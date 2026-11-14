"""Shared machinery for physics scenarios: inputs, numerics, result assembly."""
import math

from ..first_order import step
from ..parsing import InputError
from . import units


class Inputs:
    """The user's choices for one scenario: SI values, effects and the question."""

    def __init__(self, scenario, raw):
        self.scenario = scenario
        self.fields = {f["id"]: f for f in scenario["fields"]}
        raw = raw or {}
        preset = None
        for p in scenario["presets"]:
            if p["id"] == raw.get("preset"):
                preset = p
        if preset is None and "values" not in raw:
            preset = scenario["presets"][0]
        base_values = dict(preset["values"]) if preset else {}
        base_effects = dict(preset.get("effects", {})) if preset else {}
        given = raw.get("values")
        self.units = {}
        self.values = {}
        if given is not None:
            source = given          # the form was filled in: a blank box means unknown
        else:
            # No form (a preset, a test or a launch intent): field defaults, then the preset.
            source = {f["id"]: {"value": f["default"][0], "unit": f["default"][1]}
                      for f in scenario["fields"] if f.get("default")}
            source.update({k: {"value": v[0], "unit": v[1]} for k, v in base_values.items()})
        for fid, f in self.fields.items():
            entry = source.get(fid) or {}
            unit = entry.get("unit") or (f["default"][1] if f.get("default") else units.choices(f["quantity"])[0]["id"])
            self.units[fid] = unit
            self.values[fid] = units.parse(entry.get("value"), f["quantity"], unit, f["label"])
        self.effects = {}
        chosen = raw.get("effects") if raw.get("effects") is not None else base_effects
        for e in scenario["effects"]:
            value = chosen.get(e["id"], e["default"])
            valid = [o["id"] for o in e["options"]]
            self.effects[e["id"]] = value if value in valid else e["default"]
        questions = [q["id"] for q in scenario["questions"]]
        self.question = raw.get("question") if raw.get("question") in questions else (
            preset.get("question") if preset and preset.get("question") in questions else questions[0])

    def get(self, fid):
        return self.values.get(fid)

    def need(self, fid, why=None):
        """A value that must be known; raises a message naming the field otherwise."""
        value = self.values.get(fid)
        f = self.fields[fid]
        if value is None:
            raise InputError("To %s, fill in %s." % (why or "answer this", f["label"].lower()))
        if f.get("positive") and value <= 0:
            raise InputError("%s must be greater than zero." % f["label"])
        if f.get("nonnegative") and value < 0:
            raise InputError("%s cannot be negative." % f["label"])
        return value

    def on(self, eid, option="on"):
        return self.effects.get(eid) == option

    def show(self, fid, value):
        """Text for a quantity in the unit the user chose for that field."""
        return units.both(value, self.fields[fid]["quantity"], self.units.get(fid))

    def tex(self, fid, value):
        return units.tex(value, self.fields[fid]["quantity"], self.units.get(fid))

    def known_rows(self, ids):
        """Table rows listing the values used, as typed and in SI."""
        rows = []
        for fid in ids:
            value = self.values.get(fid)
            if value is None:
                continue
            f = self.fields[fid]
            rows.append([f["label"], f["symbol"],
                         units.tex(value, f["quantity"], self.units.get(fid)),
                         units.tex(value, f["quantity"], units.choices(f["quantity"])[0]["id"])])
        return rows


def knowns_step(inp, ids):
    s = step("List the known values",
             "Everything is converted to SI units before it goes into the equation.")
    s["table"] = {"headers": ["\\text{quantity}", "\\text{symbol}", "\\text{given}", "\\text{SI}"],
                  "rows": [[("\\text{%s}" % r[0]), r[1], r[2], r[3]] for r in inp.known_rows(ids)]}
    return s


def rk4(deriv, state, t_end, steps, stop=None):
    """Integrate state' = deriv(t, state) from 0. Returns the list of (t, state) samples.

    `stop(t, state)` may end the run early (the sample that triggered it is kept).
    """
    h = t_end / steps
    t = 0.0
    out = [(t, list(state))]
    for _ in range(steps):
        k1 = deriv(t, state)
        k2 = deriv(t + h / 2, [s + h / 2 * k for s, k in zip(state, k1)])
        k3 = deriv(t + h / 2, [s + h / 2 * k for s, k in zip(state, k2)])
        k4 = deriv(t + h, [s + h * k for s, k in zip(state, k3)])
        state = [s + h / 6 * (a + 2 * b + 2 * c + d) for s, a, b, c, d in zip(state, k1, k2, k3, k4)]
        t += h
        if not all(math.isfinite(s) for s in state):
            raise InputError("The numerical solution became unbounded; check the values entered.")
        out.append((t, list(state)))
        if stop is not None and stop(t, state):
            break
    return out


def interpolate(samples, t, index=0):
    """Linear interpolation of component `index` of rk4 samples at time t."""
    if t <= samples[0][0]:
        return samples[0][1][index]
    if t >= samples[-1][0]:
        return samples[-1][1][index]
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (t0, s0), (t1, s1) = samples[lo], samples[hi]
    w = (t - t0) / (t1 - t0)
    return s0[index] + w * (s1[index] - s0[index])


def bisect(fn, lo, hi, tol=1e-10):
    """A root of fn in [lo, hi] (fn(lo) and fn(hi) must differ in sign)."""
    flo = fn(lo)
    for _ in range(200):
        mid = (lo + hi) / 2
        fmid = fn(mid)
        if (fmid > 0) == (flo > 0):
            lo, flo = mid, fmid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, abs(hi)):
            break
    return (lo + hi) / 2


def first_crossing(fn, t_start, t_scale, t_limit):
    """The first time after t_start at which fn changes sign, or None before t_limit."""
    step_size = t_scale / 40
    t = t_start
    value = fn(t)
    while t < t_limit:
        nxt = min(t + step_size, t_limit)
        nv = fn(nxt)
        if (nv > 0) != (value > 0) or nv == 0:
            return bisect(fn, t, nxt)
        t, value = nxt, nv
        step_size *= 1.05
    return None


def sample(fn, t_max, count=240):
    """[[t, fn(t)], ...] for the graph; non-finite values are skipped."""
    out = []
    for i in range(count + 1):
        t = t_max * i / count
        try:
            v = fn(t)
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        if math.isfinite(v):
            out.append([t, v])
    return out


def compare(closed, numerical, times, scale):
    """Largest difference between two functions of time, relative to `scale`."""
    worst = 0.0
    for t in times:
        worst = max(worst, abs(closed(t) - numerical(t)))
    return worst / max(scale, 1e-300)


def linear_working(symbol, rate_tex, source_tex, final_tex, start_tex):
    """The integrating-factor working for u' + u/tau = source, as substeps.

    The physics steps quote the result; these show where it comes from, in
    the same order the first-order solver uses.
    """
    return [
        step("Write in standard form", "Divide by the coefficient of the derivative.",
             ["\\frac{d%s}{dt} + %s\\,%s = %s" % (symbol, rate_tex, symbol, source_tex)]),
        step("Multiply by the integrating factor",
             "The factor \\(\\mu = e^{t/\\tau}\\) turns the left side into the derivative of a product.",
             ["\\mu = e^{t/\\tau}", "\\frac{d}{dt}\\left[e^{t/\\tau}\\,%s\\right] = %s\\,e^{t/\\tau}" % (symbol, source_tex)]),
        step("Integrate and divide by the factor", "The constant of integration is \\(K\\).",
             ["%s(t) = %s + K\\,e^{-t/\\tau}" % (symbol, final_tex)]),
        step("Apply the initial condition", "Set \\(t = 0\\).",
             ["K = %s - %s" % (start_tex, final_tex)]),
    ]


def method(method_id, name, reason, steps, answer_tex, verified, note):
    return {"id": method_id, "name": name, "applicable": True, "reason": reason, "steps": steps,
            "general_latex": None, "particular_latex": answer_tex,
            "plot": {"general_expr": None, "particular_expr": None, "points": []},
            "verified": verified, "verify_note": note}


def finish(scenario, inp, equation_tex, headline, methods, graph):
    """Assemble the result JSON. `graph` holds the curve, ideal curve, marker and animation."""
    for m in methods:
        if not m["applicable"]:
            continue
        m["steps"].append({"title": "Answer", "text": headline["text"],
                           "latex": [m["particular_latex"]] if m["particular_latex"] else [],
                           "substeps": [], "kind": "answer"})
        if m["verify_note"]:
            kind = {True: "check-ok", False: "check-bad", None: "check-unknown"}[m["verified"]]
            m["steps"].append({"title": "Check", "text": m["verify_note"], "latex": [],
                               "substeps": [], "kind": kind})
    physics = {"scenario": scenario["id"], "headline": headline}
    physics.update(graph)
    return {"schema": 1, "ok": True, "kind": "physics", "tool": "", "title": scenario["title"],
            "order": 0, "indep": "t", "dep": "", "input_latex": equation_tex,
            "normal_latex": equation_tex, "rhs_expr": None, "ic": None,
            "summary": headline["text"], "other_types": [], "methods": methods,
            "physics": physics}
