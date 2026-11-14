"""Shared machinery for physics scenarios: inputs, numerics, result assembly."""
import math

from ..first_order import step
from ..parsing import InputError
from . import units


class Inputs:
    """The user's entries for one model: SI values (None when blank), units and effects."""

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
        given = raw.get("values")
        if given is not None:
            source = given          # the form was filled in: a blank box means unknown
        else:
            # A preset: its values only. Everything else is blank, i.e. to be found.
            source = {k: {"value": v[0], "unit": v[1]} for k, v in preset["values"].items()}
        base_effects = dict(preset.get("effects", {})) if preset else {}
        self.effects = {}
        chosen = raw.get("effects") if raw.get("effects") is not None else base_effects
        for e in scenario["effects"]:
            value = chosen.get(e["id"], e["default"])
            valid = [o["id"] for o in e["options"]]
            self.effects[e["id"]] = value if value in valid else e["default"]
        self.active = active_ids(scenario, self.effects)
        self.units = {}
        self.values = {}
        for fid, f in self.fields.items():
            entry = source.get(fid) or {}
            default_unit = f["default"][1] if f.get("default") else units.choices(f["quantity"])[0]["id"]
            # "" is a real unit id (a plain number), so only a missing unit falls back.
            unit = entry.get("unit") if entry.get("unit") is not None else default_unit
            self.units[fid] = unit
            value = units.parse(entry.get("value"), f["quantity"], unit, f["label"]) \
                if fid in self.active else None
            if value is not None:
                if f.get("positive") and value <= 0:
                    raise InputError("%s must be greater than zero." % f["label"])
                if f.get("nonnegative") and value < 0:
                    raise InputError("%s cannot be negative." % f["label"])
                if f.get("bounds") and not (f["bounds"][0] <= value <= f["bounds"][1]):
                    raise InputError("%s is outside the range this model allows (%s to %s)." % (
                        f["label"], units.text(f["bounds"][0], f["quantity"], unit),
                        units.text(f["bounds"][1], f["quantity"], unit)))
            self.values[fid] = value

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


def active_ids(scenario, effects):
    """Ids of the variables that exist for the chosen effects, in the model's order."""
    wanted = set(f["id"] for f in scenario["fields"] if f["group"] != "effect")
    for e in scenario["effects"]:
        for o in e["options"]:
            if o["id"] == effects.get(e["id"], e["default"]):
                wanted.update(o["fields"])
    return [f["id"] for f in scenario["fields"] if f["id"] in wanted]


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


# --------------------------------------------------------------------------
# Building blocks for the models
# --------------------------------------------------------------------------

_NUMERIC_CACHE = {}


class Numeric:
    """A numerically integrated solution of state' = deriv(t, state), cached and interpolated."""

    def __init__(self, deriv, state0, scale):
        self.deriv, self.state0 = deriv, list(state0)
        self.scale = max(scale, 1e-12)
        self.runs = {}

    def state(self, t):
        end = self.scale
        while end < t:
            end *= 4
            if end > self.scale * 4 ** 12:
                raise ValueError("time out of range")
        if end not in self.runs:
            self.runs[end] = rk4(self.deriv, self.state0, end, 2400)
        samples = self.runs[end]
        return [interpolate(samples, t, i) for i in range(len(self.state0))]


def numeric(key, deriv, state0, scale):
    """A Numeric for these parameters, reused while the solver searches."""
    if key not in _NUMERIC_CACHE:
        if len(_NUMERIC_CACHE) > 200:
            _NUMERIC_CACHE.clear()
        _NUMERIC_CACHE[key] = Numeric(deriv, state0, scale)
    return _NUMERIC_CACHE[key]


def oscillator(m, c, k, x0, v0, f_const=0.0, f0=0.0, w=0.0):
    """Exact solution of m x'' + c x' + k x = f_const + f0 cos(w t). Returns fn(t) -> (x, v)."""
    import cmath
    resonant = False
    X = 0j
    if f0:
        denom = complex(k - m * w * w, c * w)
        if abs(denom) < 1e-9 * max(k, m * w * w, 1e-300):
            resonant = True
        else:
            X = f0 / denom

    def particular(t):
        if resonant:
            return (f_const / k + f0 * t * math.sin(w * t) / (2 * m * w),
                    f0 * (math.sin(w * t) + w * t * math.cos(w * t)) / (2 * m * w))
        e = cmath.exp(1j * w * t)
        return f_const / k + (X * e).real, (1j * w * X * e).real

    p0, dp0 = particular(0.0)
    a0, b0 = x0 - p0, v0 - dp0
    disc = c * c - 4 * m * k
    if abs(disc) < 1e-12 * max(c * c, 4 * m * k):
        r = -c / (2 * m)
        A, B = a0, b0 - r * a0

        def homogeneous(t):
            e = math.exp(r * t)
            return (A + B * t) * e, (B + r * (A + B * t)) * e
    else:
        root = cmath.sqrt(disc)
        r1, r2 = (-c + root) / (2 * m), (-c - root) / (2 * m)
        B = (b0 - r1 * a0) / (r2 - r1)
        A = a0 - B

        def homogeneous(t):
            e1, e2 = cmath.exp(r1 * t), cmath.exp(r2 * t)
            return (A * e1 + B * e2).real, (A * r1 * e1 + B * r2 * e2).real

    def at(t):
        xh, vh = homogeneous(t)
        xp, vp = particular(t)
        return xh + xp, vh + vp

    return at


def relax(final, start, tau):
    """u(t) for u' = (final - u)/tau, u(0) = start."""
    return lambda t: final + (start - final) * math.exp(-t / tau)


def driven_relax(amp, w, start, tau):
    """u(t) for u' = (amp cos(w t) - u)/tau, u(0) = start."""
    gain = amp / math.sqrt(1 + (w * tau) ** 2)
    phi = math.atan(w * tau)
    k0 = start - gain * math.cos(phi)
    return lambda t: gain * math.cos(w * t - phi) + k0 * math.exp(-t / tau)
