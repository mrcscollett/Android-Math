"""Knowns -> unknowns solver for the physics models.

A model is a set of variables and RELATIONS among them. Some relations are
algebraic definitions (tau = R C); others tie the state at a moment to the
parameters through the solution of the model's differential equation. Given
the values the user knows, the solver repeatedly takes a relation with one
unknown left and solves it, falls back to solving two relations together,
and reports what the entered values cannot determine.
"""
import math
import time

from ..first_order import step
from ..parsing import InputError
from . import units
from .common import bisect, rk4, sample

TOLERANCE = 0.02      # entered values may disagree by this much (rounding) before it is an error
TIME_LIMIT = 10.0


class Skip(Exception):
    """A relation that does not apply for the current values (e.g. no real root)."""


class Rel:
    """One equation among variables.

    fn(v) returns (left, right) in SI for a dict of values; it may raise Skip
    or a math error where the relation is undefined. `forms` maps a variable
    id to the LaTeX of the relation rearranged for it. `traj` marks relations
    that go through the solution of the differential equation.
    """

    def __init__(self, rid, name, tex, variables, fn, forms=None, traj=False, late=False):
        self.id, self.name, self.tex = rid, name, tex
        self.late = late          # an extra (energies and the like): worked out after everything else
        self.vars = tuple(variables)
        self.fn = fn
        self.forms = forms or {}
        self.traj = traj

    def residual(self, v):
        left, right = self.fn(v)
        scale = max(abs(left), abs(right), 1e-300)
        return (left - right) / scale


def _safe(rel, v):
    try:
        r = rel.residual(v)
    except (Skip, InputError, ValueError, ZeroDivisionError, OverflowError, TypeError, KeyError):
        return None
    return r if isinstance(r, float) and math.isfinite(r) else None


def _grid(var, horizon, coarse):
    """Trial values for an unknown, ascending."""
    if var.get("bounds"):
        lo, hi = var["bounds"]
        n = 60 if coarse else 240
        return [lo + (hi - lo) * i / n for i in range(n + 1)]
    if var["quantity"] == "time" and horizon:
        n = 400
        first = [horizon * i / n for i in range(n + 1)]
        more = [horizon * (1 + 9 * i / 200) for i in range(1, 201)]
        far = [horizon * (10 + 90 * i / 200) for i in range(1, 201)]
        return first + more + far
    typical = abs(var.get("typical") or 1.0)
    step_size = 0.25 if coarse else 0.1
    decades = 9 if coarse else 15
    count = int(round(2 * decades / step_size))
    positive = [typical * 10 ** (-decades + step_size * i) for i in range(count + 1)]
    if var.get("positive"):
        return positive
    if var.get("nonnegative"):
        return [0.0] + positive
    return [-x for x in reversed(positive)] + [0.0] + positive


def solve_one(rel, uid, v, var, horizon, deadline):
    """All values of variable `uid` that satisfy `rel`, the other values being known."""
    trial = dict(v)

    def g(x):
        trial[uid] = x
        return _safe(rel, trial)

    roots = []
    hits = set()           # grid points that fit as they stand (no bisection needed)
    trail = []             # (x, residual) along the grid, None where the relation is undefined
    flat = valid = 0
    prev_x = prev_r = None
    for x in _grid(var, horizon, rel.traj and var["quantity"] != "time"):
        if time.time() > deadline:
            raise InputError("This took too long to solve. Try entering one more known value.")
        r = g(x)
        if r is None:
            prev_x = prev_r = None
            trail.append(None)
            continue
        valid += 1
        if abs(r) < 1e-5:
            flat += 1          # if most points are like this, the relation ignores this variable
        if abs(r) < 1e-12:
            roots.append(x)
            hits.add(x)
        elif prev_r is not None and prev_r != 0.0 and (r > 0) != (prev_r > 0):
            lo, hi = prev_x, x
            flo = prev_r
            ok = True
            for _ in range(80):
                mid = (lo + hi) / 2
                fm = g(mid)
                if fm is None:
                    ok = False
                    break
                if (fm > 0) == (flo > 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
                if hi - lo <= 1e-13 * max(abs(hi), abs(lo), 1e-300):
                    break
            root = (lo + hi) / 2
            check = g(root)
            # A sign change across a pole is not a root.
            if ok and check is not None and abs(check) < 1e-6:
                roots.append(root)
        prev_x, prev_r = x, r
        trail.append((x, r))
    if var["quantity"] == "time":
        roots += _dips(g, trail)       # an oscillation can meet a target twice between grid points
    ends = [p for p in trail if p is not None]
    if flat > 5 and flat >= 0.6 * valid and abs(ends[0][1]) < 1e-5 and abs(ends[-1][1]) < 1e-5:
        return []          # the relation holds whatever this variable is: it does not determine it
    if 0.0 in roots:
        # Zero itself fits, so the specks either side of it are rounding, not further answers.
        roots = [x for x in roots if x == 0.0 or x not in hits]
    out = []
    for x in roots:
        if not any(abs(x - y) <= 1e-9 * max(abs(x), abs(y), 1e-300) for y in out):
            out.append(x)
    return out


def _bisect(g, lo, flo, hi):
    for _ in range(80):
        mid = (lo + hi) / 2
        fm = g(mid)
        if fm is None:
            return None
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(abs(hi), abs(lo), 1e-300):
            break
    root = (lo + hi) / 2
    check = g(root)
    return root if check is not None and abs(check) < 1e-6 else None


def _dips(g, trail):
    """Roots the grid stepped over: the residual dips towards zero and comes back between points.

    That is two roots close together (a target met just before and just after a turning point).
    """
    found = []
    for a, b, c in zip(trail, trail[1:], trail[2:]):
        if a is None or b is None or c is None:
            continue
        (xa, ra), (xb, rb), (xc, rc) = a, b, c
        if ra == 0 or rb == 0 or rc == 0 or not ((ra > 0) == (rb > 0) == (rc > 0)):
            continue
        if not (abs(rb) <= abs(ra) and abs(rb) <= abs(rc)) or abs(rb) > 0.05:
            continue
        lo, hi = xa, xc
        best_x, best_r = xb, rb
        for _ in range(40):               # ternary search for where |residual| is smallest
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            r1, r2 = g(m1), g(m2)
            if r1 is None or r2 is None:
                break
            for x, r in ((m1, r1), (m2, r2)):
                if (r > 0) != (rb > 0) or abs(r) < abs(best_r):
                    best_x, best_r = x, r
            if (best_r > 0) != (rb > 0):
                break
            if abs(r1) < abs(r2):
                hi = m2
            else:
                lo = m1
        if (best_r > 0) != (rb > 0):
            for root in (_bisect(g, xa, ra, best_x), _bisect(g, xc, rc, best_x)):
                if root is not None:
                    found.append(root)
    return found


def _unknowns(rel, v):
    return [x for x in rel.vars if v.get(x) is None]


def _pick(roots, var):
    """The root to carry on with: the smallest time, otherwise the smallest positive value."""
    if var["quantity"] == "time":
        later = [r for r in roots if r > 1e-12] or roots
        return min(later)
    positive = [r for r in roots if r > 0]
    return min(positive) if positive else max(roots)


def run(model, effects, known, variables, deadline):
    """Solve for every unknown the relations determine.

    Returns (values, found, relations): `found` lists, in order, dicts with the
    variable id, the relation used, the chosen value and any other roots.
    """
    v = {k: known.get(k) for k in variables}
    rels = model.relations(effects)
    rels = [r for r in rels if all(x in variables for x in r.vars)]
    found = []
    dead = set()       # (relation, unknown) pairs already tried without a root: their inputs cannot change
    while True:
        horizon = model.horizon(effects, v)
        progress = False
        for stage in (0, 1, 2):
            for rel in rels:
                if (2 if rel.late else (1 if rel.traj else 0)) != stage:
                    continue
                unk = _unknowns(rel, v)
                if len(unk) != 1:
                    continue
                uid = unk[0]
                if (rel.id, uid) in dead:
                    continue
                roots = solve_one(rel, uid, v, variables[uid], horizon, deadline)
                if not roots:
                    dead.add((rel.id, uid))
                    continue
                roots = _agreeing(roots, uid, rel, rels, v) or roots
                value = _pick(roots, variables[uid])
                v[uid] = value
                found.append({"var": uid, "rel": rel, "value": value,
                              "others": [r for r in roots if r != value], "with": None})
                progress = True
                break
            if progress:
                break
        if progress:
            continue
        pair = _solve_pair(model, effects, rels, v, variables, deadline)
        if pair is None:
            break
        for item in pair:
            v[item["var"]] = item["value"]
            found.append(item)
    return v, found, rels


def _agreeing(roots, uid, rel, rels, v):
    """Of several roots, those that the other fully known relations also accept."""
    if len(roots) < 2:
        return roots
    others = [r for r in rels if r is not rel and uid in r.vars and _unknowns(r, v) == [uid]]
    if not others:
        return roots
    keep = []
    trial = dict(v)
    for x in roots:
        trial[uid] = x
        residuals = [_safe(r, trial) for r in others]
        if all(res is not None and abs(res) <= TOLERANCE for res in residuals):
            keep.append((max(abs(res) for res in residuals), x))
    if not keep:
        return []
    # Roots that fit the other relations exactly beat ones that only fit within the rounding allowance.
    best = min(score for score, _x in keep)
    return [x for score, x in keep if score <= max(10 * best, 1e-7)]


def _solve_pair(model, effects, rels, v, variables, deadline):
    """Two relations sharing the same two unknowns, solved together."""
    horizon = model.horizon(effects, v)
    # A relation with a single unknown was already tried on its own; pairing it cannot help.
    candidates = [r for r in rels if len(_unknowns(r, v)) == 2]
    for i, r1 in enumerate(candidates):
        for r2 in candidates[i + 1:]:
            both = sorted(set(_unknowns(r1, v)) | set(_unknowns(r2, v)))
            if len(both) != 2:
                continue
            for a, b in (both, both[::-1]):
                for first, second in ((r1, r2), (r2, r1)):
                    if b not in _unknowns(first, v):
                        continue
                    hit = _nested(first, second, a, b, v, variables, horizon, deadline)
                    if hit is not None:
                        va, vb = hit
                        return [{"var": a, "rel": second, "value": va, "others": [], "with": b},
                                {"var": b, "rel": first, "value": vb, "others": [], "with": a}]
    return None


def _nested(first, second, a, b, v, variables, horizon, deadline):
    """Scan `a`; for each value solve `first` for `b`, and look for `second` to hold too."""
    trial = dict(v)

    def inner(x):
        trial[a] = x
        trial[b] = None
        roots = solve_one(first, b, trial, dict(variables[b], bounds=variables[b].get("bounds")),
                          horizon, deadline)
        if not roots:
            return None, None
        y = _pick(roots, variables[b])
        trial[b] = y
        return _safe(second, trial), y

    grid = _grid(variables[a], horizon, True)
    if len(grid) > 80:
        grid = grid[::max(1, len(grid) // 80)]
    prev = None
    for x in grid:
        r, y = inner(x)
        if r is None:
            prev = None
            continue
        if prev is not None and (r > 0) != (prev[1] > 0):
            lo, flo, hi = prev[0], prev[1], x
            for _ in range(60):
                mid = (lo + hi) / 2
                fm, _y = inner(mid)
                if fm is None:
                    break
                if (fm > 0) == (flo > 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            root = (lo + hi) / 2
            fm, y = inner(root)
            if fm is not None and abs(fm) < 1e-6:
                return root, y
        prev = (x, r)
    return None


def _check_consistency(rels, v, found, inp):
    """Relations whose variables are all known must hold; a clear miss is an input error."""
    worst = 0.0
    used = {(f["rel"].id) for f in found}
    for rel in rels:
        if any(v.get(x) is None for x in rel.vars):
            continue
        r = _safe(rel, v)
        if r is None:
            continue
        worst_here = abs(r)
        if rel.id not in used and worst_here > TOLERANCE:
            left, right = rel.fn(v)
            raise InputError(
                "The entered values do not fit together. %s requires the two sides of its "
                "equation to match, but with these values they differ by %s%% (%s against %s, "
                "in SI units). Clear one of: %s." % (
                    rel.name, units.number(100 * worst_here, 3), units.number(left), units.number(right),
                    ", ".join(inp.fields[x]["label"].lower() for x in rel.vars
                              if inp.values.get(x) is not None) or "the values involved"))
        if rel.id in used or worst_here <= TOLERANCE:
            worst = max(worst, worst_here if rel.id in used else 0.0)
    return worst


def _missing(rels, v, variables):
    """For each unknown left, why, and what further value would settle it."""
    out = []
    touched = set(var["group"] for uid, var in variables.items() if v.get(uid) is not None)
    for uid, var in variables.items():
        if v.get(uid) is not None or var.get("optional"):
            continue
        # A blank "moment" is only a question once something about that moment is entered.
        if var["group"] in ("moment", "moment2") and var["group"] not in touched:
            continue
        options = []
        blocked = False
        if not any(uid in rel.vars for rel in rels):
            continue               # plays no part with these effects (e.g. mass without drag)
        for rel in rels:
            if uid not in rel.vars:
                continue
            others = [x for x in _unknowns(rel, v) if x != uid]
            if any(variables[x].get("optional") for x in others):
                continue           # an optional extra is not something to ask for
            if not others:
                blocked = True     # one unknown, yet no value satisfied the relation
                continue
            options.append(others)
        if blocked:
            reason = ("no value fits the numbers entered (for example a target that is "
                      "never reached)")
        elif options:
            # Prefer suggesting things one would measure or set over other computed quantities.
            def cost(group):
                return (sum(variables[x]["group"] in ("derived", "moment", "moment2") for x in group), len(group))
            options.sort(key=cost)
            seen, parts = set(), []
            for group in options[:2]:
                key = tuple(sorted(group))
                if key in seen:
                    continue
                seen.add(key)
                parts.append(" and ".join(variables[x]["label"].lower() for x in group))
            reason = "also enter " + ", or ".join(parts)
        else:
            continue               # it only feeds optional extras
        out.append({"id": uid, "label": var["label"], "reason": reason})
    return out


def _value_list(rel, uid, v, inp):
    parts = []
    for x in rel.vars:
        if x == uid or v.get(x) is None:
            continue
        parts.append("\\(%s = %s\\)" % (inp.fields[x]["symbol"], inp.tex(x, v[x])))
    return ", ".join(parts)


def solve(model, inp):
    spec = model.MODEL
    deadline = time.time() + TIME_LIMIT
    variables = {fid: inp.fields[fid] for fid in inp.active}
    known = {fid: inp.values[fid] for fid in inp.active}
    given = [fid for fid in inp.active if known[fid] is not None]
    if not given:
        raise InputError("Enter the values you know, then press Solve.")
    v, found, rels = run(model, inp.effects, known, variables, deadline)
    worst = _check_consistency(rels, v, found, inp)
    missing = _missing(rels, v, variables)
    if not found and missing:
        raise InputError("Nothing more can be found from these values. %s cannot be found: %s." % (
            missing[0]["label"], missing[0]["reason"]))

    steps = list(model.derivation(inp.effects))
    known_step = step("List the known values",
                      "Everything is converted to SI units before it goes into the equations.")
    known_step["table"] = {
        "headers": ["\\text{quantity}", "\\text{symbol}", "\\text{given}", "\\text{SI}"],
        "rows": [["\\text{%s}" % r[0], r[1], r[2], r[3]] for r in inp.known_rows(given)]}
    steps.append(known_step)

    for item in found:
        uid, rel = item["var"], item["rel"]
        var = variables[uid]
        text = rel.name + "."
        used = _value_list(rel, uid, v, inp)
        if used:
            text += " With " + used + "."
        lines = []
        form = rel.forms.get(uid)
        if item["with"]:
            text += (" This equation still contains the unknown %s, so it is solved together "
                     "with the equation for it, numerically." % variables[item["with"]]["label"].lower())
            lines.append(rel.tex)
        elif form:
            lines.append(form)
        else:
            lines.append(rel.tex)
            text += " This relation is solved for \\(%s\\) numerically (bisection)." % var["symbol"]
        lines.append("%s = %s" % (var["symbol"], inp.tex(uid, item["value"])))
        if item["others"]:
            text += " Other values that also fit: %s." % ", ".join(
                inp.show(uid, x) for x in sorted(item["others"])[:3])
        s = step("Find the %s" % var["label"][0].lower() + var["label"][1:], text, lines)
        steps.append(s)

    ideal_rows = _ideal_comparison(model, inp, known, v, found, deadline)
    if ideal_rows:
        s = step("Compare with the idealised model",
                 "The same known values with the friction, resistance and other losses switched off.")
        s["table"] = {"headers": ["\\text{quantity}", "\\text{with the effects}", "\\text{without}"],
                      "rows": ideal_rows}
        steps.append(s)

    summary = step("Summary", "")
    rows = []
    for fid in inp.active:
        f = variables[fid]
        if v.get(fid) is None:
            if f.get("optional"):
                continue
            rows.append(["\\text{%s}" % f["label"], f["symbol"], "\\text{--}", "\\text{not determined}"])
        else:
            rows.append(["\\text{%s}" % f["label"], f["symbol"], inp.tex(fid, v[fid]),
                         "\\text{given}" if known[fid] is not None else "\\text{found}"])
    summary["table"] = {"headers": ["\\text{quantity}", "\\text{symbol}", "\\text{value}", ""], "rows": rows}
    summary["kind"] = "answer"
    steps.append(summary)

    note = "Every found value was substituted back into the relations it appears in"
    note += " (largest mismatch %s%%)." % units.number(100 * worst, 2) if worst > 1e-9 else "; they all hold."
    agreement = _trajectory_check(model, inp.effects, v)
    verified = True
    if agreement is not None:
        if agreement < 1e-3:
            note += (" The solution of the differential equation agrees with an independent "
                     "step-by-step numerical integration.")
        else:
            verified = False
            note += " The solution of the differential equation did not match a numerical integration."
    steps.append({"title": "Check", "text": note, "latex": [], "substeps": [],
                  "kind": "check-ok" if verified else "check-bad"})

    found_list = [{"id": i["var"], "label": variables[i["var"]]["label"],
                   "symbol": variables[i["var"]]["symbol"],
                   "text": inp.show(i["var"], i["value"]), "value": i["value"]} for i in found]
    headline = {"label": "Found",
                "text": "; ".join("%s = %s" % (f["label"], f["text"]) for f in found_list[:4])
                + ("; \u2026" if len(found_list) > 4 else "")}
    method = {"id": "solve", "name": "Step by step", "applicable": True,
              "reason": spec["law"], "steps": steps, "general_latex": None, "particular_latex": None,
              "plot": {"general_expr": None, "particular_expr": None, "points": []},
              "verified": verified, "verify_note": note}
    physics = {"scenario": spec["id"], "headline": headline, "found": found_list, "missing": missing}
    physics.update(_graph(model, inp.effects, v))
    equation = model.equation(inp.effects)
    return {"schema": 1, "ok": True, "kind": "physics", "tool": "", "title": spec["title"],
            "order": 0, "indep": "t", "dep": "", "input_latex": equation, "normal_latex": equation,
            "rhs_expr": None, "ic": None, "summary": headline["text"], "other_types": [],
            "methods": [method], "physics": physics}


def _ideal_comparison(model, inp, known, v, found, deadline):
    spec = model.MODEL
    if hasattr(model, "baseline"):
        off = model.baseline(inp.effects)      # the same system without its dissipative effects
    else:
        off = {e["id"]: e["default"] for e in spec["effects"]}
    if not off or off == inp.effects or not found:
        return []
    from .common import active_ids
    ids = active_ids(spec, off)
    variables = {fid: inp.fields[fid] for fid in ids}
    try:
        v2, _found, _rels = run(model, off, {k: known.get(k) for k in ids}, variables,
                                min(deadline, time.time() + 3.0))
    except InputError:
        return []
    rows = []
    for item in found:
        uid = item["var"]
        if uid in v2 and v2[uid] is not None and known.get(uid) is None:
            if abs(v2[uid] - item["value"]) > 1e-9 * max(abs(item["value"]), 1e-300):
                rows.append(["\\text{%s}" % inp.fields[uid]["label"], inp.tex(uid, item["value"]),
                             inp.tex(uid, v2[uid])])
    return rows[:8]


def _trajectory_check(model, effects, v):
    """Relative disagreement between the model's solution and plain RK4, or None."""
    try:
        traj = model.trajectory(effects, v)
    except (Skip, ValueError, ZeroDivisionError, OverflowError, TypeError, KeyError):
        return None
    if traj is None or traj.get("ode") is None:
        return None
    deriv, state0, keys, t_end = traj["ode"]
    if not t_end or t_end <= 0 or not math.isfinite(t_end):
        return None
    try:
        samples = rk4(deriv, list(state0), t_end, 4000)
        worst = 0.0
        scale = [max(abs(s[1][i]) for s in samples) or 1.0 for i in range(len(keys))]
        for frac in (0.25, 0.5, 0.75, 1.0):
            index = int(frac * (len(samples) - 1))
            t, state = samples[index]
            exact = traj["at"](t)
            for i, key in enumerate(keys):
                worst = max(worst, abs(exact[key] - state[i]) / scale[i])
        return worst
    except (InputError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _graph(model, effects, v):
    empty = {"x_label": "", "y_label": "", "curve": [], "curve_label": "", "ideal": [],
             "ideal_label": "", "marker": None, "t_max": 1.0, "animation": None}
    try:
        g = model.graph(effects, v)
    except (Skip, ValueError, ZeroDivisionError, OverflowError, TypeError, KeyError, InputError):
        g = None
    if not g:
        return empty
    t_max = g["t_max"]
    curve = sample(g["fn"], t_max)
    if len(curve) < 2:
        return empty
    ideal = sample(g["ideal"], t_max) if g.get("ideal") else []
    values = [p[1] for p in curve]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12 * max(1.0, abs(hi)):
        lo, hi = lo - 1.0, hi + 1.0
    return {"x_label": g.get("x_label", "t (s)"), "y_label": g["y_label"], "curve": curve,
            "curve_label": g["label"], "ideal": ideal,
            "ideal_label": g.get("ideal_label", "Without the optional effects") if ideal else "",
            "marker": g.get("marker"), "t_max": t_max,
            "animation": {"kind": g.get("animation") or model.MODEL.get("animation", "capacitor"), "points": curve,
                          "min": lo, "max": hi, "label": g["y_label"]}}
