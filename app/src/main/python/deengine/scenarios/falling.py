"""A falling (or thrown) object, with optional air resistance, buoyancy and varying gravity."""
import math

from ..first_order import step
from .common import numeric
from .solver import Rel, Skip
from .spec import choice, field, preset, switch

MODEL = {
    "id": "falling", "title": "Falling object", "animation": "fall",
    "description": "Something dropped or thrown straight up or down: a skydiver, a ball, a raindrop.",
    "law": "Newton's second law along the vertical: mass times acceleration equals weight plus the other forces.",
    "fields": [
        field("m", "Mass", "m", "mass", (None, "kg"), positive=True),
        field("g", "Gravity", "g", "acceleration", ("9.81", "m/s2"), positive=True, typical=10),
        field("y0", "Starting height", "y_0", "length", (None, "m"), "initial", nonnegative=True, typical=10),
        field("v0", "Starting velocity (up is +)", "v_0", "speed", ("0", "m/s"), "initial", typical=10),
        field("b", "Linear drag coefficient", "b", "drag_linear", (None, "kg/s"), "effect", positive=True),
        field("c", "Quadratic drag coefficient", "c", "drag_quadratic", (None, "kg/m"), "effect",
              positive=True, typical=0.1),
        field("rho", "Air density", "\\rho", "density", (None, "kg/m3"), "effect", positive=True, optional=True),
        field("cd", "Drag coefficient", "C_d", "dimensionless", (None, ""), "effect", positive=True, optional=True),
        field("area", "Frontal area", "A", "area", (None, "m2"), "effect", positive=True, optional=True),
        field("vt", "Terminal speed", "v_t", "speed", (None, "m/s"), "effect", positive=True, typical=10),
        field("rhof", "Fluid density", "\\rho_f", "density", (None, "kg/m3"), "effect", positive=True),
        field("vol", "Object volume", "V", "volume", (None, "m3"), "effect", positive=True, typical=1e-3),
        field("re", "Planet radius", "R", "length", ("6371", "km"), "effect", positive=True, typical=6e6),
        field("t", "Time", "t", "time", (None, "s"), "moment", nonnegative=True),
        field("y", "Height at that time", "y", "length", (None, "m"), "moment", typical=10),
        field("v", "Velocity at that time (up is +)", "v", "speed", (None, "m/s"), "moment", typical=10),
        field("tg", "Time to reach the ground", "t_g", "time", (None, "s"), "derived", nonnegative=True),
        field("vi", "Speed at the ground", "v_i", "speed", (None, "m/s"), "derived", nonnegative=True, typical=10),
        field("ymax", "Greatest height", "y_{max}", "length", (None, "m"), "derived", typical=10),
    ],
    "effects": [
        choice("drag", "Air resistance", "A force opposing the motion that grows with speed.",
               [("none", "None", []), ("linear", "Linear (slow, tiny objects)", ["b", "vt"]),
                ("quadratic", "Quadratic (most everyday objects)", ["c", "rho", "cd", "area", "vt"])], "none"),
        switch("buoyancy", "Buoyancy", "The surrounding fluid pushes up with the weight of the fluid displaced.",
               ["rhof", "vol"]),
        switch("gravity", "Gravity weakens with height", "Inverse-square gravity, for very high drops.", ["re"]),
    ],
    "presets": [
        preset("skydiver", "Skydiver", "An 80 kg skydiver, belly to earth, jumping from 4000 m.",
               {"m": ("80", "kg"), "g": ("9.81", "m/s2"), "v0": ("0", "m/s"), "y0": ("4000", "m"),
                "rho": ("1.225", "kg/m3"), "cd": ("1.0", ""), "area": ("0.5", "m2")}, {"drag": "quadratic"}),
        preset("ball", "Dropped baseball", "A baseball dropped from a 50 m tower; where is it after 2 s?",
               {"m": ("145", "g"), "g": ("9.81", "m/s2"), "v0": ("0", "m/s"), "y0": ("50", "m"),
                "rho": ("1.225", "kg/m3"), "cd": ("0.35", ""), "area": ("42", "cm2"), "t": ("2", "s")},
               {"drag": "quadratic"}),
        preset("thrown", "Ball thrown upward", "A ball thrown up at 20 m/s from a 1.5 m hand, no air resistance.",
               {"g": ("9.81", "m/s2"), "v0": ("20", "m/s"), "y0": ("1.5", "m")}, {}),
        preset("raindrop", "Raindrop", "A raindrop is seen to fall at 6.5 m/s: what is its drag coefficient c?",
               {"m": ("4.2", "mg"), "g": ("9.81", "m/s2"), "vt": ("6.5", "m/s"), "v0": ("0", "m/s"),
                "y0": ("2", "km"), "c": ("", "g/m"), "tg": ("", "min")}, {"drag": "quadratic"}),
        preset("fog", "Fog droplet", "A 20 \u00b5m droplet, slow enough for linear (Stokes) drag.",
               {"m": ("0.0000042", "mg"), "g": ("9.81", "m/s2"), "v0": ("0", "m/s"), "y0": ("10", "m"),
                "b": ("0.0000034", "g/s"), "vt": ("", "mm/s"), "tg": ("", "min")}, {"drag": "linear"}),
    ],
}


def _ge(eff, v):
    """Net downward acceleration at ground level (gravity less buoyancy)."""
    if eff.get("buoyancy") == "on":
        return v["g"] * (1 - v["rhof"] * v["vol"] / v["m"])
    return v["g"]


def _params(eff):
    p = ["g", "v0", "y0"]
    if eff.get("drag") != "none":
        p.append("vt")
    if eff.get("buoyancy") == "on":
        p += ["m", "rhof", "vol"]
    if eff.get("gravity") == "on":
        p.append("re")
    return p


def trajectory(eff, v):
    params = _params(eff)
    if any(v.get(x) is None for x in params):
        raise Skip()
    ge, v0, y0 = _ge(eff, v), v["v0"], v["y0"]
    drag = eff.get("drag")
    vt = v.get("vt") if drag != "none" else None
    varying = eff.get("gravity") == "on"
    scale = (abs(v0) + math.sqrt(v0 * v0 + 2 * abs(ge) * max(y0, 1.0))) / max(abs(ge), 1e-9)
    if vt:
        scale = max(scale, y0 / vt)
    k = abs(ge) / vt if drag == "linear" else (abs(ge) / vt ** 2 if drag == "quadratic" else 0.0)
    re = v.get("re")

    def deriv(t, s):
        y, vel = s
        acc = -ge * (re / (re + y)) ** 2 if varying else -ge
        if drag == "linear":
            acc -= k * vel
        elif drag == "quadratic":
            acc -= k * vel * abs(vel)
        return [vel, acc]

    exact = None
    if not varying and ge > 0:
        if drag == "none":
            exact = lambda t: {"y": y0 + v0 * t - ge * t * t / 2, "v": v0 - ge * t}
        elif drag == "linear":
            def exact(t):
                # Written with expm1 and a series so that very weak drag does not cancel to nothing.
                x = ge * t / vt
                d = -math.expm1(-x)                                   # 1 - e^{-x}
                rest = (-x * x / 2 + x ** 3 / 6 - x ** 4 / 24) if x < 1e-3 else d - x
                return {"v": v0 * (1 - d) - vt * d, "y": y0 + v0 * (vt / ge) * d + (vt * vt / ge) * rest}
        elif v0 <= 0 and -v0 < vt:
            a = math.atanh(-v0 / vt)

            def exact(t):
                arg = ge * t / vt + a
                # ln(cosh) written so that it cannot overflow for long falls.
                lncosh = lambda z: (z * z / 2 - z ** 4 / 12 + z ** 6 / 45 if abs(z) < 1e-2
                                    else abs(z) + math.log1p(math.exp(-2 * abs(z))) - math.log(2))
                return {"v": -vt * math.tanh(arg), "y": y0 - (vt * vt / ge) * (lncosh(arg) - lncosh(a))}
    if exact is not None:
        return {"at": exact, "ode": (deriv, [y0, v0], ["y", "v"], scale), "exact": True, "scale": scale}
    num = numeric(("falling", drag, varying, ge, v0, y0, vt, re), deriv, [y0, v0], scale)

    def at(t):
        y, vel = num.state(t)
        return {"y": y, "v": vel}
    return {"at": at, "ode": None, "exact": False, "scale": scale}


_MEMO = {}


def _memo(name, eff, v, compute):
    key = (name, tuple(sorted(eff.items())), tuple(v.get(x) for x in _params(eff)))
    if key not in _MEMO:
        if len(_MEMO) > 4000:
            _MEMO.clear()
        try:
            _MEMO[key] = compute()
        except Skip:
            _MEMO[key] = None
    if _MEMO[key] is None:
        raise Skip()
    return _MEMO[key]


def _ground_time(eff, v):
    return _memo("ground", eff, v, lambda: _ground_raw(eff, v))


def _peak(eff, v):
    return _memo("peak", eff, v, lambda: _peak_raw(eff, v))


def _ground_raw(eff, v):
    traj = trajectory(eff, v)
    if v["y0"] <= 0 and v["v0"] <= 0:
        return 0.0
    f = lambda t: traj["at"](t)["y"]
    t, step_size = 0.0, traj["scale"] / 200
    prev = f(1e-9 * traj["scale"])
    for _ in range(4000):
        nxt = t + step_size
        cur = f(nxt)
        if cur <= 0 < prev or (cur <= 0 and t > 0):
            lo, hi = t, nxt
            for _i in range(70):
                mid = (lo + hi) / 2
                if f(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        t, prev = nxt, cur
        step_size *= 1.01
    raise Skip()


def _peak_raw(eff, v):
    if v["v0"] <= 0:
        return v["y0"]
    traj = trajectory(eff, v)
    ge = _ge(eff, v)
    if ge <= 0:
        raise Skip()
    lo, hi = 0.0, v["v0"] / ge * 1.5 + 1e-9
    if traj["at"](hi)["v"] > 0:
        raise Skip()
    for _ in range(70):
        mid = (lo + hi) / 2
        if traj["at"](mid)["v"] > 0:
            lo = mid
        else:
            hi = mid
    return traj["at"]((lo + hi) / 2)["y"]


def relations(eff):
    drag = eff.get("drag")
    buoy = eff.get("buoyancy") == "on"
    extra = ["rhof", "vol"] if buoy else []
    gsym = "g_e" if buoy else "g"
    rels = []
    if drag == "linear":
        rels.append(Rel("vt", "At terminal speed the drag force balances the net weight",
                        "b\\,v_t = m\\,%s" % gsym, ["vt", "m", "g", "b"] + extra,
                        lambda v: (v["b"] * v["vt"], v["m"] * abs(_ge(eff, v))),
                        {"vt": "v_t = \\frac{m\\,%s}{b}" % gsym, "b": "b = \\frac{m\\,%s}{v_t}" % gsym,
                         "m": "m = \\frac{b\\,v_t}{%s}" % gsym}))
    if drag == "quadratic":
        rels.append(Rel("vt", "At terminal speed the drag force balances the net weight",
                        "c\\,v_t^2 = m\\,%s" % gsym, ["vt", "m", "g", "c"] + extra,
                        lambda v: (v["c"] * v["vt"] ** 2, v["m"] * abs(_ge(eff, v))),
                        {"vt": "v_t = \\sqrt{\\frac{m\\,%s}{c}}" % gsym, "c": "c = \\frac{m\\,%s}{v_t^2}" % gsym,
                         "m": "m = \\frac{c\\,v_t^2}{%s}" % gsym}))
        rels.append(Rel("c", "The quadratic drag coefficient from the shape and the air",
                        "c = \\tfrac12\\,\\rho\\,C_d\\,A", ["c", "rho", "cd", "area"],
                        lambda v: (v["c"], 0.5 * v["rho"] * v["cd"] * v["area"]),
                        {"c": "c = \\tfrac12\\,\\rho\\,C_d\\,A", "cd": "C_d = \\frac{2c}{\\rho A}",
                         "area": "A = \\frac{2c}{\\rho C_d}", "rho": "\\rho = \\frac{2c}{C_d A}"}))
    p = _params(eff)
    simple = drag == "none" and not buoy and eff.get("gravity") != "on"
    rels.append(Rel("v", "The velocity at time t from the solution of the equation of motion",
                    "v = v(t)", ["t", "v"] + p,
                    lambda v: (v["v"] + 2 * _speed(v), trajectory(eff, v)["at"](v["t"])["v"] + 2 * _speed(v)),
                    {"v": "v = v_0 - g\\,t", "t": "t = \\frac{v_0 - v}{g}", "v0": "v_0 = v + g\\,t"} if simple else None,
                    traj=True))
    rels.append(Rel("y", "The height at time t from the solution of the equation of motion",
                    "y = y(t)", ["t", "y"] + p,
                    lambda v: (v["y"] + _span(v), trajectory(eff, v)["at"](v["t"])["y"] + _span(v)),
                    {"y": "y = y_0 + v_0 t - \\tfrac12 g t^2"} if simple else None, traj=True))
    rels.append(Rel("tg", "The object reaches the ground when its height is zero",
                    "y(t_g) = 0", ["tg"] + p, lambda v: (v["tg"], _ground_time(eff, v)),
                    {"tg": "t_g = \\frac{v_0 + \\sqrt{v_0^2 + 2 g y_0}}{g}"} if simple else None, traj=True))
    rels.append(Rel("vi", "The speed at the ground is the size of the velocity at that time",
                    "v_i = |v(t_g)|", ["vi"] + p,
                    lambda v: (v["vi"], abs(trajectory(eff, v)["at"](_ground_time(eff, v))["v"])),
                    {"vi": "v_i = \\sqrt{v_0^2 + 2 g y_0}"} if simple else None, traj=True))
    rels.append(Rel("ymax", "The greatest height is where the velocity is zero",
                    "v(t_{top}) = 0,\\quad y_{max} = y(t_{top})", ["ymax"] + p,
                    lambda v: (v["ymax"] + _span(v), _peak(eff, v) + _span(v)),
                    {"ymax": "y_{max} = y_0 + \\frac{v_0^2}{2g}"} if simple else None, traj=True))
    return rels


def _speed(v):
    """A speed scale added to both sides so velocities near zero compare sensibly."""
    return abs(v.get("v0") or 0.0) + math.sqrt(2 * (v.get("g") or 9.81) * max(abs(v.get("y0") or 0.0), 1.0))


def _span(v):
    """A length scale added to both sides so heights near zero compare sensibly."""
    return max(abs(v.get("y0") or 0.0), 1.0) + (v.get("v0") or 0.0) ** 2 / (2 * (v.get("g") or 9.81))


def equation(eff):
    terms = ["-m\\,g"]
    if eff.get("gravity") == "on":
        terms = ["-m\\,g\\left(\\frac{R}{R+y}\\right)^2"]
    if eff.get("buoyancy") == "on":
        terms.append("+ \\rho_f V g")
    if eff.get("drag") == "linear":
        terms.append("- b\\,v")
    if eff.get("drag") == "quadratic":
        terms.append("- c\\,v\\,|v|")
    return "m\\,\\frac{dv}{dt} = %s,\\qquad \\frac{dy}{dt} = v" % " ".join(terms)


def derivation(eff):
    drag = eff.get("drag")
    parts = ["\\(-mg\\) is the weight, pulling down (up is positive)."]
    if eff.get("gravity") == "on":
        parts = ["The weight weakens with height as the inverse square of the distance from the planet's centre."]
    if eff.get("buoyancy") == "on":
        parts.append("\\(\\rho_f V g\\) is the buoyant force, the weight of the fluid displaced; "
                     "it reduces gravity to an effective \\(g_e = g(1 - \\rho_f V/m)\\).")
    if drag == "linear":
        parts.append("\\(-bv\\) is air resistance proportional to speed, always against the motion.")
    if drag == "quadratic":
        parts.append("\\(-cv|v|\\) is air resistance proportional to speed squared, always against the motion.")
    steps = [step("Set up the equation of motion",
                  "Newton's second law for the vertical motion. " + " ".join(parts), [equation(eff)])]
    numerical = eff.get("gravity") == "on"
    if numerical:
        steps.append(step("Solve the equation",
                          "With gravity depending on height the equation has no elementary solution, "
                          "so it is integrated numerically (fourth-order Runge-Kutta)."))
    elif drag == "none":
        steps.append(step("Solve the equation", "The acceleration is constant, so integrate twice.",
                          ["v(t) = v_0 - g\\,t", "y(t) = y_0 + v_0 t - \\tfrac12 g t^2"]))
    elif drag == "linear":
        steps.append(step("Solve the equation",
                          "The equation is linear in \\(v\\). With \\(v_t = mg/b\\) it reads "
                          "\\(v' = -(g/v_t)(v + v_t)\\), which is solved by an integrating factor; "
                          "integrating once more gives the height.",
                          ["v(t) = -v_t + (v_0 + v_t)\\,e^{-g t/v_t}",
                           "y(t) = y_0 - v_t t + \\frac{v_t (v_0 + v_t)}{g}\\left(1 - e^{-g t/v_t}\\right)"]))
    else:
        steps.append(step("Solve the equation",
                          "While the object moves down the equation is separable. With "
                          "\\(v_t = \\sqrt{mg/c}\\) and downward speed \\(u = -v\\), "
                          "\\(u' = g(1 - u^2/v_t^2)\\). If it starts by moving up, the drag changes "
                          "direction at the top and the motion is integrated numerically instead.",
                          ["u(t) = v_t \\tanh\\!\\left(\\frac{g t}{v_t} + \\tanh^{-1}\\frac{u_0}{v_t}\\right)",
                           "y(t) = y_0 - \\frac{v_t^2}{g}\\ln\\frac{\\cosh(g t/v_t + a)}{\\cosh a},\\quad "
                           "a = \\tanh^{-1}\\frac{u_0}{v_t}"]))
    return steps


def horizon(eff, v):
    try:
        return trajectory(eff, v)["scale"] * 2
    except (Skip, ValueError, ZeroDivisionError, TypeError, KeyError):
        return None


def graph(eff, v):
    traj = trajectory(eff, v)
    try:
        t_end = _ground_time(eff, v)
    except Skip:
        t_end = 0.0
    t_max = max(t_end * 1.02, (v.get("t") or 0.0) * 1.2, traj["scale"] * 0.05)
    fn = lambda t: max(traj["at"](t)["y"], 0.0) if t_end and t > t_end else traj["at"](t)["y"]
    ideal = None
    if any(eff.get(e["id"]) != e["default"] for e in MODEL["effects"]):
        g, v0, y0 = v["g"], v["v0"], v["y0"]
        ideal = lambda t: max(y0 + v0 * t - g * t * t / 2, 0.0)
    marker = None
    if v.get("t") is not None and v.get("y") is not None:
        marker = {"t": v["t"], "y": v["y"], "label": "at time t"}
    elif t_end:
        marker = {"t": t_end, "y": 0.0, "label": "reaches the ground"}
    return {"fn": fn, "ideal": ideal, "t_max": t_max, "y_label": "height (m)", "label": "Height",
            "ideal_label": "No air resistance or other effects", "marker": marker}
