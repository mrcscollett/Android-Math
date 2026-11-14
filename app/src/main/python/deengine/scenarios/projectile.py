"""A projectile in two dimensions, with optional air resistance and wind."""
import math

from ..first_order import step
from .common import numeric
from .solver import Rel, Skip
from .spec import choice, field, preset, switch

MODEL = {
    "id": "projectile", "title": "Projectile", "animation": "fall",
    "description": "Something launched at an angle: a kicked ball, a golf drive, a basketball shot.",
    "law": "Newton's second law in two directions: gravity acts downward, and air resistance (if any) opposes the velocity.",
    "fields": [
        field("v0", "Launch speed", "v_0", "speed", (None, "m/s"), "initial", positive=True, typical=10),
        field("ang", "Launch angle", "\\theta", "angle", (None, "deg"), "initial", bounds=(0.0, math.pi / 2)),
        field("y0", "Launch height", "y_0", "length", ("0", "m"), "initial", nonnegative=True),
        field("g", "Gravity", "g", "acceleration", ("9.81", "m/s2"), positive=True, typical=10),
        field("m", "Mass", "m", "mass", (None, "kg"), "effect", positive=True),
        field("b", "Linear drag coefficient", "b", "drag_linear", (None, "kg/s"), "effect", positive=True, typical=0.1),
        field("c", "Quadratic drag coefficient", "c", "drag_quadratic", (None, "kg/m"), "effect",
              positive=True, typical=1e-3),
        field("w", "Wind speed (tailwind is +)", "w", "speed", (None, "m/s"), "effect", typical=5),
        field("t", "Time", "t", "time", (None, "s"), "moment", nonnegative=True),
        field("x", "Horizontal distance at that time", "x", "length", (None, "m"), "moment", typical=10),
        field("y", "Height at that time", "y", "length", (None, "m"), "moment", typical=10),
        field("spd", "Speed at that time", "v", "speed", (None, "m/s"), "moment", nonnegative=True, typical=10),
        field("tf", "Time of flight", "t_f", "time", (None, "s"), "derived", positive=True),
        field("rng", "Range", "R", "length", (None, "m"), "derived", nonnegative=True, typical=10),
        field("hmax", "Greatest height", "H", "length", (None, "m"), "derived", nonnegative=True, typical=10),
        field("vimp", "Landing speed", "v_i", "speed", (None, "m/s"), "derived", nonnegative=True, typical=10),
    ],
    "effects": [
        choice("drag", "Air resistance", "A force opposing the velocity through the air.",
               [("none", "None", []), ("linear", "Linear", ["m", "b"]), ("quadratic", "Quadratic", ["m", "c"])], "none"),
        switch("wind", "Wind", "A steady horizontal wind; it only matters when there is air resistance.", ["w"]),
    ],
    "presets": [
        preset("kick", "Free kick", "A ball kicked at 25 m/s, 30\u00b0 above the ground, ignoring the air.",
               {"v0": ("25", "m/s"), "ang": ("30", "deg"), "y0": ("0", "m"), "g": ("9.81", "m/s2")}, {}),
        preset("golf", "Golf drive", "A 45.9 g golf ball driven at 70 m/s and 12\u00b0, with air resistance.",
               {"v0": ("70", "m/s"), "ang": ("12", "deg"), "y0": ("0", "m"), "g": ("9.81", "m/s2"),
                "m": ("45.9", "g"), "c": ("0.000218", "kg/m")}, {"drag": "quadratic"}),
        preset("hoop", "Basketball free throw", "Released 2.1 m up at 7.3 m/s; the hoop is 4.19 m away and 3.05 m high. What angle?",
               {"v0": ("7.3", "m/s"), "y0": ("2.1", "m"), "g": ("9.81", "m/s2"), "x": ("4.19", "m"), "y": ("3.05", "m")}, {}),
        preset("jump", "Long jump", "A jumper leaves the board at 9.5 m/s and lands 8.0 m away: what take-off angle?",
               {"v0": ("9.5", "m/s"), "y0": ("0", "m"), "g": ("9.81", "m/s2"), "rng": ("8.0", "m")}, {}),
        preset("javelin", "Javelin in a headwind", "An 800 g javelin thrown at 28 m/s and 36\u00b0 from 1.8 m into a 5 m/s headwind.",
               {"v0": ("28", "m/s"), "ang": ("36", "deg"), "y0": ("1.8", "m"), "g": ("9.81", "m/s2"),
                "m": ("800", "g"), "c": ("0.002", "kg/m"), "w": ("-5", "m/s")}, {"drag": "quadratic", "wind": "on"}),
    ],
}


def _params(eff):
    p = ["v0", "ang", "y0", "g"]
    drag = eff.get("drag")
    if drag == "linear":
        p += ["m", "b"]
    if drag == "quadratic":
        p += ["m", "c"]
    if drag != "none" and eff.get("wind") == "on":
        p.append("w")
    return p


def trajectory(eff, v):
    if any(v.get(x) is None for x in _params(eff)):
        raise Skip()
    v0, ang, y0, g = v["v0"], v["ang"], v["y0"], v["g"]
    ux, uy = v0 * math.cos(ang), v0 * math.sin(ang)
    drag = eff.get("drag")
    wind = v["w"] if (drag != "none" and eff.get("wind") == "on") else 0.0
    scale = (uy + math.sqrt(uy * uy + 2 * g * max(y0, 0.0))) / g + 1e-3 * v0 / g + 1e-9

    def pack(x, y, vx, vy):
        return {"x": x, "y": y, "vx": vx, "vy": vy, "spd": math.hypot(vx, vy)}

    if drag == "none":
        return {"at": lambda t: pack(ux * t, y0 + uy * t - g * t * t / 2, ux, uy - g * t), "exact": True,
                "scale": scale, "ode": (lambda t, s: [s[2], s[3], 0.0, -g], [0.0, y0, ux, uy],
                                        ["x", "y", "vx", "vy"], scale)}
    if drag == "linear":
        k = v["b"] / v["m"]

        def at(t):
            e = math.exp(-k * t)
            return pack(wind * t + (ux - wind) * (1 - e) / k, y0 - g * t / k + (uy + g / k) * (1 - e) / k,
                        wind + (ux - wind) * e, -g / k + (uy + g / k) * e)
        return {"at": at, "exact": True, "scale": scale,
                "ode": (lambda t, s: [s[2], s[3], -k * (s[2] - wind), -g - k * s[3]], [0.0, y0, ux, uy],
                        ["x", "y", "vx", "vy"], scale)}
    k = v["c"] / v["m"]

    def deriv(t, s):
        rx, ry = s[2] - wind, s[3]
        speed = math.hypot(rx, ry)
        return [s[2], s[3], -k * speed * rx, -g - k * speed * ry]

    num = numeric(("projectile", v0, ang, y0, g, k, wind), deriv, [0.0, y0, ux, uy], scale)
    return {"at": lambda t: pack(*num.state(t)), "exact": False, "scale": scale, "ode": None}


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


def _flight(eff, v):
    return _memo("flight", eff, v, lambda: _flight_raw(eff, v))


def _apex(eff, v):
    return _memo("apex", eff, v, lambda: _apex_raw(eff, v))


def _flight_raw(eff, v):
    traj = trajectory(eff, v)
    f = lambda t: traj["at"](t)["y"]
    t, step_size = 0.0, traj["scale"] / 400
    prev = f(step_size * 1e-3)
    if prev < 0:
        raise Skip()
    for _ in range(3000):
        nxt = t + step_size
        cur = f(nxt)
        if cur <= 0 and nxt > 0:
            lo, hi = t, nxt
            for _i in range(70):
                mid = (lo + hi) / 2
                if f(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        t = nxt
    raise Skip()


def _apex_raw(eff, v):
    traj = trajectory(eff, v)
    uy = v["v0"] * math.sin(v["ang"])
    if uy <= 0:
        return v["y0"]
    lo, hi = 0.0, uy / v["g"] * 1.01 + 1e-12
    for _ in range(70):
        mid = (lo + hi) / 2
        if traj["at"](mid)["vy"] > 0:
            lo = mid
        else:
            hi = mid
    return traj["at"]((lo + hi) / 2)["y"]


def _size(v):
    return (v.get("v0") or 1.0) ** 2 / (v.get("g") or 9.81) + (v.get("y0") or 0.0) + 1e-6


def relations(eff):
    p = _params(eff)
    simple = eff.get("drag") == "none"
    state = lambda key: (lambda v: (v[key if key != "spd" else "spd"] + _size(v) * (1 if key != "spd" else 0)
                                    + (v["v0"] if key == "spd" else 0),
                                    trajectory(eff, v)["at"](v["t"])[key] + _size(v) * (1 if key != "spd" else 0)
                                    + (v["v0"] if key == "spd" else 0)))
    rels = [
        Rel("x", "The horizontal position at time t from the solution of the equations of motion", "x = x(t)",
            ["t", "x"] + p, state("x"),
            {"x": "x = v_0\\cos\\theta\\;t", "t": "t = \\frac{x}{v_0\\cos\\theta}"} if simple else None, traj=True),
        Rel("y", "The height at time t from the solution of the equations of motion", "y = y(t)",
            ["t", "y"] + p, state("y"),
            {"y": "y = y_0 + v_0\\sin\\theta\\;t - \\tfrac12 g t^2"} if simple else None, traj=True),
        Rel("spd", "The speed at time t is the size of the velocity", "v = \\sqrt{v_x^2 + v_y^2}",
            ["t", "spd"] + p, state("spd"),
            {"spd": "v = \\sqrt{(v_0\\cos\\theta)^2 + (v_0\\sin\\theta - g t)^2}"} if simple else None, traj=True),
        Rel("tf", "The flight ends when the height returns to zero", "y(t_f) = 0", ["tf"] + p,
            lambda v: (v["tf"], _flight(eff, v)),
            {"tf": "t_f = \\frac{v_0\\sin\\theta + \\sqrt{v_0^2\\sin^2\\theta + 2 g y_0}}{g}"} if simple else None,
            traj=True),
        Rel("rng", "The range is the horizontal distance covered during the flight", "R = x(t_f)", ["rng"] + p,
            lambda v: (v["rng"] + _size(v), trajectory(eff, v)["at"](_flight(eff, v))["x"] + _size(v)),
            {"rng": "R = v_0\\cos\\theta\\;t_f"} if simple else None, traj=True),
        Rel("hmax", "The greatest height is where the vertical velocity is zero", "v_y(t_{top}) = 0,\\quad H = y(t_{top})",
            ["hmax"] + p, lambda v: (v["hmax"] + _size(v), _apex(eff, v) + _size(v)),
            {"hmax": "H = y_0 + \\frac{v_0^2\\sin^2\\theta}{2g}"} if simple else None, traj=True),
        Rel("vimp", "The landing speed is the speed at the end of the flight", "v_i = v(t_f)", ["vimp"] + p,
            lambda v: (v["vimp"], trajectory(eff, v)["at"](_flight(eff, v))["spd"]),
            {"vimp": "v_i = \\sqrt{v_0^2 + 2 g y_0}"} if simple else None, traj=True),
    ]
    return rels


def equation(eff):
    drag = eff.get("drag")
    wind = drag != "none" and eff.get("wind") == "on"
    rel = "(v_x - w)" if wind else "v_x"
    if drag == "linear":
        return "m\\,v_x' = -b\\,%s,\\qquad m\\,v_y' = -m g - b\\,v_y" % rel
    if drag == "quadratic":
        return ("m\\,v_x' = -c\\,|\\mathbf{u}|\\,%s,\\qquad m\\,v_y' = -m g - c\\,|\\mathbf{u}|\\,v_y,"
                "\\qquad \\mathbf{u} = (%s,\\ v_y)" % (rel, rel))
    return "m\\,v_x' = 0,\\qquad m\\,v_y' = -m g"


def derivation(eff):
    drag = eff.get("drag")
    text = "Newton's second law in the horizontal and vertical directions. Gravity acts only vertically."
    if drag != "none":
        text += " Air resistance opposes the velocity"
        text += " relative to the moving air (the wind)." if eff.get("wind") == "on" else "."
    steps = [step("Set up the equations of motion", text,
                  [equation(eff), "v_x(0) = v_0\\cos\\theta,\\qquad v_y(0) = v_0\\sin\\theta"])]
    if drag == "none":
        steps.append(step("Solve the equations", "Each acceleration is constant, so integrate twice.",
                          ["x(t) = v_0\\cos\\theta\\;t", "y(t) = y_0 + v_0\\sin\\theta\\;t - \\tfrac12 g t^2"]))
    elif drag == "linear":
        steps.append(step("Solve the equations",
                          "The two equations are separate and linear; each velocity relaxes exponentially "
                          "with rate \\(k = b/m\\), and integrating gives the position.",
                          ["v_x(t) = w + (v_0\\cos\\theta - w)\\,e^{-kt}" if eff.get("wind") == "on"
                           else "v_x(t) = v_0\\cos\\theta\\;e^{-kt}",
                           "v_y(t) = -\\frac{g}{k} + \\left(v_0\\sin\\theta + \\frac{g}{k}\\right)e^{-kt}"]))
    else:
        steps.append(step("Solve the equations",
                          "With quadratic drag the two directions are coupled through the speed and there "
                          "is no closed-form solution, so the equations are integrated numerically "
                          "(fourth-order Runge-Kutta)."))
    return steps


def horizon(eff, v):
    if v.get("v0") and v.get("g"):
        return 2.2 * (v["v0"] + math.sqrt(v["v0"] ** 2 + 2 * v["g"] * (v.get("y0") or 0.0))) / v["g"]
    return None


def graph(eff, v):
    traj = trajectory(eff, v)
    try:
        t_end = _flight(eff, v)
    except Skip:
        t_end = traj["scale"]
    t_max = max(t_end * 1.02, (v.get("t") or 0.0) * 1.1)
    ideal = None
    if eff.get("drag") != "none":
        uy, g, y0 = v["v0"] * math.sin(v["ang"]), v["g"], v["y0"]
        ideal = lambda t: max(y0 + uy * t - g * t * t / 2, 0.0)
    marker = None
    if v.get("t") is not None and v.get("y") is not None:
        marker = {"t": v["t"], "y": v["y"], "label": "at time t"}
    return {"fn": lambda t: max(traj["at"](t)["y"], 0.0) if t > t_end else traj["at"](t)["y"], "ideal": ideal,
            "t_max": t_max, "y_label": "height (m)", "label": "Height", "ideal_label": "No air resistance",
            "marker": marker}
