"""Motion of a particle in a plane: m r'' = sum of forces.

Uniform gravity, air resistance (with wind), a uniform electric field, a
uniform magnetic field perpendicular to the plane and an inverse-square
attraction towards the origin are switches. Gravity alone is projectile
motion; the magnetic field alone gives circular (cyclotron) motion; the
central attraction alone gives orbits.
"""
import math

from ..first_order import step
from .common import numeric
from .solver import Rel, Skip
from .spec import choice, field, preset, switch

G_NEWTON = 6.6743e-11
_GRAV, _MAG, _CEN = ("gravity", ["on"]), ("bfield", ["on"]), ("central", ["on"])

MODEL = {
    "id": "plane", "title": "Motion in a plane", "animation": "fall",
    "description": "A particle moving in two dimensions: projectiles, charges in fields, orbits.",
    "law": "Newton's second law in two dimensions: mass times acceleration equals the sum of the forces.",
    "fields": [
        field("m", "Mass", "m", "mass", (None, "kg"), positive=True),
        field("g", "Gravity", "g", "acceleration", (None, "m/s2"), "effect", positive=True, typical=10,
              hint="9.81 m/s\u00b2 at the Earth's surface; it acts in the -y direction."),
        field("b", "Linear drag coefficient", "b", "drag_linear", (None, "kg/s"), "effect", positive=True),
        field("c", "Quadratic drag coefficient", "c", "drag_quadratic", (None, "kg/m"), "effect",
              positive=True, typical=0.01),
        field("w", "Wind speed along x", "w", "speed", (None, "m/s"), "effect", typical=10),
        field("q", "Charge", "q", "charge", (None, "C"), "effect", typical=1e-6),
        field("Ex", "Electric field along x", "E_x", "electric_field", (None, "V/m"), "effect", typical=1e3),
        field("Ey", "Electric field along y", "E_y", "electric_field", (None, "V/m"), "effect", typical=1e3),
        field("B", "Magnetic field (out of the plane)", "B", "magnetic_field", (None, "T"), "effect", typical=1e-3),
        field("GM", "Gravitational parameter of the central body", "GM", "grav_parameter", (None, "km3/s2"),
              "effect", positive=True, typical=4e14, hint="398600 km\u00b3/s\u00b2 for the Earth."),
        field("Mc", "Mass of the central body", "M", "mass", (None, "kg"), "effect", positive=True,
              optional=True, typical=6e24),
        field("x0", "Initial x position", "x_0", "length", (None, "m"), "initial"),
        field("y0", "Initial y position", "y_0", "length", (None, "m"), "initial"),
        field("v0", "Initial speed", "v_0", "speed", (None, "m/s"), "initial", nonnegative=True, typical=10),
        field("ang", "Initial direction above the x axis", "\\theta", "angle", (None, "deg"), "initial",
              bounds=(-math.pi, math.pi)),
        field("vx0", "Initial velocity along x", "v_{x0}", "speed", (None, "m/s"), "initial", typical=10,
              hint="Give the speed and direction, or the two components; either is enough."),
        field("vy0", "Initial velocity along y", "v_{y0}", "speed", (None, "m/s"), "initial", typical=10),
        field("t", "Time", "t", "time", (None, "s"), "moment", nonnegative=True),
        field("x", "x position at that time", "x", "length", (None, "m"), "moment"),
        field("y", "y position at that time", "y", "length", (None, "m"), "moment"),
        field("vx", "Velocity along x at that time", "v_x", "speed", (None, "m/s"), "moment", typical=10),
        field("vy", "Velocity along y at that time", "v_y", "speed", (None, "m/s"), "moment", typical=10),
        field("spd", "Speed at that time", "v", "speed", (None, "m/s"), "moment", nonnegative=True, typical=10),
        field("tf", "Time of flight (until y = 0)", "t_f", "time", (None, "s"), "derived", nonnegative=True,
              show=[_GRAV]),
        field("rng", "Range (horizontal distance covered)", "R", "length", (None, "m"), "derived", typical=10,
              show=[_GRAV]),
        field("hmax", "Greatest height", "H", "length", (None, "m"), "derived", typical=10, show=[_GRAV]),
        field("vimp", "Landing speed", "v_i", "speed", (None, "m/s"), "derived", nonnegative=True, typical=10,
              show=[_GRAV]),
        field("aimp", "Landing angle below the horizontal", "\\beta", "angle", (None, "deg"), "derived",
              bounds=(0.0, math.pi / 2), show=[_GRAV]),
        field("wc", "Cyclotron angular frequency", "\\omega_c", "angular_frequency", (None, "rad/s"), "derived",
              positive=True, typical=1e6, show=[_MAG]),
        field("Tc", "Period of one circle", "T_c", "time", (None, "s"), "derived", positive=True, typical=1e-6,
              show=[_MAG]),
        field("rc", "Radius of the circle", "r_c", "length", (None, "m"), "derived", positive=True, typical=1e-2,
              show=[_MAG]),
        field("Torb", "Orbital period", "T", "time", (None, "s"), "derived", positive=True, typical=5e3,
              show=[_CEN]),
        field("ecc", "Eccentricity", "e", "plain", (None, ""), "derived", nonnegative=True, show=[_CEN]),
        field("rp", "Closest distance (periapsis)", "r_p", "length", (None, "km"), "derived", positive=True,
              typical=7e6, show=[_CEN]),
        field("ra", "Farthest distance (apoapsis)", "r_a", "length", (None, "km"), "derived", positive=True,
              typical=7e6, show=[_CEN]),
        field("vesc", "Escape speed from the starting point", "v_{esc}", "speed", (None, "m/s"), "derived",
              positive=True, typical=1e4, show=[_CEN]),
        field("eorb", "Energy per unit mass", "\\varepsilon", "specific_energy", (None, "J/kg"), "derived",
              typical=1e7, show=[_CEN]),
        field("horb", "Angular momentum per unit mass", "h", "specific_angular_momentum", (None, "m2/s"),
              "derived", nonnegative=True, typical=5e10, show=[_CEN]),
    ],
    "effects": [
        switch("gravity", "Uniform gravity", "A constant downward acceleration g.", ["g"]),
        choice("drag", "Air resistance", "A force opposing the velocity through the air.",
               [("none", "None", []), ("linear", "Linear", ["b"]), ("quadratic", "Quadratic", ["c"])], "none"),
        switch("wind", "Wind", "The air itself moves along x, so drag acts on the velocity relative to it.", ["w"]),
        switch("efield", "Electric field", "A uniform field E pushes a charge q with force qE.", ["q", "Ex", "Ey"]),
        switch("bfield", "Magnetic field", "A uniform field out of the plane turns a moving charge: force qv\u00d7B.",
               ["q", "B"]),
        switch("central", "Central attraction", "Inverse-square attraction towards the origin, as for an orbit.",
               ["GM", "Mc"]),
    ],
    "presets": [
        preset("kick", "Free kick", "A ball kicked at 25 m/s, 30 degrees above the ground.",
               {"v0": ("25", "m/s"), "ang": ("30", "deg"), "x0": ("0", "m"), "y0": ("0", "m"), "g": ("9.81", "m/s2")},
               {"gravity": "on"}),
        preset("golf", "Golf drive", "A 45.9 g ball leaving at 70 m/s and 12 degrees, with air resistance.",
               {"v0": ("70", "m/s"), "ang": ("12", "deg"), "x0": ("0", "m"), "y0": ("0", "m"), "g": ("9.81", "m/s2"),
                "m": ("45.9", "g"), "c": ("0.000218", "kg/m")}, {"gravity": "on", "drag": "quadratic"}),
        preset("hoop", "Basketball free throw", "Released at 7.3 m/s from 2.1 m up; the hoop is 4.19 m away and 3.05 m high.",
               {"v0": ("7.3", "m/s"), "x0": ("0", "m"), "y0": ("2.1", "m"), "g": ("9.81", "m/s2"),
                "x": ("4.19", "m"), "y": ("3.05", "m")}, {"gravity": "on"}),
        preset("jump", "Long jump", "A take-off speed of 9.5 m/s and a jump of 8.0 m: the take-off angle?",
               {"v0": ("9.5", "m/s"), "x0": ("0", "m"), "y0": ("0", "m"), "g": ("9.81", "m/s2"), "rng": ("8.0", "m")},
               {"gravity": "on"}),
        preset("javelin", "Javelin in a headwind", "An 800 g javelin thrown at 28 m/s and 36 degrees into a 5 m/s headwind.",
               {"v0": ("28", "m/s"), "ang": ("36", "deg"), "x0": ("0", "m"), "y0": ("1.8", "m"), "g": ("9.81", "m/s2"),
                "m": ("800", "g"), "c": ("0.002", "kg/m"), "w": ("-5", "m/s")},
               {"gravity": "on", "drag": "quadratic", "wind": "on"}),
        preset("electron", "Electron in a magnetic field", "An electron moving at 10\u2076 m/s across a 1 mT field.",
               {"m": ("9.109*10^-31", "kg"), "q": ("-1", "e"), "B": ("1", "mT"), "v0": ("1000", "km/s"),
                "ang": ("0", "deg"), "x0": ("0", "m"), "y0": ("0", "m"), "rc": ("", "mm"), "Tc": ("", "us")},
               {"bfield": "on"}),
        preset("plates", "Electron between deflection plates", "An electron at 2\u00d710\u2077 m/s crossing a 5 kV/m field for 2 ns.",
               {"m": ("9.109*10^-31", "kg"), "q": ("-1", "e"), "Ex": ("0", "V/m"), "Ey": ("-5", "kV/m"),
                "v0": ("20000", "km/s"), "ang": ("0", "deg"), "x0": ("0", "m"), "y0": ("0", "m"),
                "t": ("0.002", "us"), "y": ("", "mm")}, {"efield": "on"}),
        preset("satellite", "Satellite in low orbit", "400 km above the Earth, moving at 7.673 km/s at right angles to the radius.",
               {"GM": ("398600", "km3/s2"), "x0": ("6771", "km"), "y0": ("0", "km"), "v0": ("7.673", "km/s"),
                "ang": ("90", "deg"), "Torb": ("", "min")}, {"central": "on"}),
        preset("transfer", "Transfer orbit", "From 6678 km with 10.25 km/s: how far out does it go?",
               {"GM": ("398600", "km3/s2"), "x0": ("6678", "km"), "y0": ("0", "km"), "v0": ("10.25", "km/s"),
                "ang": ("90", "deg"), "Torb": ("", "h")}, {"central": "on"}),
    ],
}


def _on(eff, key):
    return eff.get(key) == "on"


def _cs(ang):
    """Cosine and sine, with the rounding error at right angles removed."""
    c, s = math.cos(ang), math.sin(ang)
    return (0.0 if abs(c) < 1e-15 else c), (0.0 if abs(s) < 1e-15 else s)


def _drag(eff):
    return eff.get("drag", "none")


def _windy(eff):
    return _drag(eff) != "none" and _on(eff, "wind")


def _needs_mass(eff):
    return _drag(eff) != "none" or _on(eff, "efield") or _on(eff, "bfield")


def _params(eff):
    p = ["x0", "y0", "v0", "ang"]
    if _on(eff, "gravity"):
        p.append("g")
    if _needs_mass(eff):
        p.append("m")
    if _drag(eff) == "linear":
        p.append("b")
    if _drag(eff) == "quadratic":
        p.append("c")
    if _windy(eff):
        p.append("w")
    if _on(eff, "efield"):
        p += ["q", "Ex", "Ey"]
    if _on(eff, "bfield"):
        p += [x for x in ("q", "B") if x not in p]
    if _on(eff, "central"):
        p.append("GM")
    return p


def _only(eff, *keys):
    """True when exactly these forces act."""
    acting = set()
    for k in ("gravity", "efield", "bfield", "central"):
        if _on(eff, k):
            acting.add(k)
    if _drag(eff) != "none":
        acting.add("drag")
    return acting == set(keys)


def _orbit(v):
    """Energy and angular momentum per unit mass, and the starting radius."""
    r0 = math.hypot(v["x0"], v["y0"])
    if r0 <= 0:
        raise Skip()
    ux, uy = v["v0"] * _cs(v["ang"])[0], v["v0"] * _cs(v["ang"])[1]
    h = v["x0"] * uy - v["y0"] * ux
    energy = v["v0"] ** 2 / 2 - v["GM"] / r0
    return energy, h, r0


def _scale(eff, v):
    """A time over which the motion develops, used to set the numerical step."""
    v0 = v["v0"]
    candidates = []
    if _on(eff, "gravity"):
        uy = v0 * math.sin(v["ang"])
        candidates.append((uy + math.sqrt(uy * uy + 2 * v["g"] * max(v["y0"], 0.0))) / v["g"] + 1e-3 * v0 / v["g"] + 1e-9)
    if _on(eff, "bfield") and v["B"] != 0 and v["q"] != 0:
        candidates.append(4 * 2 * math.pi * v["m"] / abs(v["q"] * v["B"]))
    if _on(eff, "central"):
        energy, _h, r0 = _orbit(v)
        if energy < 0:
            a = -v["GM"] / (2 * energy)
            candidates.append(1.5 * 2 * math.pi * math.sqrt(a ** 3 / v["GM"]))
        else:
            candidates.append(20 * r0 / max(v0, math.sqrt(v["GM"] / r0)))
    if _on(eff, "efield"):
        acc = abs(v["q"]) * math.hypot(v["Ex"], v["Ey"]) / v["m"]
        if acc > 0:
            candidates.append(max(v0 / acc, 1e-12) * 4 if v0 > 0 else 1.0)
    if not candidates:
        return 1.0
    return min(candidates)


def trajectory(eff, v):
    if any(v.get(x) is None for x in _params(eff)):
        raise Skip()
    x0, y0, v0, ang = v["x0"], v["y0"], v["v0"], v["ang"]
    ux, uy = v0 * _cs(ang)[0], v0 * _cs(ang)[1]
    g = v["g"] if _on(eff, "gravity") else 0.0
    drag = _drag(eff)
    wind = v["w"] if _windy(eff) else 0.0
    scale = _scale(eff, v)

    def pack(x, y, vx, vy):
        return {"x": x, "y": y, "vx": vx, "vy": vy, "spd": math.hypot(vx, vy)}

    fields = _on(eff, "efield") or _on(eff, "bfield") or _on(eff, "central")
    if drag == "none" and not _on(eff, "central"):
        # Constant forces, with or without a magnetic field: both have closed forms.
        qm0 = v["q"] / v["m"] if (_on(eff, "efield") or _on(eff, "bfield")) else 0.0
        ax = qm0 * v["Ex"] if _on(eff, "efield") else 0.0
        ay = (qm0 * v["Ey"] if _on(eff, "efield") else 0.0) - g
        wc = qm0 * v["B"] if _on(eff, "bfield") else 0.0
        ode = (lambda t, s: [s[2], s[3], ax + wc * s[3], ay - wc * s[2]], [x0, y0, ux, uy],
               ["x", "y", "vx", "vy"], scale)
        if wc == 0:
            return {"at": lambda t: pack(x0 + ux * t + ax * t * t / 2, y0 + uy * t + ay * t * t / 2,
                                         ux + ax * t, uy + ay * t), "exact": True, "scale": scale, "ode": ode}
        # The velocity is a steady drift plus a part that rotates at the cyclotron frequency.
        dx, dy = ay / wc, -ax / wc
        rx, ry = ux - dx, uy - dy

        def circling(t):
            c, sn = math.cos(wc * t), math.sin(wc * t)
            return pack(x0 + dx * t + (rx * sn - ry * (c - 1)) / wc, y0 + dy * t + (rx * (c - 1) + ry * sn) / wc,
                        dx + rx * c + ry * sn, dy - rx * sn + ry * c)
        return {"at": circling, "exact": True, "scale": scale, "ode": ode}
    if not fields and drag == "linear":
        k = v["b"] / v["m"]

        def at(t):
            e = math.exp(-k * t)
            return pack(x0 + wind * t + (ux - wind) * (1 - e) / k, y0 - g * t / k + (uy + g / k) * (1 - e) / k,
                        wind + (ux - wind) * e, -g / k + (uy + g / k) * e)
        return {"at": at, "exact": True, "scale": scale,
                "ode": (lambda t, s: [s[2], s[3], -k * (s[2] - wind), -g - k * s[3]], [x0, y0, ux, uy],
                        ["x", "y", "vx", "vy"], scale)}
    m = v["m"] if _needs_mass(eff) else 1.0
    kl = v["b"] / m if drag == "linear" else 0.0
    kq = v["c"] / m if drag == "quadratic" else 0.0
    qm = v["q"] / m if (_on(eff, "efield") or _on(eff, "bfield")) else 0.0
    ex, ey = (v["Ex"], v["Ey"]) if _on(eff, "efield") else (0.0, 0.0)
    bz = v["B"] if _on(eff, "bfield") else 0.0
    gm = v["GM"] if _on(eff, "central") else 0.0

    def deriv(t, s):
        x, y, vx, vy = s
        ax, ay = 0.0, -g
        if kl or kq:
            rx, ry = vx - wind, vy
            k = kl + kq * math.hypot(rx, ry)
            ax -= k * rx
            ay -= k * ry
        if qm:
            ax += qm * (ex + vy * bz)
            ay += qm * (ey - vx * bz)
        if gm:
            r3 = math.hypot(x, y) ** 3
            if r3 <= 0:
                raise ValueError("at the centre")
            ax -= gm * x / r3
            ay -= gm * y / r3
        return [vx, vy, ax, ay]

    key = ("plane", tuple(sorted(eff.items()))) + tuple(v[x] for x in _params(eff))
    num = numeric(key, deriv, [x0, y0, ux, uy], scale)
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
    """The first time after the start at which y comes back down to zero."""
    traj = trajectory(eff, v)
    f = lambda t: traj["at"](t)["y"]
    t, step_size = 0.0, traj["scale"] / 400
    if f(step_size * 1e-3) < 0:
        raise Skip()
    for _ in range(3000):
        nxt = t + step_size
        if f(nxt) <= 0 and nxt > 0:
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
    lo, hi = 0.0, None
    t, step_size = 0.0, traj["scale"] / 400
    for _ in range(3000):
        t += step_size
        if traj["at"](t)["vy"] <= 0:
            hi = t
            lo = t - step_size
            break
    if hi is None:
        raise Skip()
    for _ in range(70):
        mid = (lo + hi) / 2
        if traj["at"](mid)["vy"] > 0:
            lo = mid
        else:
            hi = mid
    return traj["at"]((lo + hi) / 2)["y"]


def _len(eff, v):
    """A length scale for comparing positions near zero."""
    t = v.get("t") or 0.0
    size = abs(v["x0"]) + abs(v["y0"]) + v["v0"] * t + 1e-9
    if _on(eff, "efield"):
        size += abs(v["q"]) * math.hypot(v["Ex"], v["Ey"]) / v["m"] * t * t / 2
    if _on(eff, "gravity"):
        size += v["v0"] ** 2 / v["g"] + v["g"] * t * t / 2
    return 2 * size


def _vel(eff, v):
    t = v.get("t") or 0.0
    size = v["v0"] + (v["g"] * t if _on(eff, "gravity") else 0.0) + abs(v.get("w") or 0.0) + 1e-9
    if _on(eff, "efield"):
        size += abs(v["q"]) * math.hypot(v["Ex"], v["Ey"]) / v["m"] * t
    return 2 * size


def relations(eff):
    p = _params(eff)
    simple = _only(eff, "gravity") or _only(eff)
    gsym = "g" if _on(eff, "gravity") else "0"
    rels = [
        Rel("vx0", "The x component of the initial velocity", "v_{x0} = v_0\\cos\\theta", ["vx0", "v0", "ang"],
            lambda v: (v["vx0"] + 2 * v["v0"] + 1e-9, v["v0"] * _cs(v["ang"])[0] + 2 * v["v0"] + 1e-9),
            {"vx0": "v_{x0} = v_0\\cos\\theta"}),
        Rel("vy0", "The y component of the initial velocity", "v_{y0} = v_0\\sin\\theta", ["vy0", "v0", "ang"],
            lambda v: (v["vy0"] + 2 * v["v0"] + 1e-9, v["v0"] * _cs(v["ang"])[1] + 2 * v["v0"] + 1e-9),
            {"vy0": "v_{y0} = v_0\\sin\\theta"}),
        Rel("v0", "The speed is the size of the velocity", "v_0 = \\sqrt{v_{x0}^2 + v_{y0}^2}", ["v0", "vx0", "vy0"],
            lambda v: (v["v0"], math.hypot(v["vx0"], v["vy0"])), {"v0": "v_0 = \\sqrt{v_{x0}^2 + v_{y0}^2}"}),
        Rel("ang", "The direction of the velocity", "\\tan\\theta = v_{y0}/v_{x0}", ["ang", "vx0", "vy0"],
            lambda v: (v["ang"] + 10.0, math.atan2(v["vy0"], v["vx0"]) + 10.0),
            {"ang": "\\theta = \\tan^{-1}\\frac{v_{y0}}{v_{x0}}"}),
    ]

    def state(key, speed):
        size = _vel if speed else _len
        return lambda v: (v[key] + size(eff, v), trajectory(eff, v)["at"](v["t"])[key] + size(eff, v))

    rels += [
        Rel("x", "The x position at time t from the solution of the equations of motion", "x = x(t)",
            ["t", "x"] + p, state("x", False),
            {"x": "x = x_0 + v_0\\cos\\theta\\;t", "t": "t = \\frac{x - x_0}{v_0\\cos\\theta}"} if simple else None,
            traj=True),
        Rel("y", "The y position at time t from the solution of the equations of motion", "y = y(t)",
            ["t", "y"] + p, state("y", False),
            {"y": "y = y_0 + v_0\\sin\\theta\\;t - \\tfrac12 %s t^2" % gsym} if simple else None, traj=True),
        Rel("vx", "The x velocity at time t", "v_x = x'(t)", ["t", "vx"] + p, state("vx", True),
            {"vx": "v_x = v_0\\cos\\theta"} if simple else None, traj=True),
        Rel("vy", "The y velocity at time t", "v_y = y'(t)", ["t", "vy"] + p, state("vy", True),
            {"vy": "v_y = v_0\\sin\\theta - %s\\,t" % gsym} if simple else None, traj=True),
        Rel("spd", "The speed is the size of the velocity", "v = \\sqrt{v_x^2 + v_y^2}", ["spd", "vx", "vy"],
            lambda v: (v["spd"], math.hypot(v["vx"], v["vy"])), {"spd": "v = \\sqrt{v_x^2 + v_y^2}"}),
    ]
    if _on(eff, "gravity") and not _on(eff, "central"):
        plain = _only(eff, "gravity")
        rels += [
            Rel("tf", "The flight ends when y returns to zero", "y(t_f) = 0", ["tf"] + p,
                lambda v: (v["tf"], _flight(eff, v)),
                {"tf": "t_f = \\frac{v_0\\sin\\theta + \\sqrt{v_0^2\\sin^2\\theta + 2 g y_0}}{g}"} if plain else None,
                traj=True),
            Rel("rng", "The range is the horizontal distance covered during the flight", "R = x(t_f) - x_0",
                ["rng"] + p,
                lambda v: (v["rng"] + _len(eff, v), trajectory(eff, v)["at"](_flight(eff, v))["x"] - v["x0"] + _len(eff, v)),
                {"rng": "R = v_0\\cos\\theta\\;t_f"} if plain else None, traj=True),
            Rel("hmax", "The greatest height is where the vertical velocity is zero",
                "v_y(t_{top}) = 0,\\quad H = y(t_{top})", ["hmax"] + p,
                lambda v: (v["hmax"] + _len(eff, v), _apex(eff, v) + _len(eff, v)),
                {"hmax": "H = y_0 + \\frac{v_0^2\\sin^2\\theta}{2g}"} if plain else None, traj=True),
            Rel("vimp", "The landing speed is the speed at the end of the flight", "v_i = v(t_f)", ["vimp"] + p,
                lambda v: (v["vimp"], trajectory(eff, v)["at"](_flight(eff, v))["spd"]),
                {"vimp": "v_i = \\sqrt{v_0^2 + 2 g y_0}"} if plain else None, traj=True),
            Rel("aimp", "The landing angle from the velocity components at the end of the flight",
                "\\tan\\beta = -v_y(t_f)/v_x(t_f)", ["aimp"] + p,
                lambda v: (v["aimp"] + 1.0, _landing_angle(eff, v) + 1.0), None, traj=True),
        ]
    if _only(eff, "bfield"):
        rels += [
            Rel("wc", "The magnetic force qvB supplies the centripetal force, so the charge circles at a fixed rate",
                "\\omega_c = \\frac{|q| B}{m}", ["wc", "q", "B", "m"],
                lambda v: (v["wc"], abs(v["q"] * v["B"]) / v["m"]),
                {"wc": "\\omega_c = \\frac{|q| B}{m}", "m": "m = \\frac{|q| B}{\\omega_c}"}),
            Rel("Tc", "The time for one circle", "T_c = \\frac{2\\pi}{\\omega_c}", ["Tc", "wc"],
                lambda v: (v["Tc"], 2 * math.pi / v["wc"]),
                {"Tc": "T_c = \\frac{2\\pi}{\\omega_c}", "wc": "\\omega_c = \\frac{2\\pi}{T_c}"}),
            Rel("rc", "The radius of the circle follows from v = r\u03c9", "r_c = \\frac{v_0}{\\omega_c} = \\frac{m v_0}{|q| B}",
                ["rc", "v0", "wc"], lambda v: (v["rc"], v["v0"] / v["wc"]),
                {"rc": "r_c = \\frac{m v_0}{|q| B}", "v0": "v_0 = r_c\\,\\omega_c", "wc": "\\omega_c = \\frac{v_0}{r_c}"}),
        ]
    if _on(eff, "central"):
        rels.append(Rel("GM", "The gravitational parameter is Newton's constant times the central mass",
                        "GM = G\\,M", ["GM", "Mc"], lambda v: (v["GM"], G_NEWTON * v["Mc"]),
                        {"GM": "GM = G\\,M", "Mc": "M = \\frac{GM}{G}"}))
    if _only(eff, "central"):
        ov = ["x0", "y0", "v0", "ang", "GM"]
        rels += [
            Rel("vesc", "Escape speed: kinetic energy just equal to the depth of the potential well",
                "v_{esc} = \\sqrt{\\frac{2GM}{r_0}}", ["vesc", "x0", "y0", "GM"],
                lambda v: (v["vesc"], math.sqrt(2 * v["GM"] / math.hypot(v["x0"], v["y0"]))),
                {"vesc": "v_{esc} = \\sqrt{\\frac{2GM}{r_0}},\\quad r_0 = \\sqrt{x_0^2 + y_0^2}"}),
            Rel("eorb", "Energy per unit mass is conserved along the orbit",
                "\\varepsilon = \\tfrac12 v_0^2 - \\frac{GM}{r_0}", ["eorb", "x0", "y0", "v0", "GM"],
                lambda v: (v["eorb"] + 2 * v["GM"] / math.hypot(v["x0"], v["y0"]),
                           v["v0"] ** 2 / 2 + v["GM"] / math.hypot(v["x0"], v["y0"])),
                {"eorb": "\\varepsilon = \\tfrac12 v_0^2 - \\frac{GM}{r_0}"}),
            Rel("horb", "Angular momentum per unit mass is conserved along the orbit",
                "h = |x_0 v_{y0} - y_0 v_{x0}|", ["horb", "x0", "y0", "v0", "ang"],
                lambda v: (v["horb"], abs(_orbit(dict(v, GM=1.0))[1])), {"horb": "h = |x_0 v_{y0} - y_0 v_{x0}|"}),
            Rel("ecc", "The shape of the orbit from its energy and angular momentum",
                "e = \\sqrt{1 + \\frac{2\\varepsilon h^2}{(GM)^2}}", ["ecc"] + ov,
                lambda v: (v["ecc"] + 1.0, _ecc(v) + 1.0), {"ecc": "e = \\sqrt{1 + \\frac{2\\varepsilon h^2}{(GM)^2}}"}),
            Rel("rp", "The closest approach", "r_p = \\frac{h^2/GM}{1 + e}", ["rp"] + ov,
                lambda v: (v["rp"], _orbit(v)[1] ** 2 / v["GM"] / (1 + _ecc(v))),
                {"rp": "r_p = \\frac{h^2/GM}{1 + e}"}),
            Rel("ra", "The farthest point (bound orbits only)", "r_a = \\frac{h^2/GM}{1 - e}", ["ra"] + ov,
                lambda v: (v["ra"], _apoapsis(v)), {"ra": "r_a = \\frac{h^2/GM}{1 - e}"}),
            Rel("Torb", "Kepler's third law (bound orbits only)",
                "T = 2\\pi\\sqrt{\\frac{a^3}{GM}},\\quad a = -\\frac{GM}{2\\varepsilon}", ["Torb"] + ov,
                lambda v: (v["Torb"], _period(v)),
                {"Torb": "T = 2\\pi\\sqrt{\\frac{a^3}{GM}},\\quad a = -\\frac{GM}{2\\varepsilon}"}),
        ]
    return rels


def _landing_angle(eff, v):
    s = trajectory(eff, v)["at"](_flight(eff, v))
    return math.atan2(-s["vy"], abs(s["vx"]))


def _ecc(v):
    energy, h, _r0 = _orbit(v)
    return math.sqrt(max(0.0, 1 + 2 * energy * h * h / v["GM"] ** 2))


def _apoapsis(v):
    energy, h, _r0 = _orbit(v)
    if energy >= 0:
        raise Skip()
    return h * h / v["GM"] / (1 - _ecc(v)) if _ecc(v) < 1 else -v["GM"] / energy


def _period(v):
    energy, _h, _r0 = _orbit(v)
    if energy >= 0:
        raise Skip()
    a = -v["GM"] / (2 * energy)
    return 2 * math.pi * math.sqrt(a ** 3 / v["GM"])


def baseline(eff):
    """The same system without air resistance."""
    base = dict(eff, drag="none", wind="off")
    return None if base == eff else base


def equation(eff):
    xs, ys = [], []
    if _on(eff, "gravity"):
        ys.append("- m g")
    drag = _drag(eff)
    rel = "(v_x - w)" if _windy(eff) else "v_x"
    if drag == "linear":
        xs.append("- b\\,%s" % rel)
        ys.append("- b\\,v_y")
    if drag == "quadratic":
        xs.append("- c\\,|\\mathbf{u}|\\,%s" % rel)
        ys.append("- c\\,|\\mathbf{u}|\\,v_y")
    if _on(eff, "efield"):
        xs.append("+ q E_x")
        ys.append("+ q E_y")
    if _on(eff, "bfield"):
        xs.append("+ q B\\,v_y")
        ys.append("- q B\\,v_x")
    if _on(eff, "central"):
        xs.append("- \\frac{G M m\\,x}{r^3}")
        ys.append("- \\frac{G M m\\,y}{r^3}")

    def side(terms):
        text = " ".join(terms) if terms else "0"
        if text.startswith("+ "):
            text = text[2:]
        if text.startswith("- "):
            text = "-" + text[2:]
        return text
    return "m\\,v_x' = %s,\\qquad m\\,v_y' = %s" % (side(xs), side(ys))


def derivation(eff):
    parts = []
    if _on(eff, "gravity"):
        parts.append("Gravity acts only along y.")
    if _drag(eff) != "none":
        parts.append("Air resistance opposes the velocity"
                     + (" relative to the moving air (the wind)." if _windy(eff) else "."))
    if _on(eff, "efield"):
        parts.append("The electric field pushes the charge with the constant force \\(q\\mathbf{E}\\).")
    if _on(eff, "bfield"):
        parts.append("The magnetic force \\(q\\mathbf{v}\\times\\mathbf{B}\\) is always at right angles to the "
                     "velocity, so it turns the particle without changing its speed.")
    if _on(eff, "central"):
        parts.append("The central body at the origin attracts the particle with a force falling off as the "
                     "square of the distance \\(r\\).")
    if not parts:
        parts.append("No force acts, so the velocity is constant.")
    steps = [step("Set up the equations of motion", "Newton's second law along x and y. " + " ".join(parts),
                  [equation(eff), "v_x(0) = v_0\\cos\\theta,\\qquad v_y(0) = v_0\\sin\\theta"])]
    fields = _on(eff, "efield") or _on(eff, "bfield") or _on(eff, "central")
    if not _on(eff, "bfield") and not _on(eff, "central") and _drag(eff) == "none":
        steps.append(step("Solve the equations", "Each acceleration is constant, so integrate twice.",
                          ["x(t) = x_0 + v_0\\cos\\theta\\;t + \\tfrac12 a_x t^2",
                           "y(t) = y_0 + v_0\\sin\\theta\\;t + \\tfrac12 a_y t^2"]))
    elif not fields and _drag(eff) == "linear":
        steps.append(step("Solve the equations",
                          "The two equations are separate and linear; each velocity relaxes exponentially "
                          "with rate \\(k = b/m\\), and integrating gives the position.",
                          ["v_x(t) = w + (v_0\\cos\\theta - w)\\,e^{-kt}" if _windy(eff)
                           else "v_x(t) = v_0\\cos\\theta\\;e^{-kt}",
                           "v_y(t) = -\\frac{g}{k} + \\left(v_0\\sin\\theta + \\frac{g}{k}\\right)e^{-kt}"]))
    elif _on(eff, "bfield") and _drag(eff) == "none" and not _on(eff, "central"):
        steps.append(step("Solve the equations",
                          "The two equations are a linear system: differentiating one and substituting the other "
                          "gives \\(v_x'' = -\\omega_c^2 v_x\\) plus a constant, with \\(\\omega_c = |q|B/m\\). "
                          "The velocity rotates at that rate, so the path is a circle"
                          + (" carried along by a steady drift set by the constant forces."
                             if (_on(eff, "efield") or _on(eff, "gravity")) else "."),
                          ["v_x'' = -\\omega_c^2\\,v_x", "r_c = \\frac{v_0}{\\omega_c}"]))
    elif _only(eff, "central"):
        steps.append(step("Solve the equations",
                          "Energy and angular momentum are conserved, which fixes the shape of the orbit (a "
                          "conic section) and its period. The position at a given time has no elementary "
                          "formula, so it is found by numerical integration (fourth-order Runge-Kutta).",
                          ["\\varepsilon = \\tfrac12 v^2 - \\frac{GM}{r},\\qquad h = x v_y - y v_x"]))
    else:
        steps.append(step("Solve the equations",
                          "With these forces together the two directions are coupled and there is no "
                          "closed-form solution, so the equations are integrated numerically "
                          "(fourth-order Runge-Kutta)."))
    return steps


def horizon(eff, v):
    try:
        if any(v.get(x) is None for x in _params(eff)):
            if _on(eff, "gravity") and v.get("v0") and v.get("g"):
                return 2.2 * (v["v0"] + math.sqrt(v["v0"] ** 2 + 2 * v["g"] * abs(v.get("y0") or 0.0))) / v["g"]
            return None
        return 2.2 * _scale(eff, v)
    except (Skip, ValueError, ZeroDivisionError, TypeError, KeyError):
        return None


def graph(eff, v):
    traj = trajectory(eff, v)
    grounded = _on(eff, "gravity") and not _on(eff, "central")
    t_end = None
    if grounded:
        try:
            t_end = _flight(eff, v)
        except Skip:
            t_end = None
    t_max = max((t_end * 1.02) if t_end else traj["scale"], (v.get("t") or 0.0) * 1.1)
    ideal = None
    if _drag(eff) != "none" and grounded:
        uy, g, y0 = v["v0"] * math.sin(v["ang"]), v["g"], v["y0"]
        ideal = lambda t: max(y0 + uy * t - g * t * t / 2, 0.0)
    marker = None
    if v.get("t") is not None and v.get("y") is not None:
        marker = {"t": v["t"], "y": v["y"], "label": "at time t"}

    def fn(t):
        y = traj["at"](t)["y"]
        return max(y, 0.0) if (t_end and t > t_end) else y
    return {"fn": fn, "ideal": ideal, "t_max": t_max, "y_label": "y (m)", "label": "y position",
            "ideal_label": "No air resistance", "marker": marker}
