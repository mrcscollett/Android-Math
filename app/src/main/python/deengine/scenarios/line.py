"""Motion of a particle along a line: m x'' = sum of forces.

Every force is a switch: gravity (vertical, or on a surface or incline), a
constant applied force, sliding friction, air resistance, a spring, buoyancy,
a periodic driving force and gravity that weakens with height. With no force
switched on it is plain constant-acceleration kinematics.

x is measured along the line: upward when vertical, up the slope on an
incline.
"""
import math

from ..first_order import step
from ..parsing import InputError
from . import falling
from .common import interpolate, numeric, oscillator
from .solver import Rel, Skip
from .spec import choice, field, preset, switch

_CONST = [("drag", ["none"]), ("spring", ["off"]), ("driving", ["off"]), ("varying", ["off"]),
          ("friction", ["off"])]
_FREE = [("spring", ["off"]), ("driving", ["off"])]
_SPRING = [("spring", ["on"])]
_PLAIN = [("buoyancy", ["off"]), ("varying", ["off"])]

MODEL = {
    "id": "line", "title": "Motion along a line", "animation": "fall",
    "description": "A particle moving in one dimension under any mix of forces.",
    "law": "Newton's second law along the line: mass times acceleration equals the sum of the forces.",
    "fields": [
        field("m", "Mass", "m", "mass", (None, "kg"), positive=True),
        field("acc", "Constant acceleration", "a", "acceleration", (None, "m/s2"), typical=10, show=_CONST,
              hint="Enter it directly, or let it follow from the forces switched on."),
        field("g", "Gravity", "g", "acceleration", (None, "m/s2"), "effect", positive=True, typical=10,
              hint="9.81 m/s\u00b2 at the Earth's surface."),
        field("inc", "Angle of the incline", "\\alpha", "angle", (None, "deg"), "effect",
              bounds=(0.0, math.pi / 2), hint="0\u00b0 is a level surface; x points up the slope."),
        field("F", "Applied force (along +x)", "F", "force", (None, "N"), "effect", typical=10),
        field("mu", "Coefficient of sliding friction", "\\mu", "friction", (None, ""), "effect", nonnegative=True),
        field("b", "Linear drag (damping) coefficient", "b", "drag_linear", (None, "kg/s"), "effect", positive=True),
        field("c", "Quadratic drag coefficient", "c", "drag_quadratic", (None, "kg/m"), "effect",
              positive=True, typical=0.1),
        field("rho", "Air density", "\\rho", "density", (None, "kg/m3"), "effect", positive=True, optional=True),
        field("cd", "Drag coefficient", "C_d", "dimensionless", (None, ""), "effect", positive=True, optional=True),
        field("area", "Frontal area", "A", "area", (None, "m2"), "effect", positive=True, optional=True),
        field("vt", "Terminal speed", "v_t", "speed", (None, "m/s"), "effect", positive=True, typical=10),
        field("k", "Spring constant", "k", "spring_constant", (None, "N/m"), "effect", positive=True, typical=100),
        field("rhof", "Fluid density", "\\rho_f", "density", (None, "kg/m3"), "effect", positive=True),
        field("vol", "Object volume", "V", "volume", (None, "m3"), "effect", positive=True, typical=1e-3),
        field("F0", "Driving force amplitude", "F_0", "force", (None, "N"), "effect", positive=True),
        field("w", "Driving frequency", "\\omega", "angular_frequency", (None, "Hz"), "effect",
              positive=True, typical=10),
        field("re", "Planet radius", "R", "length", (None, "km"), "effect", positive=True, typical=6e6,
              hint="6371 km for the Earth."),
        field("x0", "Initial position", "x_0", "length", (None, "m"), "initial", typical=10),
        field("v0", "Initial velocity", "v_0", "speed", (None, "m/s"), "initial", typical=10),
        field("t", "Time", "t", "time", (None, "s"), "moment", nonnegative=True),
        field("x", "Position at that time", "x", "length", (None, "m"), "moment", typical=10),
        field("v", "Velocity at that time", "v", "speed", (None, "m/s"), "moment", typical=10),
        field("a", "Acceleration at that time", "a", "acceleration", (None, "m/s2"), "moment", typical=10,
              hide=_CONST),
        field("ek", "Kinetic energy at that time", "E_k", "energy", (None, "J"), "moment", nonnegative=True,
              optional=True),
        field("em", "Mechanical energy at that time", "E", "energy", (None, "J"), "moment", optional=True,
              show=_PLAIN),
        field("tg", "Time to reach x = 0", "t_g", "time", (None, "s"), "derived", nonnegative=True, optional=True, show=_FREE),
        field("vi", "Speed on reaching x = 0", "v_i", "speed", (None, "m/s"), "derived", nonnegative=True,
              typical=10, optional=True, show=_FREE),
        field("tstop", "Time until it stops (or turns round)", "t_s", "time", (None, "s"), "derived",
              nonnegative=True, optional=True, show=_FREE),
        field("xstop", "Position where it stops (greatest height when thrown up)", "x_s", "length", (None, "m"),
              "derived", typical=10, optional=True, show=_FREE),
        field("dstop", "Distance covered until it stops", "d_s", "length", (None, "m"), "derived",
              nonnegative=True, typical=10, optional=True, show=_FREE),
        field("N", "Normal force from the surface", "N", "force", (None, "N"), "derived", nonnegative=True,
              optional=True, show=[("gravity", ["surface"])]),
        field("w0", "Natural angular frequency", "\\omega_0", "angular_frequency", (None, "rad/s"), "derived",
              positive=True, typical=10, show=_SPRING),
        field("T", "Natural period", "T", "time", (None, "s"), "derived", positive=True, show=_SPRING),
        field("f", "Natural frequency", "f", "frequency", (None, "Hz"), "derived", positive=True, show=_SPRING),
        field("xeq", "Equilibrium position", "x_e", "length", (None, "m"), "derived", typical=0.01, show=_SPRING),
        field("zeta", "Damping ratio", "\\zeta", "dimensionless", (None, ""), "derived", positive=True,
              show=_SPRING + [("drag", ["linear"])]),
        field("wd", "Damped angular frequency", "\\omega_d", "angular_frequency", (None, "rad/s"), "derived",
              positive=True, typical=10, optional=True, show=_SPRING + [("drag", ["linear"])]),
        field("tau", "Decay time of the amplitude", "\\tau", "time", (None, "s"), "derived", positive=True,
              show=_SPRING + [("drag", ["linear"])]),
        field("amp", "Steady-state amplitude", "A_s", "length", (None, "m"), "derived", positive=True,
              typical=0.01, show=[("driving", ["on"]), ("drag", ["none", "linear"]), ("friction", ["off"])]),
        field("E", "Starting mechanical energy", "E_0", "energy", (None, "J"), "derived", optional=True,
              show=_PLAIN),
        field("elost", "Energy removed by drag and friction", "E_{lost}", "energy", (None, "J"), "derived",
              optional=True, show=_PLAIN + [("force", ["off"]), ("driving", ["off"])],
              hide=[("drag", ["none"]), ("friction", ["off"])]),
    ],
    "effects": [
        choice("gravity", "Gravity", "The weight, acting straight down or along a slope.",
               [("none", "None", []), ("vertical", "Vertical (x is height)", ["g"]),
                ("surface", "On a surface or incline", ["g", "inc"])], "none"),
        switch("force", "Applied force", "A constant push or pull along the line.", ["F"]),
        switch("friction", "Sliding friction", "\u03bcN against the motion; it can hold the object still.", ["mu"]),
        choice("drag", "Air resistance", "A force opposing the motion that grows with speed.",
               [("none", "None", []), ("linear", "Linear (slow motion, dampers)", ["b", "vt"]),
                ("quadratic", "Quadratic (most everyday objects)", ["c", "rho", "cd", "area", "vt"])], "none"),
        switch("spring", "Spring", "A restoring force -kx towards x = 0 (Hooke's law).", ["k"]),
        switch("buoyancy", "Buoyancy", "The surrounding fluid pushes up with the weight of the fluid displaced.",
               ["rhof", "vol"]),
        switch("driving", "Driving force", "A periodic push, F\u2080 cos(\u03c9t).", ["F0", "w"]),
        switch("varying", "Gravity weakens with height", "Inverse-square gravity, for very high drops.", ["re"]),
    ],
    "presets": [
        preset("accel", "Car accelerating", "From rest to 27 m/s in 6 s: the acceleration and the distance.",
               {"x0": ("0", "m"), "v0": ("0", "m/s"), "t": ("6", "s"), "v": ("27", "m/s")}, {}),
        preset("thrown", "Ball thrown upward", "A ball thrown up at 20 m/s from a 1.5 m hand, no air resistance.",
               {"g": ("9.81", "m/s2"), "v0": ("20", "m/s"), "x0": ("1.5", "m")}, {"gravity": "vertical"}),
        preset("skydiver", "Skydiver", "An 80 kg skydiver, belly to earth, jumping from 4000 m.",
               {"m": ("80", "kg"), "g": ("9.81", "m/s2"), "v0": ("0", "m/s"), "x0": ("4000", "m"),
                "rho": ("1.225", "kg/m3"), "cd": ("1.0", ""), "area": ("0.5", "m2")},
               {"gravity": "vertical", "drag": "quadratic"}),
        preset("ball", "Dropped baseball", "A baseball dropped from a 50 m tower; where is it after 2 s?",
               {"m": ("145", "g"), "g": ("9.81", "m/s2"), "v0": ("0", "m/s"), "x0": ("50", "m"),
                "rho": ("1.225", "kg/m3"), "cd": ("0.35", ""), "area": ("42", "cm2"), "t": ("2", "s")},
               {"gravity": "vertical", "drag": "quadratic"}),
        preset("raindrop", "Raindrop", "A raindrop is seen to fall at 6.5 m/s: what is its drag coefficient c?",
               {"m": ("4.2", "mg"), "g": ("9.81", "m/s2"), "vt": ("6.5", "m/s"), "v0": ("0", "m/s"),
                "x0": ("2", "km"), "c": ("", "g/m"), "tg": ("", "min")}, {"gravity": "vertical", "drag": "quadratic"}),
        preset("fog", "Fog droplet", "A 20 \u00b5m droplet, slow enough for linear (Stokes) drag.",
               {"m": ("0.0000042", "mg"), "g": ("9.81", "m/s2"), "v0": ("0", "m/s"), "x0": ("10", "m"),
                "b": ("0.0000034", "g/s"), "vt": ("", "mm/s"), "tg": ("", "min")},
               {"gravity": "vertical", "drag": "linear"}),
        preset("braking", "Car braking", "A car at 25 m/s brakes on a level road with \u03bc = 0.7: how far does it go?",
               {"g": ("9.81", "m/s2"), "inc": ("0", "deg"), "mu": ("0.7", ""), "x0": ("0", "m"), "v0": ("25", "m/s")},
               {"gravity": "surface", "friction": "on"}),
        preset("ramp", "Block sliding down a ramp", "Released 2 m up a 30\u00b0 ramp with \u03bc = 0.2.",
               {"g": ("9.81", "m/s2"), "inc": ("30", "deg"), "mu": ("0.2", ""), "x0": ("2", "m"), "v0": ("0", "m/s")},
               {"gravity": "surface", "friction": "on"}),
        preset("push", "Crate pushed across a floor", "A 20 kg crate pushed with 120 N against friction \u03bc = 0.4 for 3 s.",
               {"m": ("20", "kg"), "g": ("9.81", "m/s2"), "inc": ("0", "deg"), "mu": ("0.4", ""), "F": ("120", "N"),
                "x0": ("0", "m"), "v0": ("0", "m/s"), "t": ("3", "s")},
               {"gravity": "surface", "friction": "on", "force": "on"}),
        preset("car", "Car suspension", "One corner of a car (400 kg) pushed down 5 cm and released: where is it after 0.3 s?",
               {"m": ("400", "kg"), "k": ("40", "kN/m"), "x0": ("5", "cm"), "v0": ("0", "m/s"),
                "b": ("3500", "kg/s"), "t": ("0.3", "s"), "x": ("", "cm")}, {"spring": "on", "drag": "linear"}),
        preset("door", "Door closer", "What damping makes a 30 kg door on a 60 N/m closer shut without bouncing (\u03b6 = 1)?",
               {"m": ("30", "kg"), "k": ("60", "N/m"), "zeta": ("1", ""), "x0": ("60", "cm"), "v0": ("0", "m/s"),
                "t": ("2", "s"), "x": ("", "cm")}, {"spring": "on", "drag": "linear"}),
        preset("cart", "Lab cart on a spring", "A cart on a spring is timed at 1.0 s per swing: what is the spring constant?",
               {"m": ("0.5", "kg"), "T": ("1.0", "s"), "x0": ("10", "cm"), "v0": ("0", "m/s")}, {"spring": "on"}),
        preset("washer", "Washing machine", "A 60 kg drum on its mounts, shaken by an unbalanced load at 5 Hz.",
               {"m": ("60", "kg"), "k": ("60", "kN/m"), "x0": ("0", "cm"), "v0": ("0", "m/s"),
                "b": ("600", "kg/s"), "F0": ("100", "N"), "w": ("5", "Hz"), "amp": ("", "mm")},
               {"spring": "on", "drag": "linear", "driving": "on"}),
        preset("scale", "Hanging scale", "A 2 kg bag stretches a spring scale 4 cm: what is k, and how fast does it bob?",
               {"m": ("2", "kg"), "g": ("9.81", "m/s2"), "xeq": ("-4", "cm"), "x0": ("-6", "cm"), "v0": ("0", "m/s")},
               {"spring": "on", "gravity": "vertical"}),
    ],
}


def _on(eff, key):
    return eff.get(key) == "on"


def _drag(eff):
    return eff.get("drag", "none")


def _grav(eff):
    return eff.get("gravity", "none")


def _const(eff):
    return (_drag(eff) == "none" and not _on(eff, "spring") and not _on(eff, "driving")
            and not _on(eff, "varying") and not _on(eff, "friction"))


def _forced(eff):
    """True when some constant force is switched on."""
    return _grav(eff) != "none" or _on(eff, "force") or _on(eff, "buoyancy")


def _needs_mass(eff):
    return (_on(eff, "force") or _on(eff, "buoyancy") or _drag(eff) != "none" or _on(eff, "spring")
            or _on(eff, "driving"))


def _force_vars(eff):
    p = []
    if _grav(eff) != "none":
        p.append("g")
    if _grav(eff) == "surface":
        p.append("inc")
    if _on(eff, "force"):
        p.append("F")
    if _on(eff, "buoyancy"):
        p += ["rhof", "vol"]
    if _on(eff, "force") or _on(eff, "buoyancy"):
        p.append("m")
    return p


def _params(eff):
    if _const(eff):
        return ["acc", "x0", "v0"]
    p = ["x0", "v0"] + _force_vars(eff)
    if _needs_mass(eff) and "m" not in p:
        p.append("m")
    if _on(eff, "friction"):
        p.append("mu")
    if _drag(eff) == "linear":
        p.append("b")
    if _drag(eff) == "quadratic":
        p.append("c")
    if _on(eff, "spring"):
        p.append("k")
    if _on(eff, "driving"):
        p += ["F0", "w"]
    if _on(eff, "varying"):
        p.append("re")
    return p


def _check(eff):
    if _on(eff, "friction") and _grav(eff) != "surface":
        raise InputError("Sliding friction needs a surface to press on: set Gravity to "
                         "\"On a surface or incline\" (an angle of 0 is a level floor).")
    if (_on(eff, "buoyancy") or _on(eff, "varying")) and _grav(eff) != "vertical":
        raise InputError("Buoyancy and gravity that weakens with height apply to vertical motion: "
                         "set Gravity to \"Vertical\".")


def _gpar(eff, v):
    """The part of g acting back along the line."""
    if _grav(eff) == "vertical":
        return v["g"]
    if _grav(eff) == "surface":
        return v["g"] * math.sin(v["inc"])
    return 0.0


def _weight_acc(eff, v):
    """Acceleration from gravity and buoyancy (the part that weakens with height)."""
    a = -_gpar(eff, v)
    if _on(eff, "buoyancy"):
        a += v["rhof"] * v["vol"] * v["g"] / v["m"]
    return a


def _acc(eff, v):
    """Acceleration from the constant forces."""
    a = _weight_acc(eff, v)
    if _on(eff, "force"):
        a += v["F"] / v["m"]
    return a


def _friction_acc(eff, v):
    return v["mu"] * v["g"] * math.cos(v["inc"]) if _on(eff, "friction") else 0.0


def _kin_scale(a, x0, v0):
    a = abs(a)
    if a > 0:
        return (abs(v0) + math.sqrt(v0 * v0 + 2 * a * max(abs(x0), 1.0))) / a
    return max(abs(x0), 1.0) / abs(v0) if v0 else 1.0


def _sign(x):
    return (x > 0) - (x < 0)


def _friction_const(ac, fr, x0, v0):
    """Exact motion under a constant force with sliding friction: fn(t) -> (x, v, a)."""
    pieces = []          # (t_start, x, v, a)
    t1, x1 = 0.0, x0
    if v0 != 0:
        s = _sign(v0)
        a1 = ac - s * fr
        if a1 * s < 0:
            t1 = -v0 / a1
            x1 = x0 + v0 * t1 + a1 * t1 * t1 / 2
            pieces.append((0.0, x0, v0, a1))
        else:
            return lambda t: (x0 + v0 * t + a1 * t * t / 2, v0 + a1 * t, a1)
    a2 = 0.0 if abs(ac) <= fr else ac - _sign(ac) * fr
    pieces.append((t1, x1, 0.0, a2))

    def at(t):
        start, x, vel, a = pieces[-1] if t >= pieces[-1][0] else pieces[0]
        d = t - start
        return x + vel * d + a * d * d / 2, vel + a * d, a
    return at


class _Stick:
    """Runge-Kutta integration with sliding friction that can hold the object still."""

    def __init__(self, other, fr, state0, scale):
        self.other, self.fr, self.state0 = other, fr, list(state0)
        self.scale = max(scale, 1e-12)
        self.runs = {}

    def _direction(self, t, x):
        o = self.other(t, x, 0.0)
        return 0 if abs(o) <= self.fr else _sign(o)

    def _run(self, end, steps=2400):
        h = end / steps
        t, (x, vel) = 0.0, self.state0
        s = _sign(vel) or self._direction(t, x)
        out = [(t, [x, vel])]
        for _ in range(steps):
            if s == 0:
                s = self._direction(t + h, x)
                t += h
                out.append((t, [x, 0.0]))
                continue
            f = lambda tt, xx, vv: self.other(tt, xx, vv) - s * self.fr
            k1x, k1v = vel, f(t, x, vel)
            k2x, k2v = vel + h / 2 * k1v, f(t + h / 2, x + h / 2 * k1x, vel + h / 2 * k1v)
            k3x, k3v = vel + h / 2 * k2v, f(t + h / 2, x + h / 2 * k2x, vel + h / 2 * k2v)
            k4x, k4v = vel + h * k3v, f(t + h, x + h * k3x, vel + h * k3v)
            nx = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            nv = vel + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            if not (math.isfinite(nx) and math.isfinite(nv)):
                raise InputError("The numerical solution became unbounded; check the values entered.")
            if nv * s < 0:
                # It came to rest during this step: stop it there and decide whether friction holds it.
                frac = vel / (vel - nv)
                nx, nv = x + frac * (nx - x), 0.0
                s = self._direction(t + h, nx)
            x, vel, t = nx, nv, t + h
            out.append((t, [x, vel]))
        return out

    def state(self, t):
        end = self.scale
        while end < t:
            end *= 4
            if end > self.scale * 4 ** 12:
                raise ValueError("time out of range")
        if end not in self.runs:
            self.runs[end] = self._run(end)
        return [interpolate(self.runs[end], t, i) for i in (0, 1)]


_STICK = {}


def trajectory(eff, v):
    """The motion for these values: {"at": fn(t) -> {x, v, a}, ...}."""
    _check(eff)
    params = _params(eff)
    if any(v.get(x) is None for x in params):
        raise Skip()
    x0, v0 = v["x0"], v["v0"]
    if _const(eff):
        a = v["acc"]
        scale = _kin_scale(a, x0, v0)
        return {"at": lambda t: {"x": x0 + v0 * t + a * t * t / 2, "v": v0 + a * t, "a": a}, "exact": True,
                "scale": scale, "ode": (lambda t, s: [s[1], a], [x0, v0], ["x", "v"], scale)}
    ac = _acc(eff, v)
    fr = _friction_acc(eff, v)
    drag, spring, driving, varying = _drag(eff), _on(eff, "spring"), _on(eff, "driving"), _on(eff, "varying")
    m = v["m"] if _needs_mass(eff) else 1.0
    kl = v["b"] / m if drag == "linear" else 0.0
    kq = v["c"] / m if drag == "quadratic" else 0.0
    ks = v["k"] / m if spring else 0.0
    f0, w = (v["F0"] / m, v["w"]) if driving else (0.0, 0.0)
    wa = _weight_acc(eff, v)
    fa = ac - wa
    re = v["re"] if varying else None

    def other(t, x, vel):
        a = (wa * (re / (re + x)) ** 2 if varying else wa) + fa
        return a - kl * vel - kq * vel * abs(vel) - ks * x + (f0 * math.cos(w * t) if f0 else 0.0)

    deriv = lambda t, s: [s[1], other(t, s[0], s[1])]
    if spring:
        scale = 6 * 2 * math.pi / math.sqrt(ks)
    elif driving:
        scale = 6 * 2 * math.pi / w
    else:
        scale = _kin_scale(max(abs(ac), fr), x0, v0)
        if drag == "linear":
            scale = max(scale, abs(x0) * kl / abs(ac)) if ac else 5 / kl
        if drag == "quadratic":
            scale = max(scale, abs(x0) * math.sqrt(kq / abs(ac))) if ac else 5 / (kq * max(abs(v0), 1e-9))

    def pack(t, x, vel):
        a = other(t, x, vel)
        if fr:
            if vel != 0:
                a -= _sign(vel) * fr
            else:
                a = 0.0 if abs(a) <= fr else a - _sign(a) * fr
        return {"x": x, "v": vel, "a": a}

    if fr and drag == "none" and not spring and not driving and not varying:
        exact = _friction_const(ac, fr, x0, v0)

        def at(t):
            x, vel, a = exact(t)
            return {"x": x, "v": vel, "a": a}
        return {"at": at, "exact": True, "scale": scale, "ode": None}
    key = ("line", tuple(sorted(eff.items()))) + tuple(v[x] for x in params)
    if fr:
        if key not in _STICK:
            if len(_STICK) > 200:
                _STICK.clear()
            _STICK[key] = _Stick(other, fr, [x0, v0], scale)
        num = _STICK[key]
        return {"at": lambda t: pack(t, *num.state(t)), "exact": False, "scale": scale, "ode": None}
    if spring and drag != "quadratic" and not varying:
        osc = oscillator(m, kl * m, ks * m, x0, v0, ac * m, f0 * m, w)
        return {"at": lambda t: pack(t, *osc(t)), "exact": True, "scale": scale,
                "ode": (deriv, [x0, v0], ["x", "v"], scale)}
    if not spring and not driving and not varying and ac == 0:
        # Drag alone: the velocity dies away from its initial value.
        if drag == "linear":
            at = lambda t: pack(t, x0 - v0 * math.expm1(-kl * t) / kl, v0 * math.exp(-kl * t))
        else:
            at = lambda t: pack(t, x0 + _sign(v0) * math.log1p(kq * abs(v0) * t) / kq, v0 / (1 + kq * abs(v0) * t))
        return {"at": at, "exact": True, "scale": scale, "ode": (deriv, [x0, v0], ["x", "v"], scale)}
    if (not spring and not driving and ac != 0 and not (varying and (_on(eff, "force") or ac > 0))
            and (drag != "none" or varying)):
        # A constant net force with drag: the closed forms worked out in falling.py, which has the
        # force along -x. A force along +x is the mirror image.
        flip = -1.0 if ac > 0 else 1.0
        vt = abs(ac) / kl if drag == "linear" else (math.sqrt(abs(ac) / kq) if drag == "quadratic" else None)
        sub = falling.trajectory({"drag": drag, "buoyancy": "off", "gravity": "on" if varying else "off"},
                                 {"g": abs(ac), "v0": flip * v0, "y0": flip * x0, "vt": vt, "re": re})

        def at(t):
            s = sub["at"](t)
            return pack(t, flip * s["y"], flip * s["v"])
        return {"at": at, "exact": sub["exact"], "scale": sub["scale"],
                "ode": (deriv, [x0, v0], ["x", "v"], sub["scale"]) if sub["exact"] else None}
    num = numeric(key, deriv, [x0, v0], scale)
    return {"at": lambda t: pack(t, *num.state(t)), "exact": False, "scale": scale, "ode": None}


_MEMO = {}


def _memo(name, eff, v, compute):
    key = (name, tuple(sorted(eff.items())), tuple(v.get(x) for x in _params(eff)))
    if key not in _MEMO:
        if len(_MEMO) > 4000:
            _MEMO.clear()
        try:
            _MEMO[key] = compute()
        except (Skip, ValueError, InputError):
            _MEMO[key] = None
    if _MEMO[key] is None:
        raise Skip()
    return _MEMO[key]


def _zero_time(eff, v):
    """The first time after the start at which x = 0."""
    return _memo("zero", eff, v, lambda: _zero_raw(eff, v))


def _stop(eff, v):
    """(time, position) at which the velocity first reaches zero."""
    return _memo("stop", eff, v, lambda: _stop_raw(eff, v))


def _refine(f, lo, hi, positive):
    for _ in range(80):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == positive:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _zero_raw(eff, v):
    x0, v0 = v["x0"], v["v0"]
    if _const(eff):
        a = v["acc"]
        if a == 0:
            roots = [-x0 / v0] if v0 else []
        else:
            disc = v0 * v0 - 2 * a * x0
            if disc < 0:
                raise Skip()
            # Written to avoid cancellation between v0 and the square root.
            q = -(v0 + (math.sqrt(disc) if v0 >= 0 else -math.sqrt(disc)))
            roots = [q / a] + ([2 * x0 / q] if q else [])
        tiny = 1e-12 * _kin_scale(a, x0, v0)
        roots = [r for r in roots if r > tiny]
        if x0 == 0 and not roots:
            return 1e-300
        if not roots:
            raise Skip()
        return min(roots)
    traj = trajectory(eff, v)
    f = lambda t: traj["at"](t)["x"]
    start = f(1e-9 * traj["scale"])
    if x0 == 0 and (v0 == 0 or start == 0):
        return 1e-300
    positive = start > 0
    t, step_size = 0.0, traj["scale"] / 200
    while t < 400 * traj["scale"]:
        nxt = t + step_size
        if (f(nxt) > 0) != positive:
            return _refine(f, t, nxt, positive)
        t = nxt
        step_size *= 1.01
    raise Skip()


def _stop_raw(eff, v):
    x0, v0 = v["x0"], v["v0"]
    if v0 == 0:
        raise Skip()
    if _const(eff):
        a = v["acc"]
        if a * v0 >= 0:
            raise Skip()
        t = -v0 / a
        return t, x0 + v0 * t / 2
    traj = trajectory(eff, v)
    f = lambda t: traj["at"](t)["v"]
    positive = v0 > 0
    t, step_size = 0.0, traj["scale"] / 400
    while t < 400 * traj["scale"]:
        nxt = t + step_size
        if (f(nxt) > 0) != positive or f(nxt) == 0:
            ts = _refine(f, t, nxt, positive)
            return ts, traj["at"](ts)["x"]
        t = nxt
        step_size *= 1.01
    raise Skip()


def _sizes(eff, v):
    """Length and speed scales of the motion, so values near zero are compared sensibly."""
    x0, v0 = v["x0"], v["v0"]
    t = v.get("t") or 0.0
    if _const(eff):
        ac = abs(v["acc"])
        return 2 * (abs(x0) + abs(v0) * t + ac * t * t / 2) + 1e-9, 2 * (abs(v0) + ac * t) + 1e-9
    ac = abs(_acc(eff, v)) + _friction_acc(eff, v)
    m = v["m"] if _needs_mass(eff) else 1.0
    f0 = v["F0"] / m if _on(eff, "driving") else 0.0
    if _on(eff, "spring"):
        w0 = math.sqrt(v["k"] / m)
        size = abs(x0) + abs(v0) / w0 + (ac + f0) / (w0 * w0)
        return 2 * size + 1e-9, 2 * size * w0 + 1e-9
    size = abs(x0) + abs(v0) * t + (ac + f0) * t * t / 2
    return 2 * size + 1e-9, 2 * (abs(v0) + (ac + f0) * t) + 1e-9


def _energy(eff, v, x, vel):
    e = 0.5 * v["m"] * vel * vel + v["m"] * _gpar(eff, v) * x
    if _on(eff, "spring"):
        e += 0.5 * v["k"] * x * x
    return e


def relations(eff):
    _check(eff)
    const = _const(eff)
    p = _params(eff)
    fv = _force_vars(eff)
    drag = _drag(eff)
    rels = []

    def at(v, key, t=None):
        return trajectory(eff, v)["at"](v["t"] if t is None else t)[key]

    if const and _forced(eff):
        terms = []
        if _grav(eff) == "vertical":
            terms.append("-g")
        if _grav(eff) == "surface":
            terms.append("-g\\sin\\alpha")
        if _on(eff, "buoyancy"):
            terms.append("+\\frac{\\rho_f V g}{m}")
        if _on(eff, "force"):
            terms.append("+\\frac{F}{m}")
        tex = "a = " + " ".join(terms)
        rels.append(Rel("acc", "Newton's second law: the acceleration is the net force divided by the mass",
                        tex, ["acc"] + fv,
                        lambda v: (v["acc"] + _acc_size(eff, v), _acc(eff, v) + _acc_size(eff, v)), {"acc": tex}))
    if const:
        rels += [
            Rel("v", "Constant acceleration: the velocity changes steadily", "v = v_0 + a\\,t",
                ["t", "v", "acc", "v0"],
                lambda v: (v["v"] + abs(v["v0"]) + abs(v["acc"]) * v["t"] + 1e-9,
                           v["v0"] + v["acc"] * v["t"] + abs(v["v0"]) + abs(v["acc"]) * v["t"] + 1e-9),
                {"v": "v = v_0 + a\\,t", "t": "t = \\frac{v - v_0}{a}", "acc": "a = \\frac{v - v_0}{t}",
                 "v0": "v_0 = v - a\\,t"}, traj=True),
            Rel("x", "Constant acceleration: integrate the velocity", "x = x_0 + v_0 t + \\tfrac12 a t^2",
                ["t", "x"] + p, lambda v: (v["x"] + _sizes(eff, v)[0], at(v, "x") + _sizes(eff, v)[0]),
                {"x": "x = x_0 + v_0 t + \\tfrac12 a t^2", "acc": "a = \\frac{2(x - x_0 - v_0 t)}{t^2}",
                 "x0": "x_0 = x - v_0 t - \\tfrac12 a t^2", "v0": "v_0 = \\frac{x - x_0}{t} - \\tfrac12 a t"},
                traj=True),
            Rel("vx", "Eliminating the time between the two", "v^2 = v_0^2 + 2a\\,(x - x_0)",
                ["v", "v0", "acc", "x", "x0"],
                lambda v: (v["v"] ** 2 + v["v0"] ** 2 + 2 * abs(v["acc"] * (v["x"] - v["x0"])) + 1e-12,
                           v["v0"] ** 2 + 2 * v["acc"] * (v["x"] - v["x0"]) + v["v0"] ** 2
                           + 2 * abs(v["acc"] * (v["x"] - v["x0"])) + 1e-12),
                {"v": "v = \\pm\\sqrt{v_0^2 + 2a\\,(x - x_0)}", "acc": "a = \\frac{v^2 - v_0^2}{2(x - x_0)}",
                 "x": "x = x_0 + \\frac{v^2 - v_0^2}{2a}", "v0": "v_0 = \\pm\\sqrt{v^2 - 2a\\,(x - x_0)}",
                 "x0": "x_0 = x - \\frac{v^2 - v_0^2}{2a}"}),
            Rel("xv", "The displacement is the average velocity times the time", "x = x_0 + \\tfrac12 (v_0 + v)\\,t",
                ["x", "x0", "v0", "v", "t"],
                lambda v: (v["x"] + abs(v["x0"]) + (abs(v["v0"]) + abs(v["v"])) * v["t"] + 1e-9,
                           v["x0"] + (v["v0"] + v["v"]) * v["t"] / 2 + abs(v["x0"])
                           + (abs(v["v0"]) + abs(v["v"])) * v["t"] + 1e-9),
                {"x": "x = x_0 + \\tfrac12 (v_0 + v)\\,t", "t": "t = \\frac{2(x - x_0)}{v_0 + v}",
                 "v": "v = \\frac{2(x - x_0)}{t} - v_0", "v0": "v_0 = \\frac{2(x - x_0)}{t} - v",
                 "x0": "x_0 = x - \\tfrac12 (v_0 + v)\\,t"}),
        ]
    else:
        if drag != "none" and _forced(eff) and not (_on(eff, "spring") or _on(eff, "driving") or _on(eff, "friction")):
            power = "" if drag == "linear" else "^2"
            sym = "b" if drag == "linear" else "c"
            rels.append(Rel(
                "vt", "At terminal speed the drag force balances the net constant force",
                "%s\\,v_t%s = m\\,|a_c|" % (sym, power), ["vt", sym] + [x for x in fv if x != "m"] + ["m"],
                lambda v: (v[sym] * v["vt"] ** (1 if drag == "linear" else 2), v["m"] * abs(_acc(eff, v))),
                {"vt": ("v_t = \\frac{m\\,|a_c|}{b}" if drag == "linear" else "v_t = \\sqrt{\\frac{m\\,|a_c|}{c}}")
                 + ",\\quad a_c = \\text{net constant force}/m",
                 sym: "%s = \\frac{m\\,|a_c|}{v_t%s}" % (sym, power)}))
        if drag == "quadratic":
            rels.append(Rel("c", "The quadratic drag coefficient from the shape and the air",
                            "c = \\tfrac12\\,\\rho\\,C_d\\,A", ["c", "rho", "cd", "area"],
                            lambda v: (v["c"], 0.5 * v["rho"] * v["cd"] * v["area"]),
                            {"c": "c = \\tfrac12\\,\\rho\\,C_d\\,A", "cd": "C_d = \\frac{2c}{\\rho A}",
                             "area": "A = \\frac{2c}{\\rho C_d}", "rho": "\\rho = \\frac{2c}{C_d A}"}))
        rels += [
            Rel("x", "The position at time t from the solution of the equation of motion", "x = x(t)",
                ["t", "x"] + p, lambda v: (v["x"] + _sizes(eff, v)[0], at(v, "x") + _sizes(eff, v)[0]), traj=True),
            Rel("v", "The velocity at time t from the solution of the equation of motion", "v = x'(t)",
                ["t", "v"] + p, lambda v: (v["v"] + _sizes(eff, v)[1], at(v, "v") + _sizes(eff, v)[1]), traj=True),
            Rel("a", "The acceleration at time t is the net force there divided by the mass", "a = x''(t)",
                ["t", "a"] + p,
                lambda v: (v["a"] + _a_size(eff, v), at(v, "a") + _a_size(eff, v)), traj=True),
        ]
    if not _on(eff, "spring") and not _on(eff, "driving"):
        simple = const
        rels += [
            Rel("tg", "It reaches x = 0 when the position is zero", "x(t_g) = 0", ["tg"] + p,
                lambda v: (v["tg"], _zero_time(eff, v)),
                {"tg": "x_0 + v_0 t_g + \\tfrac12 a t_g^2 = 0"} if simple else None, traj=True),
            Rel("vi", "The speed on reaching x = 0", "v_i = |v(t_g)|", ["vi"] + p,
                lambda v: (v["vi"], abs(at(v, "v", _zero_time(eff, v)))),
                {"vi": "v_i = \\sqrt{v_0^2 - 2 a x_0}"} if simple else None, traj=True),
            Rel("tstop", "It stops (or turns round) when the velocity is zero", "v(t_s) = 0", ["tstop"] + p,
                lambda v: (v["tstop"], _stop(eff, v)[0]), {"tstop": "t_s = -\\frac{v_0}{a}"} if simple else None,
                traj=True),
            Rel("xstop", "The position at that instant", "x_s = x(t_s)", ["xstop"] + p,
                lambda v: (v["xstop"] + _sizes(eff, v)[0] + abs(_stop(eff, v)[1]),
                           _stop(eff, v)[1] + _sizes(eff, v)[0] + abs(_stop(eff, v)[1])),
                {"xstop": "x_s = x_0 - \\frac{v_0^2}{2a}"} if simple else None, traj=True),
            Rel("dstop", "The distance covered before it stops", "d_s = |x_s - x_0|", ["dstop", "xstop", "x0"],
                lambda v: (v["dstop"], abs(v["xstop"] - v["x0"])),
                {"dstop": "d_s = |x_s - x_0|"}, traj=True),
        ]
    if _grav(eff) == "surface":
        rels.append(Rel("N", "The surface pushes back with the part of the weight pressing into it",
                        "N = m g\\cos\\alpha", ["N", "m", "g", "inc"],
                        lambda v: (v["N"], v["m"] * v["g"] * math.cos(v["inc"])),
                        {"N": "N = m g\\cos\\alpha", "m": "m = \\frac{N}{g\\cos\\alpha}"}, late=True))
    if _on(eff, "spring"):
        rels += [
            Rel("w0", "The natural angular frequency of a mass on a spring", "\\omega_0 = \\sqrt{k/m}",
                ["w0", "k", "m"], lambda v: (v["w0"], math.sqrt(v["k"] / v["m"])),
                {"w0": "\\omega_0 = \\sqrt{\\frac{k}{m}}", "k": "k = m\\,\\omega_0^2", "m": "m = \\frac{k}{\\omega_0^2}"}),
            Rel("T", "One period is the time for a full cycle", "T = 2\\pi/\\omega_0", ["T", "w0"],
                lambda v: (v["T"], 2 * math.pi / v["w0"]),
                {"T": "T = \\frac{2\\pi}{\\omega_0}", "w0": "\\omega_0 = \\frac{2\\pi}{T}"}),
            Rel("f", "Frequency is cycles per second", "f = 1/T", ["f", "T"], lambda v: (v["f"], 1 / v["T"]),
                {"f": "f = \\frac{1}{T}", "T": "T = \\frac{1}{f}"}),
        ]
        if _forced(eff):
            rels.append(Rel("xeq", "At rest the spring force balances the constant forces",
                            "k\\,x_e = m\\,a_c", ["xeq", "k"] + [x for x in fv if x != "m"] + ["m"],
                            lambda v: (v["k"] * v["xeq"] + v["m"] * _acc_size(eff, v),
                                       v["m"] * _acc(eff, v) + v["m"] * _acc_size(eff, v)),
                            {"xeq": "x_e = \\frac{m\\,a_c}{k},\\quad a_c = \\text{net constant force}/m",
                             "k": "k = \\frac{m\\,a_c}{x_e}"}))
        if drag == "linear":
            rels += [
                Rel("zeta", "The damping ratio compares the damping with the critical value",
                    "\\zeta = \\frac{b}{2\\sqrt{k m}}", ["zeta", "b", "k", "m"],
                    lambda v: (v["zeta"], v["b"] / (2 * math.sqrt(v["k"] * v["m"]))),
                    {"zeta": "\\zeta = \\frac{b}{2\\sqrt{k m}}", "b": "b = 2\\zeta\\sqrt{k m}"}),
                Rel("wd", "Damping lowers the oscillation frequency (underdamped case only)",
                    "\\omega_d = \\omega_0\\sqrt{1 - \\zeta^2}", ["wd", "w0", "zeta"],
                    lambda v: (v["wd"], v["w0"] * math.sqrt(1 - v["zeta"] ** 2)),
                    {"wd": "\\omega_d = \\omega_0\\sqrt{1 - \\zeta^2}"}),
                Rel("tau", "The amplitude decays as e^{-t/\u03c4}", "\\tau = \\frac{2m}{b}", ["tau", "m", "b"],
                    lambda v: (v["tau"], 2 * v["m"] / v["b"]),
                    {"tau": "\\tau = \\frac{2m}{b}", "b": "b = \\frac{2m}{\\tau}", "m": "m = \\frac{b\\,\\tau}{2}"}),
            ]
    if _on(eff, "driving") and drag != "quadratic" and not _on(eff, "friction"):
        damp, spr = drag == "linear", _on(eff, "spring")
        rels.append(Rel(
            "amp", "The amplitude of the steady oscillation that remains once the transient has died away",
            "A_s = \\frac{F_0}{\\sqrt{(k - m\\omega^2)^2 + (b\\,\\omega)^2}}",
            ["amp", "F0", "m", "w"] + (["k"] if spr else []) + (["b"] if damp else []),
            lambda v: (v["amp"], v["F0"] / math.hypot((v["k"] if spr else 0.0) - v["m"] * v["w"] ** 2,
                                                       (v["b"] if damp else 0.0) * v["w"])),
            {"amp": "A_s = \\frac{F_0}{\\sqrt{(k - m\\omega^2)^2 + (b\\,\\omega)^2}}"}))
    rels.append(Rel("ek", "Kinetic energy", "E_k = \\tfrac12 m v^2", ["ek", "m", "v"],
                    lambda v: (v["ek"], 0.5 * v["m"] * v["v"] ** 2),
                    {"ek": "E_k = \\tfrac12 m v^2", "m": "m = \\frac{2E_k}{v^2}", "v": "v = \\pm\\sqrt{\\frac{2E_k}{m}}"}, late=True))
    if not _on(eff, "buoyancy") and not _on(eff, "varying"):
        ev = ["m"] + (["g"] if _grav(eff) != "none" else []) + (["inc"] if _grav(eff) == "surface" else []) \
            + (["k"] if _on(eff, "spring") else [])
        parts = "\\tfrac12 m v^2"
        if _grav(eff) == "vertical":
            parts += " + m g x"
        if _grav(eff) == "surface":
            parts += " + m g x\\sin\\alpha"
        if _on(eff, "spring"):
            parts += " + \\tfrac12 k x^2"

        def esize(v, x, vel):
            return abs(0.5 * v["m"] * vel * vel) + abs(v["m"] * _gpar(eff, v) * x) + 1e-12

        rels += [
            Rel("E", "The mechanical energy at the start: kinetic plus potential", "E_0 = " + parts.replace(
                "v^2", "v_0^2").replace("x", "x_0"), ["E", "x0", "v0"] + ev,
                lambda v: (v["E"] + esize(v, v["x0"], v["v0"]),
                           _energy(eff, v, v["x0"], v["v0"]) + esize(v, v["x0"], v["v0"])),
                {"E": "E_0 = " + parts.replace("v^2", "v_0^2").replace("x", "x_0")}, late=True),
            Rel("em", "The mechanical energy at time t: kinetic plus potential", "E = " + parts,
                ["em", "x", "v"] + ev,
                lambda v: (v["em"] + esize(v, v["x"], v["v"]),
                           _energy(eff, v, v["x"], v["v"]) + esize(v, v["x"], v["v"])), {"em": "E = " + parts}, late=True),
        ]
        if not _on(eff, "force") and not _on(eff, "driving") and (drag != "none" or _on(eff, "friction")):
            rels.append(Rel("elost", "What the mechanical energy has fallen by has gone into heat",
                            "E_{lost} = E_0 - E", ["elost", "E", "em"],
                            lambda v: (v["elost"] + abs(v["E"]) + abs(v["em"]) + 1e-12,
                                       v["E"] - v["em"] + abs(v["E"]) + abs(v["em"]) + 1e-12),
                            {"elost": "E_{lost} = E_0 - E"}, late=True))
    return rels


def _acc_size(eff, v):
    size = abs(_gpar(eff, v)) + 1e-9
    if _on(eff, "force"):
        size += abs(v["F"]) / v["m"]
    if _on(eff, "buoyancy"):
        size += v["rhof"] * v["vol"] * v["g"] / v["m"]
    return 2 * size


def _a_size(eff, v):
    length, speed = _sizes(eff, v)
    m = v["m"] if _needs_mass(eff) else 1.0
    size = abs(_acc(eff, v)) + _friction_acc(eff, v)
    if _on(eff, "spring"):
        size += v["k"] / m * length
    if _drag(eff) == "linear":
        size += v["b"] / m * speed
    if _drag(eff) == "quadratic":
        size += v["c"] / m * speed * speed
    if _on(eff, "driving"):
        size += v["F0"] / m
    return 2 * size + 1e-9


def baseline(eff):
    """The same system without friction, drag, buoyancy or the weakening of gravity."""
    base = dict(eff, drag="none", friction="off", buoyancy="off", varying="off")
    return None if base == eff else base


def equation(eff):
    if _const(eff) and not _forced(eff):
        return "x'' = a,\\qquad x' = v"
    terms = []
    if _grav(eff) == "vertical":
        terms.append("- m g\\left(\\frac{R}{R+x}\\right)^2" if _on(eff, "varying") else "- m g")
    if _grav(eff) == "surface":
        terms.append("- m g\\sin\\alpha")
    if _on(eff, "buoyancy"):
        terms.append("+ \\rho_f V g")
    if _on(eff, "force"):
        terms.append("+ F")
    if _on(eff, "friction"):
        terms.append("- \\mu N\\,\\mathrm{sgn}(v)")
    if _drag(eff) == "linear":
        terms.append("- b\\,v")
    if _drag(eff) == "quadratic":
        terms.append("- c\\,v\\,|v|")
    if _on(eff, "spring"):
        terms.append("- k\\,x")
    if _on(eff, "driving"):
        terms.append("+ F_0\\cos(\\omega t)")
    text = " ".join(terms) if terms else "0"
    if text.startswith("+ "):
        text = text[2:]
    if text.startswith("- "):
        text = "-" + text[2:]
    return "m\\,\\frac{dv}{dt} = %s,\\qquad \\frac{dx}{dt} = v" % text


def derivation(eff):
    if _const(eff) and not _forced(eff):
        return [step("Set up the equation of motion",
                     "No force is switched on, so the acceleration \\(a\\) is a constant that is entered or found.",
                     [equation(eff)]),
                step("Solve the equation", "Integrate twice, using the initial velocity and position.",
                     ["v(t) = v_0 + a\\,t", "x(t) = x_0 + v_0 t + \\tfrac12 a t^2"])]
    parts = []
    if _grav(eff) == "vertical":
        parts.append("The weight weakens with height as the inverse square of the distance from the planet's "
                     "centre." if _on(eff, "varying") else "\\(-mg\\) is the weight; x is measured upward.")
    if _grav(eff) == "surface":
        parts.append("\\(-mg\\sin\\alpha\\) is the part of the weight along the slope (x points up it); the "
                     "surface pushes back with \\(N = mg\\cos\\alpha\\).")
    if _on(eff, "buoyancy"):
        parts.append("\\(\\rho_f V g\\) is the buoyant force, the weight of the fluid displaced.")
    if _on(eff, "force"):
        parts.append("\\(F\\) is the constant applied force.")
    if _on(eff, "friction"):
        parts.append("\\(\\mu N\\) is sliding friction, always against the motion; once the object stops, "
                     "friction holds it there if the other forces are smaller than \\(\\mu N\\).")
    if _drag(eff) == "linear":
        parts.append("\\(-bv\\) is resistance proportional to speed.")
    if _drag(eff) == "quadratic":
        parts.append("\\(-cv|v|\\) is air resistance proportional to speed squared.")
    if _on(eff, "spring"):
        parts.append("\\(-kx\\) is the spring force (Hooke's law), pulling back towards x = 0.")
    if _on(eff, "driving"):
        parts.append("\\(F_0\\cos(\\omega t)\\) is the periodic driving force.")
    steps = [step("Set up the equation of motion", "Newton's second law along the line. " + " ".join(parts),
                  [equation(eff)])]
    drag, spring = _drag(eff), _on(eff, "spring")
    if _const(eff):
        steps.append(step("Solve the equation",
                          "Every force is constant, so the acceleration \\(a\\) is constant. Integrate twice.",
                          ["v(t) = v_0 + a\\,t", "x(t) = x_0 + v_0 t + \\tfrac12 a t^2"]))
    elif _on(eff, "friction") and drag == "none" and not spring and not _on(eff, "driving"):
        steps.append(step("Solve the equation",
                          "While it slides the net force is constant, so each stretch of the motion has "
                          "constant acceleration \\(a = a_c \\mp \\mu g\\cos\\alpha\\) (friction against the "
                          "velocity). When the velocity reaches zero the object stays put if "
                          "\\(|a_c| \\le \\mu g\\cos\\alpha\\); otherwise it sets off the other way.",
                          ["v(t) = v_0 + a\\,t", "x(t) = x_0 + v_0 t + \\tfrac12 a t^2"]))
    elif spring and drag != "quadratic" and not _on(eff, "friction") and not _on(eff, "varying"):
        lines = ["m r^2 + b r + k = 0" if drag == "linear" else "m r^2 + k = 0"]
        text = "The equation is linear with constant coefficients. Its characteristic equation gives the free motion: "
        if drag == "linear":
            text += ("oscillation inside a decaying envelope when \\(\\zeta < 1\\), the fastest return without "
                     "overshoot when \\(\\zeta = 1\\), and a slow return when \\(\\zeta > 1\\).")
            lines.append("x_h = e^{-t/\\tau}\\left(C_1\\cos\\omega_d t + C_2\\sin\\omega_d t\\right)\\quad(\\zeta < 1)")
        else:
            text += "a steady oscillation at \\(\\omega_0 = \\sqrt{k/m}\\)."
            lines.append("x_h = C_1\\cos\\omega_0 t + C_2\\sin\\omega_0 t")
        if _on(eff, "driving") or _forced(eff):
            text += (" A particular solution is added for the constant and periodic forces (undetermined "
                     "coefficients), and the constants are fixed by the initial position and velocity.")
        else:
            text += " The constants are fixed by the initial position and velocity."
        steps.append(step("Solve the equation", text, lines))
    elif (not spring and not _on(eff, "driving") and not _on(eff, "friction") and not _on(eff, "varying")
          and drag == "linear" and _forced(eff)):
        steps.append(step("Solve the equation",
                          "The equation is linear in \\(v\\). With the terminal speed \\(v_t = m|a_c|/b\\) the "
                          "velocity relaxes exponentially towards it (integrating factor); integrating once "
                          "more gives the position. Here for a net force in the -x direction:",
                          ["v(t) = -v_t + (v_0 + v_t)\\,e^{-|a_c| t/v_t}",
                           "x(t) = x_0 - v_t t + \\frac{v_t (v_0 + v_t)}{|a_c|}\\left(1 - e^{-|a_c| t/v_t}\\right)"]))
    elif (not spring and not _on(eff, "driving") and not _on(eff, "friction") and not _on(eff, "varying")
          and drag == "quadratic" and _forced(eff)):
        steps.append(step("Solve the equation",
                          "While the object moves in the direction of the net force the equation is separable. "
                          "With \\(v_t = \\sqrt{m|a_c|/c}\\) and speed \\(u\\) in that direction, "
                          "\\(u' = |a_c|(1 - u^2/v_t^2)\\). If it starts by moving the other way, the drag "
                          "changes direction at the turning point and the motion is integrated numerically.",
                          ["u(t) = v_t \\tanh\\!\\left(\\frac{|a_c| t}{v_t} + \\tanh^{-1}\\frac{u_0}{v_t}\\right)",
                           "x(t) = x_0 \\mp \\frac{v_t^2}{|a_c|}\\ln\\frac{\\cosh(|a_c| t/v_t + s)}{\\cosh s},\\quad "
                           "s = \\tanh^{-1}\\frac{u_0}{v_t}"]))
    else:
        steps.append(step("Solve the equation",
                          "With these forces together the equation has no elementary solution, so it is "
                          "integrated numerically (fourth-order Runge-Kutta)."))
    return steps


def horizon(eff, v):
    try:
        if _const(eff):
            a, x0, v0 = v.get("acc"), v.get("x0"), v.get("v0")
            if a is None or x0 is None or v0 is None:
                return None
            h = 2 * _kin_scale(a, x0, v0)
            if a and v.get("x") is not None:
                h = max(h, 3 * (abs(v0) / abs(a) + math.sqrt(2 * abs(v["x"] - x0) / abs(a))))
            if a and v.get("v") is not None:
                h = max(h, 1.5 * abs(v["v"] - v0) / abs(a))
            return h
        return trajectory(eff, v)["scale"] * 2
    except InputError:
        raise
    except (Skip, ValueError, ZeroDivisionError, TypeError, KeyError):
        if _on(eff, "spring") and v.get("m") and v.get("k"):
            return 8 * 2 * math.pi / math.sqrt(v["k"] / v["m"])
        return None


def graph(eff, v):
    traj = trajectory(eff, v)
    free = not _on(eff, "spring") and not _on(eff, "driving")
    t_end = None
    if free and v["x0"] != 0:
        try:
            t_end = _zero_time(eff, v)
        except Skip:
            t_end = None
    t_stop = None
    if free and t_end is None:
        try:
            t_stop = _stop(eff, v)[0]
        except Skip:
            t_stop = None
    t_max = traj["scale"]
    if t_end:
        t_max = t_end * 1.02
    elif t_stop:
        t_max = t_stop * 1.5
    t_max = max(t_max, (v.get("t") or 0.0) * 1.2)

    def clip(fn):
        if not t_end:
            return fn
        return lambda t: 0.0 if t > t_end else fn(t)

    ideal = None
    base = baseline(eff)
    if base is not None:
        try:
            v2 = dict(v)
            if _const(base):
                v2["acc"] = _acc(base, v)
            other = trajectory(base, v2)
            ideal = clip(lambda t: other["at"](t)["x"])
            if t_end:
                inner = ideal
                ideal = lambda t: max(inner(t), 0.0) if v["x0"] > 0 else min(inner(t), 0.0)
        except (Skip, KeyError, TypeError, ValueError, ZeroDivisionError):
            ideal = None
    marker = None
    if v.get("t") is not None and v.get("x") is not None:
        marker = {"t": v["t"], "y": v["x"], "label": "at time t"}
    elif t_end:
        marker = {"t": t_end, "y": 0.0, "label": "reaches x = 0"}
    elif t_stop:
        marker = {"t": t_stop, "y": traj["at"](t_stop)["x"], "label": "stops"}
    kind = "spring" if _on(eff, "spring") else "fall"
    return {"fn": clip(lambda t: traj["at"](t)["x"]), "ideal": ideal, "t_max": t_max,
            "y_label": "height (m)" if _grav(eff) == "vertical" else "position (m)", "label": "Position",
            "ideal_label": "Without friction, drag or buoyancy", "marker": marker, "animation": kind}
