"""Rotation about a fixed axis, including pendulums.

    I theta'' = -m g d sin(theta) - kappa theta - I gamma theta' + tau_0

Gravity (a point mass on a rod, or a rigid body), a torsion spring, damping
and a constant applied torque are switches. With none of them on, the system
is rotation with constant angular acceleration.
"""
import math

from ..first_order import step
from .common import numeric, oscillator
from .solver import Rel, Skip
from .spec import choice, field, preset, switch

_FREE = [("gravity", ["none"]), ("torsion", ["off"]), ("damping", ["off"])]
_PEND = ("gravity", ["point", "body"])

MODEL = {
    "id": "rotation", "title": "Rotation and pendulums", "animation": "spring",
    "description": "Turning about a fixed axis: I\u03b8'' = sum of torques. Pendulums, torsion springs, flywheels.",
    "law": "Newton's second law for rotation: moment of inertia times angular acceleration equals the net torque.",
    "fields": [
        field("L", "Length of the rod", "L", "length", (None, "m"), positive=True, show=[("gravity", ["point"])]),
        field("m", "Mass", "m", "mass", (None, "kg"), positive=True, show=[_PEND]),
        field("I", "Moment of inertia about the axis", "I", "moment_of_inertia", (None, "kg.m2"), positive=True,
              show=[("gravity", ["none", "body"])]),
        field("d", "Distance from the pivot to the centre of mass", "d", "length", (None, "m"), positive=True,
              show=[("gravity", ["body"])]),
        field("g", "Gravity", "g", "acceleration", (None, "m/s2"), positive=True, typical=10, show=[_PEND],
              hint="9.81 m/s\u00b2 at the Earth's surface."),
        field("alpha", "Angular acceleration (constant)", "\\alpha", "angular_acceleration", (None, "rad/s2"),
              show=_FREE),
        field("kappa", "Torsion constant", "\\kappa", "torsion_constant", (None, "N.m/rad"), "effect", positive=True),
        field("gam", "Damping rate", "\\gamma", "rate", (None, "1/s"), "effect", positive=True, typical=0.1,
              hint="The damping torque is I\u00b7\u03b3\u00b7\u03b8'."),
        field("tq", "Applied torque", "\\tau_0", "torque", (None, "N.m"), "effect"),
        field("th0", "Initial angle", "\\theta_0", "angle", (None, "deg"), "initial"),
        field("om0", "Initial angular velocity", "\\omega_0", "angular_frequency", (None, "rad/s"), "initial"),
        field("t", "Time", "t", "time", (None, "s"), "moment", nonnegative=True),
        field("th", "Angle at that time", "\\theta", "angle", (None, "deg"), "moment"),
        field("om", "Angular velocity at that time", "\\omega", "angular_frequency", (None, "rad/s"), "moment"),
        field("al", "Angular acceleration at that time", "\\alpha", "angular_acceleration", (None, "rad/s2"),
              "moment", hide=_FREE),
        field("revs", "Revolutions turned since the start", "N", "plain", (None, ""), "moment"),
        field("ek", "Rotational kinetic energy at that time", "E_k", "energy", (None, "J"), "moment",
              nonnegative=True, optional=True),
        field("T0", "Period of small oscillations", "T_0", "time", (None, "s"), "derived", positive=True),
        field("f0", "Frequency of small oscillations", "f_0", "frequency", (None, "Hz"), "derived", positive=True),
        field("Tex", "Exact period of the swing", "T", "time", (None, "s"), "effect", positive=True),
        field("vmax", "Speed at the bottom of the swing", "v_{max}", "speed", (None, "m/s"), "derived",
              nonnegative=True, show=[("gravity", ["point"])]),
    ],
    "effects": [
        choice("gravity", "Gravity", "A weight hanging from the pivot pulls the system back to the vertical.",
               [("none", "None", []), ("point", "Point mass on a rod", []), ("body", "Rigid body", [])], "point"),
        switch("large", "Large angles", "Use sin \u03b8 exactly instead of the small-angle approximation.", ["Tex"]),
        switch("torsion", "Torsion spring", "A restoring torque proportional to the angle.", ["kappa"]),
        switch("damping", "Damping", "A torque opposing the rotation, proportional to the angular velocity.", ["gam"]),
        switch("torque", "Applied torque", "A constant torque about the axis.", ["tq"]),
    ],
    "presets": [
        preset("clock", "Grandfather clock", "A pendulum that swings once every two seconds: how long is it?",
               {"g": ("9.81", "m/s2"), "T0": ("2", "s"), "th0": ("4", "deg"), "om0": ("0", "rad/s")}, {"gravity": "point"}),
        preset("swing", "Playground swing", "A 2.5 m swing released from 60 degrees.",
               {"L": ("2.5", "m"), "g": ("9.81", "m/s2"), "th0": ("60", "deg"), "om0": ("0", "rad/s")},
               {"gravity": "point", "large": "on"}),
        preset("foucault", "Foucault pendulum", "A 67 m pendulum released from 3 degrees.",
               {"L": ("67", "m"), "g": ("9.81", "m/s2"), "th0": ("3", "deg"), "om0": ("0", "rad/s")}, {"gravity": "point"}),
        preset("lab", "Damped lab pendulum", "A 0.5 m pendulum with damping rate 0.1 per second, after 10 s.",
               {"L": ("0.5", "m"), "g": ("9.81", "m/s2"), "th0": ("10", "deg"), "om0": ("0", "rad/s"),
                "gam": ("0.1", "1/s"), "t": ("10", "s")}, {"gravity": "point", "damping": "on"}),
        preset("gravity", "Measuring gravity", "A 1 m pendulum has a period of 2.007 s: what is g?",
               {"L": ("1", "m"), "T0": ("2.007", "s"), "th0": ("5", "deg"), "om0": ("0", "rad/s")}, {"gravity": "point"}),
        preset("stick", "Swinging metre stick", "A 150 g metre stick pivoted at one end (I = mL\u00b2/3).",
               {"m": ("150", "g"), "I": ("0.05", "kg.m2"), "d": ("0.5", "m"), "g": ("9.81", "m/s2"),
                "th0": ("5", "deg"), "om0": ("0", "rad/s")}, {"gravity": "body"}),
        preset("flywheel", "Flywheel spinning up", "A 0.5 kg\u00b7m\u00b2 flywheel driven by 2 N\u00b7m from rest for 10 s.",
               {"I": ("0.5", "kg.m2"), "tq": ("2", "N.m"), "th0": ("0", "deg"), "om0": ("0", "rad/s"), "t": ("10", "s"),
                "om": ("", "rpm")}, {"gravity": "none", "torque": "on"}),
        preset("coast", "Wheel coasting down", "A wheel at 300 rpm slows with damping rate 0.2 per second: its speed after 5 s?",
               {"om0": ("300", "rpm"), "th0": ("0", "deg"), "gam": ("0.2", "1/s"), "t": ("5", "s"), "om": ("", "rpm")},
               {"gravity": "none", "damping": "on"}),
        preset("torsion", "Torsion pendulum", "A disc of 0.002 kg\u00b7m\u00b2 on a wire twists with a 1.5 s period: the wire's torsion constant?",
               {"I": ("0.002", "kg.m2"), "T0": ("1.5", "s"), "th0": ("20", "deg"), "om0": ("0", "rad/s"),
                "kappa": ("", "mN.m/rad")}, {"gravity": "none", "torsion": "on"}),
        preset("brake", "Constant angular acceleration", "A disc slows from 120 rad/s to rest in 50 revolutions.",
               {"om0": ("120", "rad/s"), "th0": ("0", "deg"), "om": ("0", "rad/s"), "revs": ("50", "")},
               {"gravity": "none"}),
    ],
}


def _on(eff, key):
    return eff.get(key) == "on"


def _grav(eff):
    return eff.get("gravity", "point")


def _free(eff):
    """Constant angular acceleration: no gravity, torsion spring or damping."""
    return _grav(eff) == "none" and not _on(eff, "torsion") and not _on(eff, "damping")


def _ivars(eff):
    return ["m", "L"] if _grav(eff) == "point" else ["I"]


def _inertia(eff, v):
    return v["m"] * v["L"] ** 2 if _grav(eff) == "point" else v["I"]


def _avars(eff):
    return {"point": ["g", "L"], "body": ["g", "m", "d", "I"]}.get(_grav(eff), [])


def _a(eff, v):
    """Coefficient of the gravity term in theta'' (rad/s^2 per unit of sin theta)."""
    if _grav(eff) == "point":
        return v["g"] / v["L"]
    if _grav(eff) == "body":
        return v["m"] * v["g"] * v["d"] / v["I"]
    return 0.0


def _kvars(eff):
    return ["kappa"] + _ivars(eff) if _on(eff, "torsion") else []


def _stiff_vars(eff):
    out = []
    for x in _avars(eff) + _kvars(eff):
        if x not in out:
            out.append(x)
    return out


def _stiff(eff, v):
    """A + K: the square of the small-oscillation angular frequency."""
    return _a(eff, v) + (v["kappa"] / _inertia(eff, v) if _on(eff, "torsion") else 0.0)


def _params(eff):
    if _free(eff):
        return ["alpha", "th0", "om0"]
    out = ["th0", "om0"] + _stiff_vars(eff)
    if _on(eff, "damping"):
        out.append("gam")
    if _on(eff, "torque"):
        out += ["tq"] + [x for x in _ivars(eff) if x not in out]
    return out


def _accel(eff, v, th, om):
    large = _on(eff, "large") and _grav(eff) != "none"
    acc = -_a(eff, v) * (math.sin(th) if large else th)
    if _on(eff, "torsion"):
        acc -= v["kappa"] / _inertia(eff, v) * th
    if _on(eff, "damping"):
        acc -= v["gam"] * om
    if _on(eff, "torque"):
        acc += v["tq"] / _inertia(eff, v)
    return acc


def trajectory(eff, v):
    if any(v.get(x) is None for x in _params(eff)):
        raise Skip()
    th0, om0 = v["th0"], v["om0"]
    if _free(eff):
        al = v["alpha"]
        scale = max(abs(om0) / abs(al) if al else 0.0, 1.0)
        return {"at": lambda t: {"th": th0 + om0 * t + al * t * t / 2, "om": om0 + al * t}, "exact": True,
                "scale": scale, "ode": (lambda t, s: [s[1], al], [th0, om0], ["th", "om"], scale)}
    stiff = _stiff(eff, v)
    gam = v["gam"] if _on(eff, "damping") else 0.0
    drive = v["tq"] / _inertia(eff, v) if _on(eff, "torque") else 0.0
    large = _on(eff, "large") and _grav(eff) != "none"
    deriv = lambda t, s: [s[1], _accel(eff, v, s[0], s[1])]
    if stiff > 0:
        scale = 2 * math.pi / math.sqrt(stiff)
    else:
        scale = 1 / gam if gam else 1.0
    if not large:
        if stiff > 0:
            osc = oscillator(1.0, gam, stiff, th0, om0, f_const=drive)

            def at(t):
                th, om = osc(t)
                return {"th": th, "om": om}
        else:
            # theta'' + gamma theta' = drive: the angular velocity relaxes to drive/gamma.
            wf = drive / gam

            def at(t):
                e = math.exp(-gam * t)
                return {"th": th0 + wf * t + (om0 - wf) * (1 - e) / gam, "om": wf + (om0 - wf) * e}
        return {"at": at, "exact": True, "scale": scale, "ode": (deriv, [th0, om0], ["th", "om"], 2 * scale)}
    key = ("rotation", tuple(sorted(eff.items()))) + tuple(v[x] for x in _params(eff))
    num = numeric(key, deriv, [th0, om0], 3 * scale)

    def at(t):
        th, om = num.state(t)
        return {"th": th, "om": om}
    return {"at": at, "exact": False, "scale": scale, "ode": None}


def _agm_k(k):
    """Complete elliptic integral of the first kind, K(k), by the arithmetic-geometric mean."""
    a, b = 1.0, math.sqrt(1 - k * k)
    for _ in range(40):
        a, b = (a + b) / 2, math.sqrt(a * b)
    return math.pi / (2 * a)


def _true_period(eff, v):
    a = _a(eff, v)
    c = math.cos(v["th0"]) - v["om0"] ** 2 / (2 * a)
    if c <= -1 or c >= 1:
        raise Skip()          # going right round, or not moving
    amplitude = math.acos(c)
    return 4 / math.sqrt(a) * _agm_k(math.sin(amplitude / 2))


def _size(eff, v):
    """An angle scale, so that values near zero compare against the size of the motion."""
    stiff = 0.0
    try:
        stiff = _stiff(eff, v) if not _free(eff) else 0.0
    except (KeyError, TypeError):
        stiff = 0.0
    rate = math.sqrt(stiff) if stiff > 0 else 1.0
    return abs(v["th0"]) + abs(v["om0"]) / rate + 1e-3


def relations(eff):
    grav, free = _grav(eff), _free(eff)
    large = _on(eff, "large") and grav != "none"
    plain = not _on(eff, "torsion") and not _on(eff, "damping") and not _on(eff, "torque")
    iv = _ivars(eff)
    isym = "m L^2" if grav == "point" else "I"
    rels = []
    if free and _on(eff, "torque"):
        rels.append(Rel("alpha", "The applied torque sets the angular acceleration", "%s\\,\\alpha = \\tau_0" % isym,
                        ["alpha", "tq"] + iv, lambda v: (_inertia(eff, v) * v["alpha"] + 1e-9, v["tq"] + 1e-9),
                        {"alpha": "\\alpha = \\frac{\\tau_0}{%s}" % isym, "tq": "\\tau_0 = %s\\,\\alpha" % isym}))
    if free:
        rels.append(Rel("kin", "With constant angular acceleration, the speed and the angle are linked without the time",
                        "\\omega^2 = \\omega_0^2 + 2\\alpha(\\theta - \\theta_0)", ["om", "om0", "alpha", "th", "th0"],
                        lambda v: (v["om"] ** 2 + v["om0"] ** 2 + 1e-9,
                                   2 * v["om0"] ** 2 + 2 * v["alpha"] * (v["th"] - v["th0"]) + 1e-9),
                        {"alpha": "\\alpha = \\frac{\\omega^2 - \\omega_0^2}{2(\\theta - \\theta_0)}",
                         "th": "\\theta = \\theta_0 + \\frac{\\omega^2 - \\omega_0^2}{2\\alpha}"}))
    sv = _stiff_vars(eff)
    if sv:
        if grav == "point" and not _on(eff, "torsion"):
            tex, forms = "T_0 = 2\\pi\\sqrt{L/g}", {
                "T0": "T_0 = 2\\pi\\sqrt{\\frac{L}{g}}", "L": "L = g\\left(\\frac{T_0}{2\\pi}\\right)^2",
                "g": "g = L\\left(\\frac{2\\pi}{T_0}\\right)^2"}
        elif grav == "body" and not _on(eff, "torsion"):
            tex, forms = "T_0 = 2\\pi\\sqrt{\\frac{I}{m g d}}", {
                "T0": "T_0 = 2\\pi\\sqrt{\\frac{I}{m g d}}", "I": "I = m g d\\left(\\frac{T_0}{2\\pi}\\right)^2"}
        elif grav == "none":
            tex, forms = "T_0 = 2\\pi\\sqrt{I/\\kappa}", {
                "T0": "T_0 = 2\\pi\\sqrt{\\frac{I}{\\kappa}}", "kappa": "\\kappa = I\\left(\\frac{2\\pi}{T_0}\\right)^2",
                "I": "I = \\kappa\\left(\\frac{T_0}{2\\pi}\\right)^2"}
        else:
            tex, forms = "T_0 = \\frac{2\\pi}{\\sqrt{m g d/I + \\kappa/I}}", None
        rels.append(Rel("T0", "Small oscillations are simple harmonic; their period depends only on the system",
                        tex, ["T0"] + sv, lambda v: (v["T0"], 2 * math.pi / math.sqrt(_stiff(eff, v))), forms))
        rels.append(Rel("f0", "Frequency is oscillations per second", "f_0 = 1/T_0", ["f0", "T0"],
                        lambda v: (v["f0"], 1 / v["T0"]), {"f0": "f_0 = \\frac{1}{T_0}", "T0": "T_0 = \\frac{1}{f_0}"}))
    if grav == "point" and plain:
        rels.append(Rel(
            "vmax", "Energy conservation between the release point and the bottom of the swing",
            "\\tfrac12 v_{max}^2 = g L (1 - \\cos\\theta_0) + \\tfrac12 (L\\omega_0)^2", ["vmax", "g", "L", "th0", "om0"],
            lambda v: (v["vmax"], math.sqrt(
                (2 * v["g"] * v["L"] * (1 - math.cos(v["th0"])) if large else v["g"] * v["L"] * v["th0"] ** 2)
                + (v["L"] * v["om0"]) ** 2)),
            {"vmax": "v_{max} = \\sqrt{2 g L (1 - \\cos\\theta_0) + (L\\omega_0)^2}" if large
             else "v_{max} = \\sqrt{g L\\,\\theta_0^2 + (L\\omega_0)^2}"}))
    if large and plain:
        root = "\\sqrt{\\frac{L}{g}}" if grav == "point" else "\\sqrt{\\frac{I}{m g d}}"
        rels.append(Rel("Tex", "The exact period of a large swing is longer than the small-swing value",
                        "T = 4%s\\;K\\!\\left(\\sin\\frac{\\theta_{max}}{2}\\right)" % root,
                        ["Tex", "th0", "om0"] + _avars(eff), lambda v: (v["Tex"], _true_period(eff, v)),
                        {"Tex": "T = 4%s\\;K\\!\\left(\\sin\\frac{\\theta_{max}}{2}\\right)" % root}))
    p = _params(eff)
    rate = lambda v: (math.sqrt(_stiff(eff, v)) if (not free and _stiff(eff, v) > 0) else 1.0)
    simple_th = {"th": "\\theta = \\theta_0 + \\omega_0 t + \\tfrac12\\alpha t^2"} if free else None
    simple_om = {"om": "\\omega = \\omega_0 + \\alpha t", "t": "t = \\frac{\\omega - \\omega_0}{\\alpha}",
                 "alpha": "\\alpha = \\frac{\\omega - \\omega_0}{t}", "om0": "\\omega_0 = \\omega - \\alpha t"} if free else None
    rels += [
        Rel("th", "The angle at time t from the solution of the equation of motion", "\\theta = \\theta(t)",
            ["t", "th"] + p,
            lambda v: (v["th"] + 2 * _span(eff, v), trajectory(eff, v)["at"](v["t"])["th"] + 2 * _span(eff, v)),
            simple_th, traj=True),
        Rel("om", "The angular velocity at time t from the solution of the equation of motion", "\\omega = \\theta'(t)",
            ["t", "om"] + p,
            lambda v: (v["om"] + 2 * _span(eff, v) * rate(v) + abs(v["om0"]),
                       trajectory(eff, v)["at"](v["t"])["om"] + 2 * _span(eff, v) * rate(v) + abs(v["om0"])),
            simple_om, traj=True),
        Rel("revs", "One revolution is 2\u03c0 radians", "N = \\frac{\\theta - \\theta_0}{2\\pi}", ["revs", "th", "th0"],
            lambda v: (v["revs"] + 1.0 + abs(v["th0"]), (v["th"] - v["th0"]) / (2 * math.pi) + 1.0 + abs(v["th0"])),
            {"revs": "N = \\frac{\\theta - \\theta_0}{2\\pi}", "th": "\\theta = \\theta_0 + 2\\pi N"}),
        Rel("ek", "Rotational kinetic energy", "E_k = \\tfrac12 %s\\,\\omega^2" % isym, ["ek", "om"] + iv,
            lambda v: (v["ek"], 0.5 * _inertia(eff, v) * v["om"] ** 2),
            {"ek": "E_k = \\tfrac12 %s\\,\\omega^2" % isym}, late=True),
    ]
    if not free:
        accel_vars = [x for x in p if x not in ("th0", "om0")]
        rels.append(Rel("al", "The angular acceleration is the net torque divided by the moment of inertia",
                        "\\alpha = \\theta''", ["al", "th", "om"] + accel_vars,
                        lambda v: (v["al"] + _al_size(eff, v), _accel(eff, v, v["th"], v["om"]) + _al_size(eff, v)),
                        None))
    return rels


def _al_size(eff, v):
    """Twice the sum of the sizes of the terms in the angular acceleration."""
    size = _stiff(eff, v) * (abs(v["th"]) + 1.0)
    if _on(eff, "damping"):
        size += v["gam"] * abs(v["om"])
    if _on(eff, "torque"):
        size += abs(v["tq"]) / _inertia(eff, v)
    return 2 * size + 1e-6


def _span(eff, v):
    size = _size(eff, v)
    if _free(eff):
        t = v.get("t") or 0.0
        size += abs(v["om0"]) * t + abs(v.get("alpha") or 0.0) * t * t / 2
    elif _on(eff, "torque"):
        try:
            stiff = _stiff(eff, v)
            if stiff > 0:
                size += abs(v["tq"]) / _inertia(eff, v) / stiff
            else:
                size += abs(v["tq"]) / _inertia(eff, v) * (v.get("t") or 1.0) ** 2
        except (KeyError, TypeError, ZeroDivisionError):
            pass
    if _grav(eff) == "none" and _on(eff, "damping") and not _on(eff, "torsion"):
        size += abs(v["om0"]) / (v.get("gam") or 1.0)
    return size


def baseline(eff):
    """The same system without damping."""
    base = dict(eff, damping="off")
    return None if base == eff else base


def equation(eff):
    grav = _grav(eff)
    if _free(eff):
        return "I\\,\\theta'' = \\tau_0" if _on(eff, "torque") else "\\theta'' = \\alpha\\ \\text{(constant)}"
    large = _on(eff, "large") and grav != "none"
    ang = "\\sin\\theta" if large else "\\theta"
    if grav == "point":
        lhs, terms = "m L^2\\,\\theta''", ["-m g L\\,%s" % ang]
    elif grav == "body":
        lhs, terms = "I\\,\\theta''", ["-m g d\\,%s" % ang]
    else:
        lhs, terms = "I\\,\\theta''", []
    isym = "m L^2" if grav == "point" else "I"
    if _on(eff, "torsion"):
        terms.append("- \\kappa\\,\\theta")
    if _on(eff, "damping"):
        terms.append("- %s\\,\\gamma\\,\\theta'" % isym)
    if _on(eff, "torque"):
        terms.append("+ \\tau_0")
    rhs = " ".join(terms) if terms else "0"
    if rhs.startswith("- "):
        rhs = "-" + rhs[2:]
    if rhs.startswith("+ "):
        rhs = rhs[2:]
    return "%s = %s" % (lhs, rhs)


def derivation(eff):
    grav = _grav(eff)
    if _free(eff):
        text = ("Nothing but " + ("a constant torque" if _on(eff, "torque") else "a constant angular acceleration")
                + " acts, so the angular acceleration is constant.")
        return [step("Set up the equation of motion", text, [equation(eff)]),
                step("Solve the equation", "Integrate twice.",
                     ["\\omega(t) = \\omega_0 + \\alpha t", "\\theta(t) = \\theta_0 + \\omega_0 t + \\tfrac12\\alpha t^2"])]
    large = _on(eff, "large") and grav != "none"
    parts = []
    if grav == "point":
        parts.append("The weight \\(mg\\) acts at distance \\(L\\) from the pivot, giving the restoring torque "
                     "\\(-mgL\\sin\\theta\\)" + ("." if large else ", with \\(\\sin\\theta \\approx \\theta\\) for small angles."))
    if grav == "body":
        parts.append("The weight acts at the centre of mass, a distance \\(d\\) from the pivot, giving the "
                     "restoring torque \\(-mgd\\sin\\theta\\)" + ("." if large else ", with \\(\\sin\\theta \\approx \\theta\\) for small angles."))
    if _on(eff, "torsion"):
        parts.append("\\(-\\kappa\\theta\\) is the torsion spring, twisting back towards \\(\\theta = 0\\).")
    if _on(eff, "damping"):
        parts.append("The damping torque opposes the rotation in proportion to the angular velocity.")
    if _on(eff, "torque"):
        parts.append("\\(\\tau_0\\) is the constant applied torque.")
    steps = [step("Set up the equation of motion", "Newton's second law for rotation about the axis. " + " ".join(parts),
                  [equation(eff)])]
    if large:
        steps.append(step("Solve the equation",
                          "With \\(\\sin\\theta\\) kept exactly the equation is nonlinear and has no elementary "
                          "solution, so it is integrated numerically (fourth-order Runge-Kutta)."))
    elif grav == "none" and not _on(eff, "torsion"):
        steps.append(step("Solve the equation",
                          "The equation is first-order linear in the angular velocity, which relaxes "
                          "exponentially to \\(\\omega_f = \\tau_0/(I\\gamma)\\); integrating once more gives the angle.",
                          ["\\omega(t) = \\omega_f + (\\omega_0 - \\omega_f)\\,e^{-\\gamma t}",
                           "\\theta(t) = \\theta_0 + \\omega_f t + \\frac{\\omega_0 - \\omega_f}{\\gamma}\\left(1 - e^{-\\gamma t}\\right)"]))
    else:
        lines = ["\\theta'' + \\gamma\\,\\theta' + \\Omega^2\\theta = \\text{const}" if _on(eff, "damping")
                 else "\\theta'' + \\Omega^2\\theta = \\text{const}",
                 "\\Omega = \\frac{2\\pi}{T_0}"]
        if _on(eff, "damping"):
            lines.append("\\theta(t) = \\theta_e + e^{-\\gamma t/2}\\left(A\\cos\\omega_d t + B\\sin\\omega_d t\\right),"
                         "\\quad \\omega_d = \\sqrt{\\Omega^2 - \\gamma^2/4}")
        else:
            lines.append("\\theta(t) = \\theta_e + (\\theta_0 - \\theta_e)\\cos\\Omega t + \\frac{\\omega_0}{\\Omega}\\sin\\Omega t")
        steps.append(step("Solve the equation",
                          "Dividing by the moment of inertia gives a second-order linear equation with constant "
                          "coefficients. Its characteristic equation gives the oscillation; \\(\\theta_e\\) is the "
                          "angle at which the torques balance, and the constants come from the initial angle and "
                          "angular velocity.", lines))
    return steps


def horizon(eff, v):
    try:
        return trajectory(eff, v)["scale"] * 3
    except (Skip, ValueError, ZeroDivisionError, TypeError, KeyError):
        return None


def graph(eff, v):
    traj = trajectory(eff, v)
    t_max = max(traj["scale"] * (3 if not _free(eff) else 1.5), (v.get("t") or 0.0) * 1.2)
    ideal = None
    if _on(eff, "damping") and not _free(eff):
        plain = dict(eff, damping="off")
        try:
            base = trajectory(plain, v)
            ideal = lambda t: math.degrees(base["at"](t)["th"])
        except (Skip, ZeroDivisionError):
            ideal = None
    marker = None
    if v.get("t") is not None and v.get("th") is not None:
        marker = {"t": v["t"], "y": math.degrees(v["th"]), "label": "at time t"}
    return {"fn": lambda t: math.degrees(traj["at"](t)["th"]), "ideal": ideal, "t_max": t_max,
            "y_label": "angle (\u00b0)", "label": "Angle", "ideal_label": "Without damping", "marker": marker}
