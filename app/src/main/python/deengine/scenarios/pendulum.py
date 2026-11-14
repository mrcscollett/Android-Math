"""A pendulum, small swings or large, with optional damping."""
import math

from ..first_order import step
from .common import numeric, oscillator
from .solver import Rel, Skip
from .spec import field, preset, switch

MODEL = {
    "id": "pendulum", "title": "Pendulum", "animation": "spring",
    "description": "A weight swinging on a string or rod: a clock pendulum, a playground swing, a wrecking ball.",
    "law": "Newton's second law for rotation: the component of gravity along the arc pulls the bob back towards the bottom.",
    "fields": [
        field("L", "Length", "L", "length", (None, "m"), positive=True),
        field("g", "Gravity", "g", "acceleration", ("9.81", "m/s2"), positive=True, typical=10),
        field("th0", "Starting angle", "\\theta_0", "angle", (None, "deg"), "initial", bounds=(-3.1, 3.1)),
        field("om0", "Starting angular velocity", "\\omega_0", "angular_frequency", ("0", "rad/s"), "initial"),
        field("gam", "Damping rate", "\\gamma", "rate", (None, "1/s"), "effect", positive=True, typical=0.1),
        field("Tex", "True period for this swing", "T", "time", (None, "s"), "effect", positive=True),
        field("t", "Time", "t", "time", (None, "s"), "moment", nonnegative=True),
        field("th", "Angle at that time", "\\theta", "angle", (None, "deg"), "moment", bounds=(-3.14, 3.14)),
        field("om", "Angular velocity at that time", "\\omega", "angular_frequency", (None, "rad/s"), "moment"),
        field("T0", "Small-swing period", "T_0", "time", (None, "s"), "derived", positive=True),
        field("f0", "Small-swing frequency", "f_0", "frequency", (None, "Hz"), "derived", positive=True),
        field("vmax", "Speed at the bottom", "v_{max}", "speed", (None, "m/s"), "derived", nonnegative=True),
    ],
    "effects": [
        switch("large", "Large swings", "Keep sin \u03b8 instead of approximating it by \u03b8 (needed beyond about 15\u00b0).",
               ["Tex"]),
        switch("damping", "Damping", "Air resistance and friction at the pivot, proportional to angular velocity.",
               ["gam"]),
    ],
    "presets": [
        preset("clock", "Grandfather clock", "A clock pendulum must tick once a second (period 2 s): how long is it?",
               {"g": ("9.81", "m/s2"), "T0": ("2", "s"), "th0": ("4", "deg"), "om0": ("0", "rad/s")}, {}),
        preset("swing", "Playground swing", "A 2.5 m swing pulled back 60\u00b0 and released.",
               {"L": ("2.5", "m"), "g": ("9.81", "m/s2"), "th0": ("60", "deg"), "om0": ("0", "rad/s")},
               {"large": "on"}),
        preset("foucault", "Foucault pendulum", "The 67 m pendulum in the Panth\u00e9on, released from 3\u00b0.",
               {"L": ("67", "m"), "g": ("9.81", "m/s2"), "th0": ("3", "deg"), "om0": ("0", "rad/s")}, {}),
        preset("lab", "Damped lab pendulum", "A 0.5 m pendulum released from 10\u00b0 with damping rate 0.1 per second: its angle after 10 s.",
               {"L": ("0.5", "m"), "g": ("9.81", "m/s2"), "th0": ("10", "deg"), "om0": ("0", "rad/s"),
                "gam": ("0.1", "1/s"), "t": ("10", "s")}, {"damping": "on"}),
        preset("gravity", "Measuring gravity", "A 1.000 m pendulum takes 2.007 s per swing: what is g there?",
               {"L": ("1", "m"), "T0": ("2.007", "s"), "th0": ("5", "deg"), "om0": ("0", "rad/s")}, {}),
    ],
}


def _params(eff):
    return ["L", "g", "th0", "om0"] + (["gam"] if eff.get("damping") == "on" else [])


def trajectory(eff, v):
    if any(v.get(x) is None for x in _params(eff)):
        raise Skip()
    w2 = v["g"] / v["L"]
    gam = v["gam"] if eff.get("damping") == "on" else 0.0
    scale = 6 * 2 * math.pi / math.sqrt(w2)
    if eff.get("large") == "on":
        deriv = lambda t, s: [s[1], -w2 * math.sin(s[0]) - gam * s[1]]
        num = numeric(("pendulum", w2, gam, v["th0"], v["om0"]), deriv, [v["th0"], v["om0"]], scale)

        def at(t):
            th, om = num.state(t)
            return {"th": th, "om": om}
        return {"at": at, "exact": False, "scale": scale, "ode": None}
    at2 = oscillator(1.0, gam, w2, v["th0"], v["om0"])

    def at(t):
        th, om = at2(t)
        return {"th": th, "om": om}
    return {"at": at, "exact": True, "scale": scale,
            "ode": (lambda t, s: [s[1], -w2 * s[0] - gam * s[1]], [v["th0"], v["om0"]], ["th", "om"], scale)}


def _true_period(v):
    """Exact period of an undamped pendulum released from rest, by the arithmetic-geometric mean."""
    if abs(v["om0"]) > 1e-12 or abs(v["th0"]) < 1e-9:
        raise Skip()
    a, b = 1.0, math.cos(abs(v["th0"]) / 2)
    for _ in range(40):
        a, b = (a + b) / 2, math.sqrt(a * b)
    return 2 * math.pi * math.sqrt(v["L"] / v["g"]) / a


def relations(eff):
    large, damping = eff.get("large") == "on", eff.get("damping") == "on"
    rels = [
        Rel("T0", "The period of small swings depends only on the length and gravity",
            "T_0 = 2\\pi\\sqrt{L/g}", ["T0", "L", "g"], lambda v: (v["T0"], 2 * math.pi * math.sqrt(v["L"] / v["g"])),
            {"T0": "T_0 = 2\\pi\\sqrt{\\frac{L}{g}}", "L": "L = g\\left(\\frac{T_0}{2\\pi}\\right)^2",
             "g": "g = L\\left(\\frac{2\\pi}{T_0}\\right)^2"}),
        Rel("f0", "Frequency is swings per second", "f_0 = 1/T_0", ["f0", "T0"], lambda v: (v["f0"], 1 / v["T0"]),
            {"f0": "f_0 = \\frac{1}{T_0}", "T0": "T_0 = \\frac{1}{f_0}"}),
    ]
    if not damping:
        rels.append(Rel(
            "vmax", "Energy conservation between the release point and the bottom of the swing",
            "\\tfrac12 v_{max}^2 = g L (1 - \\cos\\theta_0) + \\tfrac12 (L\\omega_0)^2", ["vmax", "g", "L", "th0", "om0"],
            lambda v: (v["vmax"], math.sqrt(
                (2 * v["g"] * v["L"] * (1 - math.cos(v["th0"])) if large else v["g"] * v["L"] * v["th0"] ** 2)
                + (v["L"] * v["om0"]) ** 2)),
            {"vmax": "v_{max} = \\sqrt{2 g L (1 - \\cos\\theta_0) + (L\\omega_0)^2}" if large
             else "v_{max} = \\sqrt{g L\\,\\theta_0^2 + (L\\omega_0)^2}"}))
    if large and not damping:
        rels.append(Rel("Tex", "The exact period of a large swing released from rest is longer than the small-swing value",
                        "T = 4\\sqrt{\\frac{L}{g}}\\;K\\!\\left(\\sin\\frac{\\theta_0}{2}\\right)", ["Tex", "L", "g", "th0", "om0"],
                        lambda v: (v["Tex"], _true_period(v)),
                        {"Tex": "T = 4\\sqrt{\\frac{L}{g}}\\;K\\!\\left(\\sin\\frac{\\theta_0}{2}\\right)"}))
    p = _params(eff)
    size = lambda v: abs(v["th0"]) + abs(v["om0"]) * math.sqrt(v["L"] / v["g"]) + 1e-6
    rels += [
        Rel("th", "The angle at time t from the solution of the equation of motion", "\\theta = \\theta(t)",
            ["t", "th"] + p, lambda v: (v["th"] + 2 * size(v), trajectory(eff, v)["at"](v["t"])["th"] + 2 * size(v)),
            traj=True),
        Rel("om", "The angular velocity at time t from the solution of the equation of motion", "\\omega = \\theta'(t)",
            ["t", "om"] + p,
            lambda v: (v["om"] + 2 * size(v) * math.sqrt(v["g"] / v["L"]),
                       trajectory(eff, v)["at"](v["t"])["om"] + 2 * size(v) * math.sqrt(v["g"] / v["L"])), traj=True),
    ]
    return rels


def equation(eff):
    restoring = "\\frac{g}{L}\\sin\\theta" if eff.get("large") == "on" else "\\frac{g}{L}\\,\\theta"
    damping = " + \\gamma\\,\\theta'" if eff.get("damping") == "on" else ""
    return "\\theta''%s + %s = 0" % (damping, restoring)


def derivation(eff):
    large = eff.get("large") == "on"
    text = ("The bob moves on an arc of radius \\(L\\). The component of gravity along the arc is "
            "\\(-mg\\sin\\theta\\), so \\(mL\\theta'' = -mg\\sin\\theta\\); the mass cancels.")
    if not large:
        text += " For small swings \\(\\sin\\theta \\approx \\theta\\), which makes the equation linear."
    if eff.get("damping") == "on":
        text += " \\(\\gamma\\theta'\\) is the damping."
    steps = [step("Set up the equation of motion", text, [equation(eff)])]
    if large:
        steps.append(step("Solve the equation",
                          "With \\(\\sin\\theta\\) kept the equation is nonlinear and has no solution in elementary "
                          "functions, so the motion is integrated numerically (fourth-order Runge-Kutta). The "
                          "period, though, has an exact formula in terms of an elliptic integral."))
    elif eff.get("damping") == "on":
        steps.append(step("Solve the equation",
                          "Linear with constant coefficients: the characteristic equation "
                          "\\(r^2 + \\gamma r + g/L = 0\\) gives a decaying oscillation.",
                          ["\\theta(t) = e^{-\\gamma t/2}\\left(C_1\\cos\\omega_d t + C_2\\sin\\omega_d t\\right),"
                           "\\quad \\omega_d = \\sqrt{\\frac{g}{L} - \\frac{\\gamma^2}{4}}"]))
    else:
        steps.append(step("Solve the equation",
                          "Linear with constant coefficients: the characteristic equation \\(r^2 + g/L = 0\\) "
                          "gives simple harmonic motion.",
                          ["\\theta(t) = \\theta_0\\cos\\Omega t + \\frac{\\omega_0}{\\Omega}\\sin\\Omega t,"
                           "\\quad \\Omega = \\sqrt{\\frac{g}{L}}"]))
    return steps


def horizon(eff, v):
    if v.get("L") and v.get("g"):
        return 8 * 2 * math.pi * math.sqrt(v["L"] / v["g"])
    return None


def graph(eff, v):
    traj = trajectory(eff, v)
    t_max = max(traj["scale"] * 0.6, (v.get("t") or 0.0) * 1.2)
    ideal = None
    if eff.get("large") == "on" or eff.get("damping") == "on":
        free = oscillator(1.0, 0.0, v["g"] / v["L"], v["th0"], v["om0"])
        ideal = lambda t: math.degrees(free(t)[0])
    marker = None
    if v.get("t") is not None and v.get("th") is not None:
        marker = {"t": v["t"], "y": math.degrees(v["th"]), "label": "at time t"}
    return {"fn": lambda t: math.degrees(traj["at"](t)["th"]), "ideal": ideal, "t_max": t_max,
            "y_label": "angle (\u00b0)", "label": "Angle", "ideal_label": "Small swings, no damping", "marker": marker}
