"""A mass on a spring, with optional damping, a driving force and gravity."""
import math

from ..first_order import step
from .common import oscillator
from .solver import Rel, Skip
from .spec import field, preset, switch

MODEL = {
    "id": "spring", "title": "Mass on a spring", "animation": "spring",
    "description": "Anything that bounces on a spring: a car on its suspension, a door closer, a lab cart.",
    "law": "Newton's second law with Hooke's law: the spring pulls back in proportion to the stretch.",
    "fields": [
        field("m", "Mass", "m", "mass", (None, "kg"), positive=True),
        field("k", "Spring constant", "k", "spring_constant", (None, "N/m"), positive=True, typical=100),
        field("x0", "Starting displacement", "x_0", "length", (None, "m"), "initial", typical=0.1),
        field("v0", "Starting velocity", "v_0", "speed", ("0", "m/s"), "initial"),
        field("c", "Damping coefficient", "c", "damping", (None, "kg/s"), "effect", positive=True, typical=10),
        field("zeta", "Damping ratio", "\\zeta", "dimensionless", (None, ""), "effect", positive=True),
        field("wd", "Damped angular frequency", "\\omega_d", "angular_frequency", (None, "rad/s"), "effect",
              positive=True, typical=10, optional=True),
        field("tau", "Decay time of the amplitude", "\\tau", "time", (None, "s"), "effect", positive=True),
        field("F0", "Driving force amplitude", "F_0", "force", (None, "N"), "effect", positive=True),
        field("w", "Driving frequency", "\\omega", "angular_frequency", (None, "Hz"), "effect",
              positive=True, typical=10),
        field("amp", "Steady-state amplitude", "A", "length", (None, "m"), "effect", positive=True, typical=0.01),
        field("g", "Gravity", "g", "acceleration", ("9.81", "m/s2"), "effect", positive=True, typical=10),
        field("sag", "Static sag", "x_s", "length", (None, "m"), "effect", positive=True, typical=0.01),
        field("t", "Time", "t", "time", (None, "s"), "moment", nonnegative=True),
        field("x", "Displacement at that time", "x", "length", (None, "m"), "moment", typical=0.1),
        field("v", "Velocity at that time", "v", "speed", (None, "m/s"), "moment"),
        field("w0", "Natural angular frequency", "\\omega_0", "angular_frequency", (None, "rad/s"), "derived",
              positive=True, typical=10),
        field("T", "Natural period", "T", "time", (None, "s"), "derived", positive=True),
        field("f", "Natural frequency", "f", "frequency", (None, "Hz"), "derived", positive=True),
        field("E", "Starting energy", "E_0", "energy", (None, "J"), "derived", nonnegative=True),
    ],
    "effects": [
        switch("damping", "Damping", "A friction force proportional to speed (a shock absorber).",
               ["c", "zeta", "wd", "tau"]),
        switch("driving", "Driving force", "A periodic push, F\u2080 cos(\u03c9t).", ["F0", "w", "amp"]),
        switch("gravity", "Gravity (vertical spring)", "The weight stretches the spring to a new rest position.",
               ["g", "sag"]),
    ],
    "presets": [
        preset("car", "Car suspension", "One corner of a car (400 kg) pushed down 5 cm and released: where is it after 0.3 s?",
               {"m": ("400", "kg"), "k": ("40", "kN/m"), "x0": ("5", "cm"), "v0": ("0", "m/s"),
                "c": ("3500", "kg/s"), "t": ("0.3", "s"), "x": ("", "cm")}, {"damping": "on"}),
        preset("door", "Door closer", "What damping makes a 30 kg door on a 60 N/m closer shut without bouncing (\u03b6 = 1)?",
               {"m": ("30", "kg"), "k": ("60", "N/m"), "zeta": ("1", ""), "x0": ("60", "cm"), "v0": ("0", "m/s"),
                "t": ("2", "s"), "x": ("", "cm")}, {"damping": "on"}),
        preset("cart", "Lab cart", "A cart on a spring is timed at 1.0 s per swing: what is the spring constant?",
               {"m": ("0.5", "kg"), "T": ("1.0", "s"), "x0": ("10", "cm"), "v0": ("0", "m/s")}, {}),
        preset("washer", "Washing machine", "A 60 kg drum on its mounts, shaken by an unbalanced load at 5 Hz.",
               {"m": ("60", "kg"), "k": ("60", "kN/m"), "x0": ("0", "cm"), "v0": ("0", "m/s"),
                "c": ("600", "kg/s"), "F0": ("100", "N"), "w": ("5", "Hz"), "amp": ("", "mm")}, {"damping": "on", "driving": "on"}),
        preset("scale", "Hanging scale", "A 2 kg bag stretches a spring scale 4 cm: what is k, and how fast does it bob?",
               {"m": ("2", "kg"), "g": ("9.81", "m/s2"), "sag": ("4", "cm"), "x0": ("6", "cm"), "v0": ("0", "m/s")}, {"gravity": "on"}),
    ],
}


def _params(eff):
    p = ["m", "k", "x0", "v0"]
    if eff.get("damping") == "on":
        p.append("c")
    if eff.get("driving") == "on":
        p += ["F0", "w"]
    if eff.get("gravity") == "on":
        p.append("g")
    return p


def trajectory(eff, v):
    if any(v.get(x) is None for x in _params(eff)):
        raise Skip()
    m, k = v["m"], v["k"]
    c = v["c"] if eff.get("damping") == "on" else 0.0
    f0, w = (v["F0"], v["w"]) if eff.get("driving") == "on" else (0.0, 0.0)
    fc = m * v["g"] if eff.get("gravity") == "on" else 0.0
    at2 = oscillator(m, c, k, v["x0"], v["v0"], fc, f0, w)
    scale = 6 * 2 * math.pi / math.sqrt(k / m)

    def deriv(t, s):
        return [s[1], (fc + f0 * math.cos(w * t) - c * s[1] - k * s[0]) / m]

    def at(t):
        x, vel = at2(t)
        return {"x": x, "v": vel}
    return {"at": at, "ode": (deriv, [v["x0"], v["v0"]], ["x", "v"], scale), "exact": True, "scale": scale}


def relations(eff):
    rels = [
        Rel("w0", "The natural angular frequency of a mass on a spring", "\\omega_0 = \\sqrt{k/m}",
            ["w0", "k", "m"], lambda v: (v["w0"], math.sqrt(v["k"] / v["m"])),
            {"w0": "\\omega_0 = \\sqrt{\\frac{k}{m}}", "k": "k = m\\,\\omega_0^2", "m": "m = \\frac{k}{\\omega_0^2}"}),
        Rel("T", "One period is the time for a full cycle", "T = 2\\pi/\\omega_0", ["T", "w0"],
            lambda v: (v["T"], 2 * math.pi / v["w0"]),
            {"T": "T = \\frac{2\\pi}{\\omega_0}", "w0": "\\omega_0 = \\frac{2\\pi}{T}"}),
        Rel("f", "Frequency is cycles per second", "f = 1/T", ["f", "T"], lambda v: (v["f"], 1 / v["T"]),
            {"f": "f = \\frac{1}{T}", "T": "T = \\frac{1}{f}"}),
        Rel("E", "The starting energy is spring energy plus kinetic energy",
            "E_0 = \\tfrac12 k x_0^2 + \\tfrac12 m v_0^2", ["E", "k", "x0", "m", "v0"],
            lambda v: (v["E"], 0.5 * v["k"] * v["x0"] ** 2 + 0.5 * v["m"] * v["v0"] ** 2),
            {"E": "E_0 = \\tfrac12 k x_0^2 + \\tfrac12 m v_0^2"}),
    ]
    if eff.get("damping") == "on":
        rels += [
            Rel("zeta", "The damping ratio compares the damping with the critical value",
                "\\zeta = \\frac{c}{2\\sqrt{k m}}", ["zeta", "c", "k", "m"],
                lambda v: (v["zeta"], v["c"] / (2 * math.sqrt(v["k"] * v["m"]))),
                {"zeta": "\\zeta = \\frac{c}{2\\sqrt{k m}}", "c": "c = 2\\zeta\\sqrt{k m}"}),
            Rel("wd", "Damping lowers the oscillation frequency (underdamped case only)",
                "\\omega_d = \\omega_0\\sqrt{1 - \\zeta^2}", ["wd", "w0", "zeta"],
                lambda v: (v["wd"], v["w0"] * math.sqrt(1 - v["zeta"] ** 2)),
                {"wd": "\\omega_d = \\omega_0\\sqrt{1 - \\zeta^2}"}),
            Rel("tau", "The amplitude decays as e^{-t/\u03c4}", "\\tau = \\frac{2m}{c}", ["tau", "m", "c"],
                lambda v: (v["tau"], 2 * v["m"] / v["c"]),
                {"tau": "\\tau = \\frac{2m}{c}", "c": "c = \\frac{2m}{\\tau}", "m": "m = \\frac{c\\,\\tau}{2}"}),
        ]
    if eff.get("driving") == "on":
        damp = eff.get("damping") == "on"
        rels.append(Rel(
            "amp", "The amplitude of the steady oscillation that remains once the transient has died away",
            "A = \\frac{F_0}{\\sqrt{(k - m\\omega^2)^2 + (c\\,\\omega)^2}}",
            ["amp", "F0", "k", "m", "w"] + (["c"] if damp else []),
            lambda v: (v["amp"], v["F0"] / math.hypot(v["k"] - v["m"] * v["w"] ** 2,
                                                       (v["c"] if damp else 0.0) * v["w"])),
            {"amp": "A = \\frac{F_0}{\\sqrt{(k - m\\omega^2)^2 + (c\\,\\omega)^2}}"}))
    if eff.get("gravity") == "on":
        rels.append(Rel("sag", "At rest the spring force balances the weight", "k\\,x_s = m\\,g",
                        ["sag", "m", "g", "k"], lambda v: (v["k"] * v["sag"], v["m"] * v["g"]),
                        {"sag": "x_s = \\frac{m g}{k}", "k": "k = \\frac{m g}{x_s}", "m": "m = \\frac{k x_s}{g}"}))
    p = _params(eff)
    scale = lambda v: abs(v.get("x0") or 0.0) + abs(v.get("v0") or 0.0) * math.sqrt(v["m"] / v["k"]) + 1e-6
    rels.append(Rel("x", "The displacement at time t from the solution of the equation of motion",
                    "x = x(t)", ["t", "x"] + p,
                    lambda v: (v["x"] + _size(eff, v), trajectory(eff, v)["at"](v["t"])["x"] + _size(eff, v)),
                    traj=True))
    rels.append(Rel("v", "The velocity at time t from the solution of the equation of motion",
                    "v = x'(t)", ["t", "v"] + p,
                    lambda v: (v["v"] + _size(eff, v) * math.sqrt(v["k"] / v["m"]),
                               trajectory(eff, v)["at"](v["t"])["v"] + _size(eff, v) * math.sqrt(v["k"] / v["m"])),
                    traj=True))
    return rels


def _size(eff, v):
    """A displacement scale, so that values near zero are compared against the size of the motion."""
    size = abs(v["x0"]) + abs(v["v0"]) * math.sqrt(v["m"] / v["k"])
    if eff.get("driving") == "on":
        size += v["F0"] / v["k"]
    if eff.get("gravity") == "on":
        size += v["m"] * v["g"] / v["k"]
    return 2 * size + 1e-9


def equation(eff):
    left = "m\\,x''"
    if eff.get("damping") == "on":
        left += " + c\\,x'"
    left += " + k\\,x"
    right = []
    if eff.get("driving") == "on":
        right.append("F_0\\cos(\\omega t)")
    if eff.get("gravity") == "on":
        right.append("m\\,g")
    return "%s = %s" % (left, " + ".join(right) or "0")


def derivation(eff):
    parts = ["\\(kx\\) is the spring force (Hooke's law), pulling back towards the unstretched position."]
    if eff.get("damping") == "on":
        parts.append("\\(cx'\\) is the damping force, proportional to speed.")
    if eff.get("driving") == "on":
        parts.append("\\(F_0\\cos(\\omega t)\\) is the periodic driving force.")
    if eff.get("gravity") == "on":
        parts.append("\\(mg\\) is the weight, with \\(x\\) measured downward from the unstretched length.")
    steps = [step("Set up the equation of motion", "Newton's second law. " + " ".join(parts), [equation(eff)])]
    lines = ["m r^2 + c r + k = 0" if eff.get("damping") == "on" else "m r^2 + k = 0"]
    text = ("The equation is linear with constant coefficients. Its characteristic equation gives the free "
            "motion: ")
    if eff.get("damping") == "on":
        text += ("oscillation inside a decaying envelope when \\(\\zeta < 1\\), the fastest return without "
                 "overshoot when \\(\\zeta = 1\\), and a slow return when \\(\\zeta > 1\\).")
        lines.append("x_h = e^{-t/\\tau}\\left(C_1\\cos\\omega_d t + C_2\\sin\\omega_d t\\right)\\quad(\\zeta < 1)")
    else:
        text += "a steady oscillation at \\(\\omega_0 = \\sqrt{k/m}\\)."
        lines.append("x_h = C_1\\cos\\omega_0 t + C_2\\sin\\omega_0 t")
    if eff.get("driving") == "on" or eff.get("gravity") == "on":
        text += (" A particular solution is added for the right-hand side (undetermined coefficients), "
                 "and the constants are fixed by the starting displacement and velocity.")
    else:
        text += " The constants are fixed by the starting displacement and velocity."
    steps.append(step("Solve the equation", text, lines))
    return steps


def horizon(eff, v):
    if v.get("m") and v.get("k"):
        return 8 * 2 * math.pi / math.sqrt(v["k"] / v["m"])
    return None


def graph(eff, v):
    traj = trajectory(eff, v)
    t_max = max(traj["scale"], (v.get("t") or 0.0) * 1.2)
    ideal = None
    if any(eff.get(e["id"]) == "on" for e in MODEL["effects"]):
        free = oscillator(v["m"], 0.0, v["k"], v["x0"], v["v0"])
        ideal = lambda t: free(t)[0]
    marker = None
    if v.get("t") is not None and v.get("x") is not None:
        marker = {"t": v["t"], "y": v["x"], "label": "at time t"}
    return {"fn": lambda t: traj["at"](t)["x"], "ideal": ideal, "t_max": t_max,
            "y_label": "displacement (m)", "label": "Displacement",
            "ideal_label": "Spring alone (no damping, driving or gravity)", "marker": marker}
