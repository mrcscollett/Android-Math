"""An RL circuit: the current in an inductor building up or dying away."""
import math

from ..first_order import step
from .circuits import response
from .solver import Rel, Skip
from .spec import choice, field, preset, switch

MODEL = {
    "id": "rl", "title": "RL circuit", "animation": "capacitor",
    "description": "A coil and a resistor: a relay, a motor winding, an electromagnet switching on or off.",
    "law": "Kirchhoff's voltage law: the source voltage equals the resistor drop plus the inductor's back-EMF, L di/dt.",
    "fields": [
        field("L", "Inductance", "L", "inductance", (None, "mH"), positive=True, typical=1e-2),
        field("R", "Resistance", "R", "resistance", (None, "ohm"), positive=True, typical=10),
        field("i0", "Starting current", "I_0", "current", ("0", "A"), "initial"),
        field("V", "Battery voltage", "V_s", "voltage", (None, "V"), "effect", typical=10),
        field("if_", "Final current", "I_f", "current", (None, "A"), "effect"),
        field("ef", "Final stored energy", "E_f", "energy", (None, "J"), "effect", nonnegative=True),
        field("vamp", "AC source amplitude", "V_a", "voltage", (None, "V"), "effect", positive=True),
        field("freq", "AC source frequency", "f", "frequency", (None, "Hz"), "effect", positive=True, typical=50),
        field("iac", "Steady current amplitude", "A", "current", (None, "A"), "effect", positive=True),
        field("r", "Internal resistance of the source", "r", "resistance", (None, "ohm"), "effect", positive=True),
        field("t", "Time", "t", "time", (None, "s"), "moment", nonnegative=True, typical=1e-3),
        field("i", "Current at that time", "i", "current", (None, "A"), "moment"),
        field("vl", "Inductor voltage at that time", "v_L", "voltage", (None, "V"), "moment"),
        field("el", "Stored energy at that time", "E_L", "energy", (None, "J"), "moment", nonnegative=True),
        field("tau", "Time constant", "\\tau", "time", (None, "s"), "derived", positive=True, typical=1e-3),
    ],
    "effects": [
        choice("source", "Source", "What drives the circuit.",
               [("battery", "Battery (current builds up)", ["V", "if_", "ef"]),
                ("none", "None (current dies away)", []),
                ("ac", "AC source", ["vamp", "freq", "iac"])], "battery"),
        switch("internal", "Internal resistance", "A real source has some resistance of its own, in series.", ["r"]),
    ],
    "presets": [
        preset("relay", "Relay coil", "A 12 V relay (200 mH, 240 \u03a9) pulls in at 35 mA: how long after switch-on?",
               {"L": ("200", "mH"), "R": ("240", "ohm"), "V": ("12", "V"), "i0": ("0", "A"), "i": ("35", "mA"),
                "t": ("", "ms"), "tau": ("", "ms"), "if_": ("", "mA"), "ef": ("", "mJ"), "el": ("", "mJ")},
               {"source": "battery"}),
        preset("magnet", "Electromagnet", "A 2 H lifting magnet with 4 \u03a9 on 48 V: final current and stored energy.",
               {"L": ("2", "H"), "R": ("4", "ohm"), "V": ("48", "V"), "i0": ("0", "A")}, {"source": "battery"}),
        preset("flyback", "Coil switched off", "A coil carrying 3 A is shorted through 15 \u03a9 and falls to 1 A in 2 ms: what is L?",
               {"R": ("15", "ohm"), "i0": ("3", "A"), "t": ("2", "ms"), "i": ("1", "A"), "tau": ("", "ms"),
                "el": ("", "mJ")}, {"source": "none"}),
        preset("mains", "Motor winding on AC", "A 50 mH, 10 \u03a9 winding on a 325 V, 50 Hz supply.",
               {"L": ("50", "mH"), "R": ("10", "ohm"), "vamp": ("325", "V"), "freq": ("50", "Hz"), "i0": ("0", "A"), "tau": ("", "ms")},
               {"source": "ac"}),
    ],
}


def _series(eff, v):
    return v["R"] + (v["r"] if eff.get("internal") == "on" else 0.0)


def _rvars(eff):
    return ["R"] + (["r"] if eff.get("internal") == "on" else [])


def _params(eff):
    source = eff.get("source")
    p = ["tau", "i0"]
    if source == "battery":
        p.append("if_")
    if source == "ac":
        p += ["vamp", "freq"] + _rvars(eff)
    return p


def trajectory(eff, v):
    if any(v.get(x) is None for x in _params(eff)):
        raise Skip()
    source, tau, start = eff.get("source"), v["tau"], v["i0"]
    if source == "battery":
        fn, target = response(source, v["if_"], start, tau), (lambda t: v["if_"])
    elif source == "ac":
        amp, freq = v["vamp"] / _series(eff, v), v["freq"]
        fn = response(source, 0.0, start, tau, amp, freq)
        target = lambda t: amp * math.cos(2 * math.pi * freq * t)
    else:
        fn, target = response(source, 0.0, start, tau), (lambda t: 0.0)
    scale = 5 * tau if source != "ac" else max(5 * tau, 3 / v["freq"])
    return {"at": lambda t: {"i": fn(t)}, "exact": True, "scale": scale,
            "ode": (lambda t, s: [(target(t) - s[0]) / tau], [start], ["i"], scale)}


def _source_voltage(eff, v):
    source = eff.get("source")
    if source == "battery":
        return v["V"]
    if source == "ac":
        return v["vamp"] * math.cos(2 * math.pi * v["freq"] * v["t"])
    return 0.0


def relations(eff):
    source = eff.get("source")
    internal = eff.get("internal") == "on"
    rv = _rvars(eff)
    res = "R + r" if internal else "R"
    rels = [Rel("tau", "The time constant of a coil and a resistor", "\\tau = \\frac{L}{%s}" % res, ["tau", "L"] + rv,
                lambda v: (v["tau"], v["L"] / _series(eff, v)),
                {"tau": "\\tau = \\frac{L}{%s}" % res, "L": "L = \\tau\\,(%s)" % res,
                 "R": "R = \\frac{L}{\\tau}" if not internal else None})]
    if source == "battery":
        rels += [
            Rel("if_", "Once the current is steady the inductor has no voltage across it, so Ohm's law gives the current",
                "I_f = \\frac{V_s}{%s}" % res, ["if_", "V"] + rv, lambda v: (v["if_"] * _series(eff, v), v["V"]),
                {"if_": "I_f = \\frac{V_s}{%s}" % res, "V": "V_s = I_f\\,(%s)" % res,
                 "R": "R = \\frac{V_s}{I_f}" if not internal else None}),
            Rel("ef", "The energy stored in the magnetic field at the final current", "E_f = \\tfrac12 L I_f^2",
                ["ef", "L", "if_"], lambda v: (v["ef"], 0.5 * v["L"] * v["if_"] ** 2),
                {"ef": "E_f = \\tfrac12 L I_f^2", "L": "L = \\frac{2E_f}{I_f^2}"}),
        ]
    if source == "ac":
        rels.append(Rel("iac", "The steady current amplitude: the source amplitude over the impedance",
                        "A = \\frac{V_a}{\\sqrt{(%s)^2 + (2\\pi f L)^2}}" % res, ["iac", "vamp", "freq", "L"] + rv,
                        lambda v: (v["iac"], v["vamp"] / math.hypot(_series(eff, v), 2 * math.pi * v["freq"] * v["L"])),
                        {"iac": "A = \\frac{V_a}{\\sqrt{(%s)^2 + (2\\pi f L)^2}}" % res}))
    forms = None
    if source == "battery":
        forms = {"i": "i = I_f + (I_0 - I_f)\\,e^{-t/\\tau}", "t": "t = \\tau\\,\\ln\\frac{I_0 - I_f}{i - I_f}",
                 "tau": "\\tau = \\frac{t}{\\ln\\frac{I_0 - I_f}{i - I_f}}"}
    elif source == "none":
        forms = {"i": "i = I_0\\,e^{-t/\\tau}", "t": "t = \\tau\\,\\ln\\frac{I_0}{i}",
                 "tau": "\\tau = \\frac{t}{\\ln(I_0/i)}", "i0": "I_0 = i\\,e^{t/\\tau}"}
    size = lambda v: abs(v.get("i0") or 0.0) + abs(v.get("if_") or 0.0) + (
        abs(v["vamp"] / _series(eff, v)) if source == "ac" else 0.0) + 1e-12
    rels.append(Rel("i", "The current at time t from the solution of the circuit equation", "i = i(t)",
                    ["t", "i"] + _params(eff),
                    lambda v: (v["i"] + 2 * size(v), trajectory(eff, v)["at"](v["t"])["i"] + 2 * size(v)), forms,
                    traj=True))
    rels += [
        Rel("el", "Energy stored in an inductor", "E_L = \\tfrac12 L i^2", ["el", "L", "i"],
            lambda v: (v["el"], 0.5 * v["L"] * v["i"] ** 2), {"el": "E_L = \\tfrac12 L i^2"}),
        Rel("vl", "The inductor takes whatever voltage the resistor does not",
            "v_L = v_{source} - (%s)\\,i" % res,
            ["vl", "i"] + rv + (["V"] if source == "battery" else []) + (["vamp", "freq", "t"] if source == "ac" else []),
            lambda v: (v["vl"] + 2 * (abs(_source_voltage(eff, v)) + abs(_series(eff, v) * v["i"])) + 1e-12,
                       _source_voltage(eff, v) - _series(eff, v) * v["i"]
                       + 2 * (abs(_source_voltage(eff, v)) + abs(_series(eff, v) * v["i"])) + 1e-12),
            {"vl": "v_L = v_{source} - (%s)\\,i" % res}),
    ]
    return rels


def equation(eff):
    res = "(R + r)" if eff.get("internal") == "on" else "R"
    right = {"battery": "V_s", "none": "0", "ac": "V_a\\cos(2\\pi f t)"}[eff.get("source")]
    return "L\\,\\frac{di}{dt} + %s\\,i = %s" % (res, right)


def derivation(eff):
    source = eff.get("source")
    text = ("Kirchhoff's voltage law around the loop. \\(L\\,di/dt\\) is the inductor's back-EMF, which "
            "opposes any change in the current; \\(Ri\\) is the resistor's voltage drop.")
    if eff.get("internal") == "on":
        text += " The source's internal resistance \\(r\\) adds to \\(R\\)."
    steps = [step("Set up the circuit equation", text, [equation(eff)])]
    if source == "ac":
        steps.append(step("Solve the equation",
                          "The equation is first-order and linear: a steady oscillation at the source "
                          "frequency plus a transient that dies away in a few time constants.",
                          ["i(t) = A\\cos(2\\pi f t - \\varphi) + K\\,e^{-t/\\tau},\\quad \\tan\\varphi = 2\\pi f\\tau"]))
    else:
        steps.append(step("Solve the equation",
                          "The equation is first-order and linear, \\(i' + i/\\tau = I_f/\\tau\\). An integrating "
                          "factor \\(e^{t/\\tau}\\) gives an exponential approach to the final current.",
                          ["i(t) = I_f + (I_0 - I_f)\\,e^{-t/\\tau}" if source == "battery" else "i(t) = I_0\\,e^{-t/\\tau}"]))
    return steps


def horizon(eff, v):
    return 6 * v["tau"] if v.get("tau") else None


def graph(eff, v):
    traj = trajectory(eff, v)
    t_max = max(traj["scale"], (v.get("t") or 0.0) * 1.2)
    ideal = None
    if eff.get("internal") == "on" and v.get("R") and v.get("L"):
        source, tau = eff.get("source"), v["L"] / v["R"]
        if source == "battery" and v.get("V") is not None:
            ideal = response(source, v["V"] / v["R"], v["i0"], tau)
        elif source == "none":
            ideal = response(source, 0.0, v["i0"], tau)
        elif source == "ac":
            ideal = response(source, 0.0, v["i0"], tau, v["vamp"] / v["R"], v["freq"])
    marker = None
    if v.get("t") is not None and v.get("i") is not None:
        marker = {"t": v["t"], "y": v["i"], "label": "at time t"}
    return {"fn": lambda t: traj["at"](t)["i"], "ideal": ideal, "t_max": t_max, "y_label": "current (A)",
            "label": "Current", "ideal_label": "Ideal source", "marker": marker}
