"""An RC circuit: a capacitor charging or discharging through a resistor."""
import math

from ..first_order import step
from .circuits import response
from .solver import Rel, Skip
from .spec import choice, field, preset, switch

MODEL = {
    "id": "rc", "title": "RC circuit", "animation": "capacitor",
    "description": "A capacitor charging or discharging through a resistor: a camera flash, a timer, a filter.",
    "law": "Kirchhoff's voltage law around the loop: the source voltage equals the resistor drop plus the capacitor voltage.",
    "fields": [
        field("R", "Resistance", "R", "resistance", (None, "ohm"), positive=True, typical=1e3),
        field("C", "Capacitance", "C", "capacitance", (None, "uF"), positive=True, typical=1e-6),
        field("vc0", "Starting capacitor voltage", "V_0", "voltage", ("0", "V"), "initial"),
        field("V", "Battery voltage", "V_s", "voltage", (None, "V"), "effect", typical=10),
        field("vf", "Final capacitor voltage", "V_f", "voltage", (None, "V"), "effect", typical=10),
        field("qf", "Final charge", "Q_f", "charge", (None, "C"), "effect", typical=1e-5),
        field("ef", "Final stored energy", "E_f", "energy", (None, "J"), "effect", nonnegative=True, typical=1e-3),
        field("vamp", "AC source amplitude", "V_a", "voltage", (None, "V"), "effect", positive=True),
        field("freq", "AC source frequency", "f", "frequency", (None, "Hz"), "effect", positive=True, typical=1e3),
        field("aac", "Steady capacitor amplitude", "A", "voltage", (None, "V"), "effect", positive=True),
        field("phase", "Phase lag", "\\varphi", "angle", (None, "deg"), "effect", bounds=(0.0, math.pi / 2)),
        field("fc", "Cutoff frequency", "f_c", "frequency", (None, "Hz"), "effect", positive=True, typical=1e3),
        field("r", "Internal resistance of the source", "r", "resistance", (None, "ohm"), "effect",
              positive=True, typical=10),
        field("rl", "Leakage resistance", "R_L", "resistance", (None, "Mohm"), "effect", positive=True, typical=1e6),
        field("t", "Time", "t", "time", (None, "s"), "moment", nonnegative=True, typical=1e-3),
        field("vc", "Capacitor voltage at that time", "v_C", "voltage", (None, "V"), "moment"),
        field("q", "Charge at that time", "q", "charge", (None, "C"), "moment", typical=1e-5),
        field("i", "Current at that time", "i", "current", (None, "A"), "moment", typical=1e-3),
        field("ec", "Stored energy at that time", "E_C", "energy", (None, "J"), "moment", nonnegative=True,
              typical=1e-3),
        field("tau", "Time constant", "\\tau", "time", (None, "s"), "derived", positive=True, typical=1e-3),
    ],
    "effects": [
        choice("source", "Source", "What drives the circuit.",
               [("battery", "Battery (charging)", ["V", "vf", "qf", "ef"]),
                ("discharge", "None (discharging)", []),
                ("ac", "AC source", ["vamp", "freq", "aac", "phase", "fc"])], "battery"),
        switch("internal", "Internal resistance", "A real source has some resistance of its own, in series.", ["r"]),
        switch("leak", "Leaky capacitor", "A real capacitor slowly leaks through a large parallel resistance.", ["rl"]),
    ],
    "presets": [
        preset("flash", "Camera flash", "A 120 \u00b5F flash capacitor charging to 300 V through 2 k\u03a9: when is it at 285 V (95%)?",
               {"R": ("2", "kohm"), "C": ("120", "uF"), "V": ("300", "V"), "vc0": ("0", "V"), "vc": ("285", "V"),
                "t": ("", "ms"), "tau": ("", "ms"), "i": ("", "mA"), "q": ("", "mC"), "qf": ("", "mC")},
               {"source": "battery"}),
        preset("timer", "Timing circuit", "A 5 V timer: how long until the capacitor reaches 3.3 V?",
               {"R": ("100", "kohm"), "C": ("10", "uF"), "V": ("5", "V"), "vc0": ("0", "V"), "vc": ("3.3", "V"),
                "i": ("", "uA"), "q": ("", "uC"), "qf": ("", "uC"), "ec": ("", "uJ"), "ef": ("", "uJ")},
               {"source": "battery"}),
        preset("touch", "Touch sensor", "A touch pad charges to 2.0 V in 110 \u00b5s through 1 M\u03a9 from 3.3 V: what is its capacitance?",
               {"R": ("1", "Mohm"), "V": ("3.3", "V"), "vc0": ("0", "V"), "t": ("110", "us"), "vc": ("2.0", "V"),
                "C": ("", "pF"), "tau": ("", "us"), "i": ("", "uA"), "q": ("", "nC"), "qf": ("", "nC")},
               {"source": "battery"}),
        preset("defib", "Defibrillator", "A 32 \u00b5F capacitor at 5 kV discharging through a 50 \u03a9 chest: the state after 2 ms.",
               {"R": ("50", "ohm"), "C": ("32", "uF"), "vc0": ("5", "kV"), "t": ("2", "ms"), "tau": ("", "ms"),
                "vc": ("", "kV"), "q": ("", "mC")},
               {"source": "discharge"}),
        preset("filter", "Audio low-pass filter", "A 1 k\u03a9, 100 nF filter fed with a 5 kHz tone.",
               {"R": ("1", "kohm"), "C": ("100", "nF"), "vamp": ("1", "V"), "vc0": ("0", "V"), "freq": ("5", "kHz"),
                "tau": ("", "us"), "fc": ("", "kHz")},
               {"source": "ac"}),
    ],
}


def _series(eff, v):
    return v["R"] + (v["r"] if eff.get("internal") == "on" else 0.0)


def _gain(eff, v):
    """Fraction of the source voltage the capacitor settles at (less than 1 with leakage)."""
    if eff.get("leak") == "on":
        return v["rl"] / (_series(eff, v) + v["rl"])
    return 1.0


def _rvars(eff):
    return ["R"] + (["r"] if eff.get("internal") == "on" else []) + (["rl"] if eff.get("leak") == "on" else [])


def _params(eff):
    source = eff.get("source")
    p = ["tau", "vc0"]
    if source == "battery":
        p.append("vf")
    if source == "ac":
        p += ["vamp", "freq"] + (_rvars(eff) if eff.get("leak") == "on" else [])
    return p


def trajectory(eff, v):
    if any(v.get(x) is None for x in _params(eff)):
        raise Skip()
    source, tau, start = eff.get("source"), v["tau"], v["vc0"]
    if source == "battery":
        fn = response(source, v["vf"], start, tau)
        target = lambda t: v["vf"]
    elif source == "ac":
        amp, freq = v["vamp"] * _gain(eff, v), v["freq"]
        fn = response(source, 0.0, start, tau, amp, freq)
        target = lambda t: amp * math.cos(2 * math.pi * freq * t)
    else:
        fn = response(source, 0.0, start, tau)
        target = lambda t: 0.0
    scale = 5 * tau if source != "ac" else max(5 * tau, 3 / v["freq"])
    return {"at": lambda t: {"vc": fn(t)}, "exact": True, "scale": scale,
            "ode": (lambda t, s: [(target(t) - s[0]) / tau], [start], ["vc"], scale)}


def _source_voltage(eff, v):
    source = eff.get("source")
    if source == "battery":
        return v["V"]
    if source == "ac":
        return v["vamp"] * math.cos(2 * math.pi * v["freq"] * v["t"])
    return 0.0


def relations(eff):
    source = eff.get("source")
    leak, internal = eff.get("leak") == "on", eff.get("internal") == "on"
    rv = _rvars(eff)
    plain = not leak and not internal
    rels = [Rel("tau", "The time constant is the resistance seen by the capacitor times the capacitance",
                "\\tau = R\\,C" if plain else "\\tau = R_{eq}\\,C", ["tau", "C"] + rv,
                lambda v: (v["tau"], v["C"] * _series(eff, v) * _gain(eff, v)),
                {"tau": "\\tau = R\\,C", "R": "R = \\frac{\\tau}{C}", "C": "C = \\frac{\\tau}{R}"} if plain
                else {"tau": "\\tau = R_{eq}\\,C,\\quad R_{eq} = \\frac{(R + r)\\,R_L}{R + r + R_L}" if leak
                      else "\\tau = (R + r)\\,C"})]
    if source == "battery":
        rels += [
            Rel("vf", "The capacitor settles at the source voltage (reduced by the leakage divider, if any)",
                "V_f = V_s" if not leak else "V_f = V_s\\,\\frac{R_L}{R + r + R_L}", ["vf", "V"] + (rv if leak else []),
                lambda v: (v["vf"], v["V"] * _gain(eff, v)),
                {"vf": "V_f = V_s" if not leak else "V_f = V_s\\,\\frac{R_L}{R + r + R_L}",
                 "V": "V_s = V_f" if not leak else None}),
            Rel("qf", "The final charge", "Q_f = C\\,V_f", ["qf", "C", "vf"], lambda v: (v["qf"], v["C"] * v["vf"]),
                {"qf": "Q_f = C\\,V_f", "C": "C = \\frac{Q_f}{V_f}", "vf": "V_f = \\frac{Q_f}{C}"}),
            Rel("ef", "The final stored energy", "E_f = \\tfrac12 C V_f^2", ["ef", "C", "vf"],
                lambda v: (v["ef"], 0.5 * v["C"] * v["vf"] ** 2),
                {"ef": "E_f = \\tfrac12 C V_f^2", "C": "C = \\frac{2E_f}{V_f^2}"}),
        ]
    if source == "ac":
        w = lambda v: 2 * math.pi * v["freq"] * v["tau"]
        rels += [
            Rel("aac", "Once the start-up transient has gone, the capacitor voltage is a smaller copy of the source",
                "A = \\frac{V_a}{\\sqrt{1 + (2\\pi f\\tau)^2}}", ["aac", "vamp", "freq", "tau"] + (rv if leak else []),
                lambda v: (v["aac"], v["vamp"] * _gain(eff, v) / math.sqrt(1 + w(v) ** 2)),
                {"aac": "A = \\frac{V_a}{\\sqrt{1 + (2\\pi f\\tau)^2}}"}),
            Rel("phase", "The capacitor voltage lags the source", "\\tan\\varphi = 2\\pi f\\tau",
                ["phase", "freq", "tau"], lambda v: (v["phase"] + 1.0, math.atan(w(v)) + 1.0),
                {"phase": "\\varphi = \\tan^{-1}(2\\pi f\\tau)"}),
            Rel("fc", "The cutoff frequency, where the output has fallen to 71% of the input",
                "f_c = \\frac{1}{2\\pi\\tau}", ["fc", "tau"], lambda v: (v["fc"], 1 / (2 * math.pi * v["tau"])),
                {"fc": "f_c = \\frac{1}{2\\pi\\tau}", "tau": "\\tau = \\frac{1}{2\\pi f_c}"}),
        ]
    p = _params(eff)
    forms = None
    if source == "battery":
        forms = {"vc": "v_C = V_f + (V_0 - V_f)\\,e^{-t/\\tau}", "t": "t = \\tau\\,\\ln\\frac{V_0 - V_f}{v_C - V_f}",
                 "tau": "\\tau = \\frac{t}{\\ln\\frac{V_0 - V_f}{v_C - V_f}}"}
    elif source == "discharge":
        forms = {"vc": "v_C = V_0\\,e^{-t/\\tau}", "t": "t = \\tau\\,\\ln\\frac{V_0}{v_C}",
                 "tau": "\\tau = \\frac{t}{\\ln(V_0/v_C)}", "vc0": "V_0 = v_C\\,e^{t/\\tau}"}
    size = lambda v: abs(v.get("vc0") or 0.0) + abs(v.get("vf") or 0.0) + abs(v.get("vamp") or 0.0) + 1e-9
    rels.append(Rel("vc", "The capacitor voltage at time t from the solution of the circuit equation",
                    "v_C = v_C(t)", ["t", "vc"] + p,
                    lambda v: (v["vc"] + 2 * size(v), trajectory(eff, v)["at"](v["t"])["vc"] + 2 * size(v)),
                    forms, traj=True))
    rels += [
        Rel("q", "Charge on a capacitor", "q = C\\,v_C", ["q", "C", "vc"], lambda v: (v["q"], v["C"] * v["vc"]),
            {"q": "q = C\\,v_C", "C": "C = \\frac{q}{v_C}", "vc": "v_C = \\frac{q}{C}"}),
        Rel("ec", "Energy stored in a capacitor", "E_C = \\tfrac12 C v_C^2", ["ec", "C", "vc"],
            lambda v: (v["ec"], 0.5 * v["C"] * v["vc"] ** 2), {"ec": "E_C = \\tfrac12 C v_C^2"}),
        Rel("i", "Ohm's law for the series resistance gives the current from the source",
            "i = \\frac{v_{source} - v_C}{R}" if not internal else "i = \\frac{v_{source} - v_C}{R + r}",
            ["i", "vc", "R"] + (["r"] if internal else []) + (["V"] if source == "battery" else [])
            + (["vamp", "freq", "t"] if source == "ac" else []),
            lambda v: (v["i"] * _series(eff, v) + 2 * (abs(v["vc"]) + abs(_source_voltage(eff, v))) + 1e-12,
                       _source_voltage(eff, v) - v["vc"] + 2 * (abs(v["vc"]) + abs(_source_voltage(eff, v))) + 1e-12),
            {"i": "i = \\frac{v_{source} - v_C}{R}" if not internal else "i = \\frac{v_{source} - v_C}{R + r}"}),
    ]
    return rels


def equation(eff):
    source = eff.get("source")
    res = "(R + r)" if eff.get("internal") == "on" else "R"
    right = {"battery": "V_s", "discharge": "0", "ac": "V_a\\cos(2\\pi f t)"}[source]
    if eff.get("leak") == "on":
        return "%s\\left(C\\,\\frac{dv_C}{dt} + \\frac{v_C}{R_L}\\right) + v_C = %s" % (res, right)
    return "%s\\,C\\,\\frac{dv_C}{dt} + v_C = %s" % (res, right)


def derivation(eff):
    source = eff.get("source")
    parts = ["The current through the resistor is the rate at which the capacitor charges, \\(i = C\\,dv_C/dt\\)."]
    if eff.get("internal") == "on":
        parts.append("The source's internal resistance \\(r\\) adds to \\(R\\).")
    if eff.get("leak") == "on":
        parts.append("Part of the current leaks through \\(R_L\\) instead of charging the capacitor.")
    steps = [step("Set up the circuit equation", "Kirchhoff's voltage law around the loop. " + " ".join(parts),
                  [equation(eff)])]
    if source == "ac":
        steps.append(step("Solve the equation",
                          "The equation is first-order and linear. Its solution is a steady oscillation at the "
                          "source frequency plus a transient that dies away in a few time constants.",
                          ["v_C(t) = A\\cos(2\\pi f t - \\varphi) + K\\,e^{-t/\\tau}"]))
    else:
        steps.append(step("Solve the equation",
                          "The equation is first-order and linear, \\(v_C' + v_C/\\tau = V_f/\\tau\\). An "
                          "integrating factor \\(e^{t/\\tau}\\) gives an exponential approach to the final value.",
                          ["v_C(t) = V_f + (V_0 - V_f)\\,e^{-t/\\tau}" if source == "battery"
                           else "v_C(t) = V_0\\,e^{-t/\\tau}"]))
    return steps


def horizon(eff, v):
    if v.get("tau"):
        return 6 * v["tau"]
    return None


def graph(eff, v):
    traj = trajectory(eff, v)
    t_max = max(traj["scale"], (v.get("t") or 0.0) * 1.2)
    ideal = None
    if (eff.get("internal") == "on" or eff.get("leak") == "on") and v.get("R") and v.get("C"):
        source = eff.get("source")
        tau = v["R"] * v["C"]
        if source == "battery" and v.get("V") is not None:
            ideal = response(source, v["V"], v["vc0"], tau)
        elif source == "discharge":
            ideal = response(source, 0.0, v["vc0"], tau)
        elif source == "ac":
            ideal = response(source, 0.0, v["vc0"], tau, v["vamp"], v["freq"])
    marker = None
    if v.get("t") is not None and v.get("vc") is not None:
        marker = {"t": v["t"], "y": v["vc"], "label": "at time t"}
    return {"fn": lambda t: traj["at"](t)["vc"], "ideal": ideal, "t_max": t_max,
            "y_label": "capacitor voltage (V)", "label": "Capacitor voltage",
            "ideal_label": "Ideal source and capacitor", "marker": marker}
