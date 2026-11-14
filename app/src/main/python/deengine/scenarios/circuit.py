"""A series circuit built from a resistor, an inductor and a capacitor.

The components present are switches, so one system covers RC, RL, LC and RLC
circuits, with no source, a battery or an AC source:

    L i' + R i + v_C = V(t),      C v_C' = i

The RC and RL cases use the first-order relations in rc.py and rl.py; the
circuits with both L and C are second order and are handled here.
"""
import math

from ..first_order import step
from ..parsing import InputError
from . import rc, rl
from .common import numeric, oscillator
from .solver import Rel, Skip
from .spec import choice, field, preset, switch

_R, _L, _C = ("res", ["on"]), ("ind", ["on"]), ("cap", ["on"])
_BATT, _AC = ("source", ["battery"]), ("source", ["ac"])

MODEL = {
    "id": "circuit", "title": "Series circuit", "animation": "capacitor",
    "description": "Resistor, inductor and capacitor in series, with no source, a battery or AC.",
    "law": "Kirchhoff's voltage law: the voltages around the loop add up to the source voltage.",
    "fields": [
        field("R", "Resistance", "R", "resistance", (None, "ohm"), positive=True, typical=1e3, show=[_R]),
        field("L", "Inductance", "L", "inductance", (None, "mH"), positive=True, typical=1e-2, show=[_L]),
        field("C", "Capacitance", "C", "capacitance", (None, "uF"), positive=True, typical=1e-6, show=[_C]),
        field("vc0", "Initial capacitor voltage", "V_0", "voltage", (None, "V"), "initial", show=[_C]),
        field("i0", "Initial current", "I_0", "current", (None, "A"), "initial", typical=0.1, show=[_L]),
        field("V", "Battery voltage", "V_s", "voltage", (None, "V"), "effect", typical=10),
        field("vf", "Final capacitor voltage", "V_f", "voltage", (None, "V"), "effect", typical=10, show=[_C]),
        field("qf", "Final charge", "Q_f", "charge", (None, "C"), "effect", typical=1e-5, show=[_C]),
        field("if_", "Final current", "I_f", "current", (None, "A"), "effect", typical=0.1,
              show=[_L], hide=[_C]),
        field("ef", "Final stored energy", "E_f", "energy", (None, "J"), "effect", nonnegative=True, typical=1e-3),
        field("vamp", "AC source amplitude", "V_a", "voltage", (None, "V"), "effect", positive=True),
        field("freq", "AC source frequency", "f", "frequency", (None, "Hz"), "effect", positive=True, typical=1e3),
        field("aac", "Steady capacitor voltage amplitude", "A", "voltage", (None, "V"), "effect", positive=True,
              show=[_C], hide=[_L]),
        field("phase", "Phase lag of the capacitor voltage", "\\varphi", "angle", (None, "deg"), "effect",
              bounds=(0.0, math.pi / 2), show=[_C], hide=[_L]),
        field("fc", "Cutoff frequency", "f_c", "frequency", (None, "Hz"), "effect", positive=True, typical=1e3,
              show=[_C], hide=[_L]),
        field("iac", "Steady current amplitude", "I_a", "current", (None, "A"), "effect", positive=True,
              typical=0.1, show=[_L]),
        field("zimp", "Impedance", "Z", "resistance", (None, "ohm"), "effect", positive=True, typical=1e3,
              show=[_L, _C]),
        field("phi", "Phase of the current behind the source", "\\phi", "angle", (None, "deg"), "effect",
              bounds=(-math.pi / 2, math.pi / 2), show=[_L, _C]),
        field("r", "Internal resistance of the source", "r", "resistance", (None, "ohm"), "effect",
              positive=True, typical=10),
        field("rl", "Leakage resistance", "R_L", "resistance", (None, "Mohm"), "effect", positive=True,
              typical=1e6),
        field("t", "Time", "t", "time", (None, "s"), "moment", nonnegative=True, typical=1e-3),
        field("vc", "Capacitor voltage at that time", "v_C", "voltage", (None, "V"), "moment", show=[_C]),
        field("q", "Charge at that time", "q", "charge", (None, "C"), "moment", typical=1e-5, show=[_C]),
        field("i", "Current at that time", "i", "current", (None, "A"), "moment", typical=1e-3),
        field("vl", "Inductor voltage at that time", "v_L", "voltage", (None, "V"), "moment", show=[_L]),
        field("ec", "Energy in the capacitor at that time", "E_C", "energy", (None, "J"), "moment",
              nonnegative=True, typical=1e-4, show=[_C]),
        field("el", "Energy in the inductor at that time", "E_L", "energy", (None, "J"), "moment",
              nonnegative=True, typical=1e-4, show=[_L]),
        field("tau", "Time constant", "\\tau", "time", (None, "s"), "derived", positive=True, typical=1e-3,
              hide=[_L, _C]),
        field("w0", "Natural angular frequency", "\\omega_0", "angular_frequency", (None, "rad/s"), "derived",
              positive=True, typical=1e4, show=[_L, _C]),
        field("f0", "Natural (resonance) frequency", "f_0", "frequency", (None, "Hz"), "derived", positive=True,
              typical=1e3, show=[_L, _C]),
        field("zeta", "Damping ratio", "\\zeta", "plain", (None, ""), "derived", nonnegative=True,
              show=[_L, _C], hint="Below 1 the circuit rings; 1 is critical; above 1 it is overdamped."),
        field("wd", "Ringing angular frequency", "\\omega_d", "angular_frequency", (None, "rad/s"), "derived", optional=True,
              positive=True, typical=1e4, show=[_L, _C]),
        field("qfac", "Quality factor", "Q", "plain", (None, ""), "derived", positive=True, typical=10,
              show=[_L, _C]),
    ],
    "effects": [
        switch("res", "Resistor", "A resistance R in the loop.", []),
        switch("ind", "Inductor", "An inductance L in the loop.", []),
        switch("cap", "Capacitor", "A capacitance C in the loop.", []),
        choice("source", "Source", "What drives the circuit.",
               [("none", "None", []), ("battery", "Battery", ["V", "vf", "qf", "if_", "ef"]),
                ("ac", "AC", ["vamp", "freq", "aac", "phase", "fc", "iac", "zimp", "phi"])], "none"),
        switch("internal", "Internal resistance", "The source has its own resistance in series.", ["r"]),
        switch("leak", "Leaky capacitor", "A resistance across the capacitor lets charge drain away.", ["rl"]),
    ],
    "presets": [
        preset("flash", "Camera flash", "A 120 µF flash capacitor charging to 300 V through 2 kΩ: when is it at 285 V?",
               {"R": ("2", "kohm"), "C": ("120", "uF"), "V": ("300", "V"), "vc0": ("0", "V"), "vc": ("285", "V"),
                "t": ("", "ms"), "tau": ("", "ms"), "i": ("", "mA"), "q": ("", "mC"), "qf": ("", "mC")},
               {"res": "on", "cap": "on", "source": "battery"}),
        preset("timer", "Timing circuit", "100 kΩ and 10 µF on 5 V: when does the capacitor reach 3.3 V?",
               {"R": ("100", "kohm"), "C": ("10", "uF"), "V": ("5", "V"), "vc0": ("0", "V"), "vc": ("3.3", "V"),
                "i": ("", "uA"), "q": ("", "uC"), "qf": ("", "uC"), "ec": ("", "uJ"), "ef": ("", "uJ")},
               {"res": "on", "cap": "on", "source": "battery"}),
        preset("touch", "Touch sensor", "A pad charging through 1 MΩ reaches 2.0 V of 3.3 V in 110 µs: its capacitance?",
               {"R": ("1", "Mohm"), "V": ("3.3", "V"), "vc0": ("0", "V"), "t": ("110", "us"), "vc": ("2.0", "V"),
                "C": ("", "pF"), "tau": ("", "us"), "i": ("", "uA"), "q": ("", "nC"), "qf": ("", "nC")},
               {"res": "on", "cap": "on", "source": "battery"}),
        preset("defib", "Defibrillator", "A 32 µF capacitor at 5 kV discharging through 50 Ω of tissue.",
               {"R": ("50", "ohm"), "C": ("32", "uF"), "vc0": ("5", "kV"), "t": ("2", "ms"), "tau": ("", "ms"),
                "vc": ("", "kV"), "q": ("", "mC")}, {"res": "on", "cap": "on", "source": "none"}),
        preset("filter", "Audio low-pass filter", "1 kΩ and 100 nF driven at 5 kHz.",
               {"R": ("1", "kohm"), "C": ("100", "nF"), "vamp": ("1", "V"), "vc0": ("0", "V"), "freq": ("5", "kHz"),
                "tau": ("", "us"), "fc": ("", "kHz")}, {"res": "on", "cap": "on", "source": "ac"}),
        preset("relay", "Relay coil", "A 200 mH, 240 Ω relay on 12 V pulls in at 35 mA: how long does that take?",
               {"L": ("200", "mH"), "R": ("240", "ohm"), "V": ("12", "V"), "i0": ("0", "A"), "i": ("35", "mA"),
                "t": ("", "ms"), "tau": ("", "ms"), "if_": ("", "mA"), "ef": ("", "mJ"), "el": ("", "mJ")},
               {"res": "on", "ind": "on", "source": "battery"}),
        preset("magnet", "Electromagnet", "A 2 H, 4 Ω electromagnet switched onto 48 V.",
               {"L": ("2", "H"), "R": ("4", "ohm"), "V": ("48", "V"), "i0": ("0", "A")},
               {"res": "on", "ind": "on", "source": "battery"}),
        preset("flyback", "Coil switched off", "3 A in a coil falls to 1 A in 2 ms through 15 Ω: its inductance?",
               {"R": ("15", "ohm"), "i0": ("3", "A"), "t": ("2", "ms"), "i": ("1", "A"), "tau": ("", "ms"),
                "L": ("", "mH"), "el": ("", "mJ")}, {"res": "on", "ind": "on", "source": "none"}),
        preset("mains", "Motor winding on AC", "A 50 mH, 10 Ω winding on 325 V peak, 50 Hz mains.",
               {"L": ("50", "mH"), "R": ("10", "ohm"), "vamp": ("325", "V"), "freq": ("50", "Hz"), "i0": ("0", "A"),
                "tau": ("", "ms")}, {"res": "on", "ind": "on", "source": "ac"}),
        preset("lc", "LC oscillator", "A 10 mH coil and a 100 nF capacitor charged to 5 V, with no resistance.",
               {"L": ("10", "mH"), "C": ("100", "nF"), "vc0": ("5", "V"), "i0": ("0", "A"), "f0": ("", "kHz"),
                "t": ("20", "us"), "i": ("", "mA")}, {"ind": "on", "cap": "on", "source": "none"}),
        preset("ring", "Ringing RLC circuit", "10 Ω, 1 mH and 1 µF, the capacitor starting at 10 V.",
               {"R": ("10", "ohm"), "L": ("1", "mH"), "C": ("1", "uF"), "vc0": ("10", "V"), "i0": ("0", "A"),
                "t": ("0.2", "ms"), "f0": ("", "kHz"), "i": ("", "mA")},
               {"res": "on", "ind": "on", "cap": "on", "source": "none"}),
        preset("tuner", "Radio tuning circuit", "Which capacitance tunes a 240 µH coil to 1 MHz, and how sharp is it with 5 Ω?",
               {"R": ("5", "ohm"), "L": ("240", "uH"), "f0": ("1", "MHz"), "C": ("", "pF"), "vc0": ("1", "V"),
                "i0": ("0", "A")},
               {"res": "on", "ind": "on", "cap": "on", "source": "none"}),
    ],
}


def _on(eff, key):
    return eff.get(key) == "on"


def _kind(eff):
    """Which sub-system the switches describe: "rc", "rl" or "rlc" (also LC)."""
    res, ind, cap = _on(eff, "res"), _on(eff, "ind"), _on(eff, "cap")
    if ind and cap:
        return "rlc"
    if cap and res:
        return "rc"
    if ind and res:
        return "rl"
    raise InputError("Switch on the components in the circuit: a resistor with a capacitor (RC), a "
                     "resistor with an inductor (RL), or an inductor with a capacitor (LC or RLC).")


def _sub(eff):
    """The first-order module and the switches in its own terms."""
    kind = _kind(eff)
    source = eff.get("source", "none")
    if kind == "rc":
        return rc, {"source": "discharge" if source == "none" else source,
                    "internal": eff.get("internal", "off"), "leak": eff.get("leak", "off")}
    return rl, {"source": source, "internal": eff.get("internal", "off")}


# --------------------------------------------------------------------------
# Circuits with both an inductor and a capacitor
# --------------------------------------------------------------------------

def _rtot_vars(eff):
    return (["R"] if _on(eff, "res") else []) + (["r"] if _on(eff, "internal") else [])


def _rtot(eff, v):
    return (v["R"] if _on(eff, "res") else 0.0) + (v["r"] if _on(eff, "internal") else 0.0)


def _source_vars(eff):
    return {"battery": ["V"], "ac": ["vamp", "freq"]}.get(eff.get("source"), [])


def _params(eff):
    return ["L", "C", "vc0", "i0"] + _rtot_vars(eff) + _source_vars(eff) + (["rl"] if _on(eff, "leak") else [])


def _vs(eff, v, t):
    if eff.get("source") == "battery":
        return v["V"]
    if eff.get("source") == "ac":
        return v["vamp"] * math.cos(2 * math.pi * v["freq"] * t)
    return 0.0


def _rlc_trajectory(eff, v):
    if any(v.get(x) is None for x in _params(eff)):
        raise Skip()
    L, C, rt = v["L"], v["C"], _rtot(eff, v)
    leak = v["rl"] if _on(eff, "leak") else None
    period = 2 * math.pi * math.sqrt(L * C)
    scale = max(period, 3 * rt * C)

    def deriv(t, s):
        vc, cur = s
        return [(cur - (vc / leak if leak else 0.0)) / C, (_vs(eff, v, t) - rt * cur - vc) / L]

    state0 = [v["vc0"], v["i0"]]
    if leak is None:
        src = eff.get("source")
        osc = oscillator(L, rt, 1 / C, C * v["vc0"], v["i0"],
                         f_const=v["V"] if src == "battery" else 0.0,
                         f0=v["vamp"] if src == "ac" else 0.0,
                         w=2 * math.pi * v["freq"] if src == "ac" else 0.0)

        def at(t):
            q, cur = osc(t)
            return {"vc": q / C, "i": cur}
        return {"at": at, "exact": True, "scale": scale, "ode": (deriv, state0, ["vc", "i"], 2 * scale)}
    key = ("rlc",) + tuple(v[x] for x in _params(eff)) + (eff.get("source"),)
    num = numeric(key, deriv, state0, 4 * scale)

    def at(t):
        vc, cur = num.state(t)
        return {"vc": vc, "i": cur}
    return {"at": at, "exact": False, "scale": scale, "ode": None}


def _size(eff, v):
    """A voltage scale for comparing values near zero."""
    return (abs(v["vc0"]) + abs(v["i0"]) * math.sqrt(v["L"] / v["C"])
            + abs(v.get("V") or 0.0) + abs(v.get("vamp") or 0.0) + 1e-9)


def _rlc_relations(eff):
    p = _params(eff)
    rv = _rtot_vars(eff)
    src = eff.get("source")
    leak = _on(eff, "leak")
    rsym = {0: "0", 1: "R" if _on(eff, "res") else "r", 2: "(R + r)"}[len(rv)]
    rels = [
        Rel("w0", "The natural frequency of an inductor and a capacitor exchanging energy",
            "\\omega_0 = \\frac{1}{\\sqrt{L C}}", ["w0", "L", "C"],
            lambda v: (v["w0"], 1 / math.sqrt(v["L"] * v["C"])),
            {"w0": "\\omega_0 = \\frac{1}{\\sqrt{L C}}", "L": "L = \\frac{1}{\\omega_0^2 C}",
             "C": "C = \\frac{1}{\\omega_0^2 L}"}),
        Rel("f0", "Frequency in cycles per second", "f_0 = \\frac{\\omega_0}{2\\pi}", ["f0", "w0"],
            lambda v: (v["f0"], v["w0"] / (2 * math.pi)),
            {"f0": "f_0 = \\frac{\\omega_0}{2\\pi}", "w0": "\\omega_0 = 2\\pi f_0"}),
        Rel("q", "Charge on a capacitor", "q = C\\,v_C", ["q", "C", "vc"], lambda v: (v["q"], v["C"] * v["vc"]),
            {"q": "q = C\\,v_C", "vc": "v_C = \\frac{q}{C}", "C": "C = \\frac{q}{v_C}"}),
        Rel("ec", "Energy stored in a capacitor", "E_C = \\tfrac12 C v_C^2", ["ec", "C", "vc"],
            lambda v: (v["ec"], 0.5 * v["C"] * v["vc"] ** 2), {"ec": "E_C = \\tfrac12 C v_C^2"}),
        Rel("el", "Energy stored in an inductor", "E_L = \\tfrac12 L i^2", ["el", "L", "i"],
            lambda v: (v["el"], 0.5 * v["L"] * v["i"] ** 2), {"el": "E_L = \\tfrac12 L i^2"}),
    ]
    if rv:
        rels += [
            Rel("zeta", "The damping ratio compares the resistance with the critical value",
                "\\zeta = \\frac{%s}{2}\\sqrt{\\frac{C}{L}}" % rsym, ["zeta", "L", "C"] + rv,
                lambda v: (v["zeta"], _rtot(eff, v) / 2 * math.sqrt(v["C"] / v["L"])),
                {"zeta": "\\zeta = \\frac{%s}{2}\\sqrt{\\frac{C}{L}}" % rsym}),
            Rel("wd", "Resistance lowers the ringing frequency (underdamped case only)",
                "\\omega_d = \\omega_0\\sqrt{1 - \\zeta^2}", ["wd", "w0", "zeta"],
                lambda v: (v["wd"], v["w0"] * math.sqrt(1 - v["zeta"] ** 2)),
                {"wd": "\\omega_d = \\omega_0\\sqrt{1 - \\zeta^2}"}),
            Rel("qfac", "The quality factor measures how sharp the resonance is", "Q = \\frac{1}{2\\zeta}",
                ["qfac", "zeta"], lambda v: (v["qfac"], 1 / (2 * v["zeta"])),
                {"qfac": "Q = \\frac{1}{2\\zeta}", "zeta": "\\zeta = \\frac{1}{2Q}"}),
        ]
    if src == "battery":
        gain = (lambda v: v["rl"] / (v["rl"] + _rtot(eff, v))) if leak else (lambda v: 1.0)
        rels += [
            Rel("vf", "Once the current has died away the capacitor holds the source voltage"
                + (" (less the share lost across the leak)" if leak else ""),
                "V_f = V_s" + ("\\,\\frac{R_L}{R_L + %s}" % rsym if leak else ""),
                ["vf", "V"] + (rv + ["rl"] if leak else []),
                lambda v: (v["vf"], v["V"] * gain(v)), {"vf": "V_f = V_s" + ("\\,\\frac{R_L}{R_L + %s}" % rsym if leak else "")}),
            Rel("qf", "Charge on the capacitor at the end", "Q_f = C\\,V_f", ["qf", "C", "vf"],
                lambda v: (v["qf"], v["C"] * v["vf"]), {"qf": "Q_f = C\\,V_f", "C": "C = \\frac{Q_f}{V_f}"}),
            Rel("ef", "Energy stored in the capacitor at the end", "E_f = \\tfrac12 C V_f^2", ["ef", "C", "vf"],
                lambda v: (v["ef"], 0.5 * v["C"] * v["vf"] ** 2), {"ef": "E_f = \\tfrac12 C V_f^2"}),
        ]
    if src == "ac" and not leak:
        react = lambda v: 2 * math.pi * v["freq"] * v["L"] - 1 / (2 * math.pi * v["freq"] * v["C"])
        rels += [
            Rel("zimp", "The impedance combines the resistance with the net reactance",
                "Z = \\sqrt{%s^2 + \\left(\\omega L - \\frac{1}{\\omega C}\\right)^2}" % rsym,
                ["zimp", "freq", "L", "C"] + rv, lambda v: (v["zimp"], math.hypot(_rtot(eff, v), react(v))),
                {"zimp": "Z = \\sqrt{%s^2 + \\left(\\omega L - \\frac{1}{\\omega C}\\right)^2},\\quad \\omega = 2\\pi f" % rsym}),
            Rel("iac", "The steady current amplitude: the source amplitude over the impedance",
                "I_a = \\frac{V_a}{Z}", ["iac", "vamp", "zimp"], lambda v: (v["iac"], v["vamp"] / v["zimp"]),
                {"iac": "I_a = \\frac{V_a}{Z}", "vamp": "V_a = I_a Z", "zimp": "Z = \\frac{V_a}{I_a}"}),
        ]
        if rv:
            rels.append(Rel("phi", "The current lags the source when the inductor dominates and leads when the capacitor does",
                            "\\tan\\phi = \\frac{\\omega L - 1/(\\omega C)}{%s}" % rsym, ["phi", "freq", "L", "C"] + rv,
                            lambda v: (v["phi"] + 2.0, math.atan2(react(v), _rtot(eff, v)) + 2.0),
                            {"phi": "\\phi = \\tan^{-1}\\frac{\\omega L - 1/(\\omega C)}{%s}" % rsym}))
    rels += [
        Rel("vc", "The capacitor voltage at time t from the solution of the circuit equation", "v_C = v_C(t)",
            ["t", "vc"] + p,
            lambda v: (v["vc"] + 2 * _size(eff, v), _rlc_trajectory(eff, v)["at"](v["t"])["vc"] + 2 * _size(eff, v)),
            traj=True),
        Rel("i", "The current at time t from the solution of the circuit equation", "i = i(t)", ["t", "i"] + p,
            lambda v: (v["i"] + 2 * _size(eff, v) * math.sqrt(v["C"] / v["L"]),
                       _rlc_trajectory(eff, v)["at"](v["t"])["i"] + 2 * _size(eff, v) * math.sqrt(v["C"] / v["L"])),
            traj=True),
        Rel("vl", "The inductor takes whatever voltage the other parts of the loop do not",
            "v_L = V(t) - %s\\,i - v_C" % rsym if rv else "v_L = V(t) - v_C",
            ["vl", "t", "i", "vc"] + rv + _source_vars(eff),
            lambda v: (v["vl"] + 2 * abs(v["vc"]) + 1e-9, _vs(eff, v, v["t"]) - _rtot(eff, v) * v["i"] - v["vc"]
                       + 2 * abs(v["vc"]) + 1e-9),
            {"vl": "v_L = V(t) - %s\\,i - v_C" % rsym if rv else "v_L = V(t) - v_C"}),
    ]
    return rels


def _rlc_equation(eff):
    rv = _rtot_vars(eff)
    rsym = {0: "", 1: "R" if _on(eff, "res") else "r", 2: "(R + r)"}[len(rv)]
    src = {"battery": "V_s", "ac": "V_a\\cos(2\\pi f t)"}.get(eff.get("source"), "0")
    loop = "L\\,\\frac{di}{dt} + %sv_C = %s" % ((rsym + "\\,i + ") if rsym else "", src)
    cap = "C\\,\\frac{dv_C}{dt} = i" + (" - \\frac{v_C}{R_L}" if _on(eff, "leak") else "")
    return loop + ",\\qquad " + cap


def _rlc_derivation(eff):
    rv = _rtot_vars(eff)
    text = ("Kirchhoff's voltage law round the loop: the inductor's voltage \\(L\\,di/dt\\)"
            + (", the resistor's \\(R\\,i\\)" if rv else "")
            + " and the capacitor's \\(v_C\\) add up to the source voltage. The current is the rate "
              "at which the capacitor charges")
    text += ", less what drains through the leak." if _on(eff, "leak") else "."
    steps = [step("Set up the circuit equation", text, [_rlc_equation(eff)])]
    if _on(eff, "leak"):
        steps.append(step("Solve the equation",
                          "With the leak the two equations are solved together numerically "
                          "(fourth-order Runge-Kutta)."))
        return steps
    lines = ["L\\,q'' + R\\,q' + \\frac{q}{C} = V(t),\\qquad q = C\\,v_C" if rv
             else "L\\,q'' + \\frac{q}{C} = V(t),\\qquad q = C\\,v_C"]
    if rv:
        lines.append("\\zeta < 1:\\ q_h = e^{-\\zeta\\omega_0 t}(A\\cos\\omega_d t + B\\sin\\omega_d t)")
        lines.append("\\zeta = 1:\\ q_h = (A + B t)\\,e^{-\\omega_0 t},\\qquad "
                     "\\zeta > 1:\\ q_h = A e^{r_1 t} + B e^{r_2 t}")
    else:
        lines.append("q = A\\cos\\omega_0 t + B\\sin\\omega_0 t + q_p")
    steps.append(step(
        "Solve the equation",
        "Written for the charge this is a second-order linear equation with constant coefficients, "
        "the same as a mass on a spring (L plays the mass, R the damping, 1/C the spring). Its "
        "characteristic equation \\(L r^2 + R r + 1/C = 0\\) decides whether the circuit rings; the "
        "source adds a particular solution, and the constants come from the initial voltage and current.",
        lines))
    return steps


# --------------------------------------------------------------------------
# The interface the solver uses
# --------------------------------------------------------------------------

def relations(eff):
    if _kind(eff) == "rlc":
        return _rlc_relations(eff)
    mod, sub = _sub(eff)
    return mod.relations(sub)


def trajectory(eff, v):
    try:
        kind = _kind(eff)
    except InputError:
        raise Skip()
    if kind == "rlc":
        return _rlc_trajectory(eff, v)
    mod, sub = _sub(eff)
    return mod.trajectory(sub, v)


def baseline(eff):
    """The same circuit with an ideal source and an ideal capacitor."""
    base = dict(eff, internal="off", leak="off")
    return None if base == eff else base


def equation(eff):
    try:
        kind = _kind(eff)
    except InputError:
        return "\\text{switch on the components in the circuit}"
    if kind == "rlc":
        return _rlc_equation(eff)
    mod, sub = _sub(eff)
    return mod.equation(sub)


def derivation(eff):
    if _kind(eff) == "rlc":
        return _rlc_derivation(eff)
    mod, sub = _sub(eff)
    return mod.derivation(sub)


def horizon(eff, v):
    try:
        kind = _kind(eff)
    except InputError:
        return None
    if kind == "rlc":
        if v.get("L") and v.get("C"):
            rt = sum(v.get(x) or 0.0 for x in _rtot_vars(eff))
            return 6 * max(2 * math.pi * math.sqrt(v["L"] * v["C"]), 3 * rt * v["C"])
        return None
    mod, sub = _sub(eff)
    return mod.horizon(sub, v)


def graph(eff, v):
    if _kind(eff) != "rlc":
        mod, sub = _sub(eff)
        return mod.graph(sub, v)
    traj = _rlc_trajectory(eff, v)
    rt = _rtot(eff, v)
    t_max = max(5 * traj["scale"], (v.get("t") or 0.0) * 1.2)
    if rt > 0:
        t_max = max(min(t_max, max(8 * 2 * v["L"] / rt, 3 * traj["scale"])), (v.get("t") or 0.0) * 1.2)
    ideal = None
    if _on(eff, "internal") or _on(eff, "leak"):
        plain = dict(eff, internal="off", leak="off")
        try:
            base = _rlc_trajectory(plain, v)
            ideal = lambda t: base["at"](t)["vc"]
        except Skip:
            ideal = None
    marker = None
    if v.get("t") is not None and v.get("vc") is not None:
        marker = {"t": v["t"], "y": v["vc"], "label": "at time t"}
    return {"fn": lambda t: traj["at"](t)["vc"], "ideal": ideal, "t_max": t_max,
            "y_label": "capacitor voltage (V)", "label": "Capacitor voltage",
            "ideal_label": "Without internal resistance or leakage", "marker": marker}
