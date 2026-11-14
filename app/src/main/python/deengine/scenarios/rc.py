"""An RC circuit: Kirchhoff's voltage law gives a first-order equation for the capacitor voltage."""
import math

from ..first_order import step
from ..parsing import InputError
from . import units
from .common import (compare, finish, interpolate, knowns_step, linear_working, method, rk4,
                     sample)
from .spec import choice, field, preset, question, switch

SCENARIO = {
    "id": "rc",
    "title": "RC circuit",
    "description": "A capacitor charging or discharging through a resistor: a camera flash, a timer, "
                   "a touch sensor. Kirchhoff's voltage law gives a first-order equation.",
    "law": "Kirchhoff's voltage law",
    "fields": [
        field("R", "Resistance", "R", "resistance", ("2", "kohm"), positive=True),
        field("C", "Capacitance", "C", "capacitance", ("120", "uF"), positive=True),
        field("V", "Source voltage", "V_s", "voltage", ("300", "V"),
              hint="The battery voltage, or the peak voltage of an AC source"),
        field("vc0", "Initial capacitor voltage", "V_0", "voltage", ("0", "V")),
        field("freq", "Source frequency", "f", "frequency", ("50", "Hz"), "effect", positive=True),
        field("r_int", "Internal resistance of the source", "r", "resistance", ("5", "ohm"), "effect",
              nonnegative=True),
        field("r_leak", "Leakage resistance", "R_L", "resistance", ("1", "Mohm"), "effect", positive=True),
        field("fraction", "Fraction of the full change", "p", "dimensionless", ("95", "%"), "param",
              positive=True),
        field("v_target", "Target voltage", "V_c", "voltage", ("200", "V"), "param"),
        field("t_query", "Measured time", "t", "time", ("0.72", "s"), "param", positive=True),
    ],
    "effects": [
        choice("source", "Source", "What drives the circuit.",
               [("battery", "Battery (charging)", []), ("discharge", "None (discharging)", []),
                ("ac", "AC source", ["freq"])], "battery"),
        switch("internal", "Internal resistance", "A real source has some resistance of its own, in "
               "series with R.", ["r_int"]),
        switch("leak", "Leaky capacitor", "A real capacitor slowly leaks charge through a large "
               "resistance in parallel with it.", ["r_leak"]),
    ],
    "questions": [
        question("motion", "Voltage and current over time"),
        question("tau", "Time constant", hides=["V", "vc0"]),
        question("fraction", "Time to reach a fraction of the full change", ["fraction"]),
        question("target", "Time to reach a given voltage", ["v_target"]),
        question("energy", "Energy stored and energy lost as heat"),
        question("steady", "AC steady state: amplitude, phase and cutoff", hides=["vc0"]),
        question("infer_r", "Resistance from a measured charging time", ["fraction", "t_query"],
                 finds=["R"], hides=["V", "vc0"]),
        question("infer_c", "Capacitance from a measured charging time", ["fraction", "t_query"],
                 finds=["C"], hides=["V", "vc0"]),
    ],
    "presets": [
        preset("flash", "Camera flash", "A 120 \u00b5F flash capacitor charged to 300 V through 2 k\u03a9.",
               {"R": ("2", "kohm"), "C": ("120", "uF"), "V": ("300", "V"), "vc0": ("0", "V")},
               {"source": "battery"}, "fraction"),
        preset("timer", "Timing circuit", "A 5 V timer: how long until the capacitor reaches 3.3 V?",
               {"R": ("100", "kohm"), "C": ("10", "uF"), "V": ("5", "V"), "vc0": ("0", "V"),
                "v_target": ("3.3", "V")}, {"source": "battery"}, "target"),
        preset("touch", "Touch sensor", "A 100 pF touch pad charged through 1 M\u03a9 from 3.3 V.",
               {"R": ("1", "Mohm"), "C": ("100", "pF"), "V": ("3.3", "V"), "vc0": ("0", "V")},
               {"source": "battery"}, "tau"),
        preset("defib", "Defibrillator", "A 32 \u00b5F capacitor at 5 kV discharging through a 50 \u03a9 chest.",
               {"R": ("50", "ohm"), "C": ("32", "uF"), "V": ("0", "V"), "vc0": ("5", "kV")},
               {"source": "discharge"}, "energy"),
        preset("filter", "Audio low-pass filter", "A 1 k\u03a9, 100 nF filter fed with a 5 kHz tone.",
               {"R": ("1", "kohm"), "C": ("100", "nF"), "V": ("1", "V"), "vc0": ("0", "V"),
                "freq": ("5", "kHz")}, {"source": "ac"}, "steady"),
    ],
}


def equation(effects):
    source = effects.get("source", "battery")
    resistance = "(R + r)" if effects.get("internal") == "on" and source != "discharge" else "R"
    left = "%s\\,C\\frac{dV_c}{dt} + V_c" % resistance
    if effects.get("leak") == "on":
        left = "%s\\,C\\frac{dV_c}{dt} + \\left(1 + \\frac{%s}{R_L}\\right)V_c" % (resistance, resistance.strip("()"))
    right = {"battery": "V_s", "discharge": "0", "ac": "V_s\\cos(2\\pi f t)"}[source]
    return "%s = %s" % (left, right)


def circuit(R, C, r_int, r_leak, source):
    """Series resistance, time constant and DC gain for the chosen effects."""
    Rt = R + (r_int if source != "discharge" else 0.0)
    if r_leak:
        Rp = Rt * r_leak / (Rt + r_leak)
    else:
        Rp = Rt
    return {"Rt": Rt, "Rp": Rp, "tau": Rp * C, "gain": Rp / Rt}


def build(R, C, V, vc0, r_int, r_leak, source, w):
    """V_c(t), the current in the series resistor i(t), and the circuit constants."""
    cc = circuit(R, C, r_int, r_leak, source)
    tau, Rt, gain = cc["tau"], cc["Rt"], cc["gain"]
    if source == "ac":
        amp = V * gain / math.sqrt(1 + (w * tau) ** 2)
        phi = math.atan(w * tau)
        k0 = vc0 - amp * math.cos(phi)
        vc = lambda t: amp * math.cos(w * t - phi) + k0 * math.exp(-t / tau)
        vs = lambda t: V * math.cos(w * t)
        final = 0.0
    else:
        final = V * gain if source == "battery" else 0.0
        amp = phi = 0.0
        vc = lambda t: final + (vc0 - final) * math.exp(-t / tau)
        vs = (lambda t: V) if source == "battery" else None
    if vs is None:
        current = lambda t: -vc(t) / Rt          # the capacitor drives current back through R
    else:
        current = lambda t: (vs(t) - vc(t)) / Rt
    cc.update({"vc": vc, "i": current, "vs": vs, "final": final, "amp": amp, "phi": phi})
    return cc


def _simpson(fn, a, b, n=4000):
    h = (b - a) / n
    total = fn(a) + fn(b)
    for i in range(1, n):
        total += fn(a + i * h) * (4 if i % 2 else 2)
    return total * h / 3


def solve(inp):
    q = inp.question
    source = inp.effects["source"]
    internal = inp.on("internal") and source != "discharge"
    leak = inp.on("leak")
    eq = equation(inp.effects)
    terms = ["\\(V_c = q/C\\) is the voltage across the capacitor.",
             "The current is \\(i = C\\,dV_c/dt\\)" + (" plus the leakage current \\(V_c/R_L\\)" if leak else "")
             + ", and it drops a voltage \\(i%s\\) across the resistance." % ("(R + r)" if internal else "R")]
    if source == "battery":
        terms.append("The battery supplies the constant voltage \\(V_s\\).")
    elif source == "ac":
        terms.append("The source supplies \\(V_s\\cos(2\\pi f t)\\).")
    else:
        terms.append("There is no source: the capacitor discharges through the resistor.")
    steps = [step("Set up the equation from Kirchhoff's voltage law",
                  "Going round the loop, the voltage drops add up to the source voltage. " + " ".join(terms),
                  [eq])]

    if q in ("infer_r", "infer_c"):
        return _infer(inp, steps, eq, q, source, internal, leak)

    R = inp.need("R", "set up the equation")
    C = inp.need("C", "set up the equation")
    r_int = inp.need("r_int", "include the internal resistance") if internal else 0.0
    r_leak = inp.need("r_leak", "include the leakage") if leak else 0.0
    w = 2 * math.pi * inp.need("freq", "use an AC source") if source == "ac" else 0.0
    static = q in ("tau", "steady")
    V = 0.0 if source == "discharge" else (1.0 if q == "tau" else inp.need("V", "set up the equation"))
    vc0 = 0.0 if static else inp.need("vc0", "start from a known capacitor voltage")
    if source == "discharge" and not static and vc0 == 0:
        raise InputError("To discharge, the capacitor must start with some voltage: fill in initial "
                         "capacitor voltage.")
    shown = ["R", "C"] + ([] if source == "discharge" or q == "tau" else ["V"]) + ([] if static else ["vc0"]) \
        + ["freq", "r_int", "r_leak"] + {"fraction": ["fraction"], "target": ["v_target"]}.get(q, [])
    visible = {"R", "C", "V", "vc0", "fraction", "v_target"} | ({"freq"} if source == "ac" else set()) \
        | ({"r_int"} if internal else set()) | ({"r_leak"} if leak else set())
    steps.append(knowns_step(inp, [s for s in shown if s in visible]))

    sol = build(R, C, V, vc0, r_int, r_leak, source, w)
    tau, Rt, final = sol["tau"], sol["Rt"], sol["final"]
    vc, cur = sol["vc"], sol["i"]
    tau_formula = ("\\tau = \\frac{%s R_L}{%s + R_L}\\,C" % (("(R + r)" if internal else "R",) * 2) if leak
                   else "\\tau = %s\\,C" % ("(R + r)" if internal else "R"))
    tau_line = "%s = %s" % (tau_formula, units.tex(tau, "time"))

    if q == "tau":
        steps.append(step(
            "Read off the time constant",
            "Dividing the equation by the coefficient of \\(V_c\\) puts it in the form "
            "\\(\\tau\\,dV_c/dt + V_c = \\text{const}\\). The solution changes by a factor \\(e\\) "
            "every \\(\\tau\\)." + (" The leakage resistance acts in parallel with the series "
                                    "resistance, which shortens it." if leak else ""),
            [tau_line, "\\text{63.2\\%% of the change after } \\tau,\\quad \\text{99.3\\%% after } 5\\tau = %s"
             % units.tex(5 * tau, "time")]))
        headline = "Time constant %s; essentially complete after %s" % (
            units.text(tau, "time"), units.text(5 * tau, "time"))
        unit = build(R, C, 1.0, 0.0 if source != "discharge" else 1.0, r_int, r_leak,
                     "battery" if source == "ac" else source, 0.0)
        ok = abs(tau - circuit(R, C, r_int, r_leak, source)["Rp"] * C) < 1e-15 + 1e-12 * tau
        return finish(SCENARIO, inp, eq, {"label": "Result", "text": headline + "."},
                      [method("tau", "From the equation",
                              "The time constant is the coefficient of the derivative once the "
                              "equation is in standard form.", steps, tau_line, ok,
                              "Checked: after one time constant the response has covered 63.2% of its change.")],
                      _graph(unit, None, 5 * tau, {"t": tau, "y": unit["vc"](tau), "label": "\u03c4"},
                             "fraction of the source voltage"))

    if q == "steady":
        if source != "ac":
            raise InputError("Choose the AC source to ask about the steady state, phase and cutoff.")
        amp, phi = sol["amp"], sol["phi"]
        fc = 1 / (2 * math.pi * tau)
        lines = [tau_line,
                 "|V_c| = \\frac{V_s%s}{\\sqrt{1 + (\\omega\\tau)^2}} = %s"
                 % ("\\,R_L/(R + R_L)" if leak else "", units.tex(amp, "voltage")),
                 "\\varphi = \\tan^{-1}(\\omega\\tau) = %s" % units.tex(phi, "angle", "deg"),
                 "f_c = \\frac{1}{2\\pi\\tau} = %s" % units.tex(fc, "frequency")]
        steps.append(step(
            "Steady-state response",
            "Once the start-up transient has died away the capacitor voltage is a cosine at the "
            "source frequency, smaller than the source and lagging behind it. Substituting "
            "\\(V_c = |V_c|\\cos(\\omega t - \\varphi)\\) into the equation gives:", lines))
        ratio = amp / V if V else 0.0
        headline = ("Capacitor voltage amplitude %s (%s of the source), lagging by %s; cutoff frequency %s"
                    % (units.both(amp, "voltage", inp.units.get("V")), units.text(ratio, "dimensionless", "%"),
                       units.text(phi, "angle", "deg"), units.text(fc, "frequency")))
        steps.append(step(
            "What the cutoff means",
            "Below \\(f_c\\) the capacitor follows the source almost fully; above it the output "
            "falls off. At \\(f_c\\) itself the amplitude is \\(1/\\sqrt2 \\approx 70.7\\%%\\) of its "
            "low-frequency value and the lag is 45\u00b0. Here \\(f/f_c = %s\\)."
            % units.number_tex(w / (2 * math.pi) / fc, 3)))
        resid = abs(tau * (-amp * w * math.sin(-phi)) + amp * math.cos(-phi) - V * sol["gain"])
        ok = resid < 1e-9 * max(abs(V), 1e-12)
        t_max = max(5 * tau, 4 * 2 * math.pi / w)
        full = build(R, C, V, 0.0, r_int, r_leak, source, w)
        graph = _graph(full, None, t_max, None, "capacitor voltage (V)")
        graph["ideal"] = sample(full["vs"], t_max, 400)
        graph["ideal_label"] = "Source voltage"
        graph["curve_label"] = "Capacitor voltage"
        return finish(SCENARIO, inp, eq, {"label": "Result", "text": headline + "."},
                      [method("steady", "Steady-state response",
                              "A linear circuit driven by a sine wave settles into a sine wave at the same frequency.",
                              steps, lines[1], ok,
                              "Checked by substituting the steady-state voltage back into the equation.")],
                      graph)

    # Solve the equation for the remaining questions.
    if source == "ac":
        working = [
            step("Write in standard form", "Divide by the coefficient of the derivative.",
                 ["\\frac{dV_c}{dt} + \\frac{V_c}{\\tau} = \\frac{G V_s}{\\tau}\\cos\\omega t"]),
            step("Guess the steady part", "Try a cosine and a sine at the source frequency and match coefficients.",
                 ["V_p = a\\cos\\omega t + b\\sin\\omega t",
                  "a = \\frac{G V_s}{1 + (\\omega\\tau)^2},\\qquad b = \\frac{G V_s\\,\\omega\\tau}{1 + (\\omega\\tau)^2}"]),
            step("Add the transient and apply the initial condition",
                 "The homogeneous solution \\(K e^{-t/\\tau}\\) carries the initial voltage.",
                 ["K = V_0 - a"]),
        ]
    else:
        working = linear_working("V_c", "\\frac{1}{\\tau}", "\\frac{V_\\infty}{\\tau}", "V_\\infty", "V_0")
    if source == "ac":
        formulas = [tau_line, "V_c(t) = |V_c|\\cos(\\omega t - \\varphi) + \\left(V_0 - |V_c|\\cos\\varphi\\right)e^{-t/\\tau}",
                    "|V_c| = %s,\\qquad \\varphi = %s" % (units.tex(sol["amp"], "voltage"),
                                                          units.tex(sol["phi"], "angle", "deg"))]
    else:
        formulas = [tau_line, "V_\\infty = %s" % units.tex(final, "voltage"),
                    "V_c(t) = V_\\infty + (V_0 - V_\\infty)\\,e^{-t/\\tau}",
                    "i(t) = %s" % ("\\frac{V_s - V_c(t)}{%s}" % ("R + r" if internal else "R")
                                   if source == "battery" else "-\\frac{V_c(t)}{R}")]
    steps.append(step(
        "Solve the differential equation",
        "This is a linear first-order equation. The capacitor voltage relaxes exponentially towards "
        "its final value with time constant \\(\\tau\\)."
        + (" With leakage the final value is below the source voltage, because the leak and the "
           "series resistance form a voltage divider." if leak and source == "battery" else "")
        + (" On top of that sits the steady oscillation at the source frequency." if source == "ac" else ""),
        formulas, working))

    t_max = 5 * tau
    if source == "ac":
        t_max = max(t_max, 4 * 2 * math.pi / w)
    marker = None
    answer = formulas[2] if source != "ac" else formulas[1]
    energy = None
    if q in ("fraction", "target"):
        if source == "ac":
            raise InputError("With an AC source the voltage keeps oscillating. Choose the battery or "
                             "discharging to ask for a time.")
        if q == "fraction":
            frac = inp.need("fraction", "find that time")
            if not 0 < frac < 1:
                raise InputError("The fraction must be between 0 and 100%.")
            target = vc0 + frac * (final - vc0)
            lines = ["1 - e^{-t/\\tau} = p", "t = -\\tau\\ln(1 - p) = %s" % units.tex(-tau * math.log(1 - frac), "time")]
            what = "%s of the way to its final value (%s)" % (units.text(frac, "dimensionless", "%"),
                                                              units.both(target, "voltage", inp.units.get("V")))
        else:
            target = inp.need("v_target", "find that time")
            if (target - vc0) * (final - target) <= 0:
                raise InputError("The capacitor voltage goes from %s to %s, so it never reaches %s."
                                 % (units.text(vc0, "voltage"), units.text(final, "voltage"),
                                    units.text(target, "voltage")))
            lines = ["V_c(t) = %s" % units.tex(target, "voltage"),
                     "t = \\tau\\ln\\frac{V_0 - V_\\infty}{V_c - V_\\infty} = %s"
                     % units.tex(tau * math.log((vc0 - final) / (target - final)), "time")]
            what = units.both(target, "voltage", inp.units.get("v_target"))
        t_event = tau * math.log((vc0 - final) / (target - final))
        steps.append(step("Solve for the time",
                          "Set the solution equal to the voltage wanted and take logarithms.", lines))
        headline = "The capacitor reaches %s after %s (%s time constants)" % (
            what, units.text(t_event, "time"), units.number(t_event / tau, 3))
        answer = lines[-1]
        marker = {"t": t_event, "y": target, "label": "target"}
        t_max = max(t_event * 1.4, 3 * tau)
    elif q == "energy":
        if source == "ac":
            raise InputError("Choose the battery or discharging to ask about energy.")
        T = 10 * tau
        supplied = _simpson(lambda t: V * cur(t), 0.0, T) if source == "battery" else 0.0
        heat = _simpson(lambda t: cur(t) ** 2 * Rt, 0.0, T)
        leaked = _simpson(lambda t: vc(t) ** 2 / r_leak, 0.0, T) if leak else 0.0
        stored0, stored1 = 0.5 * C * vc0 * vc0, 0.5 * C * vc(T) ** 2
        energy = {"supplied": supplied, "heat": heat, "leak": leaked,
                  "stored_initial": stored0, "stored_final": stored1}
        lines = ["E_C = \\tfrac12 C V_c^2:\\quad %s \\to %s" % (units.tex(stored0, "energy"),
                                                                units.tex(stored1, "energy")),
                 "E_R = \\int_0^{10\\tau} i^2 %s\\,dt = %s" % ("(R + r)" if internal else "R",
                                                              units.tex(heat, "energy"))]
        if leak:
            lines.append("E_{\\text{leak}} = \\int_0^{10\\tau} \\frac{V_c^2}{R_L}\\,dt = %s" % units.tex(leaked, "energy"))
        if source == "battery":
            lines.append("E_{\\text{source}} = \\int_0^{10\\tau} V_s\\,i\\,dt = %s" % units.tex(supplied, "energy"))
        steps.append(step(
            "Follow the energy",
            "The energy stored in the capacitor is \\(\\tfrac12 C V_c^2\\). The resistor turns "
            "\\(i^2 R\\) into heat every second, so integrate that over the process (taken as "
            "\\(10\\tau\\), by which time it is complete to 0.005%).", lines))
        if source == "battery" and not leak and vc0 == 0:
            steps.append(step(
                "A general result",
                "Charging from empty through any resistor, exactly half of the energy the battery "
                "supplies ends up in the capacitor and half is lost as heat, whatever the resistance.",
                ["E_{\\text{source}} = C V_s^2,\\qquad E_C = E_R = \\tfrac12 C V_s^2"]))
        if source == "battery":
            headline = "The source supplies %s: %s is stored and %s is lost as heat" % (
                units.text(supplied, "energy"), units.text(stored1 - stored0, "energy"),
                units.text(heat + leaked, "energy"))
        else:
            headline = "The capacitor starts with %s; %s is delivered to the resistor as heat" % (
                units.text(stored0, "energy"), units.text(heat, "energy"))
            if leak:
                headline += " and %s is lost through the leak" % units.text(leaked, "energy")
        answer = lines[1]
    else:
        if source == "ac":
            headline = "After the start-up transient (%s) the capacitor voltage swings by %s" % (
                units.text(5 * tau, "time"), units.both(sol["amp"], "voltage", inp.units.get("V")))
        else:
            headline = "The capacitor voltage goes from %s to %s with time constant %s" % (
                units.both(vc0, "voltage", inp.units.get("vc0")), units.both(final, "voltage", inp.units.get("V")),
                units.text(tau, "time"))
            headline += "; the current starts at %s" % units.text(abs(cur(0)), "current")

    # Independent check against direct integration of the circuit equation.
    def deriv(t, st):
        vsrc = sol["vs"](t) if sol["vs"] else 0.0
        if source == "discharge":
            return [-st[0] / (Rt * C) - (st[0] / (r_leak * C) if leak else 0.0)]
        return [(vsrc - st[0]) / (Rt * C) - (st[0] / (r_leak * C) if leak else 0.0)]
    fastest = max(1 / tau, w)
    count = int(min(200000, max(2000, t_max * fastest * 40)))
    check = rk4(deriv, [vc0], t_max, count)
    times = [t_max * i / 10 for i in range(1, 11)]
    size = max(abs(V), abs(vc0), 1e-12)
    err = compare(vc, lambda t: interpolate(check, t, 0), times, size)
    ok = err < 1e-6
    note = ("Checked against a direct step-by-step integration of the circuit equation, and the "
            "initial voltage is reproduced.") if ok else "The formula and a direct numerical integration disagree."
    if energy is not None:
        balance = energy["supplied"] - energy["heat"] - energy["leak"] - (energy["stored_final"] - energy["stored_initial"])
        scale = max(abs(energy["supplied"]), energy["stored_initial"], energy["stored_final"], 1e-300)
        if abs(balance) < 1e-6 * scale:
            note += " The energy books balance: supplied = stored + heat" + (" + leak." if leak else ".")
        else:
            ok, note = False, "The energy supplied does not match the energy stored and dissipated."

    any_effect = internal or leak
    ideal = build(R, C, V, vc0, 0.0, 0.0, source, w) if any_effect else None
    if any_effect and q in ("motion", "fraction", "target"):
        steps.append(step(
            "Compare with the ideal case",
            "With an ideal source and no leakage the time constant would be %s instead of %s, and "
            "the final voltage %s instead of %s."
            % (units.text(ideal["tau"], "time"), units.text(tau, "time"),
               units.text(ideal["final"], "voltage"), units.text(final, "voltage"))))
    graph = _graph(sol, ideal, t_max, marker, "capacitor voltage (V)")
    if energy:
        graph["energy"] = energy
    return finish(SCENARIO, inp, eq, {"label": "Result", "text": headline + "."},
                  [method("exact", "Exact solution",
                          "A linear first-order equation has a closed-form solution.",
                          steps, answer, ok, note)], graph)


def _graph(sol, ideal, t_max, marker, label):
    curve = sample(sol["vc"], t_max, 400)
    values = [p[1] for p in curve] or [0.0]
    lo, hi = min(min(values), 0.0), max(max(values), 0.0)
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return {"x_label": "time (s)", "y_label": label,
            "curve": curve, "curve_label": "With the chosen effects" if ideal else "Capacitor voltage",
            "ideal": sample(ideal["vc"], t_max, 400) if ideal else [],
            "ideal_label": "Ideal source, no leakage", "marker": marker, "t_max": t_max,
            "animation": {"kind": "capacitor", "points": sample(sol["vc"], t_max, 240), "min": lo, "max": hi,
                          "label": "capacitor voltage (V)"}}


def _infer(inp, steps, eq, q, source, internal, leak):
    if source == "ac":
        raise InputError("Choose the battery or discharging: the measurement is the time to cover a "
                         "fraction of the charge or discharge.")
    frac = inp.need("fraction", "work back from the measurement")
    t = inp.need("t_query", "work back from the measurement")
    if not 0 < frac < 1:
        raise InputError("The fraction must be between 0 and 100%.")
    tau = -t / math.log(1 - frac)
    r_int = inp.need("r_int", "include the internal resistance") if internal else 0.0
    r_leak = inp.need("r_leak", "include the leakage") if leak else 0.0
    lines = ["1 - e^{-t/\\tau} = p \\ \\Rightarrow\\ \\tau = -\\frac{t}{\\ln(1 - p)} = %s" % units.tex(tau, "time")]
    if q == "infer_r":
        C = inp.need("C", "work out the resistance")
        Rp = tau / C
        if leak:
            if Rp >= r_leak:
                raise InputError("That charging time is longer than the leakage alone allows; check "
                                 "the leakage resistance.")
            Rt = Rp * r_leak / (r_leak - Rp)
        else:
            Rt = Rp
        R = Rt - r_int
        if R <= 0:
            raise InputError("The internal resistance alone is larger than the total resistance this "
                             "measurement implies.")
        lines.append("R%s = \\frac{\\tau}{C} = %s" % ("_{\\text{total}}" if (leak or internal) else "",
                                                       units.tex(Rp, "resistance")))
        if leak or internal:
            lines.append("R = %s" % units.tex(R, "resistance"))
        fid, value, quantity = "R", R, "resistance"
        steps.append(knowns_step(inp, ["C", "fraction", "t_query", "r_int", "r_leak"]))
    else:
        R = inp.need("R", "work out the capacitance")
        cc = circuit(R, 1.0, r_int, r_leak, source)
        C = tau / cc["Rp"]
        lines.append("C = \\frac{\\tau}{R%s} = %s" % ("_{\\text{eff}}" if (leak or internal) else "",
                                                     units.tex(C, "capacitance")))
        fid, value, quantity = "C", C, "capacitance"
        steps.append(knowns_step(inp, ["R", "fraction", "t_query", "r_int", "r_leak"]))
    steps.append(step(
        "Work back from the measured time",
        "The charge (or discharge) covers the fraction \\(p\\) of its full change in the measured "
        "time. That fixes the time constant, and the time constant fixes the unknown component.", lines))
    sol = build(R, C, 1.0, 0.0 if source == "battery" else 1.0, r_int, r_leak, source, 0.0)
    ok = abs(sol["tau"] - tau) < 1e-9 * tau
    headline = "%s: %s (time constant %s)" % (inp.fields[fid]["label"],
                                             units.both(value, quantity, inp.units.get(fid)),
                                             units.text(tau, "time"))
    return finish(SCENARIO, inp, eq, {"label": "Result", "text": headline + "."},
                  [method("measure", "From the measured time",
                          "The time to cover a given fraction depends only on the time constant.",
                          steps, lines[-1], ok,
                          "Checked: with this value the circuit covers the stated fraction in the measured time.")],
                  _graph(sol, None, max(t * 1.4, 3 * tau), {"t": t, "y": sol["vc"](t), "label": "measured"},
                         "fraction of the source voltage"))
