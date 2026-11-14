"""A mass on a spring: m x'' + c x' + k x = F0 cos(w t) (+ m g for a vertical spring)."""
import math

from ..first_order import step
from ..parsing import InputError
from . import units
from .common import (compare, finish, first_crossing, interpolate, knowns_step, method, rk4,
                     sample)
from .spec import field, preset, question, switch

SCENARIO = {
    "id": "spring",
    "title": "Mass on a spring",
    "description": "Anything that bounces back: a car on its suspension, a door closer, a cart on a "
                   "lab spring. Newton's second law with Hooke's law gives a second-order equation.",
    "law": "Newton's second law with Hooke's law",
    "fields": [
        field("m", "Mass", "m", "mass", ("400", "kg"), positive=True),
        field("k", "Spring constant", "k", "spring_constant", ("40", "kN/m"), positive=True),
        field("x0", "Initial displacement", "x_0", "length", ("5", "cm")),
        field("v0", "Initial velocity", "v_0", "speed", ("0", "m/s")),
        field("c", "Damping coefficient", "c", "damping", ("3500", "kg/s"), "effect", positive=True),
        field("F0", "Driving force amplitude", "F_0", "force", ("100", "N"), "effect", positive=True),
        field("w", "Driving frequency", "\\omega", "angular_frequency", ("1.5", "Hz"), "effect",
              positive=True),
        field("g", "Gravity", "g", "acceleration", ("9.81", "m/s2"), "effect", positive=True),
        field("fraction", "Fraction of the starting swing", "p", "dimensionless", ("5", "%"), "param",
              positive=True),
        field("t_query", "Time taken", "t", "time", ("1", "s"), "param", positive=True),
        field("T_meas", "Measured period", "T", "time", ("0.63", "s"), "param", positive=True),
    ],
    "effects": [
        switch("damping", "Damping", "A shock absorber or friction force proportional to speed.", ["c"]),
        switch("driving", "Driving force", "A periodic push F\u2080 cos(\u03c9t), such as an engine "
               "vibration or a bumpy road.", ["F0", "w"]),
        switch("gravity", "Vertical spring (gravity)", "The weight stretches the spring and shifts "
               "the rest position; x is measured downward from the unstretched length.", ["g"]),
    ],
    "questions": [
        question("motion", "Displacement over time"),
        question("frequency", "Natural frequency, period and damping regime", hides=["x0", "v0"]),
        question("decay", "Time for the swing to shrink to a fraction", ["fraction"]),
        question("steady", "Steady-state amplitude and resonance", hides=["x0", "v0"]),
        question("infer_k", "Spring constant from a measured period", ["T_meas"], finds=["k"],
                 hides=["x0", "v0"]),
        question("infer_c", "Damping from a measured decay", ["fraction", "t_query"], finds=["c"],
                 hides=["x0", "v0"]),
    ],
    "presets": [
        preset("car", "Car suspension", "One corner of a car (400 kg) pushed down 5 cm and released.",
               {"m": ("400", "kg"), "k": ("40", "kN/m"), "x0": ("5", "cm"), "v0": ("0", "m/s"),
                "c": ("3500", "kg/s")}, {"damping": "on"}, "motion"),
        preset("door", "Door closer", "A door closer tuned to shut without slamming or bouncing.",
               {"m": ("30", "kg"), "k": ("60", "N/m"), "x0": ("60", "cm"), "v0": ("0", "m/s"),
                "c": ("84.853", "kg/s")}, {"damping": "on"}, "motion"),
        preset("cart", "Lab cart", "A 0.5 kg cart on a 20 N/m spring, pulled 10 cm and released.",
               {"m": ("0.5", "kg"), "k": ("20", "N/m"), "x0": ("10", "cm"), "v0": ("0", "m/s")},
               {}, "motion"),
        preset("washer", "Washing machine", "A 60 kg drum on its mounts, shaken by an unbalanced load.",
               {"m": ("60", "kg"), "k": ("60", "kN/m"), "x0": ("0", "cm"), "v0": ("0", "m/s"),
                "c": ("600", "kg/s"), "F0": ("100", "N"), "w": ("5", "Hz")},
               {"damping": "on", "driving": "on"}, "steady"),
    ],
}


def equation(effects):
    out = "m\\frac{d^2x}{dt^2}"
    if effects.get("damping") == "on":
        out += " + c\\frac{dx}{dt}"
    out += " + k x = "
    right = []
    if effects.get("driving") == "on":
        right.append("F_0\\cos(\\omega t)")
    if effects.get("gravity") == "on":
        right.append("m g")
    return out + (" + ".join(right) if right else "0")


def _regime(gamma, w0, tolerance=1e-9):
    """The damping regime. With the default tolerance this picks the exact form of
    the solution; a looser one gives the name a person would use for it."""
    if gamma == 0:
        return "undamped"
    if abs(gamma - w0) <= tolerance * w0:
        return "critically damped"
    return "underdamped" if gamma < w0 else "overdamped"


def build(m, k, c, F0, w, g, x0, v0):
    """x(t), v(t) and the pieces of the closed-form solution (c, F0, g may be 0)."""
    w0 = math.sqrt(k / m)
    gamma = c / (2 * m)
    regime = _regime(gamma, w0)
    if regime in ("undamped", "underdamped"):
        wd = math.sqrt(w0 * w0 - gamma * gamma)
        h1 = lambda t: math.exp(-gamma * t) * math.cos(wd * t)
        h2 = lambda t: math.exp(-gamma * t) * math.sin(wd * t)
        d1 = lambda t: math.exp(-gamma * t) * (-gamma * math.cos(wd * t) - wd * math.sin(wd * t))
        d2 = lambda t: math.exp(-gamma * t) * (-gamma * math.sin(wd * t) + wd * math.cos(wd * t))
    elif regime == "critically damped":
        wd = 0.0
        h1 = lambda t: math.exp(-gamma * t)
        h2 = lambda t: t * math.exp(-gamma * t)
        d1 = lambda t: -gamma * math.exp(-gamma * t)
        d2 = lambda t: (1 - gamma * t) * math.exp(-gamma * t)
    else:
        wd = 0.0
        root = math.sqrt(gamma * gamma - w0 * w0)
        r1, r2 = -gamma + root, -gamma - root
        h1 = lambda t: math.exp(r1 * t)
        h2 = lambda t: math.exp(r2 * t)
        d1 = lambda t: r1 * math.exp(r1 * t)
        d2 = lambda t: r2 * math.exp(r2 * t)
    X = phi = 0.0
    resonant = False
    if F0:
        if c == 0 and abs(w - w0) <= 1e-9 * w0:
            resonant = True
            amp = F0 / (2 * m * w0)
            p = lambda t: amp * t * math.sin(w0 * t)
            dp = lambda t: amp * (math.sin(w0 * t) + w0 * t * math.cos(w0 * t))
        else:
            X = F0 / math.hypot(k - m * w * w, c * w)
            phi = math.atan2(c * w, k - m * w * w)
            p = lambda t: X * math.cos(w * t - phi)
            dp = lambda t: -X * w * math.sin(w * t - phi)
    else:
        p = lambda t: 0.0
        dp = lambda t: 0.0
    x_eq = m * g / k
    # Solve A h1(0) + B h2(0) = x0 - x_eq - p(0), A h1'(0) + B h2'(0) = v0 - p'(0).
    a11, a12, a21, a22 = h1(0), h2(0), d1(0), d2(0)
    b1, b2 = x0 - x_eq - p(0), v0 - dp(0)
    det = a11 * a22 - a12 * a21
    A = (b1 * a22 - a12 * b2) / det
    B = (a11 * b2 - b1 * a21) / det
    x = lambda t: x_eq + A * h1(t) + B * h2(t) + p(t)
    v = lambda t: A * d1(t) + B * d2(t) + dp(t)
    return {"x": x, "v": v, "w0": w0, "gamma": gamma, "wd": wd, "form": regime,
            "regime": _regime(gamma, w0, 2e-3), "x_eq": x_eq,
            "X": X, "phi": phi, "resonant": resonant, "A": A, "B": B}


def _formulas(sol, driven):
    regime = sol["form"]
    lines = ["\\omega_0 = \\sqrt{\\frac{k}{m}} = %s" % units.tex(sol["w0"], "angular_frequency")]
    if regime != "undamped":
        lines.append("\\gamma = \\frac{c}{2m} = %s\\ \\mathrm{s^{-1}}" % units.number_tex(sol["gamma"]))
    if regime == "undamped":
        lines.append("x_h(t) = A\\cos\\omega_0 t + B\\sin\\omega_0 t")
    elif regime == "underdamped":
        lines.append("\\omega_d = \\sqrt{\\omega_0^2 - \\gamma^2} = %s" % units.tex(sol["wd"], "angular_frequency"))
        lines.append("x_h(t) = e^{-\\gamma t}\\left(A\\cos\\omega_d t + B\\sin\\omega_d t\\right)")
    elif regime == "critically damped":
        lines.append("x_h(t) = (A + B t)\\,e^{-\\gamma t}")
    else:
        lines.append("x_h(t) = A e^{r_1 t} + B e^{r_2 t},\\quad r_{1,2} = -\\gamma \\pm \\sqrt{\\gamma^2 - \\omega_0^2}")
    if driven:
        if sol["resonant"]:
            lines.append("x_p(t) = \\frac{F_0}{2 m \\omega_0}\\,t\\sin\\omega_0 t")
        else:
            lines.append("x_p(t) = X\\cos(\\omega t - \\varphi),\\quad "
                         "X = \\frac{F_0}{\\sqrt{(k - m\\omega^2)^2 + (c\\omega)^2}}")
    lines.append("A = %s,\\qquad B = %s" % (units.number_tex(sol["A"]), units.number_tex(sol["B"])))
    return lines


def _simpson(fn, a, b, n=2000):
    h = (b - a) / n
    total = fn(a) + fn(b)
    for i in range(1, n):
        total += fn(a + i * h) * (4 if i % 2 else 2)
    return total * h / 3


def solve(inp):
    q = inp.question
    eq = equation(inp.effects)
    damping, driving, gravity = inp.on("damping"), inp.on("driving"), inp.on("gravity")
    terms = ["\\(k x\\) is the spring force (Hooke's law), pulling back towards the unstretched length."]
    if damping:
        terms.append("\\(c\\,dx/dt\\) is the damping force, proportional to speed and opposing the motion.")
    if driving:
        terms.append("\\(F_0\\cos(\\omega t)\\) is the periodic driving force.")
    if gravity:
        terms.append("\\(m g\\) is the weight, with \\(x\\) measured downward.")
    steps = [step("Set up the equation from Newton's second law",
                  "Mass times acceleration equals the sum of the forces. " + " ".join(terms), [eq])]

    if q == "infer_k":
        return _infer_k(inp, steps, eq)
    if q == "infer_c":
        return _infer_c(inp, steps, eq)

    m = inp.need("m", "set up the equation")
    k = inp.need("k", "set up the equation")
    c = inp.need("c", "include damping") if damping else 0.0
    F0 = inp.need("F0", "include the driving force") if driving else 0.0
    w = inp.need("w", "include the driving force") if driving else 0.0
    g = inp.need("g", "include gravity") if gravity else 0.0
    static = q in ("frequency", "steady")
    x0 = 0.0 if static else inp.need("x0", "start the motion")
    v0 = 0.0 if static else inp.need("v0", "start the motion")
    steps.append(knowns_step(inp, ["m", "k"] + ([] if static else ["x0", "v0"]) + ["c", "F0", "w", "g"]
                             + (["fraction"] if q == "decay" else [])))

    sol = build(m, k, c, F0, w, g, x0, v0)
    w0, gamma, regime = sol["w0"], sol["gamma"], sol["regime"]
    period = 2 * math.pi / w0
    zeta = gamma / w0
    regime_text = {
        "undamped": "There is no damping, so it oscillates for ever at the natural frequency.",
        "underdamped": "The damping ratio \\(\\zeta = %s\\) is below 1 (underdamped): it oscillates "
                       "while the swing dies away." % units.number_tex(zeta, 3),
        "critically damped": "The damping ratio is 1 to within rounding (critically damped): it returns to "
                             "rest as fast as possible without overshooting.",
        "overdamped": "The damping ratio \\(\\zeta = %s\\) is above 1 (overdamped): it creeps back "
                      "to rest without oscillating." % units.number_tex(zeta, 3),
    }[regime]

    if q == "frequency":
        lines = ["\\omega_0 = \\sqrt{\\frac{k}{m}} = %s" % units.tex(w0, "angular_frequency"),
                 "f_0 = \\frac{\\omega_0}{2\\pi} = %s,\\qquad T_0 = \\frac{2\\pi}{\\omega_0} = %s"
                 % (units.tex(w0 / (2 * math.pi), "frequency"), units.tex(period, "time"))]
        steps.append(step("Natural frequency and period",
                          "Without damping or driving the solution is a sine wave with angular "
                          "frequency \\(\\omega_0\\).", lines))
        headline = "Natural frequency %s, period %s" % (units.text(w0 / (2 * math.pi), "frequency"),
                                                        units.text(period, "time"))
        if damping:
            more = ["\\zeta = \\frac{c}{2\\sqrt{k m}} = %s" % units.number_tex(zeta, 4)]
            if regime == "underdamped":
                more.append("\\omega_d = \\omega_0\\sqrt{1 - \\zeta^2} = %s,\\qquad T_d = %s"
                            % (units.tex(sol["wd"], "angular_frequency"),
                               units.tex(2 * math.pi / sol["wd"], "time")))
            more.append("c_{\\text{crit}} = 2\\sqrt{k m} = %s" % units.tex(2 * math.sqrt(k * m), "damping"))
            steps.append(step("Damping ratio and regime", regime_text, more))
            headline += "; %s (damping ratio %s)" % (regime, units.number(zeta, 3))
        answer = lines[1]
        show = build(m, k, c, 0.0, 0.0, 0.0, 1.0, 0.0)
        t_max = 4 * period if regime in ("undamped", "underdamped") else 8 / gamma
        graph = _graph(show, None, t_max, None, "displacement for a unit release")
        ok = abs(m * w0 * w0 - k) < 1e-9 * k
        return finish(SCENARIO, inp, eq, {"label": "Result", "text": headline + "."},
                      [method("formula", "Characteristic equation",
                              "The roots of \\(m r^2 + c r + k = 0\\) give the frequency and the decay rate.",
                              steps, answer, ok, "Checked: these values satisfy the characteristic equation.")],
                      graph)

    if q == "steady":
        if not driving:
            raise InputError("Switch on the driving force to ask about the steady state and resonance.")
        if sol["resonant"]:
            raise InputError("With no damping and the driving frequency equal to the natural frequency "
                             "the amplitude grows without limit, so there is no steady state. Add damping.")
        X, phi = sol["X"], sol["phi"]
        lines = ["X = \\frac{F_0}{\\sqrt{(k - m\\omega^2)^2 + (c\\omega)^2}} = %s" % units.tex(X, "length"),
                 "\\varphi = \\tan^{-1}\\frac{c\\omega}{k - m\\omega^2} = %s" % units.tex(phi, "angle", "deg")]
        steps.append(step("Steady-state response",
                          "After the start-up transient dies away the mass moves at the driving "
                          "frequency, \\(x = X\\cos(\\omega t - \\varphi)\\). Substituting this into "
                          "the equation gives the amplitude and the phase lag.", lines))
        headline = "Steady-state amplitude %s, lagging the force by %s" % (
            units.both(X, "length", inp.units.get("x0")), units.text(phi, "angle", "deg"))
        static_x = F0 / k
        extra = ["\\frac{X}{F_0/k} = %s" % units.number_tex(X / static_x, 3)]
        if damping and w0 * w0 > 2 * gamma * gamma:
            wr = math.sqrt(w0 * w0 - 2 * gamma * gamma)
            xmax = F0 / (2 * m * gamma * math.sqrt(w0 * w0 - gamma * gamma))
            extra += ["\\omega_r = \\sqrt{\\omega_0^2 - 2\\gamma^2} = %s" % units.tex(wr, "angular_frequency", "Hz"),
                      "X_{\\max} = %s" % units.tex(xmax, "length")]
            headline += ". Resonance at %s, where the amplitude peaks at %s" % (
                units.text(wr, "angular_frequency", "Hz"), units.both(xmax, "length", inp.units.get("x0")))
            note_r = "The amplitude is largest when the driving frequency is near the natural frequency."
        elif not damping:
            extra.append("\\omega_r = \\omega_0 = %s" % units.tex(w0, "angular_frequency", "Hz"))
            headline += ". Resonance at %s, where the undamped amplitude is unbounded" % units.text(
                w0, "angular_frequency", "Hz")
            note_r = "Without damping the amplitude grows without limit as the driving frequency approaches the natural frequency."
        else:
            note_r = "The damping is heavy enough that there is no resonance peak: the amplitude only falls as the frequency rises."
        steps.append(step("Resonance", "Compared with the static stretch \\(F_0/k\\), the response is "
                          "magnified by the factor below. " + note_r, extra))
        full = build(m, k, c, F0, w, 0.0, 0.0, 0.0)
        t_max = max(8 * 2 * math.pi / w, (6 / gamma if gamma else 0.0))
        resid = abs((k - m * w * w) * X * math.cos(phi) + c * w * X * math.sin(phi) - F0)
        ok = resid < 1e-9 * F0
        graph = _graph(full, None, t_max, None, "displacement from rest")
        return finish(SCENARIO, inp, eq, {"label": "Result", "text": headline + "."},
                      [method("steady", "Steady-state response",
                              "A damped, driven oscillator settles into motion at the driving frequency.",
                              steps, lines[0], ok,
                              "Checked by substituting the steady-state motion back into the equation.")],
                      graph)

    # Motion and decay questions need the full solution.
    working = [step("Characteristic equation",
                    "Try \\(x = e^{rt}\\) in the equation with the right-hand side set to zero.",
                    ["m r^2 + c r + k = 0" if damping else "m r^2 + k = 0",
                     "r = -\\gamma \\pm \\sqrt{\\gamma^2 - \\omega_0^2}" if damping else "r = \\pm i\\,\\omega_0"])]
    if driving:
        working.append(step("Particular solution",
                            "Guess a cosine and a sine at the driving frequency (undetermined "
                            "coefficients) and match both sides."
                            + (" At resonance that guess already solves the homogeneous equation, "
                               "so it is multiplied by \\(t\\)." if sol["resonant"] else ""),
                            ["x_p = a\\cos\\omega t + b\\sin\\omega t"]))
    if gravity:
        working.append(step("Constant term", "A constant force gives a constant shift.",
                            ["x_{eq} = \\frac{m g}{k}"]))
    working.append(step("Apply the initial conditions",
                        "Set \\(x(0) = x_0\\) and \\(x'(0) = v_0\\) and solve the two equations for \\(A\\) and \\(B\\).",
                        ["A = %s,\\qquad B = %s" % (units.number_tex(sol["A"]), units.number_tex(sol["B"]))]))
    steps.append(step(
        "Solve the differential equation",
        "This is a linear second-order equation with constant coefficients. " + regime_text
        + (" The driving force adds a particular solution at the driving frequency." if driving else "")
        + (" Gravity shifts the rest position to \\(x_{eq} = m g/k = %s\\)." % units.tex(sol["x_eq"], "length")
           if gravity else ""),
        _formulas(sol, driving), working))

    x_fn, v_fn = sol["x"], sol["v"]
    if regime in ("undamped", "underdamped"):
        t_max = 5 * 2 * math.pi / (sol["wd"] or w0)
        if gamma:
            t_max = min(max(t_max, 5 / gamma), 40 * 2 * math.pi / w0) if 5 / gamma < 40 * period else 12 * period
    else:
        slow = gamma - math.sqrt(max(gamma * gamma - w0 * w0, 0.0))
        t_max = 6 / max(slow, 1e-12)
    if driving:
        t_max = max(t_max, 8 * 2 * math.pi / w)

    marker = None
    answer = "x(t) = x_h(t)" + (" + x_p(t)" if driving else "") + (" + x_{eq}" if gravity else "")
    if q == "decay":
        frac = inp.need("fraction", "find that time")
        if not damping:
            raise InputError("Without damping the swing never shrinks. Switch on damping to ask this.")
        if driving:
            raise InputError("With a driving force the motion settles to a steady swing instead of "
                             "dying away. Switch off the driving force to ask this.")
        if not 0 < frac < 1:
            raise InputError("The fraction must be between 0 and 100%.")
        if regime == "underdamped":
            t_event = -math.log(frac) / gamma
            lines = ["e^{-\\gamma t} = p", "t = -\\frac{\\ln p}{\\gamma} = %s" % units.tex(t_event, "time")]
            text = ("The swing is bounded by the envelope \\(e^{-\\gamma t}\\). Set it equal to the "
                    "fraction and solve for the time.")
            cycles = t_event * sol["wd"] / (2 * math.pi)
            tail = ", about %s oscillations" % units.number(cycles, 2)
        else:
            z0 = x0 - sol["x_eq"]
            if z0 == 0:
                raise InputError("Give a non-zero initial displacement to ask how long it takes to return.")
            t_event = first_crossing(lambda t: abs(x_fn(t) - sol["x_eq"]) - frac * abs(z0), 0.0, 1 / gamma, t_max * 50)
            if t_event is None:
                raise InputError("The displacement does not fall to that fraction.")
            lines = ["|x(t) - x_{eq}| = p\\,|x_0 - x_{eq}|", "t = %s" % units.tex(t_event, "time")]
            text = "There is no oscillation, so solve directly for when the displacement has fallen to the fraction."
            tail = ""
        steps.append(step("Solve for the time", text, lines))
        headline = "The swing falls to %s of its starting size after %s%s" % (
            units.text(frac, "dimensionless", "%"), units.text(t_event, "time"), tail)
        answer = lines[-1]
        marker = {"t": t_event, "y": x_fn(t_event), "label": units.text(frac, "dimensionless", "%")}
        t_max = max(t_event * 1.4, t_max * 0.5)
    else:
        if regime == "undamped" and not driving:
            amp = math.hypot(sol["A"], sol["B"])
            headline = "Oscillates with amplitude %s and period %s" % (
                units.both(amp, "length", inp.units.get("x0")), units.text(2 * math.pi / w0, "time"))
        elif regime == "underdamped":
            headline = "Underdamped: oscillates with period %s while the swing halves every %s" % (
                units.text(2 * math.pi / sol["wd"], "time"), units.text(math.log(2) / gamma, "time"))
        elif regime == "undamped":
            headline = "Undamped and driven: natural period %s, driving period %s" % (
                units.text(period, "time"), units.text(2 * math.pi / w, "time"))
        else:
            headline = "%s: returns to rest without oscillating" % regime.capitalize()
        if driving and not sol["resonant"]:
            headline += "; steady-state amplitude %s" % units.both(sol["X"], "length", inp.units.get("x0"))
        if sol["resonant"]:
            headline += "; driven at resonance, so the swing grows steadily"

    # Independent check against direct integration.
    def deriv(t, st):
        return [st[1], (F0 * math.cos(w * t) + m * g - c * st[1] - k * st[0]) / m]
    fastest = max(w0, w, gamma * 2)
    count = int(min(400000, max(4000, t_max * fastest * 200)))
    check = rk4(deriv, [x0, v0], t_max, count)
    times = [t_max * i / 12 for i in range(1, 13)]
    size = max([abs(x_fn(t)) for t in times] + [abs(x0), 1e-12])
    err = compare(x_fn, lambda t: interpolate(check, t, 0), times, size)
    ok = err < 1e-5
    note = ("Checked against a direct step-by-step integration of the equation, and the initial "
            "displacement and velocity are reproduced." if ok and abs(x_fn(0) - x0) < 1e-9 * max(1, abs(x0))
            else "The formula and a direct numerical integration disagree.")

    energy = None
    if damping and not driving:
        z = lambda t: x_fn(t) - sol["x_eq"]
        e0 = 0.5 * m * v0 * v0 + 0.5 * k * z(0) ** 2
        e1 = 0.5 * m * v_fn(t_max) ** 2 + 0.5 * k * z(t_max) ** 2
        lost = _simpson(lambda t: c * v_fn(t) ** 2, 0.0, t_max, 4000)
        energy = {"initial": e0, "final": e1, "dissipated": lost}
        steps.append(step(
            "Energy check",
            "The energy stored in the spring and the motion falls as the damper turns it into heat. "
            "Over the time shown, the energy lost equals the work done against damping, "
            "\\(\\int c\\,v^2\\,dt\\).",
            ["E(0) = %s,\\qquad E(%s) = %s" % (units.tex(e0, "energy"), units.tex(t_max, "time"),
                                                units.tex(e1, "energy")),
             "\\int_0^{t} c\\,v^2\\,dt = %s" % units.tex(lost, "energy")]))

    any_effect = damping or driving or gravity
    ideal = build(m, k, 0.0, 0.0, 0.0, 0.0, x0, v0) if any_effect else None
    if any_effect and q == "motion":
        steps.append(step(
            "Compare with the ideal case",
            "With no damping, driving or gravity it would swing for ever with amplitude %s and "
            "period %s." % (units.both(math.hypot(ideal["A"], ideal["B"]), "length", inp.units.get("x0")),
                            units.text(period, "time"))))
    methods = [method("exact", "Exact solution",
                      "A linear equation with constant coefficients has a closed-form solution.",
                      steps, answer, ok, note)]
    if q == "motion" and not any_effect:
        amp = math.hypot(x0, v0 / w0)
        esteps = [steps[0],
                  step("Use conservation of energy",
                       "With no damping the total energy stays constant. At the turning points the "
                       "mass is momentarily at rest, so all the energy is in the spring.",
                       ["\\tfrac12 k A^2 = \\tfrac12 k x_0^2 + \\tfrac12 m v_0^2",
                        "A = \\sqrt{x_0^2 + \\frac{m v_0^2}{k}} = %s" % units.tex(amp, "length")]),
                  step("Maximum speed", "At the centre all the energy is kinetic.",
                       ["v_{\\max} = A\\sqrt{\\frac{k}{m}} = %s" % units.tex(amp * w0, "speed")])]
        same = abs(amp - math.hypot(sol["A"], sol["B"])) < 1e-9 * max(amp, 1e-12)
        methods.append(method("energy", "Energy method", "Without damping, mechanical energy is conserved.",
                              esteps, "A = %s" % units.tex(amp, "length"), same,
                              "Agrees with the amplitude of the differential-equation solution."))
    graph = _graph(sol, ideal, t_max, marker, "displacement (m)")
    if energy:
        graph["energy"] = energy
    return finish(SCENARIO, inp, eq, {"label": "Result", "text": headline + "."}, methods, graph)


def _graph(sol, ideal, t_max, marker, label):
    curve = sample(sol["x"], t_max, 400)
    values = [p[1] for p in curve] or [0.0]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        lo, hi = lo - 1.0, hi + 1.0
    return {"x_label": "time (s)", "y_label": label if "(" in label else label + " (m)",
            "curve": curve, "curve_label": "With the chosen effects" if ideal else "Displacement",
            "ideal": sample(ideal["x"], t_max, 400) if ideal else [],
            "ideal_label": "No damping or driving", "marker": marker, "t_max": t_max,
            "animation": {"kind": "spring", "points": sample(sol["x"], t_max, 240), "min": lo, "max": hi,
                          "label": "displacement (m)"}}


def _infer_k(inp, steps, eq):
    m = inp.need("m", "work out the spring constant")
    T = inp.need("T_meas", "work out the spring constant")
    c = inp.need("c", "include damping") if inp.on("damping") else 0.0
    gamma = c / (2 * m)
    wd = 2 * math.pi / T
    k = m * (wd * wd + gamma * gamma)
    steps.append(knowns_step(inp, ["m", "T_meas", "c"]))
    lines = ["\\omega = \\frac{2\\pi}{T} = %s" % units.tex(wd, "angular_frequency")]
    if c:
        lines += ["\\omega_d^2 = \\frac{k}{m} - \\gamma^2,\\quad \\gamma = \\frac{c}{2m}",
                  "k = m\\left(\\omega_d^2 + \\gamma^2\\right) = %s" % units.tex(k, "spring_constant")]
    else:
        lines.append("k = m\\,\\omega^2 = \\frac{4\\pi^2 m}{T^2} = %s" % units.tex(k, "spring_constant"))
    steps.append(step("Work back from the period",
                      "The period fixes the angular frequency of the oscillation, and the solution "
                      "of the equation relates that frequency to the spring constant.", lines))
    sol = build(m, k, c, 0.0, 0.0, 0.0, 1.0, 0.0)
    back = 2 * math.pi / (sol["wd"] or sol["w0"])
    ok = abs(back - T) < 1e-9 * T
    headline = "Spring constant: %s" % units.both(k, "spring_constant", inp.units.get("k"))
    return finish(SCENARIO, inp, eq, {"label": "Result", "text": headline + "."},
                  [method("period", "From the period", "The oscillation frequency depends on k and m.",
                          steps, lines[-1], ok,
                          "Checked: a spring of this stiffness gives back the measured period.")],
                  _graph(sol, None, 4 * T, None, "displacement for a unit release"))


def _infer_c(inp, steps, eq):
    m = inp.need("m", "work out the damping")
    frac = inp.need("fraction", "work out the damping")
    t = inp.need("t_query", "work out the damping")
    if not 0 < frac < 1:
        raise InputError("The fraction must be between 0 and 100%.")
    gamma = -math.log(frac) / t
    c = 2 * m * gamma
    steps.append(knowns_step(inp, ["m", "k", "fraction", "t_query"]))
    lines = ["e^{-\\gamma t} = p \\ \\Rightarrow\\ \\gamma = -\\frac{\\ln p}{t} = %s\\ \\mathrm{s^{-1}}"
             % units.number_tex(gamma),
             "c = 2 m \\gamma = %s" % units.tex(c, "damping")]
    steps.append(step("Work back from the decay",
                      "For an underdamped oscillator the swing shrinks like \\(e^{-\\gamma t}\\) with "
                      "\\(\\gamma = c/2m\\). The measured decay gives \\(\\gamma\\), and so \\(c\\).", lines))
    k = inp.get("k")
    headline = "Damping coefficient: %s" % units.both(c, "damping", inp.units.get("c"))
    if k:
        zeta = gamma / math.sqrt(k / m)
        headline += " (damping ratio %s)" % units.number(zeta, 3)
        if zeta >= 1:
            raise InputError("That decay is too fast for an oscillating (underdamped) spring with "
                             "this mass and spring constant, so the envelope formula does not apply.")
    sol = build(m, k or m * (2 * math.pi) ** 2, c, 0.0, 0.0, 0.0, 1.0, 0.0)
    ok = abs(math.exp(-c / (2 * m) * t) - frac) < 1e-9
    return finish(SCENARIO, inp, eq, {"label": "Result", "text": headline + "."},
                  [method("decay", "From the decay", "The rate at which the swing shrinks depends on c and m.",
                          steps, lines[-1], ok,
                          "Checked: this damping gives back the measured decay.")],
                  _graph(sol, None, t * 1.5, {"t": t, "y": sol["x"](t), "label": units.text(frac, "dimensionless", "%")},
                         "displacement for a unit release"))
