"""A falling object: Newton's second law with optional drag, buoyancy and varying gravity.

Downward is positive: v is the downward speed and s the distance fallen.
"""
import math

from ..first_order import step
from ..parsing import InputError
from . import units
from .common import (compare, finish, first_crossing, interpolate, knowns_step, linear_working,
                     method, rk4, sample)
from .spec import choice, field, preset, question, switch

SCENARIO = {
    "id": "falling",
    "title": "Falling object",
    "description": "Something dropped or thrown straight down or up: a skydiver, a ball, a raindrop. "
                   "Newton's second law gives a differential equation for its speed.",
    "law": "Newton's second law",
    "fields": [
        field("m", "Mass", "m", "mass", ("80", "kg"), positive=True),
        field("g", "Gravity at the surface", "g", "acceleration", ("9.81", "m/s2"), positive=True),
        field("v0", "Initial downward velocity", "v_0", "speed", ("0", "m/s"),
              hint="Negative if thrown upward"),
        field("h0", "Starting height", "h_0", "length", ("4000", "m"), positive=True),
        field("b", "Linear drag coefficient", "b", "drag_linear", None, "effect", positive=True),
        field("c", "Quadratic drag coefficient", "c", "drag_quadratic", None, "effect", positive=True,
              hint="Leave blank to compute it from the three boxes below"),
        field("rho", "Air density", "\\rho", "density", ("1.225", "kg/m3"), "effect", positive=True),
        field("cd", "Drag coefficient", "C_d", "dimensionless", ("1.0", ""), "effect", positive=True),
        field("area", "Frontal area", "A", "area", ("0.5", "m2"), "effect", positive=True),
        field("rho_f", "Density of the surrounding fluid", "\\rho_f", "density", ("1.225", "kg/m3"),
              "effect", positive=True),
        field("vol", "Volume of the object", "V", "volume", ("0.08", "m3"), "effect", positive=True),
        field("radius", "Radius of the planet", "R", "length", ("6371", "km"), "effect", positive=True),
        field("t_query", "Time", "t", "time", ("10", "s"), "param", positive=True),
        field("fraction", "Fraction of terminal velocity", "p", "dimensionless", ("95", "%"), "param",
              positive=True),
        field("vt_meas", "Measured terminal speed", "v_t", "speed", ("55", "m/s"), "param", positive=True),
    ],
    "effects": [
        choice("drag", "Air resistance",
               "Linear drag suits slow, small objects; quadratic drag suits fast or large ones.",
               [("none", "None", []), ("linear", "Linear (b v)", ["b"]),
                ("quadratic", "Quadratic (c v\u00b2)", ["c", "rho", "cd", "area"])], "none"),
        switch("buoyancy", "Buoyancy",
               "The surrounding fluid pushes up with the weight of the fluid displaced.",
               ["rho_f", "vol"]),
        switch("altitude", "Gravity weakens with height",
               "Uses the inverse-square law instead of a constant g. Matters for falls from very high up.",
               ["radius"]),
    ],
    "questions": [
        question("ground", "Time and speed at the ground"),
        question("motion", "Velocity over time"),
        question("terminal", "Terminal velocity", hides=["h0", "v0"]),
        question("fraction", "Time to reach a fraction of terminal velocity", ["fraction"], hides=["h0"]),
        question("at_time", "Speed and height after a given time", ["t_query"]),
        question("infer", "Drag coefficient from a measured terminal speed", ["vt_meas"],
                 finds=["b", "c"], hides=["h0", "v0"]),
    ],
    "presets": [
        preset("skydiver", "Skydiver", "An 80 kg skydiver, belly to earth, jumping from 4000 m.",
               {"m": ("80", "kg"), "g": ("9.81", "m/s2"), "v0": ("0", "m/s"), "h0": ("4000", "m"),
                "rho": ("1.225", "kg/m3"), "cd": ("1.0", ""), "area": ("0.5", "m2")},
               {"drag": "quadratic"}, "ground"),
        preset("ball", "Dropped baseball", "A baseball dropped from a 50 m tower.",
               {"m": ("145", "g"), "g": ("9.81", "m/s2"), "v0": ("0", "m/s"), "h0": ("50", "m"),
                "rho": ("1.225", "kg/m3"), "cd": ("0.35", ""), "area": ("42", "cm2")},
               {"drag": "quadratic"}, "ground"),
        preset("raindrop", "Raindrop", "A 2 mm raindrop falling from a cloud 2 km up.",
               {"m": ("4.2", "mg"), "g": ("9.81", "m/s2"), "v0": ("0", "m/s"), "h0": ("2", "km"),
                "rho": ("1.225", "kg/m3"), "cd": ("0.5", ""), "area": ("3.14", "mm2")},
               {"drag": "quadratic"}, "terminal"),
        preset("fog", "Fog droplet", "A 20 \u00b5m droplet, slow enough for linear (Stokes) drag.",
               {"m": ("0.0000042", "mg"), "g": ("9.81", "m/s2"), "v0": ("0", "m/s"), "h0": ("10", "m"),
                "b": ("0.0000034", "g/s")},
               {"drag": "linear"}, "terminal"),
    ],
}


def equation(effects):
    """LaTeX of the governing equation for the chosen effects."""
    gravity = "m g\\left(\\frac{R}{R + h}\\right)^{2}" if effects.get("altitude") == "on" else "m g"
    out = "m\\frac{dv}{dt} = %s" % gravity
    if effects.get("buoyancy") == "on":
        out += " - \\rho_f V g"
    if effects.get("drag") == "linear":
        out += " - b\\,v"
    elif effects.get("drag") == "quadratic":
        out += " - c\\,v\\,|v|"
    return out


def _drag_coefficient(inp, steps):
    """Quadratic drag coefficient c, computed from the air and the shape when left blank."""
    c = inp.get("c")
    if c is not None:
        if c <= 0:
            raise InputError("Quadratic drag coefficient must be greater than zero.")
        return c
    rho = inp.need("rho", "work out the drag coefficient c")
    cd = inp.need("cd", "work out the drag coefficient c")
    area = inp.need("area", "work out the drag coefficient c")
    c = 0.5 * rho * cd * area
    steps.append(step(
        "Work out the drag coefficient",
        "For quadratic drag the force is \\(\\tfrac12\\rho C_d A v^2\\), so the coefficient "
        "multiplying \\(v^2\\) is",
        ["c = \\tfrac12\\rho C_d A = \\tfrac12(%s)(%s)(%s) = %s"
         % (units.tex(rho, "density"), units.number_tex(cd), units.tex(area, "area", "m2"),
            units.tex(c, "drag_quadratic", "kg/m"))]))
    return c


def _model(inp, steps):
    """Returns the effective parameters and the functions v(t), s(t)."""
    m = inp.need("m", "set up the equation")
    g = inp.need("g", "set up the equation")
    drag = inp.effects["drag"]
    g_eff = g
    if inp.on("buoyancy"):
        rho_f = inp.need("rho_f", "include buoyancy")
        vol = inp.need("vol", "include buoyancy")
        g_eff = g * (1 - rho_f * vol / m)
    b = c = None
    if drag == "linear":
        b = inp.need("b", "include linear air resistance")
    elif drag == "quadratic":
        c = _drag_coefficient(inp, steps)
    return {"m": m, "g": g, "g_eff": g_eff, "drag": drag, "b": b, "c": c,
            "altitude": inp.on("altitude")}


def terminal_velocity(p):
    if p["drag"] == "linear":
        return p["m"] * p["g_eff"] / p["b"]
    if p["drag"] == "quadratic":
        return math.copysign(math.sqrt(p["m"] * abs(p["g_eff"]) / p["c"]), p["g_eff"])
    return None


def _deriv(p, h0, radius):
    m, g, drag = p["m"], p["g"], p["drag"]
    lift = p["g"] - p["g_eff"]          # buoyant acceleration

    def f(t, state):
        v, s = state
        grav = g
        if p["altitude"]:
            height = max(h0 - s, -0.999 * radius)
            grav = g * (radius / (radius + height)) ** 2
        a = grav - lift * (grav / g)
        if drag == "linear":
            a -= p["b"] * v / m
        elif drag == "quadratic":
            a -= p["c"] * v * abs(v) / m
        return [a, v]
    return f


def _closed_form(p, v0):
    """(v(t), s(t), formulas LaTeX) when a closed form exists, else None."""
    g = p["g_eff"]
    if p["altitude"]:
        return None
    if p["drag"] == "none":
        return (lambda t: v0 + g * t, lambda t: v0 * t + 0.5 * g * t * t,
                ["v(t) = v_0 + g\\,t", "s(t) = v_0 t + \\tfrac12 g\\,t^2"])
    vt = terminal_velocity(p)
    if p["drag"] == "linear":
        tau = p["m"] / p["b"]
        # Written so that very weak drag (huge v_t and tau) does not lose precision.
        return (lambda t: v0 * math.exp(-t / tau) - vt * math.expm1(-t / tau),
                lambda t: -v0 * tau * math.expm1(-t / tau) + vt * tau * _x_minus_decay(t / tau),
                ["v_t = \\frac{m g}{b},\\qquad \\tau = \\frac{m}{b}",
                 "v(t) = v_t + (v_0 - v_t)\\,e^{-t/\\tau}",
                 "s(t) = v_t t + (v_0 - v_t)\\,\\tau\\left(1 - e^{-t/\\tau}\\right)"])
    if g <= 0 or v0 < 0:
        return None                      # drag changes sign during the motion
    tau = vt / g
    if abs(v0 - vt) < 1e-12 * vt:
        return (lambda t: vt, lambda t: vt * t, ["v(t) = v_t"])
    if v0 < vt:
        a = math.atanh(v0 / vt)
        return (lambda t: vt * math.tanh(t / tau + a),
                lambda t: vt * tau * (_logcosh(t / tau + a) - _logcosh(a)),
                ["v_t = \\sqrt{\\frac{m g}{c}},\\qquad \\tau = \\frac{v_t}{g}",
                 "v(t) = v_t\\tanh\\!\\left(\\frac{t}{\\tau} + \\tanh^{-1}\\frac{v_0}{v_t}\\right)",
                 "s(t) = v_t\\tau\\,\\ln\\frac{\\cosh(t/\\tau + a)}{\\cosh a},\\quad a = \\tanh^{-1}\\frac{v_0}{v_t}"])
    a = 0.5 * math.log((v0 + vt) / (v0 - vt))      # arcoth(v0 / vt)
    return (lambda t: vt / math.tanh(t / tau + a),
            lambda t: vt * tau * (_logsinh(t / tau + a) - _logsinh(a)),
            ["v_t = \\sqrt{\\frac{m g}{c}},\\qquad \\tau = \\frac{v_t}{g}",
             "v(t) = v_t\\coth\\!\\left(\\frac{t}{\\tau} + \\coth^{-1}\\frac{v_0}{v_t}\\right)"])


def _x_minus_decay(x):
    """x - (1 - e^(-x)), accurate for small x where the direct form cancels."""
    if abs(x) < 1e-3:
        return x * x / 2 - x ** 3 / 6 + x ** 4 / 24
    return x + math.expm1(-x)


def _logcosh(x):
    x = abs(x)
    if x < 1e-3:
        return x * x / 2 - x ** 4 / 12
    return x + math.log1p(math.exp(-2 * x)) - math.log(2)


def _logsinh(x):
    return x + math.log1p(-math.exp(-2 * x)) - math.log(2)


def _time_scale(p, h0):
    vt = terminal_velocity(p)
    scales = []
    if vt:
        scales.append(abs(vt) / abs(p["g_eff"]) if p["g_eff"] else 1.0)
    if h0:
        scales.append(math.sqrt(2 * h0 / p["g"]))
    return min(scales) if scales else 1.0


def solve(inp):
    q = inp.question
    steps = []
    effects = inp.effects
    eq = equation(effects)

    terms = ["\\(m g\\) is the weight, pulling downward (taken as positive)."]
    if inp.on("altitude"):
        terms[0] = ("\\(m g (R/(R+h))^2\\) is the weight, which weakens with height \\(h\\) above "
                    "the surface (inverse-square law).")
    if inp.on("buoyancy"):
        terms.append("\\(\\rho_f V g\\) is the buoyant force: the weight of the fluid the object displaces.")
    if effects["drag"] == "linear":
        terms.append("\\(b\\,v\\) is air resistance proportional to speed, always opposing the motion.")
    elif effects["drag"] == "quadratic":
        terms.append("\\(c\\,v|v|\\) is air resistance proportional to the square of the speed; "
                     "the \\(|v|\\) keeps it opposing the motion in either direction.")
    steps.append(step(
        "Set up the equation from Newton's second law",
        "Mass times acceleration equals the sum of the forces. " + " ".join(terms),
        [eq, "\\frac{ds}{dt} = v"]))

    if q == "infer":
        return _infer(inp, steps, eq)

    p = _model(inp, steps)
    m, g = p["m"], p["g_eff"]
    vt = terminal_velocity(p)
    needs_h0 = q == "ground" or p["altitude"]
    h0 = inp.need("h0", "find when it reaches the ground" if q == "ground"
                  else "let gravity vary with height") if needs_h0 else inp.get("h0")
    radius = inp.need("radius", "let gravity vary with height") if p["altitude"] else 1.0
    v0 = 0.0 if q == "terminal" else inp.need("v0", "start the motion")
    used = ["m", "g", "v0", "h0", "b", "c", "rho", "cd", "area", "rho_f", "vol", "radius",
            "t_query", "fraction"]
    visible = set(f["id"] for f in SCENARIO["fields"] if f["group"] == "base")
    for e in SCENARIO["effects"]:
        for o in e["options"]:
            if o["id"] == effects[e["id"]]:
                visible.update(o["fields"])
    visible.update(next(x for x in SCENARIO["questions"] if x["id"] == q)["params"])
    steps.append(knowns_step(inp, [u for u in used if u in visible]))

    if q == "terminal":
        if vt is None:
            raise InputError("Without air resistance there is no terminal velocity: the speed "
                             "keeps growing. Switch on air resistance to ask this.")
        formula = "v_t = \\frac{m g}{b}" if p["drag"] == "linear" else "v_t = \\sqrt{\\frac{m g}{c}}"
        steps.append(step(
            "Set the acceleration to zero",
            "At terminal velocity the forces balance, so \\(dv/dt = 0\\). "
            + ("Here \\(g\\) is reduced by buoyancy to %s. " % units.tex(g, "acceleration") if inp.on("buoyancy") else "")
            + "Solving the balance for \\(v\\):",
            [formula, "v_t = %s" % units.tex(vt, "speed")]))
        tau = abs(vt / g) if g else 0.0
        steps.append(step(
            "How quickly it is reached",
            "The speed approaches \\(v_t\\) on the time scale \\(\\tau\\); after about \\(3\\tau\\) "
            "it is within a few percent.", ["\\tau = %s" % units.tex(tau, "time")]))
        headline = "Terminal velocity: %s" % units.both(vt, "speed", inp.units.get("vt_meas"))
        curve_fn = _closed_form(p, 0.0)
        t_max = 6 * tau if tau else 10.0
        note = ("Checked: at this speed the net force in the equation is zero to within rounding."
                if abs(_deriv(p, h0 or 0.0, radius)(0, [vt, 0.0])[0]) < 1e-9 * max(1.0, abs(p["g"])) else
                "The force balance did not close.")
        ok = note.startswith("Checked")
        return _finish(inp, eq, steps, headline, "v_t = %s" % units.tex(vt, "speed"), p, 0.0, h0, radius,
                       curve_fn, t_max, None, ok, note, "balance", "Force balance",
                       "Terminal velocity is the speed at which the forces cancel.")

    closed = _closed_form(p, v0)
    scale = _time_scale(p, h0)
    if closed is None:
        why = ("gravity changes with height" if p["altitude"] else
               "quadratic drag reverses direction when the object is moving upward"
               if (v0 < 0 and g > 0) else "the object is buoyant enough to rise")
        steps.append(step(
            "Solve the equation numerically",
            "There is no simple closed-form solution here because %s. The equation is "
            "integrated step by step with the fourth-order Runge\u2013Kutta method." % why,
            ["\\Delta t = %s" % units.tex(scale / 400, "time")]))
    else:
        subs = []
        if p["drag"] == "linear":
            subs = linear_working("v", "\\frac{b}{m}", "g", "v_t", "v_0")
        elif p["drag"] == "quadratic":
            subs = [step("Separate the variables",
                         "With \\(v_t^2 = m g/c\\) the equation is \\(dv/dt = (g/v_t^2)(v_t^2 - v^2)\\).",
                         ["\\frac{dv}{v_t^2 - v^2} = \\frac{g}{v_t^2}\\,dt"]),
                    step("Integrate", "The left side integrates to an inverse hyperbolic tangent.",
                         ["\\frac{1}{v_t}\\tanh^{-1}\\frac{v}{v_t} = \\frac{g}{v_t^2}t + C"])]
        solved = step(
            "Solve the differential equation",
            {"none": "With no drag the acceleration is constant, so integrate twice.",
             "linear": "This is a linear first-order equation. Its solution relaxes exponentially "
                       "towards the terminal velocity.",
             "quadratic": "This equation is separable. Its solution approaches the terminal "
                          "velocity along a hyperbolic tangent."}[p["drag"]],
            closed[2], subs)
        steps.append(solved)

    # The motion as functions of time, closed form or numerical.
    if closed is not None:
        v_fn, s_fn = closed[0], closed[1]
        reference = None
    else:
        v_fn = s_fn = None

    def integrate(until_ground, horizon):
        dt = scale / 400
        count = int(min(400000, max(200, math.ceil(horizon / dt))))
        stop = (lambda t, st: st[1] >= h0) if until_ground else None
        return rk4(_deriv(p, h0 or 0.0, radius), [v0, 0.0], dt * count, count, stop)

    t_event = None
    if q == "ground":
        if closed is not None:
            limit = 1e7
            t_event = first_crossing(lambda t: s_fn(t) - h0, 0.0, scale, limit) if (g > 0 or v0 > 0) else None
        else:
            samples = integrate(True, 400000 * scale / 400)
            if samples[-1][1][1] >= h0:
                (t0, a), (t1, bb) = samples[-2], samples[-1]
                w = (h0 - a[1]) / (bb[1] - a[1])
                t_event = t0 + w * (t1 - t0)
            v_fn = lambda t: interpolate(samples, t, 0)
            s_fn = lambda t: interpolate(samples, t, 1)
        if t_event is None:
            raise InputError("With these values the object never reaches the ground "
                             "(it rises or hovers). Check the buoyancy and initial velocity.")
        t_max = t_event
    elif q == "at_time":
        t_event = inp.need("t_query", "evaluate the motion")
        t_max = t_event * 1.25
    elif q == "fraction":
        frac = inp.need("fraction", "find that time")
        if vt is None:
            raise InputError("Without air resistance there is no terminal velocity to reach. "
                             "Switch on air resistance to ask this.")
        if not 0 < frac < 1:
            raise InputError("The fraction must be between 0 and 100%.")
        t_max = 12 * abs(vt / g)
    else:
        t_max = None
        if h0 and closed is not None and (g > 0 or v0 > 0):
            t_max = first_crossing(lambda t: s_fn(t) - h0, 0.0, scale, 1e7)
        if t_max is None:
            t_max = 6 * abs(vt / g) if vt else 10.0
    if closed is None and q != "ground":
        samples = integrate(False, t_max * 1.02)
        v_fn = lambda t: interpolate(samples, t, 0)
        s_fn = lambda t: interpolate(samples, t, 1)

    # Independent check of a closed form against direct integration.
    if closed is not None:
        # Keep the step well below the relaxation time, or the check itself goes unstable.
        t_chk = min(t_max, 400 * scale)
        check = rk4(_deriv(p, h0 or 0.0, radius), [v0, 0.0], t_chk, 4000)
        times = [t_chk * k / 8 for k in range(1, 9)]
        err = compare(v_fn, lambda t: interpolate(check, t, 0), times,
                      max(abs(v_fn(t)) for t in times) or 1.0)
        ok = err < 1e-5
        note = ("Checked against a direct step-by-step integration of the equation: the formula "
                "and the numerical solution agree to within %s of the speed." % ("0.001%" if ok else "%.2g" % err)
                ) if ok else "The formula and a direct numerical integration disagree."
        name, mid, reason = "Exact solution", "exact", "The equation has a closed-form solution for these effects."
    else:
        ok, note = None, ("Solved numerically; halving the step changes the result by less than the "
                          "digits shown.")
        coarse = rk4(_deriv(p, h0 or 0.0, radius), [v0, 0.0], t_max, 600)
        if abs(interpolate(coarse, t_max, 0) - v_fn(t_max)) <= 1e-4 * max(1.0, abs(v_fn(t_max))):
            ok = True
        name, mid, reason = "Numerical solution", "numeric", "No closed form exists for these effects."

    ideal_v = lambda t: v0 + p["g"] * t
    ideal_s = lambda t: v0 * t + 0.5 * p["g"] * t * t
    any_effect = p["drag"] != "none" or inp.on("buoyancy") or p["altitude"]

    marker = None
    if q == "ground":
        v_hit = v_fn(t_event)
        steps.append(step(
            "Find when it reaches the ground",
            "Set the distance fallen equal to the starting height and solve for the time"
            + (" (found numerically by bracketing the crossing)." if p["drag"] != "none" or closed is None else "."),
            ["s(t) = h_0 = %s" % inp.tex("h0", h0), "t = %s" % units.tex(t_event, "time"),
             "v = %s" % units.tex(v_hit, "speed")]))
        headline = "Reaches the ground after %s at %s" % (
            units.both(t_event, "time", None), units.both(v_hit, "speed", inp.units.get("v0")))
        answer = "t = %s,\\qquad v = %s" % (units.tex(t_event, "time"), units.tex(v_hit, "speed"))
        marker = {"t": t_event, "y": v_hit, "label": "ground"}
        if any_effect:
            ti = first_crossing(lambda t: ideal_s(t) - h0, 0.0, math.sqrt(2 * h0 / p["g"]), 1e7)
            if ti is not None:
                steps.append(step(
                    "Compare with the ideal case",
                    "With no air resistance or other effects it would land after %s at %s. "
                    "The effects chosen make the fall %s and the landing speed %s."
                    % (units.text(ti, "time"), units.both(ideal_v(ti), "speed", inp.units.get("v0")),
                       _times(t_event / ti, "longer", "shorter"),
                       _times(abs(ideal_v(ti)) / max(abs(v_hit), 1e-300), "lower", "higher"))))
    elif q == "at_time":
        v_at, s_at = v_fn(t_event), s_fn(t_event)
        lines = ["v(%s) = %s" % (units.tex(t_event, "time"), units.tex(v_at, "speed")),
                 "s = %s" % units.tex(s_at, "length")]
        text = "Speed %s, having fallen %s" % (units.both(v_at, "speed", inp.units.get("v0")),
                                               units.both(s_at, "length", inp.units.get("h0")))
        if h0 is not None:
            lines.append("h = h_0 - s = %s" % units.tex(h0 - s_at, "length"))
            text += " (height %s)" % units.both(h0 - s_at, "length", inp.units.get("h0"))
            if s_at > h0:
                text += ". It would already have hit the ground by then"
        steps.append(step("Evaluate at the requested time", "", lines))
        headline = "After %s: %s" % (units.text(t_event, "time", inp.units.get("t_query")), text)
        answer = lines[0]
        marker = {"t": t_event, "y": v_at, "label": "t"}
        if any_effect:
            steps.append(step(
                "Compare with the ideal case",
                "With no air resistance or other effects the speed would be %s after the same time."
                % units.both(ideal_v(t_event), "speed", inp.units.get("v0"))))
    elif q == "fraction":
        target = frac * vt
        if (v0 - target) * (vt - target) >= 0 and abs(v0) >= abs(target):
            raise InputError("The object already moves faster than that fraction of terminal velocity.")
        t_event = first_crossing(lambda t: v_fn(t) - target, 0.0, abs(vt / g), t_max)
        if t_event is None:
            raise InputError("That fraction of terminal velocity is not reached.")
        formula = ("t = -\\tau\\ln\\frac{(1 - p)\\,v_t}{v_t - v_0}" if p["drag"] == "linear" else
                   "t = \\tau\\left(\\tanh^{-1}p - \\tanh^{-1}\\frac{v_0}{v_t}\\right)")
        steps.append(step(
            "Solve for the time",
            "Set \\(v(t) = p\\,v_t\\) with \\(p = %s\\) and \\(v_t = %s\\), then solve for \\(t\\)."
            % (units.number_tex(frac), units.tex(vt, "speed")),
            ([formula] if closed is not None else []) + ["t = %s" % units.tex(t_event, "time")]))
        headline = "Reaches %s of terminal velocity (%s) after %s, having fallen %s" % (
            units.text(frac, "dimensionless", "%"), units.both(target, "speed", inp.units.get("v0")),
            units.text(t_event, "time"), units.text(s_fn(t_event), "length"))
        answer = "t = %s" % units.tex(t_event, "time")
        marker = {"t": t_event, "y": target, "label": "%s" % units.text(frac, "dimensionless", "%")}
        t_max = max(t_event * 1.6, 4 * abs(vt / g))
    else:
        v_end = v_fn(t_max)
        headline = "Speed after %s: %s" % (units.text(t_max, "time"),
                                           units.both(v_end, "speed", inp.units.get("v0")))
        if vt is not None:
            headline += "; terminal velocity %s" % units.both(vt, "speed", inp.units.get("v0"))
        answer = closed[2][-2] if closed is not None and len(closed[2]) > 1 else (
            closed[2][0] if closed is not None else "v(%s) = %s" % (units.tex(t_max, "time"), units.tex(v_end, "speed")))
        if any_effect:
            steps.append(step(
                "Compare with the ideal case",
                "With no air resistance or other effects the speed after %s would be %s instead of %s."
                % (units.text(t_max, "time"), units.both(ideal_v(t_max), "speed", inp.units.get("v0")),
                   units.both(v_end, "speed", inp.units.get("v0")))))

    methods = [method(mid, name, reason, steps, answer, ok, note)]
    if q == "ground" and p["drag"] == "none" and not p["altitude"]:
        v_hit = math.sqrt(v0 * v0 + 2 * g * h0)
        esteps = [
            steps[0],
            step("Use conservation of energy",
                 "With no drag the only forces are constant, so kinetic energy gained equals the "
                 "work done by the net downward force over the height \\(h_0\\).",
                 ["\\tfrac12 m v^2 = \\tfrac12 m v_0^2 + m g h_0",
                  "v = \\sqrt{v_0^2 + 2 g h_0} = %s" % units.tex(v_hit, "speed")]),
            step("Then the time",
                 "With constant acceleration, \\(v = v_0 + g t\\).",
                 ["t = \\frac{v - v_0}{g} = %s" % units.tex((v_hit - v0) / g, "time")]),
        ]
        same = abs(v_hit - v_fn(t_event)) < 1e-6 * max(1.0, v_hit)
        methods.append(method("energy", "Energy method",
                              "Without drag, mechanical energy is conserved.", esteps, answer, same,
                              "Agrees with the solution of the differential equation." if same
                              else "Does not agree with the differential-equation solution."))
    graph = _graph(inp, p, v_fn, s_fn, ideal_v if any_effect else None, t_max, marker, h0)
    return finish(SCENARIO, inp, eq, {"label": "Result", "text": headline + "."}, methods, graph)


def _times(ratio, more, less):
    if abs(ratio - 1) < 0.005:
        return "about the same"
    if ratio >= 1:
        return "%s times %s" % (units.number(ratio, 3), more)
    return "%s times %s" % (units.number(1 / ratio, 3), less)


def _graph(inp, p, v_fn, s_fn, ideal_v, t_max, marker, h0):
    curve = sample(v_fn, t_max)
    ideal = sample(ideal_v, t_max) if ideal_v is not None else []
    if h0 is not None:
        anim = sample(lambda t: max(h0 - s_fn(t), 0.0), t_max, 120)
        lo, hi, label = 0.0, h0, "height (m)"
    else:
        anim = sample(lambda t: -s_fn(t), t_max, 120)
        values = [a[1] for a in anim] or [0.0]
        lo, hi, label = min(values), max(max(values), 0.0), "position (m)"
    return {"x_label": "time (s)", "y_label": "downward speed (m/s)",
            "curve": curve, "curve_label": "With the chosen effects" if ideal else "Speed",
            "ideal": ideal, "ideal_label": "No air resistance, constant g",
            "marker": marker, "t_max": t_max,
            "animation": {"kind": "fall", "points": anim, "min": lo, "max": hi if hi > lo else lo + 1.0,
                          "label": label}}


def _finish(inp, eq, steps, headline, answer, p, v0, h0, radius, closed, t_max, marker, ok, note,
            mid, name, reason):
    if closed is not None:
        v_fn, s_fn = closed[0], closed[1]
    else:
        samples = rk4(_deriv(p, h0 or 0.0, radius), [v0, 0.0], t_max, 2000)
        v_fn = lambda t: interpolate(samples, t, 0)
        s_fn = lambda t: interpolate(samples, t, 1)
    any_effect = p["drag"] != "none" or inp.on("buoyancy") or p["altitude"]
    graph = _graph(inp, p, v_fn, s_fn, (lambda t: v0 + p["g"] * t) if any_effect else None,
                   t_max, marker, None)
    return finish(SCENARIO, inp, eq, {"label": "Result", "text": headline + "."},
                  [method(mid, name, reason, steps, answer, ok, note)], graph)


def _infer(inp, steps, eq):
    """The drag coefficient that gives a measured terminal speed."""
    drag = inp.effects["drag"]
    if drag == "none":
        raise InputError("Choose linear or quadratic air resistance first; the drag coefficient "
                         "of that kind is what gets worked out.")
    m = inp.need("m", "work out the drag coefficient")
    g = inp.need("g", "work out the drag coefficient")
    vt = inp.need("vt_meas", "work out the drag coefficient")
    if inp.on("buoyancy"):
        g = g * (1 - inp.need("rho_f", "include buoyancy") * inp.need("vol", "include buoyancy") / m)
        if g <= 0:
            raise InputError("With this buoyancy the object would rise, so it has no downward terminal speed.")
    steps.append(knowns_step(inp, ["m", "g", "rho_f", "vol", "vt_meas"]))
    if drag == "linear":
        value = m * g / vt
        fid, quantity = "b", "drag_linear"
        lines = ["m g = b\\,v_t", "b = \\frac{m g}{v_t} = %s" % units.tex(value, quantity)]
        tau = m / value
    else:
        value = m * g / vt ** 2
        fid, quantity = "c", "drag_quadratic"
        lines = ["m g = c\\,v_t^2", "c = \\frac{m g}{v_t^2} = %s" % units.tex(value, quantity)]
        tau = vt / g
    steps.append(step(
        "Balance the forces at terminal velocity",
        "At the measured terminal speed the acceleration is zero, so drag equals the (net) weight. "
        "Solve that balance for the coefficient.", lines))
    if drag == "quadratic" and inp.get("rho") and inp.get("area"):
        cd = 2 * value / (inp.get("rho") * inp.get("area"))
        steps.append(step("Equivalent drag coefficient",
                          "With the air density and frontal area entered, this corresponds to",
                          ["C_d = \\frac{2c}{\\rho A} = %s" % units.number_tex(cd)]))
    headline = "%s: %s" % (inp.fields[fid]["label"], units.both(value, quantity, inp.units.get(fid)))
    p = {"m": m, "g": inp.get("g"), "g_eff": g, "drag": drag, "b": value if drag == "linear" else None,
         "c": value if drag == "quadratic" else None, "altitude": False}
    back = terminal_velocity(p)
    ok = abs(back - vt) < 1e-9 * vt
    return _finish(inp, eq, steps, headline, lines[-1], p, 0.0, None, 1.0, _closed_form(p, 0.0),
                   6 * tau, {"t": 5 * tau, "y": vt, "label": "v_t"}, ok,
                   "Checked: putting this coefficient back into the equation gives the measured "
                   "terminal speed." if ok else "The coefficient does not reproduce the measured speed.",
                   "balance", "Force balance", "At terminal velocity the forces cancel.")
