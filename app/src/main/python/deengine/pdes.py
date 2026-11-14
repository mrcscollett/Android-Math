"""Heat, wave and Laplace equations by separation of variables.

The steps give the eigenfunction expansion with coefficients in closed form
when SymPy can integrate them; the graph always uses numerically computed
coefficients, so it does not depend on those integrals succeeding.
"""
import math

import sympy as sp

from . import integrals
from .first_order import step
from .fourier import N, _plot as plot_fx
from .parsing import InputError, parse_expression, parse_number
from .util import run_with_timeout, tex

TERMS = 20          # nonzero terms kept in the plotted series
SCAN = 80           # how far to look for them
SYMBOLIC_TIMEOUT = 4


def _num(value):
    return repr(round(value, 12))


def _callable(expr, var):
    """A plain-Python function of one variable for numerical integration."""
    step_fn = lambda v, *rest: 0.0 if v < 0 else 1.0
    fn = sp.lambdify(var, expr, [{"Heaviside": step_fn}, "math"])

    def safe(value):
        try:
            out = float(fn(value))
        except Exception:
            return 0.0
        return out if math.isfinite(out) else 0.0
    return safe


def _simpson(fn, a, b, n=1200):
    h = (b - a) / n
    total = fn(a) + fn(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * fn(a + i * h)
    return total * h / 3


def _positive(text, label):
    if not (text or "").strip():
        raise InputError("Enter %s." % label)
    value = parse_number(text, label[0].upper() + label[1:])
    if not value.is_positive:
        raise InputError("%s must be a positive number." % (label[0].upper() + label[1:]))
    return value


def _function(text, var, label):
    if not (text or "").strip():
        raise InputError("Enter %s." % label)
    expr, symbols = parse_expression(text, [var])
    if expr.has(sp.DiracDelta):
        raise InputError("An impulse delta(...) cannot be used as initial or boundary data here.")
    return expr, symbols[var]


def _closed_form(f, x, basis, L, factor):
    """factor * integral of f * basis over (0, L) as a formula in n, or None."""
    def work():
        value = sp.integrate(f * basis, (x, 0, L))
        if value.has(sp.Integral) or value.has(sp.zoo, sp.nan, sp.oo):
            return None
        value = sp.simplify(factor * value)
        if isinstance(value, sp.Piecewise):
            value = sp.simplify(value.args[-1][0])
        return value
    if f.has(sp.Heaviside):
        return None
    try:
        return run_with_timeout(work, SYMBOLIC_TIMEOUT, None)
    except Exception:
        return None


def _coefficient_step(name, f, x, basis_tex, basis, L, factor, values, var="x"):
    """The step that works out the series coefficients."""
    formula = _closed_form(f, x, basis, L, factor)
    integral = "%s = %s\\int_0^{%s} %s\\,%s\\,d%s" % (
        name, tex(factor), tex(L), "\\left(%s\\right)" % tex(f) if f.is_Add else tex(f), basis_tex, var)
    latex = [integral]
    substeps = []
    if formula is not None:
        latex.append("%s = %s" % (name, tex(formula)))
        try:
            substeps += integrals.explain(f * basis, x, name)
        except Exception:
            pass
        text = ("The coefficients are the expansion coefficients of the data in the "
                "eigenfunctions, found with their orthogonality.")
    else:
        text = ("The coefficients are the expansion coefficients of the data in the "
                "eigenfunctions. This integral has no simple closed form in \\(n\\) here, so "
                "the values are computed numerically.")
    shown = [(k, v) for k, v in values if abs(v) > 1e-10][:5]
    if shown:
        table = step("First coefficients", "")
        table["table"] = {"headers": ["n", name.replace("_n", "_{n}")],
                          "rows": [[str(k), "%.6g" % v] for k, v in shown]}
        substeps.append(table)
    return step("Find the coefficients", text, latex, substeps)


def _separate_steps(equation_tex, separated_tex, x_problem_tex, eigen_text, eigen_tex):
    return [
        step("Separate the variables",
             "Look for solutions that are a product of a function of each variable. "
             "Substituting and dividing by the product puts each variable on its own side, so "
             "both sides equal a constant, written \\(-\\lambda\\).",
             [equation_tex, separated_tex]),
        step("Solve the eigenvalue problem",
             "The boundary conditions apply to the spatial factor. " + eigen_text,
             [x_problem_tex, eigen_tex]),
    ]


def _answer(method, terms_tex):
    method["steps"].append({"title": "Solution", "text": "", "latex": [terms_tex],
                            "substeps": [], "kind": "answer"})
    kind = {True: "check-ok", False: "check-bad", None: "check-unknown"}[method["verified"]]
    method["steps"].append({"title": "Check", "text": method["verify_note"], "latex": [],
                            "substeps": [], "kind": kind})


def _method(method_id, name, reason, steps, solution_tex, plot, verified, note):
    m = {"id": method_id, "name": name, "applicable": True, "reason": reason, "steps": steps,
         "general_latex": None, "particular_latex": solution_tex,
         "plot": {"general_expr": None, "particular_expr": plot, "points": []},
         "verified": verified, "verify_note": note}
    _answer(m, solution_tex)
    return m


def _verdict(residual, boundary, data_error, scale, what):
    """Combine the three numerical checks into (verified, note)."""
    tolerance = 2e-3 * max(scale, 1e-9)
    if residual is None or boundary is None:
        return None, "Could not be checked automatically."
    if residual > 50 * tolerance or boundary > 50 * tolerance:
        return False, "The series did not satisfy the equation and boundary conditions numerically."
    note = ("Checked numerically: the series satisfies the equation at sample points and the "
            "boundary conditions.")
    if data_error is not None:
        if data_error < 0.03 * max(scale, 1e-9):
            note += " With %d terms it reproduces %s closely." % (TERMS, what)
        else:
            note += (" With %d terms it follows %s only roughly; the series converges slowly "
                     "where the data jumps or does not meet the boundary conditions." % (TERMS, what))
    return True, note


def _range(values, extra=()):
    values = [v for v in list(values) + list(extra) if math.isfinite(v)]
    lo, hi = (min(values), max(values)) if values else (-1.0, 1.0)
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    return lo, hi


# --------------------------------------------------------------------------
# Heat equation
# --------------------------------------------------------------------------

_HEAT = {
    "00": ("Both ends held at 0", "u(0,t) = 0,\\quad u(L,t) = 0",
           "X(0) = 0,\\quad X(L) = 0"),
    "nn": ("Both ends insulated", "u_x(0,t) = 0,\\quad u_x(L,t) = 0",
           "X'(0) = 0,\\quad X'(L) = 0"),
    "0n": ("Left end at 0, right end insulated", "u(0,t) = 0,\\quad u_x(L,t) = 0",
           "X(0) = 0,\\quad X'(L) = 0"),
    "n0": ("Left end insulated, right end at 0", "u_x(0,t) = 0,\\quad u(L,t) = 0",
           "X'(0) = 0,\\quad X(L) = 0"),
    "tt": ("Ends held at constant temperatures", "u(0,t) = T_1,\\quad u(L,t) = T_2",
           "X(0) = 0,\\quad X(L) = 0"),
}


def _family(bc, L, x):
    """(mu_n formula, eigenfunction, eigenfunction TeX, has constant mode) for a condition pair."""
    if bc in ("00", "tt"):
        mu = N * sp.pi / L
        return mu, sp.sin(mu * x), "\\sin\\frac{n\\pi x}{L}", False
    if bc == "nn":
        mu = N * sp.pi / L
        return mu, sp.cos(mu * x), "\\cos\\frac{n\\pi x}{L}", True
    mu = (2 * N - 1) * sp.pi / (2 * L)
    if bc == "0n":
        return mu, sp.sin(mu * x), "\\sin\\frac{(2n-1)\\pi x}{2L}", False
    return mu, sp.cos(mu * x), "\\cos\\frac{(2n-1)\\pi x}{2L}", False


def heat(f_text, L_text, k_text, bc, t1_text, t2_text):
    if bc not in _HEAT:
        raise InputError("Choose the boundary conditions.")
    L = _positive(L_text, "the length L")
    k = _positive(k_text, "the diffusivity k")
    f, x = _function(f_text, "x", "the initial temperature f(x)")
    Lf, kf = float(L), float(k)
    steady = sp.Integer(0)
    if bc == "tt":
        if not (t1_text or "").strip() or not (t2_text or "").strip():
            raise InputError("Enter both end temperatures T1 and T2.")
        T1, T2 = parse_number(t1_text, "T1"), parse_number(t2_text, "T2")
        steady = T1 + (T2 - T1) * x / L
    data = f - steady
    mu, basis, basis_tex, constant_mode = _family(bc, L, x)
    fn = _callable(data, x)

    # Numerical coefficients for the graph.
    mean = _simpson(fn, 0.0, Lf) / Lf if constant_mode else 0.0
    modes = []          # (n, coefficient, mu value)
    for n in range(1, SCAN + 1):
        m = float(mu.subs(N, n))
        wave = (math.sin if basis.func == sp.sin else math.cos)
        c = 2 / Lf * _simpson(lambda s: fn(s) * wave(m * s), 0.0, Lf)
        if abs(c) > 1e-10:
            modes.append((n, c, m))
        if len(modes) >= TERMS:
            break
    trig = "sin" if basis.func == sp.sin else "cos"
    wave = math.sin if trig == "sin" else math.cos
    steady_fn = _callable(steady, x)

    def u(xv, tv):
        total = steady_fn(xv) + mean
        for _, c, m in modes:
            total += c * math.exp(-kf * m * m * tv) * wave(m * xv)
        return total

    parts = []
    steady_plot = plot_fx(steady, x) if steady != 0 else None
    if steady_plot:
        parts.append("(%s)" % steady_plot)
    if constant_mode and abs(mean) > 1e-12:
        parts.append(_num(mean))
    for _, c, m in modes:
        parts.append("%s*exp(-%s*y)*%s(%s*x)" % (_num(c), _num(kf * m * m), trig, _num(m)))
    plot = " + ".join(parts) if parts else "0"

    slowest = modes[0][2] if modes else float(mu.subs(N, 1))
    t_max = math.log(50.0) / (kf * slowest * slowest)
    grid = [u(Lf * i / 40, t_max * j / 20) for i in range(41) for j in range(21)]
    u_min, u_max = _range(grid, [fn(Lf * i / 40) + steady_fn(Lf * i / 40) for i in range(1, 40)])

    name, bc_tex, x_bc = _HEAT[bc]
    equation = "u_t = %s\\,u_{xx},\\qquad 0 < x < %s" % (tex(k), tex(L))
    steps = []
    if bc == "tt":
        steps.append(step(
            "Subtract the steady state",
            "The end temperatures are not zero, so first remove the temperature the rod "
            "settles to: the straight line joining \\(T_1\\) and \\(T_2\\). The remainder "
            "\\(v = u - s\\) satisfies the same equation with both ends at 0.",
            ["s(x) = T_1 + (T_2 - T_1)\\frac{x}{L} = %s" % tex(steady),
             "v(x, 0) = f(x) - s(x) = %s" % tex(sp.simplify(data))]))
    eigen_text = {
        "00": "Only \\(\\lambda > 0\\) gives non-zero solutions, and they are sines.",
        "tt": "Only \\(\\lambda > 0\\) gives non-zero solutions, and they are sines.",
        "nn": "\\(\\lambda = 0\\) gives the constant solution and \\(\\lambda > 0\\) gives cosines.",
        "0n": "Only \\(\\lambda > 0\\) works, and the conditions pick out the odd quarter-waves.",
        "n0": "Only \\(\\lambda > 0\\) works, and the conditions pick out the odd quarter-waves.",
    }[bc]
    lam_tex = ("\\lambda_n = \\left(\\frac{n\\pi}{L}\\right)^2" if bc in ("00", "tt", "nn")
               else "\\lambda_n = \\left(\\frac{(2n-1)\\pi}{2L}\\right)^2")
    steps += _separate_steps(
        "u = X(x)\\,T(t)", "\\frac{T'}{k\\,T} = \\frac{X''}{X} = -\\lambda",
        "X'' + \\lambda X = 0,\\qquad %s" % x_bc, eigen_text,
        "%s,\\qquad X_n = %s%s" % (lam_tex, basis_tex,
                                   ",\\qquad n = 1, 2, 3, \\dots" + (" \\ (\\text{and } X_0 = 1)" if constant_mode else "")))
    steps.append(step(
        "Solve the time equation",
        "Each \\(\\lambda_n\\) gives a time factor that decays; the higher modes die out fastest.",
        ["T' + k\\lambda_n T = 0 \\quad\\Rightarrow\\quad T_n = e^{-k\\lambda_n t}"]))
    lead = ("s(x) + " if bc == "tt" else "") + ("\\frac{a_0}{2} + " if constant_mode else "")
    cname = "a_n" if trig == "cos" else "b_n"
    steps.append(step(
        "Superpose",
        "The equation and boundary conditions are linear and homogeneous, so any sum of "
        "these products is also a solution.",
        ["u(x,t) = %s\\sum_{n=1}^{\\infty} %s\\,e^{-k\\lambda_n t}\\,%s" % (lead, cname, basis_tex)]))
    values = [(n, c) for n, c, _ in modes]
    target = "f(x) - s(x)" if bc == "tt" else "f(x)"
    coefficient = _coefficient_step(cname, data, x, basis_tex, basis, L, 2 / L, values)
    coefficient["text"] = ("Setting \\(t = 0\\) must give \\(%s\\). " % target) + coefficient["text"]
    if constant_mode:
        coefficient["latex"].insert(0, "\\frac{a_0}{2} = \\frac{1}{L}\\int_0^{L} f(x)\\,dx = %s" % _g(mean))
    steps.append(coefficient)

    shown = modes[:3]
    pieces = []
    if bc == "tt":
        pieces.append(tex(steady))
    if constant_mode and abs(mean) > 1e-12:
        pieces.append(_g(mean))
    for n, c, m in shown:
        rate = sp.nsimplify(sp.simplify(k * mu.subs(N, n) ** 2))
        pieces.append("%se^{%s}\\,%s" % (_lead(c), tex(-rate * T_SYM), tex(basis.subs(N, n))))
    solution = ("u(x,t) " + _join(pieces, len(modes) <= 1)) if pieces else "u(x,t) = 0"
    behaviour = {
        "00": "Every term decays, so the temperature tends to 0 everywhere.",
        "0n": "Every term decays, so the temperature tends to 0 everywhere.",
        "n0": "Every term decays, so the temperature tends to 0 everywhere.",
        "nn": "No heat can leave, so the temperature evens out to the average of the initial "
              "temperature, \\(%s\\)." % _g(mean),
        "tt": "The series part decays, so the temperature tends to the steady state \\(s(x)\\).",
    }[bc]
    steps.append(step("Long-time behaviour", behaviour))

    # Checks.
    h = Lf * 1e-3
    worst = 0.0
    for xv in (0.23 * Lf, 0.5 * Lf, 0.81 * Lf):
        for tv in (0.15 * t_max, 0.5 * t_max):
            dt = t_max * 1e-4
            ut = (u(xv, tv + dt) - u(xv, tv - dt)) / (2 * dt)
            uxx = (u(xv + h, tv) - 2 * u(xv, tv) + u(xv - h, tv)) / (h * h)
            worst = max(worst, abs(ut - kf * uxx) * t_max)
    tv = 0.3 * t_max

    def end(point, insulated, wanted):
        if insulated:
            inside = point + (h if point == 0 else -h)
            return abs((u(inside, tv) - u(point, tv)) / h) * Lf
        return abs(u(point, tv) - wanted)
    left_ins, right_ins = bc in ("nn", "n0"), bc in ("nn", "0n")
    edge = max(end(0.0, left_ins, steady_fn(0.0)), end(Lf, right_ins, steady_fn(Lf)))
    data_error = max(abs(u(Lf * i / 20, 0.0) - (fn(Lf * i / 20) + steady_fn(Lf * i / 20)))
                     for i in range(3, 18))
    verified, note = _verdict(worst, edge, data_error, u_max - u_min, "the initial temperature")

    method = _method("separation", "Separation of variables",
                     "The equation and boundary conditions are linear and homogeneous%s, so "
                     "product solutions can be superposed."
                     % (" once the steady state is removed" if bc == "tt" else ""),
                     steps, solution, plot, verified, note)
    input_latex = "%s;\\quad %s;\\quad u(x,0) = %s" % (equation, bc_tex, tex(f))
    if bc == "tt":
        input_latex += ";\\quad T_1 = %s,\\ T_2 = %s" % (tex(T1), tex(T2))
    return _result("heat", input_latex, [method],
                   "Heat equation on a rod of length L. %s. Use the slider on the graph to "
                   "watch the temperature evolve." % name,
                   {"x_min": 0.0, "x_max": Lf, "t_max": t_max, "u_min": u_min, "u_max": u_max},
                   plot_fx(f, x), False)


T_SYM = sp.Symbol("t")


def _lead(value):
    """Coefficient in front of a term: nothing for 1, a minus sign for -1."""
    shown = _g(value)
    if shown == "1":
        return ""
    if shown == "-1":
        return "-"
    return shown + "\\,"


def _join(pieces, exact):
    """Join series terms, with dots unless the series is known to stop."""
    body = " + ".join(pieces).replace("+ -", "- ")
    return ("= " + body) if exact else ("\\approx " + body + " + \\cdots")


def _g(value):
    """A short decimal for display."""
    exact = sp.nsimplify(value, rational=True, tolerance=1e-9)
    if exact.q <= 1000:
        return tex(exact)
    return "%.5g" % value


# --------------------------------------------------------------------------
# Wave equation
# --------------------------------------------------------------------------

def wave(f_text, g_text, L_text, c_text):
    L = _positive(L_text, "the length L")
    c = _positive(c_text, "the wave speed c")
    x = sp.Symbol("x", real=True)
    f = g = sp.Integer(0)
    if (f_text or "").strip():
        f, x = _function(f_text, "x", "the initial displacement f(x)")
    if (g_text or "").strip():
        g, _ = _function(g_text, "x", "the initial velocity g(x)")
    if f == 0 and g == 0:
        raise InputError("Enter an initial displacement f(x), an initial velocity g(x), or both.")
    Lf, cf = float(L), float(c)
    fn, gn = _callable(f, x), _callable(g, x)
    modes = []   # (n, a_n, b_n, wavenumber, frequency)
    for n in range(1, SCAN + 1):
        m = n * math.pi / Lf
        a = 2 / Lf * _simpson(lambda s: fn(s) * math.sin(m * s), 0.0, Lf) if f != 0 else 0.0
        b = 2 / (Lf * cf * m) * _simpson(lambda s: gn(s) * math.sin(m * s), 0.0, Lf) if g != 0 else 0.0
        if abs(a) > 1e-10 or abs(b) > 1e-10:
            modes.append((n, a, b, m, cf * m))
        if len(modes) >= TERMS:
            break

    def u(xv, tv):
        return sum((a * math.cos(w * tv) + b * math.sin(w * tv)) * math.sin(m * xv)
                   for _, a, b, m, w in modes)

    parts = []
    for _, a, b, m, w in modes:
        time = []
        if abs(a) > 1e-10:
            time.append("%s*cos(%s*y)" % (_num(a), _num(w)))
        if abs(b) > 1e-10:
            time.append("%s*sin(%s*y)" % (_num(b), _num(w)))
        parts.append("(%s)*sin(%s*x)" % (" + ".join(time), _num(m)))
    plot = " + ".join(parts) if parts else "0"
    period = 2 * Lf / cf
    t_max = 2 * period
    grid = [u(Lf * i / 40, t_max * j / 60) for i in range(41) for j in range(61)]
    u_min, u_max = _range(grid, [fn(Lf * i / 40) for i in range(41)])
    bound = max(abs(u_min), abs(u_max))
    u_min, u_max = -bound, bound

    sin_tex = "\\sin\\frac{n\\pi x}{L}"
    basis = sp.sin(N * sp.pi * x / L)
    steps = _separate_steps(
        "u = X(x)\\,T(t)", "\\frac{T''}{c^2\\,T} = \\frac{X''}{X} = -\\lambda",
        "X'' + \\lambda X = 0,\\qquad X(0) = 0,\\quad X(L) = 0",
        "Only \\(\\lambda > 0\\) gives non-zero solutions, and they are sines.",
        "\\lambda_n = \\left(\\frac{n\\pi}{L}\\right)^2,\\qquad X_n = %s" % sin_tex)
    steps.append(step(
        "Solve the time equation",
        "Each mode oscillates in time at its own frequency \\(\\omega_n = c\\,n\\pi/L\\). "
        "All of them repeat after the period \\(2L/c = %s\\)." % _g(period),
        ["T'' + c^2\\lambda_n T = 0 \\quad\\Rightarrow\\quad "
         "T_n = a_n\\cos\\omega_n t + b_n\\sin\\omega_n t"]))
    steps.append(step(
        "Superpose",
        "Any sum of these standing waves satisfies the equation and keeps both ends fixed.",
        ["u(x,t) = \\sum_{n=1}^{\\infty}\\left(a_n\\cos\\omega_n t + b_n\\sin\\omega_n t\\right)%s" % sin_tex]))
    if f != 0:
        s1 = _coefficient_step("a_n", f, x, sin_tex, basis, L, 2 / L, [(n, a) for n, a, _, _, _ in modes])
        s1["title"] = "Match the initial displacement"
        s1["text"] = "At \\(t = 0\\) the series must equal \\(f(x)\\). " + s1["text"]
        steps.append(s1)
    else:
        steps.append(step("Match the initial displacement",
                          "The string starts flat, so every \\(a_n\\) is zero.", ["a_n = 0"]))
    if g != 0:
        s2 = _coefficient_step("b_n", g, x, sin_tex, basis, L, 2 / (N * sp.pi * c),
                               [(n, b) for n, _, b, _, _ in modes])
        s2["title"] = "Match the initial velocity"
        s2["text"] = ("Differentiating in \\(t\\) and setting \\(t = 0\\) must give \\(g(x)\\), "
                      "which brings in a factor \\(1/\\omega_n\\). ") + s2["text"]
        steps.append(s2)
    else:
        steps.append(step("Match the initial velocity",
                          "The string starts at rest, so every \\(b_n\\) is zero.", ["b_n = 0"]))
    pieces = []
    for n, a, b, m, w in modes[:3]:
        omega = sp.nsimplify(sp.simplify(c * n * sp.pi / L))
        inner = []
        if abs(a) > 1e-10:
            inner.append("%s\\cos %s" % (_lead(a), tex(omega * T_SYM)))
        if abs(b) > 1e-10:
            inner.append("%s\\sin %s" % (_lead(b), tex(omega * T_SYM)))
        body = " + ".join(inner)
        pieces.append("%s\\,%s" % ("\\left(%s\\right)" % body if len(inner) > 1 else body,
                                   tex(basis.subs(N, n))))
    solution = "u(x,t) " + _join(pieces, len(modes) <= 1)

    h, dt = Lf * 1e-3, period * 1e-3
    worst = 0.0
    for xv in (0.23 * Lf, 0.5 * Lf, 0.81 * Lf):
        for tv in (0.17 * period, 0.6 * period):
            utt = (u(xv, tv + dt) - 2 * u(xv, tv) + u(xv, tv - dt)) / (dt * dt)
            uxx = (u(xv + h, tv) - 2 * u(xv, tv) + u(xv - h, tv)) / (h * h)
            worst = max(worst, abs(utt - cf * cf * uxx) * period * period / 40)
    edge = max(abs(u(0.0, 0.3 * period)), abs(u(Lf, 0.3 * period)))
    data_error = max(abs(u(Lf * i / 20, 0.0) - fn(Lf * i / 20)) for i in range(2, 19))
    smooth = max((abs(a) + abs(b)) * n * n for n, a, b, _, _ in modes) < 1e3 * max(bound, 1e-9)
    verified, note = _verdict(worst if smooth else 0.0, edge, data_error, 2 * bound,
                              "the initial displacement")
    if not smooth and verified:
        note = ("Checked numerically: the series keeps both ends fixed and reproduces the "
                "initial data. With corners in the data the truncated series is not smooth "
                "enough to test the equation by finite differences.")
    methods = [_method("separation", "Separation of variables",
                       "With both ends fixed, the motion is a sum of standing waves.",
                       steps, solution, plot, verified, note),
               _dalembert(f, g, x, L, c, u, period)]
    input_latex = ("u_{tt} = %s\\,u_{xx},\\quad 0 < x < %s;\\quad u(0,t) = u(%s,t) = 0;\\quad "
                   "u(x,0) = %s,\\quad u_t(x,0) = %s"
                   % (tex(c ** 2), tex(L), tex(L), tex(f), tex(g)))
    return _result("wave", input_latex, methods,
                   "Wave equation for a string of length L with fixed ends. Press Play on the "
                   "graph to watch it move; the motion repeats every %s." % ("%.4g" % period),
                   {"x_min": 0.0, "x_max": Lf, "t_max": t_max, "u_min": u_min, "u_max": u_max},
                   plot_fx(f, x), False)


def _dalembert(f, g, x, L, c, series_u, period):
    name = "d'Alembert's formula"
    s = sp.Symbol("s", real=True)

    def reduce(arg):
        return arg - 2 * L * sp.floor((arg + L) / (2 * L))

    def odd(arg):
        r = reduce(arg)
        return (2 * sp.Heaviside(r) - 1) * f.subs(x, sp.Abs(r))

    t = sp.Symbol("y", real=True)   # the plot evaluator's second variable stands for time
    expr = (odd(x - c * t) + odd(x + c * t)) / 2
    formula = ["u(x,t) = \\frac{F(x - ct) + F(x + ct)}{2} + \\frac{1}{2c}\\int_{x-ct}^{x+ct} G(s)\\,ds"]
    steps = [
        step("Write the general solution as two travelling waves",
             "Every solution of the wave equation is a wave moving right plus a wave moving "
             "left, each with speed \\(c\\).",
             ["u(x,t) = \\phi(x - ct) + \\psi(x + ct)"]),
        step("Fit the initial data",
             "Matching the initial displacement \\(f\\) and velocity \\(g\\) gives d'Alembert's "
             "formula. To respect the fixed ends, \\(F\\) and \\(G\\) are \\(f\\) and \\(g\\) "
             "extended as odd functions with period \\(2L\\): the waves then reflect upside "
             "down at each end.", formula),
    ]
    if g != 0:
        def work():
            value = sp.integrate(g.subs(x, s), (s, 0, x))
            return None if value.has(sp.Integral) or value.has(sp.Piecewise) else sp.simplify(value)
        base = None if g.has(sp.Heaviside) else run_with_timeout(work, SYMBOLIC_TIMEOUT, None)
        if base is None:
            return {"id": "dalembert", "name": name, "applicable": False,
                    "reason": "The integral of the initial velocity has no simple closed form, "
                              "so the formula cannot be written out here."}

        def even_integral(arg):
            return base.subs(x, sp.Abs(reduce(arg)))
        expr = expr + (even_integral(x + c * t) - even_integral(x - c * t)) / (2 * c)
        steps.append(step(
            "Integrate the initial velocity",
            "Because \\(G\\) is odd and periodic, its integral from 0 is an even periodic "
            "function, so the integral term is a difference of two of its values.",
            ["H(x) = \\int_0^{x} g(s)\\,ds = %s" % tex(base),
             "\\frac{1}{2c}\\int_{x-ct}^{x+ct} G(s)\\,ds = \\frac{H(x+ct) - H(x-ct)}{2c}"]))
    else:
        steps.append(step("Use that the string starts at rest",
                          "With \\(g = 0\\) the integral vanishes: the initial shape splits into "
                          "two half-height copies travelling in opposite directions.",
                          ["u(x,t) = \\frac{F(x - ct) + F(x + ct)}{2}"]))
    plot = _plot_xy(expr, x, t)
    verified, note = None, "Could not be checked automatically."
    if plot is not None:
        fn = sp.lambdify((x, t), expr, [{"Heaviside": lambda v, *r: 0.0 if v < 0 else 1.0}, "math"])
        Lf = float(L)
        try:
            gap = max(abs(float(fn(Lf * a, period * b)) - series_u(Lf * a, period * b))
                      for a in (0.2, 0.45, 0.7) for b in (0.1, 0.35, 0.8))
            scale = max(abs(series_u(Lf * i / 20, 0.0)) for i in range(21)) or 1.0
            if gap < 0.05 * max(scale, 1e-9) + 1e-9 or g != 0 and gap < 0.08 * max(scale, 1e-2):
                verified, note = True, ("Agrees with the series solution at sample points, and "
                                        "it reproduces the initial data exactly.")
            else:
                verified, note = None, ("The formula is exact; the truncated series differs "
                                        "from it slightly at the sample points.")
        except Exception:
            pass
    solution = formula[0] if g != 0 else "u(x,t) = \\frac{F(x - ct) + F(x + ct)}{2}"
    return _method("dalembert", name,
                   "On a string the solution is always two travelling waves, so it can be "
                   "written without a series.", steps, solution, plot, verified, note)


def _plot_xy(expr, x, t):
    """Plot string in x and y (y is the second variable), or None."""
    allowed = {"sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan", "sinh",
               "cosh", "tanh", "exp", "log", "Abs", "Heaviside", "floor"}
    try:
        if expr.has(sp.I, sp.oo, sp.zoo, sp.nan, sp.Integral, sp.Piecewise):
            return None
        if any(type(fn).__name__ not in allowed for fn in expr.atoms(sp.Function)):
            return None
        if not expr.free_symbols <= {x, t}:
            return None
        return sp.sstr(expr.xreplace({x: sp.Symbol("x"), t: sp.Symbol("y")}))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Laplace's equation on a rectangle
# --------------------------------------------------------------------------

_SIDES = {"top": ("x", "y = b"), "bottom": ("x", "y = 0"), "left": ("y", "x = 0"), "right": ("y", "x = a")}


def laplace(f_text, a_text, b_text, side):
    if side not in _SIDES:
        raise InputError("Choose which side carries the boundary values.")
    a = _positive(a_text, "the width a")
    b = _positive(b_text, "the height b")
    var, where = _SIDES[side]
    f, v = _function(f_text, var, "the boundary values f(%s)" % var)
    af, bf = float(a), float(b)
    # `span` is the side carrying the data; `depth` is the other dimension.
    span, depth = (af, bf) if var == "x" else (bf, af)
    span_sym, depth_sym = (a, b) if var == "x" else (b, a)
    fn = _callable(f, v)
    modes = []
    for n in range(1, SCAN + 1):
        m = n * math.pi / span
        coef = 2 / span * _simpson(lambda s: fn(s) * math.sin(m * s), 0.0, span)
        if abs(coef) > 1e-10:
            modes.append((n, coef, m))
        if len(modes) >= TERMS:
            break

    def ratio(m, distance):
        """sinh(m d) / sinh(m depth), written so it cannot overflow."""
        return math.exp(m * (distance - depth)) * (1 - math.exp(-2 * m * distance)) / (1 - math.exp(-2 * m * depth))

    def u(xv, yv):
        along, across = (xv, yv) if var == "x" else (yv, xv)
        distance = across if side in ("top", "right") else depth - across
        return sum(c * ratio(m, distance) * math.sin(m * along) for _, c, m in modes)

    along_name, across_name = ("x", "y") if var == "x" else ("y", "x")
    distance_text = across_name if side in ("top", "right") else "(%s - %s)" % (_num(depth), across_name)
    parts = []
    for _, c, m in modes:
        parts.append("%s*exp(%s*(%s - %s))*(1 - exp(-%s*%s))*sin(%s*%s)" % (
            _num(c / (1 - math.exp(-2 * m * depth))), _num(m), distance_text, _num(depth),
            _num(2 * m), distance_text, _num(m), along_name))
    plot = " + ".join(parts) if parts else "0"
    grid = [u(af * i / 30, bf * j / 30) for i in range(31) for j in range(31)]
    u_min, u_max = _range(grid, [fn(span * i / 30) for i in range(1, 30)])

    S, D = ("a", "b") if var == "x" else ("b", "a")
    other = "y" if var == "x" else "x"
    sin_tex = "\\sin\\frac{n\\pi %s}{%s}" % (var, S)
    far = "\\sinh\\frac{n\\pi %s}{%s}" % (other, S) if side in ("top", "right") \
        else "\\sinh\\frac{n\\pi (%s - %s)}{%s}" % (D, other, S)
    X, Y = ("X", "Y") if var == "x" else ("Y", "X")
    steps = _separate_steps(
        "u = X(x)\\,Y(y)", "\\frac{%s''}{%s} = -\\frac{%s''}{%s} = -\\lambda" % (X, X, Y, Y),
        "%s'' + \\lambda %s = 0,\\qquad %s(0) = 0,\\quad %s(%s) = 0" % (X, X, X, X, S),
        "The two sides facing each other in the \\(%s\\) direction are both held at 0, so "
        "that direction gives the eigenvalue problem; its solutions are sines." % var,
        "\\lambda_n = \\left(\\frac{n\\pi}{%s}\\right)^2,\\qquad %s_n = %s" % (S, X, sin_tex))
    steps.append(step(
        "Solve in the other direction",
        "The other factor satisfies \\(%s'' - \\lambda_n %s = 0\\), whose solutions are "
        "hyperbolic. It must vanish on the side opposite the data, which selects" % (Y, Y),
        ["%s_n = %s" % (Y, far)]))
    steps.append(step(
        "Superpose",
        "Sum the products; three sides are then at 0 whatever the coefficients are.",
        ["u(x,y) = \\sum_{n=1}^{\\infty} c_n\\,\\frac{%s}{\\sinh\\frac{n\\pi %s}{%s}}\\,%s"
         % (far, D, S, sin_tex)]))
    basis = sp.sin(N * sp.pi * v / span_sym)
    coefficient = _coefficient_step("c_n", f, v, sin_tex, basis, span_sym, 2 / span_sym,
                                    [(n, c) for n, c, _ in modes], var)
    coefficient["text"] = ("On the side \\(%s\\) the hyperbolic ratio is 1, so the series must "
                           "equal \\(f(%s)\\) there. " % (where, var)) + coefficient["text"]
    steps.append(coefficient)
    pieces = []
    for n, c, m in modes[:3]:
        k = sp.nsimplify(n * sp.pi / span_sym)
        o = sp.Symbol(other)
        arg = k * o if side in ("top", "right") else k * (depth_sym - o)
        pieces.append("%s\\frac{\\sinh %s}{\\sinh %s}\\,%s" % (
            _lead(c), "\\left(%s\\right)" % tex(arg), tex(k * depth_sym),
            tex(sp.sin(k * sp.Symbol(var)))))
    solution = "u(x,y) " + _join(pieces, len(modes) <= 1)

    h = min(af, bf) * 1e-3
    worst = 0.0
    for px in (0.3, 0.5, 0.72):
        for py in (0.3, 0.55):
            xv, yv = af * px, bf * py
            lap = (u(xv + h, yv) + u(xv - h, yv) + u(xv, yv + h) + u(xv, yv - h) - 4 * u(xv, yv)) / (h * h)
            worst = max(worst, abs(lap) * min(af, bf) ** 2 / 40)
    zero_sides = {"top": [(0, .5), (1, .5), (.5, 0)], "bottom": [(0, .5), (1, .5), (.5, 1)],
                  "left": [(.5, 0), (.5, 1), (1, .5)], "right": [(.5, 0), (.5, 1), (0, .5)]}[side]
    edge = max(abs(u(af * px, bf * py)) for px, py in zero_sides)
    def on_side(sv):
        return {"top": u(sv, bf), "bottom": u(sv, 0.0), "left": u(0.0, sv), "right": u(af, sv)}[side]
    data_error = max(abs(on_side(span * i / 20) - fn(span * i / 20)) for i in range(3, 18))
    verified, note = _verdict(worst, edge, data_error, u_max - u_min, "the boundary values")
    interior = [u(af * i / 12, bf * j / 12) for i in range(1, 12) for j in range(1, 12)]
    lo, hi = _range([fn(span * i / 60) for i in range(61)] + [0.0])
    if verified and (min(interior) < lo - 0.1 * (hi - lo) or max(interior) > hi + 0.1 * (hi - lo)):
        verified, note = False, "The interior values go outside the range of the boundary values."

    method = _method("separation", "Separation of variables",
                     "Three sides are held at 0, so product solutions can be superposed to "
                     "match the fourth.", steps, solution, plot, verified, note)
    input_latex = ("u_{xx} + u_{yy} = 0,\\quad 0 < x < %s,\\ 0 < y < %s;\\quad u = %s \\text{ on } %s,"
                   "\\ u = 0 \\text{ on the other sides}"
                   % (tex(a), tex(b), tex(f), where.replace("a", tex(a)).replace("b", tex(b))))
    return _result("laplace", input_latex, [method],
                   "Laplace's equation on a rectangle: the steady temperature when one side is "
                   "held at f and the other three at 0. The graph shows it as a heat map.",
                   {"x_min": 0.0, "x_max": af, "y_min": 0.0, "y_max": bf, "u_min": u_min, "u_max": u_max},
                   None, True)


def _result(pde, input_latex, methods, summary, domain, initial, surface):
    full = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "t_max": 0.0,
            "u_min": -1.0, "u_max": 1.0}
    full.update(domain)
    return {"schema": 1, "ok": True, "kind": "pde", "pde": pde, "order": 2, "indep": "x", "dep": "u",
            "input_latex": input_latex, "normal_latex": input_latex, "rhs_expr": None, "ic": None,
            "domain": full, "initial_expr": initial, "surface": surface,
            "summary": summary, "other_types": [], "methods": methods}


def solve(kind, f_text, g_text, p1, p2, bc, t1, t2):
    if kind == "heat":
        return heat(f_text, p1, p2, bc or "00", t1, t2)
    if kind == "wave":
        return wave(f_text, g_text, p1, p2)
    if kind == "laplace":
        return laplace(f_text, p1, p2, bc or "top")
    if kind in ("characteristics", "classify", "fd", "disk"):
        from . import pdes2
        if kind == "characteristics":   # a, b in p1, p2; c, d in t1, t2
            return pdes2.characteristics(p1, p2, t1, t2, f_text)
        if kind == "classify":          # A, B, C in p1, p2, t1
            return pdes2.classify(p1, p2, t1)
        if kind == "fd":                # L, k in p1, p2; N, dt in t1, t2; steps in g
            return pdes2.finite_differences(f_text, p1, p2, t1, t2, g_text)
        return pdes2.disk(f_text, p1)   # R in p1
    raise InputError("Choose the kind of problem.")


def preview(text, var="x"):
    """LaTeX for 'f;g' as typed so far."""
    parts = (text.split(";") + [""])[:2]
    lines = []
    for label, part in zip(("f(%s)" % var, "g(%s)" % var), parts):
        if part.strip():
            expr, _ = parse_expression(part, [var])
            lines.append("%s = %s" % (label, tex(expr)))
    return ",\\qquad ".join(lines)
