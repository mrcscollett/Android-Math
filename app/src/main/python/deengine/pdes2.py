"""More PDE tools: characteristics, classification, finite differences, the disk.

characteristics: a u_x + b u_t = c u + d with u(x, 0) = f(x) (or the general
solution when no usable initial condition is given).
classify: the type of A u_xx + B u_xy + C u_yy + ... = 0.
finite differences: the explicit (FTCS) and Crank-Nicolson schemes for the
heat equation with both ends held at 0.
disk: Laplace's equation inside a circle with the boundary values given.
"""
import math

import sympy as sp

from . import pdes
from .first_order import step
from .fourier import N, _plot as plot_fx
from .numeric import fmt
from .parsing import InputError, parse_expression, parse_number
from .util import residual_status, run_with_timeout, tex

X = sp.Symbol("x", real=True)
Y = sp.Symbol("y", real=True)
T = sp.Symbol("t", real=True)
XI = sp.Symbol("xi", real=True)
TIMEOUT = 6
MAX_LEVELS = 60


def _expr(text, names, label, default=None):
    if not (text or "").strip():
        if default is not None:
            return sp.Integer(default)
        raise InputError("Enter %s." % label)
    expr, _ = parse_expression(text, names)
    if expr.has(sp.DiracDelta, sp.Heaviside):
        raise InputError("%s cannot contain a step or an impulse."
                         % (label[0].upper() + label[1:]))
    return expr


def _plain(method_id, name, reason, steps, solution_tex, plot, verified, note, check=True):
    """A method dict; `check` False leaves out the Check step (qualitative results)."""
    m = {"id": method_id, "name": name, "applicable": True, "reason": reason, "steps": steps,
         "general_latex": None, "particular_latex": solution_tex,
         "plot": {"general_expr": None, "particular_expr": plot, "points": []},
         "verified": verified, "verify_note": note}
    if check:
        pdes._answer(m, solution_tex)
    return m


# --------------------------------------------------------------------------
# Method of characteristics
# --------------------------------------------------------------------------

def _dsolve(equation, func, ics=None):
    def work():
        return sp.dsolve(equation, func, ics=ics) if ics else sp.dsolve(equation, func)
    try:
        sol = run_with_timeout(work, TIMEOUT, None)
    except Exception:
        sol = None
    if isinstance(sol, (list, tuple)):
        sol = sol[0] if len(sol) == 1 else None
    if sol is None or sol.lhs != func or sol.rhs.has(func.func) or sol.rhs.has(sp.Integral):
        return None
    return sol.rhs


def characteristics(a_text, b_text, c_text, d_text, f_text):
    names = ["x", "t"]
    a = _expr(a_text, names, "the coefficient a of u_x")
    b = _expr(b_text, names, "the coefficient b of u_t")
    c = _expr(c_text, names, "c", 0)
    d = _expr(d_text, names, "d", 0)
    if b == 0:
        raise InputError("The coefficient of u_t cannot be 0: the method follows the "
                         "solution forward in t.")
    f = None
    if (f_text or "").strip():
        f, _ = parse_expression(f_text, ["x"])
        if f.has(sp.DiracDelta):
            raise InputError("An impulse cannot be used as the initial condition.")
    b0 = sp.simplify(b.subs(T, 0))
    ivp = f is not None and b0 != 0 and not b0.has(sp.zoo, sp.nan, sp.oo)

    speed = sp.simplify(a / b)
    Xf, Uf = sp.Function("X"), sp.Function("U")
    curve_eq = sp.Eq(Xf(T).diff(T), speed.subs(X, Xf(T)))
    if ivp:
        curve = _dsolve(curve_eq, Xf(T), {Xf(0): XI})
    else:
        curve = _dsolve(curve_eq, Xf(T))
        if curve is not None:
            constants = [s for s in curve.free_symbols if s.name.startswith("C")]
            curve = curve.subs(constants[0], XI) if len(constants) == 1 else None
    if curve is None:
        raise InputError("The characteristic equation dx/dt = a/b could not be solved in "
                         "closed form for these coefficients.")
    try:
        inverse = run_with_timeout(lambda: sp.solve(sp.Eq(X, curve), XI), TIMEOUT, [])
    except Exception:
        inverse = []
    inverse = [s for s in inverse if not s.has(sp.I)]
    if len(inverse) != 1:
        raise InputError("The characteristic curves could not be solved for the constant "
                         "that labels them.")
    invariant = sp.simplify(inverse[0])

    along = sp.Eq(Uf(T).diff(T), sp.simplify((c.subs(X, curve) * Uf(T) + d.subs(X, curve))
                                             / b.subs(X, curve)))
    start = sp.Symbol("u_0")
    if ivp:
        on_curve = _dsolve(along, Uf(T), {Uf(0): start})
    else:
        on_curve = _dsolve(along, Uf(T))
        if on_curve is not None:
            constants = [s for s in on_curve.free_symbols if s.name.startswith("C")]
            on_curve = on_curve.subs(constants[0], start) if len(constants) == 1 else None
    if on_curve is None:
        raise InputError("The equation for u along the characteristics could not be solved "
                         "in closed form.")

    equation = "%s\\,u_x + %s\\,u_t = %s" % (
        _paren(a), _paren(b), tex(sp.Symbol("u") * c + d) if (c != 0 or d != 0) else "0")
    steps = [
        step("Write the characteristic equations",
             "Along a curve \\(x(t)\\) with \\(dx/dt = a/b\\), the left side of the equation is "
             "\\(b\\) times the rate of change of \\(u\\). So the PDE becomes two ordinary "
             "differential equations.",
             ["\\frac{dx}{dt} = %s" % tex(speed),
              "\\frac{du}{dt} = %s" % tex(sp.simplify((c * sp.Symbol("u") + d) / b))]),
        step("Solve for the characteristic curves",
             ("Label each curve by the point \\(\\xi\\) where it starts at \\(t = 0\\)."
              if ivp else "Each curve is labelled by its constant of integration \\(\\xi\\).")
             + " Solving for \\(\\xi\\) gives a quantity that stays constant along each curve.",
             ["x = %s" % tex(curve), "\\xi = %s" % tex(invariant)]),
    ]
    if ivp:
        steps.append(step(
            "Solve along a characteristic",
            "On the curve that starts at \\(\\xi\\), \\(u\\) starts at \\(u_0 = f(\\xi)\\).",
            ["u = %s" % tex(on_curve), "u_0 = f(\\xi) = %s" % tex(f.subs(X, XI))]))
        u = sp.simplify(on_curve.subs(start, f.subs(X, XI)).subs(XI, invariant))
        steps.append(step("Return to x and t",
                          "Replace \\(\\xi\\) by its expression in \\(x\\) and \\(t\\).",
                          ["u(x,t) = %s" % tex(u)]))
        status = residual_status(a * sp.diff(u, X) + b * sp.diff(u, T) - c * u - d)
        at_start = residual_status(u.subs(T, 0) - f)
        if status in ("symbolic", "numeric") and at_start in ("symbolic", "numeric"):
            verified, note = True, ("Checked by substituting into the equation, and it "
                                    "reduces to f(x) at t = 0.")
        elif status == "bad" or at_start == "bad":
            verified, note = False, "Substituting back did not reproduce the equation."
        else:
            verified, note = None, "Could not be checked automatically."
        plot = pdes._plot_xy(u, X, T)
        domain = _transport_domain(u, speed)
        solution = "u(x,t) = %s" % tex(u)
        initial = plot_fx(f, X)
        summary = ("First-order equation solved along its characteristics. Use the slider on "
                   "the graph to watch the initial profile move.")
        input_latex = "%s;\\quad u(x,0) = %s" % (equation, tex(f))
    else:
        F = sp.Function("F")
        u = sp.simplify(on_curve.subs(start, F(XI)).subs(XI, invariant))
        steps.append(step(
            "Solve along a characteristic",
            "The constant of integration can be different on each curve, so it is an "
            "arbitrary function \\(F\\) of \\(\\xi\\).",
            ["u = %s" % tex(on_curve.subs(start, F(XI)))]))
        steps.append(step("Return to x and t", "", ["u(x,t) = %s" % tex(u)]))
        try:
            ok = sp.simplify((a * sp.diff(u, X) + b * sp.diff(u, T) - c * u - d).doit()) == 0
        except Exception:
            ok = False
        verified = True if ok else None
        note = ("Checked by substituting into the equation for an arbitrary function F."
                if ok else "Could not be checked automatically.")
        plot, initial, domain = None, None, {}
        solution = "u(x,t) = %s" % tex(u)
        summary = "General solution of a first-order equation by the method of characteristics."
        if f is not None:
            summary += (" The coefficient of u_t vanishes at t = 0, so a condition given "
                        "there cannot pick out one solution; F stays arbitrary.")
        input_latex = equation
    method = _plain("characteristics", "Method of characteristics",
                    "The equation is first order and linear in \\(u\\), so it reduces to "
                    "ordinary differential equations along curves in the \\(x\\)–\\(t\\) plane.",
                    steps, solution, plot, verified, note)
    return pdes._result("characteristics", input_latex, [method], summary, domain, initial, False)


def _paren(expr):
    return "\\left(%s\\right)" % tex(expr) if expr.is_Add else tex(expr)


def _transport_domain(u, speed):
    """Window for the animation: x-range, time span and the range of u."""
    t_max = 3.0
    if not speed.free_symbols and speed != 0:
        t_max = min(6.0, 6.0 / abs(float(speed)))
    step_fn = lambda v, *rest: 0.0 if v < 0 else 1.0
    values, start = [], []
    try:
        fn = sp.lambdify((X, T), u, [{"Heaviside": step_fn}, "math"])
        for i in range(81):
            xv = -8.0 + 16.0 * i / 80
            for j in range(9):
                try:
                    value = float(fn(xv, t_max * j / 8))
                except Exception:
                    continue
                if math.isfinite(value):
                    values.append(value)
                    if j == 0:
                        start.append(value)
    except Exception:
        pass
    lo, hi = pdes._range(values)
    s_lo, s_hi = pdes._range(start)
    width = max(s_hi - s_lo, 1e-9)
    # Growth or blow-up must not flatten the picture of the initial profile.
    lo, hi = max(lo, s_lo - 4 * width), min(hi, s_hi + 4 * width)
    return {"x_min": -8.0, "x_max": 8.0, "t_max": t_max, "u_min": lo, "u_max": hi}


# --------------------------------------------------------------------------
# Classification of second-order equations
# --------------------------------------------------------------------------

def _region(expr, relation):
    """LaTeX description of where expr (in one variable) is >0, <0 or =0, or None."""
    symbols = list(expr.free_symbols)
    if len(symbols) != 1:
        return None
    try:
        if relation == "=":
            roots = sp.solve(expr, symbols[0])
            real = [r for r in roots if r.is_real]
            return ",\\ ".join("%s = %s" % (tex(symbols[0]), tex(r)) for r in real) if real else None
        result = sp.solve_univariate_inequality(
            expr > 0 if relation == ">" else expr < 0, symbols[0], relational=True)
        if result is sp.false:
            return None
        return tex(result)
    except Exception:
        return None


def _line(left, slope, right):
    """LaTeX of the lines left - slope * right = const, with the sign folded in."""
    if slope == 0:
        return "%s = \\text{const}" % left
    magnitude = -slope if slope.could_extract_minus_sign() else slope
    term = right if magnitude == 1 else "%s %s" % (
        "\\left(%s\\right)" % tex(magnitude) if magnitude.is_Add else tex(magnitude), right)
    return "%s %s %s = \\text{const}" % (left, "+" if slope.could_extract_minus_sign() else "-", term)


def _family(slope, name):
    """LaTeX of the curves with dy/dx = slope, when that integrates simply."""
    try:
        if not slope.has(X, Y):
            return _line("y", slope, "x")
        if not slope.has(Y):
            antiderivative = run_with_timeout(lambda: sp.integrate(slope, X), 3, None)
            if antiderivative is not None and not antiderivative.has(sp.Integral, sp.I):
                return "y - \\left(%s\\right) = \\text{const}" % tex(antiderivative)
        if not slope.has(X):
            antiderivative = run_with_timeout(lambda: sp.integrate(1 / slope, Y), 3, None)
            if antiderivative is not None and not antiderivative.has(sp.Integral, sp.I):
                return "%s - x = \\text{const}" % tex(sp.simplify(antiderivative))
    except Exception:
        pass
    return None


def classify(a_text, b_text, c_text):
    names = ["x", "y"]
    A = _expr(a_text, names, "the coefficient A of u_xx", 0)
    B = _expr(b_text, names, "the coefficient B of u_xy", 0)
    C = _expr(c_text, names, "the coefficient C of u_yy", 0)
    if A == 0 and B == 0 and C == 0:
        raise InputError("Enter at least one of the coefficients A, B and C.")
    disc = sp.simplify(B**2 - 4 * A * C)
    equation = sp.Symbol("u_{xx}") * A + sp.Symbol("u_{xy}") * B + sp.Symbol("u_{yy}") * C
    input_latex = "%s + \\cdots = 0" % tex(equation)
    steps = [step(
        "Compute the discriminant",
        "For \\(A u_{xx} + B u_{xy} + C u_{yy} + \\cdots = 0\\) the type depends only on the "
        "sign of \\(B^2 - 4AC\\). Lower-order terms do not matter.",
        ["A = %s,\\quad B = %s,\\quad C = %s" % (tex(A), tex(B), tex(C)),
         "B^2 - 4AC = %s" % tex(disc)])]

    kinds = {"hyperbolic": ("the wave equation", "u_{\\xi\\eta} = \\text{lower-order terms}",
                            "two families of real characteristics"),
             "parabolic": ("the heat equation", "u_{\\eta\\eta} = \\text{lower-order terms}",
                           "one family of real characteristics"),
             "elliptic": ("Laplace's equation", "u_{\\xi\\xi} + u_{\\eta\\eta} = \\text{lower-order terms}",
                          "no real characteristics")}
    constant = not disc.free_symbols
    if constant:
        kind = "hyperbolic" if disc > 0 else ("parabolic" if disc == 0 else "elliptic")
        sign = {"hyperbolic": "positive", "parabolic": "zero", "elliptic": "negative"}[kind]
        steps.append(step("Read off the type",
                          "The discriminant is %s everywhere, so the equation is %s."
                          % (sign, kind)))
        verdict = "\\text{%s}" % kind
        summary = "The equation is %s (like %s)." % (kind, kinds[kind][0])
        regions = [kind]
    else:
        lines, regions = [], []
        for kind, relation in (("hyperbolic", ">"), ("parabolic", "="), ("elliptic", "<")):
            where = _region(disc, relation)
            if where is None and len(disc.free_symbols) == 1:
                continue
            regions.append(kind)
            lines.append("\\text{%s where } %s" % (
                kind, where if where else "%s %s 0" % (tex(disc), relation)))
        steps.append(step(
            "Read off the type",
            "The discriminant changes sign, so the type depends on the point: hyperbolic "
            "where it is positive, parabolic where it is zero, elliptic where it is negative.",
            lines))
        verdict = "\\text{mixed type: }" + ";\\ ".join(lines)
        summary = "The equation is of mixed type: %s in different regions." % ", ".join(regions)

    # Characteristic directions.
    if "hyperbolic" in regions or "parabolic" in regions:
        root = sp.sqrt(disc)
        if A != 0:
            slopes = [sp.simplify((B + root) / (2 * A)), sp.simplify((B - root) / (2 * A))]
            formula = "\\frac{dy}{dx} = \\frac{B \\pm \\sqrt{B^2 - 4AC}}{2A}"
            families = [_family(m, "y") for m in dict.fromkeys(slopes)]
            shown = ["\\frac{dy}{dx} = %s" % tex(m) for m in dict.fromkeys(slopes)]
        elif C != 0:
            slopes = [sp.simplify((B + root) / (2 * C)), sp.simplify((B - root) / (2 * C))]
            formula = "\\frac{dx}{dy} = \\frac{B \\pm \\sqrt{B^2 - 4AC}}{2C}"
            shown = ["\\frac{dx}{dy} = %s" % tex(m) for m in dict.fromkeys(slopes)]
            families = []
            for m in dict.fromkeys(slopes):
                families.append(_line("x", m, "y") if not m.has(X, Y) else None)
        else:
            formula = "\\text{with } A = C = 0 \\text{ the characteristics are the coordinate lines}"
            shown, families = [], ["x = \\text{const}", "y = \\text{const}"]
        curves = [c for c in families if c]
        text = ("Characteristics are the curves along which information travels. They solve "
                "the ordinary differential equation below.")
        if not constant:
            text += " They are real only in the region where the discriminant is not negative."
        latex = [formula] + shown
        if curves:
            latex.append(",\\qquad ".join(curves))
        steps.append(step("Find the characteristics", text, latex))
    else:
        steps.append(step("Find the characteristics",
                          "The discriminant is negative, so the characteristic directions are "
                          "complex: an elliptic equation has no real characteristics, and "
                          "disturbances are felt everywhere at once."))

    canon = []
    for kind in regions:
        model, form, count = kinds[kind]
        canon.append("\\text{%s: } %s" % (kind, form))
    steps.append(step(
        "Canonical form",
        "A change of variables built from the characteristics brings the second-order part "
        "to a standard shape. " + " ".join(
            "Where it is %s it behaves like %s (%s)." % (k, kinds[k][0], kinds[k][2])
            for k in regions),
        canon))
    answer = step("Type", "", [verdict])
    answer["kind"] = "answer"
    steps.append(answer)
    method = _plain("discriminant", "Discriminant test",
                    "The type of a second-order linear equation is decided by its "
                    "second-order coefficients alone.",
                    steps, verdict, None, None, "", check=False)
    return pdes._result("classify", input_latex, [method], summary, {}, None, False)


# --------------------------------------------------------------------------
# Finite differences for the heat equation
# --------------------------------------------------------------------------

def _whole(text, label, low, high):
    if not (text or "").strip():
        raise InputError("Enter %s." % (label[0].lower() + label[1:]))
    try:
        value = int((text or "").strip())
    except ValueError:
        raise InputError("%s must be a whole number." % label)
    if not low <= value <= high:
        raise InputError("%s must be between %d and %d." % (label, low, high))
    return value


def _thin(items, limit):
    if len(items) <= limit:
        return items
    stride = math.ceil((len(items) - 1) / (limit - 1))
    picked = items[::stride]
    if picked[-1] is not items[-1]:
        picked.append(items[-1])
    return picked


def _tridiagonal(lower, diag, upper, rhs):
    """Solve a tridiagonal system by forward elimination and back substitution."""
    n = len(rhs)
    c, d = [0.0] * n, [0.0] * n
    c[0], d[0] = upper / diag, rhs[0] / diag
    for i in range(1, n):
        denom = diag - lower * c[i - 1]
        c[i] = upper / denom
        d[i] = (rhs[i] - lower * d[i - 1]) / denom
    out = [0.0] * n
    out[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        out[i] = d[i] - c[i] * out[i + 1]
    return out


def finite_differences(f_text, L_text, k_text, n_text, dt_text, m_text):
    L = pdes._positive(L_text, "the length L")
    k = pdes._positive(k_text, "the diffusivity k")
    f, x = pdes._function(f_text, "x", "the initial temperature f(x)")
    intervals = _whole(n_text, "The number of intervals N", 2, 40)
    if not (dt_text or "").strip():
        raise InputError("Enter the time step.")
    dt = float(parse_number(dt_text, "The time step"))
    if dt <= 0:
        raise InputError("The time step must be a positive number.")
    count = _whole(m_text, "The number of time steps", 1, 400)
    Lf, kf = float(L), float(k)
    dx = Lf / intervals
    r = kf * dt / dx**2
    fn = pdes._callable(f, x)
    xs = [i * dx for i in range(intervals + 1)]
    first = [0.0] + [fn(v) for v in xs[1:-1]] + [0.0]

    exact = None
    try:
        series = pdes.heat(f_text, L_text, k_text, "00", "", "")
        exact_plot = series["methods"][0]["plot"]["particular_expr"]
        if exact_plot:
            xs_, ys_ = sp.Symbol("x"), sp.Symbol("y")
            exact = sp.lambdify((xs_, ys_), sp.sympify(exact_plot, locals={"x": xs_, "y": ys_}), "math")
    except Exception:
        exact_plot = None

    def run(scheme):
        levels, u = [list(first)], list(first)
        blew_up = False
        for _ in range(count):
            if scheme == "ftcs":
                new = [0.0] + [u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])
                               for i in range(1, intervals)] + [0.0]
            else:
                rhs = [(1 - r) * u[i] + r / 2 * (u[i + 1] + u[i - 1]) for i in range(1, intervals)]
                new = [0.0] + _tridiagonal(-r / 2, 1 + r, -r / 2, rhs) + [0.0]
            if any(not math.isfinite(v) or abs(v) > 1e12 for v in new):
                blew_up = True
                break
            levels.append(new)
            u = new
        return levels, blew_up

    scale_lo, scale_hi = pdes._range(first)
    width = max(scale_hi - scale_lo, 1e-9)
    columns = _thin(list(range(intervals + 1)), 9)
    headers = ["t"] + ["x = %s" % fmt(xs[i]) for i in columns]
    methods = []
    for scheme, name, formula, about in (
        ("ftcs", "Explicit scheme (FTCS)",
         "u_i^{j+1} = u_i^{j} + r\\left(u_{i+1}^{j} - 2u_i^{j} + u_{i-1}^{j}\\right)",
         "Forward difference in time, centred difference in space. Each new value comes "
         "directly from three old ones, but the scheme is stable only when \\(r \\le \\tfrac12\\)."),
        ("cn", "Crank\u2013Nicolson",
         "-\\tfrac{r}{2}u_{i-1}^{j+1} + (1+r)u_i^{j+1} - \\tfrac{r}{2}u_{i+1}^{j+1} = "
         "\\tfrac{r}{2}u_{i-1}^{j} + (1-r)u_i^{j} + \\tfrac{r}{2}u_{i+1}^{j}",
         "Averages the space difference between the old and new time levels. Each step "
         "needs a tridiagonal system solved, but it is stable for every \\(r\\) and more accurate."),
    ):
        levels, blew_up = run(scheme)
        unstable = scheme == "ftcs" and r > 0.5 + 1e-12
        steps = [
            step("Set up the grid",
                 "Divide the rod into \\(N\\) intervals and step forward in time. "
                 "\\(u_i^j\\) approximates \\(u(x_i, t_j)\\); the end values are 0.",
                 ["\\Delta x = \\frac{L}{N} = %s,\\qquad \\Delta t = %s" % (fmt(dx), fmt(dt)),
                  "x_i = i\\,\\Delta x,\\qquad t_j = j\\,\\Delta t"]),
            step("The difference formula", about, [formula]),
        ]
        stability = ("Here \\(r \\le \\tfrac12\\), so the explicit scheme is stable."
                     if scheme == "ftcs" and not unstable else
                     "Here \\(r > \\tfrac12\\), so the explicit scheme is unstable: rounding "
                     "errors grow at every step and the numbers below oscillate and blow up. "
                     "Use a smaller time step, or Crank\u2013Nicolson."
                     if unstable else
                     "Crank\u2013Nicolson is stable for any value of \\(r\\), so the time step "
                     "can be chosen for accuracy alone.")
        steps.append(step("The mesh ratio", stability,
                          ["r = \\frac{k\\,\\Delta t}{\\Delta x^2} = %s" % fmt(r)]))
        if len(levels) > 1:
            if scheme == "ftcs":
                steps.append(step(
                    "First time level with numbers",
                    "Apply the formula at the first interior point.",
                    ["u_1^{1} = %s + %s\\left(%s - 2(%s) + %s\\right) = %s"
                     % (fmt(first[1]), fmt(r), fmt(first[2]), fmt(first[1]), fmt(first[0]),
                        fmt(levels[1][1]))]))
            else:
                rhs = [(1 - r) * first[i] + r / 2 * (first[i + 1] + first[i - 1])
                       for i in range(1, intervals)]
                steps.append(step(
                    "The system for the first time level",
                    "The unknowns are the interior values at the new time. The matrix has "
                    "\\(1 + r\\) on the diagonal and \\(-r/2\\) beside it, and is the same at "
                    "every step.",
                    ["1 + r = %s,\\qquad -\\tfrac{r}{2} = %s" % (fmt(1 + r), fmt(-r / 2)),
                     "\\text{right-hand side: } " + ",\\ ".join(fmt(v) for v in rhs[:6])
                     + ("\\ \\ldots" if len(rhs) > 6 else ""),
                     "u_1^{1} = %s" % fmt(levels[1][1])]))
        table = step("Values on the grid",
                     "Each row is one time level." + (
                         " Long tables show a selection of rows and columns."
                         if len(levels) > 16 or len(columns) < intervals + 1 else ""))
        table["table"] = {"headers": headers,
                          "rows": [[fmt(j * dt)] + [fmt(levels[j][i]) for i in columns]
                                   for j in _thin(list(range(len(levels))), 16)]}
        steps.append(table)

        t_end = (len(levels) - 1) * dt
        lines = ["u(%s, %s) \\approx %s" % (fmt(xs[intervals // 2]), fmt(t_end),
                                            fmt(levels[-1][intervals // 2]))]
        error = None
        if exact is not None and not blew_up:
            try:
                error = max(abs(levels[-1][i] - float(exact(xs[i], t_end)))
                            for i in range(1, intervals))
                lines.append("\\text{largest difference from the series solution at } t = %s:\\ %s"
                             % (fmt(t_end), fmt(error)))
            except Exception:
                error = None
        text = ""
        if blew_up:
            text = ("The values exceeded \\(10^{12}\\) after %d steps and the calculation was "
                    "stopped: the scheme is unstable for this mesh ratio." % (len(levels) - 1))
        result = step("Result", text, lines)
        result["kind"] = "answer"
        steps.append(result)
        if unstable or blew_up:
            verified, note = None, "Unstable for this mesh ratio; the numbers are not meaningful."
        elif error is not None:
            ok = error < 0.25 * width
            verified = True if ok else None
            note = ("Compared with the series solution at the final time: largest difference %s."
                    % fmt(error).replace("\\times 10^", "e").replace("{", "").replace("}", ""))
        else:
            verified, note = None, "No exact solution was available to compare with."
        kept = _thin(list(range(len(levels))), MAX_LEVELS)
        clip_lo, clip_hi = scale_lo - 3 * width, scale_hi + 3 * width
        methods.append({
            "id": scheme, "name": name, "applicable": True, "reason": about, "steps": steps,
            "general_latex": None, "particular_latex": None,
            "plot": {"general_expr": None, "particular_expr": exact_plot, "points": [],
                     "levels": [{"t": j * dt,
                                 "points": [[xs[i], min(max(levels[j][i], clip_lo), clip_hi)]
                                            for i in range(intervals + 1)]} for j in kept]},
            "verified": verified, "verify_note": note})

    input_latex = ("u_t = %s\\,u_{xx},\\quad 0 < x < %s;\\quad u(0,t) = u(%s,t) = 0;\\quad "
                   "u(x,0) = %s" % (tex(k), tex(L), tex(L), tex(f)))
    summary = ("Finite-difference solution of the heat equation with mesh ratio r = %s. "
               "Switch between the two schemes to compare them."
               % fmt(r).replace("\\times 10^", "e").replace("{", "").replace("}", ""))
    if r > 0.5 + 1e-12:
        summary += " The explicit scheme is unstable here because r is above 1/2."
    lo = min(scale_lo, 0.0) - 0.05 * width
    hi = max(scale_hi, 0.0) + 0.05 * width
    return pdes._result("fd", input_latex, methods, summary,
                        {"x_min": 0.0, "x_max": Lf, "t_max": count * dt, "u_min": lo, "u_max": hi},
                        plot_fx(f, x), False)


# --------------------------------------------------------------------------
# Laplace's equation on a disk
# --------------------------------------------------------------------------

def _harmonics(n):
    """(Re, Im) of (x + i y)^n as polynomials: r^n cos(n theta) and r^n sin(n theta)."""
    expanded = sp.expand((X + sp.I * Y) ** n)
    return sp.re(expanded), sp.im(expanded)


def disk(f_text, R_text):
    R = pdes._positive(R_text, "the radius R")
    if not (f_text or "").strip():
        raise InputError("Enter the boundary values f as a function of the angle t.")
    f, symbols = parse_expression(f_text, ["t"])
    if f.has(sp.DiracDelta):
        raise InputError("An impulse cannot be used as boundary data here.")
    theta = symbols["t"]
    Rf = float(R)
    fn = pdes._callable(f, theta)

    # Midpoint rule on a grid with a cell edge at 0, so a jump there (a unit
    # step in the angle) is integrated exactly rather than smeared.
    cells = 4000
    nodes = [-math.pi + 2 * math.pi * (i + 0.5) / cells for i in range(cells)]
    data = [fn(v) for v in nodes]
    weight = 2.0 / cells      # (2 pi / cells) / pi

    a0 = sum(data) * weight
    if abs(a0) < 1e-7:
        a0 = 0.0
    modes = []       # (n, a_n, b_n)
    for n in range(1, pdes.SCAN + 1):
        an = sum(d * math.cos(n * v) for d, v in zip(data, nodes)) * weight
        bn = sum(d * math.sin(n * v) for d, v in zip(data, nodes)) * weight
        if abs(an) < 1e-7:
            an = 0.0
        if abs(bn) < 1e-7:
            bn = 0.0
        if an or bn:
            modes.append((n, an, bn))
        if len(modes) >= pdes.TERMS:
            break

    expr = sp.Float(round(a0 / 2, 12))
    for n, an, bn in modes:
        re, im = _harmonics(n)
        expr += (sp.Float(round(an / Rf**n, 14)) * re + sp.Float(round(bn / Rf**n, 14)) * im)
    plot = pdes._plot_xy(sp.expand(expr), X, Y)

    def u(xv, yv):
        radius = math.hypot(xv, yv)
        angle = math.atan2(yv, xv)
        total = a0 / 2
        for n, an, bn in modes:
            total += (radius / Rf) ** n * (an * math.cos(n * angle) + bn * math.sin(n * angle))
        return total

    # Coefficient formulas for the steps.
    def closed(weight):
        if f.has(sp.Heaviside):
            return None

        def work():
            value = sp.integrate(f * weight, (theta, -sp.pi, sp.pi))
            if value.has(sp.Integral) or value.has(sp.zoo, sp.nan, sp.oo):
                return None
            value = sp.simplify(value / sp.pi)
            if isinstance(value, sp.Piecewise):
                value = sp.simplify(value.args[-1][0])
            return value
        try:
            return run_with_timeout(work, pdes.SYMBOLIC_TIMEOUT, None)
        except Exception:
            return None
    an_formula = closed(sp.cos(N * theta))
    bn_formula = closed(sp.sin(N * theta))
    # A formula in n can miss individual n (cos 3t has only a_3, yet the
    # generic integral is 0). Show it only when it matches the first modes.
    numeric = {n: (an, bn) for n, an, bn in modes}
    if an_formula is not None and bn_formula is not None:
        for n in range(1, 9):
            try:
                got = (float(an_formula.subs(N, n)), float(bn_formula.subs(N, n)))
            except Exception:
                got = None
            want = numeric.get(n, (0.0, 0.0))
            if got is None or abs(got[0] - want[0]) > 1e-5 or abs(got[1] - want[1]) > 1e-5:
                an_formula = bn_formula = None
                break

    steps = [
        step("Write the equation in polar coordinates",
             "On a disk the boundary is \\(r = R\\), so use \\(r\\) and \\(\\theta\\) and look "
             "for product solutions \\(u = F(r)\\,G(\\theta)\\).",
             ["u_{rr} + \\frac{1}{r}u_r + \\frac{1}{r^2}u_{\\theta\\theta} = 0",
              "\\frac{r^2F'' + rF'}{F} = -\\frac{G''}{G} = \\lambda"]),
        step("The angular problem",
             "\\(G\\) must repeat when \\(\\theta\\) increases by \\(2\\pi\\). That periodicity "
             "is the boundary condition, and it forces \\(\\lambda = n^2\\).",
             ["G'' + \\lambda G = 0,\\qquad G(\\theta + 2\\pi) = G(\\theta)",
              "\\lambda_n = n^2,\\qquad G_n = \\cos n\\theta,\\ \\sin n\\theta \\quad (n = 0, 1, 2, \\ldots)"]),
        step("The radial problem",
             "The radial equation is a Cauchy\u2013Euler equation with solutions \\(r^n\\) and "
             "\\(r^{-n}\\) (or \\(1\\) and \\(\\ln r\\) for \\(n = 0\\)). The ones that blow up at "
             "the centre are discarded, because the temperature there is finite.",
             ["r^2F'' + rF' - n^2F = 0", "F_n = r^n"]),
        step("Superpose",
             "Add the product solutions. Setting \\(r = R\\) shows that the constants are the "
             "Fourier coefficients of the boundary values.",
             ["u(r,\\theta) = \\frac{a_0}{2} + \\sum_{n=1}^{\\infty}\\left(\\frac{r}{R}\\right)^n"
              "\\left(a_n\\cos n\\theta + b_n\\sin n\\theta\\right)",
              "a_n = \\frac{1}{\\pi}\\int_{-\\pi}^{\\pi} f(\\theta)\\cos n\\theta\\,d\\theta,\\qquad "
              "b_n = \\frac{1}{\\pi}\\int_{-\\pi}^{\\pi} f(\\theta)\\sin n\\theta\\,d\\theta"]),
    ]
    shown_tex = tex(f.subs(theta, sp.Symbol("theta")))
    coefficient = step("Compute the coefficients",
                       "Here \\(f(\\theta) = %s\\) for \\(-\\pi < \\theta < \\pi\\)." % shown_tex)
    if an_formula is not None and bn_formula is not None:
        coefficient["latex"] = ["a_0 = %s" % pdes._g(a0), "a_n = %s" % tex(an_formula),
                                "b_n = %s" % tex(bn_formula)]
    else:
        coefficient["text"] += (" The coefficients are listed by value (only the nonzero ones "
                                "are shown).")
        coefficient["latex"] = ["a_0 = %s" % pdes._g(a0)]
        coefficient["table"] = {"headers": ["n", "a_n", "b_n"],
                                "rows": [[str(n), fmt(an), fmt(bn)] for n, an, bn in modes[:8]]}
    steps.append(coefficient)

    pieces = []
    if abs(a0) > 1e-12:
        pieces.append(pdes._g(a0 / 2))
    ratio = "\\left(\\frac{r}{%s}\\right)" % tex(R) if R != 1 else "r"
    for n, an, bn in modes[:3]:
        power = ratio if n == 1 else "%s^{%d}" % (ratio, n)
        for value, wave in ((an, "\\cos"), (bn, "\\sin")):
            if value:
                pieces.append("%s%s%s %s\\theta" % (pdes._lead(value), power, wave,
                                                    "" if n == 1 else str(n)))
    solution = ("u(r,\\theta) " + pdes._join(pieces, len(modes) <= 3)) if pieces else "u(r,\\theta) = 0"
    steps.append(step(
        "The value at the centre",
        "At \\(r = 0\\) only the constant term survives, so the temperature at the centre is "
        "the average of the boundary values.",
        ["u(0,\\theta) = \\frac{a_0}{2} = \\frac{1}{2\\pi}\\int_{-\\pi}^{\\pi} f(\\theta)\\,d\\theta = %s"
         % pdes._g(a0 / 2)]))
    steps.append(step(
        "Poisson's integral formula",
        "Summing the series in closed form gives the solution as a weighted average of the "
        "boundary values.",
        ["u(r,\\theta) = \\frac{1}{2\\pi}\\int_{-\\pi}^{\\pi}"
         "\\frac{R^2 - r^2}{R^2 - 2Rr\\cos(\\theta - \\phi) + r^2}\\,f(\\phi)\\,d\\phi"]))

    # Checks: Laplacian, boundary values, maximum principle.
    h = Rf * 1e-3
    worst = 0.0
    for xv, yv in ((0.3 * Rf, 0.2 * Rf), (-0.4 * Rf, 0.35 * Rf), (0.1 * Rf, -0.6 * Rf)):
        lap = (u(xv + h, yv) + u(xv - h, yv) + u(xv, yv + h) + u(xv, yv - h) - 4 * u(xv, yv)) / h**2
        worst = max(worst, abs(lap) * Rf**2)
    samples = [fn(-math.pi + 2 * math.pi * (i + 0.5) / 40) for i in range(40)]
    lo, hi = pdes._range(samples)
    data_error = max(abs(u(Rf * math.cos(a), Rf * math.sin(a)) - fn(a))
                     for a in (-2.5, -1.3, -0.4, 0.7, 1.9, 2.8))
    verified, note = pdes._verdict(worst, 0.0, data_error, hi - lo, "the boundary values")
    interior = [u(0.9 * Rf * math.cos(a), 0.9 * Rf * math.sin(a)) for a in (0.1 * i for i in range(63))]
    if verified and (min(interior) < lo - 0.15 * (hi - lo) or max(interior) > hi + 0.15 * (hi - lo)):
        verified, note = False, "The interior values go outside the range of the boundary values."

    method = _plain("separation", "Separation of variables",
                    "In polar coordinates the boundary is a coordinate line, and periodicity "
                    "in the angle supplies the eigenvalue problem.",
                    steps, solution, plot, verified, note)
    input_latex = ("\\nabla^2 u = 0,\\quad r < %s;\\quad u(%s,\\theta) = %s"
                   % (tex(R), tex(R), shown_tex))
    return pdes._result(
        "disk", input_latex, [method],
        "Laplace's equation inside a circle: the steady temperature for the given boundary "
        "values. The graph shows it as a heat map over the disk.",
        {"x_min": -Rf, "x_max": Rf, "y_min": -Rf, "y_max": Rf, "u_min": lo, "u_max": hi,
         "mask": "disk"}, None, True)
