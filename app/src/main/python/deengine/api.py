"""Entry points called from the Android app. Everything returns a JSON string."""
import json

import sympy as sp

from . import (boundary as boundary_problems, first_order, fourier as fourier_series,
               higher_order, numeric as numeric_methods, second_order, systems, transforms)
from .parsing import InputError, parse_equation, parse_number
from .util import dydx, plot_expr, tex, tex_side

SCHEMA = 1


def warmup():
    """Import SymPy and run one small solve so the first real one is fast."""
    solve("y' = x*y", "0", "1")
    return "ok"


def transform(text, direction="forward"):
    """Laplace transform of f(t) ("forward") or inverse transform of F(s) ("inverse")."""
    work = transforms.inverse if direction == "inverse" else transforms.forward
    return _guard(lambda: work(text))


def system(f_text, g_text, x0_text="", y0_text="", h_text="", z0_text=""):
    """Solve or analyse x' = f, y' = g (and optionally z' = h for a 3x3 linear system)."""
    return _guard(lambda: systems.solve(f_text or "", g_text or "", x0_text or "", y0_text or "",
                                        h_text or "", z0_text or ""))


def fourier(f_text, left_text="", L_text=""):
    """Fourier series of f on (-L, L), plus half-range sine and cosine series on (0, L)."""
    return _guard(lambda: fourier_series.solve(f_text or "", left_text or "", L_text or ""))


def boundary(text, a_text, a_kind, a_value, b_text, b_kind, b_value):
    """Second-order boundary value problem; each condition is on "y" or "dy" at a point."""
    return _guard(lambda: boundary_problems.solve(text or "", a_text, a_kind, a_value,
                                                  b_text, b_kind, b_value))


def eigen_problem(kind, L_text):
    """y'' + lambda y = 0 on [0, L]; kind is dd, nn, dn or nd (d: y = 0, n: y' = 0)."""
    return _guard(lambda: boundary_problems.eigen(kind, L_text))


def pde(kind, f_text, g_text="", p1="", p2="", bc="", t1="", t2=""):
    """Heat ("heat"), wave ("wave") or Laplace ("laplace") equation by separation of variables.

    heat: f, -, L, k, bc in {00, nn, 0n, n0, tt}, T1, T2.  wave: f, g, L, c.
    laplace: f, -, a, b, side in {top, bottom, left, right}.
    characteristics (a u_x + b u_t = c u + d): f, -, a, b, -, c, d.
    classify (A u_xx + B u_xy + C u_yy): -, -, A, B, -, C.
    fd (finite differences for the heat equation): f, steps, L, k, -, N, dt.
    disk (Laplace's equation, u(R, t) = f(t) with t the angle): f, -, R.
    """
    from . import pdes as pde_module
    return _guard(lambda: pde_module.solve(kind, f_text or "", g_text or "", p1 or "", p2 or "",
                                           bc or "", t1 or "", t2 or ""))


def _tool_module(name):
    from . import (calculus as calculus_tools, complexnum, dynamics, ftransform,
                   linalg as linear_algebra, powerseries, recurrence, special, vector)
    modules = {"linalg": linear_algebra, "calculus": calculus_tools,
               "series": powerseries, "complex": complexnum,
               "special": special, "ftransform": ftransform, "recurrence": recurrence,
               "dynamics": dynamics, "vector": vector}
    if name not in modules:
        raise InputError("Unknown tool.")
    return modules[name]


def tool(name, operation, text, extra1="", extra2=""):
    """The stand-alone math tools. `name` is linalg, calculus, series, complex,
    special, ftransform, recurrence, dynamics or vector.

    linalg: rref, solve (extra1 = b), det, inverse, eigen, spaces, independence, wronskian.
    calculus: integral, definite (extra1, extra2 = limits), fractions, derivative.
    series: taylor (extra1 = centre, extra2 = order), convergence, radius (extra1 = centre).
    complex: evaluate, roots (extra1 = n).
    special: bessel (text = order), legendre / hermite / laguerre / chebyshev (text = degree),
        sturm (text = "a; b; c; d", extra1, extra2 = interval ends), expansion (extra1 = terms).
    ftransform: forward, inverse (in w, with i), sine, cosine, heat (extra1 = k).
    recurrence: solve (text such as "a(n+2) = a(n+1) + a(n)", extra1 = starting values).
    dynamics: lyapunov (text = "f; g", extra1 = your V(x, y), extra2 = equilibrium "a, b"),
        conservative (text = F(x) in x'' = F(x)), cycles (text = "f; g"),
        bifurcation (text = f(x, m)), hopf (text = "f; g" with m, extra1 = m for the graph).
    vector: gradient (text = f, extra1 = point, extra2 = direction), divcurl (text = "P; Q[; R]"),
        line (text = field or scalar, extra1 = curve "x(t); y(t)", extra2 = "a, b"),
        double (text = f(x, y), extra1 = x limits, extra2 = y limits), polar (extra1 = r limits,
        extra2 = angle limits), green (text = "P; Q", extra1 = region), flux and stokes
        (text = "P; Q; R", extra1 = surface such as "sphere 2").
    """
    def work():
        module = _tool_module(name)
        if name == "linalg":
            return module.solve(operation, text or "", extra1 or "")
        if name == "complex":
            return module.solve(operation, text or "", extra1 or "")
        return module.solve(operation, text or "", extra1 or "", extra2 or "")
    return _guard(work)


def linalg(operation, matrix_text, rhs_text=""):
    """Linear algebra on a matrix typed as rows separated by ';' (see tool)."""
    return tool("linalg", operation, matrix_text, rhs_text)


def calculus(operation, text, a_text="", b_text=""):
    """Integrals, partial fractions and derivatives (see tool)."""
    return tool("calculus", operation, text, a_text, b_text)


def series_tool(operation, text, a_text="", b_text=""):
    """Taylor series, convergence tests and radius of convergence (see tool)."""
    return tool("series", operation, text, a_text, b_text)


def complex_number(operation, text, n_text=""):
    """Complex arithmetic, polar form and roots (see tool)."""
    return tool("complex", operation, text, n_text)


def preview(text, kind="ode"):
    """Render what the user has typed so far.

    kind is "ode" for an equation, "forward" for f(t), "inverse" for F(s),
    "system" for a pair of right-hand sides written as "f;g",
    "fourier" for f(x) and an optional left-half formula written as "f;left",
    and "tool:operation" for the math tools (e.g. "linalg:det", "series:taylor").
    """
    try:
        if not text.strip():
            return json.dumps({"ok": True, "latex": "", "order": 1})
        if ":" in kind:
            # A math tool, written "tool:operation" (for example "linalg:det").
            name, operation = kind.split(":", 1)
            return json.dumps({"ok": True, "order": 0,
                               "latex": _tool_module(name).preview(operation, text)})
        if kind in ("pde", "pdey", "pdet"):
            from . import pdes as pde_module
            return json.dumps({"ok": True, "order": 0,
                               "latex": pde_module.preview(text, {"pdey": "y", "pdet": "t"}.get(kind, "x"))})
        if kind == "fourier":
            return json.dumps({"ok": True, "order": 0, "latex": fourier_series.preview(text)})
        if kind == "system":
            return json.dumps({"ok": True, "order": 1, "latex": systems.preview(text)})
        if kind in ("forward", "inverse"):
            from .parsing import parse_expression
            expr, _ = parse_expression(text, ["s" if kind == "inverse" else "t"])
            wrap = "\\mathcal{L}^{-1}\\left\\{%s\\right\\}" if kind == "inverse" \
                else "\\mathcal{L}\\left\\{%s\\right\\}"
            return json.dumps({"ok": True, "order": 0, "latex": wrap % tex(expr)})
        prob, lhs, rhs = parse_equation(text)
        return json.dumps({"ok": True, "order": prob.order,
                           "latex": "%s = %s" % (tex_side(lhs, prob), tex_side(rhs, prob))})
    except InputError as err:
        return json.dumps({"ok": False, "error": str(err)})
    except Exception:
        return json.dumps({"ok": False, "error": "I couldn't read that."
                           if ":" in kind else "I couldn't read that equation."})


def _fail(message):
    return json.dumps({"schema": SCHEMA, "ok": False, "error": message})


def _guard(work):
    try:
        return json.dumps(work())
    except InputError as err:
        return _fail(str(err))
    except Exception as err:  # never let a stack trace reach the UI
        return _fail("Something went wrong while solving this equation (%s)."
                     % type(err).__name__)


def solve(text, x0_text="", y0_text="", y1_text=""):
    """Solve an ODE by every method that applies.

    First order takes an optional y(x0) = y0; second order takes an optional
    y(x0) = y0, y'(x0) = y1.
    """
    return _guard(lambda: _solve(text, x0_text or "", y0_text or "", y1_text or ""))


def numeric(text, x0_text, y0_text, h_text, n_text):
    """Approximate y' = f(x, y), y(x0) = y0 with Euler, improved Euler and RK4."""
    return _guard(lambda: _numeric(text, x0_text or "", y0_text or "",
                                   h_text or "", n_text or ""))


def _isolate(prob, lhs, rhs):
    """Rewrite a first-order equation as y' = f(x, y)."""
    expr = sp.expand(lhs - rhs)
    try:
        poly = sp.Poly(expr, prob.yp)
    except sp.PolynomialError:
        poly = None
    if poly is None or poly.degree() != 1:
        raise InputError("I can only handle equations where y' appears once, to "
                         "the first power, so it can be isolated.")
    f = sp.simplify(-poly.nth(0) / poly.nth(1))
    if f.has(prob.yp):
        raise InputError("I couldn't isolate y' in that equation.")
    # The form the user wrote, M + N y' = 0, is the natural one to test for exactness.
    prob.user_form = (sp.expand(poly.nth(0)), sp.expand(poly.nth(1)))
    return f


def _check_point(prob, f, ic):
    try:
        at = f.subs({prob.x: ic[0], prob.y: ic[1]})
        defined = not at.has(sp.zoo, sp.nan, sp.oo) and at.is_real is not False
    except Exception:
        defined = False
    if not defined:
        raise InputError("The equation is not defined at the initial point "
                         "(%s, %s). Choose a different initial condition."
                         % (ic[0], ic[1]))


def _run_methods(methods, *args):
    out = []
    for method in methods:
        try:
            out.append(method(*args))
        except Exception as err:
            out.append({
                "id": method.__name__,
                "name": method.__name__.replace("_", " ").capitalize(),
                "applicable": False,
                "reason": "This method ran into an internal error (%s)." % type(err).__name__,
            })
    return out


def _method_list(applicable):
    names = [m["name"] for m in applicable]
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _solve(text, x0_text, y0_text, y1_text):
    prob, lhs, rhs = parse_equation(text)
    if prob.order == 2:
        return _solve_second(prob, lhs, rhs, x0_text, y0_text, y1_text)
    if prob.order > 2:
        return _solve_higher(prob, lhs, rhs, x0_text, y0_text, y1_text)
    xn = prob.indep_name

    ic = None
    if x0_text.strip() or y0_text.strip():
        if not (x0_text.strip() and y0_text.strip()):
            raise InputError("For an initial condition, fill in both %s₀ and "
                             "y₀ (or leave both empty)." % xn)
        ic = (parse_number(x0_text, "%s₀" % xn), parse_number(y0_text, "y₀"))

    f = _isolate(prob, lhs, rhs)
    input_latex = "%s = %s" % (tex_side(lhs, prob), tex_side(rhs, prob))
    normal_latex = "%s = %s" % (dydx(prob), tex(f))
    already_normal = (lhs == prob.yp and sp.simplify(rhs - f) == 0)
    if ic:
        _check_point(prob, f, ic)

    if f.has(sp.Heaviside, sp.DiracDelta):
        # Step or impulse forcing: only the Laplace method is set up for it.
        methods = _run_methods((first_order.laplace,), prob, f, ic)
        if not ic and not methods[0]["applicable"]:
            raise InputError("Equations with a step u(...) or impulse delta(...) "
                             "are solved by Laplace transforms, which need an "
                             "initial condition at %s = 0." % xn)
    else:
        methods = _run_methods(first_order.METHODS, prob, f, ic)

    prelude = None
    if not already_normal:
        prelude = first_order.step(
            "Solve for the derivative",
            "Rearrange the equation so the derivative is alone on the left.",
            [input_latex, normal_latex])
    for m in methods:
        # Exact works from the form as written, so it needs no rearranging step.
        if m["applicable"] and prelude and m["id"] not in ("exact", "phase_line"):
            m["steps"].insert(0, prelude)
        if m["applicable"]:
            _add_closing_steps(m, ic)

    applicable = [m for m in methods if m["applicable"]]
    others = []
    if not applicable:
        others = first_order.other_types(prob, f)
    if applicable:
        summary = "First-order equation. %d method%s appl%s: %s." % (
            len(applicable), "" if len(applicable) == 1 else "s",
            "ies" if len(applicable) == 1 else "y", _method_list(applicable))
    elif others:
        summary = ("None of the step-by-step methods here applies to this "
                   "equation. It looks %s; support for that type is coming. "
                   "The graph uses a numerical solution." % " / ".join(others))
    else:
        summary = ("None of the step-by-step methods here applies to this "
                   "equation. The graph uses a numerical solution.")

    return {
        "schema": SCHEMA, "ok": True, "kind": "symbolic", "order": 1,
        "indep": xn, "dep": "y",
        "input_latex": input_latex,
        "normal_latex": normal_latex,
        "rhs_expr": plot_expr(f, prob),
        "ic": None if not ic else {
            "x0": float(ic[0]), "y0": float(ic[1]),
            "latex": "y(%s) = %s" % (tex(ic[0]), tex(ic[1])),
        },
        "summary": summary,
        "other_types": others,
        "methods": methods,
    }


def _solve_second(prob, lhs, rhs, x0_text, y0_text, y1_text):
    xn = prob.indep_name
    given = [t.strip() for t in (x0_text, y0_text, y1_text)]
    ic = None
    if any(given):
        if not all(given):
            raise InputError("A second-order equation needs both y(%s₀) and "
                             "y'(%s₀): fill in all three boxes, or leave "
                             "them all empty." % (xn, xn))
        ic = (parse_number(x0_text, "%s₀" % xn),
              parse_number(y0_text, "y(%s₀)" % xn),
              parse_number(y1_text, "y'(%s₀)" % xn))

    eq = second_order.analyse(prob, lhs, rhs)
    input_latex = "%s = %s" % (tex_side(lhs, prob), tex_side(rhs, prob))
    methods = _run_methods(second_order.METHODS, eq, ic)
    for m in methods:
        if m["applicable"]:
            _add_closing_steps(m, ic)

    applicable = [m for m in methods if m["applicable"]]
    if applicable:
        summary = "Second-order linear equation. %d method%s appl%s: %s." % (
            len(applicable), "" if len(applicable) == 1 else "s",
            "ies" if len(applicable) == 1 else "y", _method_list(applicable))
    else:
        summary = ("This is a linear second-order equation with variable "
                   "coefficients that none of the methods here covers. Supported: "
                   "constant coefficients, Cauchy-Euler equations, and series "
                   "solutions about 0 for homogeneous equations with polynomial "
                   "coefficients.")
    return {
        "schema": SCHEMA, "ok": True, "kind": "symbolic", "order": 2,
        "indep": xn, "dep": "y",
        "input_latex": input_latex,
        "normal_latex": "%s = %s" % (eq.lhs_tex(), tex(eq.g)),
        "rhs_expr": None,
        "ic": None if not ic else {
            "x0": float(ic[0]), "y0": float(ic[1]), "y1": float(ic[2]),
            "latex": "y(%s) = %s,\\quad y'(%s) = %s"
                     % (tex(ic[0]), tex(ic[1]), tex(ic[0]), tex(ic[2])),
        },
        "summary": summary,
        "other_types": [],
        "methods": methods,
    }


def _solve_higher(prob, lhs, rhs, x0_text, y0_text, rest_text):
    """Order 3 or 4. rest_text lists y'(x0), y''(x0), ... separated by commas."""
    xn, n = prob.indep_name, prob.order
    names = ["y", "y'", "y''", "y'''"][:n]
    ic = None
    if x0_text.strip() or y0_text.strip() or rest_text.strip():
        rest = [part.strip() for part in rest_text.split(",") if part.strip()]
        if not (x0_text.strip() and y0_text.strip()) or len(rest) != n - 1:
            raise InputError(
                "An equation of order %d needs %d initial values at %s\u2080: y, then "
                "%s separated by commas in the last box (or leave every box empty)."
                % (n, n, xn, ", ".join(names[1:])))
        x0 = parse_number(x0_text, "%s\u2080" % xn)
        values = [parse_number(y0_text, "y(%s\u2080)" % xn)] + [
            parse_number(part, "%s(%s\u2080)" % (name, xn)) for part, name in zip(rest, names[1:])]
        ic = (x0, values)
    eq = higher_order.analyse(prob, lhs, rhs)
    input_latex = "%s = %s" % (tex_side(lhs, prob), tex_side(rhs, prob))
    methods = _run_methods(higher_order.METHODS, eq, ic)
    for m in methods:
        if m["applicable"]:
            _add_closing_steps(m, ic)
    applicable = [m for m in methods if m["applicable"]]
    ordinal = "Third" if n == 3 else "Fourth"
    if applicable:
        summary = "%s-order linear equation with constant coefficients. Solved by: %s." % (
            ordinal, _method_list(applicable))
    else:
        summary = ("%s-order linear equation that the methods here cannot solve: %s"
                   % (ordinal, "the right-hand side is outside the family undetermined "
                               "coefficients handles, or the characteristic roots have no closed form."))
    return {
        "schema": SCHEMA, "ok": True, "kind": "symbolic", "order": n,
        "indep": xn, "dep": "y",
        "input_latex": input_latex,
        "normal_latex": "%s = %s" % (eq.lhs_tex(), tex(eq.g)),
        "rhs_expr": None,
        "ic": None if not ic else {
            "x0": float(ic[0]), "y0": float(ic[1][0]), "y1": float(ic[1][1]),
            "latex": ",\\quad ".join("%s(%s) = %s" % (name, tex(ic[0]), tex(v))
                                      for name, v in zip(names, ic[1])),
        },
        "summary": summary, "other_types": [], "methods": methods,
    }


def _numeric(text, x0_text, y0_text, h_text, n_text):
    prob, lhs, rhs = parse_equation(text)
    if prob.order != 1:
        raise InputError("The numerical methods here work on first-order "
                         "equations y' = f(x, y).")
    xn = prob.indep_name
    if not (x0_text.strip() and y0_text.strip()):
        raise InputError("Numerical methods need a starting point: fill in "
                         "%s₀ and y₀." % xn)
    ic = (parse_number(x0_text, "%s₀" % xn), parse_number(y0_text, "y₀"))
    if not h_text.strip():
        raise InputError("Enter a step size h, for example 0.1.")
    h = float(parse_number(h_text, "The step size h"))
    if h == 0:
        raise InputError("The step size h cannot be 0.")
    try:
        n = int(n_text.strip())
    except ValueError:
        raise InputError("The number of steps must be a whole number.")
    if not 1 <= n <= numeric_methods.MAX_STEPS:
        raise InputError("The number of steps must be between 1 and %d."
                         % numeric_methods.MAX_STEPS)
    f = _isolate(prob, lhs, rhs)
    _check_point(prob, f, ic)
    input_latex = "%s = %s" % (tex_side(lhs, prob), tex_side(rhs, prob))
    return numeric_methods.solve(prob, f, ic, h, n, input_latex)


def _add_closing_steps(m, ic):
    """Append the 'answer' and 'check' steps shared by every method."""
    if m.get("qualitative"):
        return  # no formula to state or to substitute back
    answer = m.get("particular_latex") if ic else None
    answer = answer or m.get("general_latex")
    if answer:
        label = "Solution" if (ic and m.get("particular_latex")) else "General solution"
        extra = ""
        latex = [answer]
        if m.get("also_latex") and not (ic and m.get("particular_latex")):
            extra = "Together with the constant solutions found earlier:"
            latex.append(m["also_latex"])
        m["steps"].append({"title": label, "text": extra, "latex": latex,
                           "substeps": [], "kind": "answer"})
    kind = {True: "check-ok", False: "check-bad", None: "check-unknown"}[m["verified"]]
    m["steps"].append({"title": "Check", "text": m["verify_note"], "latex": [],
                       "substeps": [], "kind": kind})
