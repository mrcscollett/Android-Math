"""Entry points called from the Android app. Everything returns a JSON string."""
import json

import sympy as sp

from . import first_order, numeric as numeric_methods, second_order, transforms
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


def preview(text, kind="ode"):
    """Render what the user has typed so far.

    kind is "ode" for an equation, "forward" for f(t), "inverse" for F(s).
    """
    try:
        if not text.strip():
            return json.dumps({"ok": True, "latex": "", "order": 1})
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
        return json.dumps({"ok": False, "error": "I couldn't read that equation."})


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
        if m["applicable"] and prelude and m["id"] != "exact":
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
