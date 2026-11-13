"""Entry points called from the Android app. Everything returns a JSON string."""
import json

import sympy as sp

from . import first_order
from .parsing import InputError, parse_equation, parse_number
from .util import dydx, plot_expr, tex, tex_side

SCHEMA = 1


def warmup():
    """Import SymPy and run one small solve so the first real one is fast."""
    solve("y' = x*y", "0", "1")
    return "ok"


def preview(text):
    """Render what the user has typed so far."""
    try:
        if not text.strip():
            return json.dumps({"ok": True, "latex": ""})
        prob, lhs, rhs = parse_equation(text)
        return json.dumps({"ok": True, "latex": "%s = %s" % (tex_side(lhs, prob), tex_side(rhs, prob))})
    except InputError as err:
        return json.dumps({"ok": False, "error": str(err)})
    except Exception:
        return json.dumps({"ok": False, "error": "I couldn't read that equation."})


def _fail(message):
    return json.dumps({"schema": SCHEMA, "ok": False, "error": message})


def solve(text, x0_text="", y0_text=""):
    """Solve a first-order ODE, optionally with y(x0) = y0, by every method that applies."""
    try:
        return json.dumps(_solve(text, x0_text or "", y0_text or ""))
    except InputError as err:
        return _fail(str(err))
    except Exception as err:  # never let a stack trace reach the UI
        return _fail("Something went wrong while solving this equation (%s)."
                     % type(err).__name__)


def _solve(text, x0_text, y0_text):
    prob, lhs, rhs = parse_equation(text)
    xn = prob.indep_name

    ic = None
    if x0_text.strip() or y0_text.strip():
        if not (x0_text.strip() and y0_text.strip()):
            raise InputError("For an initial condition, fill in both %s₀ and "
                             "y₀ (or leave both empty)." % xn)
        ic = (parse_number(x0_text, "%s₀" % xn), parse_number(y0_text, "y₀"))

    # Isolate y'.
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
    d = dydx(prob)
    input_latex = "%s = %s" % (tex_side(lhs, prob), tex_side(rhs, prob))
    normal_latex = "%s = %s" % (d, tex(f))
    already_normal = (lhs == prob.yp and sp.simplify(rhs - f) == 0)

    if ic:
        try:
            at = f.subs({prob.x: ic[0], prob.y: ic[1]})
            defined = not at.has(sp.zoo, sp.nan, sp.oo) and at.is_real is not False
        except Exception:
            defined = False
        if not defined:
            raise InputError("The equation is not defined at the initial point "
                             "(%s, %s). Choose a different initial condition."
                             % (ic[0], ic[1]))

    methods = []
    for method in first_order.METHODS:
        try:
            methods.append(method(prob, f, ic))
        except Exception as err:
            methods.append({
                "id": method.__name__, "name": method.__name__.capitalize(),
                "applicable": False,
                "reason": "This method ran into an internal error (%s)." % type(err).__name__,
            })

    prelude = None
    if not already_normal:
        prelude = first_order.step(
            "Solve for the derivative",
            "Rearrange the equation so the derivative is alone on the left.",
            [input_latex, normal_latex])
    for m in methods:
        if m["applicable"] and prelude:
            m["steps"].insert(0, prelude)
        if m["applicable"]:
            _add_closing_steps(m, ic, prob)

    applicable = [m for m in methods if m["applicable"]]
    others = []
    if not applicable:
        others = [t for t in first_order.other_types(prob, f)
                  if t not in ("linear",)]
    if applicable:
        names = " and ".join(m["name"].lower() for m in applicable)
        summary = "First-order equation. %d method%s appl%s: %s." % (
            len(applicable), "" if len(applicable) == 1 else "s",
            "ies" if len(applicable) == 1 else "y", names)
    elif others:
        summary = ("First-order equation that is neither separable nor linear. "
                   "It looks %s; step-by-step support for that type is coming. "
                   "The graph below uses a numerical solution." % " / ".join(others))
    else:
        summary = ("First-order equation that is neither separable nor linear. "
                   "The graph below uses a numerical solution.")

    return {
        "schema": SCHEMA, "ok": True,
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


def _add_closing_steps(m, ic, prob):
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
