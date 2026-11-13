"""Laplace-transform solution of linear constant-coefficient IVPs at 0."""
import sympy as sp

from .first_order import _eq, _paren, step
from .util import TIMEOUT, run_with_timeout, tex


def solve_ivp(prob, a, b, c, g, y0, y1=None):
    """Solve a y'' + b y' + c y = g with y(0) = y0 (and y'(0) = y1 if a != 0).

    Returns (steps, solution) or (None, reason) when the transforms fail.
    """
    x, xn = prob.x, prob.indep_name
    s = sp.Symbol("s")
    Y = sp.Symbol("Y")

    if g == 0:
        G = sp.Integer(0)
    else:
        G = run_with_timeout(
            lambda: sp.laplace_transform(g, x, s, noconds=True), TIMEOUT, None)
        if G is None or G.has(sp.LaplaceTransform) or G.has(sp.Integral):
            return None, ("The Laplace transform of the right-hand side "
                          "\\(%s\\) could not be found." % tex(g))
        G = sp.simplify(G)

    rules = ["\\mathcal{L}\\{y'\\} = sY - y(0)"]
    left = sp.Integer(0)
    shown = []
    if a != 0:
        rules.insert(0, "\\mathcal{L}\\{y''\\} = s^2 Y - s\\,y(0) - y'(0)")
        left += a * (s**2 * Y - s * y0 - y1)
        shown.append("%s\\left(s^2 Y - s\\,y(0) - y'(0)\\right)" % _coef(a))
    if b != 0:
        left += b * (s * Y - y0)
        shown.append("%s\\left(sY - y(0)\\right)" % _coef(b))
    if c != 0:
        left += c * Y
        shown.append("%sY" % _coef(c))
    template = " + ".join(shown).replace("+ -", "- ")

    steps = [step(
        "Transform both sides",
        "Take the Laplace transform of each term, writing "
        "\\(Y(s) = \\mathcal{L}\\{y\\}\\). Derivatives turn into "
        "multiplication by \\(s\\), with the initial values appearing.",
        ["%s = %s" % (template, tex(G))],
        [step("Rules used", "", rules + (
            ["\\mathcal{L}\\left\\{%s\\right\\} = %s" % (tex(g), tex(G))] if g != 0 else []))])]

    values = "y(0) = %s" % tex(y0)
    if a != 0:
        values += ",\\qquad y'(0) = %s" % tex(y1)
    steps.append(step(
        "Insert the initial values",
        "Replace \\(y(0)\\)%s with the given numbers and collect the \\(Y\\) terms."
        % (" and \\(y'(0)\\)" if a != 0 else ""),
        [values, "%s = %s" % (tex(sp.collect(sp.expand(left), Y)), tex(G))]))

    solved = sp.solve(sp.Eq(left, G), Y)
    if len(solved) != 1:
        return None, "The transformed equation could not be solved for \\(Y(s)\\)."
    Ys = sp.factor(sp.together(solved[0]))
    steps.append(step("Solve for Y(s)",
                      "This is now an algebra problem.", [_eq("Y(s)", Ys)]))

    try:
        parts = sp.apart(Ys, s)
    except Exception:
        parts = Ys
    if parts != Ys:
        steps.append(step(
            "Split into partial fractions",
            "Break \\(Y(s)\\) into pieces that each appear in a table of "
            "Laplace transforms.", [_eq("Y(s)", parts)]))

    y = run_with_timeout(
        lambda: sp.inverse_laplace_transform(parts, s, x), TIMEOUT, None)
    if y is None or y.has(sp.InverseLaplaceTransform) or y.has(sp.DiracDelta):
        return None, "The inverse Laplace transform could not be found."
    # u(t) is 1 for t >= 0; shifted steps u(t - a) are part of the answer.
    y = y.replace(lambda e: isinstance(e, sp.Heaviside) and e.args[0] == x,
                  lambda e: sp.Integer(1))
    y = sp.expand(y)
    if y.has(sp.I):
        y = sp.simplify(sp.re(sp.expand_complex(y)))
    steps.append(step(
        "Transform back",
        "Take the inverse transform of each piece. The result is the solution "
        "for \\(%s \\ge 0\\)." % xn,
        [_eq("y", y)],
        [step("Table entries used", "", [
            "\\mathcal{L}^{-1}\\left\\{\\frac{1}{s - a}\\right\\} = e^{a%s}" % xn,
            "\\mathcal{L}^{-1}\\left\\{\\frac{n!}{s^{n+1}}\\right\\} = %s^n" % xn,
            "\\mathcal{L}^{-1}\\left\\{\\frac{b}{(s - a)^2 + b^2}\\right\\} = e^{a%s}\\sin b%s" % (xn, xn),
            "\\mathcal{L}^{-1}\\left\\{\\frac{s - a}{(s - a)^2 + b^2}\\right\\} = e^{a%s}\\cos b%s" % (xn, xn),
        ])]))
    return steps, y


def _coef(value):
    if value == 1:
        return ""
    if value == -1:
        return "-"
    return _paren(value) if value.is_Add else tex(value)


NAME = "Laplace transform"


def why_not(prob, constant, ic):
    """Reason the method cannot be used, or None if it can."""
    if not constant:
        return "The method needs constant coefficients."
    if not ic:
        return ("It needs initial conditions at \\(%s = 0\\); add them to see "
                "this method." % prob.indep_name)
    if ic[0] != 0:
        return ("The initial conditions are given at \\(%s = %s\\); this "
                "method is set up for conditions at 0."
                % (prob.indep_name, tex(ic[0])))
    return None


def forced(expr):
    """True when the forcing contains a unit step or an impulse."""
    return expr.has(sp.Heaviside, sp.DiracDelta)


def check_forced(residual, x):
    """Numerical check for t > 0, away from the instants where the forcing jumps."""
    import random
    rng = random.Random(11)
    good = 0
    for _ in range(60):
        point = sp.Float(rng.uniform(0.05, 8.0))
        try:
            value = complex(residual.evalf(subs={x: point}))
        except Exception:
            continue
        if value != value or abs(value) == float("inf"):
            continue
        if abs(value) > 1e-6:
            return False
        good += 1
    return True if good >= 6 else None
