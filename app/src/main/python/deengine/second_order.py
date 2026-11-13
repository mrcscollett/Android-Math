"""Second-order linear ODE methods: a y'' + b y' + c y = g(x).

Constant coefficients are solved by the characteristic equation, then
undetermined coefficients, variation of parameters and (for initial value
problems at 0) Laplace transforms. Homogeneous Cauchy-Euler equations are
handled by the substitution y = x^r.
"""
import sympy as sp
from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match

from .first_order import _eq, _not_applicable, _paren, step
from .parsing import InputError
from .util import (has_integral, integrate, is_zero, plot_expr,
                   residual_status, tex, with_abs)

_COEFF_NAMES = ["A", "B", "D", "E", "F", "G", "H", "K", "L", "M"]


class Linear2:
    """The equation a y'' + b y' + c y = g with its symbols."""

    def __init__(self, prob, a, b, c, g):
        self.prob, self.a, self.b, self.c, self.g = prob, a, b, c, g
        self.x = prob.x
        self.xn = prob.indep_name
        self.constant = not (a.has(prob.x) or b.has(prob.x) or c.has(prob.x))

    def residual(self, sol):
        x = self.x
        return (self.a * sp.diff(sol, x, 2) + self.b * sp.diff(sol, x)
                + self.c * sol - self.g)

    def lhs_tex(self):
        """LaTeX of a y'' + b y' + c y."""
        out = ""
        for coef, name in ((self.a, "y''"), (self.b, "y'"), (self.c, "y")):
            if coef == 0:
                continue
            negative = coef.could_extract_minus_sign() and not coef.is_Add
            mag = -coef if negative else coef
            body = name if mag == 1 else "%s\\,%s" % (_paren(mag), name)
            if not out:
                out = ("-" if negative else "") + body
            else:
                out += (" - " if negative else " + ") + body
        return out


def analyse(prob, lhs, rhs):
    """Read off a, b, c, g or raise InputError if the equation is not linear."""
    expr = sp.expand(lhs - rhs)
    try:
        poly = sp.Poly(expr, prob.ypp, prob.yp, prob.y)
    except sp.PolynomialError:
        poly = None
    if poly is None or poly.total_degree() != 1:
        raise InputError("Second-order support covers linear equations "
                         "a y'' + b y' + c y = g(x). In this one, y or its "
                         "derivatives appear nonlinearly.")
    a = sp.simplify(poly.coeff_monomial(prob.ypp))
    b = sp.simplify(poly.coeff_monomial(prob.yp))
    c = sp.simplify(poly.coeff_monomial(prob.y))
    g = sp.simplify(-poly.coeff_monomial(1))
    if a == 0:
        raise InputError("I couldn't find the y'' term.")
    return Linear2(prob, a, b, c, g)


def check(eq, sol, ic=None):
    """Substitute back. Returns (verified, note)."""
    prob = eq.prob
    status = residual_status(eq.residual(sol), {prob.x: ic[0]} if ic else None)
    if status in ("bad", "partial"):
        return False, "Substituting back did not reproduce the equation."
    if status == "unknown":
        return None, "Could not be checked automatically."
    how = "symbolically" if status == "symbolic" else "numerically at sample points"
    if ic:
        x0, y0, y1 = ic
        try:
            d0 = sol.subs(prob.x, x0) - y0
            d1 = sp.diff(sol, prob.x).subs(prob.x, x0) - y1
            ok = abs(complex(sp.N(d0))) < 1e-9 and abs(complex(sp.N(d1))) < 1e-9
        except Exception:
            ok = False
        if not ok:
            return False, "The initial conditions are not satisfied."
        return True, ("Checked %s by substituting back, and it meets both "
                      "initial conditions." % how)
    return True, "Checked %s by substituting back into the equation." % how


def _package(method_id, name, reason, steps, eq, general, particular, ic,
             general_tex=None):
    target = particular if particular is not None else general
    verified, note = None, "Could not be checked automatically."
    if target is not None and not has_integral(target):
        verified, note = check(eq, target, ic if particular is not None else None)
    elif target is not None:
        note = "Left in integral form; nothing to substitute back."
    prob = eq.prob
    return {
        "id": method_id, "name": name, "applicable": True, "reason": reason,
        "steps": steps,
        "general_latex": general_tex or (_eq("y", general) if general is not None else None),
        "particular_latex": _eq("y", particular) if particular is not None else None,
        "plot": {"general_expr": plot_expr(general, prob),
                 "particular_expr": plot_expr(particular, prob)},
        "verified": verified, "verify_note": note,
    }


def _apply_ic(eq, general, ic, steps):
    """Solve for C1, C2 from y(x0) = y0, y'(x0) = y1. Returns the particular solution."""
    prob = eq.prob
    x0, y0, y1 = ic
    xn = eq.xn
    try:
        at0 = general.subs(prob.x, x0)
        slope0 = sp.diff(general, prob.x).subs(prob.x, x0)
        defined = not (at0.has(sp.zoo, sp.nan, sp.oo) or slope0.has(sp.zoo, sp.nan, sp.oo))
        found = sp.solve([at0 - y0, slope0 - y1], [prob.C1, prob.C2], dict=True) if defined else []
    except Exception:
        found = []
    if not found:
        steps.append(step(
            "Apply the initial conditions",
            "The solution formula is not defined at \\(%s = %s\\), so these "
            "initial conditions cannot be applied." % (xn, tex(x0))))
        return None
    values = found[0]
    particular = sp.expand(general.subs(values))
    c1 = values.get(prob.C1, prob.C1)
    c2 = values.get(prob.C2, prob.C2)
    steps.append(step(
        "Apply the initial conditions",
        "Substitute \\(%s = %s\\) into \\(y\\) and \\(y'\\), set them equal to "
        "\\(%s\\) and \\(%s\\), and solve the two equations for \\(C_1\\) and "
        "\\(C_2\\)." % (xn, tex(x0), tex(y0), tex(y1)),
        [_eq("y", particular)],
        [step("Equations for the constants", "",
              [_eq(at0, y0), _eq(slope0, y1),
               "C_1 = %s,\\qquad C_2 = %s" % (tex(c1), tex(c2))])]))
    return particular


# --------------------------------------------------------------------------
# Homogeneous part: characteristic equation
# --------------------------------------------------------------------------

def _characteristic(eq):
    """Returns (steps, y1, y2, roots dict) for a y'' + b y' + c y = 0."""
    x = eq.x
    r = sp.Symbol("r")
    a, b, c = eq.a, eq.b, eq.c
    char = a * r**2 + b * r + c
    disc = sp.simplify(b**2 - 4 * a * c)
    roots = sp.roots(sp.Poly(char, r))
    steps = [step(
        "Write the characteristic equation",
        "For constant coefficients, try \\(y = e^{r%s}\\). Substituting and "
        "dividing by \\(e^{r%s}\\) leaves a quadratic in \\(r\\)." % (eq.xn, eq.xn),
        ["%s = 0" % eq.lhs_tex(), "%s = 0" % tex(char)])]

    if disc > 0:
        r1, r2 = sorted(roots, key=lambda v: float(v))
        y1, y2 = sp.exp(r1 * x), sp.exp(r2 * x)
        kind = ("The discriminant is positive, so there are two different real "
                "roots. Each gives an exponential solution.")
        root_tex = "r_1 = %s,\\qquad r_2 = %s" % (tex(r1), tex(r2))
    elif disc == 0:
        r1 = list(roots)[0]
        y1, y2 = sp.exp(r1 * x), x * sp.exp(r1 * x)
        kind = ("The discriminant is zero, so there is one repeated root. The "
                "second solution is the first multiplied by \\(%s\\)." % eq.xn)
        root_tex = "r = %s\\ \\text{(repeated)}" % tex(r1)
    else:
        alpha = sp.simplify(-b / (2 * a))
        beta = sp.simplify(sp.sqrt(-disc) / (2 * sp.Abs(a)))
        y1 = sp.exp(alpha * x) * sp.cos(beta * x)
        y2 = sp.exp(alpha * x) * sp.sin(beta * x)
        kind = ("The discriminant is negative, so the roots are complex, "
                "\\(r = \\alpha \\pm \\beta i\\). They give "
                "\\(e^{\\alpha %s}\\cos \\beta %s\\) and "
                "\\(e^{\\alpha %s}\\sin \\beta %s\\)." % ((eq.xn,) * 4))
        root_tex = "r = %s \\pm %s\\,i" % (tex(alpha), tex(beta)) if alpha != 0 \
            else "r = \\pm %s\\,i" % tex(beta)
    steps.append(step(
        "Solve for the roots", kind,
        [root_tex],
        [step("Quadratic formula", "",
              ["b^2 - 4ac = %s" % tex(disc),
               "r = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"])]))
    return steps, y1, y2, roots


def _homogeneous_solution(eq, y1, y2):
    return eq.prob.C1 * y1 + eq.prob.C2 * y2


def characteristic(eq, ic):
    name = "Characteristic equation"
    if not eq.constant:
        return _not_applicable("characteristic", name,
                               "The coefficients depend on \\(%s\\); this method "
                               "needs constant coefficients." % eq.xn)
    if eq.g != 0:
        return _not_applicable(
            "characteristic", name,
            "On its own this only solves the homogeneous equation (right-hand "
            "side 0). It is the first stage of the other methods here.")
    steps, y1, y2, _ = _characteristic(eq)
    general = _homogeneous_solution(eq, y1, y2)
    steps.append(step(
        "Write the general solution",
        "Combine the two independent solutions with arbitrary constants.",
        [_eq("y", general)]))
    particular = _apply_ic(eq, general, ic, steps) if ic else None
    return _package("characteristic", name,
                    "The equation is linear and homogeneous with constant "
                    "coefficients.", steps, eq, general, particular, ic)


# --------------------------------------------------------------------------
# Undetermined coefficients
# --------------------------------------------------------------------------

def _term_root(term, x):
    """For x^k e^(alpha x) cos/sin(beta x) return (alpha, beta), or None."""
    alpha, beta = sp.Integer(0), sp.Integer(0)
    for factor in sp.Mul.make_args(term):
        if not factor.has(x):
            continue
        if isinstance(factor, sp.exp):
            alpha += sp.simplify(factor.args[0] / x)
        elif isinstance(factor, (sp.sin, sp.cos)):
            beta = sp.Abs(sp.simplify(factor.args[0] / x))
        elif factor == x or (factor.is_Pow and factor.base == x and factor.exp.is_Integer):
            continue
        else:
            return None
    if alpha.has(x) or beta.has(x):
        return None
    return alpha, beta


def undetermined(eq, ic):
    name = "Undetermined coefficients"
    x = eq.x
    if not eq.constant:
        return _not_applicable("undetermined", name,
                               "This method needs constant coefficients.")
    if eq.g == 0:
        return _not_applicable("undetermined", name,
                               "The right-hand side is 0, so there is no "
                               "particular solution to guess.")
    g = sp.expand(eq.g)
    try:
        match = _undetermined_coefficients_match(g, x)
    except Exception:
        match = {"test": False}
    if not match.get("test"):
        return _not_applicable(
            "undetermined", name,
            "The right-hand side \\(%s\\) is not built from polynomials, "
            "exponentials, sines and cosines, so its derivatives do not stay "
            "in a finite family to guess from." % tex(eq.g))

    steps, y1, y2, roots = _characteristic(eq)
    yh = _homogeneous_solution(eq, y1, y2)
    steps.append(step(
        "Homogeneous solution",
        "This solves the equation with the right-hand side replaced by 0.",
        [_eq("y_h", yh)]))

    # Multiply each family by x^s where s is how many times its exponent
    # is a root of the characteristic equation (resonance).
    trial, bumped = [], False
    for term in match["trialset"]:
        ab = _term_root(term, x)
        if ab is None:
            return _not_applicable("undetermined", name,
                                   "The right-hand side is outside the forms "
                                   "this method handles.")
        alpha, beta = ab
        mult = 0
        for root, count in roots.items():
            if is_zero(sp.re(root) - alpha) and is_zero(sp.Abs(sp.im(root)) - beta):
                mult = count
        if mult:
            bumped = True
        trial.append(sp.expand(term * x**mult))
    trial = sorted(set(trial), key=lambda t: sp.default_sort_key(t))
    if len(trial) > len(_COEFF_NAMES):
        return _not_applicable("undetermined", name,
                               "The guess would need too many coefficients.")
    coeffs = [sp.Symbol(n) for n in _COEFF_NAMES[:len(trial)]]
    guess = sum(c * t for c, t in zip(coeffs, trial))
    text = ("Guess a combination of the right-hand side's terms and all of "
            "their derivatives, with unknown coefficients.")
    if bumped:
        text += (" Part of the natural guess already solves the homogeneous "
                 "equation, so that part is multiplied by \\(%s\\) (once for "
                 "each time its exponent is a root)." % eq.xn)
    steps.append(step("Guess the particular solution", text, [_eq("y_p", guess)]))

    residual = sp.expand(eq.a * sp.diff(guess, x, 2) + eq.b * sp.diff(guess, x)
                         + eq.c * guess - g)
    buckets = {}
    for term in sp.Add.make_args(residual):
        const, func = term.as_independent(x)
        buckets[func] = buckets.get(func, 0) + const
    equations = [sp.Eq(sp.expand(v), 0) for v in buckets.values() if sp.expand(v) != 0]
    try:
        solved = sp.solve(equations, coeffs, dict=True)
    except Exception:
        solved = []
    if not solved:
        return _not_applicable("undetermined", name,
                               "The coefficients could not be matched for this "
                               "right-hand side.")
    values = {c: solved[0].get(c, sp.Integer(0)) for c in coeffs}
    yp = sp.expand(guess.subs(values))
    if not is_zero(eq.residual(yp)):
        return _not_applicable("undetermined", name,
                               "The guess did not reproduce the right-hand side.")
    steps.append(step(
        "Substitute and match coefficients",
        "Put \\(y_p\\) into the equation and compare the coefficient of each "
        "kind of term on the two sides.",
        [",\\quad ".join(_eq(c, values[c]) for c in coeffs)],
        [step("Equations", "", [_eq(e.lhs, 0) for e in equations])]))
    steps.append(step("Particular solution", "", [_eq("y_p", yp)]))

    general = yh + yp
    steps.append(step(
        "Write the general solution",
        "Add the homogeneous and particular solutions.",
        ["y = y_h + y_p", _eq("y", general)]))
    particular = _apply_ic(eq, general, ic, steps) if ic else None
    return _package("undetermined", name,
                    "The coefficients are constant and the right-hand side "
                    "\\(%s\\) is made of polynomials, exponentials, sines or "
                    "cosines." % tex(eq.g), steps, eq, general, particular, ic)


# --------------------------------------------------------------------------
# Variation of parameters
# --------------------------------------------------------------------------

def variation(eq, ic):
    name = "Variation of parameters"
    x = eq.x
    if not eq.constant:
        return _not_applicable("variation", name,
                               "It needs two known solutions of the homogeneous "
                               "equation; those are found here only for "
                               "constant coefficients.")
    if eq.g.has(sp.Heaviside, sp.DiracDelta):
        return _not_applicable("variation", name,
                               "Step and impulse forcing is handled here by "
                               "Laplace transforms.")
    if eq.g == 0:
        return _not_applicable("variation", name,
                               "The right-hand side is 0, so no particular "
                               "solution is needed.")
    steps, y1, y2, _ = _characteristic(eq)
    yh = _homogeneous_solution(eq, y1, y2)
    steps.append(step(
        "Homogeneous solution",
        "This gives the two solutions \\(y_1\\) and \\(y_2\\) the method builds on.",
        [_eq("y_h", yh), "y_1 = %s,\\qquad y_2 = %s" % (tex(y1), tex(y2))]))

    wronskian = sp.simplify(y1 * sp.diff(y2, x) - y2 * sp.diff(y1, x))
    f = sp.simplify(eq.g / eq.a)
    steps.append(step(
        "Compute the Wronskian",
        "The Wronskian of \\(y_1\\) and \\(y_2\\) is \\(y_1 y_2' - y_2 y_1'\\).",
        ["W = %s" % tex(wronskian)]))
    du1 = sp.simplify(-y2 * f / wronskian)
    du2 = sp.simplify(y1 * f / wronskian)
    note = ""
    if eq.a != 1:
        note = (" First divide the equation by the coefficient of \\(y''\\), so "
                "the right-hand side is \\(f = %s\\)." % tex(f))
    steps.append(step(
        "Set up the parameters",
        "Look for \\(y_p = u_1 y_1 + u_2 y_2\\) with "
        "\\(u_1' = -\\dfrac{y_2 f}{W}\\) and \\(u_2' = \\dfrac{y_1 f}{W}\\)." + note,
        ["u_1' = %s" % tex(du1), "u_2' = %s" % tex(du2)]))
    u1 = with_abs(integrate(du1, x))
    u2 = with_abs(integrate(du2, x))
    if has_integral(u1) or has_integral(u2):
        steps.append(step(
            "Integrate",
            "At least one of these integrals has no closed form in elementary "
            "functions, so the particular solution is left in integral form.",
            ["u_1 = \\int %s\\,d%s" % (_paren(du1), eq.xn),
             "u_2 = \\int %s\\,d%s" % (_paren(du2), eq.xn)]))
        out = _package("variation", name, "", steps, eq, None, None, ic,
                       general_tex="y = %s + u_1 y_1 + u_2 y_2" % tex(yh))
        out["reason"] = _variation_reason()
        out["verify_note"] = "Left in integral form; nothing to substitute back."
        return out
    steps.append(step(
        "Integrate",
        "Integrate each one. No constants are needed here; they are already "
        "in the homogeneous solution.",
        ["u_1 = %s" % tex(u1), "u_2 = %s" % tex(u2)]))
    yp = sp.simplify(u1 * y1 + u2 * y2)
    # Drop any part of y_p that merely repeats the homogeneous solution.
    yp = _strip_homogeneous(sp.expand(yp), y1, y2, x)
    steps.append(step("Particular solution",
                      "Form \\(y_p = u_1 y_1 + u_2 y_2\\) and simplify.",
                      [_eq("y_p", yp)]))
    general = yh + yp
    steps.append(step(
        "Write the general solution",
        "Add the homogeneous and particular solutions.",
        ["y = y_h + y_p", _eq("y", general)]))
    particular = _apply_ic(eq, general, ic, steps) if ic else None
    return _package("variation", name, _variation_reason(), steps, eq, general,
                    particular, ic)


def _variation_reason():
    return ("It works for any right-hand side once the homogeneous equation is "
            "solved, at the cost of two integrals.")


def _strip_homogeneous(yp, y1, y2, x):
    """Remove terms of y_p that are just multiples of y1 or y2."""
    kept = []
    for term in sp.Add.make_args(yp):
        func = term.as_independent(x)[1]
        if any(not sp.simplify(func / basis).has(x) for basis in (y1, y2)):
            continue
        kept.append(term)
    return sp.Add(*kept) if kept else yp


# --------------------------------------------------------------------------
# Cauchy-Euler (homogeneous)
# --------------------------------------------------------------------------

def cauchy_euler(eq, ic):
    name = "Cauchy–Euler (y = x^r)"
    x = eq.x
    xn = eq.xn
    A = sp.simplify(eq.a / x**2)
    B = sp.simplify(eq.b / x)
    C = eq.c
    if eq.constant or A.has(x) or B.has(x) or C.has(x) or A == 0:
        return _not_applicable(
            "cauchy_euler", name,
            "The equation does not have the form "
            "\\(a%s^2 y'' + b%s y' + c y = 0\\)." % (xn, xn))
    if eq.g != 0:
        return _not_applicable("cauchy_euler", name,
                               "Only the homogeneous case (right-hand side 0) "
                               "is supported so far.")
    r = sp.Symbol("r")
    indicial = sp.expand(A * r * (r - 1) + B * r + C)
    disc = sp.simplify((B - A)**2 - 4 * A * C)
    roots = sp.roots(sp.Poly(indicial, r))
    steps = [step(
        "Substitute a power of %s" % xn,
        "Each term has a power of \\(%s\\) matching the order of its "
        "derivative, so try \\(y = %s^r\\). Dividing by \\(%s^r\\) leaves a "
        "quadratic in \\(r\\)." % (xn, xn, xn),
        ["%s = 0" % eq.lhs_tex(), "%s = 0" % tex(indicial)])]
    if disc > 0:
        r1, r2 = sorted(roots, key=lambda v: float(v))
        y1, y2 = x**r1, x**r2
        kind = "Two different real roots give two power solutions."
        root_tex = "r_1 = %s,\\qquad r_2 = %s" % (tex(r1), tex(r2))
    elif disc == 0:
        r1 = list(roots)[0]
        y1, y2 = x**r1, x**r1 * sp.log(x)
        kind = ("A repeated root gives one power solution; the second is that "
                "power times \\(\\ln %s\\)." % xn)
        root_tex = "r = %s\\ \\text{(repeated)}" % tex(r1)
    else:
        alpha = sp.simplify(-(B - A) / (2 * A))
        beta = sp.simplify(sp.sqrt(-disc) / (2 * sp.Abs(A)))
        y1 = x**alpha * sp.cos(beta * sp.log(x))
        y2 = x**alpha * sp.sin(beta * sp.log(x))
        kind = ("Complex roots \\(\\alpha \\pm \\beta i\\) give "
                "\\(%s^{\\alpha}\\cos(\\beta \\ln %s)\\) and "
                "\\(%s^{\\alpha}\\sin(\\beta \\ln %s)\\)." % ((xn,) * 4))
        root_tex = "r = %s \\pm %s\\,i" % (tex(alpha), tex(beta))
    steps.append(step("Solve for the roots", kind, [root_tex]))
    general = eq.prob.C1 * y1 + eq.prob.C2 * y2
    steps.append(step(
        "Write the general solution",
        "Valid for \\(%s > 0\\)." % xn, [_eq("y", general)]))
    particular = _apply_ic(eq, general, ic, steps) if ic else None
    return _package("cauchy_euler", name,
                    "It has the form \\(a%s^2 y'' + b%s y' + c y = 0\\)." % (xn, xn),
                    steps, eq, general, particular, ic)




def laplace(eq, ic):
    from . import laplace as lt
    blocked = lt.why_not(eq.prob, eq.constant, ic)
    if blocked:
        return _not_applicable("laplace", lt.NAME, blocked)
    steps, sol = lt.solve_ivp(eq.prob, eq.a, eq.b, eq.c, eq.g, ic[1], ic[2])
    if steps is None:
        return _not_applicable("laplace", lt.NAME, sol)
    if lt.forced(eq.g):
        out = _package("laplace", lt.NAME,
                       "Laplace transforms turn the step and impulse terms into "
                       "simple factors \\(e^{-as}\\), which the other methods "
                       "cannot handle directly.", steps, eq, None, None, ic)
        ok = lt.check_forced(eq.residual(sol), eq.x)
        out["particular_latex"] = _eq("y", sol)
        out["plot"]["particular_expr"] = plot_expr(sol, eq.prob)
        out["verified"] = ok
        out["verify_note"] = ("Checked numerically for \\(%s > 0\\), between the "
                              "instants where the forcing switches." % eq.xn) if ok else (
            "Could not be checked automatically." if ok is None
            else "Substituting back did not reproduce the equation.")
        return out
    return _package("laplace", lt.NAME,
                    "The equation is linear with constant coefficients and the "
                    "initial conditions are given at 0.",
                    steps, eq, None, sol, ic)


METHODS = (characteristic, undetermined, variation, cauchy_euler, laplace)
