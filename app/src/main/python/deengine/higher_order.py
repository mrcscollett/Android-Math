"""Third- and fourth-order linear equations with constant coefficients."""
import sympy as sp
from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match

from .first_order import _eq, _not_applicable, _paren, step
from .parsing import InputError
from .second_order import _COEFF_NAMES, _term_root
from .util import is_zero, plot_expr, residual_status, tex

_PRIMES = ["y", "y'", "y''", "y'''", "y''''"]


class LinearN:
    """c[n] y^(n) + ... + c[1] y' + c[0] y = g, with constant c."""

    def __init__(self, prob, coeffs, g):
        self.prob, self.c, self.g = prob, coeffs, g
        self.n = len(coeffs) - 1
        self.x, self.xn = prob.x, prob.indep_name
        self.constants = [prob.C1, prob.C2, prob.C3, prob.C4][:self.n]

    def apply(self, sol):
        return sum(c * sp.diff(sol, self.x, k) for k, c in enumerate(self.c))

    def lhs_tex(self):
        out = ""
        for k in range(self.n, -1, -1):
            coef = self.c[k]
            if coef == 0:
                continue
            negative = coef.could_extract_minus_sign()
            mag = -coef if negative else coef
            body = _PRIMES[k] if mag == 1 else "%s\\,%s" % (_paren(mag), _PRIMES[k])
            out += (("-" if negative else "") + body) if not out else ((" - " if negative else " + ") + body)
        return out


def analyse(prob, lhs, rhs):
    derivs = [prob.y, prob.yp, prob.ypp, prob.y3, prob.y4]
    expr = sp.expand(lhs - rhs)
    try:
        poly = sp.Poly(expr, *derivs)
    except sp.PolynomialError:
        poly = None
    if poly is None or poly.total_degree() != 1:
        raise InputError("Higher-order support covers linear equations with "
                         "constant coefficients. In this one, y or its "
                         "derivatives appear nonlinearly.")
    coeffs = [sp.simplify(poly.coeff_monomial(d)) for d in derivs][:prob.order + 1]
    if any(c.has(prob.x) for c in coeffs):
        raise InputError("Equations of third and fourth order are supported "
                         "only with constant coefficients.")
    return LinearN(prob, coeffs, sp.simplify(-poly.coeff_monomial(1)))


def _homogeneous(eq):
    """Returns (steps, basis, roots) or (None, reason, None)."""
    x, xn = eq.x, eq.xn
    r = sp.Symbol("r")
    char = sum(c * r**k for k, c in enumerate(eq.c))
    roots = sp.roots(sp.Poly(char, r))
    if sum(roots.values()) != eq.n or any(sp.count_ops(v) > 20 for v in roots):
        return None, ("The characteristic equation \\(%s = 0\\) has roots that "
                      "have no simple closed form." % tex(char)), None
    steps = [step(
        "Write the characteristic equation",
        "Try \\(y = e^{r%s}\\). Each derivative brings down a factor \\(r\\), "
        "leaving a polynomial of degree %d." % (xn, eq.n),
        ["%s = 0" % eq.lhs_tex(), "%s = 0" % tex(char), "%s = 0" % tex(sp.factor(char))])]
    basis, lines = [], []
    done = set()
    for root in sorted(roots, key=lambda v: (float(sp.re(v)), float(sp.im(v)))):
        mult = roots[root]
        alpha, beta = sp.re(root), sp.im(root)
        if beta == 0:
            lines.append("r = %s%s" % (tex(root), "\\ \\text{(multiplicity %d)}" % mult if mult > 1 else ""))
            basis += [x**j * sp.exp(root * x) for j in range(mult)]
        else:
            key = (alpha, sp.Abs(beta))
            if key in done:
                continue
            done.add(key)
            b = sp.Abs(beta)
            lines.append("r = %s \\pm %s\\,i%s" % (
                tex(alpha), tex(b), "\\ \\text{(multiplicity %d)}" % mult if mult > 1 else "")
                if alpha != 0 else "r = \\pm %s\\,i%s" % (
                tex(b), "\\ \\text{(multiplicity %d)}" % mult if mult > 1 else ""))
            for j in range(mult):
                basis.append(x**j * sp.exp(alpha * x) * sp.cos(b * x))
                basis.append(x**j * sp.exp(alpha * x) * sp.sin(b * x))
    steps.append(step(
        "Solve for the roots",
        "A real root \\(r\\) gives \\(e^{r%s}\\); a complex pair \\(\\alpha \\pm \\beta i\\) "
        "gives \\(e^{\\alpha %s}\\cos\\beta %s\\) and \\(e^{\\alpha %s}\\sin\\beta %s\\). "
        "A root repeated \\(m\\) times also gives those solutions multiplied by "
        "\\(%s, \\dots, %s^{m-1}\\)." % (xn, xn, xn, xn, xn, xn, xn),
        [",\\qquad ".join(lines)]))
    return steps, basis, roots


def _yh(eq, basis):
    return sum(c * b for c, b in zip(eq.constants, basis))


def _check(eq, sol, ic):
    status = residual_status(eq.apply(sol) - eq.g, {eq.x: ic[0]} if ic else None)
    if status in ("bad", "partial"):
        return False, "Substituting back did not reproduce the equation."
    if status == "unknown":
        return None, "Could not be checked automatically."
    how = "symbolically" if status == "symbolic" else "numerically at sample points"
    if ic:
        for k, value in enumerate(ic[1]):
            if abs(complex(sp.N(sp.diff(sol, eq.x, k).subs(eq.x, ic[0]) - value))) > 1e-9:
                return False, "The initial conditions are not satisfied."
        return True, "Checked %s by substituting back, and it meets all the initial conditions." % how
    return True, "Checked %s by substituting back into the equation." % how


def _apply_ic(eq, general, ic, steps):
    x0, values = ic
    eqs = [sp.diff(general, eq.x, k).subs(eq.x, x0) - v for k, v in enumerate(values)]
    found = sp.solve(eqs, eq.constants, dict=True)
    if not found:
        steps.append(step("Apply the initial conditions",
                          "These initial conditions could not be satisfied."))
        return None
    particular = sp.expand(general.subs(found[0]))
    steps.append(step(
        "Apply the initial conditions",
        "Substitute \\(%s = %s\\) into \\(y\\) and its first %d derivative%s, and solve "
        "the resulting equations for the constants."
        % (eq.xn, tex(x0), eq.n - 1, "" if eq.n == 2 else "s"),
        [_eq("y", particular)],
        [step("Equations for the constants", "",
              ["%s = %s" % (tex(e + v), tex(v)) for e, v in zip(eqs, values)]
              + [",\\qquad ".join("%s = %s" % (tex(c), tex(found[0].get(c, c))) for c in eq.constants)])]))
    return particular


def _package(method_id, name, reason, steps, eq, general, particular, ic):
    target = particular if particular is not None else general
    verified, note = _check(eq, target, ic if particular is not None else None)
    return {"id": method_id, "name": name, "applicable": True, "reason": reason, "steps": steps,
            "general_latex": _eq("y", general),
            "particular_latex": _eq("y", particular) if particular is not None else None,
            "plot": {"general_expr": plot_expr(general, eq.prob),
                     "particular_expr": plot_expr(particular, eq.prob)},
            "verified": verified, "verify_note": note}


def characteristic(eq, ic):
    name = "Characteristic equation"
    if eq.g != 0:
        return _not_applicable("characteristic", name,
                               "On its own this only solves the homogeneous equation "
                               "(right-hand side 0). It is the first stage of "
                               "undetermined coefficients.")
    steps, basis, _ = _homogeneous(eq)
    if steps is None:
        return _not_applicable("characteristic", name, basis)
    general = _yh(eq, basis)
    steps.append(step("Write the general solution",
                      "Combine the %d independent solutions with arbitrary constants." % eq.n,
                      [_eq("y", general)]))
    particular = _apply_ic(eq, general, ic, steps) if ic else None
    return _package("characteristic", name,
                    "The equation is linear and homogeneous with constant coefficients.",
                    steps, eq, general, particular, ic)


def undetermined(eq, ic):
    name = "Undetermined coefficients"
    x = eq.x
    if eq.g == 0:
        return _not_applicable("undetermined", name,
                               "The right-hand side is 0, so there is no particular solution to guess.")
    g = sp.expand(eq.g)
    try:
        match = _undetermined_coefficients_match(g, x)
    except Exception:
        match = {"test": False}
    if not match.get("test"):
        return _not_applicable(
            "undetermined", name,
            "The right-hand side \\(%s\\) is not built from polynomials, exponentials, "
            "sines and cosines." % tex(eq.g))
    steps, basis, roots = _homogeneous(eq)
    if steps is None:
        return _not_applicable("undetermined", name, basis)
    yh = _yh(eq, basis)
    steps.append(step("Homogeneous solution",
                      "This solves the equation with the right-hand side replaced by 0.",
                      [_eq("y_h", yh)]))
    trial, bumped = [], False
    for term in match["trialset"]:
        ab = _term_root(term, x)
        if ab is None:
            return _not_applicable("undetermined", name,
                                   "The right-hand side is outside the forms this method handles.")
        mult = 0
        for root, count in roots.items():
            if is_zero(sp.re(root) - ab[0]) and is_zero(sp.Abs(sp.im(root)) - ab[1]):
                mult = count
        bumped = bumped or mult > 0
        trial.append(sp.expand(term * x**mult))
    trial = sorted(set(trial), key=sp.default_sort_key)
    if len(trial) > len(_COEFF_NAMES):
        return _not_applicable("undetermined", name, "The guess would need too many coefficients.")
    coeffs = [sp.Symbol(n) for n in _COEFF_NAMES[:len(trial)]]
    guess = sum(c * t for c, t in zip(coeffs, trial))
    text = ("Guess a combination of the right-hand side's terms and all of their "
            "derivatives, with unknown coefficients.")
    if bumped:
        text += (" Part of the natural guess already solves the homogeneous equation, so "
                 "that part is multiplied by \\(%s\\) once for each time its exponent is a root."
                 % eq.xn)
    steps.append(step("Guess the particular solution", text, [_eq("y_p", guess)]))
    residual = sp.expand(eq.apply(guess) - g)
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
        return _not_applicable("undetermined", name, "The coefficients could not be matched.")
    values = {c: solved[0].get(c, sp.Integer(0)) for c in coeffs}
    yp = sp.expand(guess.subs(values))
    if not is_zero(eq.apply(yp) - eq.g):
        return _not_applicable("undetermined", name, "The guess did not reproduce the right-hand side.")
    steps.append(step("Substitute and match coefficients",
                      "Put \\(y_p\\) into the equation and compare the coefficient of each kind of term.",
                      [",\\quad ".join(_eq(c, values[c]) for c in coeffs)],
                      [step("Equations", "", [_eq(e.lhs, 0) for e in equations])]))
    steps.append(step("Particular solution", "", [_eq("y_p", yp)]))
    general = yh + yp
    steps.append(step("Write the general solution", "Add the homogeneous and particular solutions.",
                      ["y = y_h + y_p", _eq("y", general)]))
    particular = _apply_ic(eq, general, ic, steps) if ic else None
    return _package("undetermined", name,
                    "The coefficients are constant and the right-hand side \\(%s\\) is made of "
                    "polynomials, exponentials, sines or cosines." % tex(eq.g),
                    steps, eq, general, particular, ic)


METHODS = (characteristic, undetermined)
