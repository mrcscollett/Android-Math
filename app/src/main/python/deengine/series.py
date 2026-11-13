"""Series solutions of a(x) y'' + b(x) y' + c(x) y = 0 about x = 0.

At an ordinary point the solution is a power series with two free
coefficients. At a regular singular point the method of Frobenius is used:
y = x^r (1 + a1 x + a2 x^2 + ...), with r from the indicial equation.
"""
import sympy as sp

from .first_order import _eq, _not_applicable, step
from .util import plot_expr, tex

TERMS = 6   # coefficients computed beyond the leading one(s)
NAME = "Series solution"


def _coefficients(poly):
    """{power: coefficient} for a polynomial in x."""
    return {m[0]: c for m, c in poly.terms()}


def _bracket(P, Q, R, m, n, r):
    """Coefficient of a_n in the equation for the power x^(m + r - 2)."""
    return (P.get(m - n, 0) * (n + r) * (n + r - 1)
            + Q.get(m - n - 1, 0) * (n + r) + R.get(m - n - 2, 0))


def _solve_coefficients(P, Q, R, r, start, known, count):
    """Extend `known` ({index: value}) by matching powers of x.

    `start` is the power offset of the first equation that determines a new
    coefficient. Returns (values, equations) or (None, None) if an equation
    cannot be satisfied (the case that needs a logarithm).
    """
    values = dict(known)
    first = max(values) + 1
    equations = []
    for k in range(first, first + count):
        m = start + k
        rest = sum(values[n] * _bracket(P, Q, R, m, n, r) for n in values)
        lead = sp.simplify(_bracket(P, Q, R, m, k, r))
        rest = sp.simplify(rest)
        if lead == 0:
            if rest != 0:
                return None, None
            values[k] = sp.Integer(0)
        else:
            values[k] = sp.simplify(-rest / lead)
        equations.append((k, values[k]))
    return values, equations


def _series(values, r, x):
    return sum(v * x**(n + r) for n, v in sorted(values.items()))


def _asc(expr, x, dots=True):
    """LaTeX of a truncated series with powers ascending, followed by dots."""
    terms = sp.Add.make_args(sp.powsimp(sp.expand(expr)))
    ordered = sorted(terms, key=lambda term: float(term.as_coeff_exponent(x)[1]))
    out = tex(ordered[0])
    for term in ordered[1:]:
        out += (" - " + tex(-term)) if term.could_extract_minus_sign() else (" + " + tex(term))
    # A single term means every later coefficient was zero: the series stops.
    return out + " + \\cdots" if dots and len(ordered) > 1 else out


def _residual_ok(eq, sol, r, order):
    """Do the low-order terms of the equation vanish for this truncated series?"""
    x = eq.x
    try:
        res = sp.expand(sp.powsimp(sp.expand(eq.residual(sol) * x**(2 - r))))
        low = sp.series(res, x, 0, order).removeO()
        return sp.simplify(low) == 0
    except Exception:
        return None


def solve(eq, ic):
    x, xn = eq.x, eq.xn
    prob = eq.prob
    if eq.constant:
        return _not_applicable("series", NAME,
                               "With constant coefficients the other methods "
                               "give the exact solution, so a series is not needed.")
    if eq.g != 0:
        return _not_applicable("series", NAME,
                               "Only the homogeneous case (right-hand side 0) "
                               "is supported so far.")
    try:
        polys = [sp.Poly(sp.expand(e), x) for e in (eq.a, eq.b, eq.c)]
    except sp.PolynomialError:
        polys = None
    if polys is None or any(not p.domain.is_QQ and not p.domain.is_ZZ for p in polys):
        return _not_applicable("series", NAME,
                               "The coefficients are not polynomials in \\(%s\\)." % xn)
    P, Q, R = (_coefficients(p) for p in polys)
    if ic and ic[0] != 0:
        return _not_applicable("series", NAME,
                               "The series is centred at \\(%s = 0\\), so it needs "
                               "initial conditions given there." % xn)
    if P.get(0, 0) != 0:
        return _ordinary(eq, ic, P, Q, R, polys[0])
    return _frobenius(eq, ic, P, Q, R)


def _ordinary(eq, ic, P, Q, R, lead_poly):
    x, xn, prob = eq.x, eq.xn, eq.prob
    a0, a1 = sp.Symbol("a_0"), sp.Symbol("a_1")
    values, _ = _solve_coefficients(P, Q, R, 0, 0, {0: a0, 1: a1}, TERMS)
    if values is None:
        return _not_applicable("series", NAME, "The coefficients could not be determined.")

    # General recurrence in n, read off from the coefficient of x^n.
    n = sp.Symbol("n", integer=True)
    a = sp.IndexedBase("a")
    total = sp.Integer(0)
    for j, p in P.items():
        total += p * (n - j + 2) * (n - j + 1) * a[n - j + 2]
    for j, q in Q.items():
        total += q * (n - j + 1) * a[n - j + 1]
    for j, c in R.items():
        total += c * a[n - j]
    lead = P[0] * (n + 2) * (n + 1)
    recurrence = sp.factor(sp.simplify(-(total - lead * a[n + 2]) / lead))

    roots = [abs(complex(z)) for z in sp.Poly(lead_poly.as_expr(), x).nroots()] \
        if lead_poly.degree() > 0 else []
    radius = min(roots) if roots else None

    steps = [
        step("Assume a power series",
             "\\(%s = 0\\) is an ordinary point (the coefficient of \\(y''\\) is not "
             "zero there), so look for a solution as a power series and "
             "differentiate it term by term." % xn,
             ["y = \\sum_{n=0}^{\\infty} a_n %s^n" % xn,
              "y' = \\sum_{n=1}^{\\infty} n\\,a_n %s^{n-1},\\qquad "
              "y'' = \\sum_{n=2}^{\\infty} n(n-1)\\,a_n %s^{n-2}" % (xn, xn)]),
        step("Find the recurrence relation",
             "Substitute into the equation, shift the indices so every sum is in "
             "powers of \\(%s^n\\), and set the coefficient of each power to zero. "
             "Coefficients with a negative index count as zero." % xn,
             ["a_{n+2} = %s" % tex(recurrence)]),
        step("Compute the coefficients",
             "\\(a_0\\) and \\(a_1\\) stay free; every other coefficient follows from them.",
             [",\\qquad ".join("a_{%d} = %s" % (k, tex(values[k])) for k in range(2, 2 + TERMS))]),
    ]
    y1 = sum(sp.diff(values[k], a0) * x**k for k in values)
    y2 = sum(sp.diff(values[k], a1) * x**k for k in values)
    note = ("The series converges for all \\(%s\\)." % xn if radius is None else
            "The series converges at least for \\(|%s| < %s\\), the distance to the "
            "nearest point where the coefficient of \\(y''\\) is zero."
            % (xn, tex(sp.nsimplify(round(radius, 6)))))
    general_tex = "y = a_0\\left(%s\\right) + a_1\\left(%s\\right)" % (_asc(y1, x), _asc(y2, x))
    steps.append(step("Write the solution",
                      "Group the terms multiplying \\(a_0\\) and \\(a_1\\): they are two "
                      "independent solutions. Here \\(a_0 = y(0)\\) and \\(a_1 = y'(0)\\). " + note,
                      [general_tex]))

    general = prob.C1 * y1 + prob.C2 * y2
    particular, particular_tex = None, None
    if ic:
        particular = sp.expand(ic[1] * y1 + ic[2] * y2)
        particular_tex = "y = %s" % _asc(particular, x)
        steps.append(step("Apply the initial conditions",
                          "Set \\(a_0 = y(0) = %s\\) and \\(a_1 = y'(0) = %s\\)."
                          % (tex(ic[1]), tex(ic[2])), [particular_tex]))
    ok = _residual_ok(eq, sp.expand(y1 + y2), 0, TERMS)
    return _package(eq, steps,
                    "The coefficients are polynomials and \\(%s = 0\\) is an ordinary point." % xn,
                    general, particular, general_tex, particular_tex, ok)


def _frobenius(eq, ic, P, Q, R):
    x, xn, prob = eq.x, eq.xn, eq.prob
    if ic:
        return _not_applicable("series", NAME,
                               "\\(%s = 0\\) is a singular point of this equation, so "
                               "initial conditions cannot be imposed there." % xn)
    p0 = sp.limit(x * eq.b / eq.a, x, 0)
    q0 = sp.limit(x**2 * eq.c / eq.a, x, 0)
    if not (p0.is_finite and q0.is_finite):
        return _not_applicable("series", NAME,
                               "\\(%s = 0\\) is an irregular singular point; the method "
                               "of Frobenius does not apply." % xn)
    r = sp.Symbol("r")
    indicial = sp.expand(r * (r - 1) + p0 * r + q0)
    roots = sp.roots(sp.Poly(indicial, r))
    if any(not root.is_real for root in roots):
        return _not_applicable("series", NAME,
                               "The indicial equation has complex roots, which this "
                               "version does not handle.")
    ordered = sorted(roots, key=lambda v: float(v), reverse=True)
    r1 = ordered[0]
    r2 = ordered[-1]
    # The first power whose equation is not trivially satisfied gives the indicial equation.
    start = min(k for k in P if P[k] != 0)

    steps = [
        step("Classify the point",
             "The coefficient of \\(y''\\) is zero at \\(%s = 0\\), so it is a singular "
             "point. It is a regular one because these limits are finite:" % xn,
             ["p_0 = \\lim_{%s \\to 0} %s\\,\\frac{b}{a} = %s,\\qquad "
              "q_0 = \\lim_{%s \\to 0} %s^2\\,\\frac{c}{a} = %s"
              % (xn, xn, tex(p0), xn, xn, tex(q0))]),
        step("Assume a Frobenius series",
             "Look for a power series multiplied by an unknown power of \\(%s\\)." % xn,
             ["y = %s^r \\sum_{n=0}^{\\infty} a_n %s^n,\\qquad a_0 = 1" % (xn, xn)]),
        step("Solve the indicial equation",
             "The lowest power of \\(%s\\) gives a quadratic for \\(r\\)." % xn,
             ["r(r-1) + p_0 r + q_0 = 0", "%s = 0" % tex(indicial),
              "r_1 = %s,\\qquad r_2 = %s" % (tex(r1), tex(r2)) if r1 != r2
              else "r = %s\\ \\text{(repeated)}" % tex(r1)]),
    ]

    def branch(root, label):
        values, eqs = _solve_coefficients(P, Q, R, root, start, {0: sp.Integer(1)}, TERMS)
        if values is None:
            return None
        sol = _series(values, root, x)
        steps.append(step(
            "Coefficients for r = %s" % sp.sstr(root),
            "Setting the coefficient of each higher power to zero gives the "
            "\\(a_n\\) one after another.",
            [",\\qquad ".join("a_{%d} = %s" % (k, tex(v)) for k, v in eqs),
             "%s = %s" % (label, _asc(sol, x))]))
        return sol

    y1 = branch(r1, "y_1")
    if y1 is None:
        return _not_applicable("series", NAME, "The coefficients could not be determined.")
    y2 = branch(r2, "y_2") if r2 != r1 else None
    if y2 is not None:
        general = prob.C1 * y1 + prob.C2 * y2
        general_tex = "y = C_1 y_1 + C_2 y_2"
        steps.append(step("Write the general solution",
                          "The two series are independent. Valid for \\(%s > 0\\)." % xn,
                          [general_tex]))
        ok = _residual_ok(eq, y1, r1, start + TERMS) and _residual_ok(eq, y2, r2, start + TERMS)
        general_tex = "y = C_1\\left(%s\\right) + C_2\\left(%s\\right)" % (_asc(y1, x), _asc(y2, x))
    else:
        general = prob.C1 * y1
        general_tex = "y_1 = %s" % _asc(y1, x)
        steps.append(step(
            "About the second solution",
            "The roots are equal or differ by an integer in a way that makes the "
            "second solution contain \\(y_1 \\ln %s\\). That logarithmic solution "
            "is not computed here, so only \\(y_1\\) is given." % xn))
        ok = _residual_ok(eq, y1, r1, start + TERMS)
    return _package(eq, steps,
                    "The coefficients are polynomials and \\(%s = 0\\) is a regular "
                    "singular point (method of Frobenius)." % xn,
                    general, None, general_tex, None, ok)


def _package(eq, steps, reason, general, particular, general_tex, particular_tex, ok):
    note = ("The first terms were substituted back: they satisfy the equation up "
            "to the order shown. A truncated series is an approximation that is "
            "best near 0.") if ok else "The series could not be checked automatically."
    return {
        "id": "series", "name": NAME, "applicable": True, "reason": reason,
        "steps": steps,
        "general_latex": general_tex, "particular_latex": particular_tex,
        "plot": {"general_expr": plot_expr(sp.expand(general), eq.prob),
                 "particular_expr": plot_expr(particular, eq.prob) if particular is not None else None},
        "verified": True if ok else None, "verify_note": note,
    }
