"""Special functions and Sturm-Liouville theory.

Bessel functions, the classical orthogonal polynomials (Legendre, Hermite,
Laguerre, Chebyshev), the self-adjoint form of a second-order equation, and
expansions in Legendre polynomials.
"""
import math

import mpmath
import sympy as sp

from . import pieces
from .parsing import InputError, parse_expression, parse_number
from .toolkit import method, result, step
from .util import TIMEOUT, integrate, plot_expr, run_with_timeout, tex

OPERATIONS = ("bessel", "legendre", "hermite", "laguerre", "chebyshev", "sturm", "expansion")
X = sp.Symbol("x", real=True)
PLOT_RANGE = 30.0     # the Bessel series used for the graph is accurate for |x| <= 30


class _Prob:
    """Just enough of a Problem for util.plot_expr."""
    x = X
    y = sp.Symbol("y", real=True)
    C = sp.Symbol("C", real=True)
    C1 = sp.Symbol("C_1", real=True)
    C2 = sp.Symbol("C_2", real=True)


def _plot(expr):
    return plot_expr(expr, _Prob)


def _order(text, label, low, high):
    try:
        value = int((text or "").strip())
    except ValueError:
        raise InputError("%s must be a whole number from %d to %d." % (label, low, high))
    if not low <= value <= high:
        raise InputError("%s must be a whole number from %d to %d." % (label, low, high))
    return value


# --------------------------------------------------------------------------
# Bessel functions
# --------------------------------------------------------------------------

def _half_integer_forms(nu):
    """Closed forms of J_nu and Y_nu for a positive half-integer nu."""
    root = sp.sqrt(2 / (sp.pi * X))
    table = {sp.Rational(1, 2): root * sp.sin(X), sp.Rational(-1, 2): root * sp.cos(X)}
    order = sp.Rational(1, 2)
    while order < nu:                       # J_(v+1) = (2v/x) J_v - J_(v-1)
        table[order + 1] = sp.simplify(2 * order / X * table[order] - table[order - 1])
        order += 1
    order = sp.Rational(-1, 2)
    while order > -nu:                      # J_(v-1) = (2v/x) J_v - J_(v+1)
        table[order - 1] = sp.simplify(2 * order / X * table[order] - table[order + 1])
        order -= 1
    n = int(nu - sp.Rational(1, 2))
    return table[nu], sp.simplify((-1) ** (n + 1) * table[-nu])


def _series_plot(n):
    """Plot string for J_n (whole n): the power series in Horner form, blank beyond its range."""
    half = PLOT_RANGE / 2
    coeffs, k = [], 0
    while True:
        c = (-1.0) ** k / (math.factorial(k) * math.factorial(k + n) * 2.0 ** (2 * k + n))
        coeffs.append(c)
        size = half ** (2 * k + n) / (math.factorial(k) * math.factorial(k + n))
        if size < 1e-10 and k > half:
            break
        k += 1
    inner = repr(coeffs[-1])
    for c in reversed(coeffs[:-1]):
        inner = "%r + x*x*(%s)" % (c, inner)
    lead = "x**%d*" % n if n else ""
    # Dividing by the step leaves a gap outside the range instead of a wrong curve.
    return "%s(%s)/Heaviside(%r - Abs(x))" % (lead, inner, PLOT_RANGE)


def _bessel(text):
    raw = (text or "").strip()
    if not raw:
        raise InputError("Enter the order, for example 0, 1 or 1/2.")
    nu = parse_number(raw, "The order")
    if nu < 0 or 2 * nu != int(2 * nu) or nu > 6:
        raise InputError("The order must be a whole number or a half-integer "
                         "(such as 1/2 or 3/2) from 0 to 6.")
    whole = nu == int(nu)
    v = float(nu)
    name = "J_{%s}" % tex(nu)
    steps = [step(
        "Bessel's equation",
        "\\(%s\\) and \\(Y_{%s}\\) are two independent solutions of Bessel's equation of "
        "order \\(\\nu = %s\\). \\(J_\\nu\\) stays finite at \\(x = 0\\); \\(Y_\\nu\\) is "
        "unbounded there." % (name, tex(nu), tex(nu)),
        ["x^2 y'' + x y' + \\left(x^2 - %s\\right) y = 0" % tex(nu**2),
         "y = C_1 J_{%s}(x) + C_2 Y_{%s}(x)" % (tex(nu), tex(nu))])]

    terms = sum((-1) ** k / (sp.factorial(k) * sp.gamma(k + nu + 1)) * (X / 2) ** (2 * k + nu)
                for k in range(4))
    steps.append(step(
        "Series for J",
        "The method of Frobenius with \\(r = \\nu\\) gives a series that converges for every \\(x\\).",
        ["J_\\nu(x) = \\sum_{k=0}^{\\infty} \\frac{(-1)^k}{k!\\,\\Gamma(k+\\nu+1)}"
         "\\left(\\frac{x}{2}\\right)^{2k+\\nu}",
         "%s(x) = %s + \\cdots" % (name, " + ".join(
             tex(sp.simplify(t)) for t in sp.Add.make_args(sp.expand(terms))[::-1]
         ).replace("+ -", "- "))]))

    closed_j = closed_y = None
    if not whole:
        closed_j, closed_y = _half_integer_forms(nu)
        steps.append(step(
            "Closed form",
            "For half-integer order the series sums to elementary functions.",
            ["%s(x) = %s" % (name, tex(closed_j)), "Y_{%s}(x) = %s" % (tex(nu), tex(closed_y))]))

    zeros_j = [float(mpmath.besseljzero(v, k)) for k in range(1, 6)]
    zeros_y = [float(mpmath.besselyzero(v, k)) for k in range(1, 6)]
    table = step("Zeros",
                 "The positive zeros are the values that appear in problems on a disk or a "
                 "cylinder. Far from the origin they are spaced about \\(\\pi\\) apart.")
    table["table"] = {"headers": ["k", "\\text{zero of } %s" % name, "\\text{zero of } Y_{%s}" % tex(nu)],
                      "rows": [[str(k + 1), "%.6f" % zeros_j[k], "%.6f" % zeros_y[k]] for k in range(5)]}
    steps.append(table)
    steps.append(step(
        "Recurrence and derivative rules",
        "These hold for \\(J_\\nu\\) and for \\(Y_\\nu\\).",
        ["J_{\\nu-1}(x) + J_{\\nu+1}(x) = \\frac{2\\nu}{x} J_\\nu(x)",
         "J_{\\nu-1}(x) - J_{\\nu+1}(x) = 2 J_\\nu'(x)",
         "\\frac{d}{dx}\\left[x^{\\nu} J_\\nu(x)\\right] = x^{\\nu} J_{\\nu-1}(x),\\qquad "
         "\\frac{d}{dx}\\left[x^{-\\nu} J_\\nu(x)\\right] = -x^{-\\nu} J_{\\nu+1}(x)"]))
    steps.append(step(
        "Behaviour for large x",
        "Both functions oscillate with slowly shrinking amplitude.",
        ["J_\\nu(x) \\approx \\sqrt{\\frac{2}{\\pi x}}\\cos\\left(x - \\frac{\\nu\\pi}{2} - \\frac{\\pi}{4}\\right)"]))

    expr = _plot(closed_j) if closed_j is not None else _series_plot(int(nu))
    points = []
    xs = [0.1 * i for i in range(2, 301)]
    for value in xs:
        y = float(mpmath.bessely(v, value))
        if y > -3:
            points.append([value, y])
    # Independent check of what is plotted and of the zeros.
    worst = max(abs(float(mpmath.besselj(v, z))) for z in zeros_j)
    ok = worst < 1e-9
    note = ("The listed zeros make \\(%s\\) vanish to nine decimal places. The graph of "
            "\\(%s\\) is drawn for \\(%s|x| \\le %d\\)." % (name, name, "" if whole else "0 < ", PLOT_RANGE))
    if closed_j is not None:
        answer = ["%s(x) = %s" % (name, tex(closed_j))]
    elif nu == 0:
        answer = ["J_{0}(x) = \\sum_{k=0}^{\\infty} \\frac{(-1)^k}{(k!)^2}\\left(\\frac{x}{2}\\right)^{2k}"]
    else:
        answer = ["%s(x) = \\sum_{k=0}^{\\infty} \\frac{(-1)^k}{k!\\,(k+%s)!}"
                  "\\left(\\frac{x}{2}\\right)^{2k+%s}" % (name, tex(nu), tex(nu))]
    m = method("bessel", "Bessel functions",
               "Bessel's equation has a regular singular point at 0, so its solutions are "
               "Frobenius series, not elementary functions (except for half-integer order).",
               steps, answer, plot={"particular_expr": expr, "points": points},
               verified=ok, note=note)
    return result("special", "Bessel functions", "x^2 y'' + x y' + \\left(x^2 - %s\\right) y = 0" % tex(nu**2),
                  "Bessel functions of order %s." % sp.sstr(nu), [m], span=7.5,
                  legends={"exact": "J", "points": "Y"})


# --------------------------------------------------------------------------
# Orthogonal polynomials
# --------------------------------------------------------------------------

_FAMILIES = {
    "legendre": dict(
        title="Legendre polynomials", letter="P", fn=sp.legendre, lo=-1, hi=1, weight=sp.Integer(1),
        ode=lambda n, y: (1 - X**2) * y.diff(X, 2) - 2 * X * y.diff(X) + n * (n + 1) * y,
        ode_tex="(1 - x^2) y'' - 2x y' + n(n+1) y = 0",
        formula="P_n(x) = \\frac{1}{2^n n!}\\frac{d^n}{dx^n}\\left(x^2 - 1\\right)^n",
        seed=lambda n: (X**2 - 1) ** n, finish=lambda n, d: d / (2**n * sp.factorial(n)),
        recurrence="(n+1) P_{n+1}(x) = (2n+1)\\,x\\,P_n(x) - n\\,P_{n-1}(x)",
        norm=lambda n: sp.Rational(2, 2 * n + 1), norm_tex="\\frac{2}{2n+1}", span=1.0),
    "hermite": dict(
        title="Hermite polynomials", letter="H", fn=sp.hermite, lo=-sp.oo, hi=sp.oo, weight=sp.exp(-X**2),
        ode=lambda n, y: y.diff(X, 2) - 2 * X * y.diff(X) + 2 * n * y,
        ode_tex="y'' - 2x y' + 2n y = 0",
        formula="H_n(x) = (-1)^n e^{x^2}\\frac{d^n}{dx^n} e^{-x^2}",
        seed=lambda n: sp.exp(-X**2), finish=lambda n, d: (-1) ** n * sp.exp(X**2) * d,
        recurrence="H_{n+1}(x) = 2x\\,H_n(x) - 2n\\,H_{n-1}(x)",
        norm=lambda n: sp.sqrt(sp.pi) * 2**n * sp.factorial(n), norm_tex="\\sqrt{\\pi}\\,2^n n!", span=2.0),
    "laguerre": dict(
        title="Laguerre polynomials", letter="L", fn=sp.laguerre, lo=0, hi=sp.oo, weight=sp.exp(-X),
        ode=lambda n, y: X * y.diff(X, 2) + (1 - X) * y.diff(X) + n * y,
        ode_tex="x y'' + (1 - x) y' + n y = 0",
        formula="L_n(x) = \\frac{e^{x}}{n!}\\frac{d^n}{dx^n}\\left(x^n e^{-x}\\right)",
        seed=lambda n: X**n * sp.exp(-X), finish=lambda n, d: sp.exp(X) * d / sp.factorial(n),
        recurrence="(n+1) L_{n+1}(x) = (2n+1-x)\\,L_n(x) - n\\,L_{n-1}(x)",
        norm=lambda n: sp.Integer(1), norm_tex="1", span=4.0),
    "chebyshev": dict(
        title="Chebyshev polynomials", letter="T", fn=sp.chebyshevt, lo=-1, hi=1,
        weight=1 / sp.sqrt(1 - X**2),
        ode=lambda n, y: (1 - X**2) * y.diff(X, 2) - X * y.diff(X) + n**2 * y,
        ode_tex="(1 - x^2) y'' - x y' + n^2 y = 0",
        formula="T_n(\\cos\\theta) = \\cos n\\theta", seed=None, finish=None,
        recurrence="T_{n+1}(x) = 2x\\,T_n(x) - T_{n-1}(x)",
        norm=lambda n: sp.pi if n == 0 else sp.pi / 2, norm_tex="\\frac{\\pi}{2}\\ (\\pi \\text{ for } n = 0)",
        span=1.0),
}


def _weighted_integral(expr, family):
    """Numerical integral of expr times the family's weight over its interval."""
    f = sp.lambdify(X, expr * family["weight"], "mpmath")
    lo, hi = family["lo"], family["hi"]
    a = -mpmath.inf if lo == -sp.oo else float(lo)
    b = mpmath.inf if hi == sp.oo else float(hi)
    return float(mpmath.quad(f, [a, b] if (a, b) != (-mpmath.inf, mpmath.inf) else [a, 0, b]))


def _polynomial(operation, text):
    family = _FAMILIES[operation]
    n = _order(text, "The degree n", 0, 8)
    letter = family["letter"]
    poly = sp.expand(family["fn"](n, X))
    name = "%s_{%d}" % (letter, n)
    steps = [step(
        "The equation it solves",
        "\\(%s_n\\) is the polynomial solution of this equation, which arises when a "
        "boundary value problem is separated in the matching coordinates." % letter,
        [family["ode_tex"]])]
    if family["seed"] is not None:
        seed = family["seed"](n)
        derived = sp.diff(seed, X, n) if n else seed
        steps.append(step(
            "Rodrigues' formula",
            "Differentiate \\(n = %d\\) time%s and tidy up." % (n, "" if n == 1 else "s"),
            [family["formula"],
             "\\frac{d^{%d}}{dx^{%d}}\\left[%s\\right] = %s" % (n, n, tex(seed), tex(sp.factor(derived))),
             "%s(x) = %s" % (name, tex(poly))]))
        built = sp.expand(sp.simplify(family["finish"](n, derived)))
    else:
        rows = [sp.Integer(1), X]
        while len(rows) <= n:
            rows.append(sp.expand(2 * X * rows[-1] - rows[-2]))
        steps.append(step(
            "Build it from the recurrence",
            "Start from \\(T_0 = 1\\) and \\(T_1 = x\\). The defining property is "
            "\\(%s\\)." % family["formula"],
            [family["recurrence"]] + ["T_{%d}(x) = %s" % (k, tex(rows[k])) for k in range(min(n, 8) + 1)]))
        built = rows[n]
    if family["seed"] is not None:
        steps.append(step("Recurrence relation",
                          "Each polynomial follows from the two before it, which is the "
                          "practical way to generate them.", [family["recurrence"]]))
    weight = family["weight"]
    interval = "%s,\\ %s" % (tex(family["lo"]), tex(family["hi"]))
    steps.append(step(
        "Orthogonality",
        "Polynomials of different degree are orthogonal on the interval with this weight; "
        "that is what lets a function be expanded in them.",
        ["\\int_{%s}^{%s} %s\\,%s_m(x)\\,%s_n(x)\\,dx = 0\\quad (m \\ne n)"
         % (tex(family["lo"]), tex(family["hi"]), "" if weight == 1 else tex(weight) + "\\,", letter, letter),
         "\\int_{%s}^{%s} %s\\,%s_n(x)^2\\,dx = %s"
         % (tex(family["lo"]), tex(family["hi"]), "" if weight == 1 else tex(weight) + "\\,", letter,
            family["norm_tex"]),
         "\\text{for } %s:\\quad %s" % (name, tex(family["norm"](n)))]))
    if n:
        roots = sorted(float(sp.re(r)) for r in sp.Poly(poly, X).nroots())
        steps.append(step(
            "Zeros",
            "It has %d distinct real zero%s, all inside the interval \\((%s)\\)."
            % (n, "" if n == 1 else "s", interval),
            ["x \\approx " + ",\\ ".join("%.5f" % r for r in roots)]))

    ok = sp.expand(built - poly) == 0 and sp.simplify(family["ode"](n, poly)) == 0
    norm = _weighted_integral(poly**2, family)
    ok = ok and abs(norm - float(family["norm"](n))) < 1e-6 * (1 + abs(norm))
    if n:
        cross = _weighted_integral(poly * family["fn"](n - 1, X), family)
        ok = ok and abs(cross) < 1e-6 * (1 + abs(norm))
    partials = []
    for k in range(n + 1):
        expr = _plot(sp.expand(family["fn"](k, X)))
        partials.append({"n": k, "expr": expr, "label": "%s%s" % (letter, _sub(k))})
    m = method(operation, family["title"],
               "These are the polynomial solutions of a Sturm\u2013Liouville problem, so "
               "they form an orthogonal family.",
               steps, "%s(x) = %s" % (name, tex(poly)), plot={"partials": partials},
               verified=bool(ok),
               note="The polynomial satisfies its differential equation, and the orthogonality "
                    "integrals were confirmed numerically." if ok else
                    "The polynomial could not be confirmed.")
    return result("special", family["title"], "%s(x)" % name,
                  "%s of degree %d." % (family["title"][:-1], n), [m], span=family["span"])


def _sub(k):
    return "".join("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"[int(d)] for d in str(k))


# --------------------------------------------------------------------------
# Sturm-Liouville form
# --------------------------------------------------------------------------

def _end(text, label):
    raw = (text or "").strip().lower().replace(" ", "")
    if raw in ("oo", "+oo", "inf", "infinity"):
        return sp.oo
    if raw in ("-oo", "-inf", "-infinity"):
        return -sp.oo
    return parse_number(text, label)


def _sturm(text, lo_text, hi_text):
    parts = [p.strip() for p in (text or "").split(";")]
    if len(parts) != 4 or not all(parts):
        raise InputError("Enter the four coefficients a; b; c; d of "
                         "a y'' + b y' + (c + \u03bb d) y = 0, separated by semicolons. "
                         "For Bessel's equation of order 2: x^2; x; -4; x^2")
    a, b, c, d = (parse_expression(p, ["x"])[0].xreplace({sp.Symbol("x", real=True): X}) for p in parts)
    if a == 0:
        raise InputError("The coefficient a of y'' cannot be 0.")
    if d == 0:
        raise InputError("The coefficient d of \u03bb y cannot be 0.")
    lam = sp.Symbol("lambda")
    ratio = sp.simplify(b / a)
    exponent = integrate(ratio, X)
    if exponent.has(sp.Integral):
        raise InputError("The integral of b/a has no closed form, so the self-adjoint "
                         "form cannot be written explicitly.")
    p = sp.powsimp(sp.simplify(sp.exp(exponent)))
    q = sp.simplify(p * c / a)
    w = sp.simplify(p * d / a)
    # The integrating factor is fixed only up to a constant; choose the sign
    # that makes the weight positive (probing inside the interval when given).
    probe = sp.Rational(1, 2)
    try:
        if (lo_text or "").strip() and (hi_text or "").strip():
            lo0, hi0 = _end(lo_text, "The left end"), _end(hi_text, "The right end")
            if lo0 not in (sp.oo, -sp.oo) and hi0 not in (sp.oo, -sp.oo):
                probe = (lo0 + hi0) / 2 + (hi0 - lo0) / 7
        if float(sp.N(w.subs(X, probe))) < 0:
            p, q, w = sp.simplify(-p), sp.simplify(-q), sp.simplify(-w)
    except (TypeError, ValueError):
        pass
    steps = [
        step("Start from the equation",
             "Read off the coefficients.",
             ["a\\,y'' + b\\,y' + (c + \\lambda d)\\,y = 0",
              "a = %s,\\quad b = %s,\\quad c = %s,\\quad d = %s" % (tex(a), tex(b), tex(c), tex(d))]),
        step("Find the integrating factor",
             "Multiplying the equation by \\(p/a\\), with \\(p = e^{\\int b/a\\,dx}\\), makes the "
             "first two terms the derivative of \\(p\\,y'\\), because then \\(p' = p\\,b/a\\).",
             ["\\int \\frac{b}{a}\\,dx = %s" % tex(exponent), "p(x) = %s" % tex(p)]),
        step("Identify q and the weight w",
             "After multiplying by \\(p/a\\) the remaining terms give \\(q\\) and \\(w\\).",
             ["q(x) = \\frac{p\\,c}{a} = %s" % tex(q), "w(x) = \\frac{p\\,d}{a} = %s" % tex(w)]),
    ]
    lead = "y''" if p == 1 else "\\left(%s\\,y'\\right)'" % _wrap(p)
    weight = "\\lambda" if w == 1 else "\\lambda\\,%s" % _wrap(w)
    inside = weight if q == 0 else "%s + %s" % (tex(q), weight)
    form = "%s + %s y = 0" % (lead, weight + "\\," if q == 0 else "\\left(%s\\right)" % inside)
    steps.append(step("Self-adjoint (Sturm\u2013Liouville) form", "", [form]))
    steps.append(step(
        "What this form gives you",
        "With suitable boundary conditions the eigenvalues are real, and eigenfunctions "
        "belonging to different eigenvalues are orthogonal with weight \\(w\\).",
        ["\\int_a^b w(x)\\,y_m(x)\\,y_n(x)\\,dx = 0\\quad (m \\ne n)"]))

    if (lo_text or "").strip() and (hi_text or "").strip():
        lo, hi = _end(lo_text, "The left end"), _end(hi_text, "The right end")
        if not lo < hi:
            raise InputError("The left end of the interval must be less than the right end.")
        reasons = []
        if lo == -sp.oo or hi == sp.oo:
            reasons.append("the interval is infinite")
        for end, label in ((lo, "left"), (hi, "right")):
            if end in (sp.oo, -sp.oo):
                continue
            try:
                side = "+" if label == "left" else "-"
                values = [sp.limit(e, X, end, side) for e in (p, q, w)]
            except Exception:
                values = [sp.nan]
            if any(v.has(sp.oo, sp.zoo, sp.nan) for v in values):
                reasons.append("a coefficient is unbounded at the %s end" % label)
            elif values[0] == 0:
                reasons.append("\\(p\\) vanishes at the %s end" % label)
            elif values[2] == 0:
                reasons.append("\\(w\\) vanishes at the %s end" % label)
        steps.append(step(
            "Regular or singular on the interval",
            ("The problem is singular on \\(%s%s,\\ %s%s\\) because %s. Boundedness of the "
             "solution usually replaces a boundary condition at such an end."
             % ("(" if lo == -sp.oo else "[", tex(lo), tex(hi), ")" if hi == sp.oo else "]",
                " and ".join(reasons))) if reasons else
            ("The problem is regular on \\([%s, %s]\\): the interval is finite and "
             "\\(p\\), \\(q\\), \\(w\\) are continuous there with \\(p\\) and \\(w\\) not zero "
             "at the ends." % (tex(lo), tex(hi)))))

    y = sp.Function("y")(X)
    original = a * y.diff(X, 2) + b * y.diff(X) + (c + lam * d) * y
    rebuilt = sp.diff(p * y.diff(X), X) + (q + lam * w) * y
    ok = sp.simplify(sp.expand(rebuilt - p / a * original)) == 0
    m = method("sturm", "Self-adjoint form",
               "Any second-order linear equation of this shape can be put in self-adjoint "
               "form with an integrating factor.",
               steps, form, verified=bool(ok),
               note="Expanding the self-adjoint form gives back the original equation times "
                    "\\(p/a\\)." if ok else "The form could not be confirmed by expanding it.")
    return result("special", "Sturm\u2013Liouville form",
                  "%s\\,y'' + %s\\,y' + \\left(%s + \\lambda\\,%s\\right) y = 0"
                  % tuple("\\left(%s\\right)" % tex(e) if e.is_Add else tex(e) for e in (a, b, c, d)),
                  "Second-order equation rewritten in self-adjoint form.", [m])


# --------------------------------------------------------------------------
# Legendre expansion
# --------------------------------------------------------------------------

def _expansion(text, count_text):
    f, symbols = parse_expression(text, ["x"])
    f = f.xreplace({symbols["x"]: X})
    if f.has(sp.DiracDelta):
        raise InputError("An impulse cannot be expanded in Legendre polynomials.")
    count = _order(count_text, "The number of terms", 1, 9) if (count_text or "").strip() else 6
    parts = pieces.split(f, X, sp.Integer(-1), sp.Integer(1))
    fn = pieces.numeric(f, X)
    cuts = [-1.0] + [float(c) for c in pieces.breakpoints(f, X) if -1 < c < 1] + [1.0]
    coeffs, exact_all, worst = [], True, 0.0
    for n in range(count):
        P = sp.legendre(n, X)
        factor = sp.Rational(2 * n + 1, 2)
        reference = float(factor) * float(mpmath.quad(
            lambda t: fn(t) * mpmath.legendre(n, t), cuts))
        value = None
        if exact_all:
            total = sp.Integer(0)
            for a, b, expr in parts:
                piece = run_with_timeout(lambda: sp.integrate(expr * P, (X, a, b)), 3, None)
                if piece is None or piece.has(sp.Integral):
                    total = None
                    break
                total += piece
            if total is not None:
                value = sp.simplify(factor * total)
                if value.has(sp.I) or abs(float(sp.N(value)) - reference) > 1e-7:
                    value = None
            if value is None:
                exact_all = False
        if value is None:
            value = sp.Float(round(reference, 8))
        else:
            worst = max(worst, abs(float(sp.N(value)) - reference))
        coeffs.append(value)

    def partial(k):
        return sum(coeffs[n] * sp.legendre(n, X) for n in range(k))

    steps = [
        step("Coefficient formula",
             "The Legendre polynomials are orthogonal on \\([-1, 1]\\) with "
             "\\(\\int_{-1}^{1} P_n^2\\,dx = \\frac{2}{2n+1}\\), so multiplying the series by "
             "\\(P_n\\) and integrating isolates one coefficient.",
             ["f(x) = \\sum_{n=0}^{\\infty} c_n P_n(x),\\qquad "
              "c_n = \\frac{2n+1}{2}\\int_{-1}^{1} f(x)\\,P_n(x)\\,dx"]),
        step("First coefficients",
             "The first two, written out.",
             ["c_0 = \\frac{1}{2}\\int_{-1}^{1} %s\\,dx = %s" % (_wrap(f), tex(coeffs[0]))]
             + (["c_1 = \\frac{3}{2}\\int_{-1}^{1} %s\\,x\\,dx = %s" % (_wrap(f), tex(coeffs[1]))]
                if count > 1 else [])),
    ]
    table = step("All the coefficients",
                 "" if exact_all else "Where an integral has no closed form the value is "
                 "given as a decimal from numerical integration.")
    table["table"] = {"headers": ["n", "P_n(x)", "c_n"],
                      "rows": [[str(n), tex(sp.expand(sp.legendre(n, X))), tex(coeffs[n])]
                               for n in range(count)]}
    steps.append(table)
    shown = [c * sp.Symbol("P_{%d}" % n) for n, c in enumerate(coeffs) if c != 0]
    series_tex = "0" if not shown else tex(shown[0])
    for term in shown[1:]:
        series_tex += (" - " + tex(-term)) if term.could_extract_minus_sign() else (" + " + tex(term))
    answer = "f(x) \\approx %s" % series_tex
    samples = [-0.85, -0.5, -0.2, 0.15, 0.45, 0.8]
    top = sp.lambdify(X, partial(count), "mpmath")
    error = max(abs(float(top(s)) - float(fn(s))) for s in samples)
    exact_poly = error < 1e-9
    note = ("The coefficients agree with numerical integration. " +
            ("The series reproduces \\(f\\) exactly, because \\(f\\) is a polynomial of degree "
             "below %d." % count if exact_poly else
             "With %d term%s the largest error at six test points in \\((-1, 1)\\) is %.2g; "
             "it shrinks as terms are added%s." % (
                 count, "" if count == 1 else "s", error,
                 ", slowly near a jump" if len(cuts) > 2 else "")))
    mask = "/Heaviside(1 - Abs(x))"      # leave the graph blank outside [-1, 1]
    partials = []
    for k in range(1, count + 1):
        expr = _plot(sp.expand(partial(k)))
        if expr:
            partials.append({"n": k, "expr": "(%s)%s" % (expr, mask),
                             "label": "%d term%s" % (k, "" if k == 1 else "s")})
    base = _plot(f)
    m = method("expansion", "Legendre series",
               "Any reasonable function on \\([-1, 1]\\) can be expanded in Legendre "
               "polynomials, just as it can in sines and cosines.",
               steps, answer,
               plot={"particular_expr": "(%s)%s" % (base, mask) if base else None, "partials": partials},
               verified=True if worst < 1e-7 else None, note=note)
    return result("special", "Legendre expansion", "f(x) = %s" % tex(f),
                  "Expansion of f in Legendre polynomials on [-1, 1].", [m], span=1.0,
                  legends={"exact": "f(x)"})


def _wrap(expr):
    if expr.is_Add or expr.could_extract_minus_sign():
        return "\\left(%s\\right)" % tex(expr)
    return tex(expr)


def solve(operation, text, extra1="", extra2=""):
    if operation == "bessel":
        return _bessel(text)
    if operation in _FAMILIES:
        return _polynomial(operation, text)
    if operation == "sturm":
        return _sturm(text, extra1, extra2)
    if operation == "expansion":
        return _expansion(text, extra1)
    raise InputError("Unknown special-functions operation.")


def preview(operation, text):
    if operation == "bessel":
        return "J_{%s}(x),\\quad Y_{%s}(x)" % ((tex(parse_number(text, "The order")),) * 2)
    if operation in _FAMILIES:
        return "%s_{%d}(x)" % (_FAMILIES[operation]["letter"], _order(text, "The degree n", 0, 8))
    if operation == "sturm":
        parts = [p.strip() for p in text.split(";")]
        shown = [tex(parse_expression(p, ["x"])[0]) if p else "\\,?\\," for p in (parts + [""] * 4)[:4]]
        return "\\left(%s\\right) y'' + \\left(%s\\right) y' + \\left(%s + \\lambda\\left(%s\\right)\\right) y = 0" % tuple(shown)
    return "f(x) = %s" % tex(parse_expression(text, ["x"])[0])
