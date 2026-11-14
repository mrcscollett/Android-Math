"""Series tools: Taylor series, convergence tests for numerical series, and
the radius and interval of convergence of a power series."""
import re

import sympy as sp

from .parsing import InputError, parse_expression, parse_number
from .toolkit import method, not_applicable, result, step
from .util import plot_expr, run_with_timeout, tex

OPERATIONS = ("taylor", "convergence", "radius")
LIMIT_TIME = 4


class _Prob:
    def __init__(self, x):
        self.x = x
        self.y = sp.Symbol("y", real=True)
        self.C = self.C1 = self.C2 = sp.Symbol("C", real=True)


def _term(text):
    """Parse a_n, accepting n! and (..)! for factorials. Returns (a_n, n)."""
    cleaned = re.sub(r"(\w+|\([^()]*\))!", r"factorial(\1)", text or "")
    expr, symbols = parse_expression(cleaned, ["n"])
    n = sp.Symbol("n", integer=True, positive=True)
    return expr.xreplace({symbols["n"]: n}), n


def _lim(expr, n):
    """limit as n -> oo, or None when it cannot be found in time."""
    out = run_with_timeout(lambda: sp.limit(expr, n, sp.oo), LIMIT_TIME, None)
    if out is None or out.has(sp.Limit) or out.has(sp.AccumBounds) or out is sp.nan:
        return None
    return out


# --------------------------------------------------------------------------
# Taylor series
# --------------------------------------------------------------------------

def _entire(f, x):
    """True for sums and products of polynomials, exp, sin, cos, sinh, cosh of polynomials."""
    for node in sp.preorder_traversal(f):
        if isinstance(node, sp.Function) and not isinstance(node, (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh)):
            return False
        if node.is_Pow and node.base.has(x) and not (node.exp.is_Integer and node.exp >= 0):
            if not isinstance(node, sp.exp):
                return False
    return True


def _taylor(text, a_text, order_text):
    f, symbols = parse_expression(text, ["x"])
    x = symbols["x"]
    a = parse_number(a_text, "The centre a") if (a_text or "").strip() else sp.Integer(0)
    try:
        order = int((order_text or "6").strip())
    except ValueError:
        raise InputError("The order must be a whole number.")
    if not 1 <= order <= 12:
        raise InputError("The order must be between 1 and 12.")
    derivs, coeffs = [], []
    current = f
    for k in range(order + 1):
        value = sp.simplify(current.subs(x, a))
        if value.has(sp.zoo, sp.nan, sp.oo) or not value.is_finite:
            value = run_with_timeout(lambda: sp.limit(current, x, a), LIMIT_TIME, None)
            if value is None or not value.is_finite:
                raise InputError("The function or one of its derivatives is not defined at x = %s, "
                                 "so it has no Taylor series there." % sp.sstr(a))
        derivs.append(value)
        coeffs.append(sp.nsimplify(value / sp.factorial(k)))
        current = sp.diff(current, x)
    poly = sum(c * (x - a) ** k for k, c in enumerate(coeffs))
    centre = "x" if a == 0 else "(x - %s)" % tex(a) if not a.could_extract_minus_sign() else "(x + %s)" % tex(-a)

    def poly_tex(upto):
        parts = []
        for k in range(upto + 1):
            c = coeffs[k]
            if c == 0:
                continue
            power = "" if k == 0 else centre if k == 1 else "%s^{%d}" % (centre, k)
            mag = -c if c.could_extract_minus_sign() else c
            body = tex(mag) if not power else ("" if mag == 1 else tex(mag) + "\\,") + power
            parts.append(("- " if c.could_extract_minus_sign() else "+ ") + body)
        text_ = " ".join(parts).lstrip("+ ").strip() or "0"
        return text_

    rows = [[str(k), tex(sp.simplify(sp.diff(f, x, k))) if k <= 4 else "\\cdots", tex(derivs[k]), tex(coeffs[k])]
            for k in range(order + 1)]
    table = step("Differentiate and evaluate at the centre",
                 "The coefficient of \\(%s^k\\) is \\(f^{(k)}(a)/k!\\)." % centre)
    table["table"] = {"headers": ["k", "f^{(k)}(x)", "f^{(k)}(a)", "f^{(k)}(a)/k!"], "rows": rows}
    steps = [step("Write the Taylor formula", "",
                  ["f(x) = \\sum_{k=0}^{\\infty} \\frac{f^{(k)}(a)}{k!}(x - a)^k,\\qquad a = %s" % tex(a)]),
             table,
             step("Assemble the polynomial", "Through the term of degree %d:" % order,
                  ["%s \\approx %s" % (tex(f), poly_tex(order))])]

    # General term and where the series converges.
    general = run_with_timeout(lambda: sp.fps(f, x, a), LIMIT_TIME, None)
    radius_note = None
    try:
        if general is not None and hasattr(general, "infinite"):
            shown = general.infinite
            if not shown.has(sp.Piecewise) and shown.has(sp.Sum):
                steps.append(step("General term", "The pattern in the coefficients gives the whole series.",
                                  ["%s = %s" % (tex(f), tex(shown))]))
    except Exception:
        pass
    if _entire(f, x):
        radius_note = ("The function is built from polynomials, exponentials, sines and cosines, "
                       "which have no singularities, so the series converges for every \\(x\\).")
        radius = sp.oo
    else:
        radius = None
        try:
            z = sp.Symbol("z")
            sing = run_with_timeout(lambda: sp.singularities(f.subs(x, z), z, sp.S.Complexes), LIMIT_TIME, None)
            branch = [sp.solve(arg.subs(x, z), z) for arg in
                      [n.args[0] for n in sp.preorder_traversal(f) if isinstance(n, sp.log)]
                      + [n.base for n in sp.preorder_traversal(f) if n.is_Pow and n.base.has(x)
                         and not n.exp.is_Integer]]
            points = list(sing) if sing is not None and sing.is_FiniteSet else None
            if points is not None:
                points += [p for group in branch for p in group]
                if points:
                    radius = min(sp.Abs(p - a) for p in points)
                    radius = sp.nsimplify(sp.simplify(radius))
                    nearest = [p for p in points if sp.simplify(sp.Abs(p - a) - radius) == 0][0]
                    radius_note = ("The nearest point where the function fails to be analytic is "
                                   "\\(x = %s\\) (counting complex points), at distance \\(%s\\) from the "
                                   "centre. That distance is the radius of convergence."
                                   % (tex(nearest), tex(radius)))
        except Exception:
            radius = None
    if radius_note:
        steps.append(step("Where the series converges", radius_note,
                          ["R = \\infty" if radius == sp.oo else "R = %s" % tex(radius)]))
    else:
        steps.append(step("Where the series converges",
                          "The radius of convergence could not be determined automatically here."))
    prob = _Prob(x)
    partials = []
    seen = set()
    for k in range(1, order + 1):
        p = sum(c * (x - a) ** j for j, c in enumerate(coeffs[:k + 1]))
        expr = plot_expr(sp.expand(p), prob)
        if expr and expr not in seen:
            seen.add(expr)
            partials.append({"n": k, "expr": expr})
    # Check the coefficients against a direct series expansion.
    direct = run_with_timeout(lambda: sp.series(f, x, a, order + 1).removeO(), 6, None)
    ok = None if direct is None else sp.simplify(sp.expand(direct - poly)) == 0
    return result("series", "Taylor series", "%s \\text{ about } x = %s" % (tex(f), tex(a)),
                  "Taylor polynomial of degree %d about x = %s." % (order, sp.sstr(a)),
                  [method("taylor", "Derivatives at the centre", "The function is differentiable at the centre.",
                          steps, "%s \\approx %s" % (tex(f), poly_tex(order)),
                          plot={"particular_expr": plot_expr(f, prob), "partials": partials},
                          verified=ok, note="Agrees with an independent series expansion." if ok
                          else "Could not be checked automatically.")],
                  span=float(min(sp.Abs(a) + 3, 50)) / 2)


# --------------------------------------------------------------------------
# Convergence tests
# --------------------------------------------------------------------------

def _decide(a, n):
    """Run every test. Returns (methods, verdict) with verdict in
    'absolute', 'conditional', 'converges', 'diverges' or None."""
    methods = []
    verdicts = []
    absolute = sp.simplify(sp.Abs(a))
    # The signs alternate if consecutive terms always have opposite signs.
    try:
        values = [sp.N(a.subs(n, k)) for k in range(1, 9)]
        alternating = all(values[k] * values[k + 1] < 0 for k in range(7))
    except Exception:
        alternating = False
    b = absolute

    # 1. nth-term test
    L = _lim(a, n)
    if L is not None and L != 0:
        methods.append(method("nth", "nth-term test", "The terms do not tend to 0.",
                              [step("Take the limit of the terms", "If the terms of a series do not tend "
                                    "to 0, the series diverges.", ["\\lim_{n\\to\\infty} a_n = %s \\ne 0" % tex(L)])],
                              "\\text{diverges}"))
        verdicts.append("diverges")
    elif L is None and alternating and _lim(b, n) not in (0, None):
        methods.append(method("nth", "nth-term test", "The terms do not tend to 0.",
                              [step("Look at the size of the terms", "The terms alternate in sign but their "
                                    "size does not shrink to 0, so the series diverges.",
                                    ["\\lim_{n\\to\\infty} |a_n| = %s \\ne 0" % tex(_lim(b, n))])],
                              "\\text{diverges}"))
        verdicts.append("diverges")
    else:
        methods.append(not_applicable("nth", "nth-term test",
                                      "The terms tend to 0, so this test says nothing: the series may "
                                      "still converge or diverge."))

    # 2. geometric
    ratio = sp.simplify(a.subs(n, n + 1) / a)
    if not ratio.has(n) and ratio.is_number:
        first = a.subs(n, 1)
        conv = sp.Abs(ratio) < 1
        lines = ["r = \\frac{a_{n+1}}{a_n} = %s" % tex(ratio)]
        if conv:
            lines.append("\\sum_{n=1}^{\\infty} a_n = \\frac{a_1}{1 - r} = \\frac{%s}{1 - %s} = %s"
                         % (tex(first), _wrap(ratio), tex(sp.simplify(first / (1 - ratio)))))
        methods.append(method("geometric", "Geometric series", "Each term is a fixed multiple of the one before.",
                              [step("Find the common ratio", "A geometric series converges exactly when "
                                    "\\(|r| < 1\\), and then its sum is the first term over \\(1 - r\\).", lines)],
                              "\\text{converges absolutely}" if conv else "\\text{diverges}"))
        verdicts.append("absolute" if conv else "diverges")
    else:
        methods.append(not_applicable("geometric", "Geometric series",
                                      "The ratio of consecutive terms is not constant."))

    # 3. p-series
    p = sp.Wild("p", exclude=[n])
    c = sp.Wild("c", exclude=[n])
    match = absolute.match(c * n ** p)
    if match and match[c] != 0 and match[p].is_number and match[p].is_real:
        power = -match[p]
        conv = power > 1
        methods.append(method("pseries", "p-series", "The terms are a constant times a power of n.",
                              [step("Identify p", "\\(\\sum 1/n^p\\) converges exactly when \\(p > 1\\).",
                                    ["|a_n| = %s,\\qquad p = %s" % (tex(absolute), tex(power))])],
                              ("\\text{converges absolutely}" if conv else
                               "\\sum |a_n| \\text{ diverges}" if alternating else "\\text{diverges}")))
        verdicts.append("absolute" if conv else ("notabs" if alternating else "diverges"))
    else:
        methods.append(not_applicable("pseries", "p-series", "The terms are not a constant times a power of n."))

    # 4. ratio test
    R = _lim(sp.simplify(sp.Abs(a.subs(n, n + 1) / a)), n)
    if R is not None and R != 1:
        conv = R < 1
        methods.append(method("ratio", "Ratio test", "The limit of the ratio is not 1.",
                              [step("Take the limit of the ratio",
                                    "The series converges absolutely if the limit is below 1 and diverges if it is above 1.",
                                    ["\\lim_{n\\to\\infty}\\left|\\frac{a_{n+1}}{a_n}\\right| = %s" % tex(R)])],
                              "\\text{converges absolutely}" if conv else "\\text{diverges}"))
        verdicts.append("absolute" if conv else "diverges")
    else:
        methods.append(not_applicable("ratio", "Ratio test",
                                      "The limit of \\(|a_{n+1}/a_n|\\) is 1, so the test is inconclusive."
                                      if R == 1 else "The limit of the ratio could not be found."))

    # 5. root test
    Q = _lim(absolute ** (1 / n), n) if not a.has(sp.factorial) else None
    if Q is not None and Q != 1:
        conv = Q < 1
        methods.append(method("root", "Root test", "The limit of the nth root is not 1.",
                              [step("Take the limit of the nth root",
                                    "The series converges absolutely if the limit is below 1 and diverges if it is above 1.",
                                    ["\\lim_{n\\to\\infty}\\sqrt[n]{|a_n|} = %s" % tex(Q)])],
                              "\\text{converges absolutely}" if conv else "\\text{diverges}"))
        verdicts.append("absolute" if conv else "diverges")
    else:
        methods.append(not_applicable("root", "Root test",
                                      "The limit of the nth root is 1, so the test is inconclusive."
                                      if Q == 1 else "The limit of the nth root was not found (or involves a factorial)."))

    # 6. limit comparison with a p-series
    done = False
    if L == 0 and not a.has(sp.factorial):
        power = _lim(-sp.log(absolute) / sp.log(n), n)
        if power is not None and power.is_number and power.is_finite and power.is_real:
            power = sp.nsimplify(power)
            ratio_to = _lim(absolute * n ** power, n)
            if ratio_to is not None and ratio_to.is_finite and ratio_to > 0:
                conv = power > 1
                methods.append(method(
                    "comparison", "Limit comparison", "The terms behave like a power of n.",
                    [step("Compare with a p-series",
                          "If \\(|a_n| / b_n\\) tends to a positive finite number, both series behave "
                          "the same way. Here \\(b_n = 1/n^{%s}\\), which %s." % (
                              tex(power), "converges (p > 1)" if conv else "diverges (p \\le 1)"),
                          ["\\lim_{n\\to\\infty} \\frac{|a_n|}{1/n^{%s}} = %s" % (tex(power), tex(ratio_to))])],
                    ("\\text{converges absolutely}" if conv else
                     "\\sum |a_n| \\text{ diverges}" if alternating else "\\text{diverges}")))
                verdicts.append("absolute" if conv else ("notabs" if alternating else "diverges"))
                done = True
    if not done:
        methods.append(not_applicable("comparison", "Limit comparison",
                                      "The terms do not behave like a simple power of n."))

    # 7. integral test
    done = False
    if not alternating and not a.has(sp.factorial) and L == 0:
        t = sp.Symbol("t", positive=True)
        ft = a.subs(n, t)
        try:
            positive = all(sp.N(ft.subs(t, v)) > 0 for v in (1, 2, 5, 20, 100))
            decreasing = all(sp.N(sp.diff(ft, t).subs(t, v)) < 0 for v in (5, 20, 100))
        except Exception:
            positive = decreasing = False
        if positive and decreasing:
            value = run_with_timeout(lambda: sp.integrate(ft, (t, 1, sp.oo)), 6, None)
            if value is not None and not value.has(sp.Integral):
                conv = value.is_finite is True
                methods.append(method(
                    "integral", "Integral test", "The terms are positive and decreasing.",
                    [step("Compare with an improper integral",
                          "For positive, decreasing terms the series and the integral converge or diverge together.",
                          ["\\int_1^{\\infty} %s\\,dx = %s" % (tex(ft.subs(t, sp.Symbol("x"))), tex(value))])],
                    "\\text{converges}" if conv else "\\text{diverges}"))
                verdicts.append("absolute" if conv else "diverges")
                done = True
    if not done:
        methods.append(not_applicable("integral", "Integral test",
                                      "It needs positive, decreasing terms with an integral that can be evaluated."))

    # 8. alternating series test
    if alternating:
        bl = _lim(b, n)
        try:
            decreasing = all(sp.N(b.subs(n, v + 1) - b.subs(n, v)) < 0 for v in (5, 10, 20, 50, 100))
        except Exception:
            decreasing = False
        if bl == 0 and decreasing:
            methods.append(method(
                "alternating", "Alternating series test", "The signs alternate.",
                [step("Check the two conditions",
                      "An alternating series converges when the sizes of its terms decrease to 0.",
                      ["b_n = %s,\\qquad b_{n+1} \\le b_n,\\qquad \\lim_{n\\to\\infty} b_n = 0" % tex(b)])],
                "\\text{converges}"))
            verdicts.append("converges")
        else:
            methods.append(not_applicable("alternating", "Alternating series test",
                                          "The sizes of the terms do not decrease to 0."))
    else:
        methods.append(not_applicable("alternating", "Alternating series test", "The signs do not alternate."))

    if "diverges" in verdicts:
        verdict = "diverges"
    elif "absolute" in verdicts:
        verdict = "absolute"
    elif "converges" in verdicts and "notabs" in verdicts:
        verdict = "conditional"
    elif "converges" in verdicts:
        verdict = "converges"
    else:
        verdict = None
    return methods, verdict


def _wrap(value):
    return "\\left(%s\\right)" % tex(value) if value.could_extract_minus_sign() else tex(value)


_VERDICT = {
    "absolute": "The series converges absolutely.",
    "conditional": "The series converges conditionally: it converges, but the series of absolute values diverges.",
    "converges": "The series converges. Whether it converges absolutely was not settled by these tests.",
    "diverges": "The series diverges.",
    None: "None of the tests here settles this series.",
}


def _convergence(text):
    a, n = _term(text)
    if not a.has(n):
        raise InputError("Enter the general term as an expression in n, such as 1/n^2.")
    methods, verdict = _decide(a, n)
    summary = _VERDICT[verdict]
    if verdict in ("absolute", "conditional", "converges"):
        total = run_with_timeout(lambda: sp.summation(a, (n, 1, sp.oo)), 5, None)
        if total is not None and total.is_finite and not total.has(sp.Sum) and not total.has(sp.Piecewise):
            summary += " Its sum (from n = 1) is %s." % sp.sstr(total)
            for m in methods:
                if m["applicable"]:
                    m["steps"].append(step("Sum of the series", "Starting from \\(n = 1\\):",
                                           ["\\sum_{n=1}^{\\infty} %s = %s" % (tex(a), tex(total))]))
    for m in methods:
        if m["applicable"]:
            m["steps"].append(step("Overall conclusion", summary))
    return result("series", "Convergence", "\\sum_{n=1}^{\\infty} %s" % tex(a), summary, methods)


# --------------------------------------------------------------------------
# Radius and interval of convergence
# --------------------------------------------------------------------------

def _radius(text, a_text):
    c, n = _term(text)
    centre = parse_number(a_text, "The centre a") if (a_text or "").strip() else sp.Integer(0)
    shift = "x" if centre == 0 else "(x - %s)" % tex(centre) if not centre.could_extract_minus_sign() \
        else "(x + %s)" % tex(-centre)
    shown = "\\sum_{n=1}^{\\infty} %s\\,%s^n" % (_wrap(c) if c.is_Add else tex(c), shift)
    L = _lim(sp.simplify(sp.Abs(c.subs(n, n + 1) / c)), n)
    used = "ratio"
    if L is None:
        L = _lim(sp.Abs(c) ** (1 / n), n)
        used = "root"
    if L is None:
        raise InputError("Neither the ratio test nor the root test gave a limit for these coefficients.")
    R = sp.oo if L == 0 else (sp.Integer(0) if L == sp.oo else sp.nsimplify(1 / L))
    limit_tex = ("\\lim_{n\\to\\infty}\\left|\\frac{c_{n+1}}{c_n}\\right|" if used == "ratio"
                 else "\\lim_{n\\to\\infty}\\sqrt[n]{|c_n|}")
    steps = [step("Apply the %s test" % used,
                  "The series converges when \\(L\\,|%s| < 1\\), where \\(L\\) is this limit." % shift,
                  ["L = %s = %s" % (limit_tex, tex(L)),
                   "R = \\frac{1}{L} = %s" % ("\\infty" if R == sp.oo else tex(R))])]
    if R == sp.oo:
        interval = "(-\\infty, \\infty)"
        steps.append(step("Interval of convergence", "The series converges for every \\(x\\)."))
    elif R == 0:
        interval = "\\{%s\\}" % tex(centre)
        steps.append(step("Interval of convergence", "The series converges only at its centre."))
    else:
        ends = []
        for sign, point in ((-1, centre - R), (1, centre + R)):
            term = sp.simplify(c * (sign * R) ** n)
            _, verdict = _decide(term, n)
            inside = verdict in ("absolute", "conditional", "converges")
            ends.append(inside if verdict is not None else None)
            steps.append(step("Test the endpoint x = %s" % sp.sstr(point),
                              "There the series is \\(\\sum %s\\). %s" % (tex(term), _VERDICT[verdict]),
                              []))
        left = "[" if ends[0] else "("
        right = "]" if ends[1] else ")"
        interval = "%s%s,\\ %s%s" % (left, tex(centre - R), tex(centre + R), right)
        if None in ends:
            steps.append(step("Interval of convergence",
                              "An endpoint could not be settled, so it is shown as excluded; check it by hand."))
    answer = "R = %s,\\qquad \\text{interval: } %s" % ("\\infty" if R == sp.oo else tex(R), interval)
    return result("series", "Radius of convergence", shown,
                  "Radius of convergence %s." % ("infinite" if R == sp.oo else sp.sstr(R)),
                  [method("radius", "%s test" % used.capitalize(),
                          "The limit of the %s exists." % ("ratio of coefficients" if used == "ratio" else "nth root"),
                          steps, answer)])


def solve(operation, text, a_text="", b_text=""):
    if operation == "taylor":
        return _taylor(text, a_text, b_text)
    if operation == "convergence":
        return _convergence(text)
    if operation == "radius":
        return _radius(text, a_text)
    raise InputError("Unknown series operation.")


def preview(operation, text):
    if operation == "taylor":
        return tex(parse_expression(text, ["x"])[0])
    a, _ = _term(text)
    if operation == "radius":
        return "\\sum %s\\,x^n" % (_wrap(a) if a.is_Add else tex(a))
    return "\\sum_{n=1}^{\\infty} %s" % tex(a)
