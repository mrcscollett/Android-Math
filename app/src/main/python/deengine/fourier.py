"""Fourier series of a function on (-L, L), and half-range sine and cosine series on (0, L)."""
import sympy as sp

from . import integrals
from .first_order import step
from .parsing import InputError, parse_expression, parse_number
from .util import TIMEOUT, run_with_timeout, tex

N = sp.Symbol("n", integer=True, positive=True)
PARTIALS = [1, 2, 3, 5, 10, 20]
_PLOT_FUNCS = {"sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan", "sinh",
               "cosh", "tanh", "exp", "log", "Abs", "Heaviside", "floor"}


class _NoClosedForm(Exception):
    pass


def _definite(expr, x, a, b):
    value = run_with_timeout(lambda: sp.integrate(expr, (x, a, b)), TIMEOUT, None)
    if value is None or value.has(sp.Integral) or value.has(sp.zoo, sp.nan, sp.oo):
        raise _NoClosedForm()
    return sp.simplify(value)


def _generic(coef):
    """The formula valid for general n (the 'otherwise' branch of a Piecewise)."""
    if isinstance(coef, sp.Piecewise):
        return sp.simplify(coef.args[-1][0])
    return coef


def _value(coef, k):
    """Numerical value of a coefficient at n = k."""
    try:
        value = sp.N(coef.subs(N, k))
        if value.is_real and value.is_finite:
            return float(value)
    except Exception:
        pass
    return None


def _plot(expr, x):
    try:
        if expr.has(sp.I, sp.oo, sp.zoo, sp.nan, sp.Integral, sp.Piecewise):
            return None
        if any(type(f).__name__ not in _PLOT_FUNCS for f in expr.atoms(sp.Function)):
            return None
        if not expr.free_symbols <= {x}:
            return None
        return sp.sstr(expr.xreplace({x: sp.Symbol("x")}))
    except Exception:
        return None


def _num(value):
    return repr(round(value, 12))


def _partial(a0, a, b, L, count):
    """Plot string of the partial sum with harmonics 1..count (numeric coefficients)."""
    w = float(sp.pi / L)
    parts = [_num(a0 / 2)] if abs(a0) > 1e-13 else []
    for k in range(1, count + 1):
        if abs(a[k]) > 1e-13:
            parts.append("%s*cos(%s*x)" % (_num(a[k]), _num(k * w)))
        if abs(b[k]) > 1e-13:
            parts.append("%s*sin(%s*x)" % (_num(b[k]), _num(k * w)))
    return " + ".join(parts) if parts else "0"


def _cases(name, coef):
    """LaTeX lines for a coefficient, with even and odd n spelled out when they differ."""
    shown = _generic(coef)
    lines = ["%s = %s" % (name, tex(shown))]
    if shown.has((-1) ** N):
        even = sp.simplify(shown.subs((-1) ** N, 1))
        odd = sp.simplify(shown.subs((-1) ** N, -1))
        if even != odd:
            lines.append("%s = %s\\ (n\\ \\text{even}),\\qquad %s = %s\\ (n\\ \\text{odd})"
                         % (name, tex(even), name, tex(odd)))
    return lines


def _first_terms(a0, an, bn, L, x):
    """LaTeX of the series with its first few nonzero terms."""
    terms = []
    if a0 != 0:
        terms.append(sp.nsimplify(a0) / 2 if a0.is_number else a0 / 2)
    k = 1
    while len(terms) < 5 and k <= 14:
        for coef, fn in ((an, sp.cos), (bn, sp.sin)):
            try:
                value = sp.simplify(coef.subs(N, k))
            except Exception:
                value = sp.Integer(0)
            if value != 0 and value.is_finite:
                terms.append(value * fn(k * sp.pi * x / L))
        k += 1
    if not terms:
        return "0"
    out = tex(terms[0])
    for term in terms[1:]:
        out += (" - " + tex(-term)) if term.could_extract_minus_sign() else (" + " + tex(term))
    # Fewer than five terms means every later coefficient is zero: the series stops.
    return out + " + \\cdots" if len(terms) >= 5 else out


def _check(pieces, weight_factor, coef, kind, L, x):
    """Compare a coefficient formula with numerical integration for n = 1, 2, 3."""
    try:
        for k in (1, 2, 3):
            trig = (sp.cos if kind == "a" else sp.sin)(k * sp.pi * x / L)
            reference = sum(sp.Integral(f * trig, (x, lo, hi)).evalf() for f, lo, hi in pieces)
            got = _value(coef, k)
            if got is None or abs(float(weight_factor * reference) - got) > 1e-7:
                return False
        return True
    except Exception:
        return None


def _method(method_id, name, reason, pieces, factor, want_a, want_b, L, x, extension,
            interval_tex, symmetry_note):
    """One series. `pieces` is a list of (formula, lower, upper) to integrate over."""
    arg = N * sp.pi * x / L
    try:
        a0 = sp.simplify(factor * sum(_definite(f, x, lo, hi) for f, lo, hi in pieces)) if want_a else sp.Integer(0)
        an = sp.simplify(factor * sum(_definite(f * sp.cos(arg), x, lo, hi) for f, lo, hi in pieces)) if want_a else sp.Integer(0)
        bn = sp.simplify(factor * sum(_definite(f * sp.sin(arg), x, lo, hi) for f, lo, hi in pieces)) if want_b else sp.Integer(0)
    except _NoClosedForm:
        return {"id": method_id, "name": name, "applicable": False,
                "reason": "One of the coefficient integrals has no closed form in elementary functions."}

    f_tex = "f(x)"
    factor_tex = tex(factor)
    formulas, setup = [], []
    if want_a and want_b:
        formulas.append("f(x) \\sim \\frac{a_0}{2} + \\sum_{n=1}^{\\infty}\\left(a_n\\cos\\frac{n\\pi x}{L} + b_n\\sin\\frac{n\\pi x}{L}\\right)")
    elif want_a:
        formulas.append("f(x) \\sim \\frac{a_0}{2} + \\sum_{n=1}^{\\infty} a_n\\cos\\frac{n\\pi x}{L}")
    else:
        formulas.append("f(x) \\sim \\sum_{n=1}^{\\infty} b_n\\sin\\frac{n\\pi x}{L}")
    if want_a:
        formulas.append("a_n = %s\\int_{%s} %s\\cos\\frac{n\\pi x}{L}\\,dx\\quad (n = 0, 1, 2, \\dots)"
                        % (factor_tex, interval_tex, f_tex))
    if want_b:
        formulas.append("b_n = %s\\int_{%s} %s\\sin\\frac{n\\pi x}{L}\\,dx\\quad (n = 1, 2, \\dots)"
                        % (factor_tex, interval_tex, f_tex))
    steps = [step("Write the series and its coefficient formulas",
                  "Here \\(L = %s\\)." % tex(L), formulas)]

    def integral_tex(trig):
        return " + ".join("\\int_{%s}^{%s} %s\\,dx" % (
            tex(lo), tex(hi), "\\left(%s\\right)%s" % (tex(f), trig) if trig or f.is_Add else tex(f))
            for f, lo, hi in pieces)

    cos_tex = "\\cos\\left(%s\\right)" % tex(arg)
    sin_tex = "\\sin\\left(%s\\right)" % tex(arg)
    lines = []
    if want_a:
        lines += ["a_0 = %s\\left[%s\\right] = %s" % (factor_tex, integral_tex(""), tex(a0))]
        lines += ["a_n = %s\\left[%s\\right]" % (factor_tex, integral_tex(cos_tex))] + _cases("a_n", an)
    if want_b:
        lines += ["b_n = %s\\left[%s\\right]" % (factor_tex, integral_tex(sin_tex))] + _cases("b_n", bn)
    working = []
    first = pieces[-1][0]
    if want_a:
        working += integrals.explain(first * sp.cos(arg), x, "a_n")
    if want_b:
        working += integrals.explain(first * sp.sin(arg), x, "b_n")
    text = ("Evaluate the integrals. Since \\(n\\) is a whole number, "
            "\\(\\sin n\\pi = 0\\) and \\(\\cos n\\pi = (-1)^n\\).")
    if symmetry_note:
        text = symmetry_note + " " + text
    if any(isinstance(c, sp.Piecewise) for c in (an, bn)):
        text += (" The formula shown holds for general \\(n\\); the few values of \\(n\\) "
                 "where it breaks down are computed separately for the graph.")
    steps.append(step("Compute the coefficients", text, lines, working))

    series_tex = "f(x) \\sim %s" % _first_terms(a0, an, bn, L, x)
    steps.append(step("Write out the series", "Substitute the coefficients; the first nonzero terms are",
                      [series_tex]))
    steps.append(step(
        "Where it converges",
        "For a piecewise smooth function the series converges to \\(f(x)\\) wherever the "
        "periodic extension is continuous, and to the average of the left and right limits "
        "at each jump. Near a jump the partial sums overshoot by about 9% (the Gibbs phenomenon)."))

    a0_value = _value(a0, 1) or 0.0
    a_values = {k: (_value(an, k) or 0.0) for k in range(1, PARTIALS[-1] + 1)}
    b_values = {k: (_value(bn, k) or 0.0) for k in range(1, PARTIALS[-1] + 1)}
    partials = [{"n": count, "expr": _partial(a0_value, a_values, b_values, L, count)}
                for count in PARTIALS]

    checks = []
    if want_a:
        checks.append(_check(pieces, factor, an, "a", L, x))
    if want_b:
        checks.append(_check(pieces, factor, bn, "b", L, x))
    ok = all(c is True for c in checks)
    bad = any(c is False for c in checks)
    note = ("The coefficient formulas were compared with numerical integration for n = 1, 2, 3."
            if ok else "The coefficient formulas did not match numerical integration."
            if bad else "Could not be checked automatically.")
    answer = {"title": "Series", "text": "", "latex": [series_tex], "substeps": [], "kind": "answer"}
    check_step = {"title": "Check", "text": note, "latex": [], "substeps": [],
                  "kind": "check-ok" if ok else "check-bad" if bad else "check-unknown"}
    return {"id": method_id, "name": name, "applicable": True, "reason": reason,
            "steps": steps + [answer, check_step],
            "general_latex": None, "particular_latex": series_tex,
            "plot": {"general_expr": None, "particular_expr": _plot(extension, x), "points": [],
                     "partials": partials},
            "verified": True if ok else False if bad else None, "verify_note": note}


def solve(f_text, left_text, L_text):
    if not f_text.strip():
        raise InputError("Enter the function f(x).")
    if not L_text.strip():
        raise InputError("Enter the half-length L of the interval, for example pi or 1.")
    L = parse_number(L_text, "L")
    if not L.is_positive:
        raise InputError("L must be a positive number.")
    f, symbols = parse_expression(f_text, ["x"])
    x = symbols["x"]
    left = None
    if left_text.strip():
        left, _ = parse_expression(left_text, ["x"])
    for expr in (f, left):
        if expr is not None and expr.has(sp.DiracDelta):
            raise InputError("An impulse delta(...) has no ordinary Fourier series here.")

    # x reduced to the base period [-L, L).
    xr = x - 2 * L * sp.floor((x + L) / (2 * L))
    methods = []
    if left is not None:
        pieces = [(left, -L, sp.Integer(0)), (f, sp.Integer(0), L)]
        extension = left.subs(x, xr) * (1 - sp.Heaviside(xr)) + f.subs(x, xr) * sp.Heaviside(xr)
        methods.append(_method(
            "full", "Fourier series", "The function is given on the whole interval \\((-L, L)\\), "
            "one formula on each half.", pieces, 1 / L, True, True, L, x, extension, "-L}^{L", ""))
        for method_id, name in (("sine", "Sine series (half-range)"), ("cosine", "Cosine series (half-range)")):
            methods.append({"id": method_id, "name": name, "applicable": False,
                            "reason": "Half-range series extend a function given on \\((0, L)\\) only; "
                                      "here the left half is specified, so the full series is the one that applies."})
        input_latex = ("f(x) = %s\\ \\ (-%s < x < 0),\\qquad f(x) = %s\\ \\ (0 < x < %s)"
                       % (tex(left), tex(L), tex(f), tex(L)))
    else:
        even = sp.simplify(f.subs(x, -x) - f) == 0
        odd = sp.simplify(f.subs(x, -x) + f) == 0
        note = ""
        if even:
            note = ("\\(f\\) is even, so every \\(b_n\\) is zero and the series has cosines only.")
        elif odd:
            note = ("\\(f\\) is odd, so \\(a_0\\) and every \\(a_n\\) are zero and the series has sines only.")
        methods.append(_method(
            "full", "Fourier series", "The same formula is used on all of \\((-L, L)\\).",
            [(f, -L, L)], 1 / L, True, True, L, x, f.subs(x, xr), "-L}^{L", note))
        absx = sp.Abs(xr)
        methods.append(_method(
            "sine", "Sine series (half-range)",
            "Extend \\(f\\) from \\((0, L)\\) as an odd function; its series has sines only.",
            [(f, sp.Integer(0), L)], 2 / L, False, True, L, x, xr / absx * f.subs(x, absx), "0}^{L", ""))
        methods.append(_method(
            "cosine", "Cosine series (half-range)",
            "Extend \\(f\\) from \\((0, L)\\) as an even function; its series has cosines only.",
            [(f, sp.Integer(0), L)], 2 / L, True, False, L, x, f.subs(x, absx), "0}^{L", ""))
        input_latex = "f(x) = %s,\\qquad L = %s" % (tex(f), tex(L))
    return {"schema": 1, "ok": True, "kind": "fourier", "order": 0, "indep": "x", "dep": "f",
            "input_latex": input_latex, "normal_latex": input_latex, "rhs_expr": None, "ic": None,
            "span": float(L),
            "summary": "Fourier series on an interval of half-length L. Use the slider on the "
                       "graph to watch the partial sums converge.",
            "other_types": [], "methods": methods}


def preview(text):
    """LaTeX for 'f;left' as typed so far."""
    parts = (text.split(";") + [""])[:2]
    lines = []
    for label, part in zip(("f(x)", "f(x)\\ (x<0)"), parts):
        if part.strip():
            expr, _ = parse_expression(part, ["x"])
            lines.append("%s = %s" % (label, tex(expr)))
    return ",\\qquad ".join(lines)
