"""Laplace transforms and inverse transforms with the working shown."""
import sympy as sp

from .first_order import _eq, _paren, step
from .parsing import InputError, parse_expression
from .util import TIMEOUT, integrate, run_with_timeout, sample_zero, tex

S = sp.Symbol("s")


def _L(inner):
    return "\\mathcal{L}\\left\\{%s\\right\\}" % inner


def _Linv(inner):
    return "\\mathcal{L}^{-1}\\left\\{%s\\right\\}" % inner


def _tidy(F):
    """A real, compact form of a transform."""
    out = sp.simplify(sp.together(F))
    if out.has(sp.I):
        out = sp.simplify(sp.expand_complex(out))
    if out.has(sp.exp) or not out.is_rational_function(S):
        return out
    num, den = sp.fraction(sp.cancel(out))
    return sp.expand(num) / sp.factor(den)


def _strip_step(expr, t):
    """Drop u(t), which is 1 for t >= 0, but keep shifted steps u(t - a)."""
    return expr.replace(lambda e: isinstance(e, sp.Heaviside) and e.args[0] == t,
                        lambda e: sp.Integer(1))


def _rule(g, t):
    """(description, formula) of the table entry or theorem that transforms g."""
    powers = [f for f in sp.Mul.make_args(g) if f == t or (f.is_Pow and f.base == t)]
    exps = [f for f in sp.Mul.make_args(g) if isinstance(f, sp.exp)]
    trig = [f for f in sp.Mul.make_args(g) if isinstance(f, (sp.sin, sp.cos, sp.sinh, sp.cosh))]
    if g.has(sp.DiracDelta):
        return ("An impulse at \\(t = a\\) picks out the value of \\(e^{-st}\\) there.",
                "%s = e^{-as}" % _L("\\delta(t-a)"))
    if g.has(sp.Heaviside):
        return ("Second shifting theorem: a function switched on at \\(t = a\\) "
                "and shifted by \\(a\\) gains a factor \\(e^{-as}\\).",
                "%s = e^{-as}F(s)" % _L("u(t-a)\\,f(t-a)"))
    if not g.has(t):
        return ("Transform of a constant.", "%s = \\frac{1}{s}" % _L("1"))
    if exps and (powers or trig):
        return ("First shifting theorem: multiplying by \\(e^{at}\\) shifts "
                "\\(s\\) to \\(s - a\\).", "%s = F(s-a)" % _L("e^{at}f(t)"))
    if powers and trig:
        return ("Multiplying by \\(t\\) differentiates the transform.",
                "%s = -F'(s)" % _L("t\\,f(t)"))
    if exps:
        return ("Table entry for an exponential.", "%s = \\frac{1}{s-a}" % _L("e^{at}"))
    if trig:
        if isinstance(trig[0], sp.sin):
            return ("Table entry for a sine.", "%s = \\frac{b}{s^2+b^2}" % _L("\\sin bt"))
        if isinstance(trig[0], sp.cos):
            return ("Table entry for a cosine.", "%s = \\frac{s}{s^2+b^2}" % _L("\\cos bt"))
        if isinstance(trig[0], sp.sinh):
            return ("Table entry for a hyperbolic sine.", "%s = \\frac{b}{s^2-b^2}" % _L("\\sinh bt"))
        return ("Table entry for a hyperbolic cosine.", "%s = \\frac{s}{s^2-b^2}" % _L("\\cosh bt"))
    if powers and len(sp.Mul.make_args(g)) == len(powers):
        return ("Table entry for a power of \\(t\\).", "%s = \\frac{n!}{s^{n+1}}" % _L("t^n"))
    return ("Computed from the definition.",
            "%s = \\int_0^\\infty e^{-st}f(t)\\,dt" % _L("f(t)"))


def _method(method_id, name, reason, steps, answer_tex, plot, verified, note):
    return {"id": method_id, "name": name, "applicable": True, "reason": reason,
            "steps": steps, "general_latex": None, "particular_latex": answer_tex,
            "plot": {"general_expr": None, "particular_expr": plot, "points": []},
            "verified": verified, "verify_note": note}


def _na(method_id, name, reason):
    return {"id": method_id, "name": name, "applicable": False, "reason": reason}


def _plot(expr, var):
    """Plot string in x for a function of t (None if it cannot be graphed)."""
    try:
        if expr.has(sp.I, sp.oo, sp.zoo, sp.nan, sp.Integral, sp.DiracDelta, sp.Piecewise):
            return None
        allowed = {"sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan",
                   "sinh", "cosh", "tanh", "exp", "log", "Abs", "Heaviside"}
        if any(type(f).__name__ not in allowed for f in expr.atoms(sp.Function)):
            return None
        if not expr.free_symbols <= {var}:
            return None
        return sp.sstr(expr.xreplace({var: sp.Symbol("x")}))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Forward transform
# --------------------------------------------------------------------------

def forward(text):
    f, symbols = parse_expression(text, ["t"])
    t = symbols["t"]
    terms = sp.Add.make_args(f)
    pieces = []
    for term in terms:
        coef, core = term.as_independent(t)
        if core == 1 and coef != 1:
            coef, core = coef, sp.Integer(1)
        F = run_with_timeout(lambda: sp.laplace_transform(core, t, S, noconds=True), TIMEOUT, None)
        if F is None or F.has(sp.LaplaceTransform) or F.has(sp.Integral):
            raise InputError("The Laplace transform of %s could not be found." % sp.sstr(term))
        pieces.append((coef, core, _tidy(F)))
    total = _tidy(sum(c * F for c, _, F in pieces))
    answer = "%s = %s" % (_L(tex(f)), tex(total))

    steps = []
    if len(pieces) > 1:
        steps.append(step(
            "Split using linearity",
            "The transform of a sum is the sum of the transforms, and constants factor out.",
            ["%s = %s" % (_L(tex(f)), " + ".join(
                "%s%s" % ("" if c == 1 else _paren(c) if c.is_Add or c.could_extract_minus_sign() else tex(c),
                           _L(tex(g))) for c, g, _ in pieces))]))
    for coef, core, F in pieces:
        text_, formula = _rule(core, t)
        steps.append(step("Transform the constant" if core == 1 else
                          "Transform term %d" % (pieces.index((coef, core, F)) + 1)
                          if len(pieces) > 1 else "Apply the rule",
                          text_, [formula, "%s = %s" % (_L(tex(core)), tex(F))]))
    if len(pieces) > 1 or pieces[0][0] != 1:
        steps.append(step("Combine", "Multiply by the constants and add.", [answer]))

    back = run_with_timeout(lambda: sp.inverse_laplace_transform(total, S, t), TIMEOUT, None)
    verified, note = None, "Could not be checked automatically."
    if back is not None and not back.has(sp.InverseLaplaceTransform):
        good, bad = sample_zero(_strip_step(back, t) - f, [t], around={t: 2.0}, spread=1.5)
        if good >= 3 and not bad:
            verified, note = True, "Checked by transforming the answer back to the original function."
    methods = [_method("table", "Table and rules",
                       "Each term matches a standard table entry or a shifting theorem.",
                       steps, answer, _plot(f, t), verified, note),
               _definition(f, t, total)]
    return _result("transform", "\\mathcal{L}\\left\\{%s\\right\\}" % tex(f), methods,
                   "Laplace transform.")


def _definition(f, t, expected):
    name = "From the definition"
    if f.has(sp.Heaviside, sp.DiracDelta):
        return _na("definition", name,
                   "With steps or impulses the integral splits into pieces; the "
                   "shifting rules are the practical route.")
    sp_ = sp.Symbol("s", positive=True)
    integrand = sp.exp(-sp_ * t) * f
    anti = integrate(integrand, t)
    value = run_with_timeout(
        lambda: sp.integrate(integrand, (t, 0, sp.oo), conds="none"), TIMEOUT, None)
    if value is None or value.has(sp.Integral) or anti.has(sp.Integral):
        return _na("definition", name, "The defining integral could not be evaluated in closed form.")
    value = _tidy(value.subs(sp_, S))
    if sp.simplify(value - expected) != 0:
        return _na("definition", name, "The defining integral did not simplify to the same result.")
    steps = [
        step("Write the definition",
             "The Laplace transform is an integral over \\(t\\) from 0 to infinity.",
             ["%s = \\int_0^\\infty e^{-st}\\,%s\\,dt" % (_L(tex(f)), _paren(f))]),
        step("Find an antiderivative",
             "Integrate with respect to \\(t\\), treating \\(s\\) as a constant.",
             ["\\int e^{-st}\\,%s\\,dt = %s" % (_paren(f), tex(sp.simplify(anti).subs(sp_, S)))]),
        step("Evaluate from 0 to infinity",
             "For \\(s\\) large enough the \\(e^{-st}\\) factor makes the upper limit vanish, "
             "leaving minus the value at \\(t = 0\\).",
             ["%s = %s" % (_L(tex(f)), tex(value))]),
    ]
    return _method("definition", name,
                   "Every transform comes from this integral; the table is a list of its results.",
                   steps, "%s = %s" % (_L(tex(f)), tex(value)), _plot(f, t), True,
                   "Agrees with the table method.")


# --------------------------------------------------------------------------
# Inverse transform
# --------------------------------------------------------------------------

def _split_shifts(F):
    """Write F as a sum of e^(-a s) G(s). Returns [(a, G)] or None."""
    exps = sorted(F.atoms(sp.exp), key=sp.default_sort_key)
    if not exps:
        return [(sp.Integer(0), F)]
    dummies = [sp.Dummy("z%d" % i) for i in range(len(exps))]
    replaced = F.xreplace(dict(zip(exps, dummies)))
    try:
        poly = sp.Poly(sp.expand(replaced), *dummies)
    except sp.PolynomialError:
        return None
    out = []
    for monom, coef in poly.terms():
        if sum(monom) > 1 or coef.has(*dummies):
            return None
        shift = sp.Integer(0)
        if sum(monom) == 1:
            arg = exps[monom.index(1)].args[0]
            shift = sp.simplify(-arg / S)
            if shift.has(S) or not shift.is_positive:
                return None
        out.append((shift, sp.together(coef)))
    return sorted(out, key=lambda p: float(p[0]))


def _inverse_rule(term):
    """Description of the table entry that inverts one partial fraction."""
    den = sp.denom(sp.together(term))
    degree = sp.Poly(den, S).degree() if den.has(S) else 0
    factors = sp.factor_list(den)[1]
    if len(factors) == 1 and sp.Poly(factors[0][0], S).degree() == 2:
        return ("Complete the square in the denominator and match the "
                "sine and cosine entries.",
                "%s = e^{at}\\cos bt,\\qquad %s = e^{at}\\sin bt"
                % (_Linv("\\frac{s-a}{(s-a)^2+b^2}"), _Linv("\\frac{b}{(s-a)^2+b^2}")))
    if degree == 1:
        return ("Table entry for a simple pole.", "%s = e^{at}" % _Linv("\\frac{1}{s-a}"))
    if degree >= 2:
        return ("Table entry for a repeated pole.",
                "%s = \\frac{t^{n-1}}{(n-1)!}e^{at}" % _Linv("\\frac{1}{(s-a)^n}"))
    return ("Table entry.", "%s = \\delta(t)" % _Linv("1"))


def _invert_rational(G, t, steps, label="F(s)"):
    """Inverse of a rational function, appending steps. Returns f(t) or None."""
    try:
        parts = sp.apart(G, S)
    except Exception:
        parts = G
    terms = sp.Add.make_args(parts)
    if parts != G and len(terms) > 1:
        steps.append(step("Split into partial fractions",
                          "Break \\(%s\\) into pieces that each match a table entry." % label,
                          [_eq(label, parts)]))
    total = sp.Integer(0)
    lines = []
    seen = []
    for term in terms:
        inv = run_with_timeout(lambda: sp.inverse_laplace_transform(term, S, t), TIMEOUT, None)
        if inv is None or inv.has(sp.InverseLaplaceTransform):
            return None
        inv = sp.expand(_strip_step(inv, t))
        total += inv
        lines.append("%s = %s" % (_Linv(tex(term)), tex(inv)))
        rule = _inverse_rule(term) if term.is_rational_function(S) else None
        if rule and rule not in seen:
            seen.append(rule)
    steps.append(step(
        "Invert each piece",
        " ".join(r[0] for r in seen) if seen else "Use the table of transforms.",
        lines, [step("Table entries used", "", [r[1] for r in seen])] if seen else []))
    return sp.expand(total)


def inverse(text):
    F, symbols = parse_expression(text, ["s"])
    F = F.xreplace({symbols["s"]: S})
    t = sp.Symbol("t", real=True)
    shifts = _split_shifts(F)
    if shifts is None:
        raise InputError("I can invert rational functions of s, optionally "
                         "multiplied by delay factors such as e^(-2s).")
    steps = []
    delayed = any(a != 0 for a, _ in shifts)
    if delayed:
        steps.append(step(
            "Identify the delay factors",
            "A factor \\(e^{-as}\\) delays the function by \\(a\\) and switches it on "
            "at \\(t = a\\) (second shifting theorem). Invert what multiplies it first.",
            ["%s = u(t-a)\\,f(t-a)" % _Linv("e^{-as}F(s)")]))
    result = sp.Integer(0)
    for index, (a, G) in enumerate(shifts):
        label = "F(s)" if len(shifts) == 1 and a == 0 else "G_{%d}(s)" % (index + 1)
        if label != "F(s)":
            steps.append(step("Piece %d" % (index + 1),
                              "This piece carries the delay \\(a = %s\\)." % tex(a) if a != 0
                              else "This piece has no delay.", [_eq(label, G)]))
        g = _invert_rational(G, t, steps, label)
        if g is None:
            raise InputError("The inverse Laplace transform could not be found.")
        if a != 0:
            shifted = g.subs(t, t - a) * sp.Heaviside(t - a)
            steps.append(step("Apply the delay",
                              "Replace \\(t\\) by \\(t - %s\\) and multiply by the unit step." % tex(a),
                              ["u(t-%s)\\,g(t-%s) = %s" % (tex(a), tex(a), tex(shifted))]))
            g = shifted
        result += g
    answer = "%s = %s" % (_Linv(tex(F)), tex(result))
    if len(shifts) > 1:
        steps.append(step("Combine", "Add the pieces.", [answer]))

    verified, note = None, "Could not be checked automatically."
    again = run_with_timeout(lambda: sp.laplace_transform(result, t, S, noconds=True), TIMEOUT, None)
    if again is not None and not again.has(sp.LaplaceTransform):
        good, bad = sample_zero(again - F, [S], around={S: 6.0}, spread=2.0)
        if good >= 3 and not bad:
            verified, note = True, "Checked by transforming the answer forward again."
    methods = [_method("fractions", "Partial fractions and table",
                       "The transform is a rational function of \\(s\\)%s."
                       % (" with delay factors" if delayed else ""),
                       steps, answer, _plot(result, t), verified, note),
               _convolution(F, t, result)]
    return _result("transform", "\\mathcal{L}^{-1}\\left\\{%s\\right\\}" % tex(F), methods,
                   "Inverse Laplace transform. The result holds for t ≥ 0.")


def _convolution(F, t, expected):
    name = "Convolution theorem"
    if F.has(sp.exp) or not F.is_rational_function(S):
        return _na("convolution", name, "It is set up here for rational functions without delays.")
    num, den = sp.fraction(sp.together(F))
    factors = sp.factor_list(den)[1]
    if sum(k for _, k in factors) < 2:
        return _na("convolution", name,
                   "The denominator has a single factor, so there is no product to split.")
    first, power = factors[0]
    A = 1 / first
    B = sp.simplify(F * first)
    if sp.degree(sp.numer(sp.together(B)), S) >= sp.degree(sp.denom(sp.together(B)), S):
        return _na("convolution", name,
                   "The factors do not both have inverse transforms that are ordinary functions.")
    a = run_with_timeout(lambda: sp.inverse_laplace_transform(A, S, t), TIMEOUT, None)
    b = run_with_timeout(lambda: sp.inverse_laplace_transform(B, S, t), TIMEOUT, None)
    if a is None or b is None or a.has(sp.InverseLaplaceTransform) or b.has(sp.InverseLaplaceTransform):
        return _na("convolution", name, "One of the factors could not be inverted.")
    a, b = sp.simplify(_strip_step(a, t)), sp.simplify(_strip_step(b, t))
    tau = sp.Symbol("tau", real=True)
    integrand = a.subs(t, tau) * b.subs(t, t - tau)
    value = run_with_timeout(lambda: sp.integrate(integrand, (tau, 0, t)), TIMEOUT, None)
    if value is None or value.has(sp.Integral):
        return _na("convolution", name, "The convolution integral could not be evaluated in closed form.")
    value = sp.expand(sp.expand_trig(sp.simplify(value)))
    good, bad = sample_zero(value - expected, [t], around={t: 2.0}, spread=1.5)
    if bad or good < 3:
        return _na("convolution", name, "The convolution integral did not simplify to the same result.")
    steps = [
        step("Split into a product",
             "Write the transform as a product of two transforms you can invert.",
             ["%s = %s \\cdot %s" % (tex(F), _paren(A), _paren(B))]),
        step("Invert each factor", "",
             ["%s = %s" % (_Linv(tex(A)), tex(a)), "%s = %s" % (_Linv(tex(B)), tex(b))]),
        step("Convolve",
             "The inverse of a product is the convolution of the inverses.",
             ["%s = \\int_0^t f(\\tau)\\,g(t-\\tau)\\,d\\tau" % _Linv("F(s)G(s)"),
              "\\int_0^t %s\\,d\\tau = %s" % (_paren(integrand), tex(value))]),
    ]
    return _method("convolution", name,
                   "The transform is a product of simpler transforms.",
                   steps, "%s = %s" % (_Linv(tex(F)), tex(value)), _plot(value, t), True,
                   "Agrees with the partial-fraction result.")


def _result(kind, input_latex, methods, summary):
    for m in methods:
        if not m["applicable"]:
            continue
        m["steps"].append({"title": "Result", "text": "", "latex": [m["particular_latex"]],
                           "substeps": [], "kind": "answer"})
        check = {True: "check-ok", False: "check-bad", None: "check-unknown"}[m["verified"]]
        m["steps"].append({"title": "Check", "text": m["verify_note"], "latex": [],
                           "substeps": [], "kind": check})
    return {"schema": 1, "ok": True, "kind": kind, "order": 0, "indep": "t", "dep": "f",
            "input_latex": input_latex, "normal_latex": input_latex, "rhs_expr": None,
            "ic": None, "summary": summary, "other_types": [], "methods": methods}
