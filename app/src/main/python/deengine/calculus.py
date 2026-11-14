"""Calculus tools: integrals with the technique shown, definite and improper
integrals, partial fractions, and derivatives with the rules named."""
import sympy as sp
from sympy.integrals import manualintegrate as mi

from .integrals import _TABLE
from .parsing import InputError, parse_expression, parse_number
from .toolkit import method, not_applicable, result, step
from .util import plot_expr, run_with_timeout, sample_zero, tex

OPERATIONS = ("integral", "definite", "fractions", "derivative")
MAX_STEPS = 14


class _Prob:
    """Just enough of a Problem for util.plot_expr."""
    def __init__(self, x):
        self.x = x
        self.y = sp.Symbol("y", real=True)
        self.C = self.C1 = self.C2 = sp.Symbol("C", real=True)


def _paren(expr):
    return "\\left(%s\\right)" % tex(expr) if expr.is_Add or expr.could_extract_minus_sign() else tex(expr)


def _int(expr, var):
    return "\\int %s\\,d%s" % ("\\left(%s\\right)" % tex(expr) if expr.is_Add else tex(expr), tex(var))


class _Stuck(Exception):
    pass


def _value(rule):
    try:
        return rule.eval()
    except Exception:
        raise _Stuck()


def _derive(rule, out):
    """Append (title, text, latex) for every stage of the rule tree, in reading order."""
    if len(out) >= MAX_STEPS:
        return
    name = type(rule).__name__
    var = rule.variable
    here = _int(rule.integrand, var)
    if name == "DontKnowRule":
        raise _Stuck()
    if name == "AlternativeRule":
        _derive(rule.alternatives[0], out)
    elif name == "ConstantTimesRule":
        if rule.constant not in (1, -1):
            out.append(("Constant multiple", "Take the constant outside the integral.",
                        ["%s = %s\\,%s" % (here, _paren(rule.constant), _int(rule.other, var))]))
        _derive(rule.substep, out)
    elif name == "AddRule":
        out.append(("Term by term", "The integral of a sum is the sum of the integrals.",
                    ["%s = %s" % (here, " + ".join(_int(s.integrand, var) for s in rule.substeps))]))
        for sub in rule.substeps:
            _derive(sub, out)
    elif name == "URule":
        u = rule.u_var
        du = sp.diff(rule.u_func, var)
        inner = _int(rule.substep.integrand, u).replace(tex(u), "u")
        out.append(("Substitution", "Let \\(u\\) be the inner expression; its derivative appears in the integrand.",
                    ["u = %s,\\qquad du = %s\\,d%s" % (tex(rule.u_func), _paren(du), tex(var)),
                     "%s = %s" % (here, inner)]))
        _derive(rule.substep, out)
        out.append(("Substitute back", "Replace \\(u\\) by the expression in \\(%s\\)." % tex(var),
                    ["%s = %s" % (here, tex(_value(rule)))]))
    elif name == "PartsRule":
        v = _value(rule.v_step) if getattr(rule, "v_step", None) is not None else sp.integrate(rule.dv, var)
        du = sp.diff(rule.u, var)
        out.append(("Integration by parts", "Use \\(\\int u\\,dv = uv - \\int v\\,du\\).",
                    ["u = %s,\\qquad dv = %s\\,d%s" % (tex(rule.u), tex(rule.dv), tex(var)),
                     "du = %s\\,d%s,\\qquad v = %s" % (tex(du), tex(var), tex(v)),
                     "%s = %s - %s" % (here, tex(rule.u * v), _int(sp.simplify(v * du), var))]))
        if rule.second_step is not None:
            _derive(rule.second_step, out)
    elif name == "CyclicPartsRule":
        out.append(("Integration by parts, twice",
                    "Integrating by parts twice brings back the original integral. Call it "
                    "\\(I\\), move it to one side and solve for it.",
                    ["I = %s = %s" % (here, tex(_value(rule)))]))
    elif name in ("RewriteRule", "CompleteSquareRule"):
        rational = rule.integrand.is_rational_function(var)
        if name == "CompleteSquareRule":
            title, text = "Complete the square", "Complete the square in the quadratic."
        elif rational and not rule.integrand.is_polynomial(var) and sp.expand(rule.rewritten).is_Add:
            title, text = "Partial fractions", "Split the fraction into simpler ones."
        else:
            title, text = "Rewrite the integrand", "Write the integrand in a form that can be integrated."
        out.append((title, text, ["%s = %s" % (tex(rule.integrand), tex(rule.rewritten))]))
        _derive(rule.substep, out)
    elif name == "TrigSubstitutionRule":
        out.append(("Trigonometric substitution",
                    "A square root of a quadratic disappears under a trigonometric substitution.",
                    ["%s = %s" % (tex(var), tex(rule.func)),
                     "%s = %s" % (here, tex(_value(rule)))]))
    elif name == "PiecewiseRule":
        raise _Stuck()
    else:
        text = _TABLE.get(name) or "A standard integral."
        out.append(("Standard integral", text, ["%s = %s" % (here, tex(_value(rule)))]))


def _technique(rule):
    """Short name of the first real technique in a rule tree."""
    name = type(rule).__name__
    if name == "AlternativeRule":
        return _technique(rule.alternatives[0])
    if name == "ConstantTimesRule":
        return _technique(rule.substep)
    if name == "RewriteRule" and rule.integrand.is_rational_function(rule.variable) \
            and not rule.integrand.is_polynomial(rule.variable) and sp.expand(rule.rewritten).is_Add:
        return "Partial fractions"
    return {"URule": "Substitution", "PartsRule": "Integration by parts",
            "CyclicPartsRule": "Integration by parts", "RewriteRule": "Rewriting",
            "CompleteSquareRule": "Completing the square", "AddRule": "Term by term",
            "TrigSubstitutionRule": "Trigonometric substitution"}.get(name, "Standard integrals")


def _steps_for(rule):
    out = []
    _derive(rule, out)
    return [step(title, text, latex) for title, text, latex in out]


def _check_antiderivative(F, f, x):
    good, bad = sample_zero(sp.diff(F, x) - f, [x])
    if good >= 3 and not bad:
        return True, "Differentiating the answer gives back the integrand."
    if bad:
        return False, "Differentiating the answer did not give back the integrand."
    return None, "Could not be checked automatically."


def _integral(text):
    f, symbols = parse_expression(text, ["x"])
    x = symbols["x"]
    shown = "\\int %s\\,dx" % _paren(f) if f.is_Add else "\\int %s\\,dx" % tex(f)
    rule = run_with_timeout(lambda: mi.integral_steps(f, x), 10, None)
    routes = []
    if rule is not None:
        routes = list(rule.alternatives[:3]) if type(rule).__name__ == "AlternativeRule" else [rule]
    methods, seen = [], set()
    for route in routes:
        try:
            steps = _steps_for(route)
            raw = _value(route)
            F = sp.simplify(raw)
            # simplify() likes to merge sin and cos into one shifted wave; keep the plain form.
            if any(isinstance(a, (sp.sin, sp.cos)) and a.args[0].is_Add and a.args[0].has(sp.pi)
                   for a in sp.preorder_traversal(F)):
                F = sp.expand(raw)
        except _Stuck:
            continue
        if F.has(sp.Integral) or F.has(sp.Piecewise):
            continue
        name = _technique(route)
        if name in seen:
            continue
        seen.add(name)
        ok, note = _check_antiderivative(F, f, x)
        if ok is False:
            continue
        methods.append(method("route%d" % len(methods), name,
                              "This route uses %s." % name.lower(), steps,
                              "%s = %s + C" % (shown, tex(F)),
                              plot={"particular_expr": plot_expr(f, _Prob(x)),
                                    "antiderivative": plot_expr(F, _Prob(x))}, verified=ok, note=note))
    if not methods:
        F = run_with_timeout(lambda: sp.integrate(f, x), 8, None)
        if F is None or F.has(sp.Integral):
            raise InputError("This integral has no closed form in elementary functions that I can find.")
        ok, note = _check_antiderivative(F, f, x)
        methods.append(method("cas", "Computer algebra",
                              "No textbook route was found, so the result comes from a general algorithm.",
                              [step("Result of the algorithm",
                                    "The steps for this one are not available.", [])],
                              "%s = %s + C" % (shown, tex(F)),
                              plot={"particular_expr": plot_expr(f, _Prob(x)),
                                    "antiderivative": plot_expr(F, _Prob(x))}, verified=ok, note=note))
    return result("calculus", "Integral", shown, "Indefinite integral.", methods)


def _limit(text, label):
    cleaned = (text or "").strip().replace("inf", "oo")
    if cleaned in ("oo", "+oo"):
        return sp.oo
    if cleaned == "-oo":
        return -sp.oo
    if not cleaned:
        raise InputError("Enter the %s limit of integration (a number, pi, oo or -oo)." % label)
    return parse_number(cleaned, "The %s limit" % label)


def _definite(text, a_text, b_text):
    f, symbols = parse_expression(text, ["x"])
    x = symbols["x"]
    a, b = _limit(a_text, "lower"), _limit(b_text, "upper")
    shown = "\\int_{%s}^{%s} %s\\,dx" % (tex(a), tex(b), _paren(f) if f.is_Add else tex(f))
    F = run_with_timeout(lambda: sp.integrate(f, x), 8, None)
    if F is None or F.has(sp.Integral):
        raise InputError("I could not find an antiderivative for this integrand.")
    F = sp.simplify(F)
    from .integrals import explain
    steps = [step("Find an antiderivative", "", ["F(x) = %s" % tex(F)],
                  explain(f, x, "How"))]
    improper = a in (sp.oo, -sp.oo) or b in (sp.oo, -sp.oo)

    def edge(value, side):
        if value in (sp.oo, -sp.oo):
            return sp.limit(F, x, value)
        direct = F.subs(x, value)
        if direct.has(sp.zoo, sp.nan) or not direct.is_finite:
            return sp.limit(F, x, value, "+" if side == "lower" else "-")
        return direct

    try:
        top, bottom = edge(b, "upper"), edge(a, "lower")
        value = sp.simplify(top - bottom)
    except Exception:
        raise InputError("The limits of this integral could not be evaluated.")
    singular = any(v in (sp.oo, -sp.oo) for v in (top, bottom)) or value.has(sp.oo, sp.zoo, sp.nan)
    direct = run_with_timeout(lambda: sp.integrate(f, (x, a, b)), 8, None)
    if direct is not None and (direct.has(sp.oo, sp.zoo, sp.nan) or direct.has(sp.Integral)):
        singular = singular or direct.has(sp.oo, sp.zoo)
    if improper:
        steps.append(step("Write the improper integral as a limit",
                          "An infinite limit of integration means a limit of ordinary integrals.",
                          ["%s = \\lim F(x)\\Big|_{%s}^{%s}" % (shown, tex(a), tex(b))]))
    if singular:
        steps.append(step("Evaluate", "The antiderivative does not approach a finite value at a limit "
                                      "of integration, so the integral diverges.",
                          ["F \\to %s \\ (x \\to %s),\\qquad F \\to %s \\ (x \\to %s)"
                           % (tex(top), tex(b), tex(bottom), tex(a))]))
        return result("calculus", "Definite integral", shown, "The integral diverges.",
                      [method("ftc", "Fundamental theorem of calculus", "An antiderivative is available.",
                              steps, "%s \\ \\text{diverges}" % shown,
                              plot={"particular_expr": plot_expr(f, _Prob(x))})])
    steps.append(step("Evaluate at the limits",
                      "Subtract the value at the lower limit from the value at the upper limit.",
                      ["F(%s) - F(%s) = %s - %s = %s" % (tex(b), tex(a), _paren(top), _paren(bottom), tex(value))]))
    ok, note = None, "Could not be checked numerically."
    try:
        import mpmath
        fn = sp.lambdify(x, f, "mpmath")
        lo = -mpmath.inf if a == -sp.oo else float(a)
        hi = mpmath.inf if b == sp.oo else float(b)
        numeric = mpmath.quad(fn, [lo, hi])
        ok = abs(complex(numeric) - complex(sp.N(value))) < 1e-6 * (1 + abs(complex(sp.N(value))))
        note = ("Agrees with numerical integration (%s)." % sp.sstr(sp.N(value, 8)) if ok
                else "Numerical integration gives a different value; treat this answer with caution.")
    except Exception:
        pass
    return result("calculus", "Definite integral", shown, "Value: %s." % sp.sstr(value),
                  [method("ftc", "Fundamental theorem of calculus", "An antiderivative is available.",
                          steps, "%s = %s" % (shown, tex(value)),
                          plot={"particular_expr": plot_expr(f, _Prob(x))}, verified=ok, note=note)])


def _fractions(text):
    f, symbols = parse_expression(text, ["x"])
    x = symbols["x"]
    if not f.is_rational_function(x):
        raise InputError("Partial fractions apply to a ratio of polynomials in x.")
    num, den = sp.fraction(sp.cancel(sp.together(f)))
    if not den.has(x):
        raise InputError("The denominator has no x in it, so there is nothing to decompose.")
    steps = []
    poly_part = sp.Integer(0)
    if sp.degree(num, x) >= sp.degree(den, x):
        q, r = sp.div(sp.Poly(num, x), sp.Poly(den, x))
        poly_part, num = q.as_expr(), r.as_expr()
        steps.append(step("Divide first",
                          "The numerator's degree is not smaller than the denominator's, so divide.",
                          ["%s = %s + \\frac{%s}{%s}" % (tex(f), tex(poly_part), tex(num), tex(den))]))
    lead, factors = sp.factor_list(den, x)
    steps.append(step("Factor the denominator", "",
                      ["%s = %s" % (tex(sp.expand(den)), tex(sp.factor(den)))]))
    names = iter(sp.symbols("A B D E F G H K L M N P Q R"))
    terms, unknowns, simple = [], [], True
    for factor, power in factors:
        degree = sp.degree(factor, x)
        if degree > 2:
            raise InputError("The denominator has an irreducible factor of degree %d; this tool handles "
                             "linear and quadratic factors." % degree)
        if degree == 2 or power > 1:
            simple = False
        for k in range(1, power + 1):
            if degree == 1:
                c = next(names)
                unknowns.append(c)
                terms.append(c / factor**k)
            else:
                c, d = next(names), next(names)
                unknowns += [c, d]
                terms.append((c * x + d) / factor**k)
    template = sp.Add(*terms, evaluate=False)
    steps.append(step("Set up the template",
                      "Each linear factor gets a constant on top, each irreducible quadratic a linear "
                      "numerator, and a repeated factor gets one term for each power.",
                      ["\\frac{%s}{%s} = %s" % (tex(num), tex(sp.factor(den)),
                                                " + ".join(tex(t) for t in terms))]))
    cleared = sp.expand(sp.cancel(sum(terms) * den))
    equations = sp.Poly(cleared - sp.expand(num), x).all_coeffs()
    solved = sp.solve(equations, unknowns, dict=True)
    if not solved:
        raise InputError("The partial fraction coefficients could not be determined.")
    values = solved[0]
    text_ = "Multiply both sides by the denominator and match the coefficients of each power of \\(x\\)."
    if simple:
        text_ += (" With only distinct linear factors there is a shortcut (the cover-up method): "
                  "to get the constant over \\(x - a\\), cover that factor and put \\(x = a\\) in what is left.")
    steps.append(step("Solve for the constants", text_,
                      [",\\quad ".join("%s = %s" % (tex(c), tex(values.get(c, 0))) for c in unknowns)],
                      [step("Equations", "", ["%s = %s" % (tex(sp.expand(num)), tex(sp.collect(cleared, x)))]
                            + ["%s = 0" % tex(e) for e in equations if e != 0])]))
    answer_terms = [t.subs(values) for t in terms if t.subs(values) != 0]
    answer = poly_part + sum(answer_terms)
    body = " + ".join(tex(t) for t in ([poly_part] if poly_part != 0 else []) + answer_terms).replace("+ -", "- ")
    ok = sp.simplify(answer - f) == 0
    return result("calculus", "Partial fractions", tex(f), "Partial fraction decomposition.",
                  [method("fractions", "Undetermined coefficients",
                          "The expression is a ratio of polynomials.", steps,
                          "%s = %s" % (tex(f), body), verified=ok,
                          note="Putting the pieces back over a common denominator gives the original fraction.")])


def _derivative(text):
    f, symbols = parse_expression(text, ["x"])
    x = symbols["x"]
    d = sp.diff(f, x)
    steps = []
    num, den = sp.fraction(f)
    if f.is_Add:
        steps.append(step("Sum rule", "Differentiate each term separately.",
                          ["\\frac{d}{dx}\\left[%s\\right] = %s" % (
                              tex(f), " + ".join("\\frac{d}{dx}\\left[%s\\right]" % tex(t) for t in f.args))]))
    elif den.has(x) and den != 1:
        steps.append(step("Quotient rule",
                          "\\(\\left(\\dfrac{u}{v}\\right)' = \\dfrac{u'v - uv'}{v^2}\\).",
                          ["u = %s,\\qquad v = %s" % (tex(num), tex(den)),
                           "u' = %s,\\qquad v' = %s" % (tex(sp.diff(num, x)), tex(sp.diff(den, x)))]))
    elif f.is_Mul and sum(1 for a in f.args if a.has(x)) >= 2:
        parts = [a for a in f.args if a.has(x)]
        u, v = parts[0], sp.Mul(*parts[1:])
        steps.append(step("Product rule", "\\((uv)' = u'v + uv'\\).",
                          ["u = %s,\\qquad v = %s" % (tex(u), tex(v)),
                           "u' = %s,\\qquad v' = %s" % (tex(sp.diff(u, x)), tex(sp.diff(v, x)))]))
    core = [a for a in sp.preorder_traversal(f)
            if (isinstance(a, sp.Function) or a.is_Pow) and a.has(x)
            and any(arg.has(x) and arg != x for arg in a.args)]
    if core:
        outer = core[0]
        inner = [arg for arg in outer.args if arg.has(x) and arg != x][0]
        steps.append(step("Chain rule",
                          "Differentiate the outer function, keep the inner one, then multiply by "
                          "the derivative of the inner function.",
                          ["\\text{outer: } %s,\\qquad \\text{inner: } %s" % (tex(outer), tex(inner)),
                           "\\frac{d}{dx}\\left[%s\\right] = %s" % (tex(outer), tex(sp.diff(outer, x)))]))
    if not steps:
        steps.append(step("Basic rule", "Use the table of derivatives.", []))
    simplified = sp.simplify(d)
    steps.append(step("Combine and simplify", "", ["%s = %s" % (tex(d), tex(simplified))]
                      if simplified != d else [tex(d)]))
    h = sp.Rational(1, 100000)
    good, bad = sample_zero((f.subs(x, x + h) - f.subs(x, x - h)) / (2 * h) - simplified, [x], tol=1e-4)
    ok = True if good >= 3 and not bad else (False if bad else None)
    return result("calculus", "Derivative", "\\frac{d}{dx}\\left[%s\\right]" % tex(f), "Derivative.",
                  [method("rules", "Differentiation rules", "Built from the sum, product, quotient and chain rules.",
                          steps, "\\frac{d}{dx}\\left[%s\\right] = %s" % (tex(f), tex(simplified)),
                          plot={"particular_expr": plot_expr(simplified, _Prob(x))}, verified=ok,
                          note="Agrees with a numerical difference quotient at sample points." if ok
                          else "Could not be checked numerically.")])


def solve(operation, text, a_text="", b_text=""):
    if operation == "integral":
        return _integral(text)
    if operation == "definite":
        return _definite(text, a_text, b_text)
    if operation == "fractions":
        return _fractions(text)
    if operation == "derivative":
        return _derivative(text)
    raise InputError("Unknown calculus operation.")


def preview(operation, text):
    f, _ = parse_expression(text, ["x"])
    if operation in ("integral", "definite"):
        return "\\int %s\\,dx" % ("\\left(%s\\right)" % tex(f) if f.is_Add else tex(f))
    if operation == "derivative":
        return "\\frac{d}{dx}\\left[%s\\right]" % tex(f)
    return tex(f)
