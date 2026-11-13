"""Explain how an integral is done: substitution, parts, partial fractions.

SymPy's manualintegrate produces a tree of textbook rules. This module turns
that tree into a few short steps. It never raises: when the tree contains a
rule that is not described here, or takes too long, no explanation is given.
"""
import sympy as sp
from sympy.integrals import manualintegrate as mi

from .util import run_with_timeout, tex

MAX_STEPS = 5
_TABLE = {
    "PowerRule": "Power rule: \\(\\int x^n\\,dx = \\dfrac{x^{n+1}}{n+1}\\).",
    "ExpRule": "Exponential rule: \\(\\int e^{ax}\\,dx = \\dfrac{e^{ax}}{a}\\).",
    "ReciprocalRule": "Reciprocal rule: \\(\\int \\dfrac{1}{x}\\,dx = \\ln|x|\\).",
    "SinRule": "Standard integral: \\(\\int \\sin x\\,dx = -\\cos x\\).",
    "CosRule": "Standard integral: \\(\\int \\cos x\\,dx = \\sin x\\).",
    "Sec2Rule": "Standard integral: \\(\\int \\sec^2 x\\,dx = \\tan x\\).",
    "Csc2Rule": "Standard integral: \\(\\int \\csc^2 x\\,dx = -\\cot x\\).",
    "SecTanRule": "Standard integral: \\(\\int \\sec x\\tan x\\,dx = \\sec x\\).",
    "CscCotRule": "Standard integral: \\(\\int \\csc x\\cot x\\,dx = -\\csc x\\).",
    "SinhRule": "Standard integral: \\(\\int \\sinh x\\,dx = \\cosh x\\).",
    "CoshRule": "Standard integral: \\(\\int \\cosh x\\,dx = \\sinh x\\).",
    "ArctanRule": "Arctangent form: \\(\\int \\dfrac{dx}{x^2 + a^2} = \\dfrac{1}{a}\\arctan\\dfrac{x}{a}\\).",
    "ArcsinRule": "Arcsine form: \\(\\int \\dfrac{dx}{\\sqrt{a^2 - x^2}} = \\arcsin\\dfrac{x}{a}\\).",
    "ConstantRule": None,
}


class _Unknown(Exception):
    pass


def _int(expr, var):
    body = "\\left(%s\\right)" % tex(expr) if expr.is_Add else tex(expr)
    return "\\int %s\\,d%s" % (body, tex(var))


def _walk(rule, out, top=False):
    if len(out) >= MAX_STEPS:
        return
    name = type(rule).__name__
    var = rule.variable
    if name == "AlternativeRule":
        _walk(rule.alternatives[0], out, top)
    elif name == "ConstantTimesRule":
        if rule.constant not in (1, -1):
            out.append(("Constant multiple", "Take the constant outside the integral.",
                        ["%s = %s\\,%s" % (_int(rule.integrand, var),
                                           "\\left(%s\\right)" % tex(rule.constant)
                                           if rule.constant.could_extract_minus_sign() else tex(rule.constant),
                                           _int(rule.other, var))]))
        _walk(rule.substep, out)
    elif name == "AddRule":
        out.append(("Term by term", "Integrate each term separately.", []))
        for sub in rule.substeps:
            if type(sub).__name__ not in _TABLE and type(sub).__name__ != "ConstantTimesRule":
                _walk(sub, out)
            elif type(sub).__name__ == "ConstantTimesRule" and type(sub.substep).__name__ not in _TABLE:
                _walk(sub.substep, out)
    elif name == "URule":
        u = rule.u_var
        du = sp.diff(rule.u_func, var)
        out.append(("Substitution",
                    "Let \\(u\\) be the inner expression; its derivative appears in the integrand.",
                    ["u = %s,\\qquad du = %s\\,d%s" % (tex(rule.u_func), "\\left(%s\\right)" % tex(du)
                                                         if du.is_Add else tex(du), tex(var)),
                     "%s = %s" % (_int(rule.integrand, var), _int(rule.substep.integrand, u).replace(
                         tex(u), "u"))]))
        inner = rule.substep
        if type(inner).__name__ == "ConstantTimesRule":
            inner = inner.substep
        # A substitution inside a substitution reads badly; stop at one level.
        if type(inner).__name__ not in _TABLE and type(inner).__name__ != "URule":
            _walk(inner, out)
    elif name == "PartsRule":
        v = sp.integrate(rule.dv, var)
        du = sp.diff(rule.u, var)
        out.append(("Integration by parts",
                    "Use \\(\\int u\\,dv = uv - \\int v\\,du\\).",
                    ["u = %s,\\qquad dv = %s\\,d%s" % (tex(rule.u), tex(rule.dv), tex(var)),
                     "du = %s\\,d%s,\\qquad v = %s" % (tex(du), tex(var), tex(v)),
                     "%s = %s - %s" % (_int(rule.integrand, var), tex(rule.u * v),
                                       _int(sp.simplify(v * du), var))]))
        second = rule.second_step
        if type(second).__name__ == "ConstantTimesRule":
            second = second.substep
        if type(second).__name__ not in _TABLE:
            _walk(second, out)
    elif name == "CyclicPartsRule":
        out.append(("Integration by parts, twice",
                    "Integrating by parts twice brings back the original integral. "
                    "Treat it as an unknown and solve the resulting equation for it.", []))
    elif name == "RewriteRule":
        rational = rule.integrand.is_rational_function(var)
        out.append(("Partial fractions" if rational and rule.rewritten.is_Add else "Rewrite the integrand",
                    "Split the fraction into simpler ones." if rational and rule.rewritten.is_Add
                    else "Write the integrand in a form that can be integrated directly.",
                    ["%s = %s" % (tex(rule.integrand), tex(rule.rewritten))]))
        _walk(rule.substep, out)
    elif name == "CompleteSquareRule":
        out.append(("Complete the square", "Complete the square in the quadratic.",
                    ["%s = %s" % (tex(rule.integrand), tex(rule.rewritten))]))
        _walk(rule.substep, out)
    elif name in _TABLE:
        if top and _TABLE[name]:
            out.append(("Rule used", _TABLE[name], []))
    else:
        raise _Unknown(name)


def explain(integrand, var, prefix):
    """Steps (title, text, latex) describing how to integrate, titled '<prefix>: ...'."""
    if integrand.is_number or not integrand.has(var) or sp.count_ops(integrand) > 30:
        return []

    # Rational functions: show the partial-fraction split, the standard route.
    try:
        if integrand.is_rational_function(var) and not integrand.is_polynomial(var):
            parts = sp.apart(integrand, var)
            if parts != integrand and parts.is_Add:
                return [{"title": "%s: partial fractions" % prefix,
                         "text": "Split the fraction into simpler ones and integrate each term.",
                         "latex": ["%s = %s" % (tex(integrand), tex(parts))], "substeps": []}]
    except Exception:
        pass

    def work():
        rule = mi.integral_steps(sp.expand(integrand), var)
        out = []
        _walk(rule, out, top=True)
        return out

    try:
        found = run_with_timeout(work, 1.5, [])
    except Exception:
        found = []
    return [{"title": "%s: %s" % (prefix, title.lower()), "text": text, "latex": latex, "substeps": []}
            for title, text, latex in found]
