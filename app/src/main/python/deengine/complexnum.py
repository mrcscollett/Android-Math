"""Complex numbers: arithmetic in rectangular form, modulus and argument,
polar and exponential forms, powers by De Moivre's theorem, and n-th roots."""
import sympy as sp

from .parsing import InputError, parse_expression
from .toolkit import method, not_applicable, result, step
from .util import tex

OPERATIONS = ("evaluate", "roots")


def parse(text):
    """An expression in which i is the imaginary unit."""
    expr, symbols = parse_expression(text, ["i"])
    value = expr.xreplace({symbols["i"]: sp.I})
    if value.free_symbols:
        raise InputError("A complex number here may contain only numbers, i, pi and e.")
    return value


def _split_quotient(text):
    """(numerator text, denominator text) when the input is one top-level division, else None.

    SymPy clears a complex denominator as soon as it reads one, so the
    division has to be recognised in the text to show that step.
    """
    depth, places = 0, []
    for index, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in "+-" and index > 0 and text[:index].strip():
            return None
        elif depth == 0 and ch == "/":
            places.append(index)
    if len(places) != 1:
        return None
    top, bottom = text[:places[0]].strip(), text[places[0] + 1:].strip()
    return (top, bottom) if top and bottom else None


def _rect(z):
    """(re, im) simplified, or raises if z is not a finite number."""
    re, im = sp.expand_complex(z).as_real_imag()
    re, im = sp.simplify(re), sp.simplify(im)
    if re.has(sp.I) or im.has(sp.I):
        # No tidy exact form (for example i^i): fall back to decimals.
        value = complex(sp.N(z))
        parts = []
        for part in (value.real, value.imag):
            parts.append(sp.Integer(0) if abs(part) < 1e-12 else sp.Float(round(part, 10)))
        return parts[0], parts[1]
    if re.has(sp.zoo, sp.nan, sp.oo) or im.has(sp.zoo, sp.nan, sp.oo):
        raise InputError("That expression is not defined (division by zero?).")
    return sp.nsimplify(re) if re.is_Float else re, sp.nsimplify(im) if im.is_Float else im


def _form(re, im):
    """LaTeX of re + im i without clutter."""
    if im == 0:
        return tex(re)
    imag = ("" if im == 1 else "-" if im == -1 else
            ("\\left(%s\\right)" % tex(im) if im.is_Add else tex(im))) + ("i" if im in (1, -1) else "\\,i")
    if re == 0:
        return imag
    if imag.startswith("-"):
        return "%s - %s" % (tex(re), imag[1:].lstrip())
    return "%s + %s" % (tex(re), imag)


def _polar(re, im):
    r = sp.simplify(sp.sqrt(re**2 + im**2))
    theta = sp.simplify(sp.atan2(im, re)) if r != 0 else sp.Integer(0)
    return r, theta


def _num(value):
    return float(sp.N(value))


def _evaluate(text):
    z = parse(text)
    re, im = _rect(z)
    steps = [step("Read the expression", "Remember that \\(i^2 = -1\\).", ["z = %s" % tex(z)])]
    # Division: multiply top and bottom by the conjugate of the denominator.
    quotient = _split_quotient(text)
    num, den = (parse(quotient[0]), parse(quotient[1])) if quotient else (z, sp.Integer(1))
    if den.has(sp.I):
        steps[0]["latex"] = ["z = \\frac{%s}{%s}" % (tex(num), tex(den))]
        conj = sp.conjugate(den)
        steps.append(step(
            "Multiply by the conjugate",
            "Multiplying the numerator and the denominator by the conjugate of the denominator "
            "makes the denominator real.",
            ["\\frac{%s}{%s}\\cdot\\frac{%s}{%s} = \\frac{%s}{%s}"
             % (tex(num), tex(den), tex(conj), tex(conj), tex(sp.expand(num * conj)), tex(sp.expand(den * conj)))]))
    steps.append(step("Rectangular form", "Collect the real and imaginary parts.",
                      ["z = %s" % _form(re, im), "\\operatorname{Re} z = %s,\\qquad \\operatorname{Im} z = %s"
                       % (tex(re), tex(im))]))
    r, theta = _polar(re, im)
    steps.append(step("Modulus and argument",
                      "The modulus is the distance from the origin and the argument is the angle "
                      "from the positive real axis (between \\(-\\pi\\) and \\(\\pi\\)).",
                      ["|z| = \\sqrt{%s^2 + %s^2} = %s" % (_wrap(re), _wrap(im), tex(r)),
                       "\\arg z = %s%s" % (tex(theta), "" if theta.is_number and not theta.has(sp.atan)
                                           else " \\approx %s" % ("%.4f" % _num(theta)))]))
    steps.append(step("Polar and exponential forms",
                      "Euler's formula \\(e^{i\\theta} = \\cos\\theta + i\\sin\\theta\\) links the two. "
                      "It is the same fact that turns complex roots \\(\\alpha \\pm \\beta i\\) of a "
                      "characteristic equation into \\(e^{\\alpha x}\\cos\\beta x\\) and \\(e^{\\alpha x}\\sin\\beta x\\).",
                      ["z = %s\\left(\\cos %s + i\\sin %s\\right) = %s\\,e^{%s i}"
                       % (tex(r), _wrap(theta), _wrap(theta), tex(r), _wrap(theta))]))
    answer = ["z = %s" % _form(re, im), "|z| = %s,\\quad \\arg z = %s" % (tex(r), tex(theta))]
    try:
        direct = complex(sp.N(z))
        ok = abs(direct - complex(_num(re), _num(im))) < 1e-9 * (1 + abs(direct))
    except Exception:
        ok = None
    plot = {"points": [[0.0, 0.0], [_num(re), _num(im)]]}
    methods = [method("rectangular", "Rectangular arithmetic", "Works for any expression.", steps,
                      answer, plot=plot, verified=ok,
                      note="Agrees with a direct numerical evaluation." if ok else "Could not be checked numerically.")]
    # A power of a complex number: De Moivre's theorem.
    if z.is_Pow and z.exp.is_Integer and z.base.has(sp.I):
        bre, bim = _rect(z.base)
        br, bt = _polar(bre, bim)
        k = z.exp
        pr, pt = sp.simplify(br**k), sp.simplify(k * bt)
        wre, wim = sp.simplify(pr * sp.cos(pt)), sp.simplify(pr * sp.sin(pt))
        psteps = [step("Write the base in polar form", "",
                       ["%s = %s\\left(\\cos %s + i\\sin %s\\right)" % (_form(bre, bim), tex(br), _wrap(bt), _wrap(bt))]),
                  step("Apply De Moivre's theorem",
                       "Raising to a power raises the modulus to that power and multiplies the argument by it.",
                       ["\\left[r(\\cos\\theta + i\\sin\\theta)\\right]^n = r^n(\\cos n\\theta + i\\sin n\\theta)",
                        "z = %s\\left(\\cos %s + i\\sin %s\\right)" % (tex(pr), _wrap(pt), _wrap(pt))]),
                  step("Convert back", "", ["z = %s" % _form(wre, wim)])]
        same = abs(complex(_num(wre), _num(wim)) - complex(_num(re), _num(im))) < 1e-9 * (1 + abs(_num(r)))
        methods.append(method("demoivre", "De Moivre's theorem", "The expression is a whole-number power.",
                              psteps, "z = %s" % _form(wre, wim), plot=plot, verified=same,
                              note="Agrees with the rectangular calculation."))
    else:
        methods.append(not_applicable("demoivre", "De Moivre's theorem",
                                      "It applies to a whole-number power of a complex number, such as (1 + i)^8."))
    return result("complex", "Complex number", "z = %s" % tex(z), "z = %s" % sp.sstr(re + im * sp.I), methods)


def _wrap(value):
    return "\\left(%s\\right)" % tex(value) if value.could_extract_minus_sign() or value.is_Add else tex(value)


def _roots(text, n_text):
    z = parse(text)
    try:
        n = int((n_text or "").strip())
    except ValueError:
        raise InputError("Enter which roots you want as a whole number n, such as 3 for cube roots.")
    if not 2 <= n <= 12:
        raise InputError("n must be between 2 and 12.")
    re, im = _rect(z)
    r, theta = _polar(re, im)
    if r == 0:
        raise InputError("Zero has only one root, zero itself.")
    root_r = sp.root(r, n)
    rows, points, ok = [], [], True
    target = complex(_num(re), _num(im))
    for k in range(n):
        angle = sp.simplify((theta + 2 * sp.pi * k) / n)
        wre, wim = sp.simplify(root_r * sp.cos(angle)), sp.simplify(root_r * sp.sin(angle))
        exact = sp.count_ops(wre) + sp.count_ops(wim) <= 16
        shown = _form(wre, wim) if exact else "%.5f %s %.5f\\,i" % (_num(wre), "-" if _num(wim) < 0 else "+", abs(_num(wim)))
        rows.append([str(k), tex(angle), shown])
        points.append([_num(wre), _num(wim)])
        ok = ok and abs(complex(_num(wre), _num(wim)) ** n - target) < 1e-7 * (1 + abs(target))
    table = step("List the roots", "Take \\(k = 0, 1, \\ldots, %d\\); after that the angles repeat." % (n - 1))
    table["table"] = {"headers": ["k", "\\theta_k", "w_k"], "rows": rows}
    steps = [step("Write z in polar form", "", ["z = %s = %s\\,e^{%s i}" % (_form(re, im), tex(r), _wrap(theta))]),
             step("Use the root formula",
                  "The \\(n\\) roots all have modulus \\(\\sqrt[n]{r}\\) and are spread evenly around a "
                  "circle, \\(2\\pi/n\\) apart.",
                  ["w_k = \\sqrt[n]{r}\\,e^{i(\\theta + 2\\pi k)/n},\\qquad \\sqrt[%d]{r} = %s" % (n, tex(root_r))]),
             table]
    points.append(points[0])    # close the polygon the roots form
    return result("complex", "Roots", "\\sqrt[%d]{%s}" % (n, tex(z)),
                  "The %d %s roots lie on a circle of radius %s." % (
                      n, {2: "square", 3: "cube"}.get(n, "%dth" % n), sp.sstr(root_r)),
                  [method("roots", "Polar form", "Every non-zero complex number has exactly n n-th roots.",
                          steps, "w_k = %s\\,e^{i(%s + 2\\pi k)/%d},\\quad k = 0,\\ldots,%d"
                          % (tex(root_r), tex(theta), n, n - 1),
                          plot={"points": points}, verified=ok,
                          note="Raising each root to the power %d gives back z." % n)])


def solve(operation, text, n_text=""):
    if operation == "roots":
        return _roots(text, n_text)
    if operation == "evaluate":
        return _evaluate(text)
    raise InputError("Unknown complex-number operation.")


def preview(operation, text):
    quotient = _split_quotient(text)
    if quotient:
        return "\\frac{%s}{%s}" % (tex(parse(quotient[0])), tex(parse(quotient[1])))
    return tex(parse(text))
