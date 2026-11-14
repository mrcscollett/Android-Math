"""Fourier transforms on the whole line, with the convention

    F(w) = integral of f(x) e^(-i w x) dx,    f(x) = (1 / 2 pi) integral of F(w) e^(i w x) dw,

plus Fourier sine and cosine transforms on (0, infinity) and the heat
equation on the whole line solved with the heat kernel.
"""
import mpmath
import sympy as sp

from . import pieces
from .parsing import InputError, parse_expression, parse_number
from .toolkit import method, not_applicable, result, step
from .util import TIMEOUT, run_with_timeout, tex

OPERATIONS = ("forward", "inverse", "sine", "cosine", "heat")
X = sp.Symbol("x", real=True)
W = sp.Symbol("omega", real=True)
_PLOT_FUNCS = {"sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan",
               "sinh", "cosh", "tanh", "exp", "log", "Abs", "Heaviside"}
CONVENTION = ("F(\\omega) = \\int_{-\\infty}^{\\infty} f(x)\\,e^{-i\\omega x}\\,dx,\\qquad "
              "f(x) = \\frac{1}{2\\pi}\\int_{-\\infty}^{\\infty} F(\\omega)\\,e^{i\\omega x}\\,d\\omega")


def _erf_plot(arg):
    """Abramowitz and Stegun 7.1.26 (error below 2e-7), written with functions the grapher has."""
    z = sp.Abs(arg)
    t = 1 / (1 + sp.Float(0.3275911) * z)
    poly = t * (sp.Float(0.254829592) + t * (sp.Float(-0.284496736) + t * (
        sp.Float(1.421413741) + t * (sp.Float(-1.453152027) + t * sp.Float(1.061405429)))))
    return (2 * sp.Heaviside(arg) - 1) * (1 - poly * sp.exp(-z**2))


def _plot(expr, var):
    """Plot string in x for a real function of `var`, or None."""
    try:
        expr = expr.replace(lambda e: isinstance(e, sp.erfc), lambda e: 1 - sp.erf(e.args[0]))
        expr = expr.replace(lambda e: isinstance(e, sp.erf), lambda e: _erf_plot(e.args[0]))
        if expr.has(sp.I, sp.oo, sp.zoo, sp.nan, sp.Integral, sp.DiracDelta, sp.Piecewise):
            return None
        if any(type(f).__name__ not in _PLOT_FUNCS for f in expr.atoms(sp.Function)):
            return None
        if not expr.free_symbols <= {var}:
            return None
        return sp.sstr(expr.xreplace({var: sp.Symbol("x")}))
    except Exception:
        return None


def _wrap(expr):
    if expr.is_Add or expr.could_extract_minus_sign():
        return "\\left(%s\\right)" % tex(expr)
    return tex(expr)


def _real(expr):
    """expr with a vanishing imaginary part removed, when it simplifies away."""
    try:
        re, im = sp.expand_complex(expr).as_real_imag()
        if sp.simplify(im) == 0:
            return sp.simplify(re)
    except Exception:
        pass
    return expr


def _integrate_pieces(f, var, kernel, lo=-sp.oo, hi=sp.oo):
    """Sum over smooth pieces of the integral of f * kernel. Returns (value, pieces) or (None, pieces)."""
    parts = pieces.split(f, var, lo, hi)
    total = sp.Integer(0)
    for a, b, expr in parts:
        piece = run_with_timeout(
            lambda: sp.integrate(expr * kernel, (var, a, b), conds="none"), TIMEOUT, None)
        if piece is None or piece.has(sp.Integral) or piece.has(sp.oo, sp.zoo, sp.nan):
            return None, parts
        total += piece
    return total, parts


def _quad_transform(f, var, omega, sign=-1):
    """Numerical integral of f(var) e^(sign i omega var) over the line."""
    fn = pieces.numeric(f, var)
    cuts = sorted(set([-60.0, 0.0, 60.0] + [float(c) for c in pieces.breakpoints(f, var)]))
    cuts = [c for c in cuts if -60 <= c <= 60]
    if abs(fn(60.0)) + abs(fn(-60.0)) > 1e-9:
        # A slowly decaying function: integrate the oscillating tails properly.
        w = abs(omega)
        re = mpmath.quadosc(lambda t: (fn(t) + fn(-t)) * mpmath.cos(w * t), [0, mpmath.inf], omega=w)
        im = mpmath.quadosc(lambda t: (fn(t) - fn(-t)) * mpmath.sin(w * t), [0, mpmath.inf], omega=w)
        return complex(re, sign * im * (1 if omega > 0 else -1))
    re = mpmath.quad(lambda t: fn(t) * mpmath.cos(omega * t), cuts)
    im = mpmath.quad(lambda t: fn(t) * mpmath.sin(omega * t), cuts)
    return complex(re, sign * im)


def _decays(f, var, both=True):
    """Does f shrink toward 0 far from the origin? (A transform needs that.)"""
    try:
        fn = pieces.numeric(f, var)
        far = [abs(complex(fn(1000.0)))] + ([abs(complex(fn(-1000.0)))] if both else [])
        return all(v < 1e-4 for v in far)
    except Exception:
        return False


def _matches(F, var, f, x, sign=-1):
    """Does F(var) equal the transform of f(x) at a few test frequencies?"""
    try:
        for omega in (0.7, 2.3, -1.1):
            want = _quad_transform(f, x, omega, sign)
            got = complex(sp.N(F.subs(var, omega)))
            if abs(got - want) > 1e-5 * (1 + abs(want)):
                return False
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Forward transform
# --------------------------------------------------------------------------

def _table_entry(f):
    """(description, table line, shift) for the standard forms, or None."""
    coef, core = f.as_independent(X)
    if isinstance(core, sp.exp):
        arg = sp.expand(core.args[0])
        try:
            poly = sp.Poly(arg, X)
        except sp.PolynomialError:
            poly = None
        if poly is not None and poly.degree() == 2 and poly.nth(2).is_negative:
            a = -poly.nth(2)
            b = sp.simplify(poly.nth(1) / (2 * a))
            return ("a Gaussian", "\\mathcal{F}\\left\\{e^{-a x^2}\\right\\} = "
                    "\\sqrt{\\frac{\\pi}{a}}\\,e^{-\\omega^2/(4a)}", b)
        inner = core.args[0]
        if len(inner.atoms(sp.Abs)) == 1 and not inner.has(sp.Heaviside):
            ab = list(inner.atoms(sp.Abs))[0]
            rate = sp.simplify(-inner / ab)
            if not rate.has(X) and rate.is_positive:
                shift = sp.solve(ab.args[0], X)
                return ("a two-sided exponential", "\\mathcal{F}\\left\\{e^{-a|x|}\\right\\} = "
                        "\\frac{2a}{a^2 + \\omega^2}", shift[0] if shift else sp.Integer(0))
    steps_ = list(f.atoms(sp.Heaviside))
    if len(steps_) == 1 and core.has(sp.exp):
        rest = sp.simplify(core / steps_[0])
        if isinstance(rest, sp.exp) and not rest.has(sp.Heaviside):
            shift = sp.solve(steps_[0].args[0], X)
            return ("a one-sided exponential", "\\mathcal{F}\\left\\{e^{-a x}u(x)\\right\\} = "
                    "\\frac{1}{a + i\\omega}", shift[0] if shift else sp.Integer(0))
    if len(steps_) == 2 and not sp.simplify(f.subs({s: 0 for s in steps_})).has(X):
        cuts = pieces.breakpoints(f, X)
        parts = pieces.split(f, X)
        if len(cuts) == 2 and len(parts) == 1 and not parts[0][2].has(X):
            return ("a rectangular pulse", "\\mathcal{F}\\left\\{u(x+a) - u(x-a)\\right\\} = "
                    "\\frac{2\\sin a\\omega}{\\omega}", sp.simplify((cuts[0] + cuts[1]) / 2))
    return None


def _distribution(f):
    """Transforms that are only distributions: constants, cos(ax), sin(ax)."""
    total = sp.Integer(0)
    for term in sp.Add.make_args(sp.expand(f)):
        coef, core = term.as_independent(X)
        if core == 1:
            total += 2 * sp.pi * coef * sp.DiracDelta(W)
        elif isinstance(core, (sp.cos, sp.sin)):
            a = sp.simplify(core.args[0] / X)
            if a.has(X):
                return None
            if isinstance(core, sp.cos):
                total += sp.pi * coef * (sp.DiracDelta(W - a) + sp.DiracDelta(W + a))
            else:
                total += -sp.I * sp.pi * coef * (sp.DiracDelta(W - a) - sp.DiracDelta(W + a))
        else:
            return None
    return total


def _forward_value(f):
    """(F, parts, how) for an ordinary function f, or raises InputError."""
    if not _decays(f, X):
        raise InputError("That function does not die away as x grows, so its Fourier "
                         "transform is not an ordinary function. Try e^(-x^2), "
                         "e^(-abs(x)), 1/(1 + x^2) or a pulse u(x+1) - u(x-1).")
    kernel = sp.exp(-sp.I * W * X)
    F, parts = _integrate_pieces(f, X, kernel)
    candidates = []
    if F is not None:
        candidates.append(sp.simplify(F))
    if not f.has(sp.Heaviside, sp.Abs):
        k = sp.Symbol("k", real=True)
        alt = run_with_timeout(lambda: sp.fourier_transform(f, X, k), TIMEOUT, None)
        if alt is not None and not alt.has(sp.FourierTransform):
            alt = sp.simplify(alt.subs(k, W / (2 * sp.pi)))
            candidates.append(alt)
            if sp.simplify(f - f.subs(X, -X)) == 0:
                candidates.append(alt.subs(W, sp.Abs(W)))     # an even f has an even transform
    for cand in candidates:
        if not cand.has(sp.Integral) and _matches(cand, W, f, X):
            return _real(cand), parts
    raise InputError("I couldn't find that Fourier transform. The function must die away "
                     "fast enough for the integral over the whole line to converge "
                     "(for example e^(-x^2), e^(-abs(x)), or a pulse u(x+1) - u(x-1)).")


def _magnitude_plot(F):
    """(plot string, legend) of F if real, else of |F|."""
    nudge = W + sp.Float(1e-9)       # steps around removable points such as w = 0
    re, im = sp.expand_complex(F).as_real_imag()
    if sp.simplify(im) == 0:
        return _plot(F.subs(W, nudge), W), "F(\u03c9)"
    size = sp.sqrt(sp.simplify(re**2 + im**2))
    return _plot(size.subs(W, nudge), W), "|F(\u03c9)|"


def _forward(text):
    f, symbols = parse_expression(text, ["x"])
    f = f.xreplace({symbols["x"]: X})
    input_latex = "\\mathcal{F}\\left\\{%s\\right\\}" % tex(f)
    name = "Defining integral"

    # Impulses are handled by the sifting property; what is left is integrated.
    impulses, ordinary = sp.Integer(0), sp.Integer(0)
    sift_lines = []
    for term in sp.Add.make_args(sp.expand(f)):
        deltas = list(term.atoms(sp.DiracDelta))
        if not deltas:
            ordinary += term
            continue
        if len(deltas) != 1 or sp.Poly(deltas[0].args[0], X).degree() != 1:
            raise InputError("Write an impulse as delta(x - a).")
        point = sp.solve(deltas[0].args[0], X)[0]
        weight = sp.simplify(term / deltas[0]).subs(X, point)
        piece = weight * sp.exp(-sp.I * W * point)
        impulses += piece
        sift_lines.append("\\mathcal{F}\\left\\{%s\\right\\} = %s" % (tex(term), tex(piece)))

    dist = _distribution(ordinary) if ordinary != 0 else None
    if ordinary != 0 and dist is not None:
        F = sp.simplify(impulses + dist)
        steps = [step("The convention", "This tool uses", [CONVENTION]),
                 step("A function that does not die away",
                      "Constants, sines and cosines are not absolutely integrable, so their "
                      "transforms are not ordinary functions. They are impulses in frequency, "
                      "read from the table.",
                      ["\\mathcal{F}\\{1\\} = 2\\pi\\,\\delta(\\omega)",
                       "\\mathcal{F}\\{\\cos ax\\} = \\pi\\left[\\delta(\\omega - a) + \\delta(\\omega + a)\\right]",
                       "\\mathcal{F}\\{\\sin ax\\} = -i\\pi\\left[\\delta(\\omega - a) - \\delta(\\omega + a)\\right]"])]
        m = method("table", "Table of transforms",
                   "The function is a combination of constants, sines and cosines.",
                   steps, "F(\\omega) = %s" % tex(F), verified=None,
                   note="The result is a distribution, so it cannot be checked by ordinary "
                        "numerical integration.")
        return result("ftransform", "Fourier transform", input_latex,
                      "Fourier transform (a distribution).", [m])

    steps = [step("The convention", "This tool uses", [CONVENTION])]
    F = impulses
    parts = []
    if ordinary != 0:
        value, parts = _forward_value(ordinary)
        F = sp.simplify(F + value)
        if len(parts) > 1 or (parts and (parts[0][0] != -sp.oo or parts[0][1] != sp.oo)):
            steps.append(step(
                "Split where the formula changes",
                "The unit steps and absolute values change the formula at the break points, "
                "so integrate each piece on its own interval.",
                ["\\int_{%s}^{%s} %s\\,e^{-i\\omega x}\\,dx" % (tex(a), tex(b), _wrap(expr))
                 for a, b, expr in parts]))
        else:
            steps.append(step("Set up the integral", "",
                              ["F(\\omega) = \\int_{-\\infty}^{\\infty} %s\\,e^{-i\\omega x}\\,dx" % _wrap(ordinary)]))
    if sift_lines:
        steps.append(step("Impulses",
                          "An impulse at \\(x = a\\) picks out the value of \\(e^{-i\\omega x}\\) there.",
                          sift_lines))
    steps.append(step("Evaluate and simplify",
                      "The terms at infinity vanish because the function dies away.",
                      ["F(\\omega) = %s" % tex(F)]))
    plot, legend = _magnitude_plot(F) if not F.has(sp.DiracDelta) else (None, "")
    ok = True if ordinary != 0 else None
    note = ("Checked against numerical integration at three frequencies." if ordinary != 0
            else "Read directly from the sifting property of the impulse.")
    methods = [method("integral", name,
                      "The transform is defined by an integral over the whole line.",
                      steps, "F(\\omega) = %s" % tex(F), plot={"particular_expr": plot},
                      verified=ok, note=note)]

    # Use the function as typed: expanding hides forms such as e^(-(x-1)^2).
    entry = _table_entry(f) if (ordinary != 0 and impulses == 0) else None
    if entry is not None:
        what, line, shift = entry
        tsteps = [step("Recognise the form", "The function is %s%s." % (
            what, "" if shift == 0 else ", shifted to be centred at \\(x = %s\\)" % tex(shift)), [line])]
        if shift != 0:
            tsteps.append(step(
                "Shift theorem",
                "Shifting a function by \\(x_0\\) multiplies its transform by \\(e^{-i\\omega x_0}\\).",
                ["\\mathcal{F}\\{f(x - x_0)\\} = e^{-i\\omega x_0}F(\\omega),\\qquad x_0 = %s" % tex(shift)]))
        tsteps.append(step("Insert the constants", "Linearity carries any constant factor through.",
                           ["F(\\omega) = %s" % tex(F)]))
        methods.append(method("table", "Table and theorems",
                              "The function is a standard table entry%s."
                              % ("" if shift == 0 else " with a shift"),
                              tsteps, "F(\\omega) = %s" % tex(F), plot={"particular_expr": plot},
                              verified=True, note="Agrees with the defining integral."))
    elif ordinary != 0:
        methods.append(not_applicable("table", "Table and theorems",
                                      "The function is not one of the standard table forms "
                                      "(Gaussian, exponentials, rectangular pulse), so the "
                                      "integral is the direct route."))
    return result("ftransform", "Fourier transform", input_latex,
                  "Fourier transform with the convention F(\u03c9) = \u222b f(x) e^(\u2212i\u03c9x) dx.",
                  methods, span=5.0, legends={"exact": legend})


# --------------------------------------------------------------------------
# Inverse transform
# --------------------------------------------------------------------------

def _parse_frequency(text):
    expr, symbols = parse_expression(text, ["w", "i"])
    return expr.xreplace({symbols["w"]: W, symbols["i"]: sp.I})


def _inverse_rational(F):
    """Inverse of a proper rational function of i*omega by partial fractions, or None."""
    s = sp.Symbol("s")
    G = sp.together(sp.simplify(F.subs(W, -sp.I * s)))
    if not G.is_rational_function(s):
        return None
    num, den = sp.fraction(G)
    if sp.degree(num, s) >= sp.degree(den, s):
        return None
    try:
        poles = sp.roots(sp.Poly(den, s))
    except Exception:
        return None
    if sum(poles.values()) != sp.degree(den, s):
        return None
    causal, anticausal, lines = sp.Integer(0), sp.Integer(0), []
    for pole, mult in poles.items():
        if sp.re(pole) == 0:
            return None
        for m in range(1, mult + 1):
            # Coefficient of 1/(s - pole)^m in the partial fraction expansion.
            g = sp.simplify(G * (s - pole) ** mult)
            c = sp.simplify(sp.diff(g, s, mult - m).subs(s, pole) / sp.factorial(mult - m))
            if c == 0:
                continue
            term = c * X ** (m - 1) * sp.exp(pole * X) / sp.factorial(m - 1)
            lines.append("\\frac{%s}{\\left(i\\omega - %s\\right)^{%d}}" % (tex(c), _wrap(pole), m)
                         if m > 1 else "\\frac{%s}{i\\omega - %s}" % (tex(c), _wrap(pole)))
            if sp.re(pole) < 0:
                causal += term
            else:
                anticausal -= term
    return _real(sp.simplify(causal)), _real(sp.simplify(anticausal)), lines


def _inverse(text):
    F = _parse_frequency(text)
    input_latex = "\\mathcal{F}^{-1}\\left\\{%s\\right\\}" % tex(F)
    steps = [step("The convention", "This tool uses", [CONVENTION])]
    f, how, name = None, "", ""

    if F.has(sp.DiracDelta):
        total = sp.Integer(0)
        for term in sp.Add.make_args(sp.expand(F)):
            deltas = list(term.atoms(sp.DiracDelta))
            if len(deltas) != 1:
                raise InputError("Write each impulse as delta(w - a), with nothing else mixed in.")
            point = sp.solve(deltas[0].args[0], W)[0]
            total += sp.simplify(term / deltas[0]).subs(W, point) * sp.exp(sp.I * point * X) / (2 * sp.pi)
        f = _real(sp.simplify(total.rewrite(sp.cos)))
        steps.append(step("Impulses in frequency",
                          "An impulse at \\(\\omega = a\\) picks out the value of "
                          "\\(e^{i\\omega x}/2\\pi\\) there, giving a pure oscillation.",
                          ["\\mathcal{F}^{-1}\\{\\delta(\\omega - a)\\} = \\frac{e^{iax}}{2\\pi}",
                           "f(x) = %s" % tex(f)]))
        m = method("sifting", "Sifting property", "The transform is made of impulses.",
                   steps, "f(x) = %s" % tex(f), plot={"particular_expr": _plot(f, X)},
                   verified=None, note="A pure oscillation is not absolutely integrable, so "
                                       "this is not checked by numerical integration.")
        return result("ftransform", "Inverse Fourier transform", input_latex,
                      "Inverse Fourier transform.", [m], span=5.0, legends={"exact": "f(x)"})

    # sin(a w)/w is the transform of a rectangular pulse.
    ratio = sp.simplify(F * W)
    coef, core = ratio.as_independent(W)
    if isinstance(core, sp.sin) and not sp.simplify(core.args[0] / W).has(W):
        a = sp.Abs(sp.simplify(core.args[0] / W))
        sgn = 1 if sp.simplify(core.args[0] / W).is_positive else -1
        f = sgn * coef / 2 * (sp.Heaviside(X + a) - sp.Heaviside(X - a))
        name, how = "Table of transforms", "It matches the transform of a rectangular pulse."
        steps.append(step("Recognise the form", how,
                          ["\\mathcal{F}\\left\\{u(x+a) - u(x-a)\\right\\} = \\frac{2\\sin a\\omega}{\\omega}",
                           "f(x) = %s" % tex(f)]))
    if f is None:
        found = _inverse_rational(F)
        if found is not None:
            causal, anticausal, lines = found
            if anticausal != 0 and sp.simplify(anticausal - causal.subs(X, -X)) == 0:
                f = causal.subs(X, sp.Abs(X))
            else:
                f = causal * sp.Heaviside(X) + anticausal * sp.Heaviside(-X)
            name = "Partial fractions"
            how = "The transform is a rational function of \\(i\\omega\\)."
            steps.append(step("Split into partial fractions",
                              "Write the transform in terms of \\(i\\omega\\) and split it.",
                              ["%s = %s" % (tex(F), " + ".join(lines).replace("+ \\frac{-", "- \\frac{"))]))
            steps.append(step(
                "Invert each piece",
                "A pole in the left half-plane gives a function that lives on \\(x > 0\\); a "
                "pole in the right half-plane gives one on \\(x < 0\\).",
                ["\\mathcal{F}^{-1}\\left\\{\\frac{1}{a + i\\omega}\\right\\} = e^{-ax}u(x),\\qquad "
                 "\\mathcal{F}^{-1}\\left\\{\\frac{1}{a - i\\omega}\\right\\} = e^{ax}u(-x)\\quad (a > 0)",
                 "f(x) = %s" % tex(f)]))
    if f is None:
        value, parts = _integrate_pieces(F, W, sp.exp(sp.I * W * X) / (2 * sp.pi))
        candidates = [] if value is None else [sp.simplify(value)]
        if value is not None and sp.simplify(F - F.subs(W, -W)) == 0:
            candidates.append(sp.simplify(value).subs(X, sp.Abs(X)))
        for cand in candidates:
            cand = _real(cand)
            if not cand.has(sp.I) and _matches(F, W, cand, X):
                f = cand
                break
        if f is not None:
            name, how = "Inversion integral", "The transform dies away, so the inversion integral converges."
            steps.append(step("Set up the inversion integral", "",
                              ["f(x) = \\frac{1}{2\\pi}\\int_{-\\infty}^{\\infty} %s\\,e^{i\\omega x}\\,d\\omega" % _wrap(F)]))
            steps.append(step("Evaluate", "", ["f(x) = %s" % tex(f)]))
    if f is None:
        raise InputError("I couldn't invert that transform. Supported: rational functions of w "
                         "(written with i where needed, such as 1/(2 + i w)), Gaussians, "
                         "e^(-abs(w)), pulses, sin(a w)/w and impulses delta(w - a).")
    ok = _matches(F, W, f, X)
    m = method("inverse", name, how, steps, "f(x) = %s" % tex(f),
               plot={"particular_expr": _plot(f, X)}, verified=True if ok else None,
               note="Checked by transforming the answer forward numerically at three frequencies."
               if ok else "Could not be checked numerically.")
    return result("ftransform", "Inverse Fourier transform", input_latex,
                  "Inverse Fourier transform.", [m], span=5.0, legends={"exact": "f(x)"})


# --------------------------------------------------------------------------
# Sine and cosine transforms
# --------------------------------------------------------------------------

def _half_line(text, kind):
    f, symbols = parse_expression(text, ["x"])
    f = f.xreplace({symbols["x"]: X})
    if f.has(sp.DiracDelta):
        raise InputError("Impulses are not supported in the sine and cosine transforms.")
    wp = sp.Symbol("omega", positive=True)
    trig = sp.sin if kind == "sine" else sp.cos
    letter = "F_s" if kind == "sine" else "F_c"
    if not _decays(f, X, both=False):
        raise InputError("That integral does not converge: the function must die away as "
                         "x grows (for example e^(-2x) or 1/(1 + x^2)).")
    value, parts = _integrate_pieces(f, X, trig(wp * X), sp.Integer(0), sp.oo)
    candidates = [] if value is None else [sp.simplify(value)]
    if not f.has(sp.Heaviside, sp.Abs):
        # SymPy's own transforms carry a factor sqrt(2/pi) that this convention leaves out.
        builtin = sp.sine_transform if kind == "sine" else sp.cosine_transform
        k = sp.Symbol("k", positive=True)
        alt = run_with_timeout(lambda: builtin(f, X, k), TIMEOUT, None)
        if alt is not None and not alt.has(sp.SineTransform, sp.CosineTransform):
            candidates.append(sp.simplify(sp.sqrt(sp.pi / 2) * alt.subs(k, wp)))
    fn = pieces.numeric(f, X)
    cuts = sorted(set([0.0, 80.0] + [float(c) for c in pieces.breakpoints(f, X) if 0 < c < 80]))
    wave = mpmath.sin if kind == "sine" else mpmath.cos
    ok, F = False, None
    for cand in candidates:
        try:
            good = True
            for omega in (0.7, 2.3):
                if abs(fn(80.0)) > 1e-9:
                    want = float(mpmath.quadosc(lambda t: fn(t) * wave(omega * t), [0, mpmath.inf], omega=omega))
                else:
                    want = float(mpmath.quad(lambda t: fn(t) * wave(omega * t), cuts))
                got = float(sp.N(cand.subs(wp, omega)))
                if abs(got - want) > 1e-5 * (1 + abs(want)):
                    good = False
        except Exception:
            good = False
        if good:
            ok, F = True, cand
            break
    if not ok:
        raise InputError("That integral does not converge: the function must die away as "
                         "x grows (for example e^(-2x) or 1/(1 + x^2)).")
    name = "Fourier %s transform" % kind
    steps = [
        step("The definition",
             "On the half-line \\(x > 0\\) a function is represented by %ss alone, the "
             "counterpart of a half-range series." % kind,
             ["%s(\\omega) = \\int_0^{\\infty} f(x)\\,\\%s \\omega x\\,dx,\\qquad "
              "f(x) = \\frac{2}{\\pi}\\int_0^{\\infty} %s(\\omega)\\,\\%s \\omega x\\,d\\omega"
              % (letter, kind[:3], letter, kind[:3])]),
        step("Set up the integral", "",
             ["\\int_{%s}^{%s} %s\\,\\%s \\omega x\\,dx" % (tex(a), tex(b), _wrap(expr), kind[:3])
              for a, b, expr in parts]),
        step("Evaluate",
             "Integrate by parts twice (or use the table); the terms at infinity vanish.",
             ["%s(\\omega) = %s" % (letter, tex(F))]),
    ]
    m = method(kind, name,
               "The %s transform suits problems on \\(x > 0\\) with %s at \\(x = 0\\)."
               % (kind, "the value fixed at 0" if kind == "sine" else "an insulated end"),
               steps, "%s(\\omega) = %s" % (letter, tex(F)),
               plot={"particular_expr": _plot(F.subs(wp, wp + sp.Float(1e-9)), wp)}, verified=True,
               note="Checked against numerical integration at two frequencies.")
    return result("ftransform", name, "\\mathcal{F}_{%s}\\left\\{%s\\right\\}" % (kind[0], tex(f)),
                  name + ".", [m], span=5.0, legends={"exact": "%s(\u03c9), \u03c9 > 0" % letter.replace("_", "")})


# --------------------------------------------------------------------------
# Heat equation on the whole line
# --------------------------------------------------------------------------

def _heat(text, k_text):
    f, symbols = parse_expression(text, ["x"])
    f = f.xreplace({symbols["x"]: X})
    if f.has(sp.DiracDelta):
        raise InputError("For an initial impulse the solution is the heat kernel itself; "
                         "enter an ordinary function here.")
    k = parse_number(k_text, "The diffusivity k") if (k_text or "").strip() else sp.Integer(1)
    if k <= 0:
        raise InputError("The diffusivity k must be positive.")
    s = sp.Symbol("s", real=True)
    t = sp.Symbol("t", positive=True)
    kernel = sp.exp(-(X - s) ** 2 / (4 * k * t)) / sp.sqrt(4 * sp.pi * k * t)
    value, parts = _integrate_pieces(f.subs(X, s), s, kernel)
    if value is None:
        raise InputError("The heat-kernel integral for that initial temperature has no closed "
                         "form here. Try a Gaussian such as e^(-x^2), a step u(x), or a pulse "
                         "u(x+1) - u(x-1).")
    u = sp.simplify(value)
    if u.has(sp.I):
        u = _real(u)
    # Independent check: quadrature of the kernel integral, and the PDE by finite differences.
    fn = pieces.numeric(f, X)
    un = sp.lambdify((X, t), u, "mpmath")
    kf = float(k)
    ok = True
    try:
        for xv, tv in ((0.3, 0.4), (-1.2, 1.0), (2.0, 0.25)):
            cuts = sorted(set([xv - 40, xv, xv + 40] + [float(c) for c in pieces.breakpoints(f, X)]))
            want = mpmath.quad(lambda z: fn(z) * mpmath.exp(-(xv - z) ** 2 / (4 * kf * tv))
                               / mpmath.sqrt(4 * mpmath.pi * kf * tv), cuts)
            if abs(un(xv, tv) - want) > 1e-6 * (1 + abs(want)):
                ok = False
            h = 1e-3
            ut = (un(xv, tv + h) - un(xv, tv - h)) / (2 * h)
            uxx = (un(xv + h, tv) - 2 * un(xv, tv) + un(xv - h, tv)) / h**2
            if abs(ut - kf * uxx) > 1e-4 * (1 + abs(ut)):
                ok = False
    except Exception:
        ok = False
    if not ok:
        raise InputError("The heat-kernel integral for that initial temperature could not be "
                         "confirmed. The function must stay bounded as x grows.")
    steps = [
        step("The problem",
             "Heat flow on an unbounded rod, with the temperature known at \\(t = 0\\).",
             ["u_t = %s u_{xx},\\qquad u(x, 0) = %s" % ("" if k == 1 else tex(k) + "\\,", tex(f))]),
        step("Transform in x",
             "Take the Fourier transform in \\(x\\). Each derivative in \\(x\\) becomes a factor "
             "\\(i\\omega\\), so the PDE becomes an ordinary equation in \\(t\\) for each frequency.",
             ["\\hat{u}_t = -%s\\omega^2\\,\\hat{u},\\qquad \\hat{u}(\\omega, 0) = F(\\omega)"
              % ("" if k == 1 else tex(k) + "\\,"),
              "\\hat{u}(\\omega, t) = F(\\omega)\\,e^{-%s\\omega^2 t}" % ("" if k == 1 else tex(k) + "\\,")]),
        step("Transform back",
             "A product of transforms is the transform of a convolution, and "
             "\\(e^{-k\\omega^2 t}\\) is the transform of a Gaussian, the heat kernel.",
             ["u(x, t) = \\int_{-\\infty}^{\\infty} f(s)\\,\\frac{1}{\\sqrt{4\\pi k t}}"
              "\\,e^{-(x-s)^2/(4kt)}\\,ds"]),
        step("Evaluate the integral",
             "For a step or a pulse the result is written with the error function "
             "\\(\\operatorname{erf}\\)." if u.has(sp.erf, sp.erfc) else "",
             ["u(x, t) = %s" % tex(u)]),
        step("What the solution does",
             "The initial profile spreads out and flattens; sharp corners and jumps are "
             "smoothed at once, and the total heat \\(\\int u\\,dx\\) stays constant."),
    ]
    partials = []
    for tv in [0.02, 0.1, 0.3, 1, 3, 10]:
        when = sp.nsimplify(tv) / k
        expr = _plot(u.subs(t, when), X)
        if not expr:
            continue
        # The grapher has no error function, so erf is replaced by a standard
        # approximation. Keep a time only if the curve it draws is accurate.
        try:
            drawn = sp.lambdify(sp.Symbol("x"), sp.sympify(expr), [
                {"Heaviside": lambda v, *rest: 0.0 if v < 0 else 1.0}, "mpmath"])
            good = all(abs(drawn(xv) - un(xv, float(when))) < 1e-5 for xv in (-3.0, -0.8, 0.0, 1.1, 3.5))
        except Exception:
            good = False
        if good:
            partials.append({"n": len(partials) + 1, "expr": expr,
                             "label": "t = %s" % sp.sstr(sp.nsimplify(when))})
    m = method("kernel", "Fourier transform and heat kernel",
               "On the whole line there are no boundary conditions, so a Fourier transform in "
               "\\(x\\) replaces the Fourier series used on a finite rod.",
               steps, "u(x, t) = %s" % tex(u),
               plot={"particular_expr": _plot(f, X), "partials": partials}, verified=True,
               note="Checked against direct numerical integration of the kernel formula, and "
                    "against the equation \\(u_t = k u_{xx}\\) by finite differences.")
    return result("ftransform", "Heat equation on the line",
                  "u_t = %su_{xx},\\quad u(x,0) = %s" % ("" if k == 1 else tex(k) + "\\,", tex(f)),
                  "Heat equation on the whole line, solved with the heat kernel.", [m], span=3.0,
                  legends={"exact": "Initial temperature"})


def solve(operation, text, extra1="", extra2=""):
    if operation == "forward":
        return _forward(text)
    if operation == "inverse":
        return _inverse(text)
    if operation in ("sine", "cosine"):
        return _half_line(text, operation)
    if operation == "heat":
        return _heat(text, extra1)
    raise InputError("Unknown Fourier-transform operation.")


def preview(operation, text):
    if operation == "inverse":
        return "\\mathcal{F}^{-1}\\left\\{%s\\right\\}" % tex(_parse_frequency(text))
    f = parse_expression(text, ["x"])[0]
    if operation == "heat":
        return "u(x, 0) = %s" % tex(f)
    if operation in ("sine", "cosine"):
        return "\\mathcal{F}_{%s}\\left\\{%s\\right\\}" % (operation[0], tex(f))
    return "\\mathcal{F}\\left\\{%s\\right\\}" % tex(f)
