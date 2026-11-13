"""Shared helpers: LaTeX output, plot expressions, verification."""
import random
import threading

import sympy as sp
from sympy.printing.latex import LatexPrinter

# Functions the Kotlin plot evaluator understands (SymPy class name -> ok).
_PLOT_FUNCS = {
    "sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "exp", "log", "Abs", "Heaviside",
}


class _Printer(LatexPrinter):
    """SymPy's LaTeX with ln|u| written the textbook way instead of ln(|u|)."""

    def _print_log(self, expr, exp=None):
        arg = expr.args[0]
        if isinstance(arg, sp.Abs):
            inner = self._print(arg.args[0])
            if exp is None:
                return r"\ln\left|%s\right|" % inner
            return r"\ln^{%s}\left|%s\right|" % (exp, inner)
        return super()._print_log(expr, exp)

    def _print_Heaviside(self, expr, exp=None):
        """The unit step is written u(t - a), as in most ODE textbooks."""
        out = r"u\left(%s\right)" % self._print(expr.args[0])
        return out if exp is None else r"%s^{%s}" % (out, exp)


def tex(expr):
    return _Printer({"ln_notation": True}).doprint(expr)


def tex_side(expr, prob):
    """LaTeX for one side of the ODE with the y' term first, then y terms."""
    terms = sp.Add.make_args(expr)
    if len(terms) < 2 or not expr.has(prob.yp, prob.ypp):
        return tex(expr)
    def rank(term):
        if term.has(prob.ypp):
            return 0
        return 1 if term.has(prob.yp) else (2 if term.has(prob.y) else 3)
    ordered = sorted(terms, key=rank)
    out = tex(ordered[0])
    for term in ordered[1:]:
        if term.could_extract_minus_sign():
            out += " - " + tex(-term)
        else:
            out += " + " + tex(term)
    return out


def dydx(prob):
    return r"\frac{dy}{d%s}" % prob.indep_name


def too_complex(expr):
    return sp.count_ops(expr) > MAX_OPS


def signed(expr):
    """LaTeX for '+ expr', written '- |expr|' when expr is negative."""
    if expr.could_extract_minus_sign():
        return "- %s" % tex(-expr)
    return "+ %s" % tex(expr)


def has_integral(expr):
    return expr.has(sp.Integral)


try:  # SymPy recurses deeply; give worker threads room (matters on Android).
    threading.stack_size(8 * 1024 * 1024)
except (ValueError, RuntimeError):
    pass

TIMEOUT = 12  # seconds allowed for any single SymPy call that might hang
CHECK_TIMEOUT = 2  # symbolic verification falls back to a numerical check after this
MAX_OPS = 70  # explicit formulas more complicated than this are left implicit


def integrate(expr, var):
    """Integrate, returning an unevaluated Integral if SymPy can't."""
    result = run_with_timeout(lambda: sp.integrate(expr, var), TIMEOUT, None)
    if result is None:
        return sp.Integral(expr, var)
    if result.has(sp.Piecewise):
        return sp.Integral(expr, var)
    return result


def plot_expr(expr, prob):
    """A string the Kotlin evaluator can plot, or None if it can't.

    Variables are always named x (independent), y and C so the app does not
    need to know what the user called them.
    """
    if expr is None:
        return None
    try:
        if expr.has(sp.I, sp.oo, sp.zoo, sp.nan, sp.Integral, sp.Piecewise, sp.Derivative):
            return None
        for fn in expr.atoms(sp.Function):
            if type(fn).__name__ not in _PLOT_FUNCS:
                return None
        names = {prob.x: sp.Symbol("x"), prob.y: sp.Symbol("y"), prob.C: sp.Symbol("C"),
                 prob.C1: sp.Symbol("C"), prob.C2: sp.Symbol("C2")}
        if not expr.free_symbols <= set(names):
            return None
        return sp.sstr(expr.xreplace(names))
    except Exception:
        return None


def is_zero(expr):
    try:
        if expr == 0:
            return True
        simplified = sp.simplify(expr)
        if simplified == 0:
            return True
        return bool(simplified.equals(0))
    except Exception:
        return False


def sample_zero(expr, symbols, around=None, spread=1.5, tries=12, tol=1e-7):
    """Evaluate expr at random real points. Returns (points_ok, points_bad)."""
    rng = random.Random(7)
    good = bad = 0
    for _ in range(tries * 4):
        point = {}
        for s in symbols:
            centre = float(around[s]) if around and s in around else 0.0
            width = spread if not (around and s in around) else 0.25
            point[s] = sp.Float(centre + rng.uniform(-width, width))
        try:
            value = complex(expr.evalf(subs=point))
        except Exception:
            continue
        if value != value or abs(value) == float("inf") or abs(value.imag) > tol:
            continue
        scale = 1 + max([abs(float(v)) for v in point.values()] or [0]) ** 2
        if abs(value) > tol * scale:
            bad += 1
        else:
            good += 1
        if good + bad >= tries:
            break
    return good, bad


def run_with_timeout(fn, seconds, default):
    """Run fn() but give up after `seconds` (the worker is abandoned, not killed)."""
    box = {}

    def work():
        try:
            box["value"] = fn()
        except Exception:
            box["value"] = default

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(seconds)
    return box.get("value", default)


def with_abs(expr):
    """Write ln(u) as ln|u| wherever u is not known to be positive."""
    def fix(e):
        arg = e.args[0]
        return e if arg.is_positive else sp.log(sp.Abs(arg))
    return expr.replace(lambda e: isinstance(e, sp.log), fix)


def residual_status(residual, around=None):
    """Classify how well `residual` vanishes.

    Returns "symbolic", "numeric", "partial" (zero on part of the plane only),
    "bad" or "unknown". The numerical test runs first because it is fast; the
    symbolic proof is attempted only when the expression is small enough for
    SymPy to simplify quickly.
    """
    if residual == 0:
        return "symbolic"
    small = sp.count_ops(residual) <= 150
    if small:
        try:
            if sp.expand(residual) == 0:
                return "symbolic"
        except Exception:
            pass
    symbols = sorted(residual.free_symbols, key=lambda s: s.name)
    good, bad = sample_zero(residual, symbols, around)
    if bad:
        return "partial" if (good >= 3 and around is None) else "bad"
    if small and run_with_timeout(lambda: is_zero(residual), CHECK_TIMEOUT, False):
        return "symbolic"
    return "numeric" if good >= 3 else "unknown"


def hard_to_solve(expr, var):
    """True when solving expr = const for var means a cubic or worse."""
    def degree(e):
        try:
            num, den = sp.fraction(sp.together(e))
            return max(sp.Poly(num, var).degree(), sp.Poly(den, var).degree())
        except Exception:
            return None

    top = degree(expr)
    if top is not None:
        return top > 2
    for fn in expr.atoms(sp.Function):
        for arg in fn.args:
            inner = arg.args[0] if isinstance(arg, sp.Abs) else arg
            d = degree(inner)
            if d is not None and d > 2:
                return True
    return False


def check_solution(prob, f, sol, x0=None, y0=None):
    """Substitute y = sol into y' = f. Returns (verified, note)."""
    residual = sp.diff(sol, prob.x) - f.subs(prob.y, sol)
    around = {prob.x: x0} if x0 is not None else None
    status = residual_status(residual, around)
    if status == "partial":
        return None, ("Substituting back works on part of the plane only: "
                      "this formula is one branch of the solution and holds "
                      "where the roots and inverse functions in it are defined.")
    if status == "bad":
        return False, "Substituting back did not reproduce the equation."
    if status == "unknown":
        return None, "Could not be checked automatically."
    how = "symbolically" if status == "symbolic" else "numerically at sample points"
    if x0 is not None:
        at = sol.subs(prob.x, x0)
        if not (is_zero(at - y0) or abs(complex(sp.N(at - y0))) < 1e-9):
            return False, "The initial condition is not satisfied."
        return True, "Checked %s by substituting back, and it meets the initial condition." % how
    return True, "Checked %s by substituting back into the equation." % how
