"""Shared helpers: LaTeX output, plot expressions, verification."""
import random
import threading

import sympy as sp

# Functions the Kotlin plot evaluator understands (SymPy class name -> ok).
_PLOT_FUNCS = {
    "sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "exp", "log", "Abs",
}


def tex(expr):
    return sp.latex(expr, ln_notation=True)


def tex_side(expr, prob):
    """LaTeX for one side of the ODE with the y' term first, then y terms."""
    terms = sp.Add.make_args(expr)
    if len(terms) < 2 or not expr.has(prob.yp):
        return tex(expr)
    def rank(term):
        return 0 if term.has(prob.yp) else (1 if term.has(prob.y) else 2)
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


def has_integral(expr):
    return expr.has(sp.Integral)


try:  # SymPy recurses deeply; give worker threads room (matters on Android).
    threading.stack_size(8 * 1024 * 1024)
except (ValueError, RuntimeError):
    pass

TIMEOUT = 12  # seconds allowed for any single SymPy call that might hang


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
        names = {prob.x: sp.Symbol("x"), prob.y: sp.Symbol("y"), prob.C: sp.Symbol("C")}
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


def check_solution(prob, f, sol, x0=None, y0=None):
    """Substitute y = sol into y' = f. Returns (verified, note)."""
    residual = sp.diff(sol, prob.x) - f.subs(prob.y, sol)
    around = {prob.x: x0} if x0 is not None else None
    if run_with_timeout(lambda: is_zero(residual), TIMEOUT, False):
        how = "symbolically"
    else:
        symbols = sorted(residual.free_symbols, key=lambda s: s.name)
        good, bad = sample_zero(residual, symbols, around)
        if good >= 3 and bad == 0:
            how = "numerically at sample points"
        elif good >= 3 and x0 is None:
            return None, ("Substituting back works on part of the plane only: "
                          "this formula is one branch of the solution and holds "
                          "where the roots and inverse functions in it are defined.")
        elif bad:
            return False, "Substituting back did not reproduce the equation."
        else:
            return None, "Could not be checked automatically."
    if x0 is not None:
        at = sol.subs(prob.x, x0)
        if not (is_zero(at - y0) or abs(complex(sp.N(at - y0))) < 1e-9):
            return False, "The initial condition is not satisfied."
        return True, "Checked %s by substituting back, and it meets the initial condition." % how
    return True, "Checked %s by substituting back into the equation." % how
