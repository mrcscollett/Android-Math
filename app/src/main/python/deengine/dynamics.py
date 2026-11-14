"""Stability and nonlinear dynamics for planar systems and scalar equations.

Operations: Lyapunov's direct method, conservative systems x'' = F(x),
limit cycles (Bendixson, polar coordinates, Lienard), bifurcations of
x' = f(x, m), and the Hopf bifurcation of a planar system with a parameter m.
"""
import math
import re

import sympy as sp

from .parsing import InputError, parse_expression, parse_number
from .systems import _mat, _plot, classify
from .toolkit import method, not_applicable, result, step
from .util import TIMEOUT, run_with_timeout, tex

OPERATIONS = ("lyapunov", "conservative", "cycles", "bifurcation", "hopf")

X = sp.Symbol("x", real=True)
Y = sp.Symbol("y", real=True)
M = sp.Symbol("m", real=True)
T = sp.Symbol("t", real=True)
_NAMES = {X: "x", Y: "y"}


def _a(noun):
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def _clean(text):
    """The parameter may be typed as m or mu."""
    return re.sub(r"(?<![A-Za-z])mu(?![A-Za-z])", "m", text or "")


def _expr(text, names, what):
    if not text.strip():
        raise InputError("Enter %s." % what)
    expr, symbols = parse_expression(_clean(text), names)
    swaps = {symbols[n]: {"x": X, "y": Y, "m": M, "t": T}[n] for n in names}
    expr = expr.xreplace(swaps)
    if expr.has(sp.Heaviside, sp.DiracDelta):
        raise InputError("Steps and impulses are not supported here.")
    return expr


def _system(text, names=("x", "y")):
    """The pair (f, g) typed as 'f; g'."""
    parts = [p for p in _clean(text).replace("\n", ";").split(";")]
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise InputError("Enter the two right-hand sides separated by a semicolon, "
                         "for example  y; -x - y.")
    return (_expr(parts[0], list(names), "x'"), _expr(parts[1], list(names), "y'"))


def _pair(text, label):
    parts = [p for p in re.split(r"[;,]", text) if p.strip()]
    if len(parts) != 2:
        raise InputError("%s needs two numbers separated by a comma." % label)
    return parse_number(parts[0], label), parse_number(parts[1], label)


def _equilibria(f, g, limit=8):
    """Real isolated equilibria of x' = f, y' = g (possibly incomplete for periodic systems)."""
    found = run_with_timeout(lambda: sp.solve([f, g], [X, Y], dict=True), TIMEOUT, [])
    points = []
    for sol in found or []:
        if X in sol and Y in sol and not (sol[X].free_symbols or sol[Y].free_symbols):
            if sol[X].is_real and sol[Y].is_real:
                points.append((sol[X], sol[Y]))
    return sorted(set(points), key=lambda p: (float(p[0]), float(p[1])))[:limit]


def _jacobian(f, g):
    return sp.Matrix([[sp.diff(f, X), sp.diff(f, Y)], [sp.diff(g, X), sp.diff(g, Y)]])


def _circle_values(expr, point, radius, count=720):
    """Values of expr around a circle about the point, and the function of the angle."""
    fn = sp.lambdify((X, Y), expr, "math")
    cx, cy = float(point[0]), float(point[1])

    def at(angle):
        try:
            value = fn(cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            return float(value) if value == value else None
        except Exception:
            return None

    return [(2 * math.pi * k / count, at(2 * math.pi * k / count)) for k in range(count)], at


def _touches_zero(values, at, scale):
    """Does |expr| reach zero somewhere on the circle (not just come close)?"""
    usable = [(a, v) for a, v in values if v is not None]
    angle, smallest = min(usable, key=lambda p: abs(p[1]))
    if abs(smallest) < 1e-13 * scale:
        return True
    width = 2 * math.pi / len(values)
    lo, hi = angle - width, angle + width
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        v1, v2 = at(m1), at(m2)
        if v1 is None or v2 is None:
            break
        if abs(v1) < abs(v2):
            hi = m2
        else:
            lo = m1
    final = at((lo + hi) / 2)
    return final is not None and abs(final) < 1e-11 * scale


def _definiteness(expr, point):
    """'pd', 'psd', 'nd', 'nsd', 'zero' or 'indefinite' near the point.

    The sign is read on two small circles around the point. A function that
    keeps one sign but vanishes along a curve through the point (such as
    -y^2) is only semidefinite, so the circles are also searched for zeros.
    """
    signs, touches = set(), False
    for radius in (0.3, 0.03):
        values, at = _circle_values(expr, point, radius)
        usable = [v for _, v in values if v is not None]
        if not usable:
            continue
        scale = max(abs(v) for v in usable)
        if scale < 1e-14:
            signs.add(0)
            continue
        for v in usable:
            if abs(v) > 1e-13 * scale:
                signs.add(1 if v > 0 else -1)
        if not ({1, -1} <= signs):
            touches = touches or _touches_zero(values, at, scale)
    if {1, -1} <= signs:
        return "indefinite"
    if signs == {0} or not signs:
        return "zero"
    positive = 1 in signs
    if touches or 0 in signs:
        return "psd" if positive else "nsd"
    return "pd" if positive else "nd"


_WORDS = {"pd": "positive definite", "psd": "positive semidefinite", "nd": "negative definite",
          "nsd": "negative semidefinite", "zero": "identically zero", "indefinite": "indefinite"}


def _field_extras(f, g, marks, legends=None, curve=None):
    extra = {"graph": "phase", "field": {"f": _plot(f, _NAMES), "g": _plot(g, _NAMES)},
             "equilibria": marks}
    if legends:
        extra["legends"] = legends
    return extra


# --------------------------------------------------------------------------
# Lyapunov's direct method
# --------------------------------------------------------------------------

def _verdict(v_kind, vdot_kind):
    if v_kind != "pd":
        return None
    if vdot_kind == "nd":
        return "asymptotically stable"
    if vdot_kind in ("nsd", "zero"):
        return "stable"
    if vdot_kind == "pd":
        return "unstable"
    return None


def _candidates(f, g, point):
    """Quadratic candidates (label, V) in shifted coordinates around the point."""
    u, v = X - point[0], Y - point[1]
    out = [("the squared distance", u**2 + v**2)]
    # Choose the ratio of the weights to cancel the cross terms of V'.
    A = sp.Symbol("A", positive=True)
    vdot = sp.expand((2 * u * f + 2 * A * v * g).subs({X: X + point[0], Y: Y + point[1]}))
    try:
        poly = sp.Poly(vdot, X, Y)
        cross = [c for (i, j), c in poly.terms() if i % 2 == 1 and j % 2 == 1]
        for c in cross:
            for value in sp.solve(c, A):
                if value.is_positive and value != 1:
                    out.append(("weights chosen to cancel the cross terms", u**2 + value * v**2))
                    break
            if len(out) > 1:
                break
    except sp.PolynomialError:
        pass
    # The Lyapunov equation J^T P + P J = -I gives a quadratic V whenever the linearization is a sink.
    J = _jacobian(f, g).subs({X: point[0], Y: point[1]})
    p11, p12, p22 = sp.symbols("p11 p12 p22")
    P = sp.Matrix([[p11, p12], [p12, p22]])
    eqs = list(J.T * P + P * J + sp.eye(2))
    sol = sp.solve(eqs, [p11, p12, p22], dict=True)
    if sol:
        Pm = P.subs(sol[0])
        if not Pm.free_symbols and Pm[0, 0] > 0 and Pm.det() > 0:
            scale = sp.ilcm(*[sp.fraction(sp.nsimplify(e))[1] for e in Pm]) if all(
                e.is_Rational for e in Pm) else 1
            V = sp.expand(scale * (Pm[0, 0] * u**2 + 2 * Pm[0, 1] * u * v + Pm[1, 1] * v**2))
            if all(sp.simplify(V - other) != 0 for _, other in out):
                out.append(("the solution of the Lyapunov equation \\(J^{T}P + PJ = -I\\)", V))
    return out


def _lyapunov(text, v_text, point_text):
    f, g = _system(text)
    if point_text.strip():
        point = _pair(point_text, "The equilibrium point")
    else:
        point = (sp.Integer(0), sp.Integer(0))
        if sp.simplify(f.subs({X: 0, Y: 0})) != 0 or sp.simplify(g.subs({X: 0, Y: 0})) != 0:
            found = _equilibria(f, g)
            if not found:
                raise InputError("The origin is not an equilibrium of this system. Enter the "
                                 "equilibrium point to study, for example  1, 0.")
            point = found[0]
    at = {X: point[0], Y: point[1]}
    if sp.simplify(f.subs(at)) != 0 or sp.simplify(g.subs(at)) != 0:
        raise InputError("(%s, %s) is not an equilibrium: both right-hand sides must be zero "
                         "there." % (sp.sstr(point[0]), sp.sstr(point[1])))
    where = "(%s,\\ %s)" % (tex(point[0]), tex(point[1]))

    def analyse(V, intro_title, intro_text):
        vdot = sp.simplify(sp.diff(V, X) * f + sp.diff(V, Y) * g)
        shown = sp.factor(vdot)
        v_kind = _definiteness(V, point) if sp.simplify(V.subs(at)) == 0 else "indefinite"
        d_kind = _definiteness(vdot, point)
        steps = [
            step(intro_title, intro_text, ["V = %s" % tex(V)]),
            step("Check V itself",
                 "\\(V\\) is zero at the equilibrium and is %s near it." % _WORDS[v_kind]
                 if sp.simplify(V.subs(at)) == 0 else
                 "\\(V\\) is not zero at the equilibrium, so it cannot serve as a Lyapunov function."),
            step("Differentiate along solutions",
                 "By the chain rule, \\(\\dot V = V_x\\,x' + V_y\\,y'\\).",
                 ["\\dot V = %s\\left(%s\\right) + %s\\left(%s\\right)"
                  % (tex(sp.diff(V, X)), tex(f), tex(sp.diff(V, Y)), tex(g)),
                  "\\dot V = %s" % tex(shown)]),
            step("Read off the sign",
                 "Near the equilibrium \\(\\dot V\\) is %s." % _WORDS[d_kind]),
        ]
        return steps, _verdict(v_kind, d_kind), v_kind, d_kind, vdot

    conclusions = {
        "asymptotically stable": "\\(V\\) is positive definite and \\(\\dot V\\) is negative "
                                 "definite, so the equilibrium is asymptotically stable.",
        "stable": "\\(V\\) is positive definite and \\(\\dot V \\le 0\\), so the equilibrium is "
                  "stable. Because \\(\\dot V\\) is zero along a whole curve, this test alone "
                  "does not show that solutions approach the equilibrium.",
        "unstable": "\\(V\\) and \\(\\dot V\\) are both positive definite, so solutions move away: "
                    "the equilibrium is unstable.",
    }
    note = ("The signs were confirmed by evaluating \\(V\\) and \\(\\dot V\\) around two small "
            "circles centred on the equilibrium.")
    methods = []

    best = None
    tried = []
    for label, V in _candidates(f, g, point):
        steps, verdict, v_kind, d_kind, _ = analyse(
            V, "Choose a candidate", "Try a quadratic function centred on the equilibrium: %s." % label)
        tried.append((label, V, d_kind))
        rank = {"asymptotically stable": 0, "unstable": 1, "stable": 2, None: 3}[verdict]
        if best is None or rank < best[0]:
            best = (rank, steps, verdict, V)
    rank, steps, verdict, V = best
    if verdict:
        if len(tried) > 1 and tried[0][1] != V:
            first_kind = tried[0][2]
            steps.insert(0, step(
                "Why not the simplest choice",
                "With \\(V = %s\\) the derivative \\(\\dot V\\) is %s near the equilibrium, "
                "%s, so a better-shaped quadratic is needed."
                % (tex(tried[0][1]), _WORDS[first_kind],
                   "which shows stability but not that solutions approach the equilibrium"
                   if first_kind in ("nsd", "zero") else "which decides nothing")))
        steps.append(step("Conclude", conclusions[verdict]))
        methods.append(method("quadratic", "Quadratic Lyapunov function",
                              "A quadratic function centred on the equilibrium works as a "
                              "Lyapunov function here.", steps,
                              "\\text{The equilibrium } %s \\text{ is %s.}" % (where, verdict),
                              verified=True, note=note))
    else:
        methods.append(not_applicable(
            "quadratic", "Quadratic Lyapunov function",
            "None of the simple quadratic candidates tried has a derivative of one sign near "
            "the equilibrium, so they decide nothing. A different function \\(V\\) is needed; "
            "you can enter your own."))

    if v_text.strip():
        V = _expr(v_text, ["x", "y"], "V(x, y)")
        steps, verdict, v_kind, d_kind, _ = analyse(V, "Your function", "Test the function you entered.")
        steps.append(step("Conclude", conclusions[verdict] if verdict else
                          "These signs do not match any of the Lyapunov theorems, so this "
                          "\\(V\\) decides nothing about the equilibrium."))
        methods.insert(0, method(
            "given", "Your function V", "Tests the function you supplied.", steps,
            "\\text{The equilibrium } %s \\text{ is %s.}" % (where, verdict) if verdict
            else "\\text{This } V \\text{ is inconclusive.}", verified=True if verdict else None,
            note=note))

    J = _jacobian(f, g).subs(at).applyfunc(sp.simplify)
    kind, stability = classify(J)
    eigs = sorted(J.eigenvals(multiple=True), key=sp.default_sort_key)
    hyperbolic = all(sp.re(e) != 0 for e in eigs)
    lin_steps = [
        step("Linearize", "Evaluate the Jacobian at the equilibrium.",
             ["J = %s" % _mat(J), "\\lambda = %s" % ",\\ ".join(tex(e) for e in eigs)]),
        step("Conclude",
             "The linearization is %s: %s. Because no eigenvalue has zero real part, the "
             "nonlinear system behaves the same way near the point." % (_a(kind), stability)
             if hyperbolic else
             "An eigenvalue has zero real part, so the linear terms alone do not decide "
             "stability. This is exactly the situation Lyapunov's method is for."),
    ]
    if hyperbolic:
        methods.append(method("linear", "Linearization (for comparison)",
                              "The eigenvalues of the Jacobian settle the question when none "
                              "has zero real part.", lin_steps,
                              "\\text{The equilibrium } %s \\text{ is %s.}" % (where, stability)))
    else:
        methods.append(not_applicable("linear", "Linearization (for comparison)",
                                      "An eigenvalue of the Jacobian has zero real part, so "
                                      "linearization is inconclusive here."))

    marks = [{"x": float(point[0]), "y": float(point[1]), "label": "equilibrium"}]
    done = [m for m in methods if m["applicable"]]
    summary = ("Stability of the equilibrium (%s, %s) by Lyapunov's direct method."
               % (sp.sstr(point[0]), sp.sstr(point[1])))
    if not done:
        summary += " None of the tests here settles it."
    return result("dynamics", "Lyapunov stability",
                  "x' = %s,\\qquad y' = %s" % (tex(f), tex(g)), summary, methods,
                  **_field_extras(f, g, marks))


# --------------------------------------------------------------------------
# Conservative systems
# --------------------------------------------------------------------------

def _conservative(text):
    F = _expr(text, ["x"], "the force F(x)")
    U = run_with_timeout(lambda: sp.integrate(-F, X), TIMEOUT, None)
    if U is None or U.has(sp.Integral):
        raise InputError("The potential energy (minus the integral of F) has no closed form.")
    E = Y**2 / 2 + U
    roots = run_with_timeout(lambda: sp.solve(F, X), TIMEOUT, []) or []
    roots = sorted({r for r in roots if r.is_real and not r.free_symbols}, key=lambda r: float(r))[:8]
    steps = [
        step("Write as a system", "Let \\(y = x'\\) be the velocity.",
             ["x' = y,\\qquad y' = %s" % tex(F)]),
        step("Find the energy",
             "The potential energy is \\(U(x) = -\\int F\\,dx\\). The total energy is constant "
             "along every solution, because its time derivative is "
             "\\(y\\,y' + U'(x)\\,x' = y\\,F - F\\,y = 0\\).",
             ["U(x) = %s" % tex(U), "E = \\tfrac{1}{2}y^{2} + U(x) = %s" % tex(E)]),
    ]
    rows, marks = [], []
    for r in roots:
        curvature = sp.simplify(-sp.diff(F, X).subs(X, r))
        if curvature > 0:
            kind, why = "center", "minimum of U"
        elif curvature < 0:
            kind, why = "saddle point", "maximum of U"
        else:
            kind, why = "degenerate point", "U'' = 0"
        rows.append(["(%s,\\ 0)" % tex(r), tex(curvature), "\\text{%s}" % why, "\\text{%s}" % kind,
                     tex(sp.simplify(U.subs(X, r)))])
        marks.append({"x": float(r), "y": 0.0, "label": kind})
    eq_step = step(
        "Find and classify the equilibria",
        "Equilibria have \\(y = 0\\) and \\(F(x) = 0\\), the critical points of \\(U\\). A "
        "minimum of \\(U\\) is a center (closed orbits around it); a maximum is a saddle."
        if roots else "\\(F(x)\\) has no real zeros, so there are no equilibria.",
        ["F(x) = %s = 0" % tex(F)])
    if rows:
        eq_step["table"] = {"headers": ["\\text{point}", "U''", "\\text{so}", "\\text{type}", "E"],
                            "rows": rows}
    steps.append(eq_step)
    steps.append(step(
        "Read the phase portrait",
        "Every trajectory is a level curve \\(E = \\text{constant}\\). The level curves through "
        "the saddles (the separatrices) divide motion that oscillates from motion that escapes. "
        "Tap the Graph tab to draw them."))
    ok = sp.simplify(sp.diff(E, X) * Y + sp.diff(E, Y) * F) == 0
    methods = [method("energy", "Energy method",
                      "The force depends on position only, so energy is conserved.", steps,
                      "E = %s" % tex(E), verified=True if ok else None,
                      note="Checked that the energy is constant along solutions.")]
    more = "" if len(roots) < 8 and not F.has(sp.sin, sp.cos, sp.tan) else \
        " Only the equilibria SymPy lists are shown; a periodic force has more."
    return result("dynamics", "Conservative system", "x'' = %s" % tex(F),
                  "Conservative system with %d equilibrium point%s found.%s"
                  % (len(roots), "" if len(roots) == 1 else "s", more), methods,
                  **_field_extras(Y, F, marks, {"trajectories": "Level curves of the energy"}))


# --------------------------------------------------------------------------
# Limit cycles
# --------------------------------------------------------------------------

def _one_signed(expr):
    """'positive', 'negative' or None for a polynomial that is c + (even-power terms of one sign)."""
    try:
        poly = sp.Poly(sp.expand(expr), X, Y)
    except sp.PolynomialError:
        return None
    const = poly.coeff_monomial(1)
    if const == 0 or const.free_symbols:
        return None
    sign = 1 if const > 0 else -1
    for (i, j), c in poly.terms():
        if (i, j) == (0, 0):
            continue
        if i % 2 or j % 2 or c.free_symbols or (c > 0) != (sign > 0):
            return None
    return "positive" if sign > 0 else "negative"


def _polar(f, g):
    """(R(r), Omega) if r' depends on r only, else None. Omega may still depend on theta."""
    r = sp.Symbol("r", positive=True)
    th = sp.Symbol("theta", real=True)
    sub = {X: r * sp.cos(th), Y: r * sp.sin(th)}
    rdot = sp.simplify(sp.trigsimp(sp.expand(((X * f + Y * g) / r).subs(sub))))
    thdot = sp.simplify(sp.trigsimp(sp.expand(((X * g - Y * f) / r**2).subs(sub))))
    # Simplification can leave the angle in an expression that does not really
    # depend on it; test numerically and, if so, read the formula off at angle 0.
    rdot = _drop_angle(rdot, r, th)
    thdot = _drop_angle(thdot, r, th) or thdot
    if rdot is None:
        return None
    return r, th, rdot, thdot


def _drop_angle(expr, r, th):
    """expr as a function of r alone if it does not depend on the angle, else None."""
    if not expr.has(th):
        return expr
    fn = sp.lambdify((r, th), expr, "math")
    try:
        for radius in (0.4, 1.3, 2.7):
            base = fn(radius, 0.0)
            for angle in (0.7, 1.9, 3.3, 5.1):
                if abs(fn(radius, angle) - base) > 1e-9 * (1 + abs(base)):
                    return None
    except Exception:
        return None
    return sp.factor(sp.simplify(expr.subs(th, 0)))


def _cycles(text):
    f, g = _system(text)
    methods = []

    div = sp.simplify(sp.diff(f, X) + sp.diff(g, Y))
    sign = _one_signed(div)
    b_steps = [step("Compute the divergence",
                    "Bendixson's criterion: if \\(f_x + g_y\\) is not zero and keeps one sign "
                    "throughout a region without holes, no closed orbit lies entirely inside it.",
                    ["f_x + g_y = %s" % tex(div)])]
    if div == 0:
        b_steps.append(step("Conclude", "The divergence is identically zero, so the criterion "
                            "says nothing. (Zero divergence is typical of conservative systems, "
                            "which can have whole families of closed orbits.)"))
        answer = "\\text{Bendixson's criterion gives no information.}"
    elif sign:
        b_steps.append(step("Conclude", "The divergence is %s everywhere, so the system has no "
                            "closed orbits at all, and in particular no limit cycles." % sign))
        answer = "\\text{No closed orbits anywhere in the plane.}"
    else:
        b_steps.append(step(
            "Conclude",
            "The divergence changes sign across the curve \\(%s = 0\\). No closed orbit can lie "
            "entirely on one side of that curve, but one may cross it, so closed orbits are "
            "not ruled out." % tex(div)))
        answer = "\\text{Closed orbits are possible only if they cross } %s = 0." % tex(div)
    methods.append(method("bendixson", "Bendixson's criterion",
                          "A quick test that can rule closed orbits out.", b_steps, answer))

    polar = run_with_timeout(lambda: _polar(f, g), TIMEOUT, None)
    curve = None
    if polar is None:
        methods.append(not_applicable(
            "polar", "Polar coordinates",
            "In polar coordinates \\(r'\\) still depends on the angle, so the system does not "
            "reduce to a single equation for \\(r\\)."))
    else:
        r, th, rdot, thdot = polar
        p_steps = [step(
            "Change to polar coordinates",
            "Use \\(r\\,r' = x\\,x' + y\\,y'\\) and \\(r^{2}\\theta' = x\\,y' - y\\,x'\\) with "
            "\\(x = r\\cos\\theta\\), \\(y = r\\sin\\theta\\).",
            ["r' = %s" % tex(sp.factor(rdot)), "\\theta' = %s" % tex(thdot)])]
        radii = [v for v in (sp.solve(rdot, r) or []) if v.is_positive and not v.free_symbols]
        radii = sorted(set(radii), key=lambda v: float(v))
        rows = []
        for value in radii:
            slope = sp.simplify(sp.diff(rdot, r).subs(r, value))
            if slope < 0:
                kind = "stable"
            elif slope > 0:
                kind = "unstable"
            else:
                lo = sp.N(rdot.subs(r, value * sp.Rational(9, 10)))
                hi = sp.N(rdot.subs(r, value * sp.Rational(11, 10)))
                kind = "stable" if lo > 0 > hi else ("unstable" if lo < 0 < hi else "semi-stable")
            rows.append((value, kind))
        if rows and not thdot.has(th) and all(sp.simplify(thdot.subs(r, v)) != 0 for v, _ in rows):
            c_step = step(
                "Find the circular orbits",
                "A circle \\(r = r_0\\) is a closed orbit when \\(r' = 0\\) there and the angle "
                "keeps turning. It attracts nearby solutions when \\(r'\\) is positive just "
                "inside and negative just outside.",
                ["%s = 0" % tex(sp.factor(rdot))])
            c_step["table"] = {"headers": ["r_0", "\\text{limit cycle}", "\\text{direction}"],
                               "rows": [[tex(v), "\\text{%s}" % k,
                                         "\\text{counterclockwise}" if thdot.subs(r, v) > 0
                                         else "\\text{clockwise}"] for v, k in rows]}
            p_steps.append(c_step)
            answer = ",\\quad ".join("r = %s\\ \\text{(%s)}" % (tex(v), k) for v, k in rows)
            v0 = rows[0][0]
            curve = (v0 * sp.cos(T), v0 * sp.sin(T))
            ok = all(sp.simplify(rdot.subs(r, v)) == 0 for v, _ in rows)
            methods.append(method("polar", "Polar coordinates",
                                  "The radial equation separates from the angle, so circular "
                                  "limit cycles can be read off.", p_steps, answer,
                                  plot={"x_expr": _plot(curve[0], {T: "x"}),
                                        "y_expr": _plot(curve[1], {T: "x"})},
                                  verified=True if ok else None,
                                  note="Checked that r' = 0 on each circle found."))
        else:
            p_steps.append(step("Conclude", "\\(r'\\) has no positive zero where the angle keeps "
                                "turning, so there is no circular closed orbit."))
            methods.append(method("polar", "Polar coordinates",
                                  "The radial equation separates from the angle.", p_steps,
                                  "\\text{No circular limit cycle.}"))

    methods.append(_lienard(f, g))
    marks = [{"x": float(a), "y": float(b), "label": "equilibrium"} for a, b in _equilibria(f, g)]
    found = [m for m in methods if m["id"] == "polar" and m["applicable"] and curve is not None]
    summary = ("Limit cycles of a planar system. " + ("A circular limit cycle was found."
               if found else "Tap the phase plane to see how trajectories behave."))
    legends = {"exact": "Limit cycle"} if curve is not None else None
    return result("dynamics", "Limit cycles", "x' = %s,\\qquad y' = %s" % (tex(f), tex(g)),
                  summary, methods, **_field_extras(f, g, marks, legends))


def _lienard(f, g):
    name = "Li\u00e9nard's theorem"
    damping = sp.simplify(-sp.diff(g, Y))
    restoring = sp.simplify(-(g + damping * Y))
    if sp.simplify(f - Y) != 0 or damping.has(Y) or restoring.has(Y) or damping == 0:
        return not_applicable("lienard", name,
                              "It applies to equations \\(x'' + f(x)\\,x' + g(x) = 0\\), written "
                              "as \\(x' = y\\), \\(y' = -g(x) - f(x)\\,y\\). This system is not "
                              "of that form.")
    Fx = sp.integrate(damping, (X, 0, X))
    even = sp.simplify(damping.subs(X, -X) - damping) == 0
    odd = sp.simplify(restoring.subs(X, -X) + restoring) == 0
    pulls = all(sp.N(restoring.subs(X, v)) > 0 for v in (sp.Rational(1, 10), 1, 3, 10))
    zeros = sorted({z for z in (sp.solve(Fx, X) or []) if z.is_real and z > 0}, key=lambda z: float(z))
    shape = False
    if len(zeros) == 1:
        a = zeros[0]
        shape = bool(sp.N(Fx.subs(X, a / 2)) < 0 and sp.N(Fx.subs(X, 2 * a)) > 0
                     and sp.N(sp.diff(Fx, X).subs(X, 2 * a)) > 0
                     and sp.limit(Fx, X, sp.oo) == sp.oo)
    checks = [
        ["f\\ \\text{is even}", "\\text{yes}" if even else "\\text{no}"],
        ["g\\ \\text{is odd}", "\\text{yes}" if odd else "\\text{no}"],
        ["x\\,g(x) > 0\\ \\text{for } x \\ne 0", "\\text{yes}" if pulls else "\\text{no}"],
        ["F\\ \\text{has one positive zero, is negative before it and increases to } \\infty "
         "\\text{ after}", "\\text{yes}" if shape else "\\text{no}"],
    ]
    table = step("Check the hypotheses",
                 "Here \\(F(x) = \\int_0^x f(s)\\,ds\\).",
                 ["f(x) = %s,\\qquad g(x) = %s,\\qquad F(x) = %s" % (tex(damping), tex(restoring), tex(Fx))])
    table["table"] = {"headers": ["\\text{condition}", "\\text{holds?}"], "rows": checks}
    steps = [step("Write as a second-order equation",
                  "The system is \\(x'' + f(x)\\,x' + g(x) = 0\\).",
                  ["x'' + \\left(%s\\right)x' + %s = 0" % (tex(damping), tex(restoring))]), table]
    if even and odd and pulls and shape:
        steps.append(step("Conclude", "All of Li\u00e9nard's conditions hold, so the system has "
                          "exactly one closed orbit; it surrounds the origin and attracts every "
                          "other solution."))
        return method("lienard", name, "The system is a Li\u00e9nard equation.", steps,
                      "\\text{Exactly one limit cycle, and it is stable.}")
    steps.append(step("Conclude", "Not every condition holds, so the theorem does not apply. "
                      "This does not by itself rule a limit cycle out."))
    return method("lienard", name, "The system is a Li\u00e9nard equation.", steps,
                  "\\text{Li\u00e9nard's theorem gives no conclusion.}")


# --------------------------------------------------------------------------
# Bifurcations of x' = f(x, m)
# --------------------------------------------------------------------------

def _bifurcation_type(f, x0, m0):
    at = {X: x0, M: m0}
    d = lambda *v: sp.simplify(sp.diff(f, *v).subs(at))  # noqa: E731
    fm, fxx, fxm, fmm, fxxx = d(M), d(X, 2), d(X, M), d(M, 2), d(X, 3)
    if fm != 0 and fxx != 0:
        return "saddle-node", "f_m \\ne 0,\\ f_{xx} \\ne 0"
    if fm == 0 and fxx != 0 and sp.simplify(fxm**2 - fxx * fmm) > 0:
        return "transcritical", "f_m = 0,\\ f_{xx} \\ne 0,\\ f_{xm}^2 - f_{xx}f_{mm} > 0"
    if fm == 0 and fxx == 0 and fxm != 0 and fxxx != 0:
        kind = "supercritical pitchfork" if fxxx < 0 else "subcritical pitchfork"
        return kind, "f_m = 0,\\ f_{xx} = 0,\\ f_{xm} \\ne 0,\\ f_{xxx} %s 0" % ("<" if fxxx < 0 else ">")
    return "degenerate (not one of the standard types)", "\\text{the standard conditions fail}"


def _polish(f_num, fx_num, xv, mv):
    """A couple of Newton steps so a sampled root sits on the branch."""
    for _ in range(3):
        try:
            slope = fx_num(xv, mv)
            if abs(slope) < 1e-9:
                break
            xv = xv - f_num(xv, mv) / slope
        except Exception:
            break
    return xv


def _bifurcation(text):
    f = _expr(text, ["x", "m"], "f(x, m)")
    if not f.has(X):
        raise InputError("f must depend on x.")
    if not f.has(M):
        raise InputError("f must contain the parameter, written m (or mu).")
    branches = run_with_timeout(lambda: sp.solve(f, X), TIMEOUT, None)
    if not branches:
        raise InputError("The equilibria f(x, m) = 0 could not be found in closed form.")
    branches = [b for b in branches if not b.has(sp.I)][:6]
    fx = sp.diff(f, X)
    steps = [step("Find the equilibria", "Solve \\(f(x, m) = 0\\) for \\(x\\). Each solution is a "
                  "branch of equilibria that depends on the parameter.",
                  ["%s = 0" % tex(f)] + ["x = %s" % tex(b) for b in branches])]
    rows = []
    for b in branches:
        slope = sp.simplify(fx.subs(X, b))
        rows.append([tex(b), tex(slope)])
    stab = step("Test stability on each branch",
                "An equilibrium is stable where \\(f_x < 0\\) and unstable where \\(f_x > 0\\).",
                ["f_x = %s" % tex(fx)])
    stab["table"] = {"headers": ["\\text{branch } x", "f_x \\text{ on it}"], "rows": rows}
    steps.append(stab)

    found = run_with_timeout(lambda: sp.solve([f, fx], [X, M], dict=True), TIMEOUT, []) or []
    points = sorted({(s[X], s[M]) for s in found if X in s and M in s
                     and s[X].is_real and s[M].is_real and not (s[X].free_symbols or s[M].free_symbols)},
                    key=lambda p: (float(p[1]), float(p[0])))[:6]
    if points:
        rows = []
        for x0, m0 in points:
            kind, why = _bifurcation_type(f, x0, m0)
            rows.append([tex(m0), tex(x0), "\\text{%s}" % kind, why])
        bif = step("Locate and classify the bifurcations",
                   "The number or stability of equilibria can change only where \\(f = 0\\) and "
                   "\\(f_x = 0\\) together. The type follows from the next derivatives there.",
                   ["f = 0,\\qquad f_x = 0"])
        bif["table"] = {"headers": ["m", "x", "\\text{type}", "\\text{because}"], "rows": rows}
        steps.append(bif)
        answer = [",\\quad ".join("m = %s:\\ \\text{%s}" % (tex(m0), _bifurcation_type(f, x0, m0)[0])
                                  for x0, m0 in points)]
    else:
        steps.append(step("Locate the bifurcations", "\\(f = 0\\) and \\(f_x = 0\\) have no common "
                          "real solution, so the equilibria never change number or stability."))
        answer = ["\\text{No bifurcation.}"]

    # Sample the equilibria for the diagram.
    centre = float(points[0][1]) if points else 0.0
    marks = []
    f_num = sp.lambdify((X, M), f, "math")
    fx_num = sp.lambdify((X, M), fx, "math")
    is_poly = f.is_polynomial(X)
    branch_fns = [sp.lambdify(M, b, "math") for b in branches]
    for k in range(-12, 13):
        mv = centre + k * 0.25
        values = []
        if is_poly:
            try:
                roots = sp.Poly(f.subs(M, sp.Rational(mv).limit_denominator(1000)), X).nroots()
                values = [float(sp.re(z)) for z in roots if abs(sp.im(z)) < 1e-9]
            except Exception:
                values = []
        else:
            for fn in branch_fns:
                try:
                    xv = fn(mv)
                except Exception:
                    continue
                if not isinstance(xv, complex) and xv == xv:
                    values.append(float(xv))
        for xv in values:
            if abs(xv) > 50:
                continue
            xv = _polish(f_num, fx_num, xv, mv)
            try:
                sv = fx_num(xv, mv)
            except Exception:
                continue
            marks.append({"x": mv, "y": xv,
                          "label": "stable" if sv < -1e-9 else ("unstable" if sv > 1e-9 else "neutral")})
    ok = all(abs(f_num(e["y"], e["x"])) < 1e-7 for e in marks) if marks else None
    field_g = f.subs({X: Y, M: X}, simultaneous=True)
    methods = [method("branches", "Equilibrium branches",
                      "Bifurcations are found by following the equilibria as the parameter changes.",
                      steps, answer, verified=True if ok else None,
                      note="Checked that f = 0 at every plotted equilibrium.")]
    extra = {"graph": "phase", "field": {"f": "0*x", "g": _plot(field_g, _NAMES)},
             "equilibria": marks[:120],
             "legends": {"trajectories": "Motion of x at fixed m", "equilibria": "Equilibria",
                         "hint": "Across: the parameter m. Up: x. The marks sit on the branches "
                                 "of equilibria; the arrows show which way x moves, so a branch "
                                 "the arrows point toward is stable."}}
    return result("dynamics", "Bifurcation", "x' = %s" % tex(f),
                  "Bifurcations of a one-parameter equation; the parameter is m.", methods, **extra)


# --------------------------------------------------------------------------
# Hopf bifurcation
# --------------------------------------------------------------------------

def _imag(w):
    return "i" if w == 1 else "%s\\,i" % tex(w)


def _hopf(text, m_text):
    f, g = _system(text, ("x", "y", "m"))
    if not (f.has(M) or g.has(M)):
        raise InputError("The system must contain the parameter, written m (or mu).")
    found = run_with_timeout(lambda: sp.solve([f, g], [X, Y], dict=True), TIMEOUT, []) or []
    eqs = [(s[X], s[Y]) for s in found if X in s and Y in s and not (s[X].has(sp.I) or s[Y].has(sp.I))][:3]
    if not eqs:
        raise InputError("No equilibrium could be found in closed form.")
    J = _jacobian(f, g)
    steps = [step("Find the equilibrium", "Solve \\(x' = 0\\), \\(y' = 0\\).",
                  ["(x, y) = (%s,\\ %s)" % (tex(a), tex(b)) for a, b in eqs])]
    hits = []
    for a, b in eqs:
        Jm = J.subs({X: a, Y: b}).applyfunc(sp.simplify)
        tr, det = sp.simplify(Jm.trace()), sp.simplify(Jm.det())
        steps.append(step(
            "Linearize at (%s, %s)" % (sp.sstr(a), sp.sstr(b)),
            "The eigenvalues are \\(\\lambda = \\tfrac{1}{2}\\left(\\tau \\pm "
            "\\sqrt{\\tau^{2} - 4\\Delta}\\right)\\) with trace \\(\\tau\\) and determinant "
            "\\(\\Delta\\). They are purely imaginary when \\(\\tau = 0\\) and \\(\\Delta > 0\\).",
            ["J = %s" % _mat(Jm), "\\tau = %s,\\qquad \\Delta = %s" % (tex(tr), tex(det))]))
        for m0 in (sp.solve(tr, M) or []) if tr.has(M) else []:
            if not m0.is_real:
                continue
            d0 = sp.simplify(det.subs(M, m0))
            cross = sp.simplify(sp.diff(tr, M).subs(M, m0))
            if d0 > 0 and cross != 0:
                hits.append((a, b, m0, sp.sqrt(d0), cross))
    if not hits:
        steps.append(step("Conclude", "There is no parameter value at which the trace vanishes "
                          "with a positive determinant and a non-zero rate of change, so no "
                          "Hopf bifurcation occurs at these equilibria."))
        answer = "\\text{No Hopf bifurcation.}"
    else:
        rows = [[tex(m0), "\\pm %s" % _imag(w), tex(c),
                 "\\text{stable} \\to \\text{unstable}" if c > 0 else "\\text{unstable} \\to \\text{stable}"]
                for _, _, m0, w, c in hits]
        st = step("Find where the eigenvalues cross the imaginary axis",
                  "Set \\(\\tau = 0\\), check that \\(\\Delta > 0\\) there, and that the real part "
                  "\\(\\tau/2\\) really changes sign (\\(d\\tau/dm \\ne 0\\)).")
        st["table"] = {"headers": ["m", "\\lambda", "d\\tau/dm", "\\text{as } m \\text{ increases}"],
                       "rows": rows}
        steps.append(st)
        steps.append(step(
            "Conclude",
            "A pair of complex eigenvalues crosses the imaginary axis, so a Hopf bifurcation "
            "occurs: a family of periodic orbits branches from the equilibrium. Linearization "
            "alone does not tell whether the orbits are stable (supercritical) or unstable "
            "(subcritical), or on which side of the critical value they appear."))
        answer = ",\\quad ".join("m = %s\\ (\\lambda = \\pm %s)" % (tex(m0), _imag(w))
                                 for _, _, m0, w, _ in hits)

    # Direction, when the system reduces to a radial equation.
    curve = None
    m_value = parse_number(m_text, "The value of m") if m_text.strip() else (
        hits[0][2] + sp.Rational(1, 2) if hits else sp.Integer(1))
    polar = run_with_timeout(lambda: _polar(f, g), TIMEOUT, None) if hits else None
    if polar is not None:
        r, th, rdot, thdot = polar
        radii = [v for v in (sp.solve(rdot, r) or []) if v.has(M)]
        if radii:
            rad = radii[0]
            slope = sp.simplify(sp.diff(rdot, r).subs(r, rad))
            steps.append(step(
                "Direction, from polar coordinates",
                "This system reduces to a radial equation, which shows the periodic orbit "
                "directly and decides its stability.",
                ["r' = %s" % tex(sp.factor(rdot)), "r = %s" % tex(rad),
                 "\\left.\\frac{\\partial r'}{\\partial r}\\right|_{r = %s} = %s" % (tex(rad), tex(slope))]))
            rv = rad.subs(M, m_value)
            if rv.is_real and rv > 0:
                curve = (rv * sp.cos(T), rv * sp.sin(T))
    fm, gm = f.subs(M, m_value), g.subs(M, m_value)
    marks = [{"x": float(a.subs(M, m_value)), "y": float(b.subs(M, m_value)), "label": "equilibrium"}
             for a, b in eqs if a.subs(M, m_value).is_real and b.subs(M, m_value).is_real]
    plot = {"x_expr": _plot(curve[0], {T: "x"}), "y_expr": _plot(curve[1], {T: "x"})} if curve else None
    methods = [method("hopf", "Trace and determinant",
                      "A Hopf bifurcation happens where a complex pair of eigenvalues crosses "
                      "the imaginary axis.", steps, answer, plot=plot)]
    legends = {"hint": "The phase plane is drawn for m = %s. Tap to draw trajectories." % sp.sstr(m_value)}
    if curve:
        legends["exact"] = "Periodic orbit"
    return result("dynamics", "Hopf bifurcation", "x' = %s,\\qquad y' = %s" % (tex(f), tex(g)),
                  "Hopf bifurcation of a planar system with parameter m. The Graph tab uses "
                  "m = %s." % sp.sstr(m_value), methods, **_field_extras(fm, gm, marks, legends))


def solve(operation, text, extra1="", extra2=""):
    if operation == "lyapunov":
        return _lyapunov(text, extra1, extra2)
    if operation == "conservative":
        return _conservative(text)
    if operation == "cycles":
        return _cycles(text)
    if operation == "bifurcation":
        return _bifurcation(text)
    if operation == "hopf":
        return _hopf(text, extra1)
    raise InputError("Unknown operation.")


def preview(operation, text):
    if operation == "conservative":
        return "x'' = %s" % tex(_expr(text, ["x"], "F(x)"))
    if operation == "bifurcation":
        return "x' = %s" % tex(_expr(text, ["x", "m"], "f(x, m)"))
    names = ("x", "y", "m") if operation == "hopf" else ("x", "y")
    parts = (_clean(text).replace("\n", ";").split(";") + [""])[:2]
    lines = []
    for label, part in zip(("x'", "y'"), parts):
        if part.strip():
            lines.append("%s = %s" % (label, tex(_expr(part, list(names), label))))
    return ",\\qquad ".join(lines)
