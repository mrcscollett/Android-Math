"""Vector calculus: gradient, divergence and curl, potentials, line integrals,
double integrals, Green's theorem, and flux / circulation over standard surfaces."""
import re

import sympy as sp

from .integrals import explain
from .parsing import InputError, parse_expression, parse_number
from .systems import _plot
from .toolkit import method, not_applicable, result, step
from .util import TIMEOUT, run_with_timeout, tex

OPERATIONS = ("gradient", "divcurl", "line", "double", "polar", "green", "flux", "stokes")

X = sp.Symbol("x", real=True)
Y = sp.Symbol("y", real=True)
Z = sp.Symbol("z", real=True)
T = sp.Symbol("t", real=True)
_XYZ = (X, Y, Z)
_NAMES = {X: "x", Y: "y"}
_SYMBOL = {"x": X, "y": Y, "z": Z, "t": T}


def _expr(text, names, what):
    if not text.strip():
        raise InputError("Enter %s." % what)
    expr, symbols = parse_expression(text, list(names))
    expr = expr.xreplace({symbols[n]: _SYMBOL[n] for n in names})
    if expr.has(sp.Heaviside, sp.DiracDelta):
        raise InputError("Steps and impulses are not supported here.")
    return expr


def _split(text):
    """Components separated by semicolons, new lines, or commas outside parentheses."""
    parts, depth, current = [], 0, ""
    for ch in text.replace("\n", ";"):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == ";" or (ch == "," and depth == 0):
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    return [p for p in parts if p.strip()]


def _components(text, names, what, sizes=(2, 3)):
    parts = _split(text)
    if len(parts) not in sizes:
        raise InputError("%s needs %s components separated by semicolons, for example  %s."
                         % (what, " or ".join(str(n) for n in sizes),
                            "y; -x" if 2 in sizes else "x; y; z"))
    return [_expr(p, names, what) for p in parts]


def _numbers(text, count, label):
    parts = _split(text)
    if len(parts) != count:
        raise InputError("%s needs %d number%s separated by commas." % (label, count, "" if count == 1 else "s"))
    return [parse_number(p, label) for p in parts]


def _vec(entries):
    return "\\left\\langle %s \\right\\rangle" % ",\\ ".join(tex(e) for e in entries)


def _integral(expr, var, lo=None, hi=None):
    """Integrate with a time limit; raises InputError when there is no closed form."""
    args = (var,) if lo is None else ((var, lo, hi),)
    value = run_with_timeout(lambda: sp.integrate(expr, *args), TIMEOUT, None)
    if value is None or value.has(sp.Integral):
        raise InputError("An integral in this problem has no closed form: the integral of %s "
                         "with respect to %s." % (sp.sstr(expr), var))
    return sp.simplify(value)


def _valued(m, value):
    """Attach the numerical value of an integral to its method (read by tests and tools)."""
    try:
        m["value"] = float(sp.N(value))
    except Exception:
        m["value"] = None
    return m


def _close(a, b):
    try:
        return abs(complex(sp.N(a - b))) < 1e-9 * (1 + abs(complex(sp.N(b))))
    except Exception:
        return False


# --------------------------------------------------------------------------
# Scalar fields
# --------------------------------------------------------------------------

def _gradient(text, point_text, direction_text):
    names = ["x", "y", "z"] if re.search(r"(?<![A-Za-z])z(?![A-Za-z])", text) else ["x", "y"]
    f = _expr(text, names, "f")
    variables = [_SYMBOL[n] for n in names]
    grad = [sp.simplify(sp.diff(f, v)) for v in variables]
    lap = sp.simplify(sum(sp.diff(f, v, 2) for v in variables))
    letters = ", ".join(names)
    steps = [
        step("Take the partial derivatives",
             "Differentiate with respect to each variable in turn, holding the others fixed.",
             ["f_%s = %s" % (n, tex(g)) for n, g in zip(names, grad)]),
        step("Form the gradient",
             "The gradient points in the direction in which \\(f\\) increases fastest; its "
             "length is that fastest rate.", ["\\nabla f = %s" % _vec(grad)]),
        step("Laplacian", "Add the unmixed second derivatives.",
             ["\\nabla^{2} f = %s" % " + ".join("f_{%s%s}" % (n, n) for n in names),
              "\\nabla^{2} f = %s" % tex(lap)]),
    ]
    answers = ["\\nabla f = %s" % _vec(grad), "\\nabla^{2} f = %s" % tex(lap)]
    extra = {}
    point = None
    if point_text.strip():
        point = _numbers(point_text, len(names), "The point")
        at = dict(zip(variables, point))
        g0 = [sp.simplify(g.subs(at)) for g in grad]
        f0 = sp.simplify(f.subs(at))
        if any(v.has(sp.zoo, sp.nan, sp.oo) for v in g0 + [f0]):
            raise InputError("f is not differentiable at that point.")
        where = "(%s)" % ",\\ ".join(tex(p) for p in point)
        steps.append(step("Evaluate at the point", "",
                          ["f%s = %s" % (where, tex(f0)), "\\nabla f%s = %s" % (where, _vec(g0))]))
        linear = f0 + sum(g * (v - p) for g, v, p in zip(g0, variables, point))
        if len(names) == 2:
            steps.append(step(
                "Tangent plane",
                "The plane \\(z = f(a,b) + f_x(a,b)(x-a) + f_y(a,b)(y-b)\\) touches the surface "
                "\\(z = f(x, y)\\) at the point and is the best linear approximation near it.",
                ["z = %s" % tex(sp.expand(linear))]))
            answers.append("z = %s" % tex(sp.expand(linear)))
        else:
            steps.append(step("Linear approximation", "Near the point,",
                              ["f(%s) \\approx %s" % (letters, tex(sp.expand(linear)))]))
        if direction_text.strip():
            d = _numbers(direction_text, len(names), "The direction")
            norm = sp.sqrt(sum(c**2 for c in d))
            if norm == 0:
                raise InputError("The direction cannot be the zero vector.")
            unit = [sp.simplify(c / norm) for c in d]
            rate = sp.simplify(sum(g * u for g, u in zip(g0, unit)))
            steps.append(step(
                "Directional derivative",
                "Divide the direction by its length to get a unit vector \\(\\mathbf{u}\\), "
                "then take the dot product with the gradient.",
                ["\\mathbf{u} = %s" % _vec(unit),
                 "D_{\\mathbf{u}} f = \\nabla f \\cdot \\mathbf{u} = %s" % tex(rate)]))
            answers.append("D_{\\mathbf{u}} f%s = %s" % (where, tex(rate)))
    elif direction_text.strip():
        raise InputError("Give the point as well: a directional derivative is taken at a point.")

    plot = None
    if len(names) == 2:
        expr = _plot(f, _NAMES)
        if expr:
            cx, cy = (float(point[0]), float(point[1])) if point else (0.0, 0.0)
            fn = sp.lambdify((X, Y), f, "math")
            values = []
            for i in range(21):
                for j in range(21):
                    try:
                        v = fn(cx - 3 + 0.3 * i, cy - 3 + 0.3 * j)
                    except Exception:
                        continue
                    if isinstance(v, (int, float)) and v == v and abs(v) < 1e12:
                        values.append(float(v))
            if values:
                values.sort()
                lo, hi = values[len(values) // 50], values[-1 - len(values) // 50]
                if hi - lo < 1e-9:
                    lo, hi = lo - 1, hi + 1
                plot = {"particular_expr": expr}
                extra = {"graph": "surface", "surface": True,
                         "domain": {"x_min": cx - 3, "x_max": cx + 3, "y_min": cy - 3, "y_max": cy + 3,
                                    "t_max": 0.0, "u_min": lo, "u_max": hi}}
    methods = [method("partials", "Partial derivatives",
                      "Everything here comes from the first and second partial derivatives.",
                      steps, answers, plot=plot)]
    return result("vector", "Gradient", "f(%s) = %s" % (letters, tex(f)),
                  "Gradient and Laplacian of a scalar field.", methods, **extra)


# --------------------------------------------------------------------------
# Vector fields: divergence, curl, potential
# --------------------------------------------------------------------------

def _field(text, what="The field"):
    parts = _split(text)
    names = ["x", "y", "z"] if len(parts) == 3 else ["x", "y"]
    comps = _components(text, names, what)
    return comps, [_SYMBOL[n] for n in names], names


def _curl(F, variables):
    if len(F) == 2:
        return sp.simplify(sp.diff(F[1], X) - sp.diff(F[0], Y))
    P, Q, R = F
    return [sp.simplify(sp.diff(R, Y) - sp.diff(Q, Z)), sp.simplify(sp.diff(P, Z) - sp.diff(R, X)),
            sp.simplify(sp.diff(Q, X) - sp.diff(P, Y))]


def _is_conservative(F, variables):
    curl = _curl(F, variables)
    return curl == 0 if len(F) == 2 else all(c == 0 for c in curl)


def _potential(F, variables, steps=None):
    """A potential phi with grad phi = F, built by partial integration. None if it fails."""
    names = [str(v) for v in variables]
    try:
        phi = _integral(F[0], variables[0])
    except InputError:
        return None
    if steps is not None:
        rest = ", ".join(names[1:])
        steps.append(step(
            "Integrate the first component",
            "\\(\\phi_x = P\\), so integrate \\(P\\) with respect to \\(x\\). The 'constant' "
            "may depend on the other variable%s." % ("s" if len(names) == 3 else ""),
            ["\\phi = \\int %s\\,dx = %s + g(%s)" % (tex(F[0]), tex(phi), rest)]))
    for index in range(1, len(variables)):
        v = variables[index]
        missing = sp.simplify(F[index] - sp.diff(phi, v))
        if any(missing.has(u) for u in variables[:index]):
            return None
        try:
            piece = _integral(missing, v)
        except InputError:
            return None
        if steps is not None:
            steps.append(step(
                "Match the %s component" % ("second" if index == 1 else "third"),
                "Differentiate \\(\\phi\\) with respect to \\(%s\\) and compare with the "
                "component; what is left over is integrated in \\(%s\\)." % (names[index], names[index]),
                ["%s - \\phi_%s = %s" % (tex(F[index]), names[index], tex(missing)),
                 "\\text{add } %s" % tex(piece)]))
        phi = sp.simplify(phi + piece)
    if any(sp.simplify(sp.diff(phi, v) - c) != 0 for v, c in zip(variables, F)):
        return None
    return phi


def _divcurl(text):
    F, variables, names = _field(text)
    div = sp.simplify(sum(sp.diff(c, v) for c, v in zip(F, variables)))
    curl = _curl(F, variables)
    labels = "PQR"[:len(F)]
    steps = [
        step("Divergence", "Differentiate each component with respect to its own variable and add.",
             ["\\nabla \\cdot \\mathbf{F} = %s" % " + ".join("%s_%s" % (l, n) for l, n in zip(labels, names)),
              "\\nabla \\cdot \\mathbf{F} = %s" % tex(div)]),
    ]
    if len(F) == 2:
        steps.append(step("Curl", "In the plane the curl has one component, \\(Q_x - P_y\\).",
                          ["Q_x - P_y = %s" % tex(curl)]))
        curl_tex = tex(curl)
    else:
        steps.append(step(
            "Curl", "Expand the determinant with rows \\(\\mathbf{i}, \\mathbf{j}, \\mathbf{k}\\); "
            "the partial derivative operators; and the components.",
            ["\\nabla \\times \\mathbf{F} = \\left\\langle R_y - Q_z,\\ P_z - R_x,\\ Q_x - P_y \\right\\rangle",
             "\\nabla \\times \\mathbf{F} = %s" % _vec(curl)]))
        curl_tex = _vec(curl)
    methods = [method("operators", "Divergence and curl",
                      "Both are combinations of the first partial derivatives of the components.",
                      steps, ["\\nabla \\cdot \\mathbf{F} = %s" % tex(div),
                              "\\nabla \\times \\mathbf{F} = %s" % curl_tex])]
    name = "Potential function"
    if _is_conservative(F, variables):
        p_steps = [step(
            "Test whether the field is conservative",
            "The curl is zero. On a region without holes (simply connected) that means "
            "\\(\\mathbf{F} = \\nabla\\phi\\) for some potential \\(\\phi\\). Finding \\(\\phi\\) is "
            "the same procedure as solving an exact differential equation.",
            ["\\nabla \\times \\mathbf{F} = %s" % ("0" if len(F) == 2 else "\\mathbf{0}")])]
        phi = _potential(F, variables, p_steps)
        if phi is None:
            methods.append(not_applicable(name.lower().replace(" ", "_"), name,
                                          "The curl is zero, but the integrals for the potential "
                                          "have no closed form."))
        else:
            methods.append(method("potential", name, "The curl of the field is zero.", p_steps,
                                  "\\phi = %s + C" % tex(phi), verified=True,
                                  note="Checked by differentiating: the gradient of the potential "
                                       "is the field."))
    else:
        methods.append(not_applicable("potential", name,
                                      "The curl is not zero, so the field is not conservative "
                                      "and has no potential function."))
    extra = {}
    if len(F) == 2:
        extra = {"graph": "phase", "field": {"f": _plot(F[0], _NAMES), "g": _plot(F[1], _NAMES)},
                 "equilibria": [],
                 "legends": {"trajectories": "Field lines",
                             "hint": "The arrows show the direction of the field. Tap to draw the "
                                     "field line through a point."}}
    return result("vector", "Divergence and curl", "\\mathbf{F} = %s" % _vec(F),
                  "Divergence, curl and (when the curl is zero) a potential function.", methods, **extra)


# --------------------------------------------------------------------------
# Line integrals
# --------------------------------------------------------------------------

def _clamp(expr, a, b):
    """Hold the curve at its end points outside [a, b] so the drawn curve stops there."""
    held = (sp.Abs(T - a) - sp.Abs(T - b) + a + b) / 2
    return expr.subs(T, held)


def _line(text, curve_text, limits_text):
    parts = _split(text)
    curve = _components(curve_text, ["t"], "The curve r(t)") if curve_text.strip() else None
    if curve is None:
        raise InputError("Enter the curve as its components in t, for example  cos(t); sin(t).")
    a, b = _numbers(limits_text, 2, "The limits for t") if limits_text.strip() else (None, None)
    if a is None:
        raise InputError("Enter the two limits for t, for example  0, pi.")
    dim = len(curve)
    names = ["x", "y", "z"][:dim]
    variables = [_SYMBOL[n] for n in names]
    on_curve = dict(zip(variables, curve))
    velocity = [sp.simplify(sp.diff(c, T)) for c in curve]
    curve_tex = "\\mathbf{r}(t) = %s,\\quad %s \\le t \\le %s" % (_vec(curve), tex(a), tex(b))
    extra = {}
    if dim == 2:
        extra = {"graph": "phase", "equilibria": [],
                 "legends": {"exact": "The curve r(t)", "trajectories": "Field lines",
                             "hint": "The curve of integration is drawn over the field."}}
    plot = None
    if dim == 2:
        plot = {"x_expr": _plot(_clamp(curve[0], a, b), {T: "x"}),
                "y_expr": _plot(_clamp(curve[1], a, b), {T: "x"})}

    if len(parts) == 1:
        f = _expr(parts[0], names, "f")
        speed = sp.simplify(sp.sqrt(sum(v**2 for v in velocity)))
        integrand = sp.simplify(f.subs(on_curve) * speed)
        value = _integral(integrand, T, a, b)
        steps = [
            step("Parametrise", "Along the curve, \\(ds = |\\mathbf{r}'(t)|\\,dt\\).",
                 [curve_tex, "\\mathbf{r}'(t) = %s" % _vec(velocity), "|\\mathbf{r}'(t)| = %s" % tex(speed)]),
            step("Substitute", "Replace %s by the components of the curve."
                 % ", ".join("\\(%s\\)" % n for n in names),
                 ["\\int_C f\\,ds = \\int_{%s}^{%s} %s\\,dt" % (tex(a), tex(b), tex(integrand))],
                 explain(integrand, T, "The integral")),
            step("Evaluate", "", ["\\int_C f\\,ds = %s" % tex(value)]),
        ]
        ok = _check_quad(integrand, a, b, value)
        methods = [_valued(method("parametrise", "Parametrise the curve",
                                  "A line integral becomes an ordinary integral in the parameter.", steps,
                                  "\\int_C f\\,ds = %s" % tex(value), plot=plot, verified=ok,
                                  note="Agrees with numerical integration." if ok else
                                       "Could not be checked numerically."), value)]
        if dim == 2:
            extra["field"] = None
        return result("vector", "Line integral", "\\int_C %s\\,ds" % tex(f),
                      "Line integral of a scalar function along a curve.", methods, **extra)

    if len(parts) != dim:
        raise InputError("The field has %d components but the curve has %d. They must match."
                         % (len(parts), dim))
    F = [_expr(p, names, "The field") for p in parts]
    integrand = sp.simplify(sum(c.subs(on_curve) * v for c, v in zip(F, velocity)))
    value = _integral(integrand, T, a, b)
    steps = [
        step("Parametrise", "Along the curve, \\(d\\mathbf{r} = \\mathbf{r}'(t)\\,dt\\).",
             [curve_tex, "\\mathbf{r}'(t) = %s" % _vec(velocity)]),
        step("Take the dot product", "Evaluate the field on the curve and dot it with the velocity.",
             ["\\mathbf{F}(\\mathbf{r}(t)) = %s" % _vec([sp.simplify(c.subs(on_curve)) for c in F]),
              "\\mathbf{F} \\cdot \\mathbf{r}' = %s" % tex(integrand)]),
        step("Integrate", "This is the work done by the field along the curve.",
             ["\\int_C \\mathbf{F} \\cdot d\\mathbf{r} = \\int_{%s}^{%s} %s\\,dt = %s"
              % (tex(a), tex(b), tex(integrand), tex(value))],
             explain(integrand, T, "The integral")),
    ]
    ok = _check_quad(integrand, a, b, value)
    answer = "\\int_C \\mathbf{F} \\cdot d\\mathbf{r} = %s" % tex(value)
    methods = [_valued(method("parametrise", "Parametrise the curve",
                              "A line integral becomes an ordinary integral in the parameter.", steps,
                              answer, plot=plot, verified=ok,
                              note="Agrees with numerical integration." if ok else
                                   "Could not be checked numerically."), value)]
    name = "Fundamental theorem"
    if _is_conservative(F, variables):
        f_steps = [step("Check the field", "Its curl is zero, so it has a potential and the "
                        "integral depends only on the end points of the curve.")]
        phi = _potential(F, variables, f_steps)
        if phi is None:
            methods.append(not_applicable("fundamental", name, "The potential has no closed form."))
        else:
            start = [sp.simplify(c.subs(T, a)) for c in curve]
            end = [sp.simplify(c.subs(T, b)) for c in curve]
            v2 = sp.simplify(phi.subs(dict(zip(variables, end))) - phi.subs(dict(zip(variables, start))))
            f_steps.append(step(
                "Evaluate the potential at the end points",
                "\\(\\int_C \\nabla\\phi \\cdot d\\mathbf{r} = \\phi(\\text{end}) - \\phi(\\text{start})\\).",
                ["\\phi = %s" % tex(phi),
                 "\\phi(%s) - \\phi(%s) = %s" % (",\\ ".join(tex(e) for e in end),
                                                ",\\ ".join(tex(e) for e in start), tex(v2))]))
            methods.append(_valued(method(
                "fundamental", name, "The field is conservative.", f_steps,
                "\\int_C \\mathbf{F} \\cdot d\\mathbf{r} = %s" % tex(v2), plot=plot,
                verified=_close(v2, value),
                note="Agrees with the direct calculation." if _close(v2, value)
                else "Does not agree with the direct calculation."), v2))
    else:
        methods.append(not_applicable("fundamental", name,
                                      "The field is not conservative (its curl is not zero), so "
                                      "the integral depends on the path."))
    if dim == 2:
        extra["field"] = {"f": _plot(F[0], _NAMES), "g": _plot(F[1], _NAMES)}
    return result("vector", "Line integral", "\\int_C %s \\cdot d\\mathbf{r}" % _vec(F),
                  "Line integral of a vector field along a curve (work).", methods, **extra)


def _check_quad(integrand, a, b, value):
    """Compare a one-variable integral with numerical quadrature."""
    try:
        import mpmath
        fn = sp.lambdify(T, integrand, "mpmath")
        ref = mpmath.quad(fn, [float(a), float(b)])
        return True if abs(complex(sp.N(value)) - complex(ref)) < 1e-7 * (1 + abs(complex(ref))) else False
    except Exception:
        return None


# --------------------------------------------------------------------------
# Double integrals
# --------------------------------------------------------------------------

def _limits(text, names, label):
    parts = _split(text)
    if len(parts) != 2:
        raise InputError("%s needs a lower and an upper limit separated by a comma." % label)
    return [_expr(p, names, label) for p in parts]


def _double(text, x_text, y_text):
    f = _expr(text, ["x", "y"], "f(x, y)")
    if not x_text.strip() or not y_text.strip():
        raise InputError("Enter the limits for x (two numbers) and for y (numbers or "
                         "functions of x), for example  0, 1  and  0, x.")
    a, b = _numbers(x_text, 2, "The limits for x")
    lo, hi = _limits(y_text, ["x"], "The limits for y")
    inner = _integral(f, Y, lo, hi)
    value = _integral(inner, X, a, b)
    steps = [
        step("Set up the iterated integral",
             "Integrate over \\(y\\) first, between the lower and upper curves, then over \\(x\\).",
             ["\\int_{%s}^{%s}\\int_{%s}^{%s} %s\\,dy\\,dx" % (tex(a), tex(b), tex(lo), tex(hi), tex(f))]),
        step("Inner integral", "Treat \\(x\\) as a constant.",
             ["\\int_{%s}^{%s} %s\\,dy = %s" % (tex(lo), tex(hi), tex(f), tex(inner))],
             explain(f, Y, "The inner integral")),
        step("Outer integral", "",
             ["\\int_{%s}^{%s} %s\\,dx = %s" % (tex(a), tex(b), tex(inner), tex(value))],
             explain(inner, X, "The outer integral")),
    ]
    ok = _check_quad2(f, X, a, b, Y, lo, hi, value)
    answer = "\\iint_R %s\\,dA = %s" % (_wrap(f), tex(value))
    methods = [_valued(method("iterated", "Iterated integral (dy dx)",
                              "The region lies between two curves in \\(y\\) for each \\(x\\).", steps,
                              answer, verified=ok, note="Agrees with numerical integration." if ok else
                              "Could not be checked numerically."), value)]
    name = "Other order (dx dy)"
    if not lo.has(X) and not hi.has(X):
        inner2 = _integral(f, X, a, b)
        v2 = _integral(inner2, Y, lo, hi)
        methods.append(method("swapped", name,
                              "On a rectangle the order of integration can be swapped (Fubini).",
                              [step("Integrate over x first", "",
                                    ["\\int_{%s}^{%s} %s\\,dx = %s" % (tex(a), tex(b), tex(f), tex(inner2))]),
                               step("Then over y", "",
                                    ["\\int_{%s}^{%s} %s\\,dy = %s" % (tex(lo), tex(hi), tex(inner2), tex(v2))])],
                              "\\iint_R %s\\,dA = %s" % (_wrap(f), tex(v2)), verified=_close(v2, value),
                              note="Agrees with the other order."))
        _valued(methods[-1], v2)
    else:
        methods.append(not_applicable("swapped", name,
                                      "The limits for \\(y\\) depend on \\(x\\), so swapping the order "
                                      "means re-describing the region; that is not done here."))
    return result("vector", "Double integral", "\\iint_R %s\\,dA" % _wrap(f),
                  "Double integral over a region between two curves.", methods)


def _polar_integral(text, r_text, th_text):
    f = _expr(text, ["x", "y"], "f(x, y)")
    if not r_text.strip() or not th_text.strip():
        raise InputError("Enter the limits for r and for the angle, for example  0, 2  and  0, 2pi.")
    r0, r1 = _numbers(r_text, 2, "The limits for r")
    t0, t1 = _numbers(th_text, 2, "The limits for the angle")
    r = sp.Symbol("r", positive=True)
    th = sp.Symbol("theta", real=True)
    g = sp.simplify(sp.trigsimp(f.subs({X: r * sp.cos(th), Y: r * sp.sin(th)})) * r)
    inner = _integral(g, r, r0, r1)
    value = _integral(inner, th, t0, t1)
    steps = [
        step("Change to polar coordinates",
             "Put \\(x = r\\cos\\theta\\), \\(y = r\\sin\\theta\\) and replace \\(dA\\) by "
             "\\(r\\,dr\\,d\\theta\\). The extra factor \\(r\\) is the Jacobian.",
             ["\\int_{%s}^{%s}\\int_{%s}^{%s} %s\\,dr\\,d\\theta" % (tex(t0), tex(t1), tex(r0), tex(r1), tex(g))]),
        step("Inner integral (in r)", "", ["\\int_{%s}^{%s} %s\\,dr = %s" % (tex(r0), tex(r1), tex(g), tex(inner))],
             explain(g, r, "The inner integral")),
        step("Outer integral (in \u03b8)", "",
             ["\\int_{%s}^{%s} %s\\,d\\theta = %s" % (tex(t0), tex(t1), tex(inner), tex(value))]),
    ]
    ok = _check_quad2(g, th, t0, t1, r, r0, r1, value)
    return result("vector", "Double integral (polar)", "\\iint_R %s\\,dA" % _wrap(f),
                  "Double integral in polar coordinates.",
                  [_valued(method("polar", "Polar coordinates",
                                  "The region is described by limits on \\(r\\) and the angle.", steps,
                                  "\\iint_R %s\\,dA = %s" % (_wrap(f), tex(value)), verified=ok,
                                  note="Agrees with numerical integration." if ok else
                                  "Could not be checked numerically."), value)])


def _check_quad2(f, outer, a, b, inner, lo, hi, value):
    try:
        import mpmath
        fn = sp.lambdify((outer, inner), f, "mpmath")
        lo_fn = sp.lambdify(outer, lo, "mpmath")
        hi_fn = sp.lambdify(outer, hi, "mpmath")
        ref = mpmath.quad(lambda u: mpmath.quad(lambda v: fn(u, v), [lo_fn(u), hi_fn(u)]),
                          [float(a), float(b)])
        return True if abs(complex(sp.N(value)) - complex(ref)) < 1e-6 * (1 + abs(complex(ref))) else False
    except Exception:
        return None


# --------------------------------------------------------------------------
# Green's theorem
# --------------------------------------------------------------------------

def _green(text, region_text):
    F = _components(text, ["x", "y"], "The field", sizes=(2,))
    P, Q = F
    numbers = _split(region_text)
    if len(numbers) not in (1, 2, 4):
        raise InputError("Describe the region with numbers: one radius for a disc centred at the "
                         "origin, two radii for a ring, or four numbers  x1, x2, y1, y2  for a "
                         "rectangle.")
    vals = [parse_number(n, "The region") for n in numbers]
    curl = sp.simplify(sp.diff(Q, X) - sp.diff(P, Y))
    r = sp.Symbol("r", positive=True)
    th = sp.Symbol("theta", real=True)

    def circle(radius, sign):
        on = {X: radius * sp.cos(T), Y: radius * sp.sin(T)}
        integrand = sp.simplify(P.subs(on) * (-radius * sp.sin(T)) + Q.subs(on) * (radius * sp.cos(T)))
        return sign * _integral(integrand, T, 0, 2 * sp.pi), integrand

    line_steps, line_total = [], sp.Integer(0)
    if len(vals) == 4:
        x1, x2, y1, y2 = vals
        if not (x1 < x2 and y1 < y2):
            raise InputError("For a rectangle give x1 < x2 and y1 < y2.")
        region = "%s \\le x \\le %s,\\ %s \\le y \\le %s" % (tex(x1), tex(x2), tex(y1), tex(y2))
        double = _integral(_integral(curl, Y, y1, y2), X, x1, x2)
        edges = [("bottom", P.subs(Y, y1), X, x1, x2, 1), ("right", Q.subs(X, x2), Y, y1, y2, 1),
                 ("top", P.subs(Y, y2), X, x1, x2, -1), ("left", Q.subs(X, x1), Y, y1, y2, -1)]
        lines = []
        for name, integrand, var, lo, hi, sign in edges:
            piece = sign * _integral(integrand, var, lo, hi)
            line_total += piece
            lines.append("\\text{%s: } %s" % (name, tex(piece)))
        line_steps.append(step("Integrate along the four sides",
                               "Go round counterclockwise: along the bottom and top only \\(P\\,dx\\) "
                               "contributes, along the sides only \\(Q\\,dy\\).", lines))
        d_text = "Integrate over the rectangle."
        d_tex = "\\int_{%s}^{%s}\\int_{%s}^{%s} %s\\,dy\\,dx = %s" % (tex(x1), tex(x2), tex(y1), tex(y2), tex(curl), tex(double))
    else:
        inner_r, outer_r = (sp.Integer(0), vals[0]) if len(vals) == 1 else (vals[0], vals[1])
        if not (0 <= inner_r < outer_r):
            raise InputError("The radii must satisfy 0 <= inner < outer.")
        region = ("x^{2} + y^{2} \\le %s" % tex(outer_r**2) if inner_r == 0 else
                  "%s \\le x^{2} + y^{2} \\le %s" % (tex(inner_r**2), tex(outer_r**2)))
        g = sp.simplify(sp.trigsimp(curl.subs({X: r * sp.cos(th), Y: r * sp.sin(th)})) * r)
        double = _integral(_integral(g, r, inner_r, outer_r), th, 0, 2 * sp.pi)
        piece, integrand = circle(outer_r, 1)
        line_total += piece
        lines = ["\\text{outer circle, counterclockwise: } \\int_0^{2\\pi} %s\\,dt = %s" % (tex(integrand), tex(piece))]
        if inner_r != 0:
            piece2, integrand2 = circle(inner_r, -1)
            line_total += piece2
            lines.append("\\text{inner circle, clockwise: } %s" % tex(piece2))
        line_steps.append(step("Integrate around the boundary",
                               "Parametrise each circle as \\(x = R\\cos t\\), \\(y = R\\sin t\\). The "
                               "region must stay on the left, so an inner circle is traversed clockwise.",
                               lines))
        d_text = "Integrate over the region in polar coordinates."
        d_tex = "\\int_0^{2\\pi}\\int_{%s}^{%s} %s\\,dr\\,d\\theta = %s" % (tex(inner_r), tex(outer_r), tex(g), tex(double))
    line_total, double = sp.simplify(line_total), sp.simplify(double)
    agree = _close(line_total, double)
    statement = "\\oint_C P\\,dx + Q\\,dy = \\iint_D \\left(Q_x - P_y\\right)dA"
    g_steps = [
        step("State Green's theorem",
             "For a region \\(D\\) whose boundary \\(C\\) is traversed with the region on the left,",
             [statement, "D:\\ %s" % region]),
        step("Compute the integrand", "", ["Q_x - P_y = %s" % tex(curl)]),
        step("Evaluate the double integral", d_text, [d_tex]),
    ]
    line_steps.append(step("Add the pieces", "", ["\\oint_C P\\,dx + Q\\,dy = %s" % tex(line_total)]))
    note = "Both sides of Green's theorem agree." if agree else "The two sides do not agree."
    methods = [
        method("green", "Green's theorem", "Turns the boundary integral into a double integral.",
               g_steps, "\\oint_C P\\,dx + Q\\,dy = %s" % tex(double), verified=agree, note=note),
        method("direct", "Direct line integral", "Integrates along the boundary itself.",
               line_steps, "\\oint_C P\\,dx + Q\\,dy = %s" % tex(line_total), verified=agree, note=note),
    ]
    _valued(methods[0], double)
    _valued(methods[1], line_total)
    extra = {"graph": "phase", "field": {"f": _plot(P, _NAMES), "g": _plot(Q, _NAMES)}, "equilibria": [],
             "legends": {"trajectories": "Field lines", "hint": "The arrows show the field. Tap to draw a field line."}}
    return result("vector", "Green's theorem", "\\oint_C %s\\,dx + %s\\,dy" % (_wrap(P), _wrap(Q)),
                  "Circulation around a closed curve, computed two ways.", methods, **extra)


def _wrap(expr):
    return "\\left(%s\\right)" % tex(expr) if expr.is_Add else tex(expr)


# --------------------------------------------------------------------------
# Flux (divergence theorem) and circulation (Stokes' theorem)
# --------------------------------------------------------------------------

def _surface(text, allowed, example):
    words = text.strip().split(None, 1)
    if not words or words[0].lower() not in allowed:
        raise InputError("Name the surface and its size, for example  %s. Supported here: %s."
                         % (example, ", ".join(allowed)))
    kind = words[0].lower()
    numbers = [parse_number(n, "The size") for n in _split(words[1])] if len(words) > 1 else []
    if len(numbers) != allowed[kind] or any(not (n > 0) for n in numbers):
        raise InputError("A %s needs %d positive number%s after its name, for example  %s."
                         % (kind, allowed[kind], "" if allowed[kind] == 1 else "s", example))
    return kind, numbers


def _flux(text, surface_text):
    F = _components(text, ["x", "y", "z"], "The field", sizes=(3,))
    P, Q, R = F
    kind, size = _surface(surface_text, {"sphere": 1, "box": 3, "cylinder": 2}, "sphere 2")
    div = sp.simplify(sp.diff(P, X) + sp.diff(Q, Y) + sp.diff(R, Z))
    rho = sp.Symbol("rho", positive=True)
    r = sp.Symbol("r", positive=True)
    ph = sp.Symbol("phi", real=True)
    th = sp.Symbol("theta", real=True)

    if kind == "sphere":
        a = size[0]
        where = "the sphere of radius %s centred at the origin" % sp.sstr(a)
        sub = {X: rho * sp.sin(ph) * sp.cos(th), Y: rho * sp.sin(ph) * sp.sin(th), Z: rho * sp.cos(ph)}
        g = sp.simplify(div.subs(sub) * rho**2 * sp.sin(ph))
        triple = _integral(_integral(_integral(g, rho, 0, a), ph, 0, sp.pi), th, 0, 2 * sp.pi)
        t_tex = ("\\int_0^{2\\pi}\\int_0^{\\pi}\\int_0^{%s} %s\\,d\\rho\\,d\\phi\\,d\\theta = %s"
                 % (tex(a), tex(g), tex(triple)))
        t_text = "Use spherical coordinates, where \\(dV = \\rho^{2}\\sin\\phi\\,d\\rho\\,d\\phi\\,d\\theta\\)."
        on = {k: v.subs(rho, a) for k, v in sub.items()}
        normal = [sp.sin(ph) * sp.cos(th), sp.sin(ph) * sp.sin(th), sp.cos(ph)]
        h = sp.simplify(sum(c.subs(on) * n for c, n in zip(F, normal)) * a**2 * sp.sin(ph))
        direct = _integral(_integral(h, ph, 0, sp.pi), th, 0, 2 * sp.pi)
        d_lines = ["\\mathbf{n} = \\frac{1}{%s}\\langle x, y, z\\rangle,\\qquad dS = %s\\sin\\phi\\,d\\phi\\,d\\theta"
                   % (tex(a), tex(a**2)),
                   "\\int_0^{2\\pi}\\int_0^{\\pi} %s\\,d\\phi\\,d\\theta = %s" % (tex(h), tex(direct))]
        d_text = "Parametrise the sphere by the angles \\(\\phi\\) and \\(\\theta\\)."
    elif kind == "box":
        a, b, c = size
        where = "the box 0 \u2264 x \u2264 %s, 0 \u2264 y \u2264 %s, 0 \u2264 z \u2264 %s" % tuple(sp.sstr(v) for v in size)
        triple = _integral(_integral(_integral(div, Z, 0, c), Y, 0, b), X, 0, a)
        t_tex = "\\int_0^{%s}\\int_0^{%s}\\int_0^{%s} %s\\,dz\\,dy\\,dx = %s" % (tex(a), tex(b), tex(c), tex(div), tex(triple))
        t_text = "Integrate over the box."
        faces = [("x = %s" % tex(a), _integral(_integral(P.subs(X, a), Z, 0, c), Y, 0, b)),
                 ("x = 0", -_integral(_integral(P.subs(X, 0), Z, 0, c), Y, 0, b)),
                 ("y = %s" % tex(b), _integral(_integral(Q.subs(Y, b), Z, 0, c), X, 0, a)),
                 ("y = 0", -_integral(_integral(Q.subs(Y, 0), Z, 0, c), X, 0, a)),
                 ("z = %s" % tex(c), _integral(_integral(R.subs(Z, c), Y, 0, b), X, 0, a)),
                 ("z = 0", -_integral(_integral(R.subs(Z, 0), Y, 0, b), X, 0, a))]
        direct = sum(v for _, v in faces)
        d_lines = ["%s:\\ %s" % (n, tex(v)) for n, v in faces]
        d_text = ("On each face the outward normal is along one axis, so only one component of the "
                  "field contributes.")
    else:
        a, hgt = size
        where = "the closed cylinder of radius %s between z = 0 and z = %s" % (sp.sstr(a), sp.sstr(hgt))
        sub = {X: r * sp.cos(th), Y: r * sp.sin(th)}
        g = sp.simplify(div.subs(sub) * r)
        triple = _integral(_integral(_integral(g, Z, 0, hgt), r, 0, a), th, 0, 2 * sp.pi)
        t_tex = ("\\int_0^{2\\pi}\\int_0^{%s}\\int_0^{%s} %s\\,dz\\,dr\\,d\\theta = %s"
                 % (tex(a), tex(hgt), tex(g), tex(triple)))
        t_text = "Use cylindrical coordinates, where \\(dV = r\\,dz\\,dr\\,d\\theta\\)."
        side_sub = {X: a * sp.cos(th), Y: a * sp.sin(th)}
        side = _integral(_integral(sp.simplify((P.subs(side_sub) * sp.cos(th) + Q.subs(side_sub) * sp.sin(th)) * a),
                                   Z, 0, hgt), th, 0, 2 * sp.pi)
        top = _integral(_integral(sp.simplify(R.subs(Z, hgt).subs(sub) * r), r, 0, a), th, 0, 2 * sp.pi)
        bottom = -_integral(_integral(sp.simplify(R.subs(Z, 0).subs(sub) * r), r, 0, a), th, 0, 2 * sp.pi)
        direct = side + top + bottom
        d_lines = ["\\text{side: } %s" % tex(side), "\\text{top: } %s" % tex(top),
                   "\\text{bottom: } %s" % tex(bottom)]
        d_text = "Add the flux through the curved side, the top and the bottom."
    triple, direct = sp.simplify(triple), sp.simplify(direct)
    agree = _close(triple, direct)
    note = "Both sides of the divergence theorem agree." if agree else "The two sides do not agree."
    methods = [
        method("divergence", "Divergence theorem",
               "The surface is closed, so the outward flux equals the integral of the divergence "
               "over the solid inside.",
               [step("State the theorem", "",
                     ["\\iint_S \\mathbf{F} \\cdot \\mathbf{n}\\,dS = \\iiint_V \\nabla \\cdot \\mathbf{F}\\,dV"]),
                step("Compute the divergence", "", ["\\nabla \\cdot \\mathbf{F} = %s" % tex(div)]),
                step("Integrate over the solid", t_text, [t_tex])],
               "\\iint_S \\mathbf{F} \\cdot \\mathbf{n}\\,dS = %s" % tex(triple), verified=agree, note=note),
        method("direct", "Direct surface integral", "Integrates the normal component over the surface itself.",
               [step("Integrate over the surface", d_text, d_lines),
                step("Add the pieces", "", ["\\iint_S \\mathbf{F} \\cdot \\mathbf{n}\\,dS = %s" % tex(direct)])],
               "\\iint_S \\mathbf{F} \\cdot \\mathbf{n}\\,dS = %s" % tex(direct), verified=agree, note=note),
    ]
    _valued(methods[0], triple)
    _valued(methods[1], direct)
    return result("vector", "Flux", "\\iint_S %s \\cdot \\mathbf{n}\\,dS" % _vec(F),
                  "Outward flux through %s, computed two ways." % where, methods)


def _stokes(text, surface_text):
    F = _components(text, ["x", "y", "z"], "The field", sizes=(3,))
    kind, size = _surface(surface_text, {"hemisphere": 1, "disk": 1, "paraboloid": 1}, "hemisphere 1")
    a = size[0]
    curl = _curl(F, _XYZ)
    r = sp.Symbol("r", positive=True)
    th = sp.Symbol("theta", real=True)
    ph = sp.Symbol("phi", real=True)
    # The boundary is the circle of radius a in the plane z = 0, counterclockwise from above.
    on = {X: a * sp.cos(T), Y: a * sp.sin(T), Z: sp.Integer(0)}
    integrand = sp.simplify(F[0].subs(on) * (-a * sp.sin(T)) + F[1].subs(on) * (a * sp.cos(T)))
    line = _integral(integrand, T, 0, 2 * sp.pi)
    if kind == "disk":
        where = "the disc of radius %s in the plane z = 0" % sp.sstr(a)
        g = sp.simplify(curl[2].subs({X: r * sp.cos(th), Y: r * sp.sin(th), Z: 0}) * r)
        surf = _integral(_integral(g, r, 0, a), th, 0, 2 * sp.pi)
        s_text = "The normal is \\(\\mathbf{k}\\), so only the third component of the curl matters."
        s_tex = "\\int_0^{2\\pi}\\int_0^{%s} %s\\,dr\\,d\\theta = %s" % (tex(a), tex(g), tex(surf))
    elif kind == "hemisphere":
        where = "the upper half of the sphere of radius %s" % sp.sstr(a)
        sub = {X: a * sp.sin(ph) * sp.cos(th), Y: a * sp.sin(ph) * sp.sin(th), Z: a * sp.cos(ph)}
        normal = [sp.sin(ph) * sp.cos(th), sp.sin(ph) * sp.sin(th), sp.cos(ph)]
        g = sp.simplify(sum(c.subs(sub) * n for c, n in zip(curl, normal)) * a**2 * sp.sin(ph))
        surf = _integral(_integral(g, ph, 0, sp.pi / 2), th, 0, 2 * sp.pi)
        s_text = "Parametrise the hemisphere by the angles; the outward normal points up."
        s_tex = "\\int_0^{2\\pi}\\int_0^{\\pi/2} %s\\,d\\phi\\,d\\theta = %s" % (tex(g), tex(surf))
    else:
        where = "the paraboloid z = %s - x\u00b2 - y\u00b2 above the plane z = 0" % sp.sstr(a**2)
        # z = a^2 - x^2 - y^2: upward normal element is <2x, 2y, 1> dA.
        dot = curl[0] * 2 * X + curl[1] * 2 * Y + curl[2]
        g = sp.simplify(dot.subs(Z, a**2 - X**2 - Y**2).subs({X: r * sp.cos(th), Y: r * sp.sin(th)}) * r)
        surf = _integral(_integral(g, r, 0, a), th, 0, 2 * sp.pi)
        s_text = ("For the surface \\(z = h(x, y)\\) the upward normal element is "
                  "\\(\\langle -h_x, -h_y, 1\\rangle\\,dA = \\langle 2x, 2y, 1\\rangle\\,dA\\).")
        s_tex = "\\int_0^{2\\pi}\\int_0^{%s} %s\\,dr\\,d\\theta = %s" % (tex(a), tex(g), tex(surf))
    line, surf = sp.simplify(line), sp.simplify(surf)
    agree = _close(line, surf)
    note = "Both sides of Stokes' theorem agree." if agree else "The two sides do not agree."
    methods = [
        method("surface", "Surface integral of the curl", "Integrates the curl over the surface.",
               [step("State Stokes' theorem", "The boundary is traversed counterclockwise seen from above.",
                     ["\\oint_C \\mathbf{F} \\cdot d\\mathbf{r} = \\iint_S (\\nabla \\times \\mathbf{F}) \\cdot \\mathbf{n}\\,dS"]),
                step("Compute the curl", "", ["\\nabla \\times \\mathbf{F} = %s" % _vec(curl)]),
                step("Integrate over the surface", s_text, [s_tex])],
               "\\oint_C \\mathbf{F} \\cdot d\\mathbf{r} = %s" % tex(surf), verified=agree, note=note),
        method("boundary", "Line integral around the boundary",
               "Integrates the field around the bounding circle.",
               [step("Parametrise the circle", "",
                     ["\\mathbf{r}(t) = \\langle %s\\cos t,\\ %s\\sin t,\\ 0 \\rangle,\\quad 0 \\le t \\le 2\\pi"
                      % (tex(a), tex(a))]),
                step("Integrate", "", ["\\int_0^{2\\pi} %s\\,dt = %s" % (tex(integrand), tex(line))])],
               "\\oint_C \\mathbf{F} \\cdot d\\mathbf{r} = %s" % tex(line), verified=agree, note=note),
    ]
    _valued(methods[0], surf)
    _valued(methods[1], line)
    return result("vector", "Stokes' theorem", "\\oint_C %s \\cdot d\\mathbf{r}" % _vec(F),
                  "Circulation around the edge of %s, computed two ways." % where, methods)


def solve(operation, text, extra1="", extra2=""):
    if operation == "gradient":
        return _gradient(text, extra1, extra2)
    if operation == "divcurl":
        return _divcurl(text)
    if operation == "line":
        return _line(text, extra1, extra2)
    if operation == "double":
        return _double(text, extra1, extra2)
    if operation == "polar":
        return _polar_integral(text, extra1, extra2)
    if operation == "green":
        return _green(text, extra1)
    if operation == "flux":
        return _flux(text, extra1)
    if operation == "stokes":
        return _stokes(text, extra1)
    raise InputError("Unknown operation.")


def preview(operation, text):
    if operation in ("gradient", "double", "polar"):
        names = ["x", "y", "z"] if operation == "gradient" else ["x", "y"]
        return "f = %s" % tex(_expr(text, names, "f"))
    parts = _split(text)
    names = ["x", "y", "z"] if (len(parts) == 3 or operation in ("flux", "stokes")) else ["x", "y"]
    done = [_expr(p, names, "the field") for p in parts]
    if operation == "line" and len(done) == 1:
        return "f = %s" % tex(done[0])
    return "\\mathbf{F} = %s" % _vec(done)
