"""Planar autonomous systems x' = f(x, y), y' = g(x, y).

Linear homogeneous systems are solved by the eigenvalue method and by the
matrix exponential. Nonlinear (and affine) systems are analysed by
linearising at each equilibrium point.
"""
import sympy as sp

from .first_order import step
from .parsing import InputError, parse_expression, parse_number
from . import integrals
from .util import has_integral, integrate, residual_status, tex

T = sp.Symbol("t", real=True)
C1, C2 = sp.Symbol("C_1", real=True), sp.Symbol("C_2", real=True)
C3 = sp.Symbol("C_3", real=True)
_PLOT_FUNCS = {"sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan",
               "sinh", "cosh", "tanh", "exp", "log", "Abs"}


def _vec(*entries):
    return "\\begin{pmatrix}%s\\end{pmatrix}" % "\\\\".join(tex(e) for e in entries)


def _mat(M):
    rows = [" & ".join(tex(M[i, j]) for j in range(M.cols)) for i in range(M.rows)]
    return "\\begin{pmatrix}%s\\end{pmatrix}" % "\\\\".join(rows)


def _plot(expr, names):
    """Plot string with the given {symbol: name} renaming, or None."""
    try:
        if expr.has(sp.I, sp.oo, sp.zoo, sp.nan):
            return None
        if any(type(f).__name__ not in _PLOT_FUNCS for f in expr.atoms(sp.Function)):
            return None
        if not expr.free_symbols <= set(names):
            return None
        return sp.sstr(expr.xreplace({s: sp.Symbol(n) for s, n in names.items()}))
    except Exception:
        return None


def classify(M):
    """(name, stability sentence) of the equilibrium of X' = M X."""
    tr, det = sp.simplify(M.trace()), sp.simplify(M.det())
    disc = sp.simplify(tr**2 - 4 * det)
    if det < 0:
        return "saddle point", "unstable"
    if det == 0:
        return "degenerate case (a zero eigenvalue)", "not decided by the linear terms"
    if disc < 0:
        if tr == 0:
            return "center", "stable but not asymptotically stable"
        return ("spiral sink", "asymptotically stable") if tr < 0 else ("spiral source", "unstable")
    kind = "node" if disc > 0 else ("star node" if M == M[0, 0] * sp.eye(2) else "improper node")
    return ("%s (sink)" % kind, "asymptotically stable") if tr < 0 else ("%s (source)" % kind, "unstable")


def _a(noun):
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def _integer_vector(v):
    """Scale an eigenvector to have small whole-number entries when possible."""
    try:
        dens = [sp.fraction(sp.together(e))[1] for e in v]
        scale = sp.ilcm(*[int(d) for d in dens])
        v = v * scale
        g = sp.igcd(*[int(e) for e in v if e != 0]) if all(e.is_Integer for e in v) else 1
        v = v / g if g else v
        first = [e for e in v if e != 0][0]
        return -v if first.could_extract_minus_sign() else v
    except Exception:
        return v


def _closing(m, has_ic):
    answer = m["particular_latex"] if has_ic and m["particular_latex"] else m["general_latex"]
    m["steps"].append({"title": "Solution" if has_ic and m["particular_latex"] else "General solution",
                       "text": "", "latex": [answer], "substeps": [], "kind": "answer"})
    kind = {True: "check-ok", False: "check-bad", None: "check-unknown"}[m["verified"]]
    m["steps"].append({"title": "Check", "text": m["verify_note"], "latex": [],
                       "substeps": [], "kind": kind})


def _verify(A, X, ic, forcing=None):
    residual = X.diff(T) - A * X
    if forcing is not None:
        residual = residual - forcing
    statuses = [residual_status(sp.simplify(e)) for e in residual]
    if any(s in ("bad", "partial") for s in statuses):
        return False, "Substituting back did not reproduce the system."
    if any(s == "unknown" for s in statuses):
        return None, "Could not be checked automatically."
    if ic is not None:
        at0 = X.subs(T, 0)
        if any(abs(complex(sp.N(at0[i] - ic[i]))) > 1e-9 for i in range(len(ic))):
            return False, "The initial conditions are not satisfied."
        return True, "Checked by substituting back into every equation, and it meets the initial conditions."
    return True, "Checked by substituting back into every equation."


_NAMES = ("x", "y", "z")


def _components(X):
    return ",\\qquad ".join("%s(t) = %s" % (_NAMES[i], tex(X[i])) for i in range(len(X)))


def _package(method_id, name, reason, steps, A, general, particular, ic, forcing=None):
    target = particular if particular is not None else general
    verified, note = _verify(A, target, ic if particular is not None else None, forcing)
    names = {T: "x"}
    plot = {"general_expr": None, "particular_expr": None, "points": [],
            "x_expr": None, "y_expr": None}
    if particular is not None:
        for i, key in enumerate(("x_expr", "y_expr", "z_expr")[:len(particular)]):
            plot[key] = _plot(particular[i], names)
    m = {"id": method_id, "name": name, "applicable": True, "reason": reason, "steps": steps,
         "general_latex": _components(general),
         "particular_latex": _components(particular) if particular is not None else None,
         "plot": plot, "verified": verified, "verify_note": note}
    _closing(m, ic is not None)
    return m


def _apply_ic(general, ic, steps):
    n = len(ic)
    constants = [C1, C2, C3][:n]
    at0 = general.subs(T, 0)
    found = sp.solve([at0[i] - ic[i] for i in range(n)], constants, dict=True)
    if not found:
        return None
    values = found[0]
    particular = general.subs(values).applyfunc(sp.expand)
    steps.append(step(
        "Apply the initial conditions",
        "Set \\(t = 0\\) and solve the equations for the constants.",
        [_components(particular)],
        [step("Equations for the constants", "",
              ["%s = %s" % (tex(at0[i]), tex(ic[i])) for i in range(n)]
              + [",\\qquad ".join("C_%d = %s" % (i + 1, tex(values.get(c, c)))
                                   for i, c in enumerate(constants))])]))
    return particular


def _eigen_core(A):
    """Steps and general solution of X' = A X for a 2x2 matrix."""
    lam = sp.Symbol("lambda")
    tr, det = A.trace(), A.det()
    char = sp.expand(lam**2 - tr * lam + det)
    disc = sp.simplify(tr**2 - 4 * det)
    steps = [
        step("Write in matrix form", "The system is \\(\\mathbf{X}' = A\\mathbf{X}\\).",
             ["\\begin{pmatrix}x\\\\y\\end{pmatrix}' = %s\\begin{pmatrix}x\\\\y\\end{pmatrix}" % _mat(A)]),
        step("Find the eigenvalues",
             "Solutions of the form \\(\\mathbf{v}e^{\\lambda t}\\) need "
             "\\(\\det(A - \\lambda I) = 0\\).", ["%s = 0" % tex(char)]),
    ]
    vects = A.eigenvects()
    if disc > 0:
        pairs = sorted([(val, _integer_vector(vs[0])) for val, _, vs in vects], key=lambda p: float(p[0]))
        (l1, v1), (l2, v2) = pairs
        steps[-1]["latex"].append("\\lambda_1 = %s,\\qquad \\lambda_2 = %s" % (tex(l1), tex(l2)))
        steps.append(step(
            "Find the eigenvectors",
            "For each eigenvalue solve \\((A - \\lambda I)\\mathbf{v} = \\mathbf{0}\\).",
            ["\\lambda_1 = %s:\\ \\mathbf{v}_1 = %s,\\qquad \\lambda_2 = %s:\\ \\mathbf{v}_2 = %s"
             % (tex(l1), _vec(*v1), tex(l2), _vec(*v2))]))
        X1, X2 = v1 * sp.exp(l1 * T), v2 * sp.exp(l2 * T)
        vector_form = "\\mathbf{X} = C_1 %s e^{%s} + C_2 %s e^{%s}" % (
            _vec(*v1), tex(l1 * T), _vec(*v2), tex(l2 * T))
        text = "Two real eigenvalues give two independent exponential solutions."
    elif disc < 0:
        alpha = sp.simplify(tr / 2)
        beta = sp.simplify(sp.sqrt(-disc) / 2)
        target = alpha + sp.I * beta
        v = [vs[0] for val, _, vs in vects if sp.simplify(val - target) == 0][0]
        p, q = v.applyfunc(sp.re), v.applyfunc(sp.im)
        steps[-1]["latex"].append("\\lambda = %s \\pm %s\\,i" % (tex(alpha), tex(beta)) if alpha != 0
                                  else "\\lambda = \\pm %s\\,i" % tex(beta))
        steps.append(step(
            "Find a complex eigenvector",
            "For \\(\\lambda = \\alpha + \\beta i\\), split the eigenvector into real and "
            "imaginary parts, \\(\\mathbf{v} = \\mathbf{p} + i\\mathbf{q}\\).",
            ["\\mathbf{p} = %s,\\qquad \\mathbf{q} = %s" % (_vec(*p), _vec(*q))]))
        e = sp.exp(alpha * T)
        X1 = e * (p * sp.cos(beta * T) - q * sp.sin(beta * T))
        X2 = e * (p * sp.sin(beta * T) + q * sp.cos(beta * T))
        vector_form = ("\\mathbf{X} = e^{\\alpha t}\\left[C_1(\\mathbf{p}\\cos\\beta t - "
                       "\\mathbf{q}\\sin\\beta t) + C_2(\\mathbf{p}\\sin\\beta t + \\mathbf{q}\\cos\\beta t)\\right]")
        text = ("The real and imaginary parts of the complex solution are two "
                "independent real solutions.")
    else:
        l1 = sp.simplify(tr / 2)
        steps[-1]["latex"].append("\\lambda = %s\\ \\text{(repeated)}" % tex(l1))
        if A == l1 * sp.eye(2):
            v1, v2 = sp.Matrix([1, 0]), sp.Matrix([0, 1])
            steps.append(step("Find the eigenvectors",
                              "Every vector is an eigenvector, so take the two unit vectors.",
                              ["\\mathbf{v}_1 = %s,\\qquad \\mathbf{v}_2 = %s" % (_vec(*v1), _vec(*v2))]))
            X1, X2 = v1 * sp.exp(l1 * T), v2 * sp.exp(l1 * T)
            vector_form = "\\mathbf{X} = e^{%s}\\left(C_1 %s + C_2 %s\\right)" % (
                tex(l1 * T), _vec(*v1), _vec(*v2))
            text = "Both solutions share the same exponential."
        else:
            v = _integer_vector(vects[0][2][0])
            sol, params = (A - l1 * sp.eye(2)).gauss_jordan_solve(v)
            w = sol.subs({prm: 0 for prm in params})
            steps.append(step(
                "Find the eigenvector and a generalized eigenvector",
                "There is only one independent eigenvector \\(\\mathbf{v}\\). A second "
                "solution needs \\(\\mathbf{w}\\) with \\((A - \\lambda I)\\mathbf{w} = \\mathbf{v}\\).",
                ["\\mathbf{v} = %s,\\qquad \\mathbf{w} = %s" % (_vec(*v), _vec(*w))]))
            X1 = v * sp.exp(l1 * T)
            X2 = (v * T + w) * sp.exp(l1 * T)
            vector_form = "\\mathbf{X} = C_1\\mathbf{v}e^{\\lambda t} + C_2(\\mathbf{v}t + \\mathbf{w})e^{\\lambda t}"
            text = "The second solution has the form \\((\\mathbf{v}t + \\mathbf{w})e^{\\lambda t}\\)."
    general = (C1 * X1 + C2 * X2).applyfunc(sp.simplify)
    steps.append(step("Write the general solution", text, [vector_form, _components(general)]))
    return steps, general


def eigen(A, ic):
    name = "Eigenvalue method"
    steps, general = _eigen_core(A)
    particular = _apply_ic(general, ic, steps) if ic else None
    kind, stability = classify(A)
    steps.append(step("Classify the equilibrium",
                      "From the eigenvalues, the origin is %s: %s." % (_a(kind), stability)))
    return _package("eigen", name, "The system is linear with constant coefficients.",
                    steps, A, general, particular, ic)


def _expm(A, sign=1):
    """e^(sign A t) with real entries."""
    E = (sign * A * T).exp().applyfunc(sp.simplify)
    if E.has(sp.I):
        E = E.applyfunc(lambda e: sp.simplify(sp.re(sp.expand_complex(e))))
    return E


def exponential(A, ic):
    name = "Matrix exponential"
    E = _expm(A)
    steps = [
        step("Use the matrix exponential",
             "Just as \\(x' = ax\\) has solution \\(x = e^{at}x(0)\\), the system "
             "\\(\\mathbf{X}' = A\\mathbf{X}\\) has solution \\(\\mathbf{X} = e^{At}\\mathbf{X}(0)\\).",
             ["A = %s" % _mat(A),
              "e^{At} = I + At + \\frac{(At)^2}{2!} + \\frac{(At)^3}{3!} + \\cdots"]),
        step("Compute the exponential",
             "Summing the series (in practice through the eigenvalues of \\(A\\)) gives",
             ["e^{At} = %s" % _mat(E)]),
    ]
    general = (E * sp.Matrix([C1, C2])).applyfunc(sp.expand)
    steps.append(step("Multiply by the initial vector",
                      "Here \\(C_1 = x(0)\\) and \\(C_2 = y(0)\\).", [_components(general)]))
    particular = None
    if ic:
        particular = (E * sp.Matrix(list(ic))).applyfunc(sp.expand)
        steps.append(step("Insert the initial values",
                          "Put \\(x(0) = %s\\) and \\(y(0) = %s\\)." % (tex(ic[0]), tex(ic[1])),
                          [_components(particular)]))
    return _package("expm", name,
                    "For a constant matrix the solution is the matrix exponential "
                    "applied to the initial vector.", steps, A, general, particular, ic)


def linearize(f, g, x, y):
    name = "Linearization at equilibria"
    try:
        found = sp.solve([f, g], [x, y], dict=True)
    except Exception:
        found = []
    points = []
    for sol in found:
        if x in sol and y in sol and not (sol[x].free_symbols or sol[y].free_symbols):
            if sol[x].is_real and sol[y].is_real:
                points.append((sol[x], sol[y]))
    points = sorted(set(points), key=lambda p: (float(p[0]), float(p[1])))[:8]
    J = sp.Matrix([[sp.diff(f, x), sp.diff(f, y)], [sp.diff(g, x), sp.diff(g, y)]])
    steps = [step(
        "Find the equilibrium points",
        "Equilibria are where both right-hand sides are zero.",
        ["%s = 0,\\qquad %s = 0" % (tex(f), tex(g)),
         ",\\quad ".join("(%s,\\ %s)" % (tex(a), tex(b)) for a, b in points)
         if points else "\\text{no isolated equilibrium points were found}"])]
    steps.append(step(
        "Compute the Jacobian",
        "Near an equilibrium the system behaves like the linear system whose "
        "matrix is the Jacobian evaluated there.",
        ["J = \\begin{pmatrix} f_x & f_y\\\\ g_x & g_y \\end{pmatrix} = %s" % _mat(J)]))
    rows, marks = [], []
    for a, b in points:
        M = J.subs({x: a, y: b}).applyfunc(sp.simplify)
        kind, stability = classify(M)
        eigs = sorted(M.eigenvals(multiple=True), key=sp.default_sort_key)
        hyperbolic = all(sp.re(e) != 0 for e in eigs)
        note = "" if hyperbolic else (" The linear terms alone do not settle the behaviour of "
                                      "the nonlinear system here.")
        steps.append(step(
            "Classify the equilibrium at (%s, %s)" % (sp.sstr(a), sp.sstr(b)),
            "The linearization is %s: %s.%s" % (_a(kind), stability, note),
            ["J(%s,\\ %s) = %s" % (tex(a), tex(b), _mat(M)),
             "\\lambda = %s" % ",\\ ".join(tex(e) for e in eigs)]))
        rows.append(["(%s,\\ %s)" % (tex(a), tex(b)), ",\\ ".join(tex(e) for e in eigs),
                     "\\text{%s}" % kind])
        marks.append({"x": float(a), "y": float(b), "label": kind})
    if rows:
        summary = step("Summary", "")
        summary["table"] = {"headers": ["\\text{point}", "\\text{eigenvalues}", "\\text{type}"], "rows": rows}
        summary["kind"] = "answer"
        steps.append(summary)
    method = {"id": "linearize", "name": name, "applicable": True,
              "reason": "The system is not of the form \\(\\mathbf{X}' = A\\mathbf{X}\\), so its "
                        "behaviour is read off near each equilibrium point from the Jacobian.",
              "steps": steps, "general_latex": None, "particular_latex": None,
              "plot": {"general_expr": None, "particular_expr": None, "points": [],
                       "x_expr": None, "y_expr": None},
              "verified": None, "verify_note": ""}
    return method, marks


def _na(method_id, name, reason):
    return {"id": method_id, "name": name, "applicable": False, "reason": reason}


# --------------------------------------------------------------------------
# Forced linear systems X' = A X + g(t)
# --------------------------------------------------------------------------

def shift(A, b, ic):
    """Constant forcing: move the equilibrium to -A^{-1} b."""
    name = "Shift the equilibrium"
    if b.has(T):
        return _na("shift", name, "The forcing depends on \\(t\\), so there is no "
                                  "constant equilibrium to shift to.")
    if A.det() == 0:
        return _na("shift", name, "The matrix \\(A\\) is not invertible, so "
                                  "\\(A\\mathbf{X} + \\mathbf{b} = \\mathbf{0}\\) has no unique solution.")
    Xp = (-A.inv() * b).applyfunc(sp.simplify)
    steps = [
        step("Write in matrix form", "The system is \\(\\mathbf{X}' = A\\mathbf{X} + \\mathbf{b}\\) "
             "with a constant vector \\(\\mathbf{b}\\).",
             ["A = %s,\\qquad \\mathbf{b} = %s" % (_mat(A), _vec(*b))]),
        step("Find the equilibrium",
             "A constant solution needs \\(A\\mathbf{X}_p + \\mathbf{b} = \\mathbf{0}\\).",
             ["\\mathbf{X}_p = -A^{-1}\\mathbf{b} = %s" % _vec(*Xp)]),
        step("Shift the unknown",
             "With \\(\\mathbf{U} = \\mathbf{X} - \\mathbf{X}_p\\) the constant disappears: "
             "\\(\\mathbf{U}' = A\\mathbf{U}\\). Solve that by the eigenvalue method."),
    ]
    core, homogeneous = _eigen_core(A)
    steps.extend(core[1:])
    general = (homogeneous + Xp).applyfunc(sp.simplify)
    steps.append(step("Shift back", "Add the equilibrium to the homogeneous solution.",
                      ["\\mathbf{X} = \\mathbf{X}_h + \\mathbf{X}_p", _components(general)]))
    particular = _apply_ic(general, ic, steps) if ic else None
    kind, stability = classify(A)
    steps.append(step("Classify the equilibrium",
                      "The equilibrium at \\((%s,\\ %s)\\) is %s: %s."
                      % (tex(Xp[0]), tex(Xp[1]), _a(kind), stability)))
    return _package("shift", name, "The forcing is constant and \\(A\\) is invertible.",
                    steps, A, general, particular, ic, forcing=b)


def variation(A, b, ic):
    name = "Variation of parameters (systems)"
    Phi = _expm(A)
    inverse = _expm(A, -1)
    integrand = (inverse * b).applyfunc(sp.simplify)
    integral = integrand.applyfunc(lambda e: integrate(e, T))
    if any(has_integral(e) for e in integral):
        return _na("variation", name, "The integral of \\(\\Phi^{-1}\\mathbf{g}\\) has no "
                                      "closed form in elementary functions.")
    integral = integral.applyfunc(sp.simplify)
    Xp = (Phi * integral).applyfunc(lambda e: sp.expand(sp.expand_trig(sp.simplify(sp.expand(e)))))
    working = []
    for i in range(len(integrand)):
        try:
            working.extend(integrals.explain(integrand[i], T, "Row %d" % (i + 1)))
        except Exception:
            pass
    steps = [
        step("Write in matrix form", "The system is \\(\\mathbf{X}' = A\\mathbf{X} + \\mathbf{g}(t)\\).",
             ["A = %s,\\qquad \\mathbf{g}(t) = %s" % (_mat(A), _vec(*b))]),
        step("Find a fundamental matrix",
             "The matrix exponential solves the homogeneous system and equals the "
             "identity at \\(t = 0\\).", ["\\Phi(t) = e^{At} = %s" % _mat(Phi)]),
        step("Set up the formula",
             "Looking for \\(\\mathbf{X} = \\Phi(t)\\,\\mathbf{u}(t)\\) gives "
             "\\(\\mathbf{u}' = \\Phi^{-1}\\mathbf{g}\\). For the exponential, "
             "\\(\\Phi^{-1}(t) = e^{-At}\\).",
             ["\\mathbf{X} = \\Phi(t)\\left[\\mathbf{C} + \\int \\Phi^{-1}(t)\\,\\mathbf{g}(t)\\,dt\\right]",
              "\\Phi^{-1}\\mathbf{g} = %s" % _vec(*integrand)]),
        step("Integrate", "Integrate each row.",
             ["\\int \\Phi^{-1}\\mathbf{g}\\,dt = %s" % _vec(*integral)], working),
        step("Particular solution", "Multiply by \\(\\Phi(t)\\) and simplify.",
             ["\\mathbf{X}_p = %s" % _vec(*Xp)]),
    ]
    constants = sp.Matrix([C1, C2, C3][:A.rows])
    general = (Phi * constants + Xp).applyfunc(sp.expand)
    steps.append(step("Write the general solution",
                      "Add the homogeneous solution \\(\\Phi(t)\\mathbf{C}\\).",
                      [_components(general)]))
    particular = _apply_ic(general, ic, steps) if ic else None
    return _package("variation", name,
                    "It works for any forcing once the homogeneous system is solved.",
                    steps, A, general, particular, ic, forcing=b)


# --------------------------------------------------------------------------
# 3x3 linear homogeneous systems
# --------------------------------------------------------------------------

def _stability(eigenvalues):
    parts = [sp.re(e) for e in eigenvalues]
    if all(p < 0 for p in parts):
        return "asymptotically stable (every eigenvalue has negative real part)"
    if any(p > 0 for p in parts):
        return "unstable (an eigenvalue has positive real part)"
    return "not asymptotically stable (an eigenvalue has zero real part)"


def _eigen_data(A):
    """Exact eigenvalues with multiplicities and eigenvectors, or None."""
    lam = sp.Symbol("lambda")
    char = sp.expand((lam * sp.eye(A.rows) - A).det())
    roots = sp.roots(sp.Poly(char, lam))
    if sum(roots.values()) != A.rows or any(sp.count_ops(r) > 25 for r in roots):
        return char, None
    data = []
    for value, mult in roots.items():
        vectors = (A - value * sp.eye(A.rows)).nullspace()
        data.append((sp.simplify(value), mult, [v.applyfunc(sp.simplify) for v in vectors]))
    data.sort(key=lambda d: (float(sp.re(d[0])), float(sp.im(d[0]))))
    return char, data


def eigen3(A, ic):
    name = "Eigenvalue method"
    lam = sp.Symbol("lambda")
    char, data = _eigen_data(A)
    steps = [
        step("Write in matrix form", "The system is \\(\\mathbf{X}' = A\\mathbf{X}\\).",
             ["A = %s" % _mat(A)]),
        step("Find the eigenvalues", "Solve \\(\\det(A - \\lambda I) = 0\\).",
             ["%s = 0" % tex(sp.factor(char))]),
    ]
    if data is None:
        return _na("eigen", name, "The characteristic cubic \\(%s = 0\\) has no roots "
                                  "in a simple closed form." % tex(char))
    steps[-1]["latex"].append(",\\qquad ".join(
        "\\lambda = %s%s" % (tex(v), "\\ \\text{(multiplicity %d)}" % m if m > 1 else "")
        for v, m, _ in data))
    lines = []
    for value, mult, vectors in data:
        lines.append("\\lambda = %s:\\quad %s" % (tex(value), ",\\ ".join(
            _vec(*(_integer_vector(v) if not v.has(sp.I) else v)) for v in vectors)))
    steps.append(step("Find the eigenvectors",
                      "For each eigenvalue solve \\((A - \\lambda I)\\mathbf{v} = \\mathbf{0}\\).", lines))
    eigenvalues = [v for v, m, _ in data for _ in range(m)]
    defective = any(len(vectors) < mult for _, mult, vectors in data)
    if defective:
        E = _expm(A)
        general = (E * sp.Matrix([C1, C2, C3])).applyfunc(sp.expand)
        steps.append(step(
            "Handle the missing eigenvector",
            "A repeated eigenvalue has fewer independent eigenvectors than its "
            "multiplicity (the matrix is defective), so exponentials alone do not "
            "give three solutions. The matrix exponential supplies the terms with "
            "powers of \\(t\\).",
            ["e^{At} = %s" % _mat(E), _components(general)]))
    else:
        solutions = []
        for value, mult, vectors in data:
            if value.is_real:
                for v in vectors:
                    solutions.append(_integer_vector(v) * sp.exp(value * T))
            elif sp.im(value) > 0:
                alpha, beta = sp.re(value), sp.im(value)
                v = vectors[0]
                p, q = v.applyfunc(sp.re), v.applyfunc(sp.im)
                e = sp.exp(alpha * T)
                solutions.append(e * (p * sp.cos(beta * T) - q * sp.sin(beta * T)))
                solutions.append(e * (p * sp.sin(beta * T) + q * sp.cos(beta * T)))
        general = sum((c * X for c, X in zip((C1, C2, C3), solutions)),
                      sp.zeros(3, 1)).applyfunc(sp.simplify)
        steps.append(step(
            "Write the general solution",
            "Each real eigenvalue gives \\(\\mathbf{v}e^{\\lambda t}\\); a complex pair "
            "\\(\\alpha \\pm \\beta i\\) gives two real solutions from the real and "
            "imaginary parts of \\(\\mathbf{v}e^{\\lambda t}\\).",
            [_components(general)]))
    particular = _apply_ic(general, ic, steps) if ic else None
    steps.append(step("Stability", "The origin is %s." % _stability(eigenvalues)))
    return _package("eigen", name, "The system is linear with constant coefficients.",
                    steps, A, general, particular, ic)


def exponential3(A, ic):
    name = "Matrix exponential"
    _, data = _eigen_data(A)
    if data is None:
        return _na("expm", name, "The eigenvalues have no simple closed form, so the "
                                 "exponential cannot be written exactly.")
    E = _expm(A)
    steps = [
        step("Use the matrix exponential",
             "The solution of \\(\\mathbf{X}' = A\\mathbf{X}\\) is \\(\\mathbf{X} = e^{At}\\mathbf{X}(0)\\).",
             ["A = %s" % _mat(A)]),
        step("Compute the exponential", "Through the eigenvalues of \\(A\\),",
             ["e^{At} = %s" % _mat(E)]),
    ]
    general = (E * sp.Matrix([C1, C2, C3])).applyfunc(sp.expand)
    steps.append(step("Multiply by the initial vector",
                      "Here \\(C_1 = x(0)\\), \\(C_2 = y(0)\\) and \\(C_3 = z(0)\\).",
                      [_components(general)]))
    particular = None
    if ic:
        particular = (E * sp.Matrix(list(ic))).applyfunc(sp.expand)
        steps.append(step("Insert the initial values", "", [_components(particular)]))
    return _package("expm", name, "For a constant matrix the solution is the matrix "
                                  "exponential applied to the initial vector.",
                    steps, A, general, particular, ic)


def _initial(texts, labels):
    given = [t.strip() for t in texts]
    if not any(given):
        return None
    if not all(given):
        raise InputError("Fill in %s, or leave them all empty." % " and ".join(labels))
    return tuple(parse_number(t, label) for t, label in zip(texts, labels))


def _result(latex, field, marks, ic, summary, methods, dep):
    return {
        "schema": 1, "ok": True, "kind": "system", "order": 1, "indep": "t", "dep": dep,
        "input_latex": latex, "normal_latex": latex, "rhs_expr": None,
        "field": field, "equilibria": marks,
        "ic": None if not ic else dict(
            {"x0": float(ic[0]), "y0": float(ic[1]),
             "latex": ",\\quad ".join("%s(0) = %s" % (_NAMES[i], tex(v)) for i, v in enumerate(ic))},
            **({"z0": float(ic[2])} if len(ic) == 3 else {})),
        "summary": summary, "other_types": [], "methods": methods,
    }


def _solve3(f_text, g_text, h_text, texts):
    exprs = []
    symbols = None
    for text in (f_text, g_text, h_text):
        expr, symbols = parse_expression(text, ["x", "y", "z", "t"])
        exprs.append(expr)
    x, y, z, t = (symbols[n] for n in ("x", "y", "z", "t"))
    unknowns = (x, y, z)
    A = sp.Matrix([[sp.diff(e, v) for v in unknowns] for e in exprs]).applyfunc(sp.simplify)
    rest = [sp.simplify(e - sum(A[i, j] * unknowns[j] for j in range(3))) for i, e in enumerate(exprs)]
    if A.has(x, y, z, t) or any(r != 0 for r in rest):
        raise InputError("Three-equation systems are supported when they are linear "
                         "and homogeneous with constant coefficients, such as "
                         "x' = x + y, y' = y + z, z' = 2z.")
    ic = _initial(texts, ("x(0)", "y(0)", "z(0)"))
    methods = [eigen3(A, ic), exponential3(A, ic)]
    _, data = _eigen_data(A)
    if data is None:
        summary = ("Linear 3\u00d73 system whose eigenvalues have no simple closed "
                   "form, so it cannot be solved exactly here.")
    else:
        summary = "Linear 3\u00d73 system. The origin is %s." % _stability(
            [v for v, m, _ in data for _ in range(m)])
    latex = ",\\qquad ".join("%s' = %s" % (n, tex(e)) for n, e in zip(_NAMES, exprs))
    return _result(latex, None, [], ic, summary, methods, "x, y, z")


def solve(f_text, g_text, x0_text, y0_text, h_text="", z0_text=""):
    if not f_text.strip() or not g_text.strip():
        raise InputError("Enter both equations: x' and y'.")
    if h_text.strip():
        return _solve3(f_text, g_text, h_text, (x0_text, y0_text, z0_text))
    f, symbols = parse_expression(f_text, ["x", "y", "t"])
    g, _ = parse_expression(g_text, ["x", "y", "t"])
    x, y, t = symbols["x"], symbols["y"], symbols["t"]
    ic = _initial((x0_text, y0_text), ("x(0)", "y(0)"))

    A = sp.Matrix([[sp.diff(f, x), sp.diff(f, y)], [sp.diff(g, x), sp.diff(g, y)]]).applyfunc(sp.simplify)
    b = sp.Matrix([sp.simplify(f - A[0, 0] * x - A[0, 1] * y),
                   sp.simplify(g - A[1, 0] * x - A[1, 1] * y)])
    linear_part = not A.has(x, y, t) and not b.has(x, y)
    if (f.has(t) or g.has(t)) and not linear_part:
        raise InputError("The right-hand sides may depend on t only when the system "
                         "is linear in x and y with constant coefficients, as in "
                         "x' = x + 2y + e^t. Nonlinear systems must be autonomous.")
    b = b.xreplace({t: T})
    names = {x: "x", y: "y"}
    latex = "x' = %s,\\qquad y' = %s" % (tex(f), tex(g))
    field = {"f": _plot(f, names), "g": _plot(g, names)}
    homogeneous_only = ("This solves homogeneous systems \\(\\mathbf{X}' = A\\mathbf{X}\\); "
                        "here there is a forcing term.")
    marks = []

    if linear_part and b == sp.zeros(2, 1):
        methods = [eigen(A, ic), exponential(A, ic),
                   _na("linearize", "Linearization at equilibria",
                       "The system is already linear, so it can be solved exactly.")]
        kind, stability = classify(A)
        if A.det() != 0:
            marks = [{"x": 0.0, "y": 0.0, "label": kind}]
        summary = "Linear system. The origin is %s: %s." % (_a(kind), stability)
    elif linear_part:
        methods = [shift(A, b, ic), variation(A, b, ic),
                   _na("eigen", "Eigenvalue method", homogeneous_only),
                   _na("expm", "Matrix exponential", homogeneous_only)]
        if b.has(T):
            field = None
            summary = ("Linear system with forcing that depends on t. There is no "
                       "fixed direction field; the graph shows the solution curve "
                       "when initial values are given.")
        else:
            kind, stability = classify(A)
            if A.det() != 0:
                Xp = -A.inv() * b
                marks = [{"x": float(Xp[0]), "y": float(Xp[1]), "label": kind}]
                summary = ("Linear system with constant forcing. Its equilibrium at "
                           "(%s, %s) is %s: %s." % (sp.sstr(Xp[0]), sp.sstr(Xp[1]), _a(kind), stability))
            else:
                summary = "Linear system with constant forcing and a singular matrix."
    else:
        method, marks = linearize(f, g, x, y)
        reason = ("These solve linear systems; this one is nonlinear.")
        methods = [method, _na("eigen", "Eigenvalue method", reason),
                   _na("expm", "Matrix exponential", reason)]
        summary = ("Nonlinear system with %d equilibrium point%s. Tap the phase plane "
                   "to draw trajectories." % (len(marks), "" if len(marks) == 1 else "s"))
    return _result(latex, field, marks, ic, summary, methods, "x, y")


def preview(text):
    """LaTeX for 'f;g' or 'f;g;h' as typed so far."""
    parts = (text.split(";") + ["", ""])[:3]
    three = bool(parts[2].strip())
    names = ["x", "y", "z", "t"] if three else ["x", "y", "t"]
    lines = []
    for name, part in zip(("x'", "y'", "z'"), parts):
        if part.strip():
            expr, _ = parse_expression(part, names)
            lines.append("%s = %s" % (name, tex(expr)))
    return ",\\qquad ".join(lines)
