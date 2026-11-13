"""Planar autonomous systems x' = f(x, y), y' = g(x, y).

Linear homogeneous systems are solved by the eigenvalue method and by the
matrix exponential. Nonlinear (and affine) systems are analysed by
linearising at each equilibrium point.
"""
import sympy as sp

from .first_order import step
from .parsing import InputError, parse_expression, parse_number
from .util import residual_status, tex

T = sp.Symbol("t", real=True)
C1, C2 = sp.Symbol("C_1", real=True), sp.Symbol("C_2", real=True)
_PLOT_FUNCS = {"sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan",
               "sinh", "cosh", "tanh", "exp", "log", "Abs"}


def _vec(a, b):
    return "\\begin{pmatrix}%s\\\\%s\\end{pmatrix}" % (tex(a), tex(b))


def _mat(M):
    return "\\begin{pmatrix}%s & %s\\\\%s & %s\\end{pmatrix}" % tuple(tex(e) for e in M)


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


def _verify(A, X, ic):
    residual = X.diff(T) - A * X
    statuses = [residual_status(sp.simplify(e)) for e in residual]
    if any(s in ("bad", "partial") for s in statuses):
        return False, "Substituting back did not reproduce the system."
    if any(s == "unknown" for s in statuses):
        return None, "Could not be checked automatically."
    if ic is not None:
        at0 = X.subs(T, 0)
        if any(abs(complex(sp.N(at0[i] - ic[i]))) > 1e-9 for i in range(2)):
            return False, "The initial conditions are not satisfied."
        return True, "Checked by substituting back into both equations, and it meets the initial conditions."
    return True, "Checked by substituting back into both equations."


def _components(X):
    return "x(t) = %s,\\qquad y(t) = %s" % (tex(X[0]), tex(X[1]))


def _package(method_id, name, reason, steps, A, general, particular, ic):
    target = particular if particular is not None else general
    verified, note = _verify(A, target, ic if particular is not None else None)
    names = {T: "x"}
    m = {"id": method_id, "name": name, "applicable": True, "reason": reason, "steps": steps,
         "general_latex": _components(general),
         "particular_latex": _components(particular) if particular is not None else None,
         "plot": {"general_expr": None, "particular_expr": None, "points": [],
                  "x_expr": _plot(particular[0], names) if particular is not None else None,
                  "y_expr": _plot(particular[1], names) if particular is not None else None},
         "verified": verified, "verify_note": note}
    _closing(m, ic is not None)
    return m


def _apply_ic(general, ic, steps):
    at0 = general.subs(T, 0)
    found = sp.solve([at0[0] - ic[0], at0[1] - ic[1]], [C1, C2], dict=True)
    if not found:
        return None
    values = found[0]
    particular = general.subs(values).applyfunc(sp.expand)
    steps.append(step(
        "Apply the initial conditions",
        "Set \\(t = 0\\) and solve the two equations for \\(C_1\\) and \\(C_2\\).",
        [_components(particular)],
        [step("Equations for the constants", "",
              ["%s = %s" % (tex(at0[0]), tex(ic[0])), "%s = %s" % (tex(at0[1]), tex(ic[1])),
               "C_1 = %s,\\qquad C_2 = %s" % (tex(values.get(C1, C1)), tex(values.get(C2, C2)))])]))
    return particular


def eigen(A, ic):
    name = "Eigenvalue method"
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
    particular = _apply_ic(general, ic, steps) if ic else None
    kind, stability = classify(A)
    steps.append(step("Classify the equilibrium",
                      "From the eigenvalues, the origin is %s: %s." % (_a(kind), stability)))
    return _package("eigen", name, "The system is linear with constant coefficients.",
                    steps, A, general, particular, ic)


def exponential(A, ic):
    name = "Matrix exponential"
    E = (A * T).exp().applyfunc(sp.simplify)
    if E.has(sp.I):
        E = E.applyfunc(lambda e: sp.simplify(sp.re(sp.expand_complex(e))))
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


def solve(f_text, g_text, x0_text, y0_text):
    if not f_text.strip() or not g_text.strip():
        raise InputError("Enter both equations: x' and y'.")
    f, symbols = parse_expression(f_text, ["x", "y", "t"])
    g, _ = parse_expression(g_text, ["x", "y", "t"])
    x, y, t = symbols["x"], symbols["y"], symbols["t"]
    if f.has(t) or g.has(t):
        raise InputError("The right-hand sides may depend on x and y only (an "
                         "autonomous system), not on t.")
    ic = None
    if x0_text.strip() or y0_text.strip():
        if not (x0_text.strip() and y0_text.strip()):
            raise InputError("Fill in both x(0) and y(0), or leave both empty.")
        ic = (parse_number(x0_text, "x(0)"), parse_number(y0_text, "y(0)"))

    A = sp.Matrix([[sp.diff(f, x), sp.diff(f, y)], [sp.diff(g, x), sp.diff(g, y)]]).applyfunc(sp.simplify)
    linear = (not A.has(x, y) and sp.simplify(f - A[0, 0] * x - A[0, 1] * y) == 0
              and sp.simplify(g - A[1, 0] * x - A[1, 1] * y) == 0)
    marks = []
    if linear:
        methods = [eigen(A, ic), exponential(A, ic),
                   {"id": "linearize", "name": "Linearization at equilibria", "applicable": False,
                    "reason": "The system is already linear, so it can be solved exactly."}]
        kind, stability = classify(A)
        if A.det() != 0:
            marks = [{"x": 0.0, "y": 0.0, "label": kind}]
        summary = "Linear system. The origin is %s: %s." % (_a(kind), stability)
    else:
        method, marks = linearize(f, g, x, y)
        reason = ("These solve linear homogeneous systems \\(\\mathbf{X}' = A\\mathbf{X}\\); "
                  "this one is not of that form.")
        methods = [method,
                   {"id": "eigen", "name": "Eigenvalue method", "applicable": False, "reason": reason},
                   {"id": "expm", "name": "Matrix exponential", "applicable": False, "reason": reason}]
        label = ("Linear system with constant terms" if not A.has(x, y)
                 else "Nonlinear system")
        summary = ("%s with %d equilibrium point%s. Tap the phase plane to draw "
                   "trajectories." % (label, len(marks), "" if len(marks) == 1 else "s"))
    names = {x: "x", y: "y"}
    return {
        "schema": 1, "ok": True, "kind": "system", "order": 1, "indep": "t", "dep": "x, y",
        "input_latex": "x' = %s,\\qquad y' = %s" % (tex(f), tex(g)),
        "normal_latex": "x' = %s,\\qquad y' = %s" % (tex(f), tex(g)),
        "rhs_expr": None,
        "field": {"f": _plot(f, names), "g": _plot(g, names)},
        "equilibria": marks,
        "ic": None if not ic else {"x0": float(ic[0]), "y0": float(ic[1]),
                                   "latex": "x(0) = %s,\\quad y(0) = %s" % (tex(ic[0]), tex(ic[1]))},
        "summary": summary, "other_types": [], "methods": methods,
    }


def preview(text):
    """LaTeX for the pair 'f;g' as typed so far."""
    parts = (text.split(";") + [""])[:2]
    lines = []
    for name, part in zip(("x'", "y'"), parts):
        if part.strip():
            expr, _ = parse_expression(part, ["x", "y", "t"])
            lines.append("%s = %s" % (name, tex(expr)))
    return ",\\qquad ".join(lines)
