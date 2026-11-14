"""Linear algebra with the working shown: row reduction, systems, determinants,
inverses, eigenvalues, subspaces, independence and the Wronskian."""
import re

import sympy as sp

from .parsing import InputError, parse_expression, parse_number
from .toolkit import method, not_applicable, result, step
from .util import tex

MAX_SIZE = 4
OPERATIONS = ("rref", "solve", "det", "inverse", "eigen", "spaces", "independence", "wronskian")


# --------------------------------------------------------------------------
# Input and LaTeX
# --------------------------------------------------------------------------

def parse_matrix(text, label="The matrix"):
    """Rows separated by ';' or new lines, entries by spaces or commas."""
    rows = [r.strip() for r in re.split(r"[;\n]", text or "") if r.strip()]
    if not rows:
        raise InputError("%s is empty. Type rows separated by ; such as 1 2; 3 4." % label)
    data = []
    for row in rows:
        entries = [e for e in re.split(r"[,\s]+", row) if e]
        data.append([parse_number(e, "Each entry of %s" % label.lower()) for e in entries])
    width = len(data[0])
    if any(len(r) != width for r in data):
        raise InputError("Every row of %s needs the same number of entries." % label.lower())
    if len(data) > MAX_SIZE or width > MAX_SIZE:
        raise InputError("%s can be at most %d by %d." % (label, MAX_SIZE, MAX_SIZE))
    return sp.Matrix(data)


def mat(M):
    body = "\\\\".join(" & ".join(tex(e) for e in M.row(i)) for i in range(M.rows))
    return "\\begin{pmatrix}%s\\end{pmatrix}" % body


def aug(M, split):
    """[left | right] with a bar after `split` columns."""
    body = "\\\\".join(" & ".join(tex(e) for e in M.row(i)) for i in range(M.rows))
    cols = "c" * split + "|" + "c" * (M.cols - split)
    return "\\left(\\begin{array}{%s}%s\\end{array}\\right)" % (cols, body)


def vec(v):
    return "\\begin{pmatrix}%s\\end{pmatrix}" % "\\\\".join(tex(e) for e in v)


def _coef(value):
    return "\\left(%s\\right)" % tex(value) if value.could_extract_minus_sign() or value.is_Add else tex(value)


# --------------------------------------------------------------------------
# Row reduction with every operation recorded
# --------------------------------------------------------------------------

def reduce(M, limit=None):
    """Gauss-Jordan elimination on the first `limit` columns (all by default).

    Returns (R, ops, pivots, swaps, scale) where ops is a list of
    (description LaTeX, matrix after), swaps the number of row swaps and
    scale the product of the scaling factors (for determinants).
    """
    R = M.copy()
    ops, pivots = [], []
    swaps, scale = 0, sp.Integer(1)
    row = 0
    for col in range(R.cols if limit is None else limit):
        if row >= R.rows:
            break
        pivot = next((r for r in range(row, R.rows) if R[r, col] != 0), None)
        if pivot is None:
            continue
        if pivot != row:
            R.row_swap(pivot, row)
            swaps += 1
            ops.append(("R_{%d} \\leftrightarrow R_{%d}" % (row + 1, pivot + 1), R.copy()))
        if R[row, col] != 1:
            factor = R[row, col]
            R[row, :] = (R[row, :] / factor).applyfunc(sp.nsimplify)
            scale *= factor
            ops.append(("R_{%d} \\to %s R_{%d}" % (row + 1, _coef(1 / factor), row + 1), R.copy()))
        for r in range(R.rows):
            if r != row and R[r, col] != 0:
                factor = R[r, col]
                R[r, :] = (R[r, :] - factor * R[row, :]).applyfunc(sp.nsimplify)
                sign = "-" if not factor.could_extract_minus_sign() else "+"
                mag = factor if sign == "-" else -factor
                ops.append(("R_{%d} \\to R_{%d} %s %sR_{%d}"
                            % (r + 1, r + 1, sign, "" if mag == 1 else tex(mag) + "\\,", row + 1), R.copy()))
        pivots.append(col)
        row += 1
    return R, ops, pivots, swaps, scale


def _op_steps(ops, show):
    """One step per row operation; `show` renders a matrix."""
    return [step("Row operation %d" % (i + 1), "", [text, show(M)]) for i, (text, M) in enumerate(ops)]


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------

def _rref(A):
    R, ops, pivots, _, _ = reduce(A)
    steps = [step("Start", "Use row operations to reach reduced row echelon form: each pivot "
                           "is 1 and is the only non-zero entry in its column.", [mat(A)])]
    steps += _op_steps(ops, mat)
    steps.append(step("Read off the pivots",
                      "There %s %d pivot%s, in column%s %s, so the rank is %d."
                      % ("is" if len(pivots) == 1 else "are", len(pivots), "" if len(pivots) == 1 else "s",
                         "" if len(pivots) == 1 else "s", ", ".join(str(p + 1) for p in pivots) or "none",
                         len(pivots))))
    ok = R == A.rref()[0]
    return ("Row reduction", "\\operatorname{rref}%s" % mat(A), "Rank %d." % len(pivots),
            [method("rref", "Gauss–Jordan elimination", "Any matrix can be reduced this way.",
                    steps, "\\operatorname{rref}(A) = %s" % mat(R), verified=ok,
                    note="Agrees with an independent computation of the reduced form.")])


def _names(n):
    return ["x", "y", "z", "w"][:n] if n <= 4 else ["x_%d" % (i + 1) for i in range(n)]


def _solve(A, b):
    if b.cols != 1 or b.rows != A.rows:
        raise InputError("The right-hand side needs one number for each row of the matrix (%d)." % A.rows)
    n = A.cols
    names = _names(n)
    M = A.row_join(b)
    R, ops, pivots, _, _ = reduce(M, limit=n)
    steps = [step("Write the augmented matrix", "Put the right-hand side next to the coefficients.",
                  [aug(M, n)])]
    steps += _op_steps(ops, lambda X: aug(X, n))
    inconsistent = any(all(R[r, c] == 0 for c in range(n)) and R[r, n] != 0 for r in range(R.rows))
    free = [c for c in range(n) if c not in pivots]
    methods = []
    if inconsistent:
        steps.append(step("Interpret", "A row reads \\(0 = \\text{non-zero}\\), which is impossible, "
                                       "so the system has no solution."))
        summary = "No solution: the system is inconsistent."
        answer, ok, note = "\\text{no solution}", None, ""
        solution = None
    elif not free:
        solution = sp.Matrix([R[pivots.index(c), n] for c in range(n)])
        steps.append(step("Interpret", "Every variable has a pivot, so the solution is unique."))
        answer = ",\\quad ".join("%s = %s" % (names[i], tex(solution[i])) for i in range(n))
        ok = (A * solution - b).applyfunc(sp.simplify) == sp.zeros(A.rows, 1)
        note = "Substituting back gives \\(A\\mathbf{x} = \\mathbf{b}\\)."
        summary = "Exactly one solution."
    else:
        params = sp.symbols("t s r q")[:len(free)]
        particular = sp.zeros(n, 1)
        for i, c in enumerate(pivots):
            particular[c] = R[i, n]
        directions = []
        for k, f in enumerate(free):
            d = sp.zeros(n, 1)
            d[f] = 1
            for i, c in enumerate(pivots):
                d[c] = -R[i, f]
            directions.append(d)
        solution = particular
        steps.append(step(
            "Interpret",
            "The variable%s %s ha%s no pivot, so %s free. Call %s %s and solve for the others."
            % ("" if len(free) == 1 else "s", ", ".join("\\(%s\\)" % names[f] for f in free),
               "s" if len(free) == 1 else "ve", "it is" if len(free) == 1 else "they are",
               "it" if len(free) == 1 else "them", ", ".join("\\(%s\\)" % tex(p) for p in params))))
        answer = "\\mathbf{x} = %s" % vec(particular) + "".join(
            " + %s%s" % (tex(p), vec(d)) for p, d in zip(params, directions))
        ok = (A * particular - b == sp.zeros(A.rows, 1)
              and all(A * d == sp.zeros(A.rows, 1) for d in directions))
        note = ("The particular solution satisfies the system and each direction "
                "vector is sent to zero by \\(A\\).")
        summary = "Infinitely many solutions (%d free variable%s)." % (len(free), "" if len(free) == 1 else "s")
    methods.append(method("gauss", "Gaussian elimination", "Works for every linear system.",
                          steps, answer, verified=ok, note=note))

    square = A.rows == A.cols
    det = A.det() if square else None
    if square and det != 0 and n <= 3:
        lines = ["\\det A = %s" % tex(det)]
        values = []
        for i in range(n):
            Ai = A.copy()
            Ai[:, i] = b
            values.append(Ai.det() / det)
            lines.append("%s = \\frac{\\det A_{%d}}{\\det A} = \\frac{%s}{%s} = %s"
                         % (names[i], i + 1, tex(Ai.det()), tex(det), tex(values[-1])))
        csteps = [step("Compute the determinants",
                       "\\(A_i\\) is \\(A\\) with column \\(i\\) replaced by the right-hand side.", lines)]
        methods.append(method("cramer", "Cramer's rule",
                              "The matrix is square with a non-zero determinant.", csteps,
                              ",\\quad ".join("%s = %s" % (names[i], tex(values[i])) for i in range(n)),
                              verified=sp.Matrix(values) == solution,
                              note="Agrees with Gaussian elimination."))
    else:
        methods.append(not_applicable("cramer", "Cramer's rule",
                                      "It needs a square matrix (up to 3 by 3 here) with a non-zero determinant."))
    if square and det != 0:
        inv = A.inv()
        isteps = [step("Find the inverse", "See the Inverse operation for the working.",
                       ["A^{-1} = %s" % mat(inv)]),
                  step("Multiply", "", ["\\mathbf{x} = A^{-1}\\mathbf{b} = %s%s = %s"
                                        % (mat(inv), vec(b), vec(inv * b))])]
        methods.append(method("inverse", "Inverse matrix", "The matrix is invertible.", isteps,
                              "\\mathbf{x} = %s" % vec(inv * b), verified=inv * b == solution,
                              note="Agrees with Gaussian elimination."))
    else:
        methods.append(not_applicable("inverse", "Inverse matrix",
                                      "The matrix is not square or its determinant is 0, so it has no inverse."))
    return "Linear system", "%s\\mathbf{x} = %s" % (mat(A), vec(b)), summary, methods


def _minor_tex(M):
    return "\\begin{vmatrix}%s\\end{vmatrix}" % "\\\\".join(
        " & ".join(tex(e) for e in M.row(i)) for i in range(M.rows))


def _cofactor_steps(A):
    n = A.rows
    if n == 1:
        return [step("One entry", "The determinant of a 1 by 1 matrix is its entry.", [tex(A[0, 0])])]
    if n == 2:
        a, b, c, d = A
        return [step("Use the 2 by 2 formula", "\\(\\det = ad - bc\\).",
                     ["%s = %s\\cdot%s - %s\\cdot%s = %s" % (_minor_tex(A), _coef(a), _coef(d),
                                                              _coef(b), _coef(c), tex(a * d - b * c))])]
    # Expand along the row or column with the most zeros.
    lines = [("row", i, sum(1 for e in A.row(i) if e == 0)) for i in range(n)]
    lines += [("column", j, sum(1 for e in A.col(j) if e == 0)) for j in range(n)]
    kind, index, zeros = max(lines, key=lambda t: t[2])
    terms, subs, total = [], [], sp.Integer(0)
    for k in range(n):
        i, j = (index, k) if kind == "row" else (k, index)
        entry = A[i, j]
        if entry == 0:
            continue
        minor = A.minor_submatrix(i, j)
        value = minor.det()
        sign = (-1) ** (i + j)
        terms.append("%s%s%s" % ("+" if sign > 0 else "-", _coef(entry), _minor_tex(minor)))
        subs.append(step("Minor for row %d, column %d" % (i + 1, j + 1), "",
                         ["%s = %s" % (_minor_tex(minor), tex(value))]))
        total += sign * entry * value
    body = " ".join(terms).lstrip("+")
    return [step("Expand along %s %d" % (kind, index + 1),
                 "Each entry is multiplied by its minor and a sign from the checkerboard "
                 "pattern \\((-1)^{i+j}\\).%s" % (" This %s has the most zeros, so it needs the fewest minors."
                                                  % kind if zeros else ""),
                 ["%s = %s" % (_minor_tex(A), body), "= %s" % tex(total)], subs)]


def _det(A):
    if A.rows != A.cols:
        raise InputError("Only a square matrix has a determinant.")
    value = A.det()
    methods = [method("cofactor", "Cofactor expansion", "Works for any square matrix.",
                      _cofactor_steps(A), "\\det A = %s" % tex(value), verified=True,
                      note="Agrees with the row-reduction method.")]
    R, ops, pivots, swaps, scale = reduce(A)
    rsteps = [step("Reduce and keep track",
                   "A row swap changes the sign of the determinant, scaling a row by \\(c\\) "
                   "scales it by \\(c\\), and adding a multiple of one row to another leaves it unchanged.",
                   [mat(A)])]
    rsteps += _op_steps(ops, mat)
    if len(pivots) < A.rows:
        rsteps.append(step("Conclude", "A zero row appears, so the determinant is 0."))
        tracked = sp.Integer(0)
    else:
        tracked = sp.nsimplify((-1) ** swaps * scale)
        rsteps.append(step("Conclude",
                           "The reduced matrix is the identity (determinant 1). Undoing %d swap%s and "
                           "the scalings gives the determinant." % (swaps, "" if swaps == 1 else "s"),
                           ["\\det A = (-1)^{%d}\\cdot %s = %s" % (swaps, _coef(scale), tex(tracked))]))
    methods.append(method("rowreduce", "Row reduction", "Usually quicker for larger matrices.",
                          rsteps, "\\det A = %s" % tex(tracked), verified=tracked == value,
                          note="Agrees with the cofactor expansion."))
    return "Determinant", "\\det%s" % mat(A), "The determinant is %s." % sp.sstr(value), methods


def _inverse(A):
    if A.rows != A.cols:
        raise InputError("Only a square matrix can have an inverse.")
    n = A.rows
    det = A.det()
    if det == 0:
        reason = "The determinant is 0, so the matrix is singular and has no inverse."
        return ("Inverse", "%s^{-1}" % mat(A), "This matrix is singular: its determinant is 0, so it has no inverse.",
                [not_applicable("gaussjordan", "Gauss–Jordan on [A | I]", reason),
                 not_applicable("adjugate", "Adjugate formula", reason)])
    inv = A.inv()
    ok = A * inv == sp.eye(n)
    M = A.row_join(sp.eye(n))
    R, ops, _, _, _ = reduce(M, limit=n)
    gsteps = [step("Augment with the identity",
                   "Row-reduce \\([A \\mid I]\\). When the left half becomes \\(I\\), the right half is \\(A^{-1}\\).",
                   [aug(M, n)])]
    gsteps += _op_steps(ops, lambda X: aug(X, n))
    methods = [method("gaussjordan", "Gauss–Jordan on [A | I]", "The determinant is not 0.",
                      gsteps, "A^{-1} = %s" % mat(R[:, n:]), verified=ok and R[:, n:] == inv,
                      note="Multiplying gives \\(AA^{-1} = I\\).")]
    if n <= 3:
        cof = A.cofactor_matrix()
        asteps = [step("Compute the determinant", "", ["\\det A = %s" % tex(det)]),
                  step("Form the matrix of cofactors",
                       "Entry \\((i, j)\\) is \\((-1)^{i+j}\\) times the minor obtained by deleting "
                       "row \\(i\\) and column \\(j\\).", ["C = %s" % mat(cof)]),
                  step("Transpose and divide",
                       "The adjugate is the transpose of the cofactor matrix.",
                       ["A^{-1} = \\frac{1}{\\det A}C^{T} = \\frac{1}{%s}%s" % (tex(det), mat(cof.T))])]
        methods.append(method("adjugate", "Adjugate formula", "Practical for 2 by 2 and 3 by 3 matrices.",
                              asteps, "A^{-1} = %s" % mat(cof.T / det), verified=cof.T / det == inv,
                              note="Multiplying gives \\(AA^{-1} = I\\)."))
    else:
        methods.append(not_applicable("adjugate", "Adjugate formula",
                                      "It needs sixteen 3 by 3 determinants for a 4 by 4 matrix; "
                                      "row reduction is the practical route."))
    return "Inverse", "%s^{-1}" % mat(A), "The matrix is invertible.", methods


def _eigen(A):
    if A.rows != A.cols:
        raise InputError("Eigenvalues are defined for square matrices.")
    n = A.rows
    lam = sp.Symbol("lambda")
    poly = sp.factor((A - lam * sp.eye(n)).det())
    name = "Characteristic equation"
    # Roots of an irreducible cubic or quartic are unusable in exact form, so
    # only go on when the polynomial splits into linear and quadratic factors.
    simple = all(sp.Poly(f, lam).degree() <= 2 for f, _ in sp.factor_list(sp.expand(poly))[1])
    roots = sp.roots(sp.Poly(sp.expand(poly), lam)) if simple else {}
    if sum(roots.values()) < n:
        return ("Eigenvalues", "A = %s" % mat(A),
                "The characteristic polynomial has no exact roots this tool can find.",
                [not_applicable("characteristic", name,
                                "The characteristic polynomial \\(%s\\) does not factor into exact roots."
                                % tex(sp.expand(poly)))])
    steps = [step("Form the characteristic polynomial",
                  "Eigenvalues are the numbers \\(\\lambda\\) with \\(\\det(A - \\lambda I) = 0\\).",
                  ["\\det(A - \\lambda I) = %s = %s" % (_minor_tex(A - lam * sp.eye(n)), tex(poly))])]
    ordered = sorted(roots, key=sp.default_sort_key)
    steps.append(step("Solve for the eigenvalues", "",
                      [",\\quad ".join("\\lambda = %s%s" % (tex(v), "" if roots[v] == 1
                                                             else "\\ (\\text{multiplicity }%d)" % roots[v])
                                       for v in ordered)]))
    ok = True
    total_vectors = 0
    for value in ordered:
        M = (A - value * sp.eye(n)).applyfunc(sp.simplify)
        R = M.rref()[0].applyfunc(sp.simplify)
        basis = [v.applyfunc(sp.simplify) for v in M.nullspace()]
        total_vectors += len(basis)
        ok = ok and all((A * v - value * v).applyfunc(sp.simplify) == sp.zeros(n, 1) for v in basis)
        steps.append(step(
            "Eigenvectors for λ = %s" % sp.sstr(value),
            "Solve \\((A - \\lambda I)\\mathbf{v} = \\mathbf{0}\\) by row reduction. The geometric "
            "multiplicity (number of independent eigenvectors) is %d; the algebraic multiplicity is %d."
            % (len(basis), roots[value]),
            ["A - \\lambda I = %s \\ \\to\\ %s" % (mat(M), mat(R)),
             ",\\quad ".join("\\mathbf{v} = %s" % vec(v) for v in basis)]))
    answer = [",\\quad ".join("\\lambda = %s" % tex(v) for v in ordered)]
    if total_vectors == n:
        P, D = A.diagonalize()
        P, D = P.applyfunc(sp.simplify), D.applyfunc(sp.simplify)
        ok = ok and (P * D * P.inv()).applyfunc(sp.simplify) == A
        steps.append(step("Diagonalize",
                          "There are %d independent eigenvectors, so \\(A = PDP^{-1}\\) with the "
                          "eigenvectors as the columns of \\(P\\) and the eigenvalues on the diagonal of \\(D\\)." % n,
                          ["P = %s,\\qquad D = %s" % (mat(P), mat(D))]))
        answer.append("A = PDP^{-1},\\quad P = %s,\\ D = %s" % (mat(P), mat(D)))
        summary = "Diagonalizable, with %d eigenvalue%s." % (len(ordered), "" if len(ordered) == 1 else "s")
        note = "Checked \\(A\\mathbf{v} = \\lambda\\mathbf{v}\\) for each eigenvector and \\(PDP^{-1} = A\\)."
    else:
        P, J = A.jordan_form()
        P, J = P.applyfunc(sp.simplify), J.applyfunc(sp.simplify)
        ok = ok and (P * J * P.inv()).applyfunc(sp.simplify) == A
        steps.append(step("Jordan form",
                          "There are only %d independent eigenvectors for a %d by %d matrix, so it "
                          "cannot be diagonalized. Generalized eigenvectors (solutions of "
                          "\\((A - \\lambda I)\\mathbf{w} = \\mathbf{v}\\)) fill the remaining columns of "
                          "\\(P\\), and \\(A = PJP^{-1}\\) with 1s above the diagonal of \\(J\\)."
                          % (total_vectors, n, n), ["P = %s,\\qquad J = %s" % (mat(P), mat(J))]))
        answer.append("A = PJP^{-1},\\quad P = %s,\\ J = %s" % (mat(P), mat(J)))
        summary = "Not diagonalizable (defective); the Jordan form is given."
        note = "Checked \\(A\\mathbf{v} = \\lambda\\mathbf{v}\\) for each eigenvector and \\(PJP^{-1} = A\\)."
    return ("Eigenvalues", "A = %s" % mat(A), summary,
            [method("characteristic", name, "Works when the characteristic polynomial factors.",
                    steps, answer, verified=bool(ok), note=note)])


def _spaces(A):
    R, ops, pivots, _, _ = reduce(A)
    null = A.nullspace()
    rank = len(pivots)
    steps = [step("Row reduce", "Both spaces are read off the reduced row echelon form.",
                  [mat(A) + "\\ \\to\\ " + mat(R)],
                  [step("Row operations", "", [t for t, _ in ops])] if ops else []),
             step("Column space",
                  "The pivot columns of the original matrix (column%s %s) form a basis."
                  % ("" if rank == 1 else "s", ", ".join(str(p + 1) for p in pivots) or "none"),
                  [",\\quad ".join(vec(A.col(p)) for p in pivots)] if pivots else []),
             step("Null space",
                  "Solve \\(A\\mathbf{x} = \\mathbf{0}\\): one basis vector for each free variable."
                  if null else "Every column has a pivot, so only the zero vector is sent to zero.",
                  [",\\quad ".join(vec(v) for v in null)] if null else ["\\{\\mathbf{0}\\}"]),
             step("Rank and nullity",
                  "Rank plus nullity equals the number of columns (the rank–nullity theorem).",
                  ["\\operatorname{rank} = %d,\\qquad \\operatorname{nullity} = %d,\\qquad %d + %d = %d"
                   % (rank, len(null), rank, len(null), A.cols)])]
    ok = all(A * v == sp.zeros(A.rows, 1) for v in null) and rank + len(null) == A.cols
    answer = "\\operatorname{rank} = %d,\\quad \\operatorname{nullity} = %d" % (rank, len(null))
    return ("Subspaces", "A = %s" % mat(A), "Rank %d, nullity %d." % (rank, len(null)),
            [method("spaces", "Row reduction", "The reduced form shows the pivots and the free variables.",
                    steps, answer, verified=ok,
                    note="Each null-space vector is sent to zero by \\(A\\), and rank + nullity equals the number of columns.")])


def _independence(A):
    R, ops, pivots, _, _ = reduce(A)
    k = A.cols
    independent = len(pivots) == k
    steps = [step("Put the vectors in the columns of a matrix",
                  "The vectors are independent exactly when every column has a pivot.",
                  [mat(A) + "\\ \\to\\ " + mat(R)],
                  [step("Row operations", "", [t for t, _ in ops])] if ops else [])]
    if independent:
        steps.append(step("Conclude", "All %d columns have pivots, so the only combination giving "
                                      "the zero vector is the trivial one." % k))
        answer, ok = "\\text{linearly independent}", True
        note = "The rank equals the number of vectors."
    else:
        relation = A.nullspace()[0]
        scale = sp.ilcm(*[sp.fraction(e)[1] for e in relation])
        relation = relation * scale
        text = " + ".join("%s\\mathbf{v}_{%d}" % (_coef(c), i + 1) for i, c in enumerate(relation) if c != 0)
        steps.append(step("Conclude",
                          "Only %d of the %d columns have pivots, so the vectors are dependent. "
                          "A relation between them:" % (len(pivots), k), ["%s = \\mathbf{0}" % text]))
        answer = "\\text{linearly dependent}"
        ok = A * relation == sp.zeros(A.rows, 1)
        note = "The relation shown does combine the vectors to the zero vector."
    return ("Linear independence", "\\mathbf{v}_1, \\ldots, \\mathbf{v}_{%d} = \\text{columns of } %s" % (k, mat(A)),
            "The vectors are linearly %s." % ("independent" if independent else "dependent"),
            [method("independence", "Row reduction", "Independence is a question about pivots.",
                    steps, answer, verified=ok, note=note)])


def _wronskian(text):
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if not 2 <= len(parts) <= 4:
        raise InputError("Enter two to four functions of x separated by commas, such as e^x, e^(2x).")
    funcs = []
    x = None
    for part in parts:
        expr, symbols = parse_expression(part, ["x"])
        funcs.append(expr)
        x = symbols["x"]
    n = len(funcs)
    W = sp.Matrix([[sp.diff(f, x, k) for f in funcs] for k in range(n)])
    value = sp.simplify(W.det())
    steps = [step("Build the Wronskian matrix",
                  "Row 1 holds the functions, row 2 their first derivatives, and so on.",
                  ["W = %s" % _minor_tex(W)]),
             step("Evaluate the determinant", "", ["W(x) = %s" % tex(value)])]
    if value != 0:
        steps.append(step("Conclude", "The Wronskian is not identically zero, so the functions are "
                                      "linearly independent."))
        summary = "Linearly independent: the Wronskian is not identically zero."
    else:
        steps.append(step("Conclude",
                          "The Wronskian is identically zero. For functions like these (analytic, "
                          "for example solutions of one linear differential equation) that means "
                          "they are linearly dependent."))
        summary = "Linearly dependent: the Wronskian is identically zero."
    return ("Wronskian", "W\\left(%s\\right)" % ",\\ ".join(tex(f) for f in funcs), summary,
            [method("wronskian", "Wronskian determinant",
                    "A non-zero Wronskian proves independence.", steps, "W(x) = %s" % tex(value),
                    verified=None, note="")])


def solve(operation, matrix_text, rhs_text=""):
    if operation not in OPERATIONS:
        raise InputError("Unknown linear algebra operation.")
    if operation == "wronskian":
        title, shown, summary, methods = _wronskian(matrix_text)
    else:
        A = parse_matrix(matrix_text)
        if operation == "solve":
            if not (rhs_text or "").strip():
                raise InputError("Enter the right-hand side b, one number per row, such as 5; 6.")
            # b is a single column, so any separator between its entries will do.
            b = parse_matrix(";".join(e for e in re.split(r"[;,\s]+", rhs_text.strip()) if e),
                             "The right-hand side")
            title, shown, summary, methods = _solve(A, b)
        else:
            title, shown, summary, methods = {
                "rref": _rref, "det": _det, "inverse": _inverse, "eigen": _eigen,
                "spaces": _spaces, "independence": _independence}[operation](A)
    return result("linalg", title, shown, summary, methods)


def preview(operation, text):
    if operation == "wronskian":
        parts = [p for p in text.split(",") if p.strip()]
        return ",\\ ".join(tex(parse_expression(p, ["x"])[0]) for p in parts)
    return mat(parse_matrix(text))
