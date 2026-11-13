"""Boundary value problems and the eigenvalue problem y'' + lambda y = 0."""
import sympy as sp

from . import second_order
from .first_order import _eq, step
from .parsing import InputError, parse_equation, parse_number
from .util import plot_expr, tex, tex_side

_GENERAL_METHODS = (second_order.characteristic, second_order.undetermined,
                    second_order.variation, second_order.cauchy_euler, second_order.reduction)


def _general_solution(eq):
    """(sympy general solution, steps, method name) from the first method that gives a formula."""
    prob = eq.prob
    for method in _GENERAL_METHODS:
        try:
            out = method(eq, None)
        except Exception:
            continue
        text = out.get("plot", {}).get("general_expr") if out.get("applicable") else None
        if not text:
            continue
        expr = sp.sympify(text, locals={"x": prob.x, "C": prob.C1, "C2": prob.C2})
        if {prob.C1, prob.C2} <= expr.free_symbols:
            return expr, out["steps"], out["name"]
    return None, None, None


def _condition(general, prob, point, derivative, value):
    target = sp.diff(general, prob.x) if derivative else general
    try:
        at = sp.simplify(target.subs(prob.x, point))
    except Exception:
        at = sp.nan
    if at.has(sp.zoo, sp.nan, sp.oo):
        raise InputError("The general solution is not defined at %s = %s, so a condition "
                         "cannot be imposed there." % (prob.indep_name, sp.sstr(point)))
    name = "y'(%s)" % tex(point) if derivative else "y(%s)" % tex(point)
    return at - value, "%s = %s:\\quad %s = %s" % (name, tex(value), tex(at), tex(value)), name


def _kind(text, label):
    kind = (text or "y").strip().lower()
    if kind in ("y", "value"):
        return False
    if kind in ("dy", "y'", "slope"):
        return True
    raise InputError("%s must be a condition on y or on y'." % label)


def solve(text, a_text, a_kind, a_value, b_text, b_kind, b_value):
    prob, lhs, rhs = parse_equation(text)
    if prob.order != 2:
        raise InputError("Boundary value problems here are for second-order equations (with y'').")
    xn = prob.indep_name
    for value, label in ((a_text, "The first point"), (a_value, "The first boundary value"),
                         (b_text, "The second point"), (b_value, "The second boundary value")):
        if not (value or "").strip():
            raise InputError("%s is missing: fill in both points and both values." % label)
    a = parse_number(a_text, "The first point")
    b = parse_number(b_text, "The second point")
    alpha = parse_number(a_value, "The first boundary value")
    beta = parse_number(b_value, "The second boundary value")
    if a == b:
        raise InputError("The two boundary points must be different.")
    da, db = _kind(a_kind, "The first condition"), _kind(b_kind, "The second condition")

    eq = second_order.analyse(prob, lhs, rhs)
    general, steps, via = _general_solution(eq)
    if general is None:
        raise InputError("I could not find the general solution of this equation in closed "
                         "form, so the boundary conditions cannot be applied.")
    steps = list(steps)
    steps.append(step(
        "General solution",
        "Found by: %s. A boundary value problem fixes the solution at two different points, "
        "instead of fixing \\(y\\) and \\(y'\\) at one point. Unlike an initial value problem "
        "it can have exactly one solution, none, or infinitely many." % via.lower(),
        [_eq("y", general)]))

    e1, line1, name1 = _condition(general, prob, a, da, alpha)
    e2, line2, name2 = _condition(general, prob, b, db, beta)
    matrix, right = sp.linear_eq_to_matrix([e1, e2], [prob.C1, prob.C2])
    rank = matrix.rank()
    augmented = matrix.row_join(right).rank()
    steps.append(step(
        "Impose the boundary conditions",
        "Each condition gives one linear equation for \\(C_1\\) and \\(C_2\\).",
        [line1, line2]))

    particular, family, verified, note = None, None, None, ""
    if rank == 2:
        values = sp.solve([e1, e2], [prob.C1, prob.C2], dict=True)[0]
        particular = sp.expand(general.subs(values))
        outcome = "exactly one solution"
        steps.append(step(
            "Solve for the constants",
            "The two equations are independent, so there is exactly one solution.",
            ["C_1 = %s,\\qquad C_2 = %s" % (tex(values[prob.C1]), tex(values[prob.C2])),
             _eq("y", particular)]))
        ok, _ = second_order.check(eq, particular)
        try:
            slope = sp.diff(particular, prob.x)
            met = (abs(complex(sp.N((slope if da else particular).subs(prob.x, a) - alpha))) < 1e-9 and
                   abs(complex(sp.N((slope if db else particular).subs(prob.x, b) - beta))) < 1e-9)
        except Exception:
            met = False
        verified = True if (ok and met) else (False if ok is False or not met else None)
        note = ("Checked by substituting back into the equation, and it meets both boundary "
                "conditions." if verified else "The solution could not be confirmed automatically."
                if verified is None else "The check failed.")
    elif augmented > rank:
        outcome = "no solution"
        steps.append(step(
            "No solution",
            "The two equations contradict each other, so no choice of \\(C_1\\) and \\(C_2\\) "
            "satisfies both conditions. This boundary value problem has no solution."))
        note = "Nothing to substitute back: the conditions are inconsistent."
    else:
        outcome = "infinitely many solutions"
        found = sp.solve([e1, e2], [prob.C1, prob.C2], dict=True)
        values = found[0] if found else {}
        family = sp.expand(general.subs(values))
        # Whatever constant is left over is shown as C_1 so the graph slider can vary it.
        left = sorted(family.free_symbols & {prob.C1, prob.C2}, key=lambda s: s.name)
        if left and left[0] != prob.C1:
            family = family.subs(left[0], prob.C1)
        steps.append(step(
            "Infinitely many solutions",
            "The two equations say the same thing (or nothing at all), so one constant stays "
            "free. Every value of it gives a solution.", [_eq("y", family)]))
        ok, _ = second_order.check(eq, family)
        verified = True if ok else None
        note = ("Checked by substituting the whole family back into the equation."
                if ok else "The family could not be confirmed automatically.")

    answer_tex = (_eq("y", particular) if particular is not None else
                  _eq("y", family) if family is not None else "\\text{no solution}")
    steps.append({"title": "Result", "text": "This problem has %s." % outcome,
                  "latex": [answer_tex], "substeps": [], "kind": "answer"})
    if particular is not None or family is not None:
        steps.append({"title": "Check", "text": note, "latex": [], "substeps": [],
                      "kind": {True: "check-ok", False: "check-bad", None: "check-unknown"}[verified]})
    method = {
        "id": "bvp", "name": "General solution, then boundary conditions", "applicable": True,
        "reason": "The equation is linear, so its general solution has two constants for the "
                  "two conditions to determine.",
        "steps": steps,
        "general_latex": _eq("y", family) if family is not None else None,
        "particular_latex": _eq("y", particular) if particular is not None else None,
        "plot": {"general_expr": plot_expr(family, prob) if family is not None else None,
                 "particular_expr": plot_expr(particular, prob) if particular is not None else None,
                 "points": [], "partials": []},
        "verified": verified, "verify_note": note,
    }
    conditions = "%s = %s,\\quad %s = %s" % (name1, tex(alpha), name2, tex(beta))
    equation = "%s = %s" % (tex_side(lhs, prob), tex_side(rhs, prob))
    return {"schema": 1, "ok": True, "kind": "boundary", "order": 2, "indep": xn, "dep": "y",
            "input_latex": "%s,\\qquad %s" % (equation, conditions),
            "normal_latex": equation, "rhs_expr": None, "ic": None,
            "span": float(max(abs(a), abs(b))),
            "summary": "Boundary value problem with %s." % outcome,
            "other_types": [], "methods": [method]}


# --------------------------------------------------------------------------
# Eigenvalue problem y'' + lambda y = 0 on [0, L]
# --------------------------------------------------------------------------

_PAIRS = {
    "dd": ("y(0) = 0", "y(L) = 0"),
    "nn": ("y'(0) = 0", "y'(L) = 0"),
    "dn": ("y(0) = 0", "y'(L) = 0"),
    "nd": ("y'(0) = 0", "y(L) = 0"),
}


def eigen(kind_text, L_text):
    kind = (kind_text or "").strip().lower()
    if kind not in _PAIRS:
        raise InputError("Choose one of the four pairs of boundary conditions.")
    if not (L_text or "").strip():
        raise InputError("Enter the length L of the interval, for example pi or 1.")
    L = parse_number(L_text, "L")
    if not L.is_positive:
        raise InputError("L must be a positive number.")
    x = sp.Symbol("x", real=True)
    n = sp.Symbol("n", integer=True, positive=True)
    left, right = _PAIRS[kind]
    start_zero = kind[0] == "d"      # y(0) = 0 keeps the sine; y'(0) = 0 keeps the cosine
    end_zero = kind[1] == "d"
    trig = sp.sin if start_zero else sp.cos
    kept = "C_2\\sin\\mu x" if start_zero else "C_1\\cos\\mu x"
    killed = "C_1 = 0" if start_zero else "C_2 = 0"
    # The condition at L on the surviving function.
    if start_zero == end_zero:       # sin(mu L) = 0
        condition, mu = "\\sin\\mu L = 0", n * sp.pi / L
        mu_tex = "\\mu_n = \\frac{n\\pi}{L}"
    else:                            # cos(mu L) = 0
        condition, mu = "\\cos\\mu L = 0", (2 * n - 1) * sp.pi / (2 * L)
        mu_tex = "\\mu_n = \\frac{(2n-1)\\pi}{2L}"
    lam = sp.simplify(mu**2)
    y_n = trig(mu * x)
    zero_mode = kind == "nn"

    steps = [
        step("Set up the problem",
             "Find the values of \\(\\lambda\\) for which the problem has a solution other than "
             "\\(y = 0\\). The form of the solution depends on the sign of \\(\\lambda\\), so "
             "treat three cases.",
             ["y'' + \\lambda y = 0,\\qquad %s,\\qquad %s,\\qquad L = %s" % (left, right, tex(L))]),
        step("Case \u03bb < 0",
             "Write \\(\\lambda = -\\mu^2\\) with \\(\\mu > 0\\). The solutions are hyperbolic "
             "functions. The condition at 0 removes one of them, and the one that is left is "
             "never zero (nor is its slope) at \\(x = L\\), so both constants must vanish: "
             "there are no negative eigenvalues.",
             ["y = C_1\\cosh\\mu x + C_2\\sinh\\mu x \\ \\Rightarrow\\ C_1 = C_2 = 0"]),
        step("Case \u03bb = 0",
             ("The equation is \\(y'' = 0\\). Both conditions only ask for zero slope, which every "
              "constant function has, so \\(\\lambda = 0\\) is an eigenvalue with eigenfunction 1."
              if zero_mode else
              "The equation is \\(y'' = 0\\), a straight line. The two conditions force it to be "
              "the zero function, so \\(\\lambda = 0\\) is not an eigenvalue."),
             ["y = C_1 + C_2 x \\ \\Rightarrow\\ " + ("y = C_1" if zero_mode else "C_1 = C_2 = 0")]),
        step("Case \u03bb > 0",
             "Write \\(\\lambda = \\mu^2\\) with \\(\\mu > 0\\). The condition at 0 gives "
             "\\(%s\\). For a non-zero solution the condition at \\(L\\) then needs "
             "\\(%s\\), which happens only for special values of \\(\\mu\\)." % (killed, condition),
             ["y = C_1\\cos\\mu x + C_2\\sin\\mu x \\ \\Rightarrow\\ y = %s" % kept,
              condition, mu_tex + ",\\qquad n = 1, 2, 3, \\dots"]),
    ]
    pairs_tex = "\\lambda_n = %s,\\qquad y_n(x) = %s,\\qquad n = 1, 2, 3, \\dots" % (tex(lam), tex(y_n))
    lines = [pairs_tex]
    if zero_mode:
        lines.insert(0, "\\lambda_0 = 0,\\qquad y_0(x) = 1")
    steps.append(step("Eigenvalues and eigenfunctions",
                      "Each allowed \\(\\mu\\) gives one eigenvalue \\(\\lambda = \\mu^2\\) and "
                      "one eigenfunction (any constant multiple also works).", lines))

    indices = list(range(0, 5)) if zero_mode else list(range(1, 6))
    rows, partials, checks = [], [], []
    for k in indices:
        value = sp.Integer(0) if k == 0 else lam.subs(n, k)
        fn = sp.Integer(1) if k == 0 else y_n.subs(n, k)
        rows.append([str(k), tex(sp.simplify(value)), tex(fn)])
        partials.append({"n": k, "expr": sp.sstr(fn.xreplace({x: sp.Symbol("x")}))})
        slope = sp.diff(fn, x)
        checks.append(sp.simplify(sp.diff(fn, x, 2) + value * fn) == 0
                      and sp.simplify((fn if start_zero else slope).subs(x, 0)) == 0
                      and sp.simplify((fn if end_zero else slope).subs(x, L)) == 0)
    table = step("The first five", "")
    table["table"] = {"headers": ["n", "\\lambda_n", "y_n(x)"], "rows": rows}
    steps.append(table)
    steps.append(step(
        "Orthogonality",
        "Eigenfunctions belonging to different eigenvalues are orthogonal on \\([0, L]\\). "
        "This is what lets a function be expanded in them, exactly as in a Fourier series.",
        ["\\int_0^L y_m(x)\\,y_n(x)\\,dx = 0 \\qquad (m \\ne n)"]))
    ok = all(checks)
    note = ("The first five were substituted into the equation and both boundary conditions."
            if ok else "The eigenfunctions could not be confirmed automatically.")
    steps.append({"title": "Result", "text": "", "latex": lines, "substeps": [], "kind": "answer"})
    steps.append({"title": "Check", "text": note, "latex": [], "substeps": [],
                  "kind": "check-ok" if ok else "check-unknown"})
    method = {"id": "cases", "name": "Three cases for \u03bb", "applicable": True,
              "reason": "The general solution of \\(y'' + \\lambda y = 0\\) changes form with the "
                        "sign of \\(\\lambda\\).",
              "steps": steps, "general_latex": None, "particular_latex": pairs_tex,
              "plot": {"general_expr": None, "particular_expr": None, "points": [], "partials": partials},
              "verified": True if ok else None, "verify_note": note}
    problem = "y'' + \\lambda y = 0,\\qquad %s,\\quad %s" % (
        left.replace("L", tex(L)), right.replace("L", tex(L)))
    return {"schema": 1, "ok": True, "kind": "eigen", "order": 2, "indep": "x", "dep": "y",
            "input_latex": problem, "normal_latex": problem, "rhs_expr": None, "ic": None,
            "span": float(L),
            "summary": "Eigenvalue problem. Use the slider on the graph to step through the "
                       "eigenfunctions.",
            "other_types": [], "methods": [method]}
