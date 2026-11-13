"""First-order ODE methods. Each returns a dict describing one solution path.

A method dict has: id, name, applicable, reason, and when applicable also
steps, general_latex, particular_latex, plot, verified, verify_note.
"""
import sympy as sp
from sympy.solvers.ode import constantsimp

from .util import (TIMEOUT, check_solution, dydx, hard_to_solve, has_integral, integrate,
                   is_zero, plot_expr, residual_status, run_with_timeout, sample_zero,
                   signed,
                   tex, too_complex, with_abs)


def step(title, text, latex=None, substeps=None):
    return {"title": title, "text": text, "latex": list(latex or []),
            "substeps": list(substeps or [])}


def _eq(left, right):
    return "%s = %s" % (left if isinstance(left, str) else tex(left),
                        right if isinstance(right, str) else tex(right))


def _result(method_id, name, reason, steps, general, particular, prob, f,
            ic, general_tex=None, particular_tex=None, extra_verify=None):
    """Package a finished method, running the verification step."""
    x0, y0 = ic if ic else (None, None)
    verified, note = None, "Could not be checked automatically."
    target = particular if particular is not None else general
    if target is not None and not has_integral(target):
        if particular is not None:
            verified, note = check_solution(prob, f, particular, x0, y0)
        else:
            verified, note = check_solution(prob, f, general)
    elif extra_verify is not None:
        verified, note = extra_verify
    return {
        "id": method_id, "name": name, "applicable": True, "reason": reason,
        "steps": steps,
        "general_latex": general_tex or (_eq("y", general) if general is not None else None),
        "particular_latex": particular_tex or (_eq("y", particular) if particular is not None else None),
        "plot": {
            "general_expr": plot_expr(general, prob),
            "particular_expr": plot_expr(particular, prob),
        },
        "verified": verified, "verify_note": note,
    }


def _not_applicable(method_id, name, reason):
    return {"id": method_id, "name": name, "applicable": False, "reason": reason}


def _matches_ic(candidate, prob, f, x0, y0):
    """Does the candidate pass through (x0, y0) with the slope the ODE demands?"""
    try:
        at = candidate.subs(prob.x, x0)
        if at.has(sp.zoo, sp.nan, sp.oo) or at.free_symbols:
            return False
        if not (is_zero(at - y0) or abs(complex(sp.N(at - y0))) < 1e-9):
            return False
        slope = sp.diff(candidate, prob.x).subs(prob.x, x0)
        wanted = f.subs({prob.x: x0, prob.y: y0})
        return abs(complex(sp.N(slope - wanted))) < 1e-7
    except Exception:
        return False


def _solve_constant(branches, prob, f, x0, y0):
    """Find C from y(x0) = y0. Returns (equation_latex, C value, particular) or None."""
    for branch in branches:
        try:
            at = branch.subs(prob.x, x0)
            if at.has(sp.zoo, sp.nan):
                continue
            values = run_with_timeout(lambda: sp.solve(sp.Eq(at, y0), prob.C), TIMEOUT, [])
        except Exception:
            continue
        for value in values:
            if value.is_real is False or value.has(sp.zoo, sp.nan, sp.oo, sp.I):
                continue
            candidate = sp.simplify(branch.subs(prob.C, value))
            if _matches_ic(candidate, prob, f, x0, y0):
                return _eq(y0, at), value, candidate
    return None


def _absorb_exp(expr, C):
    """Rewrite e^(kC + r) as C*e^r when C only ever appears as that e^(kC)."""
    K = sp.Dummy("K")
    seen = set()

    def swap(e):
        arg = sp.expand(e.args[0])
        k = arg.coeff(C)
        rest = arg - k * C
        if k == 0 or rest.has(C) or not k.is_number:
            return e
        seen.add(k)
        return K * sp.exp(rest)

    out = expr.replace(lambda e: isinstance(e, sp.exp) and e.has(C), swap)
    if len(seen) != 1 or out.has(C):
        return expr
    return out.xreplace({K: C})


def _is_real_branch(expr, prob):
    """True if the formula takes real values somewhere (drops complex artefacts)."""
    import random
    rng = random.Random(3)
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if not symbols:
        return expr.is_real is not False
    for _ in range(30):
        point = {s: sp.Float(rng.uniform(-3, 3)) for s in symbols}
        try:
            value = complex(expr.evalf(subs=point))
        except Exception:
            continue
        if value == value and abs(value) != float("inf") and abs(value.imag) < 1e-9:
            return True
    return False


def _prefer_abs(branch, prob, f):
    """Use ln|u| in a solved branch only when that form still solves the ODE."""
    shown = with_abs(branch)
    if shown != branch and check_solution(prob, f, shown)[0] is True:
        return shown
    return branch


# --------------------------------------------------------------------------
# Separation of variables
# --------------------------------------------------------------------------

def separable(prob, f, ic):
    name = "Separation of variables"
    x, y, C = prob.x, prob.y, prob.C
    d = dydx(prob)
    xn = prob.indep_name

    parts = None
    for candidate in (f, sp.factor(f), sp.together(f)):
        try:
            parts = sp.separatevars(candidate, dict=True, symbols=(x, y))
        except Exception:
            parts = None
        if parts:
            break
    if not parts:
        return _not_applicable(
            "separable", name,
            "The right-hand side \\(%s\\) cannot be written as a product "
            "\\(g(%s)\\,h(y)\\), so the variables cannot be separated." % (tex(f), xn))

    g = sp.simplify(parts["coeff"] * parts[x])
    h = sp.simplify(parts[y])
    # Keep the sign with g so h looks natural (e.g. h = y rather than -y).
    if h.could_extract_minus_sign():
        g, h = -g, -h

    steps = []
    direct = (h == 1)
    if direct:
        reason = ("The right-hand side depends only on \\(%s\\), so the equation "
                  "is solved by integrating once." % xn)
        steps.append(step(
            "Recognise the form",
            "The right-hand side has no \\(y\\) in it. Multiply both sides by "
            "\\(d%s\\) to get \\(dy\\) alone." % xn,
            ["%s = %s" % (d, tex(g)), "dy = %s\\,d%s" % (_paren(g), xn)]))
    else:
        reason = ("The right-hand side factors as \\(g(%s)\\,h(y)\\) with "
                  "\\(g = %s\\) and \\(h = %s\\)." % (xn, tex(g), tex(h)))
        steps.append(step(
            "Recognise the form",
            "The equation is separable because the right-hand side is a "
            "function of \\(%s\\) times a function of \\(y\\)." % xn,
            ["%s = \\underbrace{%s}_{g(%s)}\\cdot\\underbrace{%s}_{h(y)}"
             % (d, tex(g), xn, tex(h))]))

    equilibria = []
    if not direct:
        try:
            equilibria = [r for r in sp.solve(h, y)
                          if r.is_real is not False and not r.free_symbols]
        except Exception:
            equilibria = []
        if equilibria:
            listed = ",\\quad ".join(_eq("y", r) for r in equilibria)
            steps.append(step(
                "Note the constant solutions",
                "Dividing by \\(h(y)\\) in the next step is only allowed where "
                "\\(h(y) \\neq 0\\). Where \\(h(y) = 0\\) the slope is zero, so "
                "these constant functions are solutions on their own:",
                [listed]))
        inv_h = sp.simplify(1 / h)
        steps.append(step(
            "Separate the variables",
            "Divide both sides by \\(h(y)\\) and multiply by \\(d%s\\), so all "
            "the \\(y\\) terms are on the left and all the \\(%s\\) terms are on "
            "the right." % (xn, xn),
            ["%s\\,dy = %s\\,d%s" % (_paren(inv_h), _paren(g), xn)]))
    else:
        inv_h = sp.Integer(1)

    # Initial condition sitting on an equilibrium: the answer is that constant.
    if ic and not direct:
        x0, y0 = ic
        if any(is_zero(r - y0) for r in equilibria):
            steps.append(step(
                "Apply the initial condition",
                "The initial value \\(y(%s) = %s\\) lies on a constant solution "
                "found above, so the solution never leaves it."
                % (tex(x0), tex(y0)),
                [_eq("y", y0)]))
            return _result("separable", name, reason, steps, None, y0, prob, f, ic)

    H = integrate(inv_h, y)
    G = integrate(g, x)
    H_show, G_show = with_abs(H), with_abs(G)
    left_int = "y" if direct else "\\int %s\\,dy" % _paren(inv_h)
    right_int = "\\int %s\\,d%s" % (_paren(g), xn)
    if has_integral(H) or has_integral(G):
        steps.append(step(
            "Integrate both sides",
            "At least one of these integrals has no closed form in elementary "
            "functions, so the solution is left in integral form. The graph "
            "uses a numerical solution instead.",
            ["%s = %s + C" % (left_int, right_int)]))
        out = _result("separable", name, reason, steps, None, None, prob, f, ic,
                      general_tex="%s = %s + C" % (left_int, right_int))
        out["verify_note"] = "Left in integral form; nothing to substitute back."
        return out

    steps.append(step(
        "Integrate both sides",
        "Integrate the left side with respect to \\(y\\) and the right side "
        "with respect to \\(%s\\). One constant of integration is enough."
        % xn,
        ["%s = %s" % (left_int, right_int), "%s = %s" % (tex(H_show), _plus_c(G_show))],
        ([] if direct else [step("Left integral", "", ["%s = %s" % (left_int, tex(H_show))])])
        + [step("Right integral", "", ["%s = %s" % (right_int, tex(G_show))])]))

    implicit = sp.Eq(H, G + C)
    implicit_tex = "%s = %s" % (tex(H_show), _plus_c(G_show))

    # Explicit form
    branches = []
    renamed = False
    if direct:
        branches = [G_show + C]
    else:
        raw = [] if hard_to_solve(H, y) else run_with_timeout(
            lambda: sp.solve(implicit, y), TIMEOUT, [])
        for sol in raw:
            if sol.has(y) or sol.has(sp.I) or has_integral(sol) or too_complex(sol):
                continue
            try:
                simp = _absorb_exp(constantsimp(sol, {C}), C)
            except Exception:
                simp = sol
            if not _is_real_branch(simp, prob):
                continue
            if simp != sol:
                renamed = True
            branches.append(_prefer_abs(simp, prob, f))

    general = None
    general_tex = implicit_tex
    if branches and not direct:
        text = "Solve this equation for \\(y\\)."
        if renamed:
            text += (" An expression made only of the arbitrary constant, such "
                     "as \\(\\pm e^{C}\\) or \\(2C\\), is itself an arbitrary "
                     "constant, so it is renamed \\(C\\).")
        if len(branches) > 1:
            text += " There is more than one branch; an initial condition picks one."
        steps.append(step("Solve for y", text, [_eq("y", b) for b in branches]))
        general = branches[0] if len(branches) == 1 else None
        general_tex = ",\\quad ".join(_eq("y", b) for b in branches)
    elif direct:
        general = branches[0]
        general_tex = _eq("y", general)
    else:
        steps.append(step(
            "General solution (implicit)",
            "This equation cannot be solved for \\(y\\) in a simple closed form, "
            "so the solution is left in implicit form.",
            [implicit_tex]))

    particular = None
    particular_tex = None
    if ic:
        x0, y0 = ic
        found = _solve_constant(branches, prob, f, x0, y0) if branches else None
        if found:
            eq_tex, value, particular = found
            steps.append(step(
                "Apply the initial condition",
                "Substitute \\(%s = %s\\) and \\(y = %s\\), then solve for \\(C\\)."
                % (xn, tex(x0), tex(y0)),
                [_eq("y", particular)],
                [step("Solve for C", "", [eq_tex, _eq("C", value)])]))
        else:
            try:
                value = sp.simplify(H_show.subs(y, y0) - G_show.subs(x, x0))
                usable = value.is_real is not False and not value.has(sp.zoo, sp.nan, sp.oo)
            except Exception:
                usable = False
            if usable:
                particular_tex = "%s = %s" % (tex(H_show), tex(G_show + value))
                steps.append(step(
                    "Apply the initial condition",
                    "Substitute \\(%s = %s\\) and \\(y = %s\\) into the implicit "
                    "solution to find \\(C\\)." % (xn, tex(x0), tex(y0)),
                    [_eq("C", value), particular_tex]))
            else:
                steps.append(step(
                    "Apply the initial condition",
                    "The point \\((%s, %s)\\) is outside the region where this "
                    "solution formula is defined, so no value of \\(C\\) fits."
                    % (tex(x0), tex(y0))))

    extra = None
    if general is None and particular is None:
        ok = is_zero(sp.diff(G, x) / sp.diff(H, y) - f)
        extra = (True, "Checked by differentiating the implicit solution.") if ok else None
    out = _result("separable", name, reason, steps, general, particular, prob, f,
                  ic, general_tex=general_tex, particular_tex=particular_tex,
                  extra_verify=extra)
    if general is None and particular is None and len(branches) > 1:
        checks = [check_solution(prob, f, b)[0] for b in branches]
        if all(c is True for c in checks):
            out["verified"] = True
            out["verify_note"] = "Each branch was checked by substituting back into the equation."
    if equilibria and out.get("general_latex"):
        out["also_latex"] = ",\\quad ".join(_eq("y", r) for r in equilibria)
    return out


def _paren(expr):
    """LaTeX for expr, parenthesised when it is a sum or starts with a minus."""
    if expr.is_Add or expr.could_extract_minus_sign():
        return "\\left(%s\\right)" % tex(expr)
    return tex(expr)


def _signed_term(coef):
    """LaTeX for '+ coef y', written as '- |coef| y' when coef is negative."""
    if coef.could_extract_minus_sign() and not coef.is_Add:
        return "- %s\\,y" % _paren(-coef)
    return "+ %s\\,y" % _paren(coef)


def _plus_c(expr):
    return "C" if expr == 0 else "%s + C" % tex(expr)


# --------------------------------------------------------------------------
# Linear: integrating factor
# --------------------------------------------------------------------------

def linear(prob, f, ic):
    name = "Integrating factor (linear)"
    x, y, C = prob.x, prob.y, prob.C
    xn = prob.indep_name
    d = dydx(prob)

    a = sp.simplify(sp.diff(f, y))
    q = sp.simplify(f - a * y)
    if a.has(y) or q.has(y):
        return _not_applicable(
            "linear", name,
            "The equation is not linear: \\(y\\) appears in a way other than a "
            "plain first power (for example squared, inside a function, or "
            "multiplying itself), so it cannot be written as "
            "\\(y' + p(%s)\\,y = q(%s)\\)." % (xn, xn))
    p = sp.simplify(-a)
    if p == 0:
        return _not_applicable(
            "linear", name,
            "There is no \\(y\\) term, so the integrating factor would be 1. "
            "This is a direct integration, shown under Separation of variables.")

    reason = ("It can be written as \\(y' + p(%s)\\,y = q(%s)\\) with "
              "\\(p = %s\\) and \\(q = %s\\)." % (xn, xn, tex(p), tex(q)))
    steps = [step(
        "Write in standard form",
        "Move the \\(y\\) term to the left so the equation reads "
        "\\(y' + p(%s)\\,y = q(%s)\\)." % (xn, xn),
        ["%s %s = %s" % (d, _signed_term(p), tex(q)),
         "p(%s) = %s,\\qquad q(%s) = %s" % (xn, tex(p), xn, tex(q))])]

    P = integrate(p, x)
    if has_integral(P):
        steps.append(step(
            "Find the integrating factor",
            "The integrating factor is \\(\\mu = e^{\\int p\\,d%s}\\), but this "
            "integral has no closed form in elementary functions. The graph "
            "uses a numerical solution instead." % xn,
            ["\\mu(%s) = e^{\\int %s\\,d%s}" % (xn, tex(p), xn)]))
        out = _result("linear", name, reason, steps, None, None, prob, f, ic,
                      general_tex="y = \\frac{1}{\\mu}\\left(\\int \\mu\\,q\\,d%s + C\\right)" % xn)
        out["verify_note"] = "Left in integral form; nothing to substitute back."
        return out
    mu = sp.powsimp(sp.simplify(sp.exp(P)))
    log_note = ""
    if P.has(sp.log):
        log_note = (" Any one integrating factor works, so the absolute value "
                    "inside the logarithm and the constant of integration are dropped.")
    steps.append(step(
        "Find the integrating factor",
        "The integrating factor is \\(\\mu = e^{\\int p\\,d%s}\\)." % xn + log_note,
        ["\\mu(%s) = e^{\\int %s\\,d%s} = %s" % (xn, tex(p), xn, tex(mu))],
        [step("Integral of p", "", ["\\int %s\\,d%s = %s" % (_paren(p), xn, tex(P))])]))

    muq = sp.simplify(mu * q)
    steps.append(step(
        "Multiply through by the integrating factor",
        "After multiplying by \\(\\mu\\), the left side is exactly the "
        "derivative of the product \\(\\mu\\,y\\) (product rule in reverse).",
        ["%s\\,%s %s = %s" % (_paren(mu), d, _signed_term(sp.simplify(mu * p)), tex(muq)),
         "\\frac{d}{d%s}\\left[%s\\,y\\right] = %s" % (xn, _paren(mu), tex(muq))]))

    R = with_abs(integrate(muq, x))
    right_int = "\\int %s\\,d%s" % (_paren(muq), xn)
    if has_integral(R):
        steps.append(step(
            "Integrate both sides",
            "This integral has no closed form in elementary functions, so the "
            "solution is left in integral form. The graph uses a numerical "
            "solution instead.",
            ["%s\\,y = %s + C" % (_paren(mu), right_int)]))
        out = _result("linear", name, reason, steps, None, None, prob, f, ic,
                      general_tex="y = \\frac{1}{%s}\\left(%s + C\\right)" % (tex(mu), right_int))
        out["verify_note"] = "Left in integral form; nothing to substitute back."
        return out
    if muq == 0:
        steps.append(step(
            "Integrate both sides",
            "The derivative of \\(\\mu\\,y\\) is zero, so \\(\\mu\\,y\\) is a constant.",
            ["%s\\,y = C" % _paren(mu)]))
    else:
        steps.append(step(
            "Integrate both sides",
            "Integrate with respect to \\(%s\\). The left side undoes the derivative." % xn,
            ["%s\\,y = %s" % (_paren(mu), right_int),
             "%s\\,y = %s" % (_paren(mu), _plus_c(R))],
            [step("Right integral", "", ["%s = %s" % (right_int, tex(R))])]))

    general = _tidy((R + C) / mu, C)
    steps.append(step(
        "Solve for y",
        "Divide both sides by the integrating factor.",
        [_eq("y", general)]))

    particular = None
    if ic:
        x0, y0 = ic
        found = _solve_constant([general], prob, f, x0, y0)
        if found:
            eq_tex, value, particular = found
            particular = _tidy(particular, C)
            steps.append(step(
                "Apply the initial condition",
                "Substitute \\(%s = %s\\) and \\(y = %s\\), then solve for \\(C\\)."
                % (xn, tex(x0), tex(y0)),
                [_eq("y", particular)],
                [step("Solve for C", "", [eq_tex, _eq("C", value)])]))
        else:
            steps.append(step(
                "Apply the initial condition",
                "The solution formula is not defined at \\(%s = %s\\) (the "
                "coefficients of the equation are singular there), so this "
                "initial condition cannot be applied." % (xn, tex(x0))))
    return _result("linear", name, reason, steps, general, particular, prob, f, ic)


def _tidy(expr, C):
    """Expand into a sum of terms, keeping whichever form is shorter."""
    expanded = sp.expand(expr)
    simplified = sp.simplify(expr)
    best = min((sp.collect(expanded, C), expanded, simplified), key=sp.count_ops)
    return best


# --------------------------------------------------------------------------
# Exact equations
# --------------------------------------------------------------------------

def exact(prob, f, ic):
    name = "Exact equation"
    x, y, C = prob.x, prob.y, prob.C
    xn = prob.indep_name

    num, den = sp.fraction(sp.together(f))
    candidates = []
    if getattr(prob, "user_form", None):
        candidates.append(prob.user_form)
    candidates.append((sp.expand(-num), sp.expand(den)))

    chosen = None
    for M, N in candidates:
        if is_zero(sp.diff(M, y) - sp.diff(N, x)):
            chosen = (M, N, None, None)
            break
    plain = separable(prob, f, None)["applicable"] or linear(prob, f, None)["applicable"]
    if chosen is None and not plain:
        # Look for an integrating factor that depends on one variable only.
        for M, N in candidates:
            gap = sp.diff(M, y) - sp.diff(N, x)
            kx = sp.simplify(gap / N) if N != 0 else None
            if kx is not None and not kx.has(y):
                mu = sp.simplify(sp.exp(integrate(kx, x)))
                if not has_integral(mu):
                    chosen = (M, N, mu, x)
                    break
            ky = sp.simplify(-gap / M) if M != 0 else None
            if ky is not None and not ky.has(x):
                mu = sp.simplify(sp.exp(integrate(ky, y)))
                if not has_integral(mu):
                    chosen = (M, N, mu, y)
                    break
    if chosen is None:
        M, N = candidates[0]
        return _not_applicable(
            "exact", name,
            "Written as \\(M\\,d%s + N\\,dy = 0\\), exactness needs "
            "\\(\\partial M/\\partial y = \\partial N/\\partial %s\\). Here they are "
            "\\(%s\\) and \\(%s\\)%s."
            % (xn, xn, tex(sp.diff(M, y)), tex(sp.diff(N, x)),
               ", and the simpler methods above already apply" if plain
               else ", and no integrating factor in one variable fixes it"))

    M, N, mu, mu_var = chosen
    steps = [step(
        "Write as M d%s + N dy = 0" % xn,
        "Move everything to one side and read off \\(M\\) and \\(N\\).",
        ["%s\\,d%s + %s\\,dy = 0" % (_paren(M), xn, _paren(N)),
         "M = %s,\\qquad N = %s" % (tex(M), tex(N))])]
    my, nx = sp.simplify(sp.diff(M, y)), sp.simplify(sp.diff(N, x))
    if mu is None:
        reason = ("In the form \\(M\\,d%s + N\\,dy = 0\\) the cross-derivatives "
                  "are equal, so the left side is the total differential of "
                  "some \\(F(%s, y)\\)." % (xn, xn))
        steps.append(step(
            "Test for exactness",
            "The equation is exact when these two partial derivatives agree.",
            ["\\frac{\\partial M}{\\partial y} = %s,\\qquad "
             "\\frac{\\partial N}{\\partial %s} = %s" % (tex(my), xn, tex(nx))]))
    else:
        vn = xn if mu_var == x else "y"
        reason = ("It is not exact as written, but multiplying by the "
                  "integrating factor \\(\\mu(%s) = %s\\) makes it exact."
                  % (vn, tex(mu)))
        ratio = ("\\frac{M_y - N_%s}{N}" % xn) if mu_var == x else ("\\frac{N_%s - M_y}{M}" % xn)
        steps.append(step(
            "Test for exactness",
            "The partial derivatives differ, so the equation is not exact yet. "
            "But \\(%s\\) depends only on \\(%s\\), so an integrating factor "
            "\\(\\mu(%s)\\) exists." % (ratio, vn, vn),
            ["\\frac{\\partial M}{\\partial y} = %s,\\qquad "
             "\\frac{\\partial N}{\\partial %s} = %s" % (tex(my), xn, tex(nx)),
             "\\mu(%s) = %s" % (vn, tex(mu))]))
        M, N = sp.simplify(mu * M), sp.simplify(mu * N)
        steps.append(step(
            "Multiply by the integrating factor",
            "The new \\(M\\) and \\(N\\) now pass the exactness test.",
            ["%s\\,d%s + %s\\,dy = 0" % (_paren(M), xn, _paren(N))]))

    part = integrate(M, x)
    if has_integral(part):
        return _not_applicable("exact", name,
                               "The integral of \\(M\\) has no elementary closed form.")
    dg = sp.simplify(N - sp.diff(part, y))
    if dg.has(x):
        return _not_applicable("exact", name,
                               "The potential function could not be completed.")
    gy = integrate(dg, y)
    if has_integral(gy):
        return _not_applicable("exact", name,
                               "The integral for \\(g(y)\\) has no elementary closed form.")
    F = sp.simplify(with_abs(part) + with_abs(gy))
    steps.append(step(
        "Integrate M with respect to %s" % xn,
        "Find \\(F\\) with \\(\\partial F/\\partial %s = M\\). The \"constant\" "
        "of integration may depend on \\(y\\), so call it \\(g(y)\\)." % xn,
        ["F = \\int %s\\,d%s = %s + g(y)" % (_paren(M), xn, tex(with_abs(part)))]))
    steps.append(step(
        "Find g(y)",
        "Differentiate \\(F\\) with respect to \\(y\\) and set it equal to "
        "\\(N\\). The \\(%s\\) terms cancel, leaving \\(g'(y)\\)." % xn,
        ["g'(y) = %s" % tex(dg), "g(y) = %s" % tex(with_abs(gy))]))
    implicit_tex = "%s = C" % tex(F)
    steps.append(step(
        "Write the implicit solution",
        "Solutions are the level curves \\(F(%s, y) = C\\)." % xn,
        [implicit_tex]))

    branches = []
    raw = [] if hard_to_solve(F, y) else run_with_timeout(
        lambda: sp.solve(sp.Eq(F, C), y), TIMEOUT, [])
    for sol in raw or []:
        if (sol.has(y) or sol.has(sp.I) or has_integral(sol) or too_complex(sol)
                or not _is_real_branch(sol, prob)):
            continue
        branches.append(sol)
    general, general_tex = None, implicit_tex
    if branches:
        text = "This can be solved for \\(y\\)."
        if len(branches) > 1:
            text += " There is more than one branch; an initial condition picks one."
        steps.append(step("Solve for y", text, [_eq("y", b) for b in branches]))
        general = branches[0] if len(branches) == 1 else None
        general_tex = ",\\quad ".join(_eq("y", b) for b in branches)

    particular, particular_tex = None, None
    if ic:
        x0, y0 = ic
        found = _solve_constant(branches, prob, f, x0, y0) if branches else None
        if found:
            eq_tex, value, particular = found
            steps.append(step(
                "Apply the initial condition",
                "Substitute \\(%s = %s\\) and \\(y = %s\\), then solve for \\(C\\)."
                % (xn, tex(x0), tex(y0)),
                [_eq("y", particular)],
                [step("Solve for C", "", [eq_tex, _eq("C", value)])]))
        else:
            value = sp.simplify(F.subs({x: x0, y: y0}))
            particular_tex = "%s = %s" % (tex(F), tex(value))
            steps.append(step(
                "Apply the initial condition",
                "Substitute \\(%s = %s\\) and \\(y = %s\\) into \\(F\\) to find \\(C\\)."
                % (xn, tex(x0), tex(y0)),
                [_eq("C", value), particular_tex]))

    extra = None
    if general is None and particular is None:
        ok = is_zero(-sp.diff(F, x) / sp.diff(F, y) - f)
        extra = (True, "Checked by differentiating the implicit solution.") if ok else None
    return _result("exact", name, reason, steps, general, particular, prob, f, ic,
                   general_tex=general_tex, particular_tex=particular_tex,
                   extra_verify=extra)


# --------------------------------------------------------------------------
# Bernoulli equations
# --------------------------------------------------------------------------

def bernoulli(prob, f, ic):
    name = "Bernoulli substitution"
    x, y, C = prob.x, prob.y, prob.C
    xn = prob.indep_name
    form = "\\(y' + p(%s)\\,y = q(%s)\\,y^n\\)" % (xn, xn)

    try:
        groups = sp.collect(sp.expand(f), y, evaluate=False)
    except Exception:
        groups = {}
    power = None
    p = sp.Integer(0)
    q = None
    ok = bool(groups)
    for key, coef in groups.items():
        if coef.has(y):
            ok = False
        elif key == y:
            p = -coef
        elif key.is_Pow and key.base == y and key.exp.is_number and not key.exp.has(y) and q is None:
            power, q = key.exp, coef
        else:
            ok = False
    if not ok or q is None or power in (0, 1):
        return _not_applicable(
            "bernoulli", name,
            "The equation does not have the form %s with \\(n \\ne 0, 1\\)." % form)
    p, q, n = sp.simplify(p), sp.simplify(q), power

    k = 1 - n
    steps = [step(
        "Write in Bernoulli form",
        "Collect the terms so the equation reads %s." % form,
        ["%s %s = %s\\,y^{%s}" % (dydx(prob), _signed_term(p), _paren(q), tex(n)),
         "p = %s,\\qquad q = %s,\\qquad n = %s" % (tex(p), tex(q), tex(n))])]
    steps.append(step(
        "Substitute v = y^(1-n)",
        "Let \\(v = y^{%s}\\). Dividing the equation by \\(y^{%s}\\) and using "
        "\\(v' = %s\\,y^{%s}\\,y'\\) turns it into a linear equation in \\(v\\)."
        % (tex(k), tex(n), tex(k), tex(-n)),
        ["v' + %s\\,v = %s" % (_paren(sp.simplify(k * p)), tex(sp.simplify(k * q)))]))

    P = integrate(sp.simplify(k * p), x)
    if has_integral(P):
        return _not_applicable("bernoulli", name,
                               "The integrating factor for the linear equation in "
                               "\\(v\\) has no elementary closed form.")
    mu = sp.powsimp(sp.simplify(sp.exp(P)))
    R = integrate(sp.simplify(mu * k * q), x)
    if has_integral(R):
        return _not_applicable("bernoulli", name,
                               "The integral in the linear equation for \\(v\\) "
                               "has no elementary closed form.")
    R = with_abs(R)
    v = _tidy((R + C) / mu, C)
    steps.append(step(
        "Solve the linear equation",
        "Use the integrating factor \\(\\mu = e^{\\int (1-n)p\\,d%s}\\), as in "
        "the integrating-factor method." % xn,
        [_eq("v", v)],
        [step("Integrating factor", "", ["\\mu = %s" % tex(mu)]),
         step("Integrate", "", ["%s\\,v = %s" % (_paren(mu), _plus_c(R))])]))

    raw = run_with_timeout(lambda: sp.solve(sp.Eq(y**k, v), y), TIMEOUT, [])
    branches = [b for b in (raw or [])
                if not b.has(y) and not b.has(sp.I) and _is_real_branch(b, prob)]
    if not branches:
        branches = [v ** (1 / k)]
    text = "Undo the substitution: \\(y^{%s} = v\\)." % tex(k)
    if n.is_positive:
        text += " Dividing by a power of \\(y\\) also lost the solution \\(y = 0\\)."
    if len(branches) > 1:
        text += " There is more than one branch; an initial condition picks one."
    steps.append(step("Return to y", text, [_eq("y", b) for b in branches]))
    general = branches[0] if len(branches) == 1 else None
    general_tex = ",\\quad ".join(_eq("y", b) for b in branches)

    particular = None
    if ic:
        x0, y0 = ic
        found = _solve_constant(branches, prob, f, x0, y0)
        if found:
            eq_tex, value, particular = found
            steps.append(step(
                "Apply the initial condition",
                "Substitute \\(%s = %s\\) and \\(y = %s\\), then solve for \\(C\\)."
                % (xn, tex(x0), tex(y0)),
                [_eq("y", particular)],
                [step("Solve for C", "", [eq_tex, _eq("C", value)])]))
        elif y0 == 0 and n.is_positive:
            particular = sp.Integer(0)
            steps.append(step("Apply the initial condition",
                              "The initial value is 0, which is the constant "
                              "solution \\(y = 0\\).", [_eq("y", 0)]))
        else:
            steps.append(step("Apply the initial condition",
                              "No value of \\(C\\) fits this initial condition "
                              "on the branches found."))
    out = _result("bernoulli", name,
                  "It has the form %s with \\(n = %s\\)." % (form, tex(n)),
                  steps, general, particular, prob, f, ic, general_tex=general_tex)
    if general is None and particular is None and len(branches) > 1:
        if all(check_solution(prob, f, b)[0] is not False for b in branches):
            out["verified"] = True
            out["verify_note"] = "Each branch was checked by substituting back into the equation."
    return out


# --------------------------------------------------------------------------
# Homogeneous equations (y = v x)
# --------------------------------------------------------------------------

def homogeneous(prob, f, ic):
    name = "Homogeneous substitution (y = vx)"
    x, y, C = prob.x, prob.y, prob.C
    xn = prob.indep_name
    v = sp.Symbol("v", real=True)

    xpos = sp.Symbol("xpos", positive=True)
    try:
        Fv = sp.simplify(f.subs(y, v * x).subs(x, xpos))
    except Exception:
        Fv = None
    if Fv is None or Fv.has(xpos):
        return _not_applicable(
            "homogeneous", name,
            "The right-hand side cannot be written as a function of "
            "\\(y/%s\\) alone." % xn)
    phi = sp.simplify(Fv - v)
    if phi == 0 or not f.has(x):
        return _not_applicable(
            "homogeneous", name,
            "The substitution gives nothing new here; the simpler methods apply.")

    if ic and ic[0] == 0:
        return _not_applicable(
            "homogeneous", name,
            "The substitution \\(y = v%s\\) divides by \\(%s\\), so it cannot "
            "use an initial condition at \\(%s = 0\\)." % (xn, xn, xn))

    steps = [step(
        "Recognise the form",
        "The right-hand side depends only on the ratio \\(y/%s\\)." % xn,
        ["%s = F\\!\\left(\\frac{y}{%s}\\right),\\qquad F(v) = %s"
         % (dydx(prob), xn, tex(Fv))])]
    steps.append(step(
        "Substitute y = v%s" % xn,
        "With \\(y = v%s\\), the product rule gives \\(y' = v + %s\\,v'\\). "
        "The equation becomes separable in \\(v\\) and \\(%s\\)." % (xn, xn, xn),
        ["v + %s\\,\\frac{dv}{d%s} = %s" % (xn, xn, tex(Fv)),
         "%s\\,\\frac{dv}{d%s} = %s" % (xn, xn, tex(phi))]))
    lines = []
    try:
        lines = [r for r in sp.solve(phi, v) if r.is_real is not False and not r.free_symbols]
    except Exception:
        lines = []
    if lines:
        steps.append(step(
            "Note the straight-line solutions",
            "Where the right side is zero, \\(v\\) is constant, so these lines "
            "through the origin are solutions on their own:",
            [",\\quad ".join(_eq("y", r * x) for r in lines)]))
    inv = sp.simplify(1 / phi)
    H = integrate(inv, v)
    if has_integral(H):
        return _not_applicable("homogeneous", name,
                               "The integral in \\(v\\) has no elementary closed form.")
    H_show = with_abs(H)
    steps.append(step(
        "Separate and integrate",
        "Put the \\(v\\) terms on the left and the \\(%s\\) terms on the right, "
        "then integrate." % xn,
        ["%s\\,dv = \\frac{1}{%s}\\,d%s" % (_paren(inv), xn, xn),
         "%s = \\ln\\left|%s\\right| + C" % (tex(H_show), xn)]))
    back = H_show.subs(v, y / x)
    implicit_tex = "%s = \\ln\\left|%s\\right| + C" % (tex(back), xn)
    steps.append(step(
        "Substitute back v = y/%s" % xn, "", [implicit_tex]))

    branches, renamed = [], False
    raw = [] if hard_to_solve(H, v) else run_with_timeout(
        lambda: sp.solve(sp.Eq(H, sp.log(x) + C), v), TIMEOUT, [])
    for sol in raw or []:
        if sol.has(v) or sol.has(sp.I) or has_integral(sol):
            continue
        cand = sp.simplify(x * sol)
        try:
            simp = _absorb_exp(constantsimp(cand, {C}), C)
        except Exception:
            simp = cand
        # A constant still trapped inside an exponential reads badly; the
        # implicit form is clearer than such a formula.
        trapped = any(e.has(C) for e in simp.atoms(sp.exp))
        if trapped or too_complex(simp) or not _is_real_branch(simp, prob):
            continue
        if simp != cand:
            renamed = True
        branches.append(_prefer_abs(simp, prob, f))
    general, general_tex = None, implicit_tex
    if branches:
        text = "Solve for \\(y\\)."
        if renamed:
            text += (" An expression made only of the arbitrary constant is "
                     "itself an arbitrary constant, so it is renamed \\(C\\).")
        if len(branches) > 1:
            text += " There is more than one branch; an initial condition picks one."
        steps.append(step("Solve for y", text, [_eq("y", b) for b in branches]))
        general = branches[0] if len(branches) == 1 else None
        general_tex = ",\\quad ".join(_eq("y", b) for b in branches)

    particular, particular_tex = None, None
    if ic:
        x0, y0 = ic
        if x0 != 0 and any(is_zero(r * x0 - y0) for r in lines):
            line = [r for r in lines if is_zero(r * x0 - y0)][0] * x
            particular = line
            steps.append(step("Apply the initial condition",
                              "The initial point lies on one of the straight-line "
                              "solutions.", [_eq("y", line)]))
        else:
            found = _solve_constant(branches, prob, f, x0, y0) if branches else None
            if found:
                eq_tex, value, particular = found
                steps.append(step(
                    "Apply the initial condition",
                    "Substitute \\(%s = %s\\) and \\(y = %s\\), then solve for \\(C\\)."
                    % (xn, tex(x0), tex(y0)),
                    [_eq("y", particular)],
                    [step("Solve for C", "", [eq_tex, _eq("C", value)])]))
            else:
                try:
                    value = sp.simplify(back.subs({x: x0, y: y0}) - sp.log(sp.Abs(x0)))
                    usable = value.is_real is not False and not value.has(sp.zoo, sp.nan, sp.oo)
                except Exception:
                    usable = False
                if usable:
                    particular_tex = "%s = \\ln\\left|%s\\right| %s" % (tex(back), xn, signed(value))
                    steps.append(step(
                        "Apply the initial condition",
                        "Substitute the initial point into the implicit solution to find \\(C\\).",
                        [_eq("C", value), particular_tex]))
                else:
                    steps.append(step("Apply the initial condition",
                                      "No value of \\(C\\) fits this initial condition."))
    extra = None
    if general is None and particular is None:
        # Differentiate the implicit relation and compare the slope with f.
        relation = H.subs(v, y / x) - sp.log(x)
        slope = -sp.diff(relation, x) / sp.diff(relation, y)
        if residual_status(slope - f) in ("symbolic", "numeric"):
            extra = (True, "Checked by differentiating the implicit solution.")
        else:
            extra = (None, "The implicit solution was not checked automatically.")
    out = _result("homogeneous", name,
                  "The right-hand side is a function of \\(y/%s\\) only." % xn,
                  steps, general, particular, prob, f, ic,
                  general_tex=general_tex, particular_tex=particular_tex,
                  extra_verify=extra)
    if general is None and particular is None and len(branches) > 1:
        if all(check_solution(prob, f, b)[0] is not False for b in branches):
            out["verified"] = True
            out["verify_note"] = "Each branch was checked by substituting back into the equation."
    return out


# --------------------------------------------------------------------------
# Laplace transform (linear, constant coefficient, condition at 0)
# --------------------------------------------------------------------------

def laplace(prob, f, ic):
    from . import laplace as lt
    x, y = prob.x, prob.y
    a = sp.simplify(sp.diff(f, y))
    q = sp.simplify(f - a * y)
    if a.has(y) or q.has(y):
        return _not_applicable("laplace", lt.NAME, "The method needs a linear equation.")
    blocked = lt.why_not(prob, not a.has(x), ic)
    if blocked:
        return _not_applicable("laplace", lt.NAME, blocked)
    steps, sol = lt.solve_ivp(prob, sp.Integer(0), sp.Integer(1), -a, q, ic[1])
    if steps is None:
        return _not_applicable("laplace", lt.NAME, sol)
    steps.insert(0, step(
        "Write in standard form", "Move the \\(y\\) term to the left.",
        ["y' %s = %s" % (_signed_term(-a), tex(q))]))
    return _result("laplace", lt.NAME,
                   "The equation is linear with constant coefficients and the "
                   "initial condition is given at 0.",
                   steps, None, sol, prob, f, ic)


METHODS = (separable, linear, exact, bernoulli, homogeneous, laplace)

_OTHER_TYPES = (
    ("Riccati", "Riccati"),
    ("separable_reduced", "reducible to separable by substitution"),
    ("linear_coefficients", "linear coefficients (shift substitution)"),
)


def other_types(prob, f):
    """Names of other standard types SymPy recognises, for 'coming later' hints."""
    try:
        func = sp.Function("y")(prob.x)
        ode = sp.Eq(func.diff(prob.x), f.subs(prob.y, func))
        hints = sp.classify_ode(ode, func)
    except Exception:
        return []
    found = []
    for key, label in _OTHER_TYPES:
        if any(h.startswith(key) for h in hints) and label not in found:
            found.append(label)
    return found
