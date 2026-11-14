"""Linear difference equations (recurrences) with constant coefficients.

Typed as, for example, a(n+2) = a(n+1) + a(n). Solved the way the matching
differential equations are: a characteristic equation for the homogeneous
part and undetermined coefficients for a right-hand side made of
polynomials in n and exponentials b^n.
"""
import re

import sympy as sp

from .parsing import InputError, parse_expression, parse_number
from .toolkit import method, not_applicable, result, step
from .util import tex

OPERATIONS = ("solve",)
_SLOTS = "pqrs"          # stand-ins for a(n), a(n+1), a(n+2), a(n+3) while parsing
_TERM = re.compile(r"a\s*\(\s*n\s*(?:([+-])\s*(\d+)\s*)?\)")
_COEFFS = ["A", "B", "D", "F", "G", "H", "K", "L"]


def _a(shift):
    return sp.Symbol("a_{n}" if shift == 0 else "a_{n+%d}" % shift)


def parse(text):
    """Returns (coeffs, g, n): sum(coeffs[j] * a(n+j)) = g(n), with coeffs[0] != 0."""
    if text.count("=") != 1:
        raise InputError("Write the recurrence with one = sign, for example "
                         "a(n+2) = a(n+1) + a(n).")
    shifts = [int((m.group(1) or "+") + (m.group(2) or "0")) for m in _TERM.finditer(text)]
    if not shifts:
        raise InputError("Write the terms of the sequence as a(n), a(n+1), a(n+2): "
                         "for example a(n+2) = a(n+1) + a(n).")
    low = min(shifts)
    order = max(shifts) - low
    if order == 0:
        raise InputError("The recurrence must relate at least two different terms, "
                         "such as a(n+1) and a(n).")
    if order > 3:
        raise InputError("Recurrences up to third order are supported "
                         "(a(n) through a(n+3)).")

    def slot(match):
        shift = int((match.group(1) or "+") + (match.group(2) or "0")) - low
        return " %s " % _SLOTS[shift]

    replaced = _TERM.sub(slot, text)
    if re.search(r"(?<![a-z])a(?![a-z])", replaced):
        raise InputError("Write each term of the sequence with its index in "
                         "brackets, such as a(n+1).")
    left, right = replaced.split("=")
    names = ["n"] + list(_SLOTS)
    lhs, symbols = parse_expression(left, names)
    rhs, _ = parse_expression(right, names)
    n = symbols["n"]
    slots = [symbols[c] for c in _SLOTS]
    expr = sp.expand(lhs - rhs)
    # The right-hand side g(n) was typed relative to the lowest index; shift it
    # so the equation is in terms of a(n), ..., a(n+order).
    expr = expr.subs(n, n - low) if low else expr
    try:
        poly = sp.Poly(expr, *slots)
    except sp.PolynomialError:
        poly = None
    if poly is None or poly.total_degree() != 1:
        raise InputError("Only linear recurrences are supported: each a(...) may "
                         "appear to the first power, not multiplied by another.")
    coeffs = [sp.simplify(poly.coeff_monomial(s)) for s in slots[:order + 1]]
    g = sp.simplify(-poly.coeff_monomial(1))
    if any(c.has(n) for c in coeffs):
        raise InputError("The coefficients of the a(...) terms must be constants, "
                         "not expressions in n.")
    if coeffs[0] == 0 or coeffs[-1] == 0:
        raise InputError("I couldn't read the order of that recurrence.")
    if g.has(*slots):
        raise InputError("Only linear recurrences are supported.")
    return coeffs, g, n


def _equation_tex(coeffs, g):
    total = sum(c * _a(j) for j, c in enumerate(coeffs))
    ordered = sorted(sp.Add.make_args(total), key=lambda t: -max(
        j for j in range(len(coeffs)) if t.has(_a(j))))
    out = tex(ordered[0])
    for term in ordered[1:]:
        out += (" - " + tex(-term)) if term.could_extract_minus_sign() else (" + " + tex(term))
    return "%s = %s" % (out, tex(g))


def _basis(roots, n):
    """Real solutions of the homogeneous recurrence, one per root counted with multiplicity."""
    out, used = [], set()
    for root in sorted(roots, key=sp.default_sort_key):
        if root in used:
            continue
        mult = roots[root]
        if root.is_real:
            used.add(root)
            for k in range(mult):
                out.append(n**k * root**n)
        else:
            conj = sp.conjugate(root)
            used.update((root, conj))
            rho, theta = sp.simplify(sp.Abs(root)), sp.simplify(sp.Abs(sp.arg(root)))
            for k in range(mult):
                out.append(n**k * rho**n * sp.cos(n * theta))
                out.append(n**k * rho**n * sp.sin(n * theta))
    return out


def _forcing_groups(g, n):
    """Split g into {base: polynomial in n} for terms poly(n) * base^n, or None."""
    groups = {}
    for term in sp.Add.make_args(sp.expand(sp.powsimp(sp.expand(g)))):
        base = sp.Integer(1)
        poly = sp.Integer(1)
        for factor in sp.Mul.make_args(term):
            if not factor.has(n):
                poly *= factor
            elif factor == n or (factor.is_Pow and factor.base == n and factor.exp.is_Integer
                                 and factor.exp > 0):
                poly *= factor
            elif factor.is_Pow and not factor.base.has(n):
                # b^(k n + m) = (b^k)^n * b^m
                exponent = sp.expand(factor.exp)
                k = exponent.coeff(n)
                rest = exponent - k * n
                if rest.has(n) or k == 0:
                    return None
                base *= factor.base**k
                poly *= factor.base**rest
            else:
                return None
        base = sp.simplify(base)
        groups[base] = groups.get(base, 0) + poly
    return groups


def _apply(coeffs, seq, n):
    """sum(c_j * seq(n + j))."""
    return sum(c * seq.subs(n, n + j) for j, c in enumerate(coeffs))


def _iterate(coeffs, g, n, start, count):
    """The first `count` terms computed directly from the recurrence."""
    values = list(start)
    order = len(coeffs) - 1
    while len(values) < count:
        m = len(values) - order
        acc = g.subs(n, m) - sum(coeffs[j] * values[m + j] for j in range(order))
        values.append(sp.simplify(acc / coeffs[order]))
    return values[:count]


def _same(a, b):
    try:
        if sp.simplify(a - b) == 0:
            return True
        scale = 1 + abs(complex(sp.N(b)))
        return abs(complex(sp.N(a - b, 30))) < 1e-9 * scale
    except Exception:
        return False


def _initial(text, order):
    parts = [p for p in re.split(r"[,;]", text or "") if p.strip()]
    if not parts:
        return None
    if len(parts) != order:
        names = ", ".join("a(%d)" % k for k in range(order))
        raise InputError("This recurrence needs %d starting value%s (%s), separated "
                         "by commas." % (order, "" if order == 1 else "s", names))
    return [parse_number(p, "a(%d)" % k) for k, p in enumerate(parts)]


def _num(value):
    f = float(sp.N(value))
    return ("%.6g" % f) if abs(f - round(f)) > 1e-12 or abs(f) > 1e9 else str(int(round(f)))


def _solve(text, start_text):
    coeffs, g, n = parse(text)
    order = len(coeffs) - 1
    start = _initial(start_text, order)
    equation = _equation_tex(coeffs, g)
    r = sp.Symbol("r")
    char = sum(c * r**j for j, c in enumerate(coeffs))
    # A cubic that does not factor has roots only through Cardano's formula,
    # which is useless to read and very slow to work with.
    reducible = all(sp.Poly(f, r).degree() <= 2 for f, _ in sp.factor_list(char, r)[1])
    roots = sp.roots(sp.Poly(char, r)) if reducible else {}
    name = "Characteristic equation"
    if sum(roots.values()) != order:
        return result("recurrence", "Difference equation", equation,
                      "The characteristic equation of this recurrence has no exact roots.",
                      [not_applicable("characteristic", name,
                                      "The characteristic equation \\(%s = 0\\) cannot be solved "
                                      "exactly, so no closed formula is given." % tex(char))]
                      + ([_iteration(coeffs, g, n, start, equation)] if start else []))

    steps = [step(
        "Write the characteristic equation",
        "Try \\(a_n = r^n\\) in the homogeneous recurrence (right-hand side 0) and divide by "
        "\\(r^n\\). This is the same idea as trying \\(e^{rx}\\) in a differential equation "
        "with constant coefficients.",
        ["%s = 0" % tex(sp.expand(char))])]
    basis = _basis(roots, n)
    listed = ",\\quad ".join(
        "r = %s%s" % (tex(root), "" if mult == 1 else "\\ (\\text{multiplicity } %d)" % mult)
        for root, mult in sorted(roots.items(), key=lambda p: sp.default_sort_key(p[0])))
    notes = []
    if any(m > 1 for m in roots.values()):
        notes.append("A repeated root contributes extra solutions multiplied by powers of \\(n\\).")
    if any(not root.is_real for root in roots):
        notes.append("A complex pair \\(\\rho e^{\\pm i\\theta}\\) contributes "
                     "\\(\\rho^n\\cos n\\theta\\) and \\(\\rho^n\\sin n\\theta\\).")
    steps.append(step("Solve for the roots", " ".join(notes) or
                      "Each root gives a solution \\(r^n\\).", [listed]))
    constants = [sp.Symbol("C_%d" % (k + 1)) for k in range(order)]
    homogeneous = sum(c * b for c, b in zip(constants, basis))
    steps.append(step("Homogeneous solution",
                      "Combine the solutions with arbitrary constants.",
                      ["a_n^{(h)} = %s" % tex(homogeneous)]))

    particular = sp.Integer(0)
    if g != 0:
        groups = _forcing_groups(g, n)
        if groups is None:
            return result("recurrence", "Difference equation", equation,
                          "The right-hand side is outside the forms handled here.",
                          [not_applicable("characteristic", name,
                                          "The right-hand side \\(%s\\) is not a combination of "
                                          "polynomials in \\(n\\) and exponentials \\(b^n\\), so "
                                          "there is no finite guess for a particular solution."
                                          % tex(g))]
                          + ([_iteration(coeffs, g, n, start, equation)] if start else []))
        unknowns, guess, resonant = [], sp.Integer(0), False
        for base, poly in groups.items():
            degree = sp.Poly(sp.expand(poly), n).degree()
            mult = 0
            for root, count in roots.items():
                if sp.simplify(root - base) == 0:
                    mult = count
            resonant = resonant or mult > 0
            for k in range(degree + 1):
                if len(unknowns) >= len(_COEFFS):
                    raise InputError("The right-hand side needs too many coefficients.")
                sym = sp.Symbol(_COEFFS[len(unknowns)])
                unknowns.append(sym)
                guess += sym * n**(k + mult) * base**n
        text_ = ("Guess a combination with the same form as the right-hand side and "
                 "unknown coefficients.")
        if resonant:
            text_ += (" Part of the natural guess already solves the homogeneous "
                      "recurrence, so that part is multiplied by \\(n\\) (once for each "
                      "time its base is a root).")
        steps.append(step("Guess a particular solution", text_, ["a_n^{(p)} = %s" % tex(guess)]))
        equations = []
        for base, poly in groups.items():
            part = sum(t for t in sp.Add.make_args(sp.expand(guess))
                       if sp.simplify(sp.powsimp(t / base**n)).is_polynomial(n))
            residual = sp.expand(sp.simplify(sp.powsimp(
                sp.expand(_apply(coeffs, part, n)) / base**n)) - poly)
            equations += [sp.Eq(c, 0) for c in sp.Poly(residual, n).all_coeffs() if c != 0]
        solved = sp.solve(equations, unknowns, dict=True)
        if not solved:
            raise InputError("The coefficients of the particular solution could not be matched.")
        values = {u: solved[0].get(u, sp.Integer(0)) for u in unknowns}
        particular = sp.expand(guess.subs(values))
        steps.append(step(
            "Substitute and match coefficients",
            "Put the guess into the recurrence and compare the two sides power by power.",
            [",\\quad ".join("%s = %s" % (tex(u), tex(values[u])) for u in unknowns),
             "a_n^{(p)} = %s" % tex(particular)]))

    general = homogeneous + particular
    steps.append(step("Write the general solution",
                      "Add the homogeneous and particular solutions." if g != 0
                      else "With right-hand side 0 this is the whole solution.",
                      ["a_n = %s" % tex(general)]))
    answer, formula = "a_n = %s" % tex(general), None
    plot, table = None, None
    verified, note = None, ""
    if start:
        eqs = [sp.Eq(general.subs(n, k), start[k]) for k in range(order)]
        found = sp.solve(eqs, constants, dict=True)
        if not found:
            raise InputError("The starting values could not be fitted.")
        formula = sp.simplify(general.subs(found[0]))
        steps.append(step(
            "Apply the starting values",
            "Set \\(n = %s\\) and solve for the constants."
            % ", ".join(str(k) for k in range(order)),
            ["a_n = %s" % tex(formula)],
            [step("Equations for the constants", "",
                  ["%s = %s" % (tex(e.lhs), tex(e.rhs)) for e in eqs]
                  + [",\\quad ".join("%s = %s" % (tex(c), tex(found[0].get(c, c)))
                                     for c in constants)])]))
        direct = _iterate(coeffs, g, n, start, 30)
        ok = all(_same(formula.subs(n, k), direct[k]) for k in range(30))
        verified = bool(ok)
        note = ("The formula reproduces the first 30 terms computed directly from the "
                "recurrence." if ok else
                "The formula does not reproduce the terms computed from the recurrence.")
        table = step("First terms", "From the formula; they agree with stepping the "
                     "recurrence forward.")
        table["table"] = {"headers": ["n", "a_n"],
                          "rows": [[str(k), _num(direct[k])] for k in range(11)]}
        steps.append(table)
        answer = "a_n = %s" % tex(formula)
        shown = [k for k in range(21) if abs(float(sp.N(direct[k]))) < 1e6]
        plot = {"points": [[float(k), float(sp.N(direct[k]))] for k in shown]}
    else:
        check = sp.simplify(_apply(coeffs, general, n) - g)
        ok = check == 0 or all(abs(complex(sp.N(check.subs(
            {n: k, **{c: 1 + i for i, c in enumerate(constants)}})))) < 1e-9 for k in range(6))
        verified = bool(ok)
        note = ("Checked by substituting the general solution back into the recurrence."
                if ok else "Substituting back did not reproduce the recurrence.")

    size = max(abs(complex(sp.N(root))) for root in roots)
    if g == 0:
        if size < 1 - 1e-12:
            behaviour = "Every root has \\(|r| < 1\\), so every solution tends to 0."
        elif size > 1 + 1e-12:
            behaviour = ("A root has \\(|r| > 1\\), so solutions generally grow without "
                         "bound, like \\(|r|^n\\) with \\(|r| = %s\\)." % _num(size))
        else:
            behaviour = ("The largest root has \\(|r| = 1\\), so solutions neither decay "
                         "nor grow geometrically.")
    else:
        behaviour = ("The homogeneous part %s; the particular solution then sets the "
                     "long-run behaviour." % ("dies out, since every root has \\(|r| < 1\\)"
                                               if size < 1 - 1e-12 else "does not die out"))
    steps.append(step("Long-run behaviour", behaviour))

    methods = [method("characteristic", name,
                      "The recurrence is linear with constant coefficients, so it is solved "
                      "like a constant-coefficient differential equation with \\(r^n\\) in "
                      "place of \\(e^{rx}\\).", steps, answer, plot=plot,
                      verified=verified, note=note)]
    if start:
        methods.append(_iteration(coeffs, g, n, start, equation))
    summary = "Linear recurrence of order %d with constant coefficients." % order
    return result("recurrence", "Difference equation", equation, summary, methods,
                  legends={"points": "Terms of the sequence"})


def _iteration(coeffs, g, n, start, equation):
    order = len(coeffs) - 1
    values = _iterate(coeffs, g, n, start, 21)
    lead = sp.solve(sum(c * _a(j) for j, c in enumerate(coeffs)) - g, _a(order))[0]
    steps = [step("Solve for the newest term",
                  "Rearrange so the term with the highest index is alone.",
                  ["%s = %s" % (tex(_a(order)), tex(lead))]),
             step("Step forward from the starting values",
                  "Each new term uses the %s before it." %
                  ("one" if order == 1 else "%d terms" % order),
                  ["a_{%d} = %s" % (k, _num(values[k])) for k in range(order, order + 3)])]
    table = step("First terms", "")
    table["table"] = {"headers": ["n", "a_n"],
                      "rows": [[str(k), _num(values[k])] for k in range(16)]}
    steps.append(table)
    shown = [k for k in range(21) if abs(float(sp.N(values[k]))) < 1e6]
    return method("iteration", "Direct iteration",
                  "Starting values are given, so the terms can be computed one after another. "
                  "This gives numbers but no formula for \\(a_n\\).",
                  steps, "a_{15} = %s" % _num(values[15]),
                  plot={"points": [[float(k), float(sp.N(values[k]))] for k in shown]})


def solve(operation, text, start_text="", unused=""):
    return _solve(text, start_text)


def preview(operation, text):
    coeffs, g, _ = parse(text)
    return _equation_tex(coeffs, g)
