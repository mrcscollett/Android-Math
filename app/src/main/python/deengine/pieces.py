"""Integrating functions that contain unit steps u(x - a) and absolute values.

SymPy's own handling of Heaviside and Abs inside improper integrals is
unreliable, so the interval is cut at every point where a step switches or an
absolute value changes sign, and each smooth piece is integrated separately.
"""
import sympy as sp

from .parsing import InputError


def breakpoints(f, x):
    """Sorted points where a Heaviside or Abs argument (linear in x) is zero."""
    found = set()
    for atom in f.atoms(sp.Heaviside, sp.Abs, sp.sign):
        arg = atom.args[0]
        if not arg.has(x):
            continue
        try:
            poly = sp.Poly(arg, x)
        except sp.PolynomialError:
            poly = None
        if poly is None or poly.degree() != 1:
            raise InputError("Inside u(...) and abs(...) use a first-degree expression "
                             "such as x - 2 or 3 - x.")
        found.add(sp.simplify(-poly.nth(0) / poly.nth(1)))
    return sorted(found, key=lambda v: float(v))


def split(f, x, lo=-sp.oo, hi=sp.oo):
    """[(a, b, expr)] with f rewritten as an ordinary expression on each (a, b)."""
    cuts = [c for c in breakpoints(f, x) if (lo == -sp.oo or c > lo) and (hi == sp.oo or c < hi)]
    edges = [lo] + cuts + [hi]
    out = []
    for a, b in zip(edges, edges[1:]):
        if a == -sp.oo and b == sp.oo:
            mid = sp.Integer(0)
        elif a == -sp.oo:
            mid = b - 1
        elif b == sp.oo:
            mid = a + 1
        else:
            mid = (a + b) / 2

        def resolve(atom):
            value = atom.args[0].subs(x, mid)
            positive = bool(value > 0)
            if isinstance(atom, sp.Heaviside):
                return sp.Integer(1) if positive else sp.Integer(0)
            if isinstance(atom, sp.sign):
                return sp.Integer(1) if positive else sp.Integer(-1)
            return atom.args[0] if positive else -atom.args[0]

        expr = f.replace(lambda e: isinstance(e, (sp.Heaviside, sp.Abs, sp.sign)) and e.args[0].has(x),
                         resolve)
        expr = sp.simplify(expr)
        if expr != 0:
            out.append((a, b, expr))
    return out


def numeric(f, x):
    """A float function of one variable for f, with steps and absolute values."""
    return sp.lambdify(x, f, [{"Heaviside": lambda v, *rest: 0.0 if v < 0 else 1.0,
                               "DiracDelta": lambda v, *rest: 0.0}, "mpmath"])
