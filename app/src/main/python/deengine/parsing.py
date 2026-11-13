"""Turn what the user typed into SymPy expressions.

The text is checked against a whitelist before it reaches SymPy's parser
(which uses eval), so only arithmetic, known function names and the
problem's variables get through.
"""
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    rationalize,
    standard_transformations,
)


class InputError(Exception):
    """A problem with the input that should be shown to the user as-is."""


_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    rationalize,
)

_FUNCTIONS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sec": sp.sec, "csc": sp.csc, "cot": sp.cot,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "ln": sp.log, "log": sp.log,
    "sqrt": sp.sqrt, "abs": sp.Abs,
}
_CONSTANTS = {"pi": sp.pi}

_UNICODE = {
    "′": "'", "’": "'", "‘": "'", "`": "'",
    "×": "*", "·": "*", "⋅": "*", "−": "-", "–": "-",
    "÷": "/", "π": "pi", "√": "sqrt", "²": "^2", "³": "^3",
}

_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-z+\-*/^().,=\s]*$")
_PRIME = "D"  # single-letter stand-in for y' while parsing


class Problem:
    """Symbols for one problem: independent variable, y and y'."""

    def __init__(self, indep="x"):
        self.indep_name = indep
        self.x = sp.Symbol(indep, real=True)
        self.y = sp.Symbol("y", real=True)
        self.yp = sp.Symbol("y'", real=True)
        self.C = sp.Symbol("C", real=True)


def _normalise_text(text):
    for bad, good in _UNICODE.items():
        text = text.replace(bad, good)
    return text.strip()


def detect_indep(text):
    """Use t when the user clearly wrote the equation in t, otherwise x."""
    text = _normalise_text(text)
    if re.search(r"dy\s*/\s*dt", text):
        return "t"
    stripped = text
    for name in sorted(list(_FUNCTIONS) + list(_CONSTANTS), key=len, reverse=True):
        stripped = re.sub(r"(?<![A-Za-z])%s(?![A-Za-z])" % name, " ", stripped)
    letters = set(re.findall(r"[A-Za-z]", stripped))
    if "t" in letters and "x" not in letters:
        return "t"
    return "x"


def _check_identifiers(text, variables):
    for match in re.finditer(r"[A-Za-z_]+", text):
        word = match.group(0)
        if word in _FUNCTIONS or word in _CONSTANTS:
            continue
        if all(ch in variables for ch in word):
            continue
        unknown = [ch for ch in word if ch not in variables]
        if len(word) > 1 and any(word.startswith(v) or word.endswith(v) for v in variables):
            raise InputError(
                "I couldn't read '%s'. Put a * or a space between variables and "
                "function names, for example x*sin(x)." % word.replace(_PRIME, "y'")
            )
        raise InputError(
            "Unknown symbol '%s'. This problem can use %s, y, y', numbers, pi, e "
            "and functions such as sin, cos, exp, ln, sqrt."
            % ("".join(unknown) if len(word) == 1 else word.replace(_PRIME, "y'"),
               (sorted(variables - {"y", _PRIME, "e"}) or ["x"])[0])
        )


def _parse_side(text, prob, allow_vars=True):
    if not text.strip():
        raise InputError("One side of the equation is empty.")
    variables = {prob.indep_name, "y", _PRIME, "e"} if allow_vars else {"e"}
    _check_identifiers(text, variables)
    if re.search(r"(?<![0-9])\.(?![0-9])", text):
        raise InputError("A decimal point needs a digit next to it.")
    try:
        expr = parse_expr(text, transformations=_TRANSFORMS,
                          local_dict=dict(_FUNCTIONS), evaluate=True)
    except Exception:
        raise InputError("I couldn't read that expression. Check for unbalanced "
                         "parentheses or a missing operator.")
    if not isinstance(expr, sp.Expr):
        raise InputError("I couldn't read that expression.")
    swaps = {
        sp.Symbol(prob.indep_name): prob.x,
        sp.Symbol("y"): prob.y,
        sp.Symbol(_PRIME): prob.yp,
        sp.Symbol("e"): sp.E,
    }
    expr = expr.xreplace(swaps)
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise InputError("That expression divides by zero or is undefined.")
    return expr


def parse_equation(text, indep=None):
    """Parse a first-order ODE. Returns (Problem, lhs, rhs)."""
    text = _normalise_text(text)
    if not text:
        raise InputError("Enter an equation, for example y' = x*y.")
    if not _ALLOWED_CHARS.match(text.replace("'", "")):
        bad = sorted(set(ch for ch in text if not _ALLOWED_CHARS.match(ch) and ch != "'"))
        raise InputError("Unexpected character: %s" % " ".join(bad))
    indep = indep or detect_indep(text)
    prob = Problem(indep)

    text = re.sub(r"d\s*\^?\s*2\s*y\s*/\s*d\s*%s\s*\^?\s*2" % indep, "y''", text)
    if "''" in text:
        raise InputError("That is a second-order equation (y''). This section "
                         "handles first-order equations; higher order arrives "
                         "in the next phase.")
    text = re.sub(r"dy\s*/\s*d%s" % indep, " y' ", text)
    text = re.sub(r"y\s*'\s*\(\s*%s\s*\)" % indep, " y' ", text)
    text = re.sub(r"y\s*\(\s*%s\s*\)" % indep, " y ", text)
    if _PRIME in text:
        raise InputError("Unknown symbol 'D'.")
    text = re.sub(r"y\s*'", " %s " % _PRIME, text)
    if "'" in text:
        raise InputError("Only y can carry a prime (y').")

    if text.count("=") != 1:
        raise InputError("Write the equation with a single = sign, "
                         "for example y' + 2y = x.")
    left, right = text.split("=")
    lhs = _parse_side(left, prob)
    rhs = _parse_side(right, prob)
    if not (lhs - rhs).has(prob.yp):
        raise InputError("The equation needs a derivative: write y' or dy/d%s." % indep)
    return prob, lhs, rhs


def parse_number(text, label):
    """Parse a constant such as 1/2, -3.5, pi or sqrt(2). Returns a SymPy number."""
    text = _normalise_text(text)
    if not _ALLOWED_CHARS.match(text) or "=" in text:
        raise InputError("%s must be a number." % label)
    try:
        value = _parse_side(text, Problem(), allow_vars=False)
    except InputError:
        raise InputError("%s must be a number such as 2, -1/3 or pi." % label)
    if value.free_symbols or value.is_real is not True or not value.is_finite:
        raise InputError("%s must be a real number." % label)
    return sp.nsimplify(value) if value.is_Float else value
