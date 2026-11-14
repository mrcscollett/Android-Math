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
    # Unit step u(t - a) and impulse delta(t - a), used with Laplace transforms.
    "u": sp.Heaviside, "delta": sp.DiracDelta,
    # Written n! in the series tool, which rewrites it to factorial(n) before parsing.
    "factorial": sp.factorial,
}
_CONSTANTS = {"pi": sp.pi}

_UNICODE = {
    "′": "'", "’": "'", "‘": "'", "`": "'",
    "×": "*", "·": "*", "⋅": "*", "−": "-", "–": "-",
    "÷": "/", "π": "pi", "√": "sqrt", "²": "^2", "³": "^3",
}

_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-z+\-*/^().,=\s]*$")
_PRIME = "D"  # single-letter stand-in for y' while parsing
_PRIME2 = "W"  # and for y'' (Q, S, N, E, I, O are taken by SymPy)
_PRIME3 = "J"  # y'''
_PRIME4 = "K"  # y''''
_PLACEHOLDERS = (_PRIME, _PRIME2, _PRIME3, _PRIME4)
MAX_ORDER = 4


class Problem:
    """Symbols for one problem: independent variable, y and y'."""

    def __init__(self, indep="x"):
        self.indep_name = indep
        self.x = sp.Symbol(indep, real=True)
        self.y = sp.Symbol("y", real=True)
        self.yp = sp.Symbol("y'", real=True)
        self.ypp = sp.Symbol("y''", real=True)
        self.y3 = sp.Symbol("y'''", real=True)
        self.y4 = sp.Symbol("y''''", real=True)
        self.C3 = sp.Symbol("C_3", real=True)
        self.C4 = sp.Symbol("C_4", real=True)
        self.C = sp.Symbol("C", real=True)
        self.C1 = sp.Symbol("C_1", real=True)
        self.C2 = sp.Symbol("C_2", real=True)
        self.order = 1


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


def _readable(word):
    for placeholder, primes in zip(reversed(_PLACEHOLDERS), ("''''", "'''", "''", "'")):
        word = word.replace(placeholder, "y" + primes)
    return word


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
                "function names, for example x*sin(x)." % _readable(word)
            )
        raise InputError(
            "Unknown symbol '%s'. This problem can use %s, y, y', numbers, pi, e "
            "and functions such as sin, cos, exp, ln, sqrt."
            % ("".join(unknown) if len(word) == 1 else _readable(word),
               (sorted(variables - {"y", "e"} - set(_PLACEHOLDERS)) or ["x"])[0])
        )


def _parse_side(text, prob, allow_vars=True, variables=None):
    if not text.strip():
        raise InputError("One side of the equation is empty.")
    if variables is None:
        variables = ({prob.indep_name, "y", "e"} | set(_PLACEHOLDERS)) if allow_vars else {"e"}
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
        sp.Symbol(_PRIME2): prob.ypp,
        sp.Symbol(_PRIME3): prob.y3,
        sp.Symbol(_PRIME4): prob.y4,
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

    for n in (4, 3, 2):
        text = re.sub(r"d\s*\^?\s*%d\s*y\s*/\s*d\s*%s\s*\^?\s*%d" % (n, indep, n),
                      "y" + "'" * n, text)
    if "'" * (MAX_ORDER + 1) in text:
        raise InputError("Equations above fourth order are not supported.")
    text = re.sub(r"dy\s*/\s*d%s" % indep, " y' ", text)
    text = re.sub(r"y\s*('+)\s*\(\s*%s\s*\)" % indep, r" y\1 ", text)
    text = re.sub(r"y\s*\(\s*%s\s*\)" % indep, " y ", text)
    for placeholder in _PLACEHOLDERS:
        if placeholder in text:
            raise InputError("Unknown symbol '%s'." % placeholder)
    for n in (4, 3, 2, 1):
        text = re.sub(r"y\s*" + "'" * n, " %s " % _PLACEHOLDERS[n - 1], text)
    if "'" in text:
        raise InputError("Only y can carry primes (y', y'', ...).")

    if text.count("=") != 1:
        raise InputError("Write the equation with a single = sign, "
                         "for example y' + 2y = x.")
    left, right = text.split("=")
    lhs = _parse_side(left, prob)
    rhs = _parse_side(right, prob)
    diff = lhs - rhs
    derivs = (prob.yp, prob.ypp, prob.y3, prob.y4)
    if not diff.has(*derivs):
        raise InputError("The equation needs a derivative: write y' or dy/d%s." % indep)
    prob.order = max(n + 1 for n, d in enumerate(derivs) if diff.has(d))
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


def parse_expression(text, names):
    """Parse an expression (no = sign) in the given variable names.

    Returns (expr, symbols) where symbols maps each name to its real Symbol.
    """
    text = _normalise_text(text)
    if not text:
        raise InputError("Enter an expression.")
    if "=" in text:
        raise InputError("Enter an expression only, without an = sign.")
    if not _ALLOWED_CHARS.match(text):
        bad = sorted(set(ch for ch in text if not _ALLOWED_CHARS.match(ch)))
        raise InputError("Unexpected character: %s" % " ".join(bad))
    prob = Problem(names[0])
    try:
        expr = _parse_side(text, prob, variables=set(names) | {"e"})
    except InputError as err:
        message = str(err)
        if message.startswith("Unknown symbol"):
            message = message.split(". ")[0] + (
                ". Here you can use %s, numbers, pi, e and functions such as "
                "sin, cos, exp, sqrt, u(...) and delta(...)." % ", ".join(names))
        raise InputError(message)
    symbols = {}
    swaps = {}
    for name in names:
        symbols[name] = sp.Symbol(name, real=True)
        swaps[sp.Symbol(name)] = symbols[name]
    swaps[prob.x] = symbols[names[0]]
    return expr.xreplace(swaps), symbols
