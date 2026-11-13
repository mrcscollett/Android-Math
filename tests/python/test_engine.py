"""Engine tests. Particular solutions are checked against an independent
numerical integration (mpmath), not against the engine's own verifier."""
import json
import math

import mpmath
import pytest
import sympy as sp

from deengine import preview, solve

X, Y, C = sp.symbols("x y C")


def run(text, x0="", y0=""):
    return json.loads(solve(text, x0, y0))


def method(result, method_id):
    return next(m for m in result["methods"] if m["id"] == method_id)


def plot_fn(expr_text):
    return sp.lambdify(X, sp.sympify(expr_text, locals={"x": X, "y": Y, "C": C}), "mpmath")


# (equation, x0, y0, methods expected to apply, x to compare at)
IVPS = [
    ("y' = x*y", "0", "1", {"separable", "linear"}, 1.0),
    ("y' = y", "0", "2", {"separable", "linear"}, 1.5),
    ("dy/dx = x/y", "0", "-1", {"separable"}, 1.0),
    ("y' = y^2", "0", "1", {"separable"}, 0.5),
    ("y' = 1 + y^2", "0", "0", {"separable"}, 1.0),
    ("y' = y(1-y)", "0", "1/2", {"separable"}, 2.0),
    ("y' = (y-1)(y+2)", "0", "0", {"separable"}, 1.0),
    ("y' = x^2", "1", "2", {"separable"}, 2.0),
    ("y' = sin(x) y", "0", "1", {"separable", "linear"}, 2.0),
    ("y' = sqrt(y)", "0", "1", {"separable"}, 1.0),
    ("y' = x e^(-y)", "0", "0", {"separable"}, 1.0),
    ("(1+x^2) y' = x y", "0", "3", {"separable", "linear"}, 2.0),
    ("x y' + y = x^2", "1", "1", {"linear"}, 2.0),
    ("y' - y/x = x", "1", "0", {"linear"}, 2.0),
    ("y' + y tan(x) = cos(x)", "0", "1", {"linear"}, 1.0),
    ("y' = x + y", "0", "1", {"linear"}, 1.0),
    ("y' + 2y = x", "0", "0", {"linear"}, 1.0),
    ("dy/dt = -0.5 y + 3", "0", "10", {"separable", "linear"}, 2.0),
    ("y' = y/x + 1", "1", "0", {"linear"}, 2.0),
    ("y' = y ln(x)", "1", "1", {"separable", "linear"}, 2.0),
    ("y' = y/(x-1)", "0", "2", {"separable", "linear"}, 0.5),
    ("y' = 3", "0", "-1", {"separable"}, 2.0),
    ("y' = -2 x y^2", "0", "1", {"separable"}, 1.0),
    ("y' = y cos(x)", "0", "-1", {"separable", "linear"}, 1.0),
    ("y' + y = e^(-x)", "0", "2", {"linear"}, 1.0),
]


@pytest.mark.parametrize("text,x0,y0,expected,x1", IVPS)
def test_ivp_matches_numerical_solution(text, x0, y0, expected, x1):
    r = run(text, x0, y0)
    assert r["ok"], r.get("error")
    applicable = {m["id"] for m in r["methods"] if m["applicable"]}
    assert applicable == expected
    f = sp.lambdify((X, Y), sp.sympify(r["rhs_expr"], locals={"x": X, "y": Y}), "mpmath")
    a, b = r["ic"]["x0"], r["ic"]["y0"]
    reference = mpmath.odefun(lambda t, v: f(t, v), a, b)(x1)
    for m in r["methods"]:
        if not m["applicable"]:
            continue
        assert m["verified"] is True, m["verify_note"]
        assert m["particular_latex"]
        assert m["steps"][-1]["title"] == "Check"
        expr = m["plot"]["particular_expr"]
        assert expr, "particular solution should be plottable"
        got = plot_fn(expr)(x1)
        assert abs(got - reference) < 1e-6 * (1 + abs(reference)), (m["id"], got, reference)
        assert abs(plot_fn(expr)(a) - b) < 1e-9


GENERAL = [
    ("y' = x*y", {"separable", "linear"}),
    ("y' + 2y = x", {"linear"}),
    ("y' = y^2", {"separable"}),
    ("y' = e^(x+y)", {"separable"}),
    ("y' = 2x y + 1", {"linear"}),
    ("y' = e^(x^2)", {"separable"}),
    ("2y' = 4", {"separable"}),
    ("y' = x/y", {"separable"}),
    ("y' = cos(y)", {"separable"}),
]


@pytest.mark.parametrize("text,expected", GENERAL)
def test_general_solution(text, expected):
    r = run(text)
    assert r["ok"], r.get("error")
    assert {m["id"] for m in r["methods"] if m["applicable"]} == expected
    for m in r["methods"]:
        if m["applicable"]:
            assert m["verified"] is not False
            assert m["general_latex"]
            assert m["particular_latex"] is None
            for s in m["steps"]:
                assert set(s) >= {"title", "text", "latex", "substeps"}


def test_general_solution_has_constant_and_solves_ode():
    r = run("y' + 2y = x")
    expr = sp.sympify(method(r, "linear")["plot"]["general_expr"], locals={"x": X, "C": C})
    assert C in expr.free_symbols
    assert sp.simplify(expr.diff(X) + 2 * expr - X) == 0


def test_both_methods_agree():
    r = run("y' = x*y", "0", "1")
    exprs = {m["plot"]["particular_expr"] for m in r["methods"]}
    assert exprs == {"exp(x**2/2)"}


def test_neither_method_applies():
    r = run("y' = x^2 + y^2", "0", "0")
    assert r["ok"]
    assert not any(m["applicable"] for m in r["methods"])
    assert all(m["reason"] for m in r["methods"])
    assert r["rhs_expr"] == "x**2 + y**2"
    assert "numerical" in r["summary"]


def test_names_other_type_when_recognised():
    r = run("y' + y = x y^2")
    assert not any(m["applicable"] for m in r["methods"])
    assert "Bernoulli" in r["other_types"]


def test_equilibrium_initial_condition():
    r = run("y' = y(1-y)", "0", "1")
    m = method(r, "separable")
    assert m["plot"]["particular_expr"] == "1"
    assert m["verified"] is True


def test_non_elementary_integral_is_reported_not_faked():
    r = run("y' = x^x")
    m = method(r, "separable")
    assert m["applicable"]
    assert m["plot"]["general_expr"] is None
    assert "\\int" in m["general_latex"]


def test_independent_variable_t():
    r = run("dy/dt = -y", "0", "1")
    assert r["indep"] == "t"
    assert "dt" in r["normal_latex"]
    assert method(r, "linear")["plot"]["particular_expr"] == "exp(-x)"


@pytest.mark.parametrize("text,fragment", [
    ("", "Enter an equation"),
    ("y'' + y = 0", "second-order"),
    ("y' = 3y + z", "Unknown symbol"),
    ("y' = xsin(x)", "couldn't read"),
    ("y'^2 = x", "first power"),
    ("y = x", "derivative"),
    ("y' = ", "empty"),
    ("y' = x = y", "single ="),
    ("y' = (x", "couldn't read"),
    ("y' = __import__('os')", "Unexpected character"),
    ("y' = x.real", "Unknown symbol"),
    ("y' = 1/0", "undefined"),
])
def test_bad_input_gives_readable_error(text, fragment):
    r = run(text)
    assert not r["ok"]
    assert fragment in r["error"]
    assert "Traceback" not in r["error"]


def test_bad_initial_conditions():
    assert "both" in run("y' = y", "1", "")["error"]
    assert "number" in run("y' = y", "a", "1")["error"]
    assert "not defined" in run("y' = y/x", "0", "1")["error"]


def test_preview():
    assert json.loads(preview("y' + 2y = x")) == {"ok": True, "latex": "y' + 2 y = x"}
    assert json.loads(preview(""))["latex"] == ""
    assert not json.loads(preview("y' = 2 +* "))["ok"]
    assert json.loads(preview("dy/dx = xy"))["latex"] == "y' = x y"


def test_unicode_input():
    r = run("y′ = 2·x − y")
    assert r["ok"] and r["rhs_expr"] == "2*x - y"


def test_export_fixtures():
    """Write sample outputs used by the Kotlin contract tests."""
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "..", "core", "src", "test", "resources")
    os.makedirs(out, exist_ok=True)
    samples = {
        "both_methods_ivp.json": ("y' = x*y", "0", "1"),
        "linear_general.json": ("y' + 2y = x", "", ""),
        "none_applicable.json": ("y' = x^2 + y^2", "0", "0"),
        "error.json": ("y' = z", "", ""),
    }
    for name, args in samples.items():
        with open(os.path.join(out, name), "w") as fh:
            fh.write(solve(*args))
