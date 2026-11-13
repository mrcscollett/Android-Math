"""Engine tests. Particular solutions are checked against an independent
numerical integration (mpmath), not against the engine's own verifier."""
import json
import math

import mpmath
import pytest
import sympy as sp

from deengine import numeric, preview, solve

X, Y, C = sp.symbols("x y C")


def run(text, x0="", y0=""):
    return json.loads(solve(text, x0, y0))


def method(result, method_id):
    return next(m for m in result["methods"] if m["id"] == method_id)


def plot_fn(expr_text):
    return sp.lambdify(X, sp.sympify(expr_text, locals={"x": X, "y": Y, "C": C}), "mpmath")


CORE = {"separable", "linear"}

# (equation, x0, y0, which of the two core methods apply, x to compare at)
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
    assert applicable & CORE == expected
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
    assert {m["id"] for m in r["methods"] if m["applicable"]} & CORE == expected
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
    exprs = {m["plot"]["particular_expr"] for m in r["methods"] if m["applicable"]}
    assert exprs == {"exp(x**2/2)"}


def test_neither_method_applies():
    r = run("y' = x^2 + y^2", "0", "0")
    assert r["ok"]
    assert not any(m["applicable"] for m in r["methods"])
    assert all(m["reason"] for m in r["methods"])
    assert r["rhs_expr"] == "x**2 + y**2"
    assert "numerical" in r["summary"]


def test_reasons_are_given_for_every_method():
    r = run("y' = x*y")
    assert [m["id"] for m in r["methods"]] == [
        "separable", "linear", "exact", "bernoulli", "homogeneous", "laplace"]
    assert all(m["reason"] for m in r["methods"])


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
    ("y''' + y = 0", "Third-order"),
    ("y'' + y^2 = 0", "nonlinearly"),
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
    assert json.loads(preview("y' + 2y = x")) == {"ok": True, "order": 1, "latex": "y' + 2 y = x"}
    assert json.loads(preview("y'' + y = 0"))["order"] == 2
    assert json.loads(preview(""))["latex"] == ""
    assert not json.loads(preview("y' = 2 +* "))["ok"]
    assert json.loads(preview("dy/dx = xy"))["latex"] == "y' = x y"


def test_unicode_input():
    r = run("y′ = 2·x − y")
    assert r["ok"] and r["rhs_expr"] == "2*x - y"


# ---- phase 2: more first-order methods ------------------------------------

# (equation, x0, y0, method ids that must apply, x to compare at)
MORE_IVPS = [
    ("2xy + (x^2 + 1) y' = 0", "0", "1", {"exact", "separable", "linear"}, 1.0),
    ("(2x + y) + (x + 2y) y' = 0", "1", "1", {"exact", "homogeneous"}, 1.3),
    ("y' + y = x y^2", "0", "1", {"bernoulli"}, 1.0),
    ("y' - y/x = y^3", "1", "1", {"bernoulli"}, 1.2),
    ("y' = (x^2 + y^2)/(x y)", "1", "2", {"exact", "bernoulli", "homogeneous"}, 2.0),
    ("y' = (x+y)/x", "1", "0", {"linear", "homogeneous"}, 2.0),
    ("y' = y/x + (y/x)^2", "1", "1", {"bernoulli", "homogeneous"}, 1.5),
    ("y' + 2y = e^(-x)", "0", "1", {"linear", "laplace"}, 1.0),
    ("y' = 3y + 2", "0", "0", {"separable", "linear", "laplace"}, 0.5),
    ("y' + y = y^2 e^x", "0", "1/2", {"bernoulli"}, 0.5),
]


@pytest.mark.parametrize("text,x0,y0,expected,x1", MORE_IVPS)
def test_more_first_order_methods(text, x0, y0, expected, x1):
    r = run(text, x0, y0)
    assert r["ok"], r.get("error")
    applicable = {m["id"] for m in r["methods"] if m["applicable"]}
    assert applicable == expected
    f = sp.lambdify((X, Y), sp.sympify(r["rhs_expr"], locals={"x": X, "y": Y}), "mpmath")
    a, b = r["ic"]["x0"], r["ic"]["y0"]
    reference = mpmath.odefun(lambda t, v: f(t, v), a, b)(x1)
    explicit = 0
    for m in r["methods"]:
        if not m["applicable"]:
            continue
        assert m["verified"] is not False, (m["id"], m["verify_note"])
        assert m["particular_latex"], m["id"]
        expr = m["plot"]["particular_expr"]
        if expr:
            explicit += 1
            got = plot_fn(expr)(x1)
            assert abs(got - reference) < 1e-6 * (1 + abs(reference)), (m["id"], got, reference)
    assert explicit >= 1


def test_exact_with_integrating_factor():
    r = run("y + (2x - y e^y) y' = 0")
    m = method(r, "exact")
    assert m["applicable"] and m["verified"] is True
    assert "integrating factor" in m["reason"]
    titles = [s["title"] for s in m["steps"]]
    assert "Multiply by the integrating factor" in titles


def test_exact_leaves_cubic_implicit_quickly():
    import time
    start = time.time()
    r = run("(3x^2 y + y^3) + (x^3 + 3 x y^2) y' = 0", "1", "1")
    assert time.time() - start < 8
    m = method(r, "exact")
    assert m["applicable"] and m["plot"]["particular_expr"] is None
    assert "= 2" in m["particular_latex"]


# ---- phase 2: second order ------------------------------------------------

# (equation, x0, y0, y'0, method ids that must apply, x to compare at)
SECOND = [
    ("y'' + 3y' + 2y = e^x", "0", "1", "0", {"undetermined", "variation", "laplace"}, 1.0),
    ("y'' + y = 0", "0", "1", "2", {"characteristic", "laplace"}, 2.0),
    ("y'' - 2y' + y = 0", "0", "1", "-1", {"characteristic", "laplace"}, 1.0),
    ("y'' + 2y' + 5y = 0", "0", "0", "1", {"characteristic", "laplace"}, 1.5),
    ("y'' + y = sin(x)", "0", "0", "0", {"undetermined", "variation", "laplace"}, 2.0),
    ("y'' - 4y' + 4y = e^(2x)", "0", "1", "1", {"undetermined", "variation", "laplace"}, 0.7),
    ("y'' - y = x e^x", "0", "1", "0", {"undetermined", "variation", "laplace"}, 1.0),
    ("y'' + 4y = x^2 + cos(2x)", "0", "0", "0", {"undetermined", "variation", "laplace"}, 1.0),
    ("2y'' + y' - y = 3x", "1", "0", "1", {"undetermined", "variation"}, 2.0),
    ("y'' = 6x", "0", "1", "2", {"undetermined", "variation", "laplace"}, 1.5),
    ("x^2 y'' + x y' - y = 0", "1", "2", "0", {"cauchy_euler"}, 2.0),
    ("x^2 y'' - 3x y' + 4y = 0", "1", "1", "0", {"cauchy_euler"}, 2.0),
    ("x^2 y'' + x y' + y = 0", "1", "1", "1", {"cauchy_euler"}, 2.0),
    ("d2y/dt2 + 4y = 0", "0", "0", "2", {"characteristic", "laplace"}, 1.0),
]


def _second_reference(text, x0, y0, y1, x1):
    """Integrate a y'' + b y' + c y = g as a first-order system with mpmath."""
    from deengine.parsing import parse_equation
    from deengine.second_order import analyse
    prob, lhs, rhs = parse_equation(text)
    eq = analyse(prob, lhs, rhs)
    rhs_fn = sp.lambdify((prob.x, prob.y, prob.yp),
                         (eq.g - eq.b * prob.yp - eq.c * prob.y) / eq.a, "mpmath")
    sol = mpmath.odefun(lambda t, v: [v[1], rhs_fn(t, v[0], v[1])],
                        float(sp.sympify(x0)), [float(sp.sympify(y0)), float(sp.sympify(y1))])
    return sol(x1)[0]


@pytest.mark.parametrize("text,x0,y0,y1,expected,x1", SECOND)
def test_second_order_ivp(text, x0, y0, y1, expected, x1):
    r = json.loads(solve(text, x0, y0, y1))
    assert r["ok"], r.get("error")
    assert r["order"] == 2 and r["rhs_expr"] is None
    assert {m["id"] for m in r["methods"] if m["applicable"]} == expected
    reference = _second_reference(text, x0, y0, y1, x1)
    answers = set()
    for m in r["methods"]:
        if not m["applicable"]:
            continue
        assert m["verified"] is True, (m["id"], m["verify_note"])
        got = plot_fn(m["plot"]["particular_expr"])(x1)
        assert abs(got - reference) < 1e-6 * (1 + abs(reference)), (m["id"], got, reference)
        answers.add(round(float(got), 9))
        assert m["steps"][-1]["title"] == "Check"
    assert len(answers) == 1


def test_second_order_general_solution_has_two_constants():
    r = run("y'' + 3y' + 2y = e^x")
    assert method(r, "laplace")["applicable"] is False
    m = method(r, "undetermined")
    C2 = sp.Symbol("C2")
    expr = sp.sympify(m["plot"]["general_expr"], locals={"x": X, "C": C, "C2": C2})
    assert {C, C2} <= expr.free_symbols
    assert sp.simplify(expr.diff(X, 2) + 3 * expr.diff(X) + 2 * expr - sp.exp(X)) == 0


def test_resonance_multiplies_the_guess():
    m = method(run("y'' + y = sin(x)"), "undetermined")
    guess = next(s for s in m["steps"] if s["title"] == "Guess the particular solution")
    assert "multiplied" in guess["text"]


def test_variation_handles_what_undetermined_cannot():
    r = run("y'' + y = tan(x)")
    assert method(r, "undetermined")["applicable"] is False
    v = method(r, "variation")
    assert v["applicable"] and v["verified"] is True


def test_second_order_errors():
    assert "all three" in json.loads(solve("y'' + y = 0", "0", "1", ""))["error"]
    r = run("x y'' + y = 0")
    assert r["ok"] and not any(m["applicable"] for m in r["methods"])
    assert "variable coefficients" in r["summary"]


# ---- phase 2: numerical methods -------------------------------------------

def test_numeric_tables_and_accuracy():
    r = json.loads(numeric("y' = x*y", "0", "1", "0.25", "4"))
    assert r["ok"] and r["kind"] == "numeric"
    assert [m["id"] for m in r["methods"]] == ["euler", "heun", "rk4"]
    exact = math.exp(0.5)
    errors = []
    for m in r["methods"]:
        pts = m["plot"]["points"]
        assert len(pts) == 5 and pts[0] == [0.0, 1.0] and abs(pts[-1][0] - 1.0) < 1e-12
        errors.append(abs(pts[-1][1] - exact))
        table = next(s for s in m["steps"] if "table" in s)["table"]
        assert len(table["rows"]) == 5
        assert all(len(row) == len(table["headers"]) for row in table["rows"])
        assert m["plot"]["particular_expr"] == "exp(x**2/2)"
    assert errors[0] > errors[1] > errors[2]
    assert abs(r["methods"][0]["plot"]["points"][-1][1] - 1.419434) < 1e-6   # hand-checked Euler
    assert errors[2] < 2e-5


def test_numeric_without_exact_solution():
    r = json.loads(numeric("y' = x^2 + y^2", "0", "0", "0.1", "10"))
    assert r["ok"]
    m = r["methods"][2]
    assert m["plot"]["particular_expr"] is None
    assert abs(m["plot"]["points"][-1][1] - 0.350232) < 1e-5
    table = next(s for s in m["steps"] if "table" in s)["table"]
    assert "\\text{exact}" not in table["headers"]


def test_numeric_stops_before_blow_up():
    r = json.loads(numeric("y' = y^2", "0", "1", "0.25", "8"))
    for m in r["methods"]:
        assert m["plot"]["points"][-1][0] < 1.0
        assert "blows up" in m["steps"][-1]["text"]


def test_numeric_negative_step_and_long_table():
    r = json.loads(numeric("y' = y", "0", "1", "-0.01", "200"))
    m = r["methods"][2]
    assert abs(m["plot"]["points"][-1][1] - math.exp(-2)) < 1e-8
    assert len(m["plot"]["points"]) == 201
    table = next(s for s in m["steps"] if "table" in s)["table"]
    assert len(table["rows"]) <= 62 and table["rows"][-1][0] == "200"


@pytest.mark.parametrize("args,fragment", [
    (("y'=y", "0", "1", "0", "5"), "cannot be 0"),
    (("y'=y", "", "1", "0.1", "5"), "starting point"),
    (("y''=y", "0", "1", "0.1", "5"), "first-order"),
    (("y'=y", "0", "1", "0.1", "x"), "whole number"),
    (("y'=y", "0", "1", "0.1", "5000"), "between 1 and"),
    (("y'=y", "0", "1", "", "5"), "step size"),
])
def test_numeric_errors(args, fragment):
    r = json.loads(numeric(*args))
    assert not r["ok"] and fragment in r["error"]


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
        "second_order_ivp.json": ("y'' + 3y' + 2y = e^x", "0", "1", "0"),
        "second_order_general.json": ("y'' + y = 0", "", "", ""),
    }
    for name, args in samples.items():
        with open(os.path.join(out, name), "w") as fh:
            fh.write(solve(*args))
    with open(os.path.join(out, "numeric.json"), "w") as fh:
        fh.write(numeric("y' = x*y", "0", "1", "0.25", "4"))
