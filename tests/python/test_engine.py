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
        if not m["applicable"] or m.get("qualitative"):
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
        if m["applicable"] and not m.get("qualitative"):
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
        "separable", "linear", "exact", "bernoulli", "homogeneous", "laplace", "phase_line"]
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
    ("y''''' + y = 0", "above fourth order"),
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
    ("y' = 3y + 2", "0", "0", {"separable", "linear", "laplace", "phase_line"}, 0.5),
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
        if not m["applicable"] or m.get("qualitative"):
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
        if not m["applicable"] or m.get("qualitative"):
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
    r = run("y'' + sin(x) y = 0")
    assert r["ok"] and not any(m["applicable"] for m in r["methods"])
    assert "variable coefficients" in r["summary"]


# ---- phase 2: numerical methods -------------------------------------------

def test_numeric_tables_and_accuracy():
    r = json.loads(numeric("y' = x*y", "0", "1", "0.25", "4"))
    assert r["ok"] and r["kind"] == "numeric"
    assert [m["id"] for m in r["methods"]] == ["euler", "heun", "rk4", "midpoint", "ab4", "rkf45"]
    exact = math.exp(0.5)
    errors = []
    for m in r["methods"]:
        pts = m["plot"]["points"]
        assert pts[0] == [0.0, 1.0] and abs(pts[-1][0] - 1.0) < 1e-12
        if m["id"] != "rkf45":
            assert len(pts) == 5
        errors.append(abs(pts[-1][1] - exact))
        table = next(s for s in m["steps"] if "table" in s)["table"]
        assert len(table["rows"]) == len(pts)
        assert all(len(row) == len(table["headers"]) for row in table["rows"])
        assert m["plot"]["particular_expr"] == "exp(x**2/2)"
    assert errors[0] > errors[1] > errors[2]
    by_id = dict(zip([m["id"] for m in r["methods"]], errors))
    assert by_id["midpoint"] < by_id["euler"] / 10
    assert by_id["rkf45"] < 1e-4
    assert abs(r["methods"][0]["plot"]["points"][-1][1] - 1.419434) < 1e-6   # hand-checked Euler
    assert errors[2] < 2e-5


def test_multistep_and_adaptive_methods():
    r = json.loads(numeric("y' = -2y + x", "0", "1", "0.05", "40"))
    exact = lambda x: 1.25 * math.exp(-2 * x) + x / 2 - 0.25
    methods = {m["id"]: m for m in r["methods"]}
    ab4 = methods["ab4"]["plot"]["points"]
    assert len(ab4) == 41 and abs(ab4[-1][1] - exact(2.0)) < 1e-5
    # Midpoint by hand: k1 = -2, k2 = f(0.025, 0.95) = -1.875, y1 = 1 - 0.05*1.875.
    assert abs(methods["midpoint"]["plot"]["points"][1][1] - 0.90625) < 1e-12
    adaptive = methods["rkf45"]["plot"]["points"]
    assert abs(adaptive[-1][0] - 2.0) < 1e-9
    assert abs(adaptive[-1][1] - exact(2.0)) < 1e-5
    steps = [b[0] - a[0] for a, b in zip(adaptive, adaptive[1:])]
    assert max(steps) > 1.5 * min(steps[:-1] or steps)   # the step size really adapts
    assert all(abs(p[1] - exact(p[0])) < 1e-4 for p in adaptive)


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


# ---- phase 3: Laplace transforms ------------------------------------------

from deengine import transform


def _numeric_laplace(expr_text, s_value):
    """Integral of e^(-s t) f(t) from 0 to infinity, computed by quadrature."""
    f = sp.lambdify(X, sp.sympify(expr_text, locals={"x": X, "Heaviside": sp.Heaviside}),
                    [{"Heaviside": lambda v, *rest: 0.0 if v < 0 else 1.0}, "mpmath"])
    breaks = [0, 1, 2, 3, 5, 10, 40]
    return mpmath.quad(lambda t: mpmath.exp(-s_value * t) * f(t), breaks)


FORWARD = ["3", "t^2 + 2", "e^(2t) sin(3t)", "t cos(2t)", "t^3 e^(-t)",
           "5 e^(-t) - 3 sinh(2t)", "u(t-2)", "(t-2) u(t-2)", "cos(t) + e^(-3t) t"]


@pytest.mark.parametrize("text", FORWARD)
def test_forward_transform_matches_quadrature(text):
    r = json.loads(transform(text, "forward"))
    assert r["ok"], r.get("error")
    table = method(r, "table")
    assert table["verified"] is True
    answer = table["particular_latex"].split(" = ", 1)[1]
    plot = table["plot"]["particular_expr"]
    assert plot
    # Recompute the transform with SymPy only to evaluate it; the reference is the integral.
    from deengine.parsing import parse_expression
    f, symbols = parse_expression(text, ["t"])
    s = sp.Symbol("s")
    F = sp.laplace_transform(f, symbols["t"], s, noconds=True)
    for s_value in (4.0, 6.5):
        reference = _numeric_laplace(plot, s_value)
        assert abs(complex(F.subs(s, s_value)) - reference) < 1e-6 * (1 + abs(reference))
    assert answer


def test_forward_methods_and_rules():
    r = json.loads(transform("e^(2t) sin(3t)"))
    assert [m["id"] for m in r["methods"]] == ["table", "definition"]
    assert all(m["applicable"] for m in r["methods"])
    assert "First shifting theorem" in method(r, "table")["steps"][0]["text"]
    assert "\\frac{3}{s^{2} - 4 s + 13}" in method(r, "definition")["particular_latex"]
    step_fn = json.loads(transform("u(t-2)"))
    assert method(step_fn, "definition")["applicable"] is False
    assert "u\\left(t - 2\\right)" in step_fn["input_latex"]


INVERSE = ["1/(s(s+1))", "(s+3)/(s^2+2s+5)", "1/(s-1)^3", "e^(-2s)/(s^2+1)",
           "e^(-s)/s + 1/(s+2)", "(2s+1)/((s-1)(s^2+4))", "1/(s^2+1)^2", "1/(s^2 - 4)"]


@pytest.mark.parametrize("text", INVERSE)
def test_inverse_transform_matches_quadrature(text):
    r = json.loads(transform(text, "inverse"))
    assert r["ok"], r.get("error")
    from deengine.parsing import parse_expression
    F, symbols = parse_expression(text, ["s"])
    answers = []
    for m in r["methods"]:
        if not m["applicable"] or m.get("qualitative"):
            continue
        assert m["verified"] is True
        for s_value in (4.0, 6.5):
            got = _numeric_laplace(m["plot"]["particular_expr"], s_value)
            want = complex(F.subs(symbols["s"], s_value))
            assert abs(got - want) < 1e-6 * (1 + abs(want)), (m["id"], got, want)
        answers.append(m["id"])
    assert "fractions" in answers


def test_inverse_steps():
    r = json.loads(transform("(2s+1)/((s-1)(s^2+4))", "inverse"))
    titles = [s["title"] for s in method(r, "fractions")["steps"]]
    assert titles[:2] == ["Split into partial fractions", "Invert each piece"]
    assert method(r, "convolution")["applicable"]
    delayed = json.loads(transform("e^(-2s)/(s^2+1)", "inverse"))
    assert "Apply the delay" in [s["title"] for s in method(delayed, "fractions")["steps"]]
    assert method(delayed, "convolution")["applicable"] is False
    assert "e^{- t} \\cos" in json.loads(transform("(s+3)/(s^2+2s+5)", "inverse"))["methods"][0]["particular_latex"]


def test_transform_errors_and_preview():
    assert "Unknown symbol 'y'" in json.loads(transform("y + 1"))["error"]
    assert "without an = sign" in json.loads(transform("t = 1"))["error"]
    assert "could not be found" in json.loads(transform("1/t"))["error"]
    assert "rational functions" in json.loads(transform("sin(s)/(e^s + 1)", "inverse"))["error"]
    assert json.loads(preview("1/(s^2+1)", "inverse"))["latex"].startswith("\\mathcal{L}^{-1}")
    assert json.loads(preview("t^2", "forward"))["ok"]


# (equation, y0, y'0 or "", x to compare at)
FORCED = [
    ("y'' + y = u(t-2)", "0", "0", 5.0),
    ("y'' + 4y = u(t-1) - u(t-2)", "1", "0", 4.0),
    ("y' + y = u(t-1)", "2", "", 3.0),
    ("y'' + 3y' + 2y = u(t-1)", "0", "1", 3.5),
]


@pytest.mark.parametrize("text,y0,y1,x1", FORCED)
def test_step_forcing_matches_numerical_solution(text, y0, y1, x1):
    r = json.loads(solve(text, "0", y0, y1))
    assert r["ok"], r.get("error")
    assert [m["id"] for m in r["methods"] if m["applicable"]] == ["laplace"]
    m = method(r, "laplace")
    assert m["verified"] is True
    expr = sp.sympify(m["plot"]["particular_expr"], locals={"x": X})
    got = float(expr.subs(X, x1))
    step = lambda v, *rest: 0.0 if v < 0 else 1.0
    from deengine.parsing import parse_equation
    prob, lhs, rhs = parse_equation(text)
    if y1:
        from deengine.second_order import analyse
        eq = analyse(prob, lhs, rhs)
        rhs_fn = sp.lambdify((prob.x, prob.y, prob.yp), (eq.g - eq.b * prob.yp - eq.c * prob.y) / eq.a,
                             [{"Heaviside": step}, "math"])
        state = [float(y0), float(y1)]
        deriv = lambda t, v: [v[1], rhs_fn(t, v[0], v[1])]
    else:
        f = sp.solve(lhs - rhs, prob.yp)[0]
        rhs_fn = sp.lambdify((prob.x, prob.y), f, [{"Heaviside": step}, "math"])
        state = [float(y0)]
        deriv = lambda t, v: [rhs_fn(t, v[0])]
    # Plain RK4 with a small step: adaptive integrators dislike the jumps.
    t, h = 0.0, 0.0005
    while t < x1 - 1e-12:
        k1 = deriv(t, state)
        k2 = deriv(t + h / 2, [a + h / 2 * b for a, b in zip(state, k1)])
        k3 = deriv(t + h / 2, [a + h / 2 * b for a, b in zip(state, k2)])
        k4 = deriv(t + h, [a + h * b for a, b in zip(state, k3)])
        state = [a + h / 6 * (p + 2 * q + 2 * u + w) for a, p, q, u, w in zip(state, k1, k2, k3, k4)]
        t += h
    assert abs(got - state[0]) < 1e-4, (got, state[0])


def test_impulse_forcing():
    r = json.loads(solve("y'' + 2y' + 5y = delta(t-1)", "0", "0", "0"))
    m = method(r, "laplace")
    assert m["applicable"] and m["verified"] is True
    expr = sp.sympify(m["plot"]["particular_expr"], locals={"x": X})
    assert float(expr.subs(X, 0.5)) == 0.0
    # After the impulse: e^{-(t-1)} sin(2(t-1)) / 2.
    assert abs(float(expr.subs(X, 1.5)) - math.exp(-0.5) * math.sin(1.0) / 2) < 1e-12
    assert "initial condition" in json.loads(solve("y' + y = u(t-1)"))["error"]


# ---- phase 3: series solutions --------------------------------------------

def _series_value(m, x1, key="particular_expr"):
    C2 = sp.Symbol("C2")
    expr = sp.sympify(m["plot"][key], locals={"x": X, "C": C, "C2": C2})
    return expr, float(expr.subs({X: x1, C: 1, C2: 0})), float(expr.subs({X: x1, C: 0, C2: 1}))


@pytest.mark.parametrize("text,y0,y1", [
    ("y'' - x y = 0", "1", "0"),
    ("y'' - x y = 0", "0", "1"),
    ("y'' + x y' + y = 0", "1", "2"),
    ("(1 - x^2) y'' - 2x y' + 6y = 0", "1", "1"),
    ("(1 + x^2) y'' + x y' - y = 0", "2", "-1"),
])
def test_power_series_matches_numerical_solution(text, y0, y1):
    r = json.loads(solve(text, "0", y0, y1))
    m = method(r, "series")
    assert m["applicable"] and m["verified"] is True
    assert "a_{n+2}" in " ".join(l for s in m["steps"] for l in s["latex"])
    expr = sp.sympify(m["plot"]["particular_expr"], locals={"x": X})
    for x1 in (0.3, 0.15):   # mpmath.odefun only integrates forwards
        reference = _second_reference(text, "0", y0, y1, x1)
        # The series is truncated after x^7, so the error is of the size of the x^8 term.
        assert abs(float(expr.subs(X, x1)) - reference) < 2e-5, (x1, float(expr.subs(X, x1)), reference)


def test_airy_coefficients_and_recurrence():
    m = method(run("y'' - x y = 0"), "series")
    text = " ".join(l for s in m["steps"] for l in s["latex"])
    assert "a_{3} = \\frac{a_{0}}{6}" in text and "a_{4} = \\frac{a_{1}}{12}" in text
    assert "a_{6} = \\frac{a_{0}}{180}" in text
    assert m["general_latex"].startswith("y = a_0\\left(1 + \\frac{x^{3}}{6}")


def test_frobenius_two_roots():
    # Bessel's equation of order 1/2: solutions are sin(x)/sqrt(x) and cos(x)/sqrt(x).
    m = method(run("x^2 y'' + x y' + (x^2 - 1/4) y = 0"), "series")
    assert m["applicable"] and m["verified"] is True
    text = " ".join(l for s in m["steps"] for l in s["latex"])
    assert "r_1 = \\frac{1}{2},\\qquad r_2 = - \\frac{1}{2}" in text
    _, first, second = _series_value(m, 0.4, "general_expr")
    assert abs(first - math.sin(0.4) / math.sqrt(0.4)) < 1e-7
    assert abs(second - math.cos(0.4) / math.sqrt(0.4)) < 1e-6


def test_frobenius_log_case_and_limits():
    m = method(run("x^2 y'' + x y' + x^2 y = 0"), "series")   # Bessel J0
    assert m["applicable"]
    assert "Find the second solution with a logarithm" in [s["title"] for s in m["steps"]]
    _, first, _ = _series_value(m, 0.5, "general_expr")
    assert abs(first - float(mpmath.besselj(0, 0.5))) < 1e-6
    assert "irregular" in method(run("x^3 y'' + y = 0"), "series")["reason"]
    assert "not polynomials" in method(run("y'' + sin(x) y = 0"), "series")["reason"]
    assert method(run("y'' + y = 0"), "series")["applicable"] is False
    assert "singular point" in method(json.loads(solve("x y'' + y = 0", "0", "1", "1")), "series")["reason"]


# ---- phase 3: systems -----------------------------------------------------

from deengine import system

LINEAR_SYSTEMS = [
    ("x + 2y", "3x + 2y", "1", "0", "saddle point"),
    ("y", "-x", "1", "0", "center"),
    ("-x - 2y", "2x - y", "1", "1", "spiral sink"),
    ("x - y", "x + 3y", "0", "1", "improper node (source)"),
    ("2x", "2y", "1", "2", "star node (source)"),
    ("-3x + y", "x - 3y", "2", "-1", "node (sink)"),
    ("x + y", "-2x + 3y", "1", "0", "spiral source"),
]


@pytest.mark.parametrize("f,g,x0,y0,kind", LINEAR_SYSTEMS)
def test_linear_system_matches_numerical_solution(f, g, x0, y0, kind):
    r = json.loads(system(f, g, x0, y0))
    assert r["ok"], r.get("error")
    assert kind in r["summary"]
    ff = sp.lambdify((X, Y), sp.sympify(r["field"]["f"], locals={"x": X, "y": Y}), "math")
    gg = sp.lambdify((X, Y), sp.sympify(r["field"]["g"], locals={"x": X, "y": Y}), "math")
    state, h, t_end = [float(x0), float(y0)], 0.0005, 0.8
    for _ in range(int(round(t_end / h))):
        def d(v):
            return [ff(*v), gg(*v)]
        k1 = d(state)
        k2 = d([a + h / 2 * b for a, b in zip(state, k1)])
        k3 = d([a + h / 2 * b for a, b in zip(state, k2)])
        k4 = d([a + h * b for a, b in zip(state, k3)])
        state = [a + h / 6 * (p + 2 * q + 2 * u + w) for a, p, q, u, w in zip(state, k1, k2, k3, k4)]
    assert [m["id"] for m in r["methods"] if m["applicable"]] == ["eigen", "expm"]
    for m in r["methods"][:2]:
        assert m["verified"] is True, m["verify_note"]
        xs = float(sp.sympify(m["plot"]["x_expr"], locals={"x": X}).subs(X, t_end))
        ys = float(sp.sympify(m["plot"]["y_expr"], locals={"x": X}).subs(X, t_end))
        assert abs(xs - state[0]) < 1e-7 * (1 + abs(state[0])) and abs(ys - state[1]) < 1e-7 * (1 + abs(state[1]))


def test_system_general_solution_and_steps():
    r = json.loads(system("x + 2y", "3x + 2y"))
    m = method(r, "eigen")
    assert m["particular_latex"] is None and "C_{1}" in m["general_latex"]
    text = " ".join(l for s in m["steps"] for l in s["latex"])
    assert "\\lambda_1 = -1,\\qquad \\lambda_2 = 4" in text
    assert r["equilibria"] == [{"x": 0.0, "y": 0.0, "label": "saddle point"}]
    repeated = json.loads(system("x - y", "x + 3y"))
    assert "generalized eigenvector" in " ".join(s["title"] for s in method(repeated, "eigen")["steps"])


def test_nonlinear_system_linearization():
    r = json.loads(system("x(1 - y)", "y(x - 1)"))   # predator-prey
    assert [m["id"] for m in r["methods"] if m["applicable"]] == ["linearize"]
    assert r["equilibria"] == [{"x": 0.0, "y": 0.0, "label": "saddle point"},
                               {"x": 1.0, "y": 1.0, "label": "center"}]
    steps = method(r, "linearize")["steps"]
    assert steps[-1]["table"]["rows"][0][2] == "\\text{saddle point}"
    assert "do not settle" in steps[3]["text"]            # a center is not conclusive
    pendulum = json.loads(system("y", "-sin(x)"))
    assert {e["label"] for e in pendulum["equilibria"]} == {"center", "saddle point"}
    shifted = json.loads(system("x + 1", "y"))
    assert shifted["equilibria"][0]["x"] == -1.0 and "constant forcing" in shifted["summary"]


def test_system_errors_and_preview():
    assert "autonomous" in json.loads(system("t x", "y"))["error"]
    assert "both equations" in json.loads(system("x", ""))["error"]
    assert "x(0) and y(0)" in json.loads(system("x", "y", "1", ""))["error"]
    assert json.loads(preview("x + 2y;3x", "system"))["latex"] == "x' = x + 2 y,\\qquad y' = 3 x"
    assert not json.loads(preview("q;x", "system"))["ok"]


# ---- group 1: higher order, integration steps, reduction of order, phase line ----

def _rk4_system(deriv, state, t0, t1, h=0.0005):
    t = t0
    for _ in range(int(round((t1 - t0) / h))):
        k1 = deriv(t, state)
        k2 = deriv(t + h / 2, [a + h / 2 * b for a, b in zip(state, k1)])
        k3 = deriv(t + h / 2, [a + h / 2 * b for a, b in zip(state, k2)])
        k4 = deriv(t + h, [a + h * b for a, b in zip(state, k3)])
        state = [a + h / 6 * (p + 2 * q + 2 * u + w) for a, p, q, u, w in zip(state, k1, k2, k3, k4)]
        t += h
    return state


# (equation, x0, y0, "y', y'', ..." list, method id, x to compare at)
HIGHER = [
    ("y''' - 6y'' + 11y' - 6y = 0", "0", "1", "0, 0", "characteristic", 0.7),
    ("y'''' - y = 0", "0", "1", "0, 0, 0", "characteristic", 1.0),
    ("y'''' + 2y'' + y = 0", "0", "0", "1, 0, 0", "characteristic", 1.2),
    ("y''' + y' = sin(x)", "0", "0", "0, 1", "undetermined", 1.5),
    ("y''' - 3y'' + 3y' - y = e^x", "0", "1", "2, 0", "undetermined", 0.8),
    ("y''' - y' = x", "0", "0", "1, 0", "undetermined", 1.0),
    ("d3y/dx3 + 8y = 0", "0", "1", "0, 0", "characteristic", 0.9),
    ("2y''' + y'' = 4", "1", "0", "1, -1", "undetermined", 2.0),
]


@pytest.mark.parametrize("text,x0,y0,rest,method_id,x1", HIGHER)
def test_higher_order_ivp(text, x0, y0, rest, method_id, x1):
    r = json.loads(solve(text, x0, y0, rest))
    assert r["ok"], r.get("error")
    assert r["order"] in (3, 4)
    assert [m["id"] for m in r["methods"] if m["applicable"]] == [method_id]
    m = method(r, method_id)
    assert m["verified"] is True, m["verify_note"]
    from deengine.parsing import parse_equation
    from deengine.higher_order import analyse
    prob, lhs, rhs = parse_equation(text)
    eq = analyse(prob, lhs, rhs)
    g = sp.lambdify(prob.x, eq.g, "math")
    c = [float(v) for v in eq.c]
    n = eq.n

    def deriv(t, v):
        top = (g(t) - sum(c[k] * v[k] for k in range(n))) / c[n]
        return list(v[1:]) + [top]

    start = [float(sp.sympify(y0))] + [float(sp.sympify(p)) for p in rest.split(",")]
    reference = _rk4_system(deriv, start, float(x0), x1)[0]
    got = float(sp.sympify(m["plot"]["particular_expr"], locals={"x": X}).subs(X, x1))
    assert abs(got - reference) < 1e-6 * (1 + abs(reference)), (got, reference)
    assert m["plot"]["general_expr"] is None


def test_higher_order_general_and_errors():
    r = run("y''' - 3y'' + 3y' - y = e^x")
    m = method(r, "undetermined")
    assert "C_{3}" in m["general_latex"] and "\\frac{x^{3} e^{x}}{6}" in m["general_latex"]
    assert "multiplied" in next(s for s in m["steps"] if s["title"] == "Guess the particular solution")["text"]
    assert "Third-order" in r["summary"]
    assert "above fourth order" in run("y''''' + y = 0")["error"]
    assert "constant coefficients" in run("y''' + x y = 0")["error"]
    assert "needs 3 initial values" in json.loads(solve("y''' = 6", "1", "0", "1"))["error"]
    cubic = run("y''' + y' + 5y = 0")
    assert cubic["ok"] and not any(m["applicable"] for m in cubic["methods"])
    assert json.loads(preview("y''' - y = 0"))["order"] == 3


def test_integration_technique_is_explained():
    from deengine.integrals import explain
    titles = lambda expr: [s["title"] for s in explain(expr, X, "Integral")]
    assert titles(X * sp.exp(2 * X))[0] == "Integral: integration by parts"
    assert titles(X * sp.cos(X**2)) == ["Integral: substitution"]
    assert titles(1 / (X * (X - 1))) == ["Integral: partial fractions"]
    assert titles(sp.exp(X) * sp.sin(X)) == ["Integral: integration by parts, twice"]
    assert titles(sp.tan(X))[0] == "Integral: rewrite the integrand"
    assert titles(sp.Integer(3)) == [] and titles(sp.exp(-X**2)) == []
    parts = explain(X * sp.exp(2 * X), X, "Integral")[0]
    assert "u = x,\\qquad dv = e^{2 x}\\,dx" in parts["latex"]
    r = run("y' + y/x = cos(x)")
    subs = [s["title"] for st in method(r, "linear")["steps"] for s in st["substeps"]]
    assert "Right integral: integration by parts" in subs
    logistic = run("y' = y(1-y)")
    subs = [s["title"] for st in method(logistic, "separable")["steps"] for s in st["substeps"]]
    assert "Left integral: partial fractions" in subs
    vop = run("y'' + y = tan(x)")
    assert any(s["title"].startswith("Integral for u") for st in method(vop, "variation")["steps"]
               for s in st["substeps"])


@pytest.mark.parametrize("text,y0,y1,expected,x1", [
    ("x y'' - (x+1) y' + y = 0", "1", "0", {"reduction"}, 2.0),
    ("x^2 y'' - 2y = x", "1", "0", {"variation"}, 2.0),
    ("x y'' - (x+1) y' + y = x^2", "0", "1", {"variation"}, 2.0),
    ("x^2 y'' - 3x y' + 4y = x^2", "1", "2", {"variation"}, 1.8),
])
def test_reduction_of_order_and_variable_coefficient_variation(text, y0, y1, expected, x1):
    r = json.loads(solve(text, "1", y0, y1))
    assert r["ok"], r.get("error")
    assert {m["id"] for m in r["methods"] if m["applicable"]} == expected
    m = next(m for m in r["methods"] if m["applicable"])
    assert m["verified"] is True, m["verify_note"]
    reference = _second_reference(text, "1", y0, y1, x1)
    got = plot_fn(m["plot"]["particular_expr"])(x1)
    assert abs(got - reference) < 1e-6 * (1 + abs(reference)), (got, reference)


def test_reduction_steps_and_limits():
    m = method(run("x y'' - (x+1) y' + y = 0"), "reduction")
    assert [s["title"] for s in m["steps"]][:2] == ["Find one solution by inspection", "Reduce the order"]
    assert m["plot"]["general_expr"] == "C*exp(x) + C2*(x + 1)"
    assert method(run("y'' + y = 0"), "reduction")["applicable"] is False
    assert "none of the simple" in method(run("y'' - x y = 0"), "reduction")["reason"]
    euler = run("x^2 y'' - 2y = x")
    assert "variation of parameters" in method(euler, "cauchy_euler")["reason"]


def test_frobenius_logarithmic_second_solution():
    m = method(run("x^2 y'' + x y' + x^2 y = 0"), "series")   # Bessel order 0
    assert m["verified"] is True
    assert "Find the second solution with a logarithm" in [s["title"] for s in m["steps"]]
    C2 = sp.Symbol("C2")
    expr = sp.sympify(m["plot"]["general_expr"], locals={"x": X, "C": C, "C2": C2})
    for xv in (0.3, 0.6):
        j0, y0 = float(mpmath.besselj(0, xv)), float(mpmath.bessely(0, xv))
        assert abs(float(expr.subs({X: xv, C: 1, C2: 0})) - j0) < 1e-6
        # J0 ln x + series equals (pi/2) Y0 + (ln 2 - gamma) J0.
        want = math.pi / 2 * y0 + (math.log(2) - float(mpmath.euler)) * j0
        assert abs(float(expr.subs({X: xv, C: 0, C2: 1})) - want) < 1e-5
    integer_gap = method(run("x y'' + y = 0"), "series")
    line = next(s for s in integer_gap["steps"] if "logarithm" in s["title"])["latex"][0]
    assert line.startswith("y_2 = - y_1 \\ln x + 1 - \\frac{3 x^{2}}{4}")
    assert integer_gap["verified"] is True


def test_phase_line():
    r = run("y' = y(1-y)", "0", "1/2")
    m = method(r, "phase_line")
    assert m["applicable"] and m["verified"] is None and m["particular_latex"] is None
    table = next(s for s in m["steps"] if "table" in s)["table"]
    assert [row[1] for row in table["rows"]] == ["-", "+", "-"]
    text = " ".join(l for s in m["steps"] for l in s["latex"])
    assert "y = 0:\\ \\text{unstable},\\qquad y = 1:\\ \\text{stable}" in text
    assert "approaches \\(y = 1\\)" in m["steps"][-1]["text"]
    assert [s["title"] for s in m["steps"]][-1] == "Follow the initial value"   # no Check step
    semi = method(run("y' = y^2 (y - 2)"), "phase_line")
    assert "y = 0:\\ \\text{semi-stable},\\qquad y = 2:\\ \\text{unstable}" in " ".join(
        l for s in semi["steps"] for l in s["latex"])
    assert "without bound" in method(run("y' = 1 + y^2", "0", "0"), "phase_line")["steps"][-1]["text"]
    assert method(run("y' = x y"), "phase_line")["applicable"] is False
    assert method(run("y' = sin(y)"), "phase_line")["applicable"] is False
    assert "stays there" in method(run("y' = y(1-y)", "0", "1"), "phase_line")["steps"][-1]["text"]


# ---- group 2: Fourier series ----------------------------------------------

from deengine import boundary, eigen_problem, fourier


def _fx(text):
    return sp.lambdify(X, sp.sympify(text.replace("^", "**"), locals={"x": X, "e": sp.E}), "mpmath")


def _quad_coefficient(pieces, L, k, kind):
    """(1/L) * sum over pieces of the integral of f(x) cos or sin(k pi x / L)."""
    trig = mpmath.cos if kind == "a" else mpmath.sin
    return sum(mpmath.quad(lambda t: f(t) * trig(k * mpmath.pi * t / L), [lo, hi])
               for f, lo, hi in pieces) / L


def _partial_coefficients(expr_text):
    """{('a', 0): value, ('a', k): ..., ('b', k): ...} read from a partial-sum plot string."""
    out = {}
    for term in expr_text.split(" + "):
        if "*" not in term:
            out[("a", 0)] = 2 * float(term)       # the constant term is a_0 / 2
            continue
        coef, rest = term.split("*", 1)
        kind = "a" if rest.startswith("cos") else "b"
        out[(kind, float(rest[4:rest.index("*x")]))] = float(coef)
    return out


# (f on (0, L), f on (-L, 0) or "", L text, L value)
FOURIER = [
    ("x", "", "pi", math.pi),
    ("x^2", "", "1", 1.0),
    ("1", "", "2", 2.0),
    ("1", "-1", "pi", math.pi),          # square wave
    ("x(1-x)", "", "1", 1.0),
    ("e^x", "", "1", 1.0),
    ("x", "0", "pi", math.pi),           # half-wave ramp
]


@pytest.mark.parametrize("text,left,L_text,L", FOURIER)
def test_fourier_coefficients_match_quadrature(text, left, L_text, L):
    r = json.loads(fourier(text, left, L_text))
    assert r["ok"], r.get("error")
    assert r["kind"] == "fourier" and [m["id"] for m in r["methods"]] == ["full", "sine", "cosine"]
    f = _fx(text.replace("x(1-x)", "x*(1-x)"))
    g = _fx(left) if left else f
    setups = {"full": ([(g, -L, 0), (f, 0, L)], True, True, 1.0),
              "sine": ([(f, 0, L)], False, True, 2.0),
              "cosine": ([(f, 0, L)], True, False, 2.0)}
    for m in r["methods"]:
        if left and m["id"] != "full":
            assert not m["applicable"] and "left half" in m["reason"]
            continue
        assert m["applicable"] and m["verified"] is True, (m["id"], m.get("verify_note"))
        pieces, want_a, want_b, scale = setups[m["id"]]
        partials = m["plot"]["partials"]
        assert [p["n"] for p in partials] == [1, 2, 3, 5, 10, 20]
        got = _partial_coefficients(partials[3]["expr"])        # harmonics 1..5
        w = math.pi / L
        for k in range(0, 6):
            for kind, wanted in (("a", want_a), ("b", want_b)):
                if kind == "b" and k == 0:
                    continue
                reference = float(scale * _quad_coefficient(pieces, L, k, kind)) if wanted else 0.0
                key = (kind, 0) if k == 0 else (kind, round(k * w, 12))
                value = got.get(key, 0.0)
                assert abs(value - reference) < 1e-8, (m["id"], kind, k, value, reference)


def test_fourier_partial_sums_approach_the_function():
    r = json.loads(fourier("x(1-x)", "", "1"))
    sine = method(r, "sine")
    errors = []
    for p in sine["plot"]["partials"]:
        s = sp.lambdify(X, sp.sympify(p["expr"], locals={"x": X}), "math")
        errors.append(max(abs(s(t) - t * (1 - t)) for t in (0.1, 0.3, 0.5, 0.8)))
    assert errors[0] > errors[2] > errors[-1] and errors[-1] < 1e-4
    # The periodic extension agrees with f on the base interval and is odd.
    ext = sp.sympify(sine["plot"]["particular_expr"], locals={"x": X})
    assert abs(float(ext.subs(X, 0.3)) - 0.21) < 1e-12
    assert abs(float(ext.subs(X, -0.3)) + 0.21) < 1e-12
    assert abs(float(ext.subs(X, 2.3)) - 0.21) < 1e-12
    cosine_ext = sp.sympify(method(r, "cosine")["plot"]["particular_expr"], locals={"x": X})
    assert abs(float(cosine_ext.subs(X, -0.3)) - 0.21) < 1e-12


def test_fourier_symmetry_cases_and_text():
    odd = method(json.loads(fourier("x", "", "pi")), "full")
    text = " ".join(s["text"] for s in odd["steps"])
    assert "is odd" in text and "average of the left and right limits" in text
    assert "2 \\sin{\\left(x \\right)} - \\sin{\\left(2 x \\right)}" in odd["particular_latex"]
    even = method(json.loads(fourier("x^2", "", "1")), "full")
    assert "is even" in " ".join(s["text"] for s in even["steps"])
    square = method(json.loads(fourier("1", "-1", "pi")), "full")
    lines = " ".join(l for s in square["steps"] for l in s["latex"])
    assert "(n\\ \\text{even})" in lines and "(n\\ \\text{odd})" in lines
    assert "\\frac{4 \\sin{\\left(x \\right)}}{\\pi}" in square["particular_latex"]
    assert json.loads(fourier("1", "", "2"))["methods"][0]["particular_latex"] == "f(x) \\sim 1"


def test_fourier_errors_and_preview():
    assert "half-length L" in json.loads(fourier("x", "", ""))["error"]
    assert "positive" in json.loads(fourier("x", "", "0"))["error"]
    assert "positive" in json.loads(fourier("x", "", "-2"))["error"]
    assert "Unknown symbol 'q'" in json.loads(fourier("q", "", "1"))["error"]
    assert "Enter the function" in json.loads(fourier("", "", "1"))["error"]
    assert json.loads(preview("x;-x", "fourier"))["latex"] == "f(x) = x,\\qquad f(x)\\ (x<0) = - x"
    assert not json.loads(preview("x;q", "fourier"))["ok"]


# ---- group 2: boundary value and eigenvalue problems ----------------------

def _shoot(text, a, da, alpha, b, db, beta):
    """Solve the linear BVP by shooting: two RK4 runs from a, combined linearly."""
    from deengine.parsing import parse_equation
    from deengine.second_order import analyse
    prob, lhs, rhs = parse_equation(text)
    eq = analyse(prob, lhs, rhs)
    acc = sp.lambdify((prob.x, prob.y, prob.yp), (eq.g - eq.b * prob.yp - eq.c * prob.y) / eq.a, "math")

    def run(y, v, stop):
        t, h = a, (stop - a) / 4000
        for _ in range(4000):
            def d(tt, s):
                return [s[1], acc(tt, s[0], s[1])]
            st = [y, v]
            k1 = d(t, st)
            k2 = d(t + h / 2, [p + h / 2 * q for p, q in zip(st, k1)])
            k3 = d(t + h / 2, [p + h / 2 * q for p, q in zip(st, k2)])
            k4 = d(t + h, [p + h * q for p, q in zip(st, k3)])
            y, v = [p + h / 6 * (q1 + 2 * q2 + 2 * q3 + q4) for p, q1, q2, q3, q4 in zip(st, k1, k2, k3, k4)]
            t += h
        return y, v

    # Start values meeting the condition at a, with the other quantity 0 or 1.
    starts = [(0.0, alpha), (1.0, alpha)] if da else [(alpha, 0.0), (alpha, 1.0)]

    def end(start, stop):
        y, v = run(start[0], start[1], stop)
        return v if db else y

    e0, e1 = end(starts[0], b), end(starts[1], b)
    weight = (beta - e0) / (e1 - e0)
    start = (starts[0][0] + weight * (starts[1][0] - starts[0][0]),
             starts[0][1] + weight * (starts[1][1] - starts[0][1]))
    mid = a + 0.6 * (b - a)
    return mid, run(start[0], start[1], mid)[0]


# (equation, a, kind, alpha, b, kind, beta)
BVPS = [
    ("y'' + y = 0", "0", "y", "0", "1", "y", "1"),
    ("y'' - y = x", "0", "y", "0", "1", "dy", "0"),
    ("y'' = 6x", "0", "dy", "1", "1", "y", "0"),
    ("y'' + 3y' + 2y = e^x", "0", "y", "1", "1", "y", "0"),
    ("x^2 y'' + x y' - y = 0", "1", "y", "1", "2", "y", "2"),
    ("y'' + 4y = 0", "0", "dy", "1", "1", "dy", "-1"),
    ("x^2 y'' - 2y = x", "1", "y", "0", "2", "y", "1"),
]


@pytest.mark.parametrize("text,a,ka,va,b,kb,vb", BVPS)
def test_boundary_value_problem_matches_shooting(text, a, ka, va, b, kb, vb):
    r = json.loads(boundary(text, a, ka, va, b, kb, vb))
    assert r["ok"], r.get("error")
    assert r["kind"] == "boundary" and "exactly one solution" in r["summary"]
    m = r["methods"][0]
    assert m["verified"] is True and m["steps"][-1]["title"] == "Check"
    expr = sp.sympify(m["plot"]["particular_expr"], locals={"x": X})
    mid, reference = _shoot(text, float(a), ka == "dy", float(va), float(b), kb == "dy", float(vb))
    assert abs(float(expr.subs(X, mid)) - reference) < 1e-6 * (1 + abs(reference))


def test_boundary_solvability_outcomes():
    one = json.loads(boundary("y'' + y = 0", "0", "y", "0", "pi/2", "y", "1"))
    assert one["methods"][0]["plot"]["particular_expr"] == "sin(x)"
    many = json.loads(boundary("y'' + y = 0", "0", "y", "0", "pi", "y", "0"))
    assert "infinitely many" in many["summary"]
    assert many["methods"][0]["plot"]["general_expr"] == "C*sin(x)"
    assert many["methods"][0]["particular_latex"] is None
    none = json.loads(boundary("y'' + y = 0", "0", "y", "0", "pi", "y", "1"))
    assert "no solution" in none["summary"]
    m = none["methods"][0]
    assert m["plot"]["particular_expr"] is None and m["plot"]["general_expr"] is None
    assert m["steps"][-1]["title"] == "Result" and "No solution" in [s["title"] for s in m["steps"]]


def test_boundary_errors():
    assert "second-order" in json.loads(boundary("y' = y", "0", "y", "0", "1", "y", "0"))["error"]
    assert "must be different" in json.loads(boundary("y'' + y = 0", "0", "y", "0", "0", "y", "1"))["error"]
    assert "missing" in json.loads(boundary("y'' + y = 0", "0", "y", "", "1", "y", "1"))["error"]
    assert "closed form" in json.loads(boundary("y'' + sin(x) y = 0", "0", "y", "0", "1", "y", "1"))["error"]
    assert "not defined" in json.loads(boundary("x^2 y'' + x y' - y = 0", "0", "y", "0", "1", "y", "1"))["error"]
    assert "condition on y" in json.loads(boundary("y'' + y = 0", "0", "z", "0", "1", "y", "1"))["error"]


@pytest.mark.parametrize("kind,L_text,L", [("dd", "pi", math.pi), ("nn", "2", 2.0),
                                          ("dn", "1", 1.0), ("nd", "3", 3.0)])
def test_eigenvalue_problem(kind, L_text, L):
    r = json.loads(eigen_problem(kind, L_text))
    assert r["ok"] and r["kind"] == "eigen"
    m = r["methods"][0]
    assert m["verified"] is True
    titles = [s["title"] for s in m["steps"]]
    assert titles[1:4] == ["Case λ < 0", "Case λ = 0", "Case λ > 0"]
    table = next(s for s in m["steps"] if "table" in s)["table"]
    assert len(table["rows"]) == 5 and len(m["plot"]["partials"]) == 5
    start_value, end_value = kind[0] == "d", kind[1] == "d"
    for p in m["plot"]["partials"]:
        k = p["n"]
        fn = sp.sympify(p["expr"], locals={"x": X})
        if kind in ("dd", "nn"):
            lam = (k * math.pi / L) ** 2
        else:
            lam = ((2 * k - 1) * math.pi / (2 * L)) ** 2
        # Each eigenfunction satisfies the equation and both conditions.
        assert abs(float((fn.diff(X, 2) + lam * fn).subs(X, 0.37 * L))) < 1e-9
        assert abs(float((fn if start_value else fn.diff(X)).subs(X, 0))) < 1e-12
        assert abs(float((fn if end_value else fn.diff(X)).subs(X, L))) < 1e-9
    assert (m["plot"]["partials"][0]["n"] == 0) == (kind == "nn")
    # Orthogonality of the first two.
    f0, f1 = (sp.lambdify(X, sp.sympify(p["expr"], locals={"x": X}), "mpmath") for p in m["plot"]["partials"][:2])
    assert abs(mpmath.quad(lambda t: f0(t) * f1(t), [0, L])) < 1e-10


def test_eigenvalue_problem_errors():
    assert "four pairs" in json.loads(eigen_problem("xx", "1"))["error"]
    assert "length L" in json.loads(eigen_problem("dd", ""))["error"]
    assert "positive" in json.loads(eigen_problem("dd", "-1"))["error"]


# ---- group 3: forced and 3x3 systems --------------------------------------

def _system_rk4(exprs, start, t_end, h=0.0005):
    """Hand RK4 for a system given as plot-style strings in x, y, z and t."""
    Z, Tt = sp.Symbol("z"), sp.Symbol("t")
    fns = [sp.lambdify((Tt, X, Y, Z), sp.sympify(e, locals={"x": X, "y": Y, "z": Z, "t": Tt, "e": sp.E}), "math")
           for e in exprs]
    state = list(start) + [0.0] * (3 - len(start))
    t = 0.0

    def d(tt, v):
        out = [fn(tt, v[0], v[1], v[2]) for fn in fns]
        return out + [0.0] * (3 - len(out))
    for _ in range(int(round(t_end / h))):
        k1 = d(t, state)
        k2 = d(t + h / 2, [a + h / 2 * b for a, b in zip(state, k1)])
        k3 = d(t + h / 2, [a + h / 2 * b for a, b in zip(state, k2)])
        k4 = d(t + h, [a + h * b for a, b in zip(state, k3)])
        state = [a + h / 6 * (p + 2 * q + 2 * u + w) for a, p, q, u, w in zip(state, k1, k2, k3, k4)]
        t += h
    return state


def _sys_at(m, key, t_value):
    return float(sp.sympify(m["plot"][key], locals={"x": X}).subs(X, t_value))


# (f, g, python forms of f and g, x0, y0, applicable ids)
FORCED_SYSTEMS = [
    ("x + 2y + 1", "3x + 2y - 2", "x + 2*y + 1", "3*x + 2*y - 2", "1", "0", ["shift", "variation"]),
    ("y", "-x + 1", "y", "-x + 1", "2", "0", ["shift", "variation"]),
    ("x + 2y + e^t", "3x + 2y", "x + 2*y + exp(t)", "3*x + 2*y", "0", "0", ["variation"]),
    ("y + t", "-x", "y + t", "-x", "1", "0", ["variation"]),
    ("-x + sin(t)", "-2y + t^2", "-x + sin(t)", "-2*y + t**2", "1", "-1", ["variation"]),
    ("x + 2y + 1", "2x + 4y", "x + 2*y + 1", "2*x + 4*y", "0", "1", ["variation"]),
]


@pytest.mark.parametrize("f,g,pf,pg,x0,y0,ids", FORCED_SYSTEMS)
def test_forced_system_matches_numerical_solution(f, g, pf, pg, x0, y0, ids):
    r = json.loads(system(f, g, x0, y0))
    assert r["ok"], r.get("error")
    assert [m["id"] for m in r["methods"] if m["applicable"]] == ids
    reference = _system_rk4([pf, pg], [float(x0), float(y0)], 0.7)
    for m in r["methods"]:
        if not m["applicable"]:
            continue
        assert m["verified"] is True, m["verify_note"]
        assert abs(_sys_at(m, "x_expr", 0.7) - reference[0]) < 1e-7 * (1 + abs(reference[0]))
        assert abs(_sys_at(m, "y_expr", 0.7) - reference[1]) < 1e-7 * (1 + abs(reference[1]))
    time_dependent = "t" in pf or "t" in pg
    assert (r["field"] is None) == time_dependent


def test_forced_system_general_solution_and_marks():
    r = json.loads(system("x + 2y + 1", "3x + 2y - 2"))
    assert r["equilibria"] == [{"x": 1.5, "y": -1.25, "label": "saddle point"}]
    for m in r["methods"][:2]:
        assert m["verified"] is True and "C_{1}" in m["general_latex"] and m["particular_latex"] is None
    assert "Shift back" in [s["title"] for s in method(r, "shift")["steps"]]
    forced = json.loads(system("-x + sin(t)", "-2y + t^2"))
    assert method(forced, "variation")["verified"] is True
    assert "depends on" in method(forced, "shift")["reason"]
    assert "linear in x and y" in json.loads(system("x y + t", "y"))["error"]


# (f, g, h, python forms, start, defective?)
THREE_SYSTEMS = [
    ("x + y", "y + z", "2z", ["x + y", "y + z", "2*z"], ["1", "0", "1"], True),
    ("y", "-x", "-z", ["y", "-x", "-z"], ["1", "0", "2"], False),
    ("2x", "2y", "3z", ["2*x", "2*y", "3*z"], ["1", "1", "1"], False),
    ("x + y", "y + z", "z", ["x + y", "y + z", "z"], ["1", "1", "1"], True),
    # Eigenvalues -1, -1, 4 with a single eigenvector for -1.
    ("x + 2y", "3x + 2y", "-z + x", ["x + 2*y", "3*x + 2*y", "-z + x"], ["1", "0", "0"], True),
    ("x + 2y", "3x + 2y", "2z + x", ["x + 2*y", "3*x + 2*y", "2*z + x"], ["1", "0", "0"], False),
]


@pytest.mark.parametrize("f,g,h,py,start,defective", THREE_SYSTEMS)
def test_three_by_three_matches_numerical_solution(f, g, h, py, start, defective):
    r = json.loads(system(f, g, start[0], start[1], h, start[2]))
    assert r["ok"], r.get("error")
    assert r["field"] is None and r["ic"]["z0"] == float(start[2])
    assert [m["id"] for m in r["methods"] if m["applicable"]] == ["eigen", "expm"]
    reference = _system_rk4(py, [float(v) for v in start], 0.6)
    for m in r["methods"]:
        assert m["verified"] is True, m["verify_note"]
        for key, want in zip(("x_expr", "y_expr", "z_expr"), reference):
            assert abs(_sys_at(m, key, 0.6) - want) < 1e-7 * (1 + abs(want)), (m["id"], key)
    titles = [s["title"] for s in method(r, "eigen")["steps"]]
    assert ("Handle the missing eigenvector" in titles) == defective


def test_three_by_three_general_and_errors():
    r = json.loads(system("y", "-x", "", "", "-z", ""))
    m = method(r, "eigen")
    assert all(c in m["general_latex"] for c in ("C_{1}", "C_{2}", "C_{3}", "z(t)"))
    assert "zero real part" in r["summary"]
    assert "linear and homogeneous" in json.loads(system("x + y", "y", "", "", "x z", ""))["error"]
    cubic = json.loads(system("y", "z", "", "", "-x - y", ""))
    assert cubic["ok"] and not any(m["applicable"] for m in cubic["methods"])
    assert "x(0) and y(0) and z(0)" in json.loads(system("x", "y", "1", "", "z", ""))["error"]
    assert json.loads(preview("x;y;z+t", "system"))["latex"].endswith("z' = t + z")
    assert "asymptotically stable (every" in json.loads(system("-x", "-2y", "", "", "-3z", ""))["summary"]


# ---- PDEs: heat, wave, Laplace --------------------------------------------

from deengine import pde


def _pde_fn(expr_text):
    """u(x, y) from a plot string, where y is time (or the second space variable)."""
    step = lambda v, *rest: 0.0 if v < 0 else 1.0
    expr = sp.sympify(expr_text, locals={"x": X, "y": Y, "Heaviside": sp.Heaviside, "floor": sp.floor})
    fn = sp.lambdify((X, Y), expr, [{"Heaviside": step, "floor": math.floor}, "math"])
    return lambda a, b: float(fn(a, b))


def _pde(*args):
    r = json.loads(pde(*args))
    assert r["ok"], r.get("error")
    assert r["kind"] == "pde" and r["rhs_expr"] is None
    return r


def test_heat_single_mode_is_exact():
    r = _pde("heat", "sin(pi x)", "", "1", "1", "00", "", "")
    m = method(r, "separation")
    assert m["verified"] is True
    u = _pde_fn(m["plot"]["particular_expr"])
    for xv, tv in ((0.3, 0.05), (0.5, 0.2), (0.8, 0.01)):
        assert abs(u(xv, tv) - math.exp(-math.pi ** 2 * tv) * math.sin(math.pi * xv)) < 1e-9
    assert "\\approx" not in m["particular_latex"]
    assert abs(r["domain"]["t_max"] - math.log(50) / math.pi ** 2) < 1e-9
    assert r["initial_expr"] == "sin(pi*x)" and r["surface"] is False


def _heat_residual(u, k, L, t_max):
    worst = 0.0
    h, dt = L * 1e-3, t_max * 1e-4
    for xv in (0.2 * L, 0.55 * L, 0.9 * L):
        for tv in (0.1 * t_max, 0.6 * t_max):
            ut = (u(xv, tv + dt) - u(xv, tv - dt)) / (2 * dt)
            uxx = (u(xv + h, tv) - 2 * u(xv, tv) + u(xv - h, tv)) / (h * h)
            worst = max(worst, abs(ut - k * uxx))
    return worst


@pytest.mark.parametrize("f,L,k,bc,t1,t2", [
    ("x(1-x)", "1", "1/2", "00", "", ""),
    ("x", "pi", "1", "nn", "", ""),
    ("1", "2", "1", "0n", "", ""),
    ("x", "1", "1", "n0", "", ""),
    ("0", "1", "1", "tt", "0", "100"),
    ("sin(pi x)", "1", "1", "tt", "1", "3"),
])
def test_heat_equation_and_boundary_conditions(f, L, k, bc, t1, t2):
    r = _pde("heat", f, "", L, k, bc, t1, t2)
    m = method(r, "separation")
    assert m["verified"] is True, m["verify_note"]
    u = _pde_fn(m["plot"]["particular_expr"])
    Lf, kf, t_max = float(sp.sympify(L)), float(sp.sympify(k)), r["domain"]["t_max"]
    scale = r["domain"]["u_max"] - r["domain"]["u_min"]
    assert _heat_residual(u, kf, Lf, t_max) < 1e-3 * scale / t_max
    tv, h = 0.4 * t_max, 1e-5 * Lf
    left = (u(h, tv) - u(0.0, tv)) / h if bc in ("nn", "n0") else u(0.0, tv) - (float(t1) if bc == "tt" else 0.0)
    right = (u(Lf, tv) - u(Lf - h, tv)) / h if bc in ("nn", "0n") else u(Lf, tv) - (float(t2) if bc == "tt" else 0.0)
    assert abs(left) < 1e-3 * scale and abs(right) < 1e-3 * scale
    # The initial data is reproduced away from the ends.
    f0 = sp.lambdify(X, sp.sympify(r["initial_expr"], locals={"x": X}), "math")
    assert max(abs(u(Lf * i / 10, 0.0) - f0(Lf * i / 10)) for i in range(3, 8)) < 0.05 * scale
    assert [s["title"] for s in m["steps"]][-2:] == ["Solution", "Check"]


def test_heat_long_time_limits():
    insulated = _pde("heat", "x", "", "pi", "1", "nn", "", "")
    u = _pde_fn(method(insulated, "separation")["plot"]["particular_expr"])
    assert abs(u(1.0, 50.0) - math.pi / 2) < 1e-9          # the mean of f
    assert "average" in " ".join(s["text"] for s in method(insulated, "separation")["steps"])
    held = _pde("heat", "0", "", "1", "1", "tt", "0", "100")
    u = _pde_fn(method(held, "separation")["plot"]["particular_expr"])
    assert abs(u(0.3, 50.0) - 30.0) < 1e-9                 # the steady state 100 x
    assert "Subtract the steady state" == method(held, "separation")["steps"][0]["title"]
    quarter = _pde("heat", "1", "", "2", "1", "0n", "", "")
    assert "2n-1" in " ".join(l for s in method(quarter, "separation")["steps"] for l in s["latex"])


def test_heat_coefficient_formula_in_steps():
    r = _pde("heat", "x(1-x)", "", "1", "1", "00", "", "")
    step_ = next(s for s in method(r, "separation")["steps"] if s["title"] == "Find the coefficients")
    assert any("b_n = " in l and "\\int" not in l for l in step_["latex"])
    table = next(s for s in step_["substeps"] if "table" in s)["table"]
    assert table["rows"][0][0] == "1" and abs(float(table["rows"][0][1]) - 8 / math.pi ** 3) < 1e-5


def test_wave_standing_mode_and_period():
    r = _pde("wave", "sin(pi x)", "", "1", "2", "", "", "")
    assert [m["id"] for m in r["methods"]] == ["separation", "dalembert"]
    assert abs(r["domain"]["t_max"] - 2.0) < 1e-12 and r["domain"]["u_min"] == -1.0
    for m in r["methods"]:
        assert m["verified"] is True
        u = _pde_fn(m["plot"]["particular_expr"])
        for xv, tv in ((0.3, 0.1), (0.7, 0.45), (0.5, 1.3)):
            assert abs(u(xv, tv) - math.cos(2 * math.pi * tv) * math.sin(math.pi * xv)) < 1e-9, m["id"]


def _plucked(xv, tv):
    """Exact plucked-string solution (L = 1, c = 1) from the odd periodic extension."""
    def tri(s):
        s = (s + 1) % 2 - 1
        sign = -1.0 if s < 0 else 1.0
        s = abs(s)
        return sign * (s if s < 0.5 else 1 - s)
    return (tri(xv - tv) + tri(xv + tv)) / 2


def test_plucked_string_matches_dalembert():
    r = _pde("wave", "x (1 - u(x - 1/2)) + (1 - x) u(x - 1/2)", "", "1", "1", "", "", "")
    series = _pde_fn(method(r, "separation")["plot"]["particular_expr"])
    travelling = _pde_fn(method(r, "dalembert")["plot"]["particular_expr"])
    for xv, tv in ((0.2, 0.1), (0.5, 0.3), (0.75, 0.9), (0.4, 1.7), (0.6, 2.4)):
        exact = _plucked(xv, tv)
        assert abs(travelling(xv, tv) - exact) < 1e-12
        assert abs(series(xv, tv) - exact) < 0.02
    assert abs(travelling(0.3, 2.0) - travelling(0.3, 0.0)) < 1e-12   # period 2L/c


def test_wave_with_initial_velocity():
    r = _pde("wave", "", "sin(pi x)", "1", "1", "", "", "")
    for m in r["methods"]:
        assert m["applicable"] and m["verified"] is True, m["id"]
        u = _pde_fn(m["plot"]["particular_expr"])
        for xv, tv in ((0.3, 0.2), (0.6, 0.7), (0.5, 1.4)):
            assert abs(u(xv, tv) - math.sin(math.pi * tv) * math.sin(math.pi * xv) / math.pi) < 1e-9, m["id"]
    both = _pde("wave", "x(1-x)", "x", "1", "1", "", "", "")
    a = _pde_fn(method(both, "separation")["plot"]["particular_expr"])
    b = _pde_fn(method(both, "dalembert")["plot"]["particular_expr"])
    assert max(abs(a(xv, tv) - b(xv, tv)) for xv in (0.2, 0.5, 0.8) for tv in (0.3, 1.1)) < 0.02
    h = 1e-5
    assert abs((b(0.4, h) - b(0.4, 0.0)) / h - 0.4) < 1e-3          # u_t(x, 0) = g(x) = x
    assert abs(b(0.0, 0.7)) < 1e-12 and abs(b(1.0, 0.7)) < 1e-9


def test_laplace_single_mode_and_sides():
    r = _pde("laplace", "sin(pi x)", "", "1", "1", "top", "", "")
    m = method(r, "separation")
    assert m["verified"] is True and r["surface"] is True and r["initial_expr"] is None
    u = _pde_fn(m["plot"]["particular_expr"])
    for xv, yv in ((0.3, 0.4), (0.5, 0.9), (0.8, 0.1)):
        exact = math.sinh(math.pi * yv) / math.sinh(math.pi) * math.sin(math.pi * xv)
        assert abs(u(xv, yv) - exact) < 1e-9
    assert r["domain"]["y_max"] == 1.0 and r["domain"]["x_max"] == 1.0


@pytest.mark.parametrize("f,a,b,side", [
    ("x(2-x)", "2", "1", "bottom"), ("y(1-y)", "1", "1", "left"),
    ("1", "1", "3", "right"), ("x", "1", "2", "top"), ("sin(pi y)", "30", "1", "right"),
])
def test_laplace_harmonic_and_bounded(f, a, b, side):
    r = _pde("laplace", f, "", a, b, side, "", "")
    m = method(r, "separation")
    assert m["verified"] is True, m["verify_note"]
    u = _pde_fn(m["plot"]["particular_expr"])
    af, bf = float(a), float(b)
    h = min(af, bf) * 1e-3
    xv, yv = 0.37 * af, 0.58 * bf
    mean = (u(xv + h, yv) + u(xv - h, yv) + u(xv, yv + h) + u(xv, yv - h)) / 4
    scale = r["domain"]["u_max"] - r["domain"]["u_min"]
    assert abs(mean - u(xv, yv)) < 1e-5 * scale                      # mean-value property
    data = sp.lambdify(sp.Symbol("v"), sp.sympify(f.replace("^", "**").replace("(2-x)", "*(2-x)")
                                                  .replace("(1-y)", "*(1-y)").replace("pi y", "pi*y")
                                                  .replace("x", "v").replace("y", "v")), "math")
    span = af if side in ("top", "bottom") else bf
    values = [data(span * i / 50) for i in range(51)] + [0.0]
    inside = [u(af * i / 9, bf * j / 9) for i in range(1, 9) for j in range(1, 9)]
    assert min(inside) > min(values) - 0.05 * scale and max(inside) < max(values) + 0.05 * scale   # maximum principle
    assert all(math.isfinite(v) for v in inside)
    # The three other sides are at 0.
    zero = {"top": [(0.0, 0.5), (1.0, 0.5), (0.5, 0.0)], "bottom": [(0.0, 0.5), (1.0, 0.5), (0.5, 1.0)],
            "left": [(0.5, 0.0), (0.5, 1.0), (1.0, 0.5)], "right": [(0.5, 0.0), (0.5, 1.0), (0.0, 0.5)]}[side]
    assert max(abs(u(af * px, bf * py)) for px, py in zero) < 1e-6 * max(scale, 1.0)


@pytest.mark.parametrize("args,fragment", [
    (("heat", "", "", "1", "1", "00", "", ""), "initial temperature"),
    (("heat", "x", "", "0", "1", "00", "", ""), "length L must be a positive"),
    (("heat", "x", "", "1", "-1", "00", "", ""), "diffusivity k must be a positive"),
    (("heat", "x", "", "", "1", "00", "", ""), "Enter the length L"),
    (("heat", "q", "", "1", "1", "00", "", ""), "Unknown symbol"),
    (("heat", "x", "", "1", "1", "tt", "1", ""), "both end temperatures"),
    (("heat", "x", "", "1", "1", "zz", "", ""), "boundary conditions"),
    (("heat", "delta(x-1)", "", "2", "1", "00", "", ""), "impulse"),
    (("wave", "", "", "1", "1", "", "", ""), "initial displacement"),
    (("wave", "x", "", "1", "0", "", "", ""), "wave speed c must be a positive"),
    (("laplace", "x", "", "1", "1", "up", "", ""), "which side"),
    (("laplace", "y", "", "1", "1", "top", "", ""), "Unknown symbol"),
    (("foo", "x", "", "1", "1", "", "", ""), "heat, wave or Laplace"),
])
def test_pde_errors(args, fragment):
    r = json.loads(pde(*args))
    assert not r["ok"] and fragment in r["error"]


def test_pde_preview_and_speed():
    import time
    assert json.loads(preview("x(1-x);sin(x)", "pde"))["latex"].startswith("f(x) = ")
    assert json.loads(preview("y^2", "pdey"))["latex"] == "f(y) = y^{2}"
    assert not json.loads(preview("y^2", "pde"))["ok"]
    start = time.time()
    _pde("heat", "x(1-x)", "", "1", "1", "00", "", "")
    _pde("wave", "x(1-x)", "x", "1", "1", "", "", "")
    _pde("laplace", "x(2-x)", "", "2", "1", "bottom", "", "")
    assert time.time() - start < 6


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
    with open(os.path.join(out, "inverse_laplace.json"), "w") as fh:
        fh.write(transform("e^(-2s)/(s^2+1)", "inverse"))
    with open(os.path.join(out, "system_linear.json"), "w") as fh:
        fh.write(system("-x - 2y", "2x - y", "1", "1"))
    with open(os.path.join(out, "system_nonlinear.json"), "w") as fh:
        fh.write(system("x(1 - y)", "y(x - 1)"))
    with open(os.path.join(out, "system_forced.json"), "w") as fh:
        fh.write(system("y + t", "-x", "1", "0"))
    with open(os.path.join(out, "system_three.json"), "w") as fh:
        fh.write(system("y", "-x", "1", "0", "-z", "2"))
    with open(os.path.join(out, "pde_heat.json"), "w") as fh:
        fh.write(pde("heat", "x(1-x)", "", "1", "1", "00", "", ""))
    with open(os.path.join(out, "pde_wave.json"), "w") as fh:
        fh.write(pde("wave", "sin(pi x)", "", "1", "2", "", "", ""))
    with open(os.path.join(out, "pde_laplace.json"), "w") as fh:
        fh.write(pde("laplace", "sin(pi x)", "", "1", "2", "top", "", ""))
    with open(os.path.join(out, "fourier.json"), "w") as fh:
        fh.write(fourier("x", "", "pi"))
    with open(os.path.join(out, "boundary.json"), "w") as fh:
        fh.write(boundary("y'' + y = 0", "0", "y", "0", "pi/2", "y", "1"))
    with open(os.path.join(out, "eigen.json"), "w") as fh:
        fh.write(eigen_problem("dd", "pi"))
