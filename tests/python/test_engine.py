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
    (("foo", "x", "", "1", "1", "", "", ""), "kind of problem"),
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


# ---- second PDE patch: characteristics, classification, finite differences, disk ----

def _pde_err(*args):
    return json.loads(pde(*args))


def _uxt(expr_text):
    """u(x, t) from a plot expression (y stands for t)."""
    return sp.lambdify((X, Y), sp.sympify(expr_text, locals={"x": X, "y": Y}), "math")


@pytest.mark.parametrize("a,b,c,d,f,exact", [
    ("2", "1", "", "", "e^(-x^2)", lambda x, t: math.exp(-(x - 2 * t) ** 2)),
    ("1", "1", "-1", "", "sin(x)", lambda x, t: math.exp(-t) * math.sin(x - t)),
    ("x", "1", "", "", "1/(1+x^2)", lambda x, t: 1 / (1 + (x * math.exp(-t)) ** 2)),
    ("t", "1", "", "", "cos(x)", lambda x, t: math.cos(x - t * t / 2)),
    ("3", "2", "", "1", "x", lambda x, t: (x - 1.5 * t) + t / 2),
    ("-1", "1", "2", "", "1/(1+x^2)", lambda x, t: math.exp(2 * t) / (1 + (x + t) ** 2)),
])
def test_characteristics_match_exact_solution_and_pde(a, b, c, d, f, exact):
    r = _pde("characteristics", f, "", a, b, "", c, d)
    assert r["ok"], r.get("error")
    assert r["pde"] == "characteristics"
    m = r["methods"][0]
    assert m["verified"] is True, m["verify_note"]
    u = _uxt(m["plot"]["particular_expr"])
    coef = [sp.lambdify((X, sp.Symbol("t")), sp.sympify(text or "0", locals={"x": X}), "math")
            for text in (a, b, c, d)]
    h = 1e-5
    for x, t in ((0.3, 0.2), (-1.1, 0.7), (0.9, 1.3)):
        assert abs(u(x, t) - exact(x, t)) < 1e-9
        ux = (u(x + h, t) - u(x - h, t)) / (2 * h)
        ut = (u(x, t + h) - u(x, t - h)) / (2 * h)
        av, bv, cv, dv = (fn(x, t) for fn in coef)
        assert abs(av * ux + bv * ut - cv * u(x, t) - dv) < 1e-6
    assert abs(u(0.4, 0.0) - exact(0.4, 0.0)) < 1e-12
    titles = [s["title"] for s in m["steps"]]
    assert titles[:2] == ["Write the characteristic equations", "Solve for the characteristic curves"]
    assert r["domain"]["t_max"] > 0 and r["initial_expr"]


def test_characteristics_general_solution_and_errors():
    r = _pde("characteristics", "", "", "3", "1", "", "", "")
    m = r["methods"][0]
    assert "F{\\left(- 3 t + x \\right)}" in m["particular_latex"]
    assert m["verified"] is True and m["plot"]["particular_expr"] is None
    degenerate = _pde("characteristics", "x", "", "x", "t", "", "1", "")
    assert "t F{\\left(\\frac{x}{t} \\right)}" in degenerate["methods"][0]["particular_latex"]
    assert "vanishes at t = 0" in degenerate["summary"]
    assert "cannot be 0" in _pde_err("characteristics", "x", "", "1", "0", "", "", "")["error"]
    assert "coefficient a" in _pde_err("characteristics", "x", "", "", "1", "", "", "")["error"]
    assert "Unknown symbol" in _pde_err("characteristics", "x", "", "y", "1", "", "", "")["error"]


@pytest.mark.parametrize("A,B,C,kind,model", [
    ("1", "0", "-4", "hyperbolic", "wave"),
    ("1", "0", "0", "parabolic", "heat"),
    ("1", "0", "1", "elliptic", "Laplace"),
    ("1", "2", "1", "parabolic", "heat"),
    ("1", "4", "1", "hyperbolic", "wave"),
    ("2", "1", "3", "elliptic", "Laplace"),
    ("0", "1", "0", "hyperbolic", "wave"),
])
def test_classification_constant_coefficients(A, B, C, kind, model):
    r = _pde("classify", "", "", A, B, "", C, "")
    assert r["ok"], r.get("error")
    assert ("is %s" % kind) in r["summary"] and model in r["summary"]
    m = r["methods"][0]
    assert m["particular_latex"] == "\\text{%s}" % kind
    assert [s["title"] for s in m["steps"]] == [
        "Compute the discriminant", "Read off the type", "Find the characteristics",
        "Canonical form", "Type"]
    disc = sp.sympify(B) ** 2 - 4 * sp.sympify(A) * sp.sympify(C)
    assert ("B^2 - 4AC = %s" % sp.latex(disc)) in m["steps"][0]["latex"][1]


def test_classification_characteristics_and_mixed_type():
    wave = _pde("classify", "", "", "1", "0", "", "-4", "")["methods"][0]
    lines = " ".join(wave["steps"][2]["latex"])
    assert "y - 2 x = \\text{const}" in lines and "y + 2 x = \\text{const}" in lines
    mixed = _pde("classify", "", "", "0", "1", "", "0", "")["methods"][0]
    assert "x = \\text{const}" in " ".join(mixed["steps"][2]["latex"])
    tricomi = _pde("classify", "", "", "y", "0", "", "1", "")
    assert "mixed type" in tricomi["summary"]
    text = " ".join(tricomi["methods"][0]["steps"][1]["latex"])
    assert "\\text{hyperbolic where } y < 0" in text
    assert "\\text{parabolic where } y = 0" in text
    assert "\\text{elliptic where } 0 < y" in text
    laplace = _pde("classify", "", "", "1", "0", "", "1", "")["methods"][0]
    assert "no real characteristics" in laplace["steps"][2]["text"]
    assert "at least one" in _pde_err("classify", "", "", "", "", "", "", "")["error"]


def _fd(f, steps, L, k, n, dt):
    r = _pde("fd", f, steps, L, k, "", n, dt)
    assert r["ok"], r.get("error")
    return r


def test_finite_differences_first_level_and_accuracy():
    r = _fd("sin(pi x)", "10", "1", "1", "4", "0.02")
    assert [m["id"] for m in r["methods"]] == ["ftcs", "cn"]
    ftcs = r["methods"][0]
    levels = ftcs["plot"]["levels"]
    assert len(levels) == 11 and levels[0]["points"][0] == [0.0, 0.0]
    # By hand: dx = 0.25, r = 0.32, u_1^1 = s + r (1 - 2 s + 0) with s = sin(pi/4).
    s = math.sin(math.pi / 4)
    assert abs(levels[1]["points"][1][1] - (s + 0.32 * (1 - 2 * s))) < 1e-12
    assert abs(levels[1]["t"] - 0.02) < 1e-12
    exact = lambda x, t: math.exp(-math.pi ** 2 * t) * math.sin(math.pi * x)
    for m in r["methods"]:
        assert m["verified"] is True
        for x, u in m["plot"]["levels"][-1]["points"]:
            assert abs(u - exact(x, 0.2)) < 0.02
        table = next(st for st in m["steps"] if "table" in st)["table"]
        assert all(len(row) == len(table["headers"]) for row in table["rows"])


def test_finite_differences_converge_and_stability():
    def error(scheme, n, dt, steps):
        r = _fd("sin(pi x)", str(steps), "1", "1", str(n), dt)
        m = next(mm for mm in r["methods"] if mm["id"] == scheme)
        t_end = m["plot"]["levels"][-1]["t"]
        return max(abs(u - math.exp(-math.pi ** 2 * t_end) * math.sin(math.pi * x))
                   for x, u in m["plot"]["levels"][-1]["points"])
    # Same final time 0.1, mesh ratio 0.4 on both grids.
    assert error("ftcs", 8, "0.00625", 16) < error("ftcs", 4, "0.025", 4) / 3
    assert error("cn", 8, "0.00625", 16) < error("cn", 4, "0.025", 4) / 3
    unstable = _fd("sin(pi x) + 0.1 sin(3 pi x)", "60", "1", "1", "10", "0.01")   # r = 1
    assert "unstable" in unstable["summary"]
    ftcs, cn = unstable["methods"]
    assert ftcs["verified"] is None and "unstable" in ftcs["steps"][2]["text"]
    assert cn["verified"] is True
    final = cn["plot"]["levels"][-1]["points"]
    assert max(abs(u) for _, u in final) < 0.01          # decayed, not blown up


def test_finite_difference_errors():
    assert "number of intervals" in _pde_err("fd", "x", "5", "1", "1", "", "", "0.1")["error"]
    assert "between 2 and 40" in _pde_err("fd", "x", "5", "1", "1", "", "100", "0.1")["error"]
    assert "time step" in _pde_err("fd", "x", "5", "1", "1", "", "4", "")["error"]
    assert "number of time steps" in _pde_err("fd", "x", "", "1", "1", "", "4", "0.1")["error"]
    assert "positive" in _pde_err("fd", "x", "5", "1", "1", "", "4", "-1")["error"]


def _uxy(expr_text):
    return sp.lambdify((X, Y), sp.sympify(expr_text, locals={"x": X, "y": Y}), "math")


def test_disk_exact_cases():
    r = _pde("disk", "cos(t)", "", "2")
    assert r["ok"] and r["surface"] is True and r["domain"]["mask"] == "disk"
    u = _uxy(r["methods"][0]["plot"]["particular_expr"])
    assert abs(u(0.7, -0.4) - 0.35) < 1e-9                      # x / R
    double = _uxy(_pde("disk", "sin(2t)", "", "1")["methods"][0]["plot"]["particular_expr"])
    assert abs(double(0.3, 0.5) - 2 * 0.3 * 0.5) < 1e-9          # r^2 sin 2θ = 2xy
    constant = _pde("disk", "7", "", "3")["methods"][0]
    assert abs(_uxy(constant["plot"]["particular_expr"])(1.0, -2.0) - 7) < 1e-9
    mixed = _pde("disk", "5 + cos(3t) - 2 sin(t)", "", "2")["methods"][0]
    um = _uxy(mixed["plot"]["particular_expr"])
    x, y = 0.9, 0.6
    r3 = (math.hypot(x, y) / 2) ** 3 * math.cos(3 * math.atan2(y, x))
    assert abs(um(x, y) - (5 - y + r3)) < 1e-8
    step_ = next(s for s in mixed["steps"] if s["title"] == "Compute the coefficients")
    assert step_["table"]["rows"] == [["1", "0", "-2"], ["3", "1", "0"]]
    for m in (constant, mixed):
        assert m["verified"] is True


def test_disk_step_against_poisson_integral():
    r = _pde("disk", "u(t)", "", "1")          # 1 on the upper half of the circle, 0 below
    m = r["methods"][0]
    assert m["verified"] is True
    u = _uxy(m["plot"]["particular_expr"])
    assert abs(u(0.0, 0.0) - 0.5) < 1e-9       # the centre is the mean of the boundary values
    centre = next(s for s in m["steps"] if s["title"] == "The value at the centre")
    assert "\\frac{1}{2}" in centre["latex"][0]

    def poisson(x, y):
        rad, ang = math.hypot(x, y), math.atan2(y, x)
        kernel = lambda p: (1 - rad ** 2) / (1 - 2 * rad * mpmath.cos(ang - p) + rad ** 2)
        return float(mpmath.quad(kernel, [0, math.pi])) / (2 * math.pi)
    for x, y in ((0.2, 0.3), (-0.5, 0.1), (0.1, -0.6), (0.6, 0.5)):
        assert abs(u(x, y) - poisson(x, y)) < 2e-3
    assert "radius" in _pde_err("disk", "cos(t)", "", "")["error"]
    assert "boundary values" in _pde_err("disk", "", "", "1")["error"]
    assert "Unknown symbol" in _pde_err("disk", "cos(x)", "", "1")["error"]


# ---- math tools: linear algebra, calculus, series, complex numbers ---------

import cmath

from deengine import api as engine_api, tool


def _tool(name, op, text, a="", b=""):
    return json.loads(tool(name, op, text, a, b))


def _matrix(latex):
    """Read a pmatrix out of LaTeX as a list of rows of SymPy numbers."""
    body = latex.split("\\begin{pmatrix}", 1)[1].split("\\end{pmatrix}", 1)[0]
    from sympy.parsing.sympy_parser import parse_expr
    def number(cell):
        cell = cell.strip().replace("\\frac{", "((").replace("}{", ")/(").replace("}", "))").replace("- ", "-")
        return sp.Rational(parse_expr(cell))
    return [[number(c) for c in row.split("&")] for row in body.split("\\\\")]


def _mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


@pytest.mark.parametrize("text", ["2 1; 5 3", "1 2 3; 0 1 4; 5 6 0", "2 0 0 1; 0 1 0 0; 1 0 1 0; 0 0 0 4"])
def test_inverse_multiplies_to_identity(text):
    r = _tool("linalg", "inverse", text)
    A = [[sp.Rational(v) for v in row.split()] for row in text.split(";")]
    n = len(A)
    identity = [[sp.Integer(int(i == j)) for j in range(n)] for i in range(n)]
    for m in r["methods"]:
        if m["applicable"]:
            assert m["verified"] is True
            assert _mul(A, _matrix(m["particular_latex"])) == identity
    assert method(r, "gaussjordan")["applicable"]
    assert method(r, "adjugate")["applicable"] == (n <= 3)


def test_singular_matrix_has_no_inverse():
    r = _tool("linalg", "inverse", "1 2; 2 4")
    assert r["ok"] and not any(m["applicable"] for m in r["methods"])
    assert "singular" in r["summary"]


@pytest.mark.parametrize("text,value", [
    ("1 2; 3 4", -2), ("2 0 1; 1 3 2; 1 1 1", 0), ("1 2 3; 4 5 6; 7 8 10", -3),
    ("1 2 3 4; 0 1 0 2; 2 0 1 1; 1 1 1 1", 7), ("1/2 1; 3 4", -1), ("0 1; 1 0", -1),
])
def test_determinant_two_ways(text, value):
    r = _tool("linalg", "det", text)
    assert [m["id"] for m in r["methods"]] == ["cofactor", "rowreduce"]
    for m in r["methods"]:
        assert m["verified"] is True
        assert m["particular_latex"].endswith("= %s" % sp.latex(sp.Integer(value)))


def test_linear_systems():
    one = _tool("linalg", "solve", "2 1; 1 3", "5 10")
    assert "one solution" in one["summary"]
    assert [m["id"] for m in one["methods"] if m["applicable"]] == ["gauss", "cramer", "inverse"]
    assert method(one, "gauss")["particular_latex"] == "x = 1,\\quad y = 3"      # 2+3 = 5, 1+9 = 10
    assert all(m["verified"] for m in one["methods"])
    three = _tool("linalg", "solve", "1 1 1; 0 2 5; 2 5 -1", "6; -4; 27")
    assert method(three, "cramer")["particular_latex"] == "x = 5,\\quad y = 3,\\quad z = -2"
    none = _tool("linalg", "solve", "1 2; 2 4", "3; 7")
    assert "No solution" in none["summary"]
    many = _tool("linalg", "solve", "1 2; 2 4", "3; 6")
    assert "Infinitely many" in many["summary"] and method(many, "gauss")["verified"] is True
    assert "t\\begin{pmatrix}-2\\\\1\\end{pmatrix}" in method(many, "gauss")["particular_latex"]
    wide = _tool("linalg", "solve", "1 1 1; 0 1 2", "6 5")
    assert "1 free variable" in wide["summary"]
    assert "right-hand side" in _tool("linalg", "solve", "1 2; 3 4")["error"]
    assert "one number for each row" in _tool("linalg", "solve", "1 2; 3 4", "1 2 3")["error"]


def test_row_reduction_steps():
    r = _tool("linalg", "rref", "1 2 3; 4 5 6; 7 8 10")
    m = r["methods"][0]
    assert m["verified"] is True and "Rank 3" in r["summary"]
    assert _matrix(m["particular_latex"]) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert any("R_{2} \\to R_{2} - 4\\,R_{1}" in l for s in m["steps"] for l in s["latex"])
    assert "Rank 1" in _tool("linalg", "rref", "1 2; 2 4")["summary"]


@pytest.mark.parametrize("text,values,diagonalizable", [
    ("2 0; 0 3", [2, 3], True), ("4 1; 2 3", [2, 5], True), ("1 1; 0 1", [1], False),
    ("2 1 0; 0 2 0; 0 0 3", [2, 3], False), ("2 0 0; 0 3 4; 0 4 9", [1, 2, 11], True),
])
def test_eigenvalues(text, values, diagonalizable):
    r = _tool("linalg", "eigen", text)
    m = r["methods"][0]
    assert m["verified"] is True
    assert m["particular_latex"] == ",\\quad ".join("\\lambda = %d" % v for v in values)
    assert ("Diagonalizable" in r["summary"]) == diagonalizable
    titles = [s["title"] for s in m["steps"]]
    assert ("Diagonalize" in titles) == diagonalizable and ("Jordan form" in titles) == (not diagonalizable)
    A = sp.Matrix([[sp.Rational(v) for v in row.split()] for row in text.split(";")])
    assert sorted(A.eigenvals()) == values


def test_eigen_complex_and_unsolvable():
    r = _tool("linalg", "eigen", "0 1; -1 0")
    assert r["methods"][0]["verified"] is True and "i" in r["methods"][0]["particular_latex"]
    hard = _tool("linalg", "eigen", "1 2 3; 4 5 6; 7 8 10")
    assert hard["ok"] and not hard["methods"][0]["applicable"]


def test_subspaces_independence_wronskian():
    r = _tool("linalg", "spaces", "1 2 3; 2 4 6; 1 0 1")
    assert "Rank 2, nullity 1" in r["summary"] and r["methods"][0]["verified"] is True
    assert "dependent" in _tool("linalg", "independence", "1 2; 2 4; 3 6")["summary"]
    assert "linearly independent" in _tool("linalg", "independence", "1 0; 0 1; 1 1")["summary"]
    w = _tool("linalg", "wronskian", "e^x, e^(2x)")
    assert w["methods"][0]["particular_latex"] == "W(x) = e^{3 x}" and "independent" in w["summary"]
    assert "dependent" in _tool("linalg", "wronskian", "x, 2x")["summary"]
    assert _tool("linalg", "wronskian", "sin(x), cos(x)")["methods"][0]["particular_latex"] == "W(x) = -1"
    assert "two to four" in _tool("linalg", "wronskian", "x")["error"]


@pytest.mark.parametrize("text,fragment", [
    ("1 2; 3", "same number"), ("1 2 3 4 5", "at most 4"), ("a b; c d", "must be a number"), ("", "empty"),
])
def test_matrix_input_errors(text, fragment):
    assert fragment in _tool("linalg", "rref", text)["error"]
    assert "square" in _tool("linalg", "det", "1 2 3; 4 5 6")["error"]


INTEGRALS = ["x e^x", "x cos(x^2)", "1/(x^2-1)", "e^x sin(x)", "sin(x)^2", "1/(x^2+2x+5)",
             "x^3 - 2x + 1/x", "ln(x)", "tan(x)", "x^2 e^(3x)", "1/(x(x+1)^2)", "cos(x)^3"]


@pytest.mark.parametrize("text", INTEGRALS)
def test_indefinite_integral_by_quadrature(text):
    r = _tool("calculus", "integral", text)
    assert r["ok"], r.get("error")
    from deengine.parsing import parse_expression
    f, symbols = parse_expression(text, ["x"])
    fn = sp.lambdify(symbols["x"], f, "mpmath")
    lo, hi = 1.2, 1.5
    reference = mpmath.quad(fn, [lo, hi])
    for m in r["methods"]:
        assert m["verified"] is True
        assert len(m["steps"]) >= 3
        assert m["particular_latex"].endswith("+ C")
        F = sp.sympify(m["plot"]["antiderivative"], locals={"x": X})
        got = complex(sp.N(F.subs(X, hi) - F.subs(X, lo)))
        assert abs(got - reference) < 1e-9, (got, reference)


def test_integral_technique_names():
    names = lambda t: [m["name"] for m in _tool("calculus", "integral", t)["methods"]]
    assert names("x cos(x^2)") == ["Substitution"]
    assert names("ln(x)") == ["Integration by parts"]
    assert "Partial fractions" in [s["title"] for s in _tool("calculus", "integral", "1/(x^2-1)")["methods"][0]["steps"]]
    assert "Substitute back" in [s["title"] for s in _tool("calculus", "integral", "x cos(x^2)")["methods"][0]["steps"]]
    assert "no closed form" in _tool("calculus", "integral", "x^x")["error"]


@pytest.mark.parametrize("text,a,b,value", [
    ("x^2", "0", "3", 9.0), ("sin(x)", "0", "pi", 2.0), ("e^(-x)", "0", "oo", 1.0),
    ("1/x^2", "1", "oo", 1.0), ("1/sqrt(x)", "0", "1", 2.0), ("e^(-x^2)", "-oo", "oo", math.sqrt(math.pi)),
    ("x e^(-x)", "0", "oo", 1.0), ("1/(1+x^2)", "-oo", "oo", math.pi),
])
def test_definite_integrals(text, a, b, value):
    r = _tool("calculus", "definite", text, a, b)
    m = r["methods"][0]
    assert m["verified"] is True, m["verify_note"]
    from deengine.parsing import parse_expression
    f, symbols = parse_expression(text, ["x"])
    lo = -mpmath.inf if a == "-oo" else float(sp.sympify(a))
    hi = mpmath.inf if b == "oo" else float(sp.sympify(b))
    assert abs(mpmath.quad(sp.lambdify(symbols["x"], f, "mpmath"), [lo, hi]) - value) < 1e-8
    assert abs(float(sp.sympify(r["summary"][len("Value: "):-1])) - value) < 1e-9


def test_divergent_and_bad_definite_integrals():
    assert "diverges" in _tool("calculus", "definite", "1/x", "1", "oo")["summary"]
    assert "diverges" in _tool("calculus", "definite", "1/x", "0", "1")["summary"]
    assert "lower limit" in _tool("calculus", "definite", "x", "", "1")["error"]


@pytest.mark.parametrize("text", ["1/(x^2-1)", "(x^3+1)/(x^2-4)", "(2x+1)/((x-1)^2 (x^2+1))",
                                  "(5x-4)/(x^2-x-2)", "x^2/(x^2+1)", "1/(x^3+x)"])
def test_partial_fractions_recombine(text):
    r = _tool("calculus", "fractions", text)
    m = r["methods"][0]
    assert m["verified"] is True
    from deengine.parsing import parse_expression
    f, symbols = parse_expression(text, ["x"])
    x = symbols["x"]
    parts = sp.apart(f, x)
    for point in (0.37, 2.6, -3.1):
        assert abs(float(parts.subs(x, point)) - float(f.subs(x, point))) < 1e-9
    assert "Set up the template" in [s["title"] for s in m["steps"]]


def test_partial_fraction_values_and_errors():
    m = _tool("calculus", "fractions", "(5x-4)/(x^2-x-2)")["methods"][0]
    consts = next(s for s in m["steps"] if s["title"] == "Solve for the constants")
    assert consts["latex"][0] in ("A = 2,\\quad B = 3", "A = 3,\\quad B = 2")
    assert "cover-up" in consts["text"]
    assert "ratio of polynomials" in _tool("calculus", "fractions", "sin(x)/x")["error"]
    assert "degree 3" in _tool("calculus", "fractions", "1/(x^3+x+1)")["error"]


@pytest.mark.parametrize("text,rule", [("x^2 sin(x)", "Product rule"), ("sin(x)/x", "Quotient rule"),
                                       ("e^(3x^2)", "Chain rule"), ("x^3 + 2x", "Sum rule")])
def test_derivative_rules(text, rule):
    r = _tool("calculus", "derivative", text)
    m = r["methods"][0]
    assert m["verified"] is True and rule in [s["title"] for s in m["steps"]]
    from deengine.parsing import parse_expression
    f, symbols = parse_expression(text, ["x"])
    got = sp.sympify(m["plot"]["particular_expr"], locals={"x": X})
    want = mpmath.diff(sp.lambdify(symbols["x"], f, "mpmath"), 0.8)
    assert abs(float(got.subs(X, 0.8)) - want) < 1e-8


@pytest.mark.parametrize("text,a,order,function", [
    ("e^x", "", "6", mpmath.exp), ("sin(x)", "", "7", mpmath.sin), ("1/(1-x)", "", "6", lambda v: 1 / (1 - v)),
    ("ln(1+x)", "", "5", lambda v: mpmath.log(1 + v)), ("ln(x)", "1", "4", mpmath.log),
    ("cos(x)", "pi", "4", mpmath.cos), ("sqrt(1+x)", "", "4", lambda v: mpmath.sqrt(1 + v)),
])
def test_taylor_coefficients(text, a, order, function):
    r = _tool("series", "taylor", text, a, order)
    m = r["methods"][0]
    assert m["verified"] is True
    centre = float(sp.sympify(a)) if a else 0.0
    reference = mpmath.taylor(function, centre, int(order))
    table = next(s for s in m["steps"] if "table" in s)["table"]
    for k, row in enumerate(table["rows"]):
        got = sp.sympify(m["plot"]["partials"][-1]["expr"], locals={"x": X})
        coefficient = sp.Poly(sp.expand(got.subs(X, X + sp.nsimplify(centre, [sp.pi]))), X).coeff_monomial(X**k)
        assert abs(float(coefficient) - float(reference[k])) < 1e-9, (k, coefficient, reference[k])
    best = sp.sympify(m["plot"]["partials"][-1]["expr"], locals={"x": X})
    assert abs(float(best.subs(X, centre + 0.2)) - float(function(centre + 0.2))) < 1e-3


def test_taylor_convergence_statements():
    text = lambda t, a="": " ".join(s["text"] + " ".join(s["latex"]) for s in _tool("series", "taylor", t, a)["methods"][0]["steps"])
    assert "R = \\infty" in text("e^x")
    assert "R = 1" in text("1/(1-x)") and "R = 1" in text("ln(1+x)") and "R = 1" in text("1/(1+x^2)")
    assert "R = 2" in text("1/(1-x)", "-1")
    assert "not defined" in _tool("series", "taylor", "1/x")["error"]
    assert "between 1 and 12" in _tool("series", "taylor", "e^x", "", "40")["error"]


@pytest.mark.parametrize("text,fragment,tests", [
    ("1/n", "diverges", {"pseries", "integral"}),
    ("1/n^2", "converges absolutely", {"pseries", "integral"}),
    ("(-1)^n/n", "conditionally", {"alternating"}),
    ("(-1)^(n+1)/sqrt(n)", "conditionally", {"alternating"}),
    ("(1/2)^n", "converges absolutely", {"geometric", "ratio"}),
    ("3 (2/3)^n", "converges absolutely", {"geometric"}),
    ("(3/2)^n", "diverges", {"geometric"}),
    ("n!/n^n", "converges absolutely", {"ratio"}),
    ("2^n/n!", "converges absolutely", {"ratio"}),
    ("n/(n+1)", "diverges", {"nth"}),
    ("1/(n^2+1)", "converges absolutely", {"comparison"}),
    ("1/sqrt(n)", "diverges", {"pseries"}),
    ("(-1)^n/n^3", "converges absolutely", {"pseries"}),
])
def test_series_convergence(text, fragment, tests):
    r = _tool("series", "convergence", text)
    assert fragment in r["summary"], r["summary"]
    assert tests <= {m["id"] for m in r["methods"] if m["applicable"]}
    assert len(r["methods"]) == 8 and all(m["reason"] for m in r["methods"])


def test_series_sums_and_honesty():
    assert "pi**2/6" in _tool("series", "convergence", "1/n^2")["summary"]
    assert "sum (from n = 1) is 1." in _tool("series", "convergence", "(1/2)^n")["summary"]
    assert abs(sum(0.5 ** k for k in range(1, 60)) - 1) < 1e-12
    unsettled = _tool("series", "convergence", "1/(n ln(n+1))")
    assert "None of the tests" in unsettled["summary"] and not any(m["applicable"] for m in unsettled["methods"])
    assert "expression in n" in _tool("series", "convergence", "3")["error"]


@pytest.mark.parametrize("text,a,answer", [
    ("1/n", "", "R = 1,\\qquad \\text{interval: } [-1,\\ 1)"),
    ("1/n^2", "1", "R = 1,\\qquad \\text{interval: } [0,\\ 2]"),
    ("1/n!", "", "R = \\infty,\\qquad \\text{interval: } (-\\infty, \\infty)"),
    ("n!", "", "R = 0,\\qquad \\text{interval: } \\{0\\}"),
    ("2^n", "", "R = \\frac{1}{2},\\qquad \\text{interval: } (- \\frac{1}{2},\\ \\frac{1}{2})"),
    ("n/3^n", "2", "R = 3,\\qquad \\text{interval: } (-1,\\ 5)"),
])
def test_radius_and_interval(text, a, answer):
    assert _tool("series", "radius", text, a)["methods"][0]["particular_latex"] == answer


@pytest.mark.parametrize("text,value", [
    ("(1 + 2i)/(3 - i)", (1 + 2j) / (3 - 1j)), ("(1 + i)^8", (1 + 1j) ** 8), ("e^(i pi/3)", cmath.exp(1j * cmath.pi / 3)),
    ("sqrt(-4)", 2j), ("(2-3i)(1+i)", (2 - 3j) * (1 + 1j)), ("i^i", 1j ** 1j), ("1/i", 1 / 1j),
    ("(3+4i)^2 - 2i", (3 + 4j) ** 2 - 2j), ("5", 5 + 0j),
])
def test_complex_arithmetic(text, value):
    r = _tool("complex", "evaluate", text)
    for m in r["methods"]:
        if m["applicable"]:
            assert m["verified"] is True
            x, y = m["plot"]["points"][1]
            assert abs(complex(x, y) - value) < 1e-9
    steps = " ".join(l for s in r["methods"][0]["steps"] for l in s["latex"])
    assert "|z|" in steps


def test_complex_methods_and_forms():
    r = _tool("complex", "evaluate", "(1 + i)^8")
    assert [m["id"] for m in r["methods"] if m["applicable"]] == ["rectangular", "demoivre"]
    assert method(r, "demoivre")["particular_latex"] == "z = 16"
    q = _tool("complex", "evaluate", "(1 + 2i)/(3 - i)")
    assert "Multiply by the conjugate" in [s["title"] for s in q["methods"][0]["steps"]]
    assert q["methods"][0]["particular_latex"] == "z = \\frac{1}{10} + \\frac{7}{10}\\,i"
    polar = _tool("complex", "evaluate", "1 + i")["methods"][0]
    assert any("\\sqrt{2}" in l and "\\frac{\\pi}{4}" in l for s in polar["steps"] for l in s["latex"])
    assert "Unknown symbol 'x'" in _tool("complex", "evaluate", "x + i")["error"]


@pytest.mark.parametrize("text,n", [("1", 3), ("-8", 3), ("1+i", 2), ("16", 4), ("i", 5), ("-1", 2)])
def test_complex_roots(text, n):
    r = _tool("complex", "roots", text, str(n))
    m = r["methods"][0]
    assert m["verified"] is True
    points = m["plot"]["points"]
    assert len(points) == n + 1 and points[0] == points[-1]
    from deengine.complexnum import parse
    target = complex(sp.N(parse(text)))
    roots = [complex(x, y) for x, y in points[:-1]]
    for w in roots:
        assert abs(w ** n - target) < 1e-9
    assert len({(round(w.real, 6), round(w.imag, 6)) for w in roots}) == n
    table = next(s for s in m["steps"] if "table" in s)["table"]
    assert len(table["rows"]) == n


def test_roots_of_unity_and_errors():
    cube = _tool("complex", "roots", "1", "3")["methods"][0]
    assert cube["steps"][2]["table"]["rows"][1][2] == "- \\frac{1}{2} + \\frac{\\sqrt{3}}{2}\\,i"
    assert "whole number" in _tool("complex", "roots", "i", "")["error"]
    assert "between 2 and 12" in _tool("complex", "roots", "i", "40")["error"]
    assert "only one root" in _tool("complex", "roots", "0", "2")["error"]


def test_tool_api_and_previews():
    assert json.loads(engine_api.linalg("det", "1 2; 3 4"))["summary"] == "The determinant is -2."
    assert json.loads(engine_api.calculus("definite", "x^2", "0", "3"))["summary"] == "Value: 9."
    assert json.loads(engine_api.series_tool("radius", "1/n"))["title"] == "Radius of convergence"
    assert json.loads(engine_api.complex_number("roots", "1", "3"))["kind"] == "tool"
    assert not json.loads(tool("nope", "x", "1"))["ok"]
    assert "pmatrix" in json.loads(preview("1 2\n3 4", "linalg:det"))["latex"]
    assert json.loads(preview("x e^x", "calculus:integral"))["latex"] == "\\int x e^{x}\\,dx"
    assert json.loads(preview("(1+2i)/(3-i)", "complex:evaluate"))["latex"] == "\\frac{1 + 2 i}{3 - i}"
    assert json.loads(preview("n!/n^n", "series:convergence"))["ok"]
    assert not json.loads(preview("1 2; a", "linalg:det"))["ok"]
    # "i" is the imaginary unit only in the complex tool; elsewhere it is an unknown symbol.
    assert "Unknown symbol" in json.loads(solve("y' = i*y"))["error"]
    assert "Unknown symbol" in _tool("calculus", "integral", "i x")["error"]


# ---- math tools 2: special functions, Fourier transforms, recurrences ------

def _tool(*args):
    return json.loads(tool(*args))


def _expr_fn(text):
    """A float function of x for a plot string (with the unit step)."""
    expr = sp.sympify(text, locals={"x": X})
    return sp.lambdify(X, expr, [{"Heaviside": lambda v, *rest: 0.0 if v < 0 else 1.0}, "mpmath"])


@pytest.mark.parametrize("order", ["0", "1", "3", "6"])
def test_bessel_whole_order(order):
    r = _tool("special", "bessel", order)
    assert r["ok"], r.get("error")
    m = r["methods"][0]
    assert m["verified"] is True
    n = int(order)
    plotted = _expr_fn(m["plot"]["particular_expr"])
    for value in (-28.0, -9.3, -0.7, 0.2, 2.4, 11.1, 19.7, 29.5):
        assert abs(plotted(value) - mpmath.besselj(n, value)) < 5e-5
    for xv, yv in m["plot"]["points"][::17]:           # Y_n is sampled, not a formula
        assert abs(yv - mpmath.bessely(n, xv)) < 1e-9
    rows = next(s for s in m["steps"] if s["title"] == "Zeros")["table"]["rows"]
    for k, row in enumerate(rows, start=1):
        assert abs(float(row[1]) - mpmath.besseljzero(n, k)) < 1e-6
        assert abs(mpmath.besselj(n, float(row[1]))) < 1e-5
        assert abs(mpmath.bessely(n, float(row[2]))) < 1e-5


@pytest.mark.parametrize("order", ["1/2", "3/2", "5/2"])
def test_bessel_half_integer_closed_form(order):
    r = _tool("special", "bessel", order)
    m = r["methods"][0]
    nu = float(sp.Rational(order))
    plotted = _expr_fn(m["plot"]["particular_expr"])
    for value in (0.4, 1.7, 6.2, 14.5):
        assert abs(plotted(value) - mpmath.besselj(nu, value)) < 1e-10
    assert "Closed form" in [s["title"] for s in m["steps"]]
    assert "\\sin" in m["particular_latex"] or "\\cos" in m["particular_latex"]


def test_bessel_errors():
    assert "half-integer" in _tool("special", "bessel", "1/3")["error"]
    assert "half-integer" in _tool("special", "bessel", "7")["error"]
    assert not _tool("special", "bessel", "")["ok"]


@pytest.mark.parametrize("family,reference", [
    ("legendre", mpmath.legendre), ("hermite", mpmath.hermite),
    ("laguerre", lambda n, t: mpmath.laguerre(n, 0, t)), ("chebyshev", mpmath.chebyt),
])
def test_orthogonal_polynomials(family, reference):
    for n in (0, 1, 4, 7):
        r = _tool("special", family, str(n))
        assert r["ok"], r.get("error")
        m = r["methods"][0]
        assert m["verified"] is True
        partials = m["plot"]["partials"]
        assert [p["n"] for p in partials] == list(range(n + 1))
        top = _expr_fn(partials[-1]["expr"])
        for value in (-0.9, -0.3, 0.2, 0.75):
            assert abs(top(value) - reference(n, value)) < 1e-9 * (1 + abs(reference(n, value)))
    assert "whole number" in _tool("special", family, "9")["error"]


def test_legendre_orthogonality_and_rodrigues():
    m = _tool("special", "legendre", "3")["methods"][0]
    assert m["particular_latex"] == "P_{3}(x) = \\frac{5 x^{3}}{2} - \\frac{3 x}{2}"
    assert "Rodrigues' formula" in [s["title"] for s in m["steps"]]
    p = [_expr_fn(q["expr"]) for q in m["plot"]["partials"]]
    for i in range(4):
        for j in range(4):
            value = mpmath.quad(lambda t: p[i](t) * p[j](t), [-1, 1])
            want = 2.0 / (2 * i + 1) if i == j else 0.0
            assert abs(value - want) < 1e-10


@pytest.mark.parametrize("coeffs,lo,hi,kind", [
    ("x^2; x; -4; x^2", "0", "1", "singular"),            # Bessel, order 2
    ("1; 0; 0; 1", "0", "pi", "regular"),
    ("1-x^2; -2x; 0; 1", "-1", "1", "singular"),          # Legendre
    ("1; -2x; 0; 1", "-oo", "oo", "singular"),            # Hermite
    ("x; 1-x; 0; 1", "0", "oo", "singular"),              # Laguerre
    ("1; 2; 3; 1", "0", "1", "regular"),
])
def test_sturm_liouville_form_expands_back(coeffs, lo, hi, kind):
    r = _tool("special", "sturm", coeffs, lo, hi)
    assert r["ok"], r.get("error")
    m = r["methods"][0]
    assert m["verified"] is True
    text = " ".join(s["text"] for s in m["steps"])
    assert "problem is %s" % kind in text
    # Independent check: rebuild p, q, w from the steps' formulas by redoing the algebra.
    a, b, c, d = (sp.sympify(part.replace("^", "**").replace("2x", "2*x"), locals={"x": X})
                  for part in coeffs.split(";"))
    p = sp.exp(sp.integrate(b / a, X))
    y = sp.Function("y")(X)
    lam = sp.Symbol("lambda")
    expanded = sp.diff(p * y.diff(X), X) + (p * c / a + lam * p * d / a) * y
    assert sp.simplify(expanded - p / a * (a * y.diff(X, 2) + b * y.diff(X) + (c + lam * d) * y)) == 0


def test_sturm_weight_is_positive_and_errors():
    m = _tool("special", "sturm", "1-x^2; -2x; 0; 1", "-1", "1")["methods"][0]
    assert "\\left(1 - x^{2}\\right)\\,y'" in m["particular_latex"]
    assert "four coefficients" in _tool("special", "sturm", "1; 2")["error"]
    assert "cannot be 0" in _tool("special", "sturm", "0; 1; 1; 1")["error"]
    assert "less than" in _tool("special", "sturm", "1; 0; 0; 1", "2", "1")["error"]


@pytest.mark.parametrize("text,count", [("x^3", "4"), ("e^x", "7"), ("abs(x)", "7"),
                                        ("sin(pi x)", "8"), ("u(x)", "9")])
def test_legendre_expansion(text, count):
    r = _tool("special", "expansion", text, count)
    assert r["ok"], r.get("error")
    m = r["methods"][0]
    assert m["verified"] is True
    f = _expr_fn(m["plot"]["particular_expr"])
    rows = next(s for s in m["steps"] if "table" in s)["table"]["rows"]
    assert len(rows) == int(count)
    partials = [_expr_fn(p["expr"]) for p in m["plot"]["partials"]]
    points = (-0.7, -0.35, 0.3, 0.65)
    errors = [max(abs(s(v) - f(v)) for v in points) for s in partials]
    assert errors[-1] < errors[0] or errors[0] < 1e-12
    if text in ("x^3", "e^x", "sin(pi x)"):
        assert errors[-1] < 2e-3
    # The mask divides by a step that is 0 outside [-1, 1], which the app's
    # evaluator turns into a gap in the curve (Python raises instead).
    with pytest.raises(ZeroDivisionError):
        partials[-1](1.5)


def test_legendre_expansion_coefficients_by_quadrature():
    r = _tool("special", "expansion", "e^x", "5")
    exprs = [_expr_fn(p["expr"]) for p in r["methods"][0]["plot"]["partials"]]
    # c_n recovered from the plotted partial sum by projecting onto P_n.
    top = exprs[-1]
    for n in range(5):
        got = (2 * n + 1) / 2 * mpmath.quad(lambda t: top(t) * mpmath.legendre(n, t), [-1, 1])
        want = (2 * n + 1) / 2 * mpmath.quad(lambda t: mpmath.exp(t) * mpmath.legendre(n, t), [-1, 1])
        assert abs(got - want) < 1e-9


def _ft_reference(fn, omega, cuts):
    re = mpmath.quad(lambda t: fn(t) * mpmath.cos(omega * t), cuts)
    im = mpmath.quad(lambda t: fn(t) * mpmath.sin(omega * t), cuts)
    return complex(re, -im)


FORWARD_FT = [
    ("e^(-x^2)", lambda t: mpmath.exp(-t * t), [-12, 0, 12]),
    ("e^(-abs(x))", lambda t: mpmath.exp(-abs(t)), [-50, 0, 50]),
    ("u(x) e^(-2x)", lambda t: mpmath.exp(-2 * t) if t >= 0 else 0, [0, 30]),
    ("u(x+1) - u(x-1)", lambda t: 1.0, [-1, 1]),
    ("u(x-1) - u(x-3)", lambda t: 1.0, [1, 3]),
    ("(1 - abs(x))(u(x+1) - u(x-1))", lambda t: 1 - abs(t), [-1, 0, 1]),
    ("e^(-(x-1)^2)", lambda t: mpmath.exp(-(t - 1) ** 2), [-12, 1, 14]),
    ("x e^(-x^2)", lambda t: t * mpmath.exp(-t * t), [-12, 0, 12]),
    ("e^(-3abs(x)) cos(2x)", lambda t: mpmath.exp(-3 * abs(t)) * mpmath.cos(2 * t), [-20, 0, 20]),
]


@pytest.mark.parametrize("text,fn,cuts", FORWARD_FT)
def test_fourier_transform_matches_quadrature(text, fn, cuts):
    from deengine import ftransform
    from deengine.parsing import parse_expression
    r = _tool("ftransform", "forward", text)
    assert r["ok"], r.get("error")
    m = method(r, "integral")
    assert m["verified"] is True
    f, symbols = parse_expression(text, ["x"])
    value, _ = ftransform._forward_value(f.xreplace({symbols["x"]: ftransform.X}))
    for omega in (0.4, 1.9, -3.1):
        want = _ft_reference(fn, omega, cuts)
        got = complex(sp.N(value.subs(ftransform.W, omega)))
        assert abs(got - want) < 1e-7 * (1 + abs(want)), (omega, got, want)
        # What the app graphs is F when it is real and |F| otherwise.
        shown = _expr_fn(m["plot"]["particular_expr"])(omega)
        assert abs(abs(shown) - abs(want)) < 1e-6 * (1 + abs(want))


def test_fourier_transform_table_method_and_theorems():
    r = _tool("ftransform", "forward", "e^(-(x-1)^2)")
    table = method(r, "table")
    assert table["applicable"] and "Shift theorem" in [s["title"] for s in table["steps"]]
    assert method(_tool("ftransform", "forward", "x e^(-x^2)"), "table")["applicable"] is False
    lorentz = _tool("ftransform", "forward", "1/(1+x^2)")
    assert "\\pi e^{- \\left|{\\omega}\\right|}" in method(lorentz, "integral")["particular_latex"]
    delta = _tool("ftransform", "forward", "delta(x-2)")
    assert "e^{- 2 i \\omega}" in delta["methods"][0]["particular_latex"]
    cosine = _tool("ftransform", "forward", "cos(3x)")
    assert "\\delta\\left(\\omega - 3\\right)" in cosine["methods"][0]["particular_latex"]
    assert cosine["methods"][0]["verified"] is None


INVERSE_FT = ["1/(1+w^2)", "e^(-w^2/4)", "1/(2 + i w)", "2 sin(w)/w", "1/(1 + i w)^2",
              "e^(-abs(w))", "1/((1+i w)(3 - i w))", "(1 + i w)/((1 + i w)^2 + 4)"]


@pytest.mark.parametrize("text", INVERSE_FT)
def test_inverse_fourier_transform_round_trip(text):
    from deengine import ftransform
    r = _tool("ftransform", "inverse", text)
    assert r["ok"], r.get("error")
    m = r["methods"][0]
    assert m["verified"] is True
    f = _expr_fn(m["plot"]["particular_expr"])
    F = ftransform._parse_frequency(text)
    slow = text == "e^(-abs(w))"
    for omega in (0.6, 2.2):
        if slow:    # f = 1/(pi(1+x^2)) decays slowly: integrate the oscillating tail properly
            want = 2 * mpmath.quadosc(lambda t: f(t) * mpmath.cos(omega * t), [0, mpmath.inf], omega=omega)
            got = complex(want, 0)
        else:
            got = _ft_reference(f, omega, [-40, -3, -1, 0, 1, 3, 40])
        want = complex(sp.N(F.subs(ftransform.W, omega)))
        assert abs(got - want) < 1e-6 * (1 + abs(want)), (omega, got, want)


def test_inverse_fourier_transform_forms_and_errors():
    assert "e^{- 2 x} u\\left(x\\right)" in _tool("ftransform", "inverse", "1/(2 + i w)")["methods"][0]["particular_latex"]
    assert "\\left|{x}\\right|" in _tool("ftransform", "inverse", "1/(1+w^2)")["methods"][0]["particular_latex"]
    assert "\\cos{\\left(2 x \\right)}" in _tool(
        "ftransform", "inverse", "delta(w-2) + delta(w+2)")["methods"][0]["particular_latex"]
    assert "couldn't invert" in _tool("ftransform", "inverse", "w")["error"]
    assert "does not die away" in _tool("ftransform", "forward", "x^2")["error"]
    assert "Unknown symbol" in _tool("ftransform", "forward", "e^(-t^2)")["error"]


@pytest.mark.parametrize("kind,text,fn", [
    ("sine", "e^(-2x)", lambda t: mpmath.exp(-2 * t)),
    ("cosine", "e^(-2x)", lambda t: mpmath.exp(-2 * t)),
    ("sine", "x e^(-x)", lambda t: t * mpmath.exp(-t)),
    ("cosine", "u(1 - x)", lambda t: 1.0 if t <= 1 else 0.0),
])
def test_sine_and_cosine_transforms(kind, text, fn):
    r = _tool("ftransform", kind, text)
    assert r["ok"], r.get("error")
    shown = _expr_fn(r["methods"][0]["plot"]["particular_expr"])
    wave = mpmath.sin if kind == "sine" else mpmath.cos
    for omega in (0.5, 1.8, 4.0):
        want = mpmath.quad(lambda t: fn(t) * wave(omega * t), [0, 1, 40])
        assert abs(shown(omega) - want) < 1e-6
    assert "does not converge" in _tool("ftransform", "sine", "1")["error"]


@pytest.mark.parametrize("text,k,fn,cuts", [
    ("e^(-x^2)", "1", lambda z: mpmath.exp(-z * z), []),
    ("u(x)", "2", lambda z: 1.0 if z >= 0 else 0.0, [0]),
    ("u(x+1) - u(x-1)", "", lambda z: 1.0 if -1 <= z <= 1 else 0.0, [-1, 1]),
    ("e^(-abs(x))", "1/2", lambda z: mpmath.exp(-abs(z)), [0]),
])
def test_heat_equation_on_the_line(text, k, fn, cuts):
    r = _tool("ftransform", "heat", text, k)
    assert r["ok"], r.get("error")
    m = r["methods"][0]
    assert m["verified"] is True
    kv = float(sp.Rational(k)) if k else 1.0
    partials = m["plot"]["partials"]
    assert len(partials) >= 4 and all(p["label"].startswith("t = ") for p in partials)
    assert [p["n"] for p in partials] == list(range(1, len(partials) + 1))
    for p in partials:
        tv = float(sp.Rational(p["label"][4:]))
        u = _expr_fn(p["expr"])
        for xv in (-1.4, 0.3, 2.1):
            # Direct quadrature of the heat-kernel integral.
            want = mpmath.quad(lambda z: fn(z) * mpmath.exp(-(xv - z) ** 2 / (4 * kv * tv))
                               / mpmath.sqrt(4 * mpmath.pi * kv * tv),
                               sorted(set([xv - 40, xv, xv + 40] + cuts)))
            assert abs(u(xv) - want) < 3e-5, (p["label"], xv, u(xv), want)
    # The PDE itself, by finite differences between two of the plotted times is too coarse;
    # check u_t = k u_xx on the closed form for the Gaussian instead.
    if text == "e^(-x^2)":
        t = sp.Symbol("t", positive=True)
        u = sp.exp(-X**2 / (4 * t + 1)) / sp.sqrt(4 * t + 1)
        assert sp.simplify(u.diff(t) - u.diff(X, 2)) == 0
        assert "\\sqrt{4 t + 1}" in m["particular_latex"]


def test_heat_on_the_line_errors():
    assert "positive" in _tool("ftransform", "heat", "e^(-x^2)", "-1")["error"]
    assert not _tool("ftransform", "heat", "e^(x^2)", "1")["ok"]


RECURRENCES = [
    ("a(n+2) = a(n+1) + a(n)", "0, 1", [-1, -1, 1], lambda n: 0),                    # Fibonacci
    ("a(n+1) = 2a(n) + 3", "1", [-2, 1], lambda n: 3),
    ("a(n+2) - 4a(n+1) + 4a(n) = 2^n", "0, 1", [4, -4, 1], lambda n: 2**n),          # resonant
    ("a(n+2) + a(n) = 0", "1, 0", [1, 0, 1], lambda n: 0),                           # complex roots
    ("a(n+2) = 2a(n+1) - 2a(n)", "1, 1", [2, -2, 1], lambda n: 0),                   # complex roots
    ("a(n) = 3a(n-1) - 2a(n-2) + n", "0, 1", [2, -3, 1], lambda n: n + 2),           # resonant with 1^n
    ("a(n+3) - 6a(n+2) + 11a(n+1) - 6a(n) = 0", "1, 0, 2", [-6, 11, -6, 1], lambda n: 0),
    ("a(n+2) - 5a(n+1) + 6a(n) = n 3^n", "1, 2", [6, -5, 1], lambda n: n * 3**n),    # resonant
    ("a(n+1) = a(n) + n^2", "0", [-1, 1], lambda n: n * n),
    ("a(n+1) = a(n)/2 + 1", "5", [sp.Rational(-1, 2), 1], lambda n: 1),
    ("a(n+2) - a(n) = 2^n + 1", "0, 0", [-1, 0, 1], lambda n: 2**n + 1),
]


@pytest.mark.parametrize("text,start,coeffs,g", RECURRENCES)
def test_recurrence_formula_matches_direct_iteration(text, start, coeffs, g):
    r = _tool("recurrence", "solve", text, start)
    assert r["ok"], r.get("error")
    m = method(r, "characteristic")
    assert m["applicable"] and m["verified"] is True
    # Independent iteration with exact rationals, straight from the coefficients.
    values = [sp.Rational(v.strip()) for v in start.split(",")]
    order = len(coeffs) - 1
    while len(values) < 30:
        k = len(values) - order
        values.append((g(k) - sum(coeffs[j] * values[k + j] for j in range(order))) / coeffs[order])
    n = sp.Symbol("n")
    latex_free = m["plot"]["points"]
    for xv, yv in latex_free:
        assert abs(yv - float(values[int(xv)])) < 1e-6 * (1 + abs(float(values[int(xv)])))
    rows = next(s for s in m["steps"] if s["title"] == "First terms")["table"]["rows"]
    for k, shown in rows:
        assert abs(float(shown) - float(values[int(k)])) < 1e-5 * (1 + abs(float(values[int(k)])))
    # The closed formula itself, re-derived from the engine's solver, for 30 terms.
    from deengine import recurrence
    cs, gg, nn = recurrence.parse(text)
    assert [sp.nsimplify(c) for c in cs] == [sp.nsimplify(c) for c in coeffs] or \
        [sp.nsimplify(-c) for c in cs] == [sp.nsimplify(c) for c in coeffs]
    iteration = method(r, "iteration")
    assert iteration["applicable"]
    assert abs(float(iteration["particular_latex"].split("=")[1]) - float(values[15])) < 1e-4 * (1 + abs(float(values[15])))


def test_recurrence_closed_forms_and_general_solution():
    fib = method(_tool("recurrence", "solve", "a(n+2) = a(n+1) + a(n)", "0, 1"), "characteristic")
    assert "\\sqrt{5}" in fib["particular_latex"]
    general = _tool("recurrence", "solve", "a(n+2) = a(n+1) + a(n)")
    m = method(general, "characteristic")
    assert "C_{1}" in m["particular_latex"] and m["verified"] is True and m["plot"]["points"] == []
    assert [x["id"] for x in general["methods"]] == ["characteristic"]
    resonant = method(_tool("recurrence", "solve", "a(n+2) - 4a(n+1) + 4a(n) = 2^n", "0, 1"), "characteristic")
    guess = next(s for s in resonant["steps"] if s["title"] == "Guess a particular solution")
    assert "multiplied by" in guess["text"]
    spin = method(_tool("recurrence", "solve", "a(n+2) + a(n) = 0", "1, 0"), "characteristic")
    assert "\\cos" in spin["particular_latex"]
    decay = method(_tool("recurrence", "solve", "a(n+1) = a(n)/2", "8"), "characteristic")
    assert "tends to 0" in " ".join(s["text"] for s in decay["steps"])


def test_recurrence_limits_and_errors():
    cubic = _tool("recurrence", "solve", "a(n+3) = a(n+1) + a(n)", "1, 1, 1")
    assert method(cubic, "characteristic")["applicable"] is False
    assert method(cubic, "iteration")["applicable"]
    odd = _tool("recurrence", "solve", "a(n+1) = a(n) + sin(n)", "0")
    assert method(odd, "characteristic")["applicable"] is False
    for text, start, fragment in [
        ("a(n+1) = a(n)^2", "1", "linear"),
        ("a(n+1) = n a(n)", "1", "constants"),
        ("a(n+2) = a(n+1) + a(n)", "1", "2 starting values"),
        ("b(n+1) = b(n)", "1", "a(n)"),
        ("a(n+1) + a(n)", "", "= sign"),
        ("a(n+5) = a(n)", "", "third order"),
        ("a(n) = 2", "", "at least two"),
    ]:
        r = _tool("recurrence", "solve", text, start)
        assert not r["ok"] and fragment in r["error"], (text, r.get("error"))
    assert json.loads(preview("a(n) = 3a(n-1) - 2a(n-2) + n", "recurrence:solve"))["latex"] == \
        "a_{n+2} - 3 a_{n+1} + 2 a_{n} = n + 2"


# ---- math tools 3: stability and dynamics, vector calculus -----------------

def _rk4_plane(f, g, start, h, steps):
    """Hand RK4 for x' = f(x, y), y' = g(x, y); returns the list of states."""
    state, out = list(start), [tuple(start)]
    for _ in range(steps):
        def d(v):
            return [f(*v), g(*v)]
        k1 = d(state)
        k2 = d([a + h / 2 * b for a, b in zip(state, k1)])
        k3 = d([a + h / 2 * b for a, b in zip(state, k2)])
        k4 = d([a + h * b for a, b in zip(state, k3)])
        state = [a + h / 6 * (p + 2 * q + 2 * u + w) for a, p, q, u, w in zip(state, k1, k2, k3, k4)]
        out.append(tuple(state))
    return out


def _plane_field(r):
    f = sp.lambdify((X, Y), sp.sympify(r["field"]["f"], locals={"x": X, "y": Y}), "math")
    g = sp.lambdify((X, Y), sp.sympify(r["field"]["g"], locals={"x": X, "y": Y}), "math")
    return f, g


@pytest.mark.parametrize("system,verdict,linear_applies", [
    ("y; -x - y", "asymptotically stable", True),
    ("-x^3; -y^3", "asymptotically stable", False),
    ("-y - x^3; x - y^3", "asymptotically stable", False),
    ("y; -x", "is stable", False),
    ("x + y; y", "unstable", True),
    ("x^3; y^3", "unstable", False),
])
def test_lyapunov_conclusions(system, verdict, linear_applies):
    r = _tool("dynamics", "lyapunov", system)
    assert r["ok"], r.get("error")
    quad = method(r, "quadratic")
    assert quad["applicable"] and verdict in quad["particular_latex"]
    assert method(r, "linear")["applicable"] is linear_applies
    assert r["graph"] == "phase" and r["equilibria"][0]["x"] == 0.0
    # Independent check: follow a nearby trajectory with RK4.
    f, g = _plane_field(r)
    # Long enough for slow (cubic) decay, short enough that growth does not overflow.
    path = _rk4_plane(f, g, (0.2, 0.1), 0.01, 1000 if verdict == "unstable" else 10000)
    start, end = math.hypot(*path[0]), math.hypot(*path[-1])
    if verdict == "asymptotically stable":
        assert end < 0.6 * start
    elif verdict == "unstable":
        assert max(math.hypot(*p) for p in path) > 1.5 * start
    else:
        assert abs(end - start) < 1e-6


def test_lyapunov_derivative_matches_trajectory():
    from deengine import dynamics
    f, g = dynamics._system("-y - x^3; x - y^3")
    V = dynamics._candidates(f, g, (sp.Integer(0), sp.Integer(0)))[0][1]
    vdot = sp.lambdify((dynamics.X, dynamics.Y), sp.diff(V, dynamics.X) * f + sp.diff(V, dynamics.Y) * g, "math")
    vf = sp.lambdify((dynamics.X, dynamics.Y), V, "math")
    ff = sp.lambdify((dynamics.X, dynamics.Y), f, "math")
    gg = sp.lambdify((dynamics.X, dynamics.Y), g, "math")
    h = 1e-3
    path = _rk4_plane(ff, gg, (0.5, -0.3), h, 400)
    for k in (50, 200, 350):
        numeric = (vf(*path[k + 1]) - vf(*path[k - 1])) / (2 * h)
        assert abs(numeric - vdot(*path[k])) < 1e-5
        assert vdot(*path[k]) < 0


def test_lyapunov_semidefinite_and_given_function():
    r = _tool("dynamics", "lyapunov", "y; -sin(x) - y", "y^2/2 + 1 - cos(x)")
    given = method(r, "given")
    assert "is stable" in given["particular_latex"]              # V' = -y^2 is only semidefinite
    assert "negative semidefinite" in " ".join(s["text"] for s in given["steps"])
    assert "asymptotically stable" in method(r, "quadratic")["particular_latex"]
    none = _tool("dynamics", "lyapunov", "y - x^3; -x^3")
    assert not any(m["applicable"] for m in none["methods"]) and "None of the tests" in none["summary"]
    bad = _tool("dynamics", "lyapunov", "y; -x", "x*y")
    assert "inconclusive" in method(bad, "given")["particular_latex"]
    shifted = _tool("dynamics", "lyapunov", "x(1-x-y); y(2-x-y)", "", "0, 2")
    assert shifted["ok"] and shifted["equilibria"][0]["y"] == 2.0
    assert "not an equilibrium" in _tool("dynamics", "lyapunov", "x + 1; y", "", "0, 0")["error"]
    assert "semicolon" in _tool("dynamics", "lyapunov", "x + y")["error"]


def test_conservative_systems():
    pend = _tool("dynamics", "conservative", "-sin(x)")
    assert {(round(e["x"], 6), e["label"]) for e in pend["equilibria"]} == {(0.0, "center"), (round(math.pi, 6), "saddle point")}
    well = _tool("dynamics", "conservative", "x - x^3")
    assert [(e["x"], e["label"]) for e in well["equilibria"]] == [(-1.0, "center"), (0.0, "saddle point"), (1.0, "center")]
    assert method(well, "energy")["verified"] is True
    # Energy E = y^2/2 + x^4/4 - x^2/2 stays constant along an RK4 trajectory.
    f, g = _plane_field(well)
    energy = lambda p: p[1] ** 2 / 2 + p[0] ** 4 / 4 - p[0] ** 2 / 2
    path = _rk4_plane(f, g, (0.5, 0.4), 0.005, 4000)
    assert max(abs(energy(p) - energy(path[0])) for p in path) < 1e-8
    assert "Enter" in _tool("dynamics", "conservative", "")["error"]


def test_limit_cycles():
    r = _tool("dynamics", "cycles", "x - y - x(x^2+y^2); x + y - y(x^2+y^2)")
    polar = method(r, "polar")
    assert "r = 1\\ \\text{(stable)}" in polar["particular_latex"] and polar["verified"] is True
    assert polar["plot"]["x_expr"] == "cos(x)" and r["legends"]["exact"] == "Limit cycle"
    f, g = _plane_field(r)
    for start in ((0.1, 0.0), (2.0, 0.0)):
        end = _rk4_plane(f, g, start, 0.01, 3000)[-1]
        assert abs(math.hypot(*end) - 1.0) < 1e-3
    two = _tool("dynamics", "cycles", "-y + x(1 - x^2 - y^2)(4 - x^2 - y^2); x + y(1 - x^2 - y^2)(4 - x^2 - y^2)")
    assert "r = 1\\ \\text{(stable)},\\quad r = 2\\ \\text{(unstable)}" in method(two, "polar")["particular_latex"]
    none = _tool("dynamics", "cycles", "-x + y^3; -y - x^3")
    assert "No closed orbits" in method(none, "bendixson")["particular_latex"]
    assert method(none, "polar")["applicable"] is False
    centre = _tool("dynamics", "cycles", "y; -x")
    assert "no information" in method(centre, "bendixson")["particular_latex"]
    assert "No circular limit cycle" in method(centre, "polar")["particular_latex"]


def test_van_der_pol_by_lienard():
    r = _tool("dynamics", "cycles", "y; -x + (1 - x^2) y")
    assert "Exactly one limit cycle" in method(r, "lienard")["particular_latex"]
    f, g = _plane_field(r)
    path = _rk4_plane(f, g, (0.1, 0.0), 0.01, 6000)
    amplitude = max(abs(p[0]) for p in path[-1500:])
    assert 1.95 < amplitude < 2.05          # the van der Pol cycle has amplitude about 2
    damped = _tool("dynamics", "cycles", "y; -x - y")
    assert "no conclusion" in method(damped, "lienard")["particular_latex"]


@pytest.mark.parametrize("text,kinds", [
    ("m - x^2", {0.0: "saddle-node"}),
    ("m x - x^2", {0.0: "transcritical"}),
    ("m x - x^3", {0.0: "supercritical pitchfork"}),
    ("mu x + x^3", {0.0: "subcritical pitchfork"}),
    ("m + x - x^3", {-2 * math.sqrt(3) / 9: "saddle-node", 2 * math.sqrt(3) / 9: "saddle-node"}),
    ("x(m - x + x^2)", {0.0: "transcritical", 0.25: "saddle-node"}),
])
def test_bifurcation_types(text, kinds):
    r = _tool("dynamics", "bifurcation", text)
    assert r["ok"], r.get("error")
    table = next(s for s in method(r, "branches")["steps"] if s["title"].startswith("Locate"))["table"]
    found = {}
    for row in table["rows"]:
        value = float(sp.sympify(row[0].replace("\\frac{2 \\sqrt{3}}{9}", "2*sqrt(3)/9").replace("- ", "-")
                                 .replace("\\frac{1}{4}", "1/4")))
        found[round(value, 9)] = row[2].replace("\\text{", "").rstrip("}")
    assert found == {round(k, 9): v for k, v in kinds.items()}
    # Every plotted equilibrium really is one, and its label matches the sign of f_x.
    from deengine import dynamics
    f = dynamics._expr(text, ["x", "m"], "f")
    fn = sp.lambdify((dynamics.X, dynamics.M), f, "math")
    fx = sp.lambdify((dynamics.X, dynamics.M), sp.diff(f, dynamics.X), "math")
    assert len(r["equilibria"]) >= 10
    for e in r["equilibria"]:
        assert abs(fn(e["y"], e["x"])) < 1e-7
        slope = fx(e["y"], e["x"])
        if abs(slope) > 1e-6:
            assert e["label"] == ("stable" if slope < 0 else "unstable")
    assert r["field"]["f"] == "0*x" and "parameter m" in r["legends"]["hint"]


def test_bifurcation_and_hopf_edges():
    assert "No bifurcation" in method(_tool("dynamics", "bifurcation", "m - e^x + 1"), "branches")["particular_latex"]
    assert "parameter" in _tool("dynamics", "bifurcation", "x^2 + 1")["error"]
    hopf = _tool("dynamics", "hopf", "m x - y - x(x^2+y^2); x + m y - y(x^2+y^2)")
    m = method(hopf, "hopf")
    assert "m = 0" in m["particular_latex"] and "\\pm i" in m["particular_latex"]
    assert "Direction, from polar coordinates" in [s["title"] for s in m["steps"]]
    # At m = 1/2 the periodic orbit has radius sqrt(1/2).
    xs = sp.sympify(m["plot"]["x_expr"], locals={"x": X})
    assert abs(float(xs.subs(X, 0)) - math.sqrt(0.5)) < 1e-12
    f, g = _plane_field(hopf)
    end = _rk4_plane(f, g, (0.05, 0.0), 0.01, 4000)[-1]
    assert abs(math.hypot(*end) - math.sqrt(0.5)) < 1e-3
    vdp = _tool("dynamics", "hopf", "y; -x + (m - x^2) y", "2")
    assert "m = 0" in method(vdp, "hopf")["particular_latex"] and "m = 2" in vdp["summary"]
    assert "not tell" in " ".join(s["text"] for s in method(vdp, "hopf")["steps"])
    assert "No Hopf" in method(_tool("dynamics", "hopf", "m x; -y"), "hopf")["particular_latex"]
    assert "parameter" in _tool("dynamics", "hopf", "x; y")["error"]


def _central(fn, point, index, h=1e-5):
    up, down = list(point), list(point)
    up[index] += h
    down[index] -= h
    return (fn(*up) - fn(*down)) / (2 * h)


def test_vector_operators_match_finite_differences():
    from deengine import vector
    V = (vector.X, vector.Y, vector.Z)
    F = vector._components("x^2 y + z; y z^2 - x; x y z + sin(x)", ["x", "y", "z"], "F")
    fns = [sp.lambdify(V, c, "math") for c in F]
    point = (0.7, -0.4, 1.3)
    div = sum(_central(fns[i], point, i) for i in range(3))
    curl = [_central(fns[2], point, 1) - _central(fns[1], point, 2), _central(fns[0], point, 2) - _central(fns[2], point, 0),
            _central(fns[1], point, 0) - _central(fns[0], point, 1)]
    got_curl = [float(c.subs(dict(zip(V, point)))) for c in vector._curl(F, V)]
    assert all(abs(a - b) < 1e-6 for a, b in zip(curl, got_curl))
    r = _tool("vector", "divcurl", "x^2 y + z; y z^2 - x; x y z + sin(x)")
    shown = method(r, "operators")["steps"][0]["latex"][1].split(" = ", 1)[1]
    assert shown == sp.latex(sp.sympify("2*x*y + z**2 + x*y"), ln_notation=True)
    assert abs(float(sp.sympify("2*x*y + z**2 + x*y").subs({"x": 0.7, "y": -0.4, "z": 1.3})) - div) < 1e-6
    # Identities: curl grad = 0 and div curl = 0.
    for text in ("x^2 y z + sin(x y)", "x^3 - 3 x y^2 + z^4"):
        f = vector._expr(text, ["x", "y", "z"], "f")
        assert all(c == 0 for c in vector._curl([sp.diff(f, v) for v in V], V))
    for text in ("x^2 y; y^2 z; z^2 x", "sin(y); x z; x y^2"):
        G = vector._curl(vector._components(text, ["x", "y", "z"], "F"), V)
        assert sp.simplify(sum(sp.diff(c, v) for c, v in zip(G, V))) == 0


def test_gradient_and_directional_derivative():
    r = _tool("vector", "gradient", "x^2 y + sin(y)", "1, 0", "3, 4")
    m = method(r, "partials")
    fn = lambda x, y: x ** 2 * y + math.sin(y)
    gx, gy = _central(fn, (1.0, 0.0), 0), _central(fn, (1.0, 0.0), 1)
    text = " ".join(l for s in m["steps"] for l in s["latex"])
    assert "\\nabla f(1,\\ 0) = \\left\\langle 0,\\ 2 \\right\\rangle" in text
    assert abs(gx - 0) < 1e-8 and abs(gy - 2) < 1e-8
    assert "D_{\\mathbf{u}} f = \\nabla f \\cdot \\mathbf{u} = \\frac{8}{5}" in text      # (0, 2).(3, 4)/5
    assert "z = 2 y" in text                                                            # tangent plane
    assert r["graph"] == "surface" and r["surface"] is True
    d = r["domain"]
    assert d["x_min"] == -2.0 and d["x_max"] == 4.0 and d["u_min"] < d["u_max"]
    assert m["plot"]["particular_expr"] == "x**2*y + sin(y)"
    three = _tool("vector", "gradient", "x y z + z^2", "1, 2, 3", "1, 0, 0")
    assert three["ok"] and "graph" not in three
    assert "= 6" in " ".join(l for s in method(three, "partials")["steps"] for l in s["latex"])
    assert "point as well" in _tool("vector", "gradient", "x^2", "", "1, 0")["error"]
    assert "zero vector" in _tool("vector", "gradient", "x^2 + y", "1, 1", "0, 0")["error"]


def test_potential_functions():
    from deengine import vector
    for text, names in (("2x y; x^2 + 3y^2", ["x", "y"]), ("y z; x z; x y", ["x", "y", "z"]),
                        ("e^x cos(y) + y z; x z - e^x sin(y); x y + z", ["x", "y", "z"])):
        F = vector._components(text, names, "F")
        V = [vector._SYMBOL[n] for n in names]
        phi = vector._potential(F, V)
        assert phi is not None and all(sp.simplify(sp.diff(phi, v) - c) == 0 for v, c in zip(V, F))
        assert method(_tool("vector", "divcurl", text), "potential")["verified"] is True
    assert method(_tool("vector", "divcurl", "-y; x"), "potential")["applicable"] is False
    assert _tool("vector", "divcurl", "-y; x")["graph"] == "phase"
    assert "components" in _tool("vector", "divcurl", "x")["error"]


@pytest.mark.parametrize("field,curve,limits,expected", [
    ("-y; x", "cos(t); sin(t)", "0, 2pi", 2 * math.pi),
    ("2x y; x^2 + 3y^2", "t; t^2", "0, 1", 2.0),
    ("2x y; x^2 + 3y^2", "t; t", "0, 1", 2.0),              # same end points: path independent
    ("y z; x z; x y", "t; t^2; t^3", "0, 1", 1.0),
    ("x^2; x y", "t; 2t", "0, 1", None),
    ("x + y", "cos(t); sin(t)", "0, pi/2", 2.0),
    ("x y", "t; t^2", "0, 1", None),
])
def test_line_integrals_match_quadrature(field, curve, limits, expected):
    from deengine import vector
    r = _tool("vector", "line", field, curve, limits)
    assert r["ok"], r.get("error")
    comps = vector._components(curve, ["t"], "r")
    names = ["x", "y", "z"][:len(comps)]
    parts = vector._split(field)
    a, b = [float(sp.sympify(v.replace("pi", "*pi") if v.strip()[0].isdigit() and "pi" in v else v))
            for v in limits.split(",")]
    cf = [sp.lambdify(vector.T, c, "mpmath") for c in comps]
    df = [sp.lambdify(vector.T, sp.diff(c, vector.T), "mpmath") for c in comps]
    variables = [vector._SYMBOL[n] for n in names]
    if len(parts) == 1:
        f = sp.lambdify(variables, vector._expr(parts[0], names, "f"), "mpmath")
        reference = mpmath.quad(lambda t: f(*[c(t) for c in cf]) * mpmath.sqrt(sum(d(t) ** 2 for d in df)), [a, b])
    else:
        F = [sp.lambdify(variables, vector._expr(p, names, "F"), "mpmath") for p in parts]
        reference = mpmath.quad(lambda t: sum(Fi(*[c(t) for c in cf]) * d(t) for Fi, d in zip(F, df)), [a, b])
    for m in r["methods"]:
        if m["applicable"]:
            assert abs(m["value"] - float(reference)) < 1e-8, (m["id"], m["value"], reference)
            assert m["verified"] is True
    if expected is not None:
        assert abs(float(reference) - expected) < 1e-9
    if field.startswith("2x y"):
        assert method(r, "fundamental")["applicable"]
    if len(comps) == 2:
        assert r["graph"] == "phase"
        # The drawn curve stops at the end points of the parameter interval.
        xs = sp.sympify(r["methods"][0]["plot"]["x_expr"], locals={"x": X})
        assert abs(float(xs.subs(X, b + 5)) - float(xs.subs(X, b))) < 1e-12


def test_line_integral_errors():
    assert "must match" in _tool("vector", "line", "y; x", "t; t; t", "0, 1")["error"]
    assert "curve" in _tool("vector", "line", "y; x", "", "0, 1")["error"]
    assert "limits" in _tool("vector", "line", "y; x", "t; t", "")["error"]


@pytest.mark.parametrize("op,text,a,b,ref", [
    ("double", "x y", "0, 1", "0, x", lambda: mpmath.quad(lambda x: mpmath.quad(lambda y: x * y, [0, x]), [0, 1])),
    ("double", "x^2 + y", "0, 2", "1, 3", lambda: mpmath.quad(lambda x, y: x ** 2 + y, [0, 2], [1, 3])),
    ("double", "x + y", "0, 1", "x^2, x", lambda: mpmath.quad(lambda x: mpmath.quad(lambda y: x + y, [x ** 2, x]), [0, 1])),
    ("double", "e^(x+y)", "0, 1", "0, 1", lambda: (mpmath.e - 1) ** 2),
    ("polar", "x^2 + y^2", "0, 2", "0, 2pi", lambda: 8 * mpmath.pi),
    ("polar", "e^(-x^2-y^2)", "0, 1", "0, pi", lambda: mpmath.pi / 2 * (1 - mpmath.exp(-1))),
    ("polar", "x", "0, 1", "0, pi/2", lambda: mpmath.mpf(1) / 3),
])
def test_double_integrals(op, text, a, b, ref):
    r = _tool("vector", op, text, a, b)
    assert r["ok"], r.get("error")
    want = float(ref())
    tolerance = 1e-8
    for m in r["methods"]:
        if m["applicable"]:
            assert abs(m["value"] - want) < tolerance * (1 + abs(want)), (m["id"], m["value"], want)
    if op == "double" and "x" not in b:
        assert method(r, "swapped")["applicable"]
    assert "limits" in _tool("vector", op, text, "", "")["error"]


@pytest.mark.parametrize("field,region,expected", [
    ("-y; x", "2", 8 * math.pi),                       # 2 * area
    ("x y; x^2", "0, 1, 0, 2", 1.0),
    ("-y^3; x^3", "1, 2", 45 * math.pi / 2),           # 3 * integral of r^2 over the ring
    ("x^2 - y; x + y^2", "0, 3, 1, 2", 6.0),           # curl = 2, area 3
    ("e^x; e^y", "1", 0.0),                            # conservative
])
def test_greens_theorem_both_sides(field, region, expected):
    r = _tool("vector", "green", field, region)
    assert r["ok"], r.get("error")
    green, direct = method(r, "green"), method(r, "direct")
    assert green["verified"] is True and direct["verified"] is True
    assert abs(green["value"] - expected) < 1e-9 and abs(direct["value"] - expected) < 1e-9
    assert "region" in _tool("vector", "green", field, "1, 2, 3")["error"].lower()


@pytest.mark.parametrize("op,field,surface,expected", [
    ("flux", "x; y; z", "sphere 2", 32 * math.pi),             # 3 * volume
    ("flux", "x^2; y^2; z^2", "box 1, 2, 3", 36.0),
    ("flux", "x y; y z; z x", "cylinder 1, 2", 2 * math.pi),
    ("flux", "x^3; y^3; z^3", "sphere 1", 12 * math.pi / 5),
    ("flux", "y; -x; 0*z", "cylinder 2, 1", 0.0),
    ("stokes", "-y; x; z", "hemisphere 1", 2 * math.pi),
    ("stokes", "z; x; y", "paraboloid 2", 4 * math.pi),
    ("stokes", "y^2; x; z^2", "disk 3", 9 * math.pi),
    ("stokes", "-y^3; x^3; z", "hemisphere 2", 24 * math.pi),  # 3 * integral of r^2 over the disc
])
def test_flux_and_stokes_two_ways(op, field, surface, expected):
    r = _tool("vector", op, field, surface)
    assert r["ok"], r.get("error")
    values = [m["value"] for m in r["methods"]]
    assert len(values) == 2 and all(m["verified"] is True for m in r["methods"])
    assert all(abs(v - expected) < 1e-9 * (1 + abs(expected)) for v in values)


def test_surface_errors_and_previews():
    assert "Supported here" in _tool("vector", "flux", "x; y; z", "cone 1")["error"]
    assert "positive" in _tool("vector", "flux", "x; y; z", "sphere -1")["error"]
    assert "3 components" in _tool("vector", "stokes", "x; y", "disk 1")["error"]
    assert json.loads(preview("y; -x", "dynamics:lyapunov"))["latex"] == "x' = y,\\qquad y' = - x"
    assert json.loads(preview("mu x - x^3", "dynamics:bifurcation"))["latex"] == "x' = m x - x^{3}"
    assert json.loads(preview("x; y; z", "vector:flux"))["latex"].startswith("\\mathbf{F}")
    assert not json.loads(preview("q", "vector:gradient"))["ok"]


# ---- physics scenarios ----------------------------------------------------

from deengine import physics, physics_catalog
from deengine.scenarios import units as punits


def _vals(**kw):
    return {k: {"value": str(v[0]), "unit": v[1]} for k, v in kw.items()}


def _phys(scenario, values=None, effects=None, question=None, preset=None):
    raw = {}
    if values is not None:
        raw["values"] = values
        raw["effects"] = effects or {}
    elif effects is not None:
        raw["effects"] = effects
    if preset:
        raw["preset"] = preset
    if question:
        raw["question"] = question
    return json.loads(physics(scenario, json.dumps(raw)))


def _rk4(deriv, state, t_end, n):
    """A plain RK4 written here, so the engine is not checked against itself."""
    h = t_end / n
    t = 0.0
    out = [(0.0, list(state))]
    for _ in range(n):
        k1 = deriv(t, state)
        k2 = deriv(t + h / 2, [s + h / 2 * k for s, k in zip(state, k1)])
        k3 = deriv(t + h / 2, [s + h / 2 * k for s, k in zip(state, k2)])
        k4 = deriv(t + h, [s + h * k for s, k in zip(state, k3)])
        state = [s + h / 6 * (a + 2 * b + 2 * c + d) for s, a, b, c, d in zip(state, k1, k2, k3, k4)]
        t += h
        out.append((t, list(state)))
    return out


def _at(samples, t, i=0):
    for (t0, a), (t1, b) in zip(samples, samples[1:]):
        if t0 <= t <= t1:
            w = (t - t0) / (t1 - t0)
            return a[i] + w * (b[i] - a[i])
    return samples[-1][1][i]


def _curve_matches(result, reference, tol=2e-4):
    curve = result["physics"]["curve"]
    scale = max(abs(p[1]) for p in curve) or 1.0
    for t, y in curve[10::23]:
        assert abs(y - reference(t)) < tol * scale, (t, y, reference(t))


FALL_CASES = [
    # (mass kg, drag, coefficient, v0, buoyancy (rho_f, vol) or None, h0)
    (80.0, "none", None, 0.0, None, 500.0),
    (80.0, "linear", 12.0, 0.0, None, 500.0),
    (80.0, "linear", 12.0, 30.0, None, 2000.0),
    (0.145, "quadratic", 0.0009, 0.0, None, 60.0),
    (0.145, "quadratic", 0.0009, 60.0, None, 300.0),      # starts above terminal speed
    (80.0, "quadratic", 0.3, 5.0, (1.225, 0.08), 3000.0),
    (2.0, "linear", 0.8, -6.0, None, 40.0),               # thrown upward
    (0.145, "quadratic", 0.0009, -15.0, None, 60.0),      # thrown upward: numerical path
]


@pytest.mark.parametrize("m,drag,coef,v0,buoy,h0", FALL_CASES)
def test_falling_matches_independent_integration(m, drag, coef, v0, buoy, h0):
    g = 9.81
    values = _vals(m=(m, "kg"), g=(g, "m/s2"), v0=(v0, "m/s"), h0=(h0, "m"))
    effects = {"drag": drag}
    if drag == "linear":
        values.update(_vals(b=(coef, "kg/s")))
    elif drag == "quadratic":
        values.update(_vals(c=(coef, "kg/m")))
    lift = 0.0
    if buoy:
        values.update(_vals(rho_f=(buoy[0], "kg/m3"), vol=(buoy[1], "m3")))
        effects["buoyancy"] = "on"
        lift = buoy[0] * buoy[1] * g / m

    def deriv(t, st):
        v = st[0]
        a = g - lift
        if drag == "linear":
            a -= coef * v / m
        elif drag == "quadratic":
            a -= coef * v * abs(v) / m
        return [a, v]

    r = _phys("falling", values, effects, "ground")
    assert r["ok"], r.get("error")
    t_end = r["physics"]["t_max"]
    ref = _rk4(deriv, [v0, 0.0], t_end, 20000)
    _curve_matches(r, lambda t: _at(ref, t, 0))
    # The event: distance fallen equals the starting height at the reported time.
    assert abs(_at(ref, t_end, 1) - h0) < 1e-3 * h0
    marker = r["physics"]["marker"]
    assert abs(marker["t"] - t_end) < 1e-9 and abs(marker["y"] - _at(ref, t_end, 0)) < 1e-3 * (1 + abs(marker["y"]))
    assert all(mm["verified"] is True for mm in r["methods"] if mm["applicable"])
    assert r["methods"][0]["id"] == ("numeric" if (drag == "quadratic" and v0 < 0) else "exact")
    if drag == "none" and not buoy:
        assert [mm["id"] for mm in r["methods"]] == ["exact", "energy"]
        assert r["physics"]["ideal"] == []
    else:
        assert len(r["physics"]["ideal"]) > 100


def test_falling_terminal_velocity_and_inference_round_trip():
    lin = _phys("falling", _vals(m=(80, "kg"), g=(9.81, "m/s2"), b=(12, "kg/s")), {"drag": "linear"}, "terminal")
    assert abs(lin["physics"]["curve"][-1][1] - 80 * 9.81 / 12) < 0.01 * 65.4
    assert "65.4 m/s" in lin["summary"]
    quad = _phys("falling", _vals(m=(80, "kg"), g=(9.81, "m/s2"), rho=(1.225, "kg/m3"), cd=(1.0, ""),
                                  area=(0.5, "m2")), {"drag": "quadratic"}, "terminal")
    vt = math.sqrt(80 * 9.81 / (0.5 * 1.225 * 1.0 * 0.5))
    assert punits.text(vt, "speed") in quad["summary"]
    assert "Work out the drag coefficient" in [s["title"] for s in quad["methods"][0]["steps"]]
    # Feed the terminal speed back in and recover the coefficient.
    back = _phys("falling", _vals(m=(80, "kg"), g=(9.81, "m/s2"), vt_meas=(vt, "m/s")), {"drag": "quadratic"}, "infer")
    assert punits.text(0.5 * 1.225 * 0.5, "drag_quadratic", "kg/m") in back["summary"]
    back_lin = _phys("falling", _vals(m=(80, "kg"), g=(9.81, "m/s2"), vt_meas=(65.4, "m/s")), {"drag": "linear"}, "infer")
    assert "12 kg/s" in back_lin["summary"]


def test_falling_fraction_at_time_and_limits():
    values = _vals(m=(80, "kg"), g=(9.81, "m/s2"), v0=(0, "m/s"), b=(12, "kg/s"), fraction=(95, "%"), h0=(4000, "m"))
    r = _phys("falling", values, {"drag": "linear"}, "fraction")
    tau = 80 / 12
    assert abs(r["physics"]["marker"]["t"] - (-tau * math.log(0.05))) < 1e-6
    at = _phys("falling", dict(values, **_vals(t_query=(3, "s"))), {"drag": "linear"}, "at_time")
    vt = 80 * 9.81 / 12
    assert abs(at["physics"]["marker"]["y"] - vt * (1 - math.exp(-3 / tau))) < 1e-9
    # Drag tending to zero recovers free fall.
    tiny = _phys("falling", _vals(m=(80, "kg"), g=(9.81, "m/s2"), v0=(0, "m/s"), h0=(100, "m"), b=(1e-9, "kg/s")),
                 {"drag": "linear"}, "ground")
    free = _phys("falling", _vals(m=(80, "kg"), g=(9.81, "m/s2"), v0=(0, "m/s"), h0=(100, "m")), {"drag": "none"}, "ground")
    assert abs(tiny["physics"]["t_max"] - free["physics"]["t_max"]) < 1e-6
    assert abs(free["physics"]["t_max"] - math.sqrt(200 / 9.81)) < 1e-9


def test_falling_gravity_varying_with_height():
    values = _vals(m=(100, "kg"), g=(9.81, "m/s2"), v0=(0, "m/s"), h0=(400, "km"), radius=(6371, "km"))
    r = _phys("falling", values, {"altitude": "on"}, "ground")
    assert r["methods"][0]["id"] == "numeric" and r["methods"][0]["verified"] is True
    R, h = 6.371e6, 4e5
    v_exact = math.sqrt(2 * 9.81 * R * h / (R + h))      # energy conservation in an inverse-square field
    assert abs(r["physics"]["marker"]["y"] - v_exact) < 1e-4 * v_exact
    assert "numerically" in " ".join(s["title"] for s in r["methods"][0]["steps"]).lower()


SPRING_CASES = [
    # m, k, c, F0, w (rad/s), g, x0, v0
    (0.5, 20.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0),
    (0.5, 20.0, 0.0, 0.0, 0.0, 0.0, 0.05, -0.4),
    (400.0, 40000.0, 3500.0, 0.0, 0.0, 0.0, 0.05, 0.0),          # underdamped
    (30.0, 60.0, 2 * math.sqrt(30 * 60.0), 0.0, 0.0, 0.0, 0.6, 0.0),  # critically damped
    (2.0, 8.0, 20.0, 0.0, 0.0, 0.0, 0.3, 1.0),                    # overdamped
    (60.0, 60000.0, 600.0, 100.0, 31.4, 0.0, 0.0, 0.0),           # damped and driven
    (1.0, 16.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0),                    # driven at resonance, undamped
    (1.0, 16.0, 0.0, 2.0, 3.0, 0.0, 0.1, 0.0),                    # driven off resonance, undamped
    (0.5, 20.0, 0.4, 0.0, 0.0, 9.81, 0.0, 0.0),                   # vertical spring released at natural length
    (0.5, 20.0, 0.4, 1.5, 5.0, 9.81, 0.1, 0.2),                   # everything on
]


@pytest.mark.parametrize("m,k,c,F0,w,g,x0,v0", SPRING_CASES)
def test_spring_matches_independent_integration(m, k, c, F0, w, g, x0, v0):
    values = _vals(m=(m, "kg"), k=(k, "N/m"), x0=(x0, "m"), v0=(v0, "m/s"))
    effects = {}
    if c:
        values.update(_vals(c=(c, "kg/s")))
        effects["damping"] = "on"
    if F0:
        values.update(_vals(F0=(F0, "N"), w=(w, "rad/s")))
        effects["driving"] = "on"
    if g:
        values.update(_vals(g=(g, "m/s2")))
        effects["gravity"] = "on"
    r = _phys("spring", values, effects, "motion")
    assert r["ok"], r.get("error")
    t_end = r["physics"]["t_max"]
    ref = _rk4(lambda t, st: [st[1], (F0 * math.cos(w * t) + m * g - c * st[1] - k * st[0]) / m],
               [x0, v0], t_end, 60000)
    _curve_matches(r, lambda t: _at(ref, t, 0), tol=5e-4)
    assert r["physics"]["curve"][0][1] == pytest.approx(x0, abs=1e-12)
    assert all(mm["verified"] is True for mm in r["methods"])
    if c and not F0:
        e = r["physics"]["energy"]            # energy lost equals the work done against damping
        assert e["initial"] - e["final"] == pytest.approx(e["dissipated"], rel=1e-5)
    if not (c or F0 or g):
        assert [mm["id"] for mm in r["methods"]] == ["exact", "energy"]


def test_spring_frequency_regimes_and_resonance():
    f = _phys("spring", _vals(m=(0.5, "kg"), k=(20, "N/m")), {}, "frequency")
    assert punits.text(math.sqrt(40) / (2 * math.pi), "frequency") in f["summary"]
    for c, word in ((3500, "underdamped"), (8000, "critically damped"), (20000, "overdamped")):
        r = _phys("spring", _vals(m=(400, "kg"), k=(40, "kN/m"), c=(c, "kg/s")), {"damping": "on"}, "frequency")
        assert word in r["summary"], r["summary"]
    s = _phys("spring", preset="washer", question="steady")
    m, k, c, F0, w = 60.0, 60000.0, 600.0, 100.0, 2 * math.pi * 5
    X = F0 / math.hypot(k - m * w * w, c * w)
    assert punits.text(X, "length", "cm") in s["summary"]
    assert "Resonance" in s["summary"]
    assert "driving force" in _phys("spring", preset="cart", question="steady")["error"]
    assert "no steady state" in _phys("spring", _vals(m=(1, "kg"), k=(16, "N/m"), F0=(2, "N"), w=(4, "rad/s")),
                                      {"driving": "on"}, "steady")["error"]


def test_spring_decay_and_inference_round_trips():
    values = _vals(m=(400, "kg"), k=(40, "kN/m"), x0=(5, "cm"), v0=(0, "m/s"), c=(3500, "kg/s"), fraction=(5, "%"))
    d = _phys("spring", values, {"damping": "on"}, "decay")
    t = d["physics"]["marker"]["t"]
    assert t == pytest.approx(-math.log(0.05) / (3500 / 800), rel=1e-9)
    # A measured decay gives the damping back.
    back = _phys("spring", _vals(m=(400, "kg"), k=(40, "kN/m"), fraction=(5, "%"), t_query=(t, "s")), {}, "infer_c")
    assert "3500 N" in back["summary"]
    # A measured period gives the spring constant back, with and without damping.
    wd = math.sqrt(40000 / 400 - (3500 / 800) ** 2)
    k_back = _phys("spring", _vals(m=(400, "kg"), c=(3500, "kg/s"), T_meas=(2 * math.pi / wd, "s")),
                   {"damping": "on"}, "infer_k")
    assert "40 kN/m" in k_back["summary"]
    k_plain = _phys("spring", _vals(m=(0.5, "kg"), T_meas=(2 * math.pi / math.sqrt(40), "s")), {}, "infer_k")
    assert "20 N/m" in k_plain["summary"]
    # Damping tending to zero recovers the undamped motion.
    tiny = _phys("spring", _vals(m=(0.5, "kg"), k=(20, "N/m"), x0=(0.1, "m"), v0=(0, "m/s"), c=(1e-9, "kg/s")),
                 {"damping": "on"}, "motion")
    w0 = math.sqrt(40)
    for tt, y in tiny["physics"]["curve"][::40]:
        assert abs(y - 0.1 * math.cos(w0 * tt)) < 1e-7
    assert "never shrinks" in _phys("spring", preset="cart", question="decay")["error"]


RC_CASES = [
    # R, C, V, vc0, source, r_int, r_leak, freq
    (2000.0, 120e-6, 300.0, 0.0, "battery", 0.0, 0.0, 0.0),
    (2000.0, 120e-6, 300.0, 50.0, "battery", 150.0, 0.0, 0.0),
    (2000.0, 120e-6, 300.0, 0.0, "battery", 0.0, 30000.0, 0.0),
    (2000.0, 120e-6, 300.0, 400.0, "battery", 100.0, 50000.0, 0.0),
    (50.0, 32e-6, 0.0, 5000.0, "discharge", 0.0, 0.0, 0.0),
    (50.0, 32e-6, 0.0, 5000.0, "discharge", 0.0, 400.0, 0.0),
    (1000.0, 100e-9, 1.0, 0.0, "ac", 0.0, 0.0, 5000.0),
    (1000.0, 100e-9, 2.0, 0.5, "ac", 200.0, 20000.0, 800.0),
]


@pytest.mark.parametrize("R,C,V,vc0,source,r_int,r_leak,freq", RC_CASES)
def test_rc_matches_independent_integration(R, C, V, vc0, source, r_int, r_leak, freq):
    values = _vals(R=(R, "ohm"), C=(C, "F"), V=(V, "V"), vc0=(vc0, "V"))
    effects = {"source": source}
    if r_int:
        values.update(_vals(r_int=(r_int, "ohm")))
        effects["internal"] = "on"
    if r_leak:
        values.update(_vals(r_leak=(r_leak, "ohm")))
        effects["leak"] = "on"
    if source == "ac":
        values.update(_vals(freq=(freq, "Hz")))
    r = _phys("rc", values, effects, "motion")
    assert r["ok"], r.get("error")
    Rt = R + (r_int if source != "discharge" else 0.0)
    w = 2 * math.pi * freq

    def deriv(t, st):
        vs = {"battery": V, "discharge": 0.0, "ac": V * math.cos(w * t)}[source]
        return [(vs - st[0]) / (Rt * C) - (st[0] / (r_leak * C) if r_leak else 0.0)]

    t_end = r["physics"]["t_max"]
    ref = _rk4(deriv, [vc0], t_end, 40000)
    _curve_matches(r, lambda t: _at(ref, t, 0))
    assert r["methods"][0]["verified"] is True
    assert (len(r["physics"]["ideal"]) > 0) == bool((r_int and source != "discharge") or r_leak)


def test_rc_times_energy_and_inference():
    flash = _vals(R=(2, "kohm"), C=(120, "uF"), V=(300, "V"), vc0=(0, "V"))
    tau = 2000 * 120e-6
    assert "240 ms" in _phys("rc", flash, {}, "tau")["summary"]
    frac = _phys("rc", dict(flash, **_vals(fraction=(95, "%"))), {}, "fraction")
    assert frac["physics"]["marker"]["t"] == pytest.approx(-tau * math.log(0.05), rel=1e-12)
    tgt = _phys("rc", dict(flash, **_vals(v_target=(200, "V"))), {}, "target")
    assert tgt["physics"]["marker"]["t"] == pytest.approx(tau * math.log(3.0), rel=1e-12)
    assert "never reaches" in _phys("rc", dict(flash, **_vals(v_target=(400, "V"))), {}, "target")["error"]
    # Energy: the battery's energy is the stored energy plus the heat, half each from empty.
    e = _phys("rc", flash, {}, "energy")["physics"]["energy"]
    assert e["supplied"] == pytest.approx(120e-6 * 300 ** 2, rel=1e-4)
    assert e["stored_final"] == pytest.approx(0.5 * 120e-6 * 300 ** 2, rel=1e-4)
    assert e["supplied"] == pytest.approx(e["heat"] + e["stored_final"] - e["stored_initial"], rel=1e-6)
    leaky = _phys("rc", dict(flash, **_vals(r_leak=(30, "kohm"), r_int=(100, "ohm"))),
                  {"leak": "on", "internal": "on"}, "energy")
    el = leaky["physics"]["energy"]
    assert el["supplied"] == pytest.approx(el["heat"] + el["leak"] + el["stored_final"], rel=1e-6)
    assert leaky["methods"][0]["verified"] is True
    d = _phys("rc", preset="defib", question="energy")["physics"]["energy"]
    assert d["heat"] == pytest.approx(0.5 * 32e-6 * 5000 ** 2, rel=1e-4)
    # Round trips: a measured charging time gives R and C back, also with the extra effects.
    t95 = -tau * math.log(0.05)
    assert "2 kΩ" in _phys("rc", _vals(C=(120, "uF"), fraction=(95, "%"), t_query=(t95, "s")), {}, "infer_r")["summary"]
    assert "120 µF" in _phys("rc", _vals(R=(2, "kohm"), fraction=(95, "%"), t_query=(t95, "s")), {}, "infer_c")["summary"]
    Rt, RL = 2100.0, 30000.0
    tau2 = Rt * RL / (Rt + RL) * 120e-6
    back = _phys("rc", _vals(C=(120, "uF"), fraction=(50, "%"), t_query=(tau2 * math.log(2), "s"),
                             r_int=(100, "ohm"), r_leak=(30, "kohm")), {"internal": "on", "leak": "on"}, "infer_r")
    assert "2 kΩ" in back["summary"]
    # A very large leakage resistance changes nothing.
    big = _phys("rc", dict(flash, **_vals(r_leak=(1e15, "ohm"))), {"leak": "on"}, "motion")
    plain = _phys("rc", flash, {}, "motion")
    assert big["physics"]["curve"][200][1] == pytest.approx(plain["physics"]["curve"][200][1], rel=1e-9)


def test_rc_ac_steady_state():
    r = _phys("rc", preset="filter", question="steady")
    tau, w = 1000 * 100e-9, 2 * math.pi * 5000
    amp = 1 / math.sqrt(1 + (w * tau) ** 2)
    assert punits.text(amp, "voltage", "V") in r["summary"]
    assert punits.text(math.atan(w * tau), "angle", "deg") in r["summary"]
    assert "1.592 kHz" in r["summary"]
    assert "AC source" in _phys("rc", preset="flash", question="steady")["error"]


def test_units_conversions_and_formatting():
    for quantity in punits.QUANTITIES:
        for choice in punits.choices(quantity):
            si = punits.to_si(3.7, quantity, choice["id"])
            assert punits.from_si(si, quantity, choice["id"]) == pytest.approx(3.7, rel=1e-12)
    assert punits.to_si(100, "temperature", "C") == pytest.approx(373.15)
    assert punits.to_si(212, "temperature", "F") == pytest.approx(373.15)
    assert punits.from_si(273.15, "temperature", "F") == pytest.approx(32.0)
    assert punits.to_si(60, "speed", "mph") == pytest.approx(26.8224)
    assert punits.to_si(1, "angular_frequency", "Hz") == pytest.approx(2 * math.pi)
    assert punits.text(1.2e-4, "capacitance") == "120 µF"
    assert punits.text(2400, "resistance") == "2.4 kΩ"
    assert punits.text(0.9999999, "time") == "1 s"
    assert punits.both(53.4, "speed", "km/h") == "192.2 km/h (53.4 m/s)"
    assert punits.number(123456789) == "1.235×10^8" and punits.number(0) == "0"
    with pytest.raises(Exception):
        punits.to_si(1, "mass", "V")


def test_physics_mixed_units_give_the_same_answer():
    si = _phys("falling", _vals(m=(80, "kg"), g=(9.81, "m/s2"), v0=(0, "m/s"), h0=(1000, "m"), b=(12, "kg/s")),
               {"drag": "linear"}, "ground")
    mixed = _phys("falling", _vals(m=(80000, "g"), g=(9.81 / 0.3048, "ft/s2"), v0=(0, "mph"), h0=(1, "km"),
                                   b=(12000, "g/s")), {"drag": "linear"}, "ground")
    assert mixed["physics"]["t_max"] == pytest.approx(si["physics"]["t_max"], rel=1e-9)
    assert "km/h" not in si["summary"]
    kmh = _phys("falling", _vals(m=(80, "kg"), g=(9.81, "m/s2"), v0=(0, "km/h"), h0=(1000, "m"), b=(12, "kg/s")),
                {"drag": "linear"}, "ground")
    assert "km/h" in kmh["summary"] and "m/s" in kmh["summary"]      # shown in the user's unit and in SI


@pytest.mark.parametrize("scenario,values,effects,question,fragment", [
    ("falling", _vals(g=(9.81, "m/s2"), v0=(0, "m/s"), h0=(50, "m")), {}, "ground", "fill in mass"),
    ("falling", _vals(m=(1, "kg"), g=(9.81, "m/s2"), v0=(0, "m/s")), {}, "ground", "fill in starting height"),
    ("falling", _vals(m=(1, "kg"), g=(9.81, "m/s2"), v0=(0, "m/s"), h0=(5, "m")), {"drag": "linear"}, "ground",
     "fill in linear drag coefficient"),
    ("falling", _vals(m=(1, "kg"), g=(9.81, "m/s2"), v0=(0, "m/s"), h0=(5, "m")), {"drag": "quadratic"}, "ground",
     "fill in air density"),
    ("falling", _vals(m=(-1, "kg"), g=(9.81, "m/s2"), v0=(0, "m/s"), h0=(5, "m")), {}, "ground", "greater than zero"),
    ("falling", _vals(m=(1, "kg"), g=(9.81, "m/s2")), {}, "terminal", "no terminal velocity"),
    ("falling", _vals(m=("abc", "kg"), g=(9.81, "m/s2")), {}, "ground", "Mass must be a number"),
    ("falling", _vals(m=(1, "V"), g=(9.81, "m/s2")), {}, "ground", "cannot be used for a mass"),
    ("falling", _vals(m=(0.01, "kg"), g=(9.81, "m/s2"), v0=(0, "m/s"), h0=(50, "m"), b=(0.05, "kg/s"),
                      rho_f=(1.225, "kg/m3"), vol=(0.02, "m3")), {"drag": "linear", "buoyancy": "on"}, "ground",
     "never reaches the ground"),
    ("spring", _vals(m=(1, "kg")), {}, "motion", "fill in spring constant"),
    ("spring", _vals(m=(1, "kg"), k=(4, "N/m"), x0=(0.1, "m"), v0=(0, "m/s")), {"damping": "on"}, "motion",
     "fill in damping coefficient"),
    ("rc", _vals(R=(1, "kohm"), V=(5, "V"), vc0=(0, "V")), {}, "motion", "fill in capacitance"),
    ("rc", _vals(R=(1, "kohm"), C=(1, "uF"), V=(0, "V"), vc0=(0, "V")), {"source": "discharge"}, "motion",
     "must start with some voltage"),
    ("rc", _vals(R=(1, "kohm"), C=(1, "uF"), V=(5, "V"), vc0=(0, "V")), {"source": "ac"}, "motion",
     "fill in source frequency"),
])
def test_physics_missing_or_bad_inputs_name_the_field(scenario, values, effects, question, fragment):
    r = _phys(scenario, values, effects, question)
    assert not r["ok"] and fragment in r["error"], r.get("error")
    assert "Traceback" not in r["error"]


def test_physics_catalog_and_equation_preview():
    cat = json.loads(physics_catalog())
    assert [s["id"] for s in cat["scenarios"]] == ["falling", "spring", "rc"]
    assert len(cat["upcoming"]) >= 5
    for sc in cat["scenarios"]:
        ids = {f["id"] for f in sc["fields"]}
        assert all(f["units"] and f["group"] in ("base", "effect", "param") for f in sc["fields"])
        for e in sc["effects"]:
            assert e["default"] in [o["id"] for o in e["options"]]
            assert all(fid in ids for o in e["options"] for fid in o["fields"])
        for q in sc["questions"]:
            assert set(q["params"]) | set(q["finds"]) | set(q["hides"]) <= ids
        for p in sc["presets"]:
            assert set(p["values"]) <= ids
            for q in sc["questions"]:        # every preset with every question solves or explains why not
                r = _phys(sc["id"], preset=p["id"], question=q["id"])
                assert r["ok"] or (r["error"] and "went wrong" not in r["error"]), (sc["id"], p["id"], q["id"], r)
                if r["ok"]:
                    assert r["kind"] == "physics" and r["physics"]["curve"] and r["physics"]["animation"]["points"]
                    assert r["methods"][0]["steps"][0]["title"].startswith("Set up the equation")
                    assert r["methods"][0]["steps"][-1]["title"] == "Check"
    assert json.loads(preview('{"effects": {"drag": "quadratic", "buoyancy": "on"}}', "physics:falling"))["latex"] == \
        "m\\frac{dv}{dt} = m g - \\rho_f V g - c\\,v\\,|v|"
    assert "c\\frac{dx}{dt}" in json.loads(preview('{"effects": {"damping": "on"}}', "physics:spring"))["latex"]
    assert "R_L" in json.loads(preview('{"effects": {"leak": "on"}}', "physics:rc"))["latex"]
    assert "Unknown physics scenario" in json.loads(physics("warp", ""))["error"]
    assert "could not be read" in json.loads(physics("rc", "{oops"))["error"]


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
    extra = {
        "pde_characteristics.json": ("characteristics", "e^(-x^2)", "", "2", "1", "", "", ""),
        "pde_classify.json": ("classify", "", "", "y", "0", "", "1", ""),
        "pde_fd.json": ("fd", "sin(pi x)", "10", "1", "1", "", "4", "0.02"),
        "pde_disk.json": ("disk", "u(t)", "", "1"),
    }
    for name, args in extra.items():
        with open(os.path.join(out, name), "w") as fh:
            fh.write(pde(*args))
    with open(os.path.join(out, "fourier.json"), "w") as fh:
        fh.write(fourier("x", "", "pi"))
    with open(os.path.join(out, "boundary.json"), "w") as fh:
        fh.write(boundary("y'' + y = 0", "0", "y", "0", "pi/2", "y", "1"))
    with open(os.path.join(out, "eigen.json"), "w") as fh:
        fh.write(eigen_problem("dd", "pi"))
    for name, args in {
        "tool_linalg.json": ("linalg", "solve", "2 1; 1 3", "5 10"),
        "tool_calculus.json": ("calculus", "integral", "x cos(x^2)"),
        "tool_series.json": ("series", "taylor", "sin(x)", "", "7"),
        "tool_complex.json": ("complex", "roots", "-8", "3"),
        "tool_special.json": ("special", "legendre", "3"),
        "tool_bessel.json": ("special", "bessel", "0"),
        "tool_ftransform.json": ("ftransform", "heat", "u(x+1) - u(x-1)", "1"),
        "tool_recurrence.json": ("recurrence", "solve", "a(n+2) = a(n+1) + a(n)", "0, 1"),
        "tool_dynamics.json": ("dynamics", "cycles", "x - y - x(x^2+y^2); x + y - y(x^2+y^2)"),
        "tool_bifurcation.json": ("dynamics", "bifurcation", "m x - x^3"),
        "tool_vector.json": ("vector", "line", "-y; x", "cos(t); sin(t)", "0, pi"),
        "tool_gradient.json": ("vector", "gradient", "x^2 - y^2", "1, 1"),
    }.items():
        with open(os.path.join(out, name), "w") as fh:
            fh.write(tool(*args))
    with open(os.path.join(out, "physics_catalog.json"), "w") as fh:
        fh.write(physics_catalog())
    with open(os.path.join(out, "physics_falling.json"), "w") as fh:
        fh.write(physics("falling", json.dumps({"preset": "skydiver", "question": "ground"})))
    with open(os.path.join(out, "physics_spring.json"), "w") as fh:
        fh.write(physics("spring", json.dumps({"preset": "car", "question": "motion"})))
    with open(os.path.join(out, "physics_rc.json"), "w") as fh:
        fh.write(physics("rc", json.dumps({"preset": "flash", "question": "fraction"})))
