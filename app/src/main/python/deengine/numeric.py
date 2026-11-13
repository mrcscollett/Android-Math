"""Numerical methods for y' = f(x, y), y(x0) = y0: Euler, improved Euler, RK4.

Each method returns its formula, the first step worked with numbers, and a
table of every step, compared with the exact solution when one is known.
"""
import math

import sympy as sp

from . import first_order
from .first_order import step
from .util import plot_expr, tex

MAX_STEPS = 1000
MAX_ROWS = 60


def fmt(value):
    if value is None:
        return ""
    if value == 0:
        return "0"
    if abs(value) >= 1e6 or abs(value) < 1e-4:
        mantissa, exponent = ("%.4e" % value).split("e")
        return "%s \\times 10^{%d}" % (mantissa.rstrip("0").rstrip("."), int(exponent))
    return ("%.6f" % value).rstrip("0").rstrip(".")


def _exact_solution(prob, f, ic):
    """A verified closed-form solution of the IVP, if a symbolic method finds one."""
    for method in (first_order.separable, first_order.linear):
        try:
            out = method(prob, f, ic)
        except Exception:
            continue
        if out.get("applicable") and out.get("verified") and out["plot"]["particular_expr"]:
            expr = sp.sympify(out["plot"]["particular_expr"], locals={"x": prob.x})
            try:
                return expr, sp.lambdify(prob.x, expr, "math"), out["particular_latex"]
            except Exception:
                continue
    return None, None, None


def _run(stepper, fn, x0, y0, h, n):
    """Returns rows of (i, x, y, slopes) and a note if it had to stop early."""
    rows, x, y, note = [], x0, y0, None
    for i in range(n + 1):
        if i == n:
            rows.append((i, x, y, None))
            break
        try:
            y_next, slopes = stepper(fn, x, y, h)
            if not math.isfinite(y_next) or abs(y_next) > 1e150:
                raise OverflowError
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            rows.append((i, x, y, None))
            note = ("The method stopped at step %d: the next value is undefined "
                    "or too large (the solution may blow up there)." % i)
            break
        rows.append((i, x, y, slopes))
        x, y = x0 + (i + 1) * h, y_next
    return rows, note


def _euler(fn, x, y, h):
    k1 = fn(x, y)
    return y + h * k1, (k1,)


def _heun(fn, x, y, h):
    k1 = fn(x, y)
    k2 = fn(x + h, y + h * k1)
    return y + h * (k1 + k2) / 2, (k1, k2)


def _rk4(fn, x, y, h):
    k1 = fn(x, y)
    k2 = fn(x + h / 2, y + h * k1 / 2)
    k3 = fn(x + h / 2, y + h * k2 / 2)
    k4 = fn(x + h, y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6, (k1, k2, k3, k4)


def _midpoint(fn, x, y, h):
    k1 = fn(x, y)
    k2 = fn(x + h / 2, y + h * k1 / 2)
    return y + h * k2, (k1, k2)


def _ab4_runner(fn, x0, y0, h, n):
    """Adams-Bashforth 4-step, started with three RK4 steps."""
    rows, note = [], None
    x, y = x0, y0
    history = []
    for i in range(n + 1):
        if i == n:
            rows.append((i, x, y, None))
            break
        try:
            f_now = fn(x, y)
            history.append(f_now)
            if len(history) < 4:
                y_next, _ = _rk4(fn, x, y, h)
            else:
                f0, f1, f2, f3 = history[-1], history[-2], history[-3], history[-4]
                y_next = y + h * (55 * f0 - 59 * f1 + 37 * f2 - 9 * f3) / 24
            if not math.isfinite(y_next) or abs(y_next) > 1e150:
                raise OverflowError
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            rows.append((i, x, y, None))
            note = ("The method stopped at step %d: the next value is undefined "
                    "or too large (the solution may blow up there)." % i)
            break
        rows.append((i, x, y, (f_now,)))
        x, y = x0 + (i + 1) * h, y_next
    return rows, note


TOLERANCE = 1e-6


def _rkf45_runner(fn, x0, y0, h, n):
    """Runge-Kutta-Fehlberg with step-size control, from x0 to x0 + n h."""
    rows, note = [], None
    end = x0 + n * h
    direction = 1.0 if h > 0 else -1.0
    step_size = h
    x, y = x0, y0
    i = 0
    while (end - x) * direction > 1e-12 * max(1.0, abs(end)):
        if i >= MAX_STEPS:
            note = "Stopped after %d steps without reaching the end point." % MAX_STEPS
            break
        if (x + step_size - end) * direction > 0:
            step_size = end - x
        try:
            k1 = fn(x, y)
            k2 = fn(x + step_size / 4, y + step_size * k1 / 4)
            k3 = fn(x + 3 * step_size / 8, y + step_size * (3 * k1 + 9 * k2) / 32)
            k4 = fn(x + 12 * step_size / 13,
                    y + step_size * (1932 * k1 - 7200 * k2 + 7296 * k3) / 2197)
            k5 = fn(x + step_size, y + step_size * (439 * k1 / 216 - 8 * k2
                                                     + 3680 * k3 / 513 - 845 * k4 / 4104))
            k6 = fn(x + step_size / 2, y + step_size * (-8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565
                                                         + 1859 * k4 / 4104 - 11 * k5 / 40))
            y4 = y + step_size * (25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5)
            y5 = y + step_size * (16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430
                                  - 9 * k5 / 50 + 2 * k6 / 55)
            error = abs(y5 - y4)
            if not math.isfinite(y5) or abs(y5) > 1e150:
                raise OverflowError
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            if abs(step_size) > 1e-10:
                step_size /= 4
                continue
            note = ("The method stopped near %s: the solution is undefined or "
                    "blows up there." % fmt(x))
            break
        allowed = TOLERANCE * max(1.0, abs(y))
        if error <= allowed or abs(step_size) < 1e-10:
            rows.append((i, x, y, (step_size, error)))
            x, y = x + step_size, y5
            i += 1
        factor = 4.0 if error == 0 else min(4.0, max(0.1, 0.84 * (allowed / error) ** 0.25))
        step_size *= factor
    rows.append((i, x, y, None))
    return rows, note


_ab4_runner.columns = ["f_n"]
_rkf45_runner.columns = ["h_n", "\\text{error est.}"]


_SPECS = [
    ("euler", "Euler", _euler,
     "Follows the slope at the start of each step. Simple, but the error is "
     "proportional to the step size \\(h\\).",
     ["y_{n+1} = y_n + h\\,f(x_n, y_n)"],
     ["k_1 = f(x_n, y_n)"]),
    ("heun", "Improved Euler (Heun)", _heun,
     "Averages the slope at the start of the step with the slope at the "
     "predicted end. The error is proportional to \\(h^2\\).",
     ["k_1 = f(x_n, y_n)", "k_2 = f(x_n + h,\\ y_n + h\\,k_1)",
      "y_{n+1} = y_n + \\tfrac{h}{2}\\,(k_1 + k_2)"],
     ["k_1", "k_2"]),
    ("rk4", "Runge–Kutta (RK4)", _rk4,
     "Combines four slope samples per step. The error is proportional to "
     "\\(h^4\\), so it is far more accurate for the same step size.",
     ["k_1 = f(x_n, y_n)", "k_2 = f(x_n + \\tfrac{h}{2},\\ y_n + \\tfrac{h}{2}k_1)",
      "k_3 = f(x_n + \\tfrac{h}{2},\\ y_n + \\tfrac{h}{2}k_2)",
      "k_4 = f(x_n + h,\\ y_n + h\\,k_3)",
      "y_{n+1} = y_n + \\tfrac{h}{6}\\,(k_1 + 2k_2 + 2k_3 + k_4)"],
     ["k_1", "k_2", "k_3", "k_4"]),
    ("midpoint", "Midpoint (RK2)", _midpoint,
     "Takes a half step to estimate the slope at the middle of the interval "
     "and uses that slope for the whole step. The error is proportional to \\(h^2\\).",
     ["k_1 = f(x_n, y_n)", "k_2 = f(x_n + \\tfrac{h}{2},\\ y_n + \\tfrac{h}{2}k_1)",
      "y_{n+1} = y_n + h\\,k_2"],
     ["k_1", "k_2"]),
    ("ab4", "Adams\u2013Bashforth (4-step)", _ab4_runner,
     "A multistep method: it reuses the slopes from the last four points "
     "instead of sampling new ones, so each step costs one evaluation of "
     "\\(f\\). The first three steps are taken with RK4 to get it started. "
     "The error is proportional to \\(h^4\\).",
     ["f_n = f(x_n, y_n)",
      "y_{n+1} = y_n + \\tfrac{h}{24}\\,(55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3})"],
     []),
    ("rkf45", "Adaptive RK45 (Fehlberg)", _rkf45_runner,
     "Computes a fourth- and a fifth-order estimate from the same six slopes. "
     "Their difference estimates the error, and the step size is shrunk or "
     "grown to keep it below a tolerance of \\(10^{-6}\\). Your \\(h\\) is "
     "only the first trial step.",
     ["\\text{error} \\approx |y^{(5)}_{n+1} - y^{(4)}_{n+1}|",
      "h_{\\text{new}} = 0.84\\,h\\left(\\frac{\\text{tolerance}}{\\text{error}}\\right)^{1/4}"],
     []),
]


def solve(prob, f, ic, h, n, input_latex):
    x0, y0 = float(ic[0]), float(ic[1])
    xn = prob.indep_name
    fn = sp.lambdify((prob.x, prob.y), f, "math")
    exact_expr, exact_fn, exact_latex = _exact_solution(prob, f, ic)

    def exact_at(x):
        if exact_fn is None:
            return None
        try:
            value = float(exact_fn(x))
            return value if math.isfinite(value) else None
        except Exception:
            return None

    # If the exact solution blows up inside the interval, anything a method
    # reports beyond that point is meaningless, so stop there.
    blow_up = None
    if exact_fn is not None:
        limit = 1e8 * (1 + abs(y0))
        for i in range(n * 8 + 1):
            xs = x0 + h * i / 8
            value = exact_at(xs)
            if value is None or abs(value) > limit:
                blow_up = xs
                break
    usable = n if blow_up is None else max(1, int(abs((blow_up - x0) / h)) - 0)
    usable = min(n, usable)
    if blow_up is not None and abs(x0 + usable * h - x0) >= abs(blow_up - x0):
        usable = max(1, usable - 1)

    methods = []
    for method_id, name, stepper, about, formula, slope_names in _SPECS:
        custom = hasattr(stepper, "columns")
        if custom:
            rows, note = stepper(fn, x0, y0, h, usable)
        else:
            rows, note = _run(stepper, fn, x0, y0, h, usable)
        if blow_up is not None and note is None:
            note = ("The exact solution blows up near \\(%s = %s\\), so the "
                    "table stops before that point. No step-by-step method "
                    "can follow a solution through a blow-up." % (xn, fmt(blow_up)))
        if custom:
            slope_headers = stepper.columns
        else:
            slope_headers = ["f(%s_n, y_n)" % xn] if method_id == "euler" else slope_names
        headers = ["n", "%s_n" % xn, "y_n"] + slope_headers
        if exact_fn is not None:
            headers += ["\\text{exact}", "\\text{error}"]
        table_rows = []
        for i, x, y, slopes in _thin(rows):
            cells = [str(i), fmt(x), fmt(y)]
            cells += [fmt(k) for k in slopes] if slopes else [""] * len(slope_headers)
            if exact_fn is not None:
                e = exact_at(x)
                cells += [fmt(e), fmt(abs(y - e)) if e is not None else ""]
            table_rows.append(cells)

        steps = [step("The method", "Starting from \\((%s_0, y_0)\\), repeat with step size \\(h\\):" % xn,
                      [f_line.replace("x_n", "%s_n" % xn) for f_line in formula]
                      + ["f(%s, y) = %s,\\qquad h = %s,\\qquad (%s_0, y_0) = (%s, %s)"
                         % (xn, tex(f), fmt(h), xn, fmt(x0), fmt(y0))])]
        if len(rows) > 1 and rows[0][3] and not custom:
            slopes = rows[0][3]
            lines = ["%s = %s" % (label.split(" = ")[0] if method_id != "euler" else "f(%s_0, y_0)" % xn, fmt(k))
                     for label, k in zip(slope_names, slopes)]
            lines.append("y_1 = %s" % fmt(rows[1][2]))
            steps.append(step("First step with numbers",
                              "Put \\(%s_0 = %s\\), \\(y_0 = %s\\) and \\(h = %s\\) into the formulas."
                              % (xn, fmt(x0), fmt(y0), fmt(h)), lines))
        table_step = step("All steps",
                          "Each row uses the row above it." + (
                              " Long tables show a selection of rows." if len(rows) > MAX_ROWS else ""))
        table_step["table"] = {"headers": headers, "rows": table_rows}
        steps.append(table_step)

        last = rows[-1]
        lines = ["y(%s) \\approx %s" % (fmt(last[1]), fmt(last[2]))]
        text = note or ""
        e = exact_at(last[1])
        if e is not None:
            lines.append("\\text{exact: } %s,\\qquad \\text{error: } %s" % (fmt(e), fmt(abs(last[2] - e))))
        result = step("Result", text, lines)
        result["kind"] = "answer"
        steps.append(result)

        methods.append({
            "id": method_id, "name": name, "applicable": True, "reason": about,
            "steps": steps,
            "general_latex": None, "particular_latex": exact_latex,
            "plot": {
                "general_expr": None,
                "particular_expr": plot_expr(exact_expr, prob) if exact_expr is not None else None,
                "points": [[r[1], r[2]] for r in rows],
            },
            "verified": None, "verify_note": note or "",
        })

    summary = ("Numerical solution with step size %s over %d steps. Switch "
               "between the methods to compare them." % (fmt(h), n))
    if exact_fn is not None:
        summary += " The exact solution is known, so each table shows the error."
    if blow_up is not None:
        summary += " The exact solution blows up near %s = %s." % (xn, fmt(blow_up))
    return {
        "schema": 1, "ok": True, "kind": "numeric", "order": 1,
        "indep": xn, "dep": "y",
        "input_latex": input_latex,
        "normal_latex": "%s' = %s" % ("y", tex(f)),
        "rhs_expr": plot_expr(f, prob),
        "ic": {"x0": x0, "y0": y0, "latex": "y(%s) = %s" % (tex(ic[0]), tex(ic[1]))},
        "summary": summary, "other_types": [], "methods": methods,
    }


def _thin(rows):
    if len(rows) <= MAX_ROWS:
        return rows
    stride = math.ceil((len(rows) - 11) / (MAX_ROWS - 11))
    picked = rows[:10] + rows[10:-1:stride]
    if picked[-1] is not rows[-1]:
        picked.append(rows[-1])
    return picked
