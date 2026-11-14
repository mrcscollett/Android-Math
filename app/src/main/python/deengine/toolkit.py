"""Shared pieces for the stand-alone math tools (linear algebra, calculus, series, complex numbers)."""
from .first_order import step  # noqa: F401  (re-exported for the tool modules)


def method(method_id, name, reason, steps, answer=None, plot=None, verified=None, note=""):
    """A finished method. `answer` is the LaTeX shown in the Result step."""
    data = {"general_expr": None, "particular_expr": None, "points": []}
    data.update(plot or {})
    return {"id": method_id, "name": name, "applicable": True, "reason": reason,
            "steps": steps, "general_latex": None, "particular_latex": answer,
            "plot": data, "verified": verified, "verify_note": note}


def not_applicable(method_id, name, reason):
    return {"id": method_id, "name": name, "applicable": False, "reason": reason}


def result(tool, title, input_latex, summary, methods, **extra):
    """The result JSON for a tool; appends the Result and Check steps to each method."""
    for m in methods:
        if not m["applicable"]:
            continue
        if m["particular_latex"]:
            answers = m["particular_latex"] if isinstance(m["particular_latex"], list) else [m["particular_latex"]]
            m["steps"].append({"title": "Result", "text": "", "latex": answers,
                               "substeps": [], "kind": "answer"})
            m["particular_latex"] = answers[0]
        if m["verify_note"]:
            kind = {True: "check-ok", False: "check-bad", None: "check-unknown"}[m["verified"]]
            m["steps"].append({"title": "Check", "text": m["verify_note"], "latex": [],
                               "substeps": [], "kind": kind})
    out = {"schema": 1, "ok": True, "kind": "tool", "tool": tool, "title": title, "order": 0,
           "indep": "x", "dep": "", "input_latex": input_latex, "normal_latex": input_latex,
           "rhs_expr": None, "ic": None, "summary": summary, "other_types": [],
           "methods": methods}
    out.update(extra)
    return out
