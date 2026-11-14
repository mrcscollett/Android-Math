"""Exponential decay and growth: radioactivity, drug elimination, populations."""
import math

from ..first_order import step
from .solver import Rel, Skip
from .spec import choice, field, preset

MODEL = {
    "id": "decay", "title": "Exponential decay and growth", "animation": "capacitor",
    "description": "A quantity that changes in proportion to itself: radioactive decay, a drug leaving the body, bacteria multiplying.",
    "law": "The rate of change is proportional to the amount present.",
    "fields": [
        field("lam", "Rate constant", "\\lambda", "rate", (None, "1/y"), positive=True, typical=1e-6),
        field("th", "Half-life (or doubling time)", "t_{1/2}", "time", (None, "y"), "derived", positive=True, typical=1e5),
        field("tau", "Mean lifetime (time constant)", "\\tau", "time", (None, "y"), "derived", positive=True, typical=1e5),
        field("N0", "Starting amount", "N_0", "amount", (None, ""), "initial", positive=True, optional=True),
        field("t", "Time", "t", "time", (None, "y"), "moment", nonnegative=True, typical=1e5),
        field("frac", "Fraction of the start present", "N/N_0", "dimensionless", (None, "%"), "moment", positive=True),
        field("N", "Amount at that time", "N", "amount", (None, ""), "moment", positive=True, optional=True),
        field("t2", "Second time", "t_2", "time", (None, "y"), "moment2", nonnegative=True, typical=1e5),
        field("N2", "Amount at the second time", "N_2", "amount", (None, ""), "moment2", positive=True),
    ],
    "effects": [
        choice("kind", "Direction", "Whether the amount shrinks or grows.",
               [("decay", "Decay", []), ("growth", "Growth", [])], "decay"),
    ],
    "presets": [
        preset("carbon", "Carbon dating", "A bone has 25% of its original carbon-14 (half-life 5730 years): how old is it?",
               {"th": ("5730", "y"), "frac": ("25", "%")}, {"kind": "decay"}),
        preset("iodine", "Iodine-131 dose", "400 MBq of iodine-131 (half-life 8.02 days): how much is left after 30 days?",
               {"th": ("8.02", "d"), "N0": ("400", "MBq"), "t": ("30", "d"), "N": ("", "MBq"),
                "lam": ("", "1/d"), "tau": ("", "d")}, {"kind": "decay"}),
        preset("drug", "Drug elimination", "A 200 mg dose is down to 150 mg after 2 hours: what is left after 12 hours?",
               {"N0": ("200", "mg"), "t": ("2", "h"), "N": ("150", "mg"), "t2": ("12", "h"), "N2": ("", "mg"),
                "th": ("", "h"), "tau": ("", "h"), "lam": ("", "1/h")}, {"kind": "decay"}),
        preset("bacteria", "Bacteria doubling", "1000 bacteria doubling every 20 minutes: when are there a million?",
               {"th": ("20", "min"), "N0": ("1000", ""), "N": ("1000000", ""), "t": ("", "min"), "frac": ("", ""),
                "lam": ("", "1/min"), "tau": ("", "min")}, {"kind": "growth"}),
    ],
}


def _sign(eff):
    return 1.0 if eff.get("kind") == "growth" else -1.0


def trajectory(eff, v):
    if v.get("lam") is None:
        raise Skip()
    lam, sign = v["lam"], _sign(eff)
    scale = (3 if sign > 0 else 5) / lam
    return {"at": lambda t: {"frac": math.exp(sign * lam * t)}, "exact": True, "scale": scale,
            "ode": (lambda t, s: [sign * lam * s[0]], [1.0], ["frac"], scale)}


def relations(eff):
    sign = _sign(eff)
    sg = "" if sign > 0 else "-"
    word = "doubles" if sign > 0 else "halves"
    ratio = lambda a, b: "\\ln\\frac{%s}{%s}" % ((a, b) if sign > 0 else (b, a))
    return [
        Rel("th", "The amount %s in a fixed time" % word, "t_{1/2} = \\frac{\\ln 2}{\\lambda}", ["th", "lam"],
            lambda v: (v["th"], math.log(2) / v["lam"]),
            {"th": "t_{1/2} = \\frac{\\ln 2}{\\lambda}", "lam": "\\lambda = \\frac{\\ln 2}{t_{1/2}}"}),
        Rel("tau", "The time constant is the reciprocal of the rate constant", "\\tau = 1/\\lambda", ["tau", "lam"],
            lambda v: (v["tau"], 1 / v["lam"]),
            {"tau": "\\tau = \\frac{1}{\\lambda}", "lam": "\\lambda = \\frac{1}{\\tau}"}),
        Rel("frac", "The fraction present at time t from the solution of the rate equation",
            "\\frac{N}{N_0} = e^{%s\\lambda t}" % sg, ["frac", "lam", "t"],
            lambda v: (math.log(v["frac"]) + 50.0, sign * v["lam"] * v["t"] + 50.0),
            {"frac": "\\frac{N}{N_0} = e^{%s\\lambda t}" % sg,
             "t": "t = \\frac{1}{\\lambda}%s" % ratio("N", "N_0"),
             "lam": "\\lambda = \\frac{1}{t}%s" % ratio("N", "N_0")}, traj=True),
        Rel("N", "The amount is that fraction of the starting amount", "N = N_0\\cdot\\frac{N}{N_0}",
            ["N", "N0", "frac"], lambda v: (v["N"], v["N0"] * v["frac"]),
            {"N": "N = N_0\\,e^{%s\\lambda t}" % sg, "N0": "N_0 = \\frac{N}{N/N_0}", "frac": "\\frac{N}{N_0} = \\frac{N}{N_0}"}),
        Rel("N2", "The same solution at the second time", "N_2 = N_0\\,e^{%s\\lambda t_2}" % sg,
            ["N2", "N0", "lam", "t2"],
            lambda v: (math.log(v["N2"] / v["N0"]) + 50.0, sign * v["lam"] * v["t2"] + 50.0),
            {"N2": "N_2 = N_0\\,e^{%s\\lambda t_2}" % sg, "t2": "t_2 = \\frac{1}{\\lambda}%s" % ratio("N_2", "N_0"),
             "lam": "\\lambda = \\frac{1}{t_2}%s" % ratio("N_2", "N_0")}, traj=True),
    ]


def equation(eff):
    return "\\frac{dN}{dt} = %s\\lambda\\,N" % ("" if _sign(eff) > 0 else "-")


def derivation(eff):
    growth = _sign(eff) > 0
    sg = "" if growth else "-"
    return [
        step("Set up the equation",
             "Each unit of the quantity has the same chance of %s per unit time, so the total rate of "
             "change is proportional to the amount present." % ("reproducing" if growth else "decaying"),
             [equation(eff)]),
        step("Solve the equation", "The equation is separable: divide by \\(N\\) and integrate both sides.",
             ["\\int \\frac{dN}{N} = %s\\int \\lambda\\,dt" % sg, "\\ln N = %s\\lambda t + C" % sg,
              "N(t) = N_0\\,e^{%s\\lambda t}" % sg]),
    ]


def horizon(eff, v):
    return (40 if _sign(eff) > 0 else 12) / v["lam"] if v.get("lam") else None


def graph(eff, v):
    traj = trajectory(eff, v)
    times = [x for x in (v.get("t"), v.get("t2")) if x is not None]
    t_max = max([traj["scale"] * 0.8] + [x * 1.2 for x in times])
    n0 = v.get("N0")
    scale = n0 if n0 else 100.0
    marker = None
    if v.get("t") is not None and v.get("frac") is not None:
        marker = {"t": v["t"], "y": scale * v["frac"], "label": "at time t"}
    return {"fn": lambda t: scale * traj["at"](t)["frac"], "ideal": None, "t_max": t_max,
            "y_label": "amount" if n0 else "percent of the start", "label": "Amount", "marker": marker}
