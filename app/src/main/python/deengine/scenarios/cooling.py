"""Newton's law of cooling (or heating), with an optional steady heat source."""
import math

from ..first_order import step
from .solver import Rel, Skip
from .spec import field, preset, switch

MODEL = {
    "id": "cooling", "title": "Cooling and heating", "animation": "capacitor",
    "description": "An object settling to the temperature of its surroundings: coffee cooling, a roast heating, a body after death.",
    "law": "Newton's law of cooling: the temperature changes at a rate proportional to its difference from the surroundings.",
    "fields": [
        field("T0", "Starting temperature", "T_0", "temperature", (None, "C"), "initial", typical=300),
        field("Ta", "Surrounding temperature", "T_a", "temperature", (None, "C"), typical=300),
        field("k", "Cooling constant", "k", "rate", (None, "1/min"), positive=True, typical=1e-3),
        field("s", "Heating rate of the source", "s", "temperature_rate", (None, "K/min"), "effect",
              positive=True, typical=0.01),
        field("Tf", "Final temperature", "T_f", "temperature", (None, "C"), "effect", typical=300),
        field("t", "Time", "t", "time", (None, "min"), "moment", nonnegative=True, typical=600),
        field("T", "Temperature at that time", "T", "temperature", (None, "C"), "moment", typical=300),
        field("t2", "Second time", "t_2", "time", (None, "min"), "moment2", nonnegative=True, typical=600),
        field("T2", "Temperature at the second time", "T_2", "temperature", (None, "C"), "moment2", typical=300),
        field("tau", "Time constant", "\\tau", "time", (None, "min"), "derived", positive=True, typical=600),
        field("th", "Time for the difference to halve", "t_{1/2}", "time", (None, "min"), "derived",
              positive=True, typical=600),
    ],
    "effects": [
        switch("source", "Heat source", "A heater (or the object's own heat) warming it at a steady rate.",
               ["s", "Tf"]),
    ],
    "presets": [
        preset("coffee", "Cooling coffee", "Coffee at 90 \u00b0C in a 22 \u00b0C room is 75 \u00b0C after 5 minutes: when is it 60 \u00b0C?",
               {"T0": ("90", "C"), "Ta": ("22", "C"), "t": ("5", "min"), "T": ("75", "C"), "T2": ("60", "C")}, {}),
        preset("forensic", "Time since death", "A body at 29.5 \u00b0C in an 18 \u00b0C room, cooling constant 0.19 per hour.",
               {"T0": ("37", "C"), "Ta": ("18", "C"), "k": ("0.19", "1/h"), "T": ("29.5", "C")}, {}),
        preset("roast", "Roast in the oven", "A 5 \u00b0C roast in a 180 \u00b0C oven reaches 40 \u00b0C in 30 min: when is it 70 \u00b0C?",
               {"T0": ("5", "C"), "Ta": ("180", "C"), "t": ("30", "min"), "T": ("40", "C"), "T2": ("70", "C")}, {}),
        preset("chip", "Processor warming up", "A chip in 25 \u00b0C air heats at 3 \u00b0C/s with cooling constant 0.05 per second.",
               {"T0": ("25", "C"), "Ta": ("25", "C"), "k": ("0.05", "1/s"), "s": ("180", "K/min"), "t": ("30", "s")},
               {"source": "on"}),
    ],
}


def _final(eff, v):
    return v["Ta"] + (v["s"] / v["k"] if eff.get("source") == "on" else 0.0)


def _params(eff):
    return ["T0", "Ta", "k"] + (["s"] if eff.get("source") == "on" else [])


def trajectory(eff, v):
    if any(v.get(x) is None for x in _params(eff)):
        raise Skip()
    tf, t0, k = _final(eff, v), v["T0"], v["k"]
    s = v["s"] if eff.get("source") == "on" else 0.0
    return {"at": lambda t: {"T": tf + (t0 - tf) * math.exp(-k * t)}, "exact": True, "scale": 5 / k,
            "ode": (lambda t, st: [-k * (st[0] - v["Ta"]) + s], [t0], ["T"], 5 / k)}


def _reading(eff, tvar, Tvar, label):
    source = eff.get("source") == "on"
    target = "T_f" if source else "T_a"

    def fn(v):
        tf = _final(eff, v)
        size = 2 * abs(v["T0"] - tf) + 1e-9
        return v[Tvar] - tf + size, (v["T0"] - tf) * math.exp(-v["k"] * v[tvar]) + size

    sym_t, sym_T = ("t", "T") if tvar == "t" else ("t_2", "T_2")
    forms = {Tvar: "%s = %s + (T_0 - %s)\\,e^{-k %s}" % (sym_T, target, target, sym_t),
             tvar: "%s = \\frac{1}{k}\\ln\\frac{T_0 - %s}{%s - %s}" % (sym_t, target, sym_T, target)}
    if not source:
        forms["k"] = "k = \\frac{1}{%s}\\ln\\frac{T_0 - T_a}{%s - T_a}" % (sym_t, sym_T)
        forms["T0"] = "T_0 = T_a + (%s - T_a)\\,e^{k %s}" % (sym_T, sym_t)
    return Rel("T" + ("" if tvar == "t" else "2"), label,
               "%s - %s = (T_0 - %s)\\,e^{-k %s}" % (sym_T, target, target, sym_t),
               [tvar, Tvar] + _params(eff), fn, forms, traj=True)


def relations(eff):
    rels = [
        Rel("tau", "The time constant is the reciprocal of the cooling constant", "\\tau = 1/k", ["tau", "k"],
            lambda v: (v["tau"], 1 / v["k"]), {"tau": "\\tau = \\frac{1}{k}", "k": "k = \\frac{1}{\\tau}"}),
        Rel("th", "The temperature difference halves in a fixed time", "t_{1/2} = \\frac{\\ln 2}{k}", ["th", "k"],
            lambda v: (v["th"], math.log(2) / v["k"]),
            {"th": "t_{1/2} = \\frac{\\ln 2}{k}", "k": "k = \\frac{\\ln 2}{t_{1/2}}"}),
    ]
    if eff.get("source") == "on":
        rels.append(Rel("Tf", "The temperature settles where the heating balances the loss to the surroundings",
                        "T_f = T_a + \\frac{s}{k}", ["Tf", "Ta", "s", "k"],
                        lambda v: (v["Tf"] - v["Ta"], v["s"] / v["k"]),
                        {"Tf": "T_f = T_a + \\frac{s}{k}", "s": "s = k\\,(T_f - T_a)", "k": "k = \\frac{s}{T_f - T_a}"}))
    rels.append(_reading(eff, "t", "T", "The temperature at time t from the solution of the cooling equation"))
    rels.append(_reading(eff, "t2", "T2", "The same solution at the second time"))
    return rels


def equation(eff):
    return "\\frac{dT}{dt} = -k\\,(T - T_a)" + (" + s" if eff.get("source") == "on" else "")


def derivation(eff):
    text = ("Newton's law of cooling: the rate of change of temperature is proportional to the difference "
            "from the surroundings; the minus sign drives \\(T\\) towards \\(T_a\\) from either side.")
    if eff.get("source") == "on":
        text += " \\(s\\) is the steady heating rate of the source."
    target = "T_f" if eff.get("source") == "on" else "T_a"
    return [
        step("Set up the equation", text, [equation(eff)]),
        step("Solve the equation",
             "The equation is separable (and linear): the difference from the final temperature decays "
             "exponentially.",
             ["\\int \\frac{dT}{T - %s} = -\\int k\\,dt" % target,
              "T(t) = %s + (T_0 - %s)\\,e^{-kt}" % (target, target)]),
    ]


def horizon(eff, v):
    return 6 / v["k"] if v.get("k") else None


def graph(eff, v):
    traj = trajectory(eff, v)
    times = [x for x in (v.get("t"), v.get("t2")) if x is not None]
    t_max = max([traj["scale"] * 0.8] + [x * 1.2 for x in times])
    ideal = None
    if eff.get("source") == "on":
        ideal = lambda t: v["Ta"] + (v["T0"] - v["Ta"]) * math.exp(-v["k"] * t) - 273.15
    marker = None
    if v.get("t") is not None and v.get("T") is not None:
        marker = {"t": v["t"], "y": v["T"] - 273.15, "label": "first reading"}
    if v.get("t2") is not None and v.get("T2") is not None:
        marker = {"t": v["t2"], "y": v["T2"] - 273.15, "label": "second reading"}
    return {"fn": lambda t: traj["at"](t)["T"] - 273.15, "ideal": ideal, "t_max": t_max,
            "y_label": "temperature (\u00b0C)", "label": "Temperature", "ideal_label": "Without the heat source",
            "marker": marker}
