"""First-order change: a quantity that decays, relaxes towards a level, or grows.

    decay:   y' = -k y + s
    level:   y' = -k (y - y_inf) + s
    growth:  y' =  k y + s

"What is changing?" only chooses the names and units of the rows; the
mathematics is the same for a temperature, an amount, a voltage and so on.
"""
import math

from ..first_order import step
from .common import linear_working
from .solver import Rel, Skip
from .spec import choice, field, preset, switch

# id, option label, noun, quantity, rate quantity, symbol, default unit, rate unit, label of the level
KINDS = [
    ("temp", "Temperature", "temperature", "temperature", "temperature_rate", "T", "C", "K/s", "Surrounding temperature"),
    ("amt", "Amount or mass", "amount", "amount", "amount_rate", "N", "", "/s", "Level it approaches"),
    ("conc", "Concentration", "concentration", "concentration", "concentration_rate", "c", "mol/L", "/s", "Level it approaches"),
    ("volt", "Voltage", "voltage", "voltage", "voltage_rate", "V", "V", "V/s", "Voltage it approaches"),
    ("cur", "Current", "current", "current", "current_rate", "I", "A", "A/s", "Current it approaches"),
    ("spd", "Speed", "speed", "speed", "acceleration", "v", "m/s", "m/s2", "Speed it approaches"),
    ("pop", "Population", "population", "plain", "plain_rate", "P", "", "/s", "Level it approaches"),
    ("gen", "Something else", "value", "plain", "plain_rate", "y", "", "/s", "Level it approaches"),
]
_BY_ID = {k[0]: k for k in KINDS}


def _fields():
    out = []
    per_kind = {"y0": [], "yinf": [], "s": [], "yf": [], "y": [], "y2": []}
    for kid, _label, noun, q, rq, sym, unit, runit, level in KINDS:
        only = [("what", [kid])]
        per_kind["y0"].append(field("y0_" + kid, "Initial " + noun, sym + "_0", q, (None, unit), "initial", show=only))
        per_kind["yinf"].append(field("yinf_" + kid, level, sym + "_\\infty", q, (None, unit), "setup",
                                      show=only + [("law", ["level"])]))
        per_kind["s"].append(field("s_" + kid, "Source rate", "s", rq, (None, runit), "effect", show=only,
                                   hint="How fast the source alone would change the %s." % noun))
        per_kind["yf"].append(field("yf_" + kid, "Final %s with the source" % noun, sym + "_f", q, (None, unit),
                                    "effect", show=only + [("law", ["decay", "level"])]))
        per_kind["y"].append(field("y_" + kid, noun.capitalize() + " at that time", sym, q, (None, unit), "moment",
                                   show=only))
        per_kind["y2"].append(field("y2_" + kid, noun.capitalize() + " at the second time", sym + "_2", q,
                                    (None, unit), "moment2", show=only))
    out += per_kind["yinf"]
    out.append(field("k", "Rate constant", "k", "rate", (None, "1/s"), positive=True, typical=0.01))
    out += per_kind["y0"] + per_kind["s"] + per_kind["yf"]
    out.append(field("t", "Time", "t", "time", (None, "s"), "moment", nonnegative=True, typical=10))
    out += per_kind["y"]
    out.append(field("frac", "Fraction of the initial difference left (growth: growth factor)", "f",
                     "dimensionless", (None, ""), "moment", positive=True,
                     hint="Decay: 0.25 means a quarter is left. Growth: 8 means eight times the start."))
    out.append(field("t2", "Second time", "t_2", "time", (None, "s"), "moment2", nonnegative=True, typical=10))
    out += per_kind["y2"]
    out.append(field("tau", "Time constant", "\\tau", "time", (None, "s"), "derived", positive=True, typical=100))
    out.append(field("th", "Half-life (growth: doubling time)", "t_{1/2}", "time", (None, "s"), "derived",
                     positive=True, typical=100))
    return out


def _ex(pid, label, text, kind, law, values, source=False):
    eff = {"what": kind, "law": law}
    if source:
        eff["source"] = "on"
    return preset(pid, label, text, {(k if k in ("k", "t", "t2", "frac", "tau", "th") else k + "_" + kind): v
                                     for k, v in values.items()}, eff)


MODEL = {
    "id": "change", "title": "First-order change", "animation": "capacitor",
    "description": "Decay, growth, or relaxing towards a level: y' = -k (y - level) + source.",
    "law": "The rate of change is proportional to how far the quantity is from where it is heading.",
    "fields": _fields(),
    "effects": [
        choice("what", "What is changing?", "This only sets the names and units of the rows.",
               [(k[0], k[1], []) for k in KINDS], "gen"),
        choice("law", "How it changes", "Decay towards zero, relax towards a level, or grow.",
               [("decay", "Decays to zero", []), ("level", "Approaches a level", []), ("growth", "Grows", [])],
               "decay"),
        switch("source", "Constant source", "Something adds to (or removes from) the quantity at a steady rate.",
               ["s_" + k[0] for k in KINDS] + ["yf_" + k[0] for k in KINDS]),
    ],
    "presets": [
        _ex("coffee", "Cooling coffee", "Coffee at 90 °C in a 22 °C room is 75 °C after 5 minutes; when is it 60 °C?",
            "temp", "level", {"y0": ("90", "C"), "yinf": ("22", "C"), "t": ("5", "min"), "y": ("75", "C"),
                              "y2": ("60", "C"), "t2": ("", "min"), "k": ("", "1/min")}),
        _ex("forensic", "Time since death", "A body at 29.5 °C in an 18 °C room, cooling constant 0.19 per hour.",
            "temp", "level", {"y0": ("37", "C"), "yinf": ("18", "C"), "k": ("0.19", "1/h"), "y": ("29.5", "C"),
                              "t": ("", "h")}),
        _ex("roast", "Roast in the oven", "A 5 °C roast in a 180 °C oven reads 40 °C after 30 minutes; when is it 70 °C?",
            "temp", "level", {"y0": ("5", "C"), "yinf": ("180", "C"), "t": ("30", "min"), "y": ("40", "C"),
                              "y2": ("70", "C"), "t2": ("", "min"), "k": ("", "1/h")}),
        _ex("chip", "Processor warming up", "A chip heated at 180 K/min while losing heat to 25 °C air.",
            "temp", "level", {"y0": ("25", "C"), "yinf": ("25", "C"), "k": ("0.05", "1/s"), "s": ("180", "K/min"),
                              "t": ("30", "s")}, source=True),
        _ex("carbon", "Carbon dating", "A sample has 25% of its carbon-14 left; the half-life is 5730 years.",
            "amt", "decay", {"th": ("5730", "y"), "frac": ("25", "%"), "t": ("", "y")}),
        _ex("iodine", "Iodine-131 dose", "400 MBq of iodine-131 (half-life 8.02 days): how much is left after 30 days?",
            "amt", "decay", {"th": ("8.02", "d"), "y0": ("400", "MBq"), "t": ("30", "d"), "y": ("", "MBq"),
                             "k": ("", "1/d"), "tau": ("", "d")}),
        _ex("drug", "Drug elimination", "200 mg of a drug falls to 150 mg in 2 hours; how much is left after 12 hours?",
            "amt", "decay", {"y0": ("200", "mg"), "t": ("2", "h"), "y": ("150", "mg"), "t2": ("12", "h"),
                             "y2": ("", "mg"), "th": ("", "h"), "tau": ("", "h"), "k": ("", "1/h")}),
        _ex("bacteria", "Bacteria doubling", "A culture doubles every 20 minutes; how long from 1000 to a million cells?",
            "pop", "growth", {"th": ("20", "min"), "y0": ("1000", ""), "y": ("1000000", ""), "t": ("", "min"),
                              "k": ("", "1/min"), "tau": ("", "min")}),
    ],
}


def _ids(eff):
    """The row ids for the chosen kind, and which optional rows exist."""
    kid = eff.get("what", "gen")
    if kid not in _BY_ID:
        kid = "gen"
    law = eff.get("law", "decay")
    src = eff.get("source") == "on"
    n = {x: x + "_" + kid for x in ("y0", "yinf", "s", "yf", "y", "y2")}
    return n, law, src, _BY_ID[kid][5]


def _extra(n, law, src):
    return ([n["yinf"]] if law == "level" else []) + ([n["s"]] if src else [])


def _target(v, n, law, src):
    """Where the quantity is heading (decay/level), or the unstable balance point (growth)."""
    base = v[n["yinf"]] if law == "level" else 0.0
    if not src:
        return base
    return base + v[n["s"]] / v["k"] if law != "growth" else -v[n["s"]] / v["k"]


def _span(v, n, law, src):
    """A size for the quantity, so values near zero compare against the size of the change."""
    return abs(v[n["y0"]]) + abs(_target(v, n, law, src)) + 1e-9


def _value(v, n, law, src, t):
    target = _target(v, n, law, src)
    sign = 1.0 if law == "growth" else -1.0
    return target + (v[n["y0"]] - target) * math.exp(sign * v["k"] * t)


def relations(eff):
    n, law, src, sym = _ids(eff)
    grow = law == "growth"
    extra = _extra(n, law, src)
    ft = "f = e^{k t}" if grow else "f = e^{-k t}"
    tgt = "%s_f" % sym if src else ("%s_\\infty" % sym if law == "level" else "0")
    if grow:
        tgt = "-s/k" if src else "0"
    rels = [
        Rel("tau", "The time constant is the reciprocal of the rate constant", "\\tau = 1/k", ["tau", "k"],
            lambda v: (v["tau"], 1 / v["k"]), {"tau": "\\tau = \\frac{1}{k}", "k": "k = \\frac{1}{\\tau}"}),
        Rel("th", "The time for the difference to %s" % ("double" if grow else "halve"),
            "t_{1/2} = \\frac{\\ln 2}{k}", ["th", "k"], lambda v: (v["th"], math.log(2) / v["k"]),
            {"th": "t_{1/2} = \\frac{\\ln 2}{k}", "k": "k = \\frac{\\ln 2}{t_{1/2}}"}),
        Rel("frac", "The difference from where it is heading %s exponentially" % ("grows" if grow else "shrinks"),
            ft, ["frac", "k", "t"],
            lambda v: (math.log(v["frac"]), (1.0 if grow else -1.0) * v["k"] * v["t"]),
            {"frac": ft, "t": "t = \\frac{\\ln f}{k}" if grow else "t = -\\frac{\\ln f}{k}",
             "k": "k = \\frac{\\ln f}{t}" if grow else "k = -\\frac{\\ln f}{t}"}),
    ]
    if src and not grow:
        base = "%s_\\infty + " % sym if law == "level" else ""
        rels.append(Rel("yf", "With a constant source the quantity settles where the source balances the loss",
                        "%s_f = %s\\frac{s}{k}" % (sym, base), [n["yf"], n["s"], "k"] + ([n["yinf"]] if law == "level" else []),
                        lambda v: (v[n["yf"]] + _scale(v, n, law), _target(v, n, law, True) + _scale(v, n, law)),
                        {n["yf"]: "%s_f = %s\\frac{s}{k}" % (sym, base)}))
    form = "%s = %s + f\\,(%s_0 - %s)" % (sym, tgt, sym, tgt) if tgt != "0" else "%s = f\\,%s_0" % (sym, sym)
    rels.append(Rel(
        "y", "The value at time t is where it is heading plus what is left of the initial difference", form,
        [n["y"], "frac", n["y0"]] + extra + (["k"] if src else []),
        lambda v: (v[n["y"]] + _span(v, n, law, src),
                   _target(v, n, law, src) + v["frac"] * (v[n["y0"]] - _target(v, n, law, src)) + _span(v, n, law, src)),
        {n["y"]: form, "frac": ("f = \\frac{%s - %s}{%s_0 - %s}" % (sym, tgt, sym, tgt)) if tgt != "0"
         else "f = \\frac{%s}{%s_0}" % (sym, sym)}))
    exp2 = "e^{k t_2}" if grow else "e^{-k t_2}"
    form2 = ("%s_2 = %s + (%s_0 - %s)\\,%s" % (sym, tgt, sym, tgt, exp2)) if tgt != "0" \
        else "%s_2 = %s_0\\,%s" % (sym, sym, exp2)
    rels.append(Rel(
        "y2", "The same solution read at the second time", form2,
        [n["y2"], "t2", n["y0"], "k"] + extra,
        lambda v: (v[n["y2"]] + _span(v, n, law, src), _value(v, n, law, src, v["t2"]) + _span(v, n, law, src)),
        {n["y2"]: form2}, traj=True))
    return rels


def _scale(v, n, law):
    return abs(v[n["yinf"]]) if law == "level" else 0.0


def trajectory(eff, v):
    n, law, src, _sym = _ids(eff)
    need = [n["y0"], "k"] + _extra(n, law, src)
    if any(v.get(x) is None for x in need):
        raise Skip()
    k = v["k"]
    base = v[n["yinf"]] if law == "level" else 0.0
    s = v[n["s"]] if src else 0.0
    sign = 1.0 if law == "growth" else -1.0
    deriv = lambda _t, st: [sign * k * (st[0] - base) + s] if law != "growth" else [k * st[0] + s]
    return {"at": lambda t: {"y": _value(v, n, law, src, t)}, "exact": True, "scale": 1 / k,
            "ode": (deriv, [v[n["y0"]]], ["y"], 3 / k)}


def baseline(eff):
    return None        # nothing here is a loss to switch off


def equation(eff):
    _n, law, src, sym = _ids(eff)
    tail = " + s" if src else ""
    if law == "growth":
        return "\\frac{d%s}{dt} = k\\,%s%s" % (sym, sym, tail)
    if law == "level":
        return "\\frac{d%s}{dt} = -k\\,(%s - %s_\\infty)%s" % (sym, sym, sym, tail)
    return "\\frac{d%s}{dt} = -k\\,%s%s" % (sym, sym, tail)


def derivation(eff):
    _n, law, src, sym = _ids(eff)
    if law == "growth":
        text = "The quantity grows at a rate proportional to how much of it there is."
    elif law == "level":
        text = ("The quantity changes at a rate proportional to its distance from the level it is "
                "approaching, \\(%s_\\infty\\) (Newton's law of cooling is the best-known case)." % sym)
    else:
        text = "The quantity is lost at a rate proportional to how much of it is left."
    if src:
        text += " The constant \\(s\\) is a source adding to it at a steady rate."
    steps = [step("Set up the differential equation", text, [equation(eff)])]
    if law == "growth":
        tgt = "-s/k" if src else None
        lines = ["%s(t) = \\left(%s_0 + \\frac{s}{k}\\right)e^{k t} - \\frac{s}{k}" % (sym, sym)] if src \
            else ["%s(t) = %s_0\\,e^{k t}" % (sym, sym)]
        steps.append(step("Solve the equation",
                          "The equation is linear with constant coefficients; separating the variables "
                          "gives exponential growth.", lines))
        return steps
    tgt = "%s_f" % sym if src else ("%s_\\infty" % sym if law == "level" else "0")
    line = ("%s(t) = %s + (%s_0 - %s)\\,e^{-k t}" % (sym, tgt, sym, tgt)) if tgt != "0" \
        else "%s(t) = %s_0\\,e^{-k t}" % (sym, sym)
    steps.append(step(
        "Solve the equation",
        "The equation is first-order linear. The difference from where the quantity is heading "
        "obeys \\(u' = -k\\,u\\), so it decays exponentially.", [line],
        linear_working(sym, "k", "k\\,%s" % tgt if tgt != "0" else "0", tgt, "%s_0" % sym)))
    return steps


def horizon(eff, v):
    return 6 / v["k"] if v.get("k") else None


def graph(eff, v):
    n, law, src, sym = _ids(eff)
    try:
        traj = trajectory(eff, v)
    except Skip:
        if not v.get("k"):
            raise
        # The amounts themselves are not known, but the rate is: draw the fraction instead.
        sign = 1.0 if law == "growth" else -1.0
        t_end = max((5 if sign < 0 else 4) / v["k"], (v.get("t") or 0.0) * 1.2)
        marker = None
        if v.get("t") is not None and v.get("frac") is not None:
            marker = {"t": v["t"], "y": v["frac"], "label": "at time t"}
        return {"fn": lambda t: math.exp(sign * v["k"] * t), "ideal": None, "t_max": t_end,
                "y_label": "fraction of the initial difference", "label": "Fraction", "ideal_label": "",
                "marker": marker}
    t_max = max(5 / v["k"] if law != "growth" else 4 / v["k"], (v.get("t") or 0.0) * 1.2, (v.get("t2") or 0.0) * 1.1)
    ideal = None
    if src:
        ideal = lambda t: _value(v, n, law, False, t)
    marker = None
    if v.get("t") is not None and v.get(n["y"]) is not None:
        marker = {"t": v["t"], "y": v[n["y"]], "label": "at time t"}
    kind = _BY_ID[eff.get("what", "gen")] if eff.get("what") in _BY_ID else _BY_ID["gen"]
    unit = {"temperature": "K", "voltage": "V", "current": "A", "speed": "m/s"}.get(kind[3], "")
    return {"fn": lambda t: traj["at"](t)["y"], "ideal": ideal, "t_max": t_max,
            "y_label": kind[2] + (" (%s)" % unit if unit else ""), "label": kind[1],
            "ideal_label": "Without the source", "marker": marker}
