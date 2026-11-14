"""Physics scenarios: real situations modelled by differential equations.

Each scenario module declares its fields, optional effects, questions and
presets in a SCENARIO dict, and provides equation(effects) and solve(inputs).
"""
import json

from ..parsing import InputError
from . import falling, rc, spring
from .common import Inputs

SCENARIOS = [falling, spring, rc]
UPCOMING = ["Projectile", "Pendulum", "Rocket", "Cooling and heating", "RL and RLC circuits",
            "Charged particle in a field", "Electromagnetic induction", "Radioactive decay"]


def _module(scenario_id):
    for mod in SCENARIOS:
        if mod.SCENARIO["id"] == scenario_id:
            return mod
    raise InputError("Unknown physics scenario: %s." % scenario_id)


def _raw(inputs_text):
    if not inputs_text or not inputs_text.strip():
        return {}
    try:
        raw = json.loads(inputs_text)
    except ValueError:
        raise InputError("The scenario inputs could not be read.")
    if not isinstance(raw, dict):
        raise InputError("The scenario inputs could not be read.")
    return raw


def catalog():
    return {"ok": True, "scenarios": [m.SCENARIO for m in SCENARIOS], "upcoming": UPCOMING}


def solve(scenario_id, inputs_text):
    mod = _module(scenario_id)
    return mod.solve(Inputs(mod.SCENARIO, _raw(inputs_text)))


def equation(scenario_id, inputs_text):
    """LaTeX of the governing equation for the switches chosen so far."""
    mod = _module(scenario_id)
    raw = _raw(inputs_text)
    effects = {e["id"]: e["default"] for e in mod.SCENARIO["effects"]}
    effects.update({k: v for k, v in (raw.get("effects") or {}).items() if k in effects})
    return mod.equation(effects)
