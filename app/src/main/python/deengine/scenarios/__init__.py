"""Physics: general systems governed by differential equations.

Each system is one equation of motion with optional forces or components as
switches, not a particular situation. Each system module declares its variables, optional effects and presets in a
MODEL dict and provides relations(effects), derivation(effects),
equation(effects), horizon(effects, v), trajectory(effects, v) and
graph(effects, v). The user fills in the variables they know; solver.py
finds the rest from the relations.
"""
import json

from ..parsing import InputError
from . import change, circuit, line, plane, rotation, solver
from .common import Inputs

MODELS = [line, plane, rotation, change, circuit]
UPCOMING = ["Rocket (changing mass)", "Draining tank", "Coupled oscillators", "Electromagnetic induction"]
# Ids of the earlier situation-based models, so saved history and old links still open something.
LEGACY = {"falling": "line", "spring": "line", "projectile": "plane", "pendulum": "rotation",
          "cooling": "change", "decay": "change", "rc": "circuit", "rl": "circuit"}


def _module(model_id):
    model_id = LEGACY.get(model_id, model_id)
    for mod in MODELS:
        if mod.MODEL["id"] == model_id:
            return mod
    raise InputError("Unknown physics model: %s." % model_id)


def _raw(inputs_text):
    if not inputs_text or not inputs_text.strip():
        return {}
    try:
        raw = json.loads(inputs_text)
    except ValueError:
        raise InputError("The inputs could not be read.")
    if not isinstance(raw, dict):
        raise InputError("The inputs could not be read.")
    return raw


def catalog():
    return {"ok": True, "scenarios": [m.MODEL for m in MODELS], "upcoming": UPCOMING}


def solve(model_id, inputs_text):
    mod = _module(model_id)
    return solver.solve(mod, Inputs(mod.MODEL, _raw(inputs_text)))


def equation(model_id, inputs_text):
    """LaTeX of the governing equation for the switches chosen so far."""
    mod = _module(model_id)
    raw = _raw(inputs_text)
    effects = {e["id"]: e["default"] for e in mod.MODEL["effects"]}
    effects.update({k: v for k, v in (raw.get("effects") or {}).items() if k in effects})
    return mod.equation(effects)
