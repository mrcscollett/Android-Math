"""Small constructors for the declarative part of a scenario."""
from . import units


def field(fid, label, symbol, quantity, default=None, group="setup", positive=False,
          nonnegative=False, hint="", bounds=None, typical=1.0, optional=False,
          show=None, hide=None):
    """A variable row. `default` is (value text, unit id) or (None, unit id) for a blank box.

    group: "setup" (parameters), "initial" (initial conditions), "moment" (the
    state at one time), "moment2" (a second reading), "derived", or "effect"
    (shown only with its effect). `bounds` (SI) and `typical` (an SI magnitude)
    guide the numerical search when this variable is the unknown. An `optional`
    variable is left out of the "could not be found" list.

    `show` and `hide` make a row depend on the switches: each is a list of
    (effect id, [option ids]) conditions. The row exists only while every
    `show` condition holds, and is removed while every `hide` condition holds.
    """
    return {"id": fid, "label": label, "symbol": symbol, "quantity": quantity,
            "units": units.choices(quantity), "default": list(default) if default else None,
            "group": group, "positive": positive, "nonnegative": nonnegative, "hint": hint,
            "bounds": list(bounds) if bounds else None, "typical": typical, "optional": optional,
            "show": [{"effect": c[0], "in": list(c[1])} for c in (show or [])],
            "hide": [{"effect": c[0], "in": list(c[1])} for c in (hide or [])]}


def switch(eid, label, description, fields):
    """An optional effect that is either off or on."""
    return {"id": eid, "label": label, "description": description, "kind": "switch",
            "default": "off",
            "options": [{"id": "off", "label": "Off", "fields": []},
                        {"id": "on", "label": "On", "fields": list(fields)}]}


def choice(eid, label, description, options, default):
    """An effect with several alternatives; options are (id, label, fields)."""
    return {"id": eid, "label": label, "description": description, "kind": "choice",
            "default": default,
            "options": [{"id": o[0], "label": o[1], "fields": list(o[2])} for o in options]}


def preset(pid, label, description, values, effects=None):
    """An optional example: a real situation and the values known in it, {field: (value text, unit id)}.

    Fields it does not mention are left blank, i.e. they are what gets found.
    """
    return {"id": pid, "label": label, "description": description,
            "values": {k: list(v) for k, v in values.items()},
            "effects": dict(effects or {}), "question": None}
