"""Small constructors for the declarative part of a scenario."""
from . import units


def field(fid, label, symbol, quantity, default=None, group="base", positive=False,
          nonnegative=False, hint=""):
    """An input box. `default` is (value text, unit id); group is base, effect or param."""
    return {"id": fid, "label": label, "symbol": symbol, "quantity": quantity,
            "units": units.choices(quantity), "default": list(default) if default else None,
            "group": group, "positive": positive, "nonnegative": nonnegative, "hint": hint}


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


def question(qid, label, params=(), finds=(), hides=()):
    """Something the scenario can answer. `params` are extra boxes it needs, `finds`
    the fields it computes (shown as 'to find'), `hides` base fields it does not use."""
    return {"id": qid, "label": label, "params": list(params), "finds": list(finds),
            "hides": list(hides)}


def preset(pid, label, description, values, effects=None, question=None):
    """A realistic starting point: values are {field: (value text, unit id)}."""
    return {"id": pid, "label": label, "description": description,
            "values": {k: list(v) for k, v in values.items()},
            "effects": dict(effects or {}), "question": question}
