"""Units for the physics scenarios: conversion to SI and readable output."""
import math

from ..parsing import InputError, parse_number

# quantity -> list of (unit id, label, LaTeX, factor to SI, offset). The first entry is the SI unit.
_U = {
    "length": [("m", "m", "m", 1, 0), ("cm", "cm", "cm", 0.01, 0), ("mm", "mm", "mm", 1e-3, 0),
               ("km", "km", "km", 1e3, 0), ("ft", "ft", "ft", 0.3048, 0), ("in", "in", "in", 0.0254, 0),
               ("mi", "mi", "mi", 1609.344, 0)],
    "time": [("s", "s", "s", 1, 0), ("ms", "ms", "ms", 1e-3, 0), ("us", "\u00b5s", "\\mu s", 1e-6, 0),
             ("min", "min", "min", 60, 0), ("h", "h", "h", 3600, 0),
             ("d", "days", "d", 86400, 0), ("y", "years", "yr", 31557600, 0)],
    "rate": [("1/s", "1/s", "s^{-1}", 1, 0), ("1/min", "1/min", "min^{-1}", 1 / 60, 0),
             ("1/h", "1/h", "h^{-1}", 1 / 3600, 0), ("1/d", "1/day", "d^{-1}", 1 / 86400, 0),
             ("1/y", "1/year", "yr^{-1}", 1 / 31557600, 0)],
    "temperature_rate": [("K/s", "K/s", "K/s", 1, 0), ("K/min", "\u00b0C/min", "K/min", 1 / 60, 0),
                         ("K/h", "\u00b0C/h", "K/h", 1 / 3600, 0)],
    "amount": [("", "units", "", 1, 0), ("g", "g", "g", 1, 0), ("mg", "mg", "mg", 1e-3, 0),
               ("kg", "kg", "kg", 1e3, 0), ("Bq", "Bq", "Bq", 1, 0), ("MBq", "MBq", "MBq", 1e6, 0),
               ("mol", "mol", "mol", 1, 0)],
    "amount_rate": [("/s", "per s", "s^{-1}", 1, 0), ("/min", "per min", "min^{-1}", 1 / 60, 0),
                    ("/h", "per h", "h^{-1}", 1 / 3600, 0), ("/d", "per day", "d^{-1}", 1 / 86400, 0)],
    "mass": [("kg", "kg", "kg", 1, 0), ("g", "g", "g", 1e-3, 0), ("mg", "mg", "mg", 1e-6, 0),
             ("lb", "lb", "lb", 0.45359237, 0)],
    "speed": [("m/s", "m/s", "m/s", 1, 0), ("mm/s", "mm/s", "mm/s", 1e-3, 0), ("cm/s", "cm/s", "cm/s", 0.01, 0), ("km/h", "km/h", "km/h", 1 / 3.6, 0),
              ("mph", "mph", "mph", 0.44704, 0), ("ft/s", "ft/s", "ft/s", 0.3048, 0)],
    "acceleration": [("m/s2", "m/s\u00b2", "m/s^2", 1, 0), ("ft/s2", "ft/s\u00b2", "ft/s^2", 0.3048, 0),
                     ("g0", "g", "g_0", 9.80665, 0)],
    "force": [("N", "N", "N", 1, 0), ("kN", "kN", "kN", 1e3, 0), ("lbf", "lbf", "lbf", 4.4482216, 0)],
    "energy": [("J", "J", "J", 1, 0), ("kJ", "kJ", "kJ", 1e3, 0), ("mJ", "mJ", "mJ", 1e-3, 0),
               ("uJ", "\u00b5J", "\\mu J", 1e-6, 0)],
    "power": [("W", "W", "W", 1, 0), ("kW", "kW", "kW", 1e3, 0), ("mW", "mW", "mW", 1e-3, 0)],
    "angle": [("rad", "rad", "rad", 1, 0), ("deg", "\u00b0", "^\\circ", math.pi / 180, 0)],
    "frequency": [("Hz", "Hz", "Hz", 1, 0), ("kHz", "kHz", "kHz", 1e3, 0), ("MHz", "MHz", "MHz", 1e6, 0)],
    "angular_frequency": [("rad/s", "rad/s", "rad/s", 1, 0), ("Hz", "Hz", "Hz", 2 * math.pi, 0),
                          ("rpm", "rpm", "rpm", 2 * math.pi / 60, 0)],
    "temperature": [("K", "K", "K", 1, 0), ("C", "\u00b0C", "^\\circ C", 1, 273.15),
                    ("F", "\u00b0F", "^\\circ F", 5 / 9, 459.67 * 5 / 9)],
    "voltage": [("V", "V", "V", 1, 0), ("mV", "mV", "mV", 1e-3, 0), ("kV", "kV", "kV", 1e3, 0)],
    "current": [("A", "A", "A", 1, 0), ("mA", "mA", "mA", 1e-3, 0), ("uA", "\u00b5A", "\\mu A", 1e-6, 0)],
    "resistance": [("ohm", "\u03a9", "\\Omega", 1, 0), ("kohm", "k\u03a9", "k\\Omega", 1e3, 0),
                   ("Mohm", "M\u03a9", "M\\Omega", 1e6, 0)],
    "capacitance": [("F", "F", "F", 1, 0), ("mF", "mF", "mF", 1e-3, 0), ("uF", "\u00b5F", "\\mu F", 1e-6, 0),
                    ("nF", "nF", "nF", 1e-9, 0), ("pF", "pF", "pF", 1e-12, 0)],
    "inductance": [("H", "H", "H", 1, 0), ("mH", "mH", "mH", 1e-3, 0), ("uH", "\u00b5H", "\\mu H", 1e-6, 0)],
    "charge": [("C", "C", "C", 1, 0), ("mC", "mC", "mC", 1e-3, 0), ("uC", "\u00b5C", "\\mu C", 1e-6, 0),
               ("nC", "nC", "nC", 1e-9, 0)],
    "area": [("m2", "m\u00b2", "m^2", 1, 0), ("cm2", "cm\u00b2", "cm^2", 1e-4, 0),
             ("mm2", "mm\u00b2", "mm^2", 1e-6, 0), ("ft2", "ft\u00b2", "ft^2", 0.09290304, 0)],
    "volume": [("m3", "m\u00b3", "m^3", 1, 0), ("L", "L", "L", 1e-3, 0), ("cm3", "cm\u00b3", "cm^3", 1e-6, 0)],
    "density": [("kg/m3", "kg/m\u00b3", "kg/m^3", 1, 0), ("g/cm3", "g/cm\u00b3", "g/cm^3", 1e3, 0)],
    "drag_linear": [("kg/s", "kg/s", "kg/s", 1, 0), ("g/s", "g/s", "g/s", 1e-3, 0)],
    "drag_quadratic": [("kg/m", "kg/m", "kg/m", 1, 0), ("g/m", "g/m", "g/m", 1e-3, 0)],
    "spring_constant": [("N/m", "N/m", "N/m", 1, 0), ("kN/m", "kN/m", "kN/m", 1e3, 0),
                        ("N/cm", "N/cm", "N/cm", 100, 0), ("lbf/in", "lbf/in", "lbf/in", 175.12684, 0)],
    "damping": [("kg/s", "N\u00b7s/m", "N\\,s/m", 1, 0), ("kN.s/m", "kN\u00b7s/m", "kN\\,s/m", 1e3, 0)],
    "dimensionless": [("", "", "", 1, 0), ("%", "%", "\\%", 0.01, 0)],
}

QUANTITIES = sorted(_U)


def choices(quantity):
    """Unit choices for the app's picker: [{id, label}]."""
    return [{"id": u[0], "label": u[1]} for u in _U[quantity]]


def _find(quantity, unit):
    if quantity not in _U:
        raise InputError("Unknown kind of quantity: %s." % quantity)
    for u in _U[quantity]:
        if u[0] == unit:
            return u
    raise InputError("The unit '%s' cannot be used for a %s." % (unit, quantity.replace("_", " ")))


def to_si(value, quantity, unit):
    u = _find(quantity, unit)
    return value * u[3] + u[4]


def from_si(value, quantity, unit):
    u = _find(quantity, unit)
    return (value - u[4]) / u[3]


def parse(text, quantity, unit, label):
    """A typed value with its unit, as an SI float. Blank text means unknown (None)."""
    if text is None or not str(text).strip():
        return None
    number = float(parse_number(str(text), label))
    return to_si(number, quantity, unit or _U[quantity][0][0])


def number(value, digits=4):
    """A number to `digits` significant figures, without needless zeros."""
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    magnitude = math.floor(math.log10(abs(value)))
    if magnitude >= 6 or magnitude < -4:
        mantissa = value / 10 ** magnitude
        text = ("%.*f" % (digits - 1, mantissa)).rstrip("0").rstrip(".")
        return "%s\u00d710^%d" % (text, magnitude)
    decimals = max(0, digits - 1 - magnitude)
    text = "%.*f" % (decimals, value)
    return text.rstrip("0").rstrip(".") if "." in text else text


def number_tex(value, digits=4):
    text = number(value, digits)
    if "\u00d710^" in text:
        mantissa, power = text.split("\u00d710^")
        return "%s \\times 10^{%s}" % (mantissa, power)
    return text


def best_unit(value, quantity, prefer=None):
    """The unit to show a result in: the one the user typed, else a readable prefix."""
    if prefer is not None:
        return _find(quantity, prefer)
    if quantity == "temperature" or value == 0:
        return _U[quantity][0]
    metric = [u for u in _U[quantity] if u[4] == 0 and _is_power_of_ten(u[3])]
    best = _U[quantity][0]
    def shown(u):
        # Rounded as it will be printed, so 0.9999999 s reads as 1 s and not 1000 ms.
        return float("%.4g" % (abs(value) / u[3]))

    for u in metric:
        if 1 <= shown(u) < 1000 and (not (1 <= shown(best) < 1000) or u is _U[quantity][0]):
            best = u
    return best


def _is_power_of_ten(x):
    return abs(math.log10(x) - round(math.log10(x))) < 1e-9


def text(value, quantity, prefer=None, digits=4):
    """Plain text such as '53.4 m/s'."""
    u = best_unit(value, quantity, prefer)
    shown = number(from_si(value, quantity, u[0]), digits)
    return shown + ("" if not u[1] else (u[1] if u[1] in ("%", "\u00b0") else " " + u[1]))


def tex(value, quantity, prefer=None, digits=4):
    """LaTeX such as 53.4\\ \\text{m/s}."""
    u = best_unit(value, quantity, prefer)
    shown = number_tex(from_si(value, quantity, u[0]), digits)
    return shown + ("" if not u[2] else "\\ \\mathrm{%s}" % u[2])


def both(value, quantity, prefer=None):
    """Text in the preferred unit, with the SI value added when that differs."""
    main = text(value, quantity, prefer)
    if quantity in ("amount", "amount_rate"):
        return main            # these have no single SI unit to echo
    si = text(value, quantity, _U[quantity][0][0])
    return main if main == si else "%s (%s)" % (main, si)
