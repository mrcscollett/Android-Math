"""Shared pieces for the first-order circuits (RC and RL)."""
import math

from .common import driven_relax, relax


def response(source, final, start, tau, amp=0.0, freq=0.0):
    """u(t) relaxing towards `final` (battery / none) or following an AC source."""
    if source == "ac":
        return driven_relax(amp, 2 * math.pi * freq, start, tau)
    return relax(final, start, tau)
