"""Surrogate-safety baselines the paper compares against: TTC and TTS.

The paper cites both as established measures without formalizing them; the
standard definitions below are used and stated explicitly so the benchmark
comparison is reproducible.
"""

import math
from typing import Iterable, Optional, Tuple


def ttc_point_mass(
    p1: Tuple[float, float],
    v1: Tuple[float, float],
    p2: Tuple[float, float],
    v2: Tuple[float, float],
    radius: float,
) -> float:
    """Time until two constant-velocity discs of combined `radius` touch.

    Returns +inf when they never do. Solves |dp + dv t| = radius for the
    smallest non-negative root; if already overlapping, returns 0.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dvx, dvy = v2[0] - v1[0], v2[1] - v1[1]

    c = dx * dx + dy * dy - radius * radius
    if c <= 0:
        return 0.0
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (dx * dvx + dy * dvy)
    if a == 0.0:
        return math.inf
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return math.inf
    root = (-b - math.sqrt(disc)) / (2.0 * a)
    if root < 0:
        return math.inf
    return root


def min_ttc(av, foes: Iterable, radius_fn=None) -> float:
    """Minimum TTC between the AV and any foe.

    `av` and each foe expose .x, .y, .vx, .vy and .dims (VehicleDims). Each
    body is approximated by a disc of radius (length + width) / 4 - between
    the inscribed and circumscribed circles - and the combined radius is
    their sum. A disc approximation, stated as such.
    """
    best = math.inf
    for foe in foes:
        if radius_fn is None:
            r = (av.dims.length + av.dims.width) / 4.0 + (
                foe.dims.length + foe.dims.width
            ) / 4.0
        else:
            r = radius_fn(av, foe)
        t = ttc_point_mass(
            (av.x, av.y), (av.vx, av.vy), (foe.x, foe.y), (foe.vx, foe.vy), r
        )
        best = min(best, t)
    return best


def tts_margin(speed_ms: float, ttc: float, decel_ms2: float = 9.81,
               reaction_time_s: float = 1.0) -> float:
    """Time-to-stop margin: TTC minus the time needed to react and stop.

    Negative margin: even full braking after the reaction time cannot avert
    the TTC event. Used as a risk score via its negation.
    """
    if speed_ms < 0:
        raise ValueError("speed must be non-negative")
    time_to_stop = reaction_time_s + speed_ms / decel_ms2
    if math.isinf(ttc):
        return math.inf
    return ttc - time_to_stop
