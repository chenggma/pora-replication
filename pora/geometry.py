"""Safety box geometry (paper section III-B).

All coordinates in this module are in the AV frame: origin at the AV
centroid, +x pointing along the AV heading, units in meters. The box is
centered on the AV position and aligned with its heading, per the paper.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VehicleDims:
    length: float
    width: float

    def __post_init__(self):
        if self.length <= 0 or self.width <= 0:
            raise ValueError("vehicle dimensions must be positive")


def stopping_sight_distance(speed_ms, reaction_time_s=1.0, decel_ms2=9.81):
    """SSD in meters, SI form: v*r + v^2 / (2a).

    The paper writes the AASHTO form in km/h units with a 1 g deceleration.
    The reaction time it uses is not stated unambiguously in the arXiv HTML;
    it is a parameter here (default 1.0 s) and a documented gap.
    """
    if speed_ms < 0:
        raise ValueError("speed must be non-negative")
    return speed_ms * reaction_time_s + speed_ms * speed_ms / (2.0 * decel_ms2)


@dataclass(frozen=True)
class SafetyBox:
    """Dynamic safety box Phi and guaranteed-collision subarea phi.

    Paper definitions:
      Phi width   = w_AV + max_n l_n                     (constant)
      Phi length  = l_AV + max_n l_n + SSD(v)            (speed-dependent)
      phi width   = w_AV + min_n w_n
      phi length  = l_AV + min_n w_n
    phi is clamped to Phi so the nesting phi within Phi always holds.
    """

    av: VehicleDims
    fleet_max_length: float
    fleet_min_width: float
    speed_ms: float
    reaction_time_s: float = 1.0
    decel_ms2: float = 9.81

    @property
    def ssd(self):
        return stopping_sight_distance(
            self.speed_ms, self.reaction_time_s, self.decel_ms2
        )

    @property
    def phi_length(self):
        return self.av.length + self.fleet_max_length + self.ssd

    @property
    def phi_width(self):
        return self.av.width + self.fleet_max_length

    @property
    def core_length(self):
        return min(self.av.length + self.fleet_min_width, self.phi_length)

    @property
    def core_width(self):
        return min(self.av.width + self.fleet_min_width, self.phi_width)

    def conditional_collision_probability(self, x, y):
        """P(C | O) for AV-frame coordinates, vectorized.

        1 inside the guaranteed-collision subarea phi, 0 outside the safety
        box Phi, and a decay f in between. The paper leaves f unspecified
        ("can be derived through calibration"); here f is the linear decay of
        the nested-rectangle interpolation coordinate: f = 1 - t where t is
        how far the cell sits between the phi boundary (t=0) and the Phi
        boundary (t=1) along the more binding axis. See README,
        "Reproducibility gaps".
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        hx_out, hy_out = self.phi_length / 2.0, self.phi_width / 2.0
        hx_in, hy_in = self.core_length / 2.0, self.core_width / 2.0

        # Interpolation coordinate per axis; guard zero-thickness shells.
        dx = hx_out - hx_in
        dy = hy_out - hy_in
        tx = np.zeros_like(x) if dx <= 0 else (np.abs(x) - hx_in) / dx
        ty = np.zeros_like(y) if dy <= 0 else (np.abs(y) - hy_in) / dy
        t = np.clip(np.maximum(tx, ty), 0.0, 1.0)

        inside_phi = (np.abs(x) <= hx_out) & (np.abs(y) <= hy_out)
        return np.where(inside_phi, 1.0 - t, 0.0)


def fleet_extremes(fleet: Sequence[VehicleDims]):
    """(max length, min width) over surrounding traffic participants."""
    if not fleet:
        raise ValueError("fleet must contain at least one participant")
    return max(v.length for v in fleet), min(v.width for v in fleet)
