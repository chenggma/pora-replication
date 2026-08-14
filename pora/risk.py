"""Risk computation: conditional collision probability, Cox adjustment,
horizon aggregation (paper III-D and III-E).

Pipeline per prediction timestep t_k:
  P_(i,j)(C)  = P(C | O)_(i,j) * P(O)_(i,j)
  P*_(i,j)    = P_(i,j)(C) * exp(beta * dP(O)_(i,j))     for k > 1
  Phat*_(i,j) = P*_(i,j) / exp(beta)
  per-step score = max over cells of Phat*
The paper does not state the aggregation across the K timesteps; this module
reports the per-step series and uses the max over the horizon as the scalar,
both returned so callers can choose.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .geometry import SafetyBox
from .transform import av_frame_cell_centers


@dataclass
class PoraResult:
    per_step: List[float]
    scalar: float
    per_step_unadjusted: List[float]


def collision_probability(local_grid, box: SafetyBox):
    """Cell-wise P(C) = P(C|O) * P(O) on an AV-frame grid."""
    if local_grid.frame != "av":
        raise ValueError("collision_probability expects an AV-frame grid")
    xl, yl = av_frame_cell_centers(local_grid)
    pcgo = box.conditional_collision_probability(xl, yl)
    return pcgo * local_grid.data


def cox_adjust(pc, po_now, po_prev, beta):
    """Cox-model adjustment for one timestep (k > 1), already normalized.

    Phat* = P(C) * exp(beta * (P_k(O) - P_{k-1}(O))) / exp(beta)

    With dP in [-1, 1] this stays within [0, P(C)]; equality with P(C) holds
    only when the occupancy probability rose by the maximum possible amount
    (dP = +1). This is exactly the normalization the paper states; note it
    implies a static scene (dP = 0) is discounted by e^-beta relative to the
    unadjusted risk.
    """
    d_po = np.asarray(po_now, dtype=float) - np.asarray(po_prev, dtype=float)
    return np.asarray(pc, dtype=float) * np.exp(beta * d_po) / math.exp(beta)


def pora_horizon(
    local_grids: Sequence,
    boxes: Sequence[SafetyBox],
    beta: float = 1.0,
) -> PoraResult:
    """Full-horizon PORA from AV-frame occupancy grids and per-step boxes.

    local_grids[k] is the AV-frame occupancy at prediction step k; boxes[k]
    the safety box for the AV state at that step (the box length depends on
    the planned speed). Grids must share one shape so dP(O) is cell-aligned.
    """
    if len(local_grids) != len(boxes):
        raise ValueError("need one safety box per grid")
    if not local_grids:
        raise ValueError("empty horizon")
    shape0 = local_grids[0].shape
    for g in local_grids:
        if g.shape != shape0:
            raise ValueError("all grids in a horizon must share a shape")

    per_step: List[float] = []
    per_step_raw: List[float] = []
    prev_po: Optional[np.ndarray] = None
    for k, (grid, box) in enumerate(zip(local_grids, boxes)):
        pc = collision_probability(grid, box)
        per_step_raw.append(float(pc.max()))
        if k == 0 or prev_po is None:
            adjusted = pc
        else:
            adjusted = cox_adjust(pc, grid.data, prev_po, beta)
        per_step.append(float(adjusted.max()))
        prev_po = grid.data

    return PoraResult(
        per_step=per_step,
        scalar=max(per_step),
        per_step_unadjusted=per_step_raw,
    )
