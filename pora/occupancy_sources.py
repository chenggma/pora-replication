"""Simple occupancy-heatmap sources.

The paper trains a transformer-encoder / GAN-decoder predictor for its
heatmaps. This package deliberately does NOT reproduce that model: the
metric is the replication target. These sources provide physically
transparent heatmaps so the metric can be exercised and benchmarked.

constant_velocity_gaussian: each traffic participant is propagated at
constant velocity; positional uncertainty is an isotropic Gaussian whose
sigma grows linearly with lead time; the per-cell occupancy probability is
the Gaussian density integrated over the cell (approximated as density *
cell area), inflated by the participant footprint, capped at 1; several
participants combine as 1 - prod(1 - p).
"""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .geometry import VehicleDims
from .heatmap import OccupancyGrid


@dataclass(frozen=True)
class FoeState:
    x: float
    y: float
    vx: float
    vy: float
    dims: VehicleDims


def constant_velocity_gaussian(
    foes: Sequence[FoeState],
    lead_time: float,
    origin: Tuple[float, float],
    shape: Tuple[int, int],
    resolution: float,
    sigma0: float = 0.5,
    sigma_growth: float = 0.5,
):
    """Global-frame occupancy grid at `lead_time` seconds ahead.

    sigma(t) = sigma0 + sigma_growth * t, then inflated by a quarter of the
    foe's mean footprint dimension so larger vehicles occupy more cells.
    """
    ny, nx = shape
    xs = origin[0] + resolution * np.arange(nx)
    ys = origin[1] + resolution * np.arange(ny)
    gx, gy = np.meshgrid(xs, ys)

    free = np.ones((ny, nx), dtype=float)
    cell_area = resolution * resolution
    for foe in foes:
        mx = foe.x + foe.vx * lead_time
        my = foe.y + foe.vy * lead_time
        sigma = sigma0 + sigma_growth * lead_time
        sigma += (foe.dims.length + foe.dims.width) / 8.0
        d2 = (gx - mx) ** 2 + (gy - my) ** 2
        density = np.exp(-d2 / (2.0 * sigma * sigma)) / (
            2.0 * math.pi * sigma * sigma
        )
        # Footprint inflation: `density * cell_area` is the chance the foe
        # CENTER falls in the cell; multiplying by footprint/cell_area
        # approximates the chance any part of the body overlaps the cell.
        # Net: p = density * footprint, capped at 1.
        footprint = foe.dims.length * foe.dims.width
        p = np.clip(density * footprint, 0.0, 1.0)
        free *= 1.0 - p

    return OccupancyGrid(
        data=1.0 - free, resolution=resolution, origin=origin, frame="global"
    )
