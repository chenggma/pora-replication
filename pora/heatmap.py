"""Occupancy grid container.

A grid stores P(cell occupied at time t) as a (ny, nx) array. `origin` is
the world coordinate of the CENTER of cell [0, 0]; cell [j, i] center is
origin + (i * resolution, j * resolution). Row index j moves along +y.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class OccupancyGrid:
    data: np.ndarray
    resolution: float
    origin: Tuple[float, float]
    frame: str = "global"

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError("data must be 2-D (ny, nx)")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if np.any(self.data < -1e-9) or np.any(self.data > 1.0 + 1e-9):
            raise ValueError("occupancy probabilities must lie in [0, 1]")
        self.data = np.clip(self.data, 0.0, 1.0)

    @property
    def shape(self):
        return self.data.shape

    def cell_center(self, ix, jy):
        return (
            self.origin[0] + ix * self.resolution,
            self.origin[1] + jy * self.resolution,
        )

    def sample_bilinear(self, x, y):
        """Bilinearly interpolated occupancy at world points (vectorized).

        Points outside the grid return 0 - unobserved space is treated as
        free, which is the conservative-for-precision choice and is flagged
        in the README.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        gx = (x - self.origin[0]) / self.resolution
        gy = (y - self.origin[1]) / self.resolution

        ny, nx = self.data.shape
        i0 = np.floor(gx).astype(int)
        j0 = np.floor(gy).astype(int)
        fx = gx - i0
        fy = gy - j0

        out = np.zeros(np.broadcast(gx, gy).shape, dtype=float)
        for di, dj, w in (
            (0, 0, (1 - fx) * (1 - fy)),
            (1, 0, fx * (1 - fy)),
            (0, 1, (1 - fx) * fy),
            (1, 1, fx * fy),
        ):
            ii = i0 + di
            jj = j0 + dj
            valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
            contrib = np.zeros_like(out)
            contrib[valid] = self.data[jj[valid], ii[valid]] * np.asarray(w)[valid]
            out += contrib
        return out
