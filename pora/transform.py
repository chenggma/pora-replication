"""Global-frame to AV-centered-frame heatmap transformation (paper III-C).

The paper translates, rotates, and resamples the global heatmap so that the
result is aligned with the AV heading. Implementation: inverse mapping - for
every output cell center (in the AV frame) compute the corresponding world
point and bilinearly sample the global grid there. Inverse mapping avoids
the holes a forward scatter would leave.
"""

import math

import numpy as np

from .heatmap import OccupancyGrid


def to_av_frame(
    grid,
    av_x,
    av_y,
    av_heading_rad,
    half_length,
    half_width,
    resolution=None,
):
    """Resample `grid` into an AV-centered window.

    The output covers x in [-half_length, +half_length] (AV forward = +x)
    and y in [-half_width, +half_width], at `resolution` (defaults to the
    input grid's). Returns an OccupancyGrid with frame="av".
    """
    if resolution is None:
        resolution = grid.resolution
    nx = max(1, int(round(2.0 * half_length / resolution)) + 1)
    ny = max(1, int(round(2.0 * half_width / resolution)) + 1)

    xs = -half_length + resolution * np.arange(nx)
    ys = -half_width + resolution * np.arange(ny)
    xl, yl = np.meshgrid(xs, ys)

    c, s = math.cos(av_heading_rad), math.sin(av_heading_rad)
    wx = av_x + c * xl - s * yl
    wy = av_y + s * xl + c * yl

    data = grid.sample_bilinear(wx, wy)
    return OccupancyGrid(
        data=data,
        resolution=resolution,
        origin=(-half_length, -half_width),
        frame="av",
    )


def av_frame_cell_centers(local_grid):
    """(x, y) meshgrids of an AV-frame grid's cell centers."""
    ny, nx = local_grid.shape
    xs = local_grid.origin[0] + local_grid.resolution * np.arange(nx)
    ys = local_grid.origin[1] + local_grid.resolution * np.arange(ny)
    return np.meshgrid(xs, ys)
