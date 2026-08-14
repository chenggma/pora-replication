import math
import unittest

import numpy as np

from pora.heatmap import OccupancyGrid
from pora.transform import av_frame_cell_centers, to_av_frame


def hot_cell_grid(world_x, world_y, extent=21, resolution=1.0):
    """Grid with a single 1.0 cell at integer world coords, origin at
    (-extent//2, -extent//2)."""
    half = extent // 2
    data = np.zeros((extent, extent))
    ix = int(world_x + half)
    jy = int(world_y + half)
    data[jy, ix] = 1.0
    return OccupancyGrid(data, resolution, (-float(half), -float(half)))


class TestToAvFrame(unittest.TestCase):
    def test_identity_heading_zero(self):
        grid = hot_cell_grid(5, 0)
        local = to_av_frame(grid, 0.0, 0.0, 0.0, half_length=8, half_width=4)
        xl, yl = av_frame_cell_centers(local)
        peak = np.unravel_index(np.argmax(local.data), local.data.shape)
        self.assertAlmostEqual(xl[peak], 5.0)
        self.assertAlmostEqual(yl[peak], 0.0)
        self.assertAlmostEqual(float(local.data[peak]), 1.0, places=9)

    def test_rotation_quarter_turn(self):
        # Foe due north in world; AV heading north: foe should be dead ahead
        # (+x) in the AV frame.
        grid = hot_cell_grid(0, 5)
        local = to_av_frame(
            grid, 0.0, 0.0, math.pi / 2.0, half_length=8, half_width=4
        )
        xl, yl = av_frame_cell_centers(local)
        peak = np.unravel_index(np.argmax(local.data), local.data.shape)
        self.assertAlmostEqual(xl[peak], 5.0)
        self.assertAlmostEqual(abs(float(yl[peak])), 0.0)

    def test_translation(self):
        grid = hot_cell_grid(7, 3)
        local = to_av_frame(grid, 7.0, 0.0, 0.0, half_length=6, half_width=6)
        xl, yl = av_frame_cell_centers(local)
        peak = np.unravel_index(np.argmax(local.data), local.data.shape)
        self.assertAlmostEqual(xl[peak], 0.0)
        self.assertAlmostEqual(yl[peak], 3.0)

    def test_frame_tag_and_shape(self):
        grid = hot_cell_grid(0, 0)
        local = to_av_frame(grid, 0, 0, 0, half_length=5, half_width=2)
        self.assertEqual(local.frame, "av")
        ny, nx = local.shape
        self.assertEqual(nx, 11)
        self.assertEqual(ny, 5)

    def test_out_of_coverage_is_zero(self):
        grid = hot_cell_grid(5, 0, extent=11)
        # AV far away: window sees nothing.
        local = to_av_frame(grid, 100.0, 100.0, 0.0, half_length=5, half_width=5)
        self.assertEqual(float(local.data.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
