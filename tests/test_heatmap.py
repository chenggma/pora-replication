import unittest

import numpy as np

from pora.heatmap import OccupancyGrid


class TestValidation(unittest.TestCase):
    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            OccupancyGrid(np.zeros(4), 1.0, (0.0, 0.0))

    def test_rejects_bad_resolution(self):
        with self.assertRaises(ValueError):
            OccupancyGrid(np.zeros((2, 2)), 0.0, (0.0, 0.0))

    def test_rejects_out_of_range_probability(self):
        with self.assertRaises(ValueError):
            OccupancyGrid(np.array([[0.0, 1.5]]), 1.0, (0.0, 0.0))


class TestSampling(unittest.TestCase):
    def setUp(self):
        # 3x3 grid, resolution 1, origin (0,0): cell centers at 0,1,2.
        self.data = np.array(
            [
                [0.0, 0.2, 0.4],
                [0.1, 0.5, 0.9],
                [0.0, 0.3, 0.6],
            ]
        )
        self.grid = OccupancyGrid(self.data, 1.0, (0.0, 0.0))

    def test_exact_at_cell_centers(self):
        for jy in range(3):
            for ix in range(3):
                self.assertAlmostEqual(
                    float(self.grid.sample_bilinear(ix, jy)),
                    float(self.data[jy, ix]),
                    places=12,
                )

    def test_midpoint_is_mean_of_neighbors(self):
        v = float(self.grid.sample_bilinear(0.5, 1.0))
        self.assertAlmostEqual(v, (0.1 + 0.5) / 2.0, places=12)

    def test_outside_is_zero(self):
        self.assertEqual(float(self.grid.sample_bilinear(-5.0, 1.0)), 0.0)
        self.assertEqual(float(self.grid.sample_bilinear(1.0, 99.0)), 0.0)

    def test_cell_center_coordinates(self):
        self.assertEqual(self.grid.cell_center(2, 1), (2.0, 1.0))


if __name__ == "__main__":
    unittest.main()
