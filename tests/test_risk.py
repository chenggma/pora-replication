import math
import unittest

import numpy as np

from pora.geometry import SafetyBox, VehicleDims
from pora.heatmap import OccupancyGrid
from pora.risk import collision_probability, cox_adjust, pora_horizon


AV = VehicleDims(4.5, 2.0)


def make_box(speed=0.0):
    return SafetyBox(
        av=AV, fleet_max_length=5.0, fleet_min_width=1.8, speed_ms=speed
    )


def av_grid(data, resolution=0.5):
    ny, nx = np.asarray(data).shape
    origin = (
        -resolution * (nx - 1) / 2.0,
        -resolution * (ny - 1) / 2.0,
    )
    return OccupancyGrid(np.asarray(data, dtype=float), resolution, origin, frame="av")


class TestCollisionProbability(unittest.TestCase):
    def test_certain_occupancy_at_center_gives_one(self):
        data = np.zeros((5, 5))
        data[2, 2] = 1.0  # center cell -> inside phi
        pc = collision_probability(av_grid(data), make_box())
        self.assertAlmostEqual(float(pc.max()), 1.0, places=12)

    def test_empty_grid_gives_zero(self):
        pc = collision_probability(av_grid(np.zeros((5, 5))), make_box())
        self.assertEqual(float(pc.max()), 0.0)

    def test_rejects_global_frame(self):
        grid = OccupancyGrid(np.zeros((3, 3)), 1.0, (0.0, 0.0), frame="global")
        with self.assertRaises(ValueError):
            collision_probability(grid, make_box())

    def test_scales_with_occupancy(self):
        half = np.zeros((5, 5))
        half[2, 2] = 0.5
        full = np.zeros((5, 5))
        full[2, 2] = 1.0
        box = make_box()
        pc_half = collision_probability(av_grid(half), box)
        pc_full = collision_probability(av_grid(full), box)
        self.assertAlmostEqual(float(pc_half.max()), 0.5 * float(pc_full.max()))


class TestCoxAdjust(unittest.TestCase):
    def test_static_scene_discounted_by_exp_beta(self):
        pc = np.array([0.5])
        po = np.array([0.7])
        out = cox_adjust(pc, po, po, beta=1.0)
        self.assertAlmostEqual(float(out[0]), 0.5 / math.e, places=12)

    def test_maximal_increase_preserves_pc(self):
        pc = np.array([0.5])
        out = cox_adjust(pc, np.array([1.0]), np.array([0.0]), beta=2.0)
        self.assertAlmostEqual(float(out[0]), 0.5, places=12)

    def test_decreasing_occupancy_discounts_harder(self):
        pc = np.array([0.5])
        static = cox_adjust(pc, np.array([0.5]), np.array([0.5]), beta=1.0)
        falling = cox_adjust(pc, np.array([0.2]), np.array([0.5]), beta=1.0)
        self.assertLess(float(falling[0]), float(static[0]))


class TestPoraHorizon(unittest.TestCase):
    def test_first_step_unadjusted_and_scalar_is_max(self):
        data = np.zeros((5, 5))
        data[2, 2] = 1.0
        grids = [av_grid(data), av_grid(data)]
        boxes = [make_box(), make_box()]
        res = pora_horizon(grids, boxes, beta=1.0)
        self.assertAlmostEqual(res.per_step[0], 1.0, places=12)
        # Static occupancy at step 2: discounted by e^-1.
        self.assertAlmostEqual(res.per_step[1], math.exp(-1.0), places=9)
        self.assertAlmostEqual(res.scalar, 1.0, places=12)
        self.assertEqual(len(res.per_step_unadjusted), 2)

    def test_shape_mismatch_raises(self):
        g1 = av_grid(np.zeros((5, 5)))
        g2 = av_grid(np.zeros((7, 7)))
        with self.assertRaises(ValueError):
            pora_horizon([g1, g2], [make_box(), make_box()])

    def test_empty_horizon_raises(self):
        with self.assertRaises(ValueError):
            pora_horizon([], [])

    def test_box_count_mismatch_raises(self):
        g = av_grid(np.zeros((5, 5)))
        with self.assertRaises(ValueError):
            pora_horizon([g], [make_box(), make_box()])

    def test_result_bounded_by_one(self):
        rng = np.random.default_rng(20260813)
        grids = [av_grid(rng.uniform(0, 1, (9, 9))) for _ in range(5)]
        boxes = [make_box(speed=10.0)] * 5
        res = pora_horizon(grids, boxes, beta=1.5)
        self.assertLessEqual(res.scalar, 1.0 + 1e-9)
        self.assertGreaterEqual(min(res.per_step), 0.0)


if __name__ == "__main__":
    unittest.main()
