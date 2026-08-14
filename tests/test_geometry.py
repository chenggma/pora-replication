import unittest

import numpy as np

from pora.geometry import (
    SafetyBox,
    VehicleDims,
    fleet_extremes,
    stopping_sight_distance,
)


AV = VehicleDims(length=4.5, width=2.0)


def make_box(speed=0.0):
    return SafetyBox(
        av=AV,
        fleet_max_length=5.0,
        fleet_min_width=1.8,
        speed_ms=speed,
        reaction_time_s=1.0,
        decel_ms2=9.81,
    )


class TestSSD(unittest.TestCase):
    def test_zero_speed_is_zero(self):
        self.assertEqual(stopping_sight_distance(0.0), 0.0)

    def test_si_formula(self):
        # v*r + v^2/(2a) at v=10, r=1, a=9.81
        expected = 10.0 + 100.0 / (2.0 * 9.81)
        self.assertAlmostEqual(stopping_sight_distance(10.0), expected, places=9)

    def test_negative_speed_raises(self):
        with self.assertRaises(ValueError):
            stopping_sight_distance(-1.0)


class TestBoxDimensions(unittest.TestCase):
    def test_paper_formulas_at_rest(self):
        box = make_box(0.0)
        self.assertAlmostEqual(box.phi_length, 4.5 + 5.0)
        self.assertAlmostEqual(box.phi_width, 2.0 + 5.0)
        self.assertAlmostEqual(box.core_length, 4.5 + 1.8)
        self.assertAlmostEqual(box.core_width, 2.0 + 1.8)

    def test_length_grows_with_speed_width_constant(self):
        slow, fast = make_box(5.0), make_box(20.0)
        self.assertGreater(fast.phi_length, slow.phi_length)
        self.assertEqual(fast.phi_width, slow.phi_width)

    def test_core_clamped_inside_box(self):
        # Degenerate fleet: min width larger than max length + SSD headroom.
        box = SafetyBox(
            av=AV,
            fleet_max_length=0.5,
            fleet_min_width=3.0,
            speed_ms=0.0,
        )
        self.assertLessEqual(box.core_length, box.phi_length)
        self.assertLessEqual(box.core_width, box.phi_width)


class TestConditionalCollisionProbability(unittest.TestCase):
    def test_center_is_one(self):
        self.assertEqual(make_box().conditional_collision_probability(0.0, 0.0), 1.0)

    def test_outside_box_is_zero(self):
        box = make_box()
        val = box.conditional_collision_probability(box.phi_length, 0.0)
        self.assertEqual(val, 0.0)

    def test_midshell_is_half(self):
        box = make_box()
        hx_in = box.core_length / 2.0
        hx_out = box.phi_length / 2.0
        x_mid = (hx_in + hx_out) / 2.0
        val = box.conditional_collision_probability(x_mid, 0.0)
        self.assertAlmostEqual(float(val), 0.5, places=9)

    def test_monotone_decay_along_axis(self):
        box = make_box()
        xs = np.linspace(0.0, box.phi_length / 2.0 + 1.0, 50)
        vals = box.conditional_collision_probability(xs, np.zeros_like(xs))
        self.assertTrue(np.all(np.diff(vals) <= 1e-12))

    def test_symmetry(self):
        box = make_box()
        a = box.conditional_collision_probability(1.7, 0.9)
        b = box.conditional_collision_probability(-1.7, -0.9)
        self.assertAlmostEqual(float(a), float(b), places=12)

    def test_vectorized_matches_scalar(self):
        box = make_box()
        xs = np.array([0.0, 1.0, 3.0, 6.0])
        ys = np.array([0.0, 0.5, 1.5, 0.0])
        vec = box.conditional_collision_probability(xs, ys)
        for i in range(len(xs)):
            self.assertAlmostEqual(
                float(vec[i]),
                float(box.conditional_collision_probability(xs[i], ys[i])),
                places=12,
            )


class TestFleetExtremes(unittest.TestCase):
    def test_extremes(self):
        fleet = [VehicleDims(4.0, 1.8), VehicleDims(12.0, 2.5), VehicleDims(2.0, 0.8)]
        max_l, min_w = fleet_extremes(fleet)
        self.assertEqual(max_l, 12.0)
        self.assertEqual(min_w, 0.8)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            fleet_extremes([])


if __name__ == "__main__":
    unittest.main()
