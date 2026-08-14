import math
import unittest

from pora.baselines import min_ttc, ttc_point_mass, tts_margin
from pora.geometry import VehicleDims
from pora.occupancy_sources import FoeState


CAR = VehicleDims(4.0, 2.0)


class _Actor:
    def __init__(self, x, y, vx, vy, dims=CAR):
        self.x, self.y, self.vx, self.vy, self.dims = x, y, vx, vy, dims


class TestTtcPointMass(unittest.TestCase):
    def test_head_on_analytic(self):
        # Gap closes at 10 m/s from 100 m; touch at combined radius 2 m.
        t = ttc_point_mass((0, 0), (10, 0), (100, 0), (0, 0), radius=2.0)
        self.assertAlmostEqual(t, 9.8, places=9)

    def test_already_overlapping_is_zero(self):
        t = ttc_point_mass((0, 0), (0, 0), (1, 0), (0, 0), radius=2.0)
        self.assertEqual(t, 0.0)

    def test_diverging_is_inf(self):
        t = ttc_point_mass((0, 0), (-5, 0), (10, 0), (5, 0), radius=2.0)
        self.assertTrue(math.isinf(t))

    def test_parallel_same_velocity_is_inf(self):
        t = ttc_point_mass((0, 0), (7, 0), (10, 0), (7, 0), radius=2.0)
        self.assertTrue(math.isinf(t))

    def test_lateral_miss_is_inf(self):
        t = ttc_point_mass((0, 0), (10, 0), (100, 50), (0, 0), radius=2.0)
        self.assertTrue(math.isinf(t))


class TestMinTtc(unittest.TestCase):
    def test_picks_most_imminent_foe(self):
        av = _Actor(0, 0, 10, 0)
        near = _Actor(50, 0, 0, 0)
        far = _Actor(200, 0, 0, 0)
        t_both = min_ttc(av, [near, far])
        t_near = min_ttc(av, [near])
        self.assertAlmostEqual(t_both, t_near, places=12)

    def test_default_radius_is_quarter_dims_sum(self):
        av = _Actor(0, 0, 10, 0)
        foe = _Actor(100, 0, 0, 0)
        expected_r = (4.0 + 2.0) / 4.0 * 2
        t = min_ttc(av, [foe])
        self.assertAlmostEqual(t, (100.0 - expected_r) / 10.0, places=9)

    def test_no_foes_is_inf(self):
        self.assertTrue(math.isinf(min_ttc(_Actor(0, 0, 1, 0), [])))


class TestTtsMargin(unittest.TestCase):
    def test_margin_formula(self):
        # 20 m/s, a=10: stop time = 1 + 2 = 3 s; TTC 5 -> margin 2.
        m = tts_margin(20.0, ttc=5.0, decel_ms2=10.0, reaction_time_s=1.0)
        self.assertAlmostEqual(m, 2.0, places=12)

    def test_negative_margin_when_unavoidable(self):
        m = tts_margin(30.0, ttc=1.0, decel_ms2=9.81, reaction_time_s=1.0)
        self.assertLess(m, 0.0)

    def test_inf_ttc_gives_inf_margin(self):
        self.assertTrue(math.isinf(tts_margin(10.0, ttc=math.inf)))

    def test_negative_speed_raises(self):
        with self.assertRaises(ValueError):
            tts_margin(-1.0, ttc=5.0)


if __name__ == "__main__":
    unittest.main()
