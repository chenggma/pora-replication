import unittest

import numpy as np

from pora.geometry import VehicleDims
from pora.occupancy_sources import FoeState, constant_velocity_gaussian


CAR = VehicleDims(4.5, 2.0)
GRID = dict(origin=(-25.0, -25.0), shape=(101, 101), resolution=0.5)


class TestConstantVelocityGaussian(unittest.TestCase):
    def test_peak_at_propagated_position(self):
        foe = FoeState(x=0.0, y=0.0, vx=10.0, vy=0.0, dims=CAR)
        grid = constant_velocity_gaussian([foe], lead_time=1.0, **GRID)
        peak = np.unravel_index(np.argmax(grid.data), grid.data.shape)
        px, py = grid.cell_center(peak[1], peak[0])
        self.assertAlmostEqual(px, 10.0, delta=0.5)
        self.assertAlmostEqual(py, 0.0, delta=0.5)

    def test_probabilities_in_unit_interval(self):
        foe = FoeState(0.0, 0.0, 0.0, 0.0, CAR)
        grid = constant_velocity_gaussian([foe], lead_time=0.0, **GRID)
        self.assertGreaterEqual(float(grid.data.min()), 0.0)
        self.assertLessEqual(float(grid.data.max()), 1.0)

    def test_uncertainty_grows_with_lead_time(self):
        foe = FoeState(0.0, 0.0, 0.0, 0.0, CAR)
        near = constant_velocity_gaussian([foe], lead_time=0.5, **GRID)
        far = constant_velocity_gaussian([foe], lead_time=4.0, **GRID)
        self.assertGreater(float(near.data.max()), float(far.data.max()))

    def test_two_foes_union_bound(self):
        f1 = FoeState(-5.0, 0.0, 0.0, 0.0, CAR)
        f2 = FoeState(5.0, 0.0, 0.0, 0.0, CAR)
        single = constant_velocity_gaussian([f1], lead_time=0.0, **GRID)
        union = constant_velocity_gaussian([f1, f2], lead_time=0.0, **GRID)
        self.assertGreaterEqual(
            float(union.data.sum()), float(single.data.sum()) - 1e-9
        )
        self.assertLessEqual(float(union.data.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
