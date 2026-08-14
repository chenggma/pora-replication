"""End-to-end: occupancy source -> AV-frame transform -> PORA score."""

import unittest

from pora.geometry import SafetyBox, VehicleDims
from pora.occupancy_sources import FoeState, constant_velocity_gaussian
from pora.risk import pora_horizon
from pora.transform import to_av_frame


AV_DIMS = VehicleDims(4.5, 2.0)
CAR = VehicleDims(4.5, 2.0)


def score_scenario(foe, av_speed=10.0, horizon_s=3.0, dt=0.5, beta=1.0):
    """AV drives +x from origin at constant speed; PORA over the horizon."""
    steps = int(horizon_s / dt) + 1
    grids, boxes = [], []
    for k in range(steps):
        t = k * dt
        global_grid = constant_velocity_gaussian(
            [foe],
            lead_time=t,
            origin=(-10.0, -30.0),
            shape=(121, 161),
            resolution=0.5,
        )
        av_x = av_speed * t
        box = SafetyBox(
            av=AV_DIMS,
            fleet_max_length=foe.dims.length,
            fleet_min_width=foe.dims.width,
            speed_ms=av_speed,
        )
        half_len = box.phi_length / 2.0 + 2.0
        half_wid = box.phi_width / 2.0 + 2.0
        grids.append(
            to_av_frame(global_grid, av_x, 0.0, 0.0, half_len, half_wid, 0.5)
        )
        boxes.append(box)
    return pora_horizon(grids, boxes, beta=beta)


class TestEndToEnd(unittest.TestCase):
    def test_head_on_riskier_than_distant(self):
        oncoming = FoeState(x=40.0, y=0.0, vx=-10.0, vy=0.0, dims=CAR)
        distant = FoeState(x=40.0, y=-25.0, vx=0.0, vy=0.0, dims=CAR)
        risk_close = score_scenario(oncoming).scalar
        risk_far = score_scenario(distant).scalar
        self.assertGreater(risk_close, risk_far)
        self.assertGreater(risk_close, 0.01)

    def test_receding_foe_low_risk(self):
        receding = FoeState(x=30.0, y=0.0, vx=25.0, vy=0.0, dims=CAR)
        oncoming = FoeState(x=30.0, y=0.0, vx=-10.0, vy=0.0, dims=CAR)
        self.assertLess(
            score_scenario(receding).scalar, score_scenario(oncoming).scalar
        )

    def test_scores_bounded(self):
        foe = FoeState(x=15.0, y=0.0, vx=-15.0, vy=0.0, dims=CAR)
        res = score_scenario(foe)
        self.assertLessEqual(res.scalar, 1.0 + 1e-9)
        self.assertGreaterEqual(res.scalar, 0.0)
        self.assertEqual(len(res.per_step), 7)


if __name__ == "__main__":
    unittest.main()
