"""Unofficial, independent implementation of the PORA collision-risk metric.

PORA (Probabilistic Occupancy Risk Assessment) is described in:

    Wang, Yeo, Paiva, Utke, Delle Monache,
    "Dynamic Risk Assessment for Autonomous Vehicles from Spatio-Temporal
    Probabilistic Occupancy Heatmaps", arXiv:2501.16480.

The authors of this package are NOT authors of the paper. Everything here is
re-implemented from the public arXiv text alone; where the paper does not
specify a quantity, the choice made here is documented in README.md under
"Reproducibility gaps".
"""

from .geometry import VehicleDims, SafetyBox, stopping_sight_distance
from .heatmap import OccupancyGrid
from .transform import to_av_frame
from .risk import collision_probability, cox_adjust, pora_horizon, PoraResult
from .occupancy_sources import constant_velocity_gaussian
from .baselines import ttc_point_mass, min_ttc, tts_margin

__all__ = [
    "VehicleDims",
    "SafetyBox",
    "stopping_sight_distance",
    "OccupancyGrid",
    "to_av_frame",
    "collision_probability",
    "cox_adjust",
    "pora_horizon",
    "PoraResult",
    "constant_velocity_gaussian",
    "ttc_point_mass",
    "min_ttc",
    "tts_margin",
]
