"""Regenerate every figure in this repository from the package itself.

Usage:  python figures/make_figures.py
Writes PNGs next to this script. No external data: every figure is computed
from the implementation on a fixed synthetic scene, so the images are
byte-reproducible given the same library versions.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pora import (
    SafetyBox,
    VehicleDims,
    constant_velocity_gaussian,
    pora_horizon,
    to_av_frame,
)
from pora.occupancy_sources import FoeState
from pora.transform import av_frame_cell_centers

HERE = os.path.dirname(os.path.abspath(__file__))

CAR = VehicleDims(4.5, 2.0)
EGO_SPEED = 10.0
ONCOMING = FoeState(x=40.0, y=0.0, vx=-10.0, vy=0.0, dims=CAR)
CROSSING = FoeState(x=25.0, y=-18.0, vx=0.0, vy=9.0, dims=CAR)
RECEDING = FoeState(x=30.0, y=0.0, vx=25.0, vy=0.0, dims=CAR)

GRID = dict(origin=(-10.0, -30.0), shape=(121, 161), resolution=0.5)

INK = "#333333"
ACCENT = "#D55E00"  # safety box
ACCENT2 = "#009E73"  # core
SERIES = ["#0072B2", "#D55E00", "#009E73"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(colors=INK, labelsize=9)


def fig_occupancy_evolution():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, t in zip(axes, [0.0, 1.0, 2.0, 3.0]):
        g = constant_velocity_gaussian([ONCOMING, CROSSING], lead_time=t, **GRID)
        extent = [
            GRID["origin"][0],
            GRID["origin"][0] + (GRID["shape"][1] - 1) * GRID["resolution"],
            GRID["origin"][1],
            GRID["origin"][1] + (GRID["shape"][0] - 1) * GRID["resolution"],
        ]
        ax.imshow(
            g.data, origin="lower", extent=extent, cmap="Blues",
            vmin=0, vmax=1, aspect="equal",
        )
        ex = EGO_SPEED * t
        ax.plot(ex, 0, marker=(3, 0, -90), color=ACCENT, markersize=11)
        ax.add_patch(FancyArrow(ex, 0, 6, 0, width=0.4, color=ACCENT,
                                length_includes_head=True))
        ax.set_title(f"t = {t:.0f} s", fontsize=10, color=INK)
        ax.set_xlabel("x (m)", fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("y (m)", fontsize=9, color=INK)
    fig.suptitle(
        "Constant-velocity Gaussian occupancy, propagated over the horizon "
        "(ego marker moves with its plan)",
        fontsize=11, color=INK,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_occupancy_evolution.png"), dpi=160)
    plt.close(fig)


def fig_av_frame():
    box = SafetyBox(av=CAR, fleet_max_length=4.5, fleet_min_width=2.0,
                    speed_ms=EGO_SPEED)
    hl = box.phi_length / 2.0 + 3.0
    hw = box.phi_width / 2.0 + 3.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    # Left: the conditional collision probability field alone.
    xs = np.linspace(-hl, hl, 321)
    ys = np.linspace(-hw, hw, 121)
    gx, gy = np.meshgrid(xs, ys)
    pcgo = box.conditional_collision_probability(gx, gy)
    im = axes[0].imshow(
        pcgo, origin="lower", extent=[-hl, hl, -hw, hw], cmap="Blues",
        vmin=0, vmax=1, aspect="equal",
    )
    axes[0].set_title("P(C | O): 1 in the core, linear decay to the box edge",
                      fontsize=10, color=INK)

    # Right: AV-frame occupancy at t = 1.5 s with box outlines.
    t = 1.5
    g = constant_velocity_gaussian([ONCOMING, CROSSING], lead_time=t, **GRID)
    local = to_av_frame(g, EGO_SPEED * t, 0.0, 0.0, hl, hw, 0.25)
    xl, yl = av_frame_cell_centers(local)
    axes[1].imshow(
        local.data, origin="lower",
        extent=[xl.min(), xl.max(), yl.min(), yl.max()],
        cmap="Blues", vmin=0, vmax=local.data.max() or 1, aspect="equal",
    )
    axes[1].set_title("AV-frame occupancy at t = 1.5 s", fontsize=10, color=INK)

    for ax in axes:
        ax.add_patch(Rectangle(
            (-box.phi_length / 2, -box.phi_width / 2),
            box.phi_length, box.phi_width,
            fill=False, edgecolor=ACCENT, linewidth=2, label="safety box Φ",
        ))
        ax.add_patch(Rectangle(
            (-box.core_length / 2, -box.core_width / 2),
            box.core_length, box.core_width,
            fill=False, edgecolor=ACCENT2, linewidth=2, linestyle="--",
            label="core φ",
        ))
        ax.plot(0, 0, marker=(3, 0, -90), color=INK, markersize=9)
        ax.set_xlabel("ego frame x (m)", fontsize=9, color=INK)
        ax.set_ylabel("ego frame y (m)", fontsize=9, color=INK)
        style(ax)
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="probability")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_av_frame.png"), dpi=160)
    plt.close(fig)


def scenario_series(foe, beta=1.0, horizon_s=3.0, dt=0.5):
    steps = int(horizon_s / dt) + 1
    grids, boxes = [], []
    for k in range(steps):
        t = k * dt
        g = constant_velocity_gaussian([foe], lead_time=t, **GRID)
        box = SafetyBox(av=CAR, fleet_max_length=foe.dims.length,
                        fleet_min_width=foe.dims.width, speed_ms=EGO_SPEED)
        grids.append(to_av_frame(
            g, EGO_SPEED * t, 0.0, 0.0,
            box.phi_length / 2 + 2, box.phi_width / 2 + 2, 0.5,
        ))
        boxes.append(box)
    return pora_horizon(grids, boxes, beta=beta)


def fig_pora_horizon():
    ts = np.arange(0, 3.5, 0.5)
    on = scenario_series(ONCOMING)
    off = scenario_series(RECEDING)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(ts, on.per_step, color=SERIES[1], linewidth=2, marker="o",
            markersize=5, label="oncoming foe - Cox-adjusted")
    ax.plot(ts, on.per_step_unadjusted, color=SERIES[1], linewidth=1.5,
            linestyle=":", label="oncoming foe - unadjusted")
    ax.plot(ts, off.per_step, color=SERIES[2], linewidth=2, marker="o",
            markersize=5, label="receding foe - Cox-adjusted")
    ax.plot(ts, off.per_step_unadjusted, color=SERIES[2], linewidth=1.5,
            linestyle=":", label="receding foe - unadjusted")
    ax.set_xlabel("prediction step (s)", fontsize=10, color=INK)
    ax.set_ylabel("max cell risk", fontsize=10, color=INK)
    ax.set_title(
        "Per-step PORA (β = 1): rising occupancy amplified, "
        "static or falling occupancy discounted",
        fontsize=10, color=INK,
    )
    ax.legend(fontsize=8.5, framealpha=0.9)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_pora_horizon.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_occupancy_evolution()
    fig_av_frame()
    fig_pora_horizon()
    print("wrote 3 figures to", HERE)
