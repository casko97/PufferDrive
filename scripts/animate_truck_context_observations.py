from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from pufferlib.ocean.drive.drive import _EGO_TRAILER_STATE_FEATURES, binding

EXPORT_DEFAULT = Path("outputs/offline_fits/nuplan_boston_test10_paired_fits.pt")
OUTPUT_DEFAULT = Path("outputs/test_visualizations/truck_context_obs_map_000.gif")
BASE_EGO = binding.EGO_FEATURES_CLASSIC
AUG_EGO_FEATURES = BASE_EGO + 1 + _EGO_TRAILER_STATE_FEATURES
BASE_PARTNER_FEATURES = binding.PARTNER_FEATURES
AUG_PARTNER_FEATURES = binding.PARTNER_FEATURES + 1
PARTNER_COUNT = binding.MAX_AGENTS - 1
ROAD_FEATURES = binding.ROAD_FEATURES
ROAD_COUNT = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
MAX_VEH_LEN = 30.0
MAX_VEH_WIDTH = 15.0
MAX_ROAD_SEGMENT_LENGTH = 100.0


def _decode_local_obs(obs_vec: np.ndarray) -> dict:
    goal = (float(obs_vec[0] * 200.0), float(obs_vec[1] * 200.0))
    base_obs_dim = BASE_EGO + PARTNER_COUNT * BASE_PARTNER_FEATURES + ROAD_COUNT * ROAD_FEATURES
    aug_obs_dim = AUG_EGO_FEATURES + PARTNER_COUNT * AUG_PARTNER_FEATURES + ROAD_COUNT * ROAD_FEATURES
    is_augmented = len(obs_vec) == aug_obs_dim
    ego_features = AUG_EGO_FEATURES if is_augmented else BASE_EGO
    partner_features = AUG_PARTNER_FEATURES if is_augmented else BASE_PARTNER_FEATURES

    trailer = {"x": 0.0, "y": 0.0, "heading_x": 0.0, "heading_y": 0.0}
    if is_augmented:
        trailer = {
            "x": float(obs_vec[BASE_EGO + 1] * 50.0),
            "y": float(obs_vec[BASE_EGO + 2] * 50.0),
            "heading_x": float(obs_vec[BASE_EGO + 3]),
            "heading_y": float(obs_vec[BASE_EGO + 4]),
        }

    partners = []
    partner_start = ego_features
    for idx in range(PARTNER_COUNT):
        base = partner_start + idx * partner_features
        rel_x = float(obs_vec[base] * 50.0)
        rel_y = float(obs_vec[base + 1] * 50.0)
        width = float(obs_vec[base + 2] * MAX_VEH_WIDTH)
        length = float(obs_vec[base + 3] * MAX_VEH_LEN)
        heading_x = float(obs_vec[base + 4])
        heading_y = float(obs_vec[base + 5])
        type_id = float(obs_vec[base + 7]) if is_augmented else -1.0
        if rel_x == 0.0 and rel_y == 0.0 and width == 0.0 and length == 0.0:
            continue
        partners.append(
            {
                "x": rel_x,
                "y": rel_y,
                "width": width,
                "length": length,
                "heading_x": heading_x,
                "heading_y": heading_y,
                "type_id": type_id,
            }
        )

    roads = []
    road_start = partner_start + PARTNER_COUNT * partner_features
    for idx in range(ROAD_COUNT):
        base = road_start + idx * ROAD_FEATURES
        rel_x = float(obs_vec[base] * 50.0)
        rel_y = float(obs_vec[base + 1] * 50.0)
        seg_len = float(obs_vec[base + 2] * MAX_ROAD_SEGMENT_LENGTH)
        heading_x = float(obs_vec[base + 4])
        heading_y = float(obs_vec[base + 5])
        road_type = float(obs_vec[base + 6])
        if rel_x == 0.0 and rel_y == 0.0 and seg_len == 0.0:
            continue
        roads.append(
            {
                "x": rel_x,
                "y": rel_y,
                "length": seg_len,
                "heading_x": heading_x,
                "heading_y": heading_y,
                "road_type": road_type,
            }
        )

    return {"goal": goal, "trailer": trailer, "partners": partners, "roads": roads}


def _draw_local_frame(ax, obs_vec: np.ndarray, title: str):
    decoded = _decode_local_obs(obs_vec)
    ax.cla()
    ax.set_title(title, fontsize=10)
    ax.axhline(0.0, color="lightgray", linewidth=0.8)
    ax.axvline(0.0, color="lightgray", linewidth=0.8)
    ax.scatter([0.0], [0.0], s=70, color="black", label="Ego")
    goal_x, goal_y = decoded["goal"]
    ax.scatter([goal_x], [goal_y], s=40, color="green", marker="*", label="Goal")

    trailer = decoded["trailer"]
    if any(abs(trailer[k]) > 1e-6 for k in ("x", "y", "heading_x", "heading_y")):
        ax.scatter([trailer["x"]], [trailer["y"]], s=30, color="purple", label="Trailer")
        ax.plot(
            [trailer["x"], trailer["x"] + 4.0 * trailer["heading_x"]],
            [trailer["y"], trailer["y"] + 4.0 * trailer["heading_y"]],
            color="purple",
            linewidth=1.2,
        )

    for partner in decoded["partners"]:
        ax.scatter([partner["x"]], [partner["y"]], s=15, color="tab:orange")
        ax.plot(
            [partner["x"], partner["x"] + 3.0 * partner["heading_x"]],
            [partner["y"], partner["y"] + 3.0 * partner["heading_y"]],
            color="tab:orange",
            linewidth=1.0,
        )

    for road in decoded["roads"]:
        half_len = 0.5 * road["length"]
        ax.plot(
            [road["x"] - half_len * road["heading_x"], road["x"] + half_len * road["heading_x"]],
            [road["y"] - half_len * road["heading_y"], road["y"] + half_len * road["heading_y"]],
            color="tab:cyan",
            linewidth=0.9,
            alpha=0.8,
        )

    ax.set_xlim(-80.0, 80.0)
    ax.set_ylim(-80.0, 80.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


def animate_truck_context_observations(export_path: Path, map_name: str, output_path: Path) -> Path:
    payload = torch.load(export_path, map_location="cpu")
    pair = payload["pairs"][map_name]
    replay = pair["truck_context_replay"]
    if replay["status"] != "ok":
        raise ValueError(f"truck_context_replay unavailable for {map_name}: {replay.get('error')}")

    truck_obs = np.asarray(replay["truck_branch"]["obs"], dtype=np.float32)
    car_obs = np.asarray(replay["car_branch"]["obs"], dtype=np.float32)
    frame_count = min(len(truck_obs), len(car_obs))
    if frame_count <= 0:
        raise ValueError(f"no replay observations available for {map_name}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    def update(frame_idx: int):
        _draw_local_frame(axes[0], truck_obs[frame_idx], f"{map_name} truck actions\nframe={frame_idx}")
        _draw_local_frame(axes[1], car_obs[frame_idx], f"{map_name} car actions in truck scene\nframe={frame_idx}")
        if frame_idx == 0:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(loc="upper right", fontsize=8)
        return axes

    ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=150, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer=animation.PillowWriter(fps=6))
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Animate cached truck-context replay observations.")
    parser.add_argument("--input", type=Path, default=EXPORT_DEFAULT)
    parser.add_argument("--map-name", type=str, default="map_000.bin")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    output_path = animate_truck_context_observations(args.input, args.map_name, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
