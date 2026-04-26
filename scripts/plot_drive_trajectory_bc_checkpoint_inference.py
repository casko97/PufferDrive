#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pufferlib.ocean.drive.drive import _decode_trajectory_position
from pufferlib.ocean.drive.trajectory_bc import _build_policy_env_spec, TrajectoryBCEnvConfig
from pufferlib.ocean.drive.trajectory_bc_viz import (
    _base_obs_dim,
    _decode_history_slot,
    _decode_observation_road_segments,
    _parse_roads_with_fallback,
    _set_zoomed_limits,
    _transform_points_to_ego_frame,
    _TRAJECTORY_FEATURES,
    _TRAJECTORY_HORIZON,
    _TRAJECTORY_HISTORY_FEATURES,
    collect_visualization_sample,
)
from pufferlib.ocean.torch import Drive as DrivePolicy


def _load_policy(checkpoint_path: Path, device: torch.device) -> tuple[DrivePolicy, dict]:
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload["config"]
    env = _build_policy_env_spec(config["env"]["dynamics_model"])
    policy = DrivePolicy(env, **config["policy"])
    policy.load_state_dict(payload["model_state_dict"])
    policy.to(device)
    policy.eval()
    return policy, config


def _predict_trajectory(policy: DrivePolicy, observation: np.ndarray, device: torch.device) -> np.ndarray:
    obs = torch.from_numpy(observation[None]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        action_dist, _value = policy(obs)
    if hasattr(action_dist, "mean"):
        pred = action_dist.mean
    elif hasattr(action_dist, "loc"):
        pred = action_dist.loc
    else:
        pred = action_dist
    return pred.detach().cpu().numpy().reshape(_TRAJECTORY_HORIZON, _TRAJECTORY_FEATURES)


def _decode_xy(trajectory_block: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    if valid_mask is None:
        valid_mask = np.ones((trajectory_block.shape[0],), dtype=bool)
    if not np.any(valid_mask):
        return np.zeros((0, 2), dtype=np.float32)
    xs = [_decode_trajectory_position(float(value)) for value in trajectory_block[valid_mask, 0]]
    ys = [_decode_trajectory_position(float(value)) for value in trajectory_block[valid_mask, 1]]
    return np.stack([np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)], axis=1)


def render_checkpoint_inference_plot(
    *,
    checkpoint_path: Path,
    map_path: Path,
    output_path: Path,
    timestep: int | None = None,
    ego_row: int = 0,
    road_source: str = "map",
) -> Path:
    device = torch.device("cpu")
    policy, config = _load_policy(checkpoint_path, device)
    env_config = TrajectoryBCEnvConfig(**config["env"])

    observations, targets, current_states, _partner_ids, _ego_ids, resolved_timestep = collect_visualization_sample(
        map_path=map_path,
        timestep=timestep,
        env_config=env_config,
    )
    observation = observations[ego_row].astype(np.float32)
    target = targets[ego_row].reshape(_TRAJECTORY_HORIZON, _TRAJECTORY_FEATURES)
    prediction = _predict_trajectory(policy, observation, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_observation = observation[:_base_obs_dim()]
    ego_hist_start = _base_obs_dim()
    partner_hist_start = ego_hist_start + (_TRAJECTORY_HORIZON * _TRAJECTORY_HISTORY_FEATURES)
    ego_history = observation[ego_hist_start:partner_hist_start].reshape(_TRAJECTORY_HORIZON, _TRAJECTORY_HISTORY_FEATURES)
    partner_history = observation[partner_hist_start:].reshape(-1, _TRAJECTORY_HORIZON, _TRAJECTORY_HISTORY_FEATURES)

    current_x = float(current_states["x"][ego_row])
    current_y = float(current_states["y"][ego_row])
    current_heading = float(current_states["heading"][ego_row])

    if road_source == "map":
        roads = _parse_roads_with_fallback(map_path)
        road_segments = []
        for road in roads:
            road_x = road.x if hasattr(road, "x") else road["x"]
            road_y = road.y if hasattr(road, "y") else road["y"]
            road_type = road.entity_type if hasattr(road, "entity_type") else road["entity_type"]
            local_x, local_y = _transform_points_to_ego_frame(road_x, road_y, current_x, current_y, current_heading)
            road_segments.append({"x": local_x, "y": local_y, "type": road_type})
    else:
        road_segments = _decode_observation_road_segments(base_observation, treat_length_as_half_segment=False)

    ego_xy = _decode_history_slot(ego_history)
    partner_xy_groups = []
    for slot_block in partner_history:
        partner_xy = _decode_history_slot(slot_block)
        if partner_xy.shape[0] > 0:
            partner_xy_groups.append(partner_xy)

    target_xy = _decode_xy(target, target[:, 4] > 0.5)
    pred_valid = 1.0 / (1.0 + np.exp(-prediction[:, 4])) > 0.5
    pred_xy = _decode_xy(prediction, pred_valid)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    for road in road_segments:
        color = "#777777" if road["type"] == 4 else "#222222"
        linewidth = 1.0 if road["type"] == 4 else 1.6
        ax.plot(road["x"], road["y"], color=color, linewidth=linewidth, alpha=0.85, zorder=0)

    if ego_xy.size:
        ax.plot(ego_xy[:, 0], ego_xy[:, 1], color="tab:blue", linewidth=2.2, marker="o", markersize=3, label="Ego history")
    plotted_partner = False
    for partner_xy in partner_xy_groups:
        ax.plot(
            partner_xy[:, 0],
            partner_xy[:, 1],
            color="tab:orange",
            linewidth=1.6,
            linestyle="--",
            marker="o",
            markersize=2.5,
            alpha=0.8,
            label=None if plotted_partner else "Partner history",
        )
        plotted_partner = True
    if target_xy.size:
        ax.plot(target_xy[:, 0], target_xy[:, 1], color="tab:green", linewidth=2.6, marker="x", markersize=4, label="Target future")
    if pred_xy.size:
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="tab:purple", linewidth=2.6, marker="s", markersize=3, label="Predicted future")

    ax.scatter([0.0], [0.0], color="crimson", s=50, zorder=5, label="Current ego")
    _set_zoomed_limits(ax, [ego_xy, target_xy, pred_xy] + partner_xy_groups)
    ax.set_title(f"Checkpoint inference ({checkpoint_path.name}, {map_path.name}, t={resolved_timestep})")
    ax.set_xlabel("x in current ego frame [m]")
    ax.set_ylabel("y in current ego frame [m]")
    ax.axhline(0.0, color="0.9", linewidth=0.8)
    ax.axvline(0.0, color="0.9", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trajectory BC checkpoint inference on one scenario sample.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to `best.pt` or `last.pt`.")
    parser.add_argument("--map-path", type=Path, required=True, help="Path to the `.bin` scenario.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("tests/test-viz/trajectory_bc_checkpoint_inference.png"),
        help="Where to save the rendered inference plot.",
    )
    parser.add_argument("--timestep", type=int, default=None, help="Optional logged timestep override.")
    parser.add_argument("--road-source", choices=("map", "observation"), default="map")
    args = parser.parse_args()

    result = render_checkpoint_inference_plot(
        checkpoint_path=args.checkpoint_path.expanduser().resolve(),
        map_path=args.map_path.expanduser().resolve(),
        output_path=args.output_path.expanduser().resolve(),
        timestep=args.timestep,
        road_source=args.road_source,
    )
    print(result)


if __name__ == "__main__":
    main()
