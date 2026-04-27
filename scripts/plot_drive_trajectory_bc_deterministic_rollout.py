#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from matplotlib.patches import Circle

import pufferlib
from pufferlib import pufferl
from pufferlib.ocean.drive.trajectory_bc_viz import (
    _base_obs_dim,
    _decode_history_slot,
    _decode_observation_road_segments,
    _parse_roads_with_fallback,
    _transform_points_to_ego_frame,
    _trajectory_history_feature_dims,
    _TRAJECTORY_FEATURES,
    _TRAJECTORY_HORIZON,
)
from pufferlib.pytorch import eval_action_from_logits
from scripts.animate_drive_model_trajectories import build_args


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    return plt, FuncAnimation, PillowWriter


@dataclass
class RolloutFrame:
    timestep: int
    current_x: float
    current_y: float
    current_heading: float
    ego_history_xy: np.ndarray
    partner_histories_xy: list[np.ndarray]
    goal_xy: np.ndarray
    goal_radius: float
    target_xy: np.ndarray
    predicted_xy: np.ndarray
    rollout_xy: np.ndarray
    scenario_id: int
    observation_road_segments: list[dict[str, np.ndarray | int]]
    control_action: np.ndarray
    reward: float = 0.0
    collision_state: int = 0
    offroad_flag: bool = False
    reached_goal: bool = False
    stopped: bool = False
    speed: float = 0.0
    stop_reason: str = ""
    interrupted_after_step: bool = False
    interruption_reason: str = ""


@dataclass
class AnchoredDumpFrame:
    map_name: str
    timestep: int
    time_s: float
    current_world: list[float]
    current_anchor_xy: np.ndarray
    ego_history_xy_anchor: np.ndarray
    partner_histories_xy_anchor: list[np.ndarray]
    goal_xy_anchor: np.ndarray
    goal_radius: float
    predicted_xy_anchor: np.ndarray
    target_xy_anchor: np.ndarray
    rollout_xy_anchor: np.ndarray
    control_action: np.ndarray
    reward: float
    collision_state: int
    offroad_flag: bool
    reached_goal: bool
    stopped: bool
    speed: float
    stop_reason: str
    interrupted_after_step: bool
    interruption_reason: str


def _decode_xy(trajectory_block: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if not np.any(valid_mask):
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(
        [
            trajectory_block[valid_mask, 0] / 0.02,
            trajectory_block[valid_mask, 1] / 0.02,
        ],
        axis=1,
    ).astype(np.float32)


def _prediction_xy(prediction: np.ndarray) -> np.ndarray:
    pred_valid = 1.0 / (1.0 + np.exp(-prediction[:, 4])) > 0.5
    return _decode_xy(prediction, pred_valid)


def _target_xy(target: np.ndarray) -> np.ndarray:
    target_valid = target[:, 4] > 0.5
    return _decode_xy(target, target_valid)


def _goal_xy_from_base_observation(base_observation: np.ndarray) -> np.ndarray:
    if base_observation.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([[float(base_observation[0]) * 200.0, float(base_observation[1]) * 200.0]], dtype=np.float32)


def _ground_truth_history_xy(
    trajectories: dict[str, np.ndarray],
    *,
    row: int,
    current_idx: int,
    current_x: float,
    current_y: float,
    current_heading: float,
    history_horizon: int = _TRAJECTORY_HORIZON,
) -> np.ndarray:
    if current_idx < 0:
        return np.zeros((0, 2), dtype=np.float32)

    start_idx = max(0, current_idx - history_horizon + 1)
    world_x = trajectories["x"][row, 0, start_idx : current_idx + 1]
    world_y = trajectories["y"][row, 0, start_idx : current_idx + 1]
    valid = trajectories["valid"][row, 0, start_idx : current_idx + 1] > 0
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32)

    world_x = world_x[valid]
    world_y = world_y[valid]
    rel_x, rel_y = _transform_points_to_ego_frame(world_x, world_y, current_x, current_y, current_heading)
    return np.stack([rel_x, rel_y], axis=1).astype(np.float32)


def _transform_trace_to_current_ego(
    world_xy: np.ndarray,
    current_x: float,
    current_y: float,
    current_heading: float,
) -> np.ndarray:
    if world_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rel_x, rel_y = _transform_points_to_ego_frame(
        world_xy[:, 0], world_xy[:, 1], current_x, current_y, current_heading
    )
    return np.stack([rel_x, rel_y], axis=1).astype(np.float32)


def _transform_points_between_ego_frames(
    xy: np.ndarray,
    *,
    from_x: float,
    from_y: float,
    from_heading: float,
    to_x: float,
    to_y: float,
    to_heading: float,
) -> np.ndarray:
    if xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    cos_from = math.cos(from_heading)
    sin_from = math.sin(from_heading)
    world_x = from_x + (xy[:, 0] * cos_from - xy[:, 1] * sin_from)
    world_y = from_y + (xy[:, 0] * sin_from + xy[:, 1] * cos_from)
    rel_x, rel_y = _transform_points_to_ego_frame(world_x, world_y, to_x, to_y, to_heading)
    return np.stack([rel_x, rel_y], axis=1).astype(np.float32)


def _frame_limits(
    frame: RolloutFrame,
    *,
    min_half_span: float = 12.0,
    span_scale: float = 0.6,
) -> tuple[tuple[float, float], tuple[float, float]]:
    groups = [frame.rollout_xy, frame.ego_history_xy, frame.target_xy, frame.predicted_xy] + frame.partner_histories_xy
    valid_groups = [group for group in groups if group is not None and group.size > 0]
    if not valid_groups:
        return (-20.0, 20.0), (-20.0, 20.0)

    stacked = np.concatenate(valid_groups, axis=0)
    x_min = float(np.min(stacked[:, 0]))
    x_max = float(np.max(stacked[:, 0]))
    y_min = float(np.min(stacked[:, 1]))
    y_max = float(np.max(stacked[:, 1]))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half_span = max(min_half_span, span_scale * max(x_max - x_min, y_max - y_min, 1.0))
    return (x_center - half_span, x_center + half_span), (y_center - half_span, y_center + half_span)


def _anchored_dump_frame_limits(
    frame: AnchoredDumpFrame,
    *,
    min_half_span: float = 6.0,
    span_scale: float = 0.35,
) -> tuple[tuple[float, float], tuple[float, float]]:
    groups = [frame.rollout_xy_anchor, frame.target_xy_anchor, frame.predicted_xy_anchor]
    valid_groups = [group for group in groups if group is not None and group.size > 0]
    if not valid_groups:
        return (-20.0, 20.0), (-20.0, 20.0)
    stacked = np.concatenate(valid_groups + [frame.current_anchor_xy[None, :]], axis=0)
    x_min = float(np.min(stacked[:, 0]))
    x_max = float(np.max(stacked[:, 0]))
    y_min = float(np.min(stacked[:, 1]))
    y_max = float(np.max(stacked[:, 1]))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half_span = max(min_half_span, span_scale * max(x_max - x_min, y_max - y_min, 1.0))
    return (x_center - half_span, x_center + half_span), (y_center - half_span, y_center + half_span)


def _load_anchored_dump_frames(json_path: Path) -> list[AnchoredDumpFrame]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    frames: list[AnchoredDumpFrame] = []
    for item in payload:
        frames.append(
            AnchoredDumpFrame(
                map_name=str(item["map_name"]),
                timestep=int(item["timestep"]),
                time_s=float(item["time_s"]),
                current_world=list(item["current_world"]),
                current_anchor_xy=np.asarray(item["current_anchor_xy"], dtype=np.float32),
                ego_history_xy_anchor=np.asarray(item.get("ego_history_xy_anchor", []), dtype=np.float32),
                partner_histories_xy_anchor=[
                    np.asarray(partner_xy, dtype=np.float32)
                    for partner_xy in item.get("partner_histories_xy_anchor", [])
                ],
                goal_xy_anchor=np.asarray(item.get("goal_xy_anchor", []), dtype=np.float32),
                goal_radius=float(item.get("goal_radius", 0.0)),
                predicted_xy_anchor=np.asarray(item.get("predicted_xy_anchor", []), dtype=np.float32),
                target_xy_anchor=np.asarray(item.get("target_xy_anchor", []), dtype=np.float32),
                rollout_xy_anchor=np.asarray(item.get("rollout_xy_anchor", []), dtype=np.float32),
                control_action=np.asarray(item["control_action"], dtype=np.float32),
                reward=float(item.get("reward", 0.0)),
                collision_state=int(item.get("collision_state", 0)),
                offroad_flag=bool(item.get("offroad_flag", False)),
                reached_goal=bool(item.get("reached_goal", False)),
                stopped=bool(item.get("stopped", False)),
                speed=float(item.get("speed", 0.0)),
                stop_reason=str(item.get("stop_reason", "")),
                interrupted_after_step=bool(item.get("interrupted_after_step", False)),
                interruption_reason=str(item.get("interruption_reason", "")),
            )
        )
    return frames


def _to_anchored_dump_frame(frame: RolloutFrame, anchor: RolloutFrame, map_name: str) -> AnchoredDumpFrame:
    current_xy = _transform_points_between_ego_frames(
        np.zeros((1, 2), dtype=np.float32),
        from_x=frame.current_x,
        from_y=frame.current_y,
        from_heading=frame.current_heading,
        to_x=anchor.current_x,
        to_y=anchor.current_y,
        to_heading=anchor.current_heading,
    )[0]
    return AnchoredDumpFrame(
        map_name=map_name,
        timestep=frame.timestep,
        time_s=frame.timestep / 10.0,
        current_world=[frame.current_x, frame.current_y, frame.current_heading],
        current_anchor_xy=current_xy,
        ego_history_xy_anchor=_transform_points_between_ego_frames(
            frame.ego_history_xy,
            from_x=frame.current_x,
            from_y=frame.current_y,
            from_heading=frame.current_heading,
            to_x=anchor.current_x,
            to_y=anchor.current_y,
            to_heading=anchor.current_heading,
        ),
        partner_histories_xy_anchor=[
            _transform_points_between_ego_frames(
                partner_xy,
                from_x=frame.current_x,
                from_y=frame.current_y,
                from_heading=frame.current_heading,
                to_x=anchor.current_x,
                to_y=anchor.current_y,
                to_heading=anchor.current_heading,
            )
            for partner_xy in frame.partner_histories_xy
        ],
        goal_xy_anchor=_transform_points_between_ego_frames(
            frame.goal_xy,
            from_x=frame.current_x,
            from_y=frame.current_y,
            from_heading=frame.current_heading,
            to_x=anchor.current_x,
            to_y=anchor.current_y,
            to_heading=anchor.current_heading,
        ),
        goal_radius=float(frame.goal_radius),
        predicted_xy_anchor=_transform_points_between_ego_frames(
            frame.predicted_xy,
            from_x=frame.current_x,
            from_y=frame.current_y,
            from_heading=frame.current_heading,
            to_x=anchor.current_x,
            to_y=anchor.current_y,
            to_heading=anchor.current_heading,
        ),
        target_xy_anchor=_transform_points_between_ego_frames(
            frame.target_xy,
            from_x=frame.current_x,
            from_y=frame.current_y,
            from_heading=frame.current_heading,
            to_x=anchor.current_x,
            to_y=anchor.current_y,
            to_heading=anchor.current_heading,
        ),
        rollout_xy_anchor=_transform_points_between_ego_frames(
            frame.rollout_xy,
            from_x=frame.current_x,
            from_y=frame.current_y,
            from_heading=frame.current_heading,
            to_x=anchor.current_x,
            to_y=anchor.current_y,
            to_heading=anchor.current_heading,
        ),
        control_action=np.asarray(frame.control_action, dtype=np.float32),
        reward=float(frame.reward),
        collision_state=int(frame.collision_state),
        offroad_flag=bool(frame.offroad_flag),
        reached_goal=bool(frame.reached_goal),
        stopped=bool(frame.stopped),
        speed=float(frame.speed),
        stop_reason=str(frame.stop_reason),
        interrupted_after_step=bool(frame.interrupted_after_step),
        interruption_reason=str(frame.interruption_reason),
    )


def _render_anchored_grid(
    *,
    frames: list[AnchoredDumpFrame],
    output_path: Path,
    columns: int,
    road_segments: list[dict[str, np.ndarray | int]] | None = None,
    anchor_current_world: list[float] | None = None,
    show_diagnostics: bool = True,
) -> Path:
    plt, _FuncAnimation, _PillowWriter = _load_matplotlib()
    if not frames:
        raise ValueError("No frames to render")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = max(1, int(columns))
    rows = int(math.ceil(len(frames) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.0 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, columns)

    all_pts = []
    for frame in frames:
        for group in (
            frame.ego_history_xy_anchor,
            frame.goal_xy_anchor,
            frame.target_xy_anchor,
            frame.predicted_xy_anchor,
            frame.rollout_xy_anchor,
            *frame.partner_histories_xy_anchor,
        ):
            if group is not None and group.size > 0:
                all_pts.append(group)
        all_pts.append(frame.current_anchor_xy[None, :])
    stacked = np.concatenate(all_pts, axis=0)
    x_min = float(np.min(stacked[:, 0]))
    x_max = float(np.max(stacked[:, 0]))
    y_min = float(np.min(stacked[:, 1]))
    y_max = float(np.max(stacked[:, 1]))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half_span = max(6.0, 0.35 * max(x_max - x_min, y_max - y_min, 1.0))
    xlim = (x_center - half_span, x_center + half_span)
    ylim = (y_center - half_span, y_center + half_span)

    anchor_x = anchor_y = anchor_heading = 0.0
    if anchor_current_world is not None:
        anchor_x, anchor_y, anchor_heading = anchor_current_world

    for idx, ax in enumerate(axes.flat):
        if idx >= len(frames):
            ax.axis("off")
            continue
        frame = frames[idx]
        previous_frame = frames[idx - 1] if idx > 0 else None
        if road_segments:
            for road in road_segments:
                local_x, local_y = _transform_points_to_ego_frame(
                    road["x"], road["y"], anchor_x, anchor_y, anchor_heading
                )
                color = "#777777" if road["type"] == 4 else "#222222"
                linewidth = 1.0 if road["type"] == 4 else 1.5
                ax.plot(local_x, local_y, color=color, linewidth=linewidth, alpha=0.85, zorder=0)
        if frame.ego_history_xy_anchor.size:
            ax.plot(
                frame.ego_history_xy_anchor[:, 0],
                frame.ego_history_xy_anchor[:, 1],
                color="tab:blue",
                linewidth=2.0,
                marker="o",
                markersize=2.5,
                label="GT ego history",
            )
        if frame.goal_xy_anchor.size:
            ax.scatter(
                frame.goal_xy_anchor[:, 0],
                frame.goal_xy_anchor[:, 1],
                color="limegreen",
                s=50,
                marker="*",
                zorder=6,
                label="Goal" if idx == min(1, len(frames) - 1) else None,
            )
            goal_circle = Circle(
                (float(frame.goal_xy_anchor[0, 0]), float(frame.goal_xy_anchor[0, 1])),
                radius=float(frame.goal_radius),
                fill=False,
                edgecolor="limegreen",
                linewidth=1.2,
                alpha=0.75,
                zorder=5,
            )
            ax.add_patch(goal_circle)
        if frame.target_xy_anchor.size:
            ax.plot(
                frame.target_xy_anchor[:, 0],
                frame.target_xy_anchor[:, 1],
                color="tab:green",
                linewidth=2.0,
                marker="x",
                markersize=3.0,
                label="GT future",
            )
        if frame.predicted_xy_anchor.size:
            ax.plot(
                frame.predicted_xy_anchor[:, 0],
                frame.predicted_xy_anchor[:, 1],
                color="tab:purple",
                linewidth=2.0,
                marker="s",
                markersize=2.5,
                label="Predicted future",
            )
        if previous_frame is not None and previous_frame.predicted_xy_anchor.size:
            prev_one_step = previous_frame.predicted_xy_anchor[0]
            ax.plot(
                [previous_frame.current_anchor_xy[0], prev_one_step[0]],
                [previous_frame.current_anchor_xy[1], prev_one_step[1]],
                color="0.45",
                linewidth=1.6,
                linestyle="--",
                alpha=0.95,
                label="Prev predicted +0.1s" if idx == 1 else None,
            )
            ax.scatter(
                [prev_one_step[0]],
                [prev_one_step[1]],
                color="black",
                s=24,
                marker="X",
                zorder=6,
                label="Prev predicted pos" if idx == 1 else None,
            )
        if frame.rollout_xy_anchor.size:
            ax.plot(
                frame.rollout_xy_anchor[:, 0],
                frame.rollout_xy_anchor[:, 1],
                color="crimson",
                linewidth=2.1,
                label="Executed rollout",
            )
        for partner_idx, partner_xy in enumerate(frame.partner_histories_xy_anchor):
            if partner_xy.size:
                ax.plot(
                    partner_xy[:, 0],
                    partner_xy[:, 1],
                    color="tab:orange",
                    linewidth=1.3,
                    linestyle="--",
                    marker="o",
                    markersize=2.0,
                    alpha=0.8,
                    label="Partner history" if partner_idx == 0 else None,
                )
        ax.scatter(
            [frame.current_anchor_xy[0]],
            [frame.current_anchor_xy[1]],
            color="crimson",
            s=25,
            zorder=6,
            label="Current ego",
        )
        if frame.interrupted_after_step:
            ax.scatter(
                [frame.current_anchor_xy[0]],
                [frame.current_anchor_xy[1]],
                facecolors="none",
                edgecolors="red",
                s=120,
                linewidths=1.8,
                zorder=7,
                label="Interrupted after step" if idx == min(1, len(frames) - 1) else None,
            )
            ax.text(
                0.02,
                0.98,
                f"Interrupted ({frame.interruption_reason or 'unknown'})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="red",
                bbox={"facecolor": "white", "edgecolor": "red", "alpha": 0.85, "boxstyle": "round,pad=0.2"},
            )
        if show_diagnostics:
            flags = [
                f"v={frame.speed:.2f}",
                f"r={frame.reward:.2f}",
                f"coll={frame.collision_state}",
                f"offroad={int(frame.offroad_flag)}",
                f"stopped={int(frame.stopped)}",
                f"goal={int(frame.reached_goal)}",
                f"reason_stop={frame.stop_reason or '-'}",
            ]
            ax.text(
                0.02,
                0.02,
                "\n".join(flags),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
                color="0.2",
                bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.8, "boxstyle": "round,pad=0.2"},
            )
        ax.axhline(0.0, color="0.9", linewidth=0.8)
        ax.axvline(0.0, color="0.9", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"t = {frame.time_s:.1f}s", fontsize=9)
        if idx == min(1, len(frames) - 1):
            ax.legend(loc="best", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)

    suffix = " (json-backed)" if road_segments is None else ""
    fig.suptitle(
        f"{frames[0].map_name} | deterministic rollout | per-step grid in fixed t=0 frame{suffix}",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def collect_deterministic_rollout_frames(
    *,
    config_path: Path,
    checkpoint_path: Path,
    map_path: Path,
    device: str,
    road_source: str = "map",
    control_substeps: int = 1,
    start_seconds: float = 0.0,
) -> tuple[list[RolloutFrame], list[dict[str, np.ndarray | int]], str]:
    args, temp_root = build_args(config_path, checkpoint_path, map_path, device)
    vecenv = None
    try:
        vecenv = pufferl.load_env("puffer_drive", args)
        policy = pufferl.load_policy(args, vecenv, env_name="puffer_drive")
        policy.eval()
        driver = vecenv.driver_env

        obs, _ = vecenv.reset()
        num_agents = vecenv.observation_space.shape[0]
        state = {}
        if args["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        sim_steps = int(args["env"]["episode_length"] - args["env"]["init_steps"])
        if road_source == "map":
            roads = _parse_roads_with_fallback(map_path)
            road_segments = [
                {
                    "x": np.asarray(road.x if hasattr(road, "x") else road["x"], dtype=np.float32),
                    "y": np.asarray(road.y if hasattr(road, "y") else road["y"], dtype=np.float32),
                    "type": int(road.entity_type if hasattr(road, "entity_type") else road["entity_type"]),
                }
                for road in roads
            ]
        else:
            road_segments = []

        rollout_world_positions: list[list[float]] = []
        frames: list[RolloutFrame] = []
        scenario_id: int | None = None
        control_substeps = max(int(control_substeps), 1)
        control_sub_dt = float(driver.dt) / float(control_substeps)
        start_timestep = max(0, int(round(float(start_seconds) / max(float(driver.dt), 1e-6))))
        start_timestep = min(start_timestep, max(sim_steps - 1, 0))

        if start_timestep > 0:
            if driver.is_trajectory_observation or driver.is_trajectory_action:
                driver._clear_trajectory_history()
            for warmup_timestep in range(start_timestep + 1):
                obs, _, _, _, _ = driver.set_logged_timestep(
                    warmup_timestep,
                    reset_history=False,
                )

        current_logged_timestep = int(driver.tick)
        current_logged_timestep = min(max(current_logged_timestep, 0), max(sim_steps - 1, 0))

        for time_idx in range(current_logged_timestep, sim_steps):
            agent_state = driver.get_global_agent_state()
            diagnostics = driver.get_agent_diagnostics()
            current_x = float(agent_state["x"][0])
            current_y = float(agent_state["y"][0])
            current_heading = float(agent_state["heading"][0])
            rollout_world_positions.append([current_x, current_y])

            observation = np.asarray(obs[0], dtype=np.float32)
            base_dim = _base_obs_dim(driver.dynamics_model, driver.observation_mode_str)
            _ego_history_features, partner_history_features = _trajectory_history_feature_dims(
                driver.observation_mode_str
            )
            base_observation = observation[:base_dim]
            ego_hist_start = base_dim
            partner_hist_start = ego_hist_start + (_TRAJECTORY_HORIZON * _ego_history_features)
            partner_history = observation[partner_hist_start:].reshape(
                -1, _TRAJECTORY_HORIZON, partner_history_features
            )

            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, _value = policy.forward_eval(ob_tensor, state)
                action, _logprob, _entropy = eval_action_from_logits(logits, deterministic=True)
                action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)

            prediction = action_np[0].reshape(_TRAJECTORY_HORIZON, _TRAJECTORY_FEATURES)
            control_action = driver._trajectory_to_control_actions(prediction[None, ...])[0]
            target = driver.get_trajectory_targets(normalize=True)[0].reshape(_TRAJECTORY_HORIZON, _TRAJECTORY_FEATURES)
            gt = driver.get_ground_truth_trajectories()
            scenario_id = int(gt["scenario_id"][0, 0])
            gt_ego_history = _ground_truth_history_xy(
                gt,
                row=0,
                current_idx=int(driver.tick),
                current_x=current_x,
                current_y=current_y,
                current_heading=current_heading,
            )
            goal_xy = _goal_xy_from_base_observation(base_observation)

            observation_road_segments = (
                []
                if road_source == "map"
                else _decode_observation_road_segments(
                    base_observation,
                    dynamics_model=driver.dynamics_model,
                    observation_mode=driver.observation_mode_str,
                    treat_length_as_half_segment=False,
                )
            )
            frames.append(
                RolloutFrame(
                    timestep=int(driver.tick),
                    current_x=current_x,
                    current_y=current_y,
                    current_heading=current_heading,
                    ego_history_xy=gt_ego_history,
                    partner_histories_xy=[
                        partner_xy
                        for partner_xy in (_decode_history_slot(slot_block) for slot_block in partner_history)
                        if partner_xy.shape[0] > 0
                    ],
                    goal_xy=goal_xy,
                    goal_radius=float(args["env"]["goal_radius"]),
                    target_xy=_target_xy(target),
                    predicted_xy=_prediction_xy(prediction),
                    rollout_xy=_transform_trace_to_current_ego(
                        np.asarray(rollout_world_positions, dtype=np.float32),
                        current_x,
                        current_y,
                        current_heading,
                    ),
                    scenario_id=scenario_id,
                    observation_road_segments=observation_road_segments,
                    control_action=np.asarray(control_action, dtype=np.float32),
                    reward=float(driver.rewards[0]),
                    collision_state=int(diagnostics["collision_state"][0]),
                    offroad_flag=bool(diagnostics["offroad_flag"][0]),
                    reached_goal=bool(diagnostics["reached_goal"][0]),
                    stopped=bool(diagnostics["stopped"][0]),
                    speed=float(diagnostics["speed"][0]),
                )
            )

            interrupted_after_step = False
            interruption_reason = ""
            if control_substeps == 1:
                obs, _rewards, terminals, truncations, _ = vecenv.step(action_np)
                interrupted_after_step = bool(terminals[0] or truncations[0])
                if interrupted_after_step:
                    interruption_reason = "terminal" if bool(terminals[0]) else "truncation"
            else:
                for substep_idx in range(control_substeps):
                    if substep_idx > 0:
                        with torch.no_grad():
                            ob_tensor = torch.as_tensor(obs).to(device)
                            logits, _value = policy.forward_eval(ob_tensor, state)
                            action, _logprob, _entropy = eval_action_from_logits(logits, deterministic=True)
                            action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
                        if isinstance(logits, torch.distributions.Normal):
                            action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)

                    alpha = float(substep_idx + 1) / float(control_substeps)
                    obs, _rewards, terminals, truncations, _ = driver.physics_substep(
                        action_np,
                        sub_dt=control_sub_dt,
                        alpha=alpha,
                        update_history=False,
                    )
                    if bool(terminals[0] or truncations[0]):
                        interrupted_after_step = True
                        interruption_reason = "terminal" if bool(terminals[0]) else "truncation"
                        break

                if not interrupted_after_step:
                    obs, _rewards, terminals, truncations, _ = driver.advance_logged_timestep()
                    interrupted_after_step = bool(terminals[0] or truncations[0])
                    if interrupted_after_step:
                        interruption_reason = "terminal" if bool(terminals[0]) else "truncation"

            frames[-1].interrupted_after_step = interrupted_after_step
            frames[-1].interruption_reason = interruption_reason
            if frames[-1].stopped:
                if frames[-1].reached_goal:
                    frames[-1].stop_reason = "goal_stop"
                elif frames[-1].offroad_flag:
                    frames[-1].stop_reason = "offroad_stop"
                elif frames[-1].collision_state == 1:
                    frames[-1].stop_reason = "collision_stop"
                elif frames[-1].collision_state == 2:
                    frames[-1].stop_reason = "offroad_stop"
                else:
                    frames[-1].stop_reason = "stopped_unknown"
            if interrupted_after_step:
                break

        return frames, road_segments, map_path.name
    finally:
        if vecenv is not None:
            vecenv.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        gc.collect()


def render_deterministic_rollout_animation(
    *,
    config_path: Path,
    checkpoint_path: Path,
    map_path: Path,
    output_path: Path,
    device: str = "cpu",
    road_source: str = "map",
    control_substeps: int = 1,
    start_seconds: float = 0.0,
) -> Path:
    plt, FuncAnimation, PillowWriter = _load_matplotlib()
    frames, road_segments, map_name = collect_deterministic_rollout_frames(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        map_path=map_path,
        device=device,
        road_source=road_source,
        control_substeps=control_substeps,
        start_seconds=start_seconds,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    road_artists = []
    history_line, = ax.plot([], [], color="tab:blue", linewidth=2.2, marker="o", markersize=3, label="GT ego history")
    target_line, = ax.plot([], [], color="tab:green", linewidth=2.4, marker="x", markersize=4, label="GT future")
    pred_line, = ax.plot([], [], color="tab:purple", linewidth=2.4, marker="s", markersize=3, label="Predicted future")
    rollout_line, = ax.plot([], [], color="crimson", linewidth=2.4, label="Executed rollout")
    current_dot = ax.scatter([0.0], [0.0], color="crimson", s=45, zorder=6, label="Current ego")
    partner_lines = []

    def _draw_roads(frame: RolloutFrame):
        for artist in road_artists:
            artist.remove()
        road_artists.clear()

        if road_source == "map":
            for road in road_segments:
                local_x, local_y = _transform_points_to_ego_frame(
                    road["x"],
                    road["y"],
                    frame.current_x,
                    frame.current_y,
                    frame.current_heading,
                )
                color = "#777777" if road["type"] == 4 else "#222222"
                linewidth = 1.0 if road["type"] == 4 else 1.5
                artist, = ax.plot(local_x, local_y, color=color, linewidth=linewidth, alpha=0.85, zorder=0)
                road_artists.append(artist)
        else:
            for road in frame.observation_road_segments:
                color = "#777777" if road["type"] == 4 else "#222222"
                linewidth = 1.0 if road["type"] == 4 else 1.5
                artist, = ax.plot(road["x"], road["y"], color=color, linewidth=linewidth, alpha=0.85, zorder=0)
                road_artists.append(artist)

    def init():
        frame = frames[0]
        _draw_roads(frame)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x in current ego frame [m]")
        ax.set_ylabel("y in current ego frame [m]")
        ax.axhline(0.0, color="0.9", linewidth=0.8)
        ax.axvline(0.0, color="0.9", linewidth=0.8)
        ax.legend(loc="best")
        return [history_line, target_line, pred_line, rollout_line]

    def update(frame_idx: int):
        frame = frames[frame_idx]
        _draw_roads(frame)

        history_line.set_data(
            frame.ego_history_xy[:, 0] if frame.ego_history_xy.size else [],
            frame.ego_history_xy[:, 1] if frame.ego_history_xy.size else [],
        )
        target_line.set_data(
            frame.target_xy[:, 0] if frame.target_xy.size else [],
            frame.target_xy[:, 1] if frame.target_xy.size else [],
        )
        pred_line.set_data(
            frame.predicted_xy[:, 0] if frame.predicted_xy.size else [],
            frame.predicted_xy[:, 1] if frame.predicted_xy.size else [],
        )
        rollout_line.set_data(
            frame.rollout_xy[:, 0] if frame.rollout_xy.size else [],
            frame.rollout_xy[:, 1] if frame.rollout_xy.size else [],
        )

        while len(partner_lines) < len(frame.partner_histories_xy):
            partner_line, = ax.plot(
                [],
                [],
                color="tab:orange",
                linewidth=1.5,
                linestyle="--",
                marker="o",
                markersize=2.5,
                alpha=0.8,
                label="Partner history" if len(partner_lines) == 0 else None,
            )
            partner_lines.append(partner_line)
        for idx, partner_line in enumerate(partner_lines):
            if idx < len(frame.partner_histories_xy):
                partner_xy = frame.partner_histories_xy[idx]
                partner_line.set_data(partner_xy[:, 0], partner_xy[:, 1])
            else:
                partner_line.set_data([], [])

        xlim, ylim = _frame_limits(frame)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"{map_name} | deterministic rollout | t = {frame.timestep / 10:.1f}s")

        return [history_line, target_line, pred_line, rollout_line, *partner_lines, *road_artists]

    animation = FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=100, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    return output_path


def _draw_rollout_frame(
    ax,
    frame: RolloutFrame,
    *,
    road_segments: list[dict[str, np.ndarray | int]],
    map_name: str,
    road_source: str,
    show_legend: bool,
    anchor_frame: RolloutFrame | None = None,
):
    anchor = anchor_frame or frame

    def to_anchor(points: np.ndarray) -> np.ndarray:
        return _transform_points_between_ego_frames(
            points,
            from_x=frame.current_x,
            from_y=frame.current_y,
            from_heading=frame.current_heading,
            to_x=anchor.current_x,
            to_y=anchor.current_y,
            to_heading=anchor.current_heading,
        )

    if road_source == "map":
        for road in road_segments:
            local_x, local_y = _transform_points_to_ego_frame(
                road["x"],
                road["y"],
                anchor.current_x,
                anchor.current_y,
                anchor.current_heading,
            )
            color = "#777777" if road["type"] == 4 else "#222222"
            linewidth = 1.0 if road["type"] == 4 else 1.5
            ax.plot(local_x, local_y, color=color, linewidth=linewidth, alpha=0.85, zorder=0)
    else:
        for road in frame.observation_road_segments:
            road_xy = np.stack([road["x"], road["y"]], axis=1).astype(np.float32)
            road_xy = to_anchor(road_xy)
            color = "#777777" if road["type"] == 4 else "#222222"
            linewidth = 1.0 if road["type"] == 4 else 1.5
            ax.plot(road_xy[:, 0], road_xy[:, 1], color=color, linewidth=linewidth, alpha=0.85, zorder=0)

    ego_history_xy = to_anchor(frame.ego_history_xy)
    target_xy = to_anchor(frame.target_xy)
    predicted_xy = to_anchor(frame.predicted_xy)
    rollout_xy = to_anchor(frame.rollout_xy)
    partner_histories_xy = [to_anchor(partner_xy) for partner_xy in frame.partner_histories_xy]

    if ego_history_xy.size:
        ax.plot(
            ego_history_xy[:, 0],
            ego_history_xy[:, 1],
            color="tab:blue",
            linewidth=2.0,
            marker="o",
            markersize=2.5,
            label="GT ego history",
        )
    if target_xy.size:
        ax.plot(
            target_xy[:, 0],
            target_xy[:, 1],
            color="tab:green",
            linewidth=2.0,
            marker="x",
            markersize=3.0,
            label="GT future",
        )
    if predicted_xy.size:
        ax.plot(
            predicted_xy[:, 0],
            predicted_xy[:, 1],
            color="tab:purple",
            linewidth=2.0,
            marker="s",
            markersize=2.5,
            label="Predicted future",
        )
    if rollout_xy.size:
        ax.plot(
            rollout_xy[:, 0],
            rollout_xy[:, 1],
            color="crimson",
            linewidth=2.1,
            label="Executed rollout",
        )
    for idx, partner_xy in enumerate(partner_histories_xy):
        ax.plot(
            partner_xy[:, 0],
            partner_xy[:, 1],
            color="tab:orange",
            linewidth=1.3,
            linestyle="--",
            marker="o",
            markersize=2.0,
            alpha=0.8,
            label="Partner history" if idx == 0 else None,
        )

    current_xy = to_anchor(np.zeros((1, 2), dtype=np.float32))
    ax.scatter(current_xy[:, 0], current_xy[:, 1], color="crimson", s=25, zorder=6, label="Current ego")
    ax.axhline(0.0, color="0.9", linewidth=0.8)
    ax.axvline(0.0, color="0.9", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    anchored_frame = RolloutFrame(
        timestep=frame.timestep,
        current_x=anchor.current_x,
        current_y=anchor.current_y,
        current_heading=anchor.current_heading,
        ego_history_xy=ego_history_xy,
        partner_histories_xy=partner_histories_xy,
        goal_xy=to_anchor(frame.goal_xy),
        goal_radius=frame.goal_radius,
        target_xy=target_xy,
        predicted_xy=predicted_xy,
        rollout_xy=rollout_xy,
        scenario_id=frame.scenario_id,
        observation_road_segments=frame.observation_road_segments,
        control_action=frame.control_action,
        reward=frame.reward,
        collision_state=frame.collision_state,
        offroad_flag=frame.offroad_flag,
        reached_goal=frame.reached_goal,
        stopped=frame.stopped,
        speed=frame.speed,
        stop_reason=frame.stop_reason,
        interrupted_after_step=frame.interrupted_after_step,
        interruption_reason=frame.interruption_reason,
    )
    xlim, ylim = _frame_limits(anchored_frame, min_half_span=8.0, span_scale=0.45)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(f"t = {frame.timestep / 10:.1f}s", fontsize=9)
    if show_legend:
        ax.legend(loc="best", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)


def render_deterministic_rollout_grid(
    *,
    config_path: Path,
    checkpoint_path: Path,
    map_path: Path,
    output_path: Path,
    device: str = "cpu",
    road_source: str = "map",
    columns: int = 4,
    control_substeps: int = 1,
    start_seconds: float = 0.0,
    show_diagnostics: bool = True,
) -> Path:
    frames, road_segments, map_name = collect_deterministic_rollout_frames(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        map_path=map_path,
        device=device,
        road_source=road_source,
        control_substeps=control_substeps,
        start_seconds=start_seconds,
    )
    anchor_frame = frames[0]
    anchored_frames = [_to_anchored_dump_frame(frame, anchor_frame, map_name) for frame in frames]
    return _render_anchored_grid(
        frames=anchored_frames,
        output_path=output_path,
        columns=columns,
        road_segments=road_segments if road_source == "map" else None,
        anchor_current_world=[anchor_frame.current_x, anchor_frame.current_y, anchor_frame.current_heading],
        show_diagnostics=show_diagnostics,
    )


def render_deterministic_rollout_grid_from_json(
    *,
    frames_json_path: Path,
    output_path: Path,
    columns: int = 4,
    map_path: Path | None = None,
    show_diagnostics: bool = True,
) -> Path:
    frames = _load_anchored_dump_frames(frames_json_path)
    road_segments = None
    anchor_current_world = None
    if map_path is not None:
        roads = _parse_roads_with_fallback(map_path)
        road_segments = [
            {
                "x": np.asarray(road.x if hasattr(road, "x") else road["x"], dtype=np.float32),
                "y": np.asarray(road.y if hasattr(road, "y") else road["y"], dtype=np.float32),
                "type": int(road.entity_type if hasattr(road, "entity_type") else road["entity_type"]),
            }
            for road in roads
        ]
        anchor_current_world = frames[0].current_world
    return _render_anchored_grid(
        frames=frames,
        output_path=output_path,
        columns=columns,
        road_segments=road_segments,
        anchor_current_world=anchor_current_world,
        show_diagnostics=show_diagnostics,
    )


def dump_rollout_frames_json(
    *,
    config_path: Path,
    checkpoint_path: Path,
    map_path: Path,
    output_path: Path,
    device: str = "cpu",
    road_source: str = "map",
    control_substeps: int = 1,
    start_seconds: float = 0.0,
) -> Path:
    frames, _road_segments, map_name = collect_deterministic_rollout_frames(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        map_path=map_path,
        device=device,
        road_source=road_source,
        control_substeps=control_substeps,
        start_seconds=start_seconds,
    )
    anchor = frames[0]
    payload = []
    for frame in frames:
        anchored = _to_anchored_dump_frame(frame, anchor, map_name)
        payload.append(
            {
                "map_name": anchored.map_name,
                "timestep": anchored.timestep,
                "time_s": anchored.time_s,
                "current_world": anchored.current_world,
                "current_anchor_xy": anchored.current_anchor_xy.tolist(),
                "ego_history_xy_anchor": anchored.ego_history_xy_anchor.tolist(),
                "partner_histories_xy_anchor": [partner_xy.tolist() for partner_xy in anchored.partner_histories_xy_anchor],
                "goal_xy_anchor": anchored.goal_xy_anchor.tolist(),
                "goal_radius": anchored.goal_radius,
                "predicted_xy_anchor": anchored.predicted_xy_anchor.tolist(),
                "target_xy_anchor": anchored.target_xy_anchor.tolist(),
                "rollout_xy_anchor": anchored.rollout_xy_anchor.tolist(),
                "control_action": anchored.control_action.tolist(),
                "reward": anchored.reward,
                "collision_state": anchored.collision_state,
                "offroad_flag": anchored.offroad_flag,
                "reached_goal": anchored.reached_goal,
                "stopped": anchored.stopped,
                "speed": anchored.speed,
                "stop_reason": anchored.stop_reason,
                "interrupted_after_step": anchored.interrupted_after_step,
                "interruption_reason": anchored.interruption_reason,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a deterministic step-by-step simulator rollout using BC trajectory predictions."
    )
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("tests/test-viz/trajectory_bc_deterministic_rollout.gif"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--road-source", choices=("map", "observation"), default="map")
    parser.add_argument("--render-mode", choices=("gif", "grid", "json"), default="gif")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--control-substeps", type=int, default=1)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--dump-json", type=Path, default=None)
    parser.add_argument("--frames-json-path", type=Path, default=None)
    parser.add_argument("--hide-diagnostics", action="store_true")
    args = parser.parse_args()

    kwargs = dict(
        config_path=args.config_path.expanduser().resolve(),
        checkpoint_path=args.checkpoint_path.expanduser().resolve(),
        map_path=args.map_path.expanduser().resolve(),
        device=args.device,
        road_source=args.road_source,
        control_substeps=args.control_substeps,
        start_seconds=args.start_seconds,
    )
    if args.render_mode == "grid" and args.frames_json_path is not None:
        result = render_deterministic_rollout_grid_from_json(
            frames_json_path=args.frames_json_path.expanduser().resolve(),
            output_path=args.output_path.expanduser().resolve(),
            columns=args.columns,
            map_path=args.map_path.expanduser().resolve(),
            show_diagnostics=not args.hide_diagnostics,
        )
    elif args.render_mode == "grid":
        result = render_deterministic_rollout_grid(
            **kwargs,
            output_path=args.output_path.expanduser().resolve(),
            columns=args.columns,
            show_diagnostics=not args.hide_diagnostics,
        )
    elif args.render_mode == "json":
        output_path = args.dump_json or args.output_path
        result = dump_rollout_frames_json(**kwargs, output_path=output_path.expanduser().resolve())
    else:
        result = render_deterministic_rollout_animation(**kwargs, output_path=args.output_path.expanduser().resolve())
    print(result)
    if args.dump_json is not None and args.render_mode != "json":
        dump_path = dump_rollout_frames_json(**kwargs, output_path=args.dump_json.expanduser().resolve())
        print(dump_path)


if __name__ == "__main__":
    main()
