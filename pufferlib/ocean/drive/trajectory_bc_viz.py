from __future__ import annotations

from pathlib import Path

import math
import os
import struct
import numpy as np

from pufferlib.ocean.drive.trajectory_bc import (
    _PARTNER_POS_SCALE,
    _TRAJECTORY_FEATURES,
    _TRAJECTORY_HISTORY_FEATURES,
    _TRAJECTORY_HORIZON,
    _append_history,
    _base_obs_dim,
    _build_policy_base_observations,
    _build_trajectory_history_observations,
    _build_trajectory_targets,
    _decode_partner_states,
    _discover_map_paths,
    _get_active_agent_ids,
    _get_ego_trailer_obs_features,
    _get_global_agent_state,
    _get_global_agent_types,
    _get_ground_truth,
    _get_partner_ids,
    _get_partner_types,
    _get_sdc_trailer_state,
    _init_env_handle,
    _raw_base_obs_dim,
    _staged_map_dir,
    _trajectory_history_feature_dims,
    _trajectory_uses_augmented_base_observation,
    TrajectoryBCEnvConfig,
)
from pufferlib.ocean.drive.drive import binding
from pufferlib.ocean.drive.trailer_viz import parse_map_binary


def _transform_points_to_ego_frame(x_points, y_points, current_x: float, current_y: float, current_heading: float):
    cos_heading = np.cos(current_heading)
    sin_heading = np.sin(current_heading)
    dx = np.asarray(x_points, dtype=np.float32) - float(current_x)
    dy = np.asarray(y_points, dtype=np.float32) - float(current_y)
    rel_x = dx * cos_heading + dy * sin_heading
    rel_y = -dx * sin_heading + dy * cos_heading
    return rel_x, rel_y


def _decode_history_slot(slot_block: np.ndarray) -> np.ndarray:
    valid_mask = slot_block[:, 5] > 0.5
    if not np.any(valid_mask):
        return np.zeros((0, 2), dtype=np.float32)
    x = slot_block[valid_mask, 0] / _PARTNER_POS_SCALE
    y = slot_block[valid_mask, 1] / _PARTNER_POS_SCALE
    return np.stack([x, y], axis=1)


def _decode_trailer_history_slot(slot_block: np.ndarray) -> np.ndarray:
    if slot_block.shape[1] < 11:
        return np.zeros((0, 2), dtype=np.float32)
    trailer_heading = np.abs(slot_block[:, 9]) + np.abs(slot_block[:, 10])
    valid_mask = (slot_block[:, 5] > 0.5) & (trailer_heading > 1e-6)
    if not np.any(valid_mask):
        return np.zeros((0, 2), dtype=np.float32)
    x = slot_block[valid_mask, 7] / _PARTNER_POS_SCALE
    y = slot_block[valid_mask, 8] / _PARTNER_POS_SCALE
    return np.stack([x, y], axis=1)


def _decode_observation_road_segments(
    base_observation: np.ndarray,
    *,
    dynamics_model: str = "classic",
    observation_mode: str = "trajectory_history_32",
    treat_length_as_half_segment: bool = False,
) -> list[dict[str, np.ndarray | float | int]]:
    road_start = _base_obs_dim(dynamics_model, observation_mode) - (
        binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    )
    road_obs = base_observation[road_start : road_start + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES]
    road_obs = road_obs.reshape(binding.MAX_ROAD_SEGMENT_OBSERVATIONS, binding.ROAD_FEATURES)

    segments = []
    for slot in road_obs:
        if not np.any(np.abs(slot) > 1e-8):
            continue
        mid_x = float(slot[0]) / _PARTNER_POS_SCALE
        mid_y = float(slot[1]) / _PARTNER_POS_SCALE
        length = float(slot[2]) * 100.0
        cos_angle = float(slot[4])
        sin_angle = float(slot[5])
        road_type = int(round(float(slot[6]) + 4.0))
        half_length = length if treat_length_as_half_segment else 0.5 * length
        half_dx = half_length * cos_angle
        half_dy = half_length * sin_angle
        segments.append(
            {
                "x": np.asarray([mid_x - half_dx, mid_x + half_dx], dtype=np.float32),
                "y": np.asarray([mid_y - half_dy, mid_y + half_dy], dtype=np.float32),
                "type": road_type,
            }
        )
    return segments


def _set_zoomed_limits(ax, point_groups: list[np.ndarray], *, padding: float = 12.0, min_span: float = 20.0) -> None:
    valid_groups = [points for points in point_groups if points is not None and points.size > 0]
    if not valid_groups:
        return

    stacked = np.concatenate(valid_groups, axis=0)
    x_min = float(np.min(stacked[:, 0]))
    x_max = float(np.max(stacked[:, 0]))
    y_min = float(np.min(stacked[:, 1]))
    y_max = float(np.max(stacked[:, 1]))

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x_half = max(0.5 * (x_max - x_min) + padding, 0.5 * min_span)
    y_half = max(0.5 * (y_max - y_min) + padding, 0.5 * min_span)

    ax.set_xlim(x_center - x_half, x_center + x_half)
    ax.set_ylim(y_center - y_half, y_center + y_half)


def _parse_roads_with_fallback(map_path: str | Path):
    try:
        parsed = parse_map_binary(str(map_path))
        return parsed["roads"]
    except Exception:
        roads = []
        with open(map_path, "rb") as file_obj:
            _ = struct.unpack("<i", file_obj.read(4))[0]
            num_tracks_to_predict = struct.unpack("<i", file_obj.read(4))[0]
            file_obj.seek(4 * num_tracks_to_predict, os.SEEK_CUR)
            num_objects = struct.unpack("<i", file_obj.read(4))[0]
            num_roads = struct.unpack("<i", file_obj.read(4))[0]

            for _ in range(num_objects):
                _ = struct.unpack("<i", file_obj.read(4))[0]
                _ = struct.unpack("<i", file_obj.read(4))[0]
                _ = struct.unpack("<i", file_obj.read(4))[0]
                trajectory_length = struct.unpack("<i", file_obj.read(4))[0]
                file_obj.seek(4 * trajectory_length * 3, os.SEEK_CUR)
                file_obj.seek(4 * trajectory_length * 3, os.SEEK_CUR)
                file_obj.seek(4 * trajectory_length, os.SEEK_CUR)
                file_obj.seek(4 * trajectory_length, os.SEEK_CUR)
                file_obj.seek((6 * 4) + 4, os.SEEK_CUR)

            for _ in range(num_roads):
                _scenario_id = struct.unpack("<i", file_obj.read(4))[0]
                entity_type = struct.unpack("<i", file_obj.read(4))[0]
                _entity_id = struct.unpack("<i", file_obj.read(4))[0]
                array_size = struct.unpack("<i", file_obj.read(4))[0]
                x = np.fromfile(file_obj, dtype=np.float32, count=array_size)
                y = np.fromfile(file_obj, dtype=np.float32, count=array_size)
                file_obj.seek((4 * array_size) + (6 * 4) + 4, os.SEEK_CUR)
                roads.append({"entity_type": entity_type, "x": x, "y": y})
        return roads


def _seconds_to_steps(seconds: float, dt: float) -> int:
    return max(1, int(round(float(seconds) / max(float(dt), 1e-6))))


def _select_training_like_timesteps(
    trajectories: dict[str, np.ndarray],
    *,
    ego_row: int,
    dt: float,
    min_history_seconds: float,
    min_future_seconds: float,
    stride_seconds: float,
    start_offset_steps: int = 0,
    end_offset_steps: int = 0,
) -> list[int]:
    valid = trajectories["valid"][ego_row].astype(bool)
    num_steps = int(valid.shape[0])
    history_steps = _seconds_to_steps(min_history_seconds, dt)
    future_steps = _seconds_to_steps(min_future_seconds, dt)
    stride_steps = _seconds_to_steps(stride_seconds, dt)

    start_timestep = max(history_steps - 1, int(start_offset_steps))
    end_timestep = min(num_steps - future_steps - 1, num_steps - 2 - int(end_offset_steps))
    if end_timestep < start_timestep:
        return []

    selected = []
    for timestep in range(start_timestep, end_timestep + 1, stride_steps):
        history_ok = np.all(valid[timestep - (history_steps - 1) : timestep + 1])
        future_ok = np.all(valid[timestep + 1 : timestep + 1 + future_steps])
        if history_ok and future_ok:
            selected.append(timestep)
    return selected


def _resolve_visualization_timestep(
    trajectories: dict[str, np.ndarray],
    *,
    ego_row: int,
    requested_timestep: int | None,
) -> int:
    num_steps = int(trajectories["valid"].shape[1])
    min_history_timestep = _TRAJECTORY_HORIZON - 1
    max_future_timestep = num_steps - _TRAJECTORY_HORIZON - 1
    if max_future_timestep < 0:
        max_future_timestep = num_steps - 2

    if requested_timestep is not None:
        return max(0, min(int(requested_timestep), max(num_steps - 2, 0)))

    valid = trajectories["valid"][ego_row].astype(bool)
    candidate_timesteps = []
    for timestep in range(max(0, min_history_timestep), max(max_future_timestep, min_history_timestep) + 1):
        history_ok = np.all(valid[timestep - (_TRAJECTORY_HORIZON - 1) : timestep + 1])
        future_ok = np.all(valid[timestep + 1 : timestep + 1 + _TRAJECTORY_HORIZON])
        if history_ok and future_ok:
            candidate_timesteps.append(timestep)

    if candidate_timesteps:
        return int(candidate_timesteps[len(candidate_timesteps) // 2])

    fallback = (min_history_timestep + max(max_future_timestep, min_history_timestep)) // 2
    return max(0, min(int(fallback), max(num_steps - 2, 0)))


def collect_visualization_sample(map_path: str | Path, timestep: int | None, env_config: TrajectoryBCEnvConfig):
    from pufferlib.ocean.drive.drive import binding

    map_path = Path(map_path).expanduser().resolve()
    with _staged_map_dir(map_path) as staged_dir:
        env_handle = _init_env_handle(staged_dir, env_config)
        try:
            binding.env_reset(env_handle, 0)
            active_count = int(binding.env_get_active_agent_count(env_handle))
            if active_count <= 0:
                raise RuntimeError(f"No active agents in scenario: {map_path}")

            trajectories = _get_ground_truth(env_handle, active_count, env_config.episode_length, env_config.init_steps)
            timestep = _resolve_visualization_timestep(trajectories, ego_row=0, requested_timestep=timestep)
            ego_ids = _get_active_agent_ids(env_handle, active_count)
            history: dict[int, list[np.ndarray]] = {}

            for logged_timestep in range(timestep + 1):
                binding.env_set_logged_timestep(env_handle, logged_timestep)
                raw_base_observations = np.zeros(
                    (active_count, _raw_base_obs_dim(env_config.dynamics_model)),
                    dtype=np.float32,
                )
                binding.env_copy_observations(env_handle, raw_base_observations)
                current_states = _get_global_agent_state(env_handle, active_count)
                partner_ids = _get_partner_ids(env_handle, active_count)
                extended_observation = _trajectory_uses_augmented_base_observation(env_config.observation_mode)
                ego_types = _get_global_agent_types(env_handle, active_count) if extended_observation else None
                partner_types = _get_partner_types(env_handle, active_count) if extended_observation else None
                ego_trailer_features = (
                    _get_ego_trailer_obs_features(env_handle, active_count) if extended_observation else None
                )
                trailer_state = _get_sdc_trailer_state(env_handle) if extended_observation else None
                current_speeds = raw_base_observations[:, 2] * 100.0

                for row in range(active_count):
                    trailer_valid = 0.0
                    trailer_x = 0.0
                    trailer_y = 0.0
                    trailer_heading = 0.0
                    if (
                        extended_observation
                        and ego_trailer_features is not None
                        and trailer_state is not None
                        and int(trailer_state["has_trailer"][0]) > 0
                    ):
                        feature_values = (
                            float(ego_trailer_features["rel_x"][row]),
                            float(ego_trailer_features["rel_y"][row]),
                            float(ego_trailer_features["rel_heading_x"][row]),
                            float(ego_trailer_features["rel_heading_y"][row]),
                        )
                        if any(abs(value) > 1e-8 for value in feature_values):
                            trailer_valid = 1.0
                            trailer_x = float(trailer_state["x"][0])
                            trailer_y = float(trailer_state["y"][0])
                            trailer_heading = float(trailer_state["heading"][0])

                    _append_history(
                        history,
                        int(ego_ids[row]),
                        float(current_states["x"][row]),
                        float(current_states["y"][row]),
                        float(current_states["heading"][row]),
                        float(current_speeds[row]),
                        policy_type=int(ego_types[row]) if ego_types is not None else 0,
                        trailer_x=trailer_x,
                        trailer_y=trailer_y,
                        trailer_heading=trailer_heading,
                        trailer_valid=trailer_valid,
                    )

                for entity_id, state in _decode_partner_states(
                    raw_base_observations,
                    current_states,
                    partner_ids,
                    dynamics_model=env_config.dynamics_model,
                    partner_types=partner_types,
                ).items():
                    _append_history(history, entity_id, state[0], state[1], state[2], state[3], policy_type=state[5])

            base_observations = _build_policy_base_observations(
                raw_base_observations,
                env_handle,
                active_count,
                env_config,
                ego_types=ego_types,
                partner_types=partner_types,
                ego_trailer_features=ego_trailer_features,
            )
            observations = _build_trajectory_history_observations(
                base_observations=base_observations,
                ego_states=current_states,
                ego_ids=ego_ids,
                partner_ids=partner_ids,
                history=history,
                dynamics_model=env_config.dynamics_model,
                observation_mode=env_config.observation_mode,
            )
            targets = _build_trajectory_targets(
                trajectories=trajectories,
                current_states=current_states,
                current_timestep=timestep,
                dt=env_config.dt,
            )
            return observations, targets, current_states, partner_ids, ego_ids, timestep
        finally:
            binding.env_close(env_handle)


def render_trajectory_bc_sample_plot(
    *,
    map_path: str | Path,
    output_path: str | Path,
    timestep: int | None = None,
    ego_row: int = 0,
    env_config: TrajectoryBCEnvConfig | None = None,
    title: str | None = None,
    road_source: str = "map",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env_config = env_config or TrajectoryBCEnvConfig()
    map_path = Path(map_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    observations, targets, current_states, _partner_ids, _ego_ids, resolved_timestep = collect_visualization_sample(
        map_path=map_path,
        timestep=timestep,
        env_config=env_config,
    )
    if road_source not in {"map", "observation", "observation_half_length"}:
        raise ValueError(
            f"road_source must be 'map', 'observation', or 'observation_half_length', got: {road_source}"
        )
    roads = _parse_roads_with_fallback(map_path)

    current_x = float(current_states["x"][ego_row])
    current_y = float(current_states["y"][ego_row])
    current_heading = float(current_states["heading"][ego_row])
    base_dim = _base_obs_dim(env_config.dynamics_model, env_config.observation_mode)
    ego_history_features, partner_history_features = _trajectory_history_feature_dims(env_config.observation_mode)
    base_observation = observations[ego_row, :base_dim]

    ego_hist_start = base_dim
    partner_hist_start = ego_hist_start + (_TRAJECTORY_HORIZON * ego_history_features)
    ego_history = observations[ego_row, ego_hist_start:partner_hist_start].reshape(
        _TRAJECTORY_HORIZON,
        ego_history_features,
    )
    partner_history = observations[ego_row, partner_hist_start:].reshape(
        -1, _TRAJECTORY_HORIZON, partner_history_features
    )
    target = targets[ego_row].reshape(_TRAJECTORY_HORIZON, _TRAJECTORY_FEATURES)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    if road_source == "map":
        road_segments = []
        for road in roads:
            road_x = road.x if hasattr(road, "x") else road["x"]
            road_y = road.y if hasattr(road, "y") else road["y"]
            road_type = road.entity_type if hasattr(road, "entity_type") else road["entity_type"]
            local_x, local_y = _transform_points_to_ego_frame(road_x, road_y, current_x, current_y, current_heading)
            road_segments.append({"x": local_x, "y": local_y, "type": road_type})
    elif road_source == "observation":
        road_segments = _decode_observation_road_segments(
            base_observation,
            dynamics_model=env_config.dynamics_model,
            observation_mode=env_config.observation_mode,
            treat_length_as_half_segment=False,
        )
    else:
        road_segments = _decode_observation_road_segments(
            base_observation,
            dynamics_model=env_config.dynamics_model,
            observation_mode=env_config.observation_mode,
            treat_length_as_half_segment=True,
        )

    for road in road_segments:
        road_x = road["x"]
        road_y = road["y"]
        road_type = road["type"]
        color = "#777777" if road_type == 4 else "#222222"
        linewidth = 1.0 if road_type == 4 else 1.6
        ax.plot(road_x, road_y, color=color, linewidth=linewidth, alpha=0.85, zorder=0)

    ego_xy = _decode_history_slot(ego_history)
    if ego_xy.shape[0] > 0:
        ax.plot(
            ego_xy[:, 0],
            ego_xy[:, 1],
            color="tab:blue",
            linewidth=2.4,
            marker="o",
            markersize=3,
            label="Ego history",
        )

    trailer_xy = _decode_trailer_history_slot(ego_history)
    if trailer_xy.shape[0] > 0:
        ax.plot(
            trailer_xy[:, 0],
            trailer_xy[:, 1],
            color="tab:red",
            linewidth=2.0,
            linestyle="-.",
            marker="s",
            markersize=3,
            label="Trailer history",
        )

    plotted_partner = False
    for slot_block in partner_history:
        partner_xy = _decode_history_slot(slot_block)
        if partner_xy.shape[0] == 0:
            continue
        ax.plot(
            partner_xy[:, 0],
            partner_xy[:, 1],
            color="tab:orange",
            linewidth=1.8,
            linestyle="--",
            marker="o",
            markersize=2.5,
            alpha=0.8,
            label=None if plotted_partner else "Partner history",
        )
        plotted_partner = True

    target_valid = target[:, 4] > 0.5
    target_xy = np.zeros((0, 2), dtype=np.float32)
    if np.any(target_valid):
        target_xy = np.stack(
            [target[target_valid, 0] / _PARTNER_POS_SCALE, target[target_valid, 1] / _PARTNER_POS_SCALE],
            axis=1,
        )
        ax.plot(
            target_xy[:, 0],
            target_xy[:, 1],
            color="tab:green",
            linewidth=2.6,
            marker="x",
            markersize=4,
            label="Ego future target",
        )

    ax.scatter([0.0], [0.0], color="crimson", s=50, zorder=5, label="Current ego")
    _set_zoomed_limits(
        ax,
        [ego_xy, trailer_xy, target_xy] + [_decode_history_slot(slot_block) for slot_block in partner_history],
    )
    road_title = {
        "map": "map-road",
        "observation": "obs-road",
        "observation_half_length": "obs-road-half-len",
    }[road_source]
    ax.set_title(title or f"Trajectory BC sample sanity view ({map_path.name}, t={resolved_timestep}, {road_title})")
    ax.set_xlabel("x in current ego frame [m]")
    ax.set_ylabel("y in current ego frame [m]")
    ax.axhline(0.0, color="0.9", linewidth=0.8)
    ax.axvline(0.0, color="0.9", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def render_trajectory_bc_single_bin_grid(
    *,
    map_path: str | Path,
    output_path: str | Path,
    min_history_seconds: float = 1.0,
    min_future_seconds: float = 1.0,
    stride_seconds: float = 3.2,
    env_config: TrajectoryBCEnvConfig | None = None,
    ego_row: int = 0,
    road_source: str = "map",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env_config = env_config or TrajectoryBCEnvConfig()
    map_path = Path(map_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from pufferlib.ocean.drive.drive import binding

    if road_source not in {"map", "observation", "observation_half_length"}:
        raise ValueError(
            f"road_source must be 'map', 'observation', or 'observation_half_length', got: {road_source}"
        )
    roads = _parse_roads_with_fallback(map_path)
    samples = []
    with _staged_map_dir(map_path) as staged_dir:
        env_handle = _init_env_handle(staged_dir, env_config)
        try:
            binding.env_reset(env_handle, 0)
            active_count = int(binding.env_get_active_agent_count(env_handle))
            if active_count <= 0:
                raise RuntimeError(f"No active agents in scenario: {map_path}")

            trajectories = _get_ground_truth(env_handle, active_count, env_config.episode_length, env_config.init_steps)
            ego_ids = _get_active_agent_ids(env_handle, active_count)
            timesteps = _select_training_like_timesteps(
                trajectories,
                ego_row=ego_row,
                dt=env_config.dt,
                min_history_seconds=min_history_seconds,
                min_future_seconds=min_future_seconds,
                stride_seconds=stride_seconds,
            )
            history: dict[int, list[np.ndarray]] = {}

            if not timesteps:
                raise RuntimeError(
                    f"No timesteps satisfy history>={min_history_seconds}s, future>={min_future_seconds}s, "
                    f"stride={stride_seconds}s for {map_path.name}"
                )

            selected_set = set(timesteps)
            for logged_timestep in range(max(timesteps) + 1):
                binding.env_set_logged_timestep(env_handle, logged_timestep)
                raw_base_observations = np.zeros(
                    (active_count, _raw_base_obs_dim(env_config.dynamics_model)),
                    dtype=np.float32,
                )
                binding.env_copy_observations(env_handle, raw_base_observations)
                current_states = _get_global_agent_state(env_handle, active_count)
                partner_ids = _get_partner_ids(env_handle, active_count)
                extended_observation = _trajectory_uses_augmented_base_observation(env_config.observation_mode)
                ego_types = _get_global_agent_types(env_handle, active_count) if extended_observation else None
                partner_types = _get_partner_types(env_handle, active_count) if extended_observation else None
                ego_trailer_features = (
                    _get_ego_trailer_obs_features(env_handle, active_count) if extended_observation else None
                )
                trailer_state = _get_sdc_trailer_state(env_handle) if extended_observation else None
                current_speeds = raw_base_observations[:, 2] * 100.0

                for row in range(active_count):
                    trailer_valid = 0.0
                    trailer_x = 0.0
                    trailer_y = 0.0
                    trailer_heading = 0.0
                    if (
                        extended_observation
                        and ego_trailer_features is not None
                        and trailer_state is not None
                        and int(trailer_state["has_trailer"][0]) > 0
                    ):
                        feature_values = (
                            float(ego_trailer_features["rel_x"][row]),
                            float(ego_trailer_features["rel_y"][row]),
                            float(ego_trailer_features["rel_heading_x"][row]),
                            float(ego_trailer_features["rel_heading_y"][row]),
                        )
                        if any(abs(value) > 1e-8 for value in feature_values):
                            trailer_valid = 1.0
                            trailer_x = float(trailer_state["x"][0])
                            trailer_y = float(trailer_state["y"][0])
                            trailer_heading = float(trailer_state["heading"][0])

                    _append_history(
                        history,
                        int(ego_ids[row]),
                        float(current_states["x"][row]),
                        float(current_states["y"][row]),
                        float(current_states["heading"][row]),
                        float(current_speeds[row]),
                        policy_type=int(ego_types[row]) if ego_types is not None else 0,
                        trailer_x=trailer_x,
                        trailer_y=trailer_y,
                        trailer_heading=trailer_heading,
                        trailer_valid=trailer_valid,
                    )

                for entity_id, state in _decode_partner_states(
                    raw_base_observations,
                    current_states,
                    partner_ids,
                    dynamics_model=env_config.dynamics_model,
                    partner_types=partner_types,
                ).items():
                    _append_history(history, entity_id, state[0], state[1], state[2], state[3], policy_type=state[5])

                if logged_timestep not in selected_set:
                    continue

                base_observations = _build_policy_base_observations(
                    raw_base_observations,
                    env_handle,
                    active_count,
                    env_config,
                    ego_types=ego_types,
                    partner_types=partner_types,
                    ego_trailer_features=ego_trailer_features,
                )
                observations = _build_trajectory_history_observations(
                    base_observations=base_observations,
                    ego_states=current_states,
                    ego_ids=ego_ids,
                    partner_ids=partner_ids,
                    history=history,
                    dynamics_model=env_config.dynamics_model,
                    observation_mode=env_config.observation_mode,
                )
                targets = _build_trajectory_targets(
                    trajectories=trajectories,
                    current_states=current_states,
                    current_timestep=logged_timestep,
                    dt=env_config.dt,
                )
                samples.append((logged_timestep, observations.copy(), targets.copy(), current_states.copy()))
        finally:
            binding.env_close(env_handle)

    num_samples = len(samples)
    columns = min(3, num_samples)
    rows = int(math.ceil(num_samples / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for plot_idx, (timestep, observations, targets, current_states) in enumerate(samples):
        row_idx = plot_idx // columns
        col_idx = plot_idx % columns
        ax = axes[row_idx, col_idx]

        current_x = float(current_states["x"][ego_row])
        current_y = float(current_states["y"][ego_row])
        current_heading = float(current_states["heading"][ego_row])
        base_dim = _base_obs_dim(env_config.dynamics_model, env_config.observation_mode)
        ego_history_features, partner_history_features = _trajectory_history_feature_dims(env_config.observation_mode)
        base_observation = observations[ego_row, :base_dim]

        ego_hist_start = base_dim
        partner_hist_start = ego_hist_start + (_TRAJECTORY_HORIZON * ego_history_features)
        ego_history = observations[ego_row, ego_hist_start:partner_hist_start].reshape(
            _TRAJECTORY_HORIZON,
            ego_history_features,
        )
        partner_history = observations[ego_row, partner_hist_start:].reshape(
            -1, _TRAJECTORY_HORIZON, partner_history_features
        )
        target = targets[ego_row].reshape(_TRAJECTORY_HORIZON, _TRAJECTORY_FEATURES)

        if road_source == "map":
            road_segments = []
            for road in roads:
                road_x = road.x if hasattr(road, "x") else road["x"]
                road_y = road.y if hasattr(road, "y") else road["y"]
                road_type = road.entity_type if hasattr(road, "entity_type") else road["entity_type"]
                local_x, local_y = _transform_points_to_ego_frame(road_x, road_y, current_x, current_y, current_heading)
                road_segments.append({"x": local_x, "y": local_y, "type": road_type})
        elif road_source == "observation":
            road_segments = _decode_observation_road_segments(
                base_observation,
                dynamics_model=env_config.dynamics_model,
                observation_mode=env_config.observation_mode,
                treat_length_as_half_segment=False,
            )
        else:
            road_segments = _decode_observation_road_segments(
                base_observation,
                dynamics_model=env_config.dynamics_model,
                observation_mode=env_config.observation_mode,
                treat_length_as_half_segment=True,
            )

        for road in road_segments:
            road_x = road["x"]
            road_y = road["y"]
            road_type = road["type"]
            color = "#777777" if road_type == 4 else "#222222"
            linewidth = 1.0 if road_type == 4 else 1.6
            ax.plot(road_x, road_y, color=color, linewidth=linewidth, alpha=0.85, zorder=0)

        ego_xy = _decode_history_slot(ego_history)
        if ego_xy.shape[0] > 0:
            ax.plot(ego_xy[:, 0], ego_xy[:, 1], color="tab:blue", linewidth=2.0, marker="o", markersize=2.5)

        trailer_xy = _decode_trailer_history_slot(ego_history)
        if trailer_xy.shape[0] > 0:
            ax.plot(
                trailer_xy[:, 0],
                trailer_xy[:, 1],
                color="tab:red",
                linewidth=1.7,
                linestyle="-.",
                marker="s",
                markersize=2.2,
            )

        partner_xy_groups = []
        for slot_block in partner_history:
            partner_xy = _decode_history_slot(slot_block)
            if partner_xy.shape[0] == 0:
                continue
            partner_xy_groups.append(partner_xy)
            ax.plot(
                partner_xy[:, 0],
                partner_xy[:, 1],
                color="tab:orange",
                linewidth=1.5,
                linestyle="--",
                marker="o",
                markersize=2,
                alpha=0.8,
            )

        target_valid = target[:, 4] > 0.5
        target_xy = np.zeros((0, 2), dtype=np.float32)
        if np.any(target_valid):
            target_xy = np.stack(
                [target[target_valid, 0] / _PARTNER_POS_SCALE, target[target_valid, 1] / _PARTNER_POS_SCALE],
                axis=1,
            )
            ax.plot(target_xy[:, 0], target_xy[:, 1], color="tab:green", linewidth=2.1, marker="x", markersize=3)

        ax.scatter([0.0], [0.0], color="crimson", s=35, zorder=5)
        _set_zoomed_limits(ax, [ego_xy, trailer_xy, target_xy] + partner_xy_groups)
        ax.set_title(f"t={timestep}")
        ax.axhline(0.0, color="0.92", linewidth=0.8)
        ax.axvline(0.0, color="0.92", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    for empty_idx in range(num_samples, rows * columns):
        row_idx = empty_idx // columns
        col_idx = empty_idx % columns
        axes[row_idx, col_idx].axis("off")

    fig.suptitle(
        f"Trajectory BC extracted samples ({map_path.name})\n"
        f"min_history={min_history_seconds:.1f}s, min_future={min_future_seconds:.1f}s, stride={stride_seconds:.1f}s, roads={road_source}"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def render_trajectory_bc_dataset_plots(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    num_scenarios: int = 4,
    timestep: int | None = None,
    start_index: int = 0,
    env_config: TrajectoryBCEnvConfig | None = None,
) -> list[Path]:
    env_config = env_config or TrajectoryBCEnvConfig()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    map_paths = _discover_map_paths(dataset_dir)
    selected = map_paths[int(start_index) : int(start_index) + max(1, int(num_scenarios))]

    outputs = []
    for offset, map_path in enumerate(selected):
        output_path = output_dir / f"{start_index + offset:04d}_{map_path.stem}_t{timestep}.png"
        outputs.append(
            render_trajectory_bc_sample_plot(
                map_path=map_path,
                output_path=output_path,
                timestep=timestep,
                env_config=env_config,
            )
        )
    return outputs
