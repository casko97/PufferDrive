from __future__ import annotations

import argparse
import json
import logging
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pufferlib.ocean.drive.drive import binding
from scripts.export_paired_offline_fits import (
    OBS_MODE_DEFAULT,
    OBS_MODE_EXTENDED,
    _extract_ground_truth,
    _init_env,
    _normalize_action_source,
    _record_observation,
    _recover_reference_controls_from_trajectory,
)

LOGGER = logging.getLogger("export_real_control_bc_dataset")

DEFAULT_OUTPUT_DIR = Path("outputs/bc_datasets/real_control_bc")
DEFAULT_LOG_EVERY = 25
ACTION_SOURCE_TRAJECTORY = "trajectory"
EGO_DYNAMICS_OBS_KEY = "obs_default_plus_ego_dynamics"
VELOCITY_XY_OBS_KEY = "obs_default_vxy_speed25"
MOTION_PREV_CONTROL_OBS_KEY = "obs_default_motion_prev_control"
MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY = "obs_default_motion_prev_control_road_controls"
_LEGACY_CLASSIC_ACCEL_SCALE = 4.0
_STEERING_SCALE = 1.0
_YAW_RATE_SCALE = np.pi
_DEFAULT_SPEED_SCALE = 100.0
_VELOCITY_COMPONENT_SCALE = 25.0
_REL_POSITION_SCALE = 0.02
_ROAD_CONTROL_RADIUS_METERS = 30.0
_ROAD_TYPE_TO_NAME = {
    7: "stop_sign",
    8: "crosswalk",
    9: "speed_bump",
    10: "driveway",
}


@dataclass(frozen=True)
class SourceScenario:
    shard_name: str
    source_path: Path
    source_index: int
    scenario_id: str | None
    scenario_type: str | None
    original_map_name: str


def _stage_source_map(source_bin: Path) -> tempfile.TemporaryDirectory[str]:
    tmp_dir = tempfile.TemporaryDirectory(prefix=f"real_control_bc_{source_bin.stem}_")
    staged_map = Path(tmp_dir.name) / "map_000.bin"
    shutil.copy2(source_bin, staged_map)
    return tmp_dir


def _scenario_sort_key(item):
    idx, scenario = item
    split_index = scenario.get("split_index", idx)
    try:
        split_index = int(split_index)
    except (TypeError, ValueError):
        split_index = idx
    return split_index


def _scenario_source_path(source_dir: Path, scenario: dict) -> Path:
    map_name = scenario.get("map_name")
    if map_name:
        split_path = source_dir / str(map_name)
        if split_path.exists():
            return split_path
    for key in ("source_dataset_map_path", "source_map_path", "original_map_path"):
        value = scenario.get(key)
        if value:
            candidate = Path(str(value))
            if candidate.exists():
                return candidate
    if not map_name:
        raise KeyError("scenario is missing map_name/source_dataset_map_path")
    return source_dir / str(map_name)


def load_source_scenarios(source_dir: Path, max_maps: int | None = None) -> list[SourceScenario]:
    source_dir = Path(source_dir)
    manifest_path = source_dir / "selection_manifest.json"
    scenarios: list[SourceScenario] = []
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text())
        rows = list(payload.get("scenarios", []))
        for fallback_idx, scenario in sorted(enumerate(rows), key=_scenario_sort_key):
            split_index = scenario.get("split_index", fallback_idx)
            try:
                split_index = int(split_index)
            except (TypeError, ValueError):
                split_index = int(fallback_idx)
            source_path = _scenario_source_path(source_dir, scenario)
            shard_map_name = Path(str(scenario.get("map_name", f"map_{split_index:03d}.bin"))).name
            scenarios.append(
                SourceScenario(
                    shard_name=f"{Path(shard_map_name).stem}.pt",
                    source_path=source_path,
                    source_index=split_index,
                    scenario_id=scenario.get("scenario_id"),
                    scenario_type=scenario.get("scenario_type"),
                    original_map_name=source_path.name,
                )
            )
    else:
        for idx, source_path in enumerate(sorted(source_dir.glob("map_*.bin"))):
            scenarios.append(
                SourceScenario(
                    shard_name=f"{source_path.stem}.pt",
                    source_path=source_path,
                    source_index=idx,
                    scenario_id=None,
                    scenario_type=None,
                    original_map_name=source_path.name,
                )
            )
    if max_maps is not None and int(max_maps) > 0:
        scenarios = scenarios[: int(max_maps)]
    if not scenarios:
        raise FileNotFoundError(f"No source scenarios found under {source_dir}")
    return scenarios


def _wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    wrapped = (np.asarray(delta, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped.astype(np.float32, copy=False)


def _build_default_plus_ego_dynamics_obs(
    obs_default: np.ndarray,
    *,
    heading: np.ndarray,
    ref_accel: np.ndarray,
    ref_steer: np.ndarray,
    dt: float,
) -> np.ndarray:
    obs_default = np.asarray(obs_default, dtype=np.float32)
    heading = np.asarray(heading, dtype=np.float32)
    ref_accel = np.asarray(ref_accel, dtype=np.float32)
    ref_steer = np.asarray(ref_steer, dtype=np.float32)
    if obs_default.ndim != 2:
        raise ValueError(f"obs_default must be rank-2, got shape {obs_default.shape}")
    num_steps = int(obs_default.shape[0])
    if heading.shape[0] != num_steps or ref_accel.shape[0] != num_steps or ref_steer.shape[0] != num_steps:
        raise ValueError(
            "ego-dynamics feature inputs must match observation length: "
            f"obs={num_steps} heading={heading.shape[0]} accel={ref_accel.shape[0]} steer={ref_steer.shape[0]}"
        )

    yaw_rate = np.zeros((num_steps,), dtype=np.float32)
    if num_steps > 1:
        heading_delta = _wrap_angle_delta(heading[1:] - heading[:-1])
        yaw_rate[1:] = heading_delta / max(float(dt), 1e-6)
        yaw_rate[0] = yaw_rate[1]

    accel_feature = np.clip(ref_accel / _LEGACY_CLASSIC_ACCEL_SCALE, -1.0, 1.0)
    steer_feature = np.clip(ref_steer / _STEERING_SCALE, -1.0, 1.0)
    yaw_rate_feature = np.clip(yaw_rate / _YAW_RATE_SCALE, -1.0, 1.0)
    heading_cos = np.cos(heading).astype(np.float32, copy=False)
    heading_sin = np.sin(heading).astype(np.float32, copy=False)

    ego_dynamics = np.stack(
        [
            heading_cos,
            heading_sin,
            yaw_rate_feature.astype(np.float32, copy=False),
            steer_feature.astype(np.float32, copy=False),
            accel_feature.astype(np.float32, copy=False),
        ],
        axis=1,
    )
    return np.concatenate([obs_default, ego_dynamics], axis=1).astype(np.float32, copy=False)


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    num_steps = int(values.shape[0])
    if num_steps == 0:
        return np.zeros((0,), dtype=np.float32)
    if num_steps == 1:
        return np.zeros((1,), dtype=np.float32)
    dt_safe = max(float(dt), 1e-6)
    return np.gradient(values, dt_safe).astype(np.float32, copy=False)


def _build_default_velocity_xy_obs(
    obs_default: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    dt: float,
) -> np.ndarray:
    obs_default = np.asarray(obs_default, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    heading = np.asarray(heading, dtype=np.float32)
    if obs_default.ndim != 2:
        raise ValueError(f"obs_default must be rank-2, got shape {obs_default.shape}")
    num_steps = int(obs_default.shape[0])
    if x.shape[0] != num_steps or y.shape[0] != num_steps or heading.shape[0] != num_steps:
        raise ValueError(
            "velocity-xy feature inputs must match observation length: "
            f"obs={num_steps} x={x.shape[0]} y={y.shape[0]} heading={heading.shape[0]}"
        )

    ego_features = int(binding.EGO_FEATURES_CLASSIC)
    partner_features = int(binding.PARTNER_FEATURES)
    max_partner_objects = int(binding.MAX_AGENTS - 1)
    partner_block_width = partner_features * max_partner_objects
    road_start = ego_features + partner_block_width
    if obs_default.shape[1] < road_start:
        raise ValueError(
            f"obs_default width {obs_default.shape[1]} is too small for ego={ego_features} "
            f"partner_block={partner_block_width}"
        )

    cos_heading = np.cos(heading).astype(np.float32, copy=False)
    sin_heading = np.sin(heading).astype(np.float32, copy=False)
    vx_global = _finite_difference(x, dt)
    vy_global = _finite_difference(y, dt)
    ego_vx_local = cos_heading * vx_global + sin_heading * vy_global
    ego_vy_local = -sin_heading * vx_global + cos_heading * vy_global

    ego_vx_feature = np.clip(ego_vx_local / _VELOCITY_COMPONENT_SCALE, -1.0, 1.0)
    ego_vy_feature = np.clip(ego_vy_local / _VELOCITY_COMPONENT_SCALE, -1.0, 1.0)

    ego_block = np.stack(
        [
            obs_default[:, 0],
            obs_default[:, 1],
            ego_vx_feature.astype(np.float32, copy=False),
            ego_vy_feature.astype(np.float32, copy=False),
            obs_default[:, 3],
            obs_default[:, 4],
            obs_default[:, 5],
            obs_default[:, 6],
        ],
        axis=1,
    )

    partner_block = obs_default[:, ego_features:road_start].reshape(num_steps, max_partner_objects, partner_features)
    partner_signed_speed = partner_block[..., 6] * _DEFAULT_SPEED_SCALE
    partner_vx_local = np.clip((partner_signed_speed * partner_block[..., 4]) / _VELOCITY_COMPONENT_SCALE, -1.0, 1.0)
    partner_vy_local = np.clip((partner_signed_speed * partner_block[..., 5]) / _VELOCITY_COMPONENT_SCALE, -1.0, 1.0)
    partner_velocity_block = np.concatenate(
        [
            partner_block[..., :6],
            partner_vx_local[..., None].astype(np.float32, copy=False),
            partner_vy_local[..., None].astype(np.float32, copy=False),
        ],
        axis=2,
    )
    road_block = obs_default[:, road_start:]
    return np.concatenate(
        [
            ego_block.astype(np.float32, copy=False),
            partner_velocity_block.reshape(num_steps, -1).astype(np.float32, copy=False),
            road_block.astype(np.float32, copy=False),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _build_default_motion_prev_control_obs(
    obs_default: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    ref_accel: np.ndarray,
    ref_steer: np.ndarray,
    dt: float,
) -> np.ndarray:
    obs_default = np.asarray(obs_default, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    heading = np.asarray(heading, dtype=np.float32)
    ref_accel = np.asarray(ref_accel, dtype=np.float32)
    ref_steer = np.asarray(ref_steer, dtype=np.float32)
    if obs_default.ndim != 2:
        raise ValueError(f"obs_default must be rank-2, got shape {obs_default.shape}")
    num_steps = int(obs_default.shape[0])
    if (
        x.shape[0] != num_steps
        or y.shape[0] != num_steps
        or heading.shape[0] != num_steps
        or ref_accel.shape[0] != num_steps
        or ref_steer.shape[0] != num_steps
    ):
        raise ValueError(
            "motion-prev-control feature inputs must match observation length: "
            f"obs={num_steps} x={x.shape[0]} y={y.shape[0]} heading={heading.shape[0]} "
            f"accel={ref_accel.shape[0]} steer={ref_steer.shape[0]}"
        )

    vx_global = _finite_difference(x, dt)
    vy_global = _finite_difference(y, dt)
    cos_heading = np.cos(heading).astype(np.float32, copy=False)
    sin_heading = np.sin(heading).astype(np.float32, copy=False)
    ego_vx_local = cos_heading * vx_global + sin_heading * vy_global
    ego_vy_local = -sin_heading * vx_global + cos_heading * vy_global
    ego_vx_feature = np.clip(ego_vx_local / _VELOCITY_COMPONENT_SCALE, -1.0, 1.0)
    ego_vy_feature = np.clip(ego_vy_local / _VELOCITY_COMPONENT_SCALE, -1.0, 1.0)

    prev_steer = np.zeros((num_steps,), dtype=np.float32)
    prev_accel = np.zeros((num_steps,), dtype=np.float32)
    if num_steps > 1:
        prev_steer[1:] = ref_steer[:-1]
        prev_accel[1:] = ref_accel[:-1]
    prev_steer_feature = np.clip(prev_steer / _STEERING_SCALE, -1.0, 1.0)
    prev_accel_feature = np.clip(prev_accel / _LEGACY_CLASSIC_ACCEL_SCALE, -1.0, 1.0)

    motion_prev_control = np.stack(
        [
            cos_heading,
            sin_heading,
            ego_vx_feature.astype(np.float32, copy=False),
            ego_vy_feature.astype(np.float32, copy=False),
            prev_steer_feature.astype(np.float32, copy=False),
            prev_accel_feature.astype(np.float32, copy=False),
        ],
        axis=1,
    )
    return np.concatenate([obs_default, motion_prev_control], axis=1).astype(np.float32, copy=False)


def _load_special_road_control_points(source_path: Path) -> dict[str, np.ndarray]:
    source_path = Path(source_path)
    road_points = {name: [] for name in _ROAD_TYPE_TO_NAME.values()}
    with source_path.open("rb") as handle:
        _ = struct.unpack("i", handle.read(4))[0]  # sdc_track_index
        num_tracks_to_predict = struct.unpack("i", handle.read(4))[0]
        if num_tracks_to_predict > 0:
            handle.seek(4 * int(num_tracks_to_predict), 1)
        num_objects = struct.unpack("i", handle.read(4))[0]
        num_roads = struct.unpack("i", handle.read(4))[0]
        num_entities = int(num_objects) + int(num_roads)
        for _ in range(num_entities):
            _scenario_id, entity_type, _entity_id, array_size = struct.unpack("4i", handle.read(16))
            array_size = int(array_size)
            traj_x = np.fromfile(handle, dtype=np.float32, count=array_size)
            traj_y = np.fromfile(handle, dtype=np.float32, count=array_size)
            _traj_z = np.fromfile(handle, dtype=np.float32, count=array_size)
            if int(entity_type) in (1, 2, 3):
                np.fromfile(handle, dtype=np.float32, count=array_size)  # vx
                np.fromfile(handle, dtype=np.float32, count=array_size)  # vy
                np.fromfile(handle, dtype=np.float32, count=array_size)  # vz
                np.fromfile(handle, dtype=np.float32, count=array_size)  # heading
                np.fromfile(handle, dtype=np.int32, count=array_size)  # valid
            handle.seek(6 * 4 + 4, 1)  # width/length/height + goal xyz + mark_as_expert

            road_name = _ROAD_TYPE_TO_NAME.get(int(entity_type))
            if road_name is None or traj_x.size == 0 or traj_y.size == 0:
                continue
            road_points[road_name].append(np.stack([traj_x, traj_y], axis=1).astype(np.float32, copy=False))

    return {
        name: (
            np.concatenate(point_sets, axis=0).astype(np.float32, copy=False)
            if point_sets
            else np.zeros((0, 2), dtype=np.float32)
        )
        for name, point_sets in road_points.items()
    }


def _build_motion_prev_control_road_controls_obs(
    obs_motion_prev_control: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    road_points_by_type: dict[str, np.ndarray],
    radius_m: float = _ROAD_CONTROL_RADIUS_METERS,
) -> np.ndarray:
    obs_motion_prev_control = np.asarray(obs_motion_prev_control, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    heading = np.asarray(heading, dtype=np.float32)
    if obs_motion_prev_control.ndim != 2:
        raise ValueError(f"obs_motion_prev_control must be rank-2, got shape {obs_motion_prev_control.shape}")
    num_steps = int(obs_motion_prev_control.shape[0])
    if x.shape[0] != num_steps or y.shape[0] != num_steps or heading.shape[0] != num_steps:
        raise ValueError(
            "road-control feature inputs must match observation length: "
            f"obs={num_steps} x={x.shape[0]} y={y.shape[0]} heading={heading.shape[0]}"
        )

    cos_heading = np.cos(heading).astype(np.float32, copy=False)
    sin_heading = np.sin(heading).astype(np.float32, copy=False)
    radius_sq = float(radius_m) * float(radius_m)
    road_feature_rows = []
    for step_idx in range(num_steps):
        step_features = []
        ego_x = float(x[step_idx])
        ego_y = float(y[step_idx])
        for road_name in ("stop_sign", "crosswalk", "speed_bump", "driveway"):
            points = road_points_by_type.get(road_name)
            if points is None or int(points.shape[0]) == 0:
                step_features.extend((0.0, 0.0, 0.0))
                continue
            dx = points[:, 0] - ego_x
            dy = points[:, 1] - ego_y
            dist_sq = dx * dx + dy * dy
            best_idx = int(np.argmin(dist_sq))
            best_dist_sq = float(dist_sq[best_idx])
            if not np.isfinite(best_dist_sq) or best_dist_sq > radius_sq:
                step_features.extend((0.0, 0.0, 0.0))
                continue
            rel_x = dx[best_idx] * cos_heading[step_idx] + dy[best_idx] * sin_heading[step_idx]
            rel_y = -dx[best_idx] * sin_heading[step_idx] + dy[best_idx] * cos_heading[step_idx]
            step_features.extend(
                (
                    1.0,
                    float(np.clip(rel_x * _REL_POSITION_SCALE, -1.0, 1.0)),
                    float(np.clip(rel_y * _REL_POSITION_SCALE, -1.0, 1.0)),
                )
            )
        road_feature_rows.append(step_features)
    road_feature_array = np.asarray(road_feature_rows, dtype=np.float32)
    return np.concatenate([obs_motion_prev_control, road_feature_array], axis=1).astype(np.float32, copy=False)


def build_real_control_bc_payload(source_path: Path, *, action_source: str = ACTION_SOURCE_TRAJECTORY) -> dict[str, torch.Tensor]:
    action_source = _normalize_action_source(action_source)
    if action_source != ACTION_SOURCE_TRAJECTORY:
        raise ValueError(f"Unsupported real-control action source: {action_source!r}")

    staged = _stage_source_map(source_path)
    map_dir = Path(staged.name)
    try:
        env_handle, obs, actions, rewards, terminals, truncs = _init_env(map_dir)
        try:
            binding.env_reset(env_handle, 0)
            active_count = binding.env_get_active_agent_count(env_handle)
            if active_count <= 0:
                raise RuntimeError("no_active_agents")

            gt = _extract_ground_truth(env_handle, active_count)
            if gt["x"].shape[0] < 2:
                raise RuntimeError("trajectory_too_short")
            state_x = np.zeros(1, dtype=np.float32)
            state_y = np.zeros(1, dtype=np.float32)
            state_z = np.zeros(1, dtype=np.float32)
            state_heading = np.zeros(1, dtype=np.float32)
            state_id = np.zeros(1, dtype=np.int32)
            state_length = np.zeros(1, dtype=np.float32)
            state_width = np.zeros(1, dtype=np.float32)
            binding.get_global_agent_state(
                env_handle,
                state_x,
                state_y,
                state_z,
                state_heading,
                state_id,
                state_length,
                state_width,
            )
            geometry_length = float(state_length[0])

            recovered = _recover_reference_controls_from_trajectory(
                x=gt["x"],
                y=gt["y"],
                heading=gt["heading"],
                vehicle_length=geometry_length,
                dt=0.1,
            )
            road_points_by_type = _load_special_road_control_points(source_path)
            num_steps = int(recovered["actions"].shape[0])

            obs_rows_default = []
            obs_rows_extended = []
            for step_idx in range(num_steps):
                binding.env_set_logged_timestep(env_handle, step_idx)
                obs_record = _record_observation(env_handle, active_count)
                obs_rows_default.append(obs_record[OBS_MODE_DEFAULT])
                obs_rows_extended.append(obs_record[OBS_MODE_EXTENDED])
        finally:
            binding.env_close(env_handle)

        obs_array_default = np.stack(obs_rows_default).astype(np.float32)
        obs_tensor_default = torch.as_tensor(obs_array_default, dtype=torch.float32)
        obs_tensor_extended = torch.as_tensor(np.stack(obs_rows_extended).astype(np.float32), dtype=torch.float32)
        obs_tensor_default_plus_ego_dynamics = torch.as_tensor(
            _build_default_plus_ego_dynamics_obs(
                obs_array_default,
                heading=gt["heading"][:num_steps],
                ref_accel=recovered["ref_accel"][:num_steps],
                ref_steer=recovered["ref_steer"][:num_steps],
                dt=0.1,
            ),
            dtype=torch.float32,
        )
        obs_tensor_velocity_xy = torch.as_tensor(
            _build_default_velocity_xy_obs(
                obs_array_default,
                x=gt["x"][:num_steps],
                y=gt["y"][:num_steps],
                heading=gt["heading"][:num_steps],
                dt=0.1,
            ),
            dtype=torch.float32,
        )
        obs_tensor_motion_prev_control = torch.as_tensor(
            _build_default_motion_prev_control_obs(
                obs_array_default,
                x=gt["x"][:num_steps],
                y=gt["y"][:num_steps],
                heading=gt["heading"][:num_steps],
                ref_accel=recovered["ref_accel"][:num_steps],
                ref_steer=recovered["ref_steer"][:num_steps],
                dt=0.1,
            ),
            dtype=torch.float32,
        )
        obs_tensor_motion_prev_control_road_controls = torch.as_tensor(
            _build_motion_prev_control_road_controls_obs(
                obs_tensor_motion_prev_control.numpy(),
                x=gt["x"][:num_steps],
                y=gt["y"][:num_steps],
                heading=gt["heading"][:num_steps],
                road_points_by_type=road_points_by_type,
            ),
            dtype=torch.float32,
        )
        action_tensor = torch.as_tensor(recovered["actions"][:num_steps], dtype=torch.long)
        timestep = torch.arange(num_steps, dtype=torch.long)
        sequence_row_index = torch.arange(num_steps, dtype=torch.long)
        sequence_length = torch.full((num_steps,), num_steps, dtype=torch.long)
        return {
            "obs": obs_tensor_default,
            "obs_default": obs_tensor_default,
            EGO_DYNAMICS_OBS_KEY: obs_tensor_default_plus_ego_dynamics,
            VELOCITY_XY_OBS_KEY: obs_tensor_velocity_xy,
            MOTION_PREV_CONTROL_OBS_KEY: obs_tensor_motion_prev_control,
            MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY: obs_tensor_motion_prev_control_road_controls,
            "obs_sdc_only_with_trailer": obs_tensor_extended,
            "action": action_tensor,
            "timestep": timestep,
            "sequence_row_index": sequence_row_index,
            "sequence_length": sequence_length,
            "trajectory_ref_accel": torch.as_tensor(recovered["ref_accel"][:num_steps], dtype=torch.float32),
            "trajectory_ref_steer": torch.as_tensor(recovered["ref_steer"][:num_steps], dtype=torch.float32),
        }
    finally:
        staged.cleanup()


def export_real_control_bc_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    max_maps: int | None = None,
    log_every: int = DEFAULT_LOG_EVERY,
) -> Path:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    scenarios = load_source_scenarios(source_dir, max_maps=max_maps)
    manifest_rows = []
    total_steps = 0
    effective_log_every = max(1, int(log_every))

    for idx, scenario in enumerate(scenarios, start=1):
        payload = build_real_control_bc_payload(scenario.source_path, action_source=ACTION_SOURCE_TRAJECTORY)
        row_count = int(payload["action"].shape[0])
        shard_payload = {
            "obs": payload["obs"],
            "obs_default": payload["obs_default"],
            EGO_DYNAMICS_OBS_KEY: payload[EGO_DYNAMICS_OBS_KEY],
            VELOCITY_XY_OBS_KEY: payload[VELOCITY_XY_OBS_KEY],
            MOTION_PREV_CONTROL_OBS_KEY: payload[MOTION_PREV_CONTROL_OBS_KEY],
            MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY: payload[MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY],
            "obs_sdc_only_with_trailer": payload["obs_sdc_only_with_trailer"],
            "action": payload["action"],
            "map_id": torch.full((row_count,), int(scenario.source_index), dtype=torch.long),
            "timestep": payload["timestep"],
            "sequence_id": torch.full((row_count,), int(scenario.source_index), dtype=torch.long),
            "sequence_row_index": payload["sequence_row_index"],
            "sequence_length": payload["sequence_length"],
            "trajectory_ref_accel": payload["trajectory_ref_accel"],
            "trajectory_ref_steer": payload["trajectory_ref_steer"],
        }
        shard_path = output_dir / scenario.shard_name
        torch.save(shard_payload, shard_path)
        total_steps += row_count
        manifest_rows.append(
            {
                "shard_name": scenario.shard_name,
                "source_index": int(scenario.source_index),
                "source_map_path": str(scenario.source_path),
                "original_map_name": scenario.original_map_name,
                "scenario_id": scenario.scenario_id,
                "scenario_type": scenario.scenario_type,
                "num_steps": row_count,
            }
        )
        if idx % effective_log_every == 0 or idx == len(scenarios):
            LOGGER.info("Real-control BC export progress | maps=%d/%d rows=%d", idx, len(scenarios), total_steps)

    manifest = {
        "name": output_dir.name,
        "source_dir": str(source_dir),
        "count": len(manifest_rows),
        "row_count": int(total_steps),
        "obs_mode": OBS_MODE_DEFAULT,
        "observation_keys": [
            "obs",
            "obs_default",
            EGO_DYNAMICS_OBS_KEY,
            VELOCITY_XY_OBS_KEY,
            MOTION_PREV_CONTROL_OBS_KEY,
            MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY,
            "obs_sdc_only_with_trailer",
        ],
        "obs_default_dim": int(payload["obs_default"].shape[1]) if manifest_rows else 0,
        f"{EGO_DYNAMICS_OBS_KEY}_dim": int(payload[EGO_DYNAMICS_OBS_KEY].shape[1]) if manifest_rows else 0,
        f"{VELOCITY_XY_OBS_KEY}_dim": int(payload[VELOCITY_XY_OBS_KEY].shape[1]) if manifest_rows else 0,
        f"{MOTION_PREV_CONTROL_OBS_KEY}_dim": int(payload[MOTION_PREV_CONTROL_OBS_KEY].shape[1]) if manifest_rows else 0,
        f"{MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY}_dim": int(payload[MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY].shape[1]) if manifest_rows else 0,
        "obs_sdc_only_with_trailer_dim": int(payload["obs_sdc_only_with_trailer"].shape[1]) if manifest_rows else 0,
        "action_source": ACTION_SOURCE_TRAJECTORY,
        "maps": manifest_rows,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a BC shard dataset with observations and trajectory-derived real-control labels.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-maps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    output_dir = export_real_control_bc_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        max_maps=(args.max_maps if args.max_maps > 0 else None),
        log_every=args.log_every,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
