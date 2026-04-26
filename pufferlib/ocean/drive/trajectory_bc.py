from __future__ import annotations

import ast
import configparser
import json
import math
import os
import random
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import gymnasium
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from pufferlib.ocean.drive.drive import (
    _DYNAMICS_MODEL_IDS,
    _EMPTY_PARTNER_EPS,
    _MAX_SPEED,
    _PARTNER_POS_SCALE,
    _TRAJECTORY_ACTION_DIM,
    _TRAJECTORY_FEATURES,
    _TRAJECTORY_HISTORY_FEATURES,
    _TRAJECTORY_HORIZON,
    _TRAJECTORY_OBSERVATION_MODE,
    _clip_float,
    _encode_trajectory_heading,
    _encode_trajectory_position,
    _encode_trajectory_speed,
    _make_history_state,
    _wrap_to_pi,
    binding,
)
from pufferlib.ocean.drive.trajectory_supervision import masked_trajectory_loss, trajectory_metrics
from pufferlib.ocean.torch import Drive as DrivePolicy


def _parse_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _load_ini(path: str | Path) -> dict[str, dict[str, Any]]:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    with Path(path).open("r", encoding="utf-8") as file_obj:
        parser.read_file(file_obj)

    config: dict[str, dict[str, Any]] = {}
    for section in parser.sections():
        config[section] = {}
        for key, value in parser[section].items():
            config[section][key] = _parse_value(value)
    return config


def _discover_map_paths(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    manifest_path = root / "dataset_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = payload.get("maps") or payload.get("entries") or []
        paths = []
        for row in rows:
            relative_path = row.get("relative_path") or row.get("path") or row.get("map_name")
            if not relative_path:
                continue
            map_path = (root / relative_path).resolve()
            if map_path.exists():
                paths.append(map_path)
        if paths:
            return sorted(paths)

    top_level = sorted(root.glob("*.bin"))
    if top_level:
        return top_level
    recursive = sorted(root.rglob("*.bin"))
    if recursive:
        return recursive
    raise FileNotFoundError(f"No .bin scenarios found under {root}")


def _split_map_paths(
    map_paths: list[Path],
    *,
    val_fraction: float,
    seed: int,
    max_maps: int,
) -> tuple[list[Path], list[Path]]:
    ordered = list(map_paths)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if max_maps > 0:
        ordered = ordered[:max_maps]

    if not ordered:
        raise ValueError("No maps available after applying max_maps")

    if val_fraction <= 0.0 or len(ordered) == 1:
        return ordered, []

    num_val = max(1, int(round(len(ordered) * float(val_fraction))))
    num_val = min(num_val, len(ordered) - 1)
    return ordered[num_val:], ordered[:num_val]


def _select_map_subset(
    map_paths: list[Path],
    *,
    seed: int,
    max_maps: int,
) -> list[Path]:
    ordered = list(map_paths)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if max_maps > 0:
        ordered = ordered[:max_maps]
    if not ordered:
        raise ValueError("No maps available after applying max_maps")
    return ordered


def _mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _resolve_training_device(requested: str | None) -> torch.device:
    request = "auto" if requested is None else str(requested).strip().lower()

    if request in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if request.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        if _mps_is_available():
            print(f"[trajectory-bc] requested device '{requested}' unavailable; falling back to 'mps'")
            return torch.device("mps")
        print(f"[trajectory-bc] requested device '{requested}' unavailable; falling back to 'cpu'")
        return torch.device("cpu")

    if request == "mps":
        if _mps_is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            print("[trajectory-bc] requested device 'mps' unavailable; falling back to 'cuda'")
            return torch.device("cuda")
        print("[trajectory-bc] requested device 'mps' unavailable; falling back to 'cpu'")
        return torch.device("cpu")

    if request == "cpu":
        return torch.device("cpu")

    return torch.device(requested)


def _control_mode_id(value: str) -> int:
    mapping = {
        "control_vehicles": 0,
        "control_agents": 1,
        "control_wosac": 2,
        "control_sdc_only": 3,
    }
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported control_mode: {value}")
    return mapping[key]


def _init_mode_id(value: str) -> int:
    mapping = {
        "create_all_valid": 0,
        "create_only_controlled": 1,
    }
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported init_mode: {value}")
    return mapping[key]


def _base_obs_dim() -> int:
    return (
        binding.EGO_FEATURES_CLASSIC
        + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
        + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    )


def _trajectory_obs_dim() -> int:
    return _base_obs_dim() + (_TRAJECTORY_HORIZON * _TRAJECTORY_HISTORY_FEATURES) + (
        (binding.MAX_AGENTS - 1) * _TRAJECTORY_HORIZON * _TRAJECTORY_HISTORY_FEATURES
    )


def _build_policy_env_spec(dynamics_model: str) -> SimpleNamespace:
    return SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(_trajectory_obs_dim(),),
            dtype=np.float32,
        ),
        single_action_space=gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(_TRAJECTORY_ACTION_DIM,),
            dtype=np.float32,
        ),
        observation_mode=0,
        max_partner_objects=binding.MAX_AGENTS - 1,
        partner_features=binding.PARTNER_FEATURES,
        max_road_objects=binding.MAX_ROAD_SEGMENT_OBSERVATIONS,
        road_features=binding.ROAD_FEATURES,
        dynamics_model=dynamics_model,
        ego_features=binding.EGO_FEATURES_CLASSIC,
        type_classes=binding.POLICY_TYPE_CLASS_COUNT,
        is_trajectory_action=True,
        trajectory_base_obs_dim=_base_obs_dim(),
        trajectory_ego_history_dim=_TRAJECTORY_HORIZON * _TRAJECTORY_HISTORY_FEATURES,
        trajectory_partner_history_dim=(binding.MAX_AGENTS - 1) * _TRAJECTORY_HORIZON * _TRAJECTORY_HISTORY_FEATURES,
        trajectory_history_horizon=_TRAJECTORY_HORIZON,
        trajectory_history_features=_TRAJECTORY_HISTORY_FEATURES,
    )


@dataclass
class TrajectoryBCTrainConfig:
    dataset_dir: str
    output_dir: str
    val_dataset_dir: str | None = None
    device: str = "cuda"
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    num_workers: int = 0
    val_fraction: float = 0.1
    max_maps: int = -1
    max_val_maps: int = -1
    save_best: bool = True
    log_interval: int = 25
    seed: int = 42
    grad_clip_norm: float = 1.0
    max_train_samples_per_epoch: int = -1
    max_val_samples: int = -1
    sample_stride: int = 1
    sample_start_offset: int = 0
    sample_end_offset: int = 0
    wandb: bool = False
    wandb_project: str = "pufferdrive"
    wandb_group: str = "trajectory_bc"
    wandb_name: str | None = None
    wandb_tag: str | None = None
    wandb_resume_id: str | None = None
    resume_from: str | None = None


@dataclass
class TrajectoryBCEnvConfig:
    dynamics_model: str = "classic"
    dt: float = 0.1
    episode_length: int = 91
    init_steps: int = 0
    init_mode: str = "create_all_valid"
    control_mode: str = "control_sdc_only"
    goal_behavior: int = 2
    goal_target_distance: float = 30.0
    goal_radius: float = 2.0
    goal_speed: float = 100.0
    collision_behavior: int = 0
    offroad_behavior: int = 0
    termination_mode: int = 1
    observation_mode: str = _TRAJECTORY_OBSERVATION_MODE


@dataclass
class TrajectoryBCExperimentConfig:
    train: TrajectoryBCTrainConfig
    env: TrajectoryBCEnvConfig
    policy: dict[str, Any]

    @classmethod
    def from_ini(cls, config_path: str | Path) -> "TrajectoryBCExperimentConfig":
        payload = _load_ini(config_path)
        train_cfg = payload.get("bc_train", {})
        env_cfg = payload.get("env", {})
        policy_cfg = payload.get("policy", {})

        train = TrajectoryBCTrainConfig(
            dataset_dir=str(train_cfg["dataset_dir"]),
            output_dir=str(train_cfg["output_dir"]),
            val_dataset_dir=None if train_cfg.get("val_dataset_dir") in {None, "none", "None", ""} else str(train_cfg.get("val_dataset_dir")),
            device=str(train_cfg.get("device", "cuda")),
            epochs=int(train_cfg.get("epochs", 20)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            learning_rate=float(train_cfg.get("learning_rate", 0.003)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            num_workers=int(train_cfg.get("num_workers", 0)),
            val_fraction=float(train_cfg.get("val_fraction", 0.1)),
            max_maps=int(train_cfg.get("max_maps", -1)),
            max_val_maps=int(train_cfg.get("max_val_maps", -1)),
            save_best=bool(train_cfg.get("save_best", True)),
            log_interval=int(train_cfg.get("log_interval", 25)),
            seed=int(train_cfg.get("seed", 42)),
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
            max_train_samples_per_epoch=int(train_cfg.get("max_train_samples_per_epoch", -1)),
            max_val_samples=int(train_cfg.get("max_val_samples", -1)),
            sample_stride=int(train_cfg.get("sample_stride", 1)),
            sample_start_offset=int(train_cfg.get("sample_start_offset", 0)),
            sample_end_offset=int(train_cfg.get("sample_end_offset", 0)),
            wandb=bool(train_cfg.get("wandb", False)),
            wandb_project=str(train_cfg.get("wandb_project", "pufferdrive")),
            wandb_group=str(train_cfg.get("wandb_group", "trajectory_bc")),
            wandb_name=train_cfg.get("wandb_name"),
            wandb_tag=train_cfg.get("wandb_tag"),
            wandb_resume_id=train_cfg.get("wandb_resume_id"),
            resume_from=None if train_cfg.get("resume_from") in {None, "none", "None", ""} else str(train_cfg.get("resume_from")),
        )
        env = TrajectoryBCEnvConfig(
            dynamics_model=str(env_cfg.get("dynamics_model", "classic")),
            dt=float(env_cfg.get("dt", 0.1)),
            episode_length=int(env_cfg.get("episode_length", 91)),
            init_steps=int(env_cfg.get("init_steps", 0)),
            init_mode=str(env_cfg.get("init_mode", "create_all_valid")),
            control_mode=str(env_cfg.get("control_mode", "control_sdc_only")),
            goal_behavior=int(env_cfg.get("goal_behavior", 2)),
            goal_target_distance=float(env_cfg.get("goal_target_distance", 30.0)),
            goal_radius=float(env_cfg.get("goal_radius", 2.0)),
            goal_speed=float(env_cfg.get("goal_speed", 100.0)),
            collision_behavior=int(env_cfg.get("collision_behavior", 0)),
            offroad_behavior=int(env_cfg.get("offroad_behavior", 0)),
            termination_mode=int(env_cfg.get("termination_mode", 1)),
            observation_mode=str(env_cfg.get("observation_mode", _TRAJECTORY_OBSERVATION_MODE)),
        )
        return cls(train=train, env=env, policy=dict(policy_cfg))


@contextmanager
def _staged_map_dir(map_path: Path) -> Iterator[Path]:
    temp_dir = tempfile.TemporaryDirectory(prefix=f"trajectory_bc_{map_path.stem}_")
    staged_dir = Path(temp_dir.name)
    staged_map_path = staged_dir / "map_000.bin"
    try:
        try:
            os.symlink(map_path, staged_map_path)
        except OSError:
            shutil.copy2(map_path, staged_map_path)
        yield staged_dir
    finally:
        temp_dir.cleanup()


def _init_env_handle(map_dir: Path, env_config: TrajectoryBCEnvConfig):
    obs = np.zeros((binding.MAX_AGENTS, _base_obs_dim()), dtype=np.float32)
    actions = np.zeros(binding.MAX_AGENTS, dtype=np.int32)
    rewards = np.zeros(binding.MAX_AGENTS, dtype=np.float32)
    terminals = np.zeros(binding.MAX_AGENTS, dtype=np.uint8)
    truncations = np.zeros(binding.MAX_AGENTS, dtype=np.uint8)

    env_handle = binding.env_init(
        obs,
        actions,
        rewards,
        terminals,
        truncations,
        0,
        human_agent_idx=0,
        ini_file="pufferlib/config/ocean/drive.ini",
        map_dir=str(map_dir),
        map_id=0,
        max_agents=int(binding.MAX_AGENTS),
        max_controlled_agents=int(binding.MAX_AGENTS),
        init_steps=int(env_config.init_steps),
        init_mode=_init_mode_id(env_config.init_mode),
        control_mode=_control_mode_id(env_config.control_mode),
        goal_behavior=int(env_config.goal_behavior),
        goal_target_distance=float(env_config.goal_target_distance),
        goal_radius=float(env_config.goal_radius),
        goal_speed=float(env_config.goal_speed),
        collision_behavior=int(env_config.collision_behavior),
        offroad_behavior=int(env_config.offroad_behavior),
        termination_mode=int(env_config.termination_mode),
        dt=float(env_config.dt),
        episode_length=int(env_config.episode_length),
        dynamics_model=int(_DYNAMICS_MODEL_IDS.get(env_config.dynamics_model, 0)),
        force_zero_trailer_articulation_at_init=0,
    )
    return env_handle


def _get_global_agent_state(env_handle, active_count: int) -> dict[str, np.ndarray]:
    states = {
        "x": np.zeros(active_count, dtype=np.float32),
        "y": np.zeros(active_count, dtype=np.float32),
        "z": np.zeros(active_count, dtype=np.float32),
        "heading": np.zeros(active_count, dtype=np.float32),
        "id": np.zeros(active_count, dtype=np.int32),
        "length": np.zeros(active_count, dtype=np.float32),
        "width": np.zeros(active_count, dtype=np.float32),
    }
    binding.get_global_agent_state(
        env_handle,
        states["x"],
        states["y"],
        states["z"],
        states["heading"],
        states["id"],
        states["length"],
        states["width"],
    )
    return states


def _get_partner_ids(env_handle, active_count: int) -> np.ndarray:
    ids = np.zeros((active_count, binding.MAX_AGENTS - 1), dtype=np.int32)
    binding.get_partner_ids(env_handle, ids)
    return ids


def _get_ground_truth(env_handle, active_count: int, episode_length: int, init_steps: int) -> dict[str, np.ndarray]:
    horizon = episode_length - init_steps
    trajectories = {
        "x": np.zeros((active_count, horizon), dtype=np.float32),
        "y": np.zeros((active_count, horizon), dtype=np.float32),
        "z": np.zeros((active_count, horizon), dtype=np.float32),
        "heading": np.zeros((active_count, horizon), dtype=np.float32),
        "valid": np.zeros((active_count, horizon), dtype=np.int32),
        "id": np.zeros(active_count, dtype=np.int32),
        "is_vehicle": np.zeros(active_count, dtype=np.int32),
        "scenario_id": np.zeros(active_count, dtype=np.int32),
    }
    binding.get_ground_truth_trajectories(
        env_handle,
        trajectories["x"],
        trajectories["y"],
        trajectories["z"],
        trajectories["heading"],
        trajectories["valid"],
        trajectories["id"],
        trajectories["is_vehicle"],
        trajectories["scenario_id"],
    )
    return trajectories


def _get_active_agent_ids(env_handle, active_count: int) -> np.ndarray:
    scenario_ids = np.zeros(active_count, dtype=np.int32)
    agent_ids = np.zeros(active_count, dtype=np.int32)
    binding.env_get_active_agent_info(env_handle, scenario_ids, agent_ids)
    return agent_ids


def _append_history(history: dict[int, list[np.ndarray]], entity_id: int, x: float, y: float, heading: float, speed: float) -> None:
    if entity_id < 0:
        return
    entries = history.setdefault(entity_id, [])
    entries.append(_make_history_state(x, y, heading, speed, 1.0))
    if len(entries) > _TRAJECTORY_HORIZON:
        del entries[:-_TRAJECTORY_HORIZON]


def _decode_partner_states(base_observations: np.ndarray, ego_states: dict[str, np.ndarray], partner_ids: np.ndarray) -> dict[int, tuple[float, float, float, float, float]]:
    partner_states: dict[int, tuple[float, float, float, float, float]] = {}
    partner_offset = binding.EGO_FEATURES_CLASSIC
    partner_dim = (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
    partner_obs = base_observations[:, partner_offset : partner_offset + partner_dim].reshape(
        base_observations.shape[0],
        binding.MAX_AGENTS - 1,
        binding.PARTNER_FEATURES,
    )
    ego_cos = np.cos(ego_states["heading"])
    ego_sin = np.sin(ego_states["heading"])

    for row in range(base_observations.shape[0]):
        for slot in range(binding.MAX_AGENTS - 1):
            entity_id = int(partner_ids[row, slot])
            if entity_id <= 0:
                continue
            features = partner_obs[row, slot]
            if not np.any(np.abs(features) > _EMPTY_PARTNER_EPS):
                continue

            rel_x = float(features[0]) / _PARTNER_POS_SCALE
            rel_y = float(features[1]) / _PARTNER_POS_SCALE
            dx = rel_x * ego_cos[row] - rel_y * ego_sin[row]
            dy = rel_x * ego_sin[row] + rel_y * ego_cos[row]
            rel_heading = math.atan2(float(features[5]), float(features[4]))
            partner_states[entity_id] = (
                float(ego_states["x"][row] + dx),
                float(ego_states["y"][row] + dy),
                _wrap_to_pi(float(ego_states["heading"][row] + rel_heading)),
                float(features[6]) * _MAX_SPEED,
                1.0,
            )
    return partner_states


def _encode_history_block(history: list[np.ndarray] | None, current_x: float, current_y: float, current_heading: float) -> np.ndarray:
    block = np.zeros((_TRAJECTORY_HORIZON, _TRAJECTORY_HISTORY_FEATURES), dtype=np.float32)
    if not history:
        return block

    cos_heading = math.cos(current_heading)
    sin_heading = math.sin(current_heading)
    recent_states = history[-_TRAJECTORY_HORIZON:][::-1]
    for step_idx, state in enumerate(recent_states):
        dx = float(state[0] - current_x)
        dy = float(state[1] - current_y)
        rel_x = dx * cos_heading + dy * sin_heading
        rel_y = -dx * sin_heading + dy * cos_heading
        rel_heading = _wrap_to_pi(float(state[2]) - current_heading)
        block[step_idx, 0] = _encode_trajectory_position(rel_x)
        block[step_idx, 1] = _encode_trajectory_position(rel_y)
        block[step_idx, 2] = math.cos(rel_heading)
        block[step_idx, 3] = math.sin(rel_heading)
        block[step_idx, 4] = _encode_trajectory_speed(float(state[3]))
        block[step_idx, 5] = float(state[4])
    return block


def _build_trajectory_history_observations(
    base_observations: np.ndarray,
    ego_states: dict[str, np.ndarray],
    ego_ids: np.ndarray,
    partner_ids: np.ndarray,
    history: dict[int, list[np.ndarray]],
) -> np.ndarray:
    observations = np.zeros((base_observations.shape[0], _trajectory_obs_dim()), dtype=np.float32)
    observations[:, : _base_obs_dim()] = base_observations

    ego_hist_start = _base_obs_dim()
    partner_hist_start = ego_hist_start + (_TRAJECTORY_HORIZON * _TRAJECTORY_HISTORY_FEATURES)

    for row in range(base_observations.shape[0]):
        current_x = float(ego_states["x"][row])
        current_y = float(ego_states["y"][row])
        current_heading = float(ego_states["heading"][row])
        ego_id = int(ego_ids[row])

        ego_block = _encode_history_block(history.get(ego_id), current_x, current_y, current_heading)
        observations[row, ego_hist_start:partner_hist_start] = ego_block.reshape(-1)

        partner_block = np.zeros(
            (binding.MAX_AGENTS - 1, _TRAJECTORY_HORIZON, _TRAJECTORY_HISTORY_FEATURES),
            dtype=np.float32,
        )
        for slot in range(binding.MAX_AGENTS - 1):
            partner_id = int(partner_ids[row, slot])
            if partner_id <= 0:
                continue
            partner_block[slot] = _encode_history_block(history.get(partner_id), current_x, current_y, current_heading)
        observations[row, partner_hist_start:] = partner_block.reshape(-1)
    return observations


def _build_trajectory_targets(
    trajectories: dict[str, np.ndarray],
    current_states: dict[str, np.ndarray],
    *,
    current_timestep: int,
    dt: float,
) -> np.ndarray:
    targets = np.zeros((current_states["x"].shape[0], _TRAJECTORY_HORIZON, _TRAJECTORY_FEATURES), dtype=np.float32)
    start_idx = int(current_timestep) + 1

    for row in range(current_states["x"].shape[0]):
        current_x = float(current_states["x"][row])
        current_y = float(current_states["y"][row])
        current_heading = float(current_states["heading"][row])
        cos_heading = math.cos(current_heading)
        sin_heading = math.sin(current_heading)

        for step_idx in range(_TRAJECTORY_HORIZON):
            traj_idx = start_idx + step_idx
            if traj_idx >= trajectories["x"].shape[1]:
                break
            valid = float(trajectories["valid"][row, traj_idx])
            if valid <= 0.0:
                continue

            world_x = float(trajectories["x"][row, traj_idx])
            world_y = float(trajectories["y"][row, traj_idx])
            dx = world_x - current_x
            dy = world_y - current_y
            rel_x = dx * cos_heading + dy * sin_heading
            rel_y = -dx * sin_heading + dy * cos_heading
            rel_heading = _wrap_to_pi(float(trajectories["heading"][row, traj_idx]) - current_heading)

            prev_idx = max(traj_idx - 1, 0)
            prev_x = float(trajectories["x"][row, prev_idx])
            prev_y = float(trajectories["y"][row, prev_idx])
            speed = math.sqrt((world_x - prev_x) ** 2 + (world_y - prev_y) ** 2) / max(float(dt), 1e-6)

            targets[row, step_idx, 0] = _encode_trajectory_position(rel_x)
            targets[row, step_idx, 1] = _encode_trajectory_position(rel_y)
            targets[row, step_idx, 2] = _encode_trajectory_heading(rel_heading)
            targets[row, step_idx, 3] = _encode_trajectory_speed(speed)
            targets[row, step_idx, 4] = valid
    return targets.reshape(current_states["x"].shape[0], _TRAJECTORY_ACTION_DIM)


def _iter_sample_timesteps(total_steps: int, *, start_offset: int, end_offset: int, stride: int) -> range:
    effective_stride = max(1, int(stride))
    start = max(0, int(start_offset))
    end = max(0, (int(total_steps) - 1) - max(0, int(end_offset)))
    if end <= start:
        return range(0, 0, effective_stride)
    return range(start, end, effective_stride)


def iter_trajectory_bc_samples(
    map_path: str | Path,
    env_config: TrajectoryBCEnvConfig,
    *,
    sample_stride: int = 1,
    sample_start_offset: int = 0,
    sample_end_offset: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    map_path = Path(map_path).expanduser().resolve()
    with _staged_map_dir(map_path) as staged_dir:
        env_handle = _init_env_handle(staged_dir, env_config)
        try:
            binding.env_reset(env_handle, 0)
            active_count = int(binding.env_get_active_agent_count(env_handle))
            if active_count <= 0:
                return

            trajectories = _get_ground_truth(env_handle, active_count, env_config.episode_length, env_config.init_steps)
            ego_ids = _get_active_agent_ids(env_handle, active_count)
            history: dict[int, list[np.ndarray]] = {}
            sample_timesteps = list(
                _iter_sample_timesteps(
                    trajectories["x"].shape[1],
                    start_offset=sample_start_offset,
                    end_offset=sample_end_offset,
                    stride=sample_stride,
                )
            )
            if not sample_timesteps:
                return
            selected_timesteps = set(sample_timesteps)

            for timestep in range(max(sample_timesteps) + 1):
                binding.env_set_logged_timestep(env_handle, timestep)
                base_observations = np.zeros((active_count, _base_obs_dim()), dtype=np.float32)
                binding.env_copy_observations(env_handle, base_observations)
                current_states = _get_global_agent_state(env_handle, active_count)
                partner_ids = _get_partner_ids(env_handle, active_count)
                current_speeds = base_observations[:, 2] * _MAX_SPEED

                for row in range(active_count):
                    _append_history(
                        history,
                        int(ego_ids[row]),
                        float(current_states["x"][row]),
                        float(current_states["y"][row]),
                        float(current_states["heading"][row]),
                        float(current_speeds[row]),
                    )

                for entity_id, state in _decode_partner_states(base_observations, current_states, partner_ids).items():
                    _append_history(history, entity_id, state[0], state[1], state[2], state[3])

                if timestep not in selected_timesteps:
                    continue

                observations = _build_trajectory_history_observations(
                    base_observations=base_observations,
                    ego_states=current_states,
                    ego_ids=ego_ids,
                    partner_ids=partner_ids,
                    history=history,
                )
                targets = _build_trajectory_targets(
                    trajectories=trajectories,
                    current_states=current_states,
                    current_timestep=timestep,
                    dt=env_config.dt,
                )

                anchor_valid = trajectories["valid"][:, timestep] > 0
                target_valid = targets.reshape(active_count, _TRAJECTORY_HORIZON, _TRAJECTORY_FEATURES)[:, :, 4].sum(axis=1) > 0
                keep_rows = np.flatnonzero(anchor_valid & target_valid)
                for row in keep_rows:
                    yield observations[row].astype(np.float32), targets[row].astype(np.float32)
        finally:
            binding.env_close(env_handle)


class TrajectoryBCIterableDataset(IterableDataset):
    def __init__(
        self,
        map_paths: list[Path],
        env_config: TrajectoryBCEnvConfig,
        *,
        shuffle: bool,
        seed: int,
        epoch: int = 0,
        max_samples: int = -1,
        sample_stride: int = 1,
        sample_start_offset: int = 0,
        sample_end_offset: int = 0,
    ):
        self.map_paths = [Path(path) for path in map_paths]
        self.env_config = env_config
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.max_samples = int(max_samples)
        self.sample_stride = int(sample_stride)
        self.sample_start_offset = int(sample_start_offset)
        self.sample_end_offset = int(sample_end_offset)

    def with_epoch(self, epoch: int) -> "TrajectoryBCIterableDataset":
        return TrajectoryBCIterableDataset(
            map_paths=self.map_paths,
            env_config=self.env_config,
            shuffle=self.shuffle,
            seed=self.seed,
            epoch=epoch,
            max_samples=self.max_samples,
            sample_stride=self.sample_stride,
            sample_start_offset=self.sample_start_offset,
            sample_end_offset=self.sample_end_offset,
        )

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        map_paths = self.map_paths[worker_id::num_workers]
        indices = list(range(len(map_paths)))
        if self.shuffle:
            rng = random.Random(self.seed + (self.epoch * 100003) + worker_id)
            rng.shuffle(indices)

        sample_budget = -1
        if self.max_samples > 0:
            sample_budget = max(1, int(math.ceil(self.max_samples / max(num_workers, 1))))

        yielded = 0
        for index in indices:
            for observation, target in iter_trajectory_bc_samples(
                map_paths[index],
                self.env_config,
                sample_stride=self.sample_stride,
                sample_start_offset=self.sample_start_offset,
                sample_end_offset=self.sample_end_offset,
            ):
                yield {
                    "observation": torch.from_numpy(observation),
                    "target": torch.from_numpy(target),
                }
                yielded += 1
                if sample_budget > 0 and yielded >= sample_budget:
                    return


class TrajectoryBCTrainer:
    def __init__(self, experiment: TrajectoryBCExperimentConfig):
        self.experiment = experiment
        self.output_dir = Path(experiment.train.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_map_paths = _discover_map_paths(experiment.train.dataset_dir)
        self.train_maps: list[Path]
        self.val_maps: list[Path]
        if experiment.train.val_dataset_dir:
            self.train_maps = _select_map_subset(
                train_map_paths,
                seed=experiment.train.seed,
                max_maps=experiment.train.max_maps,
            )
            val_map_paths = _discover_map_paths(experiment.train.val_dataset_dir)
            self.val_maps = _select_map_subset(
                val_map_paths,
                seed=experiment.train.seed + 1,
                max_maps=experiment.train.max_val_maps,
            )
        else:
            self.train_maps, self.val_maps = _split_map_paths(
                train_map_paths,
                val_fraction=experiment.train.val_fraction,
                seed=experiment.train.seed,
                max_maps=experiment.train.max_maps,
            )

        policy_env = _build_policy_env_spec(experiment.env.dynamics_model)
        self.policy = DrivePolicy(policy_env, **experiment.policy)
        self.device = _resolve_training_device(experiment.train.device)
        print(f"[trajectory-bc] using device={self.device}")
        self.policy.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=experiment.train.learning_rate,
            weight_decay=experiment.train.weight_decay,
        )
        self.best_val_loss = float("inf")
        self.global_train_step = 0
        self.metrics_history: list[dict[str, Any]] = []
        self.wandb_logger = _TrajectoryBCWandbLogger.maybe_create(experiment, self.output_dir)
        if self.experiment.train.resume_from:
            self._load_checkpoint(Path(self.experiment.train.resume_from))

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(payload["model_state_dict"])

        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            val_metrics = metrics.get("val")
            train_metrics = metrics.get("train")
            best_loss = None
            if isinstance(val_metrics, dict):
                best_loss = val_metrics.get("loss")
            elif isinstance(train_metrics, dict):
                best_loss = train_metrics.get("loss")
            if best_loss is not None:
                self.best_val_loss = float(best_loss)

        print(f"[trajectory-bc] resumed from checkpoint={checkpoint_path}")

    def _make_loader(self, map_paths: list[Path], *, epoch: int, shuffle: bool, max_samples: int) -> DataLoader:
        dataset = TrajectoryBCIterableDataset(
            map_paths=map_paths,
            env_config=self.experiment.env,
            shuffle=shuffle,
            seed=self.experiment.train.seed,
            epoch=epoch,
            max_samples=max_samples,
            sample_stride=self.experiment.train.sample_stride,
            sample_start_offset=self.experiment.train.sample_start_offset,
            sample_end_offset=self.experiment.train.sample_end_offset,
        )
        return DataLoader(
            dataset,
            batch_size=self.experiment.train.batch_size,
            num_workers=self.experiment.train.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def _predict(self, observations: torch.Tensor) -> torch.Tensor:
        actions, _ = self.policy(observations)
        if hasattr(actions, "mean"):
            return actions.mean
        if hasattr(actions, "loc"):
            return actions.loc
        if isinstance(actions, (tuple, list)):
            return torch.cat(actions, dim=1)
        return actions

    def _run_epoch(self, loader: DataLoader, *, train: bool, epoch_idx: int) -> dict[str, float]:
        mode = "train" if train else "val"
        self.policy.train(mode == "train")

        total_loss = 0.0
        total_samples = 0
        metric_sums: dict[str, float] = {}
        start_time = time.time()

        for batch_idx, batch in enumerate(loader, start=1):
            observations = batch["observation"].to(self.device, non_blocking=True).float()
            targets = batch["target"].to(self.device, non_blocking=True).float()

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            predictions = self._predict(observations)
            loss = masked_trajectory_loss(predictions, targets)

            if train:
                loss.backward()
                if self.experiment.train.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.experiment.train.grad_clip_norm)
                self.optimizer.step()

            batch_size = int(observations.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            batch_metrics = trajectory_metrics(predictions.detach(), targets.detach())
            for key, value in {"loss_total": loss.detach(), **batch_metrics}.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_size

            if train:
                self.global_train_step += 1
                if self.wandb_logger is not None and (
                    self.experiment.train.log_interval <= 0
                    or batch_idx % self.experiment.train.log_interval == 0
                ):
                    self.wandb_logger.log_batch(
                        epoch=epoch_idx + 1,
                        batch_idx=batch_idx,
                        global_step=self.global_train_step,
                        loss=float(loss.detach().cpu()),
                        metrics=batch_metrics,
                        learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    )

            if train and self.experiment.train.log_interval > 0 and batch_idx % self.experiment.train.log_interval == 0:
                print(
                    f"[trajectory-bc] epoch={epoch_idx + 1} mode={mode} batch={batch_idx} "
                    f"loss={float(loss.detach().cpu()):.6f}"
                )

        if total_samples <= 0:
            raise RuntimeError(f"No samples produced for {mode} epoch {epoch_idx + 1}")

        summary = {
            "loss": total_loss / total_samples,
            "num_samples": float(total_samples),
            "epoch_seconds": time.time() - start_time,
        }
        for key, total in metric_sums.items():
            summary[key] = total / total_samples
        return summary

    def _save_checkpoint(self, filename: str, metrics: dict[str, Any]) -> Path:
        path = self.output_dir / filename
        payload = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": {
                "train": asdict(self.experiment.train),
                "env": asdict(self.experiment.env),
                "policy": dict(self.experiment.policy),
            },
            "train_maps": [str(path) for path in self.train_maps],
            "val_maps": [str(path) for path in self.val_maps],
        }
        torch.save(payload, path)
        if self.wandb_logger is not None:
            self.wandb_logger.log_checkpoint(path)
        return path

    def train(self) -> dict[str, Any]:
        summary = {
            "train_map_count": len(self.train_maps),
            "val_map_count": len(self.val_maps),
            "dataset_dir": self.experiment.train.dataset_dir,
            "val_dataset_dir": self.experiment.train.val_dataset_dir,
            "output_dir": str(self.output_dir),
            "device": str(self.device),
            "history": [],
        }
        if self.wandb_logger is not None:
            summary["wandb_run_id"] = self.wandb_logger.run_id

        try:
            for epoch_idx in range(self.experiment.train.epochs):
                train_loader = self._make_loader(
                    self.train_maps,
                    epoch=epoch_idx,
                    shuffle=True,
                    max_samples=self.experiment.train.max_train_samples_per_epoch,
                )
                train_metrics = self._run_epoch(train_loader, train=True, epoch_idx=epoch_idx)

                epoch_summary: dict[str, Any] = {"epoch": epoch_idx + 1, "train": train_metrics}
                if self.val_maps:
                    val_loader = self._make_loader(
                        self.val_maps,
                        epoch=epoch_idx,
                        shuffle=False,
                        max_samples=self.experiment.train.max_val_samples,
                    )
                    val_metrics = self._run_epoch(val_loader, train=False, epoch_idx=epoch_idx)
                    epoch_summary["val"] = val_metrics
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = float(val_metrics["loss"])
                        if self.experiment.train.save_best:
                            self._save_checkpoint("best.pt", epoch_summary)
                else:
                    if train_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = float(train_metrics["loss"])
                        if self.experiment.train.save_best:
                            self._save_checkpoint("best.pt", epoch_summary)

                self._save_checkpoint("last.pt", epoch_summary)
                self.metrics_history.append(epoch_summary)
                summary["history"].append(epoch_summary)
                if self.wandb_logger is not None:
                    self.wandb_logger.log_epoch(epoch_summary)
                print(
                    f"[trajectory-bc] epoch={epoch_idx + 1}/{self.experiment.train.epochs} "
                    f"train_loss={train_metrics['loss']:.6f}"
                    + (
                        f" val_loss={epoch_summary['val']['loss']:.6f}"
                        if "val" in epoch_summary
                        else ""
                    )
                )

            metrics_path = self.output_dir / "metrics.json"
            metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            config_path = self.output_dir / "training_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "train": asdict(self.experiment.train),
                        "env": asdict(self.experiment.env),
                        "policy": dict(self.experiment.policy),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if self.wandb_logger is not None:
                self.wandb_logger.log_summary(summary)
        finally:
            if self.wandb_logger is not None:
                self.wandb_logger.close()
        return summary


class _TrajectoryBCWandbLogger:
    def __init__(self, wandb_module, run):
        self.wandb = wandb_module
        self.run = run
        self.run_id = getattr(run, "id", None)

    def _define_default_metrics(self) -> None:
        define_metric = getattr(self.run, "define_metric", None)
        if define_metric is None:
            define_metric = getattr(self.wandb, "define_metric", None)
        if define_metric is None:
            return

        epoch_metrics = (
            "loss",
            "loss_total",
            "ade",
            "fde",
            "heading_error",
            "speed_error",
            "valid_accuracy",
            "num_samples",
            "epoch_seconds",
        )
        batch_metrics = (
            "loss",
            "ade",
            "fde",
            "heading_error",
            "speed_error",
            "valid_accuracy",
            "learning_rate",
            "batch_idx",
        )

        define_metric("epoch")
        define_metric("batch_step")
        for split_name in ("train", "val"):
            define_metric(f"{split_name}/*", step_metric="epoch")
            for metric_name in epoch_metrics:
                define_metric(f"{split_name}/{metric_name}", step_metric="epoch")
        define_metric("train_batch/*", step_metric="batch_step")
        for metric_name in batch_metrics:
            define_metric(f"train_batch/{metric_name}", step_metric="batch_step")

    @classmethod
    def maybe_create(cls, experiment: TrajectoryBCExperimentConfig, output_dir: Path) -> "_TrajectoryBCWandbLogger | None":
        if not experiment.train.wandb:
            return None
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "wandb logging was requested, but the wandb package is not installed in this environment."
            ) from exc

        config = {
            "train": asdict(experiment.train),
            "env": asdict(experiment.env),
            "policy": dict(experiment.policy),
            "output_dir": str(output_dir),
        }
        resume_id = experiment.train.wandb_resume_id
        run = wandb.init(
            id=resume_id or wandb.util.generate_id(),
            resume="allow",
            project=experiment.train.wandb_project,
            group=experiment.train.wandb_group,
            allow_val_change=True,
            save_code=False,
            config=config,
            name=experiment.train.wandb_name,
            tags=[experiment.train.wandb_tag] if experiment.train.wandb_tag else [],
        )
        logger = cls(wandb, run)
        logger._define_default_metrics()
        return logger

    def log_epoch(self, epoch_summary: dict[str, Any]) -> None:
        payload = {"epoch": int(epoch_summary["epoch"])}
        for split_name in ("train", "val"):
            split_metrics = epoch_summary.get(split_name)
            if not split_metrics:
                continue
            for key, value in split_metrics.items():
                if key == "num_samples":
                    payload[f"{split_name}/{key}"] = int(value)
                else:
                    payload[f"{split_name}/{key}"] = float(value)
        self.wandb.log(payload)

    def log_batch(
        self,
        *,
        epoch: int,
        batch_idx: int,
        global_step: int,
        loss: float,
        metrics: dict[str, float],
        learning_rate: float,
    ) -> None:
        payload = {
            "epoch": int(epoch),
            "batch_step": int(global_step),
            "train_batch/batch_idx": int(batch_idx),
            "train_batch/loss": float(loss),
            "train_batch/learning_rate": float(learning_rate),
        }
        for key, value in metrics.items():
            payload[f"train_batch/{key}"] = float(value)
        self.wandb.log(payload)

    def log_checkpoint(self, checkpoint_path: Path) -> None:
        try:
            artifact = self.wandb.Artifact(checkpoint_path.stem, type="model")
            artifact.add_file(str(checkpoint_path))
            self.run.log_artifact(artifact)
        except Exception:
            pass

    def log_summary(self, summary: dict[str, Any]) -> None:
        if hasattr(self.run, "summary"):
            self.run.summary["train_map_count"] = int(summary["train_map_count"])
            self.run.summary["val_map_count"] = int(summary["val_map_count"])
            self.run.summary["output_dir"] = str(summary["output_dir"])
            self.run.summary["device"] = str(summary["device"])

    def close(self) -> None:
        self.wandb.finish()


def run_trajectory_bc_training(config_path: str | Path) -> dict[str, Any]:
    experiment = TrajectoryBCExperimentConfig.from_ini(config_path)
    trainer = TrajectoryBCTrainer(experiment)
    return trainer.train()
