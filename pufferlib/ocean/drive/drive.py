import numpy as np
import gymnasium
import json
import struct
import os
import hashlib
import ast
import configparser
import random
import time
from pathlib import Path
import pufferlib
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from pufferlib.ocean.drive import binding
from pufferlib.ocean import torch as ocean_torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

_POLICY_TYPE_PADDED = 0
_EMPTY_PARTNER_EPS = 1e-8
_EGO_TRAILER_STATE_FEATURES = 4
_CLASSIC_ACCELERATION_VALUES = (-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0)
_CLASSIC_STEERING_VALUES = (-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0)
_CLASSIC_DISCRETE_ACTIONS = len(_CLASSIC_ACCELERATION_VALUES) * len(_CLASSIC_STEERING_VALUES)
_NON_KINEMATIC_PARAM_ORDER = [
    "tractor_length",
    "trailer_length",
    "width",
    "trailer_width",
    "vehicle_height",
    "trailer_height",
    "tractor2hitch",
    "trailer2hitch",
    "tractor_d_rear_axle2rear_bumper",
    "tractor_d_rear_axle2front_axle",
    "tractor_d_front_axle2front_bumper",
    "trailer_d_rear_axel2_rear_bumper",
    "trailer_d_real_axel2_front_bumper",
]
_NON_KINEMATIC_PARAM_CACHE = {}
DEFAULT_SDC_RUNTIME_TRUCK_REF_BIN = (
    "tests/artifacts/drive/traversing_traffic_light_intersection__97be27351e915863__97be27351e915863.bin"
)


def _print_mismatch(message):
    print(f"[Drive mismatch] {message}")


def _compare_c_python_env_config(env_handle, expected, context):
    if not hasattr(binding, "env_get_config"):
        return

    actual = binding.env_get_config(env_handle)
    mismatches = []
    float_keys = {
        "reward_vehicle_collision",
        "reward_offroad_collision",
        "reward_goal",
        "reward_goal_post_respawn",
        "goal_radius",
        "goal_speed",
        "goal_target_distance",
        "dt",
    }
    for key, expected_value in expected.items():
        if key not in actual:
            continue
        actual_value = actual[key]
        if key in float_keys:
            if not np.isclose(float(actual_value), float(expected_value), atol=1e-6, rtol=1e-6):
                mismatches.append((key, expected_value, actual_value))
        elif actual_value != expected_value:
            mismatches.append((key, expected_value, actual_value))

    for key, expected_value, actual_value in mismatches:
        _print_mismatch(
            f"{context}: Python expected {key}={expected_value!r}, but C resolved {key}={actual_value!r}"
        )


def _expected_c_env_config(
    *,
    action_type,
    dynamics_model,
    observation_mode,
    extend_classic_action_space,
    reward_vehicle_collision,
    reward_offroad_collision,
    reward_goal,
    reward_goal_post_respawn,
    goal_radius,
    goal_speed,
    goal_behavior,
    goal_target_distance,
    collision_behavior,
    offroad_behavior,
    dt,
    episode_length,
    termination_mode,
    init_steps,
    init_mode,
    control_mode,
    max_controlled_agents,
):
    action_type_flag = 0 if action_type == "discrete" else 1
    dynamics_flag = 0 if dynamics_model == "classic" else 1
    observation_flag = 0 if observation_mode == "default" else 1
    control_flag = {
        "control_vehicles": 0,
        "control_agents": 1,
        "control_wosac": 2,
        "control_sdc_only": 3,
    }[control_mode]
    init_flag = 0 if init_mode == "create_all_valid" else 1
    return {
        "action_type": action_type_flag,
        "dynamics_model": dynamics_flag,
        "observation_mode": observation_flag,
        "extend_classic_action_space": int(_as_bool(extend_classic_action_space)),
        "reward_vehicle_collision": reward_vehicle_collision,
        "reward_offroad_collision": reward_offroad_collision,
        "reward_goal": reward_goal,
        "reward_goal_post_respawn": reward_goal_post_respawn,
        "goal_radius": goal_radius,
        "goal_speed": goal_speed,
        "goal_behavior": goal_behavior,
        "goal_target_distance": goal_target_distance,
        "collision_behavior": collision_behavior,
        "offroad_behavior": offroad_behavior,
        "dt": dt,
        "episode_length": int(episode_length) if episode_length is not None else None,
        "termination_mode": int(termination_mode) if termination_mode is not None else 0,
        "init_steps": init_steps,
        "init_mode": init_flag,
        "control_mode": control_flag,
        "max_controlled_agents": int(max_controlled_agents),
    }


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _normalize_action_type(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("discrete", "continuous"):
            return normalized
    elif isinstance(value, (int, np.integer)):
        if int(value) == 0:
            return "discrete"
        if int(value) == 1:
            return "continuous"
    raise ValueError(f"action_type must be 'discrete' or 'continuous'. Got: {value}")


def _normalize_dynamics_model(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("classic", "jerk"):
            return normalized
    elif isinstance(value, (int, np.integer)):
        if int(value) == 0:
            return "classic"
        if int(value) == 1:
            return "jerk"
    raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {value}")


def _normalize_observation_mode(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "default":
            return "default"
        if normalized == "sdc_only_with_trailer":
            return "sdc_only_with_trailer"
    elif isinstance(value, (int, np.integer)):
        if int(value) == 0:
            return "default"
        if int(value) == 1:
            return "sdc_only_with_trailer"
    raise ValueError(f"observation_mode must be 'default' or 'sdc_only_with_trailer'. Got: {value}")


def _normalize_init_mode(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("create_all_valid", "created_all_valid"):
            return "create_all_valid"
        if normalized == "create_only_controlled":
            return "create_only_controlled"
    elif isinstance(value, (int, np.integer)):
        if int(value) == 0:
            return "create_all_valid"
        if int(value) == 1:
            return "create_only_controlled"
    raise ValueError(
        f"init_mode must be one of 'create_all_valid' or 'create_only_controlled'. Got: {value}"
    )


def _normalize_control_mode(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("control_vehicles", "control_agents", "control_wosac", "control_sdc_only"):
            return normalized
    elif isinstance(value, (int, np.integer)):
        mapping = {
            0: "control_vehicles",
            1: "control_agents",
            2: "control_wosac",
            3: "control_sdc_only",
        }
        if int(value) in mapping:
            return mapping[int(value)]
    raise ValueError(
        f"control_mode must be one of 'control_vehicles', 'control_agents', 'control_wosac', or 'control_sdc_only'. Got: {value}"
    )


def _normalize_env_config(env_cfg):
    normalized = dict(env_cfg)
    if "action_type" in normalized:
        normalized["action_type"] = _normalize_action_type(normalized["action_type"])
    if "dynamics_model" in normalized:
        normalized["dynamics_model"] = _normalize_dynamics_model(normalized["dynamics_model"])
    if "observation_mode" in normalized:
        normalized["observation_mode"] = _normalize_observation_mode(normalized["observation_mode"])
    if "init_mode" in normalized:
        normalized["init_mode"] = _normalize_init_mode(normalized["init_mode"])
    if "control_mode" in normalized:
        normalized["control_mode"] = _normalize_control_mode(normalized["control_mode"])
    return normalized


def _resolve_base_arg(args, key, default=None):
    if key in args:
        return args[key]
    return args.get("base", {}).get(key, default)


def _load_non_kinematic_vehicle_params_from_bin(binary_path):
    cached = _NON_KINEMATIC_PARAM_CACHE.get(binary_path)
    if cached is not None:
        return cached

    with open(binary_path, "rb") as f:
        _ = struct.unpack("<i", f.read(4))[0]
        num_tracks_to_predict = struct.unpack("<i", f.read(4))[0]
        f.seek(4 * num_tracks_to_predict, os.SEEK_CUR)
        num_objects = struct.unpack("<i", f.read(4))[0]
        num_roads = struct.unpack("<i", f.read(4))[0]

        for _ in range(num_objects):
            _ = struct.unpack("<i", f.read(4))[0]
            _ = struct.unpack("<i", f.read(4))[0]
            _ = struct.unpack("<i", f.read(4))[0]
            trajectory_length = struct.unpack("<i", f.read(4))[0]
            f.seek(4 * trajectory_length * 3, os.SEEK_CUR)
            f.seek((4 * trajectory_length * 4) + (4 * trajectory_length), os.SEEK_CUR)
            f.seek((6 * 4) + 4, os.SEEK_CUR)

        for _ in range(num_roads):
            _ = struct.unpack("<i", f.read(4))[0]
            _ = struct.unpack("<i", f.read(4))[0]
            _ = struct.unpack("<i", f.read(4))[0]
            array_size = struct.unpack("<i", f.read(4))[0]
            f.seek(4 * array_size * 3, os.SEEK_CUR)
            f.seek((6 * 4) + 4, os.SEEK_CUR)

        extension_magic = struct.unpack("<i", f.read(4))[0]
        extension_version = struct.unpack("<i", f.read(4))[0]
        if extension_magic != 0x54524C52:
            message = f"Extension magic mismatch in {binary_path}"
            _print_mismatch(message)
            raise ValueError(message)
        if extension_version != 2:
            message = f"Unsupported extension version {extension_version} in {binary_path}"
            _print_mismatch(message)
            raise ValueError(message)

        f.seek(4 * 2, os.SEEK_CUR)
        object_meta_count = struct.unpack("<i", f.read(4))[0]
        f.seek(object_meta_count * (8 + 4 + 4), os.SEEK_CUR)

        vehicle_param_count = struct.unpack("<i", f.read(4))[0]
        if vehicle_param_count != len(_NON_KINEMATIC_PARAM_ORDER):
            message = (
                f"Expected {len(_NON_KINEMATIC_PARAM_ORDER)} non-kinematic params in {binary_path}, "
                f"got {vehicle_param_count}"
            )
            _print_mismatch(message)
            raise ValueError(message)
        values = struct.unpack(f"<{vehicle_param_count}f", f.read(4 * vehicle_param_count))

    params = tuple(float(v) for v in values)
    _NON_KINEMATIC_PARAM_CACHE[binary_path] = params
    return params


def _sim_obs_dim_for_dynamics(dynamics_model):
    base_ego = {"classic": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}[dynamics_model]
    return base_ego + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES


def _postprocess_policy_observations(
    output_observations,
    sim_observations,
    base_ego,
    base_partner,
    partner_count,
    aug_ego,
    aug_partner,
    road_count,
    road_features,
    type_classes,
    ego_types,
    partner_types,
    ego_trailer_features,
):
    base_partner_dim = partner_count * base_partner
    base_road_start = base_ego + base_partner_dim
    road_dim = road_count * road_features

    aug_partner_dim = partner_count * aug_partner
    aug_road_start = aug_ego + aug_partner_dim

    output_observations.fill(0.0)
    output_observations[:, :base_ego] = sim_observations[:, :base_ego]

    sim_partner = sim_observations[:, base_ego:base_road_start].reshape(sim_observations.shape[0], partner_count, base_partner)
    aug_partner_view = output_observations[:, aug_ego:aug_road_start].reshape(
        output_observations.shape[0], partner_count, aug_partner
    )
    aug_partner_view[:, :, :base_partner] = sim_partner
    output_observations[:, aug_road_start : aug_road_start + road_dim] = sim_observations[
        :, base_road_start : base_road_start + road_dim
    ]

    policy_type_max = type_classes - 1
    output_observations[:, base_ego] = np.clip(ego_types, _POLICY_TYPE_PADDED, policy_type_max).astype(np.float32)
    trailer_feature_start = base_ego + 1
    output_observations[:, trailer_feature_start] = ego_trailer_features["rel_x"]
    output_observations[:, trailer_feature_start + 1] = ego_trailer_features["rel_y"]
    output_observations[:, trailer_feature_start + 2] = ego_trailer_features["rel_heading_x"]
    output_observations[:, trailer_feature_start + 3] = ego_trailer_features["rel_heading_y"]

    occupied_partner_slots = np.any(np.abs(sim_partner) > _EMPTY_PARTNER_EPS, axis=2)
    aug_partner_view[:, :, base_partner] = np.where(
        occupied_partner_slots,
        np.clip(partner_types, _POLICY_TYPE_PADDED, policy_type_max),
        _POLICY_TYPE_PADDED,
    ).astype(np.float32)


def _parse_config_value(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def load_drive_builder_config(config_path=None):
    puffer_root = Path(__file__).resolve().parents[2]
    default_ini = puffer_root / "config" / "default.ini"
    if config_path is None:
        config_path = puffer_root / "config" / "ocean" / "drive.ini"
    parser = configparser.ConfigParser()
    parser.read([str(default_ini), str(config_path)])
    args = {}
    for section in parser.sections():
        args[section] = {key: _parse_config_value(parser[section][key]) for key in parser[section]}
    return args


def _resolve_bc_config(args, output_dir=None):
    bc = dict(args.get("bc", {}))
    env = args["env"]
    bc.setdefault("output_dir", output_dir or os.path.join(env["map_dir"], "..", "bc_dataset"))
    bc.setdefault("beam_width", 8)
    bc.setdefault("planning_horizon", -1)
    bc.setdefault("match_weight_lateral", 2.5)
    bc.setdefault("match_weight_longitudinal", 1.5)
    bc.setdefault("match_weight_heading", 0.1)
    bc.setdefault("match_weight_speed", 0.02)
    bc.setdefault("match_weight_steer_change", 0.15)
    bc.setdefault("match_weight_accel_change", 0.02)
    bc.setdefault("match_weight_reverse", 1.0)
    bc.setdefault("match_weight_progress", 4.0)
    bc.setdefault("match_weight_steer_flip", 0.5)
    bc.setdefault("match_weight_ref_accel", 0.01)
    bc.setdefault("match_weight_ref_steer", 0.1)
    bc.setdefault("skip_existing_shards", True)
    bc.setdefault("max_maps", -1)
    return bc


def _resolve_bc_train_config(args, dataset_dir=None, output_dir=None):
    bc_train = dict(args.get("bc_train", {}))
    bc = _resolve_bc_config(args)
    train = dict(args.get("train", {}))
    bc_train.setdefault("dataset_dir", dataset_dir or bc.get("output_dir"))
    bc_train.setdefault("output_dir", output_dir or os.path.join(bc_train["dataset_dir"], "checkpoints"))
    bc_train.setdefault("device", train.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    bc_train.setdefault("epochs", 10)
    bc_train.setdefault("batch_size", 256)
    bc_train.setdefault("learning_rate", train.get("learning_rate", 3e-4))
    bc_train.setdefault("weight_decay", 0.0)
    bc_train.setdefault("num_workers", 0)
    bc_train.setdefault("val_fraction", 0.1)
    bc_train.setdefault("seq_len", train.get("bptt_horizon", 32))
    bc_train.setdefault("sequence_stride", bc_train["seq_len"])
    bc_train.setdefault("max_shards", -1)
    bc_train.setdefault("save_best", True)
    bc_train.setdefault("log_interval", 25)
    return bc_train


def _normalize_optional_name(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return value


def _get_discrete_action_size(env):
    action_space = env.single_action_space
    if not isinstance(action_space, gymnasium.spaces.MultiDiscrete) or len(action_space.nvec) != 1:
        message = "Offline BC trainer currently supports only discrete Drive policies with a joint action head"
        _print_mismatch(message)
        raise ValueError(message)
    return int(action_space.nvec[0])


class _StreamingBCIterableDataset(IterableDataset):
    def __init__(self, shard_paths, obs_dim, action_space_size, *, shuffle, seed):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.obs_dim = int(obs_dim)
        self.action_space_size = int(action_space_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._length = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _iter_worker_shards(self):
        shard_paths = list(self.shard_paths)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(shard_paths)
        worker = get_worker_info()
        if worker is None:
            return shard_paths
        return shard_paths[worker.id :: worker.num_workers]

    def _load_shard(self, shard_path):
        payload = torch.load(shard_path, map_location="cpu")
        _validate_bc_shard_payload(payload, self.obs_dim, self.action_space_size, shard_path)
        return payload

    def __len__(self):
        if self._length is None:
            self._length = int(self._compute_length())
        return self._length

    def _compute_length(self):
        raise NotImplementedError


class _FlatBCDataset(_StreamingBCIterableDataset):
    def _compute_length(self):
        total = 0
        for shard_path in self.shard_paths:
            payload = self._load_shard(shard_path)
            total += int(payload["action"].shape[0])
        return total

    def __iter__(self):
        row_seed = self.seed + self.epoch * 9973
        for shard_idx, shard_path in enumerate(self._iter_worker_shards()):
            payload = self._load_shard(shard_path)
            row_indices = list(range(int(payload["action"].shape[0])))
            if self.shuffle:
                rng = random.Random(row_seed + shard_idx)
                rng.shuffle(row_indices)
            obs = payload["obs"].float()
            action = payload["action"].long()
            for row_idx in row_indices:
                yield obs[row_idx], action[row_idx]


class _SequenceBCDataset(_StreamingBCIterableDataset):
    def __init__(self, shard_paths, obs_dim, action_space_size, *, seq_len, stride, shuffle, seed):
        super().__init__(shard_paths, obs_dim, action_space_size, shuffle=shuffle, seed=seed)
        self.seq_len = int(seq_len)
        self.stride = max(1, int(stride))

    def _compute_length(self):
        total = 0
        for shard_path in self.shard_paths:
            payload = self._load_shard(shard_path)
            total += len(_build_sequence_samples_from_payload(payload, self.seq_len, self.stride))
        return total

    def __iter__(self):
        sample_seed = self.seed + self.epoch * 9973
        for shard_idx, shard_path in enumerate(self._iter_worker_shards()):
            payload = self._load_shard(shard_path)
            samples = _build_sequence_samples_from_payload(payload, self.seq_len, self.stride)
            if self.shuffle:
                rng = random.Random(sample_seed + shard_idx)
                rng.shuffle(samples)
            for sample in samples:
                yield sample


def _list_bc_shards(dataset_dir, max_shards=-1):
    shard_paths = sorted(Path(dataset_dir).glob("map_*.pt"))
    if max_shards is not None and int(max_shards) > 0:
        shard_paths = shard_paths[: int(max_shards)]
    return [str(path) for path in shard_paths]


def _split_shards(shard_paths, val_fraction, seed):
    shard_paths = list(shard_paths)
    rng = random.Random(int(seed))
    rng.shuffle(shard_paths)
    if not shard_paths:
        return [], []
    if val_fraction <= 0:
        return shard_paths, []
    val_count = int(round(len(shard_paths) * float(val_fraction)))
    if len(shard_paths) > 1:
        val_count = max(1, min(len(shard_paths) - 1, val_count))
    else:
        val_count = 0
    if val_count == 0:
        return shard_paths, []
    return shard_paths[val_count:], shard_paths[:val_count]


def _validate_bc_shard_payload(payload, obs_dim, action_space_size, shard_path):
    required = {"obs", "action", "map_id", "scenario_id", "agent_id", "timestep"}
    missing = required.difference(payload.keys())
    if missing:
        message = f"BC shard {shard_path} is missing required keys: {sorted(missing)}"
        _print_mismatch(message)
        raise ValueError(message)

    obs = payload["obs"]
    action = payload["action"]
    if obs.ndim != 2:
        message = f"BC shard {shard_path} obs must be rank-2, got shape {tuple(obs.shape)}"
        _print_mismatch(message)
        raise ValueError(message)
    if int(obs.shape[1]) != int(obs_dim):
        message = f"BC shard {shard_path} observation width mismatch: expected {obs_dim}, got {int(obs.shape[1])}"
        _print_mismatch(message)
        raise ValueError(message)
    if action.ndim != 1:
        message = f"BC shard {shard_path} action must be rank-1, got shape {tuple(action.shape)}"
        _print_mismatch(message)
        raise ValueError(message)
    if int(action.shape[0]) != int(obs.shape[0]):
        message = f"BC shard {shard_path} obs/action sample count mismatch: {int(obs.shape[0])} vs {int(action.shape[0])}"
        _print_mismatch(message)
        raise ValueError(message)
    if action.numel() > 0:
        min_action = int(action.min().item())
        max_action = int(action.max().item())
        if min_action < 0 or max_action >= int(action_space_size):
            message = (
                f"BC shard {shard_path} contains invalid action ids [{min_action}, {max_action}] "
                f"for action space size {action_space_size}"
            )
            _print_mismatch(message)
            raise ValueError(message)

    metadata = payload.get("metadata", {})
    metadata_action_space = metadata.get("action_space_size")
    if metadata_action_space is not None and int(metadata_action_space) != int(action_space_size):
        message = (
            f"BC shard {shard_path} metadata action_space_size mismatch: expected {action_space_size}, "
            f"got {metadata_action_space}"
        )
        _print_mismatch(message)
        raise ValueError(message)


def _peek_bc_shard(shard_paths, obs_dim, action_space_size):
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu")
        _validate_bc_shard_payload(payload, obs_dim, action_space_size, shard_path)
        if int(payload["action"].shape[0]) > 0:
            return payload
    return None


def _build_sequence_samples_from_payload(payload, seq_len, stride):
    seq_len = int(seq_len)
    stride = max(1, int(stride))
    obs = payload["obs"].float()
    action = payload["action"].long()
    map_id = payload["map_id"].int()
    scenario_id = payload["scenario_id"].int()
    agent_id = payload["agent_id"].int()
    timestep = payload["timestep"].int()

    groups = {}
    for row_idx in range(obs.shape[0]):
        key = (int(map_id[row_idx]), int(scenario_id[row_idx]), int(agent_id[row_idx]))
        groups.setdefault(key, []).append((int(timestep[row_idx]), row_idx))

    samples = []
    for rows in groups.values():
        rows.sort(key=lambda item: item[0])
        ordered_indices = [row_idx for _, row_idx in rows]
        if not ordered_indices:
            continue
        for start_idx in range(0, len(ordered_indices), stride):
            window_indices = ordered_indices[start_idx : start_idx + seq_len]
            if not window_indices:
                continue
            valid_len = len(window_indices)
            obs_window = torch.zeros((seq_len, obs.shape[1]), dtype=torch.float32)
            action_window = torch.zeros((seq_len,), dtype=torch.int64)
            mask_window = torch.zeros((seq_len,), dtype=torch.bool)
            obs_window[:valid_len] = obs[window_indices]
            action_window[:valid_len] = action[window_indices]
            mask_window[:valid_len] = True
            samples.append((obs_window, action_window, mask_window))
    return samples


def _make_bc_loader(dataset, batch_size, shuffle, num_workers):
    if isinstance(dataset, IterableDataset):
        if len(getattr(dataset, "shard_paths", [])) == 0:
            return None
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "pin_memory": torch.cuda.is_available(),
        }
        if int(num_workers) > 0:
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(**loader_kwargs)

    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def _extract_action_logits(logits):
    if isinstance(logits, (tuple, list)):
        if len(logits) != 1:
            raise ValueError("Offline BC trainer expects a single discrete action head")
        return logits[0]
    return logits


def _run_bc_epoch(
    model,
    dataloader,
    optimizer,
    device,
    recurrent,
    desc=None,
    log_interval=0,
    batch_log_fn=None,
):
    if dataloader is None:
        return {"loss": 0.0, "accuracy": 0.0, "samples": 0, "elapsed_sec": 0.0}

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None
    progress = tqdm(
        dataloader,
        desc=desc or ("BC train" if training else "BC val"),
        unit="batch",
        leave=False,
        disable=False,
        dynamic_ncols=True,
        smoothing=0.05,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for batch_idx, batch in enumerate(progress, start=1):
        if recurrent:
            obs, action, mask = batch
            obs = obs.to(device)
            action = action.to(device)
            mask = mask.to(device)
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}
            logits, _ = model(obs, state)
            logits = _extract_action_logits(logits)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = action.reshape(-1)
            flat_mask = mask.reshape(-1)
            if not torch.any(flat_mask):
                continue
            losses = torch.nn.functional.cross_entropy(flat_logits, flat_targets, reduction="none")
            loss = losses[flat_mask].mean()
            predictions = flat_logits.argmax(dim=1)
            batch_correct = (predictions[flat_mask] == flat_targets[flat_mask]).sum().item()
            batch_samples = int(flat_mask.sum().item())
        else:
            obs, action = batch
            obs = obs.to(device)
            action = action.to(device)
            logits, _ = model(obs)
            logits = _extract_action_logits(logits)
            loss = torch.nn.functional.cross_entropy(logits, action)
            predictions = logits.argmax(dim=1)
            batch_correct = (predictions == action).sum().item()
            batch_samples = int(action.numel())

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * batch_samples
        total_correct += int(batch_correct)
        total_samples += batch_samples
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_accuracy = total_correct / total_samples
            progress.set_postfix(
                loss=f"{avg_loss:.3f}",
                acc=f"{avg_accuracy:.3f}",
                seen=f"{total_samples / 1000.0:.1f}k",
                refresh=False,
            )
            if log_interval and batch_idx % int(log_interval) == 0:
                progress_fraction = None
                if total_batches is not None and total_batches > 0:
                    progress_text = f"{batch_idx}/{total_batches} ({100.0 * batch_idx / total_batches:.1f}%)"
                    progress_fraction = batch_idx / total_batches
                else:
                    progress_text = str(batch_idx)
                print(
                    f"[BC] {progress.desc} batch={progress_text} loss={avg_loss:.4f} "
                    f"acc={avg_accuracy:.4f} seen={total_samples} elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )
                if batch_log_fn is not None:
                    batch_log_fn(
                        {
                            "batch": int(batch_idx),
                            "progress": progress_fraction,
                            "loss": float(avg_loss),
                            "accuracy": float(avg_accuracy),
                            "samples": int(total_samples),
                            "elapsed_sec": float(time.time() - start_time),
                        }
                    )

    progress.close()

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "samples": 0, "elapsed_sec": time.time() - start_time}
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
        "elapsed_sec": time.time() - start_time,
    }


def _build_bc_policy(args, env, device):
    policy_name = _resolve_base_arg(args, "policy_name")
    if policy_name is None:
        _print_mismatch("BC trainer could not resolve policy_name from config args")
        raise KeyError("policy_name")
    policy_cls = getattr(ocean_torch, policy_name)
    policy = policy_cls(env, **args["policy"])
    rnn_name = _normalize_optional_name(_resolve_base_arg(args, "rnn_name"))
    if rnn_name is not None:
        rnn_cls = getattr(ocean_torch, rnn_name)
        policy = rnn_cls(env, policy, **args["rnn"])
    return policy.to(device)


def _make_bc_training_env(env_cfg):
    env_kwargs = dict(env_cfg)
    env_kwargs["num_agents"] = 1
    env_kwargs["num_maps"] = 1
    env_kwargs["max_controlled_agents"] = 1
    env_kwargs["render_mode"] = None
    return Drive(**env_kwargs)


def train_bc_policy(args=None, dataset_dir=None, output_dir=None, logger=None):
    args = args or load_drive_builder_config()
    env_cfg = _normalize_env_config(args["env"])
    if env_cfg.get("action_type") != "discrete":
        message = "Offline BC trainer currently supports only discrete action_type"
        _print_mismatch(message)
        raise ValueError(message)

    bc_train_cfg = _resolve_bc_train_config(args, dataset_dir=dataset_dir, output_dir=output_dir)
    dataset_dir = bc_train_cfg["dataset_dir"]
    if dataset_dir is None or not os.path.isdir(dataset_dir):
        message = f"BC dataset directory not found: {dataset_dir}"
        _print_mismatch(message)
        raise FileNotFoundError(message)

    seed = int(args.get("train", {}).get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if logger is None:
        from pufferlib.pufferl import NeptuneLogger, WandbLogger

        if args.get("neptune"):
            logger = NeptuneLogger(args)
        elif args.get("wandb"):
            logger = WandbLogger(args)

    device = torch.device(bc_train_cfg["device"])
    output_path = Path(bc_train_cfg["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    print(
        f"[BC] starting training dataset_dir={dataset_dir} output_dir={output_path} "
        f"device={device} epochs={int(bc_train_cfg['epochs'])} batch_size={int(bc_train_cfg['batch_size'])}",
        flush=True,
    )

    env = _make_bc_training_env(env_cfg)
    try:
        obs_dim = int(env.single_observation_space.shape[0])
        action_space_size = _get_discrete_action_size(env)
        shard_paths = _list_bc_shards(dataset_dir, max_shards=bc_train_cfg["max_shards"])
        if not shard_paths:
            message = f"No BC shard files found in {dataset_dir}"
            _print_mismatch(message)
            raise FileNotFoundError(message)

        train_shards, val_shards = _split_shards(shard_paths, bc_train_cfg["val_fraction"], seed)
        print(
            f"[BC] found {len(shard_paths)} shards total: train={len(train_shards)} val={len(val_shards)}",
            flush=True,
        )
        print("[BC] using shard-streamed loading; training starts without full dataset preload", flush=True)

        first_train_payload = _peek_bc_shard(train_shards, obs_dim, action_space_size)

        recurrent = _normalize_optional_name(_resolve_base_arg(args, "rnn_name")) is not None
        if recurrent:
            train_dataset = _SequenceBCDataset(
                train_shards,
                obs_dim,
                action_space_size,
                seq_len=bc_train_cfg["seq_len"],
                stride=bc_train_cfg["sequence_stride"],
                shuffle=True,
                seed=seed,
            )
            val_dataset = _SequenceBCDataset(
                val_shards,
                obs_dim,
                action_space_size,
                seq_len=bc_train_cfg["seq_len"],
                stride=bc_train_cfg["sequence_stride"],
                shuffle=False,
                seed=seed,
            )
        else:
            train_dataset = _FlatBCDataset(train_shards, obs_dim, action_space_size, shuffle=True, seed=seed)
            val_dataset = _FlatBCDataset(val_shards, obs_dim, action_space_size, shuffle=False, seed=seed)

        if first_train_payload is None or int(first_train_payload["action"].shape[0]) == 0:
            message = "BC training dataset is empty after loading selected shards"
            _print_mismatch(message)
            raise ValueError(message)
        print(
            f"[BC] dataset ready recurrent={recurrent} train_shards={len(train_shards)} "
            f"val_shards={len(val_shards)} obs_dim={obs_dim} action_space={action_space_size}",
            flush=True,
        )

        train_loader = _make_bc_loader(
            train_dataset, batch_size=bc_train_cfg["batch_size"], shuffle=True, num_workers=bc_train_cfg["num_workers"]
        )
        val_loader = _make_bc_loader(
            val_dataset, batch_size=bc_train_cfg["batch_size"], shuffle=False, num_workers=bc_train_cfg["num_workers"]
        )

        policy = _build_bc_policy(args, env, device)
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=float(bc_train_cfg["learning_rate"]),
            weight_decay=float(bc_train_cfg["weight_decay"]),
        )

        history = []
        best_metric = None
        latest_path = output_path / "latest.pt"
        best_path = output_path / "best.pt"
        train_step = 0

        for epoch in range(int(bc_train_cfg["epochs"])):
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)
            if hasattr(val_dataset, "set_epoch"):
                val_dataset.set_epoch(epoch)
            print(
                f"[BC] epoch {epoch + 1}/{int(bc_train_cfg['epochs'])} starting",
                flush=True,
            )
            epoch_start_step = train_step

            def _log_train_batch(batch_metrics):
                if logger is None:
                    return
                batch_step = epoch_start_step + int(batch_metrics["samples"])
                batch_logs = {
                    "bc/train_loss_running": float(batch_metrics["loss"]),
                    "bc/train_accuracy_running": float(batch_metrics["accuracy"]),
                    "bc/train_batches_done": int(batch_metrics["batch"]),
                    "bc/train_samples_seen_epoch": int(batch_metrics["samples"]),
                    "bc/train_elapsed_sec_running": float(batch_metrics["elapsed_sec"]),
                }
                if batch_metrics["progress"] is not None:
                    batch_logs["bc/train_epoch_progress"] = float(batch_metrics["progress"])
                logger.log(batch_logs, batch_step)

            train_metrics = _run_bc_epoch(
                policy,
                train_loader,
                optimizer,
                device,
                recurrent=recurrent,
                desc=f"BC train epoch {epoch + 1}/{int(bc_train_cfg['epochs'])}",
                log_interval=bc_train_cfg["log_interval"],
                batch_log_fn=_log_train_batch,
            )
            with torch.no_grad():
                val_metrics = _run_bc_epoch(
                    policy,
                    val_loader,
                    None,
                    device,
                    recurrent=recurrent,
                    desc=f"BC val epoch {epoch + 1}/{int(bc_train_cfg['epochs'])}",
                    log_interval=0,
                )

            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_samples": int(train_metrics["samples"]),
                "train_elapsed_sec": float(train_metrics["elapsed_sec"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_samples": int(val_metrics["samples"]),
                "val_elapsed_sec": float(val_metrics["elapsed_sec"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(epoch_metrics)
            torch.save(policy.state_dict(), latest_path)

            selection_metric = epoch_metrics["val_loss"] if val_metrics["samples"] > 0 else epoch_metrics["train_loss"]
            if best_metric is None or selection_metric < best_metric:
                best_metric = selection_metric
                if _as_bool(bc_train_cfg["save_best"]):
                    torch.save(policy.state_dict(), best_path)

            train_step += epoch_metrics["train_samples"]
            if logger is not None:
                logger.log(
                    {
                        "bc/epoch": epoch_metrics["epoch"],
                        "bc/train_loss": epoch_metrics["train_loss"],
                        "bc/train_accuracy": epoch_metrics["train_accuracy"],
                        "bc/train_samples": epoch_metrics["train_samples"],
                        "bc/train_elapsed_sec": epoch_metrics["train_elapsed_sec"],
                        "bc/val_loss": epoch_metrics["val_loss"],
                        "bc/val_accuracy": epoch_metrics["val_accuracy"],
                        "bc/val_samples": epoch_metrics["val_samples"],
                        "bc/val_elapsed_sec": epoch_metrics["val_elapsed_sec"],
                        "bc/learning_rate": epoch_metrics["learning_rate"],
                        "bc/best_metric": float(best_metric),
                    },
                    train_step,
                )

            print(
                f"[BC] epoch={epoch_metrics['epoch']} train_loss={epoch_metrics['train_loss']:.4f} "
                f"train_acc={epoch_metrics['train_accuracy']:.4f} val_loss={epoch_metrics['val_loss']:.4f} "
                f"val_acc={epoch_metrics['val_accuracy']:.4f} train_sec={epoch_metrics['train_elapsed_sec']:.1f} "
                f"val_sec={epoch_metrics['val_elapsed_sec']:.1f}"
            )

        metadata = {
            "config": args,
            "bc_train": bc_train_cfg,
            "history": history,
            "train_shards": train_shards,
            "val_shards": val_shards,
            "observation_dim": obs_dim,
            "action_space_size": action_space_size,
            "recurrent": recurrent,
            "latest_path": str(latest_path),
            "best_path": str(best_path if best_path.exists() else latest_path),
        }
        metadata_path = output_path / "metrics.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        if logger is not None:
            logger.close(str(best_path if best_path.exists() else latest_path))

        return {
            "latest_path": str(latest_path),
            "best_path": str(best_path if best_path.exists() else latest_path),
            "metadata_path": str(metadata_path),
            "history": history,
            "train_shards": train_shards,
            "val_shards": val_shards,
        }
    finally:
        env.close()


def _create_builder_env_buffers(dynamics_model, max_agents):
    sim_obs_dim = _sim_obs_dim_for_dynamics(dynamics_model)
    observations = np.zeros((max_agents, sim_obs_dim), dtype=np.float32)
    actions = np.zeros(max_agents, dtype=np.int32)
    rewards = np.zeros(max_agents, dtype=np.float32)
    terminals = np.zeros(max_agents, dtype=np.uint8)
    truncations = np.zeros(max_agents, dtype=np.uint8)
    return observations, actions, rewards, terminals, truncations, sim_obs_dim


def _build_policy_observation_batch(env_cfg, active_count, sim_observations, env_handle):
    if env_cfg["observation_mode"] != "sdc_only_with_trailer":
        return sim_observations[:active_count].copy()

    base_ego = {"classic": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}[env_cfg["dynamics_model"]]
    partner_count = binding.MAX_AGENTS - 1
    base_partner = binding.PARTNER_FEATURES
    aug_ego = base_ego + 1 + _EGO_TRAILER_STATE_FEATURES
    aug_partner = base_partner + 1
    road_count = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
    road_features = binding.ROAD_FEATURES
    num_obs = aug_ego + partner_count * aug_partner + road_count * road_features

    policy_observations = np.zeros((active_count, num_obs), dtype=np.float32)
    ego_types = np.zeros(active_count, dtype=np.int32)
    partner_types = np.zeros((active_count, partner_count), dtype=np.int32)
    trailer_features = {
        "rel_x": np.zeros(active_count, dtype=np.float32),
        "rel_y": np.zeros(active_count, dtype=np.float32),
        "rel_heading_x": np.zeros(active_count, dtype=np.float32),
        "rel_heading_y": np.zeros(active_count, dtype=np.float32),
    }

    binding.get_global_agent_types(env_handle, ego_types)
    binding.env_get_partner_types(env_handle, partner_types)
    binding.env_get_ego_trailer_obs_features(
        env_handle,
        trailer_features["rel_x"],
        trailer_features["rel_y"],
        trailer_features["rel_heading_x"],
        trailer_features["rel_heading_y"],
    )
    _postprocess_policy_observations(
        policy_observations,
        sim_observations[:active_count],
        base_ego,
        base_partner,
        partner_count,
        aug_ego,
        aug_partner,
        road_count,
        road_features,
        binding.POLICY_TYPE_CLASS_COUNT,
        ego_types,
        partner_types,
        trailer_features,
    )
    return policy_observations


def _list_map_ids(map_dir, max_maps=-1):
    map_paths = sorted(Path(map_dir).glob("map_*.bin"))
    map_ids = []
    for path in map_paths:
        try:
            map_ids.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    if max_maps is not None and max_maps > 0:
        map_ids = map_ids[:max_maps]
    return map_ids


def _validate_bc_builder_support(env_cfg):
    if env_cfg["action_type"] != "discrete":
        message = (
            "offline BC dataset builder currently supports only discrete action_type "
            f"(got {env_cfg['action_type']!r})"
        )
        _print_mismatch(message)
        raise ValueError("Unsupported action_type for offline BC dataset builder")
    if env_cfg["dynamics_model"] != "classic":
        message = (
            "offline BC dataset builder currently supports only classic dynamics_model "
            f"(got {env_cfg['dynamics_model']!r})"
        )
        _print_mismatch(message)
        raise ValueError("Unsupported dynamics_model for offline BC dataset builder")


def build_bc_dataset(args=None, output_dir=None):
    args = args or load_drive_builder_config()
    env_cfg = _normalize_env_config(args["env"])
    bc_cfg = _resolve_bc_config(args, output_dir=output_dir)
    _validate_bc_builder_support(env_cfg)

    map_dir = env_cfg["map_dir"]
    if not os.path.isdir(map_dir):
        message = f"Map directory not found: {map_dir}"
        _print_mismatch(message)
        raise FileNotFoundError(message)

    shard_dir = Path(bc_cfg["output_dir"])
    shard_dir.mkdir(parents=True, exist_ok=True)
    map_ids = _list_map_ids(map_dir, max_maps=bc_cfg["max_maps"])
    max_agents = int(env_cfg.get("max_controlled_agents", -1))
    if max_agents <= 0:
        max_agents = binding.MAX_AGENTS
    max_agents = min(max_agents, binding.MAX_AGENTS)

    shard_paths = []
    for map_id in tqdm(map_ids, desc="Building BC shards", unit="map"):
        shard_path = shard_dir / f"map_{map_id:03d}.pt"
        if bc_cfg["skip_existing_shards"] and shard_path.exists():
            shard_paths.append(str(shard_path))
            continue

        obs_buf, act_buf, rew_buf, term_buf, trunc_buf, sim_obs_dim = _create_builder_env_buffers(
            env_cfg["dynamics_model"], max_agents
        )
        env_handle = binding.env_init(
            obs_buf,
            act_buf,
            rew_buf,
            term_buf,
            trunc_buf,
            0,
            human_agent_idx=0,
            observation_mode=0 if env_cfg["observation_mode"] == "default" else 1,
            reward_vehicle_collision=env_cfg["reward_vehicle_collision"],
            reward_offroad_collision=env_cfg["reward_offroad_collision"],
            reward_goal=env_cfg["reward_goal"],
            reward_goal_post_respawn=env_cfg["reward_goal_post_respawn"],
            goal_radius=env_cfg["goal_radius"],
            goal_speed=env_cfg["goal_speed"],
            goal_behavior=env_cfg["goal_behavior"],
            goal_target_distance=env_cfg["goal_target_distance"],
            collision_behavior=env_cfg["collision_behavior"],
            offroad_behavior=env_cfg["offroad_behavior"],
            dt=env_cfg["dt"],
            episode_length=env_cfg["episode_length"],
            termination_mode=env_cfg["termination_mode"],
            max_controlled_agents=max_agents,
            map_id=map_id,
            max_agents=max_agents,
            ini_file="pufferlib/config/ocean/drive.ini",
            init_steps=env_cfg["init_steps"],
                init_mode=0 if env_cfg["init_mode"] == "create_all_valid" else 1,
                control_mode={
                    "control_vehicles": 0,
                    "control_agents": 1,
                    "control_wosac": 2,
                    "control_sdc_only": 3,
                }[env_cfg["control_mode"]],
                map_dir=map_dir,
                extend_classic_action_space=int(_as_bool(env_cfg.get("extend_classic_action_space", True))),
                non_kinematic_vehicle_params_override=None,
                force_zero_trailer_articulation_at_init=int(_as_bool(env_cfg.get("force_zero_trailer_articulation_at_init", False))),
            )
        _compare_c_python_env_config(
            env_handle,
            _expected_c_env_config(
                action_type=env_cfg["action_type"],
                dynamics_model=env_cfg["dynamics_model"],
                observation_mode=env_cfg.get("observation_mode", "default"),
                extend_classic_action_space=env_cfg.get("extend_classic_action_space", True),
                reward_vehicle_collision=env_cfg["reward_vehicle_collision"],
                reward_offroad_collision=env_cfg["reward_offroad_collision"],
                reward_goal=env_cfg["reward_goal"],
                reward_goal_post_respawn=env_cfg["reward_goal_post_respawn"],
                goal_radius=env_cfg["goal_radius"],
                goal_speed=env_cfg["goal_speed"],
                goal_behavior=env_cfg["goal_behavior"],
                goal_target_distance=env_cfg["goal_target_distance"],
                collision_behavior=env_cfg["collision_behavior"],
                offroad_behavior=env_cfg["offroad_behavior"],
                dt=env_cfg["dt"],
                episode_length=env_cfg["episode_length"],
                termination_mode=env_cfg["termination_mode"],
                init_steps=env_cfg["init_steps"],
                init_mode=env_cfg["init_mode"],
                control_mode=env_cfg["control_mode"],
                max_controlled_agents=max_agents,
            ),
            f"build_bc_dataset map_{map_id:03d}",
        )

        try:
            binding.env_reset(env_handle, 0)
            active_count = binding.env_get_active_agent_count(env_handle)
            if active_count <= 0:
                continue

            scenario_ids = np.zeros(active_count, dtype=np.int32)
            agent_ids = np.zeros(active_count, dtype=np.int32)
            binding.env_get_active_agent_info(env_handle, scenario_ids, agent_ids)

            max_steps = max(0, int(env_cfg["episode_length"]) - int(env_cfg["init_steps"]) - 1)
            agent_actions = np.full((active_count, max_steps), -1, dtype=np.int32)
            agent_step_costs = np.zeros((active_count, max_steps), dtype=np.float32)
            agent_step_lat_costs = np.zeros((active_count, max_steps), dtype=np.float32)
            agent_step_lon_costs = np.zeros((active_count, max_steps), dtype=np.float32)
            agent_num_steps = np.zeros(active_count, dtype=np.int32)
            agent_total_costs = np.zeros(active_count, dtype=np.float32)
            agent_total_lat_costs = np.zeros(active_count, dtype=np.float32)
            agent_total_lon_costs = np.zeros(active_count, dtype=np.float32)

            for agent_slot in range(active_count):
                num_steps, total_cost, total_lat_cost, total_lon_cost = binding.env_fit_discrete_action_sequence(
                    env_handle,
                    agent_slot,
                    int(bc_cfg["beam_width"]),
                    int(bc_cfg["planning_horizon"]),
                    float(bc_cfg["match_weight_lateral"]),
                    float(bc_cfg["match_weight_longitudinal"]),
                float(bc_cfg["match_weight_heading"]),
                float(bc_cfg["match_weight_speed"]),
                float(bc_cfg["match_weight_steer_change"]),
                float(bc_cfg["match_weight_accel_change"]),
                float(bc_cfg["match_weight_reverse"]),
                float(bc_cfg["match_weight_progress"]),
                float(bc_cfg["match_weight_steer_flip"]),
                float(bc_cfg["match_weight_ref_accel"]),
                float(bc_cfg["match_weight_ref_steer"]),
                agent_actions[agent_slot],
                agent_step_costs[agent_slot],
                agent_step_lat_costs[agent_slot],
                    agent_step_lon_costs[agent_slot],
                )
                agent_num_steps[agent_slot] = num_steps
                agent_total_costs[agent_slot] = total_cost
                agent_total_lat_costs[agent_slot] = total_lat_cost
                agent_total_lon_costs[agent_slot] = total_lon_cost

            sample_obs = []
            sample_actions = []
            sample_scenario_ids = []
            sample_agent_ids = []
            sample_timesteps = []
            sample_total_costs = []
            sample_step_costs = []
            sample_total_lat_costs = []
            sample_total_lon_costs = []
            sample_step_lat_costs = []
            sample_step_lon_costs = []
            sample_map_ids = []

            max_num_steps = int(np.max(agent_num_steps)) if active_count > 0 else 0
            timestep_obs = np.zeros((active_count, sim_obs_dim), dtype=np.float32)
            policy_obs_dim = (
                sim_obs_dim
                if env_cfg["observation_mode"] == "default"
                else (
                    {"classic": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}[env_cfg["dynamics_model"]]
                    + 1
                    + _EGO_TRAILER_STATE_FEATURES
                    + (binding.MAX_AGENTS - 1) * (binding.PARTNER_FEATURES + 1)
                    + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
                )
            )
            for step_idx in range(max_num_steps):
                binding.env_set_logged_timestep(env_handle, int(env_cfg["init_steps"]) + step_idx)
                binding.env_copy_observations(env_handle, timestep_obs)
                policy_observations = _build_policy_observation_batch(env_cfg, active_count, timestep_obs, env_handle)

                valid_slots = np.nonzero(agent_num_steps > step_idx)[0]
                for agent_slot in valid_slots:
                    sample_obs.append(policy_observations[agent_slot].copy())
                    sample_actions.append(int(agent_actions[agent_slot, step_idx]))
                    sample_scenario_ids.append(int(scenario_ids[agent_slot]))
                    sample_agent_ids.append(int(agent_ids[agent_slot]))
                    sample_timesteps.append(int(env_cfg["init_steps"]) + step_idx)
                    sample_total_costs.append(float(agent_total_costs[agent_slot]))
                    sample_step_costs.append(float(agent_step_costs[agent_slot, step_idx]))
                    sample_total_lat_costs.append(float(agent_total_lat_costs[agent_slot]))
                    sample_total_lon_costs.append(float(agent_total_lon_costs[agent_slot]))
                    sample_step_lat_costs.append(float(agent_step_lat_costs[agent_slot, step_idx]))
                    sample_step_lon_costs.append(float(agent_step_lon_costs[agent_slot, step_idx]))
                    sample_map_ids.append(int(map_id))

            if sample_obs:
                obs_tensor = torch.from_numpy(np.stack(sample_obs).astype(np.float32))
            else:
                obs_tensor = torch.zeros((0, policy_obs_dim), dtype=torch.float32)

            payload = {
                "obs": obs_tensor,
                "action": torch.tensor(sample_actions, dtype=torch.int64),
                "scenario_id": torch.tensor(sample_scenario_ids, dtype=torch.int32),
                "agent_id": torch.tensor(sample_agent_ids, dtype=torch.int32),
                "timestep": torch.tensor(sample_timesteps, dtype=torch.int32),
                "match_cost_total": torch.tensor(sample_total_costs, dtype=torch.float32),
                "match_cost_step": torch.tensor(sample_step_costs, dtype=torch.float32),
                "match_cost_lateral_total": torch.tensor(sample_total_lat_costs, dtype=torch.float32),
                "match_cost_longitudinal_total": torch.tensor(sample_total_lon_costs, dtype=torch.float32),
                "match_cost_lateral_step": torch.tensor(sample_step_lat_costs, dtype=torch.float32),
                "match_cost_longitudinal_step": torch.tensor(sample_step_lon_costs, dtype=torch.float32),
                "map_id": torch.tensor(sample_map_ids, dtype=torch.int32),
                "metadata": {
                    "source_map": f"map_{map_id:03d}.bin",
                    "sample_count": len(sample_actions),
                    "observation_dim": int(obs_tensor.shape[1]),
                    "observation_mode": env_cfg["observation_mode"],
                    "action_space_size": (
                        _CLASSIC_DISCRETE_ACTIONS
                        if _as_bool(env_cfg.get("extend_classic_action_space", True))
                        else 7 * 13
                    ),
                    "config": {"env": env_cfg, "bc": bc_cfg},
                    "optimizer": {
                        "beam_width": int(bc_cfg["beam_width"]),
                        "planning_horizon": int(bc_cfg["planning_horizon"]),
                        "match_weight_lateral": float(bc_cfg["match_weight_lateral"]),
                        "match_weight_longitudinal": float(bc_cfg["match_weight_longitudinal"]),
                        "match_weight_heading": float(bc_cfg["match_weight_heading"]),
                        "match_weight_speed": float(bc_cfg["match_weight_speed"]),
                        "match_weight_steer_change": float(bc_cfg["match_weight_steer_change"]),
                        "match_weight_accel_change": float(bc_cfg["match_weight_accel_change"]),
                        "match_weight_reverse": float(bc_cfg["match_weight_reverse"]),
                        "match_weight_progress": float(bc_cfg["match_weight_progress"]),
                        "match_weight_steer_flip": float(bc_cfg["match_weight_steer_flip"]),
                        "match_weight_ref_accel": float(bc_cfg["match_weight_ref_accel"]),
                        "match_weight_ref_steer": float(bc_cfg["match_weight_ref_steer"]),
                        "init_steps": int(env_cfg["init_steps"]),
                    },
                },
            }
            torch.save(payload, shard_path)
            shard_paths.append(str(shard_path))
        finally:
            binding.env_close(env_handle)

    return shard_paths


class Drive(pufferlib.PufferEnv):
    def __init__(
        self,
        render_mode=None,
        report_interval=1,
        width=1280,
        height=1024,
        human_agent_idx=0,
        reward_vehicle_collision=-0.1,
        reward_offroad_collision=-0.1,
        reward_goal=1.0,
        reward_goal_post_respawn=0.5,
        goal_behavior=0,
        goal_target_distance=10.0,
        goal_radius=2.0,
        goal_speed=20.0,
        collision_behavior=0,
        offroad_behavior=0,
        dt=0.1,
        episode_length=None,
        termination_mode=None,
        resample_frequency=91,
        num_maps=100,
        num_agents=512,
        action_type="discrete",
        dynamics_model="classic",
        max_controlled_agents=-1,
        buf=None,
        seed=1,
        init_steps=0,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        observation_mode="default",
        extend_classic_action_space=True,
        map_dir="resources/drive/binaries/training",
        sequential_map_sampling=False,
        sdc_runtime_truck_override=False,
        sdc_runtime_truck_ref_bin=None,
        force_truck_params_from_ref_bin=None,
        force_zero_trailer_articulation_at_init=False,
    ):
        # env
        self.dt = dt
        self.render_mode = render_mode
        self.num_maps = num_maps
        self.report_interval = report_interval
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_goal = reward_goal
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.goal_radius = goal_radius
        self.goal_speed = goal_speed
        self.goal_behavior = goal_behavior
        self.goal_target_distance = goal_target_distance
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.human_agent_idx = human_agent_idx
        self.episode_length = episode_length
        self.termination_mode = termination_mode
        self.resample_frequency = resample_frequency
        action_type = _normalize_action_type(action_type)
        self.dynamics_model = _normalize_dynamics_model(dynamics_model)
        init_mode = _normalize_init_mode(init_mode)
        control_mode = _normalize_control_mode(control_mode)
        observation_mode = _normalize_observation_mode(observation_mode)
        self.type_classes = binding.POLICY_TYPE_CLASS_COUNT

        # Observation space calculation
        self._base_ego_features = {"classic": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}.get(
            self.dynamics_model
        )

        # Extract observation shapes from constants
        # These need to be defined in C, since they determine the shape of the arrays
        self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
        self.max_partner_objects = binding.MAX_AGENTS - 1
        self._base_partner_features = binding.PARTNER_FEATURES
        self.road_features = binding.ROAD_FEATURES

        self._sim_num_obs = (
            self._base_ego_features
            + self.max_partner_objects * self._base_partner_features
            + self.max_road_objects * self.road_features
        )

        self.init_steps = init_steps
        self.init_mode_str = init_mode
        self.control_mode_str = control_mode
        self.observation_mode_str = observation_mode
        self.extend_classic_action_space = _as_bool(extend_classic_action_space)
        self.map_dir = map_dir
        self.force_zero_trailer_articulation_at_init = _as_bool(force_zero_trailer_articulation_at_init)
        self.non_kinematic_vehicle_params_override = None
        if isinstance(sdc_runtime_truck_ref_bin, str):
            stripped_ref = sdc_runtime_truck_ref_bin.strip()
            if stripped_ref == "" or stripped_ref.lower() == "none":
                sdc_runtime_truck_ref_bin = None
        if _as_bool(sdc_runtime_truck_override):
            self.force_zero_trailer_articulation_at_init = True
            if force_truck_params_from_ref_bin is None:
                force_truck_params_from_ref_bin = (
                    sdc_runtime_truck_ref_bin
                    if sdc_runtime_truck_ref_bin is not None
                    else DEFAULT_SDC_RUNTIME_TRUCK_REF_BIN
                )
        if force_truck_params_from_ref_bin is not None:
            reference_bin = os.path.abspath(force_truck_params_from_ref_bin)
            if not os.path.exists(reference_bin):
                message = f"Truck reference artifact not found: {reference_bin}"
                _print_mismatch(message)
                raise FileNotFoundError(message)
            self.non_kinematic_vehicle_params_override = _load_non_kinematic_vehicle_params_from_bin(reference_bin)

        if self.control_mode_str == "control_vehicles":
            self.control_mode = 0
        elif self.control_mode_str == "control_agents":
            self.control_mode = 1
        elif self.control_mode_str == "control_wosac":
            self.control_mode = 2
        elif self.control_mode_str == "control_sdc_only":
            self.control_mode = 3
        else:
            message = (
                "control_mode must be one of 'control_vehicles', 'control_wosac', or 'control_agents'. "
                f"Got: {self.control_mode_str}"
            )
            _print_mismatch(message)
            raise ValueError(message)
        if self.observation_mode_str == "default":
            self.observation_mode = 0
        elif self.observation_mode_str == "sdc_only_with_trailer":
            self.observation_mode = 1
        else:
            message = (
                "observation_mode must be one of 'default' or 'sdc_only_with_trailer'. "
                f"Got: {self.observation_mode_str}"
            )
            _print_mismatch(message)
            raise ValueError(message)
        if self.observation_mode == 0:
            self.ego_features = self._base_ego_features
            self.partner_features = self._base_partner_features
        else:
            self.ego_features = self._base_ego_features + 1 + _EGO_TRAILER_STATE_FEATURES
            self.partner_features = self._base_partner_features + 1
        self.num_obs = (
            self.ego_features
            + self.max_partner_objects * self.partner_features
            + self.max_road_objects * self.road_features
        )
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)
        if self.init_mode_str == "create_all_valid":
            self.init_mode = 0
        elif self.init_mode_str == "create_only_controlled":
            self.init_mode = 1
        else:
            message = (
                f"init_mode must be one of 'create_all_valid' or 'create_only_controlled'. Got: {self.init_mode_str}"
            )
            _print_mismatch(message)
            raise ValueError(message)

        if action_type == "discrete":
            if self.dynamics_model == "classic":
                # Joint action space (assume dependence)
                action_count = _CLASSIC_DISCRETE_ACTIONS if self.extend_classic_action_space else (7 * 13)
                self.single_action_space = gymnasium.spaces.MultiDiscrete([action_count])
                # Multi discrete (assume independence)
                # self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
            elif self.dynamics_model == "jerk":
                # Joint action space (assume dependence) - 4 longitudinal × 3 lateral = 12
                self.single_action_space = gymnasium.spaces.MultiDiscrete([4 * 3])
            else:
                message = f"dynamics_model must be 'classic' or 'jerk'. Got: {self.dynamics_model}"
                _print_mismatch(message)
                raise ValueError(message)
        elif action_type == "continuous":
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            message = f"action_space must be 'discrete' or 'continuous'. Got: {action_type}"
            _print_mismatch(message)
            raise ValueError(message)

        self._action_type_flag = 0 if action_type == "discrete" else 1

        # Check if resources directory exists
        binary_path = f"{map_dir}/map_000.bin"
        if not os.path.exists(binary_path):
            message = (
                f"Required directory {binary_path} not found. Please ensure the Drive maps are downloaded "
                "and installed correctly per docs."
            )
            _print_mismatch(message)
            raise FileNotFoundError(message)

        # Check maps availability
        available_maps = len([name for name in os.listdir(map_dir) if name.endswith(".bin")])
        if num_maps > available_maps:
            message = (
                f"num_maps ({num_maps}) exceeds available maps in directory ({available_maps}). "
                "Please reduce num_maps or add more maps to resources/drive/binaries."
            )
            _print_mismatch(message)
            raise ValueError(message)
        self.max_controlled_agents = int(max_controlled_agents)

        # Iterate through all maps to count total agents that can be initialized for each map
        agent_offsets, map_ids, num_envs = binding.shared(
            map_dir=map_dir,
            num_agents=num_agents,
            num_maps=num_maps,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
            goal_behavior=self.goal_behavior,
            goal_target_distance=self.goal_target_distance,
            sequential_map_sampling=sequential_map_sampling,
            non_kinematic_vehicle_params_override=self.non_kinematic_vehicle_params_override,
            force_zero_trailer_articulation_at_init=int(self.force_zero_trailer_articulation_at_init),
        )

        # agent_offsets[-1] works in both cases, just making it explicit that num_agents is ignored if sequential_map_sampling is True
        self.num_agents = num_agents if not sequential_map_sampling else agent_offsets[-1]
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        super().__init__(buf=buf)
        self._sim_observations = self.observations
        if self.observation_mode == 1:
            self._sim_observations = np.zeros((self.num_agents, self._sim_num_obs), dtype=np.float32)
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
            env_id = binding.env_init(
                self._sim_observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                action_type=self._action_type_flag,
                human_agent_idx=human_agent_idx,
                observation_mode=self.observation_mode,
                reward_vehicle_collision=reward_vehicle_collision,
                reward_offroad_collision=reward_offroad_collision,
                reward_goal=reward_goal,
                reward_goal_post_respawn=reward_goal_post_respawn,
                goal_radius=goal_radius,
                goal_speed=goal_speed,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=dt,
                episode_length=(int(episode_length) if episode_length is not None else None),
                termination_mode=(int(self.termination_mode) if self.termination_mode is not None else 0),
                max_controlled_agents=self.max_controlled_agents,
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file="pufferlib/config/ocean/drive.ini",
                init_steps=init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=map_dir,
                extend_classic_action_space=int(self.extend_classic_action_space),
                non_kinematic_vehicle_params_override=self.non_kinematic_vehicle_params_override,
                force_zero_trailer_articulation_at_init=int(self.force_zero_trailer_articulation_at_init),
            )
            _compare_c_python_env_config(
                env_id,
                _expected_c_env_config(
                    action_type=action_type,
                    dynamics_model=dynamics_model,
                    observation_mode=observation_mode,
                    extend_classic_action_space=self.extend_classic_action_space,
                    reward_vehicle_collision=reward_vehicle_collision,
                    reward_offroad_collision=reward_offroad_collision,
                    reward_goal=reward_goal,
                    reward_goal_post_respawn=reward_goal_post_respawn,
                    goal_radius=goal_radius,
                    goal_speed=goal_speed,
                    goal_behavior=self.goal_behavior,
                    goal_target_distance=self.goal_target_distance,
                    collision_behavior=self.collision_behavior,
                    offroad_behavior=self.offroad_behavior,
                    dt=dt,
                    episode_length=episode_length,
                    termination_mode=self.termination_mode,
                    init_steps=init_steps,
                    init_mode=init_mode,
                    control_mode=control_mode,
                    max_controlled_agents=self.max_controlled_agents,
                ),
                f"Drive.__init__ env_index={i} map_id={map_ids[i]}",
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

    def _resample_vector_envs(self, seed):
        binding.vec_close(self.c_envs)
        agent_offsets, map_ids, num_envs = binding.shared(
            num_agents=self.num_agents,
            num_maps=self.num_maps,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
            goal_behavior=self.goal_behavior,
            goal_target_distance=self.goal_target_distance,
            goal_speed=self.goal_speed,
            map_dir=self.map_dir,
            sequential_map_sampling=False,
            non_kinematic_vehicle_params_override=self.non_kinematic_vehicle_params_override,
            force_zero_trailer_articulation_at_init=int(self.force_zero_trailer_articulation_at_init),
        )
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
            env_id = binding.env_init(
                self._sim_observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                action_type=self._action_type_flag,
                human_agent_idx=self.human_agent_idx,
                observation_mode=self.observation_mode,
                reward_vehicle_collision=self.reward_vehicle_collision,
                reward_offroad_collision=self.reward_offroad_collision,
                reward_goal=self.reward_goal,
                reward_goal_post_respawn=self.reward_goal_post_respawn,
                goal_radius=self.goal_radius,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                goal_speed=self.goal_speed,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=self.dt,
                episode_length=(int(self.episode_length) if self.episode_length is not None else None),
                max_controlled_agents=self.max_controlled_agents,
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file="pufferlib/config/ocean/drive.ini",
                init_steps=self.init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=self.map_dir,
                extend_classic_action_space=int(self.extend_classic_action_space),
                non_kinematic_vehicle_params_override=self.non_kinematic_vehicle_params_override,
                force_zero_trailer_articulation_at_init=int(self.force_zero_trailer_articulation_at_init),
            )
            _compare_c_python_env_config(
                env_id,
                _expected_c_env_config(
                    action_type="discrete" if self._action_type_flag == 0 else "continuous",
                    dynamics_model=self.dynamics_model,
                    observation_mode=self.observation_mode_str,
                    extend_classic_action_space=self.extend_classic_action_space,
                    reward_vehicle_collision=self.reward_vehicle_collision,
                    reward_offroad_collision=self.reward_offroad_collision,
                    reward_goal=self.reward_goal,
                    reward_goal_post_respawn=self.reward_goal_post_respawn,
                    goal_radius=self.goal_radius,
                    goal_speed=self.goal_speed,
                    goal_behavior=self.goal_behavior,
                    goal_target_distance=self.goal_target_distance,
                    collision_behavior=self.collision_behavior,
                    offroad_behavior=self.offroad_behavior,
                    dt=self.dt,
                    episode_length=self.episode_length,
                    termination_mode=self.termination_mode,
                    init_steps=self.init_steps,
                    init_mode=self.init_mode_str,
                    control_mode=self.control_mode_str,
                    max_controlled_agents=self.max_controlled_agents,
                ),
                f"Drive._resample_vector_envs env_index={i} map_id={map_ids[i]}",
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)
        binding.vec_reset(self.c_envs, seed)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        max_resample_attempts = 8
        attempts = 0
        while binding.vec_has_invalid_initial_trailer_state(self.c_envs):
            attempts += 1
            if attempts > max_resample_attempts:
                raise RuntimeError(
                    f"Exceeded {max_resample_attempts} resample attempts while rejecting invalid initial trailer states."
                )
            self._resample_vector_envs(np.random.randint(0, 2**32 - 1))
        self.tick = 0
        self._postprocess_observations()
        return self.observations, []

    def step(self, actions):
        self.terminals[:] = 0
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs, self.num_agents)
            if log:
                info.append(log)
                # print(log)
        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            self.tick = 0
            seed = np.random.randint(0, 2**32 - 1)
            self._resample_vector_envs(seed)
            self.terminals[:] = 1
        self._postprocess_observations()
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def get_global_agent_state(self, include_sdc_trailer=False, include_types=False):
        """Get current global state of all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'id', 'length', 'width' containing numpy arrays
            of shape (num_active_agents,)
        """
        num_agents = self.num_agents

        states = {
            "x": np.zeros(num_agents, dtype=np.float32),
            "y": np.zeros(num_agents, dtype=np.float32),
            "z": np.zeros(num_agents, dtype=np.float32),
            "heading": np.zeros(num_agents, dtype=np.float32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "length": np.zeros(num_agents, dtype=np.float32),
            "width": np.zeros(num_agents, dtype=np.float32),
        }

        binding.vec_get_global_agent_state(
            self.c_envs,
            states["x"],
            states["y"],
            states["z"],
            states["heading"],
            states["id"],
            states["length"],
            states["width"],
        )

        if include_sdc_trailer:
            states["sdc_trailer"] = self.get_sdc_trailer_state()
        if include_types:
            states["type"] = self.get_global_agent_types()

        return states

    def get_global_agent_types(self):
        """Get current type id for all active agents."""
        types = np.zeros(self.num_agents, dtype=np.int32)
        binding.vec_get_global_agent_types(self.c_envs, types)
        return types

    def get_partner_types(self):
        """Get partner type ids in the same ordering as partner observations."""
        types = np.zeros((self.num_agents, self.max_partner_objects), dtype=np.int32)
        binding.vec_get_partner_types(self.c_envs, types)
        return types

    def get_sdc_trailer_state(self):
        trailer = {
            "has_trailer": np.zeros(self.num_envs, dtype=np.int32),
            "x": np.zeros(self.num_envs, dtype=np.float32),
            "y": np.zeros(self.num_envs, dtype=np.float32),
            "z": np.zeros(self.num_envs, dtype=np.float32),
            "heading": np.zeros(self.num_envs, dtype=np.float32),
            "id": np.zeros(self.num_envs, dtype=np.int32),
            "length": np.zeros(self.num_envs, dtype=np.float32),
            "width": np.zeros(self.num_envs, dtype=np.float32),
        }

        binding.vec_get_sdc_trailer_state(
            self.c_envs,
            trailer["has_trailer"],
            trailer["x"],
            trailer["y"],
            trailer["z"],
            trailer["heading"],
            trailer["id"],
            trailer["length"],
            trailer["width"],
        )
        return trailer

    def get_ego_trailer_obs_features(self):
        trailer_features = {
            "rel_x": np.zeros(self.num_agents, dtype=np.float32),
            "rel_y": np.zeros(self.num_agents, dtype=np.float32),
            "rel_heading_x": np.zeros(self.num_agents, dtype=np.float32),
            "rel_heading_y": np.zeros(self.num_agents, dtype=np.float32),
        }

        binding.vec_get_ego_trailer_obs_features(
            self.c_envs,
            trailer_features["rel_x"],
            trailer_features["rel_y"],
            trailer_features["rel_heading_x"],
            trailer_features["rel_heading_y"],
        )
        return trailer_features

    def _postprocess_observations(self):
        if getattr(self, "observation_mode", 0) != 1:
            return

        base_ego = self._base_ego_features
        base_partner = self._base_partner_features
        partner_count = self.max_partner_objects

        aug_ego = self.ego_features
        aug_partner = self.partner_features

        policy_type_max = self.type_classes - 1
        ego_types = np.clip(self.get_global_agent_types(), _POLICY_TYPE_PADDED, policy_type_max).astype(np.float32)
        partner_types = np.clip(self.get_partner_types(), _POLICY_TYPE_PADDED, policy_type_max).astype(np.float32)
        ego_trailer_features = self.get_ego_trailer_obs_features()
        _postprocess_policy_observations(
            self.observations,
            self._sim_observations,
            base_ego,
            base_partner,
            partner_count,
            aug_ego,
            aug_partner,
            self.max_road_objects,
            self.road_features,
            self.type_classes,
            ego_types,
            partner_types,
            ego_trailer_features,
        )

    def get_ground_truth_trajectories(self):
        """Get ground truth trajectories for all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'valid', 'id', 'scenario_id' containing numpy arrays.
        """
        num_agents = self.num_agents

        trajectories = {
            "x": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "z": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "valid": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.int32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "is_vehicle": np.zeros(num_agents, dtype=np.int32),
            "scenario_id": np.zeros(num_agents, dtype=np.int32),
        }

        binding.vec_get_global_ground_truth_trajectories(
            self.c_envs,
            trajectories["x"],
            trajectories["y"],
            trajectories["z"],
            trajectories["heading"],
            trajectories["valid"],
            trajectories["id"],
            trajectories["is_vehicle"],
            trajectories["scenario_id"],
        )

        for key in trajectories:
            trajectories[key] = trajectories[key][:, None]

        return trajectories

    def get_road_edge_polylines(self):
        """Get road edge polylines for all scenarios.

        Returns:
            dict with keys 'x', 'y', 'lengths', 'scenario_id' containing numpy arrays.
            x, y are flattened point coordinates; lengths indicates points per polyline.
        """
        num_polylines, total_points = binding.vec_get_road_edge_counts(self.c_envs)

        polylines = {
            "x": np.zeros(total_points, dtype=np.float32),
            "y": np.zeros(total_points, dtype=np.float32),
            "lengths": np.zeros(num_polylines, dtype=np.int32),
            "scenario_id": np.zeros(num_polylines, dtype=np.int32),
        }

        binding.vec_get_road_edge_polylines(
            self.c_envs,
            polylines["x"],
            polylines["y"],
            polylines["lengths"],
            polylines["scenario_id"],
        )

        return polylines

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"]) - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))


def dist(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return dx * dx + dy * dy


def simplify_polyline(geometry, polyline_reduction_threshold, max_segment_length):
    """Simplify the given polyline using a method inspired by Visvalingham-Whyatt, optimized for Python."""
    num_points = len(geometry)
    if num_points < 3:
        return geometry  # Not enough points to simplify

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)
            if area < polyline_reduction_threshold and dist(point1, point3) <= max_segment_length:
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


def save_map_binary(map_data, output_file, unique_map_id):
    trajectory_length = 91
    """Saves map data in a binary format readable by C"""
    with open(output_file, "wb") as f:
        extension_magic = 0x54524C52
        extension_version = 2

        def stable_track_hash(value):
            if value is None:
                return 0
            digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
            return struct.unpack("<Q", digest)[0]

        def normalize_int32_id(value):
            int32_min = -(2**31)
            int32_max = 2**31 - 1

            if isinstance(value, bool):
                value = int(value)

            try:
                intval = int(value)
                if int32_min <= intval <= int32_max:
                    return intval
                return ((intval + 2**31) % 2**32) - 2**31
            except (TypeError, ValueError):
                digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).digest()
                uint32 = struct.unpack("<I", digest)[0]
                return uint32 - 2**32 if uint32 >= 2**31 else uint32

        # Get metadata
        metadata = map_data.get("metadata", {})
        sdc_track_index = metadata.get("sdc_track_index", -1)  # -1 as default if not found
        tracks_to_predict = metadata.get("tracks_to_predict", [])
        has_ego_trailer = int(bool(metadata.get("has_ego_trailer", False)))
        ego_trailer_track_index = int(metadata.get("ego_trailer_track_index", -1))
        non_kinematic_vehicle_params = metadata.get("non_kinematic_vehicle_params", {})
        if not isinstance(non_kinematic_vehicle_params, dict):
            non_kinematic_vehicle_params = {}
        non_kinematic_aliases = {
            "trailer_d_rear_axel2_rear_bumper": [
                "trailer_d_rear_axel2_rear_bumper",
                "trailer_d_rear_axle2_rear_bumper",
            ],
            "trailer_d_real_axel2_front_bumper": [
                "trailer_d_real_axel2_front_bumper",
                "trailer_d_rear_axel2_front_bumper",
                "trailer_d_rear_axle2_front_bumper",
            ],
        }

        # Write sdc_track_index
        f.write(struct.pack("i", sdc_track_index))

        # Write tracks_to_predict info (indices only)
        f.write(struct.pack("i", len(tracks_to_predict)))
        for track in tracks_to_predict:
            track_index = track.get("track_index", -1)
            f.write(struct.pack("i", track_index))

        # Count total entities
        num_objects = len(map_data.get("objects", []))
        num_roads = len(map_data.get("roads", []))
        # num_entities = num_objects + num_roads
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))
        # f.write(struct.pack('i', num_entities))
        # Write objects
        for obj in map_data.get("objects", []):
            # Write unique map id
            f.write(struct.pack("i", unique_map_id))

            # Write base entity data
            obj_type = obj.get("type", 1)
            if obj_type == "vehicle":
                obj_type = 1
            elif obj_type == "pedestrian":
                obj_type = 2
            elif obj_type == "cyclist":
                obj_type = 3
            f.write(struct.pack("i", obj_type))  # type
            f.write(struct.pack("i", normalize_int32_id(obj.get("id", 0))))  # id
            f.write(struct.pack("i", trajectory_length))  # array_size
            # Write position arrays
            positions = obj.get("position", [])
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("x", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("y", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("z", 0.0))))

            # Write velocity arrays
            velocities = obj.get("velocity", [])
            for arr, key in [(velocities, "x"), (velocities, "y"), (velocities, "z")]:
                for i in range(trajectory_length):
                    vel = arr[i] if i < len(arr) else {"x": 0.0, "y": 0.0, "z": 0.0}
                    f.write(struct.pack("f", float(vel.get(key, 0.0))))

            # Write heading and valid arrays
            headings = obj.get("heading", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}f",
                    *[float(headings[i]) if i < len(headings) else 0.0 for i in range(trajectory_length)],
                )
            )

            valids = obj.get("valid", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}i",
                    *[int(valids[i]) if i < len(valids) else 0 for i in range(trajectory_length)],
                )
            )

            # Write scalar fields
            f.write(struct.pack("f", float(obj.get("width", 0.0))))
            f.write(struct.pack("f", float(obj.get("length", 0.0))))
            f.write(struct.pack("f", float(obj.get("height", 0.0))))
            goal_pos = obj.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", obj.get("mark_as_expert", 0)))

        # Write roads
        for idx, road in enumerate(map_data.get("roads", [])):
            f.write(struct.pack("i", unique_map_id))

            geometry = road.get("geometry", [])
            road_type = road.get("map_element_id", 0)
            road_type_word = road.get("type", 0)
            if road_type_word == "lane":
                road_type = 2
            elif road_type_word == "road_edge":
                road_type = 15
            # breakpoint()
            if len(geometry) > 10 and road_type <= 16:
                geometry = simplify_polyline(geometry, 0.1, 250)
            size = len(geometry)
            # breakpoint()
            if road_type >= 0 and road_type <= 3:
                road_type = 4
            elif road_type >= 5 and road_type <= 13:
                road_type = 5
            elif road_type >= 14 and road_type <= 16:
                road_type = 6
            elif road_type == 17:
                road_type = 7
            elif road_type == 18:
                road_type = 8
            elif road_type == 19:
                road_type = 9
            elif road_type == 20:
                road_type = 10
            # Write base entity data
            f.write(struct.pack("i", road_type))  # type
            f.write(struct.pack("i", normalize_int32_id(road.get("id", 0))))  # id
            f.write(struct.pack("i", size))  # array_size

            # Write position arrays
            for coord in ["x", "y", "z"]:
                for point in geometry:
                    f.write(struct.pack("f", float(point.get(coord, 0.0))))

            # Write scalar fields
            f.write(struct.pack("f", float(road.get("width", 0.0))))
            f.write(struct.pack("f", float(road.get("length", 0.0))))
            f.write(struct.pack("f", float(road.get("height", 0.0))))
            goal_pos = road.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", road.get("mark_as_expert", 0)))

        objects = map_data.get("objects", [])
        f.write(struct.pack("i", extension_magic))
        f.write(struct.pack("i", extension_version))
        f.write(struct.pack("i", has_ego_trailer))
        f.write(struct.pack("i", ego_trailer_track_index))
        f.write(struct.pack("i", len(objects)))
        for obj_idx, obj in enumerate(objects):
            source_track_id_hash = stable_track_hash(obj.get("source_track_id"))
            is_trailer = int(has_ego_trailer and obj_idx == ego_trailer_track_index)
            parent_track_index = int(sdc_track_index if is_trailer else -1)
            f.write(struct.pack("Q", source_track_id_hash))
            f.write(struct.pack("i", is_trailer))
            f.write(struct.pack("i", parent_track_index))

        f.write(struct.pack("i", len(_NON_KINEMATIC_PARAM_ORDER)))
        for param_key in _NON_KINEMATIC_PARAM_ORDER:
            value = None
            for alias in non_kinematic_aliases.get(param_key, [param_key]):
                if alias in non_kinematic_vehicle_params:
                    value = non_kinematic_vehicle_params.get(alias)
                    break
            f.write(struct.pack("f", float(value if value is not None else 0.0)))


def load_map(map_name, unique_map_id, binary_output=None):
    """Loads a JSON map and optionally saves it as binary"""
    with open(map_name, "r") as f:
        map_data = json.load(f)

    if binary_output:
        save_map_binary(map_data, binary_output, unique_map_id)


def _process_single_map(args):
    """Worker function to process a single map file"""
    i, map_path, binary_path = args
    try:
        load_map(str(map_path), i, str(binary_path))
        return (i, map_path.name, True, None)
    except Exception as e:
        return (i, map_path.name, False, str(e))


def process_all_maps(
    data_folder="data/processed/training",
    output_folder=None,
    max_maps=50_000,
    num_workers=None,
):
    """Process all maps and save them as binaries using multiprocessing

    Args:
        data_folder: Path to the folder containing JSON map files
        output_folder: Path to save binary files (defaults to resources/drive/binaries/{dataset_name})
        max_maps: Maximum number of maps to process
        num_workers: Number of parallel workers (defaults to cpu_count())
    """
    from pathlib import Path

    if num_workers is None:
        num_workers = cpu_count()

    # Path to the training data
    data_dir = Path(data_folder)
    dataset_name = data_dir.name

    # Create the binaries directory if it doesn't exist
    if output_folder is None:
        binary_dir = Path(f"resources/drive/binaries/{dataset_name}")
    else:
        binary_dir = Path(output_folder)
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files in the training directory
    json_files = sorted(data_dir.glob("*.json"))

    # Prepare arguments for parallel processing
    tasks = []
    for i, map_path in enumerate(json_files[:max_maps]):
        binary_file = f"map_{i:03d}.bin"
        binary_path = binary_dir / binary_file
        tasks.append((i, map_path, binary_path))

    # Process maps in parallel with progress bar
    with Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap(_process_single_map, tasks), total=len(tasks), desc="Processing maps", unit="map")
        )

    # Collect statistics
    successful = sum(1 for _, _, success, _ in results if success)
    failed = sum(1 for _, _, success, _ in results if not success)

    if failed > 0:
        print(f"\nFailed {failed}/{len(results)} files:")
        for i, name, success, error in results:
            if not success:
                print(f"  {name}: {error}")


def test_performance(timeout=10, atn_cache=1024, num_agents=1024):
    import time

    env = Drive(
        num_agents=num_agents,
        num_maps=1,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        init_steps=0,
        episode_length=91,
    )

    env.reset()

    tick = 0
    actions = np.stack(
        [np.random.randint(0, space.n + 1, (atn_cache, num_agents)) for space in env.single_action_space], axis=-1
    )

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {num_agents * tick / (time.time() - start)}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Drive data, build offline BC datasets, or train BC policies")
    parser.add_argument(
        "--mode",
        choices=["convert", "build-bc", "train-bc"],
        default="convert",
        help="Whether to convert JSON maps, build offline BC dataset shards, or train a BC policy",
    )
    parser.add_argument(
        "--input-folder", type=str, default="data/processed/training", help="Path to folder containing JSON map files"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Path to save binary files or BC shards (defaults depend on mode)",
    )
    parser.add_argument("--max-maps", type=int, default=50_000, help="Maximum number of maps to process")
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Number of parallel workers for JSON conversion"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to the drive config file used by the offline BC dataset builder",
    )
    parser.add_argument("--wandb", action="store_true", help="Use wandb for BC training logging")
    parser.add_argument("--wandb-project", type=str, default="pufferlib", help="wandb project name")
    parser.add_argument("--wandb-group", type=str, default="debug", help="wandb group name")
    parser.add_argument("--wandb-name", type=str, default=None, help="Optional wandb run name")
    parser.add_argument("--tag", type=str, default=None, help="Optional run tag")
    parser.add_argument("--neptune", action="store_true", help="Use neptune for BC training logging")
    parser.add_argument("--neptune-name", type=str, default="pufferai", help="Neptune account/workspace name")
    parser.add_argument("--neptune-project", type=str, default="ablations", help="Neptune project name")

    args = parser.parse_args()

    if args.mode == "build-bc":
        config = load_drive_builder_config(args.config_path)
        config.setdefault("bc", {})
        config["bc"]["max_maps"] = args.max_maps
        build_bc_dataset(config, output_dir=args.output_folder)
    elif args.mode == "train-bc":
        config = load_drive_builder_config(args.config_path)
        config.setdefault("bc_train", {})
        config["bc_train"]["max_shards"] = args.max_maps
        config["wandb"] = args.wandb
        config["wandb_project"] = args.wandb_project
        config["wandb_group"] = args.wandb_group
        config["wandb_name"] = args.wandb_name
        config["tag"] = args.tag
        config["neptune"] = args.neptune
        config["neptune_name"] = args.neptune_name
        config["neptune_project"] = args.neptune_project
        train_bc_policy(config, output_dir=args.output_folder)
    else:
        process_all_maps(
            data_folder=args.input_folder,
            output_folder=args.output_folder,
            max_maps=args.max_maps,
            num_workers=args.num_workers,
        )
