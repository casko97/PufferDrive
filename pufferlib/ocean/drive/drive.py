import numpy as np
import gymnasium
import json
import struct
import os
import hashlib
import math
import shutil
import time
from collections import deque
from dataclasses import dataclass
from typing import Any
import pufferlib
from pufferlib.ocean.drive import binding
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

_POLICY_TYPE_PADDED = 0
_EMPTY_PARTNER_EPS = 1e-8
_EGO_TRAILER_STATE_FEATURES = 4
_DYNAMICS_MODEL_IDS = {"classic": 0, "jerk": 1, "articulated": 2}
_MAX_SPEED = 100.0
_PARTNER_POS_SCALE = 0.02
_TRAJECTORY_HORIZON = 32
_TRAJECTORY_FEATURES = 5
_TRAJECTORY_ACTION_DIM = _TRAJECTORY_HORIZON * _TRAJECTORY_FEATURES
_TRAJECTORY_HISTORY_FEATURES = 6
_TRAJECTORY_OBSERVATION_MODE = "trajectory_history_32"
_MAX_CLASSIC_ACCELERATION = 4.0
_MAX_CLASSIC_STEERING = 1.0
_DEFAULT_TRAJECTORY_LOOKAHEAD = 6.0
_DEFAULT_TRAJECTORY_ACCEL_GAIN = 0.5
_DEFAULT_TRAJECTORY_STEER_GAIN = 1.0
_DEFAULT_TRAJECTORY_TRACKER = "pure_pursuit"
_DEFAULT_TRAJECTORY_STANLEY_GAIN = 1.0
_DEFAULT_TRAJECTORY_STANLEY_SOFTENING = 1.0
_DEFAULT_TRAJECTORY_SMOOTHING_WINDOW = 1
_DEFAULT_TRAJECTORY_SMOOTHING_BLEND = 0.0
_DEFAULT_TRAJECTORY_MPC_HORIZON_STEPS = 6
_DEFAULT_TRAJECTORY_MPC_POSITION_WEIGHT = 1.0
_DEFAULT_TRAJECTORY_MPC_HEADING_WEIGHT = 0.1
_DEFAULT_TRAJECTORY_MPC_SPEED_WEIGHT = 0.2
_DEFAULT_TRAJECTORY_MPC_CONTROL_WEIGHT = 0.02
_DEFAULT_TRAJECTORY_MPC_ACCELERATION_GRID = (-4.0, -2.0, 0.0, 1.0, 2.5)
_DEFAULT_TRAJECTORY_MPC_STEERING_GRID = (-0.55, -0.35, -0.2, 0.0, 0.2, 0.35, 0.55)
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


def _startup_debug_enabled() -> bool:
    value = os.environ.get("PUFFERDRIVE_STARTUP_DEBUG", "")
    return value.strip().lower() in ("1", "true", "yes", "on")


def _startup_debug(message: str) -> None:
    if _startup_debug_enabled():
        print(f"[pufferdrive-startup] {message}", flush=True)


def postprocess_sdc_only_with_trailer_observations(
    sim_observations,
    ego_types,
    partner_types,
    ego_trailer_features,
    base_ego_features,
    base_partner_features,
    max_partner_objects,
    max_road_objects,
    road_features,
    type_classes,
):
    base_partner_dim = max_partner_objects * base_partner_features
    base_road_start = base_ego_features + base_partner_dim
    road_dim = max_road_objects * road_features

    aug_ego = base_ego_features + 1 + _EGO_TRAILER_STATE_FEATURES
    aug_partner = base_partner_features + 1
    aug_partner_dim = max_partner_objects * aug_partner
    aug_road_start = aug_ego + aug_partner_dim

    observations = np.zeros((sim_observations.shape[0], aug_ego + aug_partner_dim + road_dim), dtype=np.float32)
    observations[:, :base_ego_features] = sim_observations[:, :base_ego_features]

    sim_partner = sim_observations[:, base_ego_features:base_road_start].reshape(
        sim_observations.shape[0], max_partner_objects, base_partner_features
    )
    aug_partner_view = observations[:, aug_ego:aug_road_start].reshape(
        sim_observations.shape[0], max_partner_objects, aug_partner
    )
    aug_partner_view[:, :, :base_partner_features] = sim_partner
    observations[:, aug_road_start : aug_road_start + road_dim] = sim_observations[
        :, base_road_start : base_road_start + road_dim
    ]

    policy_type_max = type_classes - 1
    ego_types = np.clip(ego_types, _POLICY_TYPE_PADDED, policy_type_max).astype(np.float32)
    partner_types = np.clip(partner_types, _POLICY_TYPE_PADDED, policy_type_max).astype(np.float32)

    ego_type_idx = base_ego_features
    observations[:, ego_type_idx] = ego_types
    trailer_feature_start = ego_type_idx + 1
    observations[:, trailer_feature_start] = ego_trailer_features["rel_x"]
    observations[:, trailer_feature_start + 1] = ego_trailer_features["rel_y"]
    observations[:, trailer_feature_start + 2] = ego_trailer_features["rel_heading_x"]
    observations[:, trailer_feature_start + 3] = ego_trailer_features["rel_heading_y"]

    occupied_partner_slots = np.any(np.abs(sim_partner) > _EMPTY_PARTNER_EPS, axis=2)
    aug_partner_view[:, :, base_partner_features] = np.where(
        occupied_partner_slots, partner_types, _POLICY_TYPE_PADDED
    ).astype(np.float32)
    return observations


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clip_float(value, lower, upper):
    return max(lower, min(upper, float(value)))


def _encode_trajectory_position(value):
    return _clip_float(value * _PARTNER_POS_SCALE, -1.0, 1.0)


def _decode_trajectory_position(value):
    return float(value) / _PARTNER_POS_SCALE


def _encode_trajectory_heading(value):
    return _clip_float(_wrap_to_pi(value) / math.pi, -1.0, 1.0)


def _decode_trajectory_heading(value):
    return _clip_float(value, -1.0, 1.0) * math.pi


def _encode_trajectory_speed(value):
    return _clip_float(float(value) / _MAX_SPEED, 0.0, 1.0)


def _decode_trajectory_speed(value):
    return _clip_float(value, 0.0, 1.0) * _MAX_SPEED


def _make_history_state(x, y, heading, speed, valid):
    return np.array([x, y, heading, speed, float(valid)], dtype=np.float32)


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
            raise ValueError(f"Extension magic mismatch in {binary_path}")
        if extension_version != 2:
            raise ValueError(f"Unsupported extension version {extension_version} in {binary_path}")

        f.seek(4 * 2, os.SEEK_CUR)
        object_meta_count = struct.unpack("<i", f.read(4))[0]
        f.seek(object_meta_count * (8 + 4 + 4), os.SEEK_CUR)

        vehicle_param_count = struct.unpack("<i", f.read(4))[0]
        if vehicle_param_count != len(_NON_KINEMATIC_PARAM_ORDER):
            raise ValueError(
                f"Expected {len(_NON_KINEMATIC_PARAM_ORDER)} non-kinematic params in {binary_path}, got {vehicle_param_count}"
            )
        values = struct.unpack(f"<{vehicle_param_count}f", f.read(4 * vehicle_param_count))

    params = tuple(float(v) for v in values)
    _NON_KINEMATIC_PARAM_CACHE[binary_path] = params
    return params


def _normalize_optional_path(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "none":
            return None
        return stripped
    return str(value)


def _stable_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class MapDatasetEntry:
    dataset_id: int
    map_path: str
    relative_path: str
    source_name: str | None = None


class MapDatasetCatalog:
    def __init__(self, dataset_root: str, entries: list[MapDatasetEntry], manifest_path: str | None = None):
        if not entries:
            raise ValueError(f"No map binaries found in dataset: {dataset_root}")
        self.dataset_root = str(Path(dataset_root).resolve())
        self.entries = list(entries)
        self.manifest_path = manifest_path

    @classmethod
    def from_map_dir(cls, map_dir: str) -> "MapDatasetCatalog":
        dataset_root = Path(map_dir).resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Map dataset directory not found: {dataset_root}")

        manifest_path = dataset_root / "dataset_manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            rows = payload.get("maps") or payload.get("entries") or []
            entries = []
            for idx, row in enumerate(rows):
                relative_path = row.get("relative_path") or row.get("path") or row.get("map_name")
                if not relative_path:
                    continue
                map_path = (dataset_root / relative_path).resolve()
                if not map_path.exists():
                    raise FileNotFoundError(f"Manifest-referenced map file not found: {map_path}")
                entries.append(
                    MapDatasetEntry(
                        dataset_id=_stable_int(row.get("map_id"), idx),
                        map_path=str(map_path),
                        relative_path=str(Path(relative_path)),
                        source_name=row.get("source_name"),
                    )
                )
            return cls(str(dataset_root), entries, manifest_path=str(manifest_path))

        entries = []
        for idx, map_path in enumerate(sorted(dataset_root.glob("*.bin"))):
            entries.append(
                MapDatasetEntry(
                    dataset_id=idx,
                    map_path=str(map_path.resolve()),
                    relative_path=map_path.name,
                    source_name=map_path.name,
                )
            )
        return cls(str(dataset_root), entries, manifest_path=None)

    def limit(self, num_maps: int) -> list[MapDatasetEntry]:
        requested = int(num_maps)
        if requested <= 0:
            raise ValueError(f"num_maps must be > 0, got {requested}")
        if requested > len(self.entries):
            raise ValueError(
                f"num_maps ({requested}) exceeds available maps in directory ({len(self.entries)}). "
                "Please reduce num_maps or add more maps."
            )
        return list(self.entries[:requested])


class MapValidationCache:
    def __init__(self, dataset_root: str, signature: dict[str, Any]):
        self.dataset_root = Path(dataset_root).resolve()
        self.signature = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        self.cache_path = self.dataset_root / ".pufferdrive_map_validation_cache.json"
        self._payload = {"version": 1, "entries": {}}
        if self.cache_path.exists():
            try:
                with self.cache_path.open("r", encoding="utf-8") as file_obj:
                    payload = json.load(file_obj)
                if isinstance(payload, dict) and isinstance(payload.get("entries"), dict):
                    self._payload = payload
            except (OSError, json.JSONDecodeError):
                self._payload = {"version": 1, "entries": {}}

    @staticmethod
    def build_signature(
        *,
        dynamics_model: str,
        init_mode: int,
        control_mode: int,
        init_steps: int,
        max_controlled_agents: int,
        goal_behavior: int,
        goal_target_distance: float,
        force_zero_trailer_articulation_at_init: bool,
        non_kinematic_vehicle_params_override: tuple[float, ...] | None,
    ) -> dict[str, Any]:
        return {
            "dynamics_model": str(dynamics_model),
            "init_mode": int(init_mode),
            "control_mode": int(control_mode),
            "init_steps": int(init_steps),
            "max_controlled_agents": int(max_controlled_agents),
            "goal_behavior": int(goal_behavior),
            "goal_target_distance": float(goal_target_distance),
            "force_zero_trailer_articulation_at_init": bool(force_zero_trailer_articulation_at_init),
            "non_kinematic_vehicle_params_override": list(non_kinematic_vehicle_params_override)
            if non_kinematic_vehicle_params_override is not None
            else None,
        }

    def _entry_key(self, map_path: str) -> str:
        try:
            return str(Path(map_path).resolve().relative_to(self.dataset_root))
        except ValueError:
            return str(Path(map_path).resolve())

    def get(self, map_path: str) -> dict[str, Any] | None:
        row = self._payload["entries"].get(self._entry_key(map_path))
        if not row or row.get("signature") != self.signature:
            return None
        return {
            "active_agent_count": int(row.get("active_agent_count", 0)),
            "valid_for_sampling": bool(row.get("valid_for_sampling", False)),
            "invalid_initial_trailer_state": bool(row.get("invalid_initial_trailer_state", False)),
        }

    def set(self, map_path: str, metadata: dict[str, Any]) -> None:
        self._payload["entries"][self._entry_key(map_path)] = {
            "signature": self.signature,
            "active_agent_count": int(metadata.get("active_agent_count", 0)),
            "valid_for_sampling": bool(metadata.get("valid_for_sampling", False)),
            "invalid_initial_trailer_state": bool(metadata.get("invalid_initial_trailer_state", False)),
        }
        self.flush()

    def flush(self) -> None:
        with self.cache_path.open("w", encoding="utf-8") as file_obj:
            json.dump(self._payload, file_obj, indent=2, sort_keys=True)


class MapScheduler:
    def __init__(
        self,
        entries: list[MapDatasetEntry],
        *,
        schedule: str = "shuffle_once_per_epoch",
        seed: int | None = None,
        allow_live_duplicates: bool = False,
    ):
        if schedule not in {"shuffle_once_per_epoch", "sequential", "random_with_replacement"}:
            raise ValueError(f"Unsupported map_schedule: {schedule}")
        self.entries = list(entries)
        self.schedule = schedule
        self.allow_live_duplicates = bool(allow_live_duplicates)
        self._rng = np.random.default_rng(seed)
        self._order = list(range(len(self.entries)))
        self._cursor = 0
        if self.schedule == "shuffle_once_per_epoch":
            self._rng.shuffle(self._order)

    def _next_entry(self) -> MapDatasetEntry:
        if not self.entries:
            raise ValueError("MapScheduler requires at least one dataset entry")
        if self.schedule == "random_with_replacement":
            return self.entries[int(self._rng.integers(0, len(self.entries)))]
        if self._cursor >= len(self._order):
            self._cursor = 0
            if self.schedule == "shuffle_once_per_epoch":
                self._rng.shuffle(self._order)
        entry = self.entries[self._order[self._cursor]]
        self._cursor += 1
        return entry

    def select_for_agent_budget(
        self,
        num_agents: int,
        inspect_entry,
    ) -> list[tuple[MapDatasetEntry, dict[str, Any]]]:
        if num_agents <= 0:
            raise ValueError(f"num_agents must be > 0, got {num_agents}")
        selected = []
        selected_ids = set()
        total_agents = 0
        max_attempts = len(self.entries) if not self.allow_live_duplicates else max(len(self.entries), num_agents * 2)
        attempts = 0

        while total_agents < int(num_agents) and attempts < max_attempts:
            entry = self._next_entry()
            attempts += 1
            if not self.allow_live_duplicates and entry.dataset_id in selected_ids:
                continue
            metadata = inspect_entry(entry)
            if not metadata["valid_for_sampling"] or metadata["active_agent_count"] <= 0:
                continue
            selected.append((entry, metadata))
            selected_ids.add(entry.dataset_id)
            total_agents += int(metadata["active_agent_count"])

        if total_agents <= 0:
            raise ValueError("No valid maps available for the current initialization settings")
        return selected

    def select_all_valid(
        self,
        inspect_entry,
    ) -> list[tuple[MapDatasetEntry, dict[str, Any]]]:
        selected = []
        for entry in self.entries:
            metadata = inspect_entry(entry)
            if metadata["valid_for_sampling"] and metadata["active_agent_count"] > 0:
                selected.append((entry, metadata))
        if not selected:
            raise ValueError("No valid maps available for sequential map sampling")
        return selected


def _compute_agent_offsets(
    selected_entries: list[tuple[MapDatasetEntry, dict[str, Any]]],
    requested_num_agents: int | None = None,
) -> tuple[list[int], list[int], int]:
    agent_offsets = [0]
    map_ids = []
    total_agents = 0
    for entry, metadata in selected_entries:
        map_ids.append(entry.dataset_id)
        total_agents += int(metadata["active_agent_count"])
        if requested_num_agents is not None and total_agents >= int(requested_num_agents):
            total_agents = int(requested_num_agents)
            agent_offsets.append(total_agents)
            break
        agent_offsets.append(total_agents)
    return agent_offsets, map_ids, len(map_ids)


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
        goal_at_gt_traj_end=False,
        goal_radius=2.0,
        goal_speed=20.0,
        collision_behavior=0,
        offroad_behavior=0,
        dt=0.1,
        vision_range=21,
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
        map_dir="resources/drive/binaries/training",
        scenario_filter=None,
        scenario_filter_threshold_deg=None,
        scenario_filter_manifest_path=None,
        sequential_map_sampling=False,
        map_schedule="shuffle_once_per_epoch",
        map_allow_live_duplicates=False,
        map_scheduler_seed=None,
        sdc_runtime_truck_override=False,
        sdc_runtime_truck_ref_bin=None,
        force_truck_params_from_ref_bin=None,
        force_zero_trailer_articulation_at_init=False,
        trajectory_lookahead_distance=_DEFAULT_TRAJECTORY_LOOKAHEAD,
        trajectory_accel_gain=_DEFAULT_TRAJECTORY_ACCEL_GAIN,
        trajectory_steer_gain=_DEFAULT_TRAJECTORY_STEER_GAIN,
        trajectory_tracker_type=_DEFAULT_TRAJECTORY_TRACKER,
        trajectory_stanley_gain=_DEFAULT_TRAJECTORY_STANLEY_GAIN,
        trajectory_stanley_softening=_DEFAULT_TRAJECTORY_STANLEY_SOFTENING,
        trajectory_smoothing_window=_DEFAULT_TRAJECTORY_SMOOTHING_WINDOW,
        trajectory_smoothing_blend=_DEFAULT_TRAJECTORY_SMOOTHING_BLEND,
        trajectory_smoothness_xy_weight=0.0,
        trajectory_smoothness_heading_weight=0.0,
        trajectory_smoothness_speed_weight=0.0,
        trajectory_control_substeps=1,
        substep_expert_interpolation=True,
        trajectory_mpc_horizon_steps=_DEFAULT_TRAJECTORY_MPC_HORIZON_STEPS,
        trajectory_mpc_position_weight=_DEFAULT_TRAJECTORY_MPC_POSITION_WEIGHT,
        trajectory_mpc_heading_weight=_DEFAULT_TRAJECTORY_MPC_HEADING_WEIGHT,
        trajectory_mpc_speed_weight=_DEFAULT_TRAJECTORY_MPC_SPEED_WEIGHT,
        trajectory_mpc_control_weight=_DEFAULT_TRAJECTORY_MPC_CONTROL_WEIGHT,
        trajectory_history_warmstart_seconds=0.0,
        terminate_on_stop=False,
    ):
        # env
        self.dt = dt
        self.vision_range = int(vision_range)
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_goal = reward_goal
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.goal_radius = goal_radius
        self.goal_speed = goal_speed
        self.goal_behavior = goal_behavior
        self.goal_target_distance = goal_target_distance
        self.goal_at_gt_traj_end = _as_bool(goal_at_gt_traj_end)
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.human_agent_idx = human_agent_idx
        self.episode_length = episode_length
        self.termination_mode = termination_mode
        self.resample_frequency = resample_frequency
        self.dynamics_model = dynamics_model
        self.type_classes = binding.POLICY_TYPE_CLASS_COUNT

        # Observation space calculation
        self._base_ego_features = {"classic": binding.EGO_FEATURES_CLASSIC, "articulated": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}.get(
            dynamics_model
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
        self.action_type_str = action_type
        self.scenario_filter = scenario_filter
        self.scenario_filter_threshold_deg = scenario_filter_threshold_deg
        self.scenario_filter_manifest_path = scenario_filter_manifest_path
        self.original_map_dir = map_dir
        self.original_num_maps = int(num_maps)
        self.sequential_map_sampling = _as_bool(sequential_map_sampling)
        requested_map_schedule = str(map_schedule).strip().lower()
        self.map_schedule = "sequential" if self.sequential_map_sampling else requested_map_schedule
        self.map_allow_live_duplicates = False if self.sequential_map_sampling else _as_bool(map_allow_live_duplicates)
        self.map_scheduler_seed = seed if map_scheduler_seed is None else int(map_scheduler_seed)
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
                raise FileNotFoundError(f"Truck reference artifact not found: {reference_bin}")
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
            raise ValueError(
                f"control_mode must be one of 'control_vehicles', 'control_wosac', or 'control_agents'. Got: {self.control_mode_str}"
            )
        self.trajectory_horizon = _TRAJECTORY_HORIZON
        self.trajectory_action_dim = _TRAJECTORY_ACTION_DIM
        self.trajectory_features = _TRAJECTORY_FEATURES
        self.trajectory_history_horizon = _TRAJECTORY_HORIZON
        self.trajectory_history_features = _TRAJECTORY_HISTORY_FEATURES
        self.trajectory_lookahead_distance = float(trajectory_lookahead_distance)
        self.trajectory_accel_gain = float(trajectory_accel_gain)
        self.trajectory_steer_gain = float(trajectory_steer_gain)
        self.trajectory_tracker_type = str(trajectory_tracker_type).strip().lower()
        self.trajectory_stanley_gain = float(trajectory_stanley_gain)
        self.trajectory_stanley_softening = max(float(trajectory_stanley_softening), 1e-3)
        self.trajectory_smoothing_window = max(1, int(trajectory_smoothing_window))
        self.trajectory_smoothing_blend = _clip_float(float(trajectory_smoothing_blend), 0.0, 1.0)
        self.trajectory_smoothness_xy_weight = max(float(trajectory_smoothness_xy_weight), 0.0)
        self.trajectory_smoothness_heading_weight = max(float(trajectory_smoothness_heading_weight), 0.0)
        self.trajectory_smoothness_speed_weight = max(float(trajectory_smoothness_speed_weight), 0.0)
        self.trajectory_control_substeps = max(1, int(trajectory_control_substeps))
        self.substep_expert_interpolation = _as_bool(substep_expert_interpolation)
        self.trajectory_mpc_horizon_steps = max(1, int(trajectory_mpc_horizon_steps))
        self.trajectory_mpc_position_weight = float(trajectory_mpc_position_weight)
        self.trajectory_mpc_heading_weight = float(trajectory_mpc_heading_weight)
        self.trajectory_mpc_speed_weight = float(trajectory_mpc_speed_weight)
        self.trajectory_mpc_control_weight = float(trajectory_mpc_control_weight)
        self.trajectory_history_warmstart_seconds = max(float(trajectory_history_warmstart_seconds), 0.0)
        self.terminate_on_stop = _as_bool(terminate_on_stop)
        self.trajectory_mpc_acceleration_grid = np.asarray(_DEFAULT_TRAJECTORY_MPC_ACCELERATION_GRID, dtype=np.float32)
        self.trajectory_mpc_steering_grid = np.asarray(_DEFAULT_TRAJECTORY_MPC_STEERING_GRID, dtype=np.float32)
        if self.trajectory_tracker_type not in ("pure_pursuit", "stanley", "mpc"):
            raise ValueError(
                "trajectory_tracker_type must be one of 'pure_pursuit', 'stanley', or 'mpc'. "
                f"Got: {trajectory_tracker_type!r}"
            )
        self.is_trajectory_action = self.action_type_str == "trajectory"
        self.is_trajectory_observation = self.observation_mode_str == _TRAJECTORY_OBSERVATION_MODE
        self.trajectory_control_is_physical = (
            self.is_trajectory_action and self.trajectory_tracker_type == "mpc" and self.dynamics_model in ("classic", "articulated")
        )
        self._last_trajectory_control_debug: list[dict[str, float | int | bool | str]] = []
        self._needs_reset = False

        if self.observation_mode_str == "default" or self.is_trajectory_observation:
            self.observation_mode = 0
        elif self.observation_mode_str == "sdc_only_with_trailer":
            self.observation_mode = 1
        else:
            raise ValueError(
                "observation_mode must be one of 'default', 'sdc_only_with_trailer', "
                f"or '{_TRAJECTORY_OBSERVATION_MODE}'. "
                f"Got: {self.observation_mode_str}"
            )
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
        self._base_policy_num_obs = self.num_obs
        self.trajectory_base_obs_dim = self._base_policy_num_obs
        self.trajectory_ego_history_dim = 0
        self.trajectory_partner_history_dim = 0
        if self.is_trajectory_observation:
            self.trajectory_ego_history_dim = self.trajectory_history_horizon * self.trajectory_history_features
            self.trajectory_partner_history_dim = (
                self.max_partner_objects * self.trajectory_history_horizon * self.trajectory_history_features
            )
            self.num_obs = (
                self._base_policy_num_obs + self.trajectory_ego_history_dim + self.trajectory_partner_history_dim
            )
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)
        if self.init_mode_str == "create_all_valid":
            self.init_mode = 0
        elif self.init_mode_str == "create_only_controlled":
            self.init_mode = 1
        else:
            raise ValueError(
                f"init_mode must be one of 'create_all_valid' or 'create_only_controlled'. Got: {self.init_mode_str}"
            )

        if action_type == "discrete":
            if dynamics_model in ("classic", "articulated"):
                # Joint action space (assume dependence)
                self.single_action_space = gymnasium.spaces.MultiDiscrete([7 * 13])
                # Multi discrete (assume independence)
                # self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
            elif dynamics_model == "jerk":
                # Joint action space (assume dependence) - 4 longitudinal × 3 lateral = 12
                self.single_action_space = gymnasium.spaces.MultiDiscrete([4 * 3])
            else:
                raise ValueError(f"dynamics_model must be 'classic', 'articulated' or 'jerk'. Got: {dynamics_model}")
        elif action_type == "continuous":
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        elif action_type == "trajectory":
            self.single_action_space = gymnasium.spaces.Box(
                low=-1, high=1, shape=(self.trajectory_action_dim,), dtype=np.float32
            )
        else:
            raise ValueError(f"action_space must be 'discrete', 'continuous', or 'trajectory'. Got: {action_type}")

        self._action_type_flag = 0 if action_type == "discrete" else 1

        catalog_started_at = time.perf_counter()
        self.map_catalog = MapDatasetCatalog.from_map_dir(map_dir)
        self.selected_map_entries = self.map_catalog.limit(num_maps)
        self.map_dir = self.map_catalog.dataset_root
        self.num_maps = len(self.selected_map_entries)
        self.max_controlled_agents = int(max_controlled_agents)
        self._map_validation_cache = MapValidationCache(
            self.map_catalog.dataset_root,
            MapValidationCache.build_signature(
                dynamics_model=self.dynamics_model,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                init_steps=self.init_steps,
                max_controlled_agents=self.max_controlled_agents,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                force_zero_trailer_articulation_at_init=self.force_zero_trailer_articulation_at_init,
                non_kinematic_vehicle_params_override=self.non_kinematic_vehicle_params_override,
            ),
        )
        self._map_scheduler = MapScheduler(
            self.selected_map_entries,
            schedule=self.map_schedule,
            seed=self.map_scheduler_seed,
            allow_live_duplicates=self.map_allow_live_duplicates,
        )
        _startup_debug(
            "map_catalog "
            f"dataset_root={self.map_catalog.dataset_root} requested_num_maps={self.original_num_maps} "
            f"selected_count={self.num_maps} manifest={self.map_catalog.manifest_path} "
            f"elapsed_s={time.perf_counter() - catalog_started_at:.3f}"
        )

        selected_entries = self._select_live_maps(
            requested_num_agents=num_agents,
            sequential_map_sampling=self.sequential_map_sampling,
        )
        agent_offsets, map_ids, num_envs = _compute_agent_offsets(
            selected_entries,
            requested_num_agents=None if self.sequential_map_sampling else num_agents,
        )
        self.num_agents = agent_offsets[-1]
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        self._live_map_entries = [entry for entry, _ in selected_entries]
        super().__init__(buf=buf)
        if self.is_trajectory_action:
            self._policy_actions = self.actions
            self._control_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
            self._trajectory_smoothness_penalties = np.zeros(self.num_agents, dtype=np.float32)
        else:
            self._policy_actions = self.actions
            self._control_actions = self.actions
            self._trajectory_smoothness_penalties = np.zeros(self.num_agents, dtype=np.float32)
        self._sim_observations = self.observations
        if self.observation_mode == 1 or self.is_trajectory_observation:
            self._sim_observations = np.zeros((self.num_agents, self._sim_num_obs), dtype=np.float32)
        self._trajectory_entity_history = {}
        self._cached_global_agent_state = None
        self._cached_partner_ids = None
        self._suspend_trajectory_history_updates = False
        self.c_envs = self._build_vector_env(self._live_map_entries, self.agent_offsets, seed)

    def _resample_vector_envs(self, seed):
        binding.vec_close(self.c_envs)
        selected_entries = self._select_live_maps(
            requested_num_agents=self.num_agents,
            sequential_map_sampling=self.sequential_map_sampling,
        )
        agent_offsets, map_ids, num_envs = _compute_agent_offsets(
            selected_entries,
            requested_num_agents=None if self.sequential_map_sampling else self.num_agents,
        )
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        self._live_map_entries = [entry for entry, _ in selected_entries]
        self.c_envs = self._build_vector_env(self._live_map_entries, self.agent_offsets, seed)
        binding.vec_reset(self.c_envs, seed)

    def _inspect_map_entry(self, entry: MapDatasetEntry) -> dict[str, Any]:
        cached = self._map_validation_cache.get(entry.map_path)
        if cached is not None:
            return cached

        metadata = binding.inspect_map(
            map_path=entry.map_path,
            dynamics_model=_DYNAMICS_MODEL_IDS[self.dynamics_model],
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                goal_at_gt_traj_end=int(self.goal_at_gt_traj_end),
                non_kinematic_vehicle_params_override=self.non_kinematic_vehicle_params_override,
                force_zero_trailer_articulation_at_init=int(self.force_zero_trailer_articulation_at_init),
            )
        normalized = {
            "active_agent_count": int(metadata.get("active_agent_count", 0)),
            "valid_for_sampling": bool(metadata.get("valid_for_sampling", False)),
            "invalid_initial_trailer_state": bool(metadata.get("invalid_initial_trailer_state", False)),
        }
        self._map_validation_cache.set(entry.map_path, normalized)
        return normalized

    def _select_live_maps(self, requested_num_agents: int, sequential_map_sampling: bool):
        started_at = time.perf_counter()
        if sequential_map_sampling:
            selected_entries = self._map_scheduler.select_all_valid(self._inspect_map_entry)
        else:
            selected_entries = self._map_scheduler.select_for_agent_budget(
                requested_num_agents,
                self._inspect_map_entry,
            )
        total_agents = sum(int(metadata["active_agent_count"]) for _, metadata in selected_entries)
        _startup_debug(
            "map_selection "
            f"sequential={int(sequential_map_sampling)} selected_maps={len(selected_entries)} "
            f"total_agents={total_agents} elapsed_s={time.perf_counter() - started_at:.3f}"
        )
        return selected_entries

    def _build_vector_env(self, live_map_entries: list[MapDatasetEntry], agent_offsets: list[int], seed: int):
        env_ids = []
        for i, entry in enumerate(live_map_entries):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
            env_id = binding.env_init(
                self._sim_observations[cur:nxt],
                self._control_actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                action_type=self._action_type_flag,
                continuous_actions_are_physical=int(self.trajectory_control_is_physical),
                substep_expert_interpolation=int(self.substep_expert_interpolation),
                human_agent_idx=self.human_agent_idx,
                reward_vehicle_collision=self.reward_vehicle_collision,
                reward_offroad_collision=self.reward_offroad_collision,
                reward_goal=self.reward_goal,
                reward_goal_post_respawn=self.reward_goal_post_respawn,
                goal_radius=self.goal_radius,
                goal_speed=self.goal_speed,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                goal_at_gt_traj_end=int(self.goal_at_gt_traj_end),
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=self.dt,
                vision_range=self.vision_range,
                dynamics_model=_DYNAMICS_MODEL_IDS[self.dynamics_model],
                episode_length=(int(self.episode_length) if self.episode_length is not None else None),
                termination_mode=(int(self.termination_mode) if self.termination_mode is not None else 0),
                max_controlled_agents=self.max_controlled_agents,
                map_id=entry.dataset_id,
                map_path=entry.map_path,
                max_agents=nxt - cur,
                ini_file="pufferlib/config/ocean/drive.ini",
                init_steps=self.init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=self.map_dir,
                non_kinematic_vehicle_params_override=self.non_kinematic_vehicle_params_override,
                force_zero_trailer_articulation_at_init=int(self.force_zero_trailer_articulation_at_init),
            )
            env_ids.append(env_id)
        return binding.vectorize(*env_ids)

    @property
    def done(self):
        return bool(self._needs_reset)

    def reset(self, seed=None):
        if seed is None:
            seed = int(np.random.randint(0, 2**32 - 1))
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
        if self.is_trajectory_observation or self.is_trajectory_action:
            self._clear_trajectory_history()
        self._needs_reset = False
        warmstart_timestep = self._trajectory_history_warmstart_timestep()
        if warmstart_timestep > 0 and (self.is_trajectory_observation or self.is_trajectory_action):
            for timestep in range(warmstart_timestep + 1):
                self.set_logged_timestep(timestep, reset_history=False, update_history=True)
            return self.observations, []
        self._postprocess_observations()
        return self.observations, []

    def step(self, actions):
        self.terminals[:] = 0
        self.truncations[:] = 0
        self._policy_actions[:] = actions
        self._trajectory_smoothness_penalties.fill(0.0)
        terminal_stop_info = None
        if self.is_trajectory_action and self.trajectory_control_substeps > 1:
            sub_dt = float(self.dt) / float(self.trajectory_control_substeps)
            decoded_actions = self._decode_trajectory_actions(actions)
            self._control_actions[:] = self._trajectory_to_control_actions(decoded_actions)
            interrupted = False
            for substep_idx in range(self.trajectory_control_substeps):
                alpha = (float(substep_idx) + 0.5) / float(self.trajectory_control_substeps)
                binding.vec_physics_substep(self.c_envs, sub_dt, alpha)
                self._suspend_trajectory_history_updates = True
                self._postprocess_observations()
                self._suspend_trajectory_history_updates = False
                if np.any(self.terminals) or np.any(self.truncations):
                    interrupted = True
                    break
            if not interrupted:
                binding.vec_advance_logged_timestep(self.c_envs)
                self.tick += 1
        else:
            if self.is_trajectory_action:
                decoded_actions = self._decode_trajectory_actions(actions)
                self._control_actions[:] = self._trajectory_to_control_actions(decoded_actions)
            else:
                self.actions[:] = actions
            binding.vec_step(self.c_envs)
            self.tick += 1
        if self.is_trajectory_action:
            self.rewards[:] = self.rewards + self._trajectory_smoothness_penalties
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
            self._needs_reset = True
            if self.is_trajectory_observation or self.is_trajectory_action:
                self._clear_trajectory_history()
        self._postprocess_observations()
        if self.terminate_on_stop:
            terminal_stop_info = self._mark_terminals_for_stopped_agents()
            if terminal_stop_info:
                info.append({"terminal_stop_reasons": terminal_stop_info})
        if np.any(self.terminals) or np.any(self.truncations):
            self._needs_reset = True
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def physics_substep(self, actions, *, sub_dt: float, alpha: float, update_history: bool = True):
        self.terminals[:] = 0
        self.truncations[:] = 0
        self._policy_actions[:] = actions
        self._trajectory_smoothness_penalties.fill(0.0)
        if self.is_trajectory_action:
            decoded_actions = self._decode_trajectory_actions(actions)
            self._control_actions[:] = self._trajectory_to_control_actions(decoded_actions)
        else:
            self.actions[:] = actions
        binding.vec_physics_substep(self.c_envs, float(sub_dt), min(float(alpha), 1.0 - 1e-4))
        if self.is_trajectory_action:
            self.rewards[:] = self.rewards + self._trajectory_smoothness_penalties
        self._suspend_trajectory_history_updates = not bool(update_history)
        self._postprocess_observations()
        self._suspend_trajectory_history_updates = False
        return (self.observations, self.rewards, self.terminals, self.truncations, [])

    def advance_logged_timestep(self, *, update_history: bool = True):
        binding.vec_advance_logged_timestep(self.c_envs)
        self.tick += 1
        self._suspend_trajectory_history_updates = not bool(update_history)
        self._postprocess_observations()
        self._suspend_trajectory_history_updates = False
        return (self.observations, self.rewards, self.terminals, self.truncations, [])

    def set_logged_timestep(self, timestep: int, *, reset_history: bool = False, update_history: bool = True):
        binding.vec_set_logged_timestep(self.c_envs, int(timestep))
        self.tick = int(timestep)
        if reset_history and (self.is_trajectory_observation or self.is_trajectory_action):
            self._clear_trajectory_history()
        self._suspend_trajectory_history_updates = not bool(update_history)
        self._postprocess_observations()
        self._suspend_trajectory_history_updates = False
        return (self.observations, self.rewards, self.terminals, self.truncations, [])

    def _mark_terminals_for_stopped_agents(self):
        diagnostics = self.get_agent_diagnostics()
        stopped_mask = np.asarray(diagnostics["stopped"], dtype=bool)
        if not np.any(stopped_mask):
            return None

        reasons = []
        for agent_idx in np.flatnonzero(stopped_mask):
            if bool(diagnostics["reached_goal"][agent_idx]):
                reason = "goal_stop"
            elif bool(diagnostics["offroad_flag"][agent_idx]) or int(diagnostics["collision_state"][agent_idx]) == 2:
                reason = "offroad_stop"
            elif int(diagnostics["collision_state"][agent_idx]) == 1:
                reason = "collision_stop"
            else:
                reason = "stopped_unknown"
            reasons.append({"agent_idx": int(agent_idx), "reason": reason})
        self.terminals[stopped_mask] = 1
        return reasons

    def get_last_trajectory_control_debug(self):
        return list(self._last_trajectory_control_debug)

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

    def get_partner_ids(self):
        """Get partner ids in the same ordering as partner observations."""
        ids = np.zeros((self.num_agents, self.max_partner_objects), dtype=np.int32)
        binding.vec_get_partner_ids(self.c_envs, ids)
        return ids

    def get_agent_diagnostics(self):
        diagnostics = {
            "collision_state": np.zeros(self.num_agents, dtype=np.int32),
            "offroad_flag": np.zeros(self.num_agents, dtype=np.int32),
            "reached_goal": np.zeros(self.num_agents, dtype=np.int32),
            "speed": np.zeros(self.num_agents, dtype=np.float32),
            "stopped": np.zeros(self.num_agents, dtype=np.int32),
        }
        binding.vec_get_agent_diagnostics(
            self.c_envs,
            diagnostics["collision_state"],
            diagnostics["offroad_flag"],
            diagnostics["reached_goal"],
            diagnostics["speed"],
            diagnostics["stopped"],
        )
        return diagnostics

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

    def _clear_trajectory_history(self):
        self._trajectory_entity_history = {}
        self._cached_global_agent_state = None
        self._cached_partner_ids = None

    def _trajectory_history_warmstart_timestep(self):
        if self.trajectory_history_warmstart_seconds <= 0.0:
            return 0
        if self.episode_length is None:
            return 0
        sim_steps = max(int(self.episode_length) - int(self.init_steps), 0)
        if sim_steps <= 0:
            return 0
        warmstart_timestep = int(round(self.trajectory_history_warmstart_seconds / max(float(self.dt), 1e-6)))
        return min(max(warmstart_timestep, 0), sim_steps - 1)

    def _refresh_trajectory_state_cache(self):
        self._cached_global_agent_state = self.get_global_agent_state()
        self._cached_partner_ids = self.get_partner_ids()

    def _append_trajectory_state(self, entity_id, x, y, heading, speed, valid):
        if entity_id < 0:
            return
        history = self._trajectory_entity_history.get(entity_id)
        if history is None:
            history = deque(maxlen=self.trajectory_history_horizon)
            self._trajectory_entity_history[entity_id] = history
        history.append(_make_history_state(x, y, heading, speed, valid))

    def _decode_current_partner_states(self, ego_states, partner_ids):
        partner_states = {}
        partner_offset = self._base_ego_features
        partner_dim = self.max_partner_objects * self._base_partner_features
        partner_obs = self._sim_observations[:, partner_offset : partner_offset + partner_dim].reshape(
            self.num_agents, self.max_partner_objects, self._base_partner_features
        )

        ego_x = ego_states["x"]
        ego_y = ego_states["y"]
        ego_heading = ego_states["heading"]
        ego_cos = np.cos(ego_heading)
        ego_sin = np.sin(ego_heading)

        for row in range(self.num_agents):
            for slot in range(self.max_partner_objects):
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
                    float(ego_x[row] + dx),
                    float(ego_y[row] + dy),
                    _wrap_to_pi(float(ego_heading[row] + rel_heading)),
                    float(features[6]) * _MAX_SPEED,
                    1.0,
                )
        return partner_states

    def _update_trajectory_history(self):
        ego_states = self.get_global_agent_state()
        partner_ids = self.get_partner_ids()
        ego_speeds = self._sim_observations[:, 2] * _MAX_SPEED

        self._cached_global_agent_state = ego_states
        self._cached_partner_ids = partner_ids

        for idx in range(self.num_agents):
            self._append_trajectory_state(
                int(ego_states["id"][idx]),
                float(ego_states["x"][idx]),
                float(ego_states["y"][idx]),
                float(ego_states["heading"][idx]),
                float(ego_speeds[idx]),
                1.0,
            )

        for entity_id, state in self._decode_current_partner_states(ego_states, partner_ids).items():
            self._append_trajectory_state(entity_id, *state)

    def _encode_history_block(self, history, current_x, current_y, current_heading):
        block = np.zeros((self.trajectory_history_horizon, self.trajectory_history_features), dtype=np.float32)
        cos_heading = math.cos(current_heading)
        sin_heading = math.sin(current_heading)

        if history:
            recent_states = list(history)[-self.trajectory_history_horizon :]
            recent_states = recent_states[::-1]
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

    def _build_trajectory_history_observations(self):
        observations = np.zeros((self.num_agents, self.num_obs), dtype=np.float32)
        observations[:, : self._base_policy_num_obs] = self._sim_observations[:, : self._base_policy_num_obs]
        ego_hist_start = self._base_policy_num_obs
        partner_hist_start = ego_hist_start + self.trajectory_ego_history_dim

        ego_states = self._cached_global_agent_state
        partner_ids = self._cached_partner_ids

        for row in range(self.num_agents):
            ego_id = int(ego_states["id"][row])
            current_x = float(ego_states["x"][row])
            current_y = float(ego_states["y"][row])
            current_heading = float(ego_states["heading"][row])

            ego_block = self._encode_history_block(
                self._trajectory_entity_history.get(ego_id), current_x, current_y, current_heading
            )
            observations[row, ego_hist_start:partner_hist_start] = ego_block.reshape(-1)

            partner_block = np.zeros(
                (self.max_partner_objects, self.trajectory_history_horizon, self.trajectory_history_features),
                dtype=np.float32,
            )
            for slot in range(self.max_partner_objects):
                partner_id = int(partner_ids[row, slot])
                if partner_id <= 0:
                    continue
                partner_block[slot] = self._encode_history_block(
                    self._trajectory_entity_history.get(partner_id), current_x, current_y, current_heading
                )
            observations[row, partner_hist_start:] = partner_block.reshape(-1)
        return observations

    def _decode_trajectory_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape == (self.num_agents, self.trajectory_horizon, self.trajectory_features):
            return actions
        if actions.shape == (self.num_agents, self.trajectory_action_dim):
            return actions.reshape(self.num_agents, self.trajectory_horizon, self.trajectory_features)
        raise ValueError(
            f"Trajectory actions must have shape ({self.num_agents}, {self.trajectory_action_dim}) "
            f"or ({self.num_agents}, {self.trajectory_horizon}, {self.trajectory_features}). Got {actions.shape}"
        )

    def _trajectory_to_control_actions(self, trajectory_actions):
        if self._cached_global_agent_state is None:
            self._update_trajectory_history()

        low_level_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._last_trajectory_control_debug = []
        self._trajectory_smoothness_penalties.fill(0.0)
        current_speeds = self._sim_observations[:, 2] * _MAX_SPEED
        lengths = self._cached_global_agent_state["length"]

        for row in range(self.num_agents):
            traj = trajectory_actions[row]
            valid_mask = traj[:, 4] > 0.5
            if not np.any(valid_mask):
                continue

            decoded_xy = np.zeros((self.trajectory_horizon, 2), dtype=np.float32)
            decoded_speed = np.zeros(self.trajectory_horizon, dtype=np.float32)
            for step_idx in range(self.trajectory_horizon):
                decoded_xy[step_idx, 0] = _decode_trajectory_position(float(traj[step_idx, 0]))
                decoded_xy[step_idx, 1] = _decode_trajectory_position(float(traj[step_idx, 1]))
                decoded_speed[step_idx] = _decode_trajectory_speed(float(traj[step_idx, 3]))
            decoded_heading = np.asarray([_decode_trajectory_heading(float(value)) for value in traj[:, 2]], dtype=np.float32)

            decoded_xy = self._smooth_trajectory_positions(decoded_xy, valid_mask)
            self._trajectory_smoothness_penalties[row] = self._compute_trajectory_smoothness_penalty(
                decoded_xy=decoded_xy,
                decoded_heading=decoded_heading,
                decoded_speed=decoded_speed,
                valid_mask=valid_mask,
            )

            if self.trajectory_tracker_type == "mpc":
                requested_accel, requested_steer = self._mpc_trajectory_control(
                    decoded_xy=decoded_xy,
                    decoded_speed=decoded_speed,
                    valid_mask=valid_mask,
                    current_speed=float(current_speeds[row]),
                    wheelbase=max(float(lengths[row]), 1e-3),
                )
                applied_accel = _clip_float(float(requested_accel), float(self.trajectory_mpc_acceleration_grid.min()), float(self.trajectory_mpc_acceleration_grid.max()))
                applied_steer = _clip_float(float(requested_steer), float(self.trajectory_mpc_steering_grid.min()), float(self.trajectory_mpc_steering_grid.max()))
                accel_was_clamped = not math.isclose(applied_accel, float(requested_accel), rel_tol=0.0, abs_tol=1e-6)
                steer_was_clamped = not math.isclose(applied_steer, float(requested_steer), rel_tol=0.0, abs_tol=1e-6)
                if self.trajectory_control_is_physical:
                    low_level_actions[row, 0] = applied_accel
                    low_level_actions[row, 1] = applied_steer
                else:
                    low_level_actions[row, 0] = _clip_float(applied_accel / _MAX_CLASSIC_ACCELERATION, -1.0, 1.0)
                    low_level_actions[row, 1] = _clip_float(applied_steer / _MAX_CLASSIC_STEERING, -1.0, 1.0)
                self._last_trajectory_control_debug.append(
                    {
                        "agent_index": int(row),
                        "tracker_type": "mpc",
                        "controls_are_physical": bool(self.trajectory_control_is_physical),
                        "requested_acceleration": float(requested_accel),
                        "requested_steering": float(requested_steer),
                        "applied_acceleration": float(applied_accel),
                        "applied_steering": float(applied_steer),
                        "acceleration_was_clamped": bool(accel_was_clamped),
                        "steering_was_clamped": bool(steer_was_clamped),
                    }
                )
                continue

            lookahead_target = np.flatnonzero(valid_mask)[-1]
            for step_idx in np.flatnonzero(valid_mask):
                if float(np.linalg.norm(decoded_xy[step_idx])) >= self.trajectory_lookahead_distance:
                    lookahead_target = int(step_idx)
                    break

            if self.trajectory_tracker_type == "stanley":
                steering = self._stanley_trajectory_steering(
                    decoded_xy=decoded_xy,
                    valid_mask=valid_mask,
                    current_speed=float(current_speeds[row]),
                    wheelbase=max(float(lengths[row]), 1e-3),
                )
            else:
                target_x = float(decoded_xy[lookahead_target, 0])
                target_y = float(decoded_xy[lookahead_target, 1])
                lookahead_distance = max(float(np.linalg.norm(decoded_xy[lookahead_target])), 1e-3)
                alpha = math.atan2(target_y, target_x)
                curvature = (2.0 * math.sin(alpha)) / lookahead_distance
                steering = math.atan(curvature * max(float(lengths[row]), 1e-3))
            steering_cmd = _clip_float(
                (self.trajectory_steer_gain * steering) / _MAX_CLASSIC_STEERING,
                -1.0,
                1.0,
            )

            target_speed = float(decoded_speed[np.flatnonzero(valid_mask)[0]])
            speed_delta = target_speed - float(current_speeds[row])
            acceleration_cmd = _clip_float(
                (self.trajectory_accel_gain * speed_delta) / _MAX_CLASSIC_ACCELERATION,
                -1.0,
                1.0,
            )
            low_level_actions[row, 0] = acceleration_cmd
            low_level_actions[row, 1] = steering_cmd

        return low_level_actions

    def _compute_trajectory_smoothness_penalty(self, decoded_xy, decoded_heading, decoded_speed, valid_mask):
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size < 3:
            return 0.0

        contiguous_end = int(valid_indices[-1]) + 1
        if contiguous_end < 3:
            return 0.0

        penalty = 0.0
        if self.trajectory_smoothness_xy_weight > 0.0:
            deltas2_xy = decoded_xy[2:contiguous_end] - (2.0 * decoded_xy[1 : contiguous_end - 1]) + decoded_xy[: contiguous_end - 2]
            penalty -= self.trajectory_smoothness_xy_weight * float(np.linalg.norm(deltas2_xy, axis=1).mean())

        if self.trajectory_smoothness_heading_weight > 0.0:
            delta_heading_1 = np.asarray(
                [_wrap_to_pi(float(decoded_heading[idx + 1] - decoded_heading[idx])) for idx in range(contiguous_end - 1)],
                dtype=np.float32,
            )
            delta_heading_2 = np.asarray(
                [_wrap_to_pi(float(delta_heading_1[idx + 1] - delta_heading_1[idx])) for idx in range(delta_heading_1.shape[0] - 1)],
                dtype=np.float32,
            )
            if delta_heading_2.size > 0:
                penalty -= self.trajectory_smoothness_heading_weight * float(np.abs(delta_heading_2).mean())

        if self.trajectory_smoothness_speed_weight > 0.0:
            delta_speed_2 = decoded_speed[2:contiguous_end] - (2.0 * decoded_speed[1 : contiguous_end - 1]) + decoded_speed[: contiguous_end - 2]
            penalty -= self.trajectory_smoothness_speed_weight * float(np.abs(delta_speed_2).mean())

        return penalty

    def _smooth_trajectory_positions(self, decoded_xy, valid_mask):
        if self.trajectory_smoothing_window <= 1 or self.trajectory_smoothing_blend <= 0.0:
            return decoded_xy

        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size < 3:
            return decoded_xy

        horizon_xy = decoded_xy.copy()
        contiguous_end = int(valid_indices[-1]) + 1
        path_xy = horizon_xy[:contiguous_end].copy()
        window = min(self.trajectory_smoothing_window, contiguous_end)
        if window <= 1:
            return decoded_xy
        if window % 2 == 0:
            window -= 1
        if window <= 1:
            return decoded_xy

        kernel = np.ones(window, dtype=np.float32) / float(window)
        pad = window // 2
        for axis in range(2):
            padded = np.pad(path_xy[:, axis], (pad, pad), mode="edge")
            smoothed = np.convolve(padded, kernel, mode="valid")
            path_xy[:, axis] = (
                (1.0 - self.trajectory_smoothing_blend) * path_xy[:, axis]
                + self.trajectory_smoothing_blend * smoothed.astype(np.float32)
            )

        path_xy[0] = horizon_xy[0]
        horizon_xy[:contiguous_end] = path_xy
        return horizon_xy

    def _mpc_trajectory_control(self, decoded_xy, decoded_speed, valid_mask, current_speed, wheelbase):
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            return 0.0, 0.0

        horizon = min(self.trajectory_mpc_horizon_steps, int(valid_indices.size))
        target_indices = valid_indices[:horizon]
        target_xy = decoded_xy[target_indices]
        target_speed = decoded_speed[target_indices]
        target_heading = np.zeros(horizon, dtype=np.float32)
        for step_idx in range(horizon):
            if step_idx + 1 < horizon:
                delta = target_xy[step_idx + 1] - target_xy[step_idx]
            elif horizon > 1:
                delta = target_xy[step_idx] - target_xy[step_idx - 1]
            else:
                delta = target_xy[step_idx]
            target_heading[step_idx] = math.atan2(float(delta[1]), float(delta[0])) if float(np.dot(delta, delta)) > 1e-8 else 0.0

        best_cost = float("inf")
        best_action = (0.0, 0.0)
        for accel_cmd in self.trajectory_mpc_acceleration_grid:
            for steer_cmd in self.trajectory_mpc_steering_grid:
                cost = self._rollout_mpc_cost(
                    accel_cmd=float(accel_cmd),
                    steer_angle=float(steer_cmd),
                    target_xy=target_xy,
                    target_speed=target_speed,
                    target_heading=target_heading,
                    current_speed=current_speed,
                    wheelbase=wheelbase,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_action = (float(accel_cmd), float(steer_cmd))
        return best_action

    def _rollout_mpc_cost(
        self,
        *,
        accel_cmd,
        steer_angle,
        target_xy,
        target_speed,
        target_heading,
        current_speed,
        wheelbase,
    ):
        x = 0.0
        y = 0.0
        heading = 0.0
        speed = float(current_speed)
        accel = float(accel_cmd)
        cost = 0.0

        for step_idx in range(len(target_xy)):
            speed = _clip_float(speed + accel * self.dt, -2.0, 20.0)
            curvature = math.tan(steer_angle) / max(wheelbase, 1e-3)
            heading = _wrap_to_pi(heading + speed * curvature * self.dt)
            x += speed * math.cos(heading) * self.dt
            y += speed * math.sin(heading) * self.dt

            dx = x - float(target_xy[step_idx, 0])
            dy = y - float(target_xy[step_idx, 1])
            pos_cost = dx * dx + dy * dy
            heading_cost = _wrap_to_pi(heading - float(target_heading[step_idx])) ** 2
            speed_cost = (speed - float(target_speed[step_idx])) ** 2
            step_weight = 1.0 + 0.15 * step_idx
            cost += step_weight * (
                self.trajectory_mpc_position_weight * pos_cost
                + self.trajectory_mpc_heading_weight * heading_cost
                + self.trajectory_mpc_speed_weight * speed_cost
            )

        control_cost = ((accel_cmd / _MAX_CLASSIC_ACCELERATION) ** 2) + ((steer_angle / _MAX_CLASSIC_STEERING) ** 2)
        return cost + self.trajectory_mpc_control_weight * control_cost

    def _stanley_trajectory_steering(self, decoded_xy, valid_mask, current_speed, wheelbase):
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            return 0.0

        path_points = decoded_xy[valid_indices]
        if path_points.shape[0] == 1:
            target_x = float(path_points[0, 0])
            target_y = float(path_points[0, 1])
            lookahead_distance = max(float(np.linalg.norm(path_points[0])), 1e-3)
            alpha = math.atan2(target_y, target_x)
            curvature = (2.0 * math.sin(alpha)) / lookahead_distance
            return math.atan(curvature * wheelbase)

        front_axle = np.array([wheelbase, 0.0], dtype=np.float32)
        closest_distance = float("inf")
        closest_point = path_points[0]
        path_heading = math.atan2(float(path_points[1, 1] - path_points[0, 1]), float(path_points[1, 0] - path_points[0, 0]))

        for seg_idx in range(path_points.shape[0] - 1):
            start = path_points[seg_idx]
            end = path_points[seg_idx + 1]
            segment = end - start
            segment_norm_sq = float(np.dot(segment, segment))
            if segment_norm_sq <= 1e-8:
                continue
            projection = float(np.dot(front_axle - start, segment) / segment_norm_sq)
            projection = min(max(projection, 0.0), 1.0)
            projected_point = start + projection * segment
            delta = front_axle - projected_point
            distance = float(np.linalg.norm(delta))
            if distance < closest_distance:
                closest_distance = distance
                closest_point = projected_point
                path_heading = math.atan2(float(segment[1]), float(segment[0]))

        heading_error = _wrap_to_pi(path_heading)
        error_vector = front_axle - closest_point
        tangent = np.array([math.cos(path_heading), math.sin(path_heading)], dtype=np.float32)
        signed_cross_track = -float(tangent[0] * error_vector[1] - tangent[1] * error_vector[0])
        stanley_term = math.atan2(
            self.trajectory_stanley_gain * signed_cross_track,
            abs(float(current_speed)) + self.trajectory_stanley_softening,
        )
        return heading_error + stanley_term

    def _postprocess_observations(self):
        if self.is_trajectory_action and not self.is_trajectory_observation:
            if self._suspend_trajectory_history_updates:
                self._refresh_trajectory_state_cache()
            else:
                self._update_trajectory_history()
        if self.is_trajectory_observation:
            if self._suspend_trajectory_history_updates:
                self._refresh_trajectory_state_cache()
            else:
                self._update_trajectory_history()
            self.observations[:] = self._build_trajectory_history_observations()
            return
        if getattr(self, "observation_mode", 0) != 1:
            return
        self.observations[:] = postprocess_sdc_only_with_trailer_observations(
            sim_observations=self._sim_observations,
            ego_types=self.get_global_agent_types(),
            partner_types=self.get_partner_types(),
            ego_trailer_features=self.get_ego_trailer_obs_features(),
            base_ego_features=self._base_ego_features,
            base_partner_features=self._base_partner_features,
            max_partner_objects=self.max_partner_objects,
            max_road_objects=self.max_road_objects,
            road_features=self.road_features,
            type_classes=self.type_classes,
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

    def get_trajectory_targets(self, normalize=True):
        trajectories = self.get_ground_truth_trajectories()
        current_states = self._cached_global_agent_state
        if current_states is None:
            current_states = self.get_global_agent_state()

        targets = np.zeros((self.num_agents, self.trajectory_horizon, self.trajectory_features), dtype=np.float32)
        start_idx = int(self.tick) + 1

        for row in range(self.num_agents):
            current_x = float(current_states["x"][row])
            current_y = float(current_states["y"][row])
            current_heading = float(current_states["heading"][row])
            cos_heading = math.cos(current_heading)
            sin_heading = math.sin(current_heading)

            for step_idx in range(self.trajectory_horizon):
                traj_idx = start_idx + step_idx
                if traj_idx >= trajectories["x"].shape[2]:
                    break
                valid = float(trajectories["valid"][row, 0, traj_idx])
                if valid <= 0.0:
                    continue

                world_x = float(trajectories["x"][row, 0, traj_idx])
                world_y = float(trajectories["y"][row, 0, traj_idx])
                dx = world_x - current_x
                dy = world_y - current_y
                rel_x = dx * cos_heading + dy * sin_heading
                rel_y = -dx * sin_heading + dy * cos_heading
                rel_heading = _wrap_to_pi(float(trajectories["heading"][row, 0, traj_idx]) - current_heading)

                prev_idx = max(traj_idx - 1, 0)
                prev_x = float(trajectories["x"][row, 0, prev_idx])
                prev_y = float(trajectories["y"][row, 0, prev_idx])
                speed = math.sqrt((world_x - prev_x) ** 2 + (world_y - prev_y) ** 2) / max(float(self.dt), 1e-6)

                if normalize:
                    targets[row, step_idx, 0] = _encode_trajectory_position(rel_x)
                    targets[row, step_idx, 1] = _encode_trajectory_position(rel_y)
                    targets[row, step_idx, 2] = _encode_trajectory_heading(rel_heading)
                    targets[row, step_idx, 3] = _encode_trajectory_speed(speed)
                    targets[row, step_idx, 4] = valid
                else:
                    targets[row, step_idx, 0] = rel_x
                    targets[row, step_idx, 1] = rel_y
                    targets[row, step_idx, 2] = rel_heading
                    targets[row, step_idx, 3] = speed
                    targets[row, step_idx, 4] = valid

        if normalize:
            return targets.reshape(self.num_agents, self.trajectory_action_dim)
        return targets

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
            if isinstance(track, dict):
                track_index = track.get("track_index", -1)
            else:
                track_index = int(track)
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
    write_manifest=True,
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
    manifest_rows = []
    for i, map_path in enumerate(json_files[:max_maps]):
        binary_file = f"map_{i:03d}.bin"
        binary_path = binary_dir / binary_file
        tasks.append((i, map_path, binary_path))
        manifest_rows.append(
            {
                "map_id": i,
                "relative_path": binary_file,
                "source_name": map_path.name,
            }
        )

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

    if write_manifest:
        manifest_path = binary_dir / "dataset_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "dataset_name": dataset_name,
                    "source_data_dir": str(data_dir.resolve()),
                    "maps": manifest_rows,
                },
                file_obj,
                indent=2,
            )


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
    
    parser = argparse.ArgumentParser(description="Convert JSON map files to binary format")
    parser.add_argument("--input-folder", type=str, default="data/processed/training",
                        help="Path to folder containing JSON map files")
    parser.add_argument("--output-folder", type=str, default=None,
                        help="Path to save binary files (default: resources/drive/binaries/{dataset_name})")
    parser.add_argument("--max-maps", type=int, default=50_000,
                        help="Maximum number of maps to process")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores)")
    
    args = parser.parse_args()
    
    process_all_maps(
        data_folder=args.input_folder,
        output_folder=args.output_folder,
        max_maps=args.max_maps,
        num_workers=args.num_workers
    )
