from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from pufferlib.ocean.drive.drive import binding, postprocess_sdc_only_with_trailer_observations

LOGGER = logging.getLogger("export_paired_offline_fits")

FIT_BEAM_WIDTH = 8
FIT_MATCH_WEIGHT_LATERAL = 2.5
FIT_MATCH_WEIGHT_LONGITUDINAL = 1.5
FIT_MATCH_WEIGHT_HEADING = 0.1
FIT_MATCH_WEIGHT_SPEED = 0.02
FIT_MATCH_WEIGHT_STEER_CHANGE = 0.15
FIT_MATCH_WEIGHT_ACCEL_CHANGE = 0.02
FIT_MATCH_WEIGHT_REVERSE = 1.0
FIT_MATCH_WEIGHT_PROGRESS = 4.0
FIT_MATCH_WEIGHT_STEER_FLIP = 0.5
FIT_MATCH_WEIGHT_REF_ACCEL = 0.01
FIT_MATCH_WEIGHT_REF_STEER = 0.1

OFFLINE_FIT_CONTROL_MODE = 3
OFFLINE_FIT_GOAL_BEHAVIOR = 2
OFFLINE_FIT_COLLISION_BEHAVIOR = 0
OFFLINE_FIT_OFFROAD_BEHAVIOR = 0
OFFLINE_FIT_TERMINATION_MODE = 0
OFFLINE_FIT_GOAL_TARGET_DISTANCE = 30.0
OFFLINE_FIT_GOAL_RADIUS = 0.2
OFFLINE_FIT_GOAL_SPEED = 100.0
OFFLINE_FIT_DT = 0.1
OFFLINE_FIT_EPISODE_LENGTH = 91
OFFLINE_FIT_INIT_STEPS = 0

PAIRED_FITS_FORMAT_MONOLITHIC = "paired_fits_v1"
PAIRED_FITS_FORMAT_SHARDED = "sharded_paired_fits_v1"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_LOG_EVERY = 25

CAR_ROOT = Path("pufferlib/resources/drive/binaries/nuplanCarBostonTest10")
TRUCK_ROOT = Path("pufferlib/resources/drive/binaries/nuplanTruckBostonTest10")
DEFAULT_OUTPUT = Path("outputs/offline_fits/nuplan_boston_test10_paired_fits.pt")
OBS_MODE_DEFAULT = "default"
OBS_MODE_EXTENDED = "sdc_only_with_trailer"


def _obs_dim() -> int:
    return (
        binding.EGO_FEATURES_CLASSIC
        + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
        + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    )


def _obs_dim_extended() -> int:
    return (
        binding.EGO_FEATURES_CLASSIC
        + 1
        + 4
        + (binding.MAX_AGENTS - 1) * (binding.PARTNER_FEATURES + 1)
        + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    )


def _discover_shared_maps(car_root: Path, truck_root: Path, max_maps: int | None = None) -> list[str]:
    shared = sorted(set(p.name for p in car_root.glob("map_*.bin")) & set(p.name for p in truck_root.glob("map_*.bin")))
    if max_maps is not None and max_maps > 0:
        shared = shared[:max_maps]
    return shared


def _iter_chunks(items: list[str], chunk_size: int):
    effective_chunk_size = max(1, int(chunk_size))
    for start in range(0, len(items), effective_chunk_size):
        end = min(len(items), start + effective_chunk_size)
        yield start, end, items[start:end]


def _fit_settings_metadata() -> dict[str, object]:
    return {
        "control_mode": OFFLINE_FIT_CONTROL_MODE,
        "goal_behavior": OFFLINE_FIT_GOAL_BEHAVIOR,
        "collision_behavior": OFFLINE_FIT_COLLISION_BEHAVIOR,
        "offroad_behavior": OFFLINE_FIT_OFFROAD_BEHAVIOR,
        "termination_mode": OFFLINE_FIT_TERMINATION_MODE,
        "dt": OFFLINE_FIT_DT,
        "episode_length": OFFLINE_FIT_EPISODE_LENGTH,
        "saved_observation_modes": [OBS_MODE_DEFAULT, OBS_MODE_EXTENDED],
    }


def _optimizer_metadata() -> dict[str, object]:
    return {
        "beam_width": FIT_BEAM_WIDTH,
        "match_weight_lateral": FIT_MATCH_WEIGHT_LATERAL,
        "match_weight_longitudinal": FIT_MATCH_WEIGHT_LONGITUDINAL,
        "match_weight_heading": FIT_MATCH_WEIGHT_HEADING,
        "match_weight_speed": FIT_MATCH_WEIGHT_SPEED,
        "match_weight_steer_change": FIT_MATCH_WEIGHT_STEER_CHANGE,
        "match_weight_accel_change": FIT_MATCH_WEIGHT_ACCEL_CHANGE,
        "match_weight_reverse": FIT_MATCH_WEIGHT_REVERSE,
        "match_weight_progress": FIT_MATCH_WEIGHT_PROGRESS,
    }


def _build_export_metadata(car_root: Path, truck_root: Path, shared_maps: list[str]) -> dict[str, object]:
    return {
        "car_root": str(car_root),
        "truck_root": str(truck_root),
        "shared_maps": shared_maps,
        "fit_settings": _fit_settings_metadata(),
        "optimizer": _optimizer_metadata(),
    }


def load_paired_fit_manifest(export_path: Path) -> dict:
    payload = torch.load(export_path, map_location="cpu")
    if isinstance(payload, dict) and payload.get("format") == PAIRED_FITS_FORMAT_SHARDED:
        return payload
    if isinstance(payload, dict) and "pairs" in payload and "metadata" in payload:
        return {
            "format": PAIRED_FITS_FORMAT_MONOLITHIC,
            "metadata": payload["metadata"],
            "shards": [
                {
                    "path": str(export_path),
                    "map_count": int(len(payload["pairs"])),
                    "start_index": 0,
                    "end_index": int(len(payload["pairs"])),
                }
            ],
        }
    raise ValueError(f"unsupported paired fit payload at {export_path}")


def iter_paired_fit_shards(export_path: Path):
    manifest = load_paired_fit_manifest(export_path)
    if manifest["format"] == PAIRED_FITS_FORMAT_MONOLITHIC:
        payload = torch.load(export_path, map_location="cpu")
        yield manifest["metadata"], payload["pairs"], manifest["shards"][0]
        return

    for shard_info in manifest.get("shards", []):
        shard_path = Path(shard_info["path"])
        if not shard_path.exists():
            candidate_same_dir = export_path.parent / shard_path.name
            candidate_sibling_dir = export_path.parent / f"{export_path.stem}_shards" / shard_path.name
            if candidate_same_dir.exists():
                shard_path = candidate_same_dir
            elif candidate_sibling_dir.exists():
                shard_path = candidate_sibling_dir
        LOGGER.info(
            "Loading paired fit shard | path=%s map_count=%s",
            shard_path,
            shard_info.get("map_count"),
        )
        shard_payload = torch.load(shard_path, map_location="cpu")
        yield manifest["metadata"], shard_payload["pairs"], shard_info


def _summarize_pairs(pairs: dict[str, dict]) -> dict[str, int]:
    return {
        "map_count": int(len(pairs)),
        "successful_pairs": int(
            sum(
                1
                for pair in pairs.values()
                if pair["car"]["status"] == "ok" and pair["truck"]["status"] == "ok"
            )
        ),
        "successful_truck_context_replays": int(
            sum(1 for pair in pairs.values() if pair["truck_context_replay"]["status"] == "ok")
        ),
    }


def _write_paired_fit_summary(
    output_path: Path,
    shared_map_count: int,
    successful_pairs: int,
    successful_truck_context_replays: int,
    shard_count: int,
) -> None:
    summary = {
        "output_path": str(output_path),
        "shared_map_count": int(shared_map_count),
        "successful_pairs": int(successful_pairs),
        "successful_truck_context_replays": int(successful_truck_context_replays),
        "shard_count": int(shard_count),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))


def _init_env(map_dir: Path):
    obs = np.zeros((1, _obs_dim()), dtype=np.float32)
    actions = np.zeros(1, dtype=np.int32)
    rewards = np.zeros(1, dtype=np.float32)
    terminals = np.zeros(1, dtype=np.uint8)
    truncs = np.zeros(1, dtype=np.uint8)
    env_handle = binding.env_init(
        obs,
        actions,
        rewards,
        terminals,
        truncs,
        0,
        human_agent_idx=0,
        ini_file="pufferlib/config/ocean/drive.ini",
        map_dir=str(map_dir),
        map_id=0,
        max_agents=1,
        max_controlled_agents=1,
        init_steps=OFFLINE_FIT_INIT_STEPS,
        init_mode=0,
        control_mode=OFFLINE_FIT_CONTROL_MODE,
        goal_behavior=OFFLINE_FIT_GOAL_BEHAVIOR,
        goal_target_distance=OFFLINE_FIT_GOAL_TARGET_DISTANCE,
        goal_radius=OFFLINE_FIT_GOAL_RADIUS,
        goal_speed=OFFLINE_FIT_GOAL_SPEED,
        collision_behavior=OFFLINE_FIT_COLLISION_BEHAVIOR,
        offroad_behavior=OFFLINE_FIT_OFFROAD_BEHAVIOR,
        termination_mode=OFFLINE_FIT_TERMINATION_MODE,
        dt=OFFLINE_FIT_DT,
        episode_length=OFFLINE_FIT_EPISODE_LENGTH,
        dynamics_model=0,
        force_zero_trailer_articulation_at_init=0,
    )
    return env_handle, obs, actions, rewards, terminals, truncs


def _stage_map(source_bin: Path) -> tempfile.TemporaryDirectory[str]:
    tmp_dir = tempfile.TemporaryDirectory(prefix=f"offline_fit_{source_bin.stem}_")
    staged_map = Path(tmp_dir.name) / "map_000.bin"
    shutil.copy2(source_bin, staged_map)
    return tmp_dir


def _extract_ground_truth(env_handle, active_count: int):
    gt_x_full = np.zeros((active_count, OFFLINE_FIT_EPISODE_LENGTH), dtype=np.float32)
    gt_y_full = np.zeros((active_count, OFFLINE_FIT_EPISODE_LENGTH), dtype=np.float32)
    gt_z_full = np.zeros((active_count, OFFLINE_FIT_EPISODE_LENGTH), dtype=np.float32)
    gt_heading_full = np.zeros((active_count, OFFLINE_FIT_EPISODE_LENGTH), dtype=np.float32)
    gt_valid_full = np.zeros((active_count, OFFLINE_FIT_EPISODE_LENGTH), dtype=np.int32)
    gt_ids = np.zeros(active_count, dtype=np.int32)
    gt_is_vehicle = np.zeros(active_count, dtype=np.int32)
    gt_scenario_id = np.zeros(active_count, dtype=np.int32)
    binding.get_ground_truth_trajectories(
        env_handle,
        gt_x_full,
        gt_y_full,
        gt_z_full,
        gt_heading_full,
        gt_valid_full,
        gt_ids,
        gt_is_vehicle,
        gt_scenario_id,
    )
    valid = gt_valid_full[0].astype(bool)
    return {
        "x": gt_x_full[0][valid],
        "y": gt_y_full[0][valid],
        "z": gt_z_full[0][valid],
        "heading": gt_heading_full[0][valid],
        "valid": gt_valid_full[0],
        "scenario_id": int(gt_scenario_id[0]) if active_count > 0 else -1,
        "agent_id": int(gt_ids[0]) if active_count > 0 else -1,
        "is_vehicle": int(gt_is_vehicle[0]) if active_count > 0 else 0,
    }


def _record_observation(env_handle, active_count: int) -> dict:
    timestep_obs = np.zeros((active_count, _obs_dim()), dtype=np.float32)
    binding.env_copy_observations(env_handle, timestep_obs)
    ego_types = np.zeros(active_count, dtype=np.int32)
    partner_types = np.zeros((active_count, binding.MAX_AGENTS - 1), dtype=np.int32)
    trailer_features = {
        "rel_x": np.zeros(active_count, dtype=np.float32),
        "rel_y": np.zeros(active_count, dtype=np.float32),
        "rel_heading_x": np.zeros(active_count, dtype=np.float32),
        "rel_heading_y": np.zeros(active_count, dtype=np.float32),
    }
    binding.get_global_agent_types(env_handle, ego_types)
    binding.get_partner_types(env_handle, partner_types)
    binding.get_ego_trailer_obs_features(
        env_handle,
        trailer_features["rel_x"],
        trailer_features["rel_y"],
        trailer_features["rel_heading_x"],
        trailer_features["rel_heading_y"],
    )
    extended_obs = postprocess_sdc_only_with_trailer_observations(
        sim_observations=timestep_obs,
        ego_types=ego_types,
        partner_types=partner_types,
        ego_trailer_features=trailer_features,
        base_ego_features=binding.EGO_FEATURES_CLASSIC,
        base_partner_features=binding.PARTNER_FEATURES,
        max_partner_objects=binding.MAX_AGENTS - 1,
        max_road_objects=binding.MAX_ROAD_SEGMENT_OBSERVATIONS,
        road_features=binding.ROAD_FEATURES,
        type_classes=binding.POLICY_TYPE_CLASS_COUNT,
    )
    return {
        OBS_MODE_DEFAULT: timestep_obs[0].copy(),
        OBS_MODE_EXTENDED: extended_obs[0].copy(),
    }


def _record_agent_state(env_handle):
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
    return (
        float(state_x[0]),
        float(state_y[0]),
        float(state_heading[0]),
    )


def _compute_displacement_metrics(ref_x: np.ndarray, ref_y: np.ndarray, cand_x: np.ndarray, cand_y: np.ndarray) -> dict:
    aligned_steps = min(len(ref_x), len(cand_x), len(ref_y), len(cand_y))
    displacement = np.sqrt((cand_x[:aligned_steps] - ref_x[:aligned_steps]) ** 2 + (cand_y[:aligned_steps] - ref_y[:aligned_steps]) ** 2)
    return {
        "aligned_steps": int(aligned_steps),
        "displacement_per_step": displacement.astype(np.float32),
        "ade": float(displacement.mean()) if displacement.size else float("nan"),
        "fde": float(displacement[-1]) if displacement.size else float("nan"),
    }


def _replay_action_sequence(map_dir: Path, fit_actions: np.ndarray, gt: dict | None = None) -> dict:
    env_handle, obs, actions, rewards, terminals, truncs = _init_env(map_dir)
    try:
        binding.env_reset(env_handle, 0)
        active_count = binding.env_get_active_agent_count(env_handle)
        if active_count <= 0:
            raise RuntimeError("no_active_agents")

        rollout_obs = []
        rollout_obs_ext = []
        rollout_actions = []
        rollout_timestep = []
        start_x, start_y, start_heading = _record_agent_state(env_handle)
        rollout_x = [start_x]
        rollout_y = [start_y]
        rollout_heading = [start_heading]

        for step_idx, action in enumerate(fit_actions):
            obs_record = _record_observation(env_handle, active_count)
            rollout_obs.append(obs_record[OBS_MODE_DEFAULT])
            rollout_obs_ext.append(obs_record[OBS_MODE_EXTENDED])
            rollout_actions.append(int(action))
            rollout_timestep.append(OFFLINE_FIT_INIT_STEPS + step_idx)

            actions[0] = int(action)
            binding.env_step(env_handle)
            if int(terminals[0]) != 0 or int(truncs[0]) != 0:
                break

            cur_x, cur_y, cur_heading = _record_agent_state(env_handle)
            rollout_x.append(cur_x)
            rollout_y.append(cur_y)
            rollout_heading.append(cur_heading)

        replay = {
            "status": "ok",
            "num_steps": len(rollout_actions),
            "obs": np.stack(rollout_obs).astype(np.float32)
            if rollout_obs
            else np.zeros((0, _obs_dim()), dtype=np.float32),
            "obs_default": np.stack(rollout_obs).astype(np.float32)
            if rollout_obs
            else np.zeros((0, _obs_dim()), dtype=np.float32),
            "obs_sdc_only_with_trailer": np.stack(rollout_obs_ext).astype(np.float32)
            if rollout_obs_ext
            else np.zeros((0, _obs_dim_extended()), dtype=np.float32),
            "actions": np.asarray(rollout_actions, dtype=np.int32),
            "timestep": np.asarray(rollout_timestep, dtype=np.int32),
            "rollout_x": np.asarray(rollout_x, dtype=np.float32),
            "rollout_y": np.asarray(rollout_y, dtype=np.float32),
            "rollout_heading": np.asarray(rollout_heading, dtype=np.float32),
        }
        if gt is not None:
            metrics = _compute_displacement_metrics(
                np.asarray(gt["x"], dtype=np.float32),
                np.asarray(gt["y"], dtype=np.float32),
                replay["rollout_x"],
                replay["rollout_y"],
            )
            replay["self_displacement_per_step"] = metrics["displacement_per_step"]
            replay["self_ade"] = metrics["ade"]
            replay["self_fde"] = metrics["fde"]
        return replay
    finally:
        binding.env_close(env_handle)


def _fit_side(source_bin: Path) -> dict:
    staged = _stage_map(source_bin)
    map_dir = Path(staged.name)
    try:
        try:
            env_handle, obs, actions, rewards, terminals, truncs = _init_env(map_dir)
            try:
                binding.env_reset(env_handle, 0)
                active_count = binding.env_get_active_agent_count(env_handle)
                if active_count <= 0:
                    raise RuntimeError("no_active_agents")

                scenario_ids = np.zeros(active_count, dtype=np.int32)
                agent_ids = np.zeros(active_count, dtype=np.int32)
                binding.env_get_active_agent_info(env_handle, scenario_ids, agent_ids)

                gt = _extract_ground_truth(env_handle, active_count)
                planning_horizon = max(1, int(gt["x"].shape[0]) - 1)

                fit_actions = np.full(OFFLINE_FIT_EPISODE_LENGTH - 1, -1, dtype=np.int32)
                step_costs = np.zeros(OFFLINE_FIT_EPISODE_LENGTH - 1, dtype=np.float32)
                step_lat_costs = np.zeros(OFFLINE_FIT_EPISODE_LENGTH - 1, dtype=np.float32)
                step_lon_costs = np.zeros(OFFLINE_FIT_EPISODE_LENGTH - 1, dtype=np.float32)
                num_steps, total_cost, total_lat_cost, total_lon_cost = binding.env_fit_discrete_action_sequence(
                    env_handle,
                    0,
                    FIT_BEAM_WIDTH,
                    planning_horizon,
                    FIT_MATCH_WEIGHT_LATERAL,
                    FIT_MATCH_WEIGHT_LONGITUDINAL,
                    FIT_MATCH_WEIGHT_HEADING,
                    FIT_MATCH_WEIGHT_SPEED,
                    FIT_MATCH_WEIGHT_STEER_CHANGE,
                    FIT_MATCH_WEIGHT_ACCEL_CHANGE,
                    FIT_MATCH_WEIGHT_REVERSE,
                    FIT_MATCH_WEIGHT_PROGRESS,
                    FIT_MATCH_WEIGHT_STEER_FLIP,
                    FIT_MATCH_WEIGHT_REF_ACCEL,
                    FIT_MATCH_WEIGHT_REF_STEER,
                    fit_actions,
                    step_costs,
                    step_lat_costs,
                    step_lon_costs,
                )

                logged_obs = []
                logged_obs_ext = []
                for step_idx in range(int(num_steps)):
                    binding.env_set_logged_timestep(env_handle, OFFLINE_FIT_INIT_STEPS + step_idx)
                    obs_record = _record_observation(env_handle, active_count)
                    logged_obs.append(obs_record[OBS_MODE_DEFAULT])
                    logged_obs_ext.append(obs_record[OBS_MODE_EXTENDED])
            finally:
                binding.env_close(env_handle)

            replay = _replay_action_sequence(map_dir, fit_actions[:num_steps], gt=gt)
            return {
                "source_map": source_bin.name,
                "status": "ok",
                "scenario_id": int(scenario_ids[0]),
                "agent_id": int(agent_ids[0]),
                "num_steps": int(num_steps),
                "actions": fit_actions[:num_steps].astype(np.int32),
                "logged_obs": np.stack(logged_obs).astype(np.float32)
                if logged_obs
                else np.zeros((0, _obs_dim()), dtype=np.float32),
                "logged_obs_default": np.stack(logged_obs).astype(np.float32)
                if logged_obs
                else np.zeros((0, _obs_dim()), dtype=np.float32),
                "logged_obs_sdc_only_with_trailer": np.stack(logged_obs_ext).astype(np.float32)
                if logged_obs_ext
                else np.zeros((0, _obs_dim_extended()), dtype=np.float32),
                "logged_timestep": np.arange(OFFLINE_FIT_INIT_STEPS, OFFLINE_FIT_INIT_STEPS + int(num_steps), dtype=np.int32),
                "match_cost_step": step_costs[:num_steps].astype(np.float32),
                "match_cost_lateral_step": step_lat_costs[:num_steps].astype(np.float32),
                "match_cost_longitudinal_step": step_lon_costs[:num_steps].astype(np.float32),
                "match_cost_total": float(total_cost),
                "match_cost_lateral_total": float(total_lat_cost),
                "match_cost_longitudinal_total": float(total_lon_cost),
                "gt_x": gt["x"].astype(np.float32),
                "gt_y": gt["y"].astype(np.float32),
                "gt_heading": gt["heading"].astype(np.float32),
                "rollout_x": replay["rollout_x"],
                "rollout_y": replay["rollout_y"],
                "rollout_heading": replay["rollout_heading"],
                "self_displacement_per_step": replay["self_displacement_per_step"],
                "self_ade": replay["self_ade"],
                "self_fde": replay["self_fde"],
            }
        except Exception as exc:
            return {"source_map": source_bin.name, "status": "unavailable", "error": str(exc)}
    finally:
        staged.cleanup()


def _pair_metrics(car_side: dict, truck_side: dict) -> dict:
    if car_side["status"] != "ok" or truck_side["status"] != "ok":
        return {"status": "unavailable"}
    aligned_steps = min(len(car_side["rollout_x"]), len(truck_side["rollout_x"]))
    displacement = np.sqrt(
        (car_side["rollout_x"][:aligned_steps] - truck_side["rollout_x"][:aligned_steps]) ** 2
        + (car_side["rollout_y"][:aligned_steps] - truck_side["rollout_y"][:aligned_steps]) ** 2
    )
    return {
        "status": "ok",
        "aligned_steps": int(aligned_steps),
        "displacement_per_step": displacement.astype(np.float32),
        "ade": float(displacement.mean()) if displacement.size else float("nan"),
        "fde": float(displacement[-1]) if displacement.size else float("nan"),
    }


def _truck_context_replay(truck_bin: Path, car_side: dict, truck_side: dict) -> dict:
    if car_side["status"] != "ok" or truck_side["status"] != "ok":
        return {"status": "unavailable"}

    staged = _stage_map(truck_bin)
    map_dir = Path(staged.name)
    try:
        truck_gt = {
            "x": truck_side["gt_x"],
            "y": truck_side["gt_y"],
        }
        truck_branch = _replay_action_sequence(map_dir, truck_side["actions"], gt=truck_gt)
        car_branch = _replay_action_sequence(map_dir, car_side["actions"], gt=truck_gt)
        pair_similarity = _pair_metrics(
            {
                "status": "ok",
                "rollout_x": car_branch["rollout_x"],
                "rollout_y": car_branch["rollout_y"],
            },
            {
                "status": "ok",
                "rollout_x": truck_branch["rollout_x"],
                "rollout_y": truck_branch["rollout_y"],
            },
        )
        return {
            "status": "ok",
            "truck_branch": truck_branch,
            "car_branch": car_branch,
            "pair_similarity": pair_similarity,
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}
    finally:
        staged.cleanup()


def export_paired_fits(
    car_root: Path,
    truck_root: Path,
    output_path: Path,
    max_maps: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    log_every: int = DEFAULT_LOG_EVERY,
) -> Path:
    shared_maps = _discover_shared_maps(car_root, truck_root, max_maps=max_maps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _build_export_metadata(car_root, truck_root, shared_maps)
    effective_chunk_size = max(1, int(chunk_size))
    effective_log_every = max(1, int(log_every))

    LOGGER.info(
        "Preparing paired offline fit export | shared_maps=%d chunk_size=%d log_every=%d output=%s",
        len(shared_maps),
        effective_chunk_size,
        effective_log_every,
        output_path,
    )

    if len(shared_maps) <= effective_chunk_size:
        pairs: dict[str, dict] = {}
        for map_index, map_name in enumerate(shared_maps, start=1):
            car_side = _fit_side(car_root / map_name)
            truck_side = _fit_side(truck_root / map_name)
            pairs[map_name] = {
                "car": car_side,
                "truck": truck_side,
                "pair_similarity": _pair_metrics(car_side, truck_side),
                "truck_context_replay": _truck_context_replay(truck_root / map_name, car_side, truck_side),
            }
            if map_index % effective_log_every == 0 or map_index == len(shared_maps):
                LOGGER.info("Paired fit progress | processed=%d/%d", map_index, len(shared_maps))

        payload: dict[str, object] = {
            "format": PAIRED_FITS_FORMAT_MONOLITHIC,
            "metadata": metadata,
            "pairs": pairs,
        }
        torch.save(payload, output_path)
        summary = _summarize_pairs(pairs)
        _write_paired_fit_summary(
            output_path=output_path,
            shared_map_count=len(shared_maps),
            successful_pairs=summary["successful_pairs"],
            successful_truck_context_replays=summary["successful_truck_context_replays"],
            shard_count=1,
        )
        LOGGER.info(
            "Finished paired offline fit export | output=%s shared_maps=%d shard_count=1",
            output_path,
            len(shared_maps),
        )
        return output_path

    shard_dir = output_path.parent / f"{output_path.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": PAIRED_FITS_FORMAT_SHARDED,
        "metadata": metadata,
        "shards": [],
    }
    total_successful_pairs = 0
    total_successful_truck_context_replays = 0
    total_chunks = (len(shared_maps) + effective_chunk_size - 1) // effective_chunk_size

    for chunk_index, (start_index, end_index, chunk_maps) in enumerate(
        _iter_chunks(shared_maps, effective_chunk_size),
        start=1,
    ):
        LOGGER.info(
            "Starting paired fit shard | shard=%d/%d maps=%d range=[%d,%d)",
            chunk_index,
            total_chunks,
            len(chunk_maps),
            start_index,
            end_index,
        )
        chunk_pairs: dict[str, dict] = {}
        for local_index, map_name in enumerate(chunk_maps, start=1):
            car_side = _fit_side(car_root / map_name)
            truck_side = _fit_side(truck_root / map_name)
            chunk_pairs[map_name] = {
                "car": car_side,
                "truck": truck_side,
                "pair_similarity": _pair_metrics(car_side, truck_side),
                "truck_context_replay": _truck_context_replay(truck_root / map_name, car_side, truck_side),
            }
            overall_index = start_index + local_index
            if local_index % effective_log_every == 0 or local_index == len(chunk_maps):
                LOGGER.info(
                    "Paired fit shard progress | shard=%d/%d local=%d/%d overall=%d/%d",
                    chunk_index,
                    total_chunks,
                    local_index,
                    len(chunk_maps),
                    overall_index,
                    len(shared_maps),
                )

        shard_path = shard_dir / f"{output_path.stem}.part{chunk_index:05d}.pt"
        torch.save(
            {
                "format": "paired_fit_shard_v1",
                "map_names": chunk_maps,
                "pairs": chunk_pairs,
            },
            shard_path,
        )
        chunk_summary = _summarize_pairs(chunk_pairs)
        total_successful_pairs += chunk_summary["successful_pairs"]
        total_successful_truck_context_replays += chunk_summary["successful_truck_context_replays"]
        manifest["shards"].append(
            {
                "path": str(shard_path),
                "map_count": int(len(chunk_maps)),
                "start_index": int(start_index),
                "end_index": int(end_index),
                "successful_pairs": int(chunk_summary["successful_pairs"]),
                "successful_truck_context_replays": int(chunk_summary["successful_truck_context_replays"]),
            }
        )
        LOGGER.info(
            "Finished paired fit shard | shard=%d/%d output=%s maps=%d",
            chunk_index,
            total_chunks,
            shard_path,
            len(chunk_maps),
        )

    torch.save(manifest, output_path)
    _write_paired_fit_summary(
        output_path=output_path,
        shared_map_count=len(shared_maps),
        successful_pairs=total_successful_pairs,
        successful_truck_context_replays=total_successful_truck_context_replays,
        shard_count=len(manifest["shards"]),
    )
    LOGGER.info(
        "Finished paired offline fit export | output=%s shared_maps=%d shard_count=%d",
        output_path,
        len(shared_maps),
        len(manifest["shards"]),
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export paired truck/car offline fitted trajectories.")
    parser.add_argument("--car-root", type=Path, default=CAR_ROOT)
    parser.add_argument("--truck-root", type=Path, default=TRUCK_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-maps", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_path = export_paired_fits(
        args.car_root,
        args.truck_root,
        args.output,
        max_maps=(args.max_maps if args.max_maps > 0 else None),
        chunk_size=args.chunk_size,
        log_every=args.log_every,
    )
    print(output_path)


if __name__ == "__main__":
    main()
