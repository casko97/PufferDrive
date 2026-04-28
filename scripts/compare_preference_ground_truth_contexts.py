#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import DEFAULT_SDC_RUNTIME_TRUCK_REF_BIN, _load_non_kinematic_vehicle_params_from_bin
from scripts import compare_preference_rollout_models as rollout_compare
from scripts.export_paired_offline_fits import (
    FIT_BEAM_WIDTH,
    FIT_MATCH_WEIGHT_ACCEL_CHANGE,
    FIT_MATCH_WEIGHT_HEADING,
    FIT_MATCH_WEIGHT_LATERAL,
    FIT_MATCH_WEIGHT_LONGITUDINAL,
    FIT_MATCH_WEIGHT_PROGRESS,
    FIT_MATCH_WEIGHT_REF_ACCEL,
    FIT_MATCH_WEIGHT_REF_STEER,
    FIT_MATCH_WEIGHT_REVERSE,
    FIT_MATCH_WEIGHT_SPEED,
    FIT_MATCH_WEIGHT_STEER_CHANGE,
    FIT_MATCH_WEIGHT_STEER_FLIP,
    OBS_MODE_DEFAULT,
    OFFLINE_FIT_COLLISION_BEHAVIOR,
    OFFLINE_FIT_CONTROL_MODE,
    OFFLINE_FIT_DT,
    OFFLINE_FIT_EPISODE_LENGTH,
    OFFLINE_FIT_GOAL_BEHAVIOR,
    OFFLINE_FIT_GOAL_RADIUS,
    OFFLINE_FIT_GOAL_SPEED,
    OFFLINE_FIT_GOAL_TARGET_DISTANCE,
    OFFLINE_FIT_INIT_STEPS,
    OFFLINE_FIT_OFFROAD_BEHAVIOR,
    OFFLINE_FIT_TERMINATION_MODE,
    _extract_ground_truth,
    _obs_dim,
    _record_observation,
    _record_agent_state,
)


DEFAULT_SOURCE_MAP_DIR = Path("datasets/nuplanCarBostonAll_training")
DEFAULT_OUTPUT_DIR = Path("outputs/preference_ground_truth_context_compare")


@dataclass(frozen=True)
class GroundTruthContextSpec:
    name: str
    dynamics_model: int
    force_zero_trailer_articulation_at_init: int
    non_kinematic_vehicle_params_override: tuple[float, ...] | None = None


def _stage_map(source_bin: Path) -> tempfile.TemporaryDirectory[str]:
    tmp_dir = tempfile.TemporaryDirectory(prefix=f"gt_fit_{source_bin.stem}_")
    staged_map = Path(tmp_dir.name) / "map_000.bin"
    shutil.copy2(source_bin, staged_map)
    return tmp_dir


def _context_specs() -> list[GroundTruthContextSpec]:
    truck_params = _load_non_kinematic_vehicle_params_from_bin(DEFAULT_SDC_RUNTIME_TRUCK_REF_BIN)
    return [
        GroundTruthContextSpec(
            name="ground-truth-car-fit",
            dynamics_model=0,
            force_zero_trailer_articulation_at_init=0,
            non_kinematic_vehicle_params_override=None,
        ),
        GroundTruthContextSpec(
            name="ground-truth-truck-context-replay",
            dynamics_model=2,
            force_zero_trailer_articulation_at_init=1,
            non_kinematic_vehicle_params_override=truck_params,
        ),
    ]


def _init_env(map_dir: Path, context: GroundTruthContextSpec):
    obs = np.zeros((1, _obs_dim()), dtype=np.float32)
    actions = np.zeros(1, dtype=np.int32)
    rewards = np.zeros(1, dtype=np.float32)
    terminals = np.zeros(1, dtype=np.uint8)
    truncs = np.zeros(1, dtype=np.uint8)
    kwargs = dict(
        human_agent_idx=0,
        ini_file=str(rollout_compare.DEFAULT_DRIVE_CONFIG),
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
        dynamics_model=context.dynamics_model,
        force_zero_trailer_articulation_at_init=context.force_zero_trailer_articulation_at_init,
    )
    if context.non_kinematic_vehicle_params_override is not None:
        kwargs["non_kinematic_vehicle_params_override"] = context.non_kinematic_vehicle_params_override
    env_handle = binding.env_init(obs, actions, rewards, terminals, truncs, 0, **kwargs)
    return env_handle, obs, actions, rewards, terminals, truncs


def _replay_actions_in_context(map_dir: Path, fit_actions: np.ndarray, context: GroundTruthContextSpec, gt: dict[str, Any]) -> dict[str, Any]:
    env_handle, _obs, actions, _rewards, terminals, truncs = _init_env(map_dir, context)
    try:
        binding.env_reset(env_handle, 0)
        active_count = binding.env_get_active_agent_count(env_handle)
        if active_count <= 0:
            raise RuntimeError("no_active_agents")

        rollout_obs = []
        rollout_actions = []
        rollout_x = []
        rollout_y = []
        rollout_z = []
        rollout_heading = []

        start_x, start_y, start_heading = _record_agent_state(env_handle)
        rollout_x.append(start_x)
        rollout_y.append(start_y)
        rollout_z.append(float(gt["z"][0]) if len(gt["z"]) else 0.0)
        rollout_heading.append(start_heading)

        for action in fit_actions:
            obs_record = _record_observation(env_handle, active_count)
            rollout_obs.append(obs_record[OBS_MODE_DEFAULT])
            rollout_actions.append(int(action))

            actions[0] = int(action)
            binding.env_step(env_handle)
            if int(terminals[0]) != 0 or int(truncs[0]) != 0:
                break

            cur_x, cur_y, cur_heading = _record_agent_state(env_handle)
            rollout_x.append(cur_x)
            rollout_y.append(cur_y)
            rollout_z.append(float(gt["z"][min(len(rollout_z), len(gt["z"]) - 1)]) if len(gt["z"]) else 0.0)
            rollout_heading.append(cur_heading)

        num_steps = len(rollout_actions)
        zeros = np.zeros(num_steps, dtype=np.float32)
        bools = np.zeros(num_steps, dtype=bool)
        return {
            "observations": np.stack(rollout_obs).astype(np.float32) if rollout_obs else np.zeros((0, _obs_dim()), dtype=np.float32),
            "actions": np.asarray(rollout_actions, dtype=np.int64),
            "task_rewards": zeros.copy(),
            "dones": bools.copy(),
            "truncs": bools.copy(),
            "values": zeros.copy(),
            "entropy": zeros.copy(),
            "x": np.asarray(rollout_x[:num_steps], dtype=np.float32),
            "y": np.asarray(rollout_y[:num_steps], dtype=np.float32),
            "z": np.asarray(rollout_z[:num_steps], dtype=np.float32),
            "heading": np.asarray(rollout_heading[:num_steps], dtype=np.float32),
            "steps": num_steps,
            "stop_reason": "fit_replay",
        }
    finally:
        binding.env_close(env_handle)


def _fit_classic_ground_truth(map_path: Path, map_entry: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    staged = _stage_map(map_path)
    try:
        classic_context = _context_specs()[0]
        env_handle, _obs, _actions, _rewards, _terminals, _truncs = _init_env(Path(staged.name), classic_context)
        try:
            binding.env_reset(env_handle, 0)
            active_count = binding.env_get_active_agent_count(env_handle)
            if active_count <= 0:
                raise RuntimeError("no_active_agents")

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
            for step_idx in range(int(num_steps)):
                binding.env_set_logged_timestep(env_handle, OFFLINE_FIT_INIT_STEPS + step_idx)
                obs_record = _record_observation(env_handle, active_count)
                logged_obs.append(obs_record[OBS_MODE_DEFAULT])

            num_steps = int(num_steps)
            result = {
                "map_name": map_entry["map_name"],
                "map_path": str(map_path.resolve()),
                "scenario_type": map_entry["scenario_type"],
                "delta_heading_deg": float(map_entry["delta_heading_deg"]),
                "steps": num_steps,
                "stop_reason": "fit_horizon",
                "observations": np.stack(logged_obs).astype(np.float32) if logged_obs else np.zeros((0, _obs_dim()), dtype=np.float32),
                "actions": fit_actions[:num_steps].astype(np.int64),
                "task_rewards": np.zeros(num_steps, dtype=np.float32),
                "dones": np.zeros(num_steps, dtype=bool),
                "truncs": np.zeros(num_steps, dtype=bool),
                "values": np.zeros(num_steps, dtype=np.float32),
                "entropy": np.zeros(num_steps, dtype=np.float32),
                "x": np.asarray(gt["x"][:num_steps], dtype=np.float32),
                "y": np.asarray(gt["y"][:num_steps], dtype=np.float32),
                "z": np.asarray(gt["z"][:num_steps], dtype=np.float32),
                "heading": np.asarray(gt["heading"][:num_steps], dtype=np.float32),
                "length": np.zeros(num_steps, dtype=np.float32),
                "width": np.zeros(num_steps, dtype=np.float32),
                "trailer_has": np.zeros(num_steps, dtype=np.int32),
                "trailer_x": np.zeros(num_steps, dtype=np.float32),
                "trailer_y": np.zeros(num_steps, dtype=np.float32),
                "trailer_heading": np.zeros(num_steps, dtype=np.float32),
                "fit_match_cost_total": float(total_cost),
                "fit_match_cost_lateral_total": float(total_lat_cost),
                "fit_match_cost_longitudinal_total": float(total_lon_cost),
                "fit_match_cost_step": step_costs[:num_steps].astype(np.float32),
                "fit_match_cost_lateral_step": step_lat_costs[:num_steps].astype(np.float32),
                "fit_match_cost_longitudinal_step": step_lon_costs[:num_steps].astype(np.float32),
            }
            return result, fit_actions[:num_steps].astype(np.int32), gt
        finally:
            binding.env_close(env_handle)
    finally:
        staged.cleanup()


def collect_ground_truth_rollouts(sample_manifest: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts = _context_specs()
    rollouts_by_name = {context.name: [] for context in contexts}
    for map_entry in sample_manifest["maps"]:
        map_path = Path(map_entry["map_path"])
        car_rollout, fit_actions, gt = _fit_classic_ground_truth(map_path, map_entry)
        rollouts_by_name[contexts[0].name].append(car_rollout)

        staged = _stage_map(map_path)
        try:
            replay = _replay_actions_in_context(Path(staged.name), fit_actions, contexts[1], gt)
        finally:
            staged.cleanup()
        replay_rollout = dict(car_rollout)
        replay_rollout.update(replay)
        replay_rollout["stop_reason"] = replay["stop_reason"]
        rollouts_by_name[contexts[1].name].append(replay_rollout)

    payload_paths = []
    for context in contexts:
        payload = {
            "format": rollout_compare.ROLLOUT_FORMAT_VERSION,
            "model_name": context.name,
            "model_root": context.name,
            "config_path": str(rollout_compare.DEFAULT_DRIVE_CONFIG.resolve()),
            "checkpoint_path": "",
            "rollouts": rollouts_by_name[context.name],
        }
        payload_path = output_dir / f"{context.name}_rollouts.pt"
        torch.save(payload, payload_path)
        payload_paths.append(payload_path)
    return payload_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare preference reward on GT-fitted car/truck contexts.")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--source-map-dir", type=Path, default=DEFAULT_SOURCE_MAP_DIR)
    parser.add_argument("--turning-count", type=int, default=7)
    parser.add_argument("--straight-count", type=int, default=3)
    parser.add_argument("--threshold-deg", type=float, default=45.0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--preference-config", type=Path, default=rollout_compare.DEFAULT_PREFERENCE_CONFIG)
    parser.add_argument("--reward-dir", type=Path, default=rollout_compare.DEFAULT_REWARD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest_path is not None:
        sample_manifest = json.loads(args.manifest_path.resolve().read_text(encoding="utf-8"))
        manifest_path = args.manifest_path.resolve()
    else:
        manifest_path = output_dir / "sample_manifest.json"
        sample_manifest = rollout_compare.build_stratified_sample_manifest(
            source_map_dir=args.source_map_dir.resolve(),
            output_path=manifest_path,
            turning_count=args.turning_count,
            straight_count=args.straight_count,
            threshold_deg=args.threshold_deg,
            sample_seed=args.sample_seed,
            validity_checker=rollout_compare._build_map_validity_checker(),
        )

    rollout_paths = collect_ground_truth_rollouts(sample_manifest, output_dir / "rollouts")
    scored_path = rollout_compare.score_rollout_payloads(
        rollout_paths=rollout_paths,
        preference_config_path=args.preference_config.resolve(),
        reward_dir=args.reward_dir.resolve(),
        output_path=output_dir / "scored_rollouts.pt",
    )
    report_outputs = rollout_compare.generate_report(scored_path, output_dir / "report")
    print(
        json.dumps(
            {
                "sample_manifest": str(manifest_path),
                "scored_rollouts": str(scored_path),
                "report": report_outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
