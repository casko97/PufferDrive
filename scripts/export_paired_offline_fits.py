from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from pufferlib.ocean.drive.drive import binding

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

CAR_ROOT = Path("pufferlib/resources/drive/binaries/nuplanCarBostonTest10")
TRUCK_ROOT = Path("pufferlib/resources/drive/binaries/nuplanTruckBostonTest10")
DEFAULT_OUTPUT = Path("outputs/offline_fits/nuplan_boston_test10_paired_fits.pt")


def _obs_dim() -> int:
    return (
        binding.EGO_FEATURES_CLASSIC
        + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
        + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    )


def _discover_shared_maps(car_root: Path, truck_root: Path, max_maps: int | None = None) -> list[str]:
    shared = sorted(set(p.name for p in car_root.glob("map_*.bin")) & set(p.name for p in truck_root.glob("map_*.bin")))
    if max_maps is not None and max_maps > 0:
        shared = shared[:max_maps]
    return shared


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

                timestep_obs = np.zeros((active_count, _obs_dim()), dtype=np.float32)
                logged_obs = []
                for step_idx in range(int(num_steps)):
                    binding.env_set_logged_timestep(env_handle, OFFLINE_FIT_INIT_STEPS + step_idx)
                    binding.env_copy_observations(env_handle, timestep_obs)
                    logged_obs.append(timestep_obs[0].copy())
            finally:
                binding.env_close(env_handle)

            env_handle, obs, actions, rewards, terminals, truncs = _init_env(map_dir)
            try:
                binding.env_reset(env_handle, 0)
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
                rollout_x = [float(state_x[0])]
                rollout_y = [float(state_y[0])]
                rollout_heading = [float(state_heading[0])]
                for step_idx in range(int(num_steps)):
                    actions[0] = int(fit_actions[step_idx])
                    binding.env_step(env_handle)
                    if int(terminals[0]) != 0 or int(truncs[0]) != 0:
                        break
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
                    rollout_x.append(float(state_x[0]))
                    rollout_y.append(float(state_y[0]))
                    rollout_heading.append(float(state_heading[0]))
            finally:
                binding.env_close(env_handle)

            gt_x = np.asarray(gt["x"][: len(rollout_x)], dtype=np.float32)
            gt_y = np.asarray(gt["y"][: len(rollout_y)], dtype=np.float32)
            rollout_x = np.asarray(rollout_x, dtype=np.float32)
            rollout_y = np.asarray(rollout_y, dtype=np.float32)
            self_disp = np.sqrt((rollout_x - gt_x) ** 2 + (rollout_y - gt_y) ** 2)
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
                "rollout_x": rollout_x,
                "rollout_y": rollout_y,
                "rollout_heading": np.asarray(rollout_heading, dtype=np.float32),
                "self_displacement_per_step": self_disp.astype(np.float32),
                "self_ade": float(self_disp.mean()) if self_disp.size else float("nan"),
                "self_fde": float(self_disp[-1]) if self_disp.size else float("nan"),
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


def export_paired_fits(car_root: Path, truck_root: Path, output_path: Path, max_maps: int | None = None) -> Path:
    shared_maps = _discover_shared_maps(car_root, truck_root, max_maps=max_maps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "metadata": {
            "car_root": str(car_root),
            "truck_root": str(truck_root),
            "shared_maps": shared_maps,
            "fit_settings": {
                "control_mode": OFFLINE_FIT_CONTROL_MODE,
                "goal_behavior": OFFLINE_FIT_GOAL_BEHAVIOR,
                "collision_behavior": OFFLINE_FIT_COLLISION_BEHAVIOR,
                "offroad_behavior": OFFLINE_FIT_OFFROAD_BEHAVIOR,
                "termination_mode": OFFLINE_FIT_TERMINATION_MODE,
                "dt": OFFLINE_FIT_DT,
                "episode_length": OFFLINE_FIT_EPISODE_LENGTH,
            },
            "optimizer": {
                "beam_width": FIT_BEAM_WIDTH,
                "match_weight_lateral": FIT_MATCH_WEIGHT_LATERAL,
                "match_weight_longitudinal": FIT_MATCH_WEIGHT_LONGITUDINAL,
                "match_weight_heading": FIT_MATCH_WEIGHT_HEADING,
                "match_weight_speed": FIT_MATCH_WEIGHT_SPEED,
                "match_weight_steer_change": FIT_MATCH_WEIGHT_STEER_CHANGE,
                "match_weight_accel_change": FIT_MATCH_WEIGHT_ACCEL_CHANGE,
                "match_weight_reverse": FIT_MATCH_WEIGHT_REVERSE,
                "match_weight_progress": FIT_MATCH_WEIGHT_PROGRESS,
                "match_weight_steer_flip": FIT_MATCH_WEIGHT_STEER_FLIP,
                "match_weight_ref_accel": FIT_MATCH_WEIGHT_REF_ACCEL,
                "match_weight_ref_steer": FIT_MATCH_WEIGHT_REF_STEER,
            },
        },
        "pairs": {},
    }

    for map_name in shared_maps:
        car_side = _fit_side(car_root / map_name)
        truck_side = _fit_side(truck_root / map_name)
        payload["pairs"][map_name] = {
            "car": car_side,
            "truck": truck_side,
            "pair_similarity": _pair_metrics(car_side, truck_side),
        }

    torch.save(payload, output_path)
    summary = {
        "output_path": str(output_path),
        "shared_map_count": len(shared_maps),
        "successful_pairs": sum(
            1
            for pair in payload["pairs"].values()
            if pair["car"]["status"] == "ok" and pair["truck"]["status"] == "ok"
        ),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export paired truck/car offline fitted trajectories.")
    parser.add_argument("--car-root", type=Path, default=CAR_ROOT)
    parser.add_argument("--truck-root", type=Path, default=TRUCK_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-maps", type=int, default=0)
    args = parser.parse_args()

    output_path = export_paired_fits(
        args.car_root,
        args.truck_root,
        args.output,
        max_maps=(args.max_maps if args.max_maps > 0 else None),
    )
    print(output_path)


if __name__ == "__main__":
    main()
