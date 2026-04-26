#!/usr/bin/env python3
import argparse
import copy
import csv
import gc
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pufferlib
from pufferlib import pufferl
from scripts.run_packaged_drive_human_replay_eval import (
    REPO_ROOT,
    _load_packaged_config,
    _overlay_args,
    _prepare_single_map_dir,
    _select_map_paths,
)


def _repo_tmp_dir() -> Path:
    path = REPO_ROOT / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_float_list(value: str) -> list[float]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return [float(item) for item in values]


def _build_base_args(config_path: Path, checkpoint_path: Path, device: str) -> dict[str, Any]:
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        base_args = pufferl.load_config("puffer_drive")
    finally:
        sys.argv = original_argv

    packaged = _load_packaged_config(config_path)
    args = _overlay_args(base_args, packaged)
    args["load_model_path"] = str(checkpoint_path)
    args["load_id"] = None
    args["wandb"] = False
    args["neptune"] = False
    args["train"]["device"] = device
    args["train"]["compile"] = False
    args["train"]["render"] = False
    args["eval"]["wosac_num_rollouts"] = 1
    args["vec"] = {"backend": "PufferEnv", "num_envs": 1}
    if args["eval"].get("human_replay_eval"):
        args["env"]["control_mode"] = args["eval"]["human_replay_control_mode"]
        args["env"]["episode_length"] = 91
    return args


def _prepare_args_for_map(
    base_args: dict[str, Any],
    map_path: Path,
    temp_root: Path,
    *,
    tracker_type: str,
    lookahead_distance: float,
    accel_gain: float,
    steer_gain: float,
    stanley_gain: float,
    stanley_softening: float,
    trajectory_control_substeps: int,
) -> dict[str, Any]:
    args = copy.deepcopy(base_args)
    single_map_dir = _prepare_single_map_dir(map_path, temp_root)
    args["env"]["map_dir"] = str(single_map_dir)
    args["env"]["num_maps"] = 1
    args["eval"]["map_dir"] = str(single_map_dir)
    args["eval"]["wosac_num_maps"] = 1
    args["env"]["trajectory_lookahead_distance"] = float(lookahead_distance)
    args["env"]["trajectory_accel_gain"] = float(accel_gain)
    args["env"]["trajectory_steer_gain"] = float(steer_gain)
    args["env"]["trajectory_tracker_type"] = str(tracker_type)
    args["env"]["trajectory_stanley_gain"] = float(stanley_gain)
    args["env"]["trajectory_stanley_softening"] = float(stanley_softening)
    args["env"]["trajectory_control_substeps"] = max(1, int(trajectory_control_substeps))
    return args


def _choose_action(logits, *, deterministic: bool):
    action, _, _ = pufferlib.pytorch.eval_action_from_logits(logits, deterministic=deterministic)
    return action


def _evaluate_single_map(
    policy,
    base_args: dict[str, Any],
    map_path: Path,
    *,
    tracker_type: str,
    lookahead_distance: float,
    accel_gain: float,
    steer_gain: float,
    stanley_gain: float,
    stanley_softening: float,
    deterministic_policy: bool,
    control_substeps: int,
    start_seconds: float,
) -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix=f"{map_path.stem}_controller_", dir=str(_repo_tmp_dir())))
    vecenv = None
    try:
        args = _prepare_args_for_map(
            base_args,
            map_path,
            temp_root,
            tracker_type=tracker_type,
            lookahead_distance=lookahead_distance,
            accel_gain=accel_gain,
            steer_gain=steer_gain,
            stanley_gain=stanley_gain,
            stanley_softening=stanley_softening,
            trajectory_control_substeps=control_substeps,
        )
        vecenv = pufferl.load_env("puffer_drive", args)
        driver = vecenv.driver_env
        obs, _ = vecenv.reset()
        gt = driver.get_ground_truth_trajectories()

        sim_steps = int(args["env"]["episode_length"] - args["env"]["init_steps"])
        start_timestep = max(0, int(round(float(start_seconds) / max(float(driver.dt), 1e-6))))
        start_timestep = min(start_timestep, max(sim_steps - 1, 0))
        eval_steps = max(sim_steps - start_timestep, 0)
        sim_x = np.zeros(eval_steps, dtype=np.float32)
        sim_y = np.zeros(eval_steps, dtype=np.float32)
        sim_id = np.zeros(eval_steps, dtype=np.int32)
        state = {}

        device = args["train"]["device"]
        if args["train"]["use_rnn"]:
            num_agents = vecenv.observation_space.shape[0]
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        if start_timestep > 0:
            if driver.is_trajectory_observation or driver.is_trajectory_action:
                driver._clear_trajectory_history()
            for warmup_timestep in range(start_timestep + 1):
                obs, _, _, _, _ = driver.set_logged_timestep(
                    warmup_timestep,
                    reset_history=False,
                )

        for out_idx, time_idx in enumerate(range(start_timestep, sim_steps)):
            agent_state = driver.get_global_agent_state()
            sim_x[out_idx] = agent_state["x"][0]
            sim_y[out_idx] = agent_state["y"][0]
            sim_id[out_idx] = agent_state["id"][0]

            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, _ = policy.forward_eval(ob_tensor, state)
                action = _choose_action(logits, deterministic=deterministic_policy)
                action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)
            obs, _, _, _, _ = vecenv.step(action_np)

        controlled_agent_id = int(sim_id[0])
        gt_agent_ids = gt["id"][:, 0]
        controlled_matches = np.flatnonzero(gt_agent_ids == controlled_agent_id)
        controlled_idx = int(controlled_matches[0]) if controlled_matches.size > 0 else 0
        valid_mask = gt["valid"][controlled_idx, 0, start_timestep:sim_steps] > 0
        gt_x = gt["x"][controlled_idx, 0, start_timestep:sim_steps]
        gt_y = gt["y"][controlled_idx, 0, start_timestep:sim_steps]
        displacement = np.sqrt((sim_x - gt_x) ** 2 + (sim_y - gt_y) ** 2)
        ade = float(displacement[valid_mask].mean()) if np.any(valid_mask) else float("nan")
        fde = float(displacement[np.flatnonzero(valid_mask)[-1]]) if np.any(valid_mask) else float("nan")
        return {
            "map_name": map_path.name,
            "map_path": str(map_path.resolve()),
            "agent_id": controlled_agent_id,
            "start_timestep": start_timestep,
            "ade": ade,
            "fde": fde,
            "valid_steps": int(valid_mask.sum()),
        }
    finally:
        if vecenv is not None:
            vecenv.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        gc.collect()


def _bootstrap_policy(base_args: dict[str, Any], seed_map_path: Path):
    temp_root = Path(tempfile.mkdtemp(prefix=f"{seed_map_path.stem}_bootstrap_", dir=str(_repo_tmp_dir())))
    vecenv = None
    try:
        args = _prepare_args_for_map(
            base_args,
            seed_map_path,
            temp_root,
            tracker_type=str(base_args["env"].get("trajectory_tracker_type", "pure_pursuit")),
            lookahead_distance=float(base_args["env"].get("trajectory_lookahead_distance", 6.0)),
            accel_gain=float(base_args["env"].get("trajectory_accel_gain", 0.5)),
            steer_gain=float(base_args["env"].get("trajectory_steer_gain", 1.0)),
            stanley_gain=float(base_args["env"].get("trajectory_stanley_gain", 1.0)),
            stanley_softening=float(base_args["env"].get("trajectory_stanley_softening", 1.0)),
            trajectory_control_substeps=int(base_args["env"].get("trajectory_control_substeps", 1)),
        )
        vecenv = pufferl.load_env("puffer_drive", args)
        policy = pufferl.load_policy(args, vecenv, env_name="puffer_drive")
        policy.eval()
        return policy
    finally:
        if vecenv is not None:
            vecenv.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        gc.collect()


def _evaluate_controller_combo(
    policy,
    base_args: dict[str, Any],
    map_paths: list[Path],
    *,
    tracker_type: str,
    lookahead_distance: float,
    accel_gain: float,
    steer_gain: float,
    stanley_gain: float,
    stanley_softening: float,
    deterministic_policy: bool,
    control_substeps: int,
    start_seconds: float,
) -> dict[str, Any]:
    per_map = []
    for index, map_path in enumerate(map_paths, start=1):
        result = _evaluate_single_map(
            policy,
            base_args,
            map_path,
            tracker_type=tracker_type,
            lookahead_distance=lookahead_distance,
            accel_gain=accel_gain,
            steer_gain=steer_gain,
            stanley_gain=stanley_gain,
            stanley_softening=stanley_softening,
            deterministic_policy=deterministic_policy,
            control_substeps=control_substeps,
            start_seconds=start_seconds,
        )
        per_map.append(result)
        print(
            f"[controller-tune] combo tracker={tracker_type} lookahead={lookahead_distance:.3f} "
            f"accel={accel_gain:.3f} steer={steer_gain:.3f} "
            f"stanley_gain={stanley_gain:.3f} softening={stanley_softening:.3f} "
            f"map={index}/{len(map_paths)} ade={result['ade']:.4f}",
            flush=True,
        )

    ades = np.asarray([item["ade"] for item in per_map], dtype=np.float32)
    fdes = np.asarray([item["fde"] for item in per_map], dtype=np.float32)
    return {
        "tracker_type": str(tracker_type),
        "lookahead_distance": float(lookahead_distance),
        "accel_gain": float(accel_gain),
        "steer_gain": float(steer_gain),
        "stanley_gain": float(stanley_gain),
        "stanley_softening": float(stanley_softening),
        "mean_ade": float(np.nanmean(ades)),
        "median_ade": float(np.nanmedian(ades)),
        "mean_fde": float(np.nanmean(fdes)),
        "num_maps": len(per_map),
        "per_map": per_map,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune trajectory-controller gains against validation rollout ADE.")
    parser.add_argument("--config", type=Path, default=Path("pufferlib/config/ocean/drive_trajectory_eval.ini"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments_bc/drive_trajectory_bc_cpu_live_pufferdrive_bc_fixed/best.pt"),
    )
    parser.add_argument("--map-dir", type=Path, default=Path("datasets/nuplanCarBostonAll_validation"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--lookahead-grid", type=str, default="4.0,6.0,8.0")
    parser.add_argument("--accel-grid", type=str, default="0.35,0.5,0.75")
    parser.add_argument("--steer-grid", type=str, default="0.75,1.0,1.25")
    parser.add_argument("--tracker-type", type=str, default="pure_pursuit", choices=("pure_pursuit", "stanley"))
    parser.add_argument("--stanley-gain-grid", type=str, default="1.0")
    parser.add_argument("--stanley-softening-grid", type=str, default="1.0")
    parser.add_argument("--control-substeps", type=int, default=1)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--deterministic-policy", action="store_true", default=True)
    parser.add_argument("--stochastic-policy", dest="deterministic_policy", action="store_false")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments_bc/controller_tuning"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    map_paths = sorted(args.map_dir.glob("*.bin"))
    if not map_paths:
        raise FileNotFoundError(f"No .bin maps found under {args.map_dir}")
    selected_maps = _select_map_paths(map_paths, args.sample_size, args.sample_seed)

    base_args = _build_base_args(args.config, args.checkpoint, args.device)
    policy = _bootstrap_policy(base_args, selected_maps[0])

    results = []
    combos = list(
        itertools.product(
            _parse_float_list(args.lookahead_grid),
            _parse_float_list(args.accel_grid),
            _parse_float_list(args.steer_grid),
            _parse_float_list(args.stanley_gain_grid),
            _parse_float_list(args.stanley_softening_grid),
        )
    )
    print(
        f"[controller-tune] evaluating {len(combos)} controller combinations on {len(selected_maps)} validation maps",
        flush=True,
    )

    for combo_idx, (lookahead_distance, accel_gain, steer_gain, stanley_gain, stanley_softening) in enumerate(
        combos, start=1
    ):
        print(
            f"[controller-tune] starting combo {combo_idx}/{len(combos)}: "
            f"tracker={args.tracker_type} lookahead={lookahead_distance:.3f} "
            f"accel={accel_gain:.3f} steer={steer_gain:.3f} "
            f"stanley_gain={stanley_gain:.3f} softening={stanley_softening:.3f}",
            flush=True,
        )
        result = _evaluate_controller_combo(
            policy,
            base_args,
            selected_maps,
            tracker_type=args.tracker_type,
            lookahead_distance=lookahead_distance,
            accel_gain=accel_gain,
            steer_gain=steer_gain,
            stanley_gain=stanley_gain,
            stanley_softening=stanley_softening,
            deterministic_policy=args.deterministic_policy,
            control_substeps=args.control_substeps,
            start_seconds=args.start_seconds,
        )
        results.append(result)

    ranked = sorted(results, key=lambda item: (item["mean_ade"], item["mean_fde"]))
    best = ranked[0]

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "map_dir": str(args.map_dir),
        "sample_size": len(selected_maps),
        "sample_seed": int(args.sample_seed),
        "deterministic_policy": bool(args.deterministic_policy),
        "control_substeps": int(args.control_substeps),
        "start_seconds": float(args.start_seconds),
        "selected_maps": [str(path.resolve()) for path in selected_maps],
        "results": ranked,
        "best": best,
    }
    (output_dir / "controller_tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "controller_tuning_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lookahead_distance",
                "accel_gain",
                "steer_gain",
                "stanley_gain",
                "stanley_softening",
                "mean_ade",
                "median_ade",
                "mean_fde",
                "num_maps",
            ],
        )
        writer.writeheader()
        for item in ranked:
            writer.writerow({key: item[key] for key in writer.fieldnames})

    print(
        "[controller-tune] best "
        f"tracker={best['tracker_type']} lookahead={best['lookahead_distance']:.3f} "
        f"accel={best['accel_gain']:.3f} steer={best['steer_gain']:.3f} "
        f"stanley_gain={best['stanley_gain']:.3f} softening={best['stanley_softening']:.3f} "
        f"mean_ade={best['mean_ade']:.4f} mean_fde={best['mean_fde']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
