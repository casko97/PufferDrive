#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pufferlib import pufferl
from pufferlib.pytorch import eval_action_from_logits
from scripts.animate_drive_model_trajectories import build_args
from scripts.plot_drive_trajectory_bc_deterministic_rollout import (
    dump_rollout_frames_json,
    render_deterministic_rollout_grid_from_json,
)
from scripts.run_packaged_drive_human_replay_eval import _select_map_paths
from scripts.validate_trajectory_critic_warmstart import (
    DEFAULT_MAP_IDS,
    _config_warmstart_seconds,
    _load_checkpoint_state_dict,
    _load_eval_policy,
    _prepare_eval_args,
    _repo_tmp_dir,
    summarize_weight_changes,
)

DEFAULT_PPO_VALIDATION_SAMPLE_SIZE = 16


def default_map_paths(validation_map_dir: Path) -> list[Path]:
    preferred = [validation_map_dir / f"map_{map_id}.bin" for map_id in DEFAULT_MAP_IDS]
    all_map_paths = sorted(validation_map_dir.glob("map_*.bin"))
    if not all_map_paths:
        return preferred

    preferred_existing = [path for path in preferred if path.exists()]
    remaining_candidates = [path for path in all_map_paths if path.name not in {item.name for item in preferred_existing}]
    remaining_budget = max(DEFAULT_PPO_VALIDATION_SAMPLE_SIZE - len(preferred_existing), 0)
    sampled_remaining = _select_map_paths(remaining_candidates, remaining_budget, sample_seed=0)
    return sorted(preferred_existing + sampled_remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PPO fine-tuning against BC and critic warm-start baselines.")
    parser.add_argument("--config", type=Path, required=True, help="Packaged PPO config")
    parser.add_argument("--bc-checkpoint", type=Path, required=True, help="Original BC checkpoint")
    parser.add_argument("--warmstart-checkpoint", type=Path, required=True, help="Critic warm-start checkpoint")
    parser.add_argument("--ppo-checkpoint", type=Path, required=True, help="PPO fine-tuned checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for validation outputs")
    parser.add_argument(
        "--validation-map-dir",
        type=Path,
        default=Path("datasets/nuplanCarBostonAll_validation"),
        help="Validation map directory",
    )
    parser.add_argument(
        "--map-paths",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit held-out map paths; defaults to a deterministic held-out validation subset",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_PPO_VALIDATION_SAMPLE_SIZE,
        help="Held-out validation subset size when map paths are not explicitly provided",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Sampling seed for held-out validation subset selection",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed for deterministic rollout collection")
    parser.add_argument(
        "--grid-map",
        type=Path,
        default=None,
        help="Optional representative map for rollout-grid rendering; defaults to the first validation map",
    )
    return parser.parse_args()


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _collect_rollout_summary(
    *,
    config_path: Path,
    checkpoint_path: Path,
    map_path: Path,
    device: str,
    seed: int,
) -> dict[str, Any]:
    args, temp_root = _prepare_eval_args(config_path, checkpoint_path, map_path, device)
    vecenv = None
    try:
        vecenv = pufferl.load_env("puffer_drive", args)
        policy = _load_eval_policy(args, vecenv, checkpoint_path, reinitialize_value_head=False)
        driver = vecenv.driver_env
        obs, _ = vecenv.reset(seed=seed)
        state = {}
        sim_steps = int(args["env"]["episode_length"] - args["env"]["init_steps"])
        step_errors: list[float] = []
        rewards: list[float] = []
        done = False
        final_stop_reason = ""
        final_time_s = float(driver.tick) * float(driver.dt)

        while int(driver.tick) < sim_steps:
            tick = int(driver.tick)
            gt = driver.get_ground_truth_trajectories()
            agent_state = driver.get_global_agent_state()
            current_x = float(agent_state["x"][0])
            current_y = float(agent_state["y"][0])
            gt_x = float(gt["x"][0, 0, tick])
            gt_y = float(gt["y"][0, 0, tick])
            step_errors.append(float(np.hypot(current_x - gt_x, current_y - gt_y)))
            final_time_s = float(tick) * float(driver.dt)

            with torch.no_grad():
                tensor_obs = torch.as_tensor(obs).to(device)
                logits, _value = policy.forward_eval(tensor_obs, state)
                deterministic = getattr(policy, "is_trajectory_policy", False)
                action, _logprob, _entropy = eval_action_from_logits(logits, deterministic=deterministic)
                action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)

            obs, reward, terminals, truncations, info = vecenv.step(action_np)
            rewards.append(float(reward[0]))
            done = bool(terminals[0] or truncations[0])
            diagnostics = driver.get_agent_diagnostics()

            for item in info or []:
                if not isinstance(item, dict):
                    continue
                terminal_stop_reasons = item.get("terminal_stop_reasons")
                if terminal_stop_reasons:
                    final_stop_reason = str(terminal_stop_reasons[0].get("reason", ""))
                    break
            if bool(diagnostics["stopped"][0]):
                if bool(diagnostics["reached_goal"][0]):
                    final_stop_reason = "goal_stop"
                elif bool(diagnostics["offroad_flag"][0]):
                    final_stop_reason = "offroad_stop"
                elif int(diagnostics["collision_state"][0]) == 1:
                    final_stop_reason = "collision_stop"
                elif int(diagnostics["collision_state"][0]) == 2:
                    final_stop_reason = "offroad_stop"
                elif not final_stop_reason:
                    final_stop_reason = "stopped_unknown"

            if done:
                break

        ade = float(np.mean(step_errors)) if step_errors else float("nan")
        fde = float(step_errors[-1]) if step_errors else float("nan")
        return {
            "map_name": map_path.name,
            "num_steps": int(len(step_errors)),
            "ade": ade,
            "fde": fde,
            "total_reward": float(sum(rewards)),
            "done": done,
            "stop_reason": final_stop_reason,
            "final_time_s": final_time_s,
        }
    finally:
        if vecenv is not None:
            vecenv.close()
        shutil.rmtree(temp_root, ignore_errors=True)


def collect_model_rollouts(
    *,
    config_path: Path,
    checkpoint_path: Path,
    map_paths: list[Path],
    device: str,
    seed: int,
) -> list[dict[str, Any]]:
    return [
        _collect_rollout_summary(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            map_path=map_path,
            device=device,
            seed=seed,
        )
        for map_path in map_paths
    ]


def summarize_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ades = np.asarray([row["ade"] for row in rows], dtype=np.float64)
    fdes = np.asarray([row["fde"] for row in rows], dtype=np.float64)
    times = np.asarray([row["final_time_s"] for row in rows], dtype=np.float64)
    rewards = np.asarray([row["total_reward"] for row in rows], dtype=np.float64)
    stop_counts = Counter(row["stop_reason"] or "none" for row in rows)
    return {
        "mean_ade": float(np.nanmean(ades)),
        "mean_fde": float(np.nanmean(fdes)),
        "mean_total_reward": float(np.nanmean(rewards)),
        "mean_final_time_s": float(np.nanmean(times)),
        "stop_reason_counts": dict(sorted(stop_counts.items())),
        "num_rollouts": int(len(rows)),
    }


def save_rollout_csv(rows_by_model: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "map_name", "num_steps", "ade", "fde", "total_reward", "done", "stop_reason", "final_time_s"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, rows in rows_by_model.items():
            for row in rows:
                writer.writerow({"model": model_name, **row})


def make_bar_plots(rows_by_model: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, str]:
    plt = _load_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    model_names = list(rows_by_model.keys())
    mean_ades = [summarize_rollouts(rows_by_model[name])["mean_ade"] for name in model_names]
    mean_fdes = [summarize_rollouts(rows_by_model[name])["mean_fde"] for name in model_names]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(model_names, mean_ades, color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_ylabel("Mean ADE [m]")
    ax.set_title("Held-out rollout ADE")
    ade_path = output_dir / "mean_ade_bar.png"
    fig.savefig(ade_path, dpi=160)
    plt.close(fig)
    artifacts["mean_ade_bar"] = str(ade_path)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(model_names, mean_fdes, color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_ylabel("Mean FDE [m]")
    ax.set_title("Held-out rollout FDE")
    fde_path = output_dir / "mean_fde_bar.png"
    fig.savefig(fde_path, dpi=160)
    plt.close(fig)
    artifacts["mean_fde_bar"] = str(fde_path)
    return artifacts


def render_rollout_grid_triplet(
    *,
    config_path: Path,
    bc_checkpoint: Path,
    warmstart_checkpoint: Path,
    ppo_checkpoint: Path,
    map_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    warmstart_seconds = _config_warmstart_seconds(config_path)
    outputs: dict[str, str] = {}
    triplet = {
        "bc": bc_checkpoint,
        "warmstart": warmstart_checkpoint,
        "ppo": ppo_checkpoint,
    }
    for name, checkpoint in triplet.items():
        json_path = output_dir / f"{name}_rollout_frames.json"
        png_path = output_dir / f"{name}_rollout_grid.png"
        dump_rollout_frames_json(
            config_path=config_path,
            checkpoint_path=checkpoint,
            map_path=map_path,
            output_path=json_path,
            device="cpu",
            control_substeps=1,
            start_seconds=warmstart_seconds,
        )
        render_deterministic_rollout_grid_from_json(
            frames_json_path=json_path,
            output_path=png_path,
            columns=4,
            map_path=map_path,
            show_diagnostics=True,
        )
        outputs[f"{name}_json"] = str(json_path)
        outputs[f"{name}_grid"] = str(png_path)
    return outputs


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    bc_checkpoint = args.bc_checkpoint.expanduser().resolve()
    warmstart_checkpoint = args.warmstart_checkpoint.expanduser().resolve()
    ppo_checkpoint = args.ppo_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_map_dir = args.validation_map_dir.expanduser().resolve()
    if args.map_paths:
        candidate_paths = [path.expanduser().resolve() for path in args.map_paths]
    else:
        candidate_paths = [path.expanduser().resolve() for path in default_map_paths(validation_map_dir)]
        if args.sample_size is not None and len(candidate_paths) != args.sample_size:
            all_validation_maps = sorted(path.expanduser().resolve() for path in validation_map_dir.glob("map_*.bin"))
            candidate_paths = _select_map_paths(all_validation_maps, args.sample_size, args.sample_seed)
    map_paths = candidate_paths
    grid_map = (args.grid_map.expanduser().resolve() if args.grid_map else map_paths[0])

    rows_by_model = {
        "bc": collect_model_rollouts(
            config_path=config_path,
            checkpoint_path=bc_checkpoint,
            map_paths=map_paths,
            device=args.device,
            seed=args.seed,
        ),
        "warmstart": collect_model_rollouts(
            config_path=config_path,
            checkpoint_path=warmstart_checkpoint,
            map_paths=map_paths,
            device=args.device,
            seed=args.seed,
        ),
        "ppo": collect_model_rollouts(
            config_path=config_path,
            checkpoint_path=ppo_checkpoint,
            map_paths=map_paths,
            device=args.device,
            seed=args.seed,
        ),
    }
    save_rollout_csv(rows_by_model, output_dir / "rollout_metrics.csv")
    plot_artifacts = make_bar_plots(rows_by_model, output_dir)
    grid_artifacts = render_rollout_grid_triplet(
        config_path=config_path,
        bc_checkpoint=bc_checkpoint,
        warmstart_checkpoint=warmstart_checkpoint,
        ppo_checkpoint=ppo_checkpoint,
        map_path=grid_map,
        output_dir=output_dir,
    )

    ppo_weight_summary = summarize_weight_changes(
        _load_checkpoint_state_dict(warmstart_checkpoint),
        _load_checkpoint_state_dict(ppo_checkpoint),
    )
    trainability_pass = bool(
        not ppo_weight_summary["actor"]["unchanged"] and ppo_weight_summary["backbone"]["unchanged"]
    )

    summaries = {name: summarize_rollouts(rows) for name, rows in rows_by_model.items()}
    improves_over_bc = bool(
        summaries["ppo"]["mean_ade"] < summaries["bc"]["mean_ade"]
        and summaries["ppo"]["mean_fde"] < summaries["bc"]["mean_fde"]
    )
    improves_over_warmstart = bool(
        summaries["ppo"]["mean_ade"] < summaries["warmstart"]["mean_ade"]
        and summaries["ppo"]["mean_fde"] < summaries["warmstart"]["mean_fde"]
    )
    no_regression_stop_behavior = bool(
        summaries["ppo"]["mean_final_time_s"] >= summaries["warmstart"]["mean_final_time_s"]
    )

    summary = {
        "config_path": str(config_path),
        "bc_checkpoint": str(bc_checkpoint),
        "warmstart_checkpoint": str(warmstart_checkpoint),
        "ppo_checkpoint": str(ppo_checkpoint),
        "map_paths": [str(path) for path in map_paths],
        "grid_map": str(grid_map),
        "weight_summary": ppo_weight_summary,
        "per_model_rollouts": rows_by_model,
        "per_model_summary": summaries,
        "validation": {
            "actor_updated_backbone_frozen": trainability_pass,
            "ppo_improves_over_bc": improves_over_bc,
            "ppo_improves_over_warmstart": improves_over_warmstart,
            "no_regression_stop_behavior": no_regression_stop_behavior,
            "overall_pass": bool(trainability_pass and improves_over_bc and improves_over_warmstart),
        },
        "artifacts": {
            "csv": str(output_dir / "rollout_metrics.csv"),
            **plot_artifacts,
            **grid_artifacts,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
