#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import copy
import csv
import gc
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_IDS = (318, 330, 994, 1411)


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _repo_tmp_dir() -> Path:
    path = REPO_ROOT / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def _load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")
    return _strip_module_prefix(payload)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for idx, count in enumerate(counts):
            if count <= 1:
                continue
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()
    return ranks


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_value_metrics(predicted: np.ndarray, realized: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    mse = float(np.mean((predicted - realized) ** 2))
    mae = float(np.mean(np.abs(predicted - realized)))
    pearson = _safe_corrcoef(predicted, realized)
    spearman = _safe_corrcoef(_rankdata(predicted), _rankdata(realized))
    var_y = float(np.var(realized))
    explained_variance = float("nan") if var_y == 0.0 else float(1.0 - np.var(realized - predicted) / var_y)
    return {
        "mse": mse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "explained_variance": explained_variance,
    }


def summarize_weight_changes(
    reference_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | int | bool]]:
    summary: dict[str, dict[str, float | int | bool]] = {}
    groups = {
        "actor": lambda name: name.startswith("actor."),
        "value_fn": lambda name: name.startswith("value_fn."),
        "backbone": lambda name: not (name.startswith("actor.") or name.startswith("value_fn.")),
    }

    for group_name, predicate in groups.items():
        max_abs_diff = 0.0
        changed_params = 0
        total_params = 0
        for name, ref_tensor in reference_state.items():
            if not predicate(name):
                continue
            candidate_tensor = candidate_state.get(name)
            if candidate_tensor is None:
                continue
            diff = (candidate_tensor.detach().cpu() - ref_tensor.detach().cpu()).abs()
            total_params += diff.numel()
            local_max = float(diff.max().item()) if diff.numel() > 0 else 0.0
            max_abs_diff = max(max_abs_diff, local_max)
            changed_params += int(torch.count_nonzero(diff > 0).item())
        summary[group_name] = {
            "max_abs_diff": max_abs_diff,
            "changed_params": changed_params,
            "total_params": total_params,
            "unchanged": changed_params == 0,
        }
    return summary


def _make_single_map_dir(map_path: Path) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix=f"{map_path.stem}_critic_", dir=str(_repo_tmp_dir())))
    single_map_dir = temp_root / "single_map"
    single_map_dir.mkdir(parents=True, exist_ok=True)
    destination = single_map_dir / map_path.name
    if destination.exists():
        destination.unlink()
    destination.symlink_to(map_path.resolve())
    return temp_root


def _prepare_eval_args(config_path: Path, checkpoint_path: Path, map_path: Path, device: str) -> tuple[dict[str, Any], Path]:
    args, temp_root = build_args(config_path, checkpoint_path, map_path, device)
    args["train"]["trajectory_training_mode"] = "none"
    args["train"]["trajectory_reinit_value_head"] = False
    args["wandb"] = False
    args["neptune"] = False
    args["train"]["render"] = False
    return args, temp_root


def _unwrap_policy(policy):
    base_policy = policy
    while hasattr(base_policy, "policy"):
        base_policy = base_policy.policy
    return base_policy


def _load_eval_policy(
    args: dict[str, Any],
    vecenv,
    checkpoint_path: Path,
    *,
    reinitialize_value_head: bool,
):
    load_args = copy.deepcopy(args)
    load_args["load_model_path"] = str(checkpoint_path)
    load_args["load_id"] = None
    load_args["train"]["trajectory_training_mode"] = "none"
    load_args["train"]["trajectory_reinit_value_head"] = False
    policy = pufferl.load_policy(load_args, vecenv, env_name="puffer_drive")
    base_policy = _unwrap_policy(policy)
    if reinitialize_value_head:
        torch.manual_seed(0)
        base_policy.reinitialize_value_head()
    policy.eval()
    return policy


@dataclass
class RolloutStepRecord:
    map_name: str
    timestep: int
    time_s: float
    reward: float
    done: bool
    observation: np.ndarray
    stop_reason: str


def collect_rollout_dataset(
    *,
    config_path: Path,
    checkpoint_path: Path,
    map_paths: list[Path],
    device: str,
    seed: int,
) -> list[RolloutStepRecord]:
    records: list[RolloutStepRecord] = []
    for map_path in map_paths:
        args, temp_root = _prepare_eval_args(config_path, checkpoint_path, map_path, device)
        vecenv = None
        try:
            vecenv = pufferl.load_env("puffer_drive", args)
            policy = _load_eval_policy(args, vecenv, checkpoint_path, reinitialize_value_head=False)
            driver = vecenv.driver_env
            obs, _ = vecenv.reset(seed=seed)
            state = {}
            sim_steps = int(args["env"]["episode_length"] - args["env"]["init_steps"])

            while int(driver.tick) < sim_steps:
                current_timestep = int(driver.tick)
                current_time_s = float(current_timestep) * float(driver.dt)
                obs_before = np.asarray(obs[0], dtype=np.float32).copy()
                with torch.no_grad():
                    tensor_obs = torch.as_tensor(obs).to(device)
                    logits, _value = policy.forward_eval(tensor_obs, state)
                    deterministic = getattr(policy, "is_trajectory_policy", False)
                    action, _logprob, _entropy = eval_action_from_logits(logits, deterministic=deterministic)
                    action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
                if isinstance(logits, torch.distributions.Normal):
                    action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)
                obs, rewards, terminals, truncations, _info = vecenv.step(action_np)
                diagnostics = driver.get_agent_diagnostics()
                done = bool(terminals[0] or truncations[0])
                stop_reason = ""
                for item in _info or []:
                    terminal_stop_reasons = item.get("terminal_stop_reasons") if isinstance(item, dict) else None
                    if terminal_stop_reasons:
                        stop_reason = str(terminal_stop_reasons[0].get("reason", ""))
                        break
                if bool(diagnostics["stopped"][0]):
                    if bool(diagnostics["reached_goal"][0]):
                        stop_reason = "goal_stop"
                    elif bool(diagnostics["offroad_flag"][0]):
                        stop_reason = "offroad_stop"
                    elif int(diagnostics["collision_state"][0]) == 1:
                        stop_reason = "collision_stop"
                    elif int(diagnostics["collision_state"][0]) == 2:
                        stop_reason = "offroad_stop"
                    else:
                        stop_reason = "stopped_unknown"
                records.append(
                    RolloutStepRecord(
                        map_name=map_path.name,
                        timestep=current_timestep,
                        time_s=current_time_s,
                        reward=float(rewards[0]),
                        done=done,
                        observation=obs_before,
                        stop_reason=stop_reason,
                    )
                )
                if done:
                    break
            del policy
        finally:
            if vecenv is not None:
                vecenv.close()
            shutil.rmtree(temp_root, ignore_errors=True)
            gc.collect()
    return records


def attach_discounted_returns(records: list[RolloutStepRecord], gamma: float) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    grouped: dict[str, list[RolloutStepRecord]] = {}
    for record in records:
        grouped.setdefault(record.map_name, []).append(record)

    for map_name, map_records in grouped.items():
        running_return = 0.0
        returns = [0.0] * len(map_records)
        for idx in range(len(map_records) - 1, -1, -1):
            record = map_records[idx]
            running_return = record.reward + gamma * running_return * (0.0 if record.done else 1.0)
            returns[idx] = running_return
        for record, realized_return in zip(map_records, returns):
            payload.append(
                {
                    "map_name": map_name,
                    "timestep": record.timestep,
                    "time_s": record.time_s,
                    "reward": record.reward,
                    "done": record.done,
                    "stop_reason": record.stop_reason,
                    "realized_return": realized_return,
                    "observation": record.observation,
                }
            )
    return payload


def evaluate_policy_values(
    *,
    config_path: Path,
    checkpoint_path: Path,
    map_path: Path,
    device: str,
    observations: np.ndarray,
    reinitialize_value_head: bool,
) -> np.ndarray:
    args, temp_root = _prepare_eval_args(config_path, checkpoint_path, map_path, device)
    vecenv = None
    try:
        vecenv = pufferl.load_env("puffer_drive", args)
        policy = _load_eval_policy(args, vecenv, checkpoint_path, reinitialize_value_head=reinitialize_value_head)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observations, dtype=torch.float32).to(device)
            _actions, values = policy.forward_eval(obs_tensor, {})
        return values.detach().cpu().numpy().reshape(-1)
    finally:
        if vecenv is not None:
            vecenv.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        gc.collect()


def save_value_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "map_name",
                "timestep",
                "time_s",
                "reward",
                "done",
                "stop_reason",
                "realized_return",
                "trained_value",
                "baseline_value",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def make_scatter_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    realized = np.asarray([row["realized_return"] for row in rows], dtype=np.float32)
    trained = np.asarray([row["trained_value"] for row in rows], dtype=np.float32)
    baseline = np.asarray([row["baseline_value"] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(realized, baseline, s=10, alpha=0.4, label="baseline critic")
    ax.scatter(realized, trained, s=10, alpha=0.4, label="trained critic")
    min_val = float(min(realized.min(), trained.min(), baseline.min()))
    max_val = float(max(realized.max(), trained.max(), baseline.max()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Realized discounted return")
    ax.set_ylabel("Predicted value")
    ax.set_title("Critic value predictions vs realized returns")
    ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_timeseries_plot(rows: list[dict[str, Any]], map_name: str, output_path: Path) -> None:
    plt = _load_matplotlib()
    selected = [row for row in rows if row["map_name"] == map_name]
    if not selected:
        return
    time_s = [row["time_s"] for row in selected]
    realized = [row["realized_return"] for row in selected]
    trained = [row["trained_value"] for row in selected]
    baseline = [row["baseline_value"] for row in selected]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(time_s, realized, label="realized return", linewidth=2)
    ax.plot(time_s, trained, label="trained critic", linewidth=1.5)
    ax.plot(time_s, baseline, label="baseline critic", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Value / return")
    ax.set_title(f"Critic values over rollout: {map_name}")
    ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compare_rollout_behavior(
    *,
    config_path: Path,
    bc_checkpoint: Path,
    trained_checkpoint: Path,
    map_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    warmstart_seconds = _config_warmstart_seconds(config_path)
    bc_json = output_dir / "bc_rollout_frames.json"
    trained_json = output_dir / "warmstart_rollout_frames.json"
    dump_rollout_frames_json(
        config_path=config_path,
        checkpoint_path=bc_checkpoint,
        map_path=map_path,
        output_path=bc_json,
        device="cpu",
        control_substeps=1,
        start_seconds=warmstart_seconds,
    )
    dump_rollout_frames_json(
        config_path=config_path,
        checkpoint_path=trained_checkpoint,
        map_path=map_path,
        output_path=trained_json,
        device="cpu",
        control_substeps=1,
        start_seconds=warmstart_seconds,
    )

    render_deterministic_rollout_grid_from_json(
        frames_json_path=bc_json,
        output_path=output_dir / "bc_rollout_grid.png",
        columns=4,
        map_path=map_path,
        show_diagnostics=False,
    )
    render_deterministic_rollout_grid_from_json(
        frames_json_path=trained_json,
        output_path=output_dir / "warmstart_rollout_grid.png",
        columns=4,
        map_path=map_path,
        show_diagnostics=False,
    )

    bc_frames = json.loads(bc_json.read_text(encoding="utf-8"))
    trained_frames = json.loads(trained_json.read_text(encoding="utf-8"))
    max_world_diff = 0.0
    max_control_diff = 0.0
    for bc_frame, trained_frame in zip(bc_frames, trained_frames):
        for idx in range(3):
            max_world_diff = max(
                max_world_diff,
                abs(float(bc_frame["current_world"][idx]) - float(trained_frame["current_world"][idx])),
            )
        for idx in range(2):
            max_control_diff = max(
                max_control_diff,
                abs(float(bc_frame["control_action"][idx]) - float(trained_frame["control_action"][idx])),
            )
    return {
        "map_name": map_path.name,
        "bc_rollout_json": str(bc_json),
        "trained_rollout_json": str(trained_json),
        "bc_rollout_grid": str(output_dir / "bc_rollout_grid.png"),
        "trained_rollout_grid": str(output_dir / "warmstart_rollout_grid.png"),
        "max_world_diff": max_world_diff,
        "max_control_diff": max_control_diff,
    }


def default_map_paths(validation_map_dir: Path) -> list[Path]:
    return [validation_map_dir / f"map_{map_id}.bin" for map_id in DEFAULT_MAP_IDS]


def _config_warmstart_seconds(config_path: Path) -> float:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    with config_path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    if parser.has_option("env", "trajectory_history_warmstart_seconds"):
        return max(float(parser.get("env", "trajectory_history_warmstart_seconds")), 0.0)
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a short trajectory critic warm-start run.")
    parser.add_argument("--config", type=Path, required=True, help="Packaged short warm-start config")
    parser.add_argument("--bc-checkpoint", type=Path, required=True, help="Original BC checkpoint")
    parser.add_argument("--trained-checkpoint", type=Path, required=True, help="Trained warm-start checkpoint")
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
        help="Optional explicit held-out map paths; defaults to a small validation subset",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed for held-out rollout collection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve()
    bc_checkpoint = args.bc_checkpoint.expanduser().resolve()
    trained_checkpoint = args.trained_checkpoint.expanduser().resolve()
    validation_map_dir = args.validation_map_dir.expanduser().resolve()
    map_paths = [path.expanduser().resolve() for path in (args.map_paths or default_map_paths(validation_map_dir))]

    original_argv = []
    gamma = 0.98
    try:
        import sys

        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]
        base_args = pufferl.load_config("puffer_drive")
        gamma = float(base_args.get("train", {}).get("gamma", gamma))
    finally:
        if original_argv:
            sys.argv = original_argv

    rollout_records = collect_rollout_dataset(
        config_path=config_path,
        checkpoint_path=bc_checkpoint,
        map_paths=map_paths,
        device=args.device,
        seed=args.seed,
    )
    rows = attach_discounted_returns(rollout_records, gamma)
    observations = np.stack([row["observation"] for row in rows], axis=0).astype(np.float32)

    trained_values = evaluate_policy_values(
        config_path=config_path,
        checkpoint_path=trained_checkpoint,
        map_path=map_paths[0],
        device=args.device,
        observations=observations,
        reinitialize_value_head=False,
    )
    baseline_values = evaluate_policy_values(
        config_path=config_path,
        checkpoint_path=bc_checkpoint,
        map_path=map_paths[0],
        device=args.device,
        observations=observations,
        reinitialize_value_head=True,
    )

    for idx, row in enumerate(rows):
        row["trained_value"] = float(trained_values[idx])
        row["baseline_value"] = float(baseline_values[idx])
        del row["observation"]

    save_value_csv(rows, output_dir / "state_value_metrics.csv")
    make_scatter_plot(rows, output_dir / "value_scatter.png")
    for map_path in map_paths[:3]:
        make_timeseries_plot(rows, map_path.name, output_dir / f"{map_path.stem}_value_timeseries.png")

    realized = np.asarray([row["realized_return"] for row in rows], dtype=np.float64)
    trained = np.asarray([row["trained_value"] for row in rows], dtype=np.float64)
    baseline = np.asarray([row["baseline_value"] for row in rows], dtype=np.float64)
    overall_metrics = {
        "trained": compute_value_metrics(trained, realized),
        "baseline": compute_value_metrics(baseline, realized),
    }
    per_map_metrics = {}
    for map_path in map_paths:
        selected = [row for row in rows if row["map_name"] == map_path.name]
        map_realized = np.asarray([row["realized_return"] for row in selected], dtype=np.float64)
        map_trained = np.asarray([row["trained_value"] for row in selected], dtype=np.float64)
        map_baseline = np.asarray([row["baseline_value"] for row in selected], dtype=np.float64)
        per_map_metrics[map_path.name] = {
            "trained": compute_value_metrics(map_trained, map_realized),
            "baseline": compute_value_metrics(map_baseline, map_realized),
            "num_states": int(len(selected)),
        }

    weight_summary = summarize_weight_changes(
        _load_checkpoint_state_dict(bc_checkpoint),
        _load_checkpoint_state_dict(trained_checkpoint),
    )
    behavior_summary = compare_rollout_behavior(
        config_path=config_path,
        bc_checkpoint=bc_checkpoint,
        trained_checkpoint=trained_checkpoint,
        map_path=map_paths[0],
        output_dir=output_dir,
    )

    trainability_pass = bool(
        weight_summary["actor"]["unchanged"]
        and weight_summary["backbone"]["unchanged"]
        and not weight_summary["value_fn"]["unchanged"]
    )
    critic_beats_baseline = bool(
        overall_metrics["trained"]["mse"] < overall_metrics["baseline"]["mse"]
        and overall_metrics["trained"]["mae"] < overall_metrics["baseline"]["mae"]
    )
    critic_positive_correlation = bool(
        np.isfinite(overall_metrics["trained"]["pearson"])
        and np.isfinite(overall_metrics["trained"]["spearman"])
        and overall_metrics["trained"]["pearson"] > 0.0
        and overall_metrics["trained"]["spearman"] > 0.0
    )
    no_behavior_drift = bool(
        float(behavior_summary["max_world_diff"]) == 0.0 and float(behavior_summary["max_control_diff"]) == 0.0
    )
    overall_pass = bool(
        trainability_pass and critic_beats_baseline and critic_positive_correlation and no_behavior_drift
    )

    summary = {
        "config_path": str(config_path),
        "bc_checkpoint": str(bc_checkpoint),
        "trained_checkpoint": str(trained_checkpoint),
        "gamma": gamma,
        "map_paths": [str(path) for path in map_paths],
        "num_states": int(len(rows)),
        "overall_metrics": overall_metrics,
        "per_map_metrics": per_map_metrics,
        "weight_summary": weight_summary,
        "behavior_summary": behavior_summary,
        "validation": {
            "trainability_pass": trainability_pass,
            "critic_beats_baseline": critic_beats_baseline,
            "critic_positive_correlation": critic_positive_correlation,
            "no_behavior_drift": no_behavior_drift,
            "overall_pass": overall_pass,
        },
        "artifacts": {
            "csv": str(output_dir / "state_value_metrics.csv"),
            "scatter": str(output_dir / "value_scatter.png"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
