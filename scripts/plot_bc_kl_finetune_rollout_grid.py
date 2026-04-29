#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pufferlib
from pufferlib import pufferl
from scripts.run_packaged_drive_human_replay_eval import (
    _load_packaged_config,
    _overlay_args,
    _prepare_single_map_dir,
)


DEFAULT_CONFIG = Path(
    "pufferlib/resources/drive/models/car-baseline-full-boston-bc-kl-finetune-500m/training.ini"
)
DEFAULT_BASELINE_CHECKPOINT = Path(
    "/home/casko/phd-code/PufferDrive/pufferlib/resources/drive/models/"
    "car-baseline-full-boston/puffer_drive_my9g62z5/model_puffer_drive_004000.pt"
)
DEFAULT_FINETUNE_DIR = Path("/home/casko/phd-code/PufferDrive/experiments_bc_kl/puffer_drive_e94omqwk")
DEFAULT_REFERENCE_ROLLOUTS = Path(
    "/home/casko/phd-code/PufferDrive/outputs/preference_eval/"
    "preference_rollout_model_compare_full_no_resets/rollouts/"
    "car-baseline-full-boston_rollouts.pt"
)
DEFAULT_GT_CAR_ROLLOUTS = Path(
    "/home/casko/phd-code/PufferDrive/outputs/preference_eval/"
    "preference_ground_truth_context_compare_full/rollouts/"
    "ground-truth-car-fit_rollouts.pt"
)
DEFAULT_MAP_ROOT = Path("/home/casko/phd-code/PufferDrive/datasets/nuplanCarBostonAll")
DEFAULT_OUTPUT_DIR = Path(
    "/home/casko/phd-code/PufferDrive/outputs/bc_kl_eval/"
    "car_500m_from_full_boston/trajectory_grids"
)


def _load_rollouts(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _latest_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("model_puffer_drive_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model_puffer_drive_*.pt checkpoint found in {checkpoint_dir}")
    return checkpoints[-1]


def _rollout_by_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["map_name"]): row for row in payload.get("rollouts", [])}


def _source_name(rollout: dict[str, Any]) -> str:
    source = rollout.get("source_map_name")
    if source:
        return Path(str(source)).name
    map_path = rollout.get("map_path")
    if map_path:
        return Path(str(map_path)).name
    return str(rollout["map_name"])


def _map_records(reference_payload: dict[str, Any], map_root: Path) -> list[dict[str, Any]]:
    records = []
    for rollout in reference_payload.get("rollouts", []):
        source_name = _source_name(rollout)
        map_path = map_root / source_name
        if not map_path.exists():
            raise FileNotFoundError(f"source map from reference artifact not found locally: {map_path}")
        records.append(
            {
                "display_map_name": str(rollout["map_name"]),
                "source_map_name": source_name,
                "map_path": str(map_path),
                "scenario_type": rollout.get("scenario_type", ""),
                "delta_heading_deg": rollout.get("delta_heading_deg"),
            }
        )
    return records


def _base_args(config_path: Path, checkpoint_path: Path, device: str) -> dict[str, Any]:
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = pufferl.load_config("puffer_drive")
    finally:
        sys.argv = original_argv

    packaged = _load_packaged_config(config_path)
    args = _overlay_args(args, packaged)
    args["load_model_path"] = str(checkpoint_path)
    args["load_id"] = None
    args["wandb"] = False
    args["neptune"] = False
    args.setdefault("train", {})
    args["train"]["device"] = device
    args["train"]["compile"] = False
    args["train"]["render"] = False
    args["train"]["torch_deterministic"] = True
    args.setdefault("eval", {})
    args["eval"]["wosac_num_rollouts"] = 1
    args["eval"]["human_replay_eval"] = False
    args["vec"] = {"backend": "PufferEnv", "num_envs": 1}

    env = args.setdefault("env", {})
    env.update(
        {
            "num_agents": 1,
            "action_type": "discrete",
            "dynamics_model": "classic",
            "observation_mode": "default",
            "sdc_runtime_truck_override": False,
            "control_mode": "control_sdc_only",
            "episode_length": int(env.get("episode_length", 91)),
            "init_steps": 0,
            "collision_behavior": 0,
            "offroad_behavior": 0,
            "termination_mode": 1,
            "init_mode": "create_all_valid",
        }
    )
    return args


def _first_value(values: Any, default: float = 0.0) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _record_state(driver) -> dict[str, float]:
    state = driver.get_global_agent_state(include_sdc_trailer=True)
    trailer = state.get("sdc_trailer", {})
    return {
        "x": _first_value(state["x"]),
        "y": _first_value(state["y"]),
        "z": _first_value(state["z"]),
        "heading": _first_value(state["heading"]),
        "length": _first_value(state["length"]),
        "width": _first_value(state["width"]),
        "trailer_has": _first_value(trailer.get("has_trailer", [0])),
        "trailer_x": _first_value(trailer.get("x", [0.0])),
        "trailer_y": _first_value(trailer.get("y", [0.0])),
        "trailer_heading": _first_value(trailer.get("heading", [0.0])),
    }


def _action_from_logits(logits, deterministic: bool):
    if isinstance(logits, torch.distributions.Normal):
        action = logits.mean if deterministic else logits.sample()
        entropy = logits.entropy()
        return action, entropy
    if isinstance(logits, (tuple, list)):
        if len(logits) != 1:
            raise ValueError("rollout comparison supports only single-head discrete policies")
        logits = logits[0]
    if deterministic:
        action = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        return action, entropy
    action, _logprob, entropy = pufferlib.pytorch.sample_logits(logits)
    return action, entropy


def _rollout_one_map(
    *,
    base_args: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    model_name: str,
    map_record: dict[str, Any],
    seed: int,
    max_steps: int,
    deterministic: bool,
    workspace_dir: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)

    temp_root = Path(tempfile.mkdtemp(prefix=f"{map_record['source_map_name']}_{model_name}_", dir=str(workspace_dir)))
    vecenv = None
    try:
        args = copy.deepcopy(base_args)
        single_map_dir = _prepare_single_map_dir(Path(map_record["map_path"]), temp_root)
        args["env"]["map_dir"] = str(single_map_dir)
        args["env"]["num_maps"] = 1
        args["eval"]["map_dir"] = str(single_map_dir)
        args["eval"]["wosac_num_maps"] = 1

        vecenv = pufferl.load_env("puffer_drive", args)
        policy = pufferl.load_policy(args, vecenv, env_name="puffer_drive")
        policy.eval()

        obs, _info = vecenv.reset()
        driver = vecenv.driver_env
        road_edges = driver.get_road_edge_polylines()
        device = args["train"]["device"]
        state = {}
        if args["train"].get("use_rnn"):
            num_agents = int(vecenv.observation_space.shape[0])
            state = {
                "lstm_h": torch.zeros(num_agents, policy.hidden_size, device=device),
                "lstm_c": torch.zeros(num_agents, policy.hidden_size, device=device),
            }

        observations = []
        actions = []
        task_rewards = []
        dones = []
        truncs = []
        values = []
        entropy = []
        state_rows = {key: [] for key in ("x", "y", "z", "heading", "length", "width", "trailer_has", "trailer_x", "trailer_y", "trailer_heading")}
        stop_reason = "max_steps"

        for _step in range(max_steps):
            current_state = _record_state(driver)
            observations.append(np.asarray(obs[0], dtype=np.float32).copy())
            for key in state_rows:
                state_rows[key].append(current_state[key])

            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, value = policy.forward_eval(ob_tensor, state)
                action, dist_entropy = _action_from_logits(logits, deterministic=deterministic)
                action_np = action.detach().cpu().numpy().reshape(vecenv.action_space.shape)

            actions.append(int(np.asarray(action_np).reshape(-1)[0]))
            values.append(float(np.asarray(value.detach().cpu().numpy()).reshape(-1)[0]))
            entropy.append(float(np.asarray(dist_entropy.detach().cpu().numpy()).reshape(-1)[0]))

            next_obs, reward, terminal, truncation, _info = vecenv.step(action_np)
            task_rewards.append(_first_value(reward))
            dones.append(int(np.asarray(terminal).reshape(-1)[0]))
            truncs.append(int(np.asarray(truncation).reshape(-1)[0]))
            obs = next_obs

            if dones[-1]:
                stop_reason = "terminal"
                break
            if truncs[-1]:
                stop_reason = "truncated"
                break

        return {
            "map_name": map_record["display_map_name"],
            "source_map_name": map_record["source_map_name"],
            "map_path": str(Path(map_record["map_path"]).resolve()),
            "model_name": model_name,
            "scenario_type": map_record.get("scenario_type", ""),
            "delta_heading_deg": map_record.get("delta_heading_deg"),
            "steps": int(len(actions)),
            "stop_reason": stop_reason,
            "deterministic": bool(deterministic),
            "actions": np.asarray(actions, dtype=np.int32),
            "task_rewards": np.asarray(task_rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.uint8),
            "truncs": np.asarray(truncs, dtype=np.uint8),
            "values": np.asarray(values, dtype=np.float32),
            "entropy": np.asarray(entropy, dtype=np.float32),
            "observations": np.asarray(observations, dtype=np.float32),
            "road_edges": road_edges,
            **{key: np.asarray(values_, dtype=np.float32) for key, values_ in state_rows.items()},
            "runtime_overrides": {
                "dynamics_model": "classic",
                "sdc_runtime_truck_override": False,
                "observation_mode": "default",
                "action_type": "discrete",
                "control_mode": "control_sdc_only",
            },
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
        }
    finally:
        if vecenv is not None:
            vecenv.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        gc.collect()


def collect_policy_rollouts(
    *,
    model_name: str,
    config_path: Path,
    checkpoint_path: Path,
    reference_payload: dict[str, Any],
    map_root: Path,
    output_path: Path,
    device: str,
    max_steps: int,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    map_records = _map_records(reference_payload, map_root)
    base_args = _base_args(config_path, checkpoint_path, device)
    workspace_dir = output_path.parent / ".policy_compare_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    rollouts = []
    errors = []
    try:
        for idx, record in enumerate(map_records):
            try:
                print(f"[{model_name}] {idx + 1}/{len(map_records)} {record['display_map_name']}", flush=True)
                rollouts.append(
                    _rollout_one_map(
                        base_args=base_args,
                        config_path=config_path,
                        checkpoint_path=checkpoint_path,
                        model_name=model_name,
                        map_record=record,
                        seed=seed + idx,
                        max_steps=max_steps,
                        deterministic=deterministic,
                        workspace_dir=workspace_dir,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "map_name": record["display_map_name"],
                        "source_map_name": record["source_map_name"],
                        "map_path": record["map_path"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"[{model_name}] ERROR {record['display_map_name']}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)

    payload = {
        "format": "drive_policy_rollouts_v1",
        "model_name": model_name,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "rollouts": rollouts,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def _xy(rollout: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    if rollout is None:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    x = np.asarray(rollout.get("x", []), dtype=np.float32).reshape(-1)
    y = np.asarray(rollout.get("y", []), dtype=np.float32).reshape(-1)
    count = min(len(x), len(y))
    return x[:count], y[:count]


def _trajectory_bounds(*rollouts: dict[str, Any] | None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    xs = []
    ys = []
    for rollout in rollouts:
        x, y = _xy(rollout)
        if len(x) == 0:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        return None
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    min_x, max_x = float(np.min(x_all)), float(np.max(x_all))
    min_y, max_y = float(np.min(y_all)), float(np.max(y_all))
    pad = max(5.0, 0.12 * max(max_x - min_x, max_y - min_y, 1.0))
    return (min_x - pad, max_x + pad), (min_y - pad, max_y + pad)


def _plot_road_edges(
    ax,
    road_edges: dict[str, Any] | None,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None,
) -> None:
    if not road_edges:
        return
    lengths = np.asarray(road_edges.get("lengths", []), dtype=np.int64).reshape(-1)
    x = np.asarray(road_edges.get("x", []), dtype=np.float32).reshape(-1)
    y = np.asarray(road_edges.get("y", []), dtype=np.float32).reshape(-1)
    if len(lengths) == 0 or len(x) == 0 or len(y) == 0:
        return

    xlim = ylim = None
    if bounds is not None:
        xlim, ylim = bounds
    point_idx = 0
    for length in lengths:
        end_idx = point_idx + int(length)
        poly_x = x[point_idx:end_idx]
        poly_y = y[point_idx:end_idx]
        point_idx = end_idx
        if len(poly_x) < 2:
            continue
        if xlim is not None and ylim is not None:
            in_view = (
                (poly_x >= xlim[0])
                & (poly_x <= xlim[1])
                & (poly_y >= ylim[0])
                & (poly_y <= ylim[1])
            )
            if not np.any(in_view):
                continue
        ax.plot(poly_x, poly_y, color="#8a8f98", linewidth=0.55, alpha=0.42, zorder=0)


def _ade(policy_rollout: dict[str, Any] | None, gt_rollout: dict[str, Any] | None, horizon: int | None = None) -> float | None:
    px, py = _xy(policy_rollout)
    gx, gy = _xy(gt_rollout)
    count = min(len(px), len(py), len(gx), len(gy))
    if horizon is not None:
        count = min(count, horizon)
    if count <= 0:
        return None
    return float(np.sqrt((px[:count] - gx[:count]) ** 2 + (py[:count] - gy[:count]) ** 2).mean())


def _action_summary(rollout: dict[str, Any] | None) -> str:
    if rollout is None:
        return "missing"
    actions = np.asarray(rollout.get("actions", []), dtype=np.int64).reshape(-1)
    if actions.size == 0:
        return "no actions"
    unique, counts = np.unique(actions, return_counts=True)
    order = np.argsort(counts)[::-1][:3]
    top = ", ".join(f"{int(unique[i])}:{int(counts[i])}" for i in order)
    return f"{int(rollout.get('steps', actions.size))} steps, {rollout.get('stop_reason', '')}, top {top}"


def _policy_metrics(payload: dict[str, Any], gt_payload: dict[str, Any]) -> dict[str, Any]:
    policy_by_map = _rollout_by_map(payload)
    gt_by_map = _rollout_by_map(gt_payload)
    first32 = []
    aligned = []
    stop_reasons: dict[str, int] = {}
    for map_name, rollout in policy_by_map.items():
        gt = gt_by_map.get(map_name)
        first32_ade = _ade(rollout, gt, horizon=32)
        aligned_ade = _ade(rollout, gt)
        if first32_ade is not None:
            first32.append(first32_ade)
        if aligned_ade is not None:
            aligned.append(aligned_ade)
        reason = str(rollout.get("stop_reason", ""))
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    return {
        "rollout_count": len(payload.get("rollouts", [])),
        "error_count": len(payload.get("errors", [])),
        "stop_reasons": stop_reasons,
        "mean_first32_ade_m": float(np.mean(first32)) if first32 else None,
        "mean_aligned_ade_m": float(np.mean(aligned)) if aligned else None,
    }


def plot_grid(
    *,
    baseline_payload: dict[str, Any],
    finetune_payload: dict[str, Any],
    gt_payload: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    baseline_by_map = _rollout_by_map(baseline_payload)
    finetune_by_map = _rollout_by_map(finetune_payload)
    gt_by_map = _rollout_by_map(gt_payload)
    map_names = sorted(set(baseline_by_map) | set(finetune_by_map) | set(gt_by_map))

    cols = 4
    row_count = int(math.ceil(len(map_names) / cols))
    fig, axes = plt.subplots(row_count, cols, figsize=(18, max(8.8, row_count * 4.35)))
    axes = np.asarray(axes, dtype=object).reshape(row_count, cols)
    for ax in axes.flat:
        ax.axis("off")

    specs = [
        ("GT car", gt_by_map, "tab:blue", "-"),
        ("Baseline PPO", baseline_by_map, "tab:orange", "--"),
        ("BC-KL finetune", finetune_by_map, "tab:green", "--"),
    ]
    for idx, map_name in enumerate(map_names):
        ax = axes.flat[idx]
        ax.axis("on")
        baseline_rollout = baseline_by_map.get(map_name)
        finetune_rollout = finetune_by_map.get(map_name)
        gt_rollout = gt_by_map.get(map_name)
        bounds = _trajectory_bounds(gt_rollout, baseline_rollout, finetune_rollout)
        road_source = baseline_rollout or finetune_rollout
        _plot_road_edges(
            ax,
            road_source.get("road_edges") if road_source is not None else None,
            bounds,
        )
        plotted = False
        for label, lookup, color, linestyle in specs:
            rollout = lookup.get(map_name)
            x, y = _xy(rollout)
            if len(x) == 0:
                continue
            plotted = True
            highlight = max(1, min(32, len(x)))
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.1, alpha=0.24, zorder=2)
            ax.plot(x[:highlight], y[:highlight], color=color, linestyle=linestyle, linewidth=3.3, label=label, zorder=4)
            ax.scatter([x[0]], [y[0]], color=color, marker="o", s=24, zorder=5)
            ax.scatter([x[highlight - 1]], [y[highlight - 1]], color=color, marker="x", s=38, zorder=5)
        if not plotted:
            ax.text(0.5, 0.5, "missing rollouts", ha="center", va="center")

        base_ade = _ade(baseline_by_map.get(map_name), gt_by_map.get(map_name), horizon=32)
        ft_ade = _ade(finetune_by_map.get(map_name), gt_by_map.get(map_name), horizon=32)
        ade_line = (
            f"first32 ADE base={base_ade:.2f}m, kl={ft_ade:.2f}m"
            if base_ade is not None and ft_ade is not None
            else "first32 ADE unavailable"
        )
        ax.set_title(f"{map_name}\n{ade_line}", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        if bounds is not None:
            ax.set_xlim(*bounds[0])
            ax.set_ylim(*bounds[1])
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    note = (
        "Light gray lines are road-edge map context. Blue is fitted/ground-truth car trajectory. Orange is the original car-baseline-full-boston PPO checkpoint. "
        "Green is the BC-KL finetuned checkpoint. Thick segments show the first 32 steps; faint continuations show the full collected rollout. "
        "Circle marks rollout start; x marks the end of the highlighted prefix. Both PPO policies are rolled out with deterministic argmax actions unless --sample is used."
    )
    fig.suptitle("Baseline PPO vs BC-KL Finetuned PPO Car Rollouts", fontsize=15)
    fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=10, color="dimgray", wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "grid": str(output_path),
        "baseline": _policy_metrics(baseline_payload, gt_payload),
        "finetune": _policy_metrics(finetune_payload, gt_payload),
        "per_map_action_summaries": {
            map_name: {
                "baseline": _action_summary(baseline_by_map.get(map_name)),
                "finetune": _action_summary(finetune_by_map.get(map_name)),
            }
            for map_name in map_names
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline PPO vs BC-KL finetuned PPO rollouts on the same maps.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--finetune-checkpoint", type=Path, default=None)
    parser.add_argument("--finetune-dir", type=Path, default=DEFAULT_FINETUNE_DIR)
    parser.add_argument("--reference-rollouts", type=Path, default=DEFAULT_REFERENCE_ROLLOUTS)
    parser.add_argument("--gt-car-rollouts", type=Path, default=DEFAULT_GT_CAR_ROLLOUTS)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-steps", type=int, default=91)
    parser.add_argument("--seed", type=int, default=4567)
    parser.add_argument("--sample", action="store_true", help="sample from logits instead of using argmax actions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    baseline_checkpoint = args.baseline_checkpoint.resolve()
    finetune_checkpoint = (
        args.finetune_checkpoint.resolve()
        if args.finetune_checkpoint is not None
        else _latest_checkpoint(args.finetune_dir.resolve())
    )
    output_dir = args.output_dir.resolve() / finetune_checkpoint.stem
    reference_payload = _load_rollouts(args.reference_rollouts.resolve())
    gt_payload = _load_rollouts(args.gt_car_rollouts.resolve())

    baseline_rollout_path = output_dir / "baseline-full-boston_rollouts.pt"
    finetune_rollout_path = output_dir / f"{finetune_checkpoint.stem}_rollouts.pt"
    baseline_payload = collect_policy_rollouts(
        model_name="baseline-full-boston",
        config_path=config_path,
        checkpoint_path=baseline_checkpoint,
        reference_payload=reference_payload,
        map_root=args.map_root.resolve(),
        output_path=baseline_rollout_path,
        device=args.device,
        max_steps=args.max_steps,
        seed=args.seed,
        deterministic=not args.sample,
    )
    finetune_payload = collect_policy_rollouts(
        model_name="bc-kl-finetune",
        config_path=config_path,
        checkpoint_path=finetune_checkpoint,
        reference_payload=reference_payload,
        map_root=args.map_root.resolve(),
        output_path=finetune_rollout_path,
        device=args.device,
        max_steps=args.max_steps,
        seed=args.seed,
        deterministic=not args.sample,
    )
    outputs = plot_grid(
        baseline_payload=baseline_payload,
        finetune_payload=finetune_payload,
        gt_payload=gt_payload,
        output_path=output_dir / "baseline_vs_bc_kl_finetune_trajectory_grid.png",
    )
    outputs.update(
        {
            "baseline_rollouts": str(baseline_rollout_path),
            "finetune_rollouts": str(finetune_rollout_path),
            "config": str(config_path),
            "baseline_checkpoint": str(baseline_checkpoint),
            "finetune_checkpoint": str(finetune_checkpoint),
            "deterministic": not args.sample,
            "errors": {
                "baseline": baseline_payload.get("errors", []),
                "finetune": finetune_payload.get("errors", []),
            },
        }
    )
    summary_path = output_dir / "baseline_vs_bc_kl_finetune_summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
