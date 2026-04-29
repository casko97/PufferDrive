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
from pufferlib.ocean.drive.drive import (
    _load_paired_fit_manifest,
    _paired_fit_manifest_map_names,
    _resolve_paired_fit_shard_path,
)
from scripts.plot_bc_car_rollout_grid import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_GT_CAR_ROLLOUTS,
    DEFAULT_MAP_ROOT,
    DEFAULT_REFERENCE_ROLLOUTS,
    _action_from_logits,
    _base_args,
    _first_value,
    _load_rollouts,
    _map_records,
    _record_state,
)
from scripts.run_packaged_drive_human_replay_eval import _prepare_single_map_dir


DEFAULT_FIT_EXPORT = Path(
    "/home/casko/phd-code/pufferdrive-kth/pufferlib/resources/drive/preferences/offline_fits/"
    "nuplan_boston_all_chunk64_paired_offline_fits.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/casko/phd-code/PufferDrive/outputs/bc_eval/"
    "bc_car_paired_offline_fits_20260428_164632/gt_anchor_rollouts"
)


class PairedFitLookup:
    def __init__(self, fit_export: Path, *, fit_side: str = "car", obs_key: str = "logged_obs_default"):
        self.fit_export = Path(fit_export)
        self.fit_side = fit_side
        self.obs_key = obs_key
        self.manifest = _load_paired_fit_manifest(self.fit_export)
        self.shared_maps = _paired_fit_manifest_map_names(self.manifest)
        if self.shared_maps is None:
            raise ValueError("Expected paired offline-fit manifest to include metadata.shared_maps")
        self._shard_by_map: dict[str, Path] = {}
        for shard_info in self.manifest.get("shards", []):
            start = int(shard_info.get("start_index", 0))
            end = int(shard_info.get("end_index", start + int(shard_info.get("map_count", 0))))
            shard_path = _resolve_paired_fit_shard_path(self.fit_export, shard_info)
            for map_name in self.shared_maps[start:end]:
                self._shard_by_map[str(map_name)] = shard_path
        self._pair_cache: dict[Path, dict[str, Any]] = {}

    def tensors(self, map_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        map_name = Path(str(map_name)).name
        shard_path = self._shard_by_map.get(map_name)
        if shard_path is None:
            raise KeyError(f"map {map_name} not found in paired offline-fit manifest")
        if shard_path not in self._pair_cache:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            self._pair_cache[shard_path] = payload["pairs"]
        pair = self._pair_cache[shard_path][map_name]
        side = pair[self.fit_side]
        if side.get("status") != "ok":
            raise ValueError(f"map {map_name} side {self.fit_side} status={side.get('status')!r}")
        obs = np.asarray(side[self.obs_key], dtype=np.float32)
        actions = np.asarray(side["actions"], dtype=np.int64)
        timesteps = np.asarray(side["logged_timestep"], dtype=np.int64)
        return obs, actions, timesteps


def _parse_starts(raw: str) -> list[int]:
    starts = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not starts:
        raise ValueError("at least one start timestep is required")
    return sorted(set(starts))


def _warm_recurrent_state(policy, state: dict[str, torch.Tensor], obs_history: np.ndarray, device: str) -> None:
    if not state or obs_history.size == 0:
        return
    with torch.no_grad():
        for obs in obs_history:
            ob_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32, device=device)
            policy.forward_eval(ob_tensor, state)


def _rollout_anchor(
    *,
    base_args: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    map_record: dict[str, Any],
    fit_lookup: PairedFitLookup,
    start_timestep: int,
    horizon: int,
    seed: int,
    deterministic: bool,
    workspace_dir: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)

    obs_history, _fit_actions, fit_timesteps = fit_lookup.tensors(map_record["source_map_name"])
    available_steps = int(obs_history.shape[0])
    if start_timestep >= available_steps:
        raise ValueError(f"start_timestep={start_timestep} but only {available_steps} fitted steps are available")
    if fit_timesteps.shape[0] and int(fit_timesteps[start_timestep]) != int(start_timestep):
        raise ValueError(f"unexpected logged timestep at index {start_timestep}: {fit_timesteps[start_timestep]}")

    max_steps = max(1, min(int(horizon), available_steps - int(start_timestep)))
    temp_root = Path(tempfile.mkdtemp(prefix=f"{map_record['source_map_name']}_bc_anchor_{start_timestep}_", dir=str(workspace_dir)))
    vecenv = None
    try:
        args = copy.deepcopy(base_args)
        args["env"]["init_steps"] = int(start_timestep)
        args["env"]["episode_length"] = max(int(args["env"].get("episode_length", 91)), int(start_timestep) + max_steps + 1)
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
        device = args["train"]["device"]
        state = {}
        if args["train"].get("use_rnn"):
            num_agents = int(vecenv.observation_space.shape[0])
            state = {
                "lstm_h": torch.zeros(num_agents, policy.hidden_size, device=device),
                "lstm_c": torch.zeros(num_agents, policy.hidden_size, device=device),
            }
            _warm_recurrent_state(policy, state, obs_history[:start_timestep], device)

        observations = []
        actions = []
        values = []
        entropy = []
        task_rewards = []
        dones = []
        truncs = []
        state_rows = {key: [] for key in ("x", "y", "z", "heading", "length", "width")}
        stop_reason = "horizon"

        for _step in range(max_steps):
            current_state = _record_state(driver)
            observations.append(np.asarray(obs[0], dtype=np.float32).copy())
            for key in state_rows:
                state_rows[key].append(current_state[key])

            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
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
            "scenario_type": map_record.get("scenario_type", ""),
            "delta_heading_deg": map_record.get("delta_heading_deg"),
            "start_timestep": int(start_timestep),
            "horizon": int(horizon),
            "steps": int(len(actions)),
            "stop_reason": stop_reason,
            "deterministic": bool(deterministic),
            "history_warmup_steps": int(start_timestep),
            "actions": np.asarray(actions, dtype=np.int32),
            "task_rewards": np.asarray(task_rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.uint8),
            "truncs": np.asarray(truncs, dtype=np.uint8),
            "values": np.asarray(values, dtype=np.float32),
            "entropy": np.asarray(entropy, dtype=np.float32),
            "observations": np.asarray(observations, dtype=np.float32),
            **{key: np.asarray(values_, dtype=np.float32) for key, values_ in state_rows.items()},
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
        }
    finally:
        if vecenv is not None:
            vecenv.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        gc.collect()


def collect_anchor_rollouts(
    *,
    config_path: Path,
    checkpoint_path: Path,
    reference_payload: dict[str, Any],
    map_root: Path,
    fit_export: Path,
    output_path: Path,
    device: str,
    starts: list[int],
    horizon: int,
    seed: int,
    deterministic: bool,
    max_maps: int,
) -> dict[str, Any]:
    map_records = _map_records(reference_payload, map_root)
    if max_maps > 0:
        map_records = map_records[:max_maps]
    base_args = _base_args(config_path, checkpoint_path, device)
    fit_lookup = PairedFitLookup(fit_export)
    workspace_dir = output_path.parent / ".bc_anchor_workspace"
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    rollouts = []
    errors = []
    try:
        total = len(map_records) * len(starts)
        done = 0
        for map_idx, record in enumerate(map_records):
            for start in starts:
                done += 1
                try:
                    print(f"[BC anchor] {done}/{total} {record['display_map_name']} start={start}", flush=True)
                    rollouts.append(
                        _rollout_anchor(
                            base_args=base_args,
                            config_path=config_path,
                            checkpoint_path=checkpoint_path,
                            map_record=record,
                            fit_lookup=fit_lookup,
                            start_timestep=start,
                            horizon=horizon,
                            seed=seed + map_idx * 1000 + start,
                            deterministic=deterministic,
                            workspace_dir=workspace_dir,
                        )
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "map_name": record["display_map_name"],
                            "source_map_name": record["source_map_name"],
                            "start_timestep": int(start),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    print(
                        f"[BC anchor] ERROR {record['display_map_name']} start={start}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)

    payload = {
        "format": "bc_gt_anchor_rollouts_v1",
        "model_name": "bc-car-paired-offline-fits",
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "fit_export": str(fit_export),
        "starts": [int(v) for v in starts],
        "horizon": int(horizon),
        "deterministic": bool(deterministic),
        "rollouts": rollouts,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def _by_map_start(anchor_payload: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for rollout in anchor_payload.get("rollouts", []):
        out.setdefault(str(rollout["map_name"]), {})[int(rollout["start_timestep"])] = rollout
    return out


def _xy(rollout: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rollout.get("x", []), dtype=np.float32).reshape(-1)
    y = np.asarray(rollout.get("y", []), dtype=np.float32).reshape(-1)
    count = min(len(x), len(y))
    return x[:count], y[:count]


def _top_actions(actions: np.ndarray, limit: int = 3) -> str:
    if actions.size == 0:
        return "no actions"
    unique, counts = np.unique(actions.astype(np.int64), return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    return ", ".join(f"{int(unique[i])}:{int(counts[i])}" for i in order)


def plot_anchor_grid(
    *,
    anchor_payload: dict[str, Any],
    gt_payload: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    gt_by_map = {str(row["map_name"]): row for row in gt_payload.get("rollouts", [])}
    anchor_by_map = _by_map_start(anchor_payload)
    map_names = sorted(anchor_by_map)
    starts = [int(v) for v in anchor_payload["starts"]]
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(starts)))

    cols = 4
    row_count = int(math.ceil(len(map_names) / cols))
    fig, axes = plt.subplots(row_count, cols, figsize=(18, max(8.8, row_count * 4.25)))
    axes = np.asarray(axes, dtype=object).reshape(row_count, cols)
    for ax in axes.flat:
        ax.axis("off")

    ade_rows = []
    for idx, map_name in enumerate(map_names):
        ax = axes.flat[idx]
        ax.axis("on")
        gt = gt_by_map.get(map_name)
        if gt is None:
            ax.text(0.5, 0.5, "missing GT", ha="center", va="center")
            continue
        gt_x, gt_y = _xy(gt)
        ax.plot(gt_x, gt_y, color="tab:blue", linewidth=2.2, alpha=0.85, label="GT car")
        ax.scatter([gt_x[0]], [gt_y[0]], color="tab:blue", marker="o", s=18, zorder=3)
        map_ades = []
        title_bits = []
        for color, start in zip(colors, starts):
            rollout = anchor_by_map[map_name].get(start)
            if rollout is None:
                continue
            x, y = _xy(rollout)
            if len(x) == 0:
                continue
            label = f"BC from t={start}" if idx == 0 else None
            ax.plot(x, y, color=color, linewidth=2.1, linestyle="--", alpha=0.95, label=label)
            ax.scatter([x[0]], [y[0]], color=color, marker="o", s=22, zorder=3)
            ax.scatter([x[-1]], [y[-1]], color=color, marker="x", s=36, zorder=3)
            if start < len(gt_x):
                n = min(len(x), len(gt_x) - start)
                if n > 0:
                    d = np.sqrt((x[:n] - gt_x[start : start + n]) ** 2 + (y[:n] - gt_y[start : start + n]) ** 2)
                    ade = float(np.mean(d))
                    fde = float(d[-1])
                    map_ades.append(ade)
                    ade_rows.append(
                        {
                            "map_name": map_name,
                            "start_timestep": int(start),
                            "steps": int(len(x)),
                            "ade": ade,
                            "fde": fde,
                            "stop_reason": rollout.get("stop_reason", ""),
                            "top_actions": _top_actions(np.asarray(rollout.get("actions", []), dtype=np.int64)),
                        }
                    )
                    title_bits.append(f"{start}:{ade:.1f}")
        title_metric = "ADE " + ", ".join(title_bits) if title_bits else "no BC rollouts"
        ax.set_title(f"{map_name}\n{title_metric}", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    mode = "argmax" if anchor_payload.get("deterministic", True) else "sampled"
    note = (
        f"Blue is the full fitted/GT car trajectory. Dashed colored segments are {mode} BC rollouts initialized at sampled GT timesteps. "
        "The recurrent state is warmed with logged GT observations before each start timestep. Circles mark rollout starts; x markers mark rollout ends. "
        "Panel titles show ADE versus the GT continuation for each start timestep."
    )
    fig.suptitle("BC Car Short Rollouts From GT Anchor States", fontsize=15)
    fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=10, color="dimgray", wrap=True)
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    mean_ade = float(np.mean([row["ade"] for row in ade_rows])) if ade_rows else float("nan")
    mean_fde = float(np.mean([row["fde"] for row in ade_rows])) if ade_rows else float("nan")
    return {
        "grid": str(output_path),
        "anchor_count": len(anchor_payload.get("rollouts", [])),
        "error_count": len(anchor_payload.get("errors", [])),
        "mean_anchor_ade": mean_ade,
        "mean_anchor_fde": mean_fde,
        "per_anchor": ade_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot short BC car rollouts initialized from GT trajectory timesteps.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference-rollouts", type=Path, default=DEFAULT_REFERENCE_ROLLOUTS)
    parser.add_argument("--gt-car-rollouts", type=Path, default=DEFAULT_GT_CAR_ROLLOUTS)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--fit-export", type=Path, default=DEFAULT_FIT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--starts", type=str, default="0,20,40,60")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-maps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    starts = _parse_starts(args.starts)
    reference_payload = _load_rollouts(args.reference_rollouts)
    gt_payload = _load_rollouts(args.gt_car_rollouts)
    suffix = "sampled" if args.sample else "argmax"
    rollout_path = args.output_dir / f"bc-car-gt-anchor-{suffix}_rollouts.pt"
    anchor_payload = collect_anchor_rollouts(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        reference_payload=reference_payload,
        map_root=args.map_root,
        fit_export=args.fit_export,
        output_path=rollout_path,
        device=args.device,
        starts=starts,
        horizon=args.horizon,
        seed=args.seed,
        deterministic=not args.sample,
        max_maps=args.max_maps,
    )
    outputs = plot_anchor_grid(
        anchor_payload=anchor_payload,
        gt_payload=gt_payload,
        output_path=args.output_dir / f"bc_car_gt_anchor_{suffix}_trajectory_grid.png",
    )
    outputs["rollouts"] = str(rollout_path)
    outputs["errors"] = anchor_payload.get("errors", [])
    summary_path = args.output_dir / f"bc_car_gt_anchor_{suffix}_summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
