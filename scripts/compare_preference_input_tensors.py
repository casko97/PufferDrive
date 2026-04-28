#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_preference_first32_deployment import (
    DEFAULT_GT_CAR,
    DEFAULT_GT_TRUCK,
    DEFAULT_POLICY_CAR,
    DEFAULT_POLICY_TRUCK,
    DEFAULT_WINDOW_LEN,
    _first_window_features,
    _index_rollouts,
    _load_rollout_payload,
)
from scripts.evaluate_preference_train_val_dataset import (
    DEFAULT_CHECKPOINT_STEM,
    DEFAULT_OUTPUT_DIR as DEFAULT_TRAIN_VAL_OUTPUT_DIR,
    DEFAULT_PREFERENCE_PATH,
    DEFAULT_REWARD_DIR,
    _jsonable,
    _load_reward_bundle,
    iter_preference_shards,
)


DEFAULT_TRAIN_VAL_SCORED = DEFAULT_TRAIN_VAL_OUTPUT_DIR / "scored_preference_windows.pt"
DEFAULT_OUTPUT_DIR = Path("outputs/preference_eval/preference_eval_gap_investigation")


@dataclass
class TensorStats:
    obs_dim: int
    action_dim: int
    count_segments: int = 0
    count_steps: int = 0
    obs_sum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    obs_sumsq: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    obs_min: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    obs_max: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    obs_zero_count: int = 0
    action_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    action_onehot_sum_min: float = float("inf")
    action_onehot_sum_max: float = float("-inf")
    action_onehot_sum_total: float = 0.0
    action_onehot_sum_squares: float = 0.0

    def __post_init__(self) -> None:
        if self.obs_sum.size == 0:
            self.obs_sum = np.zeros(self.obs_dim, dtype=np.float64)
            self.obs_sumsq = np.zeros(self.obs_dim, dtype=np.float64)
            self.obs_min = np.full(self.obs_dim, np.inf, dtype=np.float64)
            self.obs_max = np.full(self.obs_dim, -np.inf, dtype=np.float64)
            self.action_counts = np.zeros(self.action_dim, dtype=np.int64)

    def update(self, segment: np.ndarray) -> None:
        arr = np.asarray(segment, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.reshape(1, *arr.shape)
        if arr.ndim != 3 or arr.shape[-1] != self.obs_dim + self.action_dim:
            raise ValueError(f"expected (*, T, {self.obs_dim + self.action_dim}), got {arr.shape}")
        obs = arr[..., : self.obs_dim].reshape(-1, self.obs_dim)
        actions = arr[..., self.obs_dim :].reshape(-1, self.action_dim)
        indices = np.argmax(actions, axis=1)
        onehot_sums = actions.sum(axis=1)

        self.count_segments += int(arr.shape[0])
        self.count_steps += int(obs.shape[0])
        self.obs_sum += obs.sum(axis=0, dtype=np.float64)
        self.obs_sumsq += np.square(obs, dtype=np.float64).sum(axis=0)
        self.obs_min = np.minimum(self.obs_min, obs.min(axis=0))
        self.obs_max = np.maximum(self.obs_max, obs.max(axis=0))
        self.obs_zero_count += int(np.sum(obs == 0.0))
        self.action_counts += np.bincount(indices, minlength=self.action_dim)
        self.action_onehot_sum_min = min(self.action_onehot_sum_min, float(onehot_sums.min()))
        self.action_onehot_sum_max = max(self.action_onehot_sum_max, float(onehot_sums.max()))
        self.action_onehot_sum_total += float(onehot_sums.sum())
        self.action_onehot_sum_squares += float(np.square(onehot_sums, dtype=np.float64).sum())

    def summary(self) -> dict[str, Any]:
        if self.count_steps == 0:
            return {"segments": 0, "steps": 0}
        obs_mean = self.obs_sum / self.count_steps
        obs_var = np.maximum(self.obs_sumsq / self.count_steps - obs_mean * obs_mean, 0.0)
        obs_std = np.sqrt(obs_var)
        action_total = int(self.action_counts.sum())
        top_actions = [
            {"action": int(action), "fraction": float(count / max(action_total, 1)), "count": int(count)}
            for action, count in sorted(
                enumerate(self.action_counts.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
            if count > 0
        ]
        onehot_mean = self.action_onehot_sum_total / self.count_steps
        onehot_var = max(self.action_onehot_sum_squares / self.count_steps - onehot_mean * onehot_mean, 0.0)
        return {
            "segments": int(self.count_segments),
            "steps": int(self.count_steps),
            "obs_global_mean": float(np.mean(obs_mean)),
            "obs_global_std_mean": float(np.mean(obs_std)),
            "obs_global_min": float(np.min(self.obs_min)),
            "obs_global_max": float(np.max(self.obs_max)),
            "obs_zero_fraction": float(self.obs_zero_count / (self.count_steps * self.obs_dim)),
            "obs_l2_mean": float(np.linalg.norm(obs_mean)),
            "obs_std_l2_mean": float(np.linalg.norm(obs_std)),
            "action_entropy": _entropy(self.action_counts),
            "action_unique_count": int(np.sum(self.action_counts > 0)),
            "action_onehot_sum_mean": float(onehot_mean),
            "action_onehot_sum_std": float(np.sqrt(onehot_var)),
            "action_onehot_sum_min": float(self.action_onehot_sum_min),
            "action_onehot_sum_max": float(self.action_onehot_sum_max),
            "top_actions": top_actions,
        }


def _entropy(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / total
    return float(-np.sum(probs * np.log(probs)))


def _load_scored_rows(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload["rows"])


def _update_preference_stats(
    *,
    preference_path: Path,
    scored_rows: list[dict[str, Any]],
    stats_by_name: dict[str, TensorStats],
) -> None:
    rows_by_index = {int(row["global_index"]): row for row in scored_rows}
    needed = set(rows_by_index)
    for payload, shard_info in iter_preference_shards(preference_path):
        start = int(shard_info["start_index"])
        end = int(shard_info["end_index"])
        local_indices = sorted(index - start for index in needed if start <= index < end)
        if not local_indices:
            continue
        preferred = np.asarray(payload["preferred_sa"][local_indices], dtype=np.float32)
        rejected = np.asarray(payload["rejected_sa"][local_indices], dtype=np.float32)
        for row_idx, local_index in enumerate(local_indices):
            row = rows_by_index[start + local_index]
            split = str(row["split"])
            stats_by_name[f"dataset_{split}_truck_preferred"].update(preferred[row_idx])
            stats_by_name[f"dataset_{split}_car_rejected"].update(rejected[row_idx])


def _update_rollout_stats(
    *,
    payload_path: Path,
    stats: TensorStats,
    bundle: Any,
    window_len: int,
) -> dict[str, Any]:
    payload = _load_rollout_payload(payload_path)
    rollouts = _index_rollouts(payload)
    skipped = []
    for map_name, rollout in sorted(rollouts.items()):
        try:
            features = _first_window_features(rollout, bundle, window_len)
        except Exception as exc:
            skipped.append({"map_name": map_name, "reason": str(exc), "steps": int(rollout.get("steps", 0))})
            continue
        stats.update(features)
    return {"path": str(payload_path), "skipped": skipped}


def compare_inputs(
    *,
    preference_path: Path,
    train_val_scored: Path,
    policy_car: Path,
    policy_truck: Path,
    gt_car: Path,
    gt_truck: Path,
    reward_dir: Path,
    checkpoint_stem: str,
    output_dir: Path,
    window_len: int,
    device: str,
) -> dict[str, Any]:
    bundle = _load_reward_bundle(reward_dir, checkpoint_stem=checkpoint_stem, device=device)
    names = [
        "dataset_train_truck_preferred",
        "dataset_train_car_rejected",
        "dataset_validation_truck_preferred",
        "dataset_validation_car_rejected",
        "deployment_gt_truck",
        "deployment_gt_car",
        "deployment_policy_truck",
        "deployment_policy_car",
    ]
    stats_by_name = {name: TensorStats(bundle.obs_dim, bundle.action_dim) for name in names}
    scored_rows = _load_scored_rows(train_val_scored)
    _update_preference_stats(preference_path=preference_path, scored_rows=scored_rows, stats_by_name=stats_by_name)
    rollout_inputs = {
        "deployment_gt_truck": gt_truck,
        "deployment_gt_car": gt_car,
        "deployment_policy_truck": policy_truck,
        "deployment_policy_car": policy_car,
    }
    rollout_meta = {
        name: _update_rollout_stats(
            payload_path=path,
            stats=stats_by_name[name],
            bundle=bundle,
            window_len=window_len,
        )
        for name, path in rollout_inputs.items()
    }
    summary = {
        "obs_dim": int(bundle.obs_dim),
        "action_dim": int(bundle.action_dim),
        "window_len": int(window_len),
        "inputs": {
            "preference_path": str(preference_path),
            "train_val_scored": str(train_val_scored),
            **{name: str(path) for name, path in rollout_inputs.items()},
        },
        "rollout_skips": rollout_meta,
        "groups": {name: stats.summary() for name, stats in stats_by_name.items()},
    }

    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "input_tensor_distribution_summary.json"
    json_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    lines = [
        "# Preference Input Tensor Distribution",
        "",
        f"- Expected shape per scored segment: `[32, {bundle.obs_dim + bundle.action_dim}]`",
        f"- Observation/action split: `{bundle.obs_dim}` obs + `{bundle.action_dim}` one-hot action",
        "",
        "| group | segments | obs mean | obs std mean | zero frac | unique actions | top actions | onehot sum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for name in names:
        item = summary["groups"][name]
        top_actions = ", ".join(f"{a['action']}:{a['fraction']:.2f}" for a in item.get("top_actions", [])[:5])
        onehot = (
            f"{item.get('action_onehot_sum_mean', 0.0):.3f} "
            f"[{item.get('action_onehot_sum_min', 0.0):.1f},{item.get('action_onehot_sum_max', 0.0):.1f}]"
        )
        lines.append(
            f"| `{name}` | {item.get('segments', 0)} | {item.get('obs_global_mean', 0.0):.4f} | "
            f"{item.get('obs_global_std_mean', 0.0):.4f} | {item.get('obs_zero_fraction', 0.0):.3f} | "
            f"{item.get('action_unique_count', 0)} | {top_actions} | {onehot} |"
        )
    lines.extend(
        [
            "",
            "One-hot sum should be exactly 1.0 for every step. Top-action shifts are a quick proxy for action-distribution mismatch.",
        ]
    )
    md_path = report_dir / "input_tensor_distribution_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary_json": str(json_path), "summary_md": str(md_path), "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare preference train/val and deployment reward-model input tensors.")
    parser.add_argument("--preference-path", type=Path, default=DEFAULT_PREFERENCE_PATH)
    parser.add_argument("--train-val-scored", type=Path, default=DEFAULT_TRAIN_VAL_SCORED)
    parser.add_argument("--policy-car", type=Path, default=DEFAULT_POLICY_CAR)
    parser.add_argument("--policy-truck", type=Path, default=DEFAULT_POLICY_TRUCK)
    parser.add_argument("--gt-car", type=Path, default=DEFAULT_GT_CAR)
    parser.add_argument("--gt-truck", type=Path, default=DEFAULT_GT_TRUCK)
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--checkpoint-stem", type=str, default=DEFAULT_CHECKPOINT_STEM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-len", type=int, default=DEFAULT_WINDOW_LEN)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = compare_inputs(
        preference_path=args.preference_path,
        train_val_scored=args.train_val_scored,
        policy_car=args.policy_car,
        policy_truck=args.policy_truck,
        gt_car=args.gt_car,
        gt_truck=args.gt_truck,
        reward_dir=args.reward_dir,
        checkpoint_stem=args.checkpoint_stem,
        output_dir=args.output_dir,
        window_len=args.window_len,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in outputs.items() if key != "summary"}, indent=2))


if __name__ == "__main__":
    main()
