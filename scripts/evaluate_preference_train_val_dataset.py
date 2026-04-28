#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PREFERENCE_PATH = Path(
    "/home/casko/phd-code/pufferdrive-kth/"
    "pufferlib/resources/drive/preferences/turning_preferences/nuplan_boston_all_chunk64_turning_preferences.pt"
)
DEFAULT_REWARD_DIR = Path("pufferlib/resources/drive/preferences/models/turning_all_90_10_30rounds_continue1")
DEFAULT_EVAL_REPORT = DEFAULT_REWARD_DIR / "offline_truck_context_reward_eval.json"
DEFAULT_PREFERENCE_CONFIG = Path("pufferlib/resources/drive/models/car-baseline-full-boston-pref-from-base/conf1_preference.ini")
DEFAULT_SOURCE_PAIRED_FIT = Path(
    "/home/casko/phd-code/pufferdrive-kth/"
    "pufferlib/resources/drive/preferences/offline_fits/nuplan_boston_all_chunk64_paired_offline_fits.pt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/preference_eval/preference_train_val_dataset_eval")
DEFAULT_CHECKPOINT_STEM = "offline_truck_context"
DEFAULT_GALLERY_COUNT = 8

SCORED_FORMAT_VERSION = "preference_train_val_first_windows_v1"
PREFERENCES_FORMAT_MONOLITHIC = "truck_context_preferences_v1"
PREFERENCES_FORMAT_SHARDED = "sharded_truck_context_preferences_v1"


@dataclass(frozen=True)
class RewardBundle:
    obs_dim: int
    action_dim: int
    size_segment: int
    ensemble_size: int
    reward_model: Any


@dataclass(frozen=True)
class SelectedWindow:
    global_index: int
    split: str
    map_name: str
    timestep_start: int
    timestep_end: int
    shard_path: str
    local_index: int
    metadata: dict[str, Any]

    @property
    def sort_key(self) -> tuple[int, int, int]:
        return (self.timestep_start, self.timestep_end, self.global_index)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_preference_manifest(preference_path: Path) -> dict[str, Any]:
    payload = torch.load(preference_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("format") == PREFERENCES_FORMAT_SHARDED:
        return payload
    if isinstance(payload, dict) and "preferred_sa" in payload and "rejected_sa" in payload and "labels" in payload:
        return {
            "format": PREFERENCES_FORMAT_MONOLITHIC,
            "metadata": payload.get("metadata", {}),
            "shards": [
                {
                    "path": str(preference_path),
                    "window_count": int(len(np.asarray(payload["preferred_sa"]))),
                    "start_index": 0,
                    "end_index": int(len(np.asarray(payload["preferred_sa"]))),
                }
            ],
        }
    raise ValueError(f"unsupported preference payload at {preference_path}")


def _resolve_shard_path(preference_path: Path, shard_path_value: str) -> Path:
    shard_path = Path(shard_path_value)
    if shard_path.exists():
        return shard_path
    if shard_path.is_absolute():
        return shard_path

    candidate_same_dir = preference_path.parent / shard_path.name
    candidate_sibling_dir = preference_path.parent / f"{preference_path.stem}_shards" / shard_path.name
    if candidate_same_dir.exists():
        return candidate_same_dir
    if candidate_sibling_dir.exists():
        return candidate_sibling_dir
    return shard_path


def iter_preference_shards(preference_path: Path):
    manifest = load_preference_manifest(preference_path)
    if manifest["format"] == PREFERENCES_FORMAT_MONOLITHIC:
        yield torch.load(preference_path, map_location="cpu", weights_only=False), manifest["shards"][0]
        return

    for shard_info in manifest.get("shards", []):
        shard_path = _resolve_shard_path(preference_path, str(shard_info["path"]))
        yield torch.load(shard_path, map_location="cpu", weights_only=False), shard_info


def _softmax_preferred_probability(preferred_scores: np.ndarray, rejected_scores: np.ndarray) -> np.ndarray:
    logits = np.stack([preferred_scores, rejected_scores], axis=1).astype(np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probs[:, 0].astype(np.float32)


def _load_eval_report(eval_report_path: Path) -> dict[str, Any]:
    report = json.loads(eval_report_path.read_text(encoding="utf-8"))
    if "train_indices" not in report or "validation_indices" not in report:
        raise ValueError(f"eval report missing train/validation indices: {eval_report_path}")
    return report


def _split_lookup(eval_report: dict[str, Any]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for index in eval_report["train_indices"]:
        lookup[int(index)] = "train"
    for index in eval_report["validation_indices"]:
        int_index = int(index)
        if int_index in lookup:
            raise ValueError(f"global index {int_index} appears in both train and validation splits")
        lookup[int_index] = "validation"
    return lookup


def _load_reward_summary(reward_dir: Path) -> dict[str, Any]:
    summary_path = reward_dir / "offline_truck_context_reward_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"reward summary not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _validate_dataset_metadata(manifest: dict[str, Any], reward_summary: dict[str, Any]) -> None:
    meta = manifest.get("metadata", {})
    checks = [
        ("obs_dim", int(reward_summary["obs_dim"])),
        ("action_dim", int(reward_summary["action_dim"])),
        ("total_windows", int(reward_summary["total_windows"])),
    ]
    for key, expected in checks:
        actual = int(meta[key])
        if actual != expected:
            raise ValueError(f"preference dataset metadata mismatch for {key}: expected {expected}, got {actual}")

    window_len = int(meta["window_len"])
    expected_window_len = int(reward_summary["size_segment"])
    if window_len != expected_window_len:
        raise ValueError(
            f"preference dataset window_len mismatch: expected {expected_window_len}, got {window_len}"
        )


def _load_reward_bundle(reward_dir: Path, checkpoint_stem: str, device: str) -> RewardBundle:
    summary = _load_reward_summary(reward_dir)
    obs_dim = int(summary["obs_dim"])
    action_dim = int(summary["action_dim"])
    size_segment = int(summary["size_segment"])
    ensemble_size = int(summary["ensemble_size"])
    activation = str(summary.get("activation") or "tanh")

    from preferences import reward_model as reward_model_module

    reward_model_module.device = device
    reward_model = reward_model_module.RewardModel(
        ds=obs_dim,
        da=action_dim,
        ensemble_size=ensemble_size,
        lr=3e-4,
        mb_size=1,
        size_segment=size_segment,
        capacity=max(2, ensemble_size),
        activation=activation,
    )
    reward_model.load(str(reward_dir), checkpoint_stem)
    return RewardBundle(
        obs_dim=obs_dim,
        action_dim=action_dim,
        size_segment=size_segment,
        ensemble_size=ensemble_size,
        reward_model=reward_model,
    )


def _score_pair_batch(
    bundle: RewardBundle,
    preferred: np.ndarray,
    rejected: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    if preferred.ndim != 3 or rejected.ndim != 3:
        raise ValueError(f"expected rank-3 segment tensors, got {preferred.shape} and {rejected.shape}")
    if preferred.shape != rejected.shape:
        raise ValueError(f"preferred/rejected shape mismatch: {preferred.shape} vs {rejected.shape}")
    expected_shape = (bundle.size_segment, bundle.obs_dim + bundle.action_dim)
    if tuple(preferred.shape[1:]) != expected_shape:
        raise ValueError(f"segment shape mismatch: expected (*, {expected_shape}), got {preferred.shape}")

    preferred_members: list[np.ndarray] = []
    rejected_members: list[np.ndarray] = []
    for member in range(bundle.ensemble_size):
        preferred_reward = _to_numpy(bundle.reward_model.r_hat_member(preferred, member=member))
        rejected_reward = _to_numpy(bundle.reward_model.r_hat_member(rejected, member=member))
        preferred_members.append(preferred_reward.sum(axis=1).reshape(-1).astype(np.float32))
        rejected_members.append(rejected_reward.sum(axis=1).reshape(-1).astype(np.float32))

    preferred_member_matrix = np.stack(preferred_members, axis=1)
    rejected_member_matrix = np.stack(rejected_members, axis=1)
    preferred_score = preferred_member_matrix.mean(axis=1)
    rejected_score = rejected_member_matrix.mean(axis=1)
    preferred_std = preferred_member_matrix.std(axis=1)
    rejected_std = rejected_member_matrix.std(axis=1)
    margin = preferred_score - rejected_score
    pair_uncertainty = 0.5 * (preferred_std + rejected_std)

    member_probs = []
    for member in range(bundle.ensemble_size):
        member_probs.append(
            _softmax_preferred_probability(preferred_member_matrix[:, member], rejected_member_matrix[:, member])
        )
    prob_preferred = np.mean(np.stack(member_probs, axis=1), axis=1).astype(np.float32)

    labels_long = labels.reshape(-1).astype(np.int64)
    predicted_label = np.where(prob_preferred >= 0.5, 0, 1).astype(np.int64)
    correct = predicted_label == labels_long
    confidence = np.abs(prob_preferred - 0.5) * 2.0
    probability_of_label = np.where(labels_long == 0, prob_preferred, 1.0 - prob_preferred)
    ce_loss = -np.log(np.clip(probability_of_label, 1e-8, 1.0)).astype(np.float32)
    signed_margin = np.where(labels_long == 0, margin, -margin).astype(np.float32)

    return {
        "preferred_member_scores": preferred_member_matrix.astype(np.float32),
        "rejected_member_scores": rejected_member_matrix.astype(np.float32),
        "preferred_score": preferred_score.astype(np.float32),
        "rejected_score": rejected_score.astype(np.float32),
        "preferred_std": preferred_std.astype(np.float32),
        "rejected_std": rejected_std.astype(np.float32),
        "pair_uncertainty": pair_uncertainty.astype(np.float32),
        "margin": margin.astype(np.float32),
        "signed_margin": signed_margin,
        "prob_preferred": prob_preferred,
        "predicted_label": predicted_label,
        "correct": correct,
        "confidence": confidence.astype(np.float32),
        "ce_loss": ce_loss,
    }


def select_first_windows_per_map(
    preference_path: Path,
    split_by_index: dict[int, str],
    max_windows_per_split: int | None = None,
) -> list[SelectedWindow]:
    by_map: dict[str, SelectedWindow] = {}
    seen_indices: set[int] = set()

    for payload, shard_info in iter_preference_shards(preference_path):
        metadata = list(payload.get("window_metadata", []))
        start_index = int(shard_info["start_index"])
        if len(metadata) != int(shard_info["window_count"]):
            raise ValueError(
                f"window_metadata length mismatch for {shard_info.get('path')}: "
                f"expected {shard_info['window_count']}, got {len(metadata)}"
            )

        for local_index, window_meta in enumerate(metadata):
            global_index = start_index + local_index
            if global_index in seen_indices:
                raise ValueError(f"duplicate global preference index encountered: {global_index}")
            seen_indices.add(global_index)
            split = split_by_index.get(global_index)
            if split is None:
                raise ValueError(f"global preference index {global_index} is missing from train/validation splits")

            map_name = str(window_meta.get("map_name", ""))
            if not map_name:
                raise ValueError(f"preference window {global_index} is missing map_name metadata")

            candidate = SelectedWindow(
                global_index=global_index,
                split=split,
                map_name=map_name,
                timestep_start=int(window_meta["timestep_start"]),
                timestep_end=int(window_meta["timestep_end"]),
                shard_path=str(shard_info["path"]),
                local_index=local_index,
                metadata=dict(window_meta),
            )
            existing = by_map.get(map_name)
            if existing is None or candidate.sort_key < existing.sort_key:
                by_map[map_name] = candidate

    selected = sorted(by_map.values(), key=lambda item: item.global_index)
    if max_windows_per_split is not None:
        capped: list[SelectedWindow] = []
        counts = {"train": 0, "validation": 0}
        for item in selected:
            if counts[item.split] >= max_windows_per_split:
                continue
            capped.append(item)
            counts[item.split] += 1
        selected = capped
    return selected


def score_selected_windows(
    preference_path: Path,
    selected_windows: list[SelectedWindow],
    bundle: RewardBundle,
) -> list[dict[str, Any]]:
    selected_by_index = {item.global_index: item for item in selected_windows}
    selected_indices = set(selected_by_index)
    rows: list[dict[str, Any]] = []

    for payload, shard_info in iter_preference_shards(preference_path):
        start_index = int(shard_info["start_index"])
        end_index = int(shard_info["end_index"])
        local_indices = [
            global_index - start_index
            for global_index in sorted(selected_indices)
            if start_index <= global_index < end_index
        ]
        if not local_indices:
            continue

        preferred = np.asarray(payload["preferred_sa"][local_indices], dtype=np.float32)
        rejected = np.asarray(payload["rejected_sa"][local_indices], dtype=np.float32)
        labels = np.asarray(payload["labels"][local_indices], dtype=np.float32)
        scored = _score_pair_batch(bundle, preferred, rejected, labels)

        for row_idx, local_index in enumerate(local_indices):
            global_index = start_index + local_index
            selected = selected_by_index[global_index]
            label = int(labels[row_idx].reshape(-1)[0])
            row = {
                "split": selected.split,
                "global_index": int(global_index),
                "map_name": selected.map_name,
                "timestep_start": int(selected.timestep_start),
                "timestep_end": int(selected.timestep_end),
                "label": label,
                "predicted_label": int(scored["predicted_label"][row_idx]),
                "correct": bool(scored["correct"][row_idx]),
                "prob_preferred": float(scored["prob_preferred"][row_idx]),
                "confidence": float(scored["confidence"][row_idx]),
                "ce_loss": float(scored["ce_loss"][row_idx]),
                "preferred_score": float(scored["preferred_score"][row_idx]),
                "rejected_score": float(scored["rejected_score"][row_idx]),
                "margin": float(scored["margin"][row_idx]),
                "signed_margin": float(scored["signed_margin"][row_idx]),
                "preferred_std": float(scored["preferred_std"][row_idx]),
                "rejected_std": float(scored["rejected_std"][row_idx]),
                "pair_uncertainty": float(scored["pair_uncertainty"][row_idx]),
                "preferred_member_scores": scored["preferred_member_scores"][row_idx].astype(np.float32),
                "rejected_member_scores": scored["rejected_member_scores"][row_idx].astype(np.float32),
                "shard_path": selected.shard_path,
                "local_index": int(selected.local_index),
                "window_metadata": dict(selected.metadata),
            }
            for key in (
                "scenario_delta_heading_deg",
                "truck_context_aligned_steps",
                "window_start_distance_m",
                "window_pair_ade",
                "window_pair_fde",
                "scenario_pair_ade",
                "scenario_pair_fde",
                "truck_self_ade",
                "truck_self_fde",
                "car_on_truck_ade",
                "car_on_truck_fde",
            ):
                row[key] = selected.metadata.get(key)
            rows.append(row)

    found = {int(row["global_index"]) for row in rows}
    missing = selected_indices - found
    if missing:
        raise ValueError(f"selected windows were not found while scoring: {sorted(missing)[:10]}")
    return sorted(rows, key=lambda row: int(row["global_index"]))


def _split_rows(rows: Iterable[dict[str, Any]], split: str | None = None) -> list[dict[str, Any]]:
    if split is None:
        return list(rows)
    return [row for row in rows if row["split"] == split]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [_safe_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else 0.0


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "window_count": 0,
            "map_count": 0,
            "accuracy": 0.0,
            "mean_ce_loss": 0.0,
            "mean_margin": 0.0,
            "mean_signed_margin": 0.0,
            "mean_confidence": 0.0,
            "mean_pair_uncertainty": 0.0,
            "mean_prob_preferred": 0.0,
        }
    return {
        "window_count": len(rows),
        "map_count": len({row["map_name"] for row in rows}),
        "accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "mean_ce_loss": _mean(rows, "ce_loss"),
        "mean_margin": _mean(rows, "margin"),
        "mean_signed_margin": _mean(rows, "signed_margin"),
        "mean_confidence": _mean(rows, "confidence"),
        "mean_pair_uncertainty": _mean(rows, "pair_uncertainty"),
        "mean_prob_preferred": _mean(rows, "prob_preferred"),
    }


def build_summary(
    rows: list[dict[str, Any]],
    *,
    preference_path: Path,
    reward_dir: Path,
    eval_report: dict[str, Any],
    manifest: dict[str, Any],
    max_windows_per_split: int | None,
) -> dict[str, Any]:
    by_split = {
        "train": _summary_for_rows(_split_rows(rows, "train")),
        "validation": _summary_for_rows(_split_rows(rows, "validation")),
    }
    return {
        "format": SCORED_FORMAT_VERSION,
        "selection": "first_window_per_map",
        "preference_path": str(preference_path),
        "reward_dir": str(reward_dir),
        "max_windows_per_split": max_windows_per_split,
        "dataset": {
            "total_windows": int(manifest["metadata"]["total_windows"]),
            "window_len": int(manifest["metadata"]["window_len"]),
            "obs_dim": int(manifest["metadata"]["obs_dim"]),
            "action_dim": int(manifest["metadata"]["action_dim"]),
            "shard_count": len(manifest.get("shards", [])),
        },
        "stored_split": {
            "train_windows": len(eval_report["train_indices"]),
            "validation_windows": len(eval_report["validation_indices"]),
            "validation_accuracy_all_windows": eval_report.get("validation_accuracy"),
            "validation_loss_all_windows": eval_report.get("validation_loss"),
        },
        "selected": {
            "window_count": len(rows),
            "map_count": len({row["map_name"] for row in rows}),
            "splits": by_split,
        },
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.ndarray, list, tuple, dict)):
        return json.dumps(_jsonable(value), sort_keys=True)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_window_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "global_index",
        "map_name",
        "timestep_start",
        "timestep_end",
        "label",
        "predicted_label",
        "correct",
        "prob_preferred",
        "confidence",
        "ce_loss",
        "preferred_score",
        "rejected_score",
        "margin",
        "signed_margin",
        "preferred_std",
        "rejected_std",
        "pair_uncertainty",
        "scenario_delta_heading_deg",
        "truck_context_aligned_steps",
        "window_start_distance_m",
        "window_pair_ade",
        "window_pair_fde",
        "scenario_pair_ade",
        "scenario_pair_fde",
        "truck_self_ade",
        "truck_self_fde",
        "car_on_truck_ade",
        "car_on_truck_fde",
        "preferred_member_scores",
        "rejected_member_scores",
        "shard_path",
        "local_index",
        "window_metadata",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return output_path


def write_by_map_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((row["split"], row["map_name"]), []).append(row)

    fieldnames = [
        "split",
        "map_name",
        "window_count",
        "first_global_index",
        "timestep_start",
        "timestep_end",
        "accuracy",
        "preferred_win_fraction",
        "mean_margin",
        "mean_signed_margin",
        "mean_prob_preferred",
        "mean_confidence",
        "mean_pair_uncertainty",
        "mean_preferred_score",
        "mean_rejected_score",
        "scenario_delta_heading_deg",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for (split, map_name), map_rows in sorted(by_key.items(), key=lambda item: (item[0][0], item[0][1])):
            first = min(map_rows, key=lambda row: int(row["global_index"]))
            writer.writerow(
                {
                    "split": split,
                    "map_name": map_name,
                    "window_count": len(map_rows),
                    "first_global_index": int(first["global_index"]),
                    "timestep_start": int(first["timestep_start"]),
                    "timestep_end": int(first["timestep_end"]),
                    "accuracy": float(np.mean([bool(row["correct"]) for row in map_rows])),
                    "preferred_win_fraction": float(np.mean([float(row["margin"]) > 0.0 for row in map_rows])),
                    "mean_margin": _mean(map_rows, "margin"),
                    "mean_signed_margin": _mean(map_rows, "signed_margin"),
                    "mean_prob_preferred": _mean(map_rows, "prob_preferred"),
                    "mean_confidence": _mean(map_rows, "confidence"),
                    "mean_pair_uncertainty": _mean(map_rows, "pair_uncertainty"),
                    "mean_preferred_score": _mean(map_rows, "preferred_score"),
                    "mean_rejected_score": _mean(map_rows, "rejected_score"),
                    "scenario_delta_heading_deg": first.get("scenario_delta_heading_deg"),
                }
            )
    return output_path


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float32)


def _plot_hist(ax, rows: list[dict[str, Any]], key: str, title: str) -> None:
    for split, color in (("train", "tab:blue"), ("validation", "tab:orange")):
        split_values = _values(_split_rows(rows, split), key)
        if split_values.size == 0:
            continue
        ax.hist(split_values, bins=30, alpha=0.55, density=True, label=split, color=color)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)


def _plot_note(fig, note: str, *, width: int = 145) -> None:
    wrapped = "\n".join(textwrap.wrap(note, width=width))
    fig.text(0.01, 0.012, wrapped, ha="left", va="bottom", fontsize=9, color="dimgray")


def _split_stats_note(rows: list[dict[str, Any]]) -> str:
    parts = []
    for split in ("train", "validation"):
        split_rows = _split_rows(rows, split)
        if not split_rows:
            continue
        accuracy = float(np.mean([bool(row["correct"]) for row in split_rows]))
        parts.append(f"{split}: n={len(split_rows)}, acc={accuracy:.3f}, mean margin={_mean(split_rows, 'margin'):.2f}")
    return "; ".join(parts)


def plot_reward_histograms(rows: list[dict[str, Any]], output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("preferred_score", "Truck/context (preferred) segment score"),
        ("rejected_score", "Car (rejected) segment score"),
        ("margin", "Truck/context - car margin"),
        ("prob_preferred", "P(truck/context preferred over car)"),
    ]
    for ax, (key, title) in zip(axes.flat, specs):
        _plot_hist(ax, rows, key, title)
    _plot_note(
        fig,
        "How to read: scores are reward-model sums over one 32-step first window per map. "
        "Truck/context is the preferred side and car is the rejected side. "
        "Margin = truck/context score - car score; margin > 0 and p(truck/context) > 0.5 mean the model prefers "
        f"truck/context. {_split_stats_note(rows)}",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_uncertainty(rows: list[dict[str, Any]], output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        ("preferred_std", "Truck/context (preferred) ensemble std"),
        ("rejected_std", "Car (rejected) ensemble std"),
        ("pair_uncertainty", "Mean pair uncertainty"),
    ]
    for ax, (key, title) in zip(axes, specs):
        _plot_hist(ax, rows, key, title)
    _plot_note(
        fig,
        "How to read: ensemble std measures disagreement between reward-model members; higher values mean less certain "
        "preference scoring. Pair uncertainty is the average std for the truck/context and car segments.",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _representative_by_margin(rows: list[dict[str, Any]], limit_per_side: int = 12) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for split in ("train", "validation"):
        split_rows = sorted(_split_rows(rows, split), key=lambda row: float(row["margin"]))
        candidates = split_rows[:limit_per_side] + split_rows[-limit_per_side:]
        for row in candidates:
            global_index = int(row["global_index"])
            if global_index not in seen:
                selected.append(row)
                seen.add(global_index)
    return sorted(selected, key=lambda row: (row["split"], float(row["margin"])))


def plot_margin_bars(rows: list[dict[str, Any]], output_path: Path) -> Path:
    reps = _representative_by_margin(rows)
    if not reps:
        raise ValueError("no rows available for margin plot")
    fig, ax = plt.subplots(figsize=(max(12, len(reps) * 0.35), 6))
    x = np.arange(len(reps))
    colors = ["tab:blue" if row["split"] == "train" else "tab:orange" for row in reps]
    ax.bar(x, [float(row["margin"]) for row in reps], color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['map_name']}\n{row['split']}" for row in reps], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("First-window segment margin")
    ax.set_title("Representative first-window truck/context-over-car margins by map")
    ax.grid(True, axis="y", alpha=0.2)
    _plot_note(
        fig,
        "How to read: each bar is truck/context score - car score for one representative map's first 32-step window. "
        "Positive bars favor truck/context; negative bars favor car.",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_win_fraction(rows: list[dict[str, Any]], output_path: Path) -> Path:
    reps = _representative_by_margin(rows)
    if not reps:
        raise ValueError("no rows available for win-fraction plot")
    fig, ax = plt.subplots(figsize=(max(12, len(reps) * 0.35), 5))
    x = np.arange(len(reps))
    colors = ["tab:blue" if row["split"] == "train" else "tab:orange" for row in reps]
    ax.bar(x, [1.0 if float(row["margin"]) > 0.0 else 0.0 for row in reps], color=colors, alpha=0.85)
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['map_name']}\n{row['split']}" for row in reps], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Truck/context wins first window")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("First-window truck/context win indicator by representative map")
    ax.grid(True, axis="y", alpha=0.2)
    _plot_note(
        fig,
        "How to read: for this first-window eval, 1 means truck/context score > car score for that map; "
        "0 means car scored higher. The dashed 0.5 line is a neutral reference.",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_cumulative_scores(rows: list[dict[str, Any]], output_path: Path) -> Path:
    reps = _representative_by_margin(rows, limit_per_side=8)
    if not reps:
        raise ValueError("no rows available for cumulative score plot")
    fig, ax = plt.subplots(figsize=(max(12, len(reps) * 0.45), 6))
    x = np.arange(len(reps))
    width = 0.38
    ax.bar(x - width / 2, [float(row["preferred_score"]) for row in reps], width=width, label="truck/context (preferred)")
    ax.bar(x + width / 2, [float(row["rejected_score"]) for row in reps], width=width, label="car (rejected)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['map_name']}\n{row['split']}" for row in reps], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("32-step segment reward sum")
    ax.set_title("First-window truck/context vs car segment scores")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    _plot_note(
        fig,
        "How to read: paired bars are reward-model sums over the first 32-step segment. "
        "The truck/context bar being taller means the model assigns more reward to the labeled preferred behavior.",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _iter_paired_fit_payloads_light(source_path: Path):
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "pairs" in payload:
        yield payload["pairs"]
        return
    if not (isinstance(payload, dict) and payload.get("format") == "sharded_paired_fits_v1"):
        raise ValueError(f"unsupported paired-fit payload at {source_path}")
    for shard_info in payload.get("shards", []):
        shard_path = Path(shard_info["path"])
        if not shard_path.exists():
            same_dir = source_path.parent / shard_path.name
            sibling_dir = source_path.parent / f"{source_path.stem}_shards" / shard_path.name
            if same_dir.exists():
                shard_path = same_dir
            elif sibling_dir.exists():
                shard_path = sibling_dir
        shard_payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        yield shard_payload["pairs"]


def _load_trajectory_lookup(source_paired_fit: Path | None, map_names: set[str]) -> dict[str, dict[str, Any]]:
    if source_paired_fit is None or not source_paired_fit.exists() or not map_names:
        return {}
    remaining = set(map_names)
    trajectories: dict[str, dict[str, Any]] = {}
    for pairs in _iter_paired_fit_payloads_light(source_paired_fit):
        for map_name in list(remaining):
            pair = pairs.get(map_name)
            if pair is None:
                continue
            replay = pair.get("truck_context_replay", {})
            if replay.get("status") != "ok":
                remaining.remove(map_name)
                continue
            trajectories[map_name] = {
                "preferred": replay.get("truck_branch", {}),
                "rejected": replay.get("car_branch", {}),
            }
            remaining.remove(map_name)
        if not remaining:
            break
    return trajectories


def _gallery_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    candidates = []
    incorrect = [row for row in rows if not bool(row["correct"])]
    candidates.extend(sorted(incorrect, key=lambda row: float(row["confidence"]), reverse=True))
    candidates.extend(sorted(rows, key=lambda row: float(row["margin"])))
    candidates.extend(sorted(rows, key=lambda row: float(row["margin"]), reverse=True))
    candidates.extend(sorted(rows, key=lambda row: float(row["pair_uncertainty"]), reverse=True))
    for row in candidates:
        global_index = int(row["global_index"])
        if global_index in seen:
            continue
        selected.append(row)
        seen.add(global_index)
        if len(selected) >= count:
            break
    return selected


def plot_representative_gallery(
    rows: list[dict[str, Any]],
    output_path: Path,
    source_paired_fit: Path | None,
    gallery_count: int,
) -> Path:
    reps = _gallery_rows(rows, gallery_count)
    if not reps:
        raise ValueError("no rows available for representative gallery")
    trajectories = _load_trajectory_lookup(source_paired_fit, {row["map_name"] for row in reps})

    fig, axes = plt.subplots(len(reps), 2, figsize=(12, max(3, 3 * len(reps))))
    axes = np.asarray(axes, dtype=object).reshape(len(reps), 2)
    for row_idx, row in enumerate(reps):
        traj_ax, score_ax = axes[row_idx]
        trajectory = trajectories.get(row["map_name"])
        if trajectory:
            for branch_key, label, color in (
                ("preferred", "truck/context (preferred)", "tab:blue"),
                ("rejected", "car (rejected)", "tab:orange"),
            ):
                branch = trajectory.get(branch_key, {})
                x_values = np.asarray(branch.get("rollout_x", []), dtype=np.float32)
                y_values = np.asarray(branch.get("rollout_y", []), dtype=np.float32)
                if x_values.size and y_values.size:
                    count = min(x_values.size, y_values.size)
                    traj_ax.plot(x_values[:count], y_values[:count], label=label, color=color, linewidth=1.8)
                    start = min(max(int(row["timestep_start"]), 0), count - 1)
                    end = min(max(int(row["timestep_end"]) - 1, 0), count - 1)
                    traj_ax.scatter([x_values[start], x_values[end]], [y_values[start], y_values[end]], color=color, s=18)
            traj_ax.set_aspect("equal", adjustable="box")
            traj_ax.legend(fontsize=8)
        else:
            traj_ax.text(0.5, 0.5, "paired-fit trajectory unavailable", ha="center", va="center")
        traj_ax.set_title(f"{row['map_name']} ({row['split']})")
        traj_ax.grid(True, alpha=0.2)

        score_ax.bar(["truck/context\npreferred", "car\nrejected"], [float(row["preferred_score"]), float(row["rejected_score"])])
        score_ax.set_title(
            f"idx={row['global_index']} margin={row['margin']:.3f} "
            f"p={row['prob_preferred']:.3f} correct={int(bool(row['correct']))}"
        )
        score_ax.grid(True, axis="y", alpha=0.2)
    fig.suptitle(
        "Representative windows: truck/context is preferred, car is rejected; margin = truck/context - car, p = P(truck/context wins), "
        "correct = predicted label matches dataset label",
        fontsize=10,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_markdown_report(summary: dict[str, Any], output_path: Path) -> Path:
    selected = summary["selected"]
    lines = [
        "# Preference Train/Validation First-Window Eval",
        "",
        f"- Selection: `{summary['selection']}`",
        f"- Selected windows: `{selected['window_count']}` across `{selected['map_count']}` maps",
        f"- Dataset windows: `{summary['dataset']['total_windows']}` across `{summary['dataset']['shard_count']}` shards",
        "",
        "## Stored Full Split",
        f"- train windows: `{summary['stored_split']['train_windows']}`",
        f"- validation windows: `{summary['stored_split']['validation_windows']}`",
        f"- stored validation accuracy all windows: `{summary['stored_split']['validation_accuracy_all_windows']}`",
        "",
        "## First-Window Results",
    ]
    for split in ("train", "validation"):
        stats = selected["splits"][split]
        lines.extend(
            [
                f"### {split}",
                f"- windows/maps: `{stats['window_count']}`",
                f"- accuracy: `{stats['accuracy']:.6f}`",
                f"- mean_ce_loss: `{stats['mean_ce_loss']:.6f}`",
                f"- mean_margin: `{stats['mean_margin']:.6f}`",
                f"- mean_confidence: `{stats['mean_confidence']:.6f}`",
                f"- mean_pair_uncertainty: `{stats['mean_pair_uncertainty']:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "- Each map contributes only its earliest 32-step preference window.",
            "- Split labels come from the original stored train/validation global index lists.",
            "- This report is a first-window sanity check, so its split counts and accuracies are not expected to match the all-window training summary.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_scored_payload(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> Path:
    serializable_rows = []
    for row in rows:
        serializable_rows.append({key: _jsonable(value) for key, value in row.items()})
    payload = {
        "format": SCORED_FORMAT_VERSION,
        "summary": summary,
        "rows": serializable_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def generate_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    source_paired_fit: Path | None,
    gallery_count: int,
) -> dict[str, str]:
    report_dir = output_dir / "report"
    outputs = {
        "scored_windows": str(write_scored_payload(rows, summary, output_dir / "scored_preference_windows.pt")),
        "window_csv": str(write_window_csv(rows, report_dir / "preference_reward_windows.csv")),
        "by_map_csv": str(write_by_map_csv(rows, report_dir / "preference_reward_by_map.csv")),
        "histograms": str(plot_reward_histograms(rows, report_dir / "reward_histograms.png")),
        "uncertainty": str(plot_uncertainty(rows, report_dir / "preference_uncertainty.png")),
        "margin_by_map": str(plot_margin_bars(rows, report_dir / "segment_mean_margin_by_map.png")),
        "win_fraction_by_map": str(plot_win_fraction(rows, report_dir / "segment_win_fraction_by_map.png")),
        "cumulative_rewards": str(plot_cumulative_scores(rows, report_dir / "cumulative_rewards_by_map.png")),
        "gallery": str(
            plot_representative_gallery(
                rows,
                report_dir / "representative_window_gallery.png",
                source_paired_fit=source_paired_fit,
                gallery_count=gallery_count,
            )
        ),
    }
    summary_path = report_dir / "report_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    outputs["summary_json"] = str(summary_path)
    outputs["narrative"] = str(write_markdown_report(summary, report_dir / "report_summary.md"))
    return outputs


def evaluate_first_windows(
    *,
    preference_path: Path,
    reward_dir: Path,
    eval_report_path: Path,
    output_dir: Path,
    preference_config_path: Path,
    source_paired_fit: Path | None,
    checkpoint_stem: str = DEFAULT_CHECKPOINT_STEM,
    device: str = "cpu",
    max_windows_per_split: int | None = None,
    gallery_count: int = DEFAULT_GALLERY_COUNT,
    bundle: RewardBundle | None = None,
) -> dict[str, str]:
    preference_path = preference_path.expanduser()
    reward_dir = reward_dir.expanduser()
    eval_report_path = eval_report_path.expanduser()
    output_dir = output_dir.expanduser()
    source_paired_fit = source_paired_fit.expanduser() if source_paired_fit is not None else None

    manifest = load_preference_manifest(preference_path)
    reward_summary = _load_reward_summary(reward_dir)
    _validate_dataset_metadata(manifest, reward_summary)
    eval_report = _load_eval_report(eval_report_path)
    split_by_index = _split_lookup(eval_report)
    selected = select_first_windows_per_map(
        preference_path,
        split_by_index,
        max_windows_per_split=max_windows_per_split,
    )
    if not selected:
        raise ValueError("no first windows were selected")

    effective_bundle = bundle or _load_reward_bundle(reward_dir, checkpoint_stem=checkpoint_stem, device=device)
    rows = score_selected_windows(preference_path, selected, effective_bundle)
    summary = build_summary(
        rows,
        preference_path=preference_path,
        reward_dir=reward_dir,
        eval_report=eval_report,
        manifest=manifest,
        max_windows_per_split=max_windows_per_split,
    )
    summary["preference_config_path"] = str(preference_config_path)
    summary["source_paired_fit"] = str(source_paired_fit) if source_paired_fit is not None else None
    return generate_report(
        rows,
        summary,
        output_dir,
        source_paired_fit=source_paired_fit,
        gallery_count=gallery_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the first 32-step preference window for each map in the train/validation dataset."
    )
    parser.add_argument("--preference-path", type=Path, default=DEFAULT_PREFERENCE_PATH)
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--eval-report", type=Path, default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--preference-config", type=Path, default=DEFAULT_PREFERENCE_CONFIG)
    parser.add_argument("--source-paired-fit", type=Path, default=DEFAULT_SOURCE_PAIRED_FIT)
    parser.add_argument(
        "--use-paired-fit-gallery",
        action="store_true",
        help="Load source paired-fit trajectories for the representative gallery. This can be slow on the full dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-stem", type=str, default=DEFAULT_CHECKPOINT_STEM)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-windows-per-split", type=int, default=None)
    parser.add_argument("--gallery-count", type=int, default=DEFAULT_GALLERY_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = evaluate_first_windows(
        preference_path=args.preference_path,
        reward_dir=args.reward_dir,
        eval_report_path=args.eval_report,
        preference_config_path=args.preference_config,
        source_paired_fit=args.source_paired_fit if args.use_paired_fit_gallery else None,
        output_dir=args.output_dir,
        checkpoint_stem=args.checkpoint_stem,
        device=args.device,
        max_windows_per_split=args.max_windows_per_split,
        gallery_count=args.gallery_count,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
