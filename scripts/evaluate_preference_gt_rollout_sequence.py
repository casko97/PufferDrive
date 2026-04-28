#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
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

from scripts.evaluate_preference_first32_deployment import (
    DEFAULT_CHECKPOINT_STEM,
    _index_rollouts,
    _load_rollout_payload,
    _one_hot_actions,
)
from scripts.evaluate_preference_train_val_dataset import (
    DEFAULT_REWARD_DIR,
    RewardBundle,
    _jsonable,
    _load_reward_bundle,
    _score_pair_batch,
    _softmax_preferred_probability,
    _to_numpy,
)


DEFAULT_GT_TRUCK = Path(
    "outputs/preference_eval/preference_ground_truth_context_compare_full/rollouts/"
    "ground-truth-truck-context-preferred-branch_rollouts.pt"
)
DEFAULT_GT_CAR = Path(
    "outputs/preference_eval/preference_ground_truth_context_compare_full/rollouts/"
    "ground-truth-car-context-rejected-branch_rollouts.pt"
)
DEFAULT_REFERENCE_SCORED = Path(
    "outputs/preference_eval/preference_first32_deployment_eval_truck_context_policy/scored_first32_deployment.pt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/preference_eval/preference_gt_rollout_sequence_eval")
DEFAULT_WINDOW_LEN = 32
SCORED_FORMAT_VERSION = "preference_gt_rollout_sequence_eval_v1"
DEFAULT_REFERENCE_COMPARISON = "gt_truck_context_vs_car_fit"
DEFAULT_REFERENCE_STATUS = "ok"


def _plot_note(fig, note: str, *, width: int = 145) -> None:
    fig.text(
        0.01,
        0.012,
        "\n".join(textwrap.wrap(note, width=width)),
        ha="left",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.ndarray, list, tuple, dict)):
        return json.dumps(_jsonable(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return output_path


def _load_reference_map_names(
    reference_scored: Path | None,
    *,
    comparison: str | None,
    status: str | None,
) -> list[str] | None:
    if reference_scored is None:
        return None
    if not reference_scored.exists():
        raise FileNotFoundError(f"reference scored deployment artifact not found: {reference_scored}")
    payload = torch.load(reference_scored, map_location="cpu", weights_only=False)
    rows = list(payload.get("rows", []))
    if comparison is not None:
        rows = [row for row in rows if row.get("comparison") == comparison]
    if status is not None:
        rows = [row for row in rows if row.get("status") == status]
    map_names = sorted({str(row["map_name"]) for row in rows if row.get("map_name")})
    if not map_names:
        raise ValueError(
            f"reference scored artifact selected no maps: path={reference_scored}, "
            f"comparison={comparison}, status={status}"
        )
    return map_names


def _window_features(
    rollout: dict[str, Any],
    bundle: RewardBundle,
    *,
    start: int,
    end: int,
) -> np.ndarray:
    observations = np.asarray(rollout["observations"][start:end], dtype=np.float32)
    actions = np.asarray(rollout["actions"][start:end], dtype=np.int64)
    expected_len = end - start
    if observations.shape != (expected_len, bundle.obs_dim):
        raise ValueError(
            f"observation shape mismatch for {rollout.get('map_name')}: "
            f"expected {(expected_len, bundle.obs_dim)}, got {observations.shape}"
        )
    action_features = _one_hot_actions(actions, bundle.action_dim)
    return np.concatenate([observations, action_features], axis=-1).astype(np.float32)


def _score_fixed_window(
    *,
    preferred_rollout: dict[str, Any],
    rejected_rollout: dict[str, Any],
    bundle: RewardBundle,
    start: int,
    end: int,
) -> dict[str, Any]:
    window_len = end - start
    preferred = _window_features(preferred_rollout, bundle, start=start, end=end)
    rejected = _window_features(rejected_rollout, bundle, start=start, end=end)
    scored = _score_pair_batch(
        bundle,
        preferred.reshape(1, window_len, -1),
        rejected.reshape(1, window_len, -1),
        np.zeros((1, 1), dtype=np.float32),
    )
    return {
        "label": 0,
        "predicted_label": int(scored["predicted_label"][0]),
        "correct": bool(scored["correct"][0]),
        "prob_preferred": float(scored["prob_preferred"][0]),
        "confidence": float(scored["confidence"][0]),
        "ce_loss": float(scored["ce_loss"][0]),
        "preferred_score": float(scored["preferred_score"][0]),
        "rejected_score": float(scored["rejected_score"][0]),
        "margin": float(scored["margin"][0]),
        "signed_margin": float(scored["signed_margin"][0]),
        "preferred_std": float(scored["preferred_std"][0]),
        "rejected_std": float(scored["rejected_std"][0]),
        "pair_uncertainty": float(scored["pair_uncertainty"][0]),
        "preferred_member_scores": scored["preferred_member_scores"][0].astype(np.float32),
        "rejected_member_scores": scored["rejected_member_scores"][0].astype(np.float32),
    }


def _score_variable_sequence(
    *,
    preferred_features: np.ndarray,
    rejected_features: np.ndarray,
    bundle: RewardBundle,
) -> dict[str, Any]:
    if preferred_features.ndim != 2 or rejected_features.ndim != 2:
        raise ValueError(f"expected rank-2 feature arrays, got {preferred_features.shape} and {rejected_features.shape}")
    if preferred_features.shape != rejected_features.shape:
        raise ValueError(f"preferred/rejected feature mismatch: {preferred_features.shape} vs {rejected_features.shape}")
    expected_dim = bundle.obs_dim + bundle.action_dim
    if preferred_features.shape[1] != expected_dim:
        raise ValueError(f"feature dim mismatch: expected {expected_dim}, got {preferred_features.shape[1]}")

    preferred_batch = preferred_features.reshape(1, preferred_features.shape[0], expected_dim)
    rejected_batch = rejected_features.reshape(1, rejected_features.shape[0], expected_dim)
    preferred_rewards_by_member: list[np.ndarray] = []
    rejected_rewards_by_member: list[np.ndarray] = []
    for member in range(bundle.ensemble_size):
        preferred_rewards = _to_numpy(bundle.reward_model.r_hat_member(preferred_batch, member=member)).reshape(-1)
        rejected_rewards = _to_numpy(bundle.reward_model.r_hat_member(rejected_batch, member=member)).reshape(-1)
        preferred_rewards_by_member.append(preferred_rewards.astype(np.float32))
        rejected_rewards_by_member.append(rejected_rewards.astype(np.float32))

    preferred_step_matrix = np.stack(preferred_rewards_by_member, axis=1)
    rejected_step_matrix = np.stack(rejected_rewards_by_member, axis=1)
    preferred_member_scores = preferred_step_matrix.sum(axis=0).astype(np.float32)
    rejected_member_scores = rejected_step_matrix.sum(axis=0).astype(np.float32)
    preferred_score = float(preferred_member_scores.mean())
    rejected_score = float(rejected_member_scores.mean())
    margin = preferred_score - rejected_score
    member_probs = _softmax_preferred_probability(
        preferred_member_scores.reshape(-1),
        rejected_member_scores.reshape(-1),
    )
    prob_preferred = float(member_probs.mean())
    predicted_label = 0 if prob_preferred >= 0.5 else 1
    probability_of_label = prob_preferred
    preferred_step_mean = preferred_step_matrix.mean(axis=1).astype(np.float32)
    rejected_step_mean = rejected_step_matrix.mean(axis=1).astype(np.float32)
    preferred_cumulative = np.cumsum(preferred_step_mean).astype(np.float32)
    rejected_cumulative = np.cumsum(rejected_step_mean).astype(np.float32)
    margin_cumulative = (preferred_cumulative - rejected_cumulative).astype(np.float32)
    return {
        "label": 0,
        "predicted_label": int(predicted_label),
        "correct": bool(predicted_label == 0),
        "prob_preferred": prob_preferred,
        "confidence": float(abs(prob_preferred - 0.5) * 2.0),
        "ce_loss": float(-np.log(np.clip(probability_of_label, 1e-8, 1.0))),
        "preferred_score": preferred_score,
        "rejected_score": rejected_score,
        "margin": float(margin),
        "signed_margin": float(margin),
        "preferred_std": float(preferred_member_scores.std()),
        "rejected_std": float(rejected_member_scores.std()),
        "pair_uncertainty": float(0.5 * (preferred_member_scores.std() + rejected_member_scores.std())),
        "preferred_member_scores": preferred_member_scores,
        "rejected_member_scores": rejected_member_scores,
        "preferred_step_reward": preferred_step_mean,
        "rejected_step_reward": rejected_step_mean,
        "preferred_cumulative_score": preferred_cumulative,
        "rejected_cumulative_score": rejected_cumulative,
        "margin_cumulative": margin_cumulative,
    }


def _base_row(
    *,
    preferred_rollout: dict[str, Any],
    rejected_rollout: dict[str, Any],
    aligned_steps: int,
    window_len: int,
) -> dict[str, Any]:
    return {
        "map_name": preferred_rollout["map_name"],
        "source_map_name": preferred_rollout.get("source_map_name", ""),
        "scenario_type": preferred_rollout.get("scenario_type", ""),
        "delta_heading_deg": preferred_rollout.get("delta_heading_deg"),
        "preferred_steps": int(preferred_rollout["steps"]),
        "rejected_steps": int(rejected_rollout["steps"]),
        "aligned_steps": int(aligned_steps),
        "window_len": int(window_len),
        "preferred_model": preferred_rollout.get("paired_fit_branch", "truck/context preferred"),
        "rejected_model": rejected_rollout.get("paired_fit_branch", "car rejected"),
    }


def score_gt_rollout_sequence(
    *,
    gt_truck_path: Path,
    gt_car_path: Path,
    bundle: RewardBundle,
    window_len: int,
    map_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    truck_payload = _load_rollout_payload(gt_truck_path)
    car_payload = _load_rollout_payload(gt_car_path)
    truck_rollouts = _index_rollouts(truck_payload)
    car_rollouts = _index_rollouts(car_payload)
    window_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    common_map_names = sorted(set(truck_rollouts) & set(car_rollouts))
    if map_names is None:
        selected_map_names = common_map_names
    else:
        missing = sorted(set(map_names) - set(common_map_names))
        if missing:
            raise ValueError(f"reference maps missing from GT truck/car rollout intersection: {missing}")
        selected_map_names = list(map_names)

    for map_name in selected_map_names:
        preferred_rollout = truck_rollouts[map_name]
        rejected_rollout = car_rollouts[map_name]
        aligned_steps = min(int(preferred_rollout["steps"]), int(rejected_rollout["steps"]))
        base = _base_row(
            preferred_rollout=preferred_rollout,
            rejected_rollout=rejected_rollout,
            aligned_steps=aligned_steps,
            window_len=window_len,
        )
        full_window_count = aligned_steps // window_len
        for window_index in range(full_window_count):
            start = window_index * window_len
            end = start + window_len
            row = {
                **base,
                "experiment": "nonoverlap_32_step_windows",
                "window_index": int(window_index),
                "timestep_start": int(start),
                "timestep_end": int(end),
                "scored_steps": int(window_len),
                "leftover_steps_after_last_full_window": int(aligned_steps - full_window_count * window_len),
                "status": "ok",
                "skip_reason": "",
            }
            try:
                row.update(
                    _score_fixed_window(
                        preferred_rollout=preferred_rollout,
                        rejected_rollout=rejected_rollout,
                        bundle=bundle,
                        start=start,
                        end=end,
                    )
                )
            except Exception as exc:
                row.update({"status": "error", "skip_reason": str(exc)})
            window_rows.append(row)

        full_row = {
            **base,
            "experiment": "full_aligned_trajectory",
            "window_index": -1,
            "timestep_start": 0,
            "timestep_end": int(aligned_steps),
            "scored_steps": int(aligned_steps),
            "full_32_step_windows": int(full_window_count),
            "leftover_steps_after_last_full_window": int(aligned_steps - full_window_count * window_len),
            "status": "ok" if aligned_steps > 0 else "too_short",
            "skip_reason": "" if aligned_steps > 0 else "aligned_steps<=0",
        }
        if aligned_steps > 0:
            try:
                preferred_features = _window_features(preferred_rollout, bundle, start=0, end=aligned_steps)
                rejected_features = _window_features(rejected_rollout, bundle, start=0, end=aligned_steps)
                scored = _score_variable_sequence(
                    preferred_features=preferred_features,
                    rejected_features=rejected_features,
                    bundle=bundle,
                )
                full_row.update(scored)
                for step_idx, (preferred_cum, rejected_cum, margin_cum) in enumerate(
                    zip(
                        scored["preferred_cumulative_score"],
                        scored["rejected_cumulative_score"],
                        scored["margin_cumulative"],
                    )
                ):
                    curve_rows.append(
                        {
                            "map_name": map_name,
                            "timestep": int(step_idx + 1),
                            "preferred_cumulative_score": float(preferred_cum),
                            "rejected_cumulative_score": float(rejected_cum),
                            "margin_cumulative": float(margin_cum),
                        }
                    )
            except Exception as exc:
                full_row.update({"status": "error", "skip_reason": str(exc)})
        full_rows.append(full_row)

    return window_rows, full_rows, curve_rows


def _ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "ok"]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else 0.0


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = _ok_rows(rows)
    return {
        "total_count": len(rows),
        "ok_count": len(ok),
        "error_count": sum(1 for row in rows if row.get("status") == "error"),
        "too_short_count": sum(1 for row in rows if row.get("status") == "too_short"),
        "win_rate": float(np.mean([bool(row["correct"]) for row in ok])) if ok else 0.0,
        "mean_margin": _mean(ok, "margin"),
        "mean_prob_preferred": _mean(ok, "prob_preferred"),
        "mean_confidence": _mean(ok, "confidence"),
        "mean_pair_uncertainty": _mean(ok, "pair_uncertainty"),
    }


def _summary_by_window_index(window_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_index: dict[str, Any] = {}
    for window_index in sorted({int(row["window_index"]) for row in _ok_rows(window_rows)}):
        rows = [row for row in _ok_rows(window_rows) if int(row["window_index"]) == window_index]
        by_index[str(window_index)] = {
            "window_number": int(window_index + 1),
            **_summary_for_rows(rows),
        }
    return by_index


def build_summary(
    *,
    window_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    gt_truck_path: Path,
    gt_car_path: Path,
    reward_dir: Path,
    window_len: int,
    reference_scored: Path | None,
    reference_comparison: str | None,
    reference_status: str | None,
    reference_map_names: list[str] | None,
) -> dict[str, Any]:
    return {
        "format": SCORED_FORMAT_VERSION,
        "orientation": "truck/context is preferred, car is rejected; positive margin means truck/context scores higher",
        "window_len": int(window_len),
        "inputs": {
            "gt_truck": str(gt_truck_path),
            "gt_car": str(gt_car_path),
            "reward_dir": str(reward_dir),
        },
        "reference_maps": {
            "source": str(reference_scored) if reference_scored is not None else None,
            "comparison": reference_comparison,
            "status": reference_status,
            "map_count": len(reference_map_names) if reference_map_names is not None else None,
            "map_names": reference_map_names,
        },
        "nonoverlap_windows": {
            **_summary_for_rows(window_rows),
            "by_window_index": _summary_by_window_index(window_rows),
        },
        "full_trajectory": _summary_for_rows(full_rows),
    }


def plot_window_margin_by_index(window_rows: list[dict[str, Any]], output_path: Path) -> Path:
    rows = _ok_rows(window_rows)
    if not rows:
        raise ValueError("no ok window rows available for margin-by-index plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    for map_name in sorted({row["map_name"] for row in rows}):
        map_rows = sorted([row for row in rows if row["map_name"] == map_name], key=lambda row: int(row["window_index"]))
        x = [int(row["window_index"]) + 1 for row in map_rows]
        y = [float(row["margin"]) for row in map_rows]
        ax.plot(x, y, marker="o", linewidth=1.2, alpha=0.65, label=map_name)
    by_idx = sorted({int(row["window_index"]) for row in rows})
    mean_y = [_mean([row for row in rows if int(row["window_index"]) == idx], "margin") for idx in by_idx]
    ax.plot([idx + 1 for idx in by_idx], mean_y, color="black", linewidth=3.0, marker="o", label="mean")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.set_xlabel("32-step window number")
    ax.set_ylabel("Truck/context score - car score")
    ax.set_title("GT Rollouts Scored in Consecutive 32-Step Windows")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, ncols=2)
    _plot_note(
        fig,
        "How to read: each point is one non-overlapping 32-step segment from the corrected GT rollout. "
        "Window 1 is timesteps 0-31, window 2 is 32-63, etc. Positive margin favors truck/context; negative favors car.",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_window_probability_by_index(window_rows: list[dict[str, Any]], output_path: Path) -> Path:
    rows = _ok_rows(window_rows)
    if not rows:
        raise ValueError("no ok window rows available for probability plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    for map_name in sorted({row["map_name"] for row in rows}):
        map_rows = sorted([row for row in rows if row["map_name"] == map_name], key=lambda row: int(row["window_index"]))
        ax.plot(
            [int(row["window_index"]) + 1 for row in map_rows],
            [float(row["prob_preferred"]) for row in map_rows],
            marker="o",
            linewidth=1.2,
            alpha=0.65,
            label=map_name,
        )
    by_idx = sorted({int(row["window_index"]) for row in rows})
    mean_y = [_mean([row for row in rows if int(row["window_index"]) == idx], "prob_preferred") for idx in by_idx]
    ax.plot([idx + 1 for idx in by_idx], mean_y, color="black", linewidth=3.0, marker="o", label="mean")
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.4, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("32-step window number")
    ax.set_ylabel("P(truck/context preferred over car)")
    ax.set_title("GT 32-Step Window Preference Probability")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, ncols=2)
    _plot_note(
        fig,
        "How to read: probabilities above 0.5 mean the reward model predicts truck/context beats car for that separate window.",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_window_margin_heatmap(window_rows: list[dict[str, Any]], output_path: Path) -> Path:
    rows = _ok_rows(window_rows)
    if not rows:
        raise ValueError("no ok window rows available for heatmap")
    map_names = sorted({row["map_name"] for row in rows})
    window_indices = sorted({int(row["window_index"]) for row in rows})
    values = np.full((len(map_names), len(window_indices)), np.nan, dtype=np.float32)
    index_lookup = {idx: col for col, idx in enumerate(window_indices)}
    for row in rows:
        values[map_names.index(row["map_name"]), index_lookup[int(row["window_index"])]] = float(row["margin"])
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    limit = max(limit, 1e-6)
    fig, ax = plt.subplots(figsize=(max(10, len(window_indices) * 0.8), max(5, len(map_names) * 0.35)))
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_yticks(np.arange(len(map_names)))
    ax.set_yticklabels(map_names, fontsize=8)
    ax.set_xticks(np.arange(len(window_indices)))
    ax.set_xticklabels([str(idx + 1) for idx in window_indices])
    ax.set_xlabel("32-step window number")
    ax.set_title("GT Window Margins by Map")
    fig.colorbar(image, ax=ax, label="truck/context - car margin")
    _plot_note(fig, "How to read: blue cells favor truck/context; red cells favor car. Blank cells mean that map did not have another full 32-step window.")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_window_summary(window_rows: list[dict[str, Any]], output_path: Path) -> Path:
    rows = _ok_rows(window_rows)
    if not rows:
        raise ValueError("no ok window rows available for summary plot")
    window_indices = sorted({int(row["window_index"]) for row in rows})
    labels = [str(idx + 1) for idx in window_indices]
    mean_margin = [_mean([row for row in rows if int(row["window_index"]) == idx], "margin") for idx in window_indices]
    win_rate = [
        float(np.mean([bool(row["correct"]) for row in rows if int(row["window_index"]) == idx]))
        for idx in window_indices
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))
    axes[0].bar(x, mean_margin, color="tab:blue", alpha=0.85)
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.45)
    axes[0].set_ylabel("Mean margin")
    axes[0].set_title("Mean margin by window")
    axes[1].bar(x, win_rate, color="tab:orange", alpha=0.85)
    axes[1].axhline(0.5, color="black", linewidth=1, alpha=0.45, linestyle="--")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Truck/context win fraction")
    axes[1].set_title("Truck/context win fraction by window")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("32-step window number")
        ax.grid(True, axis="y", alpha=0.2)
    _plot_note(
        fig,
        "How to read: the left panel averages the truck/context-car score margin across maps for each window. "
        "The right panel is the fraction of maps where the window margin is positive.",
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_full_scores_by_map(full_rows: list[dict[str, Any]], output_path: Path) -> Path:
    rows = sorted(_ok_rows(full_rows), key=lambda row: str(row["map_name"]))
    if not rows:
        raise ValueError("no ok full rows available for score plot")
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.75), 6))
    x = np.arange(len(rows))
    width = 0.36
    ax.bar(x - width / 2, [float(row["preferred_score"]) for row in rows], width=width, label="truck/context (preferred)")
    ax.bar(x + width / 2, [float(row["rejected_score"]) for row in rows], width=width, label="car (rejected)")
    ax.set_xticks(x)
    ax.set_xticklabels([row["map_name"] for row in rows], rotation=45, ha="right")
    ax.set_ylabel("Full aligned trajectory reward sum")
    ax.set_title("Full GT Trajectory Scores by Map")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    _plot_note(
        fig,
        "How to read: each bar is the reward-model sum over all aligned timesteps in the GT rollout. "
        "Taller truck/context bars mean the full trajectory is scored above the car trajectory.",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_full_margin_by_map(full_rows: list[dict[str, Any]], output_path: Path) -> Path:
    rows = sorted(_ok_rows(full_rows), key=lambda row: float(row["margin"]))
    if not rows:
        raise ValueError("no ok full rows available for margin plot")
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.75), 6))
    colors = ["tab:blue" if float(row["margin"]) >= 0.0 else "tab:red" for row in rows]
    ax.bar(np.arange(len(rows)), [float(row["margin"]) for row in rows], color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.45)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([row["map_name"] for row in rows], rotation=45, ha="right")
    ax.set_ylabel("Full trajectory truck/context - car margin")
    ax.set_title("Full GT Trajectory Margin by Map")
    ax.grid(True, axis="y", alpha=0.2)
    _plot_note(fig, "How to read: bars above zero favor truck/context over the full aligned rollout; bars below zero favor car.")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_full_cumulative_margin(curve_rows: list[dict[str, Any]], output_path: Path) -> Path:
    if not curve_rows:
        raise ValueError("no cumulative curve rows available")
    fig, ax = plt.subplots(figsize=(12, 6))
    for map_name in sorted({row["map_name"] for row in curve_rows}):
        rows = sorted([row for row in curve_rows if row["map_name"] == map_name], key=lambda row: int(row["timestep"]))
        ax.plot(
            [int(row["timestep"]) for row in rows],
            [float(row["margin_cumulative"]) for row in rows],
            linewidth=1.4,
            alpha=0.72,
            label=map_name,
        )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.45)
    ax.set_xlabel("Aligned rollout timestep")
    ax.set_ylabel("Cumulative truck/context - car margin")
    ax.set_title("Full GT Trajectory Cumulative Preference Margin")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, ncols=2)
    _plot_note(
        fig,
        "How to read: the curve sums per-timestep reward-model differences over the full GT rollout. "
        "Upward movement means truck/context is gaining reward relative to car; crossing below zero means the cumulative score favors car.",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_full_probability_by_map(full_rows: list[dict[str, Any]], output_path: Path) -> Path:
    rows = sorted(_ok_rows(full_rows), key=lambda row: float(row["prob_preferred"]))
    if not rows:
        raise ValueError("no ok full rows available for probability plot")
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.75), 5))
    ax.bar(np.arange(len(rows)), [float(row["prob_preferred"]) for row in rows], color="tab:purple", alpha=0.85)
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.45, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([row["map_name"] for row in rows], rotation=45, ha="right")
    ax.set_ylabel("P(truck/context preferred over car)")
    ax.set_title("Full GT Trajectory Preference Probability by Map")
    ax.grid(True, axis="y", alpha=0.2)
    _plot_note(fig, "How to read: values above 0.5 mean the reward model predicts the full truck/context trajectory beats the full car trajectory.")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_markdown(summary: dict[str, Any], output_path: Path) -> Path:
    window_summary = summary["nonoverlap_windows"]
    full_summary = summary["full_trajectory"]
    lines = [
        "# GT Rollout Sequence Preference Eval",
        "",
        f"- Window length: `{summary['window_len']}`",
        f"- Orientation: {summary['orientation']}",
        "",
        "## Non-Overlapping 32-Step Windows",
        f"- ok windows: `{window_summary['ok_count']}`",
        f"- truck/context win rate: `{window_summary['win_rate']:.6f}`",
        f"- mean margin: `{window_summary['mean_margin']:.6f}`",
        f"- mean p(truck/context): `{window_summary['mean_prob_preferred']:.6f}`",
        "",
        "## Full Aligned Trajectories",
        f"- ok trajectories: `{full_summary['ok_count']}`",
        f"- truck/context win rate: `{full_summary['win_rate']:.6f}`",
        f"- mean margin: `{full_summary['mean_margin']:.6f}`",
        f"- mean p(truck/context): `{full_summary['mean_prob_preferred']:.6f}`",
        "",
        "## Interpretation",
        "- The window experiment scores each consecutive 32-step chunk independently.",
        "- The full-trajectory experiment sums per-timestep reward-model outputs over every aligned GT rollout step.",
        "- Positive margin means truck/context is scored above car; probability above 0.5 means the model predicts truck/context wins.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_scored_payload(
    *,
    window_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": SCORED_FORMAT_VERSION,
            "summary": _jsonable(summary),
            "window_rows": [{key: _jsonable(value) for key, value in row.items()} for row in window_rows],
            "full_rows": [{key: _jsonable(value) for key, value in row.items()} for row in full_rows],
            "curve_rows": [{key: _jsonable(value) for key, value in row.items()} for row in curve_rows],
        },
        output_path,
    )
    return output_path


def evaluate_gt_rollout_sequence(
    *,
    gt_truck: Path,
    gt_car: Path,
    reward_dir: Path,
    output_dir: Path,
    checkpoint_stem: str,
    device: str,
    window_len: int,
    reference_scored: Path | None = DEFAULT_REFERENCE_SCORED,
    reference_comparison: str | None = DEFAULT_REFERENCE_COMPARISON,
    reference_status: str | None = DEFAULT_REFERENCE_STATUS,
    bundle: RewardBundle | None = None,
) -> dict[str, str]:
    effective_bundle = bundle or _load_reward_bundle(reward_dir, checkpoint_stem=checkpoint_stem, device=device)
    reference_map_names = _load_reference_map_names(
        reference_scored,
        comparison=reference_comparison,
        status=reference_status,
    )
    window_rows, full_rows, curve_rows = score_gt_rollout_sequence(
        gt_truck_path=gt_truck,
        gt_car_path=gt_car,
        bundle=effective_bundle,
        window_len=window_len,
        map_names=reference_map_names,
    )
    summary = build_summary(
        window_rows=window_rows,
        full_rows=full_rows,
        gt_truck_path=gt_truck,
        gt_car_path=gt_car,
        reward_dir=reward_dir,
        window_len=window_len,
        reference_scored=reference_scored,
        reference_comparison=reference_comparison,
        reference_status=reference_status,
        reference_map_names=reference_map_names,
    )
    report_dir = output_dir / "report"
    outputs = {
        "scored": str(
            write_scored_payload(
                window_rows=window_rows,
                full_rows=full_rows,
                curve_rows=curve_rows,
                summary=summary,
                output_path=output_dir / "scored_gt_rollout_sequence_eval.pt",
            )
        ),
        "window_csv": str(_write_csv(window_rows, report_dir / "gt_rollout_32step_windows.csv")),
        "full_csv": str(_write_csv(full_rows, report_dir / "gt_rollout_full_trajectories.csv")),
        "curves_csv": str(_write_csv(curve_rows, report_dir / "gt_rollout_cumulative_curves.csv")),
        "summary_json": str(report_dir / "gt_rollout_sequence_summary.json"),
        "summary_md": str(write_markdown(summary, report_dir / "gt_rollout_sequence_summary.md")),
        "window_margin_by_index": str(plot_window_margin_by_index(window_rows, report_dir / "window_margin_by_index.png")),
        "window_probability_by_index": str(
            plot_window_probability_by_index(window_rows, report_dir / "window_probability_by_index.png")
        ),
        "window_margin_heatmap": str(plot_window_margin_heatmap(window_rows, report_dir / "window_margin_heatmap.png")),
        "window_summary": str(plot_window_summary(window_rows, report_dir / "window_summary_by_index.png")),
        "full_scores_by_map": str(plot_full_scores_by_map(full_rows, report_dir / "full_trajectory_scores_by_map.png")),
        "full_margin_by_map": str(plot_full_margin_by_map(full_rows, report_dir / "full_trajectory_margin_by_map.png")),
        "full_probability_by_map": str(
            plot_full_probability_by_map(full_rows, report_dir / "full_trajectory_probability_by_map.png")
        ),
        "full_cumulative_margin": str(
            plot_full_cumulative_margin(curve_rows, report_dir / "full_trajectory_cumulative_margin.png")
        ),
    }
    Path(outputs["summary_json"]).write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score corrected GT truck/context vs car rollouts over consecutive windows and full trajectories."
    )
    parser.add_argument("--gt-truck", type=Path, default=DEFAULT_GT_TRUCK)
    parser.add_argument("--gt-car", type=Path, default=DEFAULT_GT_CAR)
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-stem", type=str, default=DEFAULT_CHECKPOINT_STEM)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--window-len", type=int, default=DEFAULT_WINDOW_LEN)
    parser.add_argument(
        "--reference-scored",
        type=Path,
        default=DEFAULT_REFERENCE_SCORED,
        help="Use map names from this deployment scored artifact. Pass 'none' to disable filtering.",
    )
    parser.add_argument("--reference-comparison", type=str, default=DEFAULT_REFERENCE_COMPARISON)
    parser.add_argument("--reference-status", type=str, default=DEFAULT_REFERENCE_STATUS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = evaluate_gt_rollout_sequence(
        gt_truck=args.gt_truck,
        gt_car=args.gt_car,
        reward_dir=args.reward_dir,
        output_dir=args.output_dir,
        checkpoint_stem=args.checkpoint_stem,
        device=args.device,
        window_len=args.window_len,
        reference_scored=None if str(args.reference_scored).lower() == "none" else args.reference_scored,
        reference_comparison=None if str(args.reference_comparison).lower() == "none" else args.reference_comparison,
        reference_status=None if str(args.reference_status).lower() == "none" else args.reference_status,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
