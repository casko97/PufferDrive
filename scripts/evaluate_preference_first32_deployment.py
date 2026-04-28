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

from scripts.evaluate_preference_train_val_dataset import (
    DEFAULT_REWARD_DIR,
    RewardBundle,
    _jsonable,
    _load_reward_bundle,
    _score_pair_batch,
)


DEFAULT_POLICY_CAR = Path(
    "outputs/preference_eval/preference_rollout_model_compare_full_no_resets/rollouts/"
    "car-baseline-full-boston_rollouts.pt"
)
DEFAULT_POLICY_TRUCK = Path(
    "outputs/preference_eval/preference_rollout_model_compare_full_no_resets/rollouts/"
    "truck-baseline-full-boston_rollouts.pt"
)
DEFAULT_GT_CAR = Path(
    "outputs/preference_eval/preference_ground_truth_context_compare_full/rollouts/"
    "ground-truth-car-fit_rollouts.pt"
)
DEFAULT_GT_TRUCK = Path(
    "outputs/preference_eval/preference_ground_truth_context_compare_full/rollouts/"
    "ground-truth-truck-context-replay_rollouts.pt"
)
DEFAULT_TRAIN_VAL_SUMMARY = Path("outputs/preference_eval/preference_train_val_dataset_eval/report/report_summary.json")
DEFAULT_OUTPUT_DIR = Path("outputs/preference_eval/preference_first32_deployment_eval")
DEFAULT_WINDOW_LEN = 32
DEFAULT_CHECKPOINT_STEM = "offline_truck_context"

SCORED_FORMAT_VERSION = "preference_first32_deployment_v1"


def _load_rollout_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"rollout artifact not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _index_rollouts(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(rollout["map_name"]): rollout for rollout in payload.get("rollouts", [])}


def _one_hot_actions(actions: np.ndarray, action_dim: int) -> np.ndarray:
    discrete = np.asarray(actions, dtype=np.int64).reshape(-1)
    if np.any(discrete < 0) or np.any(discrete >= action_dim):
        raise ValueError(f"action out of range [0, {action_dim - 1}]: {discrete.tolist()}")
    encoded = np.zeros((len(discrete), action_dim), dtype=np.float32)
    encoded[np.arange(len(discrete)), discrete] = 1.0
    return encoded


def _first_window_features(rollout: dict[str, Any], bundle: RewardBundle, window_len: int) -> np.ndarray:
    steps = int(rollout["steps"])
    if steps < window_len:
        raise ValueError(f"rollout too short: steps={steps}, window_len={window_len}")
    observations = np.asarray(rollout["observations"][:window_len], dtype=np.float32)
    actions = np.asarray(rollout["actions"][:window_len], dtype=np.int64)
    if observations.shape != (window_len, bundle.obs_dim):
        raise ValueError(f"observation shape mismatch: expected {(window_len, bundle.obs_dim)}, got {observations.shape}")
    action_features = _one_hot_actions(actions, bundle.action_dim)
    return np.concatenate([observations, action_features], axis=-1).astype(np.float32)


def _score_first32_pair(
    *,
    preferred_rollout: dict[str, Any],
    rejected_rollout: dict[str, Any],
    bundle: RewardBundle,
    window_len: int,
) -> dict[str, Any]:
    preferred_features = _first_window_features(preferred_rollout, bundle, window_len)
    rejected_features = _first_window_features(rejected_rollout, bundle, window_len)
    scored = _score_pair_batch(
        bundle,
        preferred_features.reshape(1, window_len, -1),
        rejected_features.reshape(1, window_len, -1),
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


def _base_row(
    *,
    comparison: str,
    preferred_model: str,
    rejected_model: str,
    preferred_rollout: dict[str, Any],
    rejected_rollout: dict[str, Any],
    window_len: int,
) -> dict[str, Any]:
    aligned_steps = min(int(preferred_rollout["steps"]), int(rejected_rollout["steps"]))
    return {
        "comparison": comparison,
        "preferred_model": preferred_model,
        "rejected_model": rejected_model,
        "map_name": preferred_rollout["map_name"],
        "scenario_type": preferred_rollout.get("scenario_type", ""),
        "delta_heading_deg": preferred_rollout.get("delta_heading_deg"),
        "preferred_steps": int(preferred_rollout["steps"]),
        "rejected_steps": int(rejected_rollout["steps"]),
        "aligned_steps": int(aligned_steps),
        "window_len": int(window_len),
        "timestep_start": 0,
        "timestep_end": int(window_len),
    }


def score_deployment_rollouts(
    *,
    policy_car_path: Path,
    policy_truck_path: Path,
    gt_car_path: Path,
    gt_truck_path: Path,
    bundle: RewardBundle,
    window_len: int,
) -> list[dict[str, Any]]:
    payloads = {
        "policy_car": _load_rollout_payload(policy_car_path),
        "policy_truck": _load_rollout_payload(policy_truck_path),
        "gt_car": _load_rollout_payload(gt_car_path),
        "gt_truck": _load_rollout_payload(gt_truck_path),
    }
    policy_car = _index_rollouts(payloads["policy_car"])
    policy_truck = _index_rollouts(payloads["policy_truck"])
    gt_car = _index_rollouts(payloads["gt_car"])
    gt_truck = _index_rollouts(payloads["gt_truck"])

    rows: list[dict[str, Any]] = []
    comparison_specs = [
        (
            "policy_truck_vs_car",
            payloads["policy_truck"].get("model_name", "truck-policy"),
            payloads["policy_car"].get("model_name", "car-policy"),
            policy_truck,
            policy_car,
        ),
        (
            "gt_truck_context_vs_car_fit",
            payloads["gt_truck"].get("model_name", "ground-truth-truck-context"),
            payloads["gt_car"].get("model_name", "ground-truth-car-fit"),
            gt_truck,
            gt_car,
        ),
    ]

    for comparison, preferred_model, rejected_model, preferred_by_map, rejected_by_map in comparison_specs:
        for map_name in sorted(set(preferred_by_map) & set(rejected_by_map)):
            preferred_rollout = preferred_by_map[map_name]
            rejected_rollout = rejected_by_map[map_name]
            row = _base_row(
                comparison=comparison,
                preferred_model=preferred_model,
                rejected_model=rejected_model,
                preferred_rollout=preferred_rollout,
                rejected_rollout=rejected_rollout,
                window_len=window_len,
            )
            if int(row["aligned_steps"]) < window_len:
                row.update({"status": "too_short", "skip_reason": f"aligned_steps<{window_len}"})
                rows.append(row)
                continue
            try:
                row.update(
                    _score_first32_pair(
                        preferred_rollout=preferred_rollout,
                        rejected_rollout=rejected_rollout,
                        bundle=bundle,
                        window_len=window_len,
                    )
                )
                row.update({"status": "ok", "skip_reason": ""})
            except Exception as exc:
                row.update({"status": "error", "skip_reason": str(exc)})
            rows.append(row)
    return rows


def _ok_rows(rows: list[dict[str, Any]], comparison: str | None = None) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == "ok" and (comparison is None or row.get("comparison") == comparison)
    ]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else 0.0


def _accuracy(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["correct"]) for row in rows])) if rows else 0.0


def build_summary(rows: list[dict[str, Any]], train_val_summary: dict[str, Any] | None) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for comparison in sorted({str(row["comparison"]) for row in rows}):
        subset_all = [row for row in rows if row["comparison"] == comparison]
        subset_ok = _ok_rows(rows, comparison)
        by_scenario = {}
        for scenario_type in sorted({str(row.get("scenario_type", "")) for row in subset_ok}):
            scenario_rows = [row for row in subset_ok if str(row.get("scenario_type", "")) == scenario_type]
            by_scenario[scenario_type] = {
                "ok_count": len(scenario_rows),
                "accuracy": _accuracy(scenario_rows),
                "mean_margin": _mean(scenario_rows, "margin"),
                "mean_prob_preferred": _mean(scenario_rows, "prob_preferred"),
                "mean_pair_uncertainty": _mean(scenario_rows, "pair_uncertainty"),
            }
        comparisons[comparison] = {
            "total_count": len(subset_all),
            "ok_count": len(subset_ok),
            "too_short_count": sum(1 for row in subset_all if row.get("status") == "too_short"),
            "error_count": sum(1 for row in subset_all if row.get("status") == "error"),
            "accuracy": _accuracy(subset_ok),
            "mean_margin": _mean(subset_ok, "margin"),
            "mean_prob_preferred": _mean(subset_ok, "prob_preferred"),
            "mean_confidence": _mean(subset_ok, "confidence"),
            "mean_pair_uncertainty": _mean(subset_ok, "pair_uncertainty"),
            "by_scenario_type": by_scenario,
        }
    return {
        "format": SCORED_FORMAT_VERSION,
        "window_len": DEFAULT_WINDOW_LEN,
        "orientation": "truck/context is preferred, car is rejected, matching the preference training labels",
        "comparisons": comparisons,
        "train_val_reference": train_val_summary.get("selected", {}) if train_val_summary else None,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.ndarray, list, tuple, dict)):
        return json.dumps(_jsonable(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
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


def _plot_note(fig, note: str, *, width: int = 140) -> None:
    fig.text(0.01, 0.012, "\n".join(textwrap.wrap(note, width=width)), ha="left", va="bottom", fontsize=9, color="dimgray")


def plot_deployment_histograms(rows: list[dict[str, Any]], output_path: Path) -> Path:
    ok = _ok_rows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [
        ("margin", "First-32 margin"),
        ("prob_preferred", "P(truck/context preferred over car)"),
        ("pair_uncertainty", "Pair uncertainty"),
    ]
    for ax, (key, title) in zip(axes, specs):
        for comparison in sorted({row["comparison"] for row in ok}):
            values = [float(row[key]) for row in ok if row["comparison"] == comparison]
            if values:
                ax.hist(values, bins=12, alpha=0.55, density=False, label=comparison)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    _plot_note(
        fig,
        "How to read: each row is the first 32 rollout steps on one sampled map. "
        "Truck/context is the preferred side and car is the rejected side; positive margin and probability above 0.5 mean the reward model prefers truck/context over car behavior.",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_margin_by_map(rows: list[dict[str, Any]], output_path: Path) -> Path:
    ok = _ok_rows(rows)
    comparisons = sorted({row["comparison"] for row in ok})
    map_names = sorted({row["map_name"] for row in ok})
    fig, ax = plt.subplots(figsize=(max(12, len(map_names) * 0.9), 6))
    x = np.arange(len(map_names))
    width = 0.36 if len(comparisons) <= 2 else 0.8 / max(len(comparisons), 1)
    for idx, comparison in enumerate(comparisons):
        values = []
        for map_name in map_names:
            match = next((row for row in ok if row["comparison"] == comparison and row["map_name"] == map_name), None)
            values.append(float(match["margin"]) if match is not None else 0.0)
        offset = (idx - (len(comparisons) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=comparison)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(map_names, rotation=45, ha="right")
    ax.set_ylabel("truck/context (preferred) score - car (rejected) score")
    ax.set_title("Deployment first-32 truck/context-over-car margin by map")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    _plot_note(fig, "How to read: bars above zero favor truck/context (preferred); bars below zero favor car (rejected). Missing bars indicate the rollout was shorter than 32 aligned steps.")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_probability_by_map(rows: list[dict[str, Any]], output_path: Path) -> Path:
    ok = _ok_rows(rows)
    comparisons = sorted({row["comparison"] for row in ok})
    map_names = sorted({row["map_name"] for row in ok})
    fig, ax = plt.subplots(figsize=(max(12, len(map_names) * 0.9), 6))
    x = np.arange(len(map_names))
    width = 0.36 if len(comparisons) <= 2 else 0.8 / max(len(comparisons), 1)
    for idx, comparison in enumerate(comparisons):
        values = []
        for map_name in map_names:
            match = next((row for row in ok if row["comparison"] == comparison and row["map_name"] == map_name), None)
            values.append(float(match["prob_preferred"]) if match is not None else 0.0)
        offset = (idx - (len(comparisons) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=comparison)
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.45, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(map_names, rotation=45, ha="right")
    ax.set_ylabel("P(truck/context preferred over car)")
    ax.set_title("Deployment first-32 truck/context-over-car probability by map")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    _plot_note(fig, "How to read: values above 0.5 mean the reward model predicts truck/context (preferred) should beat car (rejected) for that first-32 segment.")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_score_bars(rows: list[dict[str, Any]], output_path: Path) -> Path:
    ok = sorted(_ok_rows(rows), key=lambda row: (row["comparison"], float(row["margin"])))
    fig, ax = plt.subplots(figsize=(max(12, len(ok) * 0.45), 6))
    x = np.arange(len(ok))
    width = 0.38
    ax.bar(x - width / 2, [float(row["preferred_score"]) for row in ok], width=width, label="truck/context (preferred)")
    ax.bar(x + width / 2, [float(row["rejected_score"]) for row in ok], width=width, label="car (rejected)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['map_name']}\n{row['comparison']}" for row in ok], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("32-step reward-model score")
    ax.set_title("Deployment first-32 truck/context (preferred) vs car (rejected) scores")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    _plot_note(fig, "How to read: paired bars show raw 32-step reward sums. Taller truck/context (preferred) bar means the reward model prefers the truck/context side over car (rejected).")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_train_val_vs_deployment(summary: dict[str, Any], output_path: Path) -> Path:
    labels: list[str] = []
    margins: list[float] = []
    accuracies: list[float] = []
    train_val = summary.get("train_val_reference") or {}
    splits = train_val.get("splits", {})
    for split in ("train", "validation"):
        if split in splits:
            labels.append(f"{split}\nfirst-window data")
            margins.append(float(splits[split].get("mean_margin", 0.0)))
            accuracies.append(float(splits[split].get("accuracy", 0.0)))
    for comparison, stats in summary["comparisons"].items():
        labels.append(comparison.replace("_", "\n"))
        margins.append(float(stats.get("mean_margin", 0.0)))
        accuracies.append(float(stats.get("accuracy", 0.0)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))
    axes[0].bar(x, margins, color="tab:blue", alpha=0.85)
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.45)
    axes[0].set_ylabel("Mean margin")
    axes[0].set_title("Mean preference margin")
    axes[0].grid(True, axis="y", alpha=0.2)
    axes[1].bar(x, accuracies, color="tab:orange", alpha=0.85)
    axes[1].axhline(0.5, color="black", linewidth=1, alpha=0.45, linestyle="--")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Fraction truck/context preferred over car")
    axes[1].set_title("Truck/context preferred-side win rate")
    axes[1].grid(True, axis="y", alpha=0.2)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    _plot_note(
        fig,
        "How to read: train/validation bars are first-window dataset metrics; deployment bars use first 32 rollout steps. "
        "The orientation is consistent: truck/context is treated as preferred and car as rejected.",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_markdown(summary: dict[str, Any], output_path: Path) -> Path:
    lines = [
        "# Preference First-32 Deployment Eval",
        "",
        f"- Window length: `{summary['window_len']}`",
        f"- Orientation: {summary['orientation']}",
        "",
        "## Deployment Comparisons",
    ]
    for comparison, stats in summary["comparisons"].items():
        lines.extend(
            [
                f"### {comparison}",
                f"- ok / total: `{stats['ok_count']}` / `{stats['total_count']}`",
                f"- too short: `{stats['too_short_count']}`",
                f"- preferred-side win rate: `{stats['accuracy']:.6f}`",
                f"- mean margin: `{stats['mean_margin']:.6f}`",
                f"- mean p(preferred): `{stats['mean_prob_preferred']:.6f}`",
                f"- mean uncertainty: `{stats['mean_pair_uncertainty']:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "- Positive margin means the reward model scores truck/context above car for the first 32 rollout steps.",
            "- This is a deployment-style check: policy and GT rollouts are generated trajectories, not cached preference windows.",
            "- Short policy rollouts are skipped for segment scoring because a full 32-step window is required.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_scored_payload(rows: list[dict[str, Any]], summary: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": SCORED_FORMAT_VERSION,
            "summary": _jsonable(summary),
            "rows": [{key: _jsonable(value) for key, value in row.items()} for row in rows],
        },
        output_path,
    )
    return output_path


def evaluate_deployment(
    *,
    policy_car: Path,
    policy_truck: Path,
    gt_car: Path,
    gt_truck: Path,
    reward_dir: Path,
    train_val_summary_path: Path,
    output_dir: Path,
    checkpoint_stem: str,
    device: str,
    window_len: int,
    bundle: RewardBundle | None = None,
) -> dict[str, str]:
    effective_bundle = bundle or _load_reward_bundle(reward_dir, checkpoint_stem=checkpoint_stem, device=device)
    rows = score_deployment_rollouts(
        policy_car_path=policy_car,
        policy_truck_path=policy_truck,
        gt_car_path=gt_car,
        gt_truck_path=gt_truck,
        bundle=effective_bundle,
        window_len=window_len,
    )
    train_val_summary = (
        json.loads(train_val_summary_path.read_text(encoding="utf-8")) if train_val_summary_path.exists() else None
    )
    summary = build_summary(rows, train_val_summary)
    summary.update(
        {
            "inputs": {
                "policy_car": str(policy_car),
                "policy_truck": str(policy_truck),
                "gt_car": str(gt_car),
                "gt_truck": str(gt_truck),
                "reward_dir": str(reward_dir),
                "train_val_summary": str(train_val_summary_path),
            }
        }
    )

    report_dir = output_dir / "report"
    outputs = {
        "scored": str(write_scored_payload(rows, summary, output_dir / "scored_first32_deployment.pt")),
        "csv": str(write_rows_csv(rows, report_dir / "deployment_first32_windows.csv")),
        "summary_json": str(report_dir / "deployment_first32_summary.json"),
        "narrative": str(write_markdown(summary, report_dir / "deployment_first32_summary.md")),
        "histograms": str(plot_deployment_histograms(rows, report_dir / "deployment_histograms.png")),
        "margin_by_map": str(plot_margin_by_map(rows, report_dir / "deployment_margin_by_map.png")),
        "probability_by_map": str(plot_probability_by_map(rows, report_dir / "deployment_probability_by_map.png")),
        "score_bars": str(plot_score_bars(rows, report_dir / "deployment_score_bars.png")),
        "train_val_vs_deployment": str(plot_train_val_vs_deployment(summary, report_dir / "train_val_vs_deployment.png")),
    }
    Path(outputs["summary_json"]).write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score first-32-step policy and GT rollout segments with the trained preference reward model."
    )
    parser.add_argument("--policy-car", type=Path, default=DEFAULT_POLICY_CAR)
    parser.add_argument("--policy-truck", type=Path, default=DEFAULT_POLICY_TRUCK)
    parser.add_argument("--gt-car", type=Path, default=DEFAULT_GT_CAR)
    parser.add_argument("--gt-truck", type=Path, default=DEFAULT_GT_TRUCK)
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--train-val-summary", type=Path, default=DEFAULT_TRAIN_VAL_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-stem", type=str, default=DEFAULT_CHECKPOINT_STEM)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--window-len", type=int, default=DEFAULT_WINDOW_LEN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = evaluate_deployment(
        policy_car=args.policy_car,
        policy_truck=args.policy_truck,
        gt_car=args.gt_car,
        gt_truck=args.gt_truck,
        reward_dir=args.reward_dir,
        train_val_summary_path=args.train_val_summary,
        output_dir=args.output_dir,
        checkpoint_stem=args.checkpoint_stem,
        device=args.device,
        window_len=args.window_len,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
