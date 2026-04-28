#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib.preference_reward import PreferenceRewardMetadata, build_state_action_features


DEFAULT_REWARD_DIR = Path("pufferlib/resources/drive/preferences/models/turning_all_90_10_30rounds_continue1")
DEFAULT_PREFERENCE_CONFIG = Path("pufferlib/resources/drive/models/car-baseline-full-boston-pref-from-base/conf1_preference.ini")
DEFAULT_POLICY_CAR = Path("outputs/preference_rollout_model_compare_full_no_resets/rollouts/car-baseline-full-boston_rollouts.pt")
DEFAULT_POLICY_TRUCK = Path("outputs/preference_rollout_model_compare_full_no_resets/rollouts/truck-baseline-full-boston_rollouts.pt")
DEFAULT_GT_CAR = Path("outputs/preference_ground_truth_context_compare_full/rollouts/ground-truth-car-fit_rollouts.pt")
DEFAULT_GT_TRUCK = Path("outputs/preference_ground_truth_context_compare_full/rollouts/ground-truth-truck-context-replay_rollouts.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/preference_segment_scoring_full")
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class SegmentModelBundle:
    metadata: PreferenceRewardMetadata
    reward_model: Any
    action_space: gymnasium.Space
    action_type: str


def _load_rollout_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_segment_reward_model(reward_dir: Path) -> SegmentModelBundle:
    summary = json.loads((reward_dir / "offline_truck_context_reward_summary.json").read_text(encoding="utf-8"))
    metadata = PreferenceRewardMetadata(
        observation_mode=summary.get("observation_mode"),
        obs_dim=int(summary["obs_dim"]),
        action_dim=int(summary["action_dim"]),
        size_segment=int(summary["size_segment"]) if summary.get("size_segment") is not None else None,
        action_encoding=summary.get("action_encoding"),
        ensemble_size=int(summary["ensemble_size"]),
        activation=str(summary.get("activation", "tanh")),
    )

    from preferences import reward_model as reward_model_module

    reward_model_module.device = "cpu"
    reward_model = reward_model_module.RewardModel(
        ds=metadata.obs_dim,
        da=metadata.action_dim,
        ensemble_size=metadata.ensemble_size,
        lr=3e-4,
        mb_size=1,
        size_segment=int(metadata.size_segment or 1),
        capacity=max(2, metadata.ensemble_size),
        activation=metadata.activation,
    )
    reward_model.load(str(reward_dir), "offline_truck_context")
    action_space = gymnasium.spaces.MultiDiscrete([metadata.action_dim])
    return SegmentModelBundle(
        metadata=metadata,
        reward_model=reward_model,
        action_space=action_space,
        action_type="discrete",
    )


def _features_from_rollout(rollout: dict[str, Any], bundle: SegmentModelBundle) -> np.ndarray:
    return build_state_action_features(
        rollout["observations"],
        rollout["actions"],
        metadata=bundle.metadata,
        action_space=bundle.action_space,
        action_type=bundle.action_type,
    )


def _sliding_windows(features: np.ndarray, window_len: int) -> np.ndarray:
    if features.shape[0] < window_len:
        return np.zeros((0, window_len, features.shape[1]), dtype=np.float32)
    windows = [features[start : start + window_len] for start in range(0, features.shape[0] - window_len + 1)]
    return np.stack(windows).astype(np.float32)


def _segment_rewards(bundle: SegmentModelBundle, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if windows.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, bundle.metadata.ensemble_size), dtype=np.float32)

    per_member = []
    for member in range(bundle.metadata.ensemble_size):
        member_reward = bundle.reward_model.r_hat_member(windows, member=member).detach().cpu().numpy()
        member_sum = member_reward.sum(axis=1).reshape(-1).astype(np.float32)
        per_member.append(member_sum)
    member_matrix = np.stack(per_member, axis=1)
    return member_matrix.mean(axis=1), member_matrix


def _compare_rollout_pair(
    *,
    label: str,
    left_rollout: dict[str, Any],
    right_rollout: dict[str, Any],
    bundle: SegmentModelBundle,
    top_k: int,
) -> dict[str, Any]:
    window_len = int(bundle.metadata.size_segment or 1)
    aligned_steps = min(int(left_rollout["steps"]), int(right_rollout["steps"]))
    if aligned_steps < window_len:
        return {
            "comparison": label,
            "map_name": left_rollout["map_name"],
            "scenario_type": left_rollout["scenario_type"],
            "delta_heading_deg": float(left_rollout["delta_heading_deg"]),
            "aligned_steps": aligned_steps,
            "window_len": window_len,
            "window_count": 0,
            "status": "too_short",
        }

    left_features = _features_from_rollout(left_rollout, bundle)[:aligned_steps]
    right_features = _features_from_rollout(right_rollout, bundle)[:aligned_steps]
    left_windows = _sliding_windows(left_features, window_len)
    right_windows = _sliding_windows(right_features, window_len)

    left_scores, left_member = _segment_rewards(bundle, left_windows)
    right_scores, right_member = _segment_rewards(bundle, right_windows)
    margin = left_scores - right_scores

    safe_top_k = max(1, min(int(top_k), int(left_scores.size)))
    left_top_k = float(np.mean(np.sort(left_scores)[-safe_top_k:]))
    right_top_k = float(np.mean(np.sort(right_scores)[-safe_top_k:]))

    pair_probs = []
    for member in range(bundle.metadata.ensemble_size):
        logits = np.stack([left_member[:, member], right_member[:, member]], axis=1)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        pair_probs.append(probs[:, 0])
    pair_prob_mean = np.mean(np.stack(pair_probs, axis=1), axis=1)

    return {
        "comparison": label,
        "map_name": left_rollout["map_name"],
        "scenario_type": left_rollout["scenario_type"],
        "delta_heading_deg": float(left_rollout["delta_heading_deg"]),
        "aligned_steps": aligned_steps,
        "window_len": window_len,
        "window_count": int(left_scores.size),
        "status": "ok",
        "left_mean_window_score": float(np.mean(left_scores)),
        "right_mean_window_score": float(np.mean(right_scores)),
        "mean_window_margin": float(np.mean(margin)),
        "median_window_margin": float(np.median(margin)),
        "top_k_left_mean_window_score": left_top_k,
        "top_k_right_mean_window_score": right_top_k,
        "top_k_margin": float(left_top_k - right_top_k),
        "left_win_fraction": float(np.mean(margin > 0.0)),
        "tie_fraction": float(np.mean(np.isclose(margin, 0.0))),
        "mean_pair_probability_left_beats_right": float(np.mean(pair_prob_mean)),
    }


def _index_rollouts(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {rollout["map_name"]: rollout for rollout in payload["rollouts"]}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    by_comparison = {}
    for comparison in sorted({row["comparison"] for row in ok_rows}):
        subset = [row for row in ok_rows if row["comparison"] == comparison]
        by_comparison[comparison] = {
            "map_count": len(subset),
            "mean_window_margin": float(np.mean([row["mean_window_margin"] for row in subset])) if subset else 0.0,
            "median_window_margin": float(np.mean([row["median_window_margin"] for row in subset])) if subset else 0.0,
            "mean_left_win_fraction": float(np.mean([row["left_win_fraction"] for row in subset])) if subset else 0.0,
            "mean_pair_probability_left_beats_right": float(
                np.mean([row["mean_pair_probability_left_beats_right"] for row in subset])
            )
            if subset
            else 0.0,
            "mean_top_k_margin": float(np.mean([row["top_k_margin"] for row in subset])) if subset else 0.0,
        }
    return {
        "total_rows": len(rows),
        "ok_rows": len(ok_rows),
        "comparisons": by_comparison,
    }


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_markdown(rows: list[dict[str, Any]], summary: dict[str, Any], output_path: Path) -> Path:
    lines = [
        "# Segment Scoring Summary",
        "",
        "32-step matched sliding-window comparisons using the reward model as a true segment scorer.",
        "",
    ]
    for comparison, stats in summary["comparisons"].items():
        lines.append(f"## {comparison}")
        lines.append(f"- map_count: `{stats['map_count']}`")
        lines.append(f"- mean_window_margin: `{stats['mean_window_margin']:.4f}`")
        lines.append(f"- mean_left_win_fraction: `{stats['mean_left_win_fraction']:.4f}`")
        lines.append(f"- mean_pair_probability_left_beats_right: `{stats['mean_pair_probability_left_beats_right']:.4f}`")
        lines.append(f"- mean_top_k_margin: `{stats['mean_top_k_margin']:.4f}`")
        lines.append("")

    lines.extend(
        [
            "| Comparison | Map | Type | Windows | Mean margin | Median margin | Left win frac | Pair prob | Top-k margin |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["status"] != "ok":
            continue
        lines.append(
            f"| {row['comparison']} | {row['map_name']} | {row['scenario_type']} | {row['window_count']} | "
            f"{row['mean_window_margin']:.3f} | {row['median_window_margin']:.3f} | "
            f"{row['left_win_fraction']:.3f} | {row['mean_pair_probability_left_beats_right']:.3f} | {row['top_k_margin']:.3f} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _plot_margin_bars(rows: list[dict[str, Any]], output_path: Path) -> Path:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    comparisons = sorted({row["comparison"] for row in ok_rows})
    map_names = sorted({row["map_name"] for row in ok_rows})
    x = np.arange(len(map_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(map_names) * 1.2), 6))
    for idx, comparison in enumerate(comparisons):
        values = []
        for map_name in map_names:
            match = next((row for row in ok_rows if row["comparison"] == comparison and row["map_name"] == map_name), None)
            values.append(float(match["mean_window_margin"]) if match is not None else 0.0)
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=comparison)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(map_names, rotation=45, ha="right")
    ax.set_ylabel("Mean 32-step window margin")
    ax.set_title("Segment-level preference margin by map")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_win_fraction(rows: list[dict[str, Any]], output_path: Path) -> Path:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    comparisons = sorted({row["comparison"] for row in ok_rows})
    map_names = sorted({row["map_name"] for row in ok_rows})
    x = np.arange(len(map_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(map_names) * 1.2), 6))
    for idx, comparison in enumerate(comparisons):
        values = []
        for map_name in map_names:
            match = next((row for row in ok_rows if row["comparison"] == comparison and row["map_name"] == map_name), None)
            values.append(float(match["left_win_fraction"]) if match is not None else 0.0)
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=comparison)
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.4, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(map_names, rotation=45, ha="right")
    ax.set_ylabel("Fraction of 32-step windows left > right")
    ax.set_title("Segment-level win fraction by map")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score rollout artifacts with 32-step segment scoring.")
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--policy-car", type=Path, default=DEFAULT_POLICY_CAR)
    parser.add_argument("--policy-truck", type=Path, default=DEFAULT_POLICY_TRUCK)
    parser.add_argument("--gt-car", type=Path, default=DEFAULT_GT_CAR)
    parser.add_argument("--gt-truck", type=Path, default=DEFAULT_GT_TRUCK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = _load_segment_reward_model(args.reward_dir.resolve())
    policy_car = _index_rollouts(_load_rollout_payload(args.policy_car.resolve()))
    policy_truck = _index_rollouts(_load_rollout_payload(args.policy_truck.resolve()))
    gt_car = _index_rollouts(_load_rollout_payload(args.gt_car.resolve()))
    gt_truck = _index_rollouts(_load_rollout_payload(args.gt_truck.resolve()))

    map_names = sorted(set(policy_car) & set(policy_truck) & set(gt_car) & set(gt_truck))
    rows = []
    for map_name in map_names:
        rows.append(
            _compare_rollout_pair(
                label="policy_car_vs_truck",
                left_rollout=policy_car[map_name],
                right_rollout=policy_truck[map_name],
                bundle=bundle,
                top_k=args.top_k,
            )
        )
        rows.append(
            _compare_rollout_pair(
                label="gt_car_vs_truck_context",
                left_rollout=gt_car[map_name],
                right_rollout=gt_truck[map_name],
                bundle=bundle,
                top_k=args.top_k,
            )
        )

    summary = _summary(rows)
    output_dir = args.output_dir.resolve()
    csv_path = _write_csv(rows, output_dir / "segment_scoring_summary.csv")
    md_path = _write_markdown(rows, summary, output_dir / "segment_scoring_summary.md")
    json_path = output_dir / "segment_scoring_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    margin_plot = _plot_margin_bars(rows, output_dir / "segment_mean_margin_by_map.png")
    win_plot = _plot_win_fraction(rows, output_dir / "segment_win_fraction_by_map.png")

    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "markdown": str(md_path),
                "summary_json": str(json_path),
                "margin_plot": str(margin_plot),
                "win_fraction_plot": str(win_plot),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
