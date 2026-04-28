#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
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


DEFAULT_SCORED = Path("outputs/preference_eval/preference_first32_deployment_eval_truck_context_policy/scored_first32_deployment.pt")
DEFAULT_POLICY_TRUCK = Path(
    "outputs/preference_eval/preference_rollout_model_compare_truck_context/rollouts/"
    "truck-baseline-full-boston_truck-context_rollouts.pt"
)
DEFAULT_POLICY_CAR = Path(
    "outputs/preference_eval/preference_rollout_model_compare_truck_context/rollouts/"
    "car-baseline-full-boston_truck-context_rollouts.pt"
)
DEFAULT_GT_TRUCK = Path(
    "outputs/preference_eval/preference_ground_truth_context_compare_full/rollouts/"
    "ground-truth-truck-context-preferred-branch_rollouts.pt"
)
DEFAULT_GT_CAR = Path(
    "outputs/preference_eval/preference_ground_truth_context_compare_full/rollouts/"
    "ground-truth-car-context-rejected-branch_rollouts.pt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/preference_eval/preference_first32_deployment_eval_truck_context_policy/report/trajectory_grids")


def _load_rows(scored_path: Path) -> list[dict[str, Any]]:
    payload = torch.load(scored_path, map_location="cpu", weights_only=False)
    return list(payload["rows"])


def _load_rollouts(path: Path) -> dict[str, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {str(row["map_name"]): row for row in payload.get("rollouts", [])}


def _xy(rollout: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rollout.get("x", []), dtype=np.float32).reshape(-1)
    y = np.asarray(rollout.get("y", []), dtype=np.float32).reshape(-1)
    count = min(len(x), len(y))
    return x[:count], y[:count]


def _plot_pair_grid(
    *,
    rows: list[dict[str, Any]],
    preferred_rollouts: dict[str, dict[str, Any]],
    rejected_rollouts: dict[str, dict[str, Any]],
    preferred_label: str,
    rejected_label: str,
    title: str,
    output_path: Path,
) -> Path:
    rows = sorted(rows, key=lambda row: str(row["map_name"]))
    cols = 4
    row_count = int(math.ceil(len(rows) / cols))
    fig, axes = plt.subplots(row_count, cols, figsize=(18, max(8.5, row_count * 4.1)))
    axes = np.asarray(axes, dtype=object).reshape(row_count, cols)
    for ax in axes.flat:
        ax.axis("off")

    for idx, row in enumerate(rows):
        ax = axes.flat[idx]
        ax.axis("on")
        map_name = str(row["map_name"])
        preferred = preferred_rollouts.get(map_name)
        rejected = rejected_rollouts.get(map_name)
        if preferred is None or rejected is None:
            ax.text(0.5, 0.5, "missing rollout", ha="center", va="center")
            ax.set_title(map_name)
            continue

        status = str(row.get("status", "ok"))
        aligned_steps = int(row.get("aligned_steps") or min(int(preferred.get("steps", 0)), int(rejected.get("steps", 0))))
        window_len = int(row.get("window_len", 32))
        segment_len = min(window_len, aligned_steps)

        for rollout, label, color in (
            (preferred, preferred_label, "tab:blue"),
            (rejected, rejected_label, "tab:orange"),
        ):
            x, y = _xy(rollout)
            if len(x) == 0:
                continue
            ax.plot(x, y, color=color, linewidth=1.0, alpha=0.25)
            highlight = max(1, min(segment_len if segment_len > 0 else len(x), len(x)))
            ax.plot(x[:highlight], y[:highlight], color=color, linewidth=3.0, label=label)
            ax.scatter([x[0]], [y[0]], color=color, marker="o", s=22, zorder=3)
            ax.scatter([x[highlight - 1]], [y[highlight - 1]], color=color, marker="x", s=32, zorder=3)

        if status == "ok":
            margin = float(row["margin"])
            probability = float(row["prob_preferred"])
            detail = f"margin={margin:.2f}, p_truck={probability:.2f}"
        else:
            detail = f"{status}: {row.get('skip_reason', '')}"
        ax.set_title(f"{map_name}\n{detail}", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    note = (
        "Blue is truck/context preferred side; orange is car/rejected side. "
        "Thin lines show each collected rollout; thick lines show the scored first-32 window, or the available aligned prefix if shorter. "
        "Circle marks the segment start, x marks the segment end. Positive margin means the reward model prefers truck/context."
    )
    fig.suptitle(title, fontsize=15)
    fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=10, color="dimgray", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _row_by_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["map_name"]): row for row in rows}


def _score_detail(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing"
    status = str(row.get("status", "ok"))
    if status != "ok":
        return f"{status}"
    return f"m={float(row['margin']):.1f}, p={float(row['prob_preferred']):.2f}"


def _plot_overlay_grid(
    *,
    policy_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    policy_truck_rollouts: dict[str, dict[str, Any]],
    policy_car_rollouts: dict[str, dict[str, Any]],
    gt_truck_rollouts: dict[str, dict[str, Any]],
    gt_car_rollouts: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    policy_by_map = _row_by_map(policy_rows)
    gt_by_map = _row_by_map(gt_rows)
    map_names = sorted(set(gt_by_map) | set(policy_by_map))
    cols = 4
    row_count = int(math.ceil(len(map_names) / cols))
    fig, axes = plt.subplots(row_count, cols, figsize=(18, max(8.8, row_count * 4.25)))
    axes = np.asarray(axes, dtype=object).reshape(row_count, cols)
    for ax in axes.flat:
        ax.axis("off")

    specs = [
        ("GT truck", gt_truck_rollouts, "tab:blue", "-"),
        ("GT car", gt_car_rollouts, "tab:orange", "-"),
        ("policy truck", policy_truck_rollouts, "tab:green", "--"),
        ("policy car", policy_car_rollouts, "tab:red", "--"),
    ]
    for idx, map_name in enumerate(map_names):
        ax = axes.flat[idx]
        ax.axis("on")
        plotted = False
        for label, rollout_lookup, color, linestyle in specs:
            rollout = rollout_lookup.get(map_name)
            if rollout is None:
                continue
            x, y = _xy(rollout)
            if len(x) == 0:
                continue
            plotted = True
            highlight = max(1, min(32, len(x)))
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.28)
            ax.plot(x[:highlight], y[:highlight], color=color, linestyle=linestyle, linewidth=2.8, label=label)
            ax.scatter([x[0]], [y[0]], color=color, marker="o", s=18, zorder=3)
            ax.scatter([x[highlight - 1]], [y[highlight - 1]], color=color, marker="x", s=30, zorder=3)

        if not plotted:
            ax.text(0.5, 0.5, "missing rollouts", ha="center", va="center")
        policy_detail = _score_detail(policy_by_map.get(map_name))
        gt_detail = _score_detail(gt_by_map.get(map_name))
        ax.set_title(f"{map_name}\nGT {gt_detail} | policy {policy_detail}", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    note = (
        "Each subplot overlays all four trajectories for the same map. Solid lines are corrected GT branches; dashed lines are policy rollouts in the shared truck context. "
        "Blue/green are truck/context; orange/red are car. Thick segments show the first 32 steps used by the deployment score when available; thin lines show the full collected rollout. "
        "Circle marks rollout start, x marks the end of the highlighted prefix. Title margins are truck/context score minus car score; positive means truck/context is preferred."
    )
    fig.suptitle("GT and Policy Trajectories on the Same Maps", fontsize=15)
    fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=10, color="dimgray", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_plots(
    *,
    scored_path: Path,
    policy_truck: Path,
    policy_car: Path,
    gt_truck: Path,
    gt_car: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows = _load_rows(scored_path)
    policy_rows = [row for row in rows if row["comparison"] == "policy_truck_vs_car"]
    gt_rows = [row for row in rows if row["comparison"] == "gt_truck_context_vs_car_fit"]
    policy_truck_rollouts = _load_rollouts(policy_truck)
    policy_car_rollouts = _load_rollouts(policy_car)
    gt_truck_rollouts = _load_rollouts(gt_truck)
    gt_car_rollouts = _load_rollouts(gt_car)

    outputs = {
        "policy_grid": str(
            _plot_pair_grid(
                rows=policy_rows,
                preferred_rollouts=policy_truck_rollouts,
                rejected_rollouts=policy_car_rollouts,
                preferred_label="truck policy in truck context",
                rejected_label="car policy in truck context",
                title="Shared Truck-Context Policy Rollouts Scored by Preference Reward",
                output_path=output_dir / "policy_truck_context_trajectory_grid.png",
            )
        ),
        "gt_grid": str(
            _plot_pair_grid(
                rows=gt_rows,
                preferred_rollouts=gt_truck_rollouts,
                rejected_rollouts=gt_car_rollouts,
                preferred_label="GT truck branch",
                rejected_label="GT car branch",
                title="Corrected GT Context Rollouts Scored by Preference Reward",
                output_path=output_dir / "corrected_gt_context_trajectory_grid.png",
            )
        ),
        "gt_policy_overlay_grid": str(
            _plot_overlay_grid(
                policy_rows=policy_rows,
                gt_rows=gt_rows,
                policy_truck_rollouts=policy_truck_rollouts,
                policy_car_rollouts=policy_car_rollouts,
                gt_truck_rollouts=gt_truck_rollouts,
                gt_car_rollouts=gt_car_rollouts,
                output_path=output_dir / "gt_policy_overlay_trajectory_grid.png",
            )
        ),
    }
    summary_path = output_dir / "trajectory_grid_summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    outputs["summary_json"] = str(summary_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trajectory grids for shared truck-context policy preference eval.")
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--policy-truck", type=Path, default=DEFAULT_POLICY_TRUCK)
    parser.add_argument("--policy-car", type=Path, default=DEFAULT_POLICY_CAR)
    parser.add_argument("--gt-truck", type=Path, default=DEFAULT_GT_TRUCK)
    parser.add_argument("--gt-car", type=Path, default=DEFAULT_GT_CAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_plots(
        scored_path=args.scored,
        policy_truck=args.policy_truck,
        policy_car=args.policy_car,
        gt_truck=args.gt_truck,
        gt_car=args.gt_car,
        output_dir=args.output_dir,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
