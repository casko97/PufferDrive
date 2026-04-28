#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MANIFEST = Path("outputs/preference_rollout_model_compare_full_no_resets/sample_manifest.json")
DEFAULT_ROLLOUT_CSV = Path("outputs/preference_rollout_model_compare_full_no_resets/report/preference_reward_rollouts.csv")
DEFAULT_GT_CSV = Path("outputs/preference_ground_truth_context_compare_full/report/preference_reward_rollouts.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/preference_combined_summary_full")


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _parse_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _parse_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def _index_by_model(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["model_name"], row["map_name"]): row for row in rows}


def _favored_name(left_name: str, left_score: float, right_name: str, right_score: float) -> str:
    if left_score > right_score:
        return left_name
    if right_score > left_score:
        return right_name
    return "tie"


def build_rows(manifest: dict[str, Any], rollout_rows: list[dict[str, str]], gt_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rollout_index = _index_by_model(rollout_rows)
    gt_index = _index_by_model(gt_rows)

    rows: list[dict[str, Any]] = []
    for item in manifest["maps"]:
        map_name = item["map_name"]
        car_rollout = rollout_index[("car-baseline-full-boston", map_name)]
        truck_rollout = rollout_index[("truck-baseline-full-boston", map_name)]
        gt_car = gt_index[("ground-truth-car-fit", map_name)]
        gt_truck = gt_index[("ground-truth-truck-context-replay", map_name)]

        car_rollout_score = _parse_float(car_rollout, "cumulative_pref_shaped")
        truck_rollout_score = _parse_float(truck_rollout, "cumulative_pref_shaped")
        gt_car_score = _parse_float(gt_car, "cumulative_pref_shaped")
        gt_truck_score = _parse_float(gt_truck, "cumulative_pref_shaped")

        row = {
            "map_name": map_name,
            "scenario_type": item["scenario_type"],
            "delta_heading_deg": float(item["delta_heading_deg"]),
            "rollout_car_score": car_rollout_score,
            "rollout_truck_score": truck_rollout_score,
            "rollout_gap_car_minus_truck": car_rollout_score - truck_rollout_score,
            "rollout_car_steps": _parse_int(car_rollout, "steps"),
            "rollout_truck_steps": _parse_int(truck_rollout, "steps"),
            "gt_car_score": gt_car_score,
            "gt_truck_context_score": gt_truck_score,
            "gt_gap_car_minus_truck": gt_car_score - gt_truck_score,
            "gt_steps": _parse_int(gt_car, "steps"),
            "rollout_favored": _favored_name("car", car_rollout_score, "truck", truck_rollout_score),
            "gt_favored": _favored_name("car", gt_car_score, "truck", gt_truck_score),
            "agreement": (
                _favored_name("car", car_rollout_score, "truck", truck_rollout_score)
                == _favored_name("car", gt_car_score, "truck", gt_truck_score)
            ),
        }
        rows.append(row)

    rows.sort(key=lambda row: (0 if row["scenario_type"] == "turning" else 1, row["map_name"]))
    return rows


def _format_score(value: float) -> str:
    return f"{value:.2f}"


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "map_name",
        "scenario_type",
        "delta_heading_deg",
        "rollout_car_score",
        "rollout_truck_score",
        "rollout_gap_car_minus_truck",
        "rollout_car_steps",
        "rollout_truck_steps",
        "gt_car_score",
        "gt_truck_context_score",
        "gt_gap_car_minus_truck",
        "gt_steps",
        "rollout_favored",
        "gt_favored",
        "agreement",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Combined Preference Summary",
        "",
        "| Map | Type | Δ heading | Rollout car | Rollout truck | Rollout gap | GT car-fit | GT truck-context | GT gap | Rollout fav | GT fav | Agree |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["map_name"],
                    row["scenario_type"],
                    _format_score(row["delta_heading_deg"]),
                    _format_score(row["rollout_car_score"]),
                    _format_score(row["rollout_truck_score"]),
                    _format_score(row["rollout_gap_car_minus_truck"]),
                    _format_score(row["gt_car_score"]),
                    _format_score(row["gt_truck_context_score"]),
                    _format_score(row["gt_gap_car_minus_truck"]),
                    str(row["rollout_favored"]),
                    str(row["gt_favored"]),
                    "yes" if row["agreement"] else "no",
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_summary_json(rows: list[dict[str, Any]], output_path: Path) -> Path:
    rollout_car_wins = sum(1 for row in rows if row["rollout_favored"] == "car")
    rollout_truck_wins = sum(1 for row in rows if row["rollout_favored"] == "truck")
    gt_car_wins = sum(1 for row in rows if row["gt_favored"] == "car")
    gt_truck_wins = sum(1 for row in rows if row["gt_favored"] == "truck")
    agreement_count = sum(1 for row in rows if row["agreement"])
    payload = {
        "map_count": len(rows),
        "rollout_car_wins": rollout_car_wins,
        "rollout_truck_wins": rollout_truck_wins,
        "gt_car_wins": gt_car_wins,
        "gt_truck_wins": gt_truck_wins,
        "agreement_count": agreement_count,
        "agreement_fraction": agreement_count / len(rows) if rows else 0.0,
        "mean_rollout_gap_car_minus_truck": sum(row["rollout_gap_car_minus_truck"] for row in rows) / len(rows) if rows else 0.0,
        "mean_gt_gap_car_minus_truck": sum(row["gt_gap_car_minus_truck"] for row in rows) / len(rows) if rows else 0.0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def plot_gap_scatter(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"turning": "#1f77b4", "straight": "#ff7f0e"}
    markers = {True: "o", False: "x"}

    for row in rows:
        ax.scatter(
            row["rollout_gap_car_minus_truck"],
            row["gt_gap_car_minus_truck"],
            color=colors.get(row["scenario_type"], "#666666"),
            marker=markers[bool(row["agreement"])],
            s=80,
            alpha=0.85,
        )
        ax.annotate(row["map_name"].replace(".bin", ""), (row["rollout_gap_car_minus_truck"], row["gt_gap_car_minus_truck"]), fontsize=7, alpha=0.8)

    lim = max(
        1.0,
        max(abs(row["rollout_gap_car_minus_truck"]) for row in rows),
        max(abs(row["gt_gap_car_minus_truck"]) for row in rows),
    ) * 1.1
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.axvline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Rollout gap: car - truck")
    ax.set_ylabel("GT-context gap: car - truck")
    ax.set_title("Rollout vs GT-context preference gaps")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_per_map_gaps(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['map_name']}\n{row['scenario_type']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 1.2), 6))
    rollout_gaps = [row["rollout_gap_car_minus_truck"] for row in rows]
    gt_gaps = [row["gt_gap_car_minus_truck"] for row in rows]
    ax.bar(
        x - width / 2,
        rollout_gaps,
        width=width,
        label="car policy rollout - truck policy rollout",
        color="#4c78a8",
    )
    ax.bar(
        x + width / 2,
        gt_gaps,
        width=width,
        label="ground-truth-car-fit - ground-truth-truck-context-replay",
        color="#f58518",
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Cumulative shaped preference gap")
    ax.set_title("Per-map comparison of rollout and GT-context preference gaps")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_per_map_four_way_scores(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['map_name']}\n{row['scenario_type']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(13, len(rows) * 1.35), 6.5))
    rollout_car = [row["rollout_car_score"] for row in rows]
    rollout_truck = [row["rollout_truck_score"] for row in rows]
    gt_car = [row["gt_car_score"] for row in rows]
    gt_truck = [row["gt_truck_context_score"] for row in rows]

    ax.bar(x - 1.5 * width, rollout_car, width=width, label="Rollout car policy", color="#4c78a8")
    ax.bar(x - 0.5 * width, rollout_truck, width=width, label="Rollout truck policy", color="#f58518")
    ax.bar(x + 0.5 * width, gt_car, width=width, label="GT car-fit", color="#54a24b")
    ax.bar(x + 1.5 * width, gt_truck, width=width, label="GT truck-context", color="#e45756")

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Cumulative shaped preference reward")
    ax.set_title("Per-map cumulative preference reward: policies and GT-context traces")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_win_agreement(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    categories = [
        ("car", "car"),
        ("car", "truck"),
        ("truck", "car"),
        ("truck", "truck"),
    ]
    counts = [sum(1 for row in rows if row["rollout_favored"] == left and row["gt_favored"] == right) for left, right in categories]
    labels = [f"rollout {left}\nGT {right}" for left, right in categories]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts, color=["#54a24b", "#e45756", "#e45756", "#54a24b"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(count), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Map count")
    ax.set_title("Agreement between rollout and GT-context winners")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_joint_gap(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['map_name']}\n{row['scenario_type']}" for row in rows]
    x = np.arange(len(rows))
    joint_gap = [
        0.5 * (row["rollout_gap_car_minus_truck"] + row["gt_gap_car_minus_truck"])
        for row in rows
    ]
    colors = ["#4c78a8" if row["scenario_type"] == "turning" else "#f58518" for row in rows]

    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 1.2), 5.8))
    ax.bar(x, joint_gap, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Joint preference gap: car - truck")
    ax.set_title("Per-map joint preference gap across rollout and GT-context")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined summary table for rollout and GT preference comparisons.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rollout-csv", type=Path, default=DEFAULT_ROLLOUT_CSV)
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = build_rows(
        manifest=manifest,
        rollout_rows=_load_csv(args.rollout_csv),
        gt_rows=_load_csv(args.gt_csv),
    )

    output_dir = args.output_dir.resolve()
    csv_path = write_csv(rows, output_dir / "combined_preference_summary.csv")
    md_path = write_markdown(rows, output_dir / "combined_preference_summary.md")
    json_path = write_summary_json(rows, output_dir / "combined_preference_summary_stats.json")
    scatter_path = plot_gap_scatter(rows, output_dir / "rollout_vs_gt_gap_scatter.png")
    gap_bars_path = plot_per_map_gaps(rows, output_dir / "per_map_gap_comparison.png")
    four_way_path = plot_per_map_four_way_scores(rows, output_dir / "per_map_four_way_cumulative_rewards.png")
    agreement_path = plot_win_agreement(rows, output_dir / "winner_agreement_counts.png")
    joint_gap_path = plot_joint_gap(rows, output_dir / "per_map_joint_gap_comparison.png")
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "markdown": str(md_path),
                "stats": str(json_path),
                "gap_scatter": str(scatter_path),
                "per_map_gaps": str(gap_bars_path),
                "per_map_four_way_scores": str(four_way_path),
                "winner_agreement": str(agreement_path),
                "joint_gap": str(joint_gap_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
