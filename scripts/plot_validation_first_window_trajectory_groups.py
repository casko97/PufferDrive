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


DEFAULT_SCORED_WINDOWS = Path("outputs/preference_eval/preference_train_val_dataset_eval/scored_preference_windows.pt")
DEFAULT_PAIRED_FITS = Path(
    "/home/casko/phd-code/pufferdrive-kth/"
    "pufferlib/resources/drive/preferences/offline_fits/nuplan_boston_all_chunk64_paired_offline_fits.pt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/preference_eval/preference_train_val_dataset_eval/report/validation/trajectory_groups")
DEFAULT_GROUP_SIZE = 12


def _load_selected_validation_rows(scored_windows_path: Path, group_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = torch.load(scored_windows_path, map_location="cpu", weights_only=False)
    validation_rows = [row for row in payload["rows"] if row["split"] == "validation"]
    validation_rows = sorted(validation_rows, key=lambda row: float(row["margin"]))
    car_preferred = validation_rows[:group_size]
    truck_preferred = validation_rows[-group_size:][::-1]
    return truck_preferred, car_preferred


def _resolve_shard_path(paired_fits_path: Path, shard_info: dict[str, Any]) -> Path:
    raw = Path(shard_info["path"])
    if raw.exists():
        return raw
    same_dir = paired_fits_path.parent / raw.name
    sibling_dir = paired_fits_path.parent / f"{paired_fits_path.stem}_shards" / raw.name
    if same_dir.exists():
        return same_dir
    if sibling_dir.exists():
        return sibling_dir
    return raw


def _load_pairs_for_maps(paired_fits_path: Path, map_names: set[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    manifest = torch.load(paired_fits_path, map_location="cpu", weights_only=False)
    shared_maps = manifest.get("metadata", {}).get("shared_maps", [])
    map_to_index = {name: idx for idx, name in enumerate(shared_maps) if name in map_names}
    needed_shards = []
    for shard in manifest.get("shards", []):
        start = int(shard["start_index"])
        end = int(shard["end_index"])
        if any(start <= idx < end for idx in map_to_index.values()):
            needed_shards.append(shard)

    pairs: dict[str, dict[str, Any]] = {}
    remaining = set(map_names)
    for shard_info in needed_shards:
        shard_path = _resolve_shard_path(paired_fits_path, shard_info)
        shard_payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_pairs = shard_payload.get("pairs", {})
        for map_name in list(remaining):
            if map_name in shard_pairs:
                pairs[map_name] = shard_pairs[map_name]
                remaining.remove(map_name)
        if not remaining:
            break
    return pairs, sorted(remaining)


def _branch_xy(pair: dict[str, Any], branch_name: str) -> tuple[np.ndarray, np.ndarray]:
    replay = pair.get("truck_context_replay", {})
    branch = replay.get(branch_name, {})
    x_values = np.asarray(branch.get("rollout_x", []), dtype=np.float32)
    y_values = np.asarray(branch.get("rollout_y", []), dtype=np.float32)
    count = min(x_values.size, y_values.size)
    return x_values[:count], y_values[:count]


def _plot_group(
    *,
    rows: list[dict[str, Any]],
    pairs: dict[str, dict[str, Any]],
    title: str,
    group_label: str,
    output_path: Path,
) -> Path:
    cols = 4
    row_count = int(math.ceil(len(rows) / cols))
    fig, axes = plt.subplots(row_count, cols, figsize=(18, max(9, row_count * 4.2)))
    axes = np.asarray(axes, dtype=object).reshape(row_count, cols)
    for ax in axes.flat:
        ax.axis("off")

    for index, row in enumerate(rows):
        ax = axes.flat[index]
        ax.axis("on")
        map_name = str(row["map_name"])
        pair = pairs.get(map_name)
        if pair is None:
            ax.text(0.5, 0.5, "trajectory missing", ha="center", va="center")
            ax.set_title(map_name)
            continue

        start = int(row["timestep_start"])
        end = int(row["timestep_end"])
        for branch_name, label, color in (
            ("truck_branch", "truck/context (preferred)", "tab:blue"),
            ("car_branch", "car (rejected)", "tab:orange"),
        ):
            x_values, y_values = _branch_xy(pair, branch_name)
            if x_values.size == 0:
                continue
            ax.plot(x_values, y_values, color=color, linewidth=1.0, alpha=0.25)
            seg_start = max(0, min(start, len(x_values) - 1))
            seg_end = max(seg_start + 1, min(end, len(x_values)))
            ax.plot(x_values[seg_start:seg_end], y_values[seg_start:seg_end], color=color, linewidth=3.0, label=label)
            ax.scatter([x_values[seg_start]], [y_values[seg_start]], color=color, marker="o", s=20)
            ax.scatter([x_values[seg_end - 1]], [y_values[seg_end - 1]], color=color, marker="x", s=28)

        margin = float(row["margin"])
        probability = float(row["prob_preferred"])
        truck_score = float(row["preferred_score"])
        car_score = float(row["rejected_score"])
        ax.set_title(
            f"{map_name}\nmargin={margin:.2f}, p_truck={probability:.2f}\ntruck={truck_score:.1f}, car={car_score:.1f}",
            fontsize=9,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        if index == 0:
            ax.legend(fontsize=8, loc="best")

    note = (
        f"{group_label}. Blue is truck/context (preferred); orange is car (rejected). "
        "Thin lines show the full paired-fit rollout; thick lines show the scored first 32-step validation window. "
        "Positive margin = truck/context preferred; negative margin = car preferred. Circle marks window start, x marks window end."
    )
    fig.suptitle(title, fontsize=15)
    fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=10, color="dimgray", wrap=True)
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_plots(
    *,
    scored_windows_path: Path,
    paired_fits_path: Path,
    output_dir: Path,
    group_size: int,
) -> dict[str, Any]:
    truck_rows, car_rows = _load_selected_validation_rows(scored_windows_path, group_size=group_size)
    selected_names = {str(row["map_name"]) for row in truck_rows + car_rows}
    pairs, missing = _load_pairs_for_maps(paired_fits_path, selected_names)

    truck_path = output_dir / "validation_truck_context_preferred_trajectory_grid.png"
    car_path = output_dir / "validation_car_preferred_trajectory_grid.png"
    _plot_group(
        rows=truck_rows,
        pairs=pairs,
        title="Validation maps where reward model prefers truck/context over car",
        group_label="Top positive-margin validation first-window maps",
        output_path=truck_path,
    )
    _plot_group(
        rows=car_rows,
        pairs=pairs,
        title="Validation maps where reward model prefers car over truck/context",
        group_label="Most negative-margin validation first-window maps",
        output_path=car_path,
    )

    summary = {
        "truck_context_preferred_plot": str(truck_path),
        "car_preferred_plot": str(car_path),
        "missing_trajectories": missing,
        "truck_context_preferred_maps": [
            {
                "map_name": row["map_name"],
                "margin": float(row["margin"]),
                "prob_truck": float(row["prob_preferred"]),
            }
            for row in truck_rows
        ],
        "car_preferred_maps": [
            {
                "map_name": row["map_name"],
                "margin": float(row["margin"]),
                "prob_truck": float(row["prob_preferred"]),
            }
            for row in car_rows
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "validation_trajectory_groups_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot validation first-window trajectory groups by reward-model margin.")
    parser.add_argument("--scored-windows", type=Path, default=DEFAULT_SCORED_WINDOWS)
    parser.add_argument("--paired-fits", type=Path, default=DEFAULT_PAIRED_FITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_plots(
        scored_windows_path=args.scored_windows,
        paired_fits_path=args.paired_fits,
        output_dir=args.output_dir,
        group_size=args.group_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
