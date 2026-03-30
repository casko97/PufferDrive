from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.animate_truck_context_observations import _decode_local_obs

EXPORT_DEFAULT = Path("outputs/offline_fits/nuplan_boston_test10_paired_fits.pt")
PREFERENCE_DEFAULT = Path("outputs/preferences/nuplan_boston_test10_truck_context_preferences.pt")
OUTPUT_DEFAULT = Path("outputs/test_visualizations/truck_context_preference_pairs_grid.png")


def _to_local_frame(x: np.ndarray, y: np.ndarray, anchor_x: float, anchor_y: float, anchor_heading: float):
    dx = x - anchor_x
    dy = y - anchor_y
    cos_h = np.cos(anchor_heading)
    sin_h = np.sin(anchor_heading)
    local_x = dx * cos_h + dy * sin_h
    local_y = -dx * sin_h + dy * cos_h
    return local_x, local_y


def _start_distance(
    candidate: int,
    truck_rollout_x: np.ndarray,
    truck_rollout_y: np.ndarray,
    car_rollout_x: np.ndarray,
    car_rollout_y: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            (truck_rollout_x[candidate] - car_rollout_x[candidate]) ** 2
            + (truck_rollout_y[candidate] - car_rollout_y[candidate]) ** 2
        )
    )


def visualize_truck_context_preferences(
    export_path: Path,
    preference_path: Path,
    output_path: Path,
    max_pairs: int = 12,
) -> Path:
    export_payload = torch.load(export_path, map_location="cpu")
    pref_payload = torch.load(preference_path, map_location="cpu")
    window_meta = pref_payload["window_metadata"]
    pref_settings = pref_payload["metadata"]
    if not window_meta:
        raise ValueError(f"No preference windows available in {preference_path}")

    selected_meta = window_meta[: min(max_pairs, len(window_meta))]
    grouped = OrderedDict()
    for meta in selected_meta:
        grouped.setdefault(meta["map_name"], []).append(meta)

    rows = len(grouped)
    cols = max(len(items) for items in grouped.values())
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows))
    axes = np.atleast_2d(axes)

    for row_idx, (map_name, map_windows) in enumerate(grouped.items()):
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(map_windows):
                ax.axis("off")
                continue

            meta = map_windows[col_idx]
            start = int(meta["timestep_start"])
            end = int(meta["timestep_end"])
            replay = export_payload["pairs"][map_name]["truck_context_replay"]
            truck_branch = replay["truck_branch"]
            car_branch = replay["car_branch"]
            min_gap_steps = int(pref_settings["min_gap_steps"])
            max_start_distance_m = float(pref_settings["max_start_distance_m"])

            truck_rollout_x = np.asarray(truck_branch["rollout_x"], dtype=np.float32)
            truck_rollout_y = np.asarray(truck_branch["rollout_y"], dtype=np.float32)
            car_rollout_x = np.asarray(car_branch["rollout_x"], dtype=np.float32)
            car_rollout_y = np.asarray(car_branch["rollout_y"], dtype=np.float32)
            truck_x = truck_rollout_x[start : end + 1]
            truck_y = truck_rollout_y[start : end + 1]
            car_x = car_rollout_x[start : end + 1]
            car_y = car_rollout_y[start : end + 1]
            truck_heading = np.asarray(truck_branch["rollout_heading"], dtype=np.float32)[start : end + 1]
            obs_window = np.asarray(truck_branch["obs"], dtype=np.float32)[start:end]
            ref_idx = min(len(obs_window) // 2, max(0, len(truck_x) - 1))
            anchor_x = float(truck_x[ref_idx])
            anchor_y = float(truck_y[ref_idx])
            anchor_heading = float(truck_heading[ref_idx])
            truck_local_x, truck_local_y = _to_local_frame(truck_x, truck_y, anchor_x, anchor_y, anchor_heading)
            car_local_x, car_local_y = _to_local_frame(car_x, car_y, anchor_x, anchor_y, anchor_heading)
            if len(obs_window) > 0:
                decoded = _decode_local_obs(obs_window[len(obs_window) // 2])
                for road in decoded["roads"]:
                    half_len = 0.5 * road["length"]
                    ax.plot(
                        [road["x"] - half_len * road["heading_x"], road["x"] + half_len * road["heading_x"]],
                        [road["y"] - half_len * road["heading_y"], road["y"] + half_len * road["heading_y"]],
                        color="tab:cyan",
                        linewidth=0.9,
                        alpha=0.35,
                        zorder=0,
                    )

            ax.plot(truck_local_x, truck_local_y, color="tab:green", linewidth=2.0, label="Preferred truck branch")
            ax.plot(car_local_x, car_local_y, color="tab:red", linewidth=2.0, linestyle="--", label="Rejected car branch")
            ax.scatter(truck_local_x[0], truck_local_y[0], color="tab:green", s=20)
            ax.scatter(car_local_x[0], car_local_y[0], color="tab:red", s=20)

            failing_label = "next_fail=none"
            failing_start = start + min_gap_steps
            search_end = min(len(truck_branch["obs"]), len(car_branch["obs"])) - (end - start)
            if failing_start <= search_end:
                failing_distance = _start_distance(
                    failing_start,
                    truck_rollout_x,
                    truck_rollout_y,
                    car_rollout_x,
                    car_rollout_y,
                )
                if failing_distance > max_start_distance_m:
                    failing_end = failing_start + (end - start)
                    fail_truck_x = truck_rollout_x[failing_start : failing_end + 1]
                    fail_truck_y = truck_rollout_y[failing_start : failing_end + 1]
                    fail_car_x = car_rollout_x[failing_start : failing_end + 1]
                    fail_car_y = car_rollout_y[failing_start : failing_end + 1]
                    fail_truck_local_x, fail_truck_local_y = _to_local_frame(
                        fail_truck_x, fail_truck_y, anchor_x, anchor_y, anchor_heading
                    )
                    fail_car_local_x, fail_car_local_y = _to_local_frame(
                        fail_car_x, fail_car_y, anchor_x, anchor_y, anchor_heading
                    )
                    ax.plot(
                        fail_truck_local_x,
                        fail_truck_local_y,
                        color="tab:olive",
                        linewidth=1.4,
                        linestyle=":",
                        alpha=0.9,
                        label="Next failing truck branch" if row_idx == 0 and col_idx == 0 else None,
                    )
                    ax.plot(
                        fail_car_local_x,
                        fail_car_local_y,
                        color="tab:purple",
                        linewidth=1.4,
                        linestyle=":",
                        alpha=0.9,
                        label="Next failing car branch" if row_idx == 0 and col_idx == 0 else None,
                    )
                    ax.scatter(fail_truck_local_x[0], fail_truck_local_y[0], color="tab:olive", s=16, alpha=0.9)
                    ax.scatter(fail_car_local_x[0], fail_car_local_y[0], color="tab:purple", s=16, alpha=0.9)
                    failing_label = f"next_fail@{failing_start} d0={failing_distance:.2f}m"

            ax.scatter([0.0], [0.0], color="black", s=18, zorder=3)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.set_title(
                f"{map_name} [{start}:{end}] truck-local frame\n"
                f"window_ade={meta['window_pair_ade']:.2f} window_fde={meta['window_pair_fde']:.2f}\n"
                f"truck_self={meta['truck_self_ade']:.2f} car_on_truck={meta['car_on_truck_ade']:.2f}\n"
                f"{failing_label}",
                fontsize=9,
            )
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize cached truck-context preference window pairs.")
    parser.add_argument("--export", type=Path, default=EXPORT_DEFAULT)
    parser.add_argument("--preferences", type=Path, default=PREFERENCE_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--max-pairs", type=int, default=12)
    args = parser.parse_args()

    output_path = visualize_truck_context_preferences(
        export_path=args.export,
        preference_path=args.preferences,
        output_path=args.output,
        max_pairs=args.max_pairs,
    )
    print(output_path)


if __name__ == "__main__":
    main()
