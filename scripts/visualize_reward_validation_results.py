from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.animate_truck_context_observations import _decode_local_obs
from scripts.build_truck_context_preferences import load_preference_manifest, iter_preference_shards
from scripts.export_paired_offline_fits import load_paired_fit_manifest

EXPORT_DEFAULT = Path("outputs/offline_fits/nuplan_boston_test10_paired_fits.pt")
PREFERENCE_DEFAULT = Path("outputs/preferences/nuplan_boston_test10_truck_context_preferences.pt")
EVAL_DEFAULT = Path("outputs/reward_model/default_holdout_run/offline_truck_context_reward_eval.json")
OUTPUT_DEFAULT = Path("outputs/test_visualizations/reward_validation_grid.png")
DEFAULT_MAX_SAMPLES = 10


def _to_local_frame(x: np.ndarray, y: np.ndarray, anchor_x: float, anchor_y: float, anchor_heading: float):
    dx = x - anchor_x
    dy = y - anchor_y
    cos_h = np.cos(anchor_heading)
    sin_h = np.sin(anchor_heading)
    local_x = dx * cos_h + dy * sin_h
    local_y = -dx * sin_h + dy * cos_h
    return local_x, local_y


def _load_needed_pairs(export_path: Path, needed_maps: set[str]) -> dict[str, dict]:
    manifest = load_paired_fit_manifest(export_path)
    ordered_maps = list(manifest.get("metadata", {}).get("shared_maps", []))
    map_to_index = {name: idx for idx, name in enumerate(ordered_maps)}
    needed_indices = {map_to_index[name] for name in needed_maps if name in map_to_index}
    needed_shards: list[dict] = []
    for shard_info in manifest.get("shards", []):
        start = int(shard_info.get("start_index", 0))
        end = int(shard_info.get("end_index", 0))
        if any(start <= idx < end for idx in needed_indices):
            needed_shards.append(shard_info)

    pairs: dict[str, dict] = {}
    for shard_info in needed_shards:
        shard_path = Path(shard_info["path"])
        if not shard_path.exists():
            candidate_same_dir = export_path.parent / shard_path.name
            candidate_sibling_dir = export_path.parent / f"{export_path.stem}_shards" / shard_path.name
            if candidate_same_dir.exists():
                shard_path = candidate_same_dir
            elif candidate_sibling_dir.exists():
                shard_path = candidate_sibling_dir
        shard_payload = torch.load(shard_path, map_location="cpu")
        shard_pairs = shard_payload["pairs"]
        for map_name in needed_maps:
            if map_name in shard_pairs and map_name not in pairs:
                pairs[map_name] = shard_pairs[map_name]
        if len(pairs) == len(needed_maps):
            break
    missing = sorted(needed_maps - set(pairs))
    if missing:
        raise KeyError(f"missing maps in export payload: {missing[:5]}")
    return pairs


def _entries_from_eval_examples(eval_payload: dict, max_samples: int | None = None) -> list[tuple[int, dict, dict]]:
    by_bucket: dict[str, list[tuple[int, dict, dict]]] = {}
    for bucket in ("incorrect_examples", "correct_examples"):
        bucket_entries = []
        for item in eval_payload.get("examples", {}).get(bucket, []):
            meta = dict(item.get("window", {}))
            bucket_entries.append((int(item["index"]), meta, item))
        by_bucket[bucket] = bucket_entries

    if max_samples is None or max_samples <= 0:
        return by_bucket["incorrect_examples"] + by_bucket["correct_examples"]

    target_incorrect = min(len(by_bucket["incorrect_examples"]), max(1, max_samples // 2))
    target_correct = max_samples - target_incorrect

    # If there are fewer correct examples than requested, backfill with incorrect ones.
    if len(by_bucket["correct_examples"]) < target_correct:
        deficit = target_correct - len(by_bucket["correct_examples"])
        target_correct = len(by_bucket["correct_examples"])
        target_incorrect = min(len(by_bucket["incorrect_examples"]), target_incorrect + deficit)

    def _prioritize(bucket_entries: list[tuple[int, dict, dict]]) -> list[tuple[int, dict, dict]]:
        grouped: dict[str, list[tuple[int, dict, dict]]] = OrderedDict()
        for entry in bucket_entries:
            grouped.setdefault(entry[1]["map_name"], []).append(entry)
        prioritized: list[tuple[int, dict, dict]] = []
        for _map_name, map_entries in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
            prioritized.extend(map_entries)
        return prioritized

    selected = _prioritize(by_bucket["incorrect_examples"])[:target_incorrect]
    selected.extend(_prioritize(by_bucket["correct_examples"])[:target_correct])
    return selected[:max_samples]


def _entries_from_preference_payload(preference_path: Path, eval_payload: dict) -> list[tuple[int, dict, dict | None]]:
    val_indices = list(eval_payload.get("validation_indices", []))
    if not val_indices:
        raise ValueError(f"No validation indices found in {eval_payload}")

    by_global_index = {}
    for item in eval_payload.get("examples", {}).get("correct_examples", []):
        by_global_index[int(val_indices[int(item["index"])])] = item
    for item in eval_payload.get("examples", {}).get("incorrect_examples", []):
        by_global_index[int(val_indices[int(item["index"])])] = item

    manifest = load_preference_manifest(preference_path)
    needed = set(int(i) for i in val_indices)
    window_meta_by_index: dict[int, dict] = {}
    for shard_payload, shard_info in iter_preference_shards(preference_path):
        start = int(shard_info["start_index"])
        metas = list(shard_payload.get("window_metadata", []))
        for local_idx, meta in enumerate(metas):
            global_idx = start + local_idx
            if global_idx in needed:
                window_meta_by_index[global_idx] = meta
        if len(window_meta_by_index) == len(needed):
            break

    entries = []
    for global_index in val_indices:
        entries.append((int(global_index), window_meta_by_index[int(global_index)], by_global_index.get(int(global_index))))
    return entries


def visualize_reward_validation_results(
    export_path: Path,
    preference_path: Path,
    evaluation_path: Path,
    output_path: Path,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> Path:
    eval_payload = json.loads(evaluation_path.read_text())
    entries = _entries_from_eval_examples(eval_payload, max_samples=max_samples)
    if not entries:
        entries = _entries_from_preference_payload(preference_path, eval_payload)
        if max_samples > 0:
            entries = entries[:max_samples]
    if not entries:
        raise ValueError(f"No validation examples available in {evaluation_path}")

    needed_maps = {meta["map_name"] for _, meta, _ in entries}
    export_pairs = _load_needed_pairs(export_path, needed_maps)

    panel_count = len(entries)
    cols = min(3, panel_count)
    rows = ceil(panel_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.8 * rows))
    axes = np.asarray(axes, dtype=object).reshape(rows, cols)

    for panel_idx, (global_index, meta, eval_item) in enumerate(entries):
        row_idx = panel_idx // cols
        col_idx = panel_idx % cols
        ax = axes[row_idx, col_idx]
        map_name = meta["map_name"]
        start = int(meta["timestep_start"])
        end = int(meta["timestep_end"])
        replay = export_pairs[map_name]["truck_context_replay"]
        truck_branch = replay["truck_branch"]
        car_branch = replay["car_branch"]

        truck_rollout_x = np.asarray(truck_branch["rollout_x"], dtype=np.float32)
        truck_rollout_y = np.asarray(truck_branch["rollout_y"], dtype=np.float32)
        car_rollout_x = np.asarray(car_branch["rollout_x"], dtype=np.float32)
        car_rollout_y = np.asarray(car_branch["rollout_y"], dtype=np.float32)
        truck_heading = np.asarray(truck_branch["rollout_heading"], dtype=np.float32)[start : end + 1]

        truck_x = truck_rollout_x[start : end + 1]
        truck_y = truck_rollout_y[start : end + 1]
        car_x = car_rollout_x[start : end + 1]
        car_y = car_rollout_y[start : end + 1]
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

        correct = True if eval_item is None else bool(eval_item["correct"])
        border_color = "tab:green" if correct else "tab:red"
        prob = float("nan") if eval_item is None else float(eval_item["prob_preferred_first"])
        conf = float("nan") if eval_item is None else float(eval_item["confidence"])

        ax.plot(truck_local_x, truck_local_y, color="tab:green", linewidth=2.0, label="Preferred truck branch")
        ax.plot(car_local_x, car_local_y, color="tab:red", linewidth=2.0, linestyle="--", label="Rejected car branch")
        ax.scatter(truck_local_x[0], truck_local_y[0], color="tab:green", s=20)
        ax.scatter(car_local_x[0], car_local_y[0], color="tab:red", s=20)
        ax.scatter([0.0], [0.0], color="black", s=18, zorder=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"{map_name} idx={global_index} [{start}:{end}]\n"
            f"correct={correct} prob_pref_first={prob:.3f} conf={conf:.3f}\n"
            f"window_ade={meta['window_pair_ade']:.2f} start_d={meta['window_start_distance_m']:.2f}m",
            fontsize=9,
        )
        if panel_idx == 0:
            ax.legend(fontsize=8)

    for panel_idx in range(panel_count, rows * cols):
        row_idx = panel_idx // cols
        col_idx = panel_idx % cols
        fig.delaxes(axes[row_idx, col_idx])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize held-out reward-model validation results.")
    parser.add_argument("--export", type=Path, default=EXPORT_DEFAULT)
    parser.add_argument("--preferences", type=Path, default=PREFERENCE_DEFAULT)
    parser.add_argument("--evaluation", type=Path, default=EVAL_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    args = parser.parse_args()

    output_path = visualize_reward_validation_results(
        export_path=args.export,
        preference_path=args.preferences,
        evaluation_path=args.evaluation,
        output_path=args.output,
        max_samples=args.max_samples,
    )
    print(output_path)


if __name__ == "__main__":
    main()
