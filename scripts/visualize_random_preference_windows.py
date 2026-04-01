from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.animate_truck_context_observations import _decode_local_obs
from scripts.build_truck_context_preferences import load_preference_manifest
from scripts.export_paired_offline_fits import iter_paired_fit_shards, load_paired_fit_manifest


EXPORT_DEFAULT = Path("outputs/offline_fits/nuplan_boston_test10_paired_fits.pt")
PREFERENCE_DEFAULT = Path("outputs/preferences/nuplan_boston_test10_truck_context_preferences.pt")
OUTPUT_DEFAULT = Path("outputs/test_visualizations/truck_context_preference_random_grid.png")


def _to_local_frame(x: np.ndarray, y: np.ndarray, anchor_x: float, anchor_y: float, anchor_heading: float):
    dx = x - anchor_x
    dy = y - anchor_y
    cos_h = np.cos(anchor_heading)
    sin_h = np.sin(anchor_heading)
    local_x = dx * cos_h + dy * sin_h
    local_y = -dx * sin_h + dy * cos_h
    return local_x, local_y


def _window_count(manifest: dict) -> int:
    metadata_count = manifest.get("metadata", {}).get("total_windows")
    if metadata_count is not None:
        return int(metadata_count)
    return int(sum(int(shard.get("window_count", 0)) for shard in manifest.get("shards", [])))


def _resolve_moved_shard_path(base_artifact_path: Path, shard_path_str: str) -> Path:
    shard_path = Path(shard_path_str)
    if shard_path.exists():
        return shard_path
    sibling_candidate = base_artifact_path.parent / f"{base_artifact_path.stem}_shards" / shard_path.name
    if sibling_candidate.exists():
        return sibling_candidate
    raise FileNotFoundError(f"Could not resolve shard path: {shard_path}")


def _sample_window_metadata(preference_path: Path, sample_count: int, seed: int) -> list[dict]:
    manifest = load_preference_manifest(preference_path)
    total_windows = _window_count(manifest)
    if total_windows <= 0:
        raise ValueError(f"No preference windows available in {preference_path}")

    actual_count = min(int(sample_count), total_windows)
    rng = random.Random(seed)
    target_indices = sorted(rng.sample(range(total_windows), actual_count))

    sampled: list[dict] = []
    remaining = list(target_indices)
    for shard_info in manifest.get("shards", []):
        if not remaining:
            break
        shard_start = int(shard_info.get("start_index", 0))
        shard_end = int(shard_info.get("end_index", shard_start + int(shard_info.get("window_count", 0))))
        local_targets = [idx for idx in remaining if shard_start <= idx < shard_end]
        if not local_targets:
            continue

        shard_payload = torch.load(_resolve_moved_shard_path(preference_path, shard_info["path"]), map_location="cpu")
        shard_meta = list(shard_payload["window_metadata"])
        for global_idx in local_targets:
            sampled.append(dict(shard_meta[global_idx - shard_start]))
        remaining = [idx for idx in remaining if idx not in local_targets]

    if len(sampled) != actual_count:
        raise RuntimeError(f"Sampled {len(sampled)} windows, expected {actual_count}")
    return sampled


def _load_selected_pairs(export_path: Path, map_names: set[str]) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    pending = set(map_names)
    manifest = load_paired_fit_manifest(export_path)

    def _estimated_shard_index(map_name: str) -> int | None:
        stem = Path(map_name).stem
        if not stem.startswith("map_"):
            return None
        suffix = stem.split("_", 1)[1]
        if not suffix.isdigit():
            return None
        return int(suffix) // 64 + 1

    # Fast path: map names in these exports follow the shard chunking order.
    for map_name in list(pending):
        shard_index = _estimated_shard_index(map_name)
        if shard_index is None:
            continue
        shard_list = manifest.get("shards", [])
        if not (1 <= shard_index <= len(shard_list)):
            continue
        shard_info = shard_list[shard_index - 1]
        shard_payload = torch.load(_resolve_moved_shard_path(export_path, shard_info["path"]), map_location="cpu")
        pairs = shard_payload["pairs"]
        if map_name in pairs:
            selected[map_name] = pairs[map_name]
            pending.remove(map_name)

    if not pending:
        return selected

    try:
        for _fit_metadata, pairs, _shard_info in iter_paired_fit_shards(export_path):
            for map_name in list(pending):
                if map_name in pairs:
                    selected[map_name] = pairs[map_name]
                    pending.remove(map_name)
            if not pending:
                break
    except FileNotFoundError:
        for shard_info in manifest.get("shards", []):
            shard_payload = torch.load(_resolve_moved_shard_path(export_path, shard_info["path"]), map_location="cpu")
            pairs = shard_payload["pairs"]
            for map_name in list(pending):
                if map_name in pairs:
                    selected[map_name] = pairs[map_name]
                    pending.remove(map_name)
            if not pending:
                break
    if pending:
        raise KeyError(f"Missing selected maps in fit export: {sorted(pending)}")
    return selected


def visualize_random_preference_windows(
    export_path: Path,
    preference_path: Path,
    output_path: Path,
    *,
    sample_count: int = 10,
    seed: int = 42,
) -> Path:
    sampled_meta = _sample_window_metadata(preference_path, sample_count=sample_count, seed=seed)
    selected_pairs = _load_selected_pairs(export_path, {meta["map_name"] for meta in sampled_meta})

    cols = 2
    rows = int(np.ceil(len(sampled_meta) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, max(4.5, 4.5 * rows)))
    axes = np.asarray(axes).reshape(rows, cols)

    for plot_idx, meta in enumerate(sampled_meta):
        row_idx = plot_idx // cols
        col_idx = plot_idx % cols
        ax = axes[row_idx, col_idx]

        map_name = meta["map_name"]
        start = int(meta["timestep_start"])
        end = int(meta["timestep_end"])
        pair = selected_pairs[map_name]
        replay = pair["truck_context_replay"]
        if replay.get("status") != "ok":
            ax.text(0.5, 0.5, f"{map_name}\nreplay unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue

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

        ax.plot(truck_local_x, truck_local_y, color="tab:green", linewidth=2.0, label="Preferred truck branch")
        ax.plot(car_local_x, car_local_y, color="tab:red", linewidth=2.0, linestyle="--", label="Rejected car branch")
        ax.scatter(truck_local_x[0], truck_local_y[0], color="tab:green", s=20)
        ax.scatter(car_local_x[0], car_local_y[0], color="tab:red", s=20)
        ax.scatter([0.0], [0.0], color="black", s=18, zorder=3)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"{map_name} [{start}:{end}] sample={plot_idx + 1}\n"
            f"window_ade={meta['window_pair_ade']:.2f} window_fde={meta['window_pair_fde']:.2f}\n"
            f"truck_self={meta['truck_self_ade']:.2f} car_on_truck={meta['car_on_truck_ade']:.2f}",
            fontsize=9,
        )
        if plot_idx == 0:
            ax.legend(fontsize=8)

    for plot_idx in range(len(sampled_meta), rows * cols):
        row_idx = plot_idx // cols
        col_idx = plot_idx % cols
        axes[row_idx, col_idx].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize random cached truck-context preference windows.")
    parser.add_argument("--export", type=Path, default=EXPORT_DEFAULT)
    parser.add_argument("--preferences", type=Path, default=PREFERENCE_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = visualize_random_preference_windows(
        export_path=args.export,
        preference_path=args.preferences,
        output_path=args.output,
        sample_count=max(1, int(args.sample_count)),
        seed=int(args.seed),
    )
    print(output_path)


if __name__ == "__main__":
    main()
