#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pufferlib.ocean.drive import drive as drive_module
from pufferlib.ocean.drive.drive import binding


DEFAULT_FIT_EXPORT = (
    "/home/casko/phd-code/pufferdrive-kth/pufferlib/resources/drive/preferences/offline_fits/"
    "nuplan_boston_all_chunk64_paired_offline_fits.pt"
)
DEFAULT_SOURCE_MAP_DIR = "/home/casko/phd-code/pufferdrive-kth/datasets/nuplanCarBostonAll_training"
BASE_EGO = binding.EGO_FEATURES_CLASSIC
PARTNER_FEATURES = binding.PARTNER_FEATURES
PARTNER_COUNT = binding.MAX_AGENTS - 1
ROAD_FEATURES = binding.ROAD_FEATURES
ROAD_COUNT = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
MAX_VEH_LEN = 30.0
MAX_VEH_WIDTH = 15.0
MAX_ROAD_SEGMENT_LENGTH = 100.0


def _load_first_matching_pair(fit_export: str, source_map_dir: str, fit_side: str):
    records = drive_module._load_paired_fit_source_records(source_map_dir)
    manifest = drive_module._load_paired_fit_manifest(fit_export)
    records = drive_module._filter_records_to_paired_fit_manifest(records, manifest, context="BC source split")
    if not records:
        raise ValueError("No source records matched the paired offline-fit manifest")

    record = records[0]
    shard_plans = drive_module._build_paired_fit_shard_plans(fit_export, manifest, [record])
    if not shard_plans:
        raise ValueError("Could not resolve a shard for the first source record")
    shard_payload = torch.load(shard_plans[0]["path"], map_location="cpu")
    pair = shard_payload["pairs"][record.fit_map_name]
    return record, pair


def _segment_boundaries(obs_dim: int):
    ego = 7
    partners = 31 * 7
    roads = 128 * 7
    if ego + partners + roads != obs_dim:
        raise ValueError(f"Unexpected observation width {obs_dim}; expected default-mode 1120")
    return {
        "ego": (0, ego),
        "partners": (ego, ego + partners),
        "roads": (ego + partners, ego + partners + roads),
    }


def _plot_samples(samples, output_path: Path, *, stack_len: int, fit_side: str):
    obs_dim = int(samples[0]["obs_dim"])
    boundaries = _segment_boundaries(obs_dim)
    frame_labels = [f"t-{idx}" if idx > 0 else "t" for idx in range(stack_len)]

    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 4.5 * len(samples)), constrained_layout=True)
    if len(samples) == 1:
        axes = np.asarray([axes])

    for row_idx, sample in enumerate(samples):
        stacked = sample["stacked_obs"]
        raw = stacked.reshape(stack_len, obs_dim)

        ax_heat = axes[row_idx, 0]
        image = ax_heat.imshow(raw, aspect="auto", cmap="coolwarm")
        ax_heat.set_title(
            f"Sample {row_idx + 1}: joint={sample['joint_action']} accel={sample['accel_idx']} steer={sample['steer_idx']}"
        )
        ax_heat.set_ylabel("Stack frame")
        ax_heat.set_yticks(np.arange(stack_len))
        ax_heat.set_yticklabels(frame_labels)
        for name, (_start, end) in boundaries.items():
            ax_heat.axvline(end - 0.5, color="white", linewidth=0.8, alpha=0.8)
        ax_heat.text(3, -0.8, "ego", fontsize=9)
        ax_heat.text(boundaries["partners"][0] + 3, -0.8, "partners", fontsize=9)
        ax_heat.text(boundaries["roads"][0] + 3, -0.8, "roads", fontsize=9)
        plt.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)

        ax_mean = axes[row_idx, 1]
        for frame_idx in range(stack_len):
            ax_mean.plot(raw[frame_idx], linewidth=0.8, label=frame_labels[frame_idx])
        ax_mean.set_title("Flattened feature traces by stacked frame")
        ax_mean.set_xlabel("Feature index")
        ax_mean.set_ylabel("Value")
        for _name, (_start, end) in boundaries.items():
            ax_mean.axvline(end - 0.5, color="0.75", linewidth=0.8)
        ax_mean.legend(loc="upper right", fontsize=8)

        ax_stats = axes[row_idx, 2]
        block_names = []
        block_norms = []
        block_deltas = []
        current = raw[0]
        previous = raw[1] if stack_len > 1 else raw[0]
        for name, (start, end) in boundaries.items():
            block_names.append(name)
            block_norms.append(float(np.linalg.norm(current[start:end])))
            block_deltas.append(float(np.linalg.norm(current[start:end] - previous[start:end])))
        x = np.arange(len(block_names))
        width = 0.38
        ax_stats.bar(x - width / 2, block_norms, width=width, label="||frame t||")
        ax_stats.bar(x + width / 2, block_deltas, width=width, label="||t - t-1||")
        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels(block_names)
        ax_stats.set_title("Block norms and frame-to-frame change")
        ax_stats.legend(fontsize=8)

    fig.suptitle(
        f"Stacked BC input sanity ({fit_side})\nFrames are ordered current -> older history inside the flat tensor",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _decode_local_obs(obs_vec: np.ndarray) -> dict:
    goal = (float(obs_vec[0] * 200.0), float(obs_vec[1] * 200.0))
    partners = []
    partner_start = BASE_EGO
    for idx in range(PARTNER_COUNT):
        base = partner_start + idx * PARTNER_FEATURES
        rel_x = float(obs_vec[base] * 50.0)
        rel_y = float(obs_vec[base + 1] * 50.0)
        width = float(obs_vec[base + 2] * MAX_VEH_WIDTH)
        length = float(obs_vec[base + 3] * MAX_VEH_LEN)
        heading_x = float(obs_vec[base + 4])
        heading_y = float(obs_vec[base + 5])
        if rel_x == 0.0 and rel_y == 0.0 and width == 0.0 and length == 0.0:
            continue
        partners.append(
            {
                "slot": idx,
                "x": rel_x,
                "y": rel_y,
                "width": width,
                "length": length,
                "heading_x": heading_x,
                "heading_y": heading_y,
            }
        )

    roads = []
    road_start = partner_start + PARTNER_COUNT * PARTNER_FEATURES
    for idx in range(ROAD_COUNT):
        base = road_start + idx * ROAD_FEATURES
        rel_x = float(obs_vec[base] * 50.0)
        rel_y = float(obs_vec[base + 1] * 50.0)
        seg_len = float(obs_vec[base + 2] * MAX_ROAD_SEGMENT_LENGTH)
        heading_x = float(obs_vec[base + 4])
        heading_y = float(obs_vec[base + 5])
        if rel_x == 0.0 and rel_y == 0.0 and seg_len == 0.0:
            continue
        roads.append(
            {
                "x": rel_x,
                "y": rel_y,
                "length": seg_len,
                "heading_x": heading_x,
                "heading_y": heading_y,
            }
        )
    return {"goal": goal, "partners": partners, "roads": roads}


def _plot_geometry_samples(samples, output_path: Path, *, stack_len: int, fit_side: str):
    columns = min(3, len(samples))
    rows = int(np.ceil(len(samples) / max(1, columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 5.5 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, columns)
    frame_colors = plt.cm.viridis(np.linspace(0.15, 0.95, stack_len))

    for ax in axes.flat[len(samples) :]:
        ax.axis("off")

    for sample_idx, sample in enumerate(samples):
        ax = axes.flat[sample_idx]
        raw = sample["stacked_obs"].reshape(stack_len, sample["obs_dim"])
        decoded_frames = [_decode_local_obs(raw[frame_idx]) for frame_idx in range(stack_len)]

        current_roads = decoded_frames[0]["roads"]
        for road in current_roads:
            half_len = 0.5 * road["length"]
            ax.plot(
                [road["x"] - half_len * road["heading_x"], road["x"] + half_len * road["heading_x"]],
                [road["y"] - half_len * road["heading_y"], road["y"] + half_len * road["heading_y"]],
                color="0.7",
                linewidth=0.9,
                alpha=0.9,
                zorder=0,
            )

        goal_points = np.asarray([frame["goal"] for frame in decoded_frames], dtype=np.float32)
        ax.plot(goal_points[:, 0], goal_points[:, 1], color="tab:green", linewidth=1.8, alpha=0.85, zorder=2)
        ax.scatter(goal_points[:, 0], goal_points[:, 1], c=frame_colors, s=22, zorder=3)

        partner_by_slot = {}
        for frame_idx, frame in enumerate(decoded_frames):
            for partner in frame["partners"]:
                partner_by_slot.setdefault(int(partner["slot"]), []).append((frame_idx, partner))
        for slot, entries in partner_by_slot.items():
            if len(entries) < 2:
                continue
            xs = [partner["x"] for _, partner in entries]
            ys = [partner["y"] for _, partner in entries]
            ax.plot(xs, ys, linewidth=1.2, alpha=0.45, color="tab:orange", zorder=1)
            for frame_idx, partner in entries:
                ax.scatter(
                    [partner["x"]],
                    [partner["y"]],
                    c=[frame_colors[frame_idx]],
                    s=10,
                    alpha=0.8,
                    zorder=2,
                )

        ax.scatter([0.0], [0.0], color="black", s=36, marker="x", zorder=4)
        ax.set_title(
            f"sample={sample['sample_index']} t={sample['logged_timestep']} "
            f"a=({sample['accel_idx']},{sample['steer_idx']})"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-80.0, 80.0)
        ax.set_ylim(-80.0, 80.0)
        ax.grid(True, alpha=0.18)
        ax.set_xlabel("local x [m]")
        ax.set_ylabel("local y [m]")

    fig.suptitle(
        f"Stacked BC local-scene trajectory sanity ({fit_side})\n"
        "Goal and partner slots over stacked frames, current frame in brighter colors",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot stacked BC input tensors for sanity checks.")
    parser.add_argument("--fit-export", default=DEFAULT_FIT_EXPORT)
    parser.add_argument("--source-map-dir", default=DEFAULT_SOURCE_MAP_DIR)
    parser.add_argument("--fit-side", default="car", choices=("car", "truck"))
    parser.add_argument("--obs-key", default="logged_obs_default")
    parser.add_argument("--stack-len", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="/home/casko/phd-code/PufferDrive/outputs/bc_input_sanity/stacked_iid",
    )
    args = parser.parse_args()

    record, pair = _load_first_matching_pair(args.fit_export, args.source_map_dir, args.fit_side)
    obs, actions, timesteps = drive_module._validate_paired_fit_side_with_timestep(
        pair,
        fit_map_name=record.fit_map_name,
        fit_side=args.fit_side,
        obs_key=args.obs_key,
        obs_dim=int(pair[args.fit_side][args.obs_key].shape[1]),
        action_space_size=91,
    )

    dataset = drive_module._PairedOfflineFitStackedIIDDataset(
        args.fit_export,
        [record],
        int(obs.shape[1]),
        91,
        stack_len=args.stack_len,
        fit_side=args.fit_side,
        obs_key=args.obs_key,
        extend_classic_action_space=False,
        shuffle=False,
        seed=0,
        shard_shuffle_buffer=1,
    )
    dataset_samples = list(dataset)
    if not dataset_samples:
        raise ValueError("No stacked samples were produced")

    selected = []
    selected_rows = list(range(min(args.num_samples, len(dataset_samples))))
    for sample_idx in selected_rows:
        stacked_obs, accel_idx, steer_idx = dataset_samples[sample_idx]
        row_idx = args.stack_len - 1 + sample_idx
        selected.append(
            {
                "sample_index": int(sample_idx),
                "source_map_name": record.source_map_name,
                "fit_map_name": record.fit_map_name,
                "row_idx": int(row_idx),
                "logged_timestep": int(timesteps[row_idx].item()),
                "joint_action": int(actions[row_idx].item()),
                "accel_idx": int(accel_idx.item()),
                "steer_idx": int(steer_idx.item()),
                "acceleration_values": list(drive_module._CLASSIC_ACCELERATION_VALUES_LEGACY),
                "steering_values": list(drive_module._CLASSIC_STEERING_VALUES),
                "obs_dim": int(obs.shape[1]),
                "stack_len": int(args.stack_len),
                "stacked_obs": stacked_obs.numpy(),
            }
        )

    output_dir = Path(args.output_dir)
    png_path = output_dir / f"stacked_iid_input_sanity_{args.fit_side}.png"
    traj_path = output_dir / f"stacked_iid_input_trajectory_sanity_{args.fit_side}.png"
    json_path = output_dir / f"stacked_iid_input_sanity_{args.fit_side}.json"

    _plot_samples(selected, png_path, stack_len=args.stack_len, fit_side=args.fit_side)
    _plot_geometry_samples(selected, traj_path, stack_len=args.stack_len, fit_side=args.fit_side)

    json_payload = {
        "fit_side": args.fit_side,
        "source_map_name": record.source_map_name,
        "fit_map_name": record.fit_map_name,
        "obs_key": args.obs_key,
        "obs_dim": int(obs.shape[1]),
        "stack_len": int(args.stack_len),
        "frame_order": "current_to_past",
        "samples": [
            {
                key: value
                for key, value in sample.items()
                if key != "stacked_obs"
            }
            for sample in selected
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print(f"saved_plot={png_path}")
    print(f"saved_trajectory_plot={traj_path}")
    print(f"saved_json={json_path}")


if __name__ == "__main__":
    main()
