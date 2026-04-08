from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
import sys

import gymnasium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib.preference_reward import PreferenceRewardMetadata, build_state_action_features
from preferences import reward_model as reward_model_module
from scripts.animate_truck_context_observations import _decode_local_obs
from scripts.build_truck_context_preferences import iter_preference_shards
from scripts.export_paired_offline_fits import iter_paired_fit_shards

ACCELERATION_VALUES = np.asarray([-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0], dtype=np.float32)
STEERING_VALUES = np.asarray(
    [-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0], dtype=np.float32
)

DEFAULT_PREFERENCE_PATH = Path(
    "pufferlib/resources/drive/preferences/turning_preferences/nuplan_boston_all_chunk64_turning_preferences.pt"
)
DEFAULT_REWARD_DIR = Path("outputs/reward_model/turning_all_90_10_30rounds_continue1")
DEFAULT_OUTPUT_SUBDIR = Path("preference_reward_diagnostics/validation_action_landscapes")
EGO_BODY_LENGTH = 4.8
EGO_BODY_WIDTH = 2.0
TRAILER_BODY_LENGTH = 10.0
TRAILER_BODY_WIDTH = 2.4
TRAJECTORY_DT_SECONDS = 0.1


def _vehicle_corners(
    center_x: float,
    center_y: float,
    heading_x: float,
    heading_y: float,
    length: float,
    width: float,
) -> np.ndarray:
    heading = np.asarray([heading_x, heading_y], dtype=np.float32)
    norm = float(np.linalg.norm(heading))
    if norm < 1e-6:
        heading = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        heading = heading / norm
    lateral = np.asarray([-heading[1], heading[0]], dtype=np.float32)
    half_length = 0.5 * float(length)
    half_width = 0.5 * float(width)
    center = np.asarray([center_x, center_y], dtype=np.float32)
    return np.stack(
        [
            center + half_length * heading + half_width * lateral,
            center + half_length * heading - half_width * lateral,
            center - half_length * heading - half_width * lateral,
            center - half_length * heading + half_width * lateral,
        ],
        axis=0,
    )


def _draw_vehicle_box(
    ax,
    *,
    center_x: float,
    center_y: float,
    heading_x: float,
    heading_y: float,
    length: float,
    width: float,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    zorder: int,
):
    corners = _vehicle_corners(center_x, center_y, heading_x, heading_y, length, width)
    patch = Polygon(corners, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)
    nose = np.asarray([center_x, center_y], dtype=np.float32) + 0.5 * float(length) * np.asarray(
        [heading_x, heading_y], dtype=np.float32
    )
    ax.plot([center_x, nose[0]], [center_y, nose[1]], color=edgecolor, linewidth=1.4, zorder=zorder + 1)


def _draw_heading_triangle(
    ax,
    *,
    center_x: float,
    center_y: float,
    heading_x: float,
    heading_y: float,
    length: float,
    width: float,
    facecolor: str,
    edgecolor: str,
    zorder: int,
):
    heading = np.asarray([heading_x, heading_y], dtype=np.float32)
    norm = float(np.linalg.norm(heading))
    if norm < 1e-6:
        heading = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        heading = heading / norm
    lateral = np.asarray([-heading[1], heading[0]], dtype=np.float32)
    center = np.asarray([center_x, center_y], dtype=np.float32)
    tip = center + float(length) * heading
    base_center = center + 0.25 * float(length) * heading
    left = base_center + 0.5 * float(width) * lateral
    right = base_center - 0.5 * float(width) * lateral
    patch = Polygon(np.stack([tip, left, right], axis=0), closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.9, zorder=zorder)
    ax.add_patch(patch)


def _action_to_pair(action_idx: int) -> tuple[int, int]:
    accel_idx = int(action_idx) // len(STEERING_VALUES)
    steer_idx = int(action_idx) % len(STEERING_VALUES)
    return accel_idx, steer_idx


def _action_label(action_idx: int) -> str:
    accel_idx, steer_idx = _action_to_pair(action_idx)
    return f"a={ACCELERATION_VALUES[accel_idx]:.3g}, s={STEERING_VALUES[steer_idx]:.3g}"


def _reshape_action_values(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(len(ACCELERATION_VALUES), len(STEERING_VALUES))


def _extract_action_index(sa_step: np.ndarray, obs_dim: int) -> int:
    return int(np.argmax(np.asarray(sa_step[obs_dim:], dtype=np.float32)))


def _load_reward_model(reward_dir: Path):
    summary = json.loads((reward_dir / "offline_truck_context_reward_summary.json").read_text(encoding="utf-8"))
    metadata = PreferenceRewardMetadata(
        observation_mode=summary.get("observation_mode"),
        obs_dim=int(summary["obs_dim"]),
        action_dim=int(summary["action_dim"]),
        size_segment=int(summary["size_segment"]) if summary.get("size_segment") is not None else None,
        action_encoding=str(summary.get("action_encoding", "one_hot")),
        ensemble_size=int(summary["ensemble_size"]),
        activation=str(summary.get("activation", "tanh")),
    )
    reward_model = reward_model_module.RewardModel(
        ds=metadata.obs_dim,
        da=metadata.action_dim,
        ensemble_size=metadata.ensemble_size,
        lr=3e-4,
        mb_size=1,
        size_segment=1,
        capacity=max(2, metadata.ensemble_size),
        activation=metadata.activation,
    )
    reward_model.train_batch_size = 1
    reward_model.load(str(reward_dir), "offline_truck_context")
    return metadata, reward_model


def _score_action_grid(obs: np.ndarray, metadata: PreferenceRewardMetadata, reward_model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    repeated_obs = np.repeat(obs, metadata.action_dim, axis=0)
    actions = np.arange(metadata.action_dim, dtype=np.int64)
    features = build_state_action_features(
        repeated_obs,
        actions,
        metadata=metadata,
        action_space=gymnasium.spaces.Discrete(metadata.action_dim),
        action_type="discrete",
    )
    segment_features = np.expand_dims(features, axis=1)
    member_scores = []
    for member in range(metadata.ensemble_size):
        scores = reward_model.r_hat_member(segment_features, member=member).detach().cpu().numpy().reshape(-1)
        member_scores.append(scores.astype(np.float32))
    member_scores = np.stack(member_scores, axis=0)
    return member_scores.mean(axis=0), member_scores.std(axis=0), member_scores


def _score_sequence_conditioned_action_grid(
    preferred_sa: np.ndarray,
    *,
    metadata: PreferenceRewardMetadata,
    reward_model,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_sequence = np.asarray(preferred_sa, dtype=np.float32)
    if base_sequence.ndim != 2:
        raise ValueError(f"expected preferred_sa to have shape (T, D), got {base_sequence.shape}")
    if base_sequence.shape[1] != metadata.obs_dim + metadata.action_dim:
        raise ValueError(
            f"sequence feature dim mismatch: expected {metadata.obs_dim + metadata.action_dim}, got {base_sequence.shape[1]}"
        )

    candidate_sequences = np.repeat(base_sequence[None, :, :], metadata.action_dim, axis=0)
    candidate_sequences[:, step, metadata.obs_dim :] = 0.0
    candidate_sequences[np.arange(metadata.action_dim), step, metadata.obs_dim + np.arange(metadata.action_dim)] = 1.0

    member_totals = []
    for member in range(metadata.ensemble_size):
        step_rewards = reward_model.r_hat_member(candidate_sequences, member=member).detach().cpu().numpy().reshape(
            metadata.action_dim, base_sequence.shape[0]
        )
        member_totals.append(step_rewards[:, step].astype(np.float32))
    member_totals = np.stack(member_totals, axis=0)
    return member_totals.mean(axis=0), member_totals.std(axis=0), member_totals


def _collect_validation_windows(preference_path: Path, validation_indices: list[int]) -> dict[int, dict]:
    needed = set(int(idx) for idx in validation_indices)
    windows: dict[int, dict] = {}
    for shard_payload, shard_info in iter_preference_shards(preference_path):
        start_index = int(shard_info["start_index"])
        preferred_sa = np.asarray(shard_payload["preferred_sa"], dtype=np.float32)
        rejected_sa = np.asarray(shard_payload["rejected_sa"], dtype=np.float32)
        labels = np.asarray(shard_payload["labels"], dtype=np.float32)
        window_metadata = list(shard_payload.get("window_metadata", []))
        for local_idx in range(len(preferred_sa)):
            global_idx = start_index + local_idx
            if global_idx not in needed:
                continue
            windows[global_idx] = {
                "preferred_sa": preferred_sa[local_idx],
                "rejected_sa": rejected_sa[local_idx],
                "label": labels[local_idx],
                "window": window_metadata[local_idx] if window_metadata else {"index": global_idx},
            }
        if len(windows) == len(needed):
            break
    missing = sorted(needed - set(windows))
    if missing:
        raise KeyError(f"Missing validation windows: {missing[:10]}")
    return windows


def _load_export_pairs(export_path: Path, needed_maps: set[str]) -> dict[str, dict]:
    pairs_by_map: dict[str, dict] = {}
    for _metadata, pairs, _shard_info in iter_paired_fit_shards(export_path):
        for map_name in needed_maps:
            if map_name in pairs and map_name not in pairs_by_map:
                pairs_by_map[map_name] = pairs[map_name]
        if len(pairs_by_map) == len(needed_maps):
            break
    missing = sorted(needed_maps - set(pairs_by_map))
    if missing:
        raise KeyError(f"Missing paired offline-fit maps: {missing[:10]}")
    return pairs_by_map


def _select_examples(eval_payload: dict, max_examples: int) -> list[tuple[int, dict | None]]:
    examples = eval_payload.get("examples", {})
    incorrect = list(examples.get("incorrect_examples", []))
    correct = list(examples.get("correct_examples", []))
    selected: list[tuple[int, dict | None]] = []
    incorrect_count = min(len(incorrect), max(1, max_examples // 2))
    correct_count = min(len(correct), max_examples - incorrect_count)
    for item in incorrect[:incorrect_count]:
        selected.append((int(item["index"]), item))
    for item in correct[:correct_count]:
        selected.append((int(item["index"]), item))
    return selected[:max_examples]


def _to_local_frame(x: np.ndarray, y: np.ndarray, anchor_x: float, anchor_y: float, anchor_heading: float):
    dx = np.asarray(x, dtype=np.float32) - float(anchor_x)
    dy = np.asarray(y, dtype=np.float32) - float(anchor_y)
    cos_h = np.cos(float(anchor_heading))
    sin_h = np.sin(float(anchor_heading))
    local_x = dx * cos_h + dy * sin_h
    local_y = -dx * sin_h + dy * cos_h
    return local_x, local_y


def _trajectory_speed(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.size <= 1:
        return np.zeros_like(x, dtype=np.float32)
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx * dx + dy * dy) / TRAJECTORY_DT_SECONDS
    speed = np.concatenate([speed[:1], speed], axis=0)
    return speed.astype(np.float32)


def _plot_colored_trajectory(ax, x: np.ndarray, y: np.ndarray, speed: np.ndarray, cmap: str, *, linewidth: float, zorder: int, linestyle: str = "solid"):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    speed = np.asarray(speed, dtype=np.float32)
    if x.size < 2:
        ax.scatter(x, y, s=18, c=speed if speed.size else "#000000", cmap=cmap, zorder=zorder)
        return
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_speed = 0.5 * (speed[:-1] + speed[1:])
    collection = LineCollection(segments, cmap=cmap, linewidths=linewidth, zorder=zorder)
    if linestyle != "solid":
        collection.set_linestyle(linestyle)
    collection.set_array(segment_speed)
    ax.add_collection(collection)


def _draw_scene(ax, obs: np.ndarray, trajectory_data: dict | None = None):
    decoded = _decode_local_obs(np.asarray(obs, dtype=np.float32))
    ax.set_facecolor("#fcfcfb")
    for road in decoded["roads"]:
        half_len = 0.5 * road["length"]
        ax.plot(
            [road["x"] - half_len * road["heading_x"], road["x"] + half_len * road["heading_x"]],
            [road["y"] - half_len * road["heading_y"], road["y"] + half_len * road["heading_y"]],
            color="#d3d3cf",
            linewidth=1.0,
            alpha=0.9,
            zorder=0,
        )
    for partner in decoded["partners"]:
        _draw_vehicle_box(
            ax,
            center_x=partner["x"],
            center_y=partner["y"],
            heading_x=partner["heading_x"],
            heading_y=partner["heading_y"],
            length=max(2.5, partner["length"]),
            width=max(1.2, partner["width"]),
            facecolor="#d9d7d0",
            edgecolor="#b8b8b2",
            alpha=0.9,
            zorder=1,
        )
    goal_x, goal_y = decoded["goal"]
    trailer = decoded.get("trailer", {"x": 0.0, "y": 0.0, "heading_x": 0.0, "heading_y": 0.0})
    has_trailer_pose = any(abs(float(trailer[k])) > 1e-6 for k in ("x", "y", "heading_x", "heading_y"))

    _draw_vehicle_box(
        ax,
        center_x=0.0,
        center_y=0.0,
        heading_x=1.0,
        heading_y=0.0,
        length=EGO_BODY_LENGTH,
        width=EGO_BODY_WIDTH,
        facecolor="#69a36d",
        edgecolor="#2f7d32",
        alpha=0.95,
        zorder=3,
    )
    _draw_heading_triangle(
        ax,
        center_x=0.3,
        center_y=0.0,
        heading_x=1.0,
        heading_y=0.0,
        length=1.1,
        width=0.85,
        facecolor="#245d28",
        edgecolor="white",
        zorder=5,
    )
    if has_trailer_pose:
        _draw_vehicle_box(
            ax,
            center_x=float(trailer["x"]),
            center_y=float(trailer["y"]),
            heading_x=float(trailer["heading_x"]),
            heading_y=float(trailer["heading_y"]),
            length=TRAILER_BODY_LENGTH,
            width=TRAILER_BODY_WIDTH,
            facecolor="#9dc7a0",
            edgecolor="#2f7d32",
            alpha=0.8,
            zorder=2,
        )
        ax.plot(
            [0.0, float(trailer["x"])],
            [0.0, float(trailer["y"])],
            color="#4f6f52",
            linewidth=1.2,
            alpha=0.9,
            zorder=2,
        )
    ax.scatter([goal_x], [goal_y], s=34, color="#7b8f3b", marker="*", zorder=2)
    if trajectory_data is not None:
        pref_local_x = np.asarray(trajectory_data["preferred_local_x"], dtype=np.float32)
        pref_local_y = np.asarray(trajectory_data["preferred_local_y"], dtype=np.float32)
        rej_local_x = np.asarray(trajectory_data["rejected_local_x"], dtype=np.float32)
        rej_local_y = np.asarray(trajectory_data["rejected_local_y"], dtype=np.float32)
        gt_local_x = np.asarray(trajectory_data["gt_local_x"], dtype=np.float32)
        gt_local_y = np.asarray(trajectory_data["gt_local_y"], dtype=np.float32)
        pref_speed = np.asarray(trajectory_data["preferred_speed"], dtype=np.float32)
        rej_speed = np.asarray(trajectory_data["rejected_speed"], dtype=np.float32)
        gt_speed = np.asarray(trajectory_data["gt_speed"], dtype=np.float32)
        _plot_colored_trajectory(ax, gt_local_x, gt_local_y, gt_speed, "Blues", linewidth=1.8, zorder=1, linestyle="--")
        _plot_colored_trajectory(ax, pref_local_x, pref_local_y, pref_speed, "Oranges", linewidth=2.4, zorder=4)
        _plot_colored_trajectory(ax, rej_local_x, rej_local_y, rej_speed, "Purples", linewidth=2.4, zorder=4)
        ax.scatter(pref_local_x[0], pref_local_y[0], s=18, color="#f28e2b", zorder=5)
        ax.scatter(rej_local_x[0], rej_local_y[0], s=18, color="#8b5fbf", zorder=5)
    ax.set_xlim(-35.0, 35.0)
    ax.set_ylim(-35.0, 35.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#ecece7", alpha=0.6, linewidth=0.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Scenario Snapshot", fontsize=10)


def _annotate_action(ax, action_idx: int, marker: str, color: str, label: str):
    accel_idx, steer_idx = _action_to_pair(action_idx)
    ax.scatter(
        [steer_idx],
        [accel_idx],
        marker=marker,
        s=120,
        color=color,
        linewidths=1.5,
        edgecolors="white",
        label=label,
        zorder=3,
    )


def _plot_heatmap(ax, values: np.ndarray, title: str, preferred_idx: int, rejected_idx: int, argmax_idx: int, cmap: str):
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    _annotate_action(ax, preferred_idx, "o", "tab:green", "Preferred GT")
    _annotate_action(ax, rejected_idx, "X", "tab:red", "Rejected GT")
    _annotate_action(ax, argmax_idx, "*", "gold", "Reward argmax")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Steering bin")
    ax.set_ylabel("Acceleration bin")
    ax.set_xticks(np.arange(len(STEERING_VALUES)))
    ax.set_xticklabels([f"{value:.2g}" for value in STEERING_VALUES], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(ACCELERATION_VALUES)))
    ax.set_yticklabels([f"{value:.2g}" for value in ACCELERATION_VALUES], fontsize=8)
    return image


def _rank_desc(values: np.ndarray, action_idx: int) -> int:
    order = np.argsort(-np.asarray(values, dtype=np.float32))
    matches = np.flatnonzero(order == int(action_idx))
    return int(matches[0]) + 1


def _save_example_grid(
    output_path: Path,
    example_rows: list[dict],
):
    if not example_rows:
        return
    rows = len(example_rows)
    fig, axes = plt.subplots(rows, 3, figsize=(15.5, max(4.5, 4.7 * rows)), constrained_layout=True)
    axes = np.asarray(axes, dtype=object).reshape(rows, 3)
    for row_idx, row in enumerate(example_rows):
        _draw_scene(axes[row_idx, 0], row["obs"], trajectory_data=row.get("trajectory_data"))
        axes[row_idx, 0].set_title(f"Scene at action timestep t={row['snapshot_step']}")
        mean_im = _plot_heatmap(
            axes[row_idx, 1],
            row["mean_grid"],
            "Sequence-Conditioned Action Reward Mean\n"
            + f"replace t={row['snapshot_step']} with dataset pref={row['preferred_score']:.3f} ({row['preferred_action_label']})\n"
            + f"replace t={row['snapshot_step']} with dataset rej={row['rejected_score']:.3f} ({row['rejected_action_label']})",
            preferred_idx=row["preferred_idx"],
            rejected_idx=row["rejected_idx"],
            argmax_idx=row["argmax_idx"],
            cmap="viridis",
        )
        std_im = _plot_heatmap(
            axes[row_idx, 2],
            row["std_grid"],
            f"Sequence-Conditioned Ensemble Std\nargmax={_action_label(row['argmax_idx'])}",
            preferred_idx=row["preferred_idx"],
            rejected_idx=row["rejected_idx"],
            argmax_idx=row["argmax_idx"],
            cmap="magma",
        )
        meta = row["window"]
        eval_item = row.get("eval_item")
        prefix = (
            f"{meta['map_name']} global_idx={row['global_index']} local_val_idx={row['local_validation_index']} "
            f"correct={bool(eval_item['correct'])} conf={float(eval_item['confidence']):.3f}"
            if eval_item is not None
            else f"{meta['map_name']} global_idx={row['global_index']}"
        )
        axes[row_idx, 0].set_title(
            prefix
            + "\n"
            + f"turn={meta.get('scenario_delta_heading_deg', float('nan')):.1f}deg "
            + f"pref_rank={row['preferred_rank']} rej_rank={row['rejected_rank']}\n"
            + "Only the selected timestep action is changed; the rest of the preferred validation window stays fixed",
            fontsize=10,
        )
        if row_idx == 0:
            axes[row_idx, 1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
            handles = [
                plt.Line2D([0], [0], color="#1f3a5f", linewidth=1.4, linestyle="--", label="GT trajectory"),
                plt.Line2D([0], [0], color="#f28e2b", linewidth=2.0, label="Preferred truck"),
                plt.Line2D([0], [0], color="#8b5fbf", linewidth=2.0, label="Rejected car"),
            ]
            axes[row_idx, 0].legend(handles=handles, loc="lower left", fontsize=8)
        fig.colorbar(mean_im, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
        fig.colorbar(std_im, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_overview_figure(output_path: Path, summary: dict):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = [
        ("Mean Reward", summary["mean_reward_grid"], "viridis"),
        ("Mean Ensemble Std", summary["mean_std_grid"], "magma"),
        ("Reward Argmax Frequency", summary["argmax_freq_grid"], "Blues"),
        ("Preferred GT Frequency", summary["preferred_freq_grid"], "Greens"),
        ("Rejected GT Frequency", summary["rejected_freq_grid"], "Reds"),
        ("Argmax - Preferred Freq", summary["argmax_minus_preferred_grid"], "coolwarm"),
    ]
    for ax, (title, values, cmap) in zip(axes.flat, panels):
        image = ax.imshow(np.asarray(values, dtype=np.float32), aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Steering bin")
        ax.set_ylabel("Acceleration bin")
        ax.set_xticks(np.arange(len(STEERING_VALUES)))
        ax.set_xticklabels([f"{value:.2g}" for value in STEERING_VALUES], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(ACCELERATION_VALUES)))
        ax.set_yticklabels([f"{value:.2g}" for value in ACCELERATION_VALUES], fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze action-level behavior of a learned preference reward model.")
    parser.add_argument("--preferences", type=Path, default=DEFAULT_PREFERENCE_PATH)
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--evaluation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--scoring-mode",
        type=str,
        default="one_step",
        choices=["one_step", "sequence_conditioned"],
        help="How to score candidate actions: isolated one-step scoring or full-sequence-conditioned scoring.",
    )
    parser.add_argument(
        "--max-validation-windows",
        type=int,
        default=None,
        help="Optional cap on how many validation windows to score. Useful for faster subset regenerations.",
    )
    parser.add_argument(
        "--snapshot-step",
        type=int,
        default=-1,
        help="Timestep inside the 32-step window to replace. Use -1 for the last/current action in the window.",
    )
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()

    evaluation_path = args.evaluation if args.evaluation is not None else (args.reward_dir / "offline_truck_context_reward_eval.json")
    eval_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    validation_indices = [int(idx) for idx in eval_payload.get("validation_indices", [])]
    if not validation_indices:
        raise ValueError(f"No validation indices found in {evaluation_path}")
    if args.max_validation_windows is not None and args.max_validation_windows > 0:
        validation_indices = validation_indices[: args.max_validation_windows]

    metadata, reward_model = _load_reward_model(args.reward_dir)
    windows_by_index = _collect_validation_windows(args.preferences, validation_indices)
    preference_payload = torch.load(args.preferences, map_location="cpu")
    export_path = Path(preference_payload["metadata"]["source_export"])
    export_pairs = _load_export_pairs(export_path, {payload["window"]["map_name"] for payload in windows_by_index.values()})

    validation_rows = []
    mean_reward_sum = np.zeros((metadata.action_dim,), dtype=np.float64)
    mean_std_sum = np.zeros((metadata.action_dim,), dtype=np.float64)
    argmax_counts = np.zeros((metadata.action_dim,), dtype=np.int64)
    preferred_counts = np.zeros((metadata.action_dim,), dtype=np.int64)
    rejected_counts = np.zeros((metadata.action_dim,), dtype=np.int64)

    for local_validation_index, global_idx in enumerate(validation_indices):
        payload = windows_by_index[global_idx]
        preferred_sa = np.asarray(payload["preferred_sa"], dtype=np.float32)
        rejected_sa = np.asarray(payload["rejected_sa"], dtype=np.float32)
        if args.snapshot_step < 0:
            step = len(preferred_sa) - 1
        else:
            step = min(max(0, args.snapshot_step), len(preferred_sa) - 1)
        preferred_step = preferred_sa[step]
        rejected_step = rejected_sa[step]
        obs = np.asarray(preferred_step[: metadata.obs_dim], dtype=np.float32)
        preferred_idx = _extract_action_index(preferred_step, metadata.obs_dim)
        rejected_idx = _extract_action_index(rejected_step, metadata.obs_dim)
        if args.scoring_mode == "one_step":
            mean_scores, std_scores, _ = _score_action_grid(obs, metadata=metadata, reward_model=reward_model)
        else:
            mean_scores, std_scores, _ = _score_sequence_conditioned_action_grid(
                preferred_sa,
                metadata=metadata,
                reward_model=reward_model,
                step=step,
            )
        argmax_idx = int(np.argmax(mean_scores))
        map_name = payload["window"]["map_name"]
        pair = export_pairs[map_name]
        replay = pair["truck_context_replay"]
        truck_branch = replay["truck_branch"]
        car_branch = replay["car_branch"]
        start = int(payload["window"]["timestep_start"])
        end = int(payload["window"]["timestep_end"])
        traj_step = min(step, end - start - 1 if end > start else 0)
        anchor_idx = start + traj_step
        anchor_x = float(np.asarray(truck_branch["rollout_x"], dtype=np.float32)[anchor_idx])
        anchor_y = float(np.asarray(truck_branch["rollout_y"], dtype=np.float32)[anchor_idx])
        anchor_heading = float(np.asarray(truck_branch["rollout_heading"], dtype=np.float32)[anchor_idx])
        pref_x = np.asarray(truck_branch["rollout_x"], dtype=np.float32)
        pref_y = np.asarray(truck_branch["rollout_y"], dtype=np.float32)
        rej_x = np.asarray(car_branch["rollout_x"], dtype=np.float32)
        rej_y = np.asarray(car_branch["rollout_y"], dtype=np.float32)
        gt_x = np.asarray(pair["truck"]["gt_x"], dtype=np.float32)
        gt_y = np.asarray(pair["truck"]["gt_y"], dtype=np.float32)
        pref_local_x, pref_local_y = _to_local_frame(pref_x, pref_y, anchor_x, anchor_y, anchor_heading)
        rej_local_x, rej_local_y = _to_local_frame(rej_x, rej_y, anchor_x, anchor_y, anchor_heading)
        gt_local_x, gt_local_y = _to_local_frame(gt_x, gt_y, anchor_x, anchor_y, anchor_heading)
        pref_speed = _trajectory_speed(pref_x, pref_y)
        rej_speed = _trajectory_speed(rej_x, rej_y)
        gt_speed = _trajectory_speed(gt_x, gt_y)

        mean_reward_sum += mean_scores
        mean_std_sum += std_scores
        argmax_counts[argmax_idx] += 1
        preferred_counts[preferred_idx] += 1
        rejected_counts[rejected_idx] += 1

        validation_rows.append(
            {
                "global_index": int(global_idx),
                "local_validation_index": int(local_validation_index),
                "window": payload["window"],
                "obs": obs,
                "snapshot_step": int(step),
                "mean_scores": mean_scores,
                "std_scores": std_scores,
                "preferred_idx": preferred_idx,
                "rejected_idx": rejected_idx,
                "preferred_action_label": _action_label(preferred_idx),
                "rejected_action_label": _action_label(rejected_idx),
                "preferred_score": float(mean_scores[preferred_idx]),
                "rejected_score": float(mean_scores[rejected_idx]),
                "argmax_idx": argmax_idx,
                "trajectory_data": {
                    "preferred_local_x": pref_local_x.tolist(),
                    "preferred_local_y": pref_local_y.tolist(),
                    "preferred_speed": pref_speed.tolist(),
                    "rejected_local_x": rej_local_x.tolist(),
                    "rejected_local_y": rej_local_y.tolist(),
                    "rejected_speed": rej_speed.tolist(),
                    "gt_local_x": gt_local_x.tolist(),
                    "gt_local_y": gt_local_y.tolist(),
                    "gt_speed": gt_speed.tolist(),
                },
                "preferred_rank": _rank_desc(mean_scores, preferred_idx),
                "rejected_rank": _rank_desc(mean_scores, rejected_idx),
                "pref_minus_rej": float(mean_scores[preferred_idx] - mean_scores[rejected_idx]),
                "argmax_matches_preferred": bool(argmax_idx == preferred_idx),
                "argmax_matches_rejected": bool(argmax_idx == rejected_idx),
            }
        )

    count = max(1, len(validation_rows))
    mean_reward_grid = _reshape_action_values(mean_reward_sum / count)
    mean_std_grid = _reshape_action_values(mean_std_sum / count)
    argmax_freq_grid = _reshape_action_values(argmax_counts / count)
    preferred_freq_grid = _reshape_action_values(preferred_counts / count)
    rejected_freq_grid = _reshape_action_values(rejected_counts / count)

    resolved_snapshot_steps = sorted({int(row["snapshot_step"]) for row in validation_rows})

    summary = {
        "reward_dir": str(args.reward_dir),
        "preferences": str(args.preferences),
        "evaluation": str(evaluation_path),
        "scoring_mode": args.scoring_mode,
        "validation_windows": int(len(validation_rows)),
        "requested_snapshot_step": int(args.snapshot_step),
        "resolved_snapshot_step": int(resolved_snapshot_steps[0]) if len(resolved_snapshot_steps) == 1 else resolved_snapshot_steps,
        "sequence_conditioned": args.scoring_mode == "sequence_conditioned",
        "sequence_conditioning_description": (
            "Each candidate action is scored by replacing the selected timestep action inside the full preferred "
            "validation window while keeping the rest of the 32-step sequence fixed. The plotted score is the reward "
            "assigned at that selected timestep, not the sum over the whole sequence."
            if args.scoring_mode == "sequence_conditioned"
            else "Each candidate action is scored only from the selected timestep observation/action pair with no "
            "sequence context."
        ),
        "preferred_over_rejected_rate": float(np.mean([row["pref_minus_rej"] > 0.0 for row in validation_rows])),
        "argmax_matches_preferred_rate": float(np.mean([row["argmax_matches_preferred"] for row in validation_rows])),
        "argmax_matches_rejected_rate": float(np.mean([row["argmax_matches_rejected"] for row in validation_rows])),
        "preferred_rank_mean": float(np.mean([row["preferred_rank"] for row in validation_rows])),
        "preferred_rank_median": float(np.median([row["preferred_rank"] for row in validation_rows])),
        "rejected_rank_mean": float(np.mean([row["rejected_rank"] for row in validation_rows])),
        "rejected_rank_median": float(np.median([row["rejected_rank"] for row in validation_rows])),
        "preferred_score_mean": float(np.mean([row["preferred_score"] for row in validation_rows])),
        "rejected_score_mean": float(np.mean([row["rejected_score"] for row in validation_rows])),
        "pref_minus_rej_mean": float(np.mean([row["pref_minus_rej"] for row in validation_rows])),
        "pref_minus_rej_median": float(np.median([row["pref_minus_rej"] for row in validation_rows])),
        "mean_reward_grid": mean_reward_grid.tolist(),
        "mean_std_grid": mean_std_grid.tolist(),
        "argmax_freq_grid": argmax_freq_grid.tolist(),
        "preferred_freq_grid": preferred_freq_grid.tolist(),
        "rejected_freq_grid": rejected_freq_grid.tolist(),
        "argmax_minus_preferred_grid": (argmax_freq_grid - preferred_freq_grid).tolist(),
        "worst_pref_minus_rej_examples": [
            {
                "global_index": int(row["global_index"]),
                "local_validation_index": int(row["local_validation_index"]),
                "pref_minus_rej": float(row["pref_minus_rej"]),
                "preferred_rank": int(row["preferred_rank"]),
                "rejected_rank": int(row["rejected_rank"]),
                "preferred_action_label": row["preferred_action_label"],
                "rejected_action_label": row["rejected_action_label"],
                "argmax_idx": int(row["argmax_idx"]),
                "argmax_action_label": _action_label(row["argmax_idx"]),
                "window": row["window"],
            }
            for row in sorted(validation_rows, key=lambda row: row["pref_minus_rej"])[:10]
        ],
    }

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        if args.snapshot_step < 0:
            suffix = "laststep"
        else:
            suffix = "default"
        output_subdir = f"validation_action_landscapes_{args.scoring_mode}_{suffix}"
        output_dir = args.reward_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_snapshot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    overview_summary = {
        "mean_reward_grid": mean_reward_grid,
        "mean_std_grid": mean_std_grid,
        "argmax_freq_grid": argmax_freq_grid,
        "preferred_freq_grid": preferred_freq_grid,
        "rejected_freq_grid": rejected_freq_grid,
        "argmax_minus_preferred_grid": argmax_freq_grid - preferred_freq_grid,
    }
    _save_overview_figure(output_dir / "validation_action_overview.png", overview_summary)

    example_lookup = {int(row["local_validation_index"]): row for row in validation_rows}
    example_rows = []
    for local_idx, eval_item in _select_examples(eval_payload, args.max_examples):
        row = example_lookup.get(int(local_idx))
        if row is None:
            continue
        example_rows.append(
            {
                **row,
                "eval_item": eval_item,
                "mean_grid": _reshape_action_values(row["mean_scores"]),
                "std_grid": _reshape_action_values(row["std_scores"]),
            }
        )
    _save_example_grid(output_dir / "validation_example_action_landscapes.png", example_rows)

    print(output_dir)


if __name__ == "__main__":
    main()
