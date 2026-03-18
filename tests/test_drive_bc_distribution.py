import json
from pathlib import Path

import numpy as np
import pytest
import torch


CLASSIC_ACCELERATION_VALUES = np.asarray((-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0), dtype=np.float32)
CLASSIC_STEERING_VALUES = np.asarray(
    (-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0),
    dtype=np.float32,
)
CLASSIC_ACTION_SPACE = int(CLASSIC_ACCELERATION_VALUES.shape[0] * CLASSIC_STEERING_VALUES.shape[0])
TRAINING_WINDOWS_STRIDE10_DATASET_DIR = Path("pufferlib/resources/drive/bc_dataset_training_full_windows_stride10")


def test_bc_training_stride10_dataset_action_distribution_summary():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_dir = TRAINING_WINDOWS_STRIDE10_DATASET_DIR
    if not dataset_dir.is_dir():
        pytest.skip(f"Training stride-10 BC dataset not found: {dataset_dir}")

    shard_paths = sorted(dataset_dir.glob("map_*.pt"))
    if not shard_paths:
        pytest.skip(f"No BC shard files found in {dataset_dir}")

    action_counts = np.zeros(CLASSIC_ACTION_SPACE, dtype=np.int64)
    total_samples = 0

    for shard_path in shard_paths:
        payload = torch.load(shard_path)
        actions = payload.get("action")
        if actions is None or int(actions.numel()) == 0:
            continue
        actions_np = actions.cpu().numpy().astype(np.int64, copy=False)
        total_samples += int(actions_np.size)
        action_counts += np.bincount(actions_np, minlength=CLASSIC_ACTION_SPACE)

    assert total_samples > 0
    assert int(action_counts.sum()) == total_samples

    joint_counts = action_counts.reshape(len(CLASSIC_ACCELERATION_VALUES), len(CLASSIC_STEERING_VALUES))
    accel_counts = joint_counts.sum(axis=1)
    steer_counts = joint_counts.sum(axis=0)

    nonzero_actions = int(np.count_nonzero(action_counts))
    sample_probs = action_counts / float(total_samples)
    nonzero_probs = sample_probs[sample_probs > 0]
    entropy = float(-(nonzero_probs * np.log(nonzero_probs)).sum())

    top_action_indices = np.argsort(action_counts)[::-1][:10]
    top_actions = []
    for action_id in top_action_indices:
        count = int(action_counts[action_id])
        if count <= 0:
            continue
        accel_idx = int(action_id // len(CLASSIC_STEERING_VALUES))
        steer_idx = int(action_id % len(CLASSIC_STEERING_VALUES))
        top_actions.append(
            {
                "action_id": int(action_id),
                "count": count,
                "fraction": float(count / total_samples),
                "acceleration": float(CLASSIC_ACCELERATION_VALUES[accel_idx]),
                "steering": float(CLASSIC_STEERING_VALUES[steer_idx]),
            }
        )

    plot_dir = Path("outputs/test_visualizations")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "bc_training_stride10_action_distribution.png"
    summary_path = plot_dir / "bc_training_stride10_action_distribution.json"

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), squeeze=False)
    ax_joint, ax_accel, ax_steer = axes[:, 0]

    ax_joint.bar(np.arange(CLASSIC_ACTION_SPACE), action_counts, color="tab:blue", width=0.9)
    ax_joint.set_title(
        f"Joint action distribution | samples={total_samples:,} | nonzero_actions={nonzero_actions}/{CLASSIC_ACTION_SPACE} | entropy={entropy:.2f}"
    )
    ax_joint.set_xlabel("joint action id")
    ax_joint.set_ylabel("count")
    ax_joint.grid(True, axis="y", alpha=0.25)

    accel_labels = [f"{v:g}" for v in CLASSIC_ACCELERATION_VALUES]
    ax_accel.bar(np.arange(len(CLASSIC_ACCELERATION_VALUES)), accel_counts, color="tab:orange", width=0.75)
    ax_accel.set_title("Acceleration marginal")
    ax_accel.set_xlabel("acceleration [m/s^2]")
    ax_accel.set_ylabel("count")
    ax_accel.set_xticks(np.arange(len(CLASSIC_ACCELERATION_VALUES)))
    ax_accel.set_xticklabels(accel_labels)
    ax_accel.grid(True, axis="y", alpha=0.25)

    steer_labels = [f"{v:.3g}" for v in CLASSIC_STEERING_VALUES]
    ax_steer.bar(np.arange(len(CLASSIC_STEERING_VALUES)), steer_counts, color="tab:green", width=0.75)
    ax_steer.set_title("Steering marginal")
    ax_steer.set_xlabel("steering")
    ax_steer.set_ylabel("count")
    ax_steer.set_xticks(np.arange(len(CLASSIC_STEERING_VALUES)))
    ax_steer.set_xticklabels(steer_labels)
    ax_steer.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    summary = {
        "dataset_dir": str(dataset_dir),
        "shard_count": len(shard_paths),
        "total_samples": int(total_samples),
        "action_space_size": int(CLASSIC_ACTION_SPACE),
        "nonzero_actions": nonzero_actions,
        "entropy": entropy,
        "top_actions": top_actions,
        "acceleration_counts": accel_counts.tolist(),
        "steering_counts": steer_counts.tolist(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    assert plot_path.exists()
    assert summary_path.exists()
    assert nonzero_actions > 10
