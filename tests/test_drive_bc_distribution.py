import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.create_balanced_bc_window_subset import create_balanced_bc_window_subset


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


def test_balanced_subset_builder_counts_all_actions_within_window(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "subset"
    source_dir.mkdir()

    action_a = 0
    action_b = 1
    action_c = 2

    payload = {
        "action": torch.tensor([action_a, action_a, action_a, action_b, action_a, action_c], dtype=torch.int64),
        "window_metadata": {
            "version": 1,
            "seq_len": 2,
            "stride": 1,
            "window_count": 3,
            "window_indices": torch.tensor(
                [
                    [0, 1],
                    [2, 3],
                    [4, 5],
                ],
                dtype=torch.int64,
            ),
            "valid_lengths": torch.tensor([2, 2, 2], dtype=torch.int64),
        },
        "metadata": {
            "map_id": 0,
        },
    }
    torch.save(payload, source_dir / "map_000.pt")

    summary = create_balanced_bc_window_subset(
        source_dir,
        output_dir,
        window_set_name="window_metadata",
        target_count_per_action=1,
        seed=0,
        log_every=1,
    )

    assert summary["source_total_windows"] == 3
    assert summary["source_total_action_occurrences"] == 6
    assert summary["source_action_counts"][action_a] == 4
    assert summary["source_action_counts"][action_b] == 1
    assert summary["source_action_counts"][action_c] == 1

    assert summary["actual_target_count_per_action"] == 1
    assert summary["subset_total_windows"] == 2
    assert summary["subset_total_action_occurrences"] == 4
    assert summary["subset_nonzero_actions"] == 3
    assert summary["subset_action_counts"][action_a] == 2
    assert summary["subset_action_counts"][action_b] == 1
    assert summary["subset_action_counts"][action_c] == 1

    subset_payload = torch.load(output_dir / "map_000.pt", map_location="cpu")
    subset_manifest = subset_payload["window_metadata"]
    assert int(subset_manifest["window_count"]) == 2
    selected_rows = subset_manifest["window_indices"][:, 0].tolist()
    assert sorted(selected_rows) == [2, 4]

    summary_path = output_dir / "subset_summary.json"
    assert summary_path.exists()
    written_summary = json.loads(summary_path.read_text())
    assert written_summary["subset_total_action_occurrences"] == 4


def test_fractional_rarity_rebalancing_preserves_window_count_and_upsamples_rare_windows(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "subset"
    source_dir.mkdir()

    action_common = 0
    action_rare = 1

    payload = {
        "action": torch.tensor(
            [
                action_common,
                action_common,
                action_common,
                action_common,
                action_common,
                action_common,
                action_common,
                action_rare,
            ],
            dtype=torch.int64,
        ),
        "window_metadata": {
            "version": 1,
            "seq_len": 2,
            "stride": 1,
            "window_count": 4,
            "window_indices": torch.tensor(
                [
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],
                ],
                dtype=torch.int64,
            ),
            "valid_lengths": torch.tensor([2, 2, 2, 2], dtype=torch.int64),
        },
        "metadata": {
            "map_id": 0,
        },
    }
    torch.save(payload, source_dir / "map_000.pt")

    summary = create_balanced_bc_window_subset(
        source_dir,
        output_dir,
        window_set_name="window_metadata",
        rebalance_strategy="fractional_rarity",
        balance_fraction=1.0,
        target_window_fraction=1.0,
        max_rarity_multiplier=10.0,
        target_count_per_action=1,
        seed=0,
        log_every=1,
    )

    assert summary["rebalance_strategy"] == "fractional_rarity"
    assert summary["source_total_windows"] == 4
    assert summary["subset_total_windows"] == 4
    assert summary["source_total_action_occurrences"] == 8
    assert summary["subset_total_action_occurrences"] == 8
    assert summary["source_action_counts"][action_common] == 7
    assert summary["source_action_counts"][action_rare] == 1
    assert summary["subset_action_counts"][action_rare] > summary["source_action_counts"][action_rare]

    subset_payload = torch.load(output_dir / "map_000.pt", map_location="cpu")
    subset_manifest = subset_payload["window_metadata"]
    assert int(subset_manifest["window_count"]) == 4
    selected_rows = subset_manifest["window_indices"][:, 0].tolist()
    assert selected_rows.count(6) >= 2
