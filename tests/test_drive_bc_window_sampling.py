from pathlib import Path

import numpy as np
import pytest
import torch

from pufferlib.ocean.drive import drive as drive_module


CLASSIC_ACTION_SPACE = 9 * 13
TRAINING_WINDOWS_STRIDE10_DATASET_DIR = Path("pufferlib/resources/drive/bc_dataset_training_full_windows_stride10")


def _sample_action_counts(dataset):
    counts = np.zeros((CLASSIC_ACTION_SPACE,), dtype=np.int64)
    total_windows = 0
    for _, actions, mask in dataset:
        valid_actions = actions[mask].cpu().numpy().astype(np.int64, copy=False)
        counts += np.bincount(valid_actions, minlength=CLASSIC_ACTION_SPACE)
        total_windows += 1
    return counts, total_windows


def _balance_metrics(counts):
    positive = counts[counts > 0].astype(np.float64)
    probs = positive / float(positive.sum())
    entropy = float(-(probs * np.log(probs)).sum())
    max_entropy = float(np.log(max(positive.size, 2)))
    normalized_entropy = entropy / max_entropy
    imbalance_ratio = float(positive.max() / positive.min())
    uniform_target = float(positive.sum()) / float(positive.size)
    mean_abs_relative_deviation = float(np.mean(np.abs(positive - uniform_target) / uniform_target))
    balance_score = 1.0 / (1.0 + mean_abs_relative_deviation)
    return {
        "normalized_entropy": normalized_entropy,
        "imbalance_ratio": imbalance_ratio,
        "balance_score": balance_score,
        "positive_actions": int(positive.size),
    }


def _evaluate_real_prefix_window_rebalancing(window_balance_fraction, *, plot_stem):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_dir = TRAINING_WINDOWS_STRIDE10_DATASET_DIR
    if not dataset_dir.is_dir():
        pytest.skip(f"Training stride-10 BC dataset not found: {dataset_dir}")

    shard_paths = sorted(dataset_dir.glob("map_*.pt"))[:32]
    if not shard_paths:
        pytest.skip(f"No BC shard files found in {dataset_dir}")

    first_payload = torch.load(shard_paths[0], map_location="cpu")
    obs_dim = int(first_payload["obs"].shape[1])

    baseline = drive_module._SequenceBCDataset(
        [str(path) for path in shard_paths],
        obs_dim=obs_dim,
        action_space_size=CLASSIC_ACTION_SPACE,
        seq_len=32,
        stride=10,
        shuffle=True,
        seed=0,
        require_embedded=True,
        rebalance_windows=False,
    )
    baseline.set_epoch(0)
    baseline_counts, baseline_windows = _sample_action_counts(baseline)
    baseline_metrics = _balance_metrics(baseline_counts)

    rebalanced = drive_module._SequenceBCDataset(
        [str(path) for path in shard_paths],
        obs_dim=obs_dim,
        action_space_size=CLASSIC_ACTION_SPACE,
        seq_len=32,
        stride=10,
        shuffle=True,
        seed=0,
        require_embedded=True,
        rebalance_windows=True,
        window_balance_fraction=window_balance_fraction,
        window_balance_max_multiplier=10.0,
    )
    rebalanced.set_epoch(0)
    rebalanced_counts, rebalanced_windows = _sample_action_counts(rebalanced)
    rebalanced_metrics = _balance_metrics(rebalanced_counts)

    assert baseline_windows > 0
    assert rebalanced_windows == baseline_windows
    assert rebalanced_metrics["positive_actions"] == baseline_metrics["positive_actions"]
    assert rebalanced_metrics["normalized_entropy"] >= baseline_metrics["normalized_entropy"]
    assert rebalanced_metrics["imbalance_ratio"] <= baseline_metrics["imbalance_ratio"]
    assert rebalanced_metrics["balance_score"] >= baseline_metrics["balance_score"]

    ratio = rebalanced_counts.astype(np.float64) / np.maximum(baseline_counts.astype(np.float64), 1.0)
    changed_action_ids = np.argsort(np.abs(ratio - 1.0))[::-1]
    changed_action_ids = [
        int(action_id)
        for action_id in changed_action_ids
        if int(baseline_counts[action_id] + rebalanced_counts[action_id]) > 0
    ][:12]
    assert changed_action_ids

    plot_dir = Path("outputs/test_visualizations")
    plot_dir.mkdir(parents=True, exist_ok=True)
    diff_plot_path = plot_dir / f"{plot_stem}_distribution_diff.png"
    full_plot_path = plot_dir / f"{plot_stem}_full_action_space.png"

    x = np.arange(len(changed_action_ids), dtype=np.float32)
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), squeeze=False)
    ax_counts, ax_ratio = axes[:, 0]
    ax_counts.bar(x - width / 2, baseline_counts[changed_action_ids], width=width, label="baseline")
    ax_counts.bar(x + width / 2, rebalanced_counts[changed_action_ids], width=width, label="rebalanced")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels([str(action_id) for action_id in changed_action_ids], rotation=45, ha="right")
    ax_counts.set_xlabel("joint action id")
    ax_counts.set_ylabel("sampled action occurrences")
    ax_counts.set_title(
        "Real prefix recurrent window rebalancing\n"
        f"baseline_score={baseline_metrics['balance_score']:.3f}, "
        f"rebalanced_score={rebalanced_metrics['balance_score']:.3f}, "
        f"fraction={window_balance_fraction:.2f}"
    )
    ax_counts.legend()
    ax_counts.grid(True, axis="y", alpha=0.25)

    ax_ratio.bar(x, ratio[changed_action_ids], color="tab:green", width=0.6)
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([str(action_id) for action_id in changed_action_ids], rotation=45, ha="right")
    ax_ratio.set_xlabel("joint action id")
    ax_ratio.set_ylabel("rebalanced / baseline")
    ax_ratio.set_title("Largest action-distribution shifts on real shard prefix")
    ax_ratio.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(diff_plot_path)
    plt.close(fig)

    full_x = np.arange(CLASSIC_ACTION_SPACE)
    full_ratio = rebalanced_counts.astype(np.float64) / np.maximum(baseline_counts.astype(np.float64), 1.0)
    full_changed = rebalanced_counts - baseline_counts
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), squeeze=False)
    ax0, ax1, ax2 = axes[:, 0]
    ax0.bar(full_x, baseline_counts, width=0.9, alpha=0.6, label="baseline")
    ax0.bar(full_x, rebalanced_counts, width=0.6, alpha=0.6, label="rebalanced")
    ax0.set_title(
        f"Full action space counts | windows={baseline_windows} -> {rebalanced_windows} | "
        f"fraction={window_balance_fraction:.2f}"
    )
    ax0.set_xlabel("joint action id")
    ax0.set_ylabel("sampled action occurrences")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend()

    ax1.bar(full_x, full_ratio, width=0.9, color="tab:green")
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_title("Full action space ratio: rebalanced / baseline")
    ax1.set_xlabel("joint action id")
    ax1.set_ylabel("ratio")
    ax1.grid(True, axis="y", alpha=0.25)

    colors = np.where(full_changed >= 0, "tab:blue", "tab:red")
    ax2.bar(full_x, full_changed, width=0.9, color=colors)
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_title("Full action space signed change: rebalanced - baseline")
    ax2.set_xlabel("joint action id")
    ax2.set_ylabel("delta count")
    ax2.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(full_plot_path)
    plt.close(fig)

    assert diff_plot_path.exists()
    assert full_plot_path.exists()


def test_sequence_bc_dataset_can_rebalance_windows_toward_rare_actions(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shard_path = tmp_path / "map_000.pt"
    action_common = 0
    action_rare = 1
    action_medium = 2
    payload = {
        "obs": torch.zeros((12, 2), dtype=torch.float32),
        "action": torch.tensor(
            [
                action_common,
                action_common,
                action_common,
                action_common,
                action_common,
                action_common,
                action_common,
                action_medium,
                action_common,
                action_rare,
                action_medium,
                action_rare,
            ],
            dtype=torch.int64,
        ),
        "map_id": torch.zeros((12,), dtype=torch.int64),
        "timestep": torch.arange(12, dtype=torch.int64),
        "sequence_id": torch.zeros((12,), dtype=torch.int64),
        "sequence_row_index": torch.arange(12, dtype=torch.int64),
        "sequence_length": torch.full((12,), 12, dtype=torch.int64),
        "window_metadata": {
            "version": 1,
            "seq_len": 2,
            "stride": 1,
            "window_count": 6,
            "window_indices": torch.tensor(
                [
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],
                    [8, 9],
                    [10, 11],
                ],
                dtype=torch.int64,
            ),
            "valid_lengths": torch.tensor([2, 2, 2, 2, 2, 2], dtype=torch.int64),
        },
    }
    torch.save(payload, shard_path)

    baseline = drive_module._SequenceBCDataset(
        [str(shard_path)],
        obs_dim=2,
        action_space_size=CLASSIC_ACTION_SPACE,
        seq_len=2,
        stride=1,
        shuffle=True,
        seed=0,
        require_embedded=True,
        rebalance_windows=False,
    )
    baseline.set_epoch(0)
    baseline_samples = list(baseline)
    baseline_counts = _sample_action_counts(baseline)[0]
    baseline_metrics = _balance_metrics(baseline_counts)

    rebalanced = drive_module._SequenceBCDataset(
        [str(shard_path)],
        obs_dim=2,
        action_space_size=CLASSIC_ACTION_SPACE,
        seq_len=2,
        stride=1,
        shuffle=True,
        seed=0,
        require_embedded=True,
        rebalance_windows=True,
        window_balance_fraction=1.0,
        window_balance_max_multiplier=10.0,
    )
    rebalanced.set_epoch(0)
    rebalanced_samples = list(rebalanced)
    rebalanced_counts = _sample_action_counts(rebalanced)[0]
    rebalanced_metrics = _balance_metrics(rebalanced_counts)

    assert len(baseline_samples) == 6
    assert len(rebalanced_samples) == 6
    assert int(baseline_counts.sum()) == 12
    assert int(rebalanced_counts.sum()) == 12
    assert baseline_counts[action_common] == 8
    assert baseline_counts[action_medium] == 2
    assert baseline_counts[action_rare] == 2
    assert rebalanced_counts[action_rare] >= baseline_counts[action_rare]
    assert rebalanced_counts[action_medium] >= baseline_counts[action_medium]
    assert rebalanced_metrics["normalized_entropy"] >= baseline_metrics["normalized_entropy"]
    assert rebalanced_metrics["imbalance_ratio"] <= baseline_metrics["imbalance_ratio"]
    assert rebalanced_metrics["balance_score"] >= baseline_metrics["balance_score"]

    plot_dir = Path("outputs/test_visualizations")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "bc_train_window_rebalancing_distribution_diff.png"
    x = np.arange(3, dtype=np.float32)
    plotted_action_ids = np.asarray([action_common, action_medium, action_rare], dtype=np.int64)
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), squeeze=False)
    ax_counts, ax_ratio = axes[:, 0]
    ax_counts.bar(x - width / 2, baseline_counts[plotted_action_ids], width=width, label="baseline")
    ax_counts.bar(x + width / 2, rebalanced_counts[plotted_action_ids], width=width, label="rebalanced")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels([str(int(action_id)) for action_id in plotted_action_ids])
    ax_counts.set_xlabel("action id")
    ax_counts.set_ylabel("sampled count")
    ax_counts.set_title(
        "Trainer window rebalancing action counts\n"
        f"baseline_score={baseline_metrics['balance_score']:.3f}, "
        f"rebalanced_score={rebalanced_metrics['balance_score']:.3f}"
    )
    ax_counts.legend()
    ax_counts.grid(True, axis="y", alpha=0.25)

    ratio = rebalanced_counts[plotted_action_ids].astype(np.float64) / np.maximum(
        baseline_counts[plotted_action_ids].astype(np.float64), 1.0
    )
    ax_ratio.bar(x, ratio, color="tab:green", width=0.6)
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([str(int(action_id)) for action_id in plotted_action_ids])
    ax_ratio.set_xlabel("action id")
    ax_ratio.set_ylabel("rebalanced / baseline")
    ax_ratio.set_title("Distribution shift over sampled recurrent windows")
    ax_ratio.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    assert plot_path.exists()


def test_recurrent_window_rebalancing_improves_real_dataset_balance_prefix():
    _evaluate_real_prefix_window_rebalancing(1.0, plot_stem="bc_train_window_rebalancing_real_prefix")


def test_recurrent_window_rebalancing_real_prefix_mild_plot():
    _evaluate_real_prefix_window_rebalancing(0.2, plot_stem="bc_train_window_rebalancing_real_prefix_mild")
