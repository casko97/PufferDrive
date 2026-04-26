#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb


DEFAULT_METRICS = (
    "loss",
    "ade",
    "fde",
    "heading_error",
    "speed_error",
    "valid_accuracy",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot trajectory BC epoch metrics from a W&B run."
    )
    parser.add_argument("--project", required=True, help="W&B project, e.g. user/proj")
    parser.add_argument("--run-id", required=True, help="W&B run id")
    parser.add_argument("--output-path", type=Path, required=True, help="Output PNG path")
    parser.add_argument(
        "--samples",
        type=int,
        default=20000,
        help="Maximum number of W&B history rows to request",
    )
    return parser.parse_args()


def _collect_epoch_history(run, metrics: tuple[str, ...], samples: int) -> list[dict]:
    rows = run.history(samples=samples, pandas=False)
    by_epoch: dict[int, dict] = {}
    metric_keys = {
        *(f"train/{metric}" for metric in metrics),
        *(f"val/{metric}" for metric in metrics),
    }

    for row in rows:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        if not any(key in row for key in metric_keys):
            continue
        epoch_int = int(epoch)
        entry = by_epoch.setdefault(epoch_int, {"epoch": epoch_int, "train": {}, "val": {}})
        for metric in metrics:
            train_key = f"train/{metric}"
            val_key = f"val/{metric}"
            if train_key in row and row[train_key] is not None:
                entry["train"][metric] = float(row[train_key])
            if val_key in row and row[val_key] is not None:
                entry["val"][metric] = float(row[val_key])

    history = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    if not history:
        raise ValueError(f"No epoch-level trajectory metrics found in W&B run {run.id}")
    return history


def _collect_series(history: list[dict], split: str, metric: str) -> np.ndarray:
    values = []
    for epoch_entry in history:
        split_entry = epoch_entry.get(split, {})
        value = split_entry.get(metric)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float32)


def plot_metrics_from_wandb(
    project: str,
    run_id: str,
    output_path: Path,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    samples: int = 20000,
) -> Path:
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    history = _collect_epoch_history(run, metrics, samples=samples)
    epochs = np.asarray([int(entry["epoch"]) for entry in history], dtype=np.int32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_panels = len(metrics)
    cols = 2
    rows = int(np.ceil(num_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for panel_idx, metric in enumerate(metrics):
        row_idx = panel_idx // cols
        col_idx = panel_idx % cols
        ax = axes[row_idx, col_idx]

        train_values = _collect_series(history, "train", metric)
        val_values = _collect_series(history, "val", metric)

        ax.plot(epochs, train_values, marker="o", linewidth=2.0, color="tab:blue", label="train")
        if not np.all(np.isnan(val_values)):
            ax.plot(epochs, val_values, marker="o", linewidth=2.0, color="tab:orange", label="val")
            best_idx = int(np.nanargmax(val_values)) if metric == "valid_accuracy" else int(np.nanargmin(val_values))
            ax.scatter([epochs[best_idx]], [val_values[best_idx]], color="crimson", s=40, zorder=4)

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)
        ax.legend(loc="best")

    for empty_idx in range(num_panels, rows * cols):
        row_idx = empty_idx // cols
        col_idx = empty_idx % cols
        axes[row_idx, col_idx].axis("off")

    run_name = getattr(run, "name", run_id)
    fig.suptitle(f"Trajectory BC metrics from W&B\n{run_name} ({run.id})")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    args = _parse_args()
    result = plot_metrics_from_wandb(
        project=args.project,
        run_id=args.run_id,
        output_path=args.output_path.expanduser().resolve(),
        samples=args.samples,
    )
    print(result)


if __name__ == "__main__":
    main()
