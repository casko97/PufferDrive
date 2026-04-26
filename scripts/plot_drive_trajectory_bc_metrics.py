#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = (
    "loss",
    "ade",
    "fde",
    "heading_error",
    "speed_error",
    "valid_accuracy",
)


def _load_history(metrics_path: Path) -> list[dict]:
    payload = json.loads(metrics_path.read_text())
    history = payload.get("history", [])
    if not history:
        raise ValueError(f"No history found in {metrics_path}")
    return history


def _collect_series(history: list[dict], split: str, metric: str) -> np.ndarray:
    values = []
    for epoch_entry in history:
        split_entry = epoch_entry.get(split, {})
        value = split_entry.get(metric)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float32)


def plot_trajectory_bc_metrics(
    metrics_path: Path,
    output_path: Path,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> Path:
    history = _load_history(metrics_path)
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

        best_idx = int(np.nanargmin(val_values if metric != "valid_accuracy" else -val_values)) if not np.all(np.isnan(val_values)) else None
        if best_idx is not None:
            ax.scatter([epochs[best_idx]], [val_values[best_idx]], color="crimson", s=40, zorder=4)

        pretty_title = metric.replace("_", " ").title()
        ax.set_title(pretty_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)
        ax.legend(loc="best")

    for empty_idx in range(num_panels, rows * cols):
        row_idx = empty_idx // cols
        col_idx = empty_idx % cols
        axes[row_idx, col_idx].axis("off")

    fig.suptitle(f"Trajectory BC training metrics\n{metrics_path.parent.name}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/val metrics from a trajectory BC metrics.json file.")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        required=True,
        help="Path to the BC `metrics.json` file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to `<metrics_dir>/metrics_overview.png`.",
    )
    args = parser.parse_args()

    metrics_path = args.metrics_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve() if args.output_path else metrics_path.parent / "metrics_overview.png"
    result = plot_trajectory_bc_metrics(metrics_path, output_path)
    print(result)


if __name__ == "__main__":
    main()
