#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scripts.animate_drive_model_trajectories import collect_rollout, plot_road_edges, scenario_bounds
from scripts.analyze_human_replay_turning_buckets import classify_map_turning


GT_COLOR = "#1F3A5F"
LEFT_COLOR = "#0B6E4F"
RIGHT_COLOR = "#C84C09"


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def select_rows_from_csv(
    csv_path: Path,
    threshold_deg: float,
    top_best: int,
    top_worst: int,
    turning_only: bool,
) -> list[dict[str, object]]:
    rows = load_csv_rows(csv_path)
    selected = []
    for row in rows:
        bucket_row = classify_map_turning(Path(row["map_path"]), threshold_deg=threshold_deg)
        if turning_only and bucket_row["bucket"] != "turning":
            continue
        score = float(row["score"])
        selected.append(
            {
                **row,
                "score_float": score,
                "delta_heading_deg": float(bucket_row["delta_heading_deg"]),
                "bucket": bucket_row["bucket"],
            }
        )

    selected.sort(key=lambda row: (row["score_float"], float(row["completion_rate"]), -float(row["offroad_per_agent"])))
    best = sorted(selected[-top_best:], key=lambda row: row["score_float"], reverse=True) if top_best else []
    worst = sorted(selected[:top_worst], key=lambda row: row["score_float"]) if top_worst else []
    return best + worst


def build_scenarios(
    rows: list[dict[str, object]],
    left_config: Path,
    left_checkpoint: Path,
    right_config: Path,
    right_checkpoint: Path,
    device: str,
) -> list[dict[str, object]]:
    scenarios = []
    for row in rows:
        map_path = Path(str(row["map_path"]))
        left = collect_rollout(left_config, left_checkpoint, map_path, device)
        right = collect_rollout(right_config, right_checkpoint, map_path, device)
        xlim, ylim = scenario_bounds(left, right)
        scenarios.append(
            {
                "map_name": map_path.name,
                "scenario_id": left["scenario_id"],
                "left": left,
                "right": right,
                "xlim": xlim,
                "ylim": ylim,
                "score": float(row.get("score_float", row.get("score", 0.0))),
                "delta_heading_deg": float(row.get("delta_heading_deg", 0.0)),
            }
        )
    return scenarios


def render_static_grid(
    scenarios: list[dict[str, object]],
    output_path: Path,
    title: str,
    columns: int,
    left_label: str,
    right_label: str,
) -> None:
    columns = max(1, min(columns, len(scenarios)))
    rows = math.ceil(len(scenarios) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5.6 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // columns, idx % columns]
        plot_road_edges(ax, scenario["left"]["road_edges"], scenario["scenario_id"])
        ax.plot(
            scenario["left"]["gt_x"],
            scenario["left"]["gt_y"],
            linestyle="--",
            color=GT_COLOR,
            linewidth=1.2,
            alpha=0.9,
            label="human replay",
        )
        ax.plot(
            scenario["left"]["sim_x"],
            scenario["left"]["sim_y"],
            color=LEFT_COLOR,
            linewidth=2.0,
            alpha=0.95,
            label=left_label,
        )
        ax.plot(
            scenario["right"]["sim_x"],
            scenario["right"]["sim_y"],
            color=RIGHT_COLOR,
            linewidth=2.0,
            alpha=0.95,
            label=right_label,
        )
        ax.scatter([scenario["left"]["sim_x"][-1]], [scenario["left"]["sim_y"][-1]], color=LEFT_COLOR, s=18, zorder=3)
        ax.scatter([scenario["right"]["sim_x"][-1]], [scenario["right"]["sim_y"][-1]], color=RIGHT_COLOR, s=18, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(*scenario["xlim"])
        ax.set_ylim(*scenario["ylim"])
        ax.set_title(
            f"{scenario['map_name'].replace('.bin', '')}\nscore={scenario['score']:.2f}, "
            f"turn={scenario['delta_heading_deg']:.1f} deg",
            fontsize=10,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    for idx in range(len(scenarios), rows * columns):
        axes[idx // columns, idx % columns].axis("off")

    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render a static multi-scenario trajectory comparison grid for two PufferDrive policies.")
    parser.add_argument("--left-config", type=Path, required=True)
    parser.add_argument("--left-checkpoint", type=Path, required=True)
    parser.add_argument("--right-config", type=Path, required=True)
    parser.add_argument("--right-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-label", type=str, default="left policy")
    parser.add_argument("--right-label", type=str, default="right policy")
    parser.add_argument("--title", type=str, default="Trajectory Comparison")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--map-path", type=Path, action="append")
    parser.add_argument("--scenario-csv", type=Path)
    parser.add_argument("--top-best", type=int, default=4)
    parser.add_argument("--top-worst", type=int, default=4)
    parser.add_argument("--turning-only", action="store_true")
    parser.add_argument("--turning-threshold-deg", type=float, default=45.0)
    args = parser.parse_args()

    rows = []
    if args.map_path:
        for map_path in args.map_path:
            bucket_row = classify_map_turning(map_path.resolve(), threshold_deg=args.turning_threshold_deg)
            rows.append(
                {
                    "map_path": str(map_path.resolve()),
                    "score_float": 0.0,
                    "delta_heading_deg": float(bucket_row["delta_heading_deg"]),
                }
            )
    elif args.scenario_csv:
        rows = select_rows_from_csv(
            csv_path=args.scenario_csv.resolve(),
            threshold_deg=args.turning_threshold_deg,
            top_best=args.top_best,
            top_worst=args.top_worst,
            turning_only=args.turning_only,
        )
    else:
        raise ValueError("Provide either one or more --map-path values or a --scenario-csv")

    if not rows:
        raise ValueError("No scenarios selected for plotting")

    scenarios = build_scenarios(
        rows=rows,
        left_config=args.left_config.resolve(),
        left_checkpoint=args.left_checkpoint.resolve(),
        right_config=args.right_config.resolve(),
        right_checkpoint=args.right_checkpoint.resolve(),
        device=args.device,
    )
    render_static_grid(
        scenarios=scenarios,
        output_path=args.output.resolve(),
        title=args.title,
        columns=args.columns,
        left_label=args.left_label,
        right_label=args.right_label,
    )


if __name__ == "__main__":
    main()
