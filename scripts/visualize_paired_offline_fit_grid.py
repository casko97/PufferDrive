from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_INPUT = Path("outputs/offline_fits/nuplan_boston_test10_paired_fits.pt")
DEFAULT_OUTPUT = Path("outputs/test_visualizations/paired_offline_fit_grid.png")


def _pair_metrics(car_side: dict, truck_side: dict) -> dict:
    if car_side.get("status") != "ok" or truck_side.get("status") != "ok":
        return {"status": "unavailable"}
    aligned_steps = min(len(car_side["rollout_x"]), len(truck_side["rollout_x"]))
    displacement = np.sqrt(
        (car_side["rollout_x"][:aligned_steps] - truck_side["rollout_x"][:aligned_steps]) ** 2
        + (car_side["rollout_y"][:aligned_steps] - truck_side["rollout_y"][:aligned_steps]) ** 2
    )
    return {
        "status": "ok",
        "aligned_steps": int(aligned_steps),
        "ade": float(displacement.mean()) if displacement.size else float("nan"),
        "fde": float(displacement[-1]) if displacement.size else float("nan"),
    }


def visualize_paired_offline_fit_grid(export_path: Path, output_path: Path, max_maps: int | None = None) -> Path:
    payload = torch.load(export_path, map_location="cpu")
    pairs: dict[str, dict] = payload["pairs"]
    map_names = sorted(pairs.keys())
    if max_maps is not None and max_maps > 0:
        map_names = map_names[:max_maps]
    if not map_names:
        raise ValueError(f"No map pairs found in {export_path}")

    fig, axes = plt.subplots(len(map_names), 2, figsize=(12, max(4, 3.5 * len(map_names))))
    if len(map_names) == 1:
        axes = np.asarray([axes])

    for row_idx, map_name in enumerate(map_names):
        car_result = pairs[map_name]["car"]
        truck_result = pairs[map_name]["truck"]
        pair_metrics = _pair_metrics(car_result, truck_result)
        ok_row_results = [result for result in (car_result, truck_result) if result.get("status") == "ok"]

        shared_limits = None
        if ok_row_results:
            xs = np.concatenate([np.concatenate((result["gt_x"], result["rollout_x"])) for result in ok_row_results])
            ys = np.concatenate([np.concatenate((result["gt_y"], result["rollout_y"])) for result in ok_row_results])
            x_min = float(xs.min())
            x_max = float(xs.max())
            y_min = float(ys.min())
            y_max = float(ys.max())
            x_pad = max(0.5, 0.08 * max(1e-6, x_max - x_min))
            y_pad = max(0.5, 0.08 * max(1e-6, y_max - y_min))
            shared_limits = (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)

        for col_idx, (label, result) in enumerate((("car", car_result), ("truck", truck_result))):
            ax = axes[row_idx, col_idx]
            if result.get("status") == "ok":
                if car_result.get("status") == "ok":
                    ax.plot(car_result["gt_x"], car_result["gt_y"], label="Car GT", linewidth=1.9, color="tab:blue")
                    ax.scatter(car_result["gt_x"][0], car_result["gt_y"][0], s=14, color="tab:blue")
                if truck_result.get("status") == "ok":
                    ax.plot(
                        truck_result["gt_x"],
                        truck_result["gt_y"],
                        label="Truck GT",
                        linewidth=1.9,
                        color="tab:orange",
                    )
                    ax.scatter(truck_result["gt_x"][0], truck_result["gt_y"][0], s=14, color="tab:orange")
                ax.plot(
                    result["rollout_x"],
                    result["rollout_y"],
                    label=f"{label.title()} rollout",
                    linewidth=1.8,
                    linestyle="--",
                    color="tab:green",
                )
                if shared_limits is not None:
                    ax.set_xlim(shared_limits[0], shared_limits[1])
                    ax.set_ylim(shared_limits[2], shared_limits[3])
                pair_line = ""
                if pair_metrics["status"] == "ok":
                    pair_line = (
                        f"\npair_ade={pair_metrics['ade']:.2f} "
                        f"pair_fde={pair_metrics['fde']:.2f} "
                        f"pair_n={pair_metrics['aligned_steps']}"
                    )
                ax.set_title(
                    f"{label} {Path(map_name).stem}\n"
                    f"steps={result['num_steps']} "
                    f"cost={result['match_cost_total']:.2f} "
                    f"mean={result['self_ade']:.2f} "
                    f"final={result['self_fde']:.2f}"
                    f"{pair_line}",
                    fontsize=9,
                )
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.25)
                if row_idx == 0:
                    ax.legend(fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"unavailable\n{result.get('error', 'unknown error')}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    wrap=True,
                )
                ax.set_title(f"{label} {Path(map_name).stem}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a grid of paired offline fit trajectories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-maps", type=int, default=0)
    args = parser.parse_args()

    output_path = visualize_paired_offline_fit_grid(
        export_path=args.input,
        output_path=args.output,
        max_maps=(args.max_maps if args.max_maps > 0 else None),
    )
    print(output_path)


if __name__ == "__main__":
    main()
