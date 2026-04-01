from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from scripts.export_paired_offline_fits import _pair_metrics, iter_paired_fit_shards


DEFAULT_INPUT = Path("outputs/offline_fits/nuplan_boston_test10_paired_fits.pt")
DEFAULT_MAX_SNIPPETS = 10
DEFAULT_FPS = 6
DEFAULT_INTERVAL_MS = 160


def _default_output_dir(export_path: Path) -> Path:
    return export_path.with_name(f"{export_path.stem}_videos")


def _select_pairs(export_path: Path, max_snippets: int) -> list[tuple[str, dict]]:
    selected: list[tuple[str, dict]] = []
    for _metadata, pairs, _shard_info in iter_paired_fit_shards(export_path):
        for map_name in sorted(pairs.keys()):
            pair = pairs[map_name]
            if pair.get("car", {}).get("status") != "ok" or pair.get("truck", {}).get("status") != "ok":
                continue
            selected.append((map_name, pair))
            if len(selected) >= max_snippets:
                return selected
    return selected


def _shared_limits(car_result: dict, truck_result: dict) -> tuple[float, float, float, float] | None:
    ok_results = [result for result in (car_result, truck_result) if result.get("status") == "ok"]
    if not ok_results:
        return None

    xs = np.concatenate([np.concatenate((result["gt_x"], result["rollout_x"])) for result in ok_results])
    ys = np.concatenate([np.concatenate((result["gt_y"], result["rollout_y"])) for result in ok_results])
    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())
    x_pad = max(0.5, 0.08 * max(1e-6, x_max - x_min))
    y_pad = max(0.5, 0.08 * max(1e-6, y_max - y_min))
    return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)


def _draw_fit_panel(
    ax,
    *,
    label: str,
    map_name: str,
    active_result: dict,
    car_result: dict,
    truck_result: dict,
    pair_metrics: dict,
    frame_idx: int,
    limits: tuple[float, float, float, float] | None,
) -> None:
    ax.cla()
    if car_result.get("status") == "ok":
        ax.plot(car_result["gt_x"], car_result["gt_y"], label="Car GT", linewidth=1.9, color="tab:blue", alpha=0.9)
        ax.scatter(car_result["gt_x"][0], car_result["gt_y"][0], s=14, color="tab:blue")
    if truck_result.get("status") == "ok":
        ax.plot(
            truck_result["gt_x"],
            truck_result["gt_y"],
            label="Truck GT",
            linewidth=1.9,
            color="tab:orange",
            alpha=0.9,
        )
        ax.scatter(truck_result["gt_x"][0], truck_result["gt_y"][0], s=14, color="tab:orange")

    rollout_steps = min(frame_idx + 1, len(active_result["rollout_x"]))
    ax.plot(
        active_result["rollout_x"][:rollout_steps],
        active_result["rollout_y"][:rollout_steps],
        label=f"{label.title()} rollout",
        linewidth=2.2,
        linestyle="--",
        color="tab:green",
    )
    if rollout_steps > 0:
        ax.scatter(
            [active_result["rollout_x"][rollout_steps - 1]],
            [active_result["rollout_y"][rollout_steps - 1]],
            s=18,
            color="tab:green",
        )

    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])

    pair_line = ""
    if pair_metrics.get("status") == "ok":
        pair_line = (
            f"\npair_ade={pair_metrics['ade']:.2f} "
            f"pair_fde={pair_metrics['fde']:.2f} "
            f"pair_n={pair_metrics['aligned_steps']}"
        )
    ax.set_title(
        f"{label} {Path(map_name).stem}\n"
        f"frame={frame_idx + 1}/{len(active_result['rollout_x'])} "
        f"steps={active_result['num_steps']} "
        f"cost={active_result['match_cost_total']:.2f} "
        f"mean={active_result['self_ade']:.2f} "
        f"final={active_result['self_fde']:.2f}"
        f"{pair_line}",
        fontsize=9,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def _save_static_grid(
    sampled_pairs: list[tuple[str, dict]],
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(len(sampled_pairs), 2, figsize=(12, max(4, 3.5 * len(sampled_pairs))))
    if len(sampled_pairs) == 1:
        axes = np.asarray([axes])

    for row_idx, (map_name, pair) in enumerate(sampled_pairs):
        car_result = pair["car"]
        truck_result = pair["truck"]
        pair_metrics = _pair_metrics(car_result, truck_result)
        limits = _shared_limits(car_result, truck_result)

        for col_idx, (label, result) in enumerate((("car", car_result), ("truck", truck_result))):
            ax = axes[row_idx, col_idx]
            _draw_fit_panel(
                ax,
                label=label,
                map_name=map_name,
                active_result=result,
                car_result=car_result,
                truck_result=truck_result,
                pair_metrics=pair_metrics,
                frame_idx=len(result["rollout_x"]) - 1,
                limits=limits,
            )
            if row_idx == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_snippet_animation(
    *,
    map_name: str,
    pair: dict,
    output_path: Path,
    fps: int,
    interval_ms: int,
) -> Path:
    car_result = pair["car"]
    truck_result = pair["truck"]
    pair_metrics = _pair_metrics(car_result, truck_result)
    limits = _shared_limits(car_result, truck_result)
    frame_count = max(1, min(len(car_result["rollout_x"]), len(truck_result["rollout_x"])))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    def update(frame_idx: int):
        _draw_fit_panel(
            axes[0],
            label="car",
            map_name=map_name,
            active_result=car_result,
            car_result=car_result,
            truck_result=truck_result,
            pair_metrics=pair_metrics,
            frame_idx=frame_idx,
            limits=limits,
        )
        _draw_fit_panel(
            axes[1],
            label="truck",
            map_name=map_name,
            active_result=truck_result,
            car_result=car_result,
            truck_result=truck_result,
            pair_metrics=pair_metrics,
            frame_idx=frame_idx,
            limits=limits,
        )
        if frame_idx == 0:
            axes[0].legend(fontsize=8, loc="best")
            axes[1].legend(fontsize=8, loc="best")
        return axes

    ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=interval_ms, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def _json_safe_pair_metrics(car_result: dict, truck_result: dict) -> dict[str, object]:
    pair_metrics = _pair_metrics(car_result, truck_result)
    if pair_metrics.get("status") != "ok":
        return {"status": pair_metrics.get("status", "unavailable")}
    return {
        "status": "ok",
        "aligned_steps": int(pair_metrics["aligned_steps"]),
        "ade": float(pair_metrics["ade"]),
        "fde": float(pair_metrics["fde"]),
    }


def visualize_paired_offline_fit_snippets(
    export_path: Path,
    output_dir: Path,
    *,
    max_snippets: int = DEFAULT_MAX_SNIPPETS,
    fps: int = DEFAULT_FPS,
    interval_ms: int = DEFAULT_INTERVAL_MS,
) -> dict[str, object]:
    sampled_pairs = _select_pairs(export_path, max_snippets=max_snippets)
    if not sampled_pairs:
        raise ValueError(f"No successful map pairs found in {export_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_items = []

    for idx, (map_name, pair) in enumerate(sampled_pairs, start=1):
        stem = Path(map_name).stem
        gif_path = output_dir / f"{idx:02d}_{stem}_fit.gif"
        _save_snippet_animation(
            map_name=map_name,
            pair=pair,
            output_path=gif_path,
            fps=fps,
            interval_ms=interval_ms,
        )
        summary_items.append(
            {
                "index": idx,
                "map_name": map_name,
                "gif_path": str(gif_path),
                "pair_metrics": _json_safe_pair_metrics(pair["car"], pair["truck"]),
                "car_steps": int(len(pair["car"]["rollout_x"])),
                "truck_steps": int(len(pair["truck"]["rollout_x"])),
            }
        )

    grid_path = output_dir / "all_examples_fit_grid.png"
    _save_static_grid(sampled_pairs, grid_path)

    summary = {
        "export_path": str(export_path),
        "output_dir": str(output_dir),
        "snippet_count": len(sampled_pairs),
        "grid_path": str(grid_path),
        "items": summary_items,
    }
    summary_path = output_dir / "visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a small set of paired offline fit snippet animations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-snippets", type=int, default=DEFAULT_MAX_SNIPPETS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    args = parser.parse_args()

    export_path = args.input.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else _default_output_dir(export_path)
    summary = visualize_paired_offline_fit_snippets(
        export_path=export_path,
        output_dir=output_dir,
        max_snippets=max(1, int(args.max_snippets)),
        fps=max(1, int(args.fps)),
        interval_ms=max(1, int(args.interval_ms)),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
