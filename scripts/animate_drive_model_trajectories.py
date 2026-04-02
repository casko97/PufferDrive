#!/usr/bin/env python3
import argparse
import gc
import math
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from pufferlib import pufferl
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator
from scripts.run_packaged_drive_human_replay_eval import (
    _load_packaged_config,
    _overlay_args,
    _prepare_single_map_dir,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SANDY_COLOR = "#0B6E4F"
APRICOT_COLOR = "#C84C09"
GT_COLOR = "#1F3A5F"


def build_args(config_path: Path, checkpoint_path: Path, map_path: Path, device: str) -> tuple[dict, Path]:
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        base_args = pufferl.load_config("puffer_drive")
    finally:
        sys.argv = original_argv

    packaged = _load_packaged_config(config_path)
    args = _overlay_args(base_args, packaged)
    args["load_model_path"] = str(checkpoint_path)
    args["load_id"] = None
    args["wandb"] = False
    args["neptune"] = False
    args["train"]["device"] = device
    args["train"]["compile"] = False
    args["train"]["render"] = False
    args["eval"]["wosac_num_rollouts"] = 1
    args["vec"] = {"backend": "PufferEnv", "num_envs": 1}

    temp_root = Path(tempfile.mkdtemp(prefix=f"{map_path.stem}_traj_", dir=str(REPO_ROOT / "tmp")))
    single_map_dir = _prepare_single_map_dir(map_path, temp_root)
    args["env"]["map_dir"] = str(single_map_dir)
    args["env"]["num_maps"] = 1
    args["eval"]["map_dir"] = str(single_map_dir)
    args["eval"]["wosac_num_maps"] = 1
    if args["eval"].get("human_replay_eval"):
        args["env"]["control_mode"] = args["eval"]["human_replay_control_mode"]
        args["env"]["episode_length"] = 91

    return args, temp_root


def collect_rollout(config_path: Path, checkpoint_path: Path, map_path: Path, device: str) -> dict:
    args, temp_root = build_args(config_path, checkpoint_path, map_path, device)
    vecenv = None
    try:
        vecenv = pufferl.load_env("puffer_drive", args)
        policy = pufferl.load_policy(args, vecenv, env_name="puffer_drive")
        policy.eval()

        evaluator = WOSACEvaluator(args)
        gt_traj = evaluator.collect_ground_truth_trajectories(vecenv)
        sim_traj = evaluator.collect_simulated_trajectories(args, vecenv, policy)
        road_edges = vecenv.driver_env.get_road_edge_polylines()

        sim_agent_id = int(sim_traj["id"][0, 0, 0])
        gt_agent_ids = gt_traj["id"][:, 0]
        matches = np.flatnonzero(gt_agent_ids == sim_agent_id)
        controlled_idx = int(matches[0]) if matches.size > 0 else 0

        scenario_id = int(gt_traj["scenario_id"][controlled_idx, 0])
        return {
            "map_name": map_path.name,
            "scenario_id": scenario_id,
            "gt_x": gt_traj["x"][controlled_idx, 0],
            "gt_y": gt_traj["y"][controlled_idx, 0],
            "sim_x": sim_traj["x"][controlled_idx, 0],
            "sim_y": sim_traj["y"][controlled_idx, 0],
            "road_edges": road_edges,
        }
    finally:
        if vecenv is not None:
            vecenv.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        gc.collect()


def plot_road_edges(ax, road_edge_polylines, scenario_id):
    lengths = road_edge_polylines["lengths"]
    scenario_ids = road_edge_polylines["scenario_id"]
    x = road_edge_polylines["x"]
    y = road_edge_polylines["y"]

    pt_idx = 0
    for length, road_scenario_id in zip(lengths, scenario_ids):
        if road_scenario_id == scenario_id:
            poly_x = x[pt_idx : pt_idx + length]
            poly_y = y[pt_idx : pt_idx + length]
            ax.plot(poly_x, poly_y, color="#666666", linewidth=0.6, alpha=0.5, zorder=0)
        pt_idx += length


def scenario_bounds(sandy: dict, apricot: dict):
    xs = []
    ys = []
    for data in (sandy, apricot):
        xs.extend(data["sim_x"].tolist())
        ys.extend(data["sim_y"].tolist())
        xs.extend(data["gt_x"].tolist())
        ys.extend(data["gt_y"].tolist())

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = max(5.0, 0.08 * max(max_x - min_x, max_y - min_y, 1.0))
    return (min_x - pad, max_x + pad), (min_y - pad, max_y + pad)


def collect_scenarios(map_paths: list[Path], sandy_config: Path, sandy_checkpoint: Path, apricot_config: Path, apricot_checkpoint: Path, device: str):
    scenarios = []
    for map_path in map_paths:
        sandy = collect_rollout(sandy_config, sandy_checkpoint, map_path, device)
        apricot = collect_rollout(apricot_config, apricot_checkpoint, map_path, device)
        scenarios.append(
            {
                "map_name": map_path.name,
                "scenario_id": sandy["scenario_id"],
                "sandy": sandy,
                "apricot": apricot,
                "xlim": scenario_bounds(sandy, apricot)[0],
                "ylim": scenario_bounds(sandy, apricot)[1],
                "frames": min(len(sandy["sim_x"]), len(apricot["sim_x"])),
            }
        )
    return scenarios


def make_overlay_animation(scenario: dict, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    plot_road_edges(ax, scenario["sandy"]["road_edges"], scenario["scenario_id"])
    ax.plot(
        scenario["sandy"]["gt_x"],
        scenario["sandy"]["gt_y"],
        linestyle="--",
        color=GT_COLOR,
        linewidth=1.2,
        alpha=0.8,
        label="human replay",
        zorder=1,
    )

    sandy_line, = ax.plot([], [], color=SANDY_COLOR, linewidth=2.3, label="sandy", zorder=2)
    sandy_dot, = ax.plot([], [], "o", color=SANDY_COLOR, markersize=5, zorder=3)
    apricot_line, = ax.plot([], [], color=APRICOT_COLOR, linewidth=2.3, label="apricot", zorder=2)
    apricot_dot, = ax.plot([], [], "o", color=APRICOT_COLOR, markersize=5, zorder=3)

    ax.set_aspect("equal")
    ax.set_xlim(*scenario["xlim"])
    ax.set_ylim(*scenario["ylim"])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(scenario["map_name"])
    ax.legend(loc="upper right")

    def init():
        sandy_line.set_data([], [])
        sandy_dot.set_data([], [])
        apricot_line.set_data([], [])
        apricot_dot.set_data([], [])
        return [sandy_line, sandy_dot, apricot_line, apricot_dot]

    def update(frame_idx):
        sx = scenario["sandy"]["sim_x"][: frame_idx + 1]
        sy = scenario["sandy"]["sim_y"][: frame_idx + 1]
        axx = scenario["apricot"]["sim_x"][: frame_idx + 1]
        axy = scenario["apricot"]["sim_y"][: frame_idx + 1]

        sandy_line.set_data(sx, sy)
        sandy_dot.set_data([sx[-1]], [sy[-1]])
        apricot_line.set_data(axx, axy)
        apricot_dot.set_data([axx[-1]], [axy[-1]])
        fig.suptitle(f"{title}\n{scenario['map_name']} | t = {frame_idx / 10:.1f}s", fontsize=13)
        return [sandy_line, sandy_dot, apricot_line, apricot_dot]

    anim = FuncAnimation(fig, update, frames=scenario["frames"], init_func=init, interval=100, blit=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=10))
    plt.close(fig)


def make_grid_animation(scenarios: list[dict], output_path: Path, title: str, columns: int):
    columns = max(1, min(columns, len(scenarios)))
    rows = math.ceil(len(scenarios) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5.5 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, columns)

    panel_artists = []
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // columns, idx % columns]
        plot_road_edges(ax, scenario["sandy"]["road_edges"], scenario["scenario_id"])
        ax.plot(
            scenario["sandy"]["gt_x"],
            scenario["sandy"]["gt_y"],
            linestyle="--",
            color=GT_COLOR,
            linewidth=1.0,
            alpha=0.8,
            label="human replay",
            zorder=1,
        )
        sandy_line, = ax.plot([], [], color=SANDY_COLOR, linewidth=2.0, label="sandy", zorder=2)
        sandy_dot, = ax.plot([], [], "o", color=SANDY_COLOR, markersize=4, zorder=3)
        apricot_line, = ax.plot([], [], color=APRICOT_COLOR, linewidth=2.0, label="apricot", zorder=2)
        apricot_dot, = ax.plot([], [], "o", color=APRICOT_COLOR, markersize=4, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(*scenario["xlim"])
        ax.set_ylim(*scenario["ylim"])
        ax.set_title(scenario["map_name"], fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
        panel_artists.append((scenario, sandy_line, sandy_dot, apricot_line, apricot_dot))

    for idx in range(len(scenarios), rows * columns):
        axes[idx // columns, idx % columns].axis("off")

    max_frames = min(s["frames"] for s in scenarios)

    def init():
        updated = []
        for _, sandy_line, sandy_dot, apricot_line, apricot_dot in panel_artists:
            sandy_line.set_data([], [])
            sandy_dot.set_data([], [])
            apricot_line.set_data([], [])
            apricot_dot.set_data([], [])
            updated.extend([sandy_line, sandy_dot, apricot_line, apricot_dot])
        return updated

    def update(frame_idx):
        updated = []
        for scenario, sandy_line, sandy_dot, apricot_line, apricot_dot in panel_artists:
            sx = scenario["sandy"]["sim_x"][: frame_idx + 1]
            sy = scenario["sandy"]["sim_y"][: frame_idx + 1]
            axx = scenario["apricot"]["sim_x"][: frame_idx + 1]
            axy = scenario["apricot"]["sim_y"][: frame_idx + 1]
            sandy_line.set_data(sx, sy)
            sandy_dot.set_data([sx[-1]], [sy[-1]])
            apricot_line.set_data(axx, axy)
            apricot_dot.set_data([axx[-1]], [axy[-1]])
            updated.extend([sandy_line, sandy_dot, apricot_line, apricot_dot])
        fig.suptitle(f"{title}\nt = {frame_idx / 10:.1f}s", fontsize=14)
        return updated

    anim = FuncAnimation(fig, update, frames=max_frames, init_func=init, interval=100, blit=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=10))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Animate PufferDrive model trajectories with overlay or scenario-grid layouts.")
    parser.add_argument("--map-path", type=Path, action="append", required=True)
    parser.add_argument("--sandy-config", type=Path, required=True)
    parser.add_argument("--sandy-checkpoint", type=Path, required=True)
    parser.add_argument("--apricot-config", type=Path, required=True)
    parser.add_argument("--apricot-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--title", type=str, default="Trajectory Comparison")
    parser.add_argument("--layout", choices=["overlay", "grid"], default="overlay")
    parser.add_argument("--columns", type=int, default=2)
    args = parser.parse_args()

    (REPO_ROOT / "tmp").mkdir(exist_ok=True)

    map_paths = [path.resolve() for path in args.map_path]
    scenarios = collect_scenarios(
        map_paths=map_paths,
        sandy_config=args.sandy_config.resolve(),
        sandy_checkpoint=args.sandy_checkpoint.resolve(),
        apricot_config=args.apricot_config.resolve(),
        apricot_checkpoint=args.apricot_checkpoint.resolve(),
        device=args.device,
    )

    if args.layout == "overlay":
        if len(scenarios) != 1:
            raise ValueError("Overlay layout expects exactly one --map-path")
        make_overlay_animation(scenarios[0], args.output.resolve(), args.title)
    else:
        make_grid_animation(scenarios, args.output.resolve(), args.title, args.columns)


if __name__ == "__main__":
    main()
