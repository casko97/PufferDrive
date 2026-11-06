import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon

from pufferlib.ocean.drive.drive import Drive


SCENARIO_LABELS = {
    "straight_forward": "Straight forward",
    "constant_steering": "Constant steering",
    "reverse_steering": "Signed reverse steering",
}
COLORS = {
    "tractor": ("#e8a35b", "#8b4513"),
    "trailer": ("#7aa6c2", "#174a68"),
    "hitch": "#8c2d70",
}


def make_env(repo_root):
    return Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(
            repo_root / "pufferlib" / "resources" / "drive" / "binaries" / "training"
        ),
        control_mode="control_sdc_only",
        max_controlled_agents=1,
        ego_vehicle_type="tractor_trailer",
        action_type="continuous",
        dynamics_model="classic",
        collision_behavior=0,
        offroad_behavior=0,
        dt=0.1,
        episode_length=91,
        resample_frequency=0,
        seed=7,
    )


def control_for(scenario, step):
    if scenario == "straight_forward":
        return (0.0, 0.0)
    if scenario == "constant_steering":
        return (0.0, 0.18)
    if scenario == "reverse_steering":
        if step < 20:
            return (-1.0, 0.0)
        return (0.0, 0.22)
    raise ValueError(f"unknown scenario: {scenario}")


def state_frame(state, time_seconds):
    row = 0
    return {
        "time": time_seconds,
        "tractor_center": state["tractor_center_pose"][row].tolist(),
        "trailer_center": state["trailer_center_pose"][row].tolist(),
        "tractor_axle": state["tractor_rear_axle_pose"][row].tolist(),
        "trailer_axle": state["trailer_axle_pose"][row].tolist(),
        "hitch": state["hitch_position"][row].tolist(),
        "tractor_velocity": state["tractor_velocity"][row].tolist(),
        "trailer_velocity": state["trailer_velocity"][row].tolist(),
        "speed": float(state["speed"][row]),
        "steering": float(state["steering_angle"][row]),
        "articulation": float(state["articulation_angle"][row]),
        "tractor_dimensions": state["tractor_dimensions"][row].tolist(),
        "trailer_dimensions": state["trailer_dimensions"][row].tolist(),
        "valid": bool(state["valid"][row]),
    }


def trailer_hitch(frame):
    x, y, heading = frame["trailer_axle"]
    return np.array(
        [
            x + 8.3 * math.cos(heading),
            y + 8.3 * math.sin(heading),
        ]
    )


def hitch_error(frame):
    return float(
        np.linalg.norm(
            np.asarray(frame["hitch"][:2], dtype=np.float64)
            - trailer_hitch(frame)
        )
    )


def collect_rollout(repo_root, scenario, steps=40):
    env = make_env(repo_root)
    frames = []
    try:
        env.reset(seed=7)
        for step in range(steps + 1):
            state = env.get_articulated_ego_state()
            if state["valid"].shape != (1,) or not state["valid"][0]:
                raise RuntimeError("bundled map did not initialize an articulated SDC")
            frames.append(state_frame(state, step * env.dt))
            if step < steps:
                action = np.asarray(
                    [control_for(scenario, step)], dtype=np.float32
                )
                env.step(action)
    finally:
        env.close()

    origin = np.asarray(frames[0]["tractor_axle"][:2], dtype=np.float64)
    for frame in frames:
        for key in (
            "tractor_center",
            "trailer_center",
            "tractor_axle",
            "trailer_axle",
            "hitch",
        ):
            frame[key][0] -= float(origin[0])
            frame[key][1] -= float(origin[1])
        frame["hitch_error"] = hitch_error(frame)
    return frames


def rectangle_corners(center_pose, dimensions):
    x, y, _, heading = center_pose
    length, width, _ = dimensions
    cos_heading = math.cos(heading)
    sin_heading = math.sin(heading)
    corners = []
    for longitudinal, lateral in (
        (length / 2, width / 2),
        (length / 2, -width / 2),
        (-length / 2, -width / 2),
        (-length / 2, width / 2),
    ):
        corners.append(
            (
                x + longitudinal * cos_heading - lateral * sin_heading,
                y + longitudinal * sin_heading + lateral * cos_heading,
            )
        )
    return corners


def draw_rig(ax, frame, alpha=0.75, annotate=False):
    for entity, dimensions, color_key in (
        ("trailer_center", "trailer_dimensions", "trailer"),
        ("tractor_center", "tractor_dimensions", "tractor"),
    ):
        face, edge = COLORS[color_key]
        ax.add_patch(
            Polygon(
                rectangle_corners(frame[entity], frame[dimensions]),
                closed=True,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.4,
                alpha=alpha,
            )
        )

    tractor_axle = frame["tractor_axle"]
    trailer_axle = frame["trailer_axle"]
    hitch = frame["hitch"]
    derived_trailer_hitch = trailer_hitch(frame)
    ax.scatter(
        [tractor_axle[0], trailer_axle[0]],
        [tractor_axle[1], trailer_axle[1]],
        color=[COLORS["tractor"][1], COLORS["trailer"][1]],
        s=20,
        zorder=6,
    )
    ax.scatter(
        [hitch[0]],
        [hitch[1]],
        color=COLORS["hitch"],
        marker="D",
        s=28,
        zorder=8,
    )
    ax.scatter(
        [derived_trailer_hitch[0]],
        [derived_trailer_hitch[1]],
        facecolors="none",
        edgecolors="#20734d",
        linewidths=1.4,
        s=65,
        zorder=9,
    )
    if annotate:
        ax.annotate(
            "shared hitch",
            hitch[:2],
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7,
            color=COLORS["hitch"],
        )
        ax.annotate(
            "rear axle",
            tractor_axle[:2],
            xytext=(7, -13),
            textcoords="offset points",
            fontsize=7,
            color=COLORS["tractor"][1],
        )
        ax.annotate(
            "trailer axle",
            trailer_axle[:2],
            xytext=(7, -13),
            textcoords="offset points",
            fontsize=7,
            color=COLORS["trailer"][1],
        )


def rollout_limits(frames):
    points = []
    for frame in frames:
        points.extend(
            rectangle_corners(
                frame["tractor_center"], frame["tractor_dimensions"]
            )
        )
        points.extend(
            rectangle_corners(
                frame["trailer_center"], frame["trailer_dimensions"]
            )
        )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    margin = 2.0
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def configure_axis(ax, frames, title):
    x_min, x_max, y_min, y_max = rollout_limits(frames)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("relative world x [m]")
    ax.set_ylabel("relative world y [m]")


def draw_paths(ax, frames, through_frame=None):
    visible = frames if through_frame is None else frames[: through_frame + 1]
    ax.plot(
        [frame["tractor_axle"][0] for frame in visible],
        [frame["tractor_axle"][1] for frame in visible],
        color=COLORS["tractor"][1],
        linewidth=1.5,
        label="tractor rear axle",
    )
    ax.plot(
        [frame["trailer_axle"][0] for frame in visible],
        [frame["trailer_axle"][1] for frame in visible],
        color=COLORS["trailer"][1],
        linewidth=1.5,
        label="trailer axle",
    )


def add_metrics(ax, frame):
    ax.text(
        0.02,
        0.98,
        f"actual Drive.step(), t={frame['time']:.1f} s\n"
        f"v={frame['speed']:+.2f} m/s, steering={frame['steering']:+.2f} rad\n"
        f"articulation={math.degrees(frame['articulation']):+.1f}°\n"
        f"hitch residual={frame['hitch_error']:.2e} m",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.9},
    )


def create_overview(scenarios, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    snapshots = (0, 10, 20, 30, 40)
    for ax, (name, frames) in zip(axes, scenarios.items()):
        draw_paths(ax, frames)
        for index, frame_index in enumerate(snapshots):
            draw_rig(
                ax,
                frames[frame_index],
                alpha=0.18 + 0.14 * index,
                annotate=frame_index == snapshots[-1],
            )
        add_metrics(ax, frames[-1])
        configure_axis(ax, frames, SCENARIO_LABELS[name])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(
        "Actual simulator tractor–trailer rollouts\n"
        "Drive.step() + get_articulated_ego_state(), 41 frames each",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_animation(name, frames, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    writer = PillowWriter(fps=10)
    with writer.saving(fig, output_path, dpi=100):
        for frame_index, frame in enumerate(frames):
            ax.clear()
            configure_axis(ax, frames, SCENARIO_LABELS[name])
            draw_paths(ax, frames, through_frame=frame_index)
            draw_rig(ax, frame, annotate=True)
            add_metrics(ax, frame)
            ax.legend(loc="lower right", fontsize=7)
            writer.grab_frame()
    plt.close(fig)


def generate_visualizations(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    scenarios = {
        name: collect_rollout(repo_root, name)
        for name in SCENARIO_LABELS
    }

    overview = output_dir / "tractor_trailer_simulator_rollouts.png"
    create_overview(scenarios, overview)
    gifs = {}
    for name, frames in scenarios.items():
        path = output_dir / f"{name}.gif"
        create_animation(name, frames, path)
        gifs[name] = path

    manifest = {
        "source": (
            "actual Drive.step() simulator rollout via "
            "get_articulated_ego_state()"
        ),
        "map": "pufferlib/resources/drive/binaries/training/map_000.bin",
        "overview": overview.name,
        "coordinate_display": "world coordinates translated by initial rear axle",
        "annotations": [
            "oriented tractor and trailer entity bounding boxes",
            "tractor rear axle and trailer axle paths",
            "coincident exported and trailer-derived hitch markers",
            "speed, steering, articulation, and hitch residual",
        ],
        "scenarios": {
            name: {
                "gif": gifs[name].name,
                "frames": len(frames),
                "duration_seconds": frames[-1]["time"],
                "all_valid": all(frame["valid"] for frame in frames),
                "minimum_speed": min(frame["speed"] for frame in frames),
                "maximum_speed": max(frame["speed"] for frame in frames),
                "final_articulation": frames[-1]["articulation"],
                "max_hitch_error": max(
                    frame["hitch_error"] for frame in frames
                ),
            }
            for name, frames in scenarios.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return overview, gifs, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate actual Drive tractor-trailer rollout visualizations"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/tractor_trailer_simulator_rollouts"),
    )
    args = parser.parse_args()
    overview, gifs, manifest = generate_visualizations(args.output_dir)
    print(overview)
    for path in gifs.values():
        print(path)
    print(manifest)


if __name__ == "__main__":
    main()
