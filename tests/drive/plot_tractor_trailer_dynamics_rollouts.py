import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon


GEOMETRY = {
    "tractor_length": 6.3,
    "tractor_width": 2.55,
    "tractor_center_offset": 1.65,
    "tractor_hitch_offset": 0.6,
    "trailer_length": 12.5,
    "trailer_width": 2.55,
    "trailer_center_offset": 4.25,
    "trailer_hitch_offset": 8.3,
}

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


def compile_and_run_native_rollouts(repo_root):
    source = (
        repo_root
        / "tests"
        / "drive"
        / "generate_tractor_trailer_dynamics_rollouts.c"
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        executable = Path(temporary_directory) / "tractor_trailer_rollouts"
        subprocess.run(
            [
                "gcc",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-std=c99",
                f"-I{repo_root}",
                str(source),
                "-lm",
                "-o",
                str(executable),
            ],
            check=True,
        )
        result = subprocess.run(
            [str(executable)], check=True, capture_output=True, text=True
        )

    rows = list(csv.DictReader(result.stdout.splitlines()))
    scenarios = {}
    for row in rows:
        parsed = {
            "frame": int(row["frame"]),
            "time": float(row["time"]),
            "xr": float(row["xr"]),
            "yr": float(row["yr"]),
            "psi": float(row["psi"]),
            "xt": float(row["xt"]),
            "yt": float(row["yt"]),
            "psi_t": float(row["psi_t"]),
            "speed": float(row["speed"]),
            "steering": float(row["steering"]),
            "hitch_error": float(row["hitch_error"]),
        }
        scenarios.setdefault(row["scenario"], []).append(parsed)
    return scenarios


def offset_point(x, y, heading, distance):
    return x + distance * math.cos(heading), y + distance * math.sin(heading)


def entity_centers(frame):
    tractor_center = offset_point(
        frame["xr"],
        frame["yr"],
        frame["psi"],
        GEOMETRY["tractor_center_offset"],
    )
    trailer_center = offset_point(
        frame["xt"],
        frame["yt"],
        frame["psi_t"],
        GEOMETRY["trailer_center_offset"],
    )
    return tractor_center, trailer_center


def hitch_points(frame):
    tractor_hitch = offset_point(
        frame["xr"],
        frame["yr"],
        frame["psi"],
        -GEOMETRY["tractor_hitch_offset"],
    )
    trailer_hitch = offset_point(
        frame["xt"],
        frame["yt"],
        frame["psi_t"],
        GEOMETRY["trailer_hitch_offset"],
    )
    return tractor_hitch, trailer_hitch


def rectangle_corners(center, heading, length, width):
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
                center[0] + longitudinal * cos_heading - lateral * sin_heading,
                center[1] + longitudinal * sin_heading + lateral * cos_heading,
            )
        )
    return corners


def draw_rig(ax, frame, alpha=0.8, annotate=False):
    tractor_center, trailer_center = entity_centers(frame)
    tractor_hitch, trailer_hitch = hitch_points(frame)
    tractor_face, tractor_edge = COLORS["tractor"]
    trailer_face, trailer_edge = COLORS["trailer"]
    ax.add_patch(
        Polygon(
            rectangle_corners(
                trailer_center,
                frame["psi_t"],
                GEOMETRY["trailer_length"],
                GEOMETRY["trailer_width"],
            ),
            closed=True,
            facecolor=trailer_face,
            edgecolor=trailer_edge,
            linewidth=1.4,
            alpha=alpha,
        )
    )
    ax.add_patch(
        Polygon(
            rectangle_corners(
                tractor_center,
                frame["psi"],
                GEOMETRY["tractor_length"],
                GEOMETRY["tractor_width"],
            ),
            closed=True,
            facecolor=tractor_face,
            edgecolor=tractor_edge,
            linewidth=1.4,
            alpha=alpha,
        )
    )
    ax.scatter(
        [frame["xr"], frame["xt"]],
        [frame["yr"], frame["yt"]],
        color=[tractor_edge, trailer_edge],
        marker="o",
        s=22,
        zorder=6,
    )
    ax.plot(
        [tractor_hitch[0], trailer_hitch[0]],
        [tractor_hitch[1], trailer_hitch[1]],
        color=COLORS["hitch"],
        linewidth=2.0,
        zorder=7,
    )
    ax.scatter(
        [tractor_hitch[0]],
        [tractor_hitch[1]],
        color=COLORS["hitch"],
        marker="D",
        s=28,
        zorder=8,
    )
    ax.scatter(
        [trailer_hitch[0]],
        [trailer_hitch[1]],
        facecolors="none",
        edgecolors="#20734d",
        linewidths=1.4,
        s=65,
        zorder=9,
    )
    if annotate:
        ax.annotate(
            "shared hitch",
            tractor_hitch,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["hitch"],
        )
        ax.annotate(
            "tractor rear axle",
            (frame["xr"], frame["yr"]),
            xytext=(8, -13),
            textcoords="offset points",
            fontsize=7,
            color=tractor_edge,
        )
        ax.annotate(
            "trailer axle",
            (frame["xt"], frame["yt"]),
            xytext=(8, -13),
            textcoords="offset points",
            fontsize=7,
            color=trailer_edge,
        )


def scenario_limits(frames):
    points = []
    for frame in frames:
        tractor_center, trailer_center = entity_centers(frame)
        points.extend(
            rectangle_corners(
                tractor_center,
                frame["psi"],
                GEOMETRY["tractor_length"],
                GEOMETRY["tractor_width"],
            )
        )
        points.extend(
            rectangle_corners(
                trailer_center,
                frame["psi_t"],
                GEOMETRY["trailer_length"],
                GEOMETRY["trailer_width"],
            )
        )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    margin = 2.0
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def draw_paths(ax, frames, through_frame=None):
    visible = frames if through_frame is None else frames[: through_frame + 1]
    ax.plot(
        [frame["xr"] for frame in visible],
        [frame["yr"] for frame in visible],
        color=COLORS["tractor"][1],
        linewidth=1.5,
        label="tractor rear axle path",
    )
    ax.plot(
        [frame["xt"] for frame in visible],
        [frame["yt"] for frame in visible],
        color=COLORS["trailer"][1],
        linewidth=1.5,
        label="trailer axle path",
    )


def add_metrics(ax, frame):
    articulation = math.degrees(frame["psi_t"] - frame["psi"])
    time_seconds = frame["time"]
    speed = frame["speed"]
    steering = frame["steering"]
    hitch_error = frame["hitch_error"]
    ax.text(
        0.02,
        0.98,
        f"t = {time_seconds:.1f} s\n"
        f"v = {speed:.1f} m/s, steering = {steering:.2f} rad\n"
        f"articulation = {articulation:+.1f}°\n"
        f"hitch residual = {hitch_error:.2e} m",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.9},
    )


def configure_axis(ax, frames, title):
    x_min, x_max, y_min, y_max = scenario_limits(frames)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def create_overview(scenarios, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (scenario, frames) in zip(axes, scenarios.items()):
        draw_paths(ax, frames)
        snapshot_indices = [0, 10, 20, 30, 40]
        for snapshot_index in snapshot_indices:
            draw_rig(
                ax,
                frames[snapshot_index],
                alpha=0.18 + 0.16 * snapshot_indices.index(snapshot_index),
                annotate=snapshot_index == snapshot_indices[-1],
            )
        add_metrics(ax, frames[-1])
        configure_axis(ax, frames, SCENARIO_LABELS[scenario])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(
        "Native coupled RK2 tractor–trailer rollouts\n"
        "oriented entity boxes, axle paths, and coincident hitch markers",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_animation(scenario, frames, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    configure_axis(ax, frames, SCENARIO_LABELS[scenario])
    writer = PillowWriter(fps=10)
    with writer.saving(fig, output_path, dpi=100):
        for frame_index, frame in enumerate(frames):
            ax.clear()
            configure_axis(ax, frames, SCENARIO_LABELS[scenario])
            draw_paths(ax, frames, through_frame=frame_index)
            draw_rig(ax, frame, alpha=0.72, annotate=True)
            add_metrics(ax, frame)
            ax.legend(loc="lower right", fontsize=7)
            writer.grab_frame()
    plt.close(fig)


def generate_visualizations(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    scenarios = compile_and_run_native_rollouts(repo_root)
    expected_order = (
        "straight_forward",
        "constant_steering",
        "reverse_steering",
    )
    scenarios = {name: scenarios[name] for name in expected_order}

    overview_path = output_dir / "tractor_trailer_dynamics_rollouts.png"
    create_overview(scenarios, overview_path)
    gif_paths = {}
    for scenario, frames in scenarios.items():
        gif_path = output_dir / f"{scenario}.gif"
        create_animation(scenario, frames, gif_path)
        gif_paths[scenario] = gif_path

    manifest = {
        "source": "native tractor_trailer_dynamics.h RK2 rollout",
        "overview": overview_path.name,
        "annotations": [
            "oriented tractor and trailer bounding boxes",
            "tractor rear axle and trailer axle",
            "coincident tractor and trailer hitch markers",
            "articulation angle",
            "hitch constraint residual",
        ],
        "scenarios": {
            name: {
                "gif": gif_paths[name].name,
                "frames": len(frames),
                "duration_seconds": frames[-1]["time"],
                "max_hitch_error": max(frame["hitch_error"] for frame in frames),
            }
            for name, frames in scenarios.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return overview_path, gif_paths, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate native tractor-trailer dynamics rollout visualizations"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/tractor_trailer_dynamics_rollouts"),
    )
    args = parser.parse_args()
    overview_path, gif_paths, manifest_path = generate_visualizations(
        args.output_dir
    )
    print(overview_path)
    for gif_path in gif_paths.values():
        print(gif_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
