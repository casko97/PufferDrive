import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


TRACTOR = {
    "center": (0.0, 0.0),
    "heading": 0.0,
    "length": 6.3,
    "width": 2.55,
}
TRAILER = {
    "center": (-6.3, 0.0),
    "heading": 0.0,
    "length": 12.5,
    "width": 2.55,
}
OBSTACLE = {
    "center": (-10.0, 0.0),
    "heading": 0.0,
    "length": 2.0,
    "width": 2.0,
}

TRACTOR_COLOR = "#e8a35b"
TRACTOR_EDGE = "#8b4513"
TRAILER_COLOR = "#7aa6c2"
TRAILER_EDGE = "#174a68"
COLLISION_COLOR = "#d9534f"
SAFE_COLOR = "#2f855a"
OBSERVATION_COLOR = "#237a57"
HIDDEN_COLOR = "#9b2c2c"


def compile_and_run_native_interaction_test(repo_root):
    source = (
        repo_root
        / "tests"
        / "drive"
        / "test_tractor_trailer_interactions.c"
    )
    raylib_include = repo_root / "raylib-5.5_linux_amd64" / "include"
    with tempfile.TemporaryDirectory() as temporary_directory:
        executable = Path(temporary_directory) / "tractor_trailer_interactions"
        subprocess.run(
            [
                "gcc",
                "-Wall",
                "-Wextra",
                "-Wno-unused-parameter",
                "-Wno-unused-variable",
                "-std=gnu99",
                "-ffunction-sections",
                "-fdata-sections",
                f"-I{repo_root}",
                f"-I{raylib_include}",
                str(source),
                "-Wl,--gc-sections",
                "-lm",
                "-o",
                str(executable),
            ],
            check=True,
        )
        result = subprocess.run(
            [str(executable)], check=True, capture_output=True, text=True
        )
    return result.stdout.strip()


def rectangle_corners(entity):
    center_x, center_y = entity["center"]
    heading = entity["heading"]
    cos_heading = math.cos(heading)
    sin_heading = math.sin(heading)
    corners = []
    for longitudinal, lateral in (
        (entity["length"] / 2, entity["width"] / 2),
        (entity["length"] / 2, -entity["width"] / 2),
        (-entity["length"] / 2, -entity["width"] / 2),
        (-entity["length"] / 2, entity["width"] / 2),
    ):
        corners.append(
            (
                center_x + longitudinal * cos_heading - lateral * sin_heading,
                center_y + longitudinal * sin_heading + lateral * cos_heading,
            )
        )
    return corners


def draw_entity(
    ax,
    entity,
    face_color,
    edge_color,
    label,
    alpha=0.58,
    linewidth=2.0,
):
    ax.add_patch(
        Polygon(
            rectangle_corners(entity),
            closed=True,
            facecolor=face_color,
            edgecolor=edge_color,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )
    )
    center_x, center_y = entity["center"]
    ax.scatter([center_x], [center_y], color=edge_color, s=28, zorder=6)


def draw_rig(ax, collided=False):
    if collided:
        draw_entity(
            ax,
            TRAILER,
            COLLISION_COLOR,
            "#8b1a1a",
            "trailer: inherited collision style",
        )
        draw_entity(
            ax,
            TRACTOR,
            COLLISION_COLOR,
            "#8b1a1a",
            "tractor: aggregated collision",
        )
    else:
        draw_entity(
            ax,
            TRAILER,
            TRAILER_COLOR,
            TRAILER_EDGE,
            "trailer rectangle",
        )
        draw_entity(
            ax,
            TRACTOR,
            TRACTOR_COLOR,
            TRACTOR_EDGE,
            "tractor rectangle",
        )

    tractor_hitch = (-2.25, 0.0)
    trailer_hitch = (-2.25, 0.0)
    ax.scatter(
        [tractor_hitch[0]],
        [tractor_hitch[1]],
        color="#7d1f5c",
        marker="D",
        s=42,
        zorder=8,
    )
    ax.scatter(
        [trailer_hitch[0]],
        [trailer_hitch[1]],
        facecolors="none",
        edgecolors="#20734d",
        linewidths=1.5,
        s=78,
        zorder=9,
    )


def configure_geometry_axis(ax, title):
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-14.5, 4.5)
    ax.set_ylim(-4.0, 4.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", alpha=0.4)


def draw_mutual_overlap_panel(ax):
    draw_rig(ax, collided=False)
    overlap_rear = -3.15
    overlap_front = -0.05
    ax.add_patch(
        Rectangle(
            (overlap_rear, -1.275),
            overlap_front - overlap_rear,
            2.55,
            facecolor="#b45f9d",
            edgecolor="#6b2f61",
            hatch="//",
            alpha=0.28,
            label="intentional box overlap",
        )
    )
    ax.text(
        0.02,
        0.97,
        "Native check:\ntractor ↔ trailer overlap = true\n"
        "linked-pair collision result = ignored\ncollision state = 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": SAFE_COLOR, "alpha": 0.92},
    )
    ax.annotate(
        "ignored only for linked pair",
        xy=(-1.6, 0.0),
        xytext=(-8.5, 2.7),
        arrowprops={"arrowstyle": "->", "color": SAFE_COLOR},
        color=SAFE_COLOR,
        fontsize=8,
    )
    configure_geometry_axis(ax, "A — Mutual tractor/trailer overlap")


def draw_trailer_collision_panel(ax):
    draw_rig(ax, collided=True)
    draw_entity(
        ax,
        OBSTACLE,
        "#4a5568",
        "#1a202c",
        "third-party vehicle",
        alpha=0.8,
    )
    ax.annotate(
        "third vehicle contacts\nonly the trailer",
        xy=OBSTACLE["center"],
        xytext=(-13.8, 2.7),
        arrowprops={"arrowstyle": "->", "color": "#1a202c"},
        fontsize=8,
    )
    ax.text(
        0.98,
        0.97,
        "Aggregated result\ntractor: VEHICLE_COLLISION\n"
        "trailer: VEHICLE_COLLISION\nSTOP/REMOVE applies to both",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": COLLISION_COLOR, "alpha": 0.92},
    )
    configure_geometry_axis(ax, "B — Trailer-only vehicle collision")


def draw_offroad_panel(ax):
    draw_rig(ax, collided=True)
    ax.plot(
        [-10.0, -10.0],
        [-3.2, 3.2],
        color="#222222",
        linewidth=3,
        label="road edge",
    )
    ax.fill_betweenx(
        [-3.2, 3.2],
        -14.5,
        -10.0,
        color="#d9c6a5",
        alpha=0.35,
        hatch="..",
        label="off-road side",
    )
    ax.annotate(
        "road edge intersects only\nthe trailer rectangle",
        xy=(-10.0, -1.275),
        xytext=(-13.9, -3.1),
        arrowprops={"arrowstyle": "->", "color": HIDDEN_COLOR},
        color=HIDDEN_COLOR,
        fontsize=8,
    )
    ax.text(
        0.98,
        0.97,
        "Aggregated result\ntractor: OFFROAD\ntrailer: OFFROAD\n"
        "off-road STOP/REMOVE applies to both",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": COLLISION_COLOR, "alpha": 0.92},
    )
    configure_geometry_axis(ax, "C — Trailer-only off-road contact")


def observation_arrow(ax, start, end, color, label, offset=0.0):
    start_point = (start[0], start[1] + offset)
    end_point = (end[0], end[1] + offset)
    ax.annotate(
        "",
        xy=end_point,
        xytext=start_point,
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.8},
    )
    midpoint = (
        0.5 * (start_point[0] + end_point[0]),
        0.5 * (start_point[1] + end_point[1]),
    )
    ax.text(midpoint[0], midpoint[1] + 0.25, label, color=color, fontsize=8, ha="center")


def draw_observation_render_panel(ax):
    draw_rig(ax, collided=False)
    observer = {
        "center": (10.0, 0.0),
        "heading": math.pi,
        "length": 4.5,
        "width": 2.0,
    }
    draw_entity(
        ax,
        observer,
        "#90cdf4",
        "#2b6cb0",
        "other controlled agent",
        alpha=0.72,
    )
    observation_arrow(
        ax,
        observer["center"],
        TRACTOR["center"],
        OBSERVATION_COLOR,
        "observes tractor",
        offset=0.75,
    )
    observation_arrow(
        ax,
        observer["center"],
        TRAILER["center"],
        OBSERVATION_COLOR,
        "observes trailer",
        offset=-0.75,
    )
    ax.plot(
        [TRACTOR["center"][0], TRAILER["center"][0]],
        [TRACTOR["center"][1], TRAILER["center"][1]],
        color=HIDDEN_COLOR,
        linestyle="--",
        linewidth=1.8,
    )
    ax.scatter(
        [-3.15],
        [0.0],
        marker="x",
        color=HIDDEN_COLOR,
        s=80,
        linewidths=2,
        zorder=10,
    )
    ax.text(
        -3.15,
        -0.8,
        "owner hides attached trailer",
        color=HIDDEN_COLOR,
        fontsize=8,
        ha="center",
    )
    ax.text(
        0.02,
        0.97,
        "Policy/action count: 1\nActor count for rig: 2\n"
        "Trailer reserves one MAX_AGENTS slot\n"
        "Render style: both blue; both red on collision\n"
        "Trailer remains a separate rectangle",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#2b6cb0", "alpha": 0.92},
    )
    ax.set_title("D — Observations, capacity, and rendering", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-14.5, 13.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", alpha=0.4)


def generate_interaction_visualization(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    native_test_output = compile_and_run_native_interaction_test(repo_root)

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    draw_mutual_overlap_panel(axes[0, 0])
    draw_trailer_collision_panel(axes[0, 1])
    draw_offroad_panel(axes[1, 0])
    draw_observation_render_panel(axes[1, 1])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.suptitle(
        "Stage 4 tractor–trailer interaction tests\n"
        "collision aggregation, off-road geometry, observation ownership, and render state",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    output_path = output_dir / "tractor_trailer_interactions.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    manifest = {
        "source": "native test_tractor_trailer_interactions.c fixture",
        "native_test_output": native_test_output,
        "artifact": output_path.name,
        "panels": {
            "mutual_overlap": "linked tractor/trailer overlap ignored",
            "trailer_collision": "trailer-only vehicle collision aggregated",
            "trailer_offroad": "trailer-only road-edge contact aggregated",
            "observation_rendering": (
                "owner hides trailer; other agent sees both; trailer inherits render state"
            ),
        },
        "actor_capacity": {
            "policy_actions_for_rig": 1,
            "actors_for_rig": 2,
            "maximum_actors_verified": 32,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output_path, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Stage 4 tractor-trailer interaction tests"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/tractor_trailer_interactions"),
    )
    args = parser.parse_args()
    output_path, manifest_path = generate_interaction_visualization(args.output_dir)
    print(output_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
