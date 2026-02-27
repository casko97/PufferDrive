#!/usr/bin/env python3
"""Generate a simple top-down schematic of tractor/trailer parameters."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon


def rect_corners(center_x, center_y, length, width):
    hl = 0.5 * length
    hw = 0.5 * width
    return [
        (center_x - hl, center_y - hw),
        (center_x + hl, center_y - hw),
        (center_x + hl, center_y + hw),
        (center_x - hl, center_y + hw),
    ]


def arrow_with_label(ax, p0, p1, text, text_offset=(0.0, 0.0), color="black", y_align="bottom"):
    arrow = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="<->",
        mutation_scale=10,
        linewidth=1.4,
        color=color,
    )
    ax.add_patch(arrow)
    mx = 0.5 * (p0[0] + p1[0]) + text_offset[0]
    my = 0.5 * (p0[1] + p1[1]) + text_offset[1]
    ax.text(mx, my, text, fontsize=9, color=color, ha="center", va=y_align)


def draw_schematic(output_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Nominal geometry for visualization only
    tractor_length = 6.0
    tractor_width = 2.4
    trailer_length = 10.0
    trailer_width = 2.5
    tractor2hitch = 1.0
    trailer2hitch = 1.5
    tractor_rear_axle2rear_bumper = 1.2
    tractor_rear_axle2front_axle = 3.8
    tractor_front_axle2front_bumper = 1.0
    trailer_rear_axle2rear_bumper = 1.6
    trailer_rear_axle2front_bumper = 8.4

    tractor_center_x = 0.0
    trailer_center_x = -8.6
    center_y = 0.0

    tractor = Polygon(
        rect_corners(tractor_center_x, center_y, tractor_length, tractor_width),
        closed=True,
        facecolor="tab:blue",
        edgecolor="tab:blue",
        alpha=0.25,
        linewidth=2,
    )
    trailer = Polygon(
        rect_corners(trailer_center_x, center_y, trailer_length, trailer_width),
        closed=True,
        facecolor="tab:orange",
        edgecolor="tab:orange",
        alpha=0.25,
        linewidth=2,
    )
    ax.add_patch(tractor)
    ax.add_patch(trailer)

    # Key x locations
    tr_rear = tractor_center_x - 0.5 * tractor_length
    tr_front = tractor_center_x + 0.5 * tractor_length
    tl_rear = trailer_center_x - 0.5 * trailer_length
    tl_front = trailer_center_x + 0.5 * trailer_length

    tr_hitch_x = tr_rear + tractor2hitch
    tl_hitch_x = tl_front - trailer2hitch

    tr_rear_axle_x = tr_rear + tractor_rear_axle2rear_bumper
    tr_front_axle_x = tr_rear_axle_x + tractor_rear_axle2front_axle
    tl_rear_axle_x = tl_rear + trailer_rear_axle2rear_bumper

    # Hitch points + linkage
    ax.plot([tr_hitch_x], [center_y], "o", color="tab:blue")
    ax.plot([tl_hitch_x], [center_y], "o", color="tab:orange")
    ax.plot([tl_hitch_x, tr_hitch_x], [center_y, center_y], "--", color="black", linewidth=1.2)
    ax.text(
        0.5 * (tr_hitch_x + tl_hitch_x),
        center_y + 0.25,
        "hitch linkage",
        fontsize=8,
        ha="center",
    )

    # Main lengths and widths
    arrow_with_label(
        ax,
        (tr_rear, 2.0),
        (tr_front, 2.0),
        "tractor_length",
        text_offset=(0.0, 0.15),
        color="tab:blue",
    )
    arrow_with_label(
        ax,
        (tl_rear, 2.0),
        (tl_front, 2.0),
        "trailer_length",
        text_offset=(0.0, 0.15),
        color="tab:orange",
    )
    arrow_with_label(
        ax,
        (tractor_center_x + 3.8, -0.5 * tractor_width),
        (tractor_center_x + 3.8, 0.5 * tractor_width),
        "width",
        text_offset=(0.35, 0.0),
        color="tab:blue",
        y_align="center",
    )
    arrow_with_label(
        ax,
        (trailer_center_x - 5.8, -0.5 * trailer_width),
        (trailer_center_x - 5.8, 0.5 * trailer_width),
        "trailer_width",
        text_offset=(-0.45, 0.0),
        color="tab:orange",
        y_align="center",
    )

    # Hitch offsets
    arrow_with_label(
        ax,
        (tr_rear, -2.0),
        (tr_hitch_x, -2.0),
        "tractor2hitch",
        text_offset=(0.0, -0.15),
        color="tab:blue",
        y_align="top",
    )
    arrow_with_label(
        ax,
        (tl_hitch_x, -2.0),
        (tl_front, -2.0),
        "trailer2hitch",
        text_offset=(0.0, -0.15),
        color="tab:orange",
        y_align="top",
    )

    # Tractor axle/bumpers
    ax.plot([tr_rear_axle_x], [0.0], marker="|", markersize=18, color="tab:blue")
    ax.plot([tr_front_axle_x], [0.0], marker="|", markersize=18, color="tab:blue")
    ax.text(tr_rear_axle_x, -0.3, "rear axle", fontsize=8, color="tab:blue", ha="center")
    ax.text(tr_front_axle_x, -0.3, "front axle", fontsize=8, color="tab:blue", ha="center")
    arrow_with_label(
        ax,
        (tr_rear, -3.15),
        (tr_rear_axle_x, -3.15),
        "tractor_d_rear_axle2rear_bumper",
        text_offset=(0.0, -0.14),
        color="tab:blue",
        y_align="top",
    )
    arrow_with_label(
        ax,
        (tr_rear_axle_x, -3.7),
        (tr_front_axle_x, -3.7),
        "tractor_d_rear_axle2front_axle",
        text_offset=(0.0, -0.14),
        color="tab:blue",
        y_align="top",
    )
    arrow_with_label(
        ax,
        (tr_front_axle_x, -4.25),
        (tr_front, -4.25),
        "tractor_d_front_axle2front_bumper",
        text_offset=(0.0, -0.14),
        color="tab:blue",
        y_align="top",
    )

    # Trailer axle/bumpers
    ax.plot([tl_rear_axle_x], [0.0], marker="|", markersize=18, color="tab:orange")
    ax.text(tl_rear_axle_x, -0.3, "rear axle", fontsize=8, color="tab:orange", ha="center")
    arrow_with_label(
        ax,
        (tl_rear, -3.15),
        (tl_rear_axle_x, -3.15),
        "trailer_d_rear_axel2_rear_bumper",
        text_offset=(0.0, -0.14),
        color="tab:orange",
        y_align="top",
    )
    arrow_with_label(
        ax,
        (tl_rear_axle_x, -3.7),
        (tl_front, -3.7),
        "trailer_d_real_axel2_front_bumper",
        text_offset=(0.0, -0.14),
        color="tab:orange",
        y_align="top",
    )

    ax.text(tractor_center_x, 1.0, "TRACTOR", color="tab:blue", fontsize=10, ha="center", weight="bold")
    ax.text(trailer_center_x, 1.0, "TRAILER", color="tab:orange", fontsize=10, ha="center", weight="bold")

    ax.set_title("Trailer Parameter Schematic (Bounding-Box View)")
    ax.set_aspect("equal")
    ax.set_xlim(-15.2, 6.5)
    ax.set_ylim(-5.2, 3.6)
    ax.axis("off")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Draw a labeled tractor/trailer parameter schematic.")
    parser.add_argument(
        "--output",
        default="outputs/trailer_parameter_schematic.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    draw_schematic(output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
