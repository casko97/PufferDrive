import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_geometry(output_path):
    tractor_rear_axle_x = 0.0
    tractor_rear_extent = 1.5
    wheelbase = 3.6
    tractor_front_overhang = 1.2
    tractor_axle_to_hitch = 0.6
    tractor_width = 2.55
    trailer_rear_extent = 2.0
    trailer_front_extent = 10.5
    trailer_front_to_hitch = 2.2
    trailer_width = 2.55

    hitch_x = tractor_rear_axle_x - tractor_axle_to_hitch
    trailer_axle_to_hitch = trailer_front_extent - trailer_front_to_hitch
    trailer_axle_x = hitch_x - trailer_axle_to_hitch
    tractor_connection_x = hitch_x
    trailer_connection_x = trailer_axle_x + trailer_axle_to_hitch
    assert abs(tractor_connection_x - trailer_connection_x) < 1e-12
    tractor_front_axle_x = tractor_rear_axle_x + wheelbase
    tractor_rear_x = tractor_rear_axle_x - tractor_rear_extent
    tractor_front_x = tractor_front_axle_x + tractor_front_overhang
    trailer_rear_x = trailer_axle_x - trailer_rear_extent
    trailer_front_x = trailer_axle_x + trailer_front_extent
    tractor_center_x = 0.5 * (tractor_rear_x + tractor_front_x)
    trailer_center_x = 0.5 * (trailer_rear_x + trailer_front_x)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.add_patch(
        Rectangle(
            (trailer_rear_x, -trailer_width / 2),
            trailer_front_x - trailer_rear_x,
            trailer_width,
            facecolor="#7aa6c2",
            edgecolor="#174a68",
            linewidth=2,
            alpha=0.55,
            label="Trailer bounding box",
        )
    )
    ax.add_patch(
        Rectangle(
            (tractor_rear_x, -tractor_width / 2),
            tractor_front_x - tractor_rear_x,
            tractor_width,
            facecolor="#e8a35b",
            edgecolor="#8b4513",
            linewidth=2,
            alpha=0.65,
            label="Tractor bounding box",
        )
    )

    overlap_rear = max(tractor_rear_x, trailer_rear_x)
    overlap_front = min(tractor_front_x, trailer_front_x)
    ax.add_patch(
        Rectangle(
            (overlap_rear, -tractor_width / 2),
            overlap_front - overlap_rear,
            tractor_width,
            facecolor="#b45f9d",
            edgecolor="none",
            alpha=0.25,
            hatch="//",
            label="Bounding-box overlap",
        )
    )

    def marker(x, label, color, marker_style="o"):
        ax.scatter([x], [0.0], s=65, color=color, marker=marker_style, zorder=5)
        ax.annotate(label, (x, 0.0), xytext=(0, 10), textcoords="offset points", ha="center", color=color)

    marker(trailer_axle_x, "Trailer axle T", "#174a68")
    ax.scatter(
        [tractor_connection_x],
        [0.0],
        s=110,
        color="#7d1f5c",
        marker="D",
        zorder=6,
        label="Tractor connection",
    )
    ax.scatter(
        [trailer_connection_x],
        [0.0],
        s=190,
        facecolors="none",
        edgecolors="#20734d",
        linewidths=2,
        zorder=7,
        label="Trailer connection",
    )
    ax.annotate("Coincident connections H", (hitch_x, 0.0), xytext=(0, 13), textcoords="offset points", ha="center")
    marker(tractor_rear_axle_x, "Tractor rear axle R", "#8b4513")
    marker(tractor_front_axle_x, "Tractor front axle", "#8b4513")
    marker(trailer_center_x, "Trailer center", "#174a68", marker_style="x")
    marker(tractor_center_x, "Tractor center", "#8b4513", marker_style="x")

    def dimension(x0, x1, y, text, color="#222222"):
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops={"arrowstyle": "<->", "color": color, "linewidth": 1.4},
        )
        ax.vlines([x0, x1], y - 0.12, y + 0.12, color=color, linewidth=1)
        ax.text(0.5 * (x0 + x1), y + 0.13, text, ha="center", va="bottom", color=color, fontsize=9)

    dimension(trailer_axle_x, hitch_x, 2.15, "b: trailer axle → hitch = 8.3 m", "#174a68")
    dimension(hitch_x, tractor_rear_axle_x, 2.75, "a: tractor rear axle → hitch = 0.6 m", "#7d1f5c")
    dimension(tractor_rear_axle_x, tractor_front_axle_x, 2.15, "L: wheelbase = 3.6 m", "#8b4513")
    dimension(trailer_rear_x, trailer_axle_x, -2.15, "rear overhang = 2.0 m", "#174a68")
    dimension(trailer_axle_x, trailer_front_x, -2.75, "axle → front bumper = 10.5 m", "#174a68")
    dimension(hitch_x, trailer_front_x, -3.35, "front bumper → hitch = 2.2 m", "#7d1f5c")
    dimension(tractor_rear_x, tractor_rear_axle_x, -2.15, "rear overhang = 1.5 m", "#8b4513")
    dimension(tractor_front_axle_x, tractor_front_x, -2.15, "front overhang = 1.2 m", "#8b4513")
    dimension(trailer_rear_x, trailer_front_x, 3.35, "trailer length = 12.5 m (aligned)")
    dimension(tractor_rear_x, tractor_front_x, 3.95, "tractor length = 6.3 m (aligned)")

    ax.arrow(tractor_center_x, 0.75, 1.0, 0.0, width=0.025, head_width=0.18, head_length=0.25, color="#333333")
    ax.text(tractor_center_x + 0.5, 1.0, "heading +x", ha="center", fontsize=9)
    ax.text(
        0.01,
        0.01,
        "Axle-derived entity-center offsets: tractor 1.65 m, trailer 4.25 m\n"
        f"Aligned bounding-box overlap: {overlap_front - overlap_rear:.1f} m\n"
        f"Tractor/trailer connection residual: {abs(tractor_connection_x - trailer_connection_x):.1e} m",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.9},
    )

    ax.set_title("Default tractor–trailer geometry (aligned headings)")
    ax.set_xlabel("Longitudinal position relative to tractor rear axle R [m]")
    ax.set_ylabel("Lateral position [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(trailer_rear_x - 0.75, tractor_front_x + 0.75)
    ax.set_ylim(-4.0, 4.6)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot the default tractor-trailer geometry")
    parser.add_argument("--output", type=Path, default=Path("/tmp/tractor_trailer_geometry.png"))
    args = parser.parse_args()
    plot_geometry(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
