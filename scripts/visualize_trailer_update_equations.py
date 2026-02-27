#!/usr/bin/env python3
"""Generate a compact schematic for trailer update equations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np


def rect_corners(cx, cy, length, width, heading):
    hl = 0.5 * length
    hw = 0.5 * width
    local = np.array(
        [
            [-hl, -hw],
            [hl, -hw],
            [hl, hw],
            [-hl, hw],
        ],
        dtype=np.float32,
    )
    c = np.cos(heading)
    s = np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    world = local @ rot.T
    world[:, 0] += cx
    world[:, 1] += cy
    return [(float(x), float(y)) for x, y in world]


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


def draw(output_path: Path):
    fig, ax = plt.subplots(figsize=(12.5, 6.5))

    # Geometry (illustrative)
    Lt = 6.0
    Lr = 10.0
    Wt = 2.4
    Wr = 2.5
    tractor2hitch = 1.0
    trailer2hitch = 1.5
    Le = max(0.5, Lr - trailer2hitch)

    xt, yt = 1.8, 0.2
    theta_t = 0.10
    theta_r = -0.20
    ut = np.array([np.cos(theta_t), np.sin(theta_t)], dtype=np.float32)
    ur = np.array([np.cos(theta_r), np.sin(theta_r)], dtype=np.float32)
    nt = np.array([-np.sin(theta_t), np.cos(theta_t)], dtype=np.float32)  # left-normal of tractor heading
    nr = np.array([-np.sin(theta_r), np.cos(theta_r)], dtype=np.float32)  # left-normal of trailer heading
    tr_center = np.array([xt, yt], dtype=np.float32)
    tr_rear_pt = tr_center - 0.5 * Lt * ut
    hitch_joint = tr_rear_pt + tractor2hitch * ut

    # Realistic relative placement: trailer center is derived from the shared hitch joint.
    tl_center = hitch_joint - (0.5 * Lr - trailer2hitch) * ur
    xr, yr = float(tl_center[0]), float(tl_center[1])

    tractor = Polygon(
        rect_corners(xt, yt, Lt, Wt, theta_t),
        closed=True,
        facecolor="tab:blue",
        edgecolor="tab:blue",
        alpha=0.2,
        linewidth=2,
    )
    trailer = Polygon(
        rect_corners(xr, yr, Lr, Wr, theta_r),
        closed=True,
        facecolor="tab:orange",
        edgecolor="tab:orange",
        alpha=0.2,
        linewidth=2,
    )
    ax.add_patch(tractor)
    ax.add_patch(trailer)

    # Hitch points and key body points
    tr_rear_pt = tr_center - 0.5 * Lt * ut
    tr_front_pt = tr_center + 0.5 * Lt * ut
    tl_rear_pt = tl_center - 0.5 * Lr * ur
    tl_front_pt = tl_center + 0.5 * Lr * ur

    hitch_tr = tr_rear_pt + tractor2hitch * ut
    hitch_tl = tl_front_pt - trailer2hitch * ur
    xh, yh = float(hitch_tr[0]), float(hitch_tr[1])
    xh_tr, yh_tr = float(hitch_tl[0]), float(hitch_tl[1])
    ax.plot([xh], [yh], marker="o", markersize=7, markerfacecolor="white", markeredgecolor="black", linewidth=1.2)
    ax.text(xh + 0.1, yh + 0.2, "shared hitch joint", fontsize=8, color="black")

    # Rear point proxy and center reconstruction arrow
    rear_proxy = hitch_tr - Le * ur
    xrear, yrear = float(rear_proxy[0]), float(rear_proxy[1])
    ax.plot([xrear], [yrear], "o", color="black", markersize=4)
    ax.add_patch(FancyArrowPatch((xrear, yrear), (xr, yr), arrowstyle="->", mutation_scale=12, linewidth=1.4, color="tab:orange"))

    # Heading arrows
    ax.add_patch(
        FancyArrowPatch(
            (xt, yt),
            (xt + 1.8 * np.cos(theta_t), yt + 1.8 * np.sin(theta_t)),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.8,
            color="tab:blue",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (xr, yr),
            (xr + 1.6 * np.cos(theta_r), yr + 1.6 * np.sin(theta_r)),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.8,
            color="tab:orange",
        )
    )
    ax.text(xt + 1.9 * np.cos(theta_t), yt + 1.9 * np.sin(theta_t), r"$\theta_{tractor}$", color="tab:blue", fontsize=11, va="center")
    ax.text(xr + 1.8 * np.cos(theta_r), yr + 1.8 * np.sin(theta_r), r"$\theta_{trailer}$", color="tab:orange", fontsize=11, va="center")

    # Main dimensions (parallel to rotated boxes, offset by body normals)
    lt_offset = 1.25 * nt
    lr_offset = -1.25 * nr
    arrow_with_label(
        ax,
        (float(tr_rear_pt[0] + lt_offset[0]), float(tr_rear_pt[1] + lt_offset[1])),
        (float(tr_front_pt[0] + lt_offset[0]), float(tr_front_pt[1] + lt_offset[1])),
        r"$L_{tractor}$",
        text_offset=(0.0, 0.12),
        color="tab:blue",
    )
    arrow_with_label(
        ax,
        (float(tl_rear_pt[0] + lr_offset[0]), float(tl_rear_pt[1] + lr_offset[1])),
        (float(tl_front_pt[0] + lr_offset[0]), float(tl_front_pt[1] + lr_offset[1])),
        r"$L_{trailer}$",
        text_offset=(0.0, -0.12),
        color="tab:orange",
        y_align="top",
    )
    trailer2hitch_offset = -0.55 * nr
    arrow_with_label(
        ax,
        (float(xh_tr + trailer2hitch_offset[0]), float(yh_tr + trailer2hitch_offset[1])),
        (float(tl_front_pt[0] + trailer2hitch_offset[0]), float(tl_front_pt[1] + trailer2hitch_offset[1])),
        "trailer2hitch",
        text_offset=(0.0, 0.1),
        color="tab:orange",
    )
    arrow_with_label(
        ax,
        (xh, yh + 0.55),
        (xrear, yrear + 0.55),
        r"$L_{effective}$",
        text_offset=(0.0, 0.1),
        color="black",
    )

    # Secondary labels
    arrow_with_label(
        ax,
        (float(tr_rear_pt[0] + 0.8 * nt[0]), float(tr_rear_pt[1] + 0.8 * nt[1])),
        (float(hitch_tr[0] + 0.8 * nt[0]), float(hitch_tr[1] + 0.8 * nt[1])),
        "tractor2hitch",
        text_offset=(0.0, 0.1),
        color="tab:blue",
    )
    ax.annotate(
        r"rear proxy $(x_{rear},y_{rear})$",
        xy=(xrear, yrear),
        xytext=(xrear - 1.9, yrear - 0.9),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=9,
    )
    ax.annotate(
        r"center update dir.",
        xy=(0.5 * (xrear + xr), 0.5 * (yrear + yr)),
        xytext=(xrear - 3.1, yrear - 1.55),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=9,
    )

    # Visualize articulation angle a = wrap(theta_t - theta_r)
    # Draw both reference rays from one center so the relationship to theta_t/theta_r is explicit.
    a_center = (xh + 0.95, yh + 0.2)
    a_radius = 0.9
    ray_len = 1.25
    ax.plot(
        [a_center[0], a_center[0] + ray_len * np.cos(theta_r)],
        [a_center[1], a_center[1] + ray_len * np.sin(theta_r)],
        color="tab:orange",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.plot(
        [a_center[0], a_center[0] + ray_len * np.cos(theta_t)],
        [a_center[1], a_center[1] + ray_len * np.sin(theta_t)],
        color="tab:blue",
        linewidth=1.2,
        alpha=0.9,
    )
    phi_lo = min(theta_r, theta_t)
    phi_hi = max(theta_r, theta_t)
    arc = np.linspace(phi_lo, phi_hi, 80)
    ax.plot(
        a_center[0] + a_radius * np.cos(arc),
        a_center[1] + a_radius * np.sin(arc),
        color="black",
        linewidth=1.8,
    )
    ax.text(a_center[0] + 0.55, a_center[1] + 0.45, r"$a$", color="black", fontsize=11)
    arc_mid = 0.5 * (phi_lo + phi_hi)
    arc_anchor = (
        a_center[0] + a_radius * np.cos(arc_mid),
        a_center[1] + a_radius * np.sin(arc_mid),
    )
    ax.annotate(
        r"$a=\mathrm{wrap}(\theta_{tractor}-\theta_{trailer})$",
        xy=arc_anchor,
        xytext=(a_center[0] + 0.65, a_center[1] + 0.62),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=8.5,
        color="0.2",
    )

    # Variable labels only (no equation text)
    ax.annotate(
        r"tractor center $(x_{tractor},y_{tractor})$",
        xy=(xt, yt),
        xytext=(xt + 1.6, yt + 1.4),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=9,
    )
    ax.annotate(
        r"trailer center $(x_{trailer},y_{trailer})$",
        xy=(xr, yr),
        xytext=(xr - 3.4, yr + 1.5),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=9,
    )
    ax.annotate(
        r"hitch point $(x_h,y_h)$",
        xy=(xh, yh),
        xytext=(xh - 3.1, yh + 1.0),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=9,
    )

    ax.text(xt, yt + 1.0, "TRACTOR", color="tab:blue", fontsize=10, ha="center", weight="bold")
    ax.text(xr, yr + 1.0, "TRAILER", color="tab:orange", fontsize=10, ha="center", weight="bold")

    ax.set_title("Trailer Update Equations (Bounding-Box Schematic)")
    ax.set_aspect("equal")
    ax.set_xlim(-15.0, 7.1)
    ax.set_ylim(-4.8, 4.0)
    ax.axis("off")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Draw a schematic for trailer update equations.")
    parser.add_argument("--output", default="outputs/trailer_update_equations_schematic.png", help="Output PNG path")
    args = parser.parse_args()
    out = Path(args.output)
    draw(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
