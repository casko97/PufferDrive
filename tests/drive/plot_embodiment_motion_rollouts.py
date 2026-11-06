"""Visualize nominal jerk-bicycle rollouts for Drive vehicle embodiments.

This mirrors the trailer-support branch's rollout plots: all entities use
oriented footprint boxes, rear-axle reference-point paths, fixed snapshots, per-scenario
metrics, GIFs, and a JSON manifest. The integrator below is a standalone,
line-for-line scalar translation of Drive's JERK update so it can run without
rebuilding the Python extension.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from matplotlib.patches import Polygon
    HAVE_MATPLOTLIB = True
except ModuleNotFoundError:
    HAVE_MATPLOTLIB = False


DT = 0.1
STEPS = 40
INITIAL_SPEED = 8.0
SNAPSHOTS = (0, 10, 20, 30, 40)

EMBODIMENTS = {
    "Small car": (3.85, 1.70, 1.50, 2.50, 0.70),
    "Large car": (4.95, 1.90, 1.50, 2.90, 0.95),
    "Van": (5.50, 2.00, 2.40, 3.45, 1.00),
    "Truck": (9.00, 2.50, 3.25, 5.20, 1.45),
    "Bus": (11.50, 2.55, 3.25, 6.00, 2.30),
}
COLORS = {
    "Small car": "#2a6fbb",
    "Large car": "#27a66a",
    "Van": "#e09b2d",
    "Truck": "#d75656",
    "Bus": "#814bb2",
}
SCENARIOS = {
    "gentle_left": "Gentle left turn",
    "hard_left": "Hard left turn",
    "s_turn": "S-turn",
    "slow_tight_left": "Low-speed tight left turn",
}


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = INITIAL_SPEED
    a_long: float = 0.0
    a_lat: float = 0.0
    steering: float = 0.0


def clip(value, lower, upper):
    return max(lower, min(value, upper))


def action_for(scenario, step):
    if scenario == "gentle_left":
        return 0.0, 0.25
    if scenario == "hard_left":
        return 0.0, 1.0
    if scenario == "s_turn":
        return 0.0, 0.70 if step < STEPS // 2 else -0.70
    if scenario == "slow_tight_left":
        return 0.0, 1.0
    raise ValueError(f"unknown scenario: {scenario}")


def integrate_jerk_bicycle(state, wheelbase, action):
    """Apply the current continuous JERK branch with nominal condition scales."""
    long_action, lat_action = action
    long_jerk = long_action * (15.0 if long_action < 0.0 else 4.0)
    lat_jerk = lat_action * 4.0

    a_long_new = state.a_long + long_jerk * DT
    a_lat_new = state.a_lat + lat_jerk * DT
    if state.a_long * a_long_new < 0.0:
        a_long_new = 0.0
    else:
        a_long_new = clip(a_long_new, -5.0, 2.5)
    if state.a_lat * a_lat_new < 0.0:
        a_lat_new = 0.0
    else:
        a_lat_new = clip(a_lat_new, -4.0, 4.0)

    v_new = state.speed + 0.5 * (a_long_new + state.a_long) * DT
    if state.speed * v_new < 0.0:
        v_new = 0.0
    else:
        v_new = clip(v_new, -2.0, 20.0)

    curvature = a_lat_new / max(v_new * v_new, 1.0e-5)
    curvature = math.copysign(max(abs(curvature), 1.0e-5), curvature)
    steering_target = math.atan(curvature * wheelbase)
    steering = clip(state.steering + clip(steering_target - state.steering, -0.6 * DT, 0.6 * DT), -0.55, 0.55)
    curvature = math.tan(steering) / wheelbase
    a_lat_new = v_new * v_new * curvature

    distance = 0.5 * (v_new + state.speed) * DT
    theta = distance * curvature
    if abs(curvature) < 1.0e-5 or abs(theta) < 1.0e-5:
        dx_local, dy_local = distance, 0.0
    else:
        dx_local = math.sin(theta) / curvature
        dy_local = (1.0 - math.cos(theta)) / curvature

    x = state.x + dx_local * math.cos(state.heading) - dy_local * math.sin(state.heading)
    y = state.y + dx_local * math.sin(state.heading) + dy_local * math.cos(state.heading)
    return State(x, y, math.atan2(math.sin(state.heading + theta), math.cos(state.heading + theta)),
                 v_new, a_long_new, a_lat_new, steering)


def collect_rollout(scenario):
    frames = {name: [] for name in EMBODIMENTS}
    initial_speed = 4.0 if scenario == "slow_tight_left" else INITIAL_SPEED
    states = {
        name: State(x=-(0.5 * dimensions[0] - dimensions[4]), speed=initial_speed)
        for name, dimensions in EMBODIMENTS.items()
    }
    for step in range(STEPS + 1):
        for name, (length, width, height, wheelbase, rear_overhang) in EMBODIMENTS.items():
            state = states[name]
            rear_axle_to_center = 0.5 * length - rear_overhang
            center_x = state.x + rear_axle_to_center * math.cos(state.heading)
            center_y = state.y + rear_axle_to_center * math.sin(state.heading)
            frames[name].append({
                "time": step * DT, "x": center_x, "y": center_y, "rear_x": state.x, "rear_y": state.y,
                "heading": state.heading, "speed": state.speed, "a_lat": state.a_lat,
                "steering": state.steering, "length": length, "width": width, "height": height,
                "wheelbase": wheelbase, "rear_overhang": rear_overhang,
            })
        if step < STEPS:
            action = action_for(scenario, step)
            for name, dimensions in EMBODIMENTS.items():
                states[name] = integrate_jerk_bicycle(states[name], dimensions[3], action)
    return frames


def rectangle_corners(frame):
    half_length, half_width = frame["length"] / 2.0, frame["width"] / 2.0
    cosine, sine = math.cos(frame["heading"]), math.sin(frame["heading"])
    return [
        (frame["x"] + longitudinal * cosine - lateral * sine,
         frame["y"] + longitudinal * sine + lateral * cosine)
        for longitudinal, lateral in ((half_length, half_width), (half_length, -half_width),
                                      (-half_length, -half_width), (-half_length, half_width))
    ]


def limits(frames):
    points = [corner for vehicle in frames.values() for frame in vehicle for corner in rectangle_corners(frame)]
    points.extend((frame["rear_x"], frame["rear_y"])
                  for vehicle in frames.values() for frame in vehicle)
    xs, ys = zip(*points)
    return min(xs) - 2.0, max(xs) + 2.0, min(ys) - 2.0, max(ys) + 2.0


def configure_axis(ax, frames, title):
    xmin, xmax, ymin, ymax = limits(frames)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", title=title,
           xlabel="rear-axle reference x [m]", ylabel="rear-axle reference y [m]")
    ax.grid(True, linestyle=":", alpha=0.4)


def draw_vehicle(ax, name, frame, alpha, label=None):
    color = COLORS[name]
    ax.add_patch(Polygon(rectangle_corners(frame), closed=True, facecolor=color,
                         edgecolor=color, linewidth=1.2, alpha=alpha))
    ax.scatter(frame["x"], frame["y"], color=color, s=13, zorder=5)
    if label:
        ax.annotate(label, (frame["x"], frame["y"]), xytext=(5, 5),
                    textcoords="offset points", color=color, fontsize=7)


def draw_paths(ax, frames, through=None):
    for name, vehicle_frames in frames.items():
        visible = vehicle_frames if through is None else vehicle_frames[:through + 1]
        ax.plot([frame["rear_x"] for frame in visible], [frame["rear_y"] for frame in visible],
                color=COLORS[name], linewidth=1.6, label=name)


def add_metrics(ax, frames, scenario):
    lines = [f"JERK model, same continuous action: {scenario.replace('_', ' ')}"]
    for name, vehicle_frames in frames.items():
        frame = vehicle_frames[-1]
        lines.append(f"{name}: δ={math.degrees(frame['steering']):+.1f}°, aₗₐₜ={frame['a_lat']:+.2f} m/s²")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=7, bbox={"facecolor": "white", "edgecolor": "#777", "alpha": 0.9})


def create_overview(rollouts, output_path):
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6.4 * len(SCENARIOS), 6.5))
    for ax, (scenario, frames) in zip(axes, rollouts.items()):
        draw_paths(ax, frames)
        for ordinal, snapshot in enumerate(SNAPSHOTS):
            for name, vehicle_frames in frames.items():
                draw_vehicle(ax, name, vehicle_frames[snapshot], 0.15 + 0.14 * ordinal,
                             name if snapshot == SNAPSHOTS[-1] else None)
        configure_axis(ax, frames, SCENARIOS[scenario])
        add_metrics(ax, frames, scenario)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(EMBODIMENTS))
    fig.suptitle("Nominal Drive JERK bicycle rollouts by embodiment\n"
                 "rear-axle reference paths and oriented footprint snapshots at 0, 1, 2, 3, and 4 seconds",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def create_animation(scenario, frames, output_path):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    writer = PillowWriter(fps=10)
    with writer.saving(fig, output_path, dpi=110):
        for index in range(STEPS + 1):
            ax.clear()
            draw_paths(ax, frames, through=index)
            for name, vehicle_frames in frames.items():
                draw_vehicle(ax, name, vehicle_frames[index], 0.72, name)
            configure_axis(ax, frames, SCENARIOS[scenario])
            add_metrics(ax, {name: vehicle_frames[:index + 1] for name, vehicle_frames in frames.items()}, scenario)
            ax.legend(loc="lower right", fontsize=7)
            writer.grab_frame()
    plt.close(fig)



def create_svg_overview(rollouts, output_path):
    """Dependency-free fallback with the same paths and footprint snapshots."""
    panel_width, panel_height, margin = 620, 520, 48
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_width * len(rollouts)}" height="610">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="18" font-weight="bold">'
        'Nominal Drive JERK bicycle rollouts by embodiment</text>',
        '<text x="20" y="49" font-family="sans-serif" font-size="12">'
        'Rear-axle paths; oriented footprints at 0, 1, 2, 3, and 4 seconds; identical continuous actions</text>',
    ]
    for panel, (scenario, frames) in enumerate(rollouts.items()):
        xmin, xmax, ymin, ymax = limits(frames)
        scale = min((panel_width - 2 * margin) / (xmax - xmin),
                    (panel_height - 2 * margin) / (ymax - ymin))
        left = panel * panel_width
        def project(point):
            x, y = point
            return (left + margin + (x - xmin) * scale,
                    70 + margin + (ymax - y) * scale)
        chunks.append(f'<rect x="{left + margin}" y="{70 + margin}" width="{panel_width - 2 * margin}" '
                      f'height="{panel_height - 2 * margin}" fill="#fafafa" stroke="#999"/>')
        chunks.append(f'<text x="{left + margin}" y="85" font-family="sans-serif" font-size="15" '
                      f'font-weight="bold">{SCENARIOS[scenario]}</text>')
        for name, vehicle_frames in frames.items():
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(project, [(f["rear_x"], f["rear_y"]) for f in vehicle_frames]))
            chunks.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[name]}" stroke-width="2"/>')
            for ordinal, snapshot in enumerate(SNAPSHOTS):
                corners = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(project, rectangle_corners(vehicle_frames[snapshot])))
                opacity = 0.18 + 0.16 * ordinal
                chunks.append(f'<polygon points="{corners}" fill="{COLORS[name]}" fill-opacity="{opacity:.2f}" '
                              f'stroke="{COLORS[name]}" stroke-width="1"/>')
            final = vehicle_frames[-1]
            x, y = project((final["x"], final["y"]))
            chunks.append(f'<text x="{x + 4:.2f}" y="{y - 4:.2f}" font-family="sans-serif" font-size="10" '
                          f'fill="{COLORS[name]}">{name}</text>')
        for row, name in enumerate(EMBODIMENTS):
            final = frames[name][-1]
            chunks.append(f'<text x="{left + margin}" y="{panel_height + 88 + 15 * row}" '
                          f'font-family="sans-serif" font-size="10" fill="{COLORS[name]}">'
                          f'{name}: final steering {math.degrees(final["steering"]):+.1f} deg, '
                          f'lateral acceleration {final["a_lat"]:+.2f} m/s²</text>')
    chunks.append("</svg>")
    output_path.write_text("\\n".join(chunks) + "\\n")

def generate_visualizations(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = {scenario: collect_rollout(scenario) for scenario in SCENARIOS}
    if HAVE_MATPLOTLIB:
        overview = output_dir / "embodiment_motion_rollouts.png"
        create_overview(rollouts, overview)
        gifs = {}
        for scenario, frames in rollouts.items():
            path = output_dir / f"{scenario}.gif"
            create_animation(scenario, frames, path)
            gifs[scenario] = path
    else:
        overview = output_dir / "embodiment_motion_rollouts.svg"
        create_svg_overview(rollouts, overview)
        gifs = {}
    manifest = {
        "source": "standalone scalar translation of Drive JERK motion update",
        "dt_seconds": DT, "initial_speed_mps": INITIAL_SPEED, "steps": STEPS,
        "overview": overview.name,
        "scenarios": {name: {"gif": gifs[name].name if name in gifs else None, "action_description": SCENARIOS[name]}
                      for name in SCENARIOS},
        "embodiments": {name: {"length_m": dims[0], "width_m": dims[1], "height_m": dims[2],
                               "wheelbase_m": dims[3], "rear_overhang_m": dims[4],
                               "rear_axle_to_center_m": 0.5 * dims[0] - dims[4]}
                         for name, dims in EMBODIMENTS.items()},
        "note": "Footprints use the simulator's geometric-center pose; trails show the rear-axle bicycle reference point.",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return overview, gifs, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/embodiment_motion_rollouts"))
    args = parser.parse_args()
    overview, gifs, manifest = generate_visualizations(args.output_dir)
    print(overview)
    for path in gifs.values():
        print(path)
    print(manifest)


if __name__ == "__main__":
    main()

