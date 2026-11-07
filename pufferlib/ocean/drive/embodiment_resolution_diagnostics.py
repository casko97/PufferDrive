"""Visualization helpers for Drive embodiment launch resolution."""

from pathlib import Path
import math
import struct

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


PRESETS = {
    1: (3.85, 1.70),
    2: (4.95, 1.90),
    3: (5.50, 2.00),
    4: (9.00, 2.50),
    5: (11.50, 2.55),
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def load_map_elements(map_path):
    """Load lane centerlines and road edges in Drive's centered world frame."""
    entities = []
    with Path(map_path).open("rb") as handle:
        read_int = lambda: struct.unpack("i", handle.read(4))[0]
        _sdc_track_index = read_int()
        num_tracks = read_int()
        handle.seek(4 * num_tracks, 1)
        num_objects, num_roads = read_int(), read_int()
        for _ in range(num_objects + num_roads):
            _scenario_id, entity_type, _entity_id, size = (read_int() for _ in range(4))
            x = np.frombuffer(handle.read(4 * size), dtype=np.float32).copy()
            y = np.frombuffer(handle.read(4 * size), dtype=np.float32).copy()
            handle.seek(4 * size, 1)  # z
            valid = None
            if entity_type <= 2:
                handle.seek(4 * size * 4, 1)  # vx, vy, vz, heading
                valid = np.frombuffer(handle.read(4 * size), dtype=np.int32).copy()
            handle.seek(28, 1)  # dimensions, goal position, expert flag
            entities.append((entity_type, x, y, valid))

    # ``get_global_agent_state`` restores the world mean, so retain the
    # binary's original coordinates here as well.
    return {
        "lanes": [(x, y) for entity_type, x, y, _ in entities if entity_type == 4],
        "road_edges": [(x, y) for entity_type, x, y, _ in entities if entity_type == 6],
    }


def _draw_map_elements(axis, map_elements):
    if not map_elements:
        return
    for x, y in map_elements["lanes"]:
        axis.plot(x, y, color="#9AA0A6", linewidth=0.65, alpha=0.55, zorder=0)
    for x, y in map_elements["road_edges"]:
        axis.plot(x, y, color="#30343B", linewidth=1.15, alpha=0.85, zorder=1)


def _dimensions(state, resolution):
    if state == 0:
        return resolution[8], resolution[9]
    return PRESETS[state]


def _corners(x, y, heading, length, width):
    cosine, sine = math.cos(heading), math.sin(heading)
    return [
        (x + longitudinal * cosine - lateral * sine, y + longitudinal * sine + lateral * cosine)
        for longitudinal, lateral in ((length / 2, width / 2), (length / 2, -width / 2),
                                      (-length / 2, -width / 2), (-length / 2, width / 2))
    ]


def plot_embodiment_resolution(states, resolution, output_path, map_file=None, title=None, clearance=0.20):
    """Plot every recorded reset-resolution state in successive subplots.

    ``states`` is returned by ``Drive.get_global_agent_state()`` and
    ``resolution`` by ``Drive.get_embodiment_resolution()``.
    """
    if clearance < 0:
        raise ValueError("clearance must be >= 0")
    output_path = Path(output_path)
    map_elements = load_map_elements(map_file) if map_file is not None else None
    rounds = max(row[7] for row in resolution)
    fig, axes = plt.subplots(1, rounds, figsize=(5 * rounds, 5), squeeze=False)
    axes = axes[0]
    all_corners = []
    for index, row in enumerate(resolution):
        for state in row[:row[7]]:
            all_corners.extend(_corners(states["x"][index], states["y"][index], states["heading"][index],
                                        *_dimensions(state, row)))
    xs, ys = zip(*all_corners)
    padding = 2.0
    for round_index, axis in enumerate(axes):
        axis.set_aspect("equal")
        _draw_map_elements(axis, map_elements)
        axis.set(xlim=(min(xs) - padding, max(xs) + padding), ylim=(min(ys) - padding, max(ys) + padding),
                 xlabel="x [m]", ylabel="y [m]", title=f"resolution round {round_index}")
        axis.grid(True, linestyle=":", alpha=0.4)
        if clearance:
            axis.text(0.02, 0.02, f"dashed = {clearance * 100:.0f} cm clearance envelope",
                      transform=axis.transAxes, fontsize=8, va="bottom",
                      bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
        for index, row in enumerate(resolution):
            state = row[min(round_index, row[7] - 1)]
            length, width = _dimensions(state, row)
            polygon = Polygon(_corners(states["x"][index], states["y"][index], states["heading"][index],
                                       length, width), closed=True, facecolor=COLORS[index % len(COLORS)],
                              edgecolor="black", alpha=0.55)
            axis.add_patch(polygon)
            if clearance:
                envelope = Polygon(_corners(states["x"][index], states["y"][index], states["heading"][index],
                                            length + 2 * clearance, width + 2 * clearance),
                                   closed=True, fill=False, edgecolor=COLORS[index % len(COLORS)],
                                   linestyle=(0, (3, 2)), linewidth=1.4, alpha=0.95)
                axis.add_patch(envelope)
            axis.annotate(f"agent {states['id'][index]}: {state}",
                          (states["x"][index], states["y"][index]), ha="center", fontsize=8)
    fig.suptitle(title or "Real-map embodiment conflict resolution (lanes/road edges; state 0 = original map geometry)")
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path
