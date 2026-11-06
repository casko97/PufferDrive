"""Diagnostics for PufferDrive's centerline-based lane/Frenet reward state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


ROAD_LANE = 4
ROAD_LINE = 5
ROAD_EDGE = 6


@dataclass(frozen=True)
class RoadPolyline:
    entity_index: int
    road_id: int
    road_type: int
    points: np.ndarray


@dataclass(frozen=True)
class LaneCase:
    name: str
    x: float
    y: float
    heading: float
    explanation: str


def _read_array(handle, count: int, dtype: str) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    data = handle.read(count * itemsize)
    if len(data) != count * itemsize:
        raise ValueError("truncated PufferDrive map binary")
    return np.frombuffer(data, dtype=dtype).copy()


def load_road_polylines(map_file: str | Path) -> list[RoadPolyline]:
    """Read road geometry directly from a PufferDrive .bin map."""
    polylines: list[RoadPolyline] = []
    with Path(map_file).open("rb") as handle:
        _sdc_index, num_tracks = struct.unpack("ii", handle.read(8))
        handle.seek(4 * num_tracks, 1)
        num_objects, num_roads = struct.unpack("ii", handle.read(8))
        num_entities = num_objects + num_roads

        for entity_index in range(num_entities):
            scenario_id, entity_type, entity_id, size = struct.unpack("iiii", handle.read(16))
            del scenario_id
            x = _read_array(handle, size, "<f4")
            y = _read_array(handle, size, "<f4")
            _read_array(handle, size, "<f4")
            if entity_type in (1, 2, 3):
                for _ in range(4):
                    _read_array(handle, size, "<f4")
                _read_array(handle, size, "<i4")
            handle.seek(6 * 4 + 4, 1)
            if entity_index >= num_objects and entity_type in (ROAD_LANE, ROAD_LINE, ROAD_EDGE):
                polylines.append(
                    RoadPolyline(
                        entity_index=entity_index,
                        road_id=entity_id,
                        road_type=entity_type,
                        points=np.column_stack((x, y)),
                    )
                )
    return polylines


def lane_polylines(map_file: str | Path) -> list[RoadPolyline]:
    return [polyline for polyline in load_road_polylines(map_file) if polyline.road_type == ROAD_LANE]


def segment_pose(
    lane: RoadPolyline,
    segment_index: int,
    *,
    fraction: float = 0.5,
    lateral_offset: float = 0.0,
    heading_offset: float = 0.0,
) -> tuple[float, float, float]:
    """Place a diagnostic pose relative to one real centerline segment."""
    start = lane.points[segment_index]
    end = lane.points[segment_index + 1]
    tangent = end - start
    length = float(np.linalg.norm(tangent))
    if length <= 1e-6:
        raise ValueError("zero-length lane segment")
    tangent /= length
    normal = np.asarray((-tangent[1], tangent[0]))
    point = start + fraction * (end - start) + lateral_offset * normal
    heading = float(np.arctan2(tangent[1], tangent[0]) + heading_offset)
    return float(point[0]), float(point[1]), heading


def map_000_edge_cases(lanes: list[RoadPolyline]) -> list[LaneCase]:
    """Cases discovered from the checked-in map_000.bin geometry."""
    by_id = {lane.road_id: lane for lane in lanes}
    required = {69, 75, 154}
    missing = required - by_id.keys()
    if missing:
        raise ValueError(f"map_000.bin is missing expected lane ids: {sorted(missing)}")

    flip_pose = segment_pose(by_id[154], 1)
    reverse_pose = segment_pose(by_id[69], 0, heading_offset=np.pi)
    cutoff_inside = segment_pose(by_id[75], 0, lateral_offset=3.9)
    cutoff_outside = segment_pose(by_id[75], 0, lateral_offset=4.1)
    return [
        LaneCase(
            "Aligned centerline",
            *flip_pose,
            "Lane 154 is selected with d≈0 and θ≈0.",
        ),
        LaneCase(
            "Heading-dependent lane switch",
            flip_pose[0],
            flip_pose[1],
            flip_pose[2] + np.pi,
            "Same point, reverse heading: +3 m changes selection to lane 162.",
        ),
        LaneCase(
            "Opposite heading retained",
            *reverse_pose,
            "Lane 69 remains selected because 0 m + 3 m is below 4 m.",
        ),
        LaneCase(
            "3.9 m accepted",
            *cutoff_inside,
            "Lane 75 remains valid immediately inside the hard threshold.",
        ),
        LaneCase(
            "4.1 m rejected",
            *cutoff_outside,
            "No lane is selected; the GIGAFLOW boundary penalty applies.",
        ),
    ]


def closest_point_on_polyline(point: np.ndarray, polyline: RoadPolyline) -> np.ndarray:
    best_point = polyline.points[0]
    best_distance = np.inf
    for start, end in zip(polyline.points[:-1], polyline.points[1:]):
        delta = end - start
        denom = float(delta @ delta)
        fraction = 0.0 if denom == 0.0 else float(np.clip((point - start) @ delta / denom, 0.0, 1.0))
        candidate = start + fraction * delta
        distance = float(np.linalg.norm(point - candidate))
        if distance < best_distance:
            best_distance = distance
            best_point = candidate
    return best_point


def plot_lane_frenet_cases(
    map_file: str | Path,
    cases: list[LaneCase],
    results: list[tuple[int, int, float, float]],
    output_file: str | Path,
) -> Path:
    """Plot real centerlines, diagnostic poses, and native C metric results."""
    from html import escape
    import textwrap

    roads = load_road_polylines(map_file)
    lanes = [road for road in roads if road.road_type == ROAD_LANE]
    lane_by_index = {lane.entity_index: lane for lane in lanes}
    columns = 2
    rows = (len(cases) + columns - 1) // columns
    panel_width, panel_height = 600, 520
    plot_size = 420
    width, height = columns * panel_width, 70 + rows * panel_height
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">',
        '<path d="M0,0 L8,4 L0,8 Z" fill="#1f77b4"/>',
        "</marker>",
        '<marker id="lane-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">',
        '<path d="M0,0 L7,3.5 L0,7 Z" fill="#777"/>',
        "</marker>",
        '<marker id="selected-lane-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">',
        '<path d="M0,0 L7,3.5 L0,7 Z" fill="#d62728"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="600" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="bold">'
        "PufferDrive centerline reformulation: distance + 3 m for heading error &gt; 30°, reject &gt; 4 m"
        "</text>",
        '<text x="600" y="50" text-anchor="middle" font-family="sans-serif" font-size="12">'
        "Gray/red arrowheads show lane direction; the blue arrow shows agent heading"
        "</text>",
    ]

    for index, (case, result) in enumerate(zip(cases, results)):
        lane_index, lane_id, heading_error, center_offset = result
        column, row = index % columns, index // columns
        panel_x, panel_y = column * panel_width, 55 + row * panel_height
        left, top = panel_x + 90, panel_y + 65
        scale = plot_size / 24.0

        def transform(point: np.ndarray) -> tuple[float, float]:
            return (
                left + (float(point[0]) - (case.x - 12.0)) * scale,
                top + ((case.y + 12.0) - float(point[1])) * scale,
            )

        def append_lane_directions(lane: RoadPolyline, marker: str, color: str, stroke_width: float) -> None:
            for segment_start, segment_end in zip(lane.points[:-1], lane.points[1:]):
                midpoint = 0.5 * (segment_start + segment_end)
                if abs(float(midpoint[0]) - case.x) > 12.0 or abs(float(midpoint[1]) - case.y) > 12.0:
                    continue
                arrow_start = 0.58 * segment_start + 0.42 * segment_end
                arrow_end = 0.42 * segment_start + 0.58 * segment_end
                start_svg = transform(arrow_start)
                end_svg = transform(arrow_end)
                svg.append(
                    f'<line x1="{start_svg[0]:.2f}" y1="{start_svg[1]:.2f}" '
                    f'x2="{end_svg[0]:.2f}" y2="{end_svg[1]:.2f}" stroke="{color}" '
                    f'stroke-width="{stroke_width}" marker-end="url(#{marker})"/>'
                )

        clip_id = f"panel-{index}"
        svg.extend(
            [
                f'<clipPath id="{clip_id}"><rect x="{left}" y="{top}" width="{plot_size}" height="{plot_size}"/></clipPath>',
                f'<text x="{panel_x + panel_width / 2}" y="{panel_y + 22}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="16" font-weight="bold">{escape(case.name)}</text>',
                f'<text x="{panel_x + panel_width / 2}" y="{panel_y + 44}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="13">lane id={lane_id}, heading error='
                f'{np.degrees(heading_error):.1f}°, signed d={center_offset:.2f} m</text>',
                f'<rect x="{left}" y="{top}" width="{plot_size}" height="{plot_size}" fill="#fafafa" stroke="#999"/>',
                f'<g clip-path="url(#{clip_id})">',
            ]
        )
        for lane in lanes:
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(transform, lane.points))
            svg.append(f'<polyline points="{points}" fill="none" stroke="#c7c7c7" stroke-width="1"/>')
            append_lane_directions(lane, "lane-arrow", "#777", 1.5)

        selected = lane_by_index.get(lane_index)
        if selected is not None:
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(transform, selected.points))
            svg.append(f'<polyline points="{points}" fill="none" stroke="#d62728" stroke-width="4"/>')
            append_lane_directions(selected, "selected-lane-arrow", "#d62728", 2.5)
            point = np.asarray((case.x, case.y))
            projection = closest_point_on_polyline(point, selected)
            point_svg = transform(point)
            projection_svg = transform(projection)
            svg.append(
                f'<line x1="{point_svg[0]:.2f}" y1="{point_svg[1]:.2f}" '
                f'x2="{projection_svg[0]:.2f}" y2="{projection_svg[1]:.2f}" '
                'stroke="#1f77b4" stroke-width="2" stroke-dasharray="6 4"/>'
            )

        start = transform(np.asarray((case.x, case.y)))
        end = transform(
            np.asarray(
                (
                    case.x + 3.0 * np.cos(case.heading),
                    case.y + 3.0 * np.sin(case.heading),
                )
            )
        )
        svg.extend(
            [
                f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" '
                'stroke="#1f77b4" stroke-width="4" marker-end="url(#arrow)"/>',
                f'<circle cx="{start[0]:.2f}" cy="{start[1]:.2f}" r="5" fill="#1f77b4"/>',
                "</g>",
            ]
        )
        for line_index, line in enumerate(textwrap.wrap(case.explanation, width=72)):
            svg.append(
                f'<text x="{panel_x + panel_width / 2}" y="{top + plot_size + 22 + 16 * line_index}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="12">{escape(line)}</text>'
            )

    svg.append("</svg>")
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg), encoding="utf-8")
    return output
