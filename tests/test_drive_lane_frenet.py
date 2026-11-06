from pathlib import Path

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.lane_frenet_diagnostics import (
    lane_polylines,
    map_000_edge_cases,
    plot_lane_frenet_cases,
)


MAP_FILE = Path("pufferlib/resources/drive/binaries/map_000.bin")


def test_centerline_frenet_edge_cases_on_real_map_binary():
    """Lock down the centerline reformulation used in place of lane polygons."""
    assert MAP_FILE.is_file()
    lanes = lane_polylines(MAP_FILE)
    assert len(lanes) == 141

    cases = map_000_edge_cases(lanes)
    poses = [(case.x, case.y, case.heading) for case in cases]
    results = binding._test_lane_frenet_from_map(str(MAP_FILE), poses)

    aligned, heading_switch, reverse_retained, cutoff_inside, cutoff_outside = results

    assert aligned[1] == 154
    assert aligned[2] == pytest.approx(0.0, abs=1e-6)
    assert aligned[3] == pytest.approx(0.0, abs=1e-4)

    # At the exact same point, reversing the pose adds the 3 m heading
    # surcharge and changes the selected real centerline from 154 to 162.
    assert heading_switch[1] == 162
    assert np.degrees(heading_switch[2]) == pytest.approx(29.824, abs=0.01)
    assert heading_switch[3] == pytest.approx(2.4866, abs=1e-3)

    # A geometrically exact match is still retained when reversed because
    # adjusted distance is 0 + 3 m, which remains inside the 4 m cutoff.
    assert reverse_retained[1] == 69
    assert abs(reverse_retained[2]) == pytest.approx(np.pi, abs=1e-6)
    assert reverse_retained[3] == pytest.approx(0.0, abs=1e-4)

    assert cutoff_inside[1] == 75
    assert cutoff_inside[3] == pytest.approx(3.9, abs=1e-3)

    # Crossing the strict 4 m threshold removes the Frenet reference entirely.
    assert cutoff_outside[0] == -1
    assert cutoff_outside[1] == -1
    assert cutoff_outside[2] == pytest.approx(np.pi, abs=1e-6)
    assert cutoff_outside[3] == 0.0


def test_lane_frenet_plot_contains_lane_direction_markers(tmp_path):
    cases = map_000_edge_cases(lane_polylines(MAP_FILE))
    poses = [(case.x, case.y, case.heading) for case in cases]
    results = binding._test_lane_frenet_from_map(str(MAP_FILE), poses)
    output = plot_lane_frenet_cases(MAP_FILE, cases, results, tmp_path / "lane_cases.svg")
    svg = output.read_text(encoding="utf-8")

    assert 'id="lane-arrow"' in svg
    assert 'id="selected-lane-arrow"' in svg
    assert svg.count('marker-end="url(#lane-arrow)"') > 0
    assert svg.count('marker-end="url(#selected-lane-arrow)"') >= 4
