import math
from pathlib import Path

import pytest

from pufferlib.ocean.drive.drive import Drive, classify_map_turning, save_map_binary


def _write_heading_map(map_path: Path, heading_start_deg: float, heading_end_deg: float, unique_map_id: int) -> None:
    length = 91
    headings_rad = [
        math.radians(heading_start_deg + (heading_end_deg - heading_start_deg) * idx / (length - 1))
        for idx in range(length)
    ]
    map_data = {
        "metadata": {
            "sdc_track_index": 0,
            "tracks_to_predict": [{"track_index": 0}],
        },
        "objects": [
            {
                "id": 1000 + unique_map_id,
                "type": "vehicle",
                "length": 4.5,
                "width": 1.9,
                "height": 1.6,
                "position": [{"x": float(idx), "y": 0.0, "z": 0.0} for idx in range(length)],
                "velocity": [{"x": 1.0, "y": 0.0, "z": 0.0} for _ in range(length)],
                "heading": headings_rad,
                "valid": [1 for _ in range(length)],
                "goalPosition": {"x": 100.0, "y": 0.0, "z": 0.0},
            }
        ],
        "roads": [],
    }
    save_map_binary(map_data, str(map_path), unique_map_id=unique_map_id)


def test_classify_map_turning_matches_heading_threshold(tmp_path):
    straight_map = tmp_path / "map_000.bin"
    turning_map = tmp_path / "map_001.bin"
    _write_heading_map(straight_map, 0.0, 10.0, unique_map_id=1)
    _write_heading_map(turning_map, 0.0, 90.0, unique_map_id=2)

    straight_row = classify_map_turning(straight_map, threshold_deg=45.0)
    turning_row = classify_map_turning(turning_map, threshold_deg=45.0)

    assert straight_row["bucket"] == "straight"
    assert turning_row["bucket"] == "turning"


def test_drive_scenario_filter_turning_builds_filtered_map_dir(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_heading_map(map_dir / "map_000.bin", 0.0, 10.0, unique_map_id=1)
    _write_heading_map(map_dir / "map_001.bin", 0.0, 90.0, unique_map_id=2)

    env = Drive(
        num_agents=1,
        num_maps=2,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="default",
        scenario_filter="turning",
        scenario_filter_threshold_deg=45.0,
    )
    filtered_dir = Path(env.map_dir)
    try:
        assert env.original_map_dir == str(map_dir)
        assert env.original_num_maps == 2
        assert env.num_maps == 1
        assert filtered_dir != map_dir
        assert (filtered_dir / "map_000.bin").exists()
        assert (filtered_dir / "scenario_filter_manifest.json").exists()
        assert env.scenario_filter_metadata["selected_count"] == 1
        assert env.scenario_filter_metadata["selected_maps"][0]["original_map_name"] == "map_001.bin"

        obs, _ = env.reset(seed=0)
        assert obs.shape[0] == env.num_agents
    finally:
        env.close()

    assert not filtered_dir.exists()


def test_drive_scenario_filter_raises_when_no_maps_match(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_heading_map(map_dir / "map_000.bin", 0.0, 10.0, unique_map_id=1)

    with pytest.raises(ValueError, match="Scenario filter did not match any maps"):
        Drive(
            num_agents=1,
            num_maps=1,
            map_dir=str(map_dir),
            episode_length=91,
            init_steps=0,
            control_mode="control_sdc_only",
            init_mode="create_all_valid",
            resample_frequency=0,
            observation_mode="default",
            scenario_filter="turning",
            scenario_filter_threshold_deg=45.0,
        )
