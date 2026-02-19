import numpy as np
import pytest

import pufferlib.ocean.drive.drive as drive_module
from pufferlib.ocean.drive.drive import Drive


def _constant_trajectory(x, y, z=0.0, steps=91):
    return [{"x": x, "y": y, "z": z} for _ in range(steps)]


def _constant_scalar(value, steps=91):
    return [value for _ in range(steps)]


def test_generated_bin_loads_and_steps_in_drive(generated_conversion_bin, monkeypatch):
    map_dir = generated_conversion_bin.parent

    # Some builds of the C binding require this kwarg explicitly.
    original_shared = drive_module.binding.shared

    def shared_with_default(*args, **kwargs):
        kwargs.setdefault("sequential_map_sampling", 0)
        return original_shared(*args, **kwargs)

    monkeypatch.setattr(drive_module.binding, "shared", shared_with_default)

    try:
        env = Drive(
            num_agents=2,
            num_maps=1,
            map_dir=str(map_dir),
            resample_frequency=0,
            episode_length=10,
            control_mode="control_vehicles",
            init_mode="create_all_valid",
        )
    except Exception as exc:
        pytest.fail(f"Failed to initialize Drive with generated binary: {exc}")

    try:
        obs, info = env.reset(seed=0)
        assert obs is not None
        assert obs.shape[0] == env.num_agents

        actions = np.zeros_like(env.actions)
        obs, rewards, terminals, truncations, info = env.step(actions)
        assert obs.shape[0] == env.num_agents
        assert rewards.shape[0] == env.num_agents
        assert terminals.shape[0] == env.num_agents
        assert truncations.shape[0] == env.num_agents
    finally:
        env.close()


def _make_map(tmp_path, scenario, unique_map_id=0):
    map_dir = tmp_path / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    map_path = map_dir / "map_000.bin"
    drive_module.save_map_binary(scenario, str(map_path), unique_map_id=unique_map_id)
    return map_dir


def _shared_patch(monkeypatch):
    # Some builds of the C binding require this kwarg explicitly.
    original_shared = drive_module.binding.shared

    def shared_with_default(*args, **kwargs):
        kwargs.setdefault("sequential_map_sampling", 0)
        return original_shared(*args, **kwargs)

    monkeypatch.setattr(drive_module.binding, "shared", shared_with_default)


def _run_steps(map_dir, reward_vehicle_collision, reward_offroad_collision, num_steps=1):
    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        resample_frequency=0,
        episode_length=10,
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        reward_vehicle_collision=reward_vehicle_collision,
        reward_offroad_collision=reward_offroad_collision,
    )
    try:
        env.reset(seed=0)
        reward_total = 0.0
        collision_flag = 0.0
        for _ in range(num_steps):
            # Neutral classic-discrete action: accel idx=3, steer idx=6 => 3*13+6 = 45.
            actions = np.full_like(env.actions, 45)
            obs, rewards, _, _, _ = env.step(actions)
            reward_total += float(rewards[0])
            collision_flag = max(collision_flag, float(obs[0][5]))
        return reward_total, collision_flag
    finally:
        env.close()


def test_trailer_pose_follow_triggers_collision_after_step(tmp_path, monkeypatch):
    # A/B test: with trailer enabled, coupled trailer motion creates collision;
    # without trailer metadata, ego should not collide in this setup.
    scenario = {
        "metadata": {
            "sdc_track_index": 0,
            "tracks_to_predict": [{"track_index": 0}],
            "has_ego_trailer": True,
            "ego_trailer_track_index": 1,
        },
        "objects": [
            {
                "id": 0,
                "source_track_id": "ego",
                "type": "vehicle",
                "position": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                "velocity": [{"x": 10.0, "y": 0.0, "z": 0.0}],
                "heading": [0.0],
                "valid": [1],
                "width": 1.0,
                "length": 4.8,
                "height": 1.7,
                "goalPosition": {"x": 40.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 1,
                "source_track_id": "ego_trailer",
                "type": "vehicle",
                "position": [{"x": -3.32, "y": 0.0, "z": 0.0}],
                "velocity": [{"x": 10.0, "y": 0.0, "z": 0.0}],
                "heading": [0.0],
                "valid": [1],
                "width": 2.0,
                "length": 4.0,
                "height": 2.0,
                "goalPosition": {"x": -3.32, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 2,
                "source_track_id": "static_obstacle",
                "type": "vehicle",
                "position": [{"x": 0.3, "y": 0.9, "z": 0.0}],
                "velocity": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                "heading": [0.0],
                "valid": [1],
                "width": 1.0,
                "length": 1.0,
                "height": 1.0,
                "goalPosition": {"x": 0.3, "y": 0.9, "z": 0.0},
                "mark_as_expert": 1,
            },
        ],
        "roads": [
            {
                "id": 100,
                "type": "lane",
                "geometry": [{"x": -20.0, "y": 0.0, "z": 0.0}, {"x": 60.0, "y": 0.0, "z": 0.0}],
                "width": 3.5,
                "length": 80.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 101,
                "type": "road_edge",
                "geometry": [{"x": -20.0, "y": 8.0, "z": 0.0}, {"x": 60.0, "y": 8.0, "z": 0.0}],
                "width": 0.2,
                "length": 80.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
        ],
    }
    map_dir = _make_map(tmp_path / "with_trailer", scenario)
    scenario_no_trailer = dict(scenario)
    scenario_no_trailer["metadata"] = dict(scenario["metadata"])
    scenario_no_trailer["metadata"]["has_ego_trailer"] = False
    scenario_no_trailer["metadata"]["ego_trailer_track_index"] = -1
    map_dir_no_trailer = _make_map(tmp_path / "without_trailer", scenario_no_trailer)
    _shared_patch(monkeypatch)

    reward_with, collision_flag_with = _run_steps(
        map_dir=map_dir,
        reward_vehicle_collision=-1.0,
        reward_offroad_collision=0.0,
        num_steps=6,
    )
    reward_without, collision_flag_without = _run_steps(
        map_dir=map_dir_no_trailer,
        reward_vehicle_collision=-1.0,
        reward_offroad_collision=0.0,
        num_steps=6,
    )

    # Use a small margin: reward scaling can differ across builds, but
    # trailer-enabled case should still be measurably worse than baseline.
    assert reward_with < reward_without - 0.1
    assert collision_flag_with > collision_flag_without


def test_trailer_only_offroad_penalty(tmp_path, monkeypatch):
    # A/B test: road edge intersects trailer footprint only; offroad penalty should
    # appear when trailer metadata is enabled.
    scenario = {
        "metadata": {
            "sdc_track_index": 0,
            "tracks_to_predict": [{"track_index": 0}],
            "has_ego_trailer": True,
            "ego_trailer_track_index": 1,
        },
        "objects": [
            {
                "id": 0,
                "source_track_id": "ego",
                "type": "vehicle",
                "position": _constant_trajectory(0.0, 0.0),
                "velocity": _constant_trajectory(0.0, 0.0),
                "heading": _constant_scalar(0.0),
                "valid": _constant_scalar(1),
                "width": 1.0,
                "length": 4.8,
                "height": 1.7,
                "goalPosition": {"x": 20.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 1,
                "source_track_id": "ego_trailer",
                "type": "vehicle",
                "position": _constant_trajectory(-3.32, 0.0),
                "velocity": _constant_trajectory(0.0, 0.0),
                "heading": _constant_scalar(0.0),
                "valid": _constant_scalar(1),
                "width": 2.0,
                "length": 4.0,
                "height": 2.0,
                "goalPosition": {"x": -3.32, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
        ],
        "roads": [
            {
                "id": 100,
                "type": "lane",
                "geometry": [{"x": -20.0, "y": 0.0, "z": 0.0}, {"x": 60.0, "y": 0.0, "z": 0.0}],
                "width": 3.5,
                "length": 80.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 101,
                "type": "road_edge",
                "geometry": [{"x": -10.0, "y": 0.6, "z": 0.0}, {"x": 2.0, "y": 0.6, "z": 0.0}],
                "width": 0.2,
                "length": 12.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
        ],
    }
    map_dir = _make_map(tmp_path / "with_trailer", scenario)
    scenario_no_trailer = dict(scenario)
    scenario_no_trailer["metadata"] = dict(scenario["metadata"])
    scenario_no_trailer["objects"] = [dict(obj) for obj in scenario["objects"]]
    scenario_no_trailer["metadata"]["has_ego_trailer"] = False
    scenario_no_trailer["metadata"]["ego_trailer_track_index"] = -1
    # Keep the A/B comparison focused on trailer-coupled offroad behavior:
    # without trailer metadata, this second box would otherwise be treated as a
    # regular nearby vehicle and can trigger plain vehicle-collision penalties,
    # masking the trailer-offroad signal this test is meant to isolate.
    scenario_no_trailer["objects"][1]["valid"] = _constant_scalar(0)
    map_dir_no_trailer = _make_map(tmp_path / "without_trailer", scenario_no_trailer)
    _shared_patch(monkeypatch)

    reward_with, collision_flag_with = _run_steps(
        map_dir=map_dir,
        reward_vehicle_collision=0.0,
        reward_offroad_collision=-1.0,
        num_steps=1,
    )
    reward_without, collision_flag_without = _run_steps(
        map_dir=map_dir_no_trailer,
        reward_vehicle_collision=0.0,
        reward_offroad_collision=-1.0,
        num_steps=1,
    )

    # Use a small margin: reward scaling can differ across builds, but
    # trailer-enabled case should still be measurably worse than baseline.
    assert reward_with < reward_without - 0.1
    assert collision_flag_with > collision_flag_without


def test_offroad_penalty_baseline_for_tractor(tmp_path, monkeypatch):
    # Baseline: ensure offroad pipeline is active when road edge intersects the tractor.
    scenario = {
        "metadata": {"sdc_track_index": 0, "tracks_to_predict": [{"track_index": 0}]},
        "objects": [
            {
                "id": 0,
                "type": "vehicle",
                "position": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                "velocity": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                "heading": [0.0],
                "valid": [1],
                "width": 1.2,
                "length": 4.8,
                "height": 1.7,
                "goalPosition": {"x": 20.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            }
        ],
        "roads": [
            {
                "id": 100,
                "type": "lane",
                "geometry": [{"x": -20.0, "y": 0.0, "z": 0.0}, {"x": 60.0, "y": 0.0, "z": 0.0}],
                "width": 3.5,
                "length": 80.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 101,
                "type": "road_edge",
                "geometry": [{"x": -5.0, "y": 0.3, "z": 0.0}, {"x": 5.0, "y": 0.3, "z": 0.0}],
                "width": 0.2,
                "length": 10.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
        ],
    }
    map_dir = _make_map(tmp_path / "tractor_offroad", scenario)
    _shared_patch(monkeypatch)

    reward, collision_flag = _run_steps(
        map_dir=map_dir,
        reward_vehicle_collision=0.0,
        reward_offroad_collision=-1.0,
        num_steps=1,
    )
    assert reward <= -0.5
    assert collision_flag > 0.5
