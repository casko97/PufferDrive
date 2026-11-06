import math
import subprocess
from pathlib import Path

import numpy as np


def test_tractor_trailer_state_export_native(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root
        / "tests"
        / "drive"
        / "test_tractor_trailer_state_export.c"
    )
    executable = tmp_path / "test_tractor_trailer_state_export"
    raylib_include = repo_root / "raylib-5.5_linux_amd64" / "include"
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
    assert result.stdout.strip() == "tractor-trailer state export tests passed"


def _make_drive(ego_vehicle_type):
    from pufferlib.ocean.drive.drive import Drive

    repo_root = Path(__file__).resolve().parents[1]
    return Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(
            repo_root / "pufferlib" / "resources" / "drive" / "binaries" / "training"
        ),
        control_mode="control_sdc_only",
        max_controlled_agents=1,
        ego_vehicle_type=ego_vehicle_type,
        action_type="continuous",
        dynamics_model="classic",
        resample_frequency=0,
        seed=7,
        episode_length=91,
    )


def _assert_hitch_geometry(state, row=0):
    tractor_axle = state["tractor_rear_axle_pose"][row]
    trailer_axle = state["trailer_axle_pose"][row]
    hitch = state["hitch_position"][row, :2]
    tractor_heading = tractor_axle[2]
    trailer_heading = trailer_axle[2]

    tractor_hitch = tractor_axle[:2] - 0.6 * np.array(
        [math.cos(tractor_heading), math.sin(tractor_heading)]
    )
    trailer_hitch = trailer_axle[:2] + 8.3 * np.array(
        [math.cos(trailer_heading), math.sin(trailer_heading)]
    )
    np.testing.assert_allclose(hitch, tractor_hitch, atol=2e-5)
    np.testing.assert_allclose(hitch, trailer_hitch, atol=2e-5)

    tractor_center = state["tractor_center_pose"][row]
    trailer_center = state["trailer_center_pose"][row]
    expected_tractor_center = tractor_axle[:2] + 1.65 * np.array(
        [math.cos(tractor_heading), math.sin(tractor_heading)]
    )
    expected_trailer_center = trailer_axle[:2] + 4.25 * np.array(
        [math.cos(trailer_heading), math.sin(trailer_heading)]
    )
    np.testing.assert_allclose(
        tractor_center[:2], expected_tractor_center, atol=2e-5
    )
    np.testing.assert_allclose(
        trailer_center[:2], expected_trailer_center, atol=2e-5
    )


def test_tractor_trailer_python_state_export_and_reset():
    rig = _make_drive("tractor_trailer")
    try:
        rig.reset(seed=7)
        state = rig.get_articulated_ego_state()
        expected_keys = {
            "valid",
            "tractor_center_pose",
            "trailer_center_pose",
            "tractor_rear_axle_pose",
            "trailer_axle_pose",
            "hitch_position",
            "tractor_velocity",
            "trailer_velocity",
            "speed",
            "steering_angle",
            "articulation_angle",
            "tractor_dimensions",
            "trailer_dimensions",
        }
        assert set(state) == expected_keys
        assert state["valid"].shape == (rig.num_envs,)
        assert state["valid"].dtype == np.bool_
        assert state["valid"].all()
        assert state["tractor_center_pose"].shape == (rig.num_envs, 4)
        assert state["hitch_position"].shape == (rig.num_envs, 3)
        np.testing.assert_allclose(
            state["tractor_dimensions"][0], [6.3, 2.55, 3.5], atol=1e-5
        )
        np.testing.assert_allclose(
            state["trailer_dimensions"][0], [12.5, 2.55, 3.5], atol=1e-5
        )
        _assert_hitch_geometry(state)

        rig.step(np.zeros((rig.num_agents, 2), dtype=np.float32))
        stepped = rig.get_articulated_ego_state()
        assert stepped["valid"].all()
        _assert_hitch_geometry(stepped)

        rig.reset(seed=11)
        reset = rig.get_articulated_ego_state()
        assert reset["valid"].all()
        _assert_hitch_geometry(reset)
    finally:
        rig.close()


def test_car_state_export_is_invalid_and_zero():
    car = _make_drive("car")
    try:
        state = car.get_articulated_ego_state()
        assert not state["valid"].any()
        for key, value in state.items():
            if key != "valid":
                assert np.count_nonzero(value) == 0
    finally:
        car.close()
