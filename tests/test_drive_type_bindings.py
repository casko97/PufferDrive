import numpy as np
import pytest
import torch

from pufferlib.ocean.drive.drive import Drive, save_map_binary
from pufferlib.ocean.torch import Drive as DrivePolicy


def _constant_traj(x, y, heading=0.0, length=91):
    return {
        "position": [{"x": float(x), "y": float(y), "z": 0.0} for _ in range(length)],
        "velocity": [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(length)],
        "heading": [float(heading) for _ in range(length)],
        "valid": [1 for _ in range(length)],
    }


def _write_type_test_map(map_dir):
    # Active-agent order is deterministic: SDC first, then remaining controllable
    # entities in object index order.
    objects = [
        {
            "id": 100,
            "type": "vehicle",
            "length": 4.5,
            "width": 1.9,
            "height": 1.6,
            **_constant_traj(0.0, 0.0),
            "goalPosition": {"x": 10.0, "y": 0.0, "z": 0.0},
        },
        {
            "id": 101,
            "type": "pedestrian",
            "length": 0.5,
            "width": 0.5,
            "height": 1.8,
            **_constant_traj(10.0, 0.0),
            "goalPosition": {"x": 20.0, "y": 0.0, "z": 0.0},
        },
        {
            "id": 102,
            "type": "cyclist",
            "length": 1.8,
            "width": 0.6,
            "height": 1.7,
            **_constant_traj(0.0, 10.0),
            "goalPosition": {"x": 0.0, "y": 20.0, "z": 0.0},
        },
        {
            "id": 103,
            "type": "vehicle",
            "length": 12.0,
            "width": 2.8,
            "height": 3.8,
            **_constant_traj(10.0, 10.0),
            "goalPosition": {"x": 20.0, "y": 10.0, "z": 0.0},
        },
    ]
    map_data = {
        "metadata": {"sdc_track_index": 0, "tracks_to_predict": [{"track_index": i} for i in range(len(objects))]},
        "objects": objects,
        "roads": [],
    }
    save_map_binary(map_data, str(map_dir / "map_000.bin"), unique_map_id=1)


def _write_trailer_type_test_map(map_dir):
    objects = [
        {
            "id": 200,
            "type": "vehicle",
            "length": 6.0,
            "width": 2.5,
            "height": 3.2,
            **_constant_traj(0.0, 0.0),
            "goalPosition": {"x": 20.0, "y": 0.0, "z": 0.0},
        },
        {
            "id": 201,
            "type": "vehicle",
            "length": 10.0,
            "width": 2.5,
            "height": 3.2,
            **_constant_traj(-8.0, 0.0),
            "goalPosition": {"x": 12.0, "y": 0.0, "z": 0.0},
        },
        {
            "id": 202,
            "type": "vehicle",
            "length": 4.5,
            "width": 1.9,
            "height": 1.6,
            **_constant_traj(10.0, 0.0),
            "goalPosition": {"x": 25.0, "y": 0.0, "z": 0.0},
        },
    ]
    map_data = {
        "metadata": {
            "sdc_track_index": 0,
            "has_ego_trailer": True,
            "ego_trailer_track_index": 1,
            "tracks_to_predict": [{"track_index": 0}, {"track_index": 2}],
        },
        "objects": objects,
        "roads": [],
    }
    save_map_binary(map_data, str(map_dir / "map_000.bin"), unique_map_id=2)


def test_drive_type_api_order_and_padding(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_type_test_map(map_dir)

    env = Drive(
        num_agents=4,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_agents",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="default",
    )
    try:
        obs, _ = env.reset(seed=0)

        # Expected active-agent type order:
        # [vehicle_single, pedestrian, cyclist, vehicle_single] -> [1, 2, 3, 1]
        expected_global = np.array([1, 2, 3, 1], dtype=np.int32)
        global_types = env.get_global_agent_types()
        np.testing.assert_array_equal(global_types, expected_global)

        # Partner ordering must match simulator partner observation ordering.
        # For each row: all other active agents in active-agent order (self removed), then zero-padding.
        expected_partner_prefix = np.array(
            [
                [2, 3, 1],  # ego row 0 sees rows 1,2,3
                [1, 3, 1],  # row 1 sees rows 0,2,3
                [1, 2, 1],  # row 2 sees rows 0,1,3
                [1, 2, 3],  # row 3 sees rows 0,1,2
            ],
            dtype=np.int32,
        )
        partner_types = env.get_partner_types()
        np.testing.assert_array_equal(partner_types[:, :3], expected_partner_prefix)
        np.testing.assert_array_equal(partner_types[:, 3:], np.zeros_like(partner_types[:, 3:]))

        # Additional consistency check: occupied partner obs slots should align with non-zero type slots.
        ego_dim = env.ego_features
        partner_dim = env.max_partner_objects * env.partner_features
        partner_obs = obs[:, ego_dim : ego_dim + partner_dim].reshape(env.num_agents, env.max_partner_objects, env.partner_features)
        occupied_partner_slots = np.any(np.abs(partner_obs) > 0.0, axis=2)
        np.testing.assert_array_equal(occupied_partner_slots, partner_types != 0)
    finally:
        env.close()


def test_augmented_observation_mode_type_slots_and_padding(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_type_test_map(map_dir)

    env = Drive(
        num_agents=4,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_agents",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="sdc_only_with_trailer",
    )
    try:
        obs, _ = env.reset(seed=0)

        assert env.ego_features == env._base_ego_features + 5
        assert env.partner_features == env._base_partner_features + 1
        assert obs.shape == (env.num_agents, env.num_obs)

        base_ego = env._base_ego_features
        base_partner = env._base_partner_features

        # Ego type id slot should be filled from API types.
        expected_global_types = np.array([1, 2, 3, 1], dtype=np.float32)
        np.testing.assert_array_equal(obs[:, base_ego], expected_global_types)
        np.testing.assert_array_equal(obs[:, base_ego + 1 : base_ego + 5], np.zeros((env.num_agents, 4), dtype=np.float32))

        aug_partner_start = env.ego_features
        aug_partner_dim = env.max_partner_objects * env.partner_features
        aug_partner_view = obs[:, aug_partner_start : aug_partner_start + aug_partner_dim].reshape(
            env.num_agents, env.max_partner_objects, env.partner_features
        )

        # Type channel appended after base partner features.
        partner_type_channel = aug_partner_view[:, :, base_partner]
        expected_partner_prefix = np.array(
            [
                [2, 3, 1],
                [1, 3, 1],
                [1, 2, 1],
                [1, 2, 3],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(partner_type_channel[:, :3], expected_partner_prefix)
        np.testing.assert_array_equal(
            partner_type_channel[:, 3:], np.zeros_like(partner_type_channel[:, 3:], dtype=np.float32)
        )

        # Empty/non-existing partner slots must stay padded with 0 in the added type field.
        base_partner_features = aug_partner_view[:, :, :base_partner]
        occupied_slots = np.any(np.abs(base_partner_features) > 1e-8, axis=2)
        np.testing.assert_array_equal(partner_type_channel[~occupied_slots], np.zeros(np.count_nonzero(~occupied_slots)))
    finally:
        env.close()


def test_augmented_observation_marks_tractor_trailer_as_vehicle_multi(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_trailer_type_test_map(map_dir)

    env = Drive(
        num_agents=2,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="sdc_only_with_trailer",
    )
    try:
        obs, _ = env.reset(seed=0)

        expected_global_types = np.array([4, 1], dtype=np.int32)
        np.testing.assert_array_equal(env.get_global_agent_types(), expected_global_types)

        base_ego = env._base_ego_features
        np.testing.assert_array_equal(obs[:, base_ego], expected_global_types.astype(np.float32))
        np.testing.assert_allclose(obs[0, base_ego + 1], -0.16, atol=1e-5)
        np.testing.assert_allclose(obs[0, base_ego + 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(obs[0, base_ego + 3], 1.0, atol=1e-5)
        np.testing.assert_allclose(obs[0, base_ego + 4], 0.0, atol=1e-5)
        np.testing.assert_array_equal(obs[1, base_ego + 1 : base_ego + 5], np.zeros(4, dtype=np.float32))

        aug_partner_start = env.ego_features
        aug_partner_dim = env.max_partner_objects * env.partner_features
        aug_partner_view = obs[:, aug_partner_start : aug_partner_start + aug_partner_dim].reshape(
            env.num_agents, env.max_partner_objects, env.partner_features
        )
        partner_type_channel = aug_partner_view[:, :, env._base_partner_features]

        # SDC/tractor row hides its own trailer and only sees the other car.
        assert partner_type_channel[0, 0] == 1.0
        # Other controlled vehicle sees the tractor and trailer bodies as multi-body vehicle features.
        assert partner_type_channel[1, 0] == 4.0
        assert partner_type_channel[1, 1] == 4.0
    finally:
        env.close()


def test_augmented_observation_policy_forward_pass_with_trailer_features(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_trailer_type_test_map(map_dir)

    env = Drive(
        num_agents=2,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="sdc_only_with_trailer",
        action_type="discrete",
    )
    try:
        obs, _ = env.reset(seed=0)
        policy = DrivePolicy(env)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            actions, value = policy(obs_tensor)

        assert isinstance(actions, tuple)
        assert len(actions) == len(env.single_action_space.nvec)
        assert actions[0].shape == (env.num_agents, int(env.single_action_space.nvec[0]))
        assert value.shape == (env.num_agents, 1)
        assert torch.isfinite(actions[0]).all()
        assert torch.isfinite(value).all()
    finally:
        env.close()
