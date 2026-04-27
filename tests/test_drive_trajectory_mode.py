import numpy as np
import torch

from pufferlib.ocean.drive.drive import Drive, save_map_binary
from pufferlib.ocean.torch import Drive as DrivePolicy


def _linear_traj(x0, y0, dx, dy, heading=0.0, length=91):
    return {
        "position": [{"x": float(x0 + dx * t), "y": float(y0 + dy * t), "z": 0.0} for t in range(length)],
        "velocity": [{"x": float(dx / 0.1), "y": float(dy / 0.1), "z": 0.0} for _ in range(length)],
        "heading": [float(heading) for _ in range(length)],
        "valid": [1 for _ in range(length)],
    }


def _write_simple_trajectory_map(map_dir):
    objects = [
        {
            "id": 100,
            "type": "vehicle",
            "length": 4.5,
            "width": 1.9,
            "height": 1.6,
            **_linear_traj(0.0, 0.0, 1.0, 0.0),
            "goalPosition": {"x": 100.0, "y": 0.0, "z": 0.0},
        },
        {
            "id": 101,
            "type": "vehicle",
            "length": 4.5,
            "width": 1.9,
            "height": 1.6,
            **_linear_traj(10.0, 0.0, 1.0, 0.0),
            "goalPosition": {"x": 110.0, "y": 0.0, "z": 0.0},
        },
    ]
    map_data = {
        "metadata": {"sdc_track_index": 0, "tracks_to_predict": [{"track_index": 0}, {"track_index": 1}]},
        "objects": objects,
        "roads": [],
    }
    save_map_binary(map_data, str(map_dir / "map_000.bin"), unique_map_id=7)


def _make_env(tmp_path, **kwargs):
    map_dir = tmp_path / "maps"
    map_dir.mkdir(parents=True)
    _write_simple_trajectory_map(map_dir)
    base_kwargs = dict(
        num_agents=2,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_agents",
        init_mode="create_all_valid",
        resample_frequency=0,
        action_type="trajectory",
        observation_mode="trajectory_history_32",
    )
    base_kwargs.update(kwargs)
    return Drive(**base_kwargs)


def _write_trailer_history_map(map_dir):
    objects = [
        {
            "id": 200,
            "type": "vehicle",
            "length": 6.0,
            "width": 2.5,
            "height": 3.2,
            **_linear_traj(0.0, 0.0, 1.0, 0.0),
            "goalPosition": {"x": 100.0, "y": 0.0, "z": 0.0},
        },
        {
            "id": 201,
            "type": "vehicle",
            "length": 10.0,
            "width": 2.5,
            "height": 3.2,
            **_linear_traj(-8.0, 0.0, 1.0, 0.0),
            "goalPosition": {"x": 90.0, "y": 0.0, "z": 0.0},
        },
        {
            "id": 202,
            "type": "vehicle",
            "length": 4.5,
            "width": 1.9,
            "height": 1.6,
            **_linear_traj(10.0, 0.0, 1.0, 0.0),
            "goalPosition": {"x": 110.0, "y": 0.0, "z": 0.0},
        },
    ]
    map_data = {
        "metadata": {
            "sdc_track_index": 0,
            "has_ego_trailer": True,
            "ego_trailer_track_index": 1,
            "tracks_to_predict": [{"track_index": 0}],
        },
        "objects": objects,
        "roads": [],
    }
    save_map_binary(map_data, str(map_dir / "map_000.bin"), unique_map_id=8)


def _make_trailer_history_env(tmp_path, **kwargs):
    map_dir = tmp_path / "maps"
    map_dir.mkdir(parents=True)
    _write_trailer_history_map(map_dir)
    base_kwargs = dict(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        resample_frequency=0,
        action_type="trajectory",
        observation_mode="trajectory_history_32_sdc_only_with_trailer",
        trajectory_history_warmstart_seconds=0.3,
    )
    base_kwargs.update(kwargs)
    return Drive(**base_kwargs)


def _write_collision_map(map_dir):
    objects = [
        {
            "id": 100,
            "type": "vehicle",
            "length": 4.5,
            "width": 1.9,
            "height": 1.6,
            **_linear_traj(0.0, 0.0, 1.0, 0.0),
            "goalPosition": {"x": 100.0, "y": 0.0, "z": 0.0},
        },
        {
            "id": 101,
            "type": "vehicle",
            "length": 4.5,
            "width": 1.9,
            "height": 1.6,
            **_linear_traj(5.0, 0.0, 0.0, 0.0),
            "goalPosition": {"x": 5.0, "y": 0.0, "z": 0.0},
        },
    ]
    map_data = {
        "metadata": {"sdc_track_index": 0, "tracks_to_predict": [{"track_index": 0}]},
        "objects": objects,
        "roads": [],
    }
    save_map_binary(map_data, str(map_dir / "map_000.bin"), unique_map_id=9)


def _make_collision_env(tmp_path, **kwargs):
    map_dir = tmp_path / "maps"
    map_dir.mkdir(parents=True)
    _write_collision_map(map_dir)
    base_kwargs = dict(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        resample_frequency=0,
        action_type="trajectory",
        observation_mode="trajectory_history_32",
        reward_vehicle_collision=-0.5,
        collision_behavior=1,
        offroad_behavior=1,
        trajectory_control_substeps=5,
    )
    base_kwargs.update(kwargs)
    return Drive(**base_kwargs)


def test_trajectory_targets_match_next_logged_steps(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.reset(seed=0)
        targets = env.get_trajectory_targets(normalize=True).reshape(env.num_agents, 32, 5)

        np.testing.assert_allclose(targets[0, 0, 0], 0.02, atol=1e-6)
        np.testing.assert_allclose(targets[0, 0, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(targets[0, 0, 2], 0.0, atol=1e-6)
        np.testing.assert_allclose(targets[0, 0, 3], 0.1, atol=1e-6)
        np.testing.assert_allclose(targets[0, 0, 4], 1.0, atol=1e-6)
    finally:
        env.close()


def test_goal_can_be_overridden_to_gt_trajectory_end(tmp_path):
    default_env = _make_env(tmp_path / "default_goal", num_agents=1)
    gt_goal_env = _make_env(tmp_path / "gt_goal", num_agents=1, goal_at_gt_traj_end=True)
    try:
        default_obs, _ = default_env.reset(seed=0)
        gt_obs, _ = gt_goal_env.reset(seed=0)

        np.testing.assert_allclose(float(default_obs[0, 0]), 0.5, atol=1e-6)
        np.testing.assert_allclose(float(gt_obs[0, 0]), 0.45, atol=1e-6)
        assert float(gt_obs[0, 0]) < float(default_obs[0, 0])
    finally:
        default_env.close()
        gt_goal_env.close()


def test_trajectory_history_observation_tracks_previous_ego_state(tmp_path):
    env = _make_env(tmp_path)
    try:
        obs, _ = env.reset(seed=0)
        ego_hist_start = env.trajectory_base_obs_dim
        ego_hist = obs[:, ego_hist_start : ego_hist_start + env.trajectory_ego_history_dim].reshape(
            env.num_agents, env.trajectory_history_horizon, env.trajectory_history_features
        )

        np.testing.assert_allclose(ego_hist[0, 0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(ego_hist[0, 0, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(ego_hist[0, 0, 2], 1.0, atol=1e-6)
        np.testing.assert_allclose(ego_hist[0, 0, 3], 0.0, atol=1e-6)
        np.testing.assert_allclose(ego_hist[0, 0, 5], 1.0, atol=1e-6)

        targets = env.get_trajectory_targets(normalize=True)
        obs, _, _, _, _ = env.step(targets)
        ego_hist = obs[:, ego_hist_start : ego_hist_start + env.trajectory_ego_history_dim].reshape(
            env.num_agents, env.trajectory_history_horizon, env.trajectory_history_features
        )

        assert ego_hist[0, 1, 5] == 1.0
        assert ego_hist[0, 1, 0] < 0.0
        np.testing.assert_allclose(ego_hist[0, 1, 1], 0.0, atol=1e-2)
    finally:
        env.close()


def test_extended_trajectory_history_contains_trailer_pose_and_types(tmp_path):
    env = _make_trailer_history_env(tmp_path)
    try:
        obs, _ = env.reset(seed=0)
        assert env.trajectory_ego_history_features == 11
        assert env.trajectory_partner_history_features == 7

        base_ego = env._base_ego_features
        base_trailer = obs[0, base_ego + 1 : base_ego + 5]
        ego_hist_start = env.trajectory_base_obs_dim
        ego_hist_end = ego_hist_start + env.trajectory_ego_history_dim
        ego_hist = obs[:, ego_hist_start:ego_hist_end].reshape(
            env.num_agents,
            env.trajectory_history_horizon,
            env.trajectory_ego_history_features,
        )

        np.testing.assert_allclose(ego_hist[0, 0, 7:11], base_trailer, atol=1e-6)
        np.testing.assert_allclose(ego_hist[0, 0, 6], 4.0, atol=1e-6)

        # Warmstart places the env at logged timestep 3. Slot 1 is the trailer at
        # timestep 2 transformed into the current ego frame, so it is 9m behind.
        np.testing.assert_allclose(ego_hist[0, 1, 7], -9.0 * 0.02, atol=1e-6)
        np.testing.assert_allclose(ego_hist[0, 1, 8], 0.0, atol=1e-6)
        np.testing.assert_allclose(ego_hist[0, 1, 9], 1.0, atol=1e-6)
        np.testing.assert_allclose(ego_hist[0, 1, 10], 0.0, atol=1e-6)

        policy = DrivePolicy(env, input_size=32, hidden_size=64)
        with torch.no_grad():
            actions, value = policy(torch.as_tensor(obs, dtype=torch.float32))
        action_loc = actions.mean if hasattr(actions, "mean") else actions.loc
        assert torch.isfinite(action_loc).all()
        assert torch.isfinite(value).all()
    finally:
        env.close()


def test_extended_partner_history_contains_type_for_active_partner(tmp_path):
    env = _make_env(tmp_path, observation_mode="trajectory_history_32_sdc_only_with_trailer")
    try:
        obs, _ = env.reset(seed=0)
        partner_hist_start = env.trajectory_base_obs_dim + env.trajectory_ego_history_dim
        partner_hist = obs[:, partner_hist_start:].reshape(
            env.num_agents,
            env.max_partner_objects,
            env.trajectory_history_horizon,
            env.trajectory_partner_history_features,
        )

        np.testing.assert_allclose(partner_hist[0, 0, 0, 5], 1.0, atol=1e-6)
        np.testing.assert_allclose(partner_hist[0, 0, 0, 6], 1.0, atol=1e-6)
    finally:
        env.close()


def test_trajectory_controller_prefers_straight_positive_acceleration(tmp_path):
    env = _make_env(tmp_path)
    try:
        env.reset(seed=0)
        action = np.zeros((env.num_agents, env.trajectory_action_dim), dtype=np.float32).reshape(env.num_agents, 32, 5)
        action[:, :, 0] = 0.04
        action[:, :, 3] = 0.2
        action[:, :, 4] = 1.0

        controls = env._trajectory_to_control_actions(action)
        np.testing.assert_allclose(controls[:, 1], np.zeros(env.num_agents), atol=1e-4)
        assert np.all(controls[:, 0] > 0.0)
    finally:
        env.close()


def test_trajectory_controller_respects_tunable_gains(tmp_path):
    base_env = _make_env(tmp_path / "base")
    steer_env = _make_env(
        tmp_path / "steer",
        trajectory_steer_gain=2.0,
        trajectory_accel_gain=1.5,
    )
    try:
        base_env.reset(seed=0)
        steer_env.reset(seed=0)
        action = np.zeros((base_env.num_agents, base_env.trajectory_action_dim), dtype=np.float32).reshape(
            base_env.num_agents, 32, 5
        )
        action[:, :, 0] = 0.06
        action[:, :, 1] = 0.02
        action[:, :, 3] = 0.12
        action[:, :, 4] = 1.0

        base_controls = base_env._trajectory_to_control_actions(action)
        tuned_controls = steer_env._trajectory_to_control_actions(action)

        assert np.all(tuned_controls[:, 0] > base_controls[:, 0])
        assert np.all(tuned_controls[:, 1] > base_controls[:, 1])
    finally:
        base_env.close()
        steer_env.close()


def test_stanley_tracker_prefers_zero_steer_on_straight_path(tmp_path):
    env = _make_env(tmp_path, trajectory_tracker_type="stanley")
    try:
        env.reset(seed=0)
        action = np.zeros((env.num_agents, env.trajectory_action_dim), dtype=np.float32).reshape(env.num_agents, 32, 5)
        action[:, :, 0] = np.linspace(0.02, 0.64, 32, dtype=np.float32)
        action[:, :, 3] = 0.2
        action[:, :, 4] = 1.0

        controls = env._trajectory_to_control_actions(action)
        np.testing.assert_allclose(controls[:, 1], np.zeros(env.num_agents), atol=1e-4)
    finally:
        env.close()


def test_stanley_tracker_corrects_left_offset_with_positive_steer(tmp_path):
    env = _make_env(
        tmp_path,
        trajectory_tracker_type="stanley",
        trajectory_stanley_gain=2.0,
        trajectory_stanley_softening=0.5,
    )
    try:
        env.reset(seed=0)
        action = np.zeros((env.num_agents, env.trajectory_action_dim), dtype=np.float32).reshape(env.num_agents, 32, 5)
        action[:, :, 0] = np.linspace(0.02, 0.64, 32, dtype=np.float32)
        action[:, :, 1] = 0.02
        action[:, :, 3] = 0.15
        action[:, :, 4] = 1.0

        controls = env._trajectory_to_control_actions(action)
        assert np.all(controls[:, 1] > 0.0)
    finally:
        env.close()


def test_mpc_tracker_prefers_straight_positive_acceleration(tmp_path):
    env = _make_env(tmp_path, trajectory_tracker_type="mpc")
    try:
        env.reset(seed=0)
        action = np.zeros((env.num_agents, env.trajectory_action_dim), dtype=np.float32).reshape(env.num_agents, 32, 5)
        action[:, :, 0] = np.linspace(0.02, 0.64, 32, dtype=np.float32)
        action[:, :, 3] = 0.15
        action[:, :, 4] = 1.0

        controls = env._trajectory_to_control_actions(action)
        np.testing.assert_allclose(controls[:, 1], np.zeros(env.num_agents), atol=0.34)
        assert np.all(controls[:, 0] >= 0.0)
    finally:
        env.close()


def test_mpc_tracker_corrects_left_offset_with_positive_steer(tmp_path):
    env = _make_env(tmp_path, trajectory_tracker_type="mpc")
    try:
        env.reset(seed=0)
        action = np.zeros((env.num_agents, env.trajectory_action_dim), dtype=np.float32).reshape(env.num_agents, 32, 5)
        action[:, :, 0] = np.linspace(0.02, 0.64, 32, dtype=np.float32)
        action[:, :, 1] = 0.04
        action[:, :, 3] = 0.15
        action[:, :, 4] = 1.0

        controls = env._trajectory_to_control_actions(action)
        assert np.all(controls[:, 1] > 0.0)
    finally:
        env.close()


def test_mpc_tracker_logs_physical_control_application(tmp_path):
    env = _make_env(tmp_path, trajectory_tracker_type="mpc")
    try:
        env.reset(seed=0)
        action = np.zeros((env.num_agents, env.trajectory_action_dim), dtype=np.float32).reshape(env.num_agents, 32, 5)
        action[:, :, 0] = np.linspace(0.02, 0.64, 32, dtype=np.float32)
        action[:, :, 3] = 0.15
        action[:, :, 4] = 1.0

        _ = env._trajectory_to_control_actions(action)
        debug = env.get_last_trajectory_control_debug()
        assert len(debug) == env.num_agents
        assert all(item["tracker_type"] == "mpc" for item in debug)
        assert all(item["controls_are_physical"] for item in debug)
        assert all("requested_acceleration" in item for item in debug)
        assert all("applied_steering" in item for item in debug)
    finally:
        env.close()


def test_trajectory_smoothing_reduces_lateral_zigzag(tmp_path):
    env = _make_env(
        tmp_path / "smooth",
        trajectory_smoothing_window=3,
        trajectory_smoothing_blend=1.0,
    )
    try:
        zigzag = np.zeros((env.trajectory_horizon, 2), dtype=np.float32)
        zigzag[:, 0] = np.linspace(1.0, 32.0, env.trajectory_horizon, dtype=np.float32)
        zigzag[:, 1] = np.where(np.arange(env.trajectory_horizon) % 2 == 0, 1.0, -1.0).astype(np.float32)
        valid_mask = np.ones(env.trajectory_horizon, dtype=bool)

        smoothed = env._smooth_trajectory_positions(zigzag, valid_mask)

        np.testing.assert_allclose(smoothed[0], zigzag[0], atol=1e-6)
        assert np.max(np.abs(smoothed[1:, 1])) < np.max(np.abs(zigzag[1:, 1]))
    finally:
        env.close()


def test_trajectory_smoothness_penalty_prefers_smoother_paths(tmp_path):
    env = _make_env(
        tmp_path / "smooth_reward",
        trajectory_smoothness_xy_weight=0.1,
    )
    try:
        straight_xy = np.zeros((env.trajectory_horizon, 2), dtype=np.float32)
        straight_xy[:, 0] = np.linspace(1.0, 32.0, env.trajectory_horizon, dtype=np.float32)
        zigzag_xy = straight_xy.copy()
        zigzag_xy[:, 1] = np.where(np.arange(env.trajectory_horizon) % 2 == 0, 1.0, -1.0).astype(np.float32)
        heading = np.zeros(env.trajectory_horizon, dtype=np.float32)
        speed = np.ones(env.trajectory_horizon, dtype=np.float32)
        valid_mask = np.ones(env.trajectory_horizon, dtype=bool)

        straight_penalty = env._compute_trajectory_smoothness_penalty(straight_xy, heading, speed, valid_mask)
        zigzag_penalty = env._compute_trajectory_smoothness_penalty(zigzag_xy, heading, speed, valid_mask)

        np.testing.assert_allclose(straight_penalty, 0.0, atol=1e-6)
        assert zigzag_penalty < straight_penalty
    finally:
        env.close()


def test_trajectory_step_adds_smoothness_reward_penalty(tmp_path):
    env = _make_env(
        tmp_path / "step_smooth_reward",
        trajectory_smoothness_xy_weight=0.1,
    )
    try:
        env.reset(seed=0)
        action = np.zeros((env.num_agents, env.trajectory_horizon, env.trajectory_features), dtype=np.float32)
        action[:, :, 0] = np.linspace(0.02, 0.64, env.trajectory_horizon, dtype=np.float32)
        action[:, :, 3] = 0.1
        action[:, :, 4] = 1.0
        zigzag_action = action.copy()
        zigzag_action[:, :, 1] = np.where(np.arange(env.trajectory_horizon) % 2 == 0, 0.04, -0.04).astype(np.float32)

        _, rewards, _, _, _ = env.step(zigzag_action.reshape(env.num_agents, env.trajectory_action_dim))

        assert np.all(env._trajectory_smoothness_penalties < 0.0)
        assert np.all(rewards < env._trajectory_smoothness_penalties + 1e-6)
        assert np.all(np.abs(rewards - env._trajectory_smoothness_penalties) < 1e-2)
    finally:
        env.close()


def test_physics_substep_advances_state_without_advancing_logged_tick(tmp_path):
    env = _make_env(tmp_path / "substep", num_agents=1)
    try:
        env.reset(seed=0)
        start_state = env.get_global_agent_state()
        start_x = float(start_state["x"][0])
        assert env.tick == 0

        targets = env.get_trajectory_targets(normalize=True)
        env.physics_substep(targets, sub_dt=env.dt / 2.0, alpha=0.5)

        mid_state = env.get_global_agent_state()
        mid_x = float(mid_state["x"][0])
        assert env.tick == 0
        assert mid_x > start_x

        env.advance_logged_timestep()
        assert env.tick == 1
    finally:
        env.close()


def test_physics_substep_can_skip_history_append_until_outer_step(tmp_path):
    env = _make_env(tmp_path / "substep_history", num_agents=1)
    try:
        obs, _ = env.reset(seed=0)
        ego_hist_start = env.trajectory_base_obs_dim

        def valid_count(observation):
            ego_hist = observation[:, ego_hist_start : ego_hist_start + env.trajectory_ego_history_dim].reshape(
                env.num_agents, env.trajectory_history_horizon, env.trajectory_history_features
            )
            return int((ego_hist[0, :, 5] > 0.5).sum())

        start_valid = valid_count(obs)
        targets = env.get_trajectory_targets(normalize=True)
        obs, _, _, _, _ = env.physics_substep(targets, sub_dt=env.dt / 5.0, alpha=0.2, update_history=False)
        after_substep_valid = valid_count(obs)
        assert after_substep_valid == start_valid

        obs, _, _, _, _ = env.advance_logged_timestep(update_history=True)
        after_outer_step_valid = valid_count(obs)
        assert after_outer_step_valid == start_valid + 1
    finally:
        env.close()


def test_reset_can_warmstart_trajectory_history_from_logged_states(tmp_path):
    env = _make_env(
        tmp_path / "warmstart_history",
        num_agents=1,
        trajectory_history_warmstart_seconds=0.3,
    )
    try:
        obs, _ = env.reset(seed=0)
        ego_hist_start = env.trajectory_base_obs_dim
        ego_hist = obs[:, ego_hist_start : ego_hist_start + env.trajectory_ego_history_dim].reshape(
            env.num_agents, env.trajectory_history_horizon, env.trajectory_history_features
        )
        valid_count = int((ego_hist[0, :, 5] > 0.5).sum())

        assert env.tick == 3
        assert valid_count == 4
    finally:
        env.close()


def test_step_with_control_substeps_only_appends_history_once_per_outer_step(tmp_path):
    env = _make_env(
        tmp_path / "step_substeps",
        num_agents=1,
        trajectory_control_substeps=5,
    )
    try:
        obs, _ = env.reset(seed=0)
        ego_hist_start = env.trajectory_base_obs_dim

        def valid_count(observation):
            ego_hist = observation[:, ego_hist_start : ego_hist_start + env.trajectory_ego_history_dim].reshape(
                env.num_agents, env.trajectory_history_horizon, env.trajectory_history_features
            )
            return int((ego_hist[0, :, 5] > 0.5).sum())

        start_valid = valid_count(obs)
        start_x = float(env.get_global_agent_state()["x"][0])
        targets = env.get_trajectory_targets(normalize=True)
        obs, _, _, _, _ = env.step(targets)
        end_x = float(env.get_global_agent_state()["x"][0])
        end_valid = valid_count(obs)

        assert env.tick == 1
        assert end_x > start_x
        assert end_valid == start_valid + 1
    finally:
        env.close()


def test_step_with_control_substeps_preserves_outer_step_collision_reward(tmp_path):
    env = _make_collision_env(tmp_path / "substep_reward")
    try:
        env.reset(seed=0)
        targets = env.get_trajectory_targets(normalize=True)
        _obs, rewards, _terminals, _truncations, _info = env.step(targets)

        np.testing.assert_allclose(float(rewards[0]), -0.5, atol=1e-6)
    finally:
        env.close()


def test_terminate_on_stop_resets_with_warmstarted_history(tmp_path):
    env = _make_collision_env(
        tmp_path / "terminate_on_stop",
        terminate_on_stop=True,
        trajectory_history_warmstart_seconds=0.3,
    )
    try:
        obs, _ = env.reset(seed=0)
        ego_hist_start = env.trajectory_base_obs_dim

        def valid_count(observation):
            ego_hist = observation[:, ego_hist_start : ego_hist_start + env.trajectory_ego_history_dim].reshape(
                env.num_agents, env.trajectory_history_horizon, env.trajectory_history_features
            )
            return int((ego_hist[0, :, 5] > 0.5).sum())

        targets = env.get_trajectory_targets(normalize=True)
        _obs, rewards, terminals, _truncations, info = env.step(targets)

        assert bool(terminals[0])
        assert env.done is True
        np.testing.assert_allclose(float(rewards[0]), -0.5, atol=1e-6)

        obs, _ = env.reset()
        assert env.tick == 3
        assert valid_count(obs) == 4
        assert env.done is False
        assert any(item.get("terminal_stop_reasons") for item in info if isinstance(item, dict))
    finally:
        env.close()
