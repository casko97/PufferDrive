import numpy as np
import pytest
import torch
import shutil
from pathlib import Path
from types import SimpleNamespace

from pufferlib.ocean.drive import drive as drive_module
from pufferlib.ocean.drive.drive import Drive, build_bc_dataset, binding, save_map_binary, train_bc_policy
from pufferlib.ocean.torch import Drive as DrivePolicy
from pufferlib.pufferl import load_policy

CLASSIC_ACTION_SPACE = 9 * 13
CLASSIC_STEERING_VALUES = np.asarray(
    (-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0), dtype=np.float32
)
CLASSIC_ACCELERATION_VALUES = np.asarray((-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0), dtype=np.float32)
TEST_MATCH_WEIGHT_LATERAL = 3.0
TEST_MATCH_WEIGHT_LONGITUDINAL = 1.5
TEST_MATCH_WEIGHT_HEADING = 0.1
TEST_MATCH_WEIGHT_SPEED = 0.05
TEST_MATCH_WEIGHT_STEER_CHANGE = 0.15
TEST_MATCH_WEIGHT_ACCEL_CHANGE = 0.02
TEST_MATCH_WEIGHT_REVERSE = 1.0
TEST_MATCH_WEIGHT_PROGRESS = 4.0
TEST_MATCH_WEIGHT_STEER_FLIP = 0.5
TEST_MATCH_WEIGHT_REF_ACCEL = 0.01
TEST_MATCH_WEIGHT_REF_STEER = 0.1
TEST_REAL_CASE_PLANNING_HORIZON = 20
TEST_GOAL_RADIUS = 0.2
TRAINING_MAP_DIR = Path("resources/drive/binaries/training")


def _make_logged_vehicle(track_id, xs, ys, goal_x, goal_y, length=4.5, width=1.9):
    velocities = []
    headings = []
    for idx in range(len(xs)):
        if idx + 1 < len(xs):
            dx = xs[idx + 1] - xs[idx]
            dy = ys[idx + 1] - ys[idx]
        else:
            dx = xs[idx] - xs[idx - 1]
            dy = ys[idx] - ys[idx - 1]
        velocities.append({"x": float(dx / 0.1), "y": float(dy / 0.1), "z": 0.0})
        headings.append(float(np.arctan2(dy, dx if abs(dx) > 1e-6 else 1e-6)))

    valid_prefix = [1] * len(xs)
    pad = 91 - len(xs)
    return {
        "id": track_id,
        "type": "vehicle",
        "position": [{"x": float(x), "y": float(y), "z": 0.0} for x, y in zip(xs, ys)]
        + [{"x": float(xs[-1]), "y": float(ys[-1]), "z": 0.0}] * pad,
        "velocity": velocities + [{"x": 0.0, "y": 0.0, "z": 0.0}] * pad,
        "heading": headings + [headings[-1]] * pad,
        "valid": valid_prefix + [0] * pad,
        "width": width,
        "length": length,
        "height": 1.6,
        "goalPosition": {"x": float(goal_x), "y": float(goal_y), "z": 0.0},
        "mark_as_expert": 0,
    }


def _write_bc_test_map(map_dir, ego_x=None, ego_y=None, map_filename="map_000.bin", unique_map_id=123):
    if ego_x is None:
        ego_x = [0.0, 0.8, 1.7, 2.7, 3.8, 5.0]
    if ego_y is None:
        ego_y = [0.0, 0.0, 0.05, 0.15, 0.30, 0.5]
    partner_x = [5.0] * 6
    partner_y = [3.0] * 6

    scenario = {
        "metadata": {"sdc_track_index": 0, "tracks_to_predict": [{"track_index": 0}]},
        "objects": [
            _make_logged_vehicle(10, ego_x, ego_y, goal_x=8.0, goal_y=1.0),
            _make_logged_vehicle(11, partner_x, partner_y, goal_x=5.0, goal_y=3.0),
        ],
        "roads": [
            {
                "id": 100,
                "type": "lane",
                "geometry": [{"x": -5.0, "y": -1.0, "z": 0.0}, {"x": 15.0, "y": 1.5, "z": 0.0}],
                "width": 3.5,
                "length": 20.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            }
        ],
    }
    save_map_binary(scenario, str(map_dir / map_filename), unique_map_id=unique_map_id)
    return np.asarray(ego_x, dtype=np.float32), np.asarray(ego_y, dtype=np.float32)


BC_TRAJECTORY_CASES = [
    {
        "name": "straight_accel",
        "ego_x": [0.0, 0.6, 1.4, 2.4, 3.6, 5.0],
        "ego_y": [0.0, 0.0, 0.0, 0.0, 0.02, 0.03],
        "max_total_cost": 45.0,
        "max_mean_disp": 0.6,
        "max_final_disp": 1.6,
    },
    {
        "name": "gentle_left_curve",
        "ego_x": [0.0, 0.8, 1.7, 2.7, 3.8, 5.0],
        "ego_y": [0.0, 0.0, 0.05, 0.15, 0.30, 0.5],
        "max_total_cost": 8.0,
        "max_mean_disp": 0.25,
        "max_final_disp": 0.6,
    },
    {
        "name": "stronger_lateral",
        "ego_x": [0.0, 0.7, 1.5, 2.2, 2.8, 3.4],
        "ego_y": [0.0, 0.08, 0.30, 0.70, 1.20, 1.80],
        "max_total_cost": 1.0,
        "max_mean_disp": 0.1,
        "max_final_disp": 0.2,
    },
]


REAL_TRAJECTORY_CASES = [
    {
        "name": "real_straight",
        "source_map": "map_027.bin",
        "max_total_cost": 2.0,
        "max_mean_disp": 0.08,
        "max_final_disp": 0.08,
        "planning_horizon": TEST_REAL_CASE_PLANNING_HORIZON,
    },
    {
        "name": "real_gentle_turn",
        "source_map": "map_020.bin",
        "max_total_cost": 1.2,
        "max_mean_disp": 0.08,
        "max_final_disp": 0.25,
        "planning_horizon": TEST_REAL_CASE_PLANNING_HORIZON,
    },
    {
        "name": "real_stronger_turn",
        "source_map": "map_037.bin",
        "max_total_cost": 0.6,
        "max_mean_disp": 0.06,
        "max_final_disp": 0.08,
        "planning_horizon": TEST_REAL_CASE_PLANNING_HORIZON,
    },
    {
        "name": "real_deceleration",
        "source_map": "map_022.bin",
        "max_total_cost": 0.6,
        "max_mean_disp": 0.035,
        "max_final_disp": 0.06,
        "planning_horizon": TEST_REAL_CASE_PLANNING_HORIZON,
    },
    {
        "name": "real_acceleration",
        "source_map": "map_050.bin",
        "max_total_cost": 0.5,
        "max_mean_disp": 0.06,
        "max_final_disp": 0.10,
        "planning_horizon": TEST_REAL_CASE_PLANNING_HORIZON,
    },
]


def _builder_args(map_dir, action_type="discrete", *, export_windows=False, window_seq_len=32, window_stride=32):
    return {
        "env": {
            "map_dir": str(map_dir),
            "num_maps": 1,
            "num_agents": 1,
            "max_controlled_agents": 1,
            "action_type": action_type,
            "dynamics_model": "classic",
            "extend_classic_action_space": True,
            "observation_mode": "default",
            "reward_vehicle_collision": -0.5,
            "reward_offroad_collision": -0.5,
            "reward_goal": 1.0,
            "reward_goal_post_respawn": 0.25,
            "goal_radius": TEST_GOAL_RADIUS,
            "goal_speed": 100.0,
            "goal_behavior": 0,
            "goal_target_distance": 30.0,
            "collision_behavior": 0,
            "offroad_behavior": 0,
            "dt": 0.1,
            "episode_length": 91,
            "termination_mode": 1,
            "init_steps": 0,
            "init_mode": "create_all_valid",
            "control_mode": "control_vehicles",
        },
        "bc": {
            "output_dir": str(map_dir / "bc"),
            "export_windows": bool(export_windows),
            "window_seq_len": int(window_seq_len),
            "window_stride": int(window_stride),
            "beam_width": 4,
            "planning_horizon": 5,
            "match_weight_lateral": TEST_MATCH_WEIGHT_LATERAL,
            "match_weight_longitudinal": TEST_MATCH_WEIGHT_LONGITUDINAL,
            "match_weight_heading": TEST_MATCH_WEIGHT_HEADING,
            "match_weight_speed": TEST_MATCH_WEIGHT_SPEED,
            "match_weight_steer_change": TEST_MATCH_WEIGHT_STEER_CHANGE,
            "match_weight_accel_change": TEST_MATCH_WEIGHT_ACCEL_CHANGE,
            "match_weight_reverse": TEST_MATCH_WEIGHT_REVERSE,
            "match_weight_progress": TEST_MATCH_WEIGHT_PROGRESS,
            "match_weight_steer_flip": TEST_MATCH_WEIGHT_STEER_FLIP,
            "match_weight_ref_accel": TEST_MATCH_WEIGHT_REF_ACCEL,
            "match_weight_ref_steer": TEST_MATCH_WEIGHT_REF_STEER,
            "skip_existing_shards": False,
            "max_maps": 1,
        },
    }


def _bc_train_args(map_dir, dataset_dir, rnn_name="Recurrent"):
    builder_env = _builder_args(map_dir)["env"]
    return {
        "package": "ocean",
        "env_name": "puffer_drive",
        "policy_name": "Drive",
        "rnn_name": rnn_name,
        "policy": {
            "input_size": 32,
            "hidden_size": 32,
        },
        "rnn": {
            "input_size": 32,
            "hidden_size": 32,
        },
        "train": {
            "device": "cpu",
            "seed": 7,
            "learning_rate": 0.01,
            "bptt_horizon": 4,
        },
        "env": {
            **builder_env,
            "num_maps": 1,
            "num_agents": 1,
            "max_controlled_agents": 1,
        },
        "bc_train": {
            "dataset_dir": str(dataset_dir),
            "output_dir": str(Path(dataset_dir) / "checkpoints"),
            "device": "cpu",
            "use_embedded_windows": False,
            "epochs": 4,
            "batch_size": 4,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "num_workers": 0,
            "val_fraction": 0.0,
            "seq_len": 4,
            "sequence_stride": 4,
            "max_shards": -1,
            "save_best": True,
            "log_interval": 0,
        },
    }


def _make_binding_env(map_dir, max_agents=1):
    obs_dim = binding.EGO_FEATURES_CLASSIC + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    observations = np.zeros((max_agents, obs_dim), dtype=np.float32)
    actions = np.zeros(max_agents, dtype=np.int32)
    rewards = np.zeros(max_agents, dtype=np.float32)
    terminals = np.zeros(max_agents, dtype=np.uint8)
    truncations = np.zeros(max_agents, dtype=np.uint8)
    env_handle = binding.env_init(
        observations,
        actions,
        rewards,
        terminals,
        truncations,
        0,
        human_agent_idx=0,
        reward_vehicle_collision=-0.5,
        reward_offroad_collision=-0.5,
        reward_goal=1.0,
        reward_goal_post_respawn=0.25,
        goal_radius=TEST_GOAL_RADIUS,
        goal_speed=100.0,
        goal_behavior=0,
        goal_target_distance=30.0,
        collision_behavior=0,
        offroad_behavior=0,
        dt=0.1,
        episode_length=91,
        termination_mode=1,
        max_controlled_agents=max_agents,
        map_id=0,
        max_agents=max_agents,
        ini_file="pufferlib/config/ocean/drive.ini",
        init_steps=0,
        init_mode=0,
        control_mode=0,
        map_dir=str(map_dir),
        non_kinematic_vehicle_params_override=None,
        force_zero_trailer_articulation_at_init=0,
        extend_classic_action_space=1,
    )
    binding.env_reset(env_handle, 0)
    return env_handle


def _fit_and_rollout_case(map_dir, case):
    gt_x, gt_y = _write_bc_test_map(map_dir, ego_x=case["ego_x"], ego_y=case["ego_y"])

    env_handle = _make_binding_env(map_dir)
    try:
        max_steps = 90
        actions = np.full(max_steps, -1, dtype=np.int32)
        step_costs = np.zeros(max_steps, dtype=np.float32)
        step_lat_costs = np.zeros(max_steps, dtype=np.float32)
        step_lon_costs = np.zeros(max_steps, dtype=np.float32)
        num_steps, total_cost, total_lat_cost, total_lon_cost = binding.env_fit_discrete_action_sequence(
            env_handle,
            0,
            12,
            5,
            TEST_MATCH_WEIGHT_LATERAL,
            TEST_MATCH_WEIGHT_LONGITUDINAL,
            TEST_MATCH_WEIGHT_HEADING,
            TEST_MATCH_WEIGHT_SPEED,
            TEST_MATCH_WEIGHT_STEER_CHANGE,
            TEST_MATCH_WEIGHT_ACCEL_CHANGE,
            TEST_MATCH_WEIGHT_REVERSE,
            TEST_MATCH_WEIGHT_PROGRESS,
            TEST_MATCH_WEIGHT_STEER_FLIP,
            TEST_MATCH_WEIGHT_REF_ACCEL,
            TEST_MATCH_WEIGHT_REF_STEER,
            actions,
            step_costs,
            step_lat_costs,
            step_lon_costs,
        )
    finally:
        binding.env_close(env_handle)

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        goal_radius=TEST_GOAL_RADIUS,
        init_steps=0,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="default",
        extend_classic_action_space=True,
    )
    try:
        env.reset(seed=0)
        rollout_x = []
        rollout_y = []
        state = env.get_global_agent_state()
        rollout_x.append(float(state["x"][0]))
        rollout_y.append(float(state["y"][0]))

        action_buffer = np.zeros(1, dtype=np.int32)
        for step_idx in range(num_steps):
            action_buffer[0] = int(actions[step_idx])
            env.step(action_buffer)
            state = env.get_global_agent_state()
            rollout_x.append(float(state["x"][0]))
            rollout_y.append(float(state["y"][0]))
    finally:
        env.close()

    rollout_x = np.asarray(rollout_x, dtype=np.float32)
    rollout_y = np.asarray(rollout_y, dtype=np.float32)
    displacement = np.sqrt((rollout_x - gt_x) ** 2 + (rollout_y - gt_y) ** 2)
    steering_values = CLASSIC_STEERING_VALUES[actions[:num_steps] % len(CLASSIC_STEERING_VALUES)]
    nonzero_steering = steering_values[np.abs(steering_values) > 1e-6]
    steering_reversals = 0
    if nonzero_steering.size > 1:
        steering_reversals = int(np.count_nonzero(np.diff(np.sign(nonzero_steering)) != 0))

    return {
        "name": case["name"],
        "gt_x": gt_x,
        "gt_y": gt_y,
        "rollout_x": rollout_x,
        "rollout_y": rollout_y,
        "actions": actions[:num_steps].copy(),
        "step_costs": step_costs[:num_steps].copy(),
        "step_lat_costs": step_lat_costs[:num_steps].copy(),
        "step_lon_costs": step_lon_costs[:num_steps].copy(),
        "num_steps": num_steps,
        "steering_values": steering_values.copy(),
        "steering_reversals": steering_reversals,
        "total_cost": float(total_cost),
        "total_lat_cost": float(total_lat_cost),
        "total_lon_cost": float(total_lon_cost),
        "mean_displacement": float(displacement.mean()),
        "final_displacement": float(displacement[-1]),
    }


def _fit_and_rollout_real_case(map_dir, case):
    source_map = TRAINING_MAP_DIR / case["source_map"]
    if not source_map.exists():
        pytest.skip(f"Training map fixture not found: {source_map}")

    shutil.copy2(source_map, map_dir / "map_000.bin")

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="default",
        extend_classic_action_space=True,
    )
    try:
        env.reset(seed=0)
        trajectories = env.get_ground_truth_trajectories()
        gt_x_all = trajectories["x"][0, 0]
        gt_y_all = trajectories["y"][0, 0]
        gt_heading_all = np.unwrap(trajectories["heading"][0, 0])
        gt_valid = trajectories["valid"][0, 0].astype(bool)
    finally:
        env.close()

    gt_x = gt_x_all[gt_valid]
    gt_y = gt_y_all[gt_valid]
    gt_heading = gt_heading_all[gt_valid]
    planning_horizon = min(int(case.get("planning_horizon", TEST_REAL_CASE_PLANNING_HORIZON)), len(gt_x) - 1)
    if planning_horizon <= 0:
        pytest.skip(f"Not enough valid GT points in {source_map}")

    env_handle = _make_binding_env(map_dir)
    try:
        max_steps = 90
        actions = np.full(max_steps, -1, dtype=np.int32)
        step_costs = np.zeros(max_steps, dtype=np.float32)
        step_lat_costs = np.zeros(max_steps, dtype=np.float32)
        step_lon_costs = np.zeros(max_steps, dtype=np.float32)
        num_steps, total_cost, total_lat_cost, total_lon_cost = binding.env_fit_discrete_action_sequence(
            env_handle,
            0,
            12,
            planning_horizon,
            TEST_MATCH_WEIGHT_LATERAL,
            TEST_MATCH_WEIGHT_LONGITUDINAL,
            TEST_MATCH_WEIGHT_HEADING,
            TEST_MATCH_WEIGHT_SPEED,
            TEST_MATCH_WEIGHT_STEER_CHANGE,
            TEST_MATCH_WEIGHT_ACCEL_CHANGE,
            TEST_MATCH_WEIGHT_REVERSE,
            TEST_MATCH_WEIGHT_PROGRESS,
            TEST_MATCH_WEIGHT_STEER_FLIP,
            TEST_MATCH_WEIGHT_REF_ACCEL,
            TEST_MATCH_WEIGHT_REF_STEER,
            actions,
            step_costs,
            step_lat_costs,
            step_lon_costs,
        )
    finally:
        binding.env_close(env_handle)

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        goal_radius=TEST_GOAL_RADIUS,
        init_steps=0,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="default",
        extend_classic_action_space=True,
    )
    try:
        env.reset(seed=0)
        rollout_x = []
        rollout_y = []
        state = env.get_global_agent_state()
        rollout_x.append(float(state["x"][0]))
        rollout_y.append(float(state["y"][0]))

        action_buffer = np.zeros(1, dtype=np.int32)
        for step_idx in range(num_steps):
            action_buffer[0] = int(actions[step_idx])
            env.step(action_buffer)
            state = env.get_global_agent_state()
            rollout_x.append(float(state["x"][0]))
            rollout_y.append(float(state["y"][0]))
    finally:
        env.close()

    rollout_x = np.asarray(rollout_x, dtype=np.float32)
    rollout_y = np.asarray(rollout_y, dtype=np.float32)
    gt_x = gt_x[: num_steps + 1]
    gt_y = gt_y[: num_steps + 1]
    displacement = np.sqrt((rollout_x - gt_x) ** 2 + (rollout_y - gt_y) ** 2)
    steering_values = CLASSIC_STEERING_VALUES[actions[:num_steps] % len(CLASSIC_STEERING_VALUES)]
    acceleration_values = CLASSIC_ACCELERATION_VALUES[actions[:num_steps] // len(CLASSIC_STEERING_VALUES)]
    nonzero_steering = steering_values[np.abs(steering_values) > 1e-6]
    steering_reversals = 0
    if nonzero_steering.size > 1:
        steering_reversals = int(np.count_nonzero(np.diff(np.sign(nonzero_steering)) != 0))

    return {
        "name": case["name"],
        "source_map": case["source_map"],
        "gt_x": gt_x,
        "gt_y": gt_y,
        "rollout_x": rollout_x,
        "rollout_y": rollout_y,
        "actions": actions[:num_steps].copy(),
        "acceleration_values": acceleration_values.copy(),
        "step_costs": step_costs[:num_steps].copy(),
        "step_lat_costs": step_lat_costs[:num_steps].copy(),
        "step_lon_costs": step_lon_costs[:num_steps].copy(),
        "num_steps": num_steps,
        "steering_values": steering_values.copy(),
        "steering_reversals": steering_reversals,
        "heading_delta": float(abs(gt_heading[num_steps] - gt_heading[0])),
        "negative_accel_fraction": float(np.mean(acceleration_values < 0.0)),
        "positive_accel_fraction": float(np.mean(acceleration_values > 0.0)),
        "min_acceleration": float(np.min(acceleration_values)),
        "max_acceleration": float(np.max(acceleration_values)),
        "total_cost": float(total_cost),
        "total_lat_cost": float(total_lat_cost),
        "total_lon_cost": float(total_lon_cost),
        "mean_displacement": float(displacement.mean()),
        "final_displacement": float(displacement[-1]),
        "path_extent": float(
            max(
                np.max(gt_x) - np.min(gt_x),
                np.max(gt_y) - np.min(gt_y),
                np.max(rollout_x) - np.min(rollout_x),
                np.max(rollout_y) - np.min(rollout_y),
            )
        ),
    }


def test_bc_builder_rejects_unsupported_action_type(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)

    with pytest.raises(ValueError):
        build_bc_dataset(_builder_args(map_dir, action_type="continuous"))


def test_discrete_sequence_fit_beam_not_worse_than_greedy(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)

    env_handle = _make_binding_env(map_dir)
    try:
        max_steps = 90
        greedy_actions = np.full(max_steps, -1, dtype=np.int32)
        greedy_costs = np.zeros(max_steps, dtype=np.float32)
        greedy_lat_costs = np.zeros(max_steps, dtype=np.float32)
        greedy_lon_costs = np.zeros(max_steps, dtype=np.float32)
        beam_actions = np.full(max_steps, -1, dtype=np.int32)
        beam_costs = np.zeros(max_steps, dtype=np.float32)
        beam_lat_costs = np.zeros(max_steps, dtype=np.float32)
        beam_lon_costs = np.zeros(max_steps, dtype=np.float32)

        greedy_steps, greedy_total, _, _ = binding.env_fit_discrete_action_sequence(
            env_handle,
            0,
            1,
            5,
            TEST_MATCH_WEIGHT_LATERAL,
            TEST_MATCH_WEIGHT_LONGITUDINAL,
            TEST_MATCH_WEIGHT_HEADING,
            TEST_MATCH_WEIGHT_SPEED,
            TEST_MATCH_WEIGHT_STEER_CHANGE,
            TEST_MATCH_WEIGHT_ACCEL_CHANGE,
            TEST_MATCH_WEIGHT_REVERSE,
            TEST_MATCH_WEIGHT_PROGRESS,
            TEST_MATCH_WEIGHT_STEER_FLIP,
            TEST_MATCH_WEIGHT_REF_ACCEL,
            TEST_MATCH_WEIGHT_REF_STEER,
            greedy_actions,
            greedy_costs,
            greedy_lat_costs,
            greedy_lon_costs,
        )
        beam_steps, beam_total, _, _ = binding.env_fit_discrete_action_sequence(
            env_handle,
            0,
            4,
            5,
            TEST_MATCH_WEIGHT_LATERAL,
            TEST_MATCH_WEIGHT_LONGITUDINAL,
            TEST_MATCH_WEIGHT_HEADING,
            TEST_MATCH_WEIGHT_SPEED,
            TEST_MATCH_WEIGHT_STEER_CHANGE,
            TEST_MATCH_WEIGHT_ACCEL_CHANGE,
            TEST_MATCH_WEIGHT_REVERSE,
            TEST_MATCH_WEIGHT_PROGRESS,
            TEST_MATCH_WEIGHT_STEER_FLIP,
            TEST_MATCH_WEIGHT_REF_ACCEL,
            TEST_MATCH_WEIGHT_REF_STEER,
            beam_actions,
            beam_costs,
            beam_lat_costs,
            beam_lon_costs,
        )

        assert greedy_steps > 0
        assert beam_steps == greedy_steps
        assert beam_total <= greedy_total + 1e-6
        assert np.all((beam_actions[:beam_steps] >= 0) & (beam_actions[:beam_steps] < CLASSIC_ACTION_SPACE))
    finally:
        binding.env_close(env_handle)


def test_action_generation_rollout_matches_gt_and_saves_plot(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_results = []
    for case in REAL_TRAJECTORY_CASES:
        case_dir = tmp_path / case["name"]
        case_dir.mkdir()
        result = _fit_and_rollout_real_case(case_dir, case)

        assert result["num_steps"] == case["planning_horizon"]
        assert result["total_cost"] >= 0.0
        assert result["total_lat_cost"] >= 0.0
        assert result["total_lon_cost"] >= 0.0
        assert result["total_cost"] < case["max_total_cost"]
        assert result["mean_displacement"] < case["max_mean_disp"]
        assert result["final_displacement"] < case["max_final_disp"]
        assert result["path_extent"] > 1.0
        assert np.all((result["actions"] >= 0) & (result["actions"] < CLASSIC_ACTION_SPACE))
        if result["name"] == "real_straight":
            assert np.max(np.abs(result["steering_values"])) <= 0.167
        elif "turn" in result["name"]:
            assert result["steering_reversals"] == 0
        if result["name"] == "real_stronger_turn":
            assert result["heading_delta"] > 0.7
            assert np.mean(np.abs(result["steering_values"])) > 0.25
        if result["name"] == "real_deceleration":
            assert result["negative_accel_fraction"] > 0.8
            assert result["positive_accel_fraction"] == 0.0
            assert result["max_acceleration"] <= -1.0
        if result["name"] == "real_acceleration":
            assert result["positive_accel_fraction"] > 0.7
            assert result["negative_accel_fraction"] == 0.0
            assert result["min_acceleration"] >= 0.0

        case_results.append(result)

    plot_dir = Path("outputs/test_visualizations")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "action_generation_rollout.png"
    fig, axes = plt.subplots(1, len(case_results), figsize=(6 * len(case_results), 4), squeeze=False)
    relative_results = []
    for result in case_results:
        origin_x = float(result["gt_x"][0])
        origin_y = float(result["gt_y"][0])
        relative_results.append(
            {
                **result,
                "gt_plot_x": result["gt_x"] - origin_x,
                "gt_plot_y": result["gt_y"] - origin_y,
                "rollout_plot_x": result["rollout_x"] - origin_x,
                "rollout_plot_y": result["rollout_y"] - origin_y,
            }
        )

    max_span = 1.0
    for result in relative_results:
        plot_x = np.concatenate([result["gt_plot_x"], result["rollout_plot_x"]])
        plot_y = np.concatenate([result["gt_plot_y"], result["rollout_plot_y"]])
        max_span = max(max_span, float(plot_x.max() - plot_x.min()), float(plot_y.max() - plot_y.min()))
    axis_span = max_span * 1.15

    for ax, result in zip(axes[0], relative_results):
        plot_x = np.concatenate([result["gt_plot_x"], result["rollout_plot_x"]])
        plot_y = np.concatenate([result["gt_plot_y"], result["rollout_plot_y"]])
        center_x = 0.5 * float(plot_x.min() + plot_x.max())
        center_y = 0.5 * float(plot_y.min() + plot_y.max())
        half_span = 0.5 * axis_span

        ax.plot(
            result["gt_plot_x"],
            result["gt_plot_y"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="GT trajectory",
        )
        ax.plot(
            result["rollout_plot_x"],
            result["rollout_plot_y"],
            marker="x",
            markersize=3,
            linewidth=1.5,
            label="Rollout from target actions",
        )
        ax.set_title(
            f"{result['name']} ({result['source_map']})\ntotal_cost={result['total_cost']:.3f}, "
            f"lat={result['total_lat_cost']:.3f}, lon={result['total_lon_cost']:.3f}\n"
            f"mean_err={result['mean_displacement']:.3f}, dh={result['heading_delta']:.3f}\n"
            f"acc=[{result['min_acceleration']:.0f},{result['max_acceleration']:.0f}], rev={result['steering_reversals']}"
        )
        ax.set_xlabel("x rel. to start [m]")
        ax.set_ylabel("y rel. to start [m]")
        ax.set_xlim(center_x - half_span, center_x + half_span)
        ax.set_ylim(center_y - half_span, center_y + half_span)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    assert plot_path.exists()


def test_bc_dataset_builder_writes_model_ready_shard(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    args = _builder_args(map_dir)

    shard_paths = build_bc_dataset(args)
    assert len(shard_paths) == 1

    shard = torch.load(shard_paths[0])
    assert set(
        [
            "obs",
            "action",
            "scenario_id",
            "agent_id",
            "sequence_id",
            "sequence_row_index",
            "sequence_length",
            "timestep",
            "match_cost_total",
            "match_cost_step",
            "match_cost_lateral_total",
            "match_cost_longitudinal_total",
            "match_cost_lateral_step",
            "match_cost_longitudinal_step",
            "map_id",
        ]
    ).issubset(shard.keys())
    assert shard["obs"].dtype == torch.float32
    assert shard["action"].dtype == torch.int64
    assert shard["sequence_id"].dtype == torch.int64
    assert shard["sequence_row_index"].dtype == torch.int32
    assert shard["sequence_length"].dtype == torch.int32
    assert shard["obs"].shape[0] == shard["action"].shape[0]
    assert shard["obs"].shape[0] > 0
    assert shard["sequence_id"].shape[0] == shard["obs"].shape[0]
    assert shard["sequence_row_index"].shape[0] == shard["obs"].shape[0]
    assert shard["sequence_length"].shape[0] == shard["obs"].shape[0]
    assert torch.unique(shard["sequence_id"]).numel() >= 1
    assert "sequence_ids" in shard["metadata"]
    assert "sequence_lengths" in shard["metadata"]
    assert len(shard["metadata"]["sequence_ids"]) == len(shard["metadata"]["sequence_lengths"])

    for sequence_id, sequence_length in zip(shard["metadata"]["sequence_ids"], shard["metadata"]["sequence_lengths"]):
        sequence_mask = shard["sequence_id"] == int(sequence_id)
        row_indices = shard["sequence_row_index"][sequence_mask]
        row_lengths = shard["sequence_length"][sequence_mask]
        assert int(sequence_mask.sum()) == int(sequence_length)
        assert torch.equal(row_indices, torch.arange(int(sequence_length), dtype=torch.int32))
        assert torch.equal(row_lengths, torch.full_like(row_lengths, int(sequence_length)))

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="default",
        extend_classic_action_space=True,
    )
    try:
        policy = DrivePolicy(env, input_size=64, hidden_size=64)
        obs_batch = shard["obs"][: min(4, shard["obs"].shape[0])]
        logits, value = policy(obs_batch)
        assert len(logits) == 1
        assert logits[0].shape[0] == obs_batch.shape[0]
        assert value.shape[0] == obs_batch.shape[0]
    finally:
        env.close()


def test_bc_dataset_builder_augmented_observation_mode_writes_augmented_width(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)

    args = _builder_args(map_dir)
    args["env"]["observation_mode"] = "sdc_only_with_trailer"
    shard_paths = build_bc_dataset(args)

    shard = torch.load(shard_paths[0])
    expected_obs_dim = (
        binding.EGO_FEATURES_CLASSIC
        + 1
        + 4
        + (binding.MAX_AGENTS - 1) * (binding.PARTNER_FEATURES + 1)
        + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    )
    assert shard["obs"].shape[1] == expected_obs_dim
    assert shard["metadata"]["observation_mode"] == "sdc_only_with_trailer"


def test_bc_trainer_recurrent_smoke(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))

    args = _bc_train_args(map_dir, Path(shard_paths[0]).parent, rnn_name="Recurrent")
    result = train_bc_policy(args)

    assert Path(result["latest_path"]).exists()
    assert Path(result["best_path"]).exists()
    assert Path(result["metadata_path"]).exists()
    assert len(result["history"]) == args["bc_train"]["epochs"]
    assert result["history"][0]["train_samples"] > 0


def test_bc_trainer_recurrent_uses_embedded_windows(tmp_path, monkeypatch):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    seq_len = 4
    stride = 4
    shard_paths = build_bc_dataset(
        _builder_args(map_dir, export_windows=True, window_seq_len=seq_len, window_stride=stride)
    )
    shard_path = Path(shard_paths[0])
    manifest_path = drive_module._sequence_manifest_path(shard_path, seq_len, stride)
    assert not manifest_path.exists()

    args = _bc_train_args(map_dir, shard_path.parent, rnn_name="Recurrent")
    args["bc_train"]["use_embedded_windows"] = True
    args["bc_train"]["seq_len"] = seq_len
    args["bc_train"]["sequence_stride"] = stride
    args["train"]["bptt_horizon"] = seq_len

    def _should_not_rebuild(*args, **kwargs):
        raise AssertionError("embedded window metadata should be used by the recurrent BC trainer")

    monkeypatch.setattr(drive_module, "_build_sequence_manifest_from_payload", _should_not_rebuild)
    result = train_bc_policy(args)

    assert Path(result["latest_path"]).exists()
    assert result["history"][0]["train_samples"] > 0
    assert not manifest_path.exists()


def test_bc_trainer_recurrent_rejects_missing_embedded_windows(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))
    shard_path = Path(shard_paths[0])

    args = _bc_train_args(map_dir, shard_path.parent, rnn_name="Recurrent")
    args["bc_train"]["use_embedded_windows"] = True
    with pytest.raises(ValueError, match="does not contain embedded window metadata"):
        train_bc_policy(args)


def test_bc_trainer_rejects_missing_sequence_id(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))
    shard_path = Path(shard_paths[0])

    shard = torch.load(shard_path)
    if "sequence_id" in shard:
        del shard["sequence_id"]
    torch.save(shard, shard_path)

    args = _bc_train_args(map_dir, shard_path.parent, rnn_name="Recurrent")
    with pytest.raises(ValueError, match="missing required keys"):
        train_bc_policy(args)


def test_bc_trainer_non_recurrent_smoke(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))

    args = _bc_train_args(map_dir, Path(shard_paths[0]).parent, rnn_name=None)
    result = train_bc_policy(args)

    assert Path(result["latest_path"]).exists()
    assert result["history"][0]["train_samples"] > 0


def test_bc_trainer_rejects_mismatched_obs_width(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))
    shard_path = Path(shard_paths[0])

    shard = torch.load(shard_path)
    shard["obs"] = torch.cat([shard["obs"], torch.zeros((shard["obs"].shape[0], 1), dtype=torch.float32)], dim=1)
    torch.save(shard, shard_path)

    args = _bc_train_args(map_dir, shard_path.parent, rnn_name="Recurrent")
    with pytest.raises(ValueError, match="observation width mismatch"):
        train_bc_policy(args)


def test_bc_trainer_rejects_out_of_range_actions(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))
    shard_path = Path(shard_paths[0])

    shard = torch.load(shard_path)
    shard["action"][0] = CLASSIC_ACTION_SPACE
    torch.save(shard, shard_path)

    args = _bc_train_args(map_dir, shard_path.parent, rnn_name=None)
    with pytest.raises(ValueError, match="invalid action ids"):
        train_bc_policy(args)


def test_bc_trainer_rejects_unexpected_timestep_spacing(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))
    shard_path = Path(shard_paths[0])

    shard = torch.load(shard_path)
    shard["timestep"][1] = shard["timestep"][0] + 2
    torch.save(shard, shard_path)

    args = _bc_train_args(map_dir, shard_path.parent, rnn_name="Recurrent")
    with pytest.raises(ValueError, match="unexpected timestep spacing"):
        train_bc_policy(args)


def test_bc_trainer_recurrent_rejects_seq_len_mismatch_with_model(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))

    args = _bc_train_args(map_dir, Path(shard_paths[0]).parent, rnn_name="Recurrent")
    args["bc_train"]["seq_len"] = 5
    args["bc_train"]["sequence_stride"] = 5
    args["train"]["bptt_horizon"] = 4
    with pytest.raises(ValueError, match="window size mismatch"):
        train_bc_policy(args)


def test_bc_trainer_checkpoint_loads_with_existing_policy_path(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))
    args = _bc_train_args(map_dir, Path(shard_paths[0]).parent, rnn_name="Recurrent")
    result = train_bc_policy(args)

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="default",
        extend_classic_action_space=True,
    )
    try:
        load_args = {
            "package": "ocean",
            "train": {"device": "cpu"},
            "policy_name": "Drive",
            "rnn_name": "Recurrent",
            "policy": args["policy"],
            "rnn": args["rnn"],
            "load_id": None,
            "load_model_path": result["best_path"],
        }
        policy = load_policy(load_args, SimpleNamespace(driver_env=env), env_name="puffer_drive")
        shard = torch.load(shard_paths[0])
        obs_batch = shard["obs"][:2]
        state = {"lstm_h": None, "lstm_c": None, "hidden": None}
        logits, values = policy(obs_batch, state)
        assert len(logits) == 1
        assert logits[0].shape[0] == obs_batch.shape[0]
        assert values.shape[0] == obs_batch.shape[0]
    finally:
        env.close()


def test_bc_trainer_can_overfit_tiny_shard(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))

    args = _bc_train_args(map_dir, Path(shard_paths[0]).parent, rnn_name=None)
    args["bc_train"].update(
        {
            "epochs": 12,
            "batch_size": 64,
            "learning_rate": 0.02,
            "val_fraction": 0.0,
        }
    )
    result = train_bc_policy(args)

    assert result["history"][0]["train_loss"] > result["history"][-1]["train_loss"]
    assert result["history"][-1]["train_accuracy"] >= result["history"][0]["train_accuracy"]


def test_bc_trainer_splits_train_val_by_shard(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir, map_filename="map_000.bin", unique_map_id=123)
    _write_bc_test_map(
        map_dir,
        ego_x=[0.0, 0.4, 0.9, 1.5, 2.2, 3.0],
        ego_y=[0.0, -0.02, -0.05, -0.08, -0.1, -0.12],
        map_filename="map_001.bin",
        unique_map_id=124,
    )
    builder_args = _builder_args(map_dir)
    builder_args["bc"]["max_maps"] = 2
    shard_paths = build_bc_dataset(builder_args)
    assert len(shard_paths) == 2

    args = _bc_train_args(map_dir, Path(shard_paths[0]).parent, rnn_name=None)
    args["bc_train"]["val_fraction"] = 0.5
    result = train_bc_policy(args)

    assert len(result["train_shards"]) == 1
    assert len(result["val_shards"]) == 1
    assert set(result["train_shards"]).isdisjoint(set(result["val_shards"]))


def test_bc_trainer_logs_metrics_to_logger(tmp_path):
    class FakeLogger:
        def __init__(self):
            self.logged = []
            self.closed_with = None

        def log(self, logs, step):
            self.logged.append((dict(logs), step))

        def close(self, model_path):
            self.closed_with = model_path

    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))

    args = _bc_train_args(map_dir, Path(shard_paths[0]).parent, rnn_name=None)
    logger = FakeLogger()
    result = train_bc_policy(args, logger=logger)

    assert logger.logged
    last_logs, last_step = logger.logged[-1]
    assert last_step > 0
    assert "bc/train_loss" in last_logs
    assert "bc/train_accuracy" in last_logs
    assert "bc/train_elapsed_sec" in last_logs
    assert "bc/val_loss" in last_logs
    assert "bc/val_accuracy" in last_logs
    assert "bc/val_elapsed_sec" in last_logs
    assert "bc/best_metric" in last_logs
    assert logger.closed_with == result["best_path"]


def test_bc_trainer_recurrent_accepts_sequence_id_without_legacy_ids(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))
    shard_path = Path(shard_paths[0])

    shard = torch.load(shard_path)
    shard["sequence_id"] = torch.zeros_like(shard["action"], dtype=torch.int64)
    del shard["scenario_id"]
    del shard["agent_id"]
    torch.save(shard, shard_path)

    args = _bc_train_args(map_dir, shard_path.parent, rnn_name="Recurrent")
    result = train_bc_policy(args)

    assert Path(result["latest_path"]).exists()
    assert result["history"][0]["train_samples"] > 0


def test_sequence_manifest_is_cached_per_shard(tmp_path, monkeypatch):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))
    shard_path = Path(shard_paths[0])

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="default",
        extend_classic_action_space=True,
    )
    try:
        obs_dim = int(env.single_observation_space.shape[0])
        action_space_size = int(env.single_action_space.nvec[0])
    finally:
        env.close()

    manifest = drive_module._load_or_build_sequence_manifest(
        shard_path,
        obs_dim,
        action_space_size,
        seq_len=32,
        stride=32,
    )
    manifest_path = drive_module._sequence_manifest_path(shard_path, 32, 32)
    assert manifest_path.exists()
    assert manifest["window_count"] > 0

    def _should_not_rebuild(*args, **kwargs):
        raise AssertionError("sequence manifest should have been reused from cache")

    monkeypatch.setattr(drive_module, "_build_sequence_manifest_from_payload", _should_not_rebuild)
    cached_manifest = drive_module._load_or_build_sequence_manifest(
        shard_path,
        obs_dim,
        action_space_size,
        seq_len=32,
        stride=32,
    )
    assert cached_manifest["window_count"] == manifest["window_count"]


def test_sequence_manifest_windows_match_payload_sequences(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))
    shard_path = Path(shard_paths[0])
    payload = torch.load(shard_path)

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="default",
        extend_classic_action_space=True,
    )
    try:
        obs_dim = int(env.single_observation_space.shape[0])
        action_space_size = int(env.single_action_space.nvec[0])
    finally:
        env.close()

    seq_len = 4
    stride = 2
    manifest = drive_module._load_or_build_sequence_manifest(
        shard_path,
        obs_dim,
        action_space_size,
        seq_len=seq_len,
        stride=stride,
    )
    manifest_samples = drive_module._build_sequence_samples_from_manifest(payload, manifest)
    direct_samples = drive_module._build_sequence_samples_from_payload(payload, seq_len, stride)

    assert manifest["window_count"] == len(direct_samples)
    assert len(manifest_samples) == len(direct_samples)
    for manifest_sample, direct_sample in zip(manifest_samples, direct_samples):
        manifest_obs, manifest_action, manifest_mask = manifest_sample
        direct_obs, direct_action, direct_mask = direct_sample
        assert torch.equal(manifest_mask, direct_mask)
        assert torch.equal(manifest_action, direct_action)
        assert torch.allclose(manifest_obs, direct_obs)


def test_sequence_manifest_processes_file_data_correctly(tmp_path):
    shard_path = tmp_path / "map_000.pt"
    payload = {
        "obs": torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
                [7.0, 70.0],
            ],
            dtype=torch.float32,
        ),
        "action": torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64),
        "map_id": torch.zeros(7, dtype=torch.int32),
        "timestep": torch.tensor([0, 1, 2, 3, 10, 11, 12], dtype=torch.int32),
        "sequence_id": torch.tensor([11, 11, 11, 11, 22, 22, 22], dtype=torch.int64),
        "metadata": {"action_space_size": 10},
    }
    torch.save(payload, shard_path)

    manifest = drive_module._load_or_build_sequence_manifest(
        shard_path,
        obs_dim=2,
        action_space_size=10,
        seq_len=3,
        stride=2,
    )
    samples = drive_module._build_sequence_samples_from_manifest(payload, manifest)

    assert manifest["window_count"] == 4
    assert len(samples) == 4

    obs0, action0, mask0 = samples[0]
    assert torch.allclose(obs0, torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
    assert torch.equal(action0, torch.tensor([0, 1, 2]))
    assert torch.equal(mask0, torch.tensor([True, True, True]))

    obs1, action1, mask1 = samples[1]
    assert torch.allclose(obs1, torch.tensor([[3.0, 30.0], [4.0, 40.0], [0.0, 0.0]]))
    assert torch.equal(action1, torch.tensor([2, 3, 0]))
    assert torch.equal(mask1, torch.tensor([True, True, False]))

    obs2, action2, mask2 = samples[2]
    assert torch.allclose(obs2, torch.tensor([[5.0, 50.0], [6.0, 60.0], [7.0, 70.0]]))
    assert torch.equal(action2, torch.tensor([4, 5, 6]))
    assert torch.equal(mask2, torch.tensor([True, True, True]))

    obs3, action3, mask3 = samples[3]
    assert torch.allclose(obs3, torch.tensor([[7.0, 70.0], [0.0, 0.0], [0.0, 0.0]]))
    assert torch.equal(action3, torch.tensor([6, 0, 0]))
    assert torch.equal(mask3, torch.tensor([True, False, False]))


def test_sequence_spans_use_exported_sequence_metadata():
    payload = {
        "obs": torch.zeros((7, 2), dtype=torch.float32),
        "action": torch.zeros((7,), dtype=torch.int64),
        "map_id": torch.zeros((7,), dtype=torch.int32),
        "timestep": torch.tensor([0, 1, 2, 10, 11, 20, 21], dtype=torch.int32),
        "sequence_id": torch.tensor([101, 101, 101, 202, 202, 303, 303], dtype=torch.int64),
        "sequence_row_index": torch.tensor([0, 1, 2, 0, 1, 0, 1], dtype=torch.int32),
        "sequence_length": torch.tensor([3, 3, 3, 2, 2, 2, 2], dtype=torch.int32),
        "metadata": {
            "sequence_ids": [101, 202, 303],
            "sequence_lengths": [3, 2, 2],
            "action_space_size": 10,
        },
    }

    spans = drive_module._sequence_spans_from_payload(payload)
    assert spans == [(101, 0, 3), (202, 3, 5), (303, 5, 7)]


def test_bc_trainer_logs_batch_metrics_to_logger(tmp_path):
    class FakeLogger:
        def __init__(self):
            self.logged = []
            self.closed_with = None

        def log(self, logs, step):
            self.logged.append((dict(logs), step))

        def close(self, model_path):
            self.closed_with = model_path

    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))

    args = _bc_train_args(map_dir, Path(shard_paths[0]).parent, rnn_name=None)
    args["bc_train"]["log_interval"] = 1
    logger = FakeLogger()
    train_bc_policy(args, logger=logger)

    batch_logs = [logs for logs, _ in logger.logged if "bc/train_loss_running" in logs]
    assert batch_logs
    last_batch_logs = batch_logs[-1]
    assert "bc/train_accuracy_running" in last_batch_logs
    assert "bc/train_batches_done" in last_batch_logs
    assert "bc/train_samples_seen_epoch" in last_batch_logs
    assert "bc/train_elapsed_sec_running" in last_batch_logs


def test_bc_trainer_emits_console_progress_messages(tmp_path, capsys):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)
    shard_paths = build_bc_dataset(_builder_args(map_dir))

    args = _bc_train_args(map_dir, Path(shard_paths[0]).parent, rnn_name=None)
    args["bc_train"]["epochs"] = 1
    train_bc_policy(args)

    captured = capsys.readouterr()
    assert "[BC] starting training" in captured.out
    assert "[BC] using shard-streamed loading" in captured.out
    assert "[BC] dataset ready" in captured.out
    assert "[BC] epoch 1/1 starting" in captured.out
    assert "[BC] epoch=1" in captured.out


def test_binding_env_init_honors_python_config_overrides(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)

    obs_dim = (
        binding.EGO_FEATURES_JERK
        + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
        + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    )
    observations = np.zeros((1, obs_dim), dtype=np.float32)
    actions = np.zeros(1, dtype=np.int32)
    rewards = np.zeros(1, dtype=np.float32)
    terminals = np.zeros(1, dtype=np.uint8)
    truncations = np.zeros(1, dtype=np.uint8)

    env_handle = binding.env_init(
        observations,
        actions,
        rewards,
        terminals,
        truncations,
        0,
        human_agent_idx=0,
        action_type=1,
        dynamics_model=1,
        observation_mode=1,
        extend_classic_action_space=0,
        reward_vehicle_collision=-0.25,
        reward_offroad_collision=-0.75,
        reward_goal=2.0,
        reward_goal_post_respawn=0.4,
        goal_radius=3.5,
        goal_speed=12.5,
        goal_behavior=2,
        goal_target_distance=44.0,
        collision_behavior=2,
        offroad_behavior=1,
        dt=0.2,
        episode_length=33,
        termination_mode=0,
        max_controlled_agents=1,
        map_id=0,
        max_agents=1,
        ini_file="pufferlib/config/ocean/drive.ini",
        init_steps=5,
        init_mode=1,
        control_mode=3,
        map_dir=str(map_dir),
        non_kinematic_vehicle_params_override=None,
        force_zero_trailer_articulation_at_init=0,
    )
    try:
        config = binding.env_get_config(env_handle)
        assert config["action_type"] == 1
        assert config["dynamics_model"] == 1
        assert config["observation_mode"] == 1
        assert config["extend_classic_action_space"] == 0
        assert config["reward_vehicle_collision"] == pytest.approx(-0.25)
        assert config["reward_offroad_collision"] == pytest.approx(-0.75)
        assert config["reward_goal"] == pytest.approx(2.0)
        assert config["reward_goal_post_respawn"] == pytest.approx(0.4)
        assert config["goal_radius"] == pytest.approx(3.5)
        assert config["goal_speed"] == pytest.approx(12.5)
        assert config["goal_behavior"] == 2
        assert config["goal_target_distance"] == pytest.approx(44.0)
        assert config["collision_behavior"] == 2
        assert config["offroad_behavior"] == 1
        assert config["dt"] == pytest.approx(0.2)
        assert config["episode_length"] == 33
        assert config["termination_mode"] == 0
        assert config["init_steps"] == 5
        assert config["init_mode"] == 1
        assert config["control_mode"] == 3
    finally:
        binding.env_close(env_handle)


def test_bc_dataset_export_preserves_configured_dt(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)

    builder_args = _builder_args(map_dir)
    builder_args["env"]["dt"] = 0.2
    builder_args["env"]["init_steps"] = 3

    shard_paths = build_bc_dataset(builder_args)
    payload = torch.load(shard_paths[0])

    assert payload["metadata"]["config"]["env"]["dt"] == pytest.approx(0.2)

    sequence_id = payload["sequence_id"]
    timestep = payload["timestep"]
    first_sequence_mask = sequence_id == sequence_id[0]
    first_sequence_timesteps = timestep[first_sequence_mask]

    assert first_sequence_timesteps.numel() >= 2
    assert int(first_sequence_timesteps[0]) == 3
    assert torch.equal(
        first_sequence_timesteps[1:] - first_sequence_timesteps[:-1],
        torch.ones(first_sequence_timesteps.numel() - 1, dtype=first_sequence_timesteps.dtype),
    )

    exported_time_seconds = first_sequence_timesteps.to(torch.float32) * float(payload["metadata"]["config"]["env"]["dt"])
    assert torch.allclose(
        exported_time_seconds[1:] - exported_time_seconds[:-1],
        torch.full((first_sequence_timesteps.numel() - 1,), 0.2, dtype=torch.float32),
    )


def test_bc_dataset_export_writes_sequence_metadata(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)

    shard_paths = build_bc_dataset(_builder_args(map_dir))
    payload = torch.load(shard_paths[0])

    assert "sequence_row_index" in payload
    assert "sequence_length" in payload
    assert payload["sequence_row_index"].dtype == torch.int32
    assert payload["sequence_length"].dtype == torch.int32
    assert payload["sequence_row_index"].shape == payload["sequence_id"].shape
    assert payload["sequence_length"].shape == payload["sequence_id"].shape

    metadata = payload["metadata"]
    assert "sequence_ids" in metadata
    assert "sequence_lengths" in metadata
    assert len(metadata["sequence_ids"]) == len(metadata["sequence_lengths"])
    assert metadata["sequence_count"] == len(metadata["sequence_ids"])

    for sequence_id, sequence_length in zip(metadata["sequence_ids"], metadata["sequence_lengths"]):
        sequence_mask = payload["sequence_id"] == int(sequence_id)
        assert int(sequence_mask.sum()) == int(sequence_length)

        sequence_row_index = payload["sequence_row_index"][sequence_mask]
        sequence_lengths = payload["sequence_length"][sequence_mask]
        sequence_timesteps = payload["timestep"][sequence_mask]

        assert torch.equal(sequence_row_index, torch.arange(int(sequence_length), dtype=torch.int32))
        assert torch.equal(sequence_lengths, torch.full_like(sequence_lengths, int(sequence_length)))
        assert torch.equal(
            sequence_timesteps,
            sequence_timesteps[0] + torch.arange(int(sequence_length), dtype=sequence_timesteps.dtype),
        )


def test_bc_dataset_export_can_embed_training_windows(tmp_path, monkeypatch):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)

    seq_len = 4
    stride = 2
    shard_paths = build_bc_dataset(
        _builder_args(map_dir, export_windows=True, window_seq_len=seq_len, window_stride=stride)
    )
    shard_path = Path(shard_paths[0])
    payload = torch.load(shard_path)

    assert "window_metadata" in payload
    window_metadata = payload["window_metadata"]
    assert int(window_metadata["seq_len"]) == seq_len
    assert int(window_metadata["stride"]) == stride
    assert int(window_metadata["window_count"]) > 0
    assert payload["metadata"]["window_count"] == int(window_metadata["window_count"])
    assert payload["metadata"]["window_seq_len"] == seq_len
    assert payload["metadata"]["window_stride"] == stride

    direct_manifest = drive_module._build_sequence_manifest_from_payload(payload, seq_len, stride)
    assert int(window_metadata["window_count"]) == int(direct_manifest["window_count"])
    assert torch.equal(window_metadata["window_indices"], direct_manifest["window_indices"])
    assert torch.equal(window_metadata["valid_lengths"], direct_manifest["valid_lengths"])

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode="default",
        extend_classic_action_space=True,
    )
    try:
        obs_dim = int(env.single_observation_space.shape[0])
        action_space_size = int(env.single_action_space.nvec[0])
    finally:
        env.close()

    def _should_not_rebuild(*args, **kwargs):
        raise AssertionError("embedded window metadata should have been reused from shard payload")

    monkeypatch.setattr(drive_module, "_build_sequence_manifest_from_payload", _should_not_rebuild)
    manifest = drive_module._load_or_build_sequence_manifest(
        shard_path,
        obs_dim,
        action_space_size,
        seq_len=seq_len,
        stride=stride,
    )

    assert int(manifest["window_count"]) == int(window_metadata["window_count"])
    assert not drive_module._sequence_manifest_path(shard_path, seq_len, stride).exists()


def test_drive_accepts_c_compatible_enum_aliases(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_bc_test_map(map_dir)

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode=0,
        init_mode="created_all_valid",
        observation_mode=0,
        action_type=0,
        dynamics_model=0,
        resample_frequency=0,
    )
    try:
        assert env.control_mode == 0
        assert env.init_mode == 0
        assert env.observation_mode == 0
        assert env.dynamics_model == "classic"
        assert env._action_type_flag == 0
    finally:
        env.close()
