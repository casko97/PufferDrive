from pathlib import Path
from types import SimpleNamespace
import sys
import shutil

import numpy as np
import pytest
import torch

from pufferlib.ocean.drive.drive import (
    Drive,
    _CLASSIC_ACCELERATION_VALUES,
    _CLASSIC_STEERING_VALUES,
    build_bc_dataset,
)
from pufferlib.pufferl import load_config, load_policy
from tests.test_drive_bin_simulator_load import (
    _load_roads_from_base_bin,
    _load_sdc_trajectory_from_base_bin,
    _load_training_car_reference_map_and_trajectory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE_DT_SECONDS = 0.1
ROLLOUT_RNN_WARMUP_STEPS = 8
APRICOT_POLICY_PATH = (
    REPO_ROOT
    / "pufferlib"
    / "resources"
    / "drive"
    / "models"
    / "apricot-voice-35"
    / "puffer_drive_3vut5lku.pt"
)
PREVIOUS_BC_POLICY_PATH = REPO_ROOT / "experiments_bc" / "bc-streaming-full-default-20260317-150151" / "best.pt"
CURRENT_BC_POLICY_PATH = REPO_ROOT / "experiments_bc" / "bc-full-windows-stride10-plateaulr-20260318-104912" / "best.pt"
VALIDATION_BIN_DIR = REPO_ROOT / "resources" / "drive" / "binaries" / "validation"


def _greedy_multidiscrete_action(logits):
    if isinstance(logits, tuple):
        return torch.stack([branch.argmax(dim=1) for branch in logits], dim=1)
    return logits.argmax(dim=1, keepdim=True)


def _movement_start_step(tractor_gt, min_distance=1.0):
    valid = tractor_gt["valid"] > 0
    x = np.asarray(tractor_gt["x"][valid], dtype=np.float32)
    y = np.asarray(tractor_gt["y"][valid], dtype=np.float32)
    if x.shape[0] < 2:
        return 0

    segment_dist = np.hypot(np.diff(x), np.diff(y))
    cumulative_dist = np.concatenate([[0.0], np.cumsum(segment_dist)])
    moving = np.flatnonzero(cumulative_dist >= float(min_distance))
    if moving.size == 0:
        return 0
    return int(moving[0])


def _cap_trajectory_at_respawn(x, y, jump_threshold=8.0):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.shape[0] < 2:
        return x, y

    step_jump = np.hypot(np.diff(x), np.diff(y))
    respawn = np.flatnonzero(step_jump > float(jump_threshold))
    if respawn.size == 0:
        return x, y

    end_idx = int(respawn[0]) + 1
    return x[:end_idx], y[:end_idx]


def _estimate_initial_speed(tractor_gt, init_steps, dt=DRIVE_DT_SECONDS):
    valid = tractor_gt["valid"] > 0
    x = np.asarray(tractor_gt["x"][valid], dtype=np.float32)
    y = np.asarray(tractor_gt["y"][valid], dtype=np.float32)
    if x.shape[0] < 2:
        return 0.0

    idx = int(max(0, min(init_steps, x.shape[0] - 1)))
    next_idx = min(idx + 1, x.shape[0] - 1)
    prev_idx = max(idx - 1, 0)
    if next_idx == idx:
        dx = float(x[idx] - x[prev_idx])
        dy = float(y[idx] - y[prev_idx])
    else:
        dx = float(x[next_idx] - x[idx])
        dy = float(y[next_idx] - y[idx])
    return float(np.hypot(dx, dy) / max(float(dt), 1e-6))


def _wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _trajectory_speed_series(x, y, dt=DRIVE_DT_SECONDS):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.shape[0] < 2:
        return np.zeros((x.shape[0],), dtype=np.float32)
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.hypot(dx, dy) / max(float(dt), 1e-6)
    return np.concatenate([speed[:1], speed]).astype(np.float32)


def _trajectory_heading_series(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.shape[0] < 2:
        return np.zeros((x.shape[0],), dtype=np.float32)
    dx = np.diff(x)
    dy = np.diff(y)
    headings = np.arctan2(dy, dx + 1e-8)
    return np.concatenate([headings[:1], headings]).astype(np.float32)


def _summarize_gt_motion(tractor_gt, init_steps):
    valid = tractor_gt["valid"] > 0
    x = np.asarray(tractor_gt["x"][valid], dtype=np.float32)
    y = np.asarray(tractor_gt["y"][valid], dtype=np.float32)
    if x.shape[0] < 2:
        return {
            "v0": 0.0,
            "v_mean": 0.0,
            "v_max": 0.0,
            "accel_peak": 0.0,
            "heading_delta": 0.0,
            "heading_rate_peak": 0.0,
        }

    start = int(max(0, min(init_steps, x.shape[0] - 2)))
    x = x[start:]
    y = y[start:]
    speed = _trajectory_speed_series(x, y)
    heading = _trajectory_heading_series(x, y)
    accel = np.diff(speed) / DRIVE_DT_SECONDS if speed.shape[0] >= 2 else np.zeros((0,), dtype=np.float32)
    heading_diff = (
        np.asarray([_wrap_angle(v) for v in np.diff(heading)], dtype=np.float32)
        if heading.shape[0] >= 2
        else np.zeros((0,), dtype=np.float32)
    )
    heading_rate = heading_diff / DRIVE_DT_SECONDS if heading_diff.size > 0 else np.zeros((0,), dtype=np.float32)
    total_heading_delta = float(np.asarray([_wrap_angle(float(np.sum(heading_diff)))])[0]) if heading_diff.size > 0 else 0.0
    return {
        "v0": float(speed[0]) if speed.size > 0 else 0.0,
        "v_mean": float(np.mean(speed)) if speed.size > 0 else 0.0,
        "v_max": float(np.max(speed)) if speed.size > 0 else 0.0,
        "accel_peak": float(np.max(np.abs(accel))) if accel.size > 0 else 0.0,
        "heading_delta": total_heading_delta,
        "heading_rate_peak": float(np.max(np.abs(heading_rate))) if heading_rate.size > 0 else 0.0,
    }


def _rollout_policy(args, map_dir, num_steps, init_steps=0, *, warmup_actions=None):
    warmup_actions = np.asarray(warmup_actions if warmup_actions is not None else [], dtype=np.int64)
    warmup_steps = int(warmup_actions.shape[0])
    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=max(48, num_steps + warmup_steps + 24),
        init_steps=init_steps,
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode=args["env"]["observation_mode"],
        action_type=args["env"]["action_type"],
        dynamics_model=args["env"]["dynamics_model"],
        extend_classic_action_space=bool(args["env"]["extend_classic_action_space"]),
    )

    try:
        policy = load_policy(args, SimpleNamespace(driver_env=env), env_name="puffer_drive")
        policy.eval()

        obs, _ = env.reset(seed=0)
        action_trace = []
        device = torch.device(args["train"]["device"])

        state = {}
        if args["train"]["use_rnn"]:
            state = {
                "lstm_h": torch.zeros(env.num_agents, policy.hidden_size, device=device),
                "lstm_c": torch.zeros(env.num_agents, policy.hidden_size, device=device),
            }

        for warmup_idx in range(warmup_steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, device=device)
                logits, _ = policy.forward_eval(obs_tensor, state)
                action_np = np.asarray([int(warmup_actions[warmup_idx])], dtype=np.int32).reshape(env.action_space.shape)

            obs, rewards, dones, truncs, _ = env.step(action_np)
            assert np.isfinite(obs).all()
            assert np.isfinite(rewards).all()
            if np.all(dones) or np.all(truncs):
                raise AssertionError("Warm-up terminated before plotted rollout start")

        start_state = env.get_global_agent_state()
        rollout_x = [float(start_state["x"][0])]
        rollout_y = [float(start_state["y"][0])]
        steps_executed = 0
        for _ in range(num_steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, device=device)
                logits, _ = policy.forward_eval(obs_tensor, state)
                action = _greedy_multidiscrete_action(logits)
                action_np = action.cpu().numpy().reshape(env.action_space.shape)
                action_trace.append(int(np.asarray(action_np).reshape(-1)[0]))

            obs, rewards, dones, truncs, _ = env.step(action_np)
            steps_executed += 1
            current_state = env.get_global_agent_state()
            rollout_x.append(float(current_state["x"][0]))
            rollout_y.append(float(current_state["y"][0]))

            assert np.isfinite(obs).all()
            assert np.isfinite(rewards).all()

            if np.all(dones) or np.all(truncs):
                break

        end_state = env.get_global_agent_state()
        distance_travelled = float(
            np.hypot(end_state["x"][0] - start_state["x"][0], end_state["y"][0] - start_state["y"][0])
        )
        assert steps_executed > 0
        assert np.isfinite(end_state["heading"]).all()

        capped_x, capped_y = _cap_trajectory_at_respawn(rollout_x, rollout_y)
        return {
            "x": capped_x,
            "y": capped_y,
            "steps_executed": steps_executed,
            "distance_travelled": distance_travelled,
            "actions": np.asarray(action_trace, dtype=np.int64),
        }
    finally:
        env.close()


def _rollout_fitted_gt_actions(args, map_dir, num_steps, init_steps=0):
    env_cfg = args["env"]
    bc_args = load_config("puffer_drive")
    bc_args["env"]["map_dir"] = str(map_dir)
    bc_args["env"]["num_maps"] = 1
    bc_args["env"]["num_agents"] = 1
    bc_args["env"]["max_controlled_agents"] = 1
    bc_args["env"]["control_mode"] = "control_sdc_only"
    bc_args["env"]["init_mode"] = "create_all_valid"
    bc_args["env"]["init_steps"] = int(init_steps)
    bc_args["bc"]["output_dir"] = str(Path(map_dir) / "bc_rollout_tmp")
    bc_args["bc"]["skip_existing_shards"] = False

    shard_paths = build_bc_dataset(bc_args)
    assert len(shard_paths) == 1
    payload = torch.load(shard_paths[0])

    sample_sequence_ids = payload["sequence_id"].cpu().numpy()
    sample_timesteps = payload["timestep"].cpu().numpy()
    sample_actions = payload["action"].cpu().numpy()
    sample_total_costs = payload["match_cost_total"].cpu().numpy()
    assert sample_sequence_ids.size > 0

    unique_sequences = np.unique(sample_sequence_ids)
    assert unique_sequences.size == 1
    sequence_id = int(unique_sequences[0])
    sequence_mask = sample_sequence_ids == sequence_id
    order = np.argsort(sample_timesteps[sequence_mask], kind="stable")
    fitted_actions = sample_actions[sequence_mask][order]
    total_cost = float(sample_total_costs[sequence_mask][0])

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=int(env_cfg["episode_length"]),
        init_steps=init_steps,
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        resample_frequency=0,
        observation_mode=env_cfg["observation_mode"],
        action_type=env_cfg["action_type"],
        dynamics_model=env_cfg["dynamics_model"],
        extend_classic_action_space=bool(env_cfg["extend_classic_action_space"]),
    )
    try:
        env.reset(seed=0)
        start_state = env.get_global_agent_state()
        rollout_x = [float(start_state["x"][0])]
        rollout_y = [float(start_state["y"][0])]

        steps_executed = 0
        rollout_limit = min(int(num_steps), int(fitted_actions.shape[0]))
        for step_idx in range(rollout_limit):
            action_np = np.asarray([int(fitted_actions[step_idx])], dtype=np.int32)
            _, rewards, dones, truncs, _ = env.step(action_np)
            steps_executed += 1
            current_state = env.get_global_agent_state()
            rollout_x.append(float(current_state["x"][0]))
            rollout_y.append(float(current_state["y"][0]))

            assert np.isfinite(rewards).all()
            if np.all(dones) or np.all(truncs):
                break

        end_state = env.get_global_agent_state()
        distance_travelled = float(
            np.hypot(end_state["x"][0] - start_state["x"][0], end_state["y"][0] - start_state["y"][0])
        )
        capped_x, capped_y = _cap_trajectory_at_respawn(rollout_x, rollout_y)
        return {
            "x": capped_x,
            "y": capped_y,
            "steps_executed": steps_executed,
            "distance_travelled": distance_travelled,
            "fitted_steps": int(fitted_actions.shape[0]),
            "total_cost": float(total_cost),
            "actions": fitted_actions[:rollout_limit].copy(),
        }
    finally:
        env.close()


def _load_validation_map_and_trajectory(tmp_path, source_name, target_name):
    source_path = VALIDATION_BIN_DIR / source_name
    if not source_path.exists():
        pytest.skip(f"Validation bin not found: {source_path}")

    map_dir = tmp_path / target_name
    map_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, map_dir / "map_000.bin")
    tractor_gt = _load_sdc_trajectory_from_base_bin(source_path)
    roads = _load_roads_from_base_bin(source_path)
    return map_dir, tractor_gt, roads


def _prepare_rollout_warmup(args, map_dir, init_steps, max_warmup_steps=ROLLOUT_RNN_WARMUP_STEPS):
    if not args["train"]["use_rnn"]:
        return {
            "effective_init_steps": int(init_steps),
            "warmup_steps": 0,
            "warmup_actions": np.zeros((0,), dtype=np.int64),
        }

    available_prefix = max(0, int(init_steps))
    desired_warmup = min(int(max_warmup_steps), available_prefix)
    if desired_warmup <= 0:
        return {
            "effective_init_steps": int(init_steps),
            "warmup_steps": 0,
            "warmup_actions": np.zeros((0,), dtype=np.int64),
        }

    warmup_init_steps = int(init_steps) - desired_warmup
    warmup_rollout = _rollout_fitted_gt_actions(
        args,
        map_dir,
        num_steps=desired_warmup,
        init_steps=warmup_init_steps,
    )
    warmup_actions = np.asarray(warmup_rollout["actions"][:desired_warmup], dtype=np.int64)
    actual_warmup_steps = int(warmup_actions.shape[0])
    return {
        "effective_init_steps": int(init_steps) - actual_warmup_steps,
        "warmup_steps": actual_warmup_steps,
        "warmup_actions": warmup_actions,
    }


def _scenario_payloads(tmp_path):
    training_map_dir, training_gt, training_roads = _load_training_car_reference_map_and_trajectory(tmp_path)
    validation_turn_map_dir, validation_turn_gt, validation_turn_roads = _load_validation_map_and_trajectory(
        tmp_path, "map_037.bin", "validation_turn_map"
    )
    validation_straight_map_dir, validation_straight_gt, validation_straight_roads = _load_validation_map_and_trajectory(
        tmp_path, "map_008.bin", "validation_straight_map"
    )

    return [
        {
            "name": "Training Turn",
            "map_dir": training_map_dir,
            "roads": training_roads,
            "tractor_gt": training_gt,
            "init_steps": 0,
            "row_label": "Early",
        },
        {
            "name": "Validation Turn",
            "map_dir": validation_turn_map_dir,
            "roads": validation_turn_roads,
            "tractor_gt": validation_turn_gt,
            "init_steps": _movement_start_step(validation_turn_gt),
            "row_label": "Early",
        },
        {
            "name": "Validation Straight",
            "map_dir": validation_straight_map_dir,
            "roads": validation_straight_roads,
            "tractor_gt": validation_straight_gt,
            "init_steps": _movement_start_step(validation_straight_gt),
            "row_label": "Early",
        },
    ]


def _diverse_validation_scenario_payloads(tmp_path):
    scenario_specs = [
        ("Validation Straight", "map_008.bin", "validation_straight_map", "movement"),
        ("Validation Broad Left", "map_007.bin", "validation_broad_left_map", "movement"),
        ("Validation Gentle Turn", "map_020.bin", "validation_gentle_turn_map", "movement"),
        ("Validation Strong Turn", "map_037.bin", "validation_strong_turn_map", "movement"),
        ("Validation Long Right", "map_057.bin", "validation_long_right_map", "movement"),
        ("Validation Sustained Right", "map_1123.bin", "validation_sustained_right_map", "movement"),
        ("Validation Deceleration", "map_022.bin", "validation_decel_map", "start"),
        ("Validation Acceleration", "map_050.bin", "validation_accel_map", "start"),
    ]

    scenarios = []
    for name, source_name, target_name, start_mode in scenario_specs:
        map_dir, tractor_gt, roads = _load_validation_map_and_trajectory(tmp_path, source_name, target_name)
        init_steps = 0 if start_mode == "start" else _movement_start_step(tractor_gt)
        scenarios.append(
            {
                "name": name,
                "map_dir": map_dir,
                "roads": roads,
                "tractor_gt": tractor_gt,
                "init_steps": init_steps,
                "row_label": "Validation",
            }
        )
    return scenarios


def _with_nonstationary_rows(scenarios):
    expanded = []
    for scenario in scenarios:
        expanded.append(scenario)

        gt_valid = scenario["tractor_gt"]["valid"] > 0
        valid_len = int(gt_valid.sum())
        start_idx = int(scenario["init_steps"])
        candidate_rows = [
            ("Moving", max(start_idx, valid_len // 4)),
            ("Mid", max(start_idx, valid_len // 2)),
            ("Late", max(start_idx, int(valid_len * 0.75))),
        ]
        seen_steps = {start_idx}
        for row_label, init_step in candidate_rows:
            init_step = min(init_step, max(valid_len - 2, start_idx))
            if init_step in seen_steps:
                continue
            expanded.append(
                {
                    **scenario,
                    "init_steps": init_step,
                    "row_label": row_label,
                }
            )
            seen_steps.add(init_step)
    return expanded


def _save_policy_rollout_plot(
    scenarios,
    *,
    output_name="bc_streaming_latest_policy_rollout.png",
    suptitle="BC policy rollouts vs apricot reference vs GT across test artifacts",
    max_cols=None,
):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path("outputs/test_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / output_name

    row_names = []
    col_names = []
    for scenario in scenarios:
        if scenario["row_label"] not in row_names:
            row_names.append(scenario["row_label"])
        if scenario["name"] not in col_names:
            col_names.append(scenario["name"])

    if max_cols is None:
        grid_row_names = row_names
        grid_col_names = col_names
        grid_lookup = {
            (scenario["row_label"], scenario["name"]): (
                grid_row_names.index(scenario["row_label"]),
                grid_col_names.index(scenario["name"]),
            )
            for scenario in scenarios
        }
    else:
        flat_labels = [f"{scenario['name']} ({scenario['row_label']})" for scenario in scenarios]
        grid_col_count = max(1, int(max_cols))
        grid_row_count = int(np.ceil(len(flat_labels) / grid_col_count))
        grid_row_names = [f"grid_row_{idx}" for idx in range(grid_row_count)]
        grid_col_names = [f"grid_col_{idx}" for idx in range(grid_col_count)]
        grid_lookup = {}
        for idx, scenario in enumerate(scenarios):
            grid_lookup[(scenario["row_label"], scenario["name"])] = (idx // grid_col_count, idx % grid_col_count)

    fig, axes = plt.subplots(
        len(grid_row_names),
        len(grid_col_names),
        figsize=(7 * len(grid_col_names), 5.5 * len(grid_row_names)),
        squeeze=False,
    )
    used_axes = set()

    for scenario in scenarios:
        row_idx, col_idx = grid_lookup[(scenario["row_label"], scenario["name"])]
        ax = axes[row_idx][col_idx]
        used_axes.add((row_idx, col_idx))
        map_roads = scenario["roads"]
        gt_x = scenario["gt_x"]
        gt_y = scenario["gt_y"]
        rollout_x = scenario["rollout_x"]
        rollout_y = scenario["rollout_y"]
        previous_bc_x = scenario["previous_bc_x"]
        previous_bc_y = scenario["previous_bc_y"]
        current_bc_x = scenario["current_bc_x"]
        current_bc_y = scenario["current_bc_y"]
        apricot_x = scenario["apricot_x"]
        apricot_y = scenario["apricot_y"]

        for road in map_roads:
            x = np.asarray(road["x"], dtype=np.float32)
            y = np.asarray(road["y"], dtype=np.float32)
            if x.size == 0 or y.size == 0:
                continue
            ax.plot(x, y, color="0.75", linewidth=1.0, alpha=0.9, zorder=1)

        ax.plot(
            gt_x,
            gt_y,
            "--",
            color="tab:blue",
            linewidth=3.0,
            dashes=(6, 4),
            label="GT trajectory",
            zorder=7,
        )
        ax.plot(
            rollout_x,
            rollout_y,
            color="tab:orange",
            linewidth=1.8,
            marker="o",
            markersize=3,
            markevery=max(1, len(rollout_x) // 8),
            label="GT-fitted target actions",
            zorder=5,
        )
        ax.plot(
            previous_bc_x,
            previous_bc_y,
            color="tab:red",
            linewidth=1.75,
            alpha=0.8,
            linestyle=":",
            label="Previous BC",
            zorder=6,
        )
        ax.plot(
            apricot_x,
            apricot_y,
            color="tab:purple",
            linewidth=1.8,
            alpha=0.85,
            label="Apricot reference",
            zorder=4,
        )
        ax.plot(
            current_bc_x,
            current_bc_y,
            color="tab:green",
            linewidth=2.2,
            linestyle="--",
            marker="D",
            markersize=3,
            markevery=(max(1, len(current_bc_x) // 8) // 2, max(1, len(current_bc_x) // 8)),
            alpha=0.95,
            label="Current best BC",
            zorder=5,
        )
        ax.scatter([gt_x[0]], [gt_y[0]], color="tab:green", s=36, label="Start", zorder=5)

        road_x_segments = [np.asarray(road["x"], dtype=np.float32) for road in map_roads if len(road["x"]) > 0]
        road_y_segments = [np.asarray(road["y"], dtype=np.float32) for road in map_roads if len(road["y"]) > 0]
        traj_x = np.concatenate([gt_x, rollout_x, previous_bc_x, current_bc_x, apricot_x])
        traj_y = np.concatenate([gt_y, rollout_y, previous_bc_y, current_bc_y, apricot_y])
        road_x = np.concatenate(road_x_segments) if road_x_segments else traj_x
        road_y = np.concatenate(road_y_segments) if road_y_segments else traj_y

        center_x = 0.5 * float(traj_x.min() + traj_x.max())
        center_y = 0.5 * float(traj_y.min() + traj_y.max())
        traj_span_x = float(traj_x.max() - traj_x.min())
        traj_span_y = float(traj_y.max() - traj_y.min())
        traj_span = max(traj_span_x, traj_span_y, 1.0)
        road_span = max(float(road_x.max() - road_x.min()), float(road_y.max() - road_y.min()), traj_span)
        span = min(max(traj_span * 1.9, 1.0), road_span * 0.75)
        pad = 0.08 * span
        half_span = 0.5 * span + pad
        min_half_span_x = max(center_x - float(traj_x.min()), float(traj_x.max()) - center_x) + 0.08 * traj_span
        min_half_span_y = max(center_y - float(traj_y.min()), float(traj_y.max()) - center_y) + 0.08 * traj_span
        half_span = max(half_span, min_half_span_x, min_half_span_y)
        ax.set_xlim(center_x - half_span, center_x + half_span)
        ax.set_ylim(center_y - half_span, center_y + half_span)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        if col_idx == 0:
            ax.set_ylabel("y [m]")
        gt_metrics = scenario.get("gt_metrics")
        horizon_summary = (
            f"warm={int(scenario.get('warmup_steps', 0))}, "
            f"n(gt/fit/prev/apr/cur)={len(gt_x)}/{len(rollout_x)}/{len(previous_bc_x)}/{len(apricot_x)}/{len(current_bc_x)}"
        )
        if gt_metrics is None:
            title = (
                f"{scenario['name']} ({scenario['row_label']}, t={scenario['sample_time']}, "
                f"v0={scenario['initial_speed']:.2f} m/s, fit_cost={scenario['fit_total_cost']:.2f})\n"
                f"{horizon_summary}"
            )
        else:
            title = (
                f"{scenario['name']} ({scenario['row_label']}, t={scenario['sample_time']})\n"
                f"v0={gt_metrics['v0']:.2f}, vmean={gt_metrics['v_mean']:.2f}, vmax={gt_metrics['v_max']:.2f}, "
                f"|a|pk={gt_metrics['accel_peak']:.2f}, dpsi={gt_metrics['heading_delta']:.2f}, "
                f"|psi_dot|pk={gt_metrics['heading_rate_peak']:.2f}\n"
                f"{horizon_summary}"
            )
        ax.set_title(title)
        ax.legend(fontsize=9)

    for row_idx in range(len(grid_row_names)):
        for col_idx in range(len(grid_col_names)):
            if (row_idx, col_idx) not in used_axes:
                axes[row_idx][col_idx].set_visible(False)

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def _decode_discrete_action_components(actions):
    actions = np.asarray(actions, dtype=np.int64)
    num_steer = len(_CLASSIC_STEERING_VALUES)
    accel_idx = actions // num_steer
    steer_idx = actions % num_steer
    return (
        np.asarray(_CLASSIC_ACCELERATION_VALUES, dtype=np.float32)[accel_idx],
        np.asarray(_CLASSIC_STEERING_VALUES, dtype=np.float32)[steer_idx],
    )


def _save_policy_action_diff_plot(scenarios):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path("outputs/test_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "bc_streaming_action_diffs.png"

    row_names = []
    col_names = []
    for scenario in scenarios:
        if scenario["row_label"] not in row_names:
            row_names.append(scenario["row_label"])
        if scenario["name"] not in col_names:
            col_names.append(scenario["name"])

    fig, axes = plt.subplots(
        len(row_names),
        len(col_names),
        figsize=(7 * len(col_names), 4.8 * len(row_names)),
        squeeze=False,
    )
    used_axes = set()
    all_lon_diffs = []
    all_lat_diffs = []

    for scenario in scenarios:
        target_actions = np.asarray(scenario["target_actions"], dtype=np.int64)
        used_actions = np.asarray(scenario["used_actions"], dtype=np.int64)
        action_len = min(target_actions.shape[0], used_actions.shape[0])
        if action_len <= 0:
            continue
        target_accel, target_steer = _decode_discrete_action_components(target_actions[:action_len])
        used_accel, used_steer = _decode_discrete_action_components(used_actions[:action_len])
        all_lon_diffs.append(target_accel - used_accel)
        all_lat_diffs.append(target_steer - used_steer)

    max_abs_y = 1.0
    if all_lon_diffs or all_lat_diffs:
        concatenated = np.concatenate([*(all_lon_diffs or []), *(all_lat_diffs or [])]).astype(np.float32)
        if concatenated.size > 0:
            max_abs_y = max(float(np.max(np.abs(concatenated))), 1.0)
    y_pad = 0.1 * max_abs_y

    for scenario in scenarios:
        row_idx = row_names.index(scenario["row_label"])
        col_idx = col_names.index(scenario["name"])
        ax = axes[row_idx][col_idx]
        used_axes.add((row_idx, col_idx))

        target_actions = np.asarray(scenario["target_actions"], dtype=np.int64)
        used_actions = np.asarray(scenario["used_actions"], dtype=np.int64)
        action_len = min(target_actions.shape[0], used_actions.shape[0])
        if action_len <= 0:
            ax.set_visible(False)
            continue

        target_accel, target_steer = _decode_discrete_action_components(target_actions[:action_len])
        used_accel, used_steer = _decode_discrete_action_components(used_actions[:action_len])
        lon_diff = target_accel - used_accel
        lat_diff = target_steer - used_steer
        step_axis = np.arange(action_len, dtype=np.int32)

        ax.axhline(0.0, color="0.5", linewidth=1.0, linestyle="--")
        ax.plot(
            step_axis,
            target_accel,
            color="tab:orange",
            linewidth=1.2,
            linestyle="--",
            alpha=0.55,
            label="target lon",
        )
        ax.plot(
            step_axis,
            target_steer,
            color="tab:blue",
            linewidth=1.2,
            linestyle="--",
            alpha=0.55,
            label="target lat",
        )
        ax.plot(step_axis, lon_diff, color="tab:orange", linewidth=2.0, marker="o", markersize=3, label="lon diff")
        ax.plot(step_axis, lat_diff, color="tab:blue", linewidth=2.0, marker="D", markersize=3, label="lat diff")
        ax.set_xlabel("step")
        if col_idx == 0:
            ax.set_ylabel("target - used")
        ax.set_ylim(-max_abs_y - y_pad, max_abs_y + y_pad)
        ax.set_title(
            f"{scenario['name']} ({scenario['row_label']}, t={scenario['sample_time']}) "
            f"mismatch={float(np.mean(target_actions[:action_len] != used_actions[:action_len])):.2f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for row_idx in range(len(row_names)):
        for col_idx in range(len(col_names)):
            if (row_idx, col_idx) not in used_axes:
                axes[row_idx][col_idx].set_visible(False)

    fig.suptitle("BC rollout action differences vs GT-fitted target actions")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def test_latest_bc_streaming_policy_short_sim_rollout(tmp_path, monkeypatch):
    if not PREVIOUS_BC_POLICY_PATH.exists():
        pytest.skip(f"Previous BC checkpoint not found: {PREVIOUS_BC_POLICY_PATH}")
    if not CURRENT_BC_POLICY_PATH.exists():
        pytest.skip(f"Current BC checkpoint not found: {CURRENT_BC_POLICY_PATH}")
    if not APRICOT_POLICY_PATH.exists():
        pytest.skip(f"Apricot policy checkpoint not found: {APRICOT_POLICY_PATH}")

    scenarios = _with_nonstationary_rows(_scenario_payloads(tmp_path))

    monkeypatch.setattr(sys, "argv", ["pytest"])
    embedded_window_args = load_config("puffer_drive")
    embedded_window_args["train"]["device"] = "cpu"
    embedded_window_args["load_model_path"] = str(CURRENT_BC_POLICY_PATH)

    previous_bc_args = load_config("puffer_drive")
    previous_bc_args["train"]["device"] = "cpu"
    previous_bc_args["load_model_path"] = str(PREVIOUS_BC_POLICY_PATH)

    apricot_args = load_config("puffer_drive")
    apricot_args["train"]["device"] = "cpu"
    apricot_args["load_model_path"] = str(APRICOT_POLICY_PATH)
    apricot_args["env"]["extend_classic_action_space"] = False

    plotted_scenarios = []
    embedded_window_distances = []
    for scenario in scenarios:
        rollout_warmup = _prepare_rollout_warmup(
            embedded_window_args,
            scenario["map_dir"],
            int(scenario["init_steps"]),
        )
        apricot_warmup = _prepare_rollout_warmup(
            apricot_args,
            scenario["map_dir"],
            int(scenario["init_steps"]),
        )
        fitted_target_rollout = _rollout_fitted_gt_actions(
            embedded_window_args, scenario["map_dir"], num_steps=24, init_steps=int(scenario["init_steps"])
        )
        embedded_window_rollout = _rollout_policy(
            embedded_window_args,
            scenario["map_dir"],
            num_steps=24,
            init_steps=int(rollout_warmup["effective_init_steps"]),
            warmup_actions=rollout_warmup["warmup_actions"],
        )
        previous_bc_rollout = _rollout_policy(
            previous_bc_args,
            scenario["map_dir"],
            num_steps=24,
            init_steps=int(rollout_warmup["effective_init_steps"]),
            warmup_actions=rollout_warmup["warmup_actions"],
        )
        apricot_rollout = _rollout_policy(
            apricot_args,
            scenario["map_dir"],
            num_steps=24,
            init_steps=int(apricot_warmup["effective_init_steps"]),
            warmup_actions=apricot_warmup["warmup_actions"],
        )
        embedded_window_distances.append(float(embedded_window_rollout["distance_travelled"]))

        gt_valid = scenario["tractor_gt"]["valid"] > 0
        gt_x_all = np.asarray(scenario["tractor_gt"]["x"][gt_valid], dtype=np.float32)
        gt_y_all = np.asarray(scenario["tractor_gt"]["y"][gt_valid], dtype=np.float32)
        gt_start = int(scenario["init_steps"])
        common_horizon = int(fitted_target_rollout["x"].shape[0])
        gt_fit_len = min(
            max(0, gt_x_all.shape[0] - gt_start),
            common_horizon,
        )
        gt_x = gt_x_all[gt_start : gt_start + gt_fit_len]
        gt_y = gt_y_all[gt_start : gt_start + gt_fit_len]

        assert gt_x.shape[0] > 1
        plotted_scenarios.append(
            {
                "name": scenario["name"],
                "row_label": scenario["row_label"],
                "sample_time": int(scenario["init_steps"]),
                "warmup_steps": int(rollout_warmup["warmup_steps"]),
                "initial_speed": _estimate_initial_speed(scenario["tractor_gt"], int(scenario["init_steps"])),
                "roads": scenario["roads"],
                "gt_x": gt_x,
                "gt_y": gt_y,
                "rollout_x": fitted_target_rollout["x"][:common_horizon].copy(),
                "rollout_y": fitted_target_rollout["y"][:common_horizon].copy(),
                "previous_bc_x": previous_bc_rollout["x"][:common_horizon].copy(),
                "previous_bc_y": previous_bc_rollout["y"][:common_horizon].copy(),
                "current_bc_x": embedded_window_rollout["x"][:common_horizon].copy(),
                "current_bc_y": embedded_window_rollout["y"][:common_horizon].copy(),
                "apricot_x": apricot_rollout["x"][:common_horizon].copy(),
                "apricot_y": apricot_rollout["y"][:common_horizon].copy(),
                "fit_total_cost": fitted_target_rollout["total_cost"],
                "target_actions": fitted_target_rollout["actions"],
                "used_actions": embedded_window_rollout["actions"],
            }
        )

    plot_path = _save_policy_rollout_plot(plotted_scenarios)
    diff_plot_path = _save_policy_action_diff_plot(plotted_scenarios)

    assert max(embedded_window_distances) > 0.05
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert diff_plot_path.exists()
    assert diff_plot_path.stat().st_size > 0


def test_diverse_validation_bc_policy_rollout_plot(tmp_path, monkeypatch):
    if not PREVIOUS_BC_POLICY_PATH.exists():
        pytest.skip(f"Previous BC checkpoint not found: {PREVIOUS_BC_POLICY_PATH}")
    if not CURRENT_BC_POLICY_PATH.exists():
        pytest.skip(f"Current BC checkpoint not found: {CURRENT_BC_POLICY_PATH}")
    if not APRICOT_POLICY_PATH.exists():
        pytest.skip(f"Apricot policy checkpoint not found: {APRICOT_POLICY_PATH}")

    scenarios = _diverse_validation_scenario_payloads(tmp_path)

    monkeypatch.setattr(sys, "argv", ["pytest"])
    current_bc_args = load_config("puffer_drive")
    current_bc_args["train"]["device"] = "cpu"
    current_bc_args["load_model_path"] = str(CURRENT_BC_POLICY_PATH)

    previous_bc_args = load_config("puffer_drive")
    previous_bc_args["train"]["device"] = "cpu"
    previous_bc_args["load_model_path"] = str(PREVIOUS_BC_POLICY_PATH)

    apricot_args = load_config("puffer_drive")
    apricot_args["train"]["device"] = "cpu"
    apricot_args["load_model_path"] = str(APRICOT_POLICY_PATH)
    apricot_args["env"]["extend_classic_action_space"] = False

    plotted_scenarios = []
    for scenario in scenarios:
        rollout_warmup = _prepare_rollout_warmup(
            current_bc_args,
            scenario["map_dir"],
            int(scenario["init_steps"]),
        )
        apricot_warmup = _prepare_rollout_warmup(
            apricot_args,
            scenario["map_dir"],
            int(scenario["init_steps"]),
        )
        fitted_target_rollout = _rollout_fitted_gt_actions(
            current_bc_args, scenario["map_dir"], num_steps=24, init_steps=int(scenario["init_steps"])
        )
        current_bc_rollout = _rollout_policy(
            current_bc_args,
            scenario["map_dir"],
            num_steps=24,
            init_steps=int(rollout_warmup["effective_init_steps"]),
            warmup_actions=rollout_warmup["warmup_actions"],
        )
        previous_bc_rollout = _rollout_policy(
            previous_bc_args,
            scenario["map_dir"],
            num_steps=24,
            init_steps=int(rollout_warmup["effective_init_steps"]),
            warmup_actions=rollout_warmup["warmup_actions"],
        )
        apricot_rollout = _rollout_policy(
            apricot_args,
            scenario["map_dir"],
            num_steps=24,
            init_steps=int(apricot_warmup["effective_init_steps"]),
            warmup_actions=apricot_warmup["warmup_actions"],
        )

        gt_valid = scenario["tractor_gt"]["valid"] > 0
        gt_x_all = np.asarray(scenario["tractor_gt"]["x"][gt_valid], dtype=np.float32)
        gt_y_all = np.asarray(scenario["tractor_gt"]["y"][gt_valid], dtype=np.float32)
        gt_start = int(scenario["init_steps"])
        common_horizon = int(fitted_target_rollout["x"].shape[0])
        gt_fit_len = min(
            max(0, gt_x_all.shape[0] - gt_start),
            common_horizon,
        )
        gt_x = gt_x_all[gt_start : gt_start + gt_fit_len]
        gt_y = gt_y_all[gt_start : gt_start + gt_fit_len]

        assert gt_x.shape[0] > 1
        plotted_scenarios.append(
            {
                "name": scenario["name"],
                "row_label": scenario["row_label"],
                "sample_time": int(scenario["init_steps"]),
                "warmup_steps": int(rollout_warmup["warmup_steps"]),
                "initial_speed": _estimate_initial_speed(scenario["tractor_gt"], int(scenario["init_steps"])),
                "gt_metrics": _summarize_gt_motion(scenario["tractor_gt"], int(scenario["init_steps"])),
                "roads": scenario["roads"],
                "gt_x": gt_x,
                "gt_y": gt_y,
                "rollout_x": fitted_target_rollout["x"][:common_horizon].copy(),
                "rollout_y": fitted_target_rollout["y"][:common_horizon].copy(),
                "previous_bc_x": previous_bc_rollout["x"][:common_horizon].copy(),
                "previous_bc_y": previous_bc_rollout["y"][:common_horizon].copy(),
                "current_bc_x": current_bc_rollout["x"][:common_horizon].copy(),
                "current_bc_y": current_bc_rollout["y"][:common_horizon].copy(),
                "apricot_x": apricot_rollout["x"][:common_horizon].copy(),
                "apricot_y": apricot_rollout["y"][:common_horizon].copy(),
                "fit_total_cost": fitted_target_rollout["total_cost"],
            }
        )

    plot_path = _save_policy_rollout_plot(
        plotted_scenarios,
        output_name="bc_streaming_diverse_validation_policy_rollout.png",
        suptitle="BC rollouts on a deliberately diverse set of validation scenarios",
        max_cols=3,
    )

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
