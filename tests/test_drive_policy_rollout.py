from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from pufferlib.ocean.drive.drive import Drive, build_bc_dataset
from pufferlib.pufferl import load_config, load_policy
from tests.test_drive_bin_simulator_load import (
    _load_boston_car_header_map_and_trajectories,
    _load_boston_reference_map_and_trajectories,
    _load_roads_from_base_bin,
    _load_training_car_reference_map_and_trajectory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE_DT_SECONDS = 0.1
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
CURRENT_BC_POLICY_PATH = REPO_ROOT / "experiments_bc" / "bc-full-windows-stride10-20260317-171857" / "best.pt"


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


def _rollout_policy(args, map_dir, num_steps, init_steps=0):
    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=max(48, num_steps + 24),
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
        start_state = env.get_global_agent_state()
        rollout_x = [float(start_state["x"][0])]
        rollout_y = [float(start_state["y"][0])]
        device = torch.device(args["train"]["device"])

        state = {}
        if args["train"]["use_rnn"]:
            state = {
                "lstm_h": torch.zeros(env.num_agents, policy.hidden_size, device=device),
                "lstm_c": torch.zeros(env.num_agents, policy.hidden_size, device=device),
            }

        steps_executed = 0
        for _ in range(num_steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, device=device)
                logits, _ = policy.forward_eval(obs_tensor, state)
                action = _greedy_multidiscrete_action(logits)
                action_np = action.cpu().numpy().reshape(env.action_space.shape)

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
        }
    finally:
        env.close()


def _scenario_payloads(tmp_path):
    training_map_dir, training_gt, training_roads = _load_training_car_reference_map_and_trajectory(tmp_path)

    boston_map_dir, boston_gt_refs = _load_boston_reference_map_and_trajectories(tmp_path)
    boston_roads = _load_roads_from_base_bin(boston_map_dir / "map_000.bin")

    header_map_dir, header_gt_refs = _load_boston_car_header_map_and_trajectories(tmp_path)
    header_roads = _load_roads_from_base_bin(header_map_dir / "map_000.bin")

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
            "name": "Boston Reference",
            "map_dir": boston_map_dir,
            "roads": boston_roads,
            "tractor_gt": boston_gt_refs["tractor"],
            "init_steps": _movement_start_step(boston_gt_refs["tractor"]),
            "row_label": "Early",
        },
        {
            "name": "Boston Car Header",
            "map_dir": header_map_dir,
            "roads": header_roads,
            "tractor_gt": header_gt_refs["tractor"],
            "init_steps": _movement_start_step(header_gt_refs["tractor"]),
            "row_label": "Early",
        },
    ]


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


def _save_policy_rollout_plot(scenarios):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path("outputs/test_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "bc_streaming_latest_policy_rollout.png"

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
        figsize=(7 * len(col_names), 5.5 * len(row_names)),
        squeeze=False,
    )
    used_axes = set()

    for scenario in scenarios:
        row_idx = row_names.index(scenario["row_label"])
        col_idx = col_names.index(scenario["name"])
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

        ax.plot(gt_x, gt_y, "--", color="tab:blue", linewidth=2.0, label="GT trajectory", zorder=3)
        ax.plot(
            rollout_x,
            rollout_y,
            color="tab:orange",
            linewidth=2.4,
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
        ax.set_title(
            f"{scenario['name']} ({scenario['row_label']}, t={scenario['sample_time']}, "
            f"v0={scenario['initial_speed']:.2f} m/s, fit_cost={scenario['fit_total_cost']:.2f})"
        )
        ax.legend(fontsize=9)

    for row_idx in range(len(row_names)):
        for col_idx in range(len(col_names)):
            if (row_idx, col_idx) not in used_axes:
                axes[row_idx][col_idx].set_visible(False)

    fig.suptitle("BC policy rollouts vs apricot reference vs GT across test artifacts")
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
        fitted_target_rollout = _rollout_fitted_gt_actions(
            embedded_window_args, scenario["map_dir"], num_steps=24, init_steps=int(scenario["init_steps"])
        )
        embedded_window_rollout = _rollout_policy(
            embedded_window_args, scenario["map_dir"], num_steps=24, init_steps=int(scenario["init_steps"])
        )
        previous_bc_rollout = _rollout_policy(
            previous_bc_args, scenario["map_dir"], num_steps=24, init_steps=int(scenario["init_steps"])
        )
        apricot_rollout = _rollout_policy(
            apricot_args, scenario["map_dir"], num_steps=24, init_steps=int(scenario["init_steps"])
        )
        embedded_window_distances.append(float(embedded_window_rollout["distance_travelled"]))

        gt_valid = scenario["tractor_gt"]["valid"] > 0
        gt_x_all = np.asarray(scenario["tractor_gt"]["x"][gt_valid], dtype=np.float32)
        gt_y_all = np.asarray(scenario["tractor_gt"]["y"][gt_valid], dtype=np.float32)
        gt_start = int(scenario["init_steps"])
        aligned_len = min(
            max(0, gt_x_all.shape[0] - gt_start),
            embedded_window_rollout["x"].shape[0],
            fitted_target_rollout["x"].shape[0],
            previous_bc_rollout["x"].shape[0],
            apricot_rollout["x"].shape[0],
        )
        gt_x = gt_x_all[gt_start : gt_start + aligned_len]
        gt_y = gt_y_all[gt_start : gt_start + aligned_len]

        assert gt_x.shape[0] > 1
        plotted_scenarios.append(
            {
                "name": scenario["name"],
                "row_label": scenario["row_label"],
                "sample_time": int(scenario["init_steps"]),
                "initial_speed": _estimate_initial_speed(scenario["tractor_gt"], int(scenario["init_steps"])),
                "roads": scenario["roads"],
                "gt_x": gt_x,
                "gt_y": gt_y,
                "rollout_x": fitted_target_rollout["x"][:aligned_len],
                "rollout_y": fitted_target_rollout["y"][:aligned_len],
                "previous_bc_x": previous_bc_rollout["x"][:aligned_len],
                "previous_bc_y": previous_bc_rollout["y"][:aligned_len],
                "current_bc_x": embedded_window_rollout["x"][:aligned_len],
                "current_bc_y": embedded_window_rollout["y"][:aligned_len],
                "apricot_x": apricot_rollout["x"][:aligned_len],
                "apricot_y": apricot_rollout["y"][:aligned_len],
                "fit_total_cost": fitted_target_rollout["total_cost"],
            }
        )

    plot_path = _save_policy_rollout_plot(plotted_scenarios)

    assert max(embedded_window_distances) > 0.05
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
