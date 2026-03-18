from pathlib import Path
from types import SimpleNamespace
import shutil
import sys

import numpy as np
import pytest
import torch

from pufferlib.ocean.drive.drive import Drive, build_bc_dataset
from pufferlib.pufferl import load_config, load_policy
from tests.test_drive_bin_simulator_load import _load_roads_from_base_bin, _load_sdc_trajectory_from_base_bin


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

        for _ in range(num_steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, device=device)
                logits, _ = policy.forward_eval(obs_tensor, state)
                action = _greedy_multidiscrete_action(logits)
                action_np = action.cpu().numpy().reshape(env.action_space.shape)

            obs, rewards, dones, truncs, _ = env.step(action_np)
            current_state = env.get_global_agent_state()
            rollout_x.append(float(current_state["x"][0]))
            rollout_y.append(float(current_state["y"][0]))

            assert np.isfinite(obs).all()
            assert np.isfinite(rewards).all()
            if np.all(dones) or np.all(truncs):
                break

        capped_x, capped_y = _cap_trajectory_at_respawn(rollout_x, rollout_y)
        return {"x": capped_x, "y": capped_y}
    finally:
        env.close()


def _rollout_fitted_gt_actions(args, map_dir, num_steps, init_steps=0):
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
    unique_sequences = np.unique(sample_sequence_ids)
    assert unique_sequences.size == 1
    sequence_mask = sample_sequence_ids == int(unique_sequences[0])
    order = np.argsort(sample_timesteps[sequence_mask], kind="stable")
    fitted_actions = sample_actions[sequence_mask][order]

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
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
        env.reset(seed=0)
        start_state = env.get_global_agent_state()
        rollout_x = [float(start_state["x"][0])]
        rollout_y = [float(start_state["y"][0])]
        rollout_limit = min(int(num_steps), int(fitted_actions.shape[0]))
        for step_idx in range(rollout_limit):
            action_np = np.asarray([int(fitted_actions[step_idx])], dtype=np.int32)
            _, rewards, dones, truncs, _ = env.step(action_np)
            current_state = env.get_global_agent_state()
            rollout_x.append(float(current_state["x"][0]))
            rollout_y.append(float(current_state["y"][0]))
            assert np.isfinite(rewards).all()
            if np.all(dones) or np.all(truncs):
                break
        capped_x, capped_y = _cap_trajectory_at_respawn(rollout_x, rollout_y)
        return {"x": capped_x, "y": capped_y}
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


def _save_policy_rollout_plot(scenarios):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path("outputs/test_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "bc_streaming_diverse_validation_policy_rollout.png"

    ncols = 3
    nrows = int(np.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)
    flat_axes = list(axes.flat)

    for ax, scenario in zip(flat_axes, scenarios):
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
        ax.plot(previous_bc_x, previous_bc_y, color="tab:red", linewidth=1.75, alpha=0.8, linestyle=":", label="Previous BC", zorder=6)
        ax.plot(apricot_x, apricot_y, color="tab:purple", linewidth=1.8, alpha=0.85, label="Apricot reference", zorder=4)
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

        traj_x = np.concatenate([gt_x, rollout_x, previous_bc_x, current_bc_x, apricot_x])
        traj_y = np.concatenate([gt_y, rollout_y, previous_bc_y, current_bc_y, apricot_y])
        center_x = 0.5 * float(traj_x.min() + traj_x.max())
        center_y = 0.5 * float(traj_y.min() + traj_y.max())
        traj_span_x = float(traj_x.max() - traj_x.min())
        traj_span_y = float(traj_y.max() - traj_y.min())
        traj_span = max(traj_span_x, traj_span_y, 1.0)
        half_span = 0.6 * traj_span + 0.08 * traj_span
        ax.set_xlim(center_x - half_span, center_x + half_span)
        ax.set_ylim(center_y - half_span, center_y + half_span)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        gt_metrics = scenario["gt_metrics"]
        ax.set_title(
            f"{scenario['name']} (t={scenario['sample_time']})\n"
            f"v0={gt_metrics['v0']:.2f}, vmean={gt_metrics['v_mean']:.2f}, vmax={gt_metrics['v_max']:.2f}, "
            f"|a|pk={gt_metrics['accel_peak']:.2f}, dpsi={gt_metrics['heading_delta']:.2f}, "
            f"|psi_dot|pk={gt_metrics['heading_rate_peak']:.2f}"
        )
        ax.legend(fontsize=8)

    for ax in flat_axes[len(scenarios):]:
        ax.set_visible(False)

    fig.suptitle("BC rollouts on a deliberately diverse set of validation scenarios")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


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
        fitted_target_rollout = _rollout_fitted_gt_actions(
            current_bc_args, scenario["map_dir"], num_steps=24, init_steps=int(scenario["init_steps"])
        )
        current_bc_rollout = _rollout_policy(
            current_bc_args, scenario["map_dir"], num_steps=24, init_steps=int(scenario["init_steps"])
        )
        previous_bc_rollout = _rollout_policy(
            previous_bc_args, scenario["map_dir"], num_steps=24, init_steps=int(scenario["init_steps"])
        )
        apricot_rollout = _rollout_policy(
            apricot_args, scenario["map_dir"], num_steps=24, init_steps=int(scenario["init_steps"])
        )

        gt_valid = scenario["tractor_gt"]["valid"] > 0
        gt_x_all = np.asarray(scenario["tractor_gt"]["x"][gt_valid], dtype=np.float32)
        gt_y_all = np.asarray(scenario["tractor_gt"]["y"][gt_valid], dtype=np.float32)
        gt_start = int(scenario["init_steps"])
        aligned_len = min(
            max(0, gt_x_all.shape[0] - gt_start),
            current_bc_rollout["x"].shape[0],
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
                "sample_time": int(scenario["init_steps"]),
                "gt_metrics": _summarize_gt_motion(scenario["tractor_gt"], int(scenario["init_steps"])),
                "roads": scenario["roads"],
                "gt_x": gt_x,
                "gt_y": gt_y,
                "rollout_x": fitted_target_rollout["x"][:aligned_len],
                "rollout_y": fitted_target_rollout["y"][:aligned_len],
                "previous_bc_x": previous_bc_rollout["x"][:aligned_len],
                "previous_bc_y": previous_bc_rollout["y"][:aligned_len],
                "current_bc_x": current_bc_rollout["x"][:aligned_len],
                "current_bc_y": current_bc_rollout["y"][:aligned_len],
                "apricot_x": apricot_rollout["x"][:aligned_len],
                "apricot_y": apricot_rollout["y"][:aligned_len],
                "initial_speed": _estimate_initial_speed(scenario["tractor_gt"], int(scenario["init_steps"])),
            }
        )

    plot_path = _save_policy_rollout_plot(plotted_scenarios)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
