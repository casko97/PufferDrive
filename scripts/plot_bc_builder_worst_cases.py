from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import torch

from pufferlib.ocean.drive.drive import Drive, binding, load_drive_builder_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACCELERATION_VALUES = np.asarray((-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0), dtype=np.float32)
NUM_STEER = 13


def _neutral_action() -> int:
    return int(np.where(ACCELERATION_VALUES == 0.0)[0][0]) * NUM_STEER + 6


def _load_worst_cases(shard_dir: Path, top_k: int) -> list[tuple[float, tuple[int, int, int]]]:
    rows = []
    for path in sorted(shard_dir.glob("map_*.pt")):
        shard = torch.load(path)
        seen: dict[tuple[int, int, int], float] = {}
        for i in range(shard["obs"].shape[0]):
            key = (int(shard["map_id"][i]), int(shard["scenario_id"][i]), int(shard["agent_id"][i]))
            seen.setdefault(key, float(shard["match_cost_total"][i]))
        rows.extend((cost, key) for key, cost in seen.items())
    rows.sort(reverse=True)
    return rows[:top_k]


def _fit_all_agents(map_path: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_dim = (
        binding.EGO_FEATURES_CLASSIC
        + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
        + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    )
    max_agents = 256
    observations = np.zeros((max_agents, obs_dim), dtype=np.float32)
    actions_buf = np.zeros(max_agents, dtype=np.int32)
    rewards = np.zeros(max_agents, dtype=np.float32)
    terminals = np.zeros(max_agents, dtype=np.uint8)
    truncations = np.zeros(max_agents, dtype=np.uint8)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bc_vis_{map_path.stem}_"))
    staged_map = tmp_dir / "map_000.bin"
    shutil.copy2(map_path, staged_map)

    env_cfg = cfg["env"]
    bc_cfg = cfg["bc"]
    env_handle = binding.env_init(
        observations,
        actions_buf,
        rewards,
        terminals,
        truncations,
        0,
        human_agent_idx=0,
        reward_vehicle_collision=env_cfg["reward_vehicle_collision"],
        reward_offroad_collision=env_cfg["reward_offroad_collision"],
        reward_goal=env_cfg["reward_goal"],
        reward_goal_post_respawn=env_cfg["reward_goal_post_respawn"],
        goal_radius=env_cfg["goal_radius"],
        goal_speed=env_cfg["goal_speed"],
        goal_behavior=env_cfg["goal_behavior"],
        goal_target_distance=env_cfg["goal_target_distance"],
        collision_behavior=env_cfg["collision_behavior"],
        offroad_behavior=env_cfg["offroad_behavior"],
        dt=env_cfg["dt"],
        episode_length=env_cfg["episode_length"],
        termination_mode=env_cfg["termination_mode"],
        max_controlled_agents=max_agents,
        map_id=0,
        max_agents=max_agents,
        ini_file="pufferlib/config/ocean/drive.ini",
        init_steps=env_cfg["init_steps"],
        init_mode=0 if env_cfg["init_mode"] == "create_all_valid" else 1,
        control_mode={
            "control_vehicles": 0,
            "control_agents": 1,
            "control_wosac": 2,
            "control_sdc_only": 3,
        }[env_cfg["control_mode"]],
        map_dir=str(tmp_dir),
        non_kinematic_vehicle_params_override=None,
        force_zero_trailer_articulation_at_init=0,
        extend_classic_action_space=1 if env_cfg.get("extend_classic_action_space", True) else 0,
    )
    binding.env_reset(env_handle, 0)

    active_count = binding.env_get_active_agent_count(env_handle)
    scenario_ids = np.zeros(active_count, dtype=np.int32)
    agent_ids = np.zeros(active_count, dtype=np.int32)
    binding.env_get_active_agent_info(env_handle, scenario_ids, agent_ids)

    max_steps = max(0, int(env_cfg["episode_length"]) - int(env_cfg["init_steps"]) - 1)
    all_actions = np.full((active_count, max_steps), _neutral_action(), dtype=np.int32)
    all_num_steps = np.zeros(active_count, dtype=np.int32)
    all_total_cost = np.zeros(active_count, dtype=np.float32)
    all_total_lat = np.zeros(active_count, dtype=np.float32)
    all_total_lon = np.zeros(active_count, dtype=np.float32)
    tmp_costs = np.zeros(max_steps, dtype=np.float32)
    tmp_lat = np.zeros(max_steps, dtype=np.float32)
    tmp_lon = np.zeros(max_steps, dtype=np.float32)

    for slot in range(active_count):
        seq = np.full(max_steps, _neutral_action(), dtype=np.int32)
        n, tc, tlat, tlon = binding.env_fit_discrete_action_sequence(
            env_handle,
            slot,
            int(bc_cfg["beam_width"]),
            int(bc_cfg["planning_horizon"]),
            float(bc_cfg["match_weight_lateral"]),
            float(bc_cfg["match_weight_longitudinal"]),
            float(bc_cfg["match_weight_heading"]),
            float(bc_cfg["match_weight_speed"]),
            float(bc_cfg["match_weight_steer_change"]),
            float(bc_cfg["match_weight_accel_change"]),
            float(bc_cfg["match_weight_reverse"]),
            float(bc_cfg["match_weight_progress"]),
            float(bc_cfg["match_weight_steer_flip"]),
            float(bc_cfg["match_weight_ref_accel"]),
            float(bc_cfg["match_weight_ref_steer"]),
            seq,
            tmp_costs,
            tmp_lat,
            tmp_lon,
        )
        all_actions[slot, :n] = seq[:n]
        all_num_steps[slot] = n
        all_total_cost[slot] = tc
        all_total_lat[slot] = tlat
        all_total_lon[slot] = tlon

    binding.env_close(env_handle)
    return staged_map, scenario_ids, agent_ids, all_actions, all_num_steps, np.stack(
        [all_total_cost, all_total_lat, all_total_lon], axis=1
    )


def _detect_respawn_reason(reward: float, cfg: dict) -> str:
    env_cfg = cfg["env"]
    if np.isclose(reward, env_cfg["reward_goal"], atol=1e-5) or np.isclose(
        reward, env_cfg["reward_goal_post_respawn"], atol=1e-5
    ):
        return "goal_respawn"
    if np.isclose(reward, env_cfg["reward_vehicle_collision"], atol=1e-5):
        return "collision_respawn"
    if np.isclose(reward, env_cfg["reward_offroad_collision"], atol=1e-5):
        return "offroad_respawn"
    return "respawn_unknown"


def _rollout_target(staged_map: Path, target_slot: int, all_actions: np.ndarray, all_num_steps: np.ndarray, cfg: dict):
    env_cfg = cfg["env"]
    active_count = all_actions.shape[0]
    env = Drive(
        num_agents=active_count,
        num_maps=1,
        map_dir=str(staged_map.parent),
        episode_length=env_cfg["episode_length"],
        init_steps=env_cfg["init_steps"],
        control_mode=env_cfg["control_mode"],
        init_mode=env_cfg["init_mode"],
        resample_frequency=0,
        observation_mode=env_cfg["observation_mode"],
        extend_classic_action_space=env_cfg.get("extend_classic_action_space", True),
    )
    try:
        env.reset(seed=0)
        gt = env.get_ground_truth_trajectories()
        gt_x_all = gt["x"][:, 0]
        gt_y_all = gt["y"][:, 0]
        gt_heading_all = np.unwrap(gt["heading"][:, 0], axis=1)
        gt_valid_all = gt["valid"][:, 0].astype(bool)

        target_steps = int(all_num_steps[target_slot])
        target_valid = gt_valid_all[target_slot]
        gt_x = gt_x_all[target_slot][target_valid][: target_steps + 1]
        gt_y = gt_y_all[target_slot][target_valid][: target_steps + 1]
        gt_heading = gt_heading_all[target_slot][target_valid][: target_steps + 1]

        state = env.get_global_agent_state()
        rollout_x = [float(state["x"][target_slot])]
        rollout_y = [float(state["y"][target_slot])]
        rewards = []

        action_vec = np.full((active_count, 1), _neutral_action(), dtype=np.int32)
        respawn_idx = None
        respawn_reason = None
        initial_xy = np.asarray([gt_x[0], gt_y[0]], dtype=np.float32)
        for step in range(target_steps):
            for slot in range(active_count):
                action_vec[slot, 0] = (
                    int(all_actions[slot, step]) if step < int(all_num_steps[slot]) else _neutral_action()
                )
            _, step_rewards, _, _, _ = env.step(action_vec)
            state = env.get_global_agent_state()
            next_xy = np.asarray([float(state["x"][target_slot]), float(state["y"][target_slot])], dtype=np.float32)
            rollout_x.append(float(next_xy[0]))
            rollout_y.append(float(next_xy[1]))
            rewards.append(float(step_rewards[target_slot]))

            if np.linalg.norm(next_xy - initial_xy) < 1e-3 and step >= 1:
                respawn_idx = step + 1
                respawn_reason = _detect_respawn_reason(rewards[-1], cfg)
                break
    finally:
        env.close()

    if respawn_idx is not None:
        gt_x = gt_x[:respawn_idx]
        gt_y = gt_y[:respawn_idx]
        gt_heading = gt_heading[:respawn_idx]
        rollout_x = rollout_x[:respawn_idx]
        rollout_y = rollout_y[:respawn_idx]

    rollout_x = np.asarray(rollout_x, dtype=np.float32)
    rollout_y = np.asarray(rollout_y, dtype=np.float32)
    disp = np.sqrt((rollout_x - gt_x) ** 2 + (rollout_y - gt_y) ** 2)
    origin_x, origin_y = float(gt_x[0]), float(gt_y[0])
    return {
        "gt_x": gt_x - origin_x,
        "gt_y": gt_y - origin_y,
        "rollout_x": rollout_x - origin_x,
        "rollout_y": rollout_y - origin_y,
        "mean_err": float(disp.mean()),
        "final_err": float(disp[-1]),
        "heading_delta": float(abs(gt_heading[-1] - gt_heading[0])) if len(gt_heading) > 1 else 0.0,
        "respawn_reason": respawn_reason or "none",
        "respawn_cut_idx": respawn_idx,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot worst BC builder cases with respawn truncation")
    parser.add_argument("--shard-dir", type=Path, default=Path("outputs/bc_builder_validation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/test_visualizations/bc_builder_worst_cases.png"))
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    cfg = load_drive_builder_config()
    worst_cases = _load_worst_cases(args.shard_dir, args.top_k)
    results = []
    for _, (map_id, scenario_id, agent_id) in worst_cases:
        map_path = Path(f"resources/drive/binaries/training/map_{map_id:03d}.bin")
        staged_map, scenario_ids, agent_ids, all_actions, all_num_steps, all_costs = _fit_all_agents(map_path, cfg)
        matches = np.where((scenario_ids == scenario_id) & (agent_ids == agent_id))[0]
        if len(matches) != 1:
            continue
        slot = int(matches[0])
        rollout = _rollout_target(staged_map, slot, all_actions, all_num_steps, cfg)
        results.append(
            {
                "map_id": map_id,
                "agent_id": agent_id,
                "total_cost": float(all_costs[slot, 0]),
                "lat_cost": float(all_costs[slot, 1]),
                "lon_cost": float(all_costs[slot, 2]),
                **rollout,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = max(1, int(np.ceil(len(results) / 2)))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows), squeeze=False)
    flat_axes = list(axes.flat)
    for ax, result in zip(flat_axes, results):
        plot_x = np.concatenate([result["gt_x"], result["rollout_x"]])
        plot_y = np.concatenate([result["gt_y"], result["rollout_y"]])
        x_min, x_max = float(plot_x.min()), float(plot_x.max())
        y_min, y_max = float(plot_y.min()), float(plot_y.max())
        span = max(1.0, x_max - x_min, y_max - y_min)
        half_span = 0.5 * span * 1.08
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)

        ax.plot(result["gt_x"], result["gt_y"], marker="o", markersize=2.5, linewidth=1.4, label="GT")
        ax.plot(result["rollout_x"], result["rollout_y"], marker="x", markersize=2.5, linewidth=1.4, label="Rollout")
        ax.set_xlim(center_x - half_span, center_x + half_span)
        ax.set_ylim(center_y - half_span, center_y + half_span)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x rel. to start [m]")
        ax.set_ylabel("y rel. to start [m]")
        ax.set_title(
            f"map_{result['map_id']:03d} agent={result['agent_id']}\n"
            f"total_cost={result['total_cost']:.1f} lat={result['lat_cost']:.1f} lon={result['lon_cost']:.1f}\n"
            f"mean_err={result['mean_err']:.2f} final_err={result['final_err']:.2f} dh={result['heading_delta']:.2f}\n"
            f"respawn={result['respawn_reason']}"
        )
        ax.legend()

    for ax in flat_axes[len(results) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
