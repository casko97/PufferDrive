import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import binding, save_map_binary
from scripts.export_paired_offline_fits import (
    FIT_BEAM_WIDTH,
    FIT_MATCH_WEIGHT_ACCEL_CHANGE,
    FIT_MATCH_WEIGHT_HEADING,
    FIT_MATCH_WEIGHT_LATERAL,
    FIT_MATCH_WEIGHT_LONGITUDINAL,
    FIT_MATCH_WEIGHT_PROGRESS,
    FIT_MATCH_WEIGHT_REF_ACCEL,
    FIT_MATCH_WEIGHT_REF_STEER,
    FIT_MATCH_WEIGHT_REVERSE,
    FIT_MATCH_WEIGHT_SPEED,
    FIT_MATCH_WEIGHT_STEER_CHANGE,
    FIT_MATCH_WEIGHT_STEER_FLIP,
    OFFLINE_FIT_COLLISION_BEHAVIOR,
    OFFLINE_FIT_CONTROL_MODE,
    OFFLINE_FIT_DT,
    OFFLINE_FIT_EPISODE_LENGTH,
    OFFLINE_FIT_GOAL_BEHAVIOR,
    OFFLINE_FIT_GOAL_RADIUS,
    OFFLINE_FIT_GOAL_SPEED,
    OFFLINE_FIT_GOAL_TARGET_DISTANCE,
    OFFLINE_FIT_OFFROAD_BEHAVIOR,
    OFFLINE_FIT_TERMINATION_MODE,
)

CAR_BOSTON_TEST_DIR = Path("pufferlib/resources/drive/binaries/nuplanCarBostonTest10")
TRUCK_BOSTON_TEST_DIR = Path("pufferlib/resources/drive/binaries/nuplanTruckBostonTest10")
VISUALIZATION_OUTPUT_DIR = Path("outputs/test_visualizations")
FIT_PLANNING_HORIZON = 5


def _logged_vehicle(track_id, xs, ys, goal_x, goal_y, steps=91, length=4.5, width=1.9):
    headings = []
    velocities = []
    valid = []
    padded_xs = []
    padded_ys = []
    last_x = float(xs[-1])
    last_y = float(ys[-1])

    for idx in range(steps):
        if idx < len(xs):
            x = float(xs[idx])
            y = float(ys[idx])
            valid.append(1)
        else:
            x = last_x
            y = last_y
            valid.append(0)
        padded_xs.append(x)
        padded_ys.append(y)

    for idx in range(steps):
        if idx + 1 < len(xs):
            dx = float(xs[idx + 1] - xs[idx])
            dy = float(ys[idx + 1] - ys[idx])
        elif idx < len(xs):
            dx = float(xs[idx] - xs[idx - 1]) if idx > 0 else 0.0
            dy = float(ys[idx] - ys[idx - 1]) if idx > 0 else 0.0
        else:
            dx = 0.0
            dy = 0.0
        headings.append(float(math.atan2(dy, dx if abs(dx) > 1e-6 else 1e-6)))
        velocities.append({"x": dx / 0.1, "y": dy / 0.1, "z": 0.0})

    return {
        "id": track_id,
        "type": "vehicle",
        "position": [{"x": x, "y": y, "z": 0.0} for x, y in zip(padded_xs, padded_ys)],
        "velocity": velocities,
        "heading": headings,
        "valid": valid,
        "width": width,
        "length": length,
        "height": 1.6,
        "goalPosition": {"x": float(goal_x), "y": float(goal_y), "z": 0.0},
        "mark_as_expert": 0,
    }


def _write_offline_fit_map(map_dir: Path):
    ego_x = [0.0, 0.8, 1.7, 2.7, 3.8, 5.0]
    ego_y = [0.0, 0.0, 0.05, 0.15, 0.30, 0.5]
    scenario = {
        "metadata": {"sdc_track_index": 0, "tracks_to_predict": [{"track_index": 0}]},
        "objects": [_logged_vehicle(10, ego_x, ego_y, goal_x=8.0, goal_y=1.0)],
        "roads": [],
    }
    save_map_binary(scenario, str(map_dir / "map_000.bin"), unique_map_id=321)


def _make_obs_dim():
    return (
        binding.EGO_FEATURES_CLASSIC
        + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
        + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    )


def _make_env_handle(map_dir: Path):
    obs = np.zeros((1, _make_obs_dim()), dtype=np.float32)
    actions = np.zeros(1, dtype=np.int32)
    rewards = np.zeros(1, dtype=np.float32)
    terminals = np.zeros(1, dtype=np.uint8)
    truncations = np.zeros(1, dtype=np.uint8)
    return binding.env_init(
        obs,
        actions,
        rewards,
        terminals,
        truncations,
        0,
        human_agent_idx=0,
        ini_file="pufferlib/config/ocean/drive.ini",
        map_dir=str(map_dir),
        map_id=0,
        max_agents=1,
        max_controlled_agents=1,
        init_steps=0,
        init_mode=0,
        control_mode=OFFLINE_FIT_CONTROL_MODE,
        goal_behavior=OFFLINE_FIT_GOAL_BEHAVIOR,
        goal_target_distance=OFFLINE_FIT_GOAL_TARGET_DISTANCE,
        goal_radius=OFFLINE_FIT_GOAL_RADIUS,
        goal_speed=OFFLINE_FIT_GOAL_SPEED,
        collision_behavior=OFFLINE_FIT_COLLISION_BEHAVIOR,
        offroad_behavior=OFFLINE_FIT_OFFROAD_BEHAVIOR,
        termination_mode=OFFLINE_FIT_TERMINATION_MODE,
        dt=OFFLINE_FIT_DT,
        episode_length=OFFLINE_FIT_EPISODE_LENGTH,
        dynamics_model=0,
        force_zero_trailer_articulation_at_init=0,
    )


def _make_env_handle_code(map_dir_expr: str, control_mode: int = OFFLINE_FIT_CONTROL_MODE, indent: str = "") -> str:
    code = f"""
obs = np.zeros((1, obs_dim), dtype=np.float32)
actions_buf = np.zeros(1, dtype=np.int32)
rewards = np.zeros(1, dtype=np.float32)
terminals = np.zeros(1, dtype=np.uint8)
truncs = np.zeros(1, dtype=np.uint8)
env_handle = binding.env_init(
    obs,
    actions_buf,
    rewards,
    terminals,
    truncs,
    0,
    human_agent_idx=0,
    ini_file="pufferlib/config/ocean/drive.ini",
    map_dir=str({map_dir_expr}),
    map_id=0,
    max_agents=1,
    max_controlled_agents=1,
    init_steps=0,
    init_mode=0,
    control_mode={control_mode},
    goal_behavior={OFFLINE_FIT_GOAL_BEHAVIOR},
    goal_target_distance={OFFLINE_FIT_GOAL_TARGET_DISTANCE},
    goal_radius={OFFLINE_FIT_GOAL_RADIUS},
    goal_speed={OFFLINE_FIT_GOAL_SPEED},
    collision_behavior={OFFLINE_FIT_COLLISION_BEHAVIOR},
    offroad_behavior={OFFLINE_FIT_OFFROAD_BEHAVIOR},
    termination_mode={OFFLINE_FIT_TERMINATION_MODE},
    dt={OFFLINE_FIT_DT},
    episode_length={OFFLINE_FIT_EPISODE_LENGTH},
    dynamics_model=0,
    force_zero_trailer_articulation_at_init=0,
)
"""
    if not indent:
        return code
    return "".join(f"{indent}{line}" if line else line for line in code.splitlines(keepends=True))


def _run_offline_fitting_smoke(map_dir_str: str):
    map_dir = Path(map_dir_str)
    env_handle = _make_env_handle(map_dir)
    try:
        binding.env_reset(env_handle, 0)

        active_count = binding.env_get_active_agent_count(env_handle)
        assert active_count == 1

        scenario_ids = np.zeros(active_count, dtype=np.int32)
        agent_ids = np.zeros(active_count, dtype=np.int32)
        binding.env_get_active_agent_info(env_handle, scenario_ids, agent_ids)
        assert int(agent_ids[0]) == 10
        assert int(scenario_ids[0]) == 321

        obs_copy = np.zeros((active_count, _make_obs_dim()), dtype=np.float32)
        binding.env_set_logged_timestep(env_handle, 1)
        binding.env_copy_observations(env_handle, obs_copy)
        assert obs_copy.shape == (1, _make_obs_dim())
        assert np.isfinite(obs_copy).all()

        actions = np.full(5, -1, dtype=np.int32)
        step_costs = np.zeros(5, dtype=np.float32)
        step_lat_costs = np.zeros(5, dtype=np.float32)
        step_lon_costs = np.zeros(5, dtype=np.float32)
        num_steps, total_cost, total_lat_cost, total_lon_cost = binding.env_fit_discrete_action_sequence(
            env_handle,
            0,
            FIT_BEAM_WIDTH,
            FIT_PLANNING_HORIZON,
            FIT_MATCH_WEIGHT_LATERAL,
            FIT_MATCH_WEIGHT_LONGITUDINAL,
            FIT_MATCH_WEIGHT_HEADING,
            FIT_MATCH_WEIGHT_SPEED,
            FIT_MATCH_WEIGHT_STEER_CHANGE,
            FIT_MATCH_WEIGHT_ACCEL_CHANGE,
            FIT_MATCH_WEIGHT_REVERSE,
            FIT_MATCH_WEIGHT_PROGRESS,
            FIT_MATCH_WEIGHT_STEER_FLIP,
            FIT_MATCH_WEIGHT_REF_ACCEL,
            FIT_MATCH_WEIGHT_REF_STEER,
            actions,
            step_costs,
            step_lat_costs,
            step_lon_costs,
        )

        assert num_steps == 5
        assert np.all(actions[:num_steps] >= 0)
        assert np.isfinite(step_costs[:num_steps]).all()
        assert np.isfinite(step_lat_costs[:num_steps]).all()
        assert np.isfinite(step_lon_costs[:num_steps]).all()
        assert np.isfinite(total_cost)
        assert np.isfinite(total_lat_cost)
        assert np.isfinite(total_lon_cost)
    finally:
        binding.env_close(env_handle)


def test_offline_fitting_bindings_smoke(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_offline_fit_map(map_dir)

    code = f"""
from pathlib import Path
import numpy as np
from pufferlib.ocean.drive.drive import binding

map_dir = Path({str(map_dir)!r})
obs_dim = binding.EGO_FEATURES_CLASSIC + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
obs = np.zeros((1, obs_dim), dtype=np.float32)
actions = np.zeros(1, dtype=np.int32)
rewards = np.zeros(1, dtype=np.float32)
terminals = np.zeros(1, dtype=np.uint8)
truncations = np.zeros(1, dtype=np.uint8)
env_handle = binding.env_init(
    obs,
    actions,
    rewards,
    terminals,
    truncations,
    0,
    human_agent_idx=0,
    ini_file="pufferlib/config/ocean/drive.ini",
    map_dir=str(map_dir),
    map_id=0,
    max_agents=1,
    max_controlled_agents=1,
    init_steps=0,
    init_mode=0,
    control_mode=0,
    goal_behavior=0,
    goal_target_distance=30.0,
    goal_radius=0.2,
    goal_speed=100.0,
    dynamics_model=0,
    force_zero_trailer_articulation_at_init=0,
)
try:
    binding.env_reset(env_handle, 0)
    active_count = binding.env_get_active_agent_count(env_handle)
    assert active_count == 1
    scenario_ids = np.zeros(active_count, dtype=np.int32)
    agent_ids = np.zeros(active_count, dtype=np.int32)
    binding.env_get_active_agent_info(env_handle, scenario_ids, agent_ids)
    assert int(agent_ids[0]) == 10
    assert int(scenario_ids[0]) == 321
    obs_copy = np.zeros((active_count, obs_dim), dtype=np.float32)
    binding.env_set_logged_timestep(env_handle, 1)
    binding.env_copy_observations(env_handle, obs_copy)
    assert np.isfinite(obs_copy).all()
    fit_actions = np.full(5, -1, dtype=np.int32)
    step_costs = np.zeros(5, dtype=np.float32)
    step_lat_costs = np.zeros(5, dtype=np.float32)
    step_lon_costs = np.zeros(5, dtype=np.float32)
    num_steps, total_cost, total_lat_cost, total_lon_cost = binding.env_fit_discrete_action_sequence(
        env_handle,
        0,
        8,
        5,
        2.5,
        1.5,
        0.1,
        0.02,
        0.15,
        0.02,
        1.0,
        4.0,
        0.5,
        0.01,
        0.1,
        fit_actions,
        step_costs,
        step_lat_costs,
        step_lon_costs,
    )
    assert num_steps == 5
    assert np.all(fit_actions[:num_steps] >= 0)
    assert np.isfinite(step_costs[:num_steps]).all()
    assert np.isfinite(step_lat_costs[:num_steps]).all()
    assert np.isfinite(step_lon_costs[:num_steps]).all()
    assert np.isfinite(total_cost)
    assert np.isfinite(total_lat_cost)
    assert np.isfinite(total_lon_cost)
finally:
    binding.env_close(env_handle)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_offline_fitting_visual_sanity_plots(tmp_path):
    shared_maps = sorted(set(p.name for p in CAR_BOSTON_TEST_DIR.glob("map_*.bin")) & set(p.name for p in TRUCK_BOSTON_TEST_DIR.glob("map_*.bin")))
    if not shared_maps:
        pytest.skip("Boston car/truck test bins not available for visual sanity plots")

    output_dir = VISUALIZATION_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    code = f"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.export_paired_offline_fits import _discover_shared_maps, _fit_side, _pair_metrics

CAR_ROOT = Path({str(CAR_BOSTON_TEST_DIR)!r})
TRUCK_ROOT = Path({str(TRUCK_BOSTON_TEST_DIR)!r})
OUTPUT_DIR = Path({str(output_dir)!r})
SHARED_MAPS = _discover_shared_maps(CAR_ROOT, TRUCK_ROOT)

case_results = {{"car": {{}}, "truck": {{}}}}

for map_name in SHARED_MAPS:
    for label, source_root in (("car", CAR_ROOT), ("truck", TRUCK_ROOT)):
        source_bin = source_root / map_name
        plot_path = OUTPUT_DIR / f"{{label}}_{{Path(map_name).stem}}_fit.png"
        result = _fit_side(source_bin)
        if result["status"] == "ok":
            fig, ax = plt.subplots(figsize=(6, 6))
            gt_x = result["gt_x"][: len(result["rollout_x"])]
            gt_y = result["gt_y"][: len(result["rollout_y"])]
            ax.plot(gt_x, gt_y, label="GT trajectory", linewidth=2.5)
            ax.plot(result["rollout_x"], result["rollout_y"], label="Fitted-action rollout", linewidth=2.0, linestyle="--")
            ax.scatter(gt_x[0], gt_y[0], label="Start", s=30)
            ax.set_title(
                f"{{label}} {{Path(map_name).stem}}\\nsteps={{result['num_steps']}} total_cost={{result['match_cost_total']:.3f}} mean_disp={{result['self_ade']:.3f}} final_disp={{result['self_fde']:.3f}}"
            )
            ax.set_aspect("equal", adjustable="box")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=160)
            plt.close(fig)
            case_results[label][map_name] = result
            print(f"saved_plot={{plot_path}}")
        else:
            case_results[label][map_name] = {{
                "status": "unavailable",
                "error": result["error"],
            }}

fig, axes = plt.subplots(len(SHARED_MAPS), 2, figsize=(12, max(4, 3.5 * len(SHARED_MAPS))))
if len(SHARED_MAPS) == 1:
    axes = np.asarray([axes])
for row_idx, map_name in enumerate(SHARED_MAPS):
    shared_limits = None
    row_results = [case_results[label][map_name] for label in ("car", "truck")]
    ok_row_results = [result for result in row_results if result["status"] == "ok"]
    pair_metrics = _pair_metrics(case_results["car"][map_name], case_results["truck"][map_name])
    if ok_row_results:
        xs = np.concatenate([np.concatenate((result["gt_x"], result["rollout_x"])) for result in ok_row_results])
        ys = np.concatenate([np.concatenate((result["gt_y"], result["rollout_y"])) for result in ok_row_results])
        x_min = float(xs.min())
        x_max = float(xs.max())
        y_min = float(ys.min())
        y_max = float(ys.max())
        x_pad = max(0.5, 0.08 * max(1e-6, x_max - x_min))
        y_pad = max(0.5, 0.08 * max(1e-6, y_max - y_min))
        shared_limits = (
            x_min - x_pad,
            x_max + x_pad,
            y_min - y_pad,
            y_max + y_pad,
        )
    for col_idx, label in enumerate(("car", "truck")):
        ax = axes[row_idx, col_idx]
        result = case_results[label][map_name]
        if result["status"] == "ok":
            car_result = case_results["car"][map_name]
            truck_result = case_results["truck"][map_name]
            if car_result["status"] == "ok":
                ax.plot(car_result["gt_x"], car_result["gt_y"], label="Car GT", linewidth=1.9, color="tab:blue")
                ax.scatter(car_result["gt_x"][0], car_result["gt_y"][0], s=14, color="tab:blue")
            if truck_result["status"] == "ok":
                ax.plot(truck_result["gt_x"], truck_result["gt_y"], label="Truck GT", linewidth=1.9, color="tab:orange")
                ax.scatter(truck_result["gt_x"][0], truck_result["gt_y"][0], s=14, color="tab:orange")
            ax.plot(result["rollout_x"], result["rollout_y"], label=f"{{label.title()}} rollout", linewidth=1.8, linestyle="--", color="tab:green")
            if shared_limits is not None:
                ax.set_xlim(shared_limits[0], shared_limits[1])
                ax.set_ylim(shared_limits[2], shared_limits[3])
            pair_line = ""
            if pair_metrics["status"] == "ok":
                pair_line = (
                    f"\\npair_ade={{pair_metrics['ade']:.2f}} "
                    f"pair_fde={{pair_metrics['fde']:.2f}} "
                    f"pair_n={{pair_metrics['aligned_steps']}}"
                )
            ax.set_title(
                f"{{label}} {{Path(map_name).stem}}\\nsteps={{result['num_steps']}} cost={{result['match_cost_total']:.2f}} mean={{result['self_ade']:.2f}} final={{result['self_fde']:.2f}}{{pair_line}}",
                fontsize=9,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, f"unavailable\\n{{result['error']}}", ha="center", va="center", fontsize=8, wrap=True)
            ax.set_title(f"{{label}} {{Path(map_name).stem}}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
fig.tight_layout()
combined_path = OUTPUT_DIR / "all_examples_fit_grid.png"
fig.savefig(combined_path, dpi=180)
plt.close(fig)
print(f"saved_plot={{combined_path}}")
ok_count = sum(
    1
    for label_results in case_results.values()
    for result in label_results.values()
    if result["status"] == "ok"
)
print(f"ok_count={{ok_count}} total={{len(SHARED_MAPS) * 2}}")
if ok_count == 0:
    raise RuntimeError("No fit visualizations were generated successfully")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    assert (output_dir / "car_map_000_fit.png").exists()
    assert (output_dir / "truck_map_000_fit.png").exists()
    assert (output_dir / "all_examples_fit_grid.png").exists()
