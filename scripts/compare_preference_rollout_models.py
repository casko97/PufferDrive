#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import configparser
import copy
import csv
import json
import random
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import gymnasium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib.preference_reward import PreferenceRewardManager, combine_preference_rewards
from scripts.analyze_human_replay_turning_buckets import classify_map_turning

DEFAULT_CAR_MODEL_ROOT = Path("pufferlib/resources/drive/models/car-baseline-full-boston")
DEFAULT_TRUCK_MODEL_ROOT = Path("pufferlib/resources/drive/models/truck-baseline-full-boston")
DEFAULT_PREFERENCE_CONFIG = Path("pufferlib/resources/drive/models/car-baseline-full-boston-pref-from-base/conf1_preference.ini")
DEFAULT_REWARD_DIR = Path("pufferlib/resources/drive/preferences/models/turning_all_90_10_30rounds_continue1")
DEFAULT_OUTPUT_DIR = Path("outputs/preference_rollout_model_compare")
DEFAULT_DRIVE_CONFIG = Path("pufferlib/config/ocean/drive.ini")
DEFAULT_THRESHOLD_DEG = 45.0
DEFAULT_TURNING_COUNT = 7
DEFAULT_STRAIGHT_COUNT = 3
DEFAULT_SAMPLE_SEED = 42
DEFAULT_DEVICE = "cpu"
DEFAULT_RESET_JUMP_THRESHOLD_M = 20.0

ROLLOUT_FORMAT_VERSION = "drive_preference_rollouts_v1"
SCORED_FORMAT_VERSION = "drive_preference_scored_rollouts_v1"

PREFERRED_CONFIG_NAMES = (
    "conf1_preference.ini",
    "conf1_discrete_preference_reward_turning_prebuilt_lowmem_obs_default.ini",
    "conf1_discrete_preference_reward_turning_prebuilt_lowmem.ini",
    "conf1_discrete_preference_reward_turning_prebuilt.ini",
    "conf.ini",
)


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_root: Path
    config_path: Path
    checkpoint_path: Path


def _parse_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _load_packaged_config(config_path: Path) -> dict[str, dict[str, Any]]:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    with config_path.open("r", encoding="utf-8") as file_obj:
        parser.read_file(file_obj)

    data: dict[str, dict[str, Any]] = {}
    for section in parser.sections():
        data[section] = {}
        for key, value in parser[section].items():
            data[section][key] = _parse_value(value)
    return data


def _resolve_path(path_str: str, *, fallback_root: Path) -> Path:
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    repo_candidate = (REPO_ROOT / raw).resolve()
    if repo_candidate.exists():
        return repo_candidate
    fallback_candidate = (fallback_root / raw).resolve()
    return fallback_candidate


def _discover_model_config(model_root: Path, config_override: Path | None) -> Path:
    if config_override is not None:
        return config_override.resolve()

    root_level = {path.name: path for path in model_root.glob("*.ini")}
    for name in PREFERRED_CONFIG_NAMES:
        path = root_level.get(name)
        if path is not None:
            return path.resolve()

    if len(root_level) == 1:
        return next(iter(root_level.values())).resolve()

    raise FileNotFoundError(f"Could not infer packaged config under {model_root}")


def _discover_checkpoint(model_root: Path, packaged: dict[str, dict[str, Any]]) -> Path:
    load_model_path = packaged.get("base", {}).get("load_model_path")
    if load_model_path:
        candidate = _resolve_path(str(load_model_path), fallback_root=model_root)
        if candidate.exists():
            return candidate

    root_pts = sorted(model_root.glob("*.pt"))
    if len(root_pts) == 1:
        return root_pts[0].resolve()

    nested_model_pts = sorted(model_root.glob("**/model_puffer_drive_*.pt"))
    if nested_model_pts:
        return nested_model_pts[-1].resolve()

    root_model_pts = sorted(model_root.glob("puffer_drive_*.pt"))
    if root_model_pts:
        return root_model_pts[-1].resolve()

    nested_pts = sorted(model_root.glob("**/*.pt"))
    if len(nested_pts) == 1:
        return nested_pts[0].resolve()

    raise FileNotFoundError(f"Could not infer checkpoint under {model_root}")


def discover_model_spec(model_root: Path, config_override: Path | None = None) -> ModelSpec:
    resolved_root = model_root.resolve()
    config_path = _discover_model_config(resolved_root, config_override)
    packaged = _load_packaged_config(config_path)
    checkpoint_path = _discover_checkpoint(resolved_root, packaged)
    return ModelSpec(
        model_name=resolved_root.name,
        model_root=resolved_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )


def _infer_source_map_dir(car_spec: ModelSpec, truck_spec: ModelSpec) -> Path:
    for spec in (car_spec, truck_spec):
        packaged = _load_packaged_config(spec.config_path)
        map_dir = packaged.get("env", {}).get("map_dir")
        if map_dir:
            candidate = _resolve_path(str(map_dir), fallback_root=spec.model_root)
            if candidate.exists():
                return candidate
    raise FileNotFoundError("Could not infer a source map directory from the model configs")


def _select_map_paths(
    map_paths: list[Path],
    sample_size: int | None,
    sample_seed: int,
) -> list[Path]:
    if sample_size is None:
        return map_paths
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if sample_size >= len(map_paths):
        return map_paths
    rng = random.Random(sample_seed)
    selected = rng.sample(map_paths, sample_size)
    return sorted(selected)


def build_stratified_sample_manifest(
    *,
    source_map_dir: Path,
    output_path: Path,
    turning_count: int,
    straight_count: int,
    threshold_deg: float,
    sample_seed: int,
    validity_checker: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    map_paths = sorted(source_map_dir.glob("map_*.bin"))
    if not map_paths:
        raise FileNotFoundError(f"No maps found under {source_map_dir}")

    turning_rows: list[dict[str, Any]] = []
    straight_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for map_path in map_paths:
        classified = classify_map_turning(map_path, threshold_deg)
        row = {
            "map_name": map_path.name,
            "map_path": str(map_path.parent.resolve() / map_path.name),
            "resolved_map_path": str(map_path.resolve()),
            "bucket": classified["bucket"],
            "delta_heading_deg": float(classified["delta_heading_deg"]),
        }
        if validity_checker is not None:
            validity = validity_checker(map_path)
            row.update(
                {
                    "valid_for_sampling": bool(validity.get("valid_for_sampling", False)),
                    "active_agent_count": int(validity.get("active_agent_count", 0)),
                    "invalid_initial_trailer_state": bool(validity.get("invalid_initial_trailer_state", False)),
                }
            )
            if not row["valid_for_sampling"] or row["active_agent_count"] <= 0:
                invalid_rows.append(row)
                continue
        if row["bucket"] == "turning":
            turning_rows.append(row)
        else:
            straight_rows.append(row)

    if len(turning_rows) < turning_count:
        qualifier = " valid" if validity_checker is not None else ""
        raise ValueError(f"Requested {turning_count} turning maps, but only found {len(turning_rows)}{qualifier}")
    if len(straight_rows) < straight_count:
        qualifier = " valid" if validity_checker is not None else ""
        raise ValueError(f"Requested {straight_count} straight maps, but only found {len(straight_rows)}{qualifier}")

    selected_turning = _select_map_paths(
        [Path(row["map_path"]) for row in turning_rows],
        turning_count,
        sample_seed,
    )
    selected_straight = _select_map_paths(
        [Path(row["map_path"]) for row in straight_rows],
        straight_count,
        sample_seed,
    )
    selected = sorted(selected_turning + selected_straight, key=lambda path: path.name)

    manifest = {
        "source_map_dir": str(source_map_dir.resolve()),
        "threshold_deg": float(threshold_deg),
        "sample_seed": int(sample_seed),
        "turning_count": int(turning_count),
        "straight_count": int(straight_count),
        "selected_count": int(len(selected)),
        "excluded_invalid_count": int(len(invalid_rows)),
        "maps": [
            {
                "map_name": path.name,
                "map_path": str(path),
                "scenario_type": classify_map_turning(path, threshold_deg)["bucket"],
                "delta_heading_deg": float(classify_map_turning(path, threshold_deg)["delta_heading_deg"]),
            }
            for path in selected
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _prepare_single_map_dir(map_path: Path, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = temp_dir / "map_000.bin"
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(map_path.resolve(), target)
    return temp_dir


def _detect_rollout_reset_reason(
    current_agent_state: dict[str, Any],
    next_agent_state: dict[str, Any],
    *,
    position_jump_threshold_m: float = DEFAULT_RESET_JUMP_THRESHOLD_M,
) -> str | None:
    current_ids = np.asarray(current_agent_state.get("id", []), dtype=np.int64).reshape(-1)
    next_ids = np.asarray(next_agent_state.get("id", []), dtype=np.int64).reshape(-1)
    if current_ids.size and next_ids.size and int(current_ids[0]) != int(next_ids[0]):
        return "agent_id_changed"

    current_x = np.asarray(current_agent_state.get("x", []), dtype=np.float32).reshape(-1)
    current_y = np.asarray(current_agent_state.get("y", []), dtype=np.float32).reshape(-1)
    next_x = np.asarray(next_agent_state.get("x", []), dtype=np.float32).reshape(-1)
    next_y = np.asarray(next_agent_state.get("y", []), dtype=np.float32).reshape(-1)
    if current_x.size and current_y.size and next_x.size and next_y.size:
        jump_m = float(np.hypot(next_x[0] - current_x[0], next_y[0] - current_y[0]))
        if jump_m > float(position_jump_threshold_m):
            return f"position_jump>{position_jump_threshold_m:g}m"

    return None


@contextmanager
def swap_default_drive_config(source_config: Path, drive_config_path: Path = DEFAULT_DRIVE_CONFIG) -> Iterator[dict[str, str]]:
    drive_config_path = drive_config_path.resolve()
    source_config = source_config.resolve()
    backup_fd, backup_name = tempfile.mkstemp(prefix="drive_ini_backup_", suffix=".ini", dir=drive_config_path.parent)
    os.close(backup_fd)
    backup_path = Path(backup_name)
    shutil.copy2(drive_config_path, backup_path)
    shutil.copy2(source_config, drive_config_path)
    try:
        yield {
            "active_config_path": str(drive_config_path),
            "backup_path": str(backup_path),
            "source_config_path": str(source_config),
        }
    finally:
        shutil.copy2(backup_path, drive_config_path)
        backup_path.unlink(missing_ok=True)


def _common_runtime_overrides(sampled_count: int, map_dir: Path, device: str) -> dict[str, Any]:
    return {
        "load_id": None,
        "wandb": False,
        "neptune": False,
        "train": {
            "device": device,
            "render": False,
            "obs_only": True,
            "torch_deterministic": True,
        },
        "vec": {
            "backend": "PufferEnv",
            "num_envs": 1,
        },
        "env": {
            "num_agents": 1,
            "action_type": "discrete",
            "dynamics_model": "articulated",
            "reward_vehicle_collision": -0.5,
            "reward_offroad_collision": -0.5,
            "dt": 0.1,
            "reward_goal": 1.0,
            "reward_goal_post_respawn": 0.25,
            "goal_radius": 2.0,
            "goal_speed": 100.0,
            "goal_behavior": 0,
            "goal_target_distance": 30.0,
            "collision_behavior": 0,
            "offroad_behavior": 0,
            "episode_length": 91,
            "resample_frequency": 910,
            "termination_mode": 1,
            "map_dir": str(map_dir),
            "num_maps": int(sampled_count),
            "init_steps": 0,
            "control_mode": "control_sdc_only",
            "observation_mode": "default",
            "sdc_runtime_truck_override": True,
            "init_mode": "create_all_valid",
        },
        "eval": {
            "map_dir": str(map_dir),
            "wosac_num_maps": int(sampled_count),
            "human_replay_eval": False,
        },
    }


def _build_map_validity_checker() -> Callable[[Path], dict[str, Any]]:
    from pufferlib.ocean.drive import binding
    from pufferlib.ocean.drive.drive import (
        DEFAULT_SDC_RUNTIME_TRUCK_REF_BIN,
        _DYNAMICS_MODEL_IDS,
        _load_non_kinematic_vehicle_params_from_bin,
    )

    non_kinematic_vehicle_params_override = _load_non_kinematic_vehicle_params_from_bin(DEFAULT_SDC_RUNTIME_TRUCK_REF_BIN)
    inspect_kwargs = {
        "dynamics_model": _DYNAMICS_MODEL_IDS["articulated"],
        "init_mode": 0,
        "control_mode": 3,
        "init_steps": 0,
        "max_controlled_agents": -1,
        "goal_behavior": 0,
        "goal_target_distance": 30.0,
        "non_kinematic_vehicle_params_override": non_kinematic_vehicle_params_override,
        "force_zero_trailer_articulation_at_init": 1,
    }

    def _check(map_path: Path) -> dict[str, Any]:
        metadata = binding.inspect_map(map_path=str(map_path.resolve()), **inspect_kwargs)
        return {
            "active_agent_count": int(metadata.get("active_agent_count", 0)),
            "valid_for_sampling": bool(metadata.get("valid_for_sampling", False)),
            "invalid_initial_trailer_state": bool(metadata.get("invalid_initial_trailer_state", False)),
        }

    return _check


def _build_runtime_args(checkpoint_path: Path, map_dir: Path, sampled_count: int, device: str) -> dict[str, Any]:
    from pufferlib import pufferl

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = pufferl.load_config("puffer_drive")
    finally:
        sys.argv = original_argv

    overrides = _common_runtime_overrides(sampled_count=sampled_count, map_dir=map_dir, device=device)
    for key, value in overrides.items():
        if isinstance(value, dict):
            args.setdefault(key, {})
            args[key].update(value)
        else:
            args[key] = value
    args["load_model_path"] = str(checkpoint_path.resolve())
    return args


def _bootstrap_policy(args: dict[str, Any], first_map_path: Path, workspace_dir: Path):
    from pufferlib import pufferl

    bootstrap_dir = _prepare_single_map_dir(first_map_path, workspace_dir / "bootstrap")
    bootstrap_args = copy.deepcopy(args)
    bootstrap_args["env"]["map_dir"] = str(bootstrap_dir)
    bootstrap_args["env"]["num_maps"] = 1
    bootstrap_args["eval"]["map_dir"] = str(bootstrap_dir)
    bootstrap_args["eval"]["wosac_num_maps"] = 1
    vecenv = pufferl.load_env("puffer_drive", bootstrap_args)
    try:
        policy = pufferl.load_policy(bootstrap_args, vecenv, env_name="puffer_drive")
        policy.eval()
        return policy
    finally:
        vecenv.close()


def _collect_single_rollout(args: dict[str, Any], policy, map_entry: dict[str, Any], workspace_dir: Path) -> dict[str, Any]:
    import pufferlib

    from pufferlib import pufferl

    map_path = Path(map_entry["map_path"]).resolve()
    scenario_dir = _prepare_single_map_dir(map_path, workspace_dir / map_path.stem)
    rollout_args = copy.deepcopy(args)
    rollout_args["env"]["map_dir"] = str(scenario_dir)
    rollout_args["env"]["num_maps"] = 1
    rollout_args["eval"]["map_dir"] = str(scenario_dir)
    rollout_args["eval"]["wosac_num_maps"] = 1
    vecenv = pufferl.load_env("puffer_drive", rollout_args)
    try:
        obs, _ = vecenv.reset(seed=int(rollout_args["train"]["seed"]))
        num_agents = vecenv.observation_space.shape[0]
        device = rollout_args["train"]["device"]
        state = {}
        if rollout_args["train"]["use_rnn"]:
            state = {
                "lstm_h": torch.zeros(num_agents, policy.hidden_size, device=device),
                "lstm_c": torch.zeros(num_agents, policy.hidden_size, device=device),
            }

        observations: list[np.ndarray] = []
        actions: list[int] = []
        task_rewards: list[float] = []
        dones: list[bool] = []
        truncs: list[bool] = []
        values: list[float] = []
        entropy_trace: list[float] = []
        x: list[float] = []
        y: list[float] = []
        z: list[float] = []
        heading: list[float] = []
        length: list[float] = []
        width: list[float] = []
        trailer_x: list[float] = []
        trailer_y: list[float] = []
        trailer_heading: list[float] = []
        trailer_has: list[int] = []
        stop_reason = "episode_length"

        episode_steps = int(rollout_args["env"]["episode_length"]) - int(rollout_args["env"]["init_steps"])
        for _step_idx in range(episode_steps):
            agent_state = vecenv.driver_env.get_global_agent_state(include_sdc_trailer=True)
            trailer_state = agent_state.get("sdc_trailer", {})

            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, value = policy.forward_eval(ob_tensor, state)
                action, _logprob, sampled_entropy = pufferlib.pytorch.sample_logits(logits)
                action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
                if isinstance(logits, torch.distributions.Normal):
                    action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)
                entropy_value = float(np.mean(np.asarray(sampled_entropy.detach().cpu().numpy()).reshape(-1)))

            observations.append(np.asarray(obs[0], dtype=np.float32).copy())
            actions.append(int(np.asarray(action_np).reshape(-1)[0]))
            values.append(float(np.asarray(value.detach().cpu().numpy()).reshape(-1)[0]))
            entropy_trace.append(entropy_value)
            x.append(float(agent_state["x"][0]))
            y.append(float(agent_state["y"][0]))
            z.append(float(agent_state["z"][0]))
            heading.append(float(agent_state["heading"][0]))
            length.append(float(agent_state["length"][0]))
            width.append(float(agent_state["width"][0]))
            trailer_has.append(int(trailer_state.get("has_trailer", np.zeros(1, dtype=np.int32))[0]) if trailer_state else 0)
            trailer_x.append(float(trailer_state.get("x", np.zeros(1, dtype=np.float32))[0]) if trailer_state else 0.0)
            trailer_y.append(float(trailer_state.get("y", np.zeros(1, dtype=np.float32))[0]) if trailer_state else 0.0)
            trailer_heading.append(
                float(trailer_state.get("heading", np.zeros(1, dtype=np.float32))[0]) if trailer_state else 0.0
            )

            obs, rewards, done_flags, trunc_flags, _info = vecenv.step(action_np)
            task_rewards.append(float(np.asarray(rewards).reshape(-1)[0]))
            dones.append(bool(np.asarray(done_flags).reshape(-1)[0]))
            truncs.append(bool(np.asarray(trunc_flags).reshape(-1)[0]))
            if dones[-1] or truncs[-1]:
                stop_reason = "terminal" if dones[-1] else "truncation"
                break

            next_agent_state = vecenv.driver_env.get_global_agent_state(include_sdc_trailer=True)
            reset_reason = _detect_rollout_reset_reason(agent_state, next_agent_state)
            if reset_reason is not None:
                stop_reason = reset_reason
                break

        return {
            "map_name": map_entry["map_name"],
            "map_path": str(map_path),
            "scenario_type": map_entry["scenario_type"],
            "delta_heading_deg": float(map_entry["delta_heading_deg"]),
            "steps": len(actions),
            "stop_reason": stop_reason,
            "observations": np.stack(observations).astype(np.float32) if observations else np.zeros((0, 0), dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "task_rewards": np.asarray(task_rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "truncs": np.asarray(truncs, dtype=bool),
            "values": np.asarray(values, dtype=np.float32),
            "entropy": np.asarray(entropy_trace, dtype=np.float32),
            "x": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
            "z": np.asarray(z, dtype=np.float32),
            "heading": np.asarray(heading, dtype=np.float32),
            "length": np.asarray(length, dtype=np.float32),
            "width": np.asarray(width, dtype=np.float32),
            "trailer_has": np.asarray(trailer_has, dtype=np.int32),
            "trailer_x": np.asarray(trailer_x, dtype=np.float32),
            "trailer_y": np.asarray(trailer_y, dtype=np.float32),
            "trailer_heading": np.asarray(trailer_heading, dtype=np.float32),
        }
    finally:
        vecenv.close()


def collect_rollouts_for_model(
    *,
    model_spec: ModelSpec,
    sample_manifest: dict[str, Any],
    output_path: Path,
    device: str,
    drive_config_path: Path = DEFAULT_DRIVE_CONFIG,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_maps = sample_manifest["maps"]
    if not sampled_maps:
        raise ValueError("sample manifest contains no maps")

    with swap_default_drive_config(model_spec.config_path, drive_config_path=drive_config_path) as swap_info:
        runtime_args = _build_runtime_args(
            checkpoint_path=model_spec.checkpoint_path,
            map_dir=Path(sample_manifest["source_map_dir"]),
            sampled_count=len(sampled_maps),
            device=device,
        )
        workspace_dir = output_path.parent / f".{model_spec.model_name}_workspace"
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        try:
            policy = _bootstrap_policy(runtime_args, Path(sampled_maps[0]["map_path"]), workspace_dir)
            rollouts = []
            for map_entry in sampled_maps:
                rollouts.append(_collect_single_rollout(runtime_args, policy, map_entry, workspace_dir))
        finally:
            if workspace_dir.exists():
                shutil.rmtree(workspace_dir)

    payload = {
        "format": ROLLOUT_FORMAT_VERSION,
        "model_name": model_spec.model_name,
        "model_root": str(model_spec.model_root),
        "config_path": str(model_spec.config_path),
        "checkpoint_path": str(model_spec.checkpoint_path),
        "drive_config_swap": swap_info,
        "sample_manifest_path": str((output_path.parent.parent / "sample_manifest.json").resolve()),
        "rollouts": rollouts,
    }
    torch.save(payload, output_path)
    return output_path


def _reward_runtime_config(preference_config_path: Path) -> dict[str, Any]:
    packaged = _load_packaged_config(preference_config_path)
    cfg = dict(packaged.get("preference_reward", {}))
    cfg["enabled"] = True
    return cfg


def _load_rollout_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != ROLLOUT_FORMAT_VERSION:
        raise ValueError(f"Unsupported rollout format at {path}")
    return payload


def _flatten_rollout_steps(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for payload in payloads:
        model_name = payload["model_name"]
        for rollout in payload["rollouts"]:
            for step_idx in range(int(rollout["steps"])):
                steps.append(
                    {
                        "model_name": model_name,
                        "map_name": rollout["map_name"],
                        "scenario_type": rollout["scenario_type"],
                        "step_idx": int(step_idx),
                        "task_reward": float(rollout["task_rewards"][step_idx]),
                        "obs": np.asarray(rollout["observations"][step_idx], dtype=np.float32),
                        "action": int(rollout["actions"][step_idx]),
                    }
                )
    steps.sort(key=lambda item: (item["model_name"], item["map_name"], item["step_idx"]))
    return steps


def _compute_offline_normalization(pref_raw_values: np.ndarray, reward_cfg: dict[str, Any]) -> dict[str, Any]:
    normalize_mode = str(reward_cfg.get("normalize_mode", "none"))
    calibration_steps = int(reward_cfg.get("calibration_steps", 0))
    if normalize_mode == "none":
        return {
            "mode": "none",
            "mean": 0.0,
            "std": 1.0,
            "used_steps": 0,
            "requested_steps": calibration_steps,
        }

    effective_steps = min(calibration_steps, int(pref_raw_values.size))
    if effective_steps <= 0:
        return {
            "mode": "none_available",
            "mean": 0.0,
            "std": 1.0,
            "used_steps": 0,
            "requested_steps": calibration_steps,
        }

    calibration_values = np.asarray(pref_raw_values[:effective_steps], dtype=np.float64)
    mean = float(np.mean(calibration_values))
    std = float(max(np.std(calibration_values), 1e-6))
    return {
        "mode": "offline_sample" if effective_steps < calibration_steps else "runtime_equivalent_prefix",
        "mean": mean,
        "std": std,
        "used_steps": int(effective_steps),
        "requested_steps": int(calibration_steps),
    }


def score_rollout_payloads(
    *,
    rollout_paths: list[Path],
    preference_config_path: Path,
    reward_dir: Path,
    output_path: Path,
) -> Path:
    payloads = [_load_rollout_payload(path) for path in rollout_paths]
    reward_cfg = _reward_runtime_config(preference_config_path)
    reward_cfg["model_dir"] = str(reward_dir.resolve())
    reward_summary = json.loads((reward_dir / "offline_truck_context_reward_summary.json").read_text(encoding="utf-8"))
    manager = PreferenceRewardManager.from_config(
        reward_cfg,
        env_config={"observation_mode": "default", "action_type": "discrete"},
        observation_space=gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(reward_summary["obs_dim"]),),
            dtype=np.float32,
        ),
        action_space=gymnasium.spaces.MultiDiscrete([int(reward_summary["action_dim"])]),
        device="cpu",
    )
    if manager is None:
        raise RuntimeError("Preference reward manager failed to initialize")

    flattened = _flatten_rollout_steps(payloads)
    pref_mean_values = []
    pref_std_values = []
    pref_raw_values = []
    for item in flattened:
        pref_mean, pref_std, _members = manager.score(
            item["obs"].reshape(1, -1),
            np.asarray([item["action"]], dtype=np.int64),
        )
        pref_mean_values.append(float(pref_mean[0]))
        pref_std_values.append(float(pref_std[0]))
        pref_raw_values.append(float(pref_mean[0] - manager.lambda_uncertainty * pref_std[0]))

    pref_mean_array = np.asarray(pref_mean_values, dtype=np.float32)
    pref_std_array = np.asarray(pref_std_values, dtype=np.float32)
    pref_raw_array = np.asarray(pref_raw_values, dtype=np.float32)
    normalization = _compute_offline_normalization(pref_raw_array, reward_cfg)

    shaped_values = []
    combined_values = []
    applied_values = []
    for idx, item in enumerate(flattened):
        apply_shaping = idx >= int(reward_cfg.get("warmup_steps", 0))
        if apply_shaping:
            combined_reward, _pref_raw_unused, pref_shaped = combine_preference_rewards(
                np.asarray([item["task_reward"]], dtype=np.float32),
                np.asarray([pref_mean_array[idx]], dtype=np.float32),
                np.asarray([pref_std_array[idx]], dtype=np.float32),
                beta=float(reward_cfg.get("beta", 0.1)),
                lambda_uncertainty=float(reward_cfg.get("lambda_uncertainty", 0.0)),
                scale=float(reward_cfg.get("scale", 1.0)),
                clip_min=reward_cfg.get("clip_min"),
                clip_max=reward_cfg.get("clip_max"),
                normalize_mean=float(normalization["mean"]),
                normalize_std=float(normalization["std"]),
            )
            shaped_values.append(float(pref_shaped[0]))
            combined_values.append(float(combined_reward[0]))
        else:
            shaped_values.append(0.0)
            combined_values.append(float(item["task_reward"]))
        applied_values.append(bool(apply_shaping))

    metrics_by_key = {
        (item["model_name"], item["map_name"], item["step_idx"]): {
            "pref_mean": float(pref_mean_array[idx]),
            "pref_std": float(pref_std_array[idx]),
            "pref_raw": float(pref_raw_array[idx]),
            "pref_shaped": float(shaped_values[idx]),
            "combined_reward": float(combined_values[idx]),
            "pref_applied": bool(applied_values[idx]),
        }
        for idx, item in enumerate(flattened)
    }

    scored_payloads = []
    for payload in payloads:
        scored_rollouts = []
        for rollout in payload["rollouts"]:
            step_metrics = [metrics_by_key[(payload["model_name"], rollout["map_name"], idx)] for idx in range(int(rollout["steps"]))]
            scored_rollout = dict(rollout)
            scored_rollout["pref_mean"] = np.asarray([m["pref_mean"] for m in step_metrics], dtype=np.float32)
            scored_rollout["pref_std"] = np.asarray([m["pref_std"] for m in step_metrics], dtype=np.float32)
            scored_rollout["pref_raw"] = np.asarray([m["pref_raw"] for m in step_metrics], dtype=np.float32)
            scored_rollout["pref_shaped"] = np.asarray([m["pref_shaped"] for m in step_metrics], dtype=np.float32)
            scored_rollout["combined_reward"] = np.asarray([m["combined_reward"] for m in step_metrics], dtype=np.float32)
            scored_rollout["pref_applied"] = np.asarray([m["pref_applied"] for m in step_metrics], dtype=bool)
            scored_rollout["cumulative_pref_shaped"] = float(np.sum(scored_rollout["pref_shaped"]))
            scored_rollout["mean_pref_raw"] = float(np.mean(scored_rollout["pref_raw"])) if rollout["steps"] else 0.0
            scored_rollout["mean_pref_std"] = float(np.mean(scored_rollout["pref_std"])) if rollout["steps"] else 0.0
            scored_rollouts.append(scored_rollout)

        scored_payloads.append(
            {
                "format": SCORED_FORMAT_VERSION,
                "model_name": payload["model_name"],
                "model_root": payload["model_root"],
                "config_path": payload["config_path"],
                "checkpoint_path": payload["checkpoint_path"],
                "reward_dir": str(reward_dir.resolve()),
                "preference_config_path": str(preference_config_path.resolve()),
                "normalization": normalization,
                "rollouts": scored_rollouts,
            }
        )

    scored_payload = {
        "format": SCORED_FORMAT_VERSION,
        "reward_dir": str(reward_dir.resolve()),
        "preference_config_path": str(preference_config_path.resolve()),
        "normalization": normalization,
        "models": scored_payloads,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scored_payload, output_path)
    return output_path


def _iter_scored_rollouts(scored_payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for model_payload in scored_payload["models"]:
        for rollout in model_payload["rollouts"]:
            record = dict(rollout)
            record["model_name"] = model_payload["model_name"]
            yield record


def _save_step_csv(scored_payload: dict[str, Any], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "model_name",
                "map_name",
                "scenario_type",
                "step_idx",
                "task_reward",
                "pref_mean",
                "pref_std",
                "pref_raw",
                "pref_shaped",
                "combined_reward",
            ]
        )
        for rollout in _iter_scored_rollouts(scored_payload):
            for step_idx in range(int(rollout["steps"])):
                writer.writerow(
                    [
                        rollout["model_name"],
                        rollout["map_name"],
                        rollout["scenario_type"],
                        step_idx,
                        float(rollout["task_rewards"][step_idx]),
                        float(rollout["pref_mean"][step_idx]),
                        float(rollout["pref_std"][step_idx]),
                        float(rollout["pref_raw"][step_idx]),
                        float(rollout["pref_shaped"][step_idx]),
                        float(rollout["combined_reward"][step_idx]),
                    ]
                )
    return csv_path


def _save_rollout_summary_csv(scored_payload: dict[str, Any], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "model_name",
                "map_name",
                "scenario_type",
                "steps",
                "cumulative_pref_shaped",
                "mean_pref_raw",
                "mean_pref_std",
                "total_task_reward",
            ]
        )
        for rollout in _iter_scored_rollouts(scored_payload):
            writer.writerow(
                [
                    rollout["model_name"],
                    rollout["map_name"],
                    rollout["scenario_type"],
                    int(rollout["steps"]),
                    float(rollout["cumulative_pref_shaped"]),
                    float(rollout["mean_pref_raw"]),
                    float(rollout["mean_pref_std"]),
                    float(np.sum(rollout["task_rewards"])),
                ]
            )
    return csv_path


def _overall_arrays(scored_payload: dict[str, Any], key: str, *, scenario_type: str | None = None) -> dict[str, np.ndarray]:
    by_model: dict[str, list[np.ndarray]] = {}
    for rollout in _iter_scored_rollouts(scored_payload):
        if scenario_type is not None and rollout["scenario_type"] != scenario_type:
            continue
        by_model.setdefault(rollout["model_name"], []).append(np.asarray(rollout[key], dtype=np.float32))
    return {
        model_name: np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
        for model_name, values in by_model.items()
    }


def _plot_histogram_comparison(scored_payload: dict[str, Any], output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_specs = [
        ("pref_raw", None, "Overall raw preference reward"),
        ("pref_shaped", None, "Overall shaped preference reward"),
        ("pref_shaped", "turning", "Turning shaped preference reward"),
        ("pref_shaped", "straight", "Straight shaped preference reward"),
    ]
    for ax, (key, scenario_type, title) in zip(axes.flat, plot_specs):
        data = _overall_arrays(scored_payload, key, scenario_type=scenario_type)
        for model_name, values in sorted(data.items()):
            if values.size == 0:
                continue
            ax.hist(values, bins=20, alpha=0.5, label=model_name, density=True)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_cumulative_rewards(scored_payload: dict[str, Any], output_path: Path) -> Path:
    rollouts = list(_iter_scored_rollouts(scored_payload))
    model_names = sorted({rollout["model_name"] for rollout in rollouts})
    map_names = sorted({rollout["map_name"] for rollout in rollouts})
    x_positions = np.arange(len(map_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(map_names)), 6))
    for model_idx, model_name in enumerate(model_names):
        values = []
        for map_name in map_names:
            matched = next((rollout for rollout in rollouts if rollout["model_name"] == model_name and rollout["map_name"] == map_name), None)
            values.append(float(matched["cumulative_pref_shaped"]) if matched is not None else 0.0)
        ax.bar(x_positions + (model_idx - 0.5) * width, values, width=width, label=model_name)

    bucket_labels = {
        rollout["map_name"]: rollout["scenario_type"]
        for rollout in rollouts
    }
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{name}\n{bucket_labels.get(name, '')}" for name in map_names], rotation=45, ha="right")
    ax.set_ylabel("Cumulative shaped preference reward")
    ax.set_title("Per-map cumulative shaped preference reward")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_uncertainty(scored_payload: dict[str, Any], output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scenario_type in zip(axes, ("turning", "straight")):
        data = _overall_arrays(scored_payload, "pref_std", scenario_type=scenario_type)
        for model_name, values in sorted(data.items()):
            if values.size == 0:
                continue
            ax.hist(values, bins=20, alpha=0.5, label=model_name, density=True)
        ax.set_title(f"{scenario_type.title()} preference uncertainty")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _select_representative_maps(scored_payload: dict[str, Any], limit_per_side: int = 2) -> list[str]:
    rollouts = list(_iter_scored_rollouts(scored_payload))
    by_map: dict[str, dict[str, float]] = {}
    for rollout in rollouts:
        by_map.setdefault(rollout["map_name"], {})[rollout["model_name"]] = float(rollout["cumulative_pref_shaped"])
    if len(by_map) == 0:
        return []
    model_names = sorted({rollout["model_name"] for rollout in rollouts})
    if len(model_names) != 2:
        return sorted(by_map)[: 2 * limit_per_side]
    left, right = model_names
    ranked = sorted(
        (
            (scores.get(left, 0.0) - scores.get(right, 0.0), map_name)
            for map_name, scores in by_map.items()
        ),
        key=lambda item: item[0],
    )
    selected = ranked[:limit_per_side] + ranked[-limit_per_side:]
    return [map_name for _delta, map_name in selected]


def _plot_representative_gallery(scored_payload: dict[str, Any], output_path: Path) -> Path:
    rollouts = list(_iter_scored_rollouts(scored_payload))
    selected_maps = _select_representative_maps(scored_payload)
    if not selected_maps:
        raise ValueError("No representative maps available for plotting")

    rows = len(selected_maps)
    fig, axes = plt.subplots(rows, 2, figsize=(12, max(4, 4 * rows)))
    axes = np.asarray(axes, dtype=object).reshape(rows, 2)
    model_names = sorted({rollout["model_name"] for rollout in rollouts})
    colors = {model_names[0]: "tab:blue", model_names[1]: "tab:orange"} if len(model_names) >= 2 else {}

    for row_idx, map_name in enumerate(selected_maps):
        trajectory_ax = axes[row_idx, 0]
        trace_ax = axes[row_idx, 1]
        matched = [rollout for rollout in rollouts if rollout["map_name"] == map_name]
        for rollout in matched:
            color = colors.get(rollout["model_name"], None)
            trajectory_ax.plot(rollout["x"], rollout["y"], label=rollout["model_name"], linewidth=2.0, color=color)
            if int(rollout["steps"]) > 0:
                trace_ax.plot(
                    np.arange(int(rollout["steps"])),
                    rollout["pref_shaped"],
                    label=rollout["model_name"],
                    linewidth=1.8,
                    color=color,
                )
        scenario_type = matched[0]["scenario_type"] if matched else ""
        trajectory_ax.set_title(f"{map_name} trajectory ({scenario_type})")
        trajectory_ax.set_aspect("equal", adjustable="box")
        trajectory_ax.grid(True, alpha=0.2)
        trace_ax.set_title(f"{map_name} per-step shaped reward")
        trace_ax.grid(True, alpha=0.2)
        trajectory_ax.legend(fontsize=8)
        trace_ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _compute_summary(scored_payload: dict[str, Any]) -> dict[str, Any]:
    rollouts = list(_iter_scored_rollouts(scored_payload))
    model_names = sorted({rollout["model_name"] for rollout in rollouts})
    summary: dict[str, Any] = {
        "normalization": scored_payload["normalization"],
        "models": {},
        "per_bucket": {},
    }

    for model_name in model_names:
        model_rollouts = [rollout for rollout in rollouts if rollout["model_name"] == model_name]
        pref_shaped = np.concatenate([rollout["pref_shaped"] for rollout in model_rollouts]) if model_rollouts else np.zeros(0)
        pref_raw = np.concatenate([rollout["pref_raw"] for rollout in model_rollouts]) if model_rollouts else np.zeros(0)
        pref_std = np.concatenate([rollout["pref_std"] for rollout in model_rollouts]) if model_rollouts else np.zeros(0)
        smoothness = float(np.mean(np.abs(np.diff(pref_shaped)))) if pref_shaped.size > 1 else 0.0
        summary["models"][model_name] = {
            "rollout_count": len(model_rollouts),
            "mean_pref_raw": float(np.mean(pref_raw)) if pref_raw.size else 0.0,
            "mean_pref_shaped": float(np.mean(pref_shaped)) if pref_shaped.size else 0.0,
            "mean_pref_std": float(np.mean(pref_std)) if pref_std.size else 0.0,
            "mean_cumulative_pref_shaped": float(np.mean([rollout["cumulative_pref_shaped"] for rollout in model_rollouts])) if model_rollouts else 0.0,
            "reward_trace_smoothness": smoothness,
        }

    for scenario_type in ("turning", "straight"):
        summary["per_bucket"][scenario_type] = {}
        for model_name in model_names:
            bucket_rollouts = [
                rollout
                for rollout in rollouts
                if rollout["model_name"] == model_name and rollout["scenario_type"] == scenario_type
            ]
            shaped = np.concatenate([rollout["pref_shaped"] for rollout in bucket_rollouts]) if bucket_rollouts else np.zeros(0)
            summary["per_bucket"][scenario_type][model_name] = {
                "rollout_count": len(bucket_rollouts),
                "mean_pref_shaped": float(np.mean(shaped)) if shaped.size else 0.0,
                "mean_cumulative_pref_shaped": float(np.mean([rollout["cumulative_pref_shaped"] for rollout in bucket_rollouts])) if bucket_rollouts else 0.0,
            }

    if len(model_names) == 2:
        left, right = model_names
        left_score = summary["models"][left]["mean_cumulative_pref_shaped"]
        right_score = summary["models"][right]["mean_cumulative_pref_shaped"]
        summary["favored_model_overall"] = left if left_score >= right_score else right

    return summary


def _write_summary_report(scored_payload: dict[str, Any], summary_path: Path) -> Path:
    summary = _compute_summary(scored_payload)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _write_narrative_report(scored_payload: dict[str, Any], output_path: Path) -> Path:
    summary = _compute_summary(scored_payload)
    model_names = sorted(summary["models"])
    favored = summary.get("favored_model_overall", model_names[0] if model_names else "unknown")
    lines = [
        "# Preference Reward Rollout Comparison",
        "",
        f"- Favored model overall: `{favored}`",
        f"- Normalization mode: `{summary['normalization']['mode']}`",
        f"- Calibration steps used: `{summary['normalization']['used_steps']}` / `{summary['normalization']['requested_steps']}`",
        "",
        "## Overall",
    ]
    for model_name in model_names:
        model_summary = summary["models"][model_name]
        lines.append(
            f"- `{model_name}`: mean_cumulative_pref_shaped={model_summary['mean_cumulative_pref_shaped']:.4f}, "
            f"mean_pref_std={model_summary['mean_pref_std']:.4f}, smoothness={model_summary['reward_trace_smoothness']:.4f}"
        )

    lines.extend(["", "## Turning vs Straight"])
    for scenario_type in ("turning", "straight"):
        lines.append(f"- `{scenario_type}`:")
        for model_name in model_names:
            bucket_summary = summary["per_bucket"][scenario_type][model_name]
            lines.append(
                f"  - `{model_name}`: mean_cumulative_pref_shaped={bucket_summary['mean_cumulative_pref_shaped']:.4f}, "
                f"rollout_count={bucket_summary['rollout_count']}"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- `mean_cumulative_pref_shaped` estimates which rollout style the reward model prefers on the sampled scenes.",
            "- `mean_pref_std` summarizes ensemble disagreement; higher values indicate greater uncertainty.",
            "- `smoothness` is the mean absolute step-to-step change in shaped reward; higher values indicate noisier reward traces.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def generate_report(scored_path: Path, report_dir: Path) -> dict[str, str]:
    scored_payload = torch.load(scored_path, map_location="cpu", weights_only=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "step_csv": str(_save_step_csv(scored_payload, report_dir / "preference_reward_steps.csv")),
        "rollout_csv": str(_save_rollout_summary_csv(scored_payload, report_dir / "preference_reward_rollouts.csv")),
        "histograms": str(_plot_histogram_comparison(scored_payload, report_dir / "reward_histograms.png")),
        "cumulative_rewards": str(_plot_cumulative_rewards(scored_payload, report_dir / "cumulative_rewards_by_map.png")),
        "uncertainty": str(_plot_uncertainty(scored_payload, report_dir / "preference_uncertainty.png")),
        "gallery": str(_plot_representative_gallery(scored_payload, report_dir / "representative_rollout_gallery.png")),
        "summary_json": str(_write_summary_report(scored_payload, report_dir / "report_summary.json")),
        "narrative": str(_write_narrative_report(scored_payload, report_dir / "report_summary.md")),
    }
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Drive model rollouts with offline preference reward scoring.")
    parser.add_argument("--car-model-root", type=Path, default=DEFAULT_CAR_MODEL_ROOT)
    parser.add_argument("--truck-model-root", type=Path, default=DEFAULT_TRUCK_MODEL_ROOT)
    parser.add_argument("--car-config", type=Path, default=None)
    parser.add_argument("--truck-config", type=Path, default=None)
    parser.add_argument("--source-map-dir", type=Path, default=None)
    parser.add_argument("--preference-config", type=Path, default=DEFAULT_PREFERENCE_CONFIG)
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--drive-config-path", type=Path, default=DEFAULT_DRIVE_CONFIG)
    parser.add_argument("--turning-count", type=int, default=DEFAULT_TURNING_COUNT)
    parser.add_argument("--straight-count", type=int, default=DEFAULT_STRAIGHT_COUNT)
    parser.add_argument("--threshold-deg", type=float, default=DEFAULT_THRESHOLD_DEG)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    car_spec = discover_model_spec(args.car_model_root, args.car_config)
    truck_spec = discover_model_spec(args.truck_model_root, args.truck_config)
    source_map_dir = args.source_map_dir.resolve() if args.source_map_dir is not None else _infer_source_map_dir(car_spec, truck_spec)

    sample_manifest_path = output_dir / "sample_manifest.json"
    validity_checker = _build_map_validity_checker()
    sample_manifest = build_stratified_sample_manifest(
        source_map_dir=source_map_dir,
        output_path=sample_manifest_path,
        turning_count=args.turning_count,
        straight_count=args.straight_count,
        threshold_deg=args.threshold_deg,
        sample_seed=args.sample_seed,
        validity_checker=validity_checker,
    )

    rollout_dir = output_dir / "rollouts"
    car_rollouts = collect_rollouts_for_model(
        model_spec=car_spec,
        sample_manifest=sample_manifest,
        output_path=rollout_dir / f"{car_spec.model_name}_rollouts.pt",
        device=args.device,
        drive_config_path=args.drive_config_path,
    )
    truck_rollouts = collect_rollouts_for_model(
        model_spec=truck_spec,
        sample_manifest=sample_manifest,
        output_path=rollout_dir / f"{truck_spec.model_name}_rollouts.pt",
        device=args.device,
        drive_config_path=args.drive_config_path,
    )

    scored_path = score_rollout_payloads(
        rollout_paths=[car_rollouts, truck_rollouts],
        preference_config_path=args.preference_config.resolve(),
        reward_dir=args.reward_dir.resolve(),
        output_path=output_dir / "scored_rollouts.pt",
    )
    report_outputs = generate_report(scored_path, output_dir / "report")
    print(json.dumps({"sample_manifest": str(sample_manifest_path), "scored_rollouts": str(scored_path), "report": report_outputs}, indent=2))


if __name__ == "__main__":
    main()
