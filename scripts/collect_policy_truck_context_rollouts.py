#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pufferlib
from pufferlib import pufferl
from scripts.run_packaged_drive_human_replay_eval import (
    _load_packaged_config,
    _overlay_args,
    _prepare_single_map_dir,
)


DEFAULT_REFERENCE = Path(
    "outputs/preference_eval/preference_rollout_model_compare_full_no_resets/rollouts/"
    "car-baseline-full-boston_rollouts.pt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/preference_eval/preference_rollout_model_compare_truck_context/rollouts")
DEFAULT_MAP_ROOT = Path("datasets/nuplanCarBostonAll")
DEFAULT_WINDOW_LEN = 91
DEFAULT_DEVICE = "cpu"

DEFAULT_CAR_CONFIG = Path(
    "pufferlib/resources/drive/models/car-baseline-full-boston/"
    "conf1_discrete_preference_reward_turning_prebuilt_lowmem_obs_default.ini"
)
DEFAULT_CAR_CHECKPOINT = Path(
    "pufferlib/resources/drive/models/car-baseline-full-boston/"
    "puffer_drive_my9g62z5/model_puffer_drive_004000.pt"
)
DEFAULT_TRUCK_CONFIG = Path("pufferlib/resources/drive/models/truck-baseline-full-boston/conf.ini")
DEFAULT_TRUCK_CHECKPOINT = Path("pufferlib/resources/drive/models/truck-baseline-full-boston/puffer_drive_1bp9qw28.pt")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _source_name(rollout: dict[str, Any]) -> str:
    if rollout.get("source_map_name"):
        return str(rollout["source_map_name"])
    if rollout.get("map_path"):
        return Path(str(rollout["map_path"])).name
    return str(rollout["map_name"])


def _reference_maps(reference_path: Path, map_root: Path) -> list[dict[str, Any]]:
    payload = torch.load(reference_path, map_location="cpu", weights_only=False)
    maps = []
    for rollout in payload.get("rollouts", []):
        source_name = _source_name(rollout)
        local_map_path = map_root / source_name
        if not local_map_path.exists():
            raise FileNotFoundError(f"source map from reference artifact not found locally: {local_map_path}")
        maps.append(
            {
                "display_map_name": str(rollout["map_name"]),
                "source_map_name": source_name,
                "map_path": local_map_path,
                "scenario_type": rollout.get("scenario_type", ""),
                "delta_heading_deg": rollout.get("delta_heading_deg"),
            }
        )
    return maps


def _base_args(config_path: Path, checkpoint_path: Path, device: str) -> dict[str, Any]:
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = pufferl.load_config("puffer_drive")
    finally:
        sys.argv = original_argv
    packaged = _load_packaged_config(config_path)
    args = _overlay_args(args, packaged)
    args["load_model_path"] = str(checkpoint_path)
    args["load_id"] = None
    args["wandb"] = False
    args["neptune"] = False
    args["train"]["device"] = device
    args["train"]["compile"] = False
    args["train"]["render"] = False
    args["eval"]["wosac_num_rollouts"] = 1
    args["vec"] = {"backend": "PufferEnv", "num_envs": 1}
    args["env"]["dynamics_model"] = "articulated"
    args["env"]["sdc_runtime_truck_override"] = True
    args["env"]["observation_mode"] = "default"
    args["env"]["action_type"] = "discrete"
    args["env"]["episode_length"] = int(args["env"].get("episode_length", DEFAULT_WINDOW_LEN))
    return args


def _with_single_map(args: dict[str, Any], map_path: Path, temp_root: Path) -> dict[str, Any]:
    scenario_args = copy.deepcopy(args)
    single_map_dir = _prepare_single_map_dir(map_path, temp_root)
    scenario_args["env"]["map_dir"] = str(single_map_dir)
    scenario_args["env"]["num_maps"] = 1
    scenario_args["eval"]["map_dir"] = str(single_map_dir)
    scenario_args["eval"]["wosac_num_maps"] = 1
    scenario_args["vec"] = {"backend": "PufferEnv", "num_envs": 1}
    return scenario_args


def _first_value(values: Any, default: float = 0.0) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _record_state(driver) -> dict[str, float]:
    state = driver.get_global_agent_state(include_sdc_trailer=True)
    trailer = state.get("sdc_trailer", {})
    return {
        "x": _first_value(state["x"]),
        "y": _first_value(state["y"]),
        "z": _first_value(state["z"]),
        "heading": _first_value(state["heading"]),
        "length": _first_value(state["length"]),
        "width": _first_value(state["width"]),
        "trailer_has": _first_value(trailer.get("has_trailer", [0])),
        "trailer_x": _first_value(trailer.get("x", [0.0])),
        "trailer_y": _first_value(trailer.get("y", [0.0])),
        "trailer_heading": _first_value(trailer.get("heading", [0.0])),
    }


def _rollout_one_map(
    *,
    base_args: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    map_record: dict[str, Any],
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    temp_root = Path(tempfile.mkdtemp(prefix=f"{map_record['source_map_name']}_policy_truck_ctx_", dir=str(REPO_ROOT / "tmp")))
    vecenv = None
    try:
        args = _with_single_map(base_args, Path(map_record["map_path"]), temp_root)
        vecenv = pufferl.load_env("puffer_drive", args)
        policy = pufferl.load_policy(args, vecenv, env_name="puffer_drive")
        policy.eval()

        obs, _info = vecenv.reset()
        driver = vecenv.driver_env
        device = args["train"]["device"]
        state = {}
        if args["train"].get("use_rnn"):
            num_agents = int(vecenv.observation_space.shape[0])
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        observations = []
        actions = []
        task_rewards = []
        dones = []
        truncs = []
        values = []
        entropy = []
        state_rows = {key: [] for key in ("x", "y", "z", "heading", "length", "width", "trailer_has", "trailer_x", "trailer_y", "trailer_heading")}
        stop_reason = "max_steps"
        last_x = None
        last_y = None

        for _step in range(max_steps):
            current_state = _record_state(driver)
            observations.append(np.asarray(obs[0], dtype=np.float32).copy())
            for key in state_rows:
                state_rows[key].append(current_state[key])

            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, value = policy.forward_eval(ob_tensor, state)
                action, _logprob, dist_entropy = pufferlib.pytorch.sample_logits(logits)
                action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)

            first_action = int(np.asarray(action_np).reshape(-1)[0])
            actions.append(first_action)
            values.append(float(np.asarray(value.detach().cpu().numpy()).reshape(-1)[0]))
            entropy.append(float(np.asarray(dist_entropy.detach().cpu().numpy()).reshape(-1)[0]))

            next_obs, reward, terminal, truncation, _info = vecenv.step(action_np)
            task_rewards.append(_first_value(reward))
            dones.append(int(np.asarray(terminal).reshape(-1)[0]))
            truncs.append(int(np.asarray(truncation).reshape(-1)[0]))

            if dones[-1]:
                stop_reason = "terminal"
                obs = next_obs
                break
            if truncs[-1]:
                stop_reason = "truncated"
                obs = next_obs
                break

            new_state = _record_state(driver)
            if last_x is None:
                last_x = current_state["x"]
                last_y = current_state["y"]
            jump = float(np.hypot(new_state["x"] - current_state["x"], new_state["y"] - current_state["y"]))
            if jump > 20.0:
                stop_reason = "position_jump>20m"
                obs = next_obs
                break
            obs = next_obs

        steps = len(actions)
        return {
            "map_name": map_record["display_map_name"],
            "source_map_name": map_record["source_map_name"],
            "map_path": str(Path(map_record["map_path"]).resolve()),
            "scenario_type": map_record.get("scenario_type", ""),
            "delta_heading_deg": map_record.get("delta_heading_deg"),
            "steps": int(steps),
            "stop_reason": stop_reason,
            "actions": np.asarray(actions, dtype=np.int32),
            "task_rewards": np.asarray(task_rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.uint8),
            "truncs": np.asarray(truncs, dtype=np.uint8),
            "values": np.asarray(values, dtype=np.float32),
            "entropy": np.asarray(entropy, dtype=np.float32),
            "observations": np.asarray(observations, dtype=np.float32),
            **{key: np.asarray(values_, dtype=np.float32) for key, values_ in state_rows.items()},
            "runtime_overrides": {
                "dynamics_model": "articulated",
                "sdc_runtime_truck_override": True,
                "observation_mode": "default",
                "action_type": "discrete",
            },
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
        }
    finally:
        if vecenv is not None:
            vecenv.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        gc.collect()


def collect_model_rollouts(
    *,
    model_name: str,
    config_path: Path,
    checkpoint_path: Path,
    map_records: list[dict[str, Any]],
    output_path: Path,
    device: str,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    base_args = _base_args(config_path, checkpoint_path, device)
    rollouts = []
    errors = []
    for index, map_record in enumerate(map_records):
        try:
            rollouts.append(
                _rollout_one_map(
                    base_args=base_args,
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    map_record=map_record,
                    seed=seed + index,
                    max_steps=max_steps,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "map_name": map_record["display_map_name"],
                    "source_map_name": map_record["source_map_name"],
                    "map_path": str(map_record["map_path"]),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

    payload = {
        "format": "drive_preference_rollouts_v1",
        "model_name": model_name,
        "model_root": str(config_path.parent),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "runtime_overrides": {
            "dynamics_model": "articulated",
            "sdc_runtime_truck_override": True,
            "observation_mode": "default",
            "action_type": "discrete",
        },
        "source_reference": str(DEFAULT_REFERENCE),
        "rollouts": rollouts,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "output_path": str(output_path),
        "model_name": model_name,
        "rollout_count": len(rollouts),
        "error_count": len(errors),
        "errors": errors,
    }


def collect_pair(
    *,
    reference: Path,
    map_root: Path,
    output_dir: Path,
    car_config: Path,
    car_checkpoint: Path,
    truck_config: Path,
    truck_checkpoint: Path,
    device: str,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    map_records = _reference_maps(reference, map_root)
    car_report = collect_model_rollouts(
        model_name="car-baseline-full-boston-truck-context",
        config_path=car_config,
        checkpoint_path=car_checkpoint,
        map_records=map_records,
        output_path=output_dir / "car-baseline-full-boston_truck-context_rollouts.pt",
        device=device,
        max_steps=max_steps,
        seed=seed,
    )
    truck_report = collect_model_rollouts(
        model_name="truck-baseline-full-boston-truck-context",
        config_path=truck_config,
        checkpoint_path=truck_checkpoint,
        map_records=map_records,
        output_path=output_dir / "truck-baseline-full-boston_truck-context_rollouts.pt",
        device=device,
        max_steps=max_steps,
        seed=seed + 1000,
    )
    report = {
        "reference": str(reference),
        "map_root": str(map_root),
        "output_dir": str(output_dir),
        "map_count": len(map_records),
        "runtime_overrides": {
            "dynamics_model": "articulated",
            "sdc_runtime_truck_override": True,
            "observation_mode": "default",
            "action_type": "discrete",
        },
        "car": car_report,
        "truck": truck_report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "truck_context_policy_rollouts_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect car/truck policy rollouts in a shared truck-context runtime.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--car-config", type=Path, default=DEFAULT_CAR_CONFIG)
    parser.add_argument("--car-checkpoint", type=Path, default=DEFAULT_CAR_CHECKPOINT)
    parser.add_argument("--truck-config", type=Path, default=DEFAULT_TRUCK_CONFIG)
    parser.add_argument("--truck-checkpoint", type=Path, default=DEFAULT_TRUCK_CHECKPOINT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_WINDOW_LEN)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_pair(
        reference=args.reference,
        map_root=args.map_root,
        output_dir=args.output_dir,
        car_config=args.car_config,
        car_checkpoint=args.car_checkpoint,
        truck_config=args.truck_config,
        truck_checkpoint=args.truck_checkpoint,
        device=args.device,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print(json.dumps(_jsonable(report), indent=2))


if __name__ == "__main__":
    main()
