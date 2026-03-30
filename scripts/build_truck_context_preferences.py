from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import torch


DEFAULT_INPUT = Path("outputs/offline_fits/nuplan_boston_test10_paired_fits.pt")
DEFAULT_OUTPUT = Path("outputs/preferences/nuplan_boston_test10_truck_context_preferences.pt")
DEFAULT_WINDOW_LEN = 32
DEFAULT_MAX_START_DISTANCE_M = 1.0
DEFAULT_MIN_TIME_DIFF_SECONDS = 2.0
DEFAULT_OBSERVATION_MODE = "default"
DEFAULT_ACTION_TYPE = "discrete"
DEFAULT_DISCRETE_ACTION_COUNT = 7 * 13

OBSERVATION_KEY_BY_MODE = {
    "default": "obs_default",
    "sdc_only_with_trailer": "obs_sdc_only_with_trailer",
    "obs": "obs",
    "obs_default": "obs_default",
    "obs_sdc_only_with_trailer": "obs_sdc_only_with_trailer",
}

EXPECTED_OBS_DIM_BY_KEY = {
    "obs_default": 1120,
    "obs_sdc_only_with_trailer": 1156,
}


def _resolve_observation_key(observation_mode: str) -> str:
    if observation_mode not in OBSERVATION_KEY_BY_MODE:
        valid = ", ".join(sorted(OBSERVATION_KEY_BY_MODE))
        raise ValueError(f"unsupported observation_mode={observation_mode!r}; valid values: {valid}")
    return OBSERVATION_KEY_BY_MODE[observation_mode]


def _validate_branch_dimensions(branch: dict, observation_key: str, expected_obs_dim: int | None) -> tuple[np.ndarray, np.ndarray]:
    if observation_key not in branch:
        available = ", ".join(sorted(branch.keys()))
        raise KeyError(f"branch missing {observation_key!r}; available keys: {available}")
    obs = np.asarray(branch[observation_key], dtype=np.float32)
    actions = np.asarray(branch["actions"], dtype=np.float32).reshape(-1, 1)
    if obs.ndim != 2:
        raise ValueError(f"expected rank-2 observations for {observation_key}, got shape {obs.shape}")
    if actions.ndim != 2 or actions.shape[1] != 1:
        raise ValueError(f"expected actions to reshape to (T, 1), got shape {actions.shape}")
    if len(obs) != len(actions):
        raise ValueError(f"obs/action length mismatch: {len(obs)} vs {len(actions)}")
    if expected_obs_dim is not None and obs.shape[1] != expected_obs_dim:
        raise ValueError(
            f"unexpected observation dimension for {observation_key}: "
            f"expected {expected_obs_dim}, got {obs.shape[1]}"
        )
    return obs, actions


def _encode_actions(actions: np.ndarray, action_type: str, discrete_action_count: int) -> np.ndarray:
    if action_type != "discrete":
        raise ValueError(f"unsupported action_type={action_type!r}; only 'discrete' is currently supported")
    discrete = actions.astype(np.int64).reshape(-1)
    if np.any(discrete < 0) or np.any(discrete >= discrete_action_count):
        raise ValueError(
            f"discrete action index out of range for one-hot encoding: valid [0, {discrete_action_count - 1}]"
        )
    encoded = np.zeros((len(discrete), discrete_action_count), dtype=np.float32)
    encoded[np.arange(len(discrete)), discrete] = 1.0
    return encoded


def _branch_to_sa(branch: dict, observation_key: str, action_type: str, discrete_action_count: int) -> np.ndarray:
    expected_obs_dim = EXPECTED_OBS_DIM_BY_KEY.get(observation_key)
    obs, actions = _validate_branch_dimensions(
        branch,
        observation_key=observation_key,
        expected_obs_dim=expected_obs_dim,
    )
    encoded_actions = _encode_actions(actions, action_type=action_type, discrete_action_count=discrete_action_count)
    return np.concatenate([obs, encoded_actions], axis=-1)


def _select_window_starts(
    aligned_steps: int,
    window_len: int,
    min_gap_steps: int,
    max_start_distance_m: float,
    truck_rollout_x: np.ndarray,
    truck_rollout_y: np.ndarray,
    car_rollout_x: np.ndarray,
    car_rollout_y: np.ndarray,
) -> list[tuple[int, int, float]]:
    if aligned_steps < window_len:
        return []

    starts: list[tuple[int, int, float]] = []
    current_start = 0
    starts.append(
        (
            current_start,
            0,
            float(
                np.sqrt(
                    (truck_rollout_x[current_start] - car_rollout_x[current_start]) ** 2
                    + (truck_rollout_y[current_start] - car_rollout_y[current_start]) ** 2
                )
            ),
        )
    )

    while True:
        search_start = current_start + min_gap_steps
        search_end = aligned_steps - window_len
        if search_start > search_end:
            break

        next_start = None
        next_distance = None
        for candidate in range(search_start, search_end + 1):
            start_distance = float(
                np.sqrt(
                    (truck_rollout_x[candidate] - car_rollout_x[candidate]) ** 2
                    + (truck_rollout_y[candidate] - car_rollout_y[candidate]) ** 2
                )
            )
            if start_distance <= max_start_distance_m:
                next_start = candidate
                next_distance = start_distance
                break

        if next_start is None or next_distance is None:
            break

        starts.append((next_start, next_start - current_start, next_distance))
        current_start = next_start

    return starts


def build_truck_context_preferences(
    export_path: Path,
    output_path: Path,
    window_len: int = DEFAULT_WINDOW_LEN,
    max_start_distance_m: float = DEFAULT_MAX_START_DISTANCE_M,
    min_time_diff_seconds: float = DEFAULT_MIN_TIME_DIFF_SECONDS,
    observation_mode: str = DEFAULT_OBSERVATION_MODE,
    action_type: str = DEFAULT_ACTION_TYPE,
    discrete_action_count: int = DEFAULT_DISCRETE_ACTION_COUNT,
) -> Path:
    payload = torch.load(export_path, map_location="cpu")
    pairs = payload["pairs"]
    dt = float(payload["metadata"]["fit_settings"]["dt"])
    min_gap_steps = int(np.ceil(min_time_diff_seconds / dt))
    observation_key = _resolve_observation_key(observation_mode)

    pref_segments = []
    rej_segments = []
    labels = []
    metadata = []
    feature_dim: int | None = None

    for map_name, pair in pairs.items():
        replay = pair.get("truck_context_replay", {"status": "unavailable"})
        if replay.get("status") != "ok":
            continue

        truck_branch = replay["truck_branch"]
        car_branch = replay["car_branch"]
        preferred_sa = _branch_to_sa(
            truck_branch,
            observation_key=observation_key,
            action_type=action_type,
            discrete_action_count=discrete_action_count,
        )
        rejected_sa = _branch_to_sa(
            car_branch,
            observation_key=observation_key,
            action_type=action_type,
            discrete_action_count=discrete_action_count,
        )
        if preferred_sa.shape[-1] != rejected_sa.shape[-1]:
            raise ValueError(
                f"preferred/rejected feature dim mismatch for {map_name}: "
                f"{preferred_sa.shape[-1]} vs {rejected_sa.shape[-1]}"
            )
        if feature_dim is None:
            feature_dim = int(preferred_sa.shape[-1])
        elif preferred_sa.shape[-1] != feature_dim:
            raise ValueError(
                f"inconsistent feature dim across maps: expected {feature_dim}, got {preferred_sa.shape[-1]} for {map_name}"
            )

        aligned_steps = min(len(preferred_sa), len(rejected_sa))
        if aligned_steps < window_len:
            continue

        truck_rollout_x = np.asarray(truck_branch["rollout_x"], dtype=np.float32)
        truck_rollout_y = np.asarray(truck_branch["rollout_y"], dtype=np.float32)
        car_rollout_x = np.asarray(car_branch["rollout_x"], dtype=np.float32)
        car_rollout_y = np.asarray(car_branch["rollout_y"], dtype=np.float32)

        window_starts = _select_window_starts(
            aligned_steps=aligned_steps,
            window_len=window_len,
            min_gap_steps=min_gap_steps,
            max_start_distance_m=max_start_distance_m,
            truck_rollout_x=truck_rollout_x,
            truck_rollout_y=truck_rollout_y,
            car_rollout_x=car_rollout_x,
            car_rollout_y=car_rollout_y,
        )
        for start, shift_steps, start_distance in window_starts:
            end = start + window_len
            pref_segments.append(preferred_sa[start:end])
            rej_segments.append(rejected_sa[start:end])
            labels.append([0.0])

            rollout_end = min(
                len(truck_rollout_x),
                len(truck_rollout_y),
                len(car_rollout_x),
                len(car_rollout_y),
                end + 1,
            )
            rollout_start = min(start, rollout_end)
            displacement = np.sqrt(
                (truck_rollout_x[rollout_start:rollout_end] - car_rollout_x[rollout_start:rollout_end]) ** 2
                + (truck_rollout_y[rollout_start:rollout_end] - car_rollout_y[rollout_start:rollout_end]) ** 2
            )
            metadata.append(
                {
                    "map_name": map_name,
                    "timestep_start": int(start),
                    "timestep_end": int(end),
                    "shift_steps_from_previous": int(shift_steps),
                    "shift_seconds_from_previous": float(shift_steps * dt),
                    "window_start_distance_m": float(start_distance),
                    "window_pair_ade": float(displacement.mean()) if displacement.size else float("nan"),
                    "window_pair_fde": float(displacement[-1]) if displacement.size else float("nan"),
                    "truck_context_aligned_steps": int(aligned_steps),
                    "scenario_pair_ade": float(replay["pair_similarity"]["ade"]),
                    "scenario_pair_fde": float(replay["pair_similarity"]["fde"]),
                    "truck_self_ade": float(truck_branch["self_ade"]),
                    "truck_self_fde": float(truck_branch["self_fde"]),
                    "car_on_truck_ade": float(car_branch["self_ade"]),
                    "car_on_truck_fde": float(car_branch["self_fde"]),
                }
            )

    ds = 0
    da = discrete_action_count if action_type == "discrete" else 0
    if pref_segments:
        ds = pref_segments[0].shape[-1] - da
        pref_array = np.asarray(pref_segments, dtype=np.float32)
        rej_array = np.asarray(rej_segments, dtype=np.float32)
        label_array = np.asarray(labels, dtype=np.float32)
    else:
        pref_array = np.zeros((0, window_len, 0), dtype=np.float32)
        rej_array = np.zeros((0, window_len, 0), dtype=np.float32)
        label_array = np.zeros((0, 1), dtype=np.float32)

    out_payload = {
        "metadata": {
            "source_export": str(export_path),
            "window_len": int(window_len),
            "min_time_diff_seconds": float(min_time_diff_seconds),
            "min_gap_steps": int(min_gap_steps),
            "max_start_distance_m": float(max_start_distance_m),
            "observation_mode": observation_mode,
            "observation_key": observation_key,
            "action_type": action_type,
            "action_encoding": "one_hot" if action_type == "discrete" else "unsupported",
            "discrete_action_count": int(discrete_action_count),
            "dt": float(dt),
            "obs_dim": int(ds),
            "action_dim": int(da),
            "preferred_label": 0,
            "total_windows": int(len(pref_segments)),
        },
        "preferred_sa": pref_array,
        "rejected_sa": rej_array,
        "labels": label_array,
        "window_metadata": metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, output_path)
    summary = {
        "output_path": str(output_path),
        "source_export": str(export_path),
        "window_len": int(window_len),
        "min_time_diff_seconds": float(min_time_diff_seconds),
        "min_gap_steps": int(min_gap_steps),
        "max_start_distance_m": float(max_start_distance_m),
        "observation_mode": observation_mode,
        "observation_key": observation_key,
        "action_type": action_type,
        "action_encoding": "one_hot" if action_type == "discrete" else "unsupported",
        "discrete_action_count": int(discrete_action_count),
        "dt": float(dt),
        "obs_dim": int(ds),
        "action_dim": int(da),
        "total_windows": int(len(pref_segments)),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return output_path


def load_preferences_into_reward_model(reward_model, preference_payload: dict) -> int:
    preferred = np.asarray(preference_payload["preferred_sa"], dtype=np.float32)
    rejected = np.asarray(preference_payload["rejected_sa"], dtype=np.float32)
    labels = np.asarray(preference_payload["labels"], dtype=np.float32)
    meta = preference_payload.get("metadata", {})
    if preferred.ndim != 3 or rejected.ndim != 3:
        raise ValueError("preference tensors must be rank-3")
    if preferred.shape != rejected.shape:
        raise ValueError(f"preferred/rejected shape mismatch: {preferred.shape} vs {rejected.shape}")
    action_type = meta.get("action_type")
    action_encoding = meta.get("action_encoding")
    if action_type == "discrete" and action_encoding != "one_hot":
        warnings.warn("scalar discrete action input is not supported; expected one-hot discrete actions", stacklevel=2)
        raise ValueError(f"unsupported discrete action encoding: {action_encoding!r}")
    if action_type is not None and action_type != "discrete":
        raise ValueError(f"unsupported action_type in preference payload: {action_type!r}")
    expected_feature_dim = reward_model.ds + reward_model.da
    if preferred.shape[2] != expected_feature_dim:
        raise ValueError(
            f"feature dimension mismatch: payload has {preferred.shape[2]}, reward model expects {expected_feature_dim}"
        )
    if preferred.shape[1] != reward_model.size_segment:
        raise ValueError(
            f"segment length mismatch: payload has {preferred.shape[1]}, reward model expects {reward_model.size_segment}"
        )
    if "obs_dim" in meta and int(meta["obs_dim"]) != reward_model.ds:
        raise ValueError(f"obs_dim mismatch: payload has {meta['obs_dim']}, reward model expects {reward_model.ds}")
    if "action_dim" in meta and int(meta["action_dim"]) != reward_model.da:
        raise ValueError(f"action_dim mismatch: payload has {meta['action_dim']}, reward model expects {reward_model.da}")
    if "window_len" in meta and int(meta["window_len"]) != reward_model.size_segment:
        raise ValueError(
            f"window_len mismatch: payload has {meta['window_len']}, reward model expects {reward_model.size_segment}"
        )
    reward_model.put_queries(preferred, rejected, labels)
    return int(preferred.shape[0])


def main():
    parser = argparse.ArgumentParser(description="Build truck-context offline preference windows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-len", type=int, default=DEFAULT_WINDOW_LEN)
    parser.add_argument("--max-start-distance-m", type=float, default=DEFAULT_MAX_START_DISTANCE_M)
    parser.add_argument("--min-time-diff-seconds", type=float, default=DEFAULT_MIN_TIME_DIFF_SECONDS)
    parser.add_argument("--observation-mode", type=str, default=DEFAULT_OBSERVATION_MODE)
    parser.add_argument("--action-type", type=str, default=DEFAULT_ACTION_TYPE)
    args = parser.parse_args()

    output_path = build_truck_context_preferences(
        export_path=args.input,
        output_path=args.output,
        window_len=args.window_len,
        max_start_distance_m=args.max_start_distance_m,
        min_time_diff_seconds=args.min_time_diff_seconds,
        observation_mode=args.observation_mode,
        action_type=args.action_type,
    )
    print(output_path)


if __name__ == "__main__":
    main()
