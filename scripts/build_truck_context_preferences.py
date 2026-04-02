from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import warnings

import numpy as np
import torch

from scripts.export_paired_offline_fits import iter_paired_fit_shards, load_paired_fit_manifest

LOGGER = logging.getLogger("build_truck_context_preferences")


DEFAULT_INPUT = Path("outputs/offline_fits/nuplan_boston_test10_paired_fits.pt")
DEFAULT_OUTPUT = Path("outputs/preferences/nuplan_boston_test10_truck_context_preferences.pt")
DEFAULT_WINDOW_LEN = 32
DEFAULT_MAX_START_DISTANCE_M = 1.0
DEFAULT_MIN_TIME_DIFF_SECONDS = 2.0
DEFAULT_OBSERVATION_MODE = "default"
DEFAULT_ACTION_TYPE = "discrete"
DEFAULT_DISCRETE_ACTION_COUNT = 7 * 13
DEFAULT_PREFERENCE_CHUNK_SIZE = 256
DEFAULT_MAP_LOG_EVERY = 50

PREFERENCES_FORMAT_MONOLITHIC = "truck_context_preferences_v1"
PREFERENCES_FORMAT_SHARDED = "sharded_truck_context_preferences_v1"

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


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _scenario_heading_delta_deg(pair: dict, replay: dict) -> float | None:
    truck_side = pair.get("truck", {})
    heading_candidates = (
        truck_side.get("gt_heading"),
        truck_side.get("rollout_heading"),
        replay.get("truck_branch", {}).get("rollout_heading"),
    )
    for heading_sequence in heading_candidates:
        if heading_sequence is None:
            continue
        heading = np.asarray(heading_sequence, dtype=np.float32).reshape(-1)
        if heading.size < 2:
            continue
        delta_rad = _wrap_angle(float(heading[-1]) - float(heading[0]))
        return float(np.degrees(delta_rad))
    return None


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


def load_preference_manifest(preference_path: Path) -> dict:
    payload = torch.load(preference_path, map_location="cpu")
    if isinstance(payload, dict) and payload.get("format") == PREFERENCES_FORMAT_SHARDED:
        return payload
    if isinstance(payload, dict) and "preferred_sa" in payload and "rejected_sa" in payload and "labels" in payload:
        return {
            "format": PREFERENCES_FORMAT_MONOLITHIC,
            "metadata": payload.get("metadata", {}),
            "shards": [
                {
                    "path": str(preference_path),
                    "window_count": int(len(np.asarray(payload["preferred_sa"]))),
                    "start_index": 0,
                    "end_index": int(len(np.asarray(payload["preferred_sa"]))),
                }
            ],
        }
    raise ValueError(f"unsupported preference payload at {preference_path}")


def iter_preference_shards(preference_path: Path):
    manifest = load_preference_manifest(preference_path)
    if manifest["format"] == PREFERENCES_FORMAT_MONOLITHIC:
        payload = torch.load(preference_path, map_location="cpu")
        yield payload, manifest["shards"][0]
        return

    for shard_info in manifest.get("shards", []):
        shard_path = Path(shard_info["path"])
        if not shard_path.exists():
            candidate_same_dir = preference_path.parent / shard_path.name
            candidate_sibling_dir = preference_path.parent / f"{preference_path.stem}_shards" / shard_path.name
            if candidate_same_dir.exists():
                shard_path = candidate_same_dir
            elif candidate_sibling_dir.exists():
                shard_path = candidate_sibling_dir
        LOGGER.debug(
            "Loading preference shard | path=%s window_count=%s",
            shard_path,
            shard_info.get("window_count"),
        )
        yield torch.load(shard_path, map_location="cpu"), shard_info


def _preference_payload(
    metadata: dict,
    preferred_sa: np.ndarray,
    rejected_sa: np.ndarray,
    labels: np.ndarray,
    window_metadata: list[dict],
) -> dict:
    return {
        "metadata": metadata,
        "preferred_sa": preferred_sa,
        "rejected_sa": rejected_sa,
        "labels": labels,
        "window_metadata": window_metadata,
    }


def _stack_or_empty(segments: list[np.ndarray], shape_tail: tuple[int, ...]) -> np.ndarray:
    if segments:
        return np.stack(segments).astype(np.float32)
    return np.zeros((0, *shape_tail), dtype=np.float32)


def _write_preference_summary(output_path: Path, metadata: dict, total_windows: int, shard_count: int) -> None:
    summary = {
        "output_path": str(output_path),
        "source_export": metadata["source_export"],
        "window_len": int(metadata["window_len"]),
        "min_time_diff_seconds": float(metadata["min_time_diff_seconds"]),
        "min_gap_steps": int(metadata["min_gap_steps"]),
        "max_start_distance_m": float(metadata["max_start_distance_m"]),
        "observation_mode": metadata["observation_mode"],
        "observation_key": metadata["observation_key"],
        "action_type": metadata["action_type"],
        "action_encoding": metadata["action_encoding"],
        "discrete_action_count": int(metadata["discrete_action_count"]),
        "dt": float(metadata["dt"]),
        "obs_dim": int(metadata["obs_dim"]),
        "action_dim": int(metadata["action_dim"]),
        "turning_threshold_deg": metadata.get("turning_threshold_deg"),
        "turning_filtered_map_count": int(metadata.get("turning_filtered_map_count", 0)),
        "total_windows": int(total_windows),
        "shard_count": int(shard_count),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))


def _flush_preference_shard(
    shard_dir: Path,
    output_path: Path,
    shard_index: int,
    metadata: dict,
    pref_segments: list[np.ndarray],
    rej_segments: list[np.ndarray],
    labels: list[list[float]],
    window_metadata: list[dict],
    feature_dim: int,
) -> tuple[Path, int]:
    shard_path = shard_dir / f"{output_path.stem}.part{shard_index:05d}.pt"
    pref_array = _stack_or_empty(pref_segments, (metadata["window_len"], feature_dim))
    rej_array = _stack_or_empty(rej_segments, (metadata["window_len"], feature_dim))
    label_array = np.asarray(labels, dtype=np.float32) if labels else np.zeros((0, 1), dtype=np.float32)
    torch.save(
        _preference_payload(metadata, pref_array, rej_array, label_array, list(window_metadata)),
        shard_path,
    )
    return shard_path, int(len(pref_array))


def build_truck_context_preferences(
    export_path: Path,
    output_path: Path,
    window_len: int = DEFAULT_WINDOW_LEN,
    max_start_distance_m: float = DEFAULT_MAX_START_DISTANCE_M,
    min_time_diff_seconds: float = DEFAULT_MIN_TIME_DIFF_SECONDS,
    turning_threshold_deg: float | None = None,
    observation_mode: str = DEFAULT_OBSERVATION_MODE,
    action_type: str = DEFAULT_ACTION_TYPE,
    discrete_action_count: int = DEFAULT_DISCRETE_ACTION_COUNT,
    chunk_size: int = DEFAULT_PREFERENCE_CHUNK_SIZE,
    map_log_every: int = DEFAULT_MAP_LOG_EVERY,
) -> Path:
    fit_manifest = load_paired_fit_manifest(export_path)
    dt = float(fit_manifest["metadata"]["fit_settings"]["dt"])
    total_map_count = int(
        fit_manifest["metadata"].get("shared_map_count")
        or sum(int(shard.get("map_count", 0)) for shard in fit_manifest.get("shards", []))
    )
    min_gap_steps = int(np.ceil(min_time_diff_seconds / dt))
    observation_key = _resolve_observation_key(observation_mode)
    effective_chunk_size = max(1, int(chunk_size))
    effective_map_log_every = max(1, int(map_log_every))

    pref_segments = []
    rej_segments = []
    labels = []
    metadata = []
    feature_dim: int | None = None
    processed_maps = 0
    turning_filtered_map_count = 0
    total_windows = 0
    shard_index = 0
    shard_infos: list[dict] = []
    shard_dir = output_path.parent / f"{output_path.stem}_shards"

    LOGGER.info(
        "Starting preference build | export=%s total_map_count=%d chunk_size=%d map_log_every=%d",
        export_path,
        total_map_count,
        effective_chunk_size,
        effective_map_log_every,
    )

    def _log_progress() -> None:
        pending_windows = total_windows + len(pref_segments)
        percent_complete = (100.0 * processed_maps / total_map_count) if total_map_count > 0 else 0.0
        LOGGER.info(
            "Preference build progress | processed_maps=%d total_map_count=%d percent_complete=%.2f total_windows=%d",
            processed_maps,
            total_map_count,
            percent_complete,
            pending_windows,
        )

    for _fit_metadata, pairs, shard_info in iter_paired_fit_shards(export_path):
        LOGGER.info(
            "Processing paired fit shard for preferences | path=%s map_count=%s",
            shard_info.get("path"),
            shard_info.get("map_count"),
        )
        for map_name, pair in pairs.items():
            processed_maps += 1
            replay = pair.get("truck_context_replay", {"status": "unavailable"})
            if replay.get("status") != "ok":
                if processed_maps % effective_map_log_every == 0:
                    _log_progress()
                continue

            scenario_delta_heading_deg = _scenario_heading_delta_deg(pair, replay)
            if (
                turning_threshold_deg is not None
                and (
                    scenario_delta_heading_deg is None
                    or abs(scenario_delta_heading_deg) <= float(turning_threshold_deg)
                )
            ):
                turning_filtered_map_count += 1
                if processed_maps % effective_map_log_every == 0:
                    _log_progress()
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
                if processed_maps % effective_map_log_every == 0:
                    _log_progress()
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
                        "scenario_delta_heading_deg": scenario_delta_heading_deg,
                        "turning_threshold_deg": float(turning_threshold_deg) if turning_threshold_deg is not None else None,
                    }
                )

                if len(pref_segments) >= effective_chunk_size:
                    if feature_dim is None:
                        raise ValueError("feature_dim unexpectedly unset while flushing preference shard")
                    shard_dir.mkdir(parents=True, exist_ok=True)
                    shard_index += 1
                    shard_metadata = {
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
                        "obs_dim": int(feature_dim - discrete_action_count),
                        "action_dim": int(discrete_action_count),
                        "turning_threshold_deg": (
                            float(turning_threshold_deg) if turning_threshold_deg is not None else None
                        ),
                        "turning_filtered_map_count": int(turning_filtered_map_count),
                        "preferred_label": 0,
                        "total_windows": int(len(pref_segments)),
                    }
                    shard_path, shard_window_count = _flush_preference_shard(
                        shard_dir=shard_dir,
                        output_path=output_path,
                        shard_index=shard_index,
                        metadata=shard_metadata,
                        pref_segments=pref_segments,
                        rej_segments=rej_segments,
                        labels=labels,
                        window_metadata=metadata,
                        feature_dim=feature_dim,
                    )
                    shard_infos.append(
                        {
                            "path": str(shard_path),
                            "window_count": int(shard_window_count),
                            "start_index": int(total_windows),
                            "end_index": int(total_windows + shard_window_count),
                        }
                    )
                    total_windows += shard_window_count
                    LOGGER.info(
                        "Wrote preference shard | shard=%d windows=%d cumulative_windows=%d output=%s",
                        shard_index,
                        shard_window_count,
                        total_windows,
                        shard_path,
                    )
                    pref_segments = []
                    rej_segments = []
                    labels = []
                    metadata = []

            if processed_maps % effective_map_log_every == 0:
                _log_progress()

    ds = (feature_dim - discrete_action_count) if feature_dim is not None else EXPECTED_OBS_DIM_BY_KEY[observation_key]
    da = discrete_action_count
    base_metadata = {
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
        "turning_threshold_deg": float(turning_threshold_deg) if turning_threshold_deg is not None else None,
        "turning_filtered_map_count": int(turning_filtered_map_count),
        "preferred_label": 0,
        "total_windows": int(total_windows + len(pref_segments)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not shard_infos:
        pref_array = _stack_or_empty(pref_segments, (window_len, ds + da))
        rej_array = _stack_or_empty(rej_segments, (window_len, ds + da))
        label_array = np.asarray(labels, dtype=np.float32) if labels else np.zeros((0, 1), dtype=np.float32)
        torch.save(_preference_payload(base_metadata, pref_array, rej_array, label_array, metadata), output_path)
        _write_preference_summary(output_path, base_metadata, len(pref_array), 1)
        LOGGER.info(
            "Finished preference build | output=%s total_windows=%d shard_count=1",
            output_path,
            len(pref_array),
        )
        return output_path

    if pref_segments:
        if feature_dim is None:
            raise ValueError("feature_dim unexpectedly unset while flushing final preference shard")
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_index += 1
        shard_path, shard_window_count = _flush_preference_shard(
            shard_dir=shard_dir,
            output_path=output_path,
            shard_index=shard_index,
            metadata=base_metadata,
            pref_segments=pref_segments,
            rej_segments=rej_segments,
            labels=labels,
            window_metadata=metadata,
            feature_dim=feature_dim,
        )
        shard_infos.append(
            {
                "path": str(shard_path),
                "window_count": int(shard_window_count),
                "start_index": int(total_windows),
                "end_index": int(total_windows + shard_window_count),
            }
        )
        total_windows += shard_window_count
        LOGGER.info(
            "Wrote final preference shard | shard=%d windows=%d cumulative_windows=%d output=%s",
            shard_index,
            shard_window_count,
            total_windows,
            shard_path,
        )

    manifest = {
        "format": PREFERENCES_FORMAT_SHARDED,
        "metadata": {**base_metadata, "total_windows": int(total_windows)},
        "shards": shard_infos,
    }
    torch.save(manifest, output_path)
    _write_preference_summary(output_path, manifest["metadata"], total_windows, len(shard_infos))
    LOGGER.info(
        "Finished preference build | output=%s total_windows=%d shard_count=%d",
        output_path,
        total_windows,
        len(shard_infos),
    )
    return output_path


def load_preferences_into_reward_model(reward_model, preference_payload: dict) -> int:
    if isinstance(preference_payload, dict) and preference_payload.get("format") == PREFERENCES_FORMAT_SHARDED:
        total_loaded = 0
        for shard_info in preference_payload.get("shards", []):
            shard_payload = torch.load(Path(shard_info["path"]), map_location="cpu")
            total_loaded += load_preferences_into_reward_model(reward_model, shard_payload)
        return total_loaded
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
    parser.add_argument(
        "--turning-threshold-deg",
        type=float,
        default=None,
        help="Optional wrapped heading-delta threshold; keep only scenarios with abs(end_heading - start_heading) above this many degrees.",
    )
    parser.add_argument("--observation-mode", type=str, default=DEFAULT_OBSERVATION_MODE)
    parser.add_argument("--action-type", type=str, default=DEFAULT_ACTION_TYPE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_PREFERENCE_CHUNK_SIZE)
    parser.add_argument("--map-log-every", type=int, default=DEFAULT_MAP_LOG_EVERY)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_path = build_truck_context_preferences(
        export_path=args.input,
        output_path=args.output,
        window_len=args.window_len,
        max_start_distance_m=args.max_start_distance_m,
        min_time_diff_seconds=args.min_time_diff_seconds,
        turning_threshold_deg=args.turning_threshold_deg,
        observation_mode=args.observation_mode,
        action_type=args.action_type,
        chunk_size=args.chunk_size,
        map_log_every=args.map_log_every,
    )
    print(output_path)


if __name__ == "__main__":
    main()
