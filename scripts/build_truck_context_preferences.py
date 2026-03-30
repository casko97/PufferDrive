from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


DEFAULT_INPUT = Path("outputs/offline_fits/nuplan_boston_test10_paired_fits.pt")
DEFAULT_OUTPUT = Path("outputs/preferences/nuplan_boston_test10_truck_context_preferences.pt")
DEFAULT_WINDOW_LEN = 32
DEFAULT_STRIDE = 32


def _branch_to_sa(branch: dict) -> np.ndarray:
    obs = np.asarray(branch["obs"], dtype=np.float32)
    actions = np.asarray(branch["actions"], dtype=np.float32).reshape(-1, 1)
    if len(obs) != len(actions):
        raise ValueError(f"obs/action length mismatch: {len(obs)} vs {len(actions)}")
    return np.concatenate([obs, actions], axis=-1)


def build_truck_context_preferences(
    export_path: Path,
    output_path: Path,
    window_len: int = DEFAULT_WINDOW_LEN,
    stride: int = DEFAULT_STRIDE,
) -> Path:
    payload = torch.load(export_path, map_location="cpu")
    pairs = payload["pairs"]

    pref_segments = []
    rej_segments = []
    labels = []
    metadata = []

    for map_name, pair in pairs.items():
        replay = pair.get("truck_context_replay", {"status": "unavailable"})
        if replay.get("status") != "ok":
            continue

        truck_branch = replay["truck_branch"]
        car_branch = replay["car_branch"]
        preferred_sa = _branch_to_sa(truck_branch)
        rejected_sa = _branch_to_sa(car_branch)

        aligned_steps = min(len(preferred_sa), len(rejected_sa))
        if aligned_steps < window_len:
            continue

        truck_rollout_x = np.asarray(truck_branch["rollout_x"], dtype=np.float32)
        truck_rollout_y = np.asarray(truck_branch["rollout_y"], dtype=np.float32)
        car_rollout_x = np.asarray(car_branch["rollout_x"], dtype=np.float32)
        car_rollout_y = np.asarray(car_branch["rollout_y"], dtype=np.float32)

        for start in range(0, aligned_steps - window_len + 1, stride):
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
    da = 1
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
            "stride": int(stride),
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
    return output_path


def load_preferences_into_reward_model(reward_model, preference_payload: dict) -> int:
    preferred = np.asarray(preference_payload["preferred_sa"], dtype=np.float32)
    rejected = np.asarray(preference_payload["rejected_sa"], dtype=np.float32)
    labels = np.asarray(preference_payload["labels"], dtype=np.float32)
    if preferred.ndim != 3 or rejected.ndim != 3:
        raise ValueError("preference tensors must be rank-3")
    if preferred.shape != rejected.shape:
        raise ValueError(f"preferred/rejected shape mismatch: {preferred.shape} vs {rejected.shape}")
    if preferred.shape[1] != reward_model.size_segment:
        raise ValueError(
            f"segment length mismatch: payload has {preferred.shape[1]}, reward model expects {reward_model.size_segment}"
        )
    reward_model.put_queries(preferred, rejected, labels)
    return int(preferred.shape[0])


def main():
    parser = argparse.ArgumentParser(description="Build truck-context offline preference windows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-len", type=int, default=DEFAULT_WINDOW_LEN)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    args = parser.parse_args()

    output_path = build_truck_context_preferences(
        export_path=args.input,
        output_path=args.output,
        window_len=args.window_len,
        stride=args.stride,
    )
    print(output_path)


if __name__ == "__main__":
    main()
