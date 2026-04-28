#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PAIRED_FITS = Path(
    "/home/casko/phd-code/pufferdrive-kth/"
    "pufferlib/resources/drive/preferences/offline_fits/nuplan_boston_all_chunk64_paired_offline_fits.pt"
)
DEFAULT_REFERENCE_GT_CAR = Path(
    "outputs/preference_eval/preference_ground_truth_context_compare_full/rollouts/"
    "ground-truth-car-fit_rollouts.pt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/preference_eval/preference_ground_truth_context_compare_full/rollouts")
DEFAULT_TRUCK_OUTPUT_NAME = "ground-truth-truck-context-preferred-branch_rollouts.pt"
DEFAULT_CAR_OUTPUT_NAME = "ground-truth-car-context-rejected-branch_rollouts.pt"


def _resolve_shard_path(paired_fits_path: Path, shard_info: dict[str, Any]) -> Path:
    raw = Path(shard_info["path"])
    if raw.exists():
        return raw
    same_dir = paired_fits_path.parent / raw.name
    sibling_dir = paired_fits_path.parent / f"{paired_fits_path.stem}_shards" / raw.name
    if same_dir.exists():
        return same_dir
    if sibling_dir.exists():
        return sibling_dir
    return raw


def _load_pairs_for_sources(paired_fits_path: Path, source_names: set[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    manifest = torch.load(paired_fits_path, map_location="cpu", weights_only=False)
    shared_maps = list(manifest.get("metadata", {}).get("shared_maps", []))
    source_to_index = {name: shared_maps.index(name) for name in source_names if name in shared_maps}
    needed_shards = []
    for shard in manifest.get("shards", []):
        start = int(shard["start_index"])
        end = int(shard["end_index"])
        if any(start <= index < end for index in source_to_index.values()):
            needed_shards.append(shard)

    pairs: dict[str, dict[str, Any]] = {}
    remaining = set(source_to_index)
    for shard in needed_shards:
        shard_path = _resolve_shard_path(paired_fits_path, shard)
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_pairs = payload.get("pairs", {})
        for source_name in list(remaining):
            if source_name in shard_pairs:
                pairs[source_name] = shard_pairs[source_name]
                remaining.remove(source_name)
        if not remaining:
            break
    missing = sorted(source_names - set(pairs))
    return pairs, missing


def _as_step_array(values: Any, steps: int, *, dtype=np.float32, default: float = 0.0) -> np.ndarray:
    if values is None:
        return np.full(steps, default, dtype=dtype)
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.size == steps:
        return arr.copy()
    if arr.size > steps:
        return arr[:steps].copy()
    if arr.size == 0:
        return np.full(steps, default, dtype=dtype)
    pad = np.full(steps - arr.size, arr[-1], dtype=dtype)
    return np.concatenate([arr, pad]).astype(dtype, copy=False)


def _trajectory_steps(branch: dict[str, Any], steps: int, key: str) -> np.ndarray:
    values = branch.get(key)
    return _as_step_array(values, steps, dtype=np.float32)


def _build_rollout(
    *,
    template: dict[str, Any],
    source_map_name: str,
    pair: dict[str, Any],
    branch_name: str,
    side_name: str,
) -> dict[str, Any]:
    replay = pair["truck_context_replay"]
    branch = replay[branch_name]
    side = pair.get(side_name, {})
    observations = np.asarray(branch["obs_default"], dtype=np.float32)
    actions = np.asarray(branch["actions"], dtype=np.int32).reshape(-1)
    steps = min(int(observations.shape[0]), int(actions.shape[0]))
    observations = observations[:steps].copy()
    actions = actions[:steps].copy()

    rollout = {
        "map_name": template["map_name"],
        "source_map_name": source_map_name,
        "map_path": template.get("map_path", ""),
        "scenario_type": template.get("scenario_type", ""),
        "delta_heading_deg": template.get("delta_heading_deg"),
        "steps": int(steps),
        "stop_reason": f"paired_fit_{branch_name}",
        "observations": observations,
        "actions": actions,
        "task_rewards": np.zeros(steps, dtype=np.float32),
        "dones": np.zeros(steps, dtype=np.uint8),
        "truncs": np.zeros(steps, dtype=np.uint8),
        "values": np.zeros(steps, dtype=np.float32),
        "entropy": np.zeros(steps, dtype=np.float32),
        "x": _trajectory_steps(branch, steps, "rollout_x"),
        "y": _trajectory_steps(branch, steps, "rollout_y"),
        "z": _as_step_array(template.get("z"), steps, dtype=np.float32),
        "heading": _trajectory_steps(branch, steps, "rollout_heading"),
        "length": _as_step_array(template.get("length"), steps, dtype=np.float32),
        "width": _as_step_array(template.get("width"), steps, dtype=np.float32),
        "trailer_has": _as_step_array(template.get("trailer_has"), steps, dtype=np.float32),
        "trailer_x": _as_step_array(template.get("trailer_x"), steps, dtype=np.float32),
        "trailer_y": _as_step_array(template.get("trailer_y"), steps, dtype=np.float32),
        "trailer_heading": _as_step_array(template.get("trailer_heading"), steps, dtype=np.float32),
        "paired_fit_branch": f"truck_context_replay.{branch_name}",
        "paired_fit_side": side_name,
        "paired_fit_source_map": source_map_name,
        "self_ade": branch.get("self_ade"),
        "self_fde": branch.get("self_fde"),
    }
    for key in (
        "match_cost_total",
        "match_cost_lateral_total",
        "match_cost_longitudinal_total",
        "match_cost_step",
        "match_cost_lateral_step",
        "match_cost_longitudinal_step",
    ):
        fit_key = f"fit_{key}" if not key.startswith("fit_") else key
        if key in side:
            value = side[key]
            rollout[fit_key] = _as_step_array(value, steps, dtype=np.float32) if str(key).endswith("_step") else float(value)
    return rollout


def _source_name(rollout: dict[str, Any]) -> str:
    if rollout.get("source_map_name"):
        return str(rollout["source_map_name"])
    if rollout.get("map_path"):
        return Path(str(rollout["map_path"])).name
    return str(rollout["map_name"])


def rebuild_rollouts(
    *,
    paired_fits: Path,
    reference_gt_car: Path,
    truck_output: Path,
    car_output: Path,
) -> dict[str, Any]:
    reference_payload = torch.load(reference_gt_car, map_location="cpu", weights_only=False)
    reference_rollouts = list(reference_payload.get("rollouts", []))
    source_names = {_source_name(rollout) for rollout in reference_rollouts}
    pairs, missing = _load_pairs_for_sources(paired_fits, source_names)

    truck_rollouts = []
    car_rollouts = []
    report_rows = []
    for template in reference_rollouts:
        source_name = _source_name(template)
        pair = pairs.get(source_name)
        if pair is None:
            report_rows.append(
                {
                    "map_name": template.get("map_name"),
                    "source_map_name": source_name,
                    "status": "missing_pair",
                }
            )
            continue
        replay = pair.get("truck_context_replay", {})
        if replay.get("status") != "ok":
            report_rows.append(
                {
                    "map_name": template.get("map_name"),
                    "source_map_name": source_name,
                    "status": "missing_truck_context_replay",
                    "error": replay.get("error", ""),
                }
            )
            continue
        truck_rollout = _build_rollout(
            template=template,
            source_map_name=source_name,
            pair=pair,
            branch_name="truck_branch",
            side_name="truck",
        )
        car_rollout = _build_rollout(
            template=template,
            source_map_name=source_name,
            pair=pair,
            branch_name="car_branch",
            side_name="car",
        )
        truck_rollouts.append(truck_rollout)
        car_rollouts.append(car_rollout)
        n = min(32, truck_rollout["steps"], car_rollout["steps"])
        report_rows.append(
            {
                "map_name": template.get("map_name"),
                "source_map_name": source_name,
                "status": "ok",
                "truck_steps": int(truck_rollout["steps"]),
                "car_steps": int(car_rollout["steps"]),
                "first32_action_equal_fraction": float(
                    np.mean(truck_rollout["actions"][:n] == car_rollout["actions"][:n])
                )
                if n
                else 0.0,
                "truck_first_actions": truck_rollout["actions"][:8].astype(int).tolist(),
                "car_first_actions": car_rollout["actions"][:8].astype(int).tolist(),
            }
        )

    truck_payload = {
        **{key: value for key, value in reference_payload.items() if key != "rollouts"},
        "format": "drive_preference_rollouts_v1",
        "model_name": "ground-truth-truck-context-preferred-branch",
        "model_root": "paired-fit-truck-context-replay",
        "checkpoint_path": "",
        "source_paired_fits": str(paired_fits),
        "reference_gt_car": str(reference_gt_car),
        "corrected_branch": "truck_context_replay.truck_branch",
        "rollouts": truck_rollouts,
    }
    car_payload = {
        **{key: value for key, value in reference_payload.items() if key != "rollouts"},
        "format": "drive_preference_rollouts_v1",
        "model_name": "ground-truth-car-context-rejected-branch",
        "model_root": "paired-fit-truck-context-replay",
        "checkpoint_path": "",
        "source_paired_fits": str(paired_fits),
        "reference_gt_car": str(reference_gt_car),
        "corrected_branch": "truck_context_replay.car_branch",
        "rollouts": car_rollouts,
    }

    truck_output.parent.mkdir(parents=True, exist_ok=True)
    car_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(truck_payload, truck_output)
    torch.save(car_payload, car_output)
    report_path = truck_output.parent / "corrected_gt_context_rollouts_report.json"
    report = {
        "paired_fits": str(paired_fits),
        "reference_gt_car": str(reference_gt_car),
        "truck_output": str(truck_output),
        "car_output": str(car_output),
        "requested_source_count": len(source_names),
        "written_truck_rollouts": len(truck_rollouts),
        "written_car_rollouts": len(car_rollouts),
        "missing_sources": missing,
        "rows": report_rows,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "report": str(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild GT context rollouts from paired-fit preference branches.")
    parser.add_argument("--paired-fits", type=Path, default=DEFAULT_PAIRED_FITS)
    parser.add_argument("--reference-gt-car", type=Path, default=DEFAULT_REFERENCE_GT_CAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--truck-output-name", type=str, default=DEFAULT_TRUCK_OUTPUT_NAME)
    parser.add_argument("--car-output-name", type=str, default=DEFAULT_CAR_OUTPUT_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    outputs = rebuild_rollouts(
        paired_fits=args.paired_fits,
        reference_gt_car=args.reference_gt_car,
        truck_output=output_dir / args.truck_output_name,
        car_output=output_dir / args.car_output_name,
    )
    print(json.dumps({key: value for key, value in outputs.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
