#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_preference_first32_deployment import (
    DEFAULT_GT_CAR,
    DEFAULT_GT_TRUCK,
    DEFAULT_POLICY_CAR,
    DEFAULT_POLICY_TRUCK,
    _index_rollouts,
    _load_rollout_payload,
    _one_hot_actions,
)
from scripts.evaluate_preference_train_val_dataset import (
    DEFAULT_CHECKPOINT_STEM,
    DEFAULT_OUTPUT_DIR as DEFAULT_TRAIN_VAL_OUTPUT_DIR,
    DEFAULT_PREFERENCE_PATH,
    DEFAULT_REWARD_DIR,
    RewardBundle,
    _jsonable,
    _load_reward_bundle,
    _score_pair_batch,
    iter_preference_shards,
)


DEFAULT_TRAIN_VAL_SCORED = DEFAULT_TRAIN_VAL_OUTPUT_DIR / "scored_preference_windows.pt"
DEFAULT_DEPLOYMENT_SCORED = Path("outputs/preference_eval/preference_first32_deployment_eval/scored_first32_deployment.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/preference_eval/preference_eval_gap_investigation")
DEFAULT_WINDOW_LEN = 32


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _load_rows(path: Path, row_key: str = "rows") -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload[row_key])


def _load_preference_segments(
    preference_path: Path,
    rows: list[dict[str, Any]],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    needed = {int(row["global_index"]) for row in rows}
    segments: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for payload, shard_info in iter_preference_shards(preference_path):
        start = int(shard_info["start_index"])
        end = int(shard_info["end_index"])
        local_indices = sorted(index - start for index in needed if start <= index < end)
        if not local_indices:
            continue
        preferred = np.asarray(payload["preferred_sa"][local_indices], dtype=np.float32)
        rejected = np.asarray(payload["rejected_sa"][local_indices], dtype=np.float32)
        for row_idx, local_index in enumerate(local_indices):
            segments[start + local_index] = (preferred[row_idx], rejected[row_idx])
    missing = sorted(needed - set(segments))
    if missing:
        raise ValueError(f"missing preference segments for global indices: {missing[:10]}")
    return segments


def _rollout_features(
    rollout: dict[str, Any],
    bundle: RewardBundle,
    *,
    start: int,
    window_len: int,
) -> np.ndarray:
    steps = int(rollout["steps"])
    if start < 0 or start + window_len > steps:
        raise ValueError(f"rollout too short for start={start}, window_len={window_len}, steps={steps}")
    observations = np.asarray(rollout["observations"][start : start + window_len], dtype=np.float32)
    actions = np.asarray(rollout["actions"][start : start + window_len], dtype=np.int64)
    action_features = _one_hot_actions(actions, bundle.action_dim)
    features = np.concatenate([observations, action_features], axis=-1).astype(np.float32)
    expected = (window_len, bundle.obs_dim + bundle.action_dim)
    if features.shape != expected:
        raise ValueError(f"rollout feature shape mismatch: expected {expected}, got {features.shape}")
    return features


def _score_features(
    preferred: np.ndarray,
    rejected: np.ndarray,
    bundle: RewardBundle,
    *,
    window_len: int,
) -> dict[str, float]:
    scored = _score_pair_batch(
        bundle,
        preferred.reshape(1, window_len, -1),
        rejected.reshape(1, window_len, -1),
        np.zeros((1, 1), dtype=np.float32),
    )
    return {
        "preferred_score": float(scored["preferred_score"][0]),
        "rejected_score": float(scored["rejected_score"][0]),
        "margin": float(scored["margin"][0]),
        "prob_preferred": float(scored["prob_preferred"][0]),
        "pair_uncertainty": float(scored["pair_uncertainty"][0]),
    }


def _feature_diff(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    diff = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    return {
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def _segment_distribution(name: str, segments: list[np.ndarray], obs_dim: int) -> dict[str, Any]:
    if not segments:
        return {"name": name, "count": 0}
    stacked = np.stack(segments, axis=0).astype(np.float32)
    obs = stacked[..., :obs_dim]
    action_one_hot = stacked[..., obs_dim:]
    action_indices = np.argmax(action_one_hot, axis=-1).reshape(-1)
    one_hot_sums = action_one_hot.sum(axis=-1).reshape(-1)
    obs_rows = obs.reshape(-1, obs_dim)
    obs_l2 = np.linalg.norm(obs_rows, axis=1)
    unique_actions, action_counts = np.unique(action_indices, return_counts=True)
    top_actions = sorted(
        (
            {"action": int(action), "fraction": float(count / action_indices.size)}
            for action, count in zip(unique_actions, action_counts)
        ),
        key=lambda item: item["fraction"],
        reverse=True,
    )[:10]
    return {
        "name": name,
        "count": int(stacked.shape[0]),
        "obs_value": _stats(obs.reshape(-1)),
        "obs_l2_per_step": _stats(obs_l2),
        "obs_zero_fraction": float(np.mean(obs == 0.0)),
        "action_index": _stats(action_indices),
        "action_one_hot_sum": _stats(one_hot_sums),
        "top_actions": top_actions,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(_jsonable(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _pair_rollout_alignment(
    left_name: str,
    right_name: str,
    left_rollouts: dict[str, dict[str, Any]],
    right_rollouts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for map_name in sorted(set(left_rollouts) & set(right_rollouts)):
        left = left_rollouts[map_name]
        right = right_rollouts[map_name]
        aligned_steps = min(int(left["steps"]), int(right["steps"]))
        left_actions = np.asarray(left["actions"][:aligned_steps], dtype=np.int64)
        right_actions = np.asarray(right["actions"][:aligned_steps], dtype=np.int64)
        left_obs = np.asarray(left["observations"][:aligned_steps], dtype=np.float32)
        right_obs = np.asarray(right["observations"][:aligned_steps], dtype=np.float32)
        rows.append(
            {
                "comparison": f"{left_name}_vs_{right_name}",
                "map_name": map_name,
                "aligned_steps": int(aligned_steps),
                "left_steps": int(left["steps"]),
                "right_steps": int(right["steps"]),
                "action_equal_fraction": float(np.mean(left_actions == right_actions)) if aligned_steps else 0.0,
                "action_diff_count": int(np.sum(left_actions != right_actions)),
                "observation_mean_abs_diff": float(np.mean(np.abs(left_obs - right_obs))) if aligned_steps else 0.0,
                "left_first_actions": left_actions[:8].tolist(),
                "right_first_actions": right_actions[:8].tolist(),
                "left_scenario_type": left.get("scenario_type", ""),
                "right_scenario_type": right.get("scenario_type", ""),
                "left_delta_heading_deg": left.get("delta_heading_deg"),
                "right_delta_heading_deg": right.get("delta_heading_deg"),
            }
        )
    return rows


def investigate(
    *,
    preference_path: Path,
    train_val_scored: Path,
    deployment_scored: Path,
    policy_car: Path,
    policy_truck: Path,
    gt_car: Path,
    gt_truck: Path,
    reward_dir: Path,
    checkpoint_stem: str,
    output_dir: Path,
    window_len: int,
    device: str,
) -> dict[str, Any]:
    bundle = _load_reward_bundle(reward_dir, checkpoint_stem=checkpoint_stem, device=device)
    train_val_rows = _load_rows(train_val_scored)
    deployment_rows = _load_rows(deployment_scored)
    train_val_by_map = {str(row["map_name"]): row for row in train_val_rows}

    rollout_payloads = {
        "policy_car": _load_rollout_payload(policy_car),
        "policy_truck": _load_rollout_payload(policy_truck),
        "gt_car": _load_rollout_payload(gt_car),
        "gt_truck": _load_rollout_payload(gt_truck),
    }
    rollout_indices = {key: _index_rollouts(payload) for key, payload in rollout_payloads.items()}
    pair_alignment_rows = (
        _pair_rollout_alignment("gt_truck", "gt_car", rollout_indices["gt_truck"], rollout_indices["gt_car"])
        + _pair_rollout_alignment(
            "policy_truck",
            "policy_car",
            rollout_indices["policy_truck"],
            rollout_indices["policy_car"],
        )
    )

    def _source_name(display_name: str) -> str:
        for key in ("gt_car", "gt_truck", "policy_car", "policy_truck"):
            rollout = rollout_indices[key].get(display_name)
            if rollout is not None and rollout.get("map_path"):
                return Path(str(rollout["map_path"])).name
        return display_name

    deployment_display_names = sorted({str(row["map_name"]) for row in deployment_rows})
    deployment_source_by_display = {name: _source_name(name) for name in deployment_display_names}
    overlapping_displays = [
        display_name
        for display_name, source_name in deployment_source_by_display.items()
        if source_name in train_val_by_map
    ]
    overlapping_rows = [train_val_by_map[deployment_source_by_display[name]] for name in overlapping_displays]
    preference_segments = _load_preference_segments(preference_path, overlapping_rows)

    deployment_gt_t0_by_map = {
        str(row["map_name"]): row
        for row in deployment_rows
        if row.get("comparison") == "gt_truck_context_vs_car_fit" and row.get("status") == "ok"
    }
    deployment_policy_t0_by_map = {
        str(row["map_name"]): row
        for row in deployment_rows
        if row.get("comparison") == "policy_truck_vs_car" and row.get("status") == "ok"
    }

    alignment_rows: list[dict[str, Any]] = []
    distribution_segments: dict[str, list[np.ndarray]] = {
        "dataset_preferred_overlap": [],
        "dataset_rejected_overlap": [],
        "gt_truck_t0": [],
        "gt_car_t0": [],
        "gt_truck_preference_start": [],
        "gt_car_preference_start": [],
        "policy_truck_t0": [],
        "policy_car_t0": [],
    }

    for display_map_name in overlapping_displays:
        source_map_name = deployment_source_by_display[display_map_name]
        row = train_val_by_map[source_map_name]
        global_index = int(row["global_index"])
        preference_start = int(row["timestep_start"])
        preference_end = int(row["timestep_end"])
        dataset_preferred, dataset_rejected = preference_segments[global_index]
        distribution_segments["dataset_preferred_overlap"].append(dataset_preferred)
        distribution_segments["dataset_rejected_overlap"].append(dataset_rejected)

        out: dict[str, Any] = {
            "map_name": display_map_name,
            "source_map_name": source_map_name,
            "split": row["split"],
            "global_index": global_index,
            "preference_timestep_start": preference_start,
            "preference_timestep_end": preference_end,
            "preference_scenario_delta_heading_deg": row.get("scenario_delta_heading_deg"),
            "dataset_margin": float(row["margin"]),
            "dataset_prob_truck": float(row["prob_preferred"]),
            "dataset_truck_score": float(row["preferred_score"]),
            "dataset_car_score": float(row["rejected_score"]),
        }

        gt_t0 = deployment_gt_t0_by_map.get(display_map_name)
        if gt_t0 is not None:
            out.update(
                {
                    "gt_t0_margin": float(gt_t0["margin"]),
                    "gt_t0_prob_truck": float(gt_t0["prob_preferred"]),
                    "gt_t0_uncertainty": float(gt_t0["pair_uncertainty"]),
                }
            )
        try:
            gt_truck_t0 = _rollout_features(
                rollout_indices["gt_truck"][display_map_name], bundle, start=0, window_len=window_len
            )
            gt_car_t0 = _rollout_features(
                rollout_indices["gt_car"][display_map_name], bundle, start=0, window_len=window_len
            )
            distribution_segments["gt_truck_t0"].append(gt_truck_t0)
            distribution_segments["gt_car_t0"].append(gt_car_t0)
        except Exception as exc:
            out["gt_t0_feature_error"] = str(exc)

        try:
            gt_truck_pref = _rollout_features(
                rollout_indices["gt_truck"][display_map_name],
                bundle,
                start=preference_start,
                window_len=window_len,
            )
            gt_car_pref = _rollout_features(
                rollout_indices["gt_car"][display_map_name],
                bundle,
                start=preference_start,
                window_len=window_len,
            )
            distribution_segments["gt_truck_preference_start"].append(gt_truck_pref)
            distribution_segments["gt_car_preference_start"].append(gt_car_pref)
            gt_pref_score = _score_features(gt_truck_pref, gt_car_pref, bundle, window_len=window_len)
            out.update(
                {
                    "gt_at_preference_start_margin": gt_pref_score["margin"],
                    "gt_at_preference_start_prob_truck": gt_pref_score["prob_preferred"],
                    "gt_at_preference_start_uncertainty": gt_pref_score["pair_uncertainty"],
                    "dataset_vs_gt_pref_truck_mean_abs": _feature_diff(dataset_preferred, gt_truck_pref)["mean_abs"],
                    "dataset_vs_gt_pref_truck_max_abs": _feature_diff(dataset_preferred, gt_truck_pref)["max_abs"],
                    "dataset_vs_gt_pref_car_mean_abs": _feature_diff(dataset_rejected, gt_car_pref)["mean_abs"],
                    "dataset_vs_gt_pref_car_max_abs": _feature_diff(dataset_rejected, gt_car_pref)["max_abs"],
                }
            )
        except Exception as exc:
            out["gt_at_preference_start_error"] = str(exc)

        policy_t0 = deployment_policy_t0_by_map.get(display_map_name)
        if policy_t0 is not None:
            out.update(
                {
                    "policy_t0_margin": float(policy_t0["margin"]),
                    "policy_t0_prob_truck": float(policy_t0["prob_preferred"]),
                    "policy_t0_uncertainty": float(policy_t0["pair_uncertainty"]),
                }
            )
        try:
            policy_truck_t0 = _rollout_features(
                rollout_indices["policy_truck"][display_map_name], bundle, start=0, window_len=window_len
            )
            policy_car_t0 = _rollout_features(
                rollout_indices["policy_car"][display_map_name], bundle, start=0, window_len=window_len
            )
            distribution_segments["policy_truck_t0"].append(policy_truck_t0)
            distribution_segments["policy_car_t0"].append(policy_car_t0)
        except Exception as exc:
            out["policy_t0_feature_error"] = str(exc)

        try:
            policy_truck_pref = _rollout_features(
                rollout_indices["policy_truck"][display_map_name],
                bundle,
                start=preference_start,
                window_len=window_len,
            )
            policy_car_pref = _rollout_features(
                rollout_indices["policy_car"][display_map_name],
                bundle,
                start=preference_start,
                window_len=window_len,
            )
            policy_pref_score = _score_features(policy_truck_pref, policy_car_pref, bundle, window_len=window_len)
            out.update(
                {
                    "policy_at_preference_start_margin": policy_pref_score["margin"],
                    "policy_at_preference_start_prob_truck": policy_pref_score["prob_preferred"],
                    "policy_at_preference_start_uncertainty": policy_pref_score["pair_uncertainty"],
                }
            )
        except Exception as exc:
            out["policy_at_preference_start_error"] = str(exc)

        for key in ("gt_truck", "gt_car", "policy_truck", "policy_car"):
            rollout = rollout_indices[key].get(display_map_name)
            if rollout is not None:
                out[f"{key}_steps"] = int(rollout["steps"])
                out[f"{key}_scenario_type"] = rollout.get("scenario_type", "")
                out[f"{key}_delta_heading_deg"] = rollout.get("delta_heading_deg")
        if out.get("preference_scenario_delta_heading_deg") is not None and out.get("gt_truck_delta_heading_deg") is not None:
            out["abs_preference_vs_gt_delta_heading_diff_deg"] = abs(
                float(out["preference_scenario_delta_heading_deg"]) - float(out["gt_truck_delta_heading_deg"])
            )
        alignment_rows.append(out)

    train_starts = [int(row["timestep_start"]) for row in train_val_rows if row["split"] == "train"]
    validation_starts = [int(row["timestep_start"]) for row in train_val_rows if row["split"] == "validation"]
    overlap_starts = [int(train_val_by_map[deployment_source_by_display[name]]["timestep_start"]) for name in overlapping_displays]

    distribution_summary = {
        name: _segment_distribution(name, segments, bundle.obs_dim)
        for name, segments in distribution_segments.items()
    }
    comparison_summary = {
        "deployment_map_count": len(deployment_display_names),
        "overlap_with_train_val_first_window_maps": len(overlapping_displays),
        "deployment_maps": deployment_display_names,
        "deployment_source_by_display_map": deployment_source_by_display,
        "overlapping_display_maps": overlapping_displays,
        "overlapping_source_maps": [deployment_source_by_display[name] for name in overlapping_displays],
        "preference_window_start": {
            "train": _stats(train_starts),
            "validation": _stats(validation_starts),
            "deployment_overlap": _stats(overlap_starts),
            "train_fraction_start_zero": float(np.mean(np.asarray(train_starts) == 0)) if train_starts else 0.0,
            "validation_fraction_start_zero": float(np.mean(np.asarray(validation_starts) == 0)) if validation_starts else 0.0,
            "deployment_overlap_fraction_start_zero": float(np.mean(np.asarray(overlap_starts) == 0)) if overlap_starts else 0.0,
        },
        "margins": {
            "dataset_overlap": _stats([row["dataset_margin"] for row in alignment_rows]),
            "gt_t0": _stats([_safe_float(row.get("gt_t0_margin")) for row in alignment_rows]),
            "gt_at_preference_start": _stats(
                [_safe_float(row.get("gt_at_preference_start_margin")) for row in alignment_rows]
            ),
            "policy_t0": _stats([_safe_float(row.get("policy_t0_margin")) for row in alignment_rows]),
            "policy_at_preference_start": _stats(
                [_safe_float(row.get("policy_at_preference_start_margin")) for row in alignment_rows]
            ),
        },
        "prob_truck": {
            "dataset_overlap": _stats([row["dataset_prob_truck"] for row in alignment_rows]),
            "gt_t0": _stats([_safe_float(row.get("gt_t0_prob_truck")) for row in alignment_rows]),
            "gt_at_preference_start": _stats(
                [_safe_float(row.get("gt_at_preference_start_prob_truck")) for row in alignment_rows]
            ),
            "policy_t0": _stats([_safe_float(row.get("policy_t0_prob_truck")) for row in alignment_rows]),
            "policy_at_preference_start": _stats(
                [_safe_float(row.get("policy_at_preference_start_prob_truck")) for row in alignment_rows]
            ),
        },
        "feature_diffs_dataset_vs_gt_preference_start": {
            "truck_mean_abs": _stats([_safe_float(row.get("dataset_vs_gt_pref_truck_mean_abs")) for row in alignment_rows]),
            "truck_max_abs": _stats([_safe_float(row.get("dataset_vs_gt_pref_truck_max_abs")) for row in alignment_rows]),
            "car_mean_abs": _stats([_safe_float(row.get("dataset_vs_gt_pref_car_mean_abs")) for row in alignment_rows]),
            "car_max_abs": _stats([_safe_float(row.get("dataset_vs_gt_pref_car_max_abs")) for row in alignment_rows]),
        },
        "scenario_identity_checks": {
            "abs_preference_vs_gt_delta_heading_diff_deg": _stats(
                [_safe_float(row.get("abs_preference_vs_gt_delta_heading_diff_deg")) for row in alignment_rows]
            )
        },
        "rollout_pair_alignment": {
            "gt_truck_vs_gt_car_action_equal_fraction": _stats(
                [
                    row["action_equal_fraction"]
                    for row in pair_alignment_rows
                    if row["comparison"] == "gt_truck_vs_gt_car"
                ]
            ),
            "gt_truck_vs_gt_car_observation_mean_abs_diff": _stats(
                [
                    row["observation_mean_abs_diff"]
                    for row in pair_alignment_rows
                    if row["comparison"] == "gt_truck_vs_gt_car"
                ]
            ),
            "policy_truck_vs_policy_car_action_equal_fraction": _stats(
                [
                    row["action_equal_fraction"]
                    for row in pair_alignment_rows
                    if row["comparison"] == "policy_truck_vs_policy_car"
                ]
            ),
            "policy_truck_vs_policy_car_observation_mean_abs_diff": _stats(
                [
                    row["observation_mean_abs_diff"]
                    for row in pair_alignment_rows
                    if row["comparison"] == "policy_truck_vs_policy_car"
                ]
            ),
        },
        "distribution_summary": distribution_summary,
    }

    report_dir = output_dir / "report"
    _write_csv(alignment_rows, report_dir / "deployment_map_alignment.csv")
    _write_csv(pair_alignment_rows, report_dir / "rollout_pair_alignment.csv")
    summary_path = report_dir / "gap_investigation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_jsonable(comparison_summary), indent=2), encoding="utf-8")

    md_path = report_dir / "gap_investigation_summary.md"
    lines = [
        "# Preference Eval Gap Investigation",
        "",
        f"- Deployment maps: `{len(deployment_display_names)}`",
        f"- Deployment maps present in train/val first-window table by `map_path` basename: `{len(overlapping_displays)}`",
        f"- Validation first-window start median: `{comparison_summary['preference_window_start']['validation']['median']:.1f}`",
        f"- Deployment-overlap first-window start median: `{comparison_summary['preference_window_start']['deployment_overlap']['median']:.1f}`",
        f"- Dataset overlap mean margin: `{comparison_summary['margins']['dataset_overlap']['mean']:.3f}`",
        f"- GT deployment t=0 mean margin: `{comparison_summary['margins']['gt_t0']['mean']:.3f}`",
        f"- GT deployment at preference start mean margin: `{comparison_summary['margins']['gt_at_preference_start']['mean']:.3f}`",
        f"- Policy deployment t=0 mean margin: `{comparison_summary['margins']['policy_t0']['mean']:.3f}`",
        f"- Policy deployment at preference start mean margin: `{comparison_summary['margins']['policy_at_preference_start']['mean']:.3f}`",
        f"- Nominal overlap delta-heading abs diff mean: `{comparison_summary['scenario_identity_checks']['abs_preference_vs_gt_delta_heading_diff_deg']['mean']:.3f}` degrees",
        f"- GT truck/car rollout action equality mean: `{comparison_summary['rollout_pair_alignment']['gt_truck_vs_gt_car_action_equal_fraction']['mean']:.3f}`",
        f"- Policy truck/car rollout action equality mean: `{comparison_summary['rollout_pair_alignment']['policy_truck_vs_policy_car_action_equal_fraction']['mean']:.3f}`",
        "",
        "Positive margin means truck/context is scored above car. See `deployment_map_alignment.csv` and "
        "`rollout_pair_alignment.csv` for per-map details.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "summary_json": str(summary_path),
        "summary_md": str(md_path),
        "alignment_csv": str(report_dir / "deployment_map_alignment.csv"),
        "rollout_alignment_csv": str(report_dir / "rollout_pair_alignment.csv"),
        "summary": comparison_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigate train/validation vs deployment preference-eval gap.")
    parser.add_argument("--preference-path", type=Path, default=DEFAULT_PREFERENCE_PATH)
    parser.add_argument("--train-val-scored", type=Path, default=DEFAULT_TRAIN_VAL_SCORED)
    parser.add_argument("--deployment-scored", type=Path, default=DEFAULT_DEPLOYMENT_SCORED)
    parser.add_argument("--policy-car", type=Path, default=DEFAULT_POLICY_CAR)
    parser.add_argument("--policy-truck", type=Path, default=DEFAULT_POLICY_TRUCK)
    parser.add_argument("--gt-car", type=Path, default=DEFAULT_GT_CAR)
    parser.add_argument("--gt-truck", type=Path, default=DEFAULT_GT_TRUCK)
    parser.add_argument("--reward-dir", type=Path, default=DEFAULT_REWARD_DIR)
    parser.add_argument("--checkpoint-stem", type=str, default=DEFAULT_CHECKPOINT_STEM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-len", type=int, default=DEFAULT_WINDOW_LEN)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = investigate(
        preference_path=args.preference_path,
        train_val_scored=args.train_val_scored,
        deployment_scored=args.deployment_scored,
        policy_car=args.policy_car,
        policy_truck=args.policy_truck,
        gt_car=args.gt_car,
        gt_truck=args.gt_truck,
        reward_dir=args.reward_dir,
        checkpoint_stem=args.checkpoint_stem,
        output_dir=args.output_dir,
        window_len=args.window_len,
        device=args.device,
    )
    printable = {key: value for key, value in outputs.items() if key != "summary"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
