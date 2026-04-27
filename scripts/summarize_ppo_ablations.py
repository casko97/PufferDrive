#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PPO ablation training status and validation results.")
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=Path("experiments_ppo_validation"),
        help="Root directory containing validation summary.json files.",
    )
    parser.add_argument(
        "--run-roots",
        type=Path,
        nargs="*",
        default=[
            Path("experiments_ppo_validation8192"),
            Path("experiments_ppo_validation8192_conservative"),
            Path("experiments_ppo_validation8192_std015"),
            Path("experiments_ppo_ablation4096_conservative"),
            Path("experiments_ppo_ablation4096_std0125"),
            Path("experiments_ppo_ablation4096_lr1e4"),
            Path("experiments_ppo_full_observable"),
            Path("experiments_ppo_smoke"),
        ],
        help="Run roots to scan for trainer_state.pt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments_ppo_reports/latest"),
        help="Directory where the report files will be written.",
    )
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _checkpoint_index(path: str | Path | None) -> int | None:
    if path is None:
        return None
    stem = Path(path).stem
    if "_" not in stem:
        return None
    suffix = stem.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _load_config_hparams(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}

    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    with config_path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)

    def get(section: str, option: str) -> Any:
        if not parser.has_option(section, option):
            return None
        return parser.get(section, option)

    return {
        "config_name": config_path.name,
        "num_maps": _as_int(get("env", "num_maps")),
        "total_timesteps": _as_int(get("train", "total_timesteps")),
        "batch_size": _as_int(get("train", "batch_size")),
        "checkpoint_interval": _as_int(get("train", "checkpoint_interval")),
        "eval_interval": _as_int(get("eval", "eval_interval")),
        "max_action_std": _as_float(get("policy", "max_action_std")),
        "ent_coef": _as_float(get("train", "ent_coef")),
        "learning_rate": _as_float(get("train", "learning_rate")),
        "clip_coef": _as_float(get("train", "clip_coef")),
        "data_dir": get("train", "data_dir"),
        "project": get("train", "project"),
    }


def collect_validation_rows(validation_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(validation_root.glob("**/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        ppo_checkpoint = Path(payload["ppo_checkpoint"]).resolve()
        run_dir = ppo_checkpoint.parent
        config_path = Path(payload["config_path"]).resolve() if payload.get("config_path") else None
        hparams = _load_config_hparams(config_path)

        bc = payload.get("per_model_summary", {}).get("bc", {})
        warmstart = payload.get("per_model_summary", {}).get("warmstart", {})
        ppo = payload.get("per_model_summary", {}).get("ppo", {})
        validation = payload.get("validation", {})
        weight = payload.get("weight_summary", {})
        map_paths = payload.get("map_paths", [])

        row = {
            "summary_path": str(summary_path.resolve()),
            "validation_name": summary_path.parent.name,
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "ppo_checkpoint": str(ppo_checkpoint),
            "checkpoint_name": ppo_checkpoint.name,
            "checkpoint_index": _checkpoint_index(ppo_checkpoint),
            "config_path": str(config_path) if config_path else "",
            "eval_map_count": len(map_paths),
            "bc_mean_ade": bc.get("mean_ade"),
            "bc_mean_fde": bc.get("mean_fde"),
            "bc_mean_total_reward": bc.get("mean_total_reward"),
            "ppo_mean_ade": ppo.get("mean_ade"),
            "ppo_mean_fde": ppo.get("mean_fde"),
            "ppo_mean_total_reward": ppo.get("mean_total_reward"),
            "ppo_mean_final_time_s": ppo.get("mean_final_time_s"),
            "ppo_minus_bc_ade": None if bc.get("mean_ade") is None or ppo.get("mean_ade") is None else float(ppo["mean_ade"]) - float(bc["mean_ade"]),
            "ppo_minus_bc_fde": None if bc.get("mean_fde") is None or ppo.get("mean_fde") is None else float(ppo["mean_fde"]) - float(bc["mean_fde"]),
            "ppo_minus_bc_reward": None if bc.get("mean_total_reward") is None or ppo.get("mean_total_reward") is None else float(ppo["mean_total_reward"]) - float(bc["mean_total_reward"]),
            "actor_updated_backbone_frozen": validation.get("actor_updated_backbone_frozen"),
            "ppo_improves_over_bc": validation.get("ppo_improves_over_bc"),
            "ppo_improves_over_warmstart": validation.get("ppo_improves_over_warmstart"),
            "no_regression_stop_behavior": validation.get("no_regression_stop_behavior"),
            "overall_pass": validation.get("overall_pass"),
            "actor_changed_params": weight.get("actor", {}).get("changed_params"),
            "actor_total_params": weight.get("actor", {}).get("total_params"),
            "value_changed_params": weight.get("value_fn", {}).get("changed_params"),
            "backbone_unchanged": weight.get("backbone", {}).get("unchanged"),
            "ppo_stop_reason_counts": json.dumps(ppo.get("stop_reason_counts", {}), sort_keys=True),
            "bc_stop_reason_counts": json.dumps(bc.get("stop_reason_counts", {}), sort_keys=True),
        }
        row.update(hparams)
        rows.append(row)
    return rows


def collect_run_status_rows(run_roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in run_roots:
        if not root.exists():
            continue
        for trainer_state_path in sorted(root.glob("**/trainer_state.pt")):
            run_dir = trainer_state_path.parent
            state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
            checkpoint_paths = sorted(run_dir.glob("model_puffer_drive_*.pt"))
            rows.append(
                {
                    "run_root": str(root),
                    "run_dir": str(run_dir.resolve()),
                    "run_name": run_dir.name,
                    "trainer_state_path": str(trainer_state_path.resolve()),
                    "global_step": state.get("global_step"),
                    "update": state.get("update"),
                    "latest_model_name": state.get("model_name"),
                    "saved_checkpoint_count": len(checkpoint_paths),
                    "latest_checkpoint_index": _checkpoint_index(state.get("model_name")),
                }
            )
    return rows


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        improve = 1 if row.get("ppo_improves_over_bc") else 0
        no_regress = 1 if row.get("no_regression_stop_behavior") else 0
        overall = 1 if row.get("overall_pass") else 0
        delta_fde = row.get("ppo_minus_bc_fde")
        delta_ade = row.get("ppo_minus_bc_ade")
        map_count = row.get("eval_map_count") or 0
        ckpt = row.get("checkpoint_index") or -1
        return (
            overall,
            improve,
            no_regress,
            map_count,
            -(delta_fde if delta_fde is not None else float("inf")),
            -(delta_ade if delta_ade is not None else float("inf")),
            -ckpt,
        )

    return max(rows, key=key)


def build_best_by_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["run_name"]].append(row)

    payload: dict[str, Any] = {}
    for run_name, run_rows in sorted(grouped.items()):
        latest = max(run_rows, key=lambda item: item.get("checkpoint_index") or -1)
        best = _best_row(run_rows)
        payload[run_name] = {
            "run_dir": latest["run_dir"],
            "config_path": latest["config_path"],
            "num_validated_checkpoints": len(run_rows),
            "latest": latest,
            "best": best,
        }
    return payload


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(payload: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validation_root = args.validation_root.expanduser().resolve()
    run_roots = [path.expanduser().resolve() for path in args.run_roots]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_rows = collect_validation_rows(validation_root)
    run_status_rows = collect_run_status_rows(run_roots)
    best_by_run = build_best_by_run(validation_rows)

    write_csv(validation_rows, output_dir / "validation_leaderboard.csv")
    write_csv(run_status_rows, output_dir / "run_status.csv")
    write_json(best_by_run, output_dir / "best_by_run.json")

    overview = {
        "validation_root": str(validation_root),
        "run_roots": [str(path) for path in run_roots],
        "num_validation_summaries": len(validation_rows),
        "num_runs_with_trainer_state": len(run_status_rows),
        "runs": sorted(best_by_run.keys()),
        "artifacts": {
            "validation_leaderboard_csv": str(output_dir / "validation_leaderboard.csv"),
            "run_status_csv": str(output_dir / "run_status.csv"),
            "best_by_run_json": str(output_dir / "best_by_run.json"),
        },
    }
    write_json(overview, output_dir / "overview.json")
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
