#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch per-run metrics from W&B for a list of run names."
    )
    parser.add_argument("--project", required=True, help="W&B project, e.g. user/proj")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run names (e.g. peachy-silence-225)",
    )
    parser.add_argument(
        "--score-key",
        default="environment/score",
        help="History key for environment score",
    )
    parser.add_argument(
        "--target-step",
        type=int,
        default=90_000_000,
        help="Target step for score lookup",
    )
    parser.add_argument(
        "--power-window-min",
        type=int,
        default=10,
        help="Start time (minutes) for GPU power averaging",
    )
    parser.add_argument(
        "--power-window-max",
        type=int,
        default=20,
        help="End time (minutes) for GPU power averaging",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="History samples per stream (lower = faster)",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path",
    )
    return parser.parse_args()


def to_iso(ts) -> str:
    if not ts:
        return ""
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(
                timezone.utc
            ).isoformat()
        except ValueError:
            return ts
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc).isoformat()
    return str(ts)


def find_power_keys(sample_row: dict) -> tuple[str | None, str | None]:
    power_percent_key = None
    power_watts_key = None
    for key in sample_row.keys():
        k = key.lower()
        if "gpu" not in k:
            continue
        if "power" in k and "percent" in k:
            power_percent_key = key
        if "power" in k and ("watt" in k or "watts" in k):
            power_watts_key = key
    return power_percent_key, power_watts_key


def fetch_run(api: wandb.Api, project: str, run_name: str):
    # Try direct id/name lookup first.
    try:
        return api.run(f"{project}/{run_name}")
    except Exception:
        pass

    # Fall back to searching by display name.
    runs = api.runs(project, filters={"display_name": run_name})
    if runs:
        return runs[0]

    # Some older runs may use "name" rather than display_name.
    runs = api.runs(project, filters={"name": run_name})
    if runs:
        return runs[0]

    raise ValueError(f"Could not find run by name: {run_name}")


def compute_score_at_step(run, score_key: str, target_step: int, samples: int):
    best_step = None
    best_val = None
    best_diff = None
    history = run.history(samples=samples, pandas=False)
    for row in history:
        if score_key not in row:
            continue
        step = row.get("_step")
        if step is None:
            continue
        diff = abs(step - target_step)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_step = step
            best_val = row.get(score_key)
    return best_step, best_val


def compute_power_avg(run, start_min: int, end_min: int, samples: int):
    start_s = start_min * 60
    end_s = end_min * 60
    history = run.history(stream="system", samples=samples, pandas=False)
    power_percent_key = None
    power_watts_key = None
    percent_vals = []
    watts_vals = []
    for row in history:
        runtime = row.get("_runtime")
        if runtime is None:
            continue
        if runtime < start_s or runtime > end_s:
            continue
        if power_percent_key is None and power_watts_key is None:
            power_percent_key, power_watts_key = find_power_keys(row)
        if power_percent_key and power_percent_key in row:
            val = row.get(power_percent_key)
            if isinstance(val, (int, float)):
                percent_vals.append(float(val))
        if power_watts_key and power_watts_key in row:
            val = row.get(power_watts_key)
            if isinstance(val, (int, float)):
                watts_vals.append(float(val))
    avg_percent = sum(percent_vals) / len(percent_vals) if percent_vals else None
    avg_watts = sum(watts_vals) / len(watts_vals) if watts_vals else None
    return power_percent_key, power_watts_key, avg_percent, avg_watts, len(percent_vals), len(watts_vals)


def main() -> int:
    args = parse_args()
    api = wandb.Api()
    rows = []

    for run_name in args.runs:
        run = fetch_run(api, args.project, run_name)
        score_step, score_val = compute_score_at_step(
            run, args.score_key, args.target_step, args.samples
        )
        (
            power_percent_key,
            power_watts_key,
            avg_percent,
            avg_watts,
            n_percent,
            n_watts,
        ) = compute_power_avg(
            run, args.power_window_min, args.power_window_max, args.samples
        )

        rows.append(
            {
                "run": run.name,
                "run_id": run.id,
                "created_at": to_iso(getattr(run, "created_at", None)),
                "score_key": args.score_key,
                "score_step": score_step,
                "score_value": score_val,
                "power_percent_key": power_percent_key,
                "power_watts_key": power_watts_key,
                "avg_power_percent": avg_percent,
                "avg_power_watts": avg_watts,
                "n_power_percent": n_percent,
                "n_power_watts": n_watts,
            }
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
