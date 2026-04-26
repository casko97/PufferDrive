#!/usr/bin/env python3
import argparse
import json
import math
import os
from datetime import datetime, timezone

import wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize W&B runs by avg GPU power vs env score and emit CSV/plot."
    )
    parser.add_argument("--project", required=True, help="W&B project, e.g. casko_kth/pufferdrive")
    parser.add_argument("--max-runs", type=int, default=50, help="Max runs to scan")
    parser.add_argument("--recent-days", type=int, default=90, help="Only include runs from last N days")
    parser.add_argument("--samples", type=int, default=2000, help="History samples per run")
    parser.add_argument("--output-csv", default="wandb_gpu_vs_score.csv", help="CSV output path")
    parser.add_argument("--output-png", default="wandb_gpu_vs_score.png", help="PNG output path")
    parser.add_argument("--score-key", default="environment/score", help="Primary score key")
    parser.add_argument("--alt-score-key", default="environment/episode_return", help="Fallback score key")
    return parser.parse_args()


def mean_last_fraction(values, frac=0.2):
    if not values:
        return None
    start = int(len(values) * (1.0 - frac))
    chunk = values[start:] if start < len(values) else values
    if not chunk:
        return None
    return sum(chunk) / len(chunk)


def pick_power_key(sample_rows):
    if not sample_rows:
        return None
    keys = set()
    for row in sample_rows:
        keys.update(row.keys())
    candidates = [k for k in keys if "gpu" in k.lower() and "power" in k.lower()]
    # Prefer power usage (%) style keys
    for k in candidates:
        kl = k.lower()
        if "usage" in kl or "util" in kl:
            return k
    # Fallback to explicit watts keys
    for k in candidates:
        if "watts" in k.lower():
            return k
    return candidates[0] if candidates else None


def pick_score_key(sample_rows, preferred, fallback):
    if not sample_rows:
        return None
    keys = set()
    for row in sample_rows:
        keys.update(row.keys())
    if preferred in keys:
        return preferred
    if fallback in keys:
        return fallback
    # Heuristic fallback
    for k in keys:
        if "score" in k.lower() and "environment" in k.lower():
            return k
    return None


def main():
    args = parse_args()
    api = wandb.Api()
    runs = api.runs(args.project, per_page=args.max_runs)

    cutoff = None
    if args.recent_days:
        cutoff = datetime.now(timezone.utc).timestamp() - args.recent_days * 86400

    rows = []
    for r in runs:
        created = r.created_at
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:
                created = None
        if cutoff and created and created.timestamp() < cutoff:
            continue

        # Sample history to infer keys and compute stats
        try:
            sample_rows = list(r.history(samples=args.samples, pandas=False))
        except Exception:
            sample_rows = []

        power_key = pick_power_key(sample_rows)
        score_key = pick_score_key(sample_rows, args.score_key, args.alt_score_key)

        power_vals = []
        score_vals = []
        for row in sample_rows:
            if power_key and power_key in row and row[power_key] is not None:
                power_vals.append(float(row[power_key]))
            if score_key and score_key in row and row[score_key] is not None:
                score_vals.append(float(row[score_key]))

        avg_power = mean_last_fraction(power_vals)
        avg_score = mean_last_fraction(score_vals)

        rows.append(
            {
                "run": r.display_name or r.name,
                "run_id": r.id,
                "created_at": created.isoformat() if created else str(r.created_at),
                "power_key": power_key,
                "score_key": score_key,
                "avg_power_watts": avg_power,
                "avg_score": avg_score,
            }
        )

    # Filter rows with both metrics
    rows = [r for r in rows if r["avg_power_watts"] is not None and r["avg_score"] is not None]

    # Write CSV
    with open(args.output_csv, "w", encoding="utf-8") as f:
        f.write("run,run_id,created_at,power_key,score_key,avg_power_watts,avg_score\n")
        for r in rows:
            f.write(
                f"{r['run']},{r['run_id']},{r['created_at']},"
                f"{r['power_key']},{r['score_key']},{r['avg_power_watts']},{r['avg_score']}\n"
            )

    # Try to plot
    try:
        import matplotlib.pyplot as plt

        xs = [r["avg_power_watts"] for r in rows]
        ys = [r["avg_score"] for r in rows]
        labels = [r["run"] for r in rows]

        plt.figure(figsize=(8, 6))
        plt.scatter(xs, ys, alpha=0.75)
        for x, y, label in zip(xs, ys, labels):
            plt.annotate(label, (x, y), fontsize=8, alpha=0.7)
        plt.xlabel("Avg GPU Power Usage (%)")
        plt.ylabel("Avg Environment Score")
        plt.title("W&B Runs: GPU Power vs Env Score (last 20%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.output_png, dpi=200)
        print(f"Wrote {args.output_png}")
    except Exception as e:
        print("Plot skipped (matplotlib not available).", e)

    print(f"Wrote {args.output_csv}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
