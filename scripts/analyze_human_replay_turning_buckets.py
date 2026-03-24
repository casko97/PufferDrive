#!/usr/bin/env python3
import argparse
import csv
import hashlib
import io
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _read_i32(file_obj) -> int:
    buf = file_obj.read(4)
    if len(buf) != 4:
        raise EOFError("unexpected EOF while reading int32")
    return struct.unpack("<i", buf)[0]


def _read_f32_array(file_obj, size: int) -> tuple[float, ...]:
    buf = file_obj.read(4 * size)
    if len(buf) != 4 * size:
        raise EOFError("unexpected EOF while reading float32 array")
    return struct.unpack(f"<{size}f", buf)


def _read_i32_array(file_obj, size: int) -> tuple[int, ...]:
    buf = file_obj.read(4 * size)
    if len(buf) != 4 * size:
        raise EOFError("unexpected EOF while reading int32 array")
    return struct.unpack(f"<{size}i", buf)


def classify_map_turning(map_path: Path, threshold_deg: float) -> dict[str, Any]:
    with map_path.open("rb") as file_obj:
        sdc_track_index = _read_i32(file_obj)
        num_tracks_to_predict = _read_i32(file_obj)
        file_obj.seek(4 * num_tracks_to_predict, 1)
        num_objects = _read_i32(file_obj)
        num_roads = _read_i32(file_obj)

        if not (0 <= sdc_track_index < num_objects):
            raise ValueError(f"invalid sdc_track_index={sdc_track_index} for {map_path}")

        start_heading = None
        end_heading = None
        first_valid_idx = None
        last_valid_idx = None

        for obj_idx in range(num_objects):
            _scenario_id = _read_i32(file_obj)
            _entity_type = _read_i32(file_obj)
            _entity_id = _read_i32(file_obj)
            trajectory_length = _read_i32(file_obj)

            file_obj.seek(4 * trajectory_length * 6, 1)  # x,y,z,vx,vy,vz
            headings = _read_f32_array(file_obj, trajectory_length)
            valids = _read_i32_array(file_obj, trajectory_length)
            file_obj.seek((6 * 4) + 4, 1)  # width,length,height,goal xyz,mark_as_expert

            if obj_idx == sdc_track_index:
                valid_indices = [i for i, valid in enumerate(valids) if valid]
                if not valid_indices:
                    raise ValueError(f"SDC has no valid timesteps in {map_path}")
                first_valid_idx = valid_indices[0]
                last_valid_idx = valid_indices[-1]
                start_heading = headings[first_valid_idx]
                end_heading = headings[last_valid_idx]

        for _ in range(num_roads):
            _scenario_id = _read_i32(file_obj)
            _entity_type = _read_i32(file_obj)
            _entity_id = _read_i32(file_obj)
            array_size = _read_i32(file_obj)
            file_obj.seek(4 * array_size * 3, 1)  # x,y,z
            file_obj.seek((6 * 4) + 4, 1)

    delta_heading_rad = _wrap_angle(float(end_heading) - float(start_heading))
    delta_heading_deg = math.degrees(delta_heading_rad)
    bucket = "turning" if abs(delta_heading_deg) > threshold_deg else "straight"
    return {
        "map_name": map_path.name,
        "map_path": str(map_path.resolve()),
        "bucket": bucket,
        "delta_heading_rad": delta_heading_rad,
        "delta_heading_deg": delta_heading_deg,
        "first_valid_idx": first_valid_idx,
        "last_valid_idx": last_valid_idx,
    }


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def sample_digest(map_names: list[str]) -> str:
    return hashlib.sha256("\n".join(map_names).encode("utf-8")).hexdigest()[:16]


def build_bucket_map(manifest: dict[str, Any], threshold_deg: float) -> dict[str, dict[str, Any]]:
    bucket_map = {}
    for map_path in manifest["map_paths"]:
        row = classify_map_turning(Path(map_path), threshold_deg)
        bucket_map[row["map_name"]] = row
    return bucket_map


def summarize_bucket_counts(bucket_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(bucket_map.values())
    turning = sum(row["bucket"] == "turning" for row in rows)
    straight = sum(row["bucket"] == "straight" for row in rows)
    top_turns = sorted(rows, key=lambda row: abs(row["delta_heading_deg"]), reverse=True)[:5]
    return {
        "sample_size": len(rows),
        "turning": turning,
        "straight": straight,
        "turning_fraction": turning / len(rows) if rows else 0.0,
        "straight_fraction": straight / len(rows) if rows else 0.0,
        "top_heading_changes": top_turns,
    }


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def load_per_scenario_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def summarize_metrics_by_bucket(
    csv_path: Path,
    bucket_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = load_per_scenario_csv(csv_path)
    if not rows:
        return {"completed_rows": 0, "buckets": {}}

    metric_keys = [key for key, value in rows[0].items() if key not in {"map_name", "map_path"} and _is_float(value)]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_maps = []

    for row in rows:
        bucket_row = bucket_map.get(row["map_name"])
        if bucket_row is None:
            missing_maps.append(row["map_name"])
            continue
        grouped[bucket_row["bucket"]].append(row)

    buckets = {}
    for bucket_name, bucket_rows in grouped.items():
        means = {}
        for metric_key in metric_keys:
            values = [float(row[metric_key]) for row in bucket_rows]
            means[metric_key] = sum(values) / len(values)
        buckets[bucket_name] = {
            "count": len(bucket_rows),
            "means": means,
        }

    return {
        "completed_rows": len(rows),
        "missing_bucket_maps": missing_maps,
        "buckets": buckets,
    }


def compare_bucket_summaries(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "left_run": left_name,
        "right_run": right_name,
        "buckets": {},
        "turning_minus_straight": {},
    }

    bucket_names = sorted(set(left_summary["buckets"]) | set(right_summary["buckets"]))
    for bucket_name in bucket_names:
        left_bucket = left_summary["buckets"].get(bucket_name, {"count": 0, "means": {}})
        right_bucket = right_summary["buckets"].get(bucket_name, {"count": 0, "means": {}})
        metric_names = sorted(set(left_bucket["means"]) | set(right_bucket["means"]))
        diffs = {}
        for metric_name in metric_names:
            left_value = left_bucket["means"].get(metric_name)
            right_value = right_bucket["means"].get(metric_name)
            if left_value is None or right_value is None:
                continue
            diffs[metric_name] = left_value - right_value

        result["buckets"][bucket_name] = {
            "left_count": left_bucket["count"],
            "right_count": right_bucket["count"],
            "left_minus_right": diffs,
        }

    for run_name, summary in ((left_name, left_summary), (right_name, right_summary)):
        straight_means = summary["buckets"].get("straight", {}).get("means", {})
        turning_means = summary["buckets"].get("turning", {}).get("means", {})
        deltas = {}
        for metric_name in sorted(set(straight_means) | set(turning_means)):
            if metric_name not in straight_means or metric_name not in turning_means:
                continue
            deltas[metric_name] = turning_means[metric_name] - straight_means[metric_name]
        result["turning_minus_straight"][run_name] = deltas

    return result


def _print_bucket_count_summary(run_name: str, summary: dict[str, Any], out) -> None:
    print(f"{run_name}: sample_size={summary['sample_size']} straight={summary['straight']} turning={summary['turning']}", file=out)


def _print_metric_block(title: str, metrics: dict[str, float], preferred_keys: list[str], out) -> None:
    print(title, file=out)
    printed = set()
    for key in preferred_keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}", file=out)
            printed.add(key)
    for key in sorted(metrics):
        if key not in printed:
            print(f"  {key}: {metrics[key]:.6f}", file=out)


def print_run_summary(run_name: str, summary: dict[str, Any], out) -> None:
    preferred_keys = [
        "score",
        "completion_rate",
        "collisions_per_agent",
        "offroad_per_agent",
        "episode_return",
        "goals_reached_this_episode",
        "goals_sampled_this_episode",
        "speed_at_goal",
        "collision_rate",
        "offroad_rate",
        "dnf_rate",
        "lane_alignment_rate",
    ]
    print(f"\n== {run_name} ==", file=out)
    print(f"completed_rows: {summary['completed_rows']}", file=out)
    for bucket_name in ("straight", "turning"):
        bucket = summary["buckets"].get(bucket_name)
        if not bucket:
            continue
        _print_metric_block(f"{bucket_name} (n={bucket['count']})", bucket["means"], preferred_keys, out)


def print_comparison(comparison: dict[str, Any], out) -> None:
    preferred_keys = [
        "score",
        "completion_rate",
        "collisions_per_agent",
        "offroad_per_agent",
        "episode_return",
        "speed_at_goal",
        "collision_rate",
        "offroad_rate",
        "dnf_rate",
        "lane_alignment_rate",
    ]
    left_name = comparison["left_run"]
    right_name = comparison["right_run"]
    print(f"\n== Comparison: {left_name} - {right_name} ==", file=out)
    for bucket_name in ("straight", "turning"):
        bucket = comparison["buckets"].get(bucket_name)
        if not bucket:
            continue
        print(f"{bucket_name} (left_n={bucket['left_count']}, right_n={bucket['right_count']})", file=out)
        metrics = bucket["left_minus_right"]
        for key in preferred_keys:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.6f}", file=out)

    print("\nturning - straight within each run", file=out)
    for run_name, metrics in comparison["turning_minus_straight"].items():
        print(run_name, file=out)
        for key in preferred_keys:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.6f}", file=out)


def infer_sample_sets(model_dir: Path) -> dict[str, dict[str, Any]]:
    manifests = sorted(model_dir.glob("*/human_replay_eval_sampled_maps.json"))
    sample_sets = {}
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        digest = sample_digest(manifest["map_names"])
        sample_sets.setdefault(
            digest,
            {
                "manifest_path": manifest_path,
                "manifest": manifest,
                "run_names": [],
            },
        )
        sample_sets[digest]["run_names"].append(manifest_path.parent.name)
    return sample_sets


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze turning vs straight buckets for human replay eval outputs.")
    parser.add_argument("model_dir", type=Path, help="Model directory containing eval subdirectories")
    parser.add_argument("--threshold-deg", type=float, default=45.0, help="Heading delta threshold for turning")
    parser.add_argument("--left-run", default="car eval 2", help="Run name to use as the left side of comparison")
    parser.add_argument("--right-run", default="truck eval 2", help="Run name to use as the right side of comparison")
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        default=[],
        help="Restrict output to these run names. Can be passed multiple times.",
    )
    parser.add_argument("--output", type=Path, help="Write the rendered summary to this output file")
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print full analysis payload as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    requested_runs = args.runs or [args.left_run, args.right_run]
    sample_sets = infer_sample_sets(model_dir)
    if not sample_sets:
        raise FileNotFoundError(f"no human_replay_eval_sampled_maps.json files found under {model_dir}")

    analysis: dict[str, Any] = {
        "model_dir": str(model_dir),
        "threshold_deg": args.threshold_deg,
        "sample_sets": {},
    }

    for digest, sample_info in sample_sets.items():
        manifest = sample_info["manifest"]
        selected_run_names = [run_name for run_name in sample_info["run_names"] if run_name in requested_runs]
        if not selected_run_names:
            continue
        bucket_map = build_bucket_map(manifest, args.threshold_deg)
        bucket_summary = summarize_bucket_counts(bucket_map)
        analysis["sample_sets"][digest] = {
            "run_names": selected_run_names,
            "manifest_path": str(sample_info["manifest_path"]),
            "bucket_summary": bucket_summary,
            "runs": {},
        }

        for run_name in selected_run_names:
            run_dir = model_dir / run_name
            csv_path = run_dir / "human_replay_eval_per_scenario.csv"
            if not csv_path.exists():
                continue
            analysis["sample_sets"][digest]["runs"][run_name] = summarize_metrics_by_bucket(csv_path, bucket_map)

        left_summary = analysis["sample_sets"][digest]["runs"].get(args.left_run)
        right_summary = analysis["sample_sets"][digest]["runs"].get(args.right_run)
        if left_summary and right_summary:
            analysis["sample_sets"][digest]["comparison"] = compare_bucket_summaries(
                left_summary=left_summary,
                right_summary=right_summary,
                left_name=args.left_run,
                right_name=args.right_run,
            )

    if not analysis["sample_sets"]:
        raise FileNotFoundError(
            f"none of the requested runs were found under {model_dir}: {', '.join(requested_runs)}"
        )

    if args.json_output:
        rendered = json.dumps(analysis, indent=2, sort_keys=True)
    else:
        out = io.StringIO()
        print(f"Model: {model_dir}", file=out)
        print(f"Turning threshold: {args.threshold_deg:.1f} deg", file=out)
        for digest, sample_info in analysis["sample_sets"].items():
            print(f"\nSample set {digest}", file=out)
            print(f"runs: {', '.join(sample_info['run_names'])}", file=out)
            _print_bucket_count_summary("shared sample", sample_info["bucket_summary"], out)
            for run_name, run_summary in sample_info["runs"].items():
                print_run_summary(run_name, run_summary, out)
            if "comparison" in sample_info:
                print_comparison(sample_info["comparison"], out)
        rendered = out.getvalue()

    if args.output is not None:
        args.output = args.output.resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    print(rendered, end="")


if __name__ == "__main__":
    main()
