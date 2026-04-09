#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

from pufferlib.ocean.drive.drive import classify_map_turning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a permanent filtered Drive dataset directory from an existing dataset split."
    )
    parser.add_argument("--source-dir", required=True, help="Source dataset directory, e.g. resources/drive/binaries/...")
    parser.add_argument("--output-dir", required=True, help="Output directory for the filtered dataset")
    parser.add_argument(
        "--scenario-filter",
        default="turning",
        choices=["turning", "straight"],
        help="Which scenario bucket to keep",
    )
    parser.add_argument(
        "--threshold-deg",
        type=float,
        default=45.0,
        help="Heading delta threshold used to classify turning vs straight",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional manifest path. Defaults to <source-dir>/selection_manifest.json if present.",
    )
    parser.add_argument(
        "--max-maps",
        type=int,
        default=None,
        help="Optional limit on how many manifest scenarios or map files to inspect",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy files instead of creating symlinks",
    )
    return parser.parse_args()


def load_payload(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_candidates(source_dir: Path, manifest_path: Path | None, max_maps: int | None):
    if manifest_path is not None and manifest_path.exists():
        payload = load_payload(manifest_path)
        scenarios = payload.get("scenarios", [])
        if max_maps is not None:
            scenarios = scenarios[:max_maps]
        for scenario in scenarios:
            map_name = scenario.get("map_name")
            if not map_name:
                continue
            yield source_dir / map_name, scenario, payload
        return

    payload = None
    map_paths = sorted(source_dir.glob("map_*.bin"))
    if max_maps is not None:
        map_paths = map_paths[:max_maps]
    for map_path in map_paths:
        yield map_path, None, payload


def materialize(src: Path, dst: Path, copy_files: bool) -> None:
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else None
    if manifest_path is None:
        default_manifest = source_dir / "selection_manifest.json"
        manifest_path = default_manifest if default_manifest.exists() else None

    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=False)

    selected_scenarios = []
    selected_maps = []
    payload_template = None
    inspected = 0
    for src_path, scenario, payload in iter_candidates(source_dir, manifest_path, args.max_maps):
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source map: {src_path}")
        if payload_template is None and payload is not None:
            payload_template = payload

        row = classify_map_turning(src_path, args.threshold_deg)
        inspected += 1
        if row["bucket"] != args.scenario_filter:
            continue

        new_idx = len(selected_maps)
        new_name = f"map_{new_idx:03d}.bin"
        dst_path = output_dir / new_name
        materialize(src_path, dst_path, copy_files=args.copy_files)

        selected_maps.append(
            {
                "filtered_map_name": new_name,
                "original_map_name": src_path.name,
                "original_map_path": str(src_path),
                "bucket": row["bucket"],
                "delta_heading_deg": float(row["delta_heading_deg"]),
            }
        )

        if scenario is not None:
            new_scenario = dict(scenario)
            new_scenario["map_name"] = new_name
            new_scenario["original_map_name"] = src_path.name
            new_scenario["turning_bucket"] = row["bucket"]
            new_scenario["delta_heading_deg"] = float(row["delta_heading_deg"])
            selected_scenarios.append(new_scenario)

    if not selected_maps:
        raise SystemExit(
            f"No maps matched scenario_filter={args.scenario_filter!r} threshold_deg={args.threshold_deg} in {source_dir}"
        )

    scenario_filter_manifest = {
        "source_map_dir": str(source_dir),
        "scenario_filter": args.scenario_filter,
        "threshold_deg": float(args.threshold_deg),
        "selected_count": len(selected_maps),
        "inspected_count": inspected,
        "maps": selected_maps,
    }
    with (output_dir / "scenario_filter_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(scenario_filter_manifest, f, indent=2)

    if payload_template is not None:
        subset_payload = dict(payload_template)
        subset_payload["count"] = len(selected_scenarios)
        subset_payload["source_split_dir"] = str(source_dir)
        subset_payload["scenario_filter"] = args.scenario_filter
        subset_payload["scenario_filter_threshold_deg"] = float(args.threshold_deg)
        subset_payload["inspected_count"] = inspected
        subset_payload["scenarios"] = selected_scenarios
        with (output_dir / "selection_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(subset_payload, f, indent=2)

    print(f"Created filtered dataset: {output_dir}")
    print(f"Scenario filter: {args.scenario_filter}")
    print(f"Threshold (deg): {args.threshold_deg}")
    print(f"Inspected maps: {inspected}")
    print(f"Selected maps: {len(selected_maps)}")


if __name__ == "__main__":
    main()
