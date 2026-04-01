from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from scripts.build_truck_context_preferences import (
    PREFERENCES_FORMAT_SHARDED,
    build_truck_context_preferences,
    iter_preference_shards,
    load_preference_manifest,
)
from scripts.export_paired_offline_fits import export_paired_fits, iter_paired_fit_shards
from pufferlib.ocean.drive.drive import save_map_binary

LOGGER = logging.getLogger("create_full_aligned_boston_bins")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_json(path: Path) -> dict:
    with path.open("r") as file_obj:
        return json.load(file_obj)


def build_filter_cli_command(
    preference_path: Path,
    fit_export_path: Path,
    output_path: Path,
    max_optimization_mean_cost: float | None = None,
    min_ade: float | None = None,
    max_ade: float | None = None,
    max_fde: float | None = None,
) -> str:
    cmd = [
        sys.executable,
        "-m",
        "scripts.create_full_aligned_boston_bins",
        "--preference-input",
        str(preference_path),
        "--fit-input",
        str(fit_export_path),
        "--filtered-preference-output",
        str(output_path),
    ]
    if max_optimization_mean_cost is not None:
        cmd.extend(["--filter-max-optimization-mean-cost", str(max_optimization_mean_cost)])
    if min_ade is not None:
        cmd.extend(["--filter-min-ade", str(min_ade)])
    if max_ade is not None:
        cmd.extend(["--filter-max-ade", str(max_ade)])
    if max_fde is not None:
        cmd.extend(["--filter-max-fde", str(max_fde)])
    return shlex.join(cmd)


def normalize_ego_to_front(map_data: dict) -> dict:
    map_data = deepcopy(map_data)
    metadata = map_data.setdefault("metadata", {})
    objects = list(map_data.get("objects", []))
    sdc_track_index = int(metadata.get("sdc_track_index", -1))
    if not (0 <= sdc_track_index < len(objects)) or sdc_track_index == 0:
        return map_data

    reordered_objects = [objects[sdc_track_index]]
    reordered_objects.extend(obj for idx, obj in enumerate(objects) if idx != sdc_track_index)

    index_map = {sdc_track_index: 0}
    next_idx = 1
    for idx in range(len(objects)):
        if idx == sdc_track_index:
            continue
        index_map[idx] = next_idx
        next_idx += 1

    metadata["sdc_track_index"] = 0

    if "ego_trailer_track_index" in metadata:
        trailer_idx = int(metadata.get("ego_trailer_track_index", -1))
        metadata["ego_trailer_track_index"] = index_map.get(trailer_idx, trailer_idx)

    remapped_tracks = []
    for track in metadata.get("tracks_to_predict", []):
        if isinstance(track, dict):
            updated_track = dict(track)
            track_index = int(updated_track.get("track_index", -1))
            updated_track["track_index"] = index_map.get(track_index, track_index)
            remapped_tracks.append(updated_track)
        else:
            track_index = int(track)
            remapped_tracks.append(index_map.get(track_index, track_index))
    metadata["tracks_to_predict"] = remapped_tracks

    map_data["objects"] = reordered_objects
    return map_data


def scenario_record(map_name: str, source_json: Path, map_data: dict, batch_index: int) -> dict:
    metadata = map_data.get("metadata", {})
    return {
        "map_name": map_name,
        "source_json": str(source_json.resolve()),
        "source_batch_index": batch_index,
        "scenario_id": source_json.stem.split("__")[-1],
        "scenario_type": source_json.stem.split("__")[0],
        "has_ego_trailer": bool(metadata.get("has_ego_trailer", False)),
        "ego_trailer_track_index": int(metadata.get("ego_trailer_track_index", -1)),
        "num_objects": len(map_data.get("objects", [])),
        "num_roads": len(map_data.get("roads", [])),
    }


def shared_json_names(truck_dir: Path, car_dir: Path) -> list[str]:
    truck_names = {path.name for path in truck_dir.glob("*.json")}
    car_names = {path.name for path in car_dir.glob("*.json")}
    shared = sorted(truck_names & car_names)
    if not shared:
        raise ValueError(f"No shared JSON filenames between {truck_dir} and {car_dir}")
    return shared


def build_aligned_bins(
    truck_source_dirs: list[Path],
    car_source_dirs: list[Path],
    truck_output_dir: Path,
    car_output_dir: Path,
    limit_per_batch: int | None = None,
) -> tuple[Path, Path]:
    if len(truck_source_dirs) != len(car_source_dirs):
        raise ValueError("truck_source_dirs and car_source_dirs must have the same length")

    LOGGER.info(
        "Starting aligned bin build | truck_batches=%d car_batches=%d output_truck=%s output_car=%s limit_per_batch=%s",
        len(truck_source_dirs),
        len(car_source_dirs),
        truck_output_dir,
        car_output_dir,
        limit_per_batch,
    )
    truck_output_dir.mkdir(parents=True, exist_ok=True)
    car_output_dir.mkdir(parents=True, exist_ok=True)

    pair_batches: list[dict] = []
    for batch_index, (truck_dir, car_dir) in enumerate(zip(truck_source_dirs, car_source_dirs)):
        shared_names = shared_json_names(truck_dir, car_dir)
        original_shared_count = len(shared_names)
        if limit_per_batch is not None and limit_per_batch > 0:
            shared_names = shared_names[:limit_per_batch]
        LOGGER.info(
            "Discovered shared scenarios for batch %d | truck_dir=%s car_dir=%s shared=%d selected=%d",
            batch_index,
            truck_dir,
            car_dir,
            original_shared_count,
            len(shared_names),
        )
        pair_batches.append(
            {
                "batch_index": batch_index,
                "truck_source_dir": str(truck_dir.resolve()),
                "car_source_dir": str(car_dir.resolve()),
                "shared_count": len(shared_names),
                "shared_names": shared_names,
            }
        )

    total_maps = sum(batch["shared_count"] for batch in pair_batches)
    width = max(3, len(str(total_maps - 1)))
    map_index = 0
    truck_scenarios = []
    car_scenarios = []

    for batch in pair_batches:
        batch_index = batch["batch_index"]
        truck_dir = Path(batch["truck_source_dir"])
        car_dir = Path(batch["car_source_dir"])
        LOGGER.info(
            "Converting batch %d to aligned bins | selected_maps=%d",
            batch_index,
            len(batch["shared_names"]),
        )
        for json_name in batch["shared_names"]:
            map_name = f"map_{map_index:0{width}d}.bin"
            truck_json = truck_dir / json_name
            car_json = car_dir / json_name

            truck_map_data = normalize_ego_to_front(load_json(truck_json))
            car_map_data = normalize_ego_to_front(load_json(car_json))

            save_map_binary(truck_map_data, str(truck_output_dir / map_name), map_index)
            save_map_binary(car_map_data, str(car_output_dir / map_name), map_index)

            truck_record = scenario_record(map_name, truck_json, truck_map_data, batch_index)
            car_record = scenario_record(map_name, car_json, car_map_data, batch_index)
            truck_record["paired_source_json"] = str(car_json.resolve())
            car_record["paired_source_json"] = str(truck_json.resolve())
            truck_scenarios.append(truck_record)
            car_scenarios.append(car_record)
            map_index += 1
            if map_index % 1000 == 0:
                LOGGER.info("Converted %d aligned map pairs so far", map_index)

    manifest_common = {
        "selection_strategy": "all shared JSON filenames across provided truck/car source batches, ordered by batch and filename",
        "count": total_maps,
        "map_name_width": width,
        "batches": [
            {
                key: value
                for key, value in batch.items()
                if key != "shared_names"
            }
            for batch in pair_batches
        ],
    }

    truck_manifest = {
        **manifest_common,
        "source_dir": [str(path.resolve()) for path in truck_source_dirs],
        "paired_with": str(car_output_dir.resolve()),
        "scenarios": truck_scenarios,
    }
    car_manifest = {
        **manifest_common,
        "source_dir": [str(path.resolve()) for path in car_source_dirs],
        "paired_with": str(truck_output_dir.resolve()),
        "scenarios": car_scenarios,
    }

    truck_manifest_path = truck_output_dir / "selection_manifest.json"
    car_manifest_path = car_output_dir / "selection_manifest.json"
    with truck_manifest_path.open("w") as file_obj:
        json.dump(truck_manifest, file_obj, indent=2)
        file_obj.write("\n")
    with car_manifest_path.open("w") as file_obj:
        json.dump(car_manifest, file_obj, indent=2)
        file_obj.write("\n")

    LOGGER.info(
        "Finished aligned bin build | total_pairs=%d truck_manifest=%s car_manifest=%s",
        total_maps,
        truck_manifest_path,
        car_manifest_path,
    )
    return truck_manifest_path, car_manifest_path


def _mean_optimization_cost(side: dict) -> float:
    num_steps = int(side.get("num_steps", 0))
    if num_steps <= 0:
        return float("inf")
    return float(side.get("match_cost_total", float("inf"))) / float(num_steps)


def filter_preference_windows(
    preference_path: Path,
    fit_export_path: Path,
    output_path: Path,
    max_optimization_mean_cost: float | None = None,
    min_ade: float | None = None,
    max_ade: float | None = None,
    max_fde: float | None = None,
) -> Path:
    LOGGER.info(
        "Filtering preference windows | preference_path=%s fit_export_path=%s max_mean_cost=%s min_ade=%s max_ade=%s max_fde=%s",
        preference_path,
        fit_export_path,
        max_optimization_mean_cost,
        min_ade,
        max_ade,
        max_fde,
    )
    allowed_maps: dict[str, dict[str, float]] = {}
    for _fit_metadata, pairs, shard_info in iter_paired_fit_shards(fit_export_path):
        LOGGER.info(
            "Evaluating fit shard for filtering | path=%s map_count=%s",
            shard_info.get("path"),
            shard_info.get("map_count"),
        )
        for map_name, pair in pairs.items():
            truck_side = pair.get("truck", {})
            car_side = pair.get("car", {})
            if truck_side.get("status") != "ok" or car_side.get("status") != "ok":
                continue

            truck_mean_cost = _mean_optimization_cost(truck_side)
            car_mean_cost = _mean_optimization_cost(car_side)
            truck_ade = float(truck_side.get("self_ade", float("inf")))
            car_ade = float(car_side.get("self_ade", float("inf")))
            truck_fde = float(truck_side.get("self_fde", float("inf")))
            car_fde = float(car_side.get("self_fde", float("inf")))

            passes = True
            if max_optimization_mean_cost is not None:
                passes = passes and truck_mean_cost < max_optimization_mean_cost and car_mean_cost < max_optimization_mean_cost
            if min_ade is not None:
                passes = passes and truck_ade > min_ade and car_ade > min_ade
            if max_ade is not None:
                passes = passes and truck_ade < max_ade and car_ade < max_ade
            if max_fde is not None:
                passes = passes and truck_fde < max_fde and car_fde < max_fde

            if passes:
                allowed_maps[map_name] = {
                    "truck_optimization_mean_cost": truck_mean_cost,
                    "car_optimization_mean_cost": car_mean_cost,
                }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output_path.parent / f"{output_path.stem}_shards"
    preference_manifest = load_preference_manifest(preference_path)
    preference_is_sharded = preference_manifest.get("format") == PREFERENCES_FORMAT_SHARDED
    shard_infos: list[dict] = []
    kept_windows = 0
    filtered_out = 0
    shard_index = 0
    first_payload_metadata: dict | None = None

    for preference_payload, shard_info in iter_preference_shards(preference_path):
        if first_payload_metadata is None:
            first_payload_metadata = dict(preference_payload.get("metadata", {}))
        preferred_sa = np.asarray(preference_payload["preferred_sa"], dtype=np.float32)
        rejected_sa = np.asarray(preference_payload["rejected_sa"], dtype=np.float32)
        labels = np.asarray(preference_payload["labels"], dtype=np.float32)
        window_metadata = list(preference_payload.get("window_metadata", []))

        keep_indices: list[int] = []
        kept_window_metadata: list[dict] = []
        for idx, window_meta in enumerate(window_metadata):
            fit_meta = allowed_maps.get(window_meta["map_name"])
            if fit_meta is None:
                filtered_out += 1
                continue
            keep_indices.append(idx)
            enriched_meta = dict(window_meta)
            enriched_meta.update(fit_meta)
            kept_window_metadata.append(enriched_meta)

        if keep_indices:
            keep_index_array = np.asarray(keep_indices, dtype=np.int64)
            preferred_kept = preferred_sa[keep_index_array]
            rejected_kept = rejected_sa[keep_index_array]
            labels_kept = labels[keep_index_array]
        else:
            preferred_kept = preferred_sa[:0]
            rejected_kept = rejected_sa[:0]
            labels_kept = labels[:0]

        kept_count = int(len(preferred_kept))
        kept_windows += kept_count
        filtered_metadata = dict(preference_payload.get("metadata", {}))
        filtered_metadata["filtering"] = {
            "source_preference_path": str(preference_path),
            "source_fit_export_path": str(fit_export_path),
            "max_optimization_mean_cost": max_optimization_mean_cost,
            "min_ade": min_ade,
            "max_ade": max_ade,
            "max_fde": max_fde,
            "kept_windows": kept_count,
            "filtered_out_windows": int(len(window_metadata) - kept_count),
        }
        filtered_metadata["total_windows"] = kept_count

        if preference_is_sharded:
            if kept_count == 0:
                LOGGER.info(
                    "Skipping empty filtered preference shard | source_path=%s filtered_out=%d",
                    shard_info.get("path"),
                    len(window_metadata),
                )
                continue
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_index += 1
            shard_path = shard_dir / f"{output_path.stem}.part{shard_index:05d}.pt"
            torch.save(
                {
                    "metadata": filtered_metadata,
                    "preferred_sa": preferred_kept,
                    "rejected_sa": rejected_kept,
                    "labels": labels_kept,
                    "window_metadata": kept_window_metadata,
                },
                shard_path,
            )
            shard_infos.append(
                {
                    "path": str(shard_path),
                    "window_count": kept_count,
                    "start_index": int(kept_windows - kept_count),
                    "end_index": int(kept_windows),
                }
            )
            LOGGER.info(
                "Filtered preference shard | shard=%d kept=%d filtered_out=%d output=%s",
                shard_index,
                    kept_count,
                    len(window_metadata) - kept_count,
                    shard_path,
                )
        else:
            torch.save(
                {
                    "metadata": filtered_metadata,
                    "preferred_sa": preferred_kept,
                    "rejected_sa": rejected_kept,
                    "labels": labels_kept,
                    "window_metadata": kept_window_metadata,
                },
                output_path,
            )

    summary = {
        "output_path": str(output_path),
        "source_preference_path": str(preference_path),
        "source_fit_export_path": str(fit_export_path),
        "max_optimization_mean_cost": max_optimization_mean_cost,
        "min_ade": min_ade,
        "max_ade": max_ade,
        "max_fde": max_fde,
        "kept_windows": int(kept_windows),
        "filtered_out_windows": int(filtered_out),
        "shard_count": int(len(shard_infos) if shard_infos else 1),
    }

    if shard_infos:
        manifest_metadata = dict(first_payload_metadata or {})
        manifest_metadata["filtering"] = {
            "source_preference_path": str(preference_path),
            "source_fit_export_path": str(fit_export_path),
            "max_optimization_mean_cost": max_optimization_mean_cost,
            "min_ade": min_ade,
            "max_ade": max_ade,
            "max_fde": max_fde,
            "kept_windows": int(kept_windows),
            "filtered_out_windows": int(filtered_out),
        }
        manifest_metadata["total_windows"] = int(kept_windows)
        torch.save(
            {
                "format": PREFERENCES_FORMAT_SHARDED,
                "metadata": manifest_metadata,
                "shards": shard_infos,
            },
            output_path,
        )
    elif preference_is_sharded:
        manifest_metadata = dict(first_payload_metadata or {})
        manifest_metadata["filtering"] = {
            "source_preference_path": str(preference_path),
            "source_fit_export_path": str(fit_export_path),
            "max_optimization_mean_cost": max_optimization_mean_cost,
            "min_ade": min_ade,
            "max_ade": max_ade,
            "max_fde": max_fde,
            "kept_windows": int(kept_windows),
            "filtered_out_windows": int(filtered_out),
        }
        manifest_metadata["total_windows"] = int(kept_windows)
        feature_dim = int(manifest_metadata.get("obs_dim", 0)) + int(manifest_metadata.get("action_dim", 0))
        window_len = int(manifest_metadata.get("window_len", 0))
        torch.save(
            {
                "metadata": manifest_metadata,
                "preferred_sa": np.zeros((0, window_len, feature_dim), dtype=np.float32),
                "rejected_sa": np.zeros((0, window_len, feature_dim), dtype=np.float32),
                "labels": np.zeros((0, 1), dtype=np.float32),
                "window_metadata": [],
            },
            output_path,
        )

    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    LOGGER.info(
        "Finished filtering preference windows | kept=%d filtered_out=%d output=%s shard_count=%d",
        kept_windows,
        filtered_out,
        output_path,
        len(shard_infos) if shard_infos else 1,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create aligned truck/car binary folders from shared JSON scenario names across one or more source batches."
    )
    parser.add_argument(
        "--truck-source-dir",
        dest="truck_source_dirs",
        action="append",
        help="Truck JSON source directory. Pass once per batch.",
    )
    parser.add_argument(
        "--car-source-dir",
        dest="car_source_dirs",
        action="append",
        help="Car JSON source directory. Pass once per batch in the same order as --truck-source-dir.",
    )
    parser.add_argument("--truck-output-dir", help="Destination folder for truck binary files.")
    parser.add_argument("--car-output-dir", help="Destination folder for car binary files.")
    parser.add_argument(
        "--limit-per-batch",
        type=int,
        default=None,
        help="Optional limit on how many shared scenarios to export from each batch, after filename sorting.",
    )
    parser.add_argument(
        "--run-fit-export",
        action="store_true",
        help="After bin conversion, export paired offline action-fitting data and replay observations.",
    )
    parser.add_argument(
        "--fit-output",
        type=Path,
        default=None,
        help="Output path for the paired offline fit artifact (.pt). Required if --run-fit-export is not inferable.",
    )
    parser.add_argument(
        "--run-preference-build",
        action="store_true",
        help="After fit export, build the truck-context preference-learning dataset.",
    )
    parser.add_argument(
        "--preference-output",
        type=Path,
        default=None,
        help="Output path for the unfiltered preference-learning artifact (.pt). Required if --run-preference-build is not inferable.",
    )
    parser.add_argument(
        "--filtered-preference-output",
        type=Path,
        default=None,
        help="Optional output path for the filtered preference-learning artifact (.pt). If omitted, a sibling *_filtered.pt path is used.",
    )
    parser.add_argument(
        "--preference-input",
        type=Path,
        default=None,
        help="Existing preference artifact (.pt) to filter directly without rebuilding bins or preferences.",
    )
    parser.add_argument(
        "--fit-input",
        type=Path,
        default=None,
        help="Existing paired offline fit artifact (.pt) to use when filtering an existing preference artifact.",
    )
    parser.add_argument(
        "--artifact-manifest-output",
        type=Path,
        default=None,
        help="Optional JSON path where all generated artifact paths and stage settings are recorded.",
    )
    parser.add_argument("--window-len", type=int, default=32)
    parser.add_argument("--max-start-distance-m", type=float, default=1.0)
    parser.add_argument("--min-gap-seconds", type=float, default=2.0)
    parser.add_argument("--observation-mode", type=str, default="default")
    parser.add_argument("--action-type", type=str, default="discrete")
    parser.add_argument("--fit-chunk-size", type=int, default=256)
    parser.add_argument("--fit-log-every", type=int, default=25)
    parser.add_argument("--preference-chunk-size", type=int, default=256)
    parser.add_argument("--preference-map-log-every", type=int, default=50)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable more verbose logging.",
    )
    parser.add_argument(
        "--filter-max-optimization-mean-cost",
        type=float,
        default=None,
        help="Optional final preference-window filter: keep only maps where both truck and car mean optimization cost are below this value.",
    )
    parser.add_argument(
        "--filter-min-ade",
        type=float,
        default=None,
        help="Optional final preference-window filter: keep only maps where both truck and car fit ADE are above this value.",
    )
    parser.add_argument(
        "--filter-max-ade",
        type=float,
        default=None,
        help="Optional final preference-window filter: keep only maps where both truck and car fit ADE are below this value.",
    )
    parser.add_argument(
        "--filter-max-fde",
        type=float,
        default=None,
        help="Optional final preference-window filter: keep only maps where both truck and car fit FDE are below this value.",
    )
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)
    LOGGER.info("Parsed pipeline arguments")

    filter_only_mode = args.preference_input is not None or args.fit_input is not None
    if filter_only_mode:
        if args.preference_input is None or args.fit_input is None:
            raise ValueError("--preference-input and --fit-input must be provided together")
        if args.truck_source_dirs or args.car_source_dirs or args.truck_output_dir or args.car_output_dir:
            LOGGER.info("Filter-only mode enabled; skipping aligned-bin creation inputs")
        if not any(
            value is not None
            for value in (
                args.filter_max_optimization_mean_cost,
                args.filter_min_ade,
                args.filter_max_ade,
                args.filter_max_fde,
            )
        ):
            raise ValueError("filter-only mode requires at least one --filter-* threshold")

        filtered_preference_output = args.filtered_preference_output
        if filtered_preference_output is None:
            filtered_preference_output = args.preference_input.with_name(f"{args.preference_input.stem}_filtered.pt")

        LOGGER.info(
            "Running filter-only preference pass | preference_input=%s fit_input=%s output=%s",
            args.preference_input,
            args.fit_input,
            filtered_preference_output,
        )
        LOGGER.info(
            "Filter-only rerun command | command=%s",
            build_filter_cli_command(
                preference_path=args.preference_input,
                fit_export_path=args.fit_input,
                output_path=filtered_preference_output,
                max_optimization_mean_cost=args.filter_max_optimization_mean_cost,
                min_ade=args.filter_min_ade,
                max_ade=args.filter_max_ade,
                max_fde=args.filter_max_fde,
            ),
        )
        filtered_preference_output = filter_preference_windows(
            preference_path=args.preference_input,
            fit_export_path=args.fit_input,
            output_path=filtered_preference_output,
            max_optimization_mean_cost=args.filter_max_optimization_mean_cost,
            min_ade=args.filter_min_ade,
            max_ade=args.filter_max_ade,
            max_fde=args.filter_max_fde,
        )
        print(f"filtered_preference_output={filtered_preference_output}")
        return

    if args.run_preference_build and not args.run_fit_export:
        raise ValueError("--run-preference-build requires --run-fit-export")
    if not args.truck_source_dirs or not args.car_source_dirs:
        raise ValueError("--truck-source-dir and --car-source-dir are required unless using --preference-input/--fit-input")
    if not args.truck_output_dir or not args.car_output_dir:
        raise ValueError("--truck-output-dir and --car-output-dir are required unless using --preference-input/--fit-input")

    truck_manifest_path, car_manifest_path = build_aligned_bins(
        truck_source_dirs=[Path(path) for path in args.truck_source_dirs],
        car_source_dirs=[Path(path) for path in args.car_source_dirs],
        truck_output_dir=Path(args.truck_output_dir),
        car_output_dir=Path(args.car_output_dir),
        limit_per_batch=args.limit_per_batch,
    )
    print(f"truck_manifest={truck_manifest_path}")
    print(f"car_manifest={car_manifest_path}")

    fit_output = args.fit_output
    if args.run_fit_export and fit_output is None:
        fit_output = Path(args.truck_output_dir).parent / f"{Path(args.truck_output_dir).name}_paired_offline_fits.pt"

    preference_output = args.preference_output
    if args.run_preference_build and preference_output is None:
        if fit_output is None:
            raise ValueError("internal error: fit output must be resolved before preference generation")
        preference_output = fit_output.with_name(f"{fit_output.stem}_preferences.pt")

    filtered_preference_output = args.filtered_preference_output
    if (
        args.run_preference_build
        and filtered_preference_output is None
        and any(
            value is not None
            for value in (
                args.filter_max_optimization_mean_cost,
                args.filter_min_ade,
                args.filter_max_ade,
                args.filter_max_fde,
            )
        )
    ):
        filtered_preference_output = preference_output.with_name(f"{preference_output.stem}_filtered.pt")

    artifact_manifest_output = args.artifact_manifest_output
    if artifact_manifest_output is None:
        artifact_manifest_output = Path(args.truck_output_dir).parent / f"{Path(args.truck_output_dir).name}_artifacts.json"

    artifact_manifest = {
        "truck_manifest": str(truck_manifest_path),
        "car_manifest": str(car_manifest_path),
        "truck_output_dir": str(Path(args.truck_output_dir).resolve()),
        "car_output_dir": str(Path(args.car_output_dir).resolve()),
        "limit_per_batch": args.limit_per_batch,
        "run_fit_export": bool(args.run_fit_export),
        "run_preference_build": bool(args.run_preference_build),
        "window_len": int(args.window_len),
        "max_start_distance_m": float(args.max_start_distance_m),
        "min_gap_seconds": float(args.min_gap_seconds),
        "observation_mode": args.observation_mode,
        "action_type": args.action_type,
        "fit_chunk_size": int(args.fit_chunk_size),
        "fit_log_every": int(args.fit_log_every),
        "preference_chunk_size": int(args.preference_chunk_size),
        "preference_map_log_every": int(args.preference_map_log_every),
        "filter_max_optimization_mean_cost": args.filter_max_optimization_mean_cost,
        "filter_min_ade": args.filter_min_ade,
        "filter_max_ade": args.filter_max_ade,
        "filter_max_fde": args.filter_max_fde,
    }

    if args.run_fit_export:
        LOGGER.info("Starting paired offline fit export | output=%s", fit_output)
        fit_output = export_paired_fits(
            car_root=Path(args.car_output_dir),
            truck_root=Path(args.truck_output_dir),
            output_path=fit_output,
            chunk_size=args.fit_chunk_size,
            log_every=args.fit_log_every,
        )
        artifact_manifest["fit_output"] = str(fit_output)
        artifact_manifest["fit_summary_json"] = str(fit_output.with_suffix(".json"))
        LOGGER.info("Finished paired offline fit export | output=%s", fit_output)
        print(f"fit_output={fit_output}")

    if args.run_preference_build:
        LOGGER.info(
            "Starting preference window extraction | output=%s window_len=%d max_start_distance_m=%.3f min_gap_seconds=%.3f",
            preference_output,
            args.window_len,
            args.max_start_distance_m,
            args.min_gap_seconds,
        )
        preference_output = build_truck_context_preferences(
            export_path=fit_output,
            output_path=preference_output,
            window_len=args.window_len,
            max_start_distance_m=args.max_start_distance_m,
            min_time_diff_seconds=args.min_gap_seconds,
            observation_mode=args.observation_mode,
            action_type=args.action_type,
            chunk_size=args.preference_chunk_size,
            map_log_every=args.preference_map_log_every,
        )
        artifact_manifest["preference_output"] = str(preference_output)
        artifact_manifest["preference_summary_json"] = str(preference_output.with_suffix(".json"))
        LOGGER.info("Finished preference window extraction | output=%s", preference_output)

        if any(
            value is not None
            for value in (
                args.filter_max_optimization_mean_cost,
                args.filter_min_ade,
                args.filter_max_ade,
                args.filter_max_fde,
            )
        ):
            filtered_preference_output = filter_preference_windows(
                preference_path=preference_output,
                fit_export_path=fit_output,
                output_path=filtered_preference_output,
                max_optimization_mean_cost=args.filter_max_optimization_mean_cost,
                min_ade=args.filter_min_ade,
                max_ade=args.filter_max_ade,
                max_fde=args.filter_max_fde,
            )
            artifact_manifest["filtered_preference_output"] = str(filtered_preference_output)
            artifact_manifest["filtered_preference_summary_json"] = str(filtered_preference_output.with_suffix(".json"))
            print(f"filtered_preference_output={filtered_preference_output}")
        print(f"preference_output={preference_output}")

    artifact_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    artifact_manifest_output.write_text(json.dumps(artifact_manifest, indent=2) + "\n")
    LOGGER.info("Wrote artifact manifest | path=%s", artifact_manifest_output)
    print(f"artifact_manifest={artifact_manifest_output}")


if __name__ == "__main__":
    main()
