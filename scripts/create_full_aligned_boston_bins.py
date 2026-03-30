from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from scripts.build_truck_context_preferences import build_truck_context_preferences
from scripts.export_paired_offline_fits import export_paired_fits
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
    preference_payload = torch.load(preference_path, map_location="cpu")
    fit_payload = torch.load(fit_export_path, map_location="cpu")

    preferred_sa = np.asarray(preference_payload["preferred_sa"], dtype=np.float32)
    rejected_sa = np.asarray(preference_payload["rejected_sa"], dtype=np.float32)
    labels = np.asarray(preference_payload["labels"], dtype=np.float32)
    window_metadata = list(preference_payload.get("window_metadata", []))
    pairs = fit_payload["pairs"]

    keep_indices: list[int] = []
    kept_window_metadata: list[dict] = []
    filtered_out = 0

    for idx, window_meta in enumerate(window_metadata):
        map_name = window_meta["map_name"]
        pair = pairs.get(map_name)
        if pair is None:
            filtered_out += 1
            continue

        truck_side = pair.get("truck", {})
        car_side = pair.get("car", {})
        if truck_side.get("status") != "ok" or car_side.get("status") != "ok":
            filtered_out += 1
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
            keep_indices.append(idx)
            enriched_meta = dict(window_meta)
            enriched_meta["truck_optimization_mean_cost"] = truck_mean_cost
            enriched_meta["car_optimization_mean_cost"] = car_mean_cost
            kept_window_metadata.append(enriched_meta)
        else:
            filtered_out += 1

    if keep_indices:
        preferred_sa = preferred_sa[np.asarray(keep_indices, dtype=np.int64)]
        rejected_sa = rejected_sa[np.asarray(keep_indices, dtype=np.int64)]
        labels = labels[np.asarray(keep_indices, dtype=np.int64)]
    else:
        preferred_sa = preferred_sa[:0]
        rejected_sa = rejected_sa[:0]
        labels = labels[:0]

    filtered_payload = dict(preference_payload)
    filtered_metadata = dict(preference_payload.get("metadata", {}))
    filtered_metadata["filtering"] = {
        "source_preference_path": str(preference_path),
        "source_fit_export_path": str(fit_export_path),
        "max_optimization_mean_cost": max_optimization_mean_cost,
        "min_ade": min_ade,
        "max_ade": max_ade,
        "max_fde": max_fde,
        "kept_windows": int(len(keep_indices)),
        "filtered_out_windows": int(filtered_out),
    }
    filtered_metadata["total_windows"] = int(len(keep_indices))
    filtered_payload["metadata"] = filtered_metadata
    filtered_payload["preferred_sa"] = preferred_sa
    filtered_payload["rejected_sa"] = rejected_sa
    filtered_payload["labels"] = labels
    filtered_payload["window_metadata"] = kept_window_metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered_payload, output_path)
    summary = {
        "output_path": str(output_path),
        "source_preference_path": str(preference_path),
        "source_fit_export_path": str(fit_export_path),
        "max_optimization_mean_cost": max_optimization_mean_cost,
        "min_ade": min_ade,
        "max_ade": max_ade,
        "max_fde": max_fde,
        "kept_windows": int(len(keep_indices)),
        "filtered_out_windows": int(filtered_out),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    LOGGER.info(
        "Finished filtering preference windows | kept=%d filtered_out=%d output=%s",
        len(keep_indices),
        filtered_out,
        output_path,
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
        required=True,
        help="Truck JSON source directory. Pass once per batch.",
    )
    parser.add_argument(
        "--car-source-dir",
        dest="car_source_dirs",
        action="append",
        required=True,
        help="Car JSON source directory. Pass once per batch in the same order as --truck-source-dir.",
    )
    parser.add_argument("--truck-output-dir", required=True, help="Destination folder for truck binary files.")
    parser.add_argument("--car-output-dir", required=True, help="Destination folder for car binary files.")
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

    if args.run_preference_build and not args.run_fit_export:
        raise ValueError("--run-preference-build requires --run-fit-export")

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
