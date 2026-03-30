import argparse
import json
from copy import deepcopy
from pathlib import Path

from pufferlib.ocean.drive.drive import save_map_binary


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def parse_map_id(map_name: str) -> int:
    stem = Path(map_name).stem
    return int(stem.split("_")[1])


def normalize_ego_to_front(map_data: dict) -> dict:
    map_data = deepcopy(map_data)
    metadata = map_data.setdefault("metadata", {})
    objects = list(map_data.get("objects", []))
    sdc_track_index = int(metadata.get("sdc_track_index", -1))
    if not (0 <= sdc_track_index < len(objects)):
        return map_data
    if sdc_track_index == 0:
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

    tracks_to_predict = metadata.get("tracks_to_predict", [])
    remapped_tracks = []
    for track in tracks_to_predict:
        if isinstance(track, dict):
            updated = dict(track)
            track_index = int(updated.get("track_index", -1))
            updated["track_index"] = index_map.get(track_index, track_index)
            remapped_tracks.append(updated)
        else:
            track_index = int(track)
            remapped_tracks.append(index_map.get(track_index, track_index))
    metadata["tracks_to_predict"] = remapped_tracks

    map_data["objects"] = reordered_objects
    return map_data


def build_sister_manifest(source_manifest: dict, car_source_dir: Path, output_dir: Path):
    scenarios = []
    for scenario in source_manifest.get("scenarios", []):
        truck_source = Path(scenario["source_json"])
        car_source = car_source_dir / truck_source.name
        if not car_source.exists():
            raise FileNotFoundError(f"Missing paired car scenario: {car_source}")

        map_data = normalize_ego_to_front(load_json(car_source))
        metadata = map_data.get("metadata", {})
        updated = dict(scenario)
        updated["source_json"] = str(car_source)
        updated["has_ego_trailer"] = bool(metadata.get("has_ego_trailer", False))
        updated["ego_trailer_track_index"] = int(metadata.get("ego_trailer_track_index", -1))
        updated["num_objects"] = len(map_data.get("objects", []))
        updated["num_roads"] = len(map_data.get("roads", []))
        scenarios.append((updated, map_data))

        output_path = output_dir / updated["map_name"]
        save_map_binary(map_data, str(output_path), parse_map_id(updated["map_name"]))

    sister_manifest = dict(source_manifest)
    sister_manifest["source_dir"] = str(car_source_dir)
    sister_manifest["paired_with"] = str(output_dir.parent / "nuplanTruckBostonTest10")
    sister_manifest["scenarios"] = [scenario for scenario, _ in scenarios]
    return sister_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Create a sister binary folder using an existing manifest and paired source JSONs."
    )
    parser.add_argument("--truck-manifest", required=True, help="Path to the existing truck selection_manifest.json")
    parser.add_argument("--truck-turning-manifest", help="Optional truck turning_extension_manifest.json")
    parser.add_argument("--car-source-dir", required=True, help="Folder containing the paired car JSON scenarios")
    parser.add_argument("--output-dir", required=True, help="Destination sister binary folder")
    args = parser.parse_args()

    truck_manifest_path = Path(args.truck_manifest)
    truck_manifest = load_json(truck_manifest_path)
    car_source_dir = Path(args.car_source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sister_manifest = build_sister_manifest(truck_manifest, car_source_dir, output_dir)
    with (output_dir / "selection_manifest.json").open("w") as f:
        json.dump(sister_manifest, f, indent=2)
        f.write("\n")

    if args.truck_turning_manifest:
        truck_turning_manifest = load_json(Path(args.truck_turning_manifest))
        turning_manifest = dict(truck_turning_manifest)
        updated_turning_scenarios = []
        for scenario in truck_turning_manifest.get("scenarios", []):
            car_source = car_source_dir / Path(scenario["source_json"]).name
            map_data = normalize_ego_to_front(load_json(car_source))
            metadata = map_data.get("metadata", {})
            updated = dict(scenario)
            updated["source_json"] = str(car_source)
            updated["has_ego_trailer"] = bool(metadata.get("has_ego_trailer", False))
            updated["ego_trailer_track_index"] = int(metadata.get("ego_trailer_track_index", -1))
            updated["num_objects"] = len(map_data.get("objects", []))
            updated["num_roads"] = len(map_data.get("roads", []))
            updated_turning_scenarios.append(updated)
        turning_manifest["scenarios"] = updated_turning_scenarios
        with (output_dir / "turning_extension_manifest.json").open("w") as f:
            json.dump(turning_manifest, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
