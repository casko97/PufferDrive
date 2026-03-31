#!/usr/bin/env python3
from __future__ import annotations

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, Tuple

# Usage:
#python3 scripts/convert_scenarios_to_bins.py --input-dirs nuplan_train waymo_output --output-root pufferlib/resources/drive/binaries --num-workers 10 --max-maps 20

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional for this helper script
    def tqdm(it: Iterable, **kwargs):  # type: ignore
        return it

from pufferlib.ocean.drive.drive import load_map


Task = Tuple[int, Path, Path]


def _convert_single(task: Task) -> Tuple[int, str, bool, str | None]:
    idx, json_path, bin_path = task
    try:
        load_map(str(json_path), idx, str(bin_path))
        return idx, json_path.name, True, None
    except Exception as exc:  # pragma: no cover - just reporting
        return idx, json_path.name, False, str(exc)


def _collect_tasks(
    input_dir: Path, output_dir: Path, max_maps: int, force: bool
) -> Tuple[list[Task], int, int]:
    json_files = sorted(input_dir.glob("*.json"))
    if max_maps is not None:
        json_files = json_files[:max_maps]

    tasks: list[Task] = []
    skipped = 0
    for idx, json_path in enumerate(json_files):
        bin_name = f"map_{idx:03d}.bin"
        bin_path = output_dir / bin_name
        if bin_path.exists() and not force:
            skipped += 1
            continue
        tasks.append((idx, json_path, bin_path))
    return tasks, len(json_files), skipped


def _process_dataset(
    input_dir: Path,
    output_root: Path,
    max_maps: int,
    num_workers: int | None,
    force: bool,
) -> None:
    if not input_dir.exists():
        print(f"[skip] Missing input dir: {input_dir}")
        return

    output_dir = output_root / input_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks, total, skipped = _collect_tasks(input_dir, output_dir, max_maps, force)
    if total == 0:
        print(f"[skip] No JSON files found in {input_dir}")
        return

    if not tasks:
        print(f"[ok] {input_dir.name}: all {total} maps already converted in {output_dir}")
        return

    if num_workers is None:
        num_workers = cpu_count()

    converted = 0
    failed = 0
    if num_workers <= 1 or len(tasks) <= 1:
        for task in tqdm(tasks, total=len(tasks), desc=f"Converting {input_dir.name}", unit="map"):
            _, _, success, _ = _convert_single(task)
            if success:
                converted += 1
            else:
                failed += 1
    else:
        with Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_convert_single, tasks),
                    total=len(tasks),
                    desc=f"Converting {input_dir.name}",
                    unit="map",
                )
            )
        for _, _, success, _ in results:
            if success:
                converted += 1
            else:
                failed += 1

    print(
        f"[done] {input_dir.name}: total={total} converted={converted} skipped={skipped} failed={failed} "
        f"output={output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert scenario JSONs to .bin, skipping scenarios already converted."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["nuplan_train", "waymo_output"],
        help="List of input folders containing scenario JSONs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="pufferlib/resources/drive/binaries",
        help="Root folder for binaries; output is grouped by input folder name.",
    )
    parser.add_argument(
        "--max-maps",
        type=int,
        default=50_000,
        help="Maximum number of maps to process per input folder.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: all CPU cores). Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild bins even if the target file already exists.",
    )

    args = parser.parse_args()
    output_root = Path(args.output_root)
    for input_dir in args.input_dirs:
        _process_dataset(Path(input_dir), output_root, args.max_maps, args.num_workers, args.force)


if __name__ == "__main__":
    main()
