#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

# python3 scripts/build_combined_bin_index.py --input-dirs /proj/rpl-soro/users/x_carsk/PufferDrive/binaries/nuplan_train /proj/rpl-soro/users/x_carsk/PufferDrive/binaries/waymo_output --output-dir /proj/rpl-soro/users/x_carsk/PufferDrive/binaries/nuplan_waymo_index --clean --mix shuffle --seed 42  --log-every 1000

 # puffer train puffer_drive \
 # --env.map-dir resources/drive/binaries/nuplan_waymo_index \
 # --env.num-maps <TOTAL_PRINTED_BY_SCRIPT>

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional for this helper script
    def tqdm(it: Iterable, **kwargs):  # type: ignore
        return it


def iter_bins(src_dir: Path) -> Iterable[Path]:
    return sorted(src_dir.glob("*.bin"))


def round_robin(lists: List[List[Path]]) -> Iterable[Path]:
    max_len = max((len(lst) for lst in lists), default=0)
    for i in range(max_len):
        for lst in lists:
            if i < len(lst):
                yield lst[i]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a combined index folder of symlinks to .bin maps from multiple sources. "
            "Bins remain in their original folders; the index contains map_###.bin links."
        )
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Input folders containing .bin files (e.g., resources/drive/binaries/nuplan_train).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output folder to hold symlinks named map_###.bin.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for map_###.bin numbering.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing map_*.bin symlinks in output-dir before building.",
    )
    parser.add_argument(
        "--mix",
        choices=["concat", "round_robin", "shuffle"],
        default="round_robin",
        help="How to mix sources before indexing (default: round_robin).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for --mix shuffle (default: 0).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print a progress log every N links (0 disables; tqdm still shows if available).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for p in output_dir.glob("map_*.bin"):
            if p.is_symlink() or p.exists():
                p.unlink()

    # Collect bins
    all_bins: List[List[Path]] = []
    per_dir_counts: List[Tuple[Path, int]] = []
    for src in map(Path, args.input_dirs):
        if not src.exists():
            print(f"[skip] Missing input dir: {src}")
            continue
        bins = list(iter_bins(src))
        all_bins.append(bins)
        per_dir_counts.append((src, len(bins)))

    if per_dir_counts:
        print("[info] Input counts:")
        for src, count in per_dir_counts:
            print(f"  {src}: {count} bins")

    if args.mix == "concat":
        mixed = [p for group in all_bins for p in group]
    elif args.mix == "shuffle":
        import random

        mixed = [p for group in all_bins for p in group]
        random.Random(args.seed).shuffle(mixed)
    else:
        mixed = list(round_robin(all_bins))

    idx = args.start_index
    total = len(mixed)
    for i, bin_path in enumerate(tqdm(mixed, total=total, desc="Linking bins", unit="bin")):
        link_path = output_dir / f"map_{idx:03d}.bin"
        if link_path.exists():
            link_path.unlink()
        link_path.symlink_to(bin_path.resolve())
        idx += 1
        if args.log_every > 0 and (i + 1) % args.log_every == 0:
            print(f"[progress] linked {i + 1}/{total}")

    print(f"[done] linked_maps={idx - args.start_index} output={output_dir}")


if __name__ == "__main__":
    main()
