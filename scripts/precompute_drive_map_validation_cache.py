#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib import pufferl
from pufferlib.ocean.drive.drive import MapDatasetCatalog, MapValidationCache
from pufferlib.ocean.drive import binding
from scripts.run_packaged_drive_train import _load_packaged_config, _overlay_args


def _load_args(config_path: Path) -> dict:
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        base_args = pufferl.load_config("puffer_drive")
    finally:
        sys.argv = original_argv
    packaged = _load_packaged_config(config_path)
    return _overlay_args(base_args, packaged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute the Drive map validation cache in a separate process.")
    parser.add_argument("--config", type=Path, required=True, help="Packaged Drive INI config")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500,
        help="Emit a progress log every N processed maps",
    )
    args_ns = parser.parse_args()

    config_path = args_ns.config.expanduser().resolve()
    args = _load_args(config_path)
    env = args["env"]

    map_dir = Path(env["map_dir"]).expanduser().resolve()
    num_maps = int(env["num_maps"])
    catalog = MapDatasetCatalog.from_map_dir(str(map_dir))
    entries = catalog.limit(num_maps)
    cache = MapValidationCache(
        str(catalog.dataset_root),
        MapValidationCache.build_signature(
            dynamics_model=str(env["dynamics_model"]),
            init_mode=int(0 if env["init_mode"] == "create_all_valid" else 1),
            control_mode=int(1 if env["control_mode"] == "control_sdc_only" else 0),
            init_steps=int(env["init_steps"]),
            max_controlled_agents=int(env.get("max_controlled_agents", env["num_agents"])),
            goal_behavior=int(env["goal_behavior"]),
            goal_target_distance=float(env["goal_target_distance"]),
            force_zero_trailer_articulation_at_init=bool(env.get("force_zero_trailer_articulation_at_init", False)),
            non_kinematic_vehicle_params_override=env.get("non_kinematic_vehicle_params_override"),
        ),
    )

    updated = 0
    skipped = 0
    total = len(entries)
    start_time = time.time()
    last_log_time = start_time
    progress_interval = max(int(args_ns.progress_interval), 1)

    def _format_eta(seconds: float) -> str:
        if not seconds or seconds < 0 or seconds == float("inf"):
            return "unknown"
        minutes, sec = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}h{minutes:02d}m{sec:02d}s"
        if minutes > 0:
            return f"{minutes:d}m{sec:02d}s"
        return f"{sec:d}s"

    def _maybe_log_progress(processed: int, *, force: bool = False) -> None:
        nonlocal last_log_time
        if processed <= 0:
            return
        now = time.time()
        if not force and processed % progress_interval != 0 and (now - last_log_time) < 15.0:
            return
        elapsed = max(now - start_time, 1e-6)
        rate = processed / elapsed
        remaining = max(total - processed, 0)
        eta = remaining / max(rate, 1e-6)
        print(
            "Drive map validation cache progress: "
            f"processed={processed}/{total} "
            f"updated={updated} skipped={skipped} "
            f"rate={rate:.1f} maps/s "
            f"elapsed={_format_eta(elapsed)} "
            f"eta={_format_eta(eta)}",
            flush=True,
        )
        last_log_time = now

    for index, entry in enumerate(entries, start=1):
        if cache.get(entry.map_path) is not None:
            skipped += 1
            _maybe_log_progress(index)
            continue
        metadata = binding.inspect_map(
            map_path=entry.map_path,
            dynamics_model=0 if env["dynamics_model"] == "classic" else (1 if env["dynamics_model"] == "jerk" else 2),
            init_mode=0 if env["init_mode"] == "create_all_valid" else 1,
            control_mode=1 if env["control_mode"] == "control_sdc_only" else 0,
            init_steps=int(env["init_steps"]),
            max_controlled_agents=int(env.get("max_controlled_agents", env["num_agents"])),
            goal_behavior=int(env["goal_behavior"]),
            goal_target_distance=float(env["goal_target_distance"]),
            goal_at_gt_traj_end=int(bool(env.get("goal_at_gt_traj_end", False))),
            non_kinematic_vehicle_params_override=env.get("non_kinematic_vehicle_params_override"),
            force_zero_trailer_articulation_at_init=int(bool(env.get("force_zero_trailer_articulation_at_init", False))),
        )
        cache.set(entry.map_path, metadata)
        updated += 1
        _maybe_log_progress(index)

    _maybe_log_progress(total, force=True)

    print(
        f"Drive map validation cache ready: dataset_root={catalog.dataset_root} "
        f"selected_maps={len(entries)} updated={updated} skipped={skipped} cache={cache.cache_path}"
        ,
        flush=True,
    )


if __name__ == "__main__":
    main()
