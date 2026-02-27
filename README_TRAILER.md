# Tractor + Trailer Integration Notes

This document summarizes the tractor+trailer adaptations in PufferDrive: what changed, how it works, and what is currently tested.

## Scope

- Added support for extended scenario JSON metadata:
  - `has_ego_trailer`
  - `ego_trailer_track_index`
  - stable origin identity via `source_track_id`
- Added simulator behavior so trailer follows ego tractor.
- Kept policy interface unchanged (policy remains trailer-unaware).

## Data Format Changes

### JSON -> BIN conversion

`pufferlib/ocean/drive/drive.py` now writes an extension block at the end of each `.bin`:

- magic: `TRLR` (`0x54524C52`)
- extension version: `2`
- scenario trailer metadata:
  - `has_ego_trailer`
  - `ego_trailer_track_index`
- per-object metadata:
  - `source_track_id` hashed to stable `uint64`
  - `is_trailer`
  - `parent_track_index`
- trailer geometry metadata:
  - `non_kinematic_vehicle_params` (13 float fields; includes `tractor2hitch` and `trailer2hitch`)

Notes:
- Converter expects extension v2 metadata and writes non-kinematic trailer geometry every time.
- Object/road IDs are normalized to deterministic signed int32 so large/string IDs do not fail conversion.

### BIN loading

`pufferlib/ocean/drive/drive.h` loader requires extension v2 (magic + version + 13 non-kinematic params).

## Simulator Behavior

### Trailer coupling

- Trailer is treated as a dependent body attached to ego tractor.
- Trailer pose is updated each step with articulation kinematics (hitch-based approximation).
- Trailer is also updated on ego respawn/reset paths.

### Control and observations

- Trailer is never directly policy-controlled.
- Trailer is spawned in scene for interaction checks even if not controlled.
- Ego partner observation excludes its own trailer (to avoid self-partner leakage).
- Policy action/observation layout is unchanged from pre-trailer setup.

### Collision and off-road checks

- For ego metrics, collision checks include both tractor box and trailer box.
- Ego<->its-own-trailer self-collision pair is excluded.
- Off-road checks include trailer footprint for ego metrics.
- Reward application is still one collision/off-road state per controlled agent per step (no double penalty stacking in one step).

## What Is Tested

## 1) Conversion extension correctness

- `tests/test_drive_json_to_bin.py`
- Verifies:
  - base binary fields
  - trailer extension block fields
  - per-object trailer metadata
  - deterministic handling of large/string IDs

## 2) Converted bin is simulator-usable

- `tests/test_drive_bin_simulator_load.py::test_generated_bin_loads_and_steps_in_drive`
- Verifies reset+step works with generated `.bin`.

## 3) Trailer contributes to collision behavior

- `tests/test_drive_bin_simulator_load.py::test_trailer_pose_follow_triggers_collision_after_step`
- A/B test with trailer metadata enabled vs disabled.
- Confirms trailer coupling can change collision outcome.

## 4) Trailer off-road A/B behavior

- `tests/test_drive_bin_simulator_load.py::test_trailer_only_offroad_penalty`
- A/B setup where trailer-enabled case is worse than baseline.
- Baseline disables the secondary trailer object in the no-trailer variant to avoid confounding with regular vehicle collision.

## 5) Baseline off-road pipeline sanity

- `tests/test_drive_bin_simulator_load.py::test_offroad_penalty_baseline_for_tractor`
- Confirms off-road penalty is active for tractor-road-edge intersections.

## 6) Visualization tooling works

- `tests/test_drive_trailer_visualization.py`
- Verifies `.bin` parsing and PNG generation for trailer/road-edge visualization.

## Visualization Utilities

### Static frames

```bash
python scripts/visualize_trailer_scene.py \
  --bin /path/to/map_000.bin \
  --output trailer_debug.png \
  --frames 0,10,20
```

Optional:
- `--no-other-objects` to only show tractor+trailer and roads.

### MP4 video generation

```bash
scripts/make_trailer_video.sh \
  --bin /path/to/map_000.bin \
  --output /tmp/trailer_debug.mp4 \
  --start 0 --end 90 --fps 10
```

## Current Constraints / Notes

- Ego collision/off-road is exposed as a single aggregated signal (`obs[..., 5]`), not separate tractor-vs-trailer flags.
- Trailer geometry uses a practical hitch/length approximation (dataset-specific exact articulation params are not yet wired).
- Policy remains intentionally trailer-unaware for now.
