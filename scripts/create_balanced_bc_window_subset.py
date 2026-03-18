from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pufferlib.ocean.drive.drive import (
    _CLASSIC_DISCRETE_ACTIONS,
    _embedded_named_sequence_manifest,
    _embedded_sequence_manifest,
    _list_bc_shards,
    _validate_sequence_manifest,
)


def _log(message):
    print(message, flush=True)


def _load_manifest(payload, window_set_name, shard_path):
    if window_set_name == "window_metadata":
        manifest = _embedded_sequence_manifest(payload, shard_path=str(shard_path))
    else:
        manifest = _embedded_named_sequence_manifest(payload, window_set_name, shard_path=str(shard_path))
    if manifest is None:
        raise ValueError(f"Shard {shard_path} does not contain window manifest '{window_set_name}'")
    return _validate_sequence_manifest(manifest, shard_path=str(shard_path))


def _window_action_labels_from_manifest(payload, manifest):
    window_count = int(manifest["window_count"])
    if window_count <= 0:
        return np.zeros((0,), dtype=np.int64)
    first_rows = manifest["window_indices"][:, 0].long()
    if torch.any(first_rows < 0):
        raise ValueError("Window manifest contains invalid first-row indices")
    return payload["action"][first_rows].cpu().numpy().astype(np.int64, copy=False)


def _subset_manifest(manifest, selected_indices):
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if selected_indices.size == 0:
        window_indices = manifest["window_indices"][:0].clone()
        valid_lengths = manifest["valid_lengths"][:0].clone()
    else:
        tensor_indices = torch.as_tensor(selected_indices, dtype=torch.long)
        window_indices = manifest["window_indices"][tensor_indices].clone()
        valid_lengths = manifest["valid_lengths"][tensor_indices].clone()
    return {
        "version": int(manifest["version"]),
        "seq_len": int(manifest["seq_len"]),
        "stride": int(manifest["stride"]),
        "window_count": int(window_indices.shape[0]),
        "window_indices": window_indices,
        "valid_lengths": valid_lengths,
    }


def _summarize_elapsed(start_time):
    elapsed = max(time.time() - start_time, 1e-6)
    return f"{elapsed:.1f}s"


def _first_pass_collect_counts(shard_paths, window_set_name, *, log_every):
    start_time = time.time()
    global_counts = np.zeros(_CLASSIC_DISCRETE_ACTIONS, dtype=np.int64)
    shard_action_counts = []
    total_windows = 0

    _log(f"[pass1] counting actions across {len(shard_paths)} shards")
    for shard_index, shard_path in enumerate(shard_paths, start=1):
        payload = torch.load(shard_path, map_location="cpu")
        manifest = _load_manifest(payload, window_set_name, shard_path)
        labels = _window_action_labels_from_manifest(payload, manifest)
        local_counts = np.bincount(labels, minlength=_CLASSIC_DISCRETE_ACTIONS)
        shard_action_counts.append(local_counts.astype(np.int64, copy=False))
        global_counts += local_counts
        total_windows += int(labels.size)

        if shard_index % log_every == 0 or shard_index == len(shard_paths):
            nonzero = int(np.count_nonzero(global_counts))
            _log(
                f"[pass1] {shard_index}/{len(shard_paths)} shards | "
                f"windows={total_windows:,} | nonzero_actions={nonzero}/{_CLASSIC_DISCRETE_ACTIONS} | "
                f"elapsed={_summarize_elapsed(start_time)}"
            )

    positive = global_counts[global_counts > 0]
    if positive.size == 0:
        raise ValueError("Source dataset does not contain any labeled windows")

    _log(
        f"[pass1] done | total_windows={int(total_windows):,} | "
        f"min_positive={int(positive.min()):,} | max_positive={int(positive.max()):,} | "
        f"elapsed={_summarize_elapsed(start_time)}"
    )
    return global_counts, shard_action_counts, int(total_windows)


def _allocate_action_quotas(shard_action_counts, global_counts, target_count_per_action, seed):
    quotas_by_shard = [defaultdict(int) for _ in range(len(shard_action_counts))]
    map_usage_by_action = {}

    for action_id, global_count in enumerate(global_counts.tolist()):
        if global_count <= 0:
            continue

        target = min(int(target_count_per_action), int(global_count))
        if target <= 0:
            continue

        available_shards = [
            shard_index
            for shard_index, local_counts in enumerate(shard_action_counts)
            if int(local_counts[action_id]) > 0
        ]
        if not available_shards:
            continue

        rng = np.random.default_rng(int(seed) + int(action_id) * 1009 + 17)
        rng.shuffle(available_shards)
        remaining = {shard_index: int(shard_action_counts[shard_index][action_id]) for shard_index in available_shards}
        allocated = 0
        active = list(available_shards)

        while allocated < target and active:
            next_active = []
            progressed = False
            for shard_index in active:
                if remaining[shard_index] <= 0:
                    continue
                quotas_by_shard[shard_index][action_id] += 1
                remaining[shard_index] -= 1
                allocated += 1
                progressed = True
                if remaining[shard_index] > 0 and allocated < target:
                    next_active.append(shard_index)
                if allocated >= target:
                    break
            if not progressed:
                break
            active = next_active

        map_usage_by_action[int(action_id)] = len(
            [shard_index for shard_index, quota_map in enumerate(quotas_by_shard) if quota_map.get(action_id, 0) > 0]
        )

    return quotas_by_shard, map_usage_by_action


def create_balanced_bc_window_subset(
    source_dir,
    output_dir,
    *,
    window_set_name="window_metadata",
    target_count_per_action,
    seed=0,
    log_every=250,
):
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    shard_paths = _list_bc_shards(source_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No BC shard files found in {source_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_count_per_action = int(target_count_per_action)
    if target_count_per_action <= 0:
        raise ValueError("target_count_per_action must be positive")

    global_counts, shard_action_counts, total_windows = _first_pass_collect_counts(
        shard_paths,
        window_set_name,
        log_every=max(1, int(log_every)),
    )

    positive = global_counts[global_counts > 0]
    feasible_target = min(target_count_per_action, int(positive.min()))
    if feasible_target < target_count_per_action:
        _log(
            f"[quota] requested target_count_per_action={target_count_per_action:,} exceeds feasible min count; "
            f"using {feasible_target:,} instead"
        )

    allocation_start = time.time()
    _log(f"[quota] allocating balanced per-action quotas with target={feasible_target:,}")
    quotas_by_shard, map_usage_by_action = _allocate_action_quotas(
        shard_action_counts,
        global_counts,
        feasible_target,
        seed,
    )
    _log(f"[quota] done | elapsed={_summarize_elapsed(allocation_start)}")

    write_start = time.time()
    _log(f"[pass2] writing subset shards to {output_dir}")
    subset_counts = np.zeros_like(global_counts)
    subset_total_windows = 0
    written_shards = 0

    for shard_index, shard_path in enumerate(shard_paths, start=1):
        payload = torch.load(shard_path, map_location="cpu")
        manifest = _load_manifest(payload, window_set_name, shard_path)
        labels = _window_action_labels_from_manifest(payload, manifest)

        quota_map = quotas_by_shard[shard_index - 1]
        selected_indices = []
        for action_id, desired in sorted(quota_map.items()):
            if desired <= 0:
                continue
            local_indices = np.flatnonzero(labels == int(action_id))
            if local_indices.size == 0:
                continue
            rng = np.random.default_rng(int(seed) + (shard_index - 1) * 10007 + int(action_id))
            chosen = rng.choice(local_indices, size=min(int(desired), int(local_indices.size)), replace=False)
            selected_indices.append(chosen.astype(np.int64, copy=False))

        if selected_indices:
            selected_indices = np.concatenate(selected_indices)
            rng = np.random.default_rng(int(seed) + (shard_index - 1) * 4099 + 29)
            rng.shuffle(selected_indices)
        else:
            selected_indices = np.zeros((0,), dtype=np.int64)

        subset_manifest = _subset_manifest(manifest, selected_indices)
        updated_payload = dict(payload)
        metadata = dict(updated_payload.get("metadata", {}))
        window_sets = dict(updated_payload.get("window_sets", {})) if isinstance(updated_payload.get("window_sets"), dict) else None

        updated_payload["window_metadata"] = subset_manifest
        if window_sets is not None:
            window_sets["base_windows"] = subset_manifest
            updated_payload["window_sets"] = window_sets

        metadata["window_count"] = int(subset_manifest["window_count"])
        metadata["window_seq_len"] = int(subset_manifest["seq_len"])
        metadata["window_stride"] = int(subset_manifest["stride"])
        metadata["subset_source_dir"] = str(source_dir)
        metadata["subset_window_set"] = str(window_set_name)
        metadata["subset_seed"] = int(seed)
        metadata["subset_target_count_per_action"] = int(feasible_target)
        updated_payload["metadata"] = metadata

        out_path = output_dir / Path(shard_path).name
        torch.save(updated_payload, out_path)

        if selected_indices.size > 0:
            selected_labels = labels[selected_indices]
            subset_counts += np.bincount(selected_labels, minlength=_CLASSIC_DISCRETE_ACTIONS)
            subset_total_windows += int(selected_labels.size)
        written_shards += 1

        if written_shards % log_every == 0 or written_shards == len(shard_paths):
            _log(
                f"[pass2] {written_shards}/{len(shard_paths)} shards | "
                f"subset_windows={subset_total_windows:,} | elapsed={_summarize_elapsed(write_start)}"
            )

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "window_set_name": str(window_set_name),
        "seed": int(seed),
        "requested_target_count_per_action": int(target_count_per_action),
        "actual_target_count_per_action": int(feasible_target),
        "action_space_size": int(_CLASSIC_DISCRETE_ACTIONS),
        "source_total_windows": int(total_windows),
        "subset_total_windows": int(subset_total_windows),
        "source_nonzero_actions": int(np.count_nonzero(global_counts)),
        "subset_nonzero_actions": int(np.count_nonzero(subset_counts)),
        "source_action_counts": global_counts.tolist(),
        "subset_action_counts": subset_counts.tolist(),
        "map_usage_by_action": map_usage_by_action,
    }
    summary_path = output_dir / "subset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _log(f"[done] wrote subset summary to {summary_path}")
    _log(
        f"[done] source_windows={int(total_windows):,} | subset_windows={int(subset_total_windows):,} | "
        f"actual_target_per_action={int(feasible_target):,}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Create a smaller balanced BC window subset while preserving complete windows and spreading selections across source shards"
    )
    parser.add_argument("--source-dir", required=True, type=str, help="Path to the source BC shard directory")
    parser.add_argument("--output-dir", required=True, type=str, help="Path to write the balanced subset shards")
    parser.add_argument(
        "--window-set-name",
        default="window_metadata",
        type=str,
        help="Embedded window manifest to subset (default: window_metadata)",
    )
    parser.add_argument(
        "--target-count-per-action",
        required=True,
        type=int,
        help="Maximum number of windows to keep per joint action in the subset",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed for deterministic subset selection")
    parser.add_argument(
        "--log-every",
        default=250,
        type=int,
        help="Emit progress logs every N shards in each pass",
    )
    args = parser.parse_args()

    summary = create_balanced_bc_window_subset(
        args.source_dir,
        args.output_dir,
        window_set_name=args.window_set_name,
        target_count_per_action=args.target_count_per_action,
        seed=args.seed,
        log_every=args.log_every,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
