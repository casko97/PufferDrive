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


def _window_action_counts_from_manifest(payload, manifest):
    window_count = int(manifest["window_count"])
    if window_count <= 0:
        return np.zeros((0, _CLASSIC_DISCRETE_ACTIONS), dtype=np.int16)

    action_tensor = payload["action"].cpu().long()
    window_indices = manifest["window_indices"].cpu().long()
    valid_lengths = manifest["valid_lengths"].cpu().long()
    counts = np.zeros((window_count, _CLASSIC_DISCRETE_ACTIONS), dtype=np.int16)

    for window_index in range(window_count):
        valid_length = int(valid_lengths[window_index].item())
        if valid_length <= 0:
            continue
        row_indices = window_indices[window_index, :valid_length]
        if torch.any(row_indices < 0):
            raise ValueError("Window manifest contains invalid row indices")
        window_actions = action_tensor[row_indices].numpy()
        counts[window_index] = np.bincount(window_actions, minlength=_CLASSIC_DISCRETE_ACTIONS).astype(
            np.int16, copy=False
        )

    return counts


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


def _action_rarity_weights(global_counts, *, max_multiplier):
    global_counts = np.asarray(global_counts, dtype=np.float64)
    positive = global_counts > 0
    if not np.any(positive):
        raise ValueError("Source dataset does not contain any positive action counts")
    uniform_target = float(global_counts[positive].sum()) / float(np.count_nonzero(positive))
    rarity = np.ones_like(global_counts, dtype=np.float64)
    rarity[positive] = uniform_target / global_counts[positive]
    rarity = np.clip(rarity, 1.0, float(max_multiplier))
    return rarity


def _first_pass_collect_counts(shard_paths, window_set_name, *, log_every):
    start_time = time.time()
    global_counts = np.zeros(_CLASSIC_DISCRETE_ACTIONS, dtype=np.int64)
    shard_action_counts = []
    total_windows = 0
    total_action_occurrences = 0

    _log(f"[pass1] counting per-window action usage across {len(shard_paths)} shards")
    for shard_index, shard_path in enumerate(shard_paths, start=1):
        payload = torch.load(shard_path, map_location="cpu")
        manifest = _load_manifest(payload, window_set_name, shard_path)
        window_action_counts = _window_action_counts_from_manifest(payload, manifest)
        local_counts = window_action_counts.sum(axis=0, dtype=np.int64)
        shard_action_counts.append(local_counts.astype(np.int64, copy=False))
        global_counts += local_counts
        total_windows += int(window_action_counts.shape[0])
        total_action_occurrences += int(local_counts.sum())

        if shard_index % log_every == 0 or shard_index == len(shard_paths):
            nonzero = int(np.count_nonzero(global_counts))
            _log(
                f"[pass1] {shard_index}/{len(shard_paths)} shards | "
                f"windows={total_windows:,} | action_occurrences={total_action_occurrences:,} | "
                f"nonzero_actions={nonzero}/{_CLASSIC_DISCRETE_ACTIONS} | "
                f"elapsed={_summarize_elapsed(start_time)}"
            )

    positive = global_counts[global_counts > 0]
    if positive.size == 0:
        raise ValueError("Source dataset does not contain any labeled windows")

    _log(
        f"[pass1] done | total_windows={int(total_windows):,} | "
        f"total_action_occurrences={int(total_action_occurrences):,} | "
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


def _select_windows_for_local_quotas(window_action_counts, quota_map, seed):
    window_count = int(window_action_counts.shape[0])
    if window_count <= 0 or not quota_map:
        return np.zeros((0,), dtype=np.int64), np.zeros((_CLASSIC_DISCRETE_ACTIONS,), dtype=np.int64)

    deficits = np.zeros((_CLASSIC_DISCRETE_ACTIONS,), dtype=np.int64)
    for action_id, desired in quota_map.items():
        if desired > 0:
            deficits[int(action_id)] = int(desired)

    if not np.any(deficits > 0):
        return np.zeros((0,), dtype=np.int64), deficits

    remaining = np.arange(window_count, dtype=np.int64)
    selected = []
    selected_counts = np.zeros((_CLASSIC_DISCRETE_ACTIONS,), dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    while remaining.size > 0 and np.any(deficits > 0):
        remaining_counts = window_action_counts[remaining]
        useful = np.minimum(remaining_counts, deficits[None, :]).sum(axis=1, dtype=np.int64)
        best_useful = int(useful.max(initial=0))
        if best_useful <= 0:
            break

        candidate_positions = np.flatnonzero(useful == best_useful)
        candidate_indices = remaining[candidate_positions]
        if candidate_indices.size > 1:
            action_density = remaining_counts[candidate_positions].sum(axis=1, dtype=np.int64)
            best_density = int(action_density.max(initial=0))
            dense_positions = candidate_positions[action_density == best_density]
            chosen_position = int(rng.choice(dense_positions))
        else:
            chosen_position = int(candidate_positions[0])

        chosen_index = int(remaining[chosen_position])
        selected.append(chosen_index)
        chosen_counts = window_action_counts[chosen_index].astype(np.int64, copy=False)
        selected_counts += chosen_counts
        deficits = np.maximum(deficits - chosen_counts, 0)
        remaining = np.delete(remaining, chosen_position)

    selected_array = np.asarray(selected, dtype=np.int64)
    return selected_array, selected_counts


def _window_sampling_weights(window_action_counts, action_rarity_weights, balance_fraction):
    window_action_counts = np.asarray(window_action_counts, dtype=np.float64)
    if window_action_counts.ndim != 2:
        raise ValueError("window_action_counts must be rank-2")
    window_lengths = window_action_counts.sum(axis=1)
    scores = np.ones((window_action_counts.shape[0],), dtype=np.float64)
    valid = window_lengths > 0
    if np.any(valid):
        rarity_score = (window_action_counts[valid] * action_rarity_weights[None, :]).sum(axis=1) / window_lengths[valid]
        scores[valid] = (1.0 - float(balance_fraction)) + float(balance_fraction) * rarity_score
    scores = np.clip(scores, 1e-12, None)
    return scores


def _sample_windows_by_rarity(
    window_action_counts,
    *,
    target_window_count,
    action_rarity_weights,
    balance_fraction,
    seed,
):
    window_count = int(window_action_counts.shape[0])
    if target_window_count <= 0 or window_count <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((_CLASSIC_DISCRETE_ACTIONS,), dtype=np.int64)

    if float(balance_fraction) <= 0.0 and int(target_window_count) == int(window_count):
        selected_indices = np.arange(window_count, dtype=np.int64)
        selected_counts = window_action_counts.sum(axis=0, dtype=np.int64)
        return selected_indices, selected_counts

    weights = _window_sampling_weights(window_action_counts, action_rarity_weights, balance_fraction)
    probs = weights / float(weights.sum())
    rng = np.random.default_rng(int(seed))
    expected = probs * float(target_window_count)
    counts = np.floor(expected).astype(np.int64)
    remaining = int(target_window_count) - int(counts.sum())
    if remaining > 0:
        residual = expected - counts
        order = np.argsort(-residual)
        if remaining < order.size:
            cutoff = residual[order[remaining - 1]]
            tied = order[residual[order] == cutoff]
            if tied.size > 1:
                rng.shuffle(tied)
                order = np.concatenate([order[residual[order] > cutoff], tied, order[residual[order] < cutoff]])
        counts[order[:remaining]] += 1
    selected_indices = np.repeat(np.arange(window_count, dtype=np.int64), counts)
    if selected_indices.size > 1:
        rng.shuffle(selected_indices)
    selected_counts = window_action_counts[selected_indices].sum(axis=0, dtype=np.int64)
    return selected_indices.astype(np.int64, copy=False), selected_counts.astype(np.int64, copy=False)


def create_balanced_bc_window_subset(
    source_dir,
    output_dir,
    *,
    window_set_name="window_metadata",
    target_count_per_action,
    seed=0,
    log_every=250,
    rebalance_strategy="hard_quota",
    balance_fraction=0.0,
    target_window_fraction=1.0,
    max_rarity_multiplier=10.0,
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
    rebalance_strategy = str(rebalance_strategy)
    if rebalance_strategy not in {"hard_quota", "fractional_rarity"}:
        raise ValueError(f"Unsupported rebalance_strategy: {rebalance_strategy}")
    balance_fraction = float(balance_fraction)
    if not 0.0 <= balance_fraction <= 1.0:
        raise ValueError("balance_fraction must be in [0, 1]")
    target_window_fraction = float(target_window_fraction)
    if target_window_fraction <= 0.0:
        raise ValueError("target_window_fraction must be positive")
    max_rarity_multiplier = float(max_rarity_multiplier)
    if max_rarity_multiplier < 1.0:
        raise ValueError("max_rarity_multiplier must be >= 1")

    global_counts, shard_action_counts, total_windows = _first_pass_collect_counts(
        shard_paths,
        window_set_name,
        log_every=max(1, int(log_every)),
    )

    feasible_target = None
    quotas_by_shard = None
    map_usage_by_action = {}
    action_rarity_weights = None

    if rebalance_strategy == "hard_quota":
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
    else:
        action_rarity_weights = _action_rarity_weights(global_counts, max_multiplier=max_rarity_multiplier)
        _log(
            f"[fractional] using balance_fraction={balance_fraction:.3f} "
            f"target_window_fraction={target_window_fraction:.3f} "
            f"max_rarity_multiplier={max_rarity_multiplier:.3f}"
        )

    write_start = time.time()
    _log(f"[pass2] writing subset shards to {output_dir}")
    subset_counts = np.zeros_like(global_counts)
    subset_total_windows = 0
    written_shards = 0

    for shard_index, shard_path in enumerate(shard_paths, start=1):
        payload = torch.load(shard_path, map_location="cpu")
        manifest = _load_manifest(payload, window_set_name, shard_path)
        window_action_counts = _window_action_counts_from_manifest(payload, manifest)
        if rebalance_strategy == "hard_quota":
            quota_map = quotas_by_shard[shard_index - 1]
            selected_indices, selected_counts = _select_windows_for_local_quotas(
                window_action_counts,
                quota_map,
                int(seed) + (shard_index - 1) * 4099 + 29,
            )
        else:
            source_window_count = int(window_action_counts.shape[0])
            target_window_count = max(1, int(round(float(source_window_count) * target_window_fraction))) if source_window_count > 0 else 0
            selected_indices, selected_counts = _sample_windows_by_rarity(
                window_action_counts,
                target_window_count=target_window_count,
                action_rarity_weights=action_rarity_weights,
                balance_fraction=balance_fraction,
                seed=int(seed) + (shard_index - 1) * 4099 + 29,
            )

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
        metadata["subset_rebalance_strategy"] = str(rebalance_strategy)
        if feasible_target is not None:
            metadata["subset_target_count_per_action"] = int(feasible_target)
        if rebalance_strategy == "fractional_rarity":
            metadata["subset_balance_fraction"] = float(balance_fraction)
            metadata["subset_target_window_fraction"] = float(target_window_fraction)
            metadata["subset_max_rarity_multiplier"] = float(max_rarity_multiplier)
        updated_payload["metadata"] = metadata

        out_path = output_dir / Path(shard_path).name
        torch.save(updated_payload, out_path)

        if selected_indices.size > 0:
            subset_counts += selected_counts
            subset_total_windows += int(selected_indices.size)
        written_shards += 1

        if written_shards % log_every == 0 or written_shards == len(shard_paths):
            _log(
                f"[pass2] {written_shards}/{len(shard_paths)} shards | "
                f"subset_windows={subset_total_windows:,} | "
                f"subset_action_occurrences={int(subset_counts.sum()):,} | "
                f"elapsed={_summarize_elapsed(write_start)}"
            )

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "window_set_name": str(window_set_name),
        "rebalance_strategy": str(rebalance_strategy),
        "seed": int(seed),
        "requested_target_count_per_action": int(target_count_per_action),
        "action_space_size": int(_CLASSIC_DISCRETE_ACTIONS),
        "source_total_windows": int(total_windows),
        "subset_total_windows": int(subset_total_windows),
        "source_total_action_occurrences": int(global_counts.sum()),
        "subset_total_action_occurrences": int(subset_counts.sum()),
        "source_nonzero_actions": int(np.count_nonzero(global_counts)),
        "subset_nonzero_actions": int(np.count_nonzero(subset_counts)),
        "source_action_counts": global_counts.tolist(),
        "subset_action_counts": subset_counts.tolist(),
        "map_usage_by_action": map_usage_by_action,
    }
    if feasible_target is not None:
        summary["actual_target_count_per_action"] = int(feasible_target)
    if rebalance_strategy == "fractional_rarity":
        summary["balance_fraction"] = float(balance_fraction)
        summary["target_window_fraction"] = float(target_window_fraction)
        summary["max_rarity_multiplier"] = float(max_rarity_multiplier)
    summary_path = output_dir / "subset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _log(f"[done] wrote subset summary to {summary_path}")
    _log(
        f"[done] source_windows={int(total_windows):,} | subset_windows={int(subset_total_windows):,} | "
        f"subset_action_occurrences={int(subset_counts.sum()):,}"
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
        default=1,
        type=int,
        help="Maximum number of windows to keep per joint action in hard_quota mode",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed for deterministic subset selection")
    parser.add_argument(
        "--rebalance-strategy",
        choices=["hard_quota", "fractional_rarity"],
        default="hard_quota",
        help="How to rebalance windows: hard equalization or mild rarity-weighted resampling",
    )
    parser.add_argument(
        "--balance-fraction",
        default=0.0,
        type=float,
        help="For fractional_rarity mode, interpolate between original sampling (0) and rarity-weighted sampling (1)",
    )
    parser.add_argument(
        "--target-window-fraction",
        default=1.0,
        type=float,
        help="For fractional_rarity mode, resample this fraction of the original windows per shard",
    )
    parser.add_argument(
        "--max-rarity-multiplier",
        default=10.0,
        type=float,
        help="For fractional_rarity mode, clip action rarity upweighting to this maximum multiplier",
    )
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
        rebalance_strategy=args.rebalance_strategy,
        balance_fraction=args.balance_fraction,
        target_window_fraction=args.target_window_fraction,
        max_rarity_multiplier=args.max_rarity_multiplier,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
