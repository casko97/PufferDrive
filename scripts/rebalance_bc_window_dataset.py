from __future__ import annotations

import argparse
import json
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


def _window_action_labels_from_manifest(payload, manifest):
    manifest = _validate_sequence_manifest(manifest)
    window_count = int(manifest["window_count"])
    if window_count <= 0:
        return np.zeros((0,), dtype=np.int64)
    first_rows = manifest["window_indices"][:, 0].long()
    if torch.any(first_rows < 0):
        raise ValueError("Window manifest contains invalid first-row indices")
    return payload["action"][first_rows].cpu().numpy().astype(np.int64, copy=False)


def _load_manifest(payload, window_set_name, shard_path):
    if window_set_name == "window_metadata":
        manifest = _embedded_sequence_manifest(payload, shard_path=str(shard_path))
    else:
        manifest = _embedded_named_sequence_manifest(payload, window_set_name, shard_path=str(shard_path))
    if manifest is None:
        raise ValueError(f"Shard {shard_path} does not contain window manifest '{window_set_name}'")
    return manifest


def _target_count(global_counts, strategy):
    nonzero = global_counts[global_counts > 0]
    if nonzero.size == 0:
        raise ValueError("No windows found for balancing")
    if strategy == "oversample_to_max":
        return int(nonzero.max())
    if strategy == "downsample_to_min":
        return int(nonzero.min())
    raise ValueError(f"Unsupported strategy: {strategy}")


def _resample_manifest(payload, manifest, global_counts, *, target_count, seed, shard_index, strategy):
    manifest = _validate_sequence_manifest(manifest)
    labels = _window_action_labels_from_manifest(payload, manifest)
    selected_indices = []

    for action_id, global_count in enumerate(global_counts.tolist()):
        if global_count <= 0:
            continue
        local_indices = np.flatnonzero(labels == action_id)
        if local_indices.size == 0:
            continue

        if strategy == "oversample_to_max":
            desired = int(round(float(local_indices.size) * float(target_count) / float(global_count)))
            desired = max(1, desired)
            replace = desired > local_indices.size
        elif strategy == "downsample_to_min":
            desired = min(local_indices.size, target_count)
            replace = False
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        rng = np.random.default_rng(int(seed) + int(shard_index) * 1009 + int(action_id))
        chosen = rng.choice(local_indices, size=desired, replace=replace)
        selected_indices.append(chosen.astype(np.int64, copy=False))

    if selected_indices:
        selected = np.concatenate(selected_indices)
        rng = np.random.default_rng(int(seed) + int(shard_index) * 4099 + 17)
        rng.shuffle(selected)
        window_indices = manifest["window_indices"][selected].clone()
        valid_lengths = manifest["valid_lengths"][selected].clone()
    else:
        window_indices = manifest["window_indices"][:0].clone()
        valid_lengths = manifest["valid_lengths"][:0].clone()

    return {
        "version": int(manifest["version"]),
        "seq_len": int(manifest["seq_len"]),
        "stride": int(manifest["stride"]),
        "window_count": int(window_indices.shape[0]),
        "window_indices": window_indices,
        "valid_lengths": valid_lengths,
    }


def rebalance_bc_window_dataset(
    source_dir,
    output_dir,
    *,
    window_set_name="base_windows",
    strategy="oversample_to_max",
    seed=0,
    skip_existing=False,
):
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    shard_paths = _list_bc_shards(source_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No BC shard files found in {source_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_space_size = _CLASSIC_DISCRETE_ACTIONS
    before_counts = np.zeros(action_space_size, dtype=np.int64)
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu")
        manifest = _load_manifest(payload, window_set_name, shard_path)
        labels = _window_action_labels_from_manifest(payload, manifest)
        before_counts += np.bincount(labels, minlength=action_space_size)

    target_count = _target_count(before_counts, strategy)
    after_counts = np.zeros_like(before_counts)

    for shard_index, shard_path in enumerate(shard_paths):
        out_path = output_dir / Path(shard_path).name
        if skip_existing and out_path.exists():
            existing_payload = torch.load(out_path, map_location="cpu")
            existing_manifest = _load_manifest(existing_payload, "window_metadata", out_path)
            labels = _window_action_labels_from_manifest(existing_payload, existing_manifest)
            after_counts += np.bincount(labels, minlength=action_space_size)
            continue

        payload = torch.load(shard_path, map_location="cpu")
        manifest = _load_manifest(payload, window_set_name, shard_path)
        rebalanced = _resample_manifest(
            payload,
            manifest,
            before_counts,
            target_count=target_count,
            seed=seed,
            shard_index=shard_index,
            strategy=strategy,
        )

        updated_payload = dict(payload)
        metadata = dict(updated_payload.get("metadata", {}))
        window_sets = dict(updated_payload.get("window_sets", {})) if isinstance(updated_payload.get("window_sets"), dict) else None

        updated_payload["window_metadata"] = rebalanced
        if window_sets is not None:
            window_sets["base_windows"] = rebalanced
            updated_payload["window_sets"] = window_sets

        metadata["window_count"] = int(rebalanced["window_count"])
        metadata["window_seq_len"] = int(rebalanced["seq_len"])
        metadata["window_stride"] = int(rebalanced["stride"])
        metadata["rebalance_strategy"] = str(strategy)
        metadata["rebalance_window_set"] = str(window_set_name)
        metadata["rebalance_source_dir"] = str(source_dir)
        metadata["rebalance_seed"] = int(seed)
        metadata["rebalance_target_count"] = int(target_count)
        updated_payload["metadata"] = metadata

        torch.save(updated_payload, out_path)

        labels = _window_action_labels_from_manifest(updated_payload, rebalanced)
        after_counts += np.bincount(labels, minlength=action_space_size)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "window_set_name": str(window_set_name),
        "strategy": str(strategy),
        "seed": int(seed),
        "action_space_size": int(action_space_size),
        "target_count_per_action": int(target_count),
        "nonzero_actions_before": int(np.count_nonzero(before_counts)),
        "nonzero_actions_after": int(np.count_nonzero(after_counts)),
        "before_action_counts": before_counts.tolist(),
        "after_action_counts": after_counts.tolist(),
    }
    (output_dir / "rebalance_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Create a window-balanced BC dataset without modifying window contents")
    parser.add_argument("--source-dir", required=True, type=str, help="Path to source BC shard directory")
    parser.add_argument("--output-dir", required=True, type=str, help="Path to write balanced BC shards")
    parser.add_argument(
        "--window-set-name",
        default="base_windows",
        type=str,
        help="Embedded window manifest to rebalance (default: base_windows)",
    )
    parser.add_argument(
        "--strategy",
        choices=["oversample_to_max", "downsample_to_min"],
        default="oversample_to_max",
        help="How to balance the window label counts",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed for deterministic resampling")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume a partial output folder by keeping already-written balanced shards",
    )
    args = parser.parse_args()

    summary = rebalance_bc_window_dataset(
        args.source_dir,
        args.output_dir,
        window_set_name=args.window_set_name,
        strategy=args.strategy,
        seed=args.seed,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
