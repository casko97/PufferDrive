from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

from scripts.build_truck_context_preferences import (
    iter_preference_shards,
    load_preference_manifest,
    load_preferences_into_reward_model,
)

DEFAULT_INPUT = Path("outputs/preferences/nuplan_boston_test10_truck_context_preferences.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/reward_model")
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_SPLIT_SEED = 42
DEFAULT_EVAL_EXAMPLES = 5
DEFAULT_CHECKPOINT_STEM = "offline_truck_context"
DEFAULT_LOG_EVERY_SHARDS = 25
DEFAULT_WANDB_PROJECT = "pufferdrive-reward"
DEFAULT_WANDB_GROUP = "offline_reward"
DEFAULT_MAX_BUFFER_WINDOWS = 256
DEFAULT_INIT_CHECKPOINT_STEM = DEFAULT_CHECKPOINT_STEM

LOGGER = logging.getLogger("offline_truck_context_reward")


def _load_reward_model_class():
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from preferences.reward_model import RewardModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RewardModel dependencies are unavailable in this environment. "
            "Install the optional reward-model dependencies (for example scipy and loralib) to run offline training."
        ) from exc
    return RewardModel


class _WandbLogger:
    def __init__(self, project: str, group: str, name: str | None, config: dict, tag: str | None = None):
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "wandb logging was requested, but the wandb package is not installed in this environment."
            ) from exc

        wandb.init(
            project=project,
            group=group,
            allow_val_change=True,
            save_code=False,
            config=config,
            name=name,
            tags=[tag] if tag else [],
        )
        self.wandb = wandb

    def log(self, metrics: dict, step: int):
        self.wandb.log(metrics, step=step)

    def close(self):
        self.wandb.finish()


def _validate_preference_payload(payload: dict) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    meta = payload["metadata"]
    preferred = np.asarray(payload["preferred_sa"], dtype=np.float32)
    rejected = np.asarray(payload["rejected_sa"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.float32)
    window_metadata = list(payload.get("window_metadata", []))

    obs_dim = int(meta["obs_dim"])
    action_dim = int(meta["action_dim"])
    size_segment = int(meta["window_len"])
    total_windows = int(meta["total_windows"])

    if preferred.ndim != 3 or rejected.ndim != 3:
        raise ValueError("cached preferences must be rank-3")
    if preferred.shape != rejected.shape:
        raise ValueError(f"preferred/rejected shape mismatch: {preferred.shape} vs {rejected.shape}")
    if labels.shape != (preferred.shape[0], 1):
        raise ValueError(f"label shape mismatch: expected {(preferred.shape[0], 1)}, got {labels.shape}")

    action_type = meta.get("action_type")
    action_encoding = meta.get("action_encoding")
    if action_type == "discrete" and action_encoding != "one_hot":
        warnings.warn("scalar discrete action input is not supported; expected one-hot discrete actions", stacklevel=2)
        raise ValueError(f"unsupported discrete action encoding: {action_encoding!r}")
    if action_type is not None and action_type != "discrete":
        raise ValueError(f"unsupported action_type in preference payload: {action_type!r}")
    if preferred.shape[0] != total_windows:
        raise ValueError(f"total_windows mismatch: metadata says {total_windows}, tensor has {preferred.shape[0]}")
    if preferred.shape[1] != size_segment:
        raise ValueError(f"window_len mismatch: metadata says {size_segment}, tensor has {preferred.shape[1]}")
    if preferred.shape[2] != obs_dim + action_dim:
        raise ValueError(
            f"feature dimension mismatch: metadata says {obs_dim}+{action_dim}={obs_dim + action_dim}, "
            f"tensor has {preferred.shape[2]}"
        )
    if total_windows <= 0:
        raise ValueError("No preference windows available in payload")
    if window_metadata and len(window_metadata) != total_windows:
        raise ValueError(f"window_metadata length mismatch: expected {total_windows}, got {len(window_metadata)}")
    return meta, preferred, rejected, labels, window_metadata


def _manifest_metadata(preference_path: Path) -> dict:
    manifest = load_preference_manifest(preference_path)
    meta = dict(manifest["metadata"])
    total_windows = int(meta["total_windows"])
    if total_windows <= 0:
        raise ValueError("No preference windows available in manifest")
    return manifest


def _split_indices(total_windows: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if total_windows < 2:
        raise ValueError(f"need at least 2 windows for train/validation split, got {total_windows}")

    rng = np.random.default_rng(seed)
    indices = np.arange(total_windows, dtype=np.int64)
    rng.shuffle(indices)
    train_count = int(np.floor(total_windows * train_fraction))
    train_count = min(max(train_count, 1), total_windows - 1)
    train_indices = np.sort(indices[:train_count])
    val_indices = np.sort(indices[train_count:])
    if len(val_indices) == 0:
        raise ValueError("validation split is empty after applying train_fraction")
    return train_indices, val_indices


def _slice_payload(payload: dict, indices: np.ndarray) -> dict:
    local_payload = dict(payload)
    local_meta = dict(local_payload.get("metadata", {}))
    preferred_raw = np.asarray(local_payload["preferred_sa"], dtype=np.float32)
    local_meta["total_windows"] = int(len(preferred_raw))
    local_payload["metadata"] = local_meta
    meta, preferred, rejected, labels, window_metadata = _validate_preference_payload(local_payload)
    idx = np.asarray(indices, dtype=np.int64)
    sliced_meta = dict(meta)
    sliced_meta["total_windows"] = int(len(idx))
    return {
        "metadata": sliced_meta,
        "preferred_sa": preferred[idx],
        "rejected_sa": rejected[idx],
        "labels": labels[idx],
        "window_metadata": [window_metadata[i] for i in idx] if window_metadata else [],
    }


def _indices_for_shard(sorted_indices: np.ndarray, start_index: int, end_index: int) -> np.ndarray:
    left = np.searchsorted(sorted_indices, start_index, side="left")
    right = np.searchsorted(sorted_indices, end_index, side="left")
    if right <= left:
        return np.zeros((0,), dtype=np.int64)
    return sorted_indices[left:right] - start_index


def _iter_subset_payloads(preference_path: Path, sorted_indices: np.ndarray):
    for shard_payload, shard_info in iter_preference_shards(preference_path):
        start_index = int(shard_info["start_index"])
        end_index = int(shard_info["end_index"])
        local_indices = _indices_for_shard(sorted_indices, start_index=start_index, end_index=end_index)
        if len(local_indices) == 0:
            continue
        yield _slice_payload(shard_payload, local_indices), shard_info


def _concat_payloads(payloads: list[dict]) -> dict:
    if not payloads:
        raise ValueError("cannot concatenate empty payload list")
    base_meta = dict(payloads[0]["metadata"])
    preferred = np.concatenate([np.asarray(payload["preferred_sa"], dtype=np.float32) for payload in payloads], axis=0)
    rejected = np.concatenate([np.asarray(payload["rejected_sa"], dtype=np.float32) for payload in payloads], axis=0)
    labels = np.concatenate([np.asarray(payload["labels"], dtype=np.float32) for payload in payloads], axis=0)
    window_metadata: list[dict] = []
    for payload in payloads:
        window_metadata.extend(list(payload.get("window_metadata", [])))
    base_meta["total_windows"] = int(len(preferred))
    return {
        "metadata": base_meta,
        "preferred_sa": preferred,
        "rejected_sa": rejected,
        "labels": labels,
        "window_metadata": window_metadata,
    }


def _iter_subset_chunks(preference_path: Path, sorted_indices: np.ndarray, max_chunk_windows: int):
    chunk_payloads: list[dict] = []
    chunk_shards: list[dict] = []
    chunk_windows = 0
    effective_chunk_windows = max(1, int(max_chunk_windows))
    for subset_payload, shard_info in _iter_subset_payloads(preference_path, sorted_indices):
        subset_count = int(len(np.asarray(subset_payload["preferred_sa"])))
        if chunk_payloads and chunk_windows + subset_count > effective_chunk_windows:
            yield _concat_payloads(chunk_payloads), {
                "shard_infos": list(chunk_shards),
                "window_count": int(chunk_windows),
            }
            chunk_payloads = []
            chunk_shards = []
            chunk_windows = 0
        chunk_payloads.append(subset_payload)
        chunk_shards.append(shard_info)
        chunk_windows += subset_count
    if chunk_payloads:
        yield _concat_payloads(chunk_payloads), {
            "shard_infos": list(chunk_shards),
            "window_count": int(chunk_windows),
        }


def _max_subset_windows_per_shard(manifest: dict, sorted_indices: np.ndarray) -> int:
    max_count = 0
    for shard_info in manifest.get("shards", []):
        start_index = int(shard_info["start_index"])
        end_index = int(shard_info["end_index"])
        local_count = len(_indices_for_shard(sorted_indices, start_index=start_index, end_index=end_index))
        max_count = max(max_count, local_count)
    return max_count


def _instantiate_reward_model(
    RewardModel,
    obs_dim: int,
    action_dim: int,
    size_segment: int,
    total_windows: int,
    ensemble_size: int,
    lr: float,
    mb_size: int,
    train_batch_size: int,
    activation: str,
):
    model = RewardModel(
        ds=obs_dim,
        da=action_dim,
        ensemble_size=ensemble_size,
        lr=lr,
        mb_size=mb_size,
        size_segment=size_segment,
        capacity=max(total_windows * 2, total_windows + 1),
        activation=activation,
    )
    model.train_batch_size = train_batch_size
    return model


def evaluate_reward_model(model, preferred_sa: np.ndarray, rejected_sa: np.ndarray, labels: np.ndarray) -> dict:
    labels_long = labels.reshape(-1).astype(np.int64)
    ensemble_probs = []
    ensemble_losses = []
    ensemble_margins = []
    member_accuracies = []
    member_confidences = []
    for member in range(model.de):
        probs = model.p_hat_member(preferred_sa, rejected_sa, member=member).detach().cpu().numpy()
        ensemble_probs.append(probs.astype(np.float32))
        r_hat1 = model.r_hat_member(preferred_sa, member=member).sum(axis=1)
        r_hat2 = model.r_hat_member(rejected_sa, member=member).sum(axis=1)
        logits = torch.cat([r_hat1, r_hat2], axis=-1)
        labels_tensor = torch.from_numpy(labels_long).long().to(logits.device)
        ensemble_losses.append(float(model.CEloss(logits, labels_tensor).detach().cpu().item()))
        margin = (r_hat1 - r_hat2).detach().cpu().numpy().reshape(-1)
        signed_margin = np.where(labels_long == 0, margin, -margin)
        ensemble_margins.append(signed_margin.astype(np.float32))
        member_pred_first = (probs >= 0.5).astype(np.int64)
        member_predicted_labels = 1 - member_pred_first
        member_correct = member_predicted_labels == labels_long
        member_accuracies.append(float(np.mean(member_correct)) if len(member_correct) > 0 else float("nan"))
        member_confidences.append(float(np.mean(np.abs(probs - 0.5) * 2.0)) if len(probs) > 0 else float("nan"))
    probs = np.mean(np.stack(ensemble_probs, axis=0), axis=0)
    signed_margin = np.mean(np.stack(ensemble_margins, axis=0), axis=0)
    pred_first = (probs >= 0.5).astype(np.int64)
    predicted_labels = 1 - pred_first
    correct = predicted_labels == labels_long
    confidence = np.abs(probs - 0.5) * 2.0
    return {
        "accuracy": float(np.mean(correct)) if len(correct) > 0 else float("nan"),
        "loss": float(np.mean(ensemble_losses)) if ensemble_losses else float("nan"),
        "margin_mean": float(np.mean(signed_margin)) if len(signed_margin) > 0 else float("nan"),
        "confidence_mean": float(np.mean(confidence)) if len(confidence) > 0 else float("nan"),
        "member_losses": [float(x) for x in ensemble_losses],
        "member_accuracies": [float(x) for x in member_accuracies],
        "member_confidences": [float(x) for x in member_confidences],
        "prob_preferred_first": probs,
        "predicted_labels": predicted_labels,
        "correct": correct,
        "confidence": confidence,
    }


def evaluate_reward_model_on_indices(
    model,
    preference_path: Path,
    sorted_indices: np.ndarray,
    max_examples: int = 0,
    stage_name: str = "eval",
    log_every_shards: int = 0,
) -> dict:
    total = 0
    total_correct = 0
    weighted_loss_sum = 0.0
    weighted_margin_sum = 0.0
    weighted_confidence_sum = 0.0
    weighted_member_loss_sum = None
    weighted_member_acc_sum = None
    weighted_member_conf_sum = None
    collected_examples = []
    total_shards = 0
    if len(sorted_indices) > 0:
        manifest = _manifest_metadata(preference_path)
        total_shards = sum(
            1
            for shard_info in manifest.get("shards", [])
            if len(
                _indices_for_shard(
                    sorted_indices,
                    start_index=int(shard_info["start_index"]),
                    end_index=int(shard_info["end_index"]),
                )
            )
            > 0
        )
    processed_shards = 0

    for subset_payload, shard_info in _iter_subset_payloads(preference_path, sorted_indices):
        _, preferred, rejected, labels, window_metadata = _validate_preference_payload(subset_payload)
        evaluation = evaluate_reward_model(model, preferred, rejected, labels)
        correct = evaluation["correct"]
        predicted_labels = evaluation["predicted_labels"]
        probs = evaluation["prob_preferred_first"]
        confidence = evaluation["confidence"]
        total += len(correct)
        total_correct += int(np.sum(correct))
        weighted_loss_sum += float(evaluation["loss"]) * len(correct)
        weighted_margin_sum += float(evaluation["margin_mean"]) * len(correct)
        weighted_confidence_sum += float(evaluation["confidence_mean"]) * len(correct)
        member_losses = np.asarray(evaluation["member_losses"], dtype=np.float64)
        member_accs = np.asarray(evaluation["member_accuracies"], dtype=np.float64)
        member_confs = np.asarray(evaluation["member_confidences"], dtype=np.float64)
        if weighted_member_loss_sum is None:
            weighted_member_loss_sum = np.zeros_like(member_losses, dtype=np.float64)
            weighted_member_acc_sum = np.zeros_like(member_accs, dtype=np.float64)
            weighted_member_conf_sum = np.zeros_like(member_confs, dtype=np.float64)
        weighted_member_loss_sum += member_losses * len(correct)
        weighted_member_acc_sum += member_accs * len(correct)
        weighted_member_conf_sum += member_confs * len(correct)

        if max_examples > 0:
            shard_start = int(shard_info["start_index"])
            for i in range(len(correct)):
                collected_examples.append(
                    {
                        "index": int(shard_start + i),
                        "predicted_label": int(predicted_labels[i]),
                        "prob_preferred_first": float(probs[i]),
                        "confidence": float(confidence[i]),
                        "correct": bool(correct[i]),
                        "window": window_metadata[i] if window_metadata else {"index": int(shard_start + i)},
                    }
                )
        processed_shards += 1
        if log_every_shards > 0 and (
            processed_shards == 1
            or processed_shards == total_shards
            or processed_shards % log_every_shards == 0
        ):
            LOGGER.info(
                "Evaluation progress | stage=%s | shard %d/%d | windows %d/%d | running_loss=%.6f | running_acc=%.6f",
                stage_name,
                processed_shards,
                total_shards,
                total,
                len(sorted_indices),
                float(weighted_loss_sum / total) if total > 0 else float("nan"),
                float(total_correct / total) if total > 0 else float("nan"),
            )

    accuracy = float(total_correct / total) if total > 0 else float("nan")
    loss = float(weighted_loss_sum / total) if total > 0 else float("nan")
    margin_mean = float(weighted_margin_sum / total) if total > 0 else float("nan")
    confidence_mean = float(weighted_confidence_sum / total) if total > 0 else float("nan")
    member_losses = (weighted_member_loss_sum / total).tolist() if total > 0 and weighted_member_loss_sum is not None else []
    member_accuracies = (weighted_member_acc_sum / total).tolist() if total > 0 and weighted_member_acc_sum is not None else []
    member_confidences = (weighted_member_conf_sum / total).tolist() if total > 0 and weighted_member_conf_sum is not None else []
    if max_examples <= 0:
        return {
            "accuracy": accuracy,
            "loss": loss,
            "margin_mean": margin_mean,
            "confidence_mean": confidence_mean,
            "member_losses": member_losses,
            "member_accuracies": member_accuracies,
            "member_confidences": member_confidences,
            "count": total,
            "examples": {"correct_examples": [], "incorrect_examples": []},
        }

    correct_examples = [item for item in collected_examples if item["correct"]]
    incorrect_examples = [item for item in collected_examples if not item["correct"]]
    correct_examples.sort(key=lambda item: item["confidence"], reverse=True)
    incorrect_examples.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "accuracy": accuracy,
        "loss": loss,
        "margin_mean": margin_mean,
        "confidence_mean": confidence_mean,
        "member_losses": member_losses,
        "member_accuracies": member_accuracies,
        "member_confidences": member_confidences,
        "count": total,
        "examples": {
            "correct_examples": correct_examples[:max_examples],
            "incorrect_examples": incorrect_examples[:max_examples],
        },
    }


def _summarize_examples(
    evaluation: dict,
    window_metadata: list[dict],
    max_examples: int,
) -> dict:
    probs = evaluation["prob_preferred_first"]
    predicted_labels = evaluation["predicted_labels"]
    correct = evaluation["correct"]
    confidence = evaluation["confidence"]

    def _entry(i: int) -> dict:
        meta = window_metadata[i] if window_metadata else {"index": int(i)}
        return {
            "index": int(i),
            "predicted_label": int(predicted_labels[i]),
            "prob_preferred_first": float(probs[i]),
            "confidence": float(confidence[i]),
            "correct": bool(correct[i]),
            "window": meta,
        }

    correct_idx = np.flatnonzero(correct)
    incorrect_idx = np.flatnonzero(~correct)
    correct_rank = correct_idx[np.argsort(-confidence[correct_idx])] if len(correct_idx) > 0 else np.array([], dtype=np.int64)
    incorrect_rank = incorrect_idx[np.argsort(-confidence[incorrect_idx])] if len(incorrect_idx) > 0 else np.array([], dtype=np.int64)
    return {
        "correct_examples": [_entry(int(i)) for i in correct_rank[:max_examples]],
        "incorrect_examples": [_entry(int(i)) for i in incorrect_rank[:max_examples]],
    }


def load_trained_reward_model(
    output_dir: Path,
    obs_dim: int,
    action_dim: int,
    size_segment: int,
    ensemble_size: int,
    activation: str = "tanh",
):
    RewardModel = _load_reward_model_class()
    model = _instantiate_reward_model(
        RewardModel=RewardModel,
        obs_dim=obs_dim,
        action_dim=action_dim,
        size_segment=size_segment,
        total_windows=max(2, ensemble_size),
        ensemble_size=ensemble_size,
        lr=3e-4,
        mb_size=1,
        train_batch_size=1,
        activation=activation,
    )
    model.load(str(output_dir), DEFAULT_CHECKPOINT_STEM)
    return model


def train_offline_truck_context_reward(
    preference_path: Path,
    output_dir: Path,
    ensemble_size: int = 1,
    rounds: int = 1,
    lr: float = 3e-4,
    mb_size: int = 32,
    train_batch_size: int = 32,
    activation: str = "tanh",
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
    eval_examples: int = DEFAULT_EVAL_EXAMPLES,
    log_every_shards: int = DEFAULT_LOG_EVERY_SHARDS,
    max_buffer_windows: int = DEFAULT_MAX_BUFFER_WINDOWS,
    wandb_enabled: bool = False,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_group: str = DEFAULT_WANDB_GROUP,
    wandb_name: str | None = None,
    tag: str | None = None,
    init_from_dir: Path | None = None,
    init_checkpoint_stem: str = DEFAULT_INIT_CHECKPOINT_STEM,
) -> dict:
    manifest = _manifest_metadata(preference_path)
    meta = manifest["metadata"]
    obs_dim = int(meta["obs_dim"])
    action_dim = int(meta["action_dim"])
    size_segment = int(meta["window_len"])
    total_windows = int(meta["total_windows"])

    train_indices, val_indices = _split_indices(total_windows, train_fraction=train_fraction, seed=split_seed)
    total_train_shards = sum(1 for shard_info in manifest.get("shards", []) if len(
        _indices_for_shard(train_indices, start_index=int(shard_info["start_index"]), end_index=int(shard_info["end_index"]))
    ) > 0)
    total_val_shards = sum(1 for shard_info in manifest.get("shards", []) if len(
        _indices_for_shard(val_indices, start_index=int(shard_info["start_index"]), end_index=int(shard_info["end_index"]))
    ) > 0)

    LOGGER.info(
        "Starting reward training: input=%s total_windows=%d train=%d val=%d obs_mode=%s action_encoding=%s "
        "obs_dim=%d action_dim=%d window_len=%d ensemble=%d rounds=%d train_fraction=%.3f split_seed=%d",
        preference_path,
        total_windows,
        len(train_indices),
        len(val_indices),
        meta.get("observation_mode"),
        meta.get("action_encoding"),
        obs_dim,
        action_dim,
        size_segment,
        ensemble_size,
        rounds,
        train_fraction,
        split_seed,
    )
    LOGGER.info(
        "Streaming configuration: train_shards=%d val_shards=%d max_train_windows_per_shard=%d mb_size=%d "
        "train_batch_size=%d lr=%g activation=%s max_buffer_windows=%d",
        total_train_shards,
        total_val_shards,
        max_buffer_windows,
        mb_size,
        train_batch_size,
        lr,
        activation,
        max_buffer_windows,
    )

    wandb_logger = None
    if wandb_enabled:
        wandb_logger = _WandbLogger(
            project=wandb_project,
            group=wandb_group,
            name=wandb_name,
            tag=tag,
            config={
                "preference_path": str(preference_path),
                "output_dir": str(output_dir),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "window_len": size_segment,
                "observation_mode": meta.get("observation_mode"),
                "action_encoding": meta.get("action_encoding"),
                "total_windows": total_windows,
                "train_windows": int(len(train_indices)),
                "validation_windows": int(len(val_indices)),
                "ensemble_size": ensemble_size,
                "rounds": rounds,
                "lr": lr,
                "mb_size": mb_size,
                "train_batch_size": train_batch_size,
                "train_fraction": train_fraction,
                "split_seed": split_seed,
                "log_every_shards": log_every_shards,
                "max_buffer_windows": max_buffer_windows,
                "init_from_dir": str(init_from_dir) if init_from_dir is not None else None,
                "init_checkpoint_stem": init_checkpoint_stem,
            },
        )
        LOGGER.info(
            "wandb logging enabled: project=%s group=%s name=%s",
            wandb_project,
            wandb_group,
            wandb_name,
        )

    RewardModel = _load_reward_model_class()
    model = _instantiate_reward_model(
        RewardModel=RewardModel,
        obs_dim=obs_dim,
        action_dim=action_dim,
        size_segment=size_segment,
        total_windows=max(1, int(max_buffer_windows)),
        ensemble_size=ensemble_size,
        lr=lr,
        mb_size=mb_size,
        train_batch_size=train_batch_size,
        activation=activation,
    )
    if init_from_dir is not None:
        LOGGER.info("Loading initial reward model checkpoints from %s with stem %s", init_from_dir, init_checkpoint_stem)
        model.load(str(init_from_dir), init_checkpoint_stem)

    inserted = int(len(train_indices))
    round_acc = []
    round_loss = []
    round_member_acc = []
    round_member_loss = []
    for round_idx in range(rounds):
        LOGGER.info("Round %d/%d: training started", round_idx + 1, rounds)
        shard_accs = []
        shard_losses = []
        shard_counts = []
        processed_shards = 0
        processed_chunks = 0
        processed_windows = 0
        for train_payload, chunk_info in _iter_subset_chunks(preference_path, train_indices, max_chunk_windows=max_buffer_windows):
            model.buffer_index = 0
            model.buffer_full = False
            loaded = load_preferences_into_reward_model(model, train_payload)
            if loaded <= 0:
                continue
            metrics = model.train_reward(return_metrics=True)
            acc = np.asarray(metrics["acc"], dtype=np.float32)
            loss = np.asarray(metrics["loss"], dtype=np.float32)
            shard_accs.append(acc)
            shard_losses.append(loss)
            shard_counts.append(loaded)
            processed_chunks += 1
            chunk_shards = list(chunk_info["shard_infos"])
            processed_shards += len(chunk_shards)
            processed_windows += loaded
            if (
                processed_chunks == 1
                or processed_shards == total_train_shards
                or processed_shards % max(log_every_shards, 1) == 0
            ):
                current_acc = np.average(np.stack(shard_accs, axis=0), axis=0, weights=np.asarray(shard_counts, dtype=np.float32))
                current_loss = np.average(np.stack(shard_losses, axis=0), axis=0, weights=np.asarray(shard_counts, dtype=np.float32))
                windows_fraction = processed_windows / max(len(train_indices), 1)
                shard_fraction = processed_shards / max(total_train_shards, 1)
                overall_fraction = ((round_idx + windows_fraction) / max(rounds, 1)) * 100.0
                LOGGER.info(
                    "Progress %.2f%% | Round %d/%d | shard %d/%d (%.2f%%) | windows %d/%d (%.2f%%) | "
                    "chunk %d | loaded=%d | current_train_loss=%s | current_train_acc=%s | "
                    "shards=%s..%s (%d shards)",
                    overall_fraction,
                    round_idx + 1,
                    rounds,
                    processed_shards,
                    total_train_shards,
                    shard_fraction * 100.0,
                    processed_windows,
                    len(train_indices),
                    windows_fraction * 100.0,
                    processed_chunks,
                    loaded,
                    np.array2string(current_loss, precision=4),
                    np.array2string(current_acc, precision=4),
                    Path(chunk_shards[0]["path"]).name,
                    Path(chunk_shards[-1]["path"]).name,
                    len(chunk_shards),
                )
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "progress/overall_percent": overall_fraction,
                            "progress/round": round_idx + 1,
                            "progress/shard_index": processed_shards,
                            "progress/shard_fraction": shard_fraction,
                            "progress/chunk_index": processed_chunks,
                            "progress/train_window_fraction": windows_fraction,
                            "preference_train/stream_ce_loss": float(np.mean(current_loss)),
                            "preference_train/stream_pair_accuracy": float(np.mean(current_acc)),
                            "preference_train/stream_windows_processed": processed_windows,
                        },
                        step=(round_idx * total_train_shards) + processed_shards,
                    )
        if not shard_accs:
            raise ValueError("No training windows were loaded for any shard")
        weights = np.asarray(shard_counts, dtype=np.float32)
        weighted_acc = np.average(np.stack(shard_accs, axis=0), axis=0, weights=weights)
        weighted_loss = np.average(np.stack(shard_losses, axis=0), axis=0, weights=weights)
        round_acc.append(weighted_acc.astype(np.float32))
        round_loss.append(weighted_loss.astype(np.float32))
        round_member_acc.append(weighted_acc.astype(np.float32))
        round_member_loss.append(weighted_loss.astype(np.float32))
        train_eval_round = evaluate_reward_model_on_indices(
            model,
            preference_path,
            train_indices,
            max_examples=0,
            stage_name=f"train_round_{round_idx + 1}",
            log_every_shards=log_every_shards,
        )
        val_eval_round = evaluate_reward_model_on_indices(
            model,
            preference_path,
            val_indices,
            max_examples=0,
            stage_name=f"val_round_{round_idx + 1}",
            log_every_shards=log_every_shards,
        )
        LOGGER.info(
            "Round %d/%d complete | processed_chunks=%d processed_shards=%d processed_windows=%d | "
            "batch_train_loss=%s batch_train_acc=%s | full_train_loss=%.6f full_train_acc=%.6f | "
            "heldout_val_loss=%.6f heldout_val_acc=%.6f",
            round_idx + 1,
            rounds,
            processed_chunks,
            processed_shards,
            processed_windows,
            np.array2string(weighted_loss, precision=4),
            np.array2string(weighted_acc, precision=4),
            train_eval_round["loss"],
            train_eval_round["accuracy"],
            val_eval_round["loss"],
            val_eval_round["accuracy"],
        )
        if wandb_logger is not None:
            wandb_metrics = {
                "round/index": round_idx + 1,
                "preference_train/batch_ce_loss": float(np.mean(weighted_loss)),
                "preference_train/batch_pair_accuracy": float(np.mean(weighted_acc)),
                "preference_train/full_ce_loss": float(train_eval_round["loss"]),
                "preference_train/full_pair_accuracy": float(train_eval_round["accuracy"]),
                "preference_train/full_margin_mean": float(train_eval_round["margin_mean"]),
                "preference_train/full_confidence_mean": float(train_eval_round["confidence_mean"]),
                "preference_val/ce_loss": float(val_eval_round["loss"]),
                "preference_val/pair_accuracy": float(val_eval_round["accuracy"]),
                "preference_val/margin_mean": float(val_eval_round["margin_mean"]),
                "preference_val/confidence_mean": float(val_eval_round["confidence_mean"]),
            }
            for member_idx, member_loss in enumerate(weighted_loss):
                wandb_metrics[f"preference_train_member/{member_idx}/batch_ce_loss"] = float(member_loss)
            for member_idx, member_acc in enumerate(weighted_acc):
                wandb_metrics[f"preference_train_member/{member_idx}/batch_pair_accuracy"] = float(member_acc)
            for member_idx, member_loss in enumerate(train_eval_round["member_losses"]):
                wandb_metrics[f"preference_train_member/{member_idx}/full_ce_loss"] = float(member_loss)
            for member_idx, member_acc in enumerate(train_eval_round["member_accuracies"]):
                wandb_metrics[f"preference_train_member/{member_idx}/full_pair_accuracy"] = float(member_acc)
            for member_idx, member_conf in enumerate(train_eval_round["member_confidences"]):
                wandb_metrics[f"preference_train_member/{member_idx}/full_confidence_mean"] = float(member_conf)
            for member_idx, member_loss in enumerate(val_eval_round["member_losses"]):
                wandb_metrics[f"preference_val_member/{member_idx}/ce_loss"] = float(member_loss)
            for member_idx, member_acc in enumerate(val_eval_round["member_accuracies"]):
                wandb_metrics[f"preference_val_member/{member_idx}/pair_accuracy"] = float(member_acc)
            for member_idx, member_conf in enumerate(val_eval_round["member_confidences"]):
                wandb_metrics[f"preference_val_member/{member_idx}/confidence_mean"] = float(member_conf)
            wandb_logger.log(wandb_metrics, step=(round_idx + 1) * total_train_shards)

    LOGGER.info("Running final train-set evaluation on %d windows across %d shards", len(train_indices), total_train_shards)
    train_eval = evaluate_reward_model_on_indices(
        model,
        preference_path,
        train_indices,
        max_examples=0,
        stage_name="train_final",
        log_every_shards=log_every_shards,
    )
    LOGGER.info("Train-set evaluation complete: loss=%.6f accuracy=%.6f count=%d", train_eval["loss"], train_eval["accuracy"], train_eval["count"])
    LOGGER.info("Running validation evaluation on %d windows across %d shards", len(val_indices), total_val_shards)
    val_eval = evaluate_reward_model_on_indices(
        model,
        preference_path,
        val_indices,
        max_examples=eval_examples,
        stage_name="val_final",
        log_every_shards=log_every_shards,
    )
    LOGGER.info("Validation evaluation complete: loss=%.6f accuracy=%.6f count=%d", val_eval["loss"], val_eval["accuracy"], val_eval["count"])

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir), DEFAULT_CHECKPOINT_STEM)
    LOGGER.info("Saved reward model checkpoints to %s", output_dir)

    evaluation_report = {
        "preference_path": str(preference_path),
        "output_dir": str(output_dir),
        "train_indices": train_indices.tolist(),
        "validation_indices": val_indices.tolist(),
        "validation_accuracy": float(val_eval["accuracy"]),
        "validation_loss": float(val_eval["loss"]),
        "validation_margin_mean": float(val_eval["margin_mean"]),
        "validation_confidence_mean": float(val_eval["confidence_mean"]),
        "validation_count": int(val_eval["count"]),
        "examples": val_eval["examples"],
    }
    evaluation_path = output_dir / "offline_truck_context_reward_eval.json"
    evaluation_path.write_text(json.dumps(evaluation_report, indent=2))
    LOGGER.info("Saved evaluation report to %s", evaluation_path)

    summary = {
        "preference_path": str(preference_path),
        "output_dir": str(output_dir),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "size_segment": size_segment,
        "observation_mode": meta.get("observation_mode"),
        "action_encoding": meta.get("action_encoding"),
        "total_windows": total_windows,
        "train_windows": int(len(train_indices)),
        "validation_windows": int(len(val_indices)),
        "inserted_windows": inserted,
        "ensemble_size": ensemble_size,
        "rounds": rounds,
        "train_fraction": float(train_fraction),
        "split_seed": int(split_seed),
        "activation": activation,
        "init_from_dir": str(init_from_dir) if init_from_dir is not None else None,
        "init_checkpoint_stem": init_checkpoint_stem if init_from_dir is not None else None,
        "round_acc": [acc.tolist() for acc in round_acc],
        "round_loss": [loss.tolist() for loss in round_loss],
        "round_member_acc": [acc.tolist() for acc in round_member_acc],
        "round_member_loss": [loss.tolist() for loss in round_member_loss],
        "final_train_member_losses": [float(x) for x in train_eval["member_losses"]],
        "final_train_member_accuracies": [float(x) for x in train_eval["member_accuracies"]],
        "final_train_member_confidences": [float(x) for x in train_eval["member_confidences"]],
        "final_train_margin_mean": float(train_eval["margin_mean"]),
        "final_train_confidence_mean": float(train_eval["confidence_mean"]),
        "final_train_loss": float(train_eval["loss"]),
        "final_train_acc": float(train_eval["accuracy"]),
        "final_validation_member_losses": [float(x) for x in val_eval["member_losses"]],
        "final_validation_member_accuracies": [float(x) for x in val_eval["member_accuracies"]],
        "final_validation_member_confidences": [float(x) for x in val_eval["member_confidences"]],
        "final_validation_margin_mean": float(val_eval["margin_mean"]),
        "final_validation_confidence_mean": float(val_eval["confidence_mean"]),
        "final_validation_loss": float(val_eval["loss"]),
        "final_validation_acc": float(val_eval["accuracy"]),
        "evaluation_report": str(evaluation_path),
    }
    summary_path = output_dir / "offline_truck_context_reward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOGGER.info(
        "Saved training summary to %s; final_train_acc=%.6f final_validation_acc=%.6f",
        summary_path,
        summary["final_train_acc"],
        summary["final_validation_acc"],
    )
    if wandb_logger is not None:
        wandb_logger.log(
            {
                "preference_final/train_ce_loss": float(summary["final_train_loss"]),
                "preference_final/train_pair_accuracy": float(summary["final_train_acc"]),
                "preference_final/train_margin_mean": float(summary["final_train_margin_mean"]),
                "preference_final/train_confidence_mean": float(summary["final_train_confidence_mean"]),
                "preference_final/val_ce_loss": float(summary["final_validation_loss"]),
                "preference_final/val_pair_accuracy": float(summary["final_validation_acc"]),
                "preference_final/val_margin_mean": float(summary["final_validation_margin_mean"]),
                "preference_final/val_confidence_mean": float(summary["final_validation_confidence_mean"]),
                **{
                    f"preference_final_train_member/{idx}/ce_loss": float(value)
                    for idx, value in enumerate(summary["final_train_member_losses"])
                },
                **{
                    f"preference_final_train_member/{idx}/pair_accuracy": float(value)
                    for idx, value in enumerate(summary["final_train_member_accuracies"])
                },
                **{
                    f"preference_final_val_member/{idx}/ce_loss": float(value)
                    for idx, value in enumerate(summary["final_validation_member_losses"])
                },
                **{
                    f"preference_final_val_member/{idx}/pair_accuracy": float(value)
                    for idx, value in enumerate(summary["final_validation_member_accuracies"])
                },
            },
            step=rounds * total_train_shards + 1,
        )
        wandb_logger.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train RewardModel from cached truck-context preferences.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mb-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--eval-examples", type=int, default=DEFAULT_EVAL_EXAMPLES)
    parser.add_argument("--log-every-shards", type=int, default=DEFAULT_LOG_EVERY_SHARDS)
    parser.add_argument("--max-buffer-windows", type=int, default=DEFAULT_MAX_BUFFER_WINDOWS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging for this run")
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-group", type=str, default=DEFAULT_WANDB_GROUP)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--init-from-dir", type=Path, default=None)
    parser.add_argument("--init-checkpoint-stem", type=str, default=DEFAULT_INIT_CHECKPOINT_STEM)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    summary = train_offline_truck_context_reward(
        preference_path=args.input,
        output_dir=args.output_dir,
        ensemble_size=args.ensemble_size,
        rounds=args.rounds,
        lr=args.lr,
        mb_size=args.mb_size,
        train_batch_size=args.train_batch_size,
        activation=args.activation,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        eval_examples=args.eval_examples,
        log_every_shards=args.log_every_shards,
        max_buffer_windows=args.max_buffer_windows,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        tag=args.tag,
        init_from_dir=args.init_from_dir,
        init_checkpoint_stem=args.init_checkpoint_stem,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
