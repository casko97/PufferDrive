from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch


@dataclass(frozen=True)
class PreferenceRewardMetadata:
    observation_mode: str | None
    obs_dim: int
    action_dim: int
    size_segment: int | None
    action_encoding: str | None
    ensemble_size: int
    activation: str


def _device_string(device: Any) -> str:
    if isinstance(device, torch.device):
        return str(device)
    if isinstance(device, int):
        return f"cuda:{device}"
    return str(device)


def _as_path(path_value: Any) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if not path.is_absolute() else path


def get_discrete_action_dim(action_space: gymnasium.Space) -> int:
    if isinstance(action_space, gymnasium.spaces.Discrete):
        return int(action_space.n)
    if isinstance(action_space, gymnasium.spaces.MultiDiscrete):
        if len(action_space.nvec) != 1:
            raise ValueError(
                "Preference reward currently supports only single-branch discrete actions; "
                f"got MultiDiscrete with shape {tuple(action_space.nvec.tolist())}"
            )
        return int(action_space.nvec[0])
    raise ValueError(
        "Preference reward currently supports only discrete action spaces compatible with one-hot encoding; "
        f"got {type(action_space).__name__}"
    )


def one_hot_encode_discrete_actions(actions: np.ndarray, action_dim: int) -> np.ndarray:
    discrete = np.asarray(actions, dtype=np.int64).reshape(-1)
    if np.any(discrete < 0) or np.any(discrete >= action_dim):
        raise ValueError(
            f"Preference reward action index out of range: valid [0, {action_dim - 1}], got {discrete.tolist()}"
        )
    encoded = np.zeros((len(discrete), action_dim), dtype=np.float32)
    encoded[np.arange(len(discrete)), discrete] = 1.0
    return encoded


def build_state_action_features(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    metadata: PreferenceRewardMetadata,
    action_space: gymnasium.Space,
    action_type: str,
) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError(f"Preference reward observations must be rank-2, got shape {obs.shape}")
    if obs.shape[1] != metadata.obs_dim:
        raise ValueError(
            f"Preference reward observation dimension mismatch: expected {metadata.obs_dim}, got {obs.shape[1]}"
        )

    if action_type != "discrete":
        raise ValueError(f"Preference reward only supports action_type='discrete', got {action_type!r}")
    if metadata.action_encoding != "one_hot":
        raise ValueError(
            "Preference reward only supports action_encoding='one_hot' at runtime, "
            f"got {metadata.action_encoding!r}"
        )

    runtime_action_dim = get_discrete_action_dim(action_space)
    if runtime_action_dim != metadata.action_dim:
        raise ValueError(
            "Preference reward action dimension mismatch: "
            f"model={metadata.action_dim} env={runtime_action_dim}"
        )

    encoded_actions = one_hot_encode_discrete_actions(actions, action_dim=metadata.action_dim)
    return np.concatenate([obs, encoded_actions], axis=-1).astype(np.float32, copy=False)


def combine_preference_rewards(
    task_reward: np.ndarray,
    pref_mean: np.ndarray,
    pref_std: np.ndarray,
    *,
    beta: float,
    lambda_uncertainty: float,
    scale: float,
    clip_min: float | None,
    clip_max: float | None,
    normalize_mean: float,
    normalize_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task = np.asarray(task_reward, dtype=np.float32)
    mean = np.asarray(pref_mean, dtype=np.float32)
    std = np.asarray(pref_std, dtype=np.float32)

    pref_raw = mean - float(lambda_uncertainty) * std
    pref_norm = (pref_raw - float(normalize_mean)) / max(float(normalize_std), 1e-6)
    pref_shaped = float(scale) * pref_norm

    lower = -np.inf if clip_min is None else float(clip_min)
    upper = np.inf if clip_max is None else float(clip_max)
    pref_shaped = np.clip(pref_shaped, lower, upper)
    total_reward = task + float(beta) * pref_shaped
    return total_reward.astype(np.float32), pref_raw.astype(np.float32), pref_shaped.astype(np.float32)


class PreferenceRewardManager:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        metadata: PreferenceRewardMetadata,
        reward_model,
        action_space: gymnasium.Space,
        action_type: str,
    ):
        self.config = dict(config)
        self.metadata = metadata
        self.reward_model = reward_model
        self.action_space = action_space
        self.action_type = action_type

        self.beta = float(self.config.get("beta", 0.1))
        self.lambda_uncertainty = float(self.config.get("lambda_uncertainty", 0.0))
        self.normalize_mode = str(self.config.get("normalize_mode", "zscore_baseline"))
        self.scale = float(self.config.get("scale", 1.0))
        self.clip_min = self.config.get("clip_min")
        self.clip_max = self.config.get("clip_max")
        self.warmup_steps = int(self.config.get("warmup_steps", 0))
        self.calibration_steps = max(0, int(self.config.get("calibration_steps", 0)))
        self.log_member_stats = bool(self.config.get("log_member_stats", False))

        self._calibration_count = 0
        self._calibration_sum = 0.0
        self._calibration_sumsq = 0.0
        self._normalization_ready = self.normalize_mode == "none" or self.calibration_steps == 0
        self._normalization_mean = 0.0
        self._normalization_std = 1.0

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        env_config: dict[str, Any],
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: Any,
    ) -> PreferenceRewardManager | None:
        cfg = dict(config or {})
        if not bool(cfg.get("enabled", False)):
            return None

        model_dir_value = cfg.get("model_dir")
        if not model_dir_value:
            raise ValueError("Preference reward is enabled, but preference_reward.model_dir is not configured")

        model_dir = _as_path(model_dir_value)
        summary_path = model_dir / "offline_truck_context_reward_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Preference reward summary not found: {summary_path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = PreferenceRewardMetadata(
            observation_mode=summary.get("observation_mode"),
            obs_dim=int(summary["obs_dim"]),
            action_dim=int(summary["action_dim"]),
            size_segment=(int(summary["size_segment"]) if summary.get("size_segment") is not None else None),
            action_encoding=summary.get("action_encoding"),
            ensemble_size=int(summary["ensemble_size"]),
            activation=str(summary.get("activation", "tanh")),
        )

        validate_preference_reward_compatibility(
            metadata=metadata,
            env_config=env_config,
            observation_space=observation_space,
            action_space=action_space,
            strict_observation_match=bool(cfg.get("strict_observation_match", True)),
        )

        from preferences import reward_model as reward_model_module

        reward_model_module.device = _device_string(device)
        reward_model = reward_model_module.RewardModel(
            ds=metadata.obs_dim,
            da=metadata.action_dim,
            ensemble_size=metadata.ensemble_size,
            lr=3e-4,
            mb_size=1,
            size_segment=1,
            capacity=max(2, metadata.ensemble_size),
            activation=metadata.activation,
        )
        reward_model.train_batch_size = 1

        checkpoint_stem = str(cfg.get("checkpoint_stem", "offline_truck_context"))
        missing = [model_dir / f"reward_model_{checkpoint_stem}_{member}.pt" for member in range(metadata.ensemble_size)]
        missing = [path for path in missing if not path.exists()]
        if missing:
            missing_str = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(f"Preference reward checkpoint(s) not found: {missing_str}")
        reward_model.load(str(model_dir), checkpoint_stem)

        return cls(
            config=cfg,
            metadata=metadata,
            reward_model=reward_model,
            action_space=action_space,
            action_type=str(env_config.get("action_type", "")),
        )

    def score(self, observations: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = build_state_action_features(
            observations,
            actions,
            metadata=self.metadata,
            action_space=self.action_space,
            action_type=self.action_type,
        )
        segment_features = np.expand_dims(features, axis=1)
        member_rewards = []
        for member in range(self.metadata.ensemble_size):
            reward = self.reward_model.r_hat_member(segment_features, member=member).detach().cpu().numpy()
            member_rewards.append(reward.reshape(-1).astype(np.float32))
        rewards = np.stack(member_rewards, axis=0)
        return rewards.mean(axis=0), rewards.std(axis=0), rewards

    def _update_calibration(self, pref_raw: np.ndarray) -> None:
        values = np.asarray(pref_raw, dtype=np.float64).reshape(-1)
        if values.size == 0 or self._normalization_ready or self.normalize_mode != "zscore_baseline":
            return

        remaining = self.calibration_steps - self._calibration_count
        if remaining <= 0:
            self._finalize_calibration()
            return

        used_values = values[:remaining]
        self._calibration_count += int(used_values.size)
        self._calibration_sum += float(np.sum(used_values))
        self._calibration_sumsq += float(np.sum(np.square(used_values)))
        if self._calibration_count >= self.calibration_steps:
            self._finalize_calibration()

    def _finalize_calibration(self) -> None:
        if self._calibration_count <= 0:
            self._normalization_mean = 0.0
            self._normalization_std = 1.0
        else:
            mean = self._calibration_sum / self._calibration_count
            variance = max((self._calibration_sumsq / self._calibration_count) - mean**2, 0.0)
            std = float(np.sqrt(variance))
            self._normalization_mean = float(mean)
            self._normalization_std = max(std, 1e-6)
        self._normalization_ready = True

    def shape_rewards(
        self,
        task_reward: np.ndarray,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        global_step: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        task = np.asarray(task_reward, dtype=np.float32).reshape(-1)
        pref_mean, pref_std, member_rewards = self.score(observations, actions)
        pref_raw = pref_mean - self.lambda_uncertainty * pref_std
        normalization_ready_before_update = self._normalization_ready
        self._update_calibration(pref_raw)

        apply_shaping = global_step >= self.warmup_steps and (
            self.normalize_mode == "none" or normalization_ready_before_update
        )

        if apply_shaping:
            total_reward, _, pref_shaped = combine_preference_rewards(
                task,
                pref_mean,
                pref_std,
                beta=self.beta,
                lambda_uncertainty=self.lambda_uncertainty,
                scale=self.scale,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                normalize_mean=self._normalization_mean,
                normalize_std=self._normalization_std,
            )
        else:
            total_reward = task.astype(np.float32)
            pref_shaped = np.zeros_like(task, dtype=np.float32)

        metrics = {
            "task_reward": float(np.mean(task)),
            "preference_reward_mean": float(np.mean(pref_mean)),
            "preference_reward_std": float(np.mean(pref_std)),
            "preference_reward_raw": float(np.mean(pref_raw)),
            "preference_reward_shaped": float(np.mean(pref_shaped)),
            "combined_reward": float(np.mean(total_reward)),
            "preference_reward_applied": float(apply_shaping),
            "preference_reward_calibration_progress": (
                float(self._calibration_count / self.calibration_steps) if self.calibration_steps > 0 else 1.0
            ),
        }
        if self.log_member_stats:
            for member_idx, rewards in enumerate(member_rewards):
                metrics[f"preference_reward_member_{member_idx}"] = float(np.mean(rewards))
        return total_reward, metrics


def validate_preference_reward_compatibility(
    *,
    metadata: PreferenceRewardMetadata,
    env_config: dict[str, Any],
    observation_space: gymnasium.Space,
    action_space: gymnasium.Space,
    strict_observation_match: bool = True,
) -> None:
    env_observation_mode = str(env_config.get("observation_mode", "default"))
    if strict_observation_match and metadata.observation_mode not in (None, env_observation_mode):
        raise ValueError(
            "Preference reward observation_mode mismatch: "
            f"model={metadata.observation_mode} env={env_observation_mode}"
        )

    if len(observation_space.shape) != 1:
        raise ValueError(f"Preference reward requires a flat observation space, got shape {observation_space.shape}")
    env_obs_dim = int(observation_space.shape[0])
    if metadata.obs_dim != env_obs_dim:
        raise ValueError(f"Preference reward obs_dim mismatch: model={metadata.obs_dim} env={env_obs_dim}")

    env_action_type = str(env_config.get("action_type", ""))
    if env_action_type != "discrete":
        raise ValueError(f"Preference reward only supports env.action_type='discrete', got {env_action_type!r}")
    if metadata.action_encoding != "one_hot":
        raise ValueError(
            "Preference reward action_encoding mismatch: "
            f"model={metadata.action_encoding} expected=one_hot"
        )

    runtime_action_dim = get_discrete_action_dim(action_space)
    if metadata.action_dim != runtime_action_dim:
        raise ValueError(
            "Preference reward action_dim mismatch: "
            f"model={metadata.action_dim} env={runtime_action_dim}"
        )
