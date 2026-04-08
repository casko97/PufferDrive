import json
from pathlib import Path

import gymnasium
import numpy as np
import pytest

from preferences.reward_model import RewardModel
from pufferlib.preference_reward import (
    PreferenceRewardManager,
    PreferenceRewardMetadata,
    build_state_action_features,
    combine_preference_rewards,
    validate_preference_reward_compatibility,
)


def _write_reward_model_dir(tmp_path: Path, *, obs_dim: int = 4, action_dim: int = 3, ensemble_size: int = 2) -> Path:
    model_dir = tmp_path / "reward_model"
    model_dir.mkdir()

    model = RewardModel(
        ds=obs_dim,
        da=action_dim,
        ensemble_size=ensemble_size,
        size_segment=1,
        capacity=8,
        activation="tanh",
        mb_size=1,
    )
    model.save(str(model_dir), "offline_truck_context")

    summary = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "size_segment": 32,
        "observation_mode": "default",
        "action_encoding": "one_hot",
        "ensemble_size": ensemble_size,
        "activation": "tanh",
    }
    (model_dir / "offline_truck_context_reward_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return model_dir


def test_preference_reward_loader_succeeds_for_matching_env(tmp_path):
    model_dir = _write_reward_model_dir(tmp_path)
    manager = PreferenceRewardManager.from_config(
        {
            "enabled": True,
            "model_dir": str(model_dir),
            "checkpoint_stem": "offline_truck_context",
            "normalize_mode": "none",
        },
        env_config={"observation_mode": "default", "action_type": "discrete"},
        observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        action_space=gymnasium.spaces.MultiDiscrete([3]),
        device="cpu",
    )

    assert manager is not None
    assert manager.metadata.observation_mode == "default"
    assert manager.metadata.action_dim == 3


def test_preference_reward_loader_fails_when_summary_missing(tmp_path):
    model_dir = tmp_path / "missing_summary"
    model_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="summary not found"):
        PreferenceRewardManager.from_config(
            {"enabled": True, "model_dir": str(model_dir), "checkpoint_stem": "offline_truck_context"},
            env_config={"observation_mode": "default", "action_type": "discrete"},
            observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            action_space=gymnasium.spaces.MultiDiscrete([3]),
            device="cpu",
        )


def test_preference_reward_validation_rejects_observation_mode_mismatch():
    metadata = PreferenceRewardMetadata(
        observation_mode="default",
        obs_dim=4,
        action_dim=3,
        size_segment=32,
        action_encoding="one_hot",
        ensemble_size=2,
        activation="tanh",
    )

    with pytest.raises(ValueError, match="observation_mode mismatch: model=default env=sdc_only_with_trailer"):
        validate_preference_reward_compatibility(
            metadata=metadata,
            env_config={"observation_mode": "sdc_only_with_trailer", "action_type": "discrete"},
            observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            action_space=gymnasium.spaces.MultiDiscrete([3]),
            strict_observation_match=True,
        )


def test_preference_reward_validation_rejects_obs_dim_mismatch():
    metadata = PreferenceRewardMetadata(
        observation_mode="default",
        obs_dim=5,
        action_dim=3,
        size_segment=32,
        action_encoding="one_hot",
        ensemble_size=2,
        activation="tanh",
    )

    with pytest.raises(ValueError, match="obs_dim mismatch: model=5 env=4"):
        validate_preference_reward_compatibility(
            metadata=metadata,
            env_config={"observation_mode": "default", "action_type": "discrete"},
            observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            action_space=gymnasium.spaces.MultiDiscrete([3]),
            strict_observation_match=True,
        )


def test_preference_reward_validation_rejects_action_dim_mismatch():
    metadata = PreferenceRewardMetadata(
        observation_mode="default",
        obs_dim=4,
        action_dim=4,
        size_segment=32,
        action_encoding="one_hot",
        ensemble_size=2,
        activation="tanh",
    )

    with pytest.raises(ValueError, match="action_dim mismatch: model=4 env=3"):
        validate_preference_reward_compatibility(
            metadata=metadata,
            env_config={"observation_mode": "default", "action_type": "discrete"},
            observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            action_space=gymnasium.spaces.MultiDiscrete([3]),
            strict_observation_match=True,
        )


def test_build_state_action_features_one_hot_encodes_actions():
    metadata = PreferenceRewardMetadata(
        observation_mode="default",
        obs_dim=4,
        action_dim=3,
        size_segment=32,
        action_encoding="one_hot",
        ensemble_size=2,
        activation="tanh",
    )
    observations = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    actions = np.asarray([[2], [0]], dtype=np.int64)

    features = build_state_action_features(
        observations,
        actions,
        metadata=metadata,
        action_space=gymnasium.spaces.MultiDiscrete([3]),
        action_type="discrete",
    )

    expected = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0],
            [4.0, 3.0, 2.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(features, expected)


def test_combine_preference_rewards_matches_formula():
    total_reward, pref_raw, pref_shaped = combine_preference_rewards(
        np.asarray([0.5, 0.2], dtype=np.float32),
        np.asarray([0.4, -0.1], dtype=np.float32),
        np.asarray([0.1, 0.2], dtype=np.float32),
        beta=0.5,
        lambda_uncertainty=2.0,
        scale=1.0,
        clip_min=None,
        clip_max=None,
        normalize_mean=0.0,
        normalize_std=1.0,
    )

    np.testing.assert_allclose(pref_raw, np.asarray([0.2, -0.5], dtype=np.float32))
    np.testing.assert_allclose(pref_shaped, np.asarray([0.2, -0.5], dtype=np.float32))
    np.testing.assert_allclose(total_reward, np.asarray([0.6, -0.05], dtype=np.float32))


def test_preference_reward_warmup_keeps_task_reward_unchanged(tmp_path):
    model_dir = _write_reward_model_dir(tmp_path)
    manager = PreferenceRewardManager.from_config(
        {
            "enabled": True,
            "model_dir": str(model_dir),
            "checkpoint_stem": "offline_truck_context",
            "normalize_mode": "none",
            "warmup_steps": 100,
            "beta": 1.0,
        },
        env_config={"observation_mode": "default", "action_type": "discrete"},
        observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        action_space=gymnasium.spaces.MultiDiscrete([3]),
        device="cpu",
    )

    def fake_score(_obs, _actions):
        mean = np.asarray([0.3, 0.6], dtype=np.float32)
        std = np.asarray([0.1, 0.2], dtype=np.float32)
        members = np.stack([mean, mean], axis=0)
        return mean, std, members

    manager.score = fake_score
    task_reward = np.asarray([0.4, 0.2], dtype=np.float32)
    total_reward, metrics = manager.shape_rewards(
        task_reward,
        np.zeros((2, 4), dtype=np.float32),
        np.zeros((2, 1), dtype=np.int64),
        global_step=0,
    )

    np.testing.assert_allclose(total_reward, task_reward)
    assert metrics["preference_reward_applied"] == 0.0


def test_preference_reward_zscore_calibration_waits_until_ready(tmp_path):
    model_dir = _write_reward_model_dir(tmp_path)
    manager = PreferenceRewardManager.from_config(
        {
            "enabled": True,
            "model_dir": str(model_dir),
            "checkpoint_stem": "offline_truck_context",
            "normalize_mode": "zscore_baseline",
            "calibration_steps": 2,
            "beta": 1.0,
        },
        env_config={"observation_mode": "default", "action_type": "discrete"},
        observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        action_space=gymnasium.spaces.MultiDiscrete([3]),
        device="cpu",
    )

    def fake_score(_obs, _actions):
        mean = np.asarray([1.0, 3.0], dtype=np.float32)
        std = np.zeros(2, dtype=np.float32)
        members = np.stack([mean, mean], axis=0)
        return mean, std, members

    manager.score = fake_score
    task_reward = np.asarray([0.2, 0.4], dtype=np.float32)

    first_total, first_metrics = manager.shape_rewards(
        task_reward,
        np.zeros((2, 4), dtype=np.float32),
        np.zeros((2, 1), dtype=np.int64),
        global_step=0,
    )
    np.testing.assert_allclose(first_total, task_reward)
    assert first_metrics["preference_reward_applied"] == 0.0

    second_total, second_metrics = manager.shape_rewards(
        task_reward,
        np.zeros((2, 4), dtype=np.float32),
        np.zeros((2, 1), dtype=np.int64),
        global_step=10,
    )
    assert second_metrics["preference_reward_applied"] == 1.0
    assert not np.allclose(second_total, task_reward)
