import numpy as np

from preferences.reward_model import RewardModel
from scripts.build_truck_context_preferences import load_preferences_into_reward_model


def test_reward_model_accepts_cached_offline_preferences():
    window_len = 4
    obs_dim = 3
    action_dim = 1
    pref_payload = {
        "metadata": {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "window_len": window_len,
            "action_type": "discrete",
            "action_encoding": "one_hot",
        },
        "preferred_sa": np.random.randn(3, window_len, obs_dim + action_dim).astype(np.float32),
        "rejected_sa": np.random.randn(3, window_len, obs_dim + action_dim).astype(np.float32),
        "labels": np.zeros((3, 1), dtype=np.float32),
    }

    model = RewardModel(
        ds=obs_dim,
        da=action_dim,
        ensemble_size=1,
        size_segment=window_len,
        capacity=8,
        mb_size=2,
    )
    model.train_batch_size = 2

    inserted = load_preferences_into_reward_model(model, pref_payload)
    assert inserted == 3
    assert model.buffer_index == 3
    acc = model.train_reward()
    assert acc.shape == (1,)


def test_reward_model_rejects_dimension_mismatch():
    window_len = 4
    obs_dim = 3
    action_dim = 1
    pref_payload = {
        "metadata": {
            "obs_dim": obs_dim + 1,
            "action_dim": action_dim,
            "window_len": window_len,
            "action_type": "discrete",
            "action_encoding": "one_hot",
        },
        "preferred_sa": np.random.randn(3, window_len, obs_dim + action_dim).astype(np.float32),
        "rejected_sa": np.random.randn(3, window_len, obs_dim + action_dim).astype(np.float32),
        "labels": np.zeros((3, 1), dtype=np.float32),
    }

    model = RewardModel(
        ds=obs_dim,
        da=action_dim,
        ensemble_size=1,
        size_segment=window_len,
        capacity=8,
        mb_size=2,
    )

    try:
        load_preferences_into_reward_model(model, pref_payload)
    except ValueError as exc:
        assert "obs_dim mismatch" in str(exc)
    else:
        raise AssertionError("expected dimension mismatch error")


def test_reward_model_rejects_scalar_discrete_encoding():
    window_len = 4
    obs_dim = 3
    action_dim = 1
    pref_payload = {
        "metadata": {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "window_len": window_len,
            "action_type": "discrete",
            "action_encoding": "scalar",
        },
        "preferred_sa": np.random.randn(3, window_len, obs_dim + action_dim).astype(np.float32),
        "rejected_sa": np.random.randn(3, window_len, obs_dim + action_dim).astype(np.float32),
        "labels": np.zeros((3, 1), dtype=np.float32),
    }

    model = RewardModel(
        ds=obs_dim,
        da=action_dim,
        ensemble_size=1,
        size_segment=window_len,
        capacity=8,
        mb_size=2,
    )

    try:
        load_preferences_into_reward_model(model, pref_payload)
    except ValueError as exc:
        assert "unsupported discrete action encoding" in str(exc)
    else:
        raise AssertionError("expected unsupported scalar discrete action error")
