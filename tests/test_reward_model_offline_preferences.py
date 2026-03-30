import pytest
import numpy as np

pytest.importorskip("scipy")
pytest.importorskip("loralib")

from preferences.reward_model import RewardModel
from scripts.build_truck_context_preferences import load_preferences_into_reward_model


def test_reward_model_accepts_cached_offline_preferences():
    window_len = 4
    obs_dim = 3
    action_dim = 1
    pref_payload = {
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
