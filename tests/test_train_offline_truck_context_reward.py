from pathlib import Path

import numpy as np
import torch

from scripts.train_offline_truck_context_reward import (
    _split_indices,
    evaluate_reward_model,
    load_trained_reward_model,
    train_offline_truck_context_reward,
)


def test_split_indices_is_deterministic():
    train_a, val_a = _split_indices(total_windows=10, train_fraction=0.8, seed=42)
    train_b, val_b = _split_indices(total_windows=10, train_fraction=0.8, seed=42)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert len(train_a) == 8
    assert len(val_a) == 2


def test_train_offline_truck_context_reward_smoke(tmp_path):
    pref_path = Path("outputs/preferences/nuplan_boston_test10_truck_context_preferences.pt")
    if not pref_path.exists():
        return

    output_dir = tmp_path / "reward"
    summary = train_offline_truck_context_reward(
        preference_path=pref_path,
        output_dir=output_dir,
        ensemble_size=1,
        rounds=1,
        mb_size=8,
        train_batch_size=8,
        train_fraction=0.8,
        split_seed=7,
        eval_examples=2,
        timestep_loss_weight=0.5,
    )
    assert summary["inserted_windows"] >= 1
    assert summary["train_windows"] >= 1
    assert summary["validation_windows"] >= 1
    assert summary["timestep_loss_weight"] == 0.5
    assert 0.0 <= summary["final_validation_acc"] <= 1.0
    assert 0.0 <= summary["final_validation_timestep_acc"] <= 1.0
    assert len(summary["round_timestep_loss"]) == 1
    assert len(summary["round_total_loss"]) == 1
    assert (output_dir / "offline_truck_context_reward_summary.json").exists()
    assert (output_dir / "offline_truck_context_reward_eval.json").exists()


def test_reload_trained_reward_model_and_score(tmp_path):
    pref_path = Path("outputs/preferences/nuplan_boston_test10_truck_context_preferences.pt")
    if not pref_path.exists():
        return

    output_dir = tmp_path / "reward"
    summary = train_offline_truck_context_reward(
        preference_path=pref_path,
        output_dir=output_dir,
        ensemble_size=1,
        rounds=1,
        mb_size=8,
        train_batch_size=8,
        train_fraction=0.8,
        split_seed=3,
        eval_examples=1,
        timestep_loss_weight=0.5,
    )
    payload = torch.load(pref_path, map_location="cpu")
    model = load_trained_reward_model(
        output_dir=output_dir,
        obs_dim=int(payload["metadata"]["obs_dim"]),
        action_dim=int(payload["metadata"]["action_dim"]),
        size_segment=int(payload["metadata"]["window_len"]),
        ensemble_size=1,
    )
    evaluation = evaluate_reward_model(
        model,
        np.asarray(payload["preferred_sa"][:2], dtype=np.float32),
        np.asarray(payload["rejected_sa"][:2], dtype=np.float32),
        np.asarray(payload["labels"][:2], dtype=np.float32),
    )
    assert 0.0 <= evaluation["accuracy"] <= 1.0
    assert 0.0 <= evaluation["timestep_accuracy"] <= 1.0
    assert evaluation["prob_preferred_first"].shape == (2,)
    assert summary["output_dir"] == str(output_dir)
    assert summary["final_validation_timestep_loss"] >= 0.0
