from pathlib import Path

import pytest

pytest.importorskip("scipy")
pytest.importorskip("loralib")

from scripts.train_offline_truck_context_reward import train_offline_truck_context_reward


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
    )
    assert summary["inserted_windows"] >= 1
    assert (output_dir / "offline_truck_context_reward_summary.json").exists()
