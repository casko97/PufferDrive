from pathlib import Path

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive.embodiment_resolution_diagnostics import plot_embodiment_resolution


MAP_DIR = str(Path(__file__).parents[1] / "resources" / "drive" / "binaries")


def test_embodiment_clearance_requires_a_nonnegative_value():
    with pytest.raises(ValueError, match="embodiment_clearance must be >= 0"):
        Drive(embodiment_clearance=-0.01)


def test_real_map_conflict_is_resolved_by_smaller_embodiments_and_visualized(tmp_path):
    # map_000 at t=70 contains a close real-trajectory vehicle configuration.
    env = Drive(num_agents=1, num_maps=1, map_dir=MAP_DIR, sequential_map_sampling=True,
                init_mode="create_only_controlled", control_mode="control_vehicles",
                dynamics_model="jerk", embodiment_randomization=True, action_type="continuous",
                episode_length=91, init_steps=70, resample_frequency=0)
    try:
        env.reset(seed=3)
        resolution = env.get_embodiment_resolution()
        assert len(resolution) == 3
        assert any(row[7] > 1 for row in resolution)
        for row in resolution:
            history = row[:row[7]]
            assert all(next_state <= state for state, next_state in zip(history, history[1:]))
            assert row[7] <= 7
        np.testing.assert_array_equal(env.observations[:, 5], np.zeros(3, dtype=np.float32))
        output = plot_embodiment_resolution(env.get_global_agent_state(), resolution, tmp_path / "resolution.svg")
        assert output.exists()
        svg = output.read_text(encoding="utf-8")
        assert "resolution round 0" in svg
        assert "resolution round 1" in svg
    finally:
        env.close()
