from scripts.validate_trajectory_critic_warmstart import DEFAULT_MAP_IDS
from scripts.validate_trajectory_ppo_finetune import DEFAULT_PPO_VALIDATION_SAMPLE_SIZE, default_map_paths


def test_default_map_paths_returns_deterministic_larger_subset(tmp_path):
    for map_id in range(20):
        (tmp_path / f"map_{map_id}.bin").write_bytes(b"demo")
    for map_id in DEFAULT_MAP_IDS:
        (tmp_path / f"map_{map_id}.bin").write_bytes(b"demo")

    selected = default_map_paths(tmp_path)

    assert len(selected) == DEFAULT_PPO_VALIDATION_SAMPLE_SIZE
    selected_names = {path.name for path in selected}
    for map_id in DEFAULT_MAP_IDS:
        assert f"map_{map_id}.bin" in selected_names
