import json
from pathlib import Path

import gymnasium
import numpy as np
import pytest

from preferences.reward_model import RewardModel
from scripts import compare_preference_rollout_models as compare_script


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


def test_build_stratified_sample_manifest_is_deterministic(tmp_path, monkeypatch):
    source_dir = tmp_path / "maps"
    source_dir.mkdir()
    for idx in range(20):
        (source_dir / f"map_{idx:03d}.bin").write_bytes(b"stub")

    def fake_classify(map_path: Path, threshold_deg: float):
        idx = int(map_path.stem.split("_")[1])
        return {
            "map_name": map_path.name,
            "map_path": str(map_path.resolve()),
            "bucket": "turning" if idx % 2 == 0 else "straight",
            "delta_heading_deg": 90.0 if idx % 2 == 0 else 0.0,
        }

    monkeypatch.setattr(compare_script, "classify_map_turning", fake_classify)
    manifest_path = tmp_path / "manifest.json"
    manifest = compare_script.build_stratified_sample_manifest(
        source_map_dir=source_dir,
        output_path=manifest_path,
        turning_count=7,
        straight_count=3,
        threshold_deg=45.0,
        sample_seed=42,
    )

    assert manifest_path.exists()
    assert len(manifest["maps"]) == 10
    assert sum(item["scenario_type"] == "turning" for item in manifest["maps"]) == 7
    assert sum(item["scenario_type"] == "straight" for item in manifest["maps"]) == 3
    assert manifest["excluded_invalid_count"] == 0

    manifest_again = compare_script.build_stratified_sample_manifest(
        source_map_dir=source_dir,
        output_path=tmp_path / "manifest_again.json",
        turning_count=7,
        straight_count=3,
        threshold_deg=45.0,
        sample_seed=42,
    )
    assert manifest["maps"] == manifest_again["maps"]


def test_build_stratified_sample_manifest_filters_invalid_maps(tmp_path, monkeypatch):
    source_dir = tmp_path / "maps"
    source_dir.mkdir()
    for idx in range(8):
        (source_dir / f"map_{idx:03d}.bin").write_bytes(b"stub")

    def fake_classify(map_path: Path, threshold_deg: float):
        idx = int(map_path.stem.split("_")[1])
        return {
            "map_name": map_path.name,
            "map_path": str(map_path.resolve()),
            "bucket": "turning" if idx < 4 else "straight",
            "delta_heading_deg": 90.0 if idx < 4 else 0.0,
        }

    def fake_validity(map_path: Path):
        idx = int(map_path.stem.split("_")[1])
        return {
            "valid_for_sampling": idx not in {1, 5},
            "active_agent_count": 1 if idx not in {1, 5} else 0,
            "invalid_initial_trailer_state": idx == 5,
        }

    monkeypatch.setattr(compare_script, "classify_map_turning", fake_classify)
    manifest = compare_script.build_stratified_sample_manifest(
        source_map_dir=source_dir,
        output_path=tmp_path / "manifest.json",
        turning_count=2,
        straight_count=2,
        threshold_deg=45.0,
        sample_seed=7,
        validity_checker=fake_validity,
    )

    selected_names = {row["map_name"] for row in manifest["maps"]}
    assert "map_001.bin" not in selected_names
    assert "map_005.bin" not in selected_names
    assert manifest["excluded_invalid_count"] == 2
    assert all(Path(row["map_path"]).exists() for row in manifest["maps"])


def test_swap_default_drive_config_restores_after_error(tmp_path):
    drive_ini = tmp_path / "drive.ini"
    source_ini = tmp_path / "source.ini"
    drive_ini.write_text("original", encoding="utf-8")
    source_ini.write_text("replacement", encoding="utf-8")

    with pytest.raises(RuntimeError):
        with compare_script.swap_default_drive_config(source_ini, drive_ini):
            assert drive_ini.read_text(encoding="utf-8") == "replacement"
            raise RuntimeError("boom")

    assert drive_ini.read_text(encoding="utf-8") == "original"


def test_detect_rollout_reset_reason_on_large_position_jump():
    current_state = {
        "id": np.asarray([7], dtype=np.int32),
        "x": np.asarray([0.0], dtype=np.float32),
        "y": np.asarray([0.0], dtype=np.float32),
    }
    next_state = {
        "id": np.asarray([7], dtype=np.int32),
        "x": np.asarray([25.0], dtype=np.float32),
        "y": np.asarray([0.0], dtype=np.float32),
    }

    reason = compare_script._detect_rollout_reset_reason(current_state, next_state, position_jump_threshold_m=20.0)
    assert reason == "position_jump>20m"


def test_detect_rollout_reset_reason_on_agent_id_change():
    current_state = {
        "id": np.asarray([7], dtype=np.int32),
        "x": np.asarray([0.0], dtype=np.float32),
        "y": np.asarray([0.0], dtype=np.float32),
    }
    next_state = {
        "id": np.asarray([8], dtype=np.int32),
        "x": np.asarray([1.0], dtype=np.float32),
        "y": np.asarray([1.0], dtype=np.float32),
    }

    reason = compare_script._detect_rollout_reset_reason(current_state, next_state)
    assert reason == "agent_id_changed"


def test_score_rollout_payloads_uses_offline_calibration_fallback(tmp_path):
    reward_dir = _write_reward_model_dir(tmp_path)
    preference_config = tmp_path / "preference.ini"
    preference_config.write_text(
        "\n".join(
            [
                "[preference_reward]",
                "enabled = true",
                f'model_dir = "{reward_dir}"',
                'checkpoint_stem = "offline_truck_context"',
                "beta = 0.01",
                'normalize_mode = "zscore_baseline"',
                "scale = 1.0",
                "clip_min = -0.5",
                "clip_max = 0.5",
                "warmup_steps = 0",
                "calibration_steps = 5000",
                "lambda_uncertainty = 0.0",
                "log_member_stats = false",
                "strict_observation_match = true",
            ]
        ),
        encoding="utf-8",
    )

    observations = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    rollout_template = {
        "map_name": "map_000.bin",
        "map_path": str((tmp_path / "map_000.bin").resolve()),
        "scenario_type": "turning",
        "delta_heading_deg": 90.0,
        "steps": 2,
        "observations": observations,
        "actions": np.asarray([0, 1], dtype=np.int64),
        "task_rewards": np.asarray([0.1, 0.2], dtype=np.float32),
        "dones": np.asarray([False, True]),
        "truncs": np.asarray([False, False]),
        "values": np.asarray([0.0, 0.0], dtype=np.float32),
        "entropy": np.asarray([0.0, 0.0], dtype=np.float32),
        "x": np.asarray([0.0, 1.0], dtype=np.float32),
        "y": np.asarray([0.0, 1.0], dtype=np.float32),
        "z": np.asarray([0.0, 0.0], dtype=np.float32),
        "heading": np.asarray([0.0, 0.1], dtype=np.float32),
        "length": np.asarray([4.0, 4.0], dtype=np.float32),
        "width": np.asarray([2.0, 2.0], dtype=np.float32),
        "trailer_has": np.asarray([0, 0], dtype=np.int32),
        "trailer_x": np.asarray([0.0, 0.0], dtype=np.float32),
        "trailer_y": np.asarray([0.0, 0.0], dtype=np.float32),
        "trailer_heading": np.asarray([0.0, 0.0], dtype=np.float32),
    }

    car_payload = {
        "format": compare_script.ROLLOUT_FORMAT_VERSION,
        "model_name": "car",
        "model_root": str(tmp_path / "car"),
        "config_path": str(tmp_path / "car.ini"),
        "checkpoint_path": str(tmp_path / "car.pt"),
        "rollouts": [dict(rollout_template)],
    }
    truck_rollout = dict(rollout_template)
    truck_rollout["map_name"] = "map_001.bin"
    truck_rollout["scenario_type"] = "straight"
    truck_payload = {
        "format": compare_script.ROLLOUT_FORMAT_VERSION,
        "model_name": "truck",
        "model_root": str(tmp_path / "truck"),
        "config_path": str(tmp_path / "truck.ini"),
        "checkpoint_path": str(tmp_path / "truck.pt"),
        "rollouts": [truck_rollout],
    }

    car_path = tmp_path / "car_rollouts.pt"
    truck_path = tmp_path / "truck_rollouts.pt"
    torch = __import__("torch")
    torch.save(car_payload, car_path)
    torch.save(truck_payload, truck_path)

    scored_path = compare_script.score_rollout_payloads(
        rollout_paths=[car_path, truck_path],
        preference_config_path=preference_config,
        reward_dir=reward_dir,
        output_path=tmp_path / "scored.pt",
    )

    scored = torch.load(scored_path, map_location="cpu", weights_only=False)
    assert scored["normalization"]["mode"] == "offline_sample"
    assert scored["normalization"]["used_steps"] == 4
    for model_payload in scored["models"]:
        for rollout in model_payload["rollouts"]:
            assert rollout["pref_raw"].shape == (2,)
            assert rollout["pref_shaped"].shape == (2,)
