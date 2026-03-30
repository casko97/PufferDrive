from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from scripts.build_truck_context_preferences import build_truck_context_preferences


def test_build_truck_context_preferences_from_synthetic_export(tmp_path):
    export_path = tmp_path / "synthetic_export.pt"
    obs_dim = 4
    replay_len = 8
    pair_payload = {
        "metadata": {"fit_settings": {"dt": 0.1}},
        "pairs": {
            "map_000.bin": {
                "truck_context_replay": {
                    "status": "ok",
                    "pair_similarity": {"ade": 1.5, "fde": 2.0},
                    "truck_branch": {
                        "obs": np.arange(replay_len * obs_dim, dtype=np.float32).reshape(replay_len, obs_dim),
                        "obs_default": np.arange(replay_len * obs_dim, dtype=np.float32).reshape(replay_len, obs_dim),
                        "obs_sdc_only_with_trailer": np.arange(replay_len * (obs_dim + 2), dtype=np.float32).reshape(replay_len, obs_dim + 2),
                        "actions": np.arange(replay_len, dtype=np.int32),
                        "rollout_x": np.linspace(0, 7, replay_len + 1, dtype=np.float32),
                        "rollout_y": np.zeros(replay_len + 1, dtype=np.float32),
                        "self_ade": 0.1,
                        "self_fde": 0.2,
                    },
                    "car_branch": {
                        "obs": (np.arange(replay_len * obs_dim, dtype=np.float32).reshape(replay_len, obs_dim) + 1.0),
                        "obs_default": (np.arange(replay_len * obs_dim, dtype=np.float32).reshape(replay_len, obs_dim) + 1.0),
                        "obs_sdc_only_with_trailer": (np.arange(replay_len * (obs_dim + 2), dtype=np.float32).reshape(replay_len, obs_dim + 2) + 10.0),
                        "actions": np.arange(replay_len, dtype=np.int32) + 10,
                        "rollout_x": np.linspace(1, 8, replay_len + 1, dtype=np.float32),
                        "rollout_y": np.zeros(replay_len + 1, dtype=np.float32),
                        "self_ade": 0.3,
                        "self_fde": 0.4,
                    },
                }
            }
        },
    }
    torch.save(pair_payload, export_path)

    output_path = tmp_path / "preferences.pt"
    build_truck_context_preferences(
        export_path,
        output_path,
        window_len=4,
        min_time_diff_seconds=0.0,
        max_start_distance_m=100.0,
        observation_mode="obs",
    )

    payload = torch.load(output_path, map_location="cpu")
    assert payload["metadata"]["total_windows"] == 2
    assert payload["metadata"]["action_type"] == "discrete"
    assert payload["metadata"]["action_encoding"] == "one_hot"
    assert payload["metadata"]["action_dim"] == 91
    assert payload["preferred_sa"].shape == (2, 4, obs_dim + 91)
    assert payload["rejected_sa"].shape == (2, 4, obs_dim + 91)
    assert payload["labels"].shape == (2, 1)
    assert payload["labels"][0, 0] == 0.0


def test_build_truck_context_preferences_allows_observation_mode_selection(tmp_path):
    export_path = tmp_path / "synthetic_export.pt"
    replay_len = 4
    pair_payload = {
        "metadata": {"fit_settings": {"dt": 0.1}},
        "pairs": {
            "map_000.bin": {
                "truck_context_replay": {
                    "status": "ok",
                    "pair_similarity": {"ade": 0.5, "fde": 0.5},
                    "truck_branch": {
                        "obs": np.zeros((replay_len, 3), dtype=np.float32),
                        "obs_default": np.zeros((replay_len, 3), dtype=np.float32),
                        "obs_sdc_only_with_trailer": np.ones((replay_len, 5), dtype=np.float32),
                        "actions": np.arange(replay_len, dtype=np.int32),
                        "rollout_x": np.arange(replay_len + 1, dtype=np.float32),
                        "rollout_y": np.zeros(replay_len + 1, dtype=np.float32),
                        "self_ade": 0.1,
                        "self_fde": 0.1,
                    },
                    "car_branch": {
                        "obs": np.zeros((replay_len, 3), dtype=np.float32),
                        "obs_default": np.zeros((replay_len, 3), dtype=np.float32),
                        "obs_sdc_only_with_trailer": np.full((replay_len, 5), 2.0, dtype=np.float32),
                        "actions": np.arange(replay_len, dtype=np.int32),
                        "rollout_x": np.arange(replay_len + 1, dtype=np.float32),
                        "rollout_y": np.zeros(replay_len + 1, dtype=np.float32),
                        "self_ade": 0.2,
                        "self_fde": 0.2,
                    },
                }
            }
        },
    }
    torch.save(pair_payload, export_path)

    output_path = tmp_path / "preferences_ext.pt"
    build_truck_context_preferences(
        export_path,
        output_path,
        window_len=4,
        min_time_diff_seconds=0.0,
        max_start_distance_m=100.0,
        observation_mode="sdc_only_with_trailer",
    )

    payload = torch.load(output_path, map_location="cpu")
    assert payload["metadata"]["observation_mode"] == "sdc_only_with_trailer"
    assert payload["metadata"]["observation_key"] == "obs_sdc_only_with_trailer"
    assert payload["preferred_sa"].shape == (1, 4, 5 + 91)


def test_build_truck_context_preferences_rejects_unexpected_observation_dim(tmp_path):
    export_path = tmp_path / "bad_export.pt"
    replay_len = 4
    pair_payload = {
        "metadata": {"fit_settings": {"dt": 0.1}},
        "pairs": {
            "map_000.bin": {
                "truck_context_replay": {
                    "status": "ok",
                    "pair_similarity": {"ade": 0.5, "fde": 0.5},
                    "truck_branch": {
                        "obs_default": np.zeros((replay_len, 7), dtype=np.float32),
                        "actions": np.arange(replay_len, dtype=np.int32),
                        "rollout_x": np.arange(replay_len + 1, dtype=np.float32),
                        "rollout_y": np.zeros(replay_len + 1, dtype=np.float32),
                        "self_ade": 0.1,
                        "self_fde": 0.1,
                    },
                    "car_branch": {
                        "obs_default": np.zeros((replay_len, 7), dtype=np.float32),
                        "actions": np.arange(replay_len, dtype=np.int32),
                        "rollout_x": np.arange(replay_len + 1, dtype=np.float32),
                        "rollout_y": np.zeros(replay_len + 1, dtype=np.float32),
                        "self_ade": 0.2,
                        "self_fde": 0.2,
                    },
                }
            }
        },
    }
    torch.save(pair_payload, export_path)

    output_path = tmp_path / "preferences.pt"
    try:
        build_truck_context_preferences(
            export_path,
            output_path,
            window_len=4,
            min_time_diff_seconds=0.0,
            max_start_distance_m=100.0,
            observation_mode="default",
        )
    except ValueError as exc:
        assert "unexpected observation dimension" in str(exc)
    else:
        raise AssertionError("expected dimension validation error")


def test_build_truck_context_preferences_cli_smoke(tmp_path):
    car_root = Path("pufferlib/resources/drive/binaries/nuplanCarBostonTest10")
    truck_root = Path("pufferlib/resources/drive/binaries/nuplanTruckBostonTest10")
    if not car_root.exists() or not truck_root.exists():
        return

    export_path = tmp_path / "paired_fits.pt"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_paired_offline_fits.py",
            "--car-root",
            str(car_root),
            "--truck-root",
            str(truck_root),
            "--output",
            str(export_path),
            "--max-maps",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    pref_path = tmp_path / "preferences.pt"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_truck_context_preferences.py",
            "--input",
            str(export_path),
            "--output",
            str(pref_path),
            "--window-len",
            "32",
            "--min-time-diff-seconds",
            "0.0",
            "--max-start-distance-m",
            "100.0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    payload = torch.load(pref_path, map_location="cpu")
    assert payload["metadata"]["window_len"] == 32
    assert payload["preferred_sa"].shape[0] >= 1
    assert pref_path.with_suffix(".json").exists()


def test_dynamic_window_shift_selection_prefers_smallest_valid_gap(tmp_path):
    export_path = tmp_path / "synthetic_shift_export.pt"
    obs_dim = 2
    replay_len = 12
    truck_rollout_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32)
    truck_rollout_y = np.zeros(replay_len + 1, dtype=np.float32)
    car_rollout_x = np.array([0.0, 1.0, 2.0, 13.0, 14.0, 5.2, 6.1, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32)
    car_rollout_y = np.zeros(replay_len + 1, dtype=np.float32)
    pair_payload = {
        "metadata": {"fit_settings": {"dt": 0.1}},
        "pairs": {
            "map_000.bin": {
                "truck_context_replay": {
                    "status": "ok",
                    "pair_similarity": {"ade": 1.0, "fde": 0.0},
                    "truck_branch": {
                        "obs": np.zeros((replay_len, obs_dim), dtype=np.float32),
                        "obs_default": np.zeros((replay_len, obs_dim), dtype=np.float32),
                        "obs_sdc_only_with_trailer": np.zeros((replay_len, obs_dim + 2), dtype=np.float32),
                        "actions": np.arange(replay_len, dtype=np.int32),
                        "rollout_x": truck_rollout_x,
                        "rollout_y": truck_rollout_y,
                        "self_ade": 0.1,
                        "self_fde": 0.1,
                    },
                    "car_branch": {
                        "obs": np.ones((replay_len, obs_dim), dtype=np.float32),
                        "obs_default": np.ones((replay_len, obs_dim), dtype=np.float32),
                        "obs_sdc_only_with_trailer": np.ones((replay_len, obs_dim + 2), dtype=np.float32),
                        "actions": np.arange(replay_len, dtype=np.int32),
                        "rollout_x": car_rollout_x,
                        "rollout_y": car_rollout_y,
                        "self_ade": 0.2,
                        "self_fde": 0.2,
                    },
                }
            }
        },
    }
    torch.save(pair_payload, export_path)

    output_path = tmp_path / "preferences.pt"
    build_truck_context_preferences(
        export_path,
        output_path,
        window_len=2,
        min_time_diff_seconds=0.3,
        max_start_distance_m=1.0,
        observation_mode="obs",
    )

    payload = torch.load(output_path, map_location="cpu")
    starts = [entry["timestep_start"] for entry in payload["window_metadata"]]
    assert starts == [0, 5, 8]
    assert payload["window_metadata"][1]["shift_steps_from_previous"] == 5
    assert payload["window_metadata"][1]["window_start_distance_m"] <= 1.0
