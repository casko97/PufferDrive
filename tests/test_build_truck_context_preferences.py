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
        "metadata": {},
        "pairs": {
            "map_000.bin": {
                "truck_context_replay": {
                    "status": "ok",
                    "pair_similarity": {"ade": 1.5, "fde": 2.0},
                    "truck_branch": {
                        "obs": np.arange(replay_len * obs_dim, dtype=np.float32).reshape(replay_len, obs_dim),
                        "actions": np.arange(replay_len, dtype=np.int32),
                        "rollout_x": np.linspace(0, 7, replay_len + 1, dtype=np.float32),
                        "rollout_y": np.zeros(replay_len + 1, dtype=np.float32),
                        "self_ade": 0.1,
                        "self_fde": 0.2,
                    },
                    "car_branch": {
                        "obs": (np.arange(replay_len * obs_dim, dtype=np.float32).reshape(replay_len, obs_dim) + 1.0),
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
    build_truck_context_preferences(export_path, output_path, window_len=4, stride=4)

    payload = torch.load(output_path, map_location="cpu")
    assert payload["metadata"]["total_windows"] == 2
    assert payload["preferred_sa"].shape == (2, 4, obs_dim + 1)
    assert payload["rejected_sa"].shape == (2, 4, obs_dim + 1)
    assert payload["labels"].shape == (2, 1)
    assert payload["labels"][0, 0] == 0.0


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
            "--stride",
            "32",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    payload = torch.load(pref_path, map_location="cpu")
    assert payload["metadata"]["window_len"] == 32
    assert payload["preferred_sa"].shape[0] >= 1
