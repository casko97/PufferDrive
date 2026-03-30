from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from scripts.export_paired_offline_fits import _fit_side, _replay_action_sequence, _stage_map, _truck_context_replay


def test_export_paired_offline_fits_smoke(tmp_path):
    car_root = Path("pufferlib/resources/drive/binaries/nuplanCarBostonTest10")
    truck_root = Path("pufferlib/resources/drive/binaries/nuplanTruckBostonTest10")
    if not car_root.exists() or not truck_root.exists():
        return

    output_path = tmp_path / "paired_fits.pt"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_paired_offline_fits.py",
            "--car-root",
            str(car_root),
            "--truck-root",
            str(truck_root),
            "--output",
            str(output_path),
            "--max-maps",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert output_path.exists()

    payload = torch.load(output_path, map_location="cpu")
    assert "metadata" in payload
    assert "pairs" in payload
    assert len(payload["pairs"]) == 1
    pair = next(iter(payload["pairs"].values()))
    assert "car" in pair and "truck" in pair and "pair_similarity" in pair and "truck_context_replay" in pair
    assert "match_cost_total" in pair["car"]
    replay = pair["truck_context_replay"]
    assert replay["status"] == "ok"
    assert "truck_branch" in replay and "car_branch" in replay and "pair_similarity" in replay
    assert replay["truck_branch"]["obs"].shape[0] == replay["truck_branch"]["actions"].shape[0]
    assert replay["car_branch"]["obs"].shape[0] == replay["car_branch"]["actions"].shape[0]


def test_truck_context_replay_obs_are_from_truck_scenario():
    car_root = Path("pufferlib/resources/drive/binaries/nuplanCarBostonTest10")
    truck_root = Path("pufferlib/resources/drive/binaries/nuplanTruckBostonTest10")
    if not car_root.exists() or not truck_root.exists():
        return

    map_name = "map_000.bin"
    car_side = _fit_side(car_root / map_name)
    truck_side = _fit_side(truck_root / map_name)
    replay = _truck_context_replay(truck_root / map_name, car_side, truck_side)

    assert car_side["status"] == "ok"
    assert truck_side["status"] == "ok"
    assert replay["status"] == "ok"

    staged = _stage_map(truck_root / map_name)
    try:
        fresh_replay = _replay_action_sequence(
            Path(staged.name),
            car_side["actions"],
            gt={"x": truck_side["gt_x"], "y": truck_side["gt_y"]},
        )
    finally:
        staged.cleanup()

    cached_obs = replay["car_branch"]["obs"]
    fresh_obs = fresh_replay["obs"]
    assert cached_obs.shape == fresh_obs.shape
    assert np.allclose(cached_obs, fresh_obs)

    car_logged_obs = car_side["logged_obs"][: len(cached_obs)]
    assert car_logged_obs.shape == cached_obs.shape
    assert not np.allclose(car_logged_obs, cached_obs)
