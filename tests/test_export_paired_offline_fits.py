from pathlib import Path
import subprocess
import sys

import torch


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
    assert "car" in pair and "truck" in pair and "pair_similarity" in pair
    assert "match_cost_total" in pair["car"]
