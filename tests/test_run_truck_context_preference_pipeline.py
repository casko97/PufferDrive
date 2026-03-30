from pathlib import Path
import subprocess
import sys

import torch


def test_pipeline_fit_and_preferences_stages(tmp_path):
    car_root = Path("pufferlib/resources/drive/binaries/nuplanCarBostonTest10")
    truck_root = Path("pufferlib/resources/drive/binaries/nuplanTruckBostonTest10")
    if not car_root.exists() or not truck_root.exists():
        return

    fit_output = tmp_path / "paired_fits.pt"
    pref_output = tmp_path / "preferences.pt"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_truck_context_preference_pipeline.py",
            "--stage",
            "fit",
            "--car-root",
            str(car_root),
            "--truck-root",
            str(truck_root),
            "--fit-output",
            str(fit_output),
            "--max-maps",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert fit_output.exists()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_truck_context_preference_pipeline.py",
            "--stage",
            "preferences",
            "--fit-output",
            str(fit_output),
            "--preference-output",
            str(pref_output),
            "--window-len",
            "32",
            "--min-gap-seconds",
            "0.0",
            "--max-start-distance-m",
            "100.0",
            "--observation-mode",
            "sdc_only_with_trailer",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert pref_output.exists()
    assert pref_output.with_suffix(".json").exists()

    payload = torch.load(pref_output, map_location="cpu")
    assert payload["metadata"]["total_windows"] >= 1
    assert payload["metadata"]["observation_mode"] == "sdc_only_with_trailer"
