import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_tractor_trailer_simulator_rollout_plots(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "tests"
        / "drive"
        / "plot_tractor_trailer_simulator_rollouts.py"
    )
    output_dir = tmp_path / "simulator_rollouts"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output_dir)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    output_paths = [Path(line) for line in result.stdout.splitlines()]
    assert len(output_paths) == 5
    assert all(path.exists() for path in output_paths)

    overview = output_dir / "tractor_trailer_simulator_rollouts.png"
    assert overview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert overview.stat().st_size > 100_000

    scenario_names = {
        "straight_forward",
        "constant_steering",
        "reverse_steering",
    }
    for name in scenario_names:
        gif_path = output_dir / f"{name}.gif"
        assert gif_path.read_bytes().startswith((b"GIF87a", b"GIF89a"))
        with Image.open(gif_path) as image:
            assert image.n_frames == 41
            assert image.width >= 600
            assert image.height >= 500

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["source"] == (
        "actual Drive.step() simulator rollout via "
        "get_articulated_ego_state()"
    )
    assert set(manifest["scenarios"]) == scenario_names
    for scenario in manifest["scenarios"].values():
        assert scenario["frames"] == 41
        assert scenario["duration_seconds"] == 4.0
        assert scenario["all_valid"]
        assert scenario["max_hitch_error"] < 1e-3

    straight = manifest["scenarios"]["straight_forward"]
    assert straight["maximum_speed"] > 0.0
    assert abs(straight["final_articulation"]) < 1e-4

    turning = manifest["scenarios"]["constant_steering"]
    assert abs(turning["final_articulation"]) > 0.01

    reverse = manifest["scenarios"]["reverse_steering"]
    assert reverse["maximum_speed"] > 0.0
    assert reverse["minimum_speed"] < 0.0
    assert abs(reverse["final_articulation"]) > 0.01
