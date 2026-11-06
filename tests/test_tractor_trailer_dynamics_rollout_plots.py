import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_tractor_trailer_dynamics_rollout_plots(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "tests"
        / "drive"
        / "plot_tractor_trailer_dynamics_rollouts.py"
    )
    output_dir = tmp_path / "rollouts"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    output_paths = [Path(line) for line in result.stdout.splitlines()]
    assert len(output_paths) == 5
    assert all(path.exists() for path in output_paths)

    overview = output_dir / "tractor_trailer_dynamics_rollouts.png"
    assert overview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert overview.stat().st_size > 25_000

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert "bounding boxes" in " ".join(manifest["annotations"])
    assert "hitch constraint residual" in manifest["annotations"]
    assert set(manifest["scenarios"]) == {
        "straight_forward",
        "constant_steering",
        "reverse_steering",
    }
    for scenario, metadata in manifest["scenarios"].items():
        assert metadata["frames"] == 41
        assert metadata["duration_seconds"] == 4.0
        assert metadata["max_hitch_error"] < 1e-4
        gif_path = output_dir / metadata["gif"]
        assert gif_path.stat().st_size > 15_000
        with Image.open(gif_path) as image:
            assert image.format == "GIF"
            assert image.n_frames == 41
