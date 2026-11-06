import json
import subprocess
import sys
from pathlib import Path


def test_tractor_trailer_interaction_plot(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "tests"
        / "drive"
        / "plot_tractor_trailer_interactions.py"
    )
    output_dir = tmp_path / "interaction_plot"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    output_paths = [Path(line) for line in result.stdout.splitlines()]
    assert len(output_paths) == 2
    assert all(path.exists() for path in output_paths)

    image = output_dir / "tractor_trailer_interactions.png"
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert image.stat().st_size > 75_000

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["native_test_output"] == (
        "tractor-trailer interaction tests passed"
    )
    assert set(manifest["panels"]) == {
        "mutual_overlap",
        "trailer_collision",
        "trailer_offroad",
        "observation_rendering",
    }
    assert manifest["actor_capacity"]["policy_actions_for_rig"] == 1
    assert manifest["actor_capacity"]["actors_for_rig"] == 2
    assert manifest["actor_capacity"]["maximum_actors_verified"] == 32
