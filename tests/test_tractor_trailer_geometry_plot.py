import subprocess
import sys
from pathlib import Path


def test_tractor_trailer_geometry_plot(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tests" / "drive" / "plot_tractor_trailer_geometry.py"
    output = tmp_path / "tractor_trailer_geometry.png"

    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert Path(result.stdout.strip()) == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000
