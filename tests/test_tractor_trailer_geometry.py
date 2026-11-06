import subprocess
from pathlib import Path


def test_tractor_trailer_geometry(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "tests" / "drive" / "test_tractor_trailer_geometry.c"
    executable = tmp_path / "test_tractor_trailer_geometry"

    subprocess.run(
        [
            "gcc",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-std=c99",
            f"-I{repo_root}",
            str(source),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "tractor-trailer geometry tests passed"
