import subprocess
from pathlib import Path


def test_tractor_trailer_interactions(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "tests" / "drive" / "test_tractor_trailer_interactions.c"
    executable = tmp_path / "test_tractor_trailer_interactions"
    raylib_include = repo_root / "raylib-5.5_linux_amd64" / "include"

    subprocess.run(
        [
            "gcc",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-Wno-unused-variable",
            "-std=gnu99",
            "-ffunction-sections",
            "-fdata-sections",
            f"-I{repo_root}",
            f"-I{raylib_include}",
            str(source),
            "-Wl,--gc-sections",
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "tractor-trailer interaction tests passed"
