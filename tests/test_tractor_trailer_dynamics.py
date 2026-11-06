import inspect
import subprocess
from pathlib import Path


def test_tractor_trailer_dynamics(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "tests" / "drive" / "test_tractor_trailer_dynamics.c"
    executable = tmp_path / "test_tractor_trailer_dynamics"
    raylib_root = repo_root / "raylib-5.5_linux_amd64"
    raylib_include = raylib_root / "include"

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
    assert result.stdout.strip() == "tractor-trailer dynamics tests passed"


def test_tractor_trailer_dynamics_api_contract():
    from pufferlib.ocean.drive.drive import Drive

    signature = inspect.signature(Drive.__init__)
    assert signature.parameters["max_articulation_angle"].default is None

    for invalid_limit in (0.0, -0.1, 3.2):
        try:
            Drive(max_articulation_angle=invalid_limit)
        except ValueError as error:
            assert "max_articulation_angle" in str(error)
        else:
            raise AssertionError("invalid articulation limit must raise ValueError")
