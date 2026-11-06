import inspect
import subprocess
from pathlib import Path


def test_tractor_trailer_lifecycle(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "tests" / "drive" / "test_tractor_trailer_lifecycle.c"
    executable = tmp_path / "test_tractor_trailer_lifecycle"
    raylib_root = repo_root / "raylib-5.5_linux_amd64"

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
            f"-I{raylib_root / 'include'}",
            str(source),
            "-Wl,--gc-sections",
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "tractor-trailer lifecycle tests passed"


def test_tractor_trailer_python_api_contract():
    from pufferlib.ocean.drive.drive import Drive

    signature = inspect.signature(Drive.__init__)
    assert signature.parameters["ego_vehicle_type"].default == "car"

    try:
        Drive(ego_vehicle_type="unsupported")
    except ValueError as error:
        assert "car" in str(error)
        assert "tractor_trailer" in str(error)
    else:
        raise AssertionError("unsupported ego_vehicle_type must raise ValueError")
