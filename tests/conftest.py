import pytest

from tests.drive_test_fixtures import write_conversion_bin


@pytest.fixture(scope="session")
def generated_conversion_bin(tmp_path_factory):
    maps_dir = tmp_path_factory.mktemp("drive_conversion_maps")
    bin_path = maps_dir / "map_000.bin"
    write_conversion_bin(bin_path, unique_map_id=42)
    return bin_path
