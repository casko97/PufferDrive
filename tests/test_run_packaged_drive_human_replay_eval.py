from pathlib import Path

from scripts.run_packaged_drive_human_replay_eval import _select_map_paths


def test_select_map_paths_resolves_symlinks(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_map = source_dir / "map_00024.bin"
    source_map.write_bytes(b"demo")

    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    alias_map = alias_dir / "map_000.bin"
    alias_map.symlink_to(source_map)

    selected = _select_map_paths([alias_map], sample_size=1, sample_seed=42)

    assert selected == [source_map.resolve()]
