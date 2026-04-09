import json
from types import SimpleNamespace

import numpy as np
import pytest

import pufferlib.ocean.drive.drive as drive_module
import pufferlib.pufferl as pufferl_module
from pufferlib.ocean.drive.drive import (
    Drive,
    MapDatasetCatalog,
    MapDatasetEntry,
    MapScheduler,
    MapValidationCache,
    save_map_binary,
)


def _write_simple_map(map_path, unique_map_id):
    length = 91
    map_data = {
        "metadata": {
            "sdc_track_index": 0,
            "tracks_to_predict": [{"track_index": 0}],
        },
        "objects": [
            {
                "id": 1000 + unique_map_id,
                "type": "vehicle",
                "length": 4.5,
                "width": 1.9,
                "height": 1.6,
                "position": [{"x": float(idx), "y": 0.0, "z": 0.0} for idx in range(length)],
                "velocity": [{"x": 1.0, "y": 0.0, "z": 0.0} for _ in range(length)],
                "heading": [0.0 for _ in range(length)],
                "valid": [1 for _ in range(length)],
                "goalPosition": {"x": 100.0, "y": 0.0, "z": 0.0},
            }
        ],
        "roads": [],
    }
    save_map_binary(map_data, str(map_path), unique_map_id=unique_map_id)


def test_map_catalog_discovers_flat_bin_directory(tmp_path):
    _write_simple_map(tmp_path / "map_b.bin", unique_map_id=2)
    _write_simple_map(tmp_path / "map_a.bin", unique_map_id=1)

    catalog = MapDatasetCatalog.from_map_dir(str(tmp_path))

    assert [entry.relative_path for entry in catalog.entries] == ["map_a.bin", "map_b.bin"]
    assert [entry.dataset_id for entry in catalog.entries] == [0, 1]


def test_map_catalog_prefers_manifest_paths(tmp_path):
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    nested_dir = maps_dir / "nested"
    nested_dir.mkdir()
    _write_simple_map(nested_dir / "scene.bin", unique_map_id=1)
    manifest = {
        "maps": [
            {
                "map_id": 7,
                "relative_path": "nested/scene.bin",
                "source_name": "scene.json",
            }
        ]
    }
    with (maps_dir / "dataset_manifest.json").open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj)

    catalog = MapDatasetCatalog.from_map_dir(str(maps_dir))

    assert len(catalog.entries) == 1
    assert catalog.entries[0].dataset_id == 7
    assert catalog.entries[0].relative_path == "nested/scene.bin"


def test_validation_cache_reuses_matching_signature(tmp_path):
    cache = MapValidationCache(
        str(tmp_path),
        MapValidationCache.build_signature(
            dynamics_model="classic",
            init_mode=0,
            control_mode=0,
            init_steps=0,
            max_controlled_agents=-1,
            goal_behavior=0,
            goal_target_distance=30.0,
            force_zero_trailer_articulation_at_init=False,
            non_kinematic_vehicle_params_override=None,
        ),
    )
    map_path = str((tmp_path / "map.bin").resolve())
    payload = {
        "active_agent_count": 3,
        "valid_for_sampling": True,
        "invalid_initial_trailer_state": False,
    }

    cache.set(map_path, payload)

    reloaded = MapValidationCache(str(tmp_path), json.loads(cache.signature))
    assert reloaded.get(map_path) == payload


def test_scheduler_shuffle_once_per_epoch_covers_dataset_before_repeat():
    entries = [
        MapDatasetEntry(dataset_id=0, map_path="/tmp/a.bin", relative_path="a.bin"),
        MapDatasetEntry(dataset_id=1, map_path="/tmp/b.bin", relative_path="b.bin"),
        MapDatasetEntry(dataset_id=2, map_path="/tmp/c.bin", relative_path="c.bin"),
    ]
    scheduler = MapScheduler(entries, schedule="shuffle_once_per_epoch", seed=123, allow_live_duplicates=False)

    def inspect(_entry):
        return {"active_agent_count": 1, "valid_for_sampling": True, "invalid_initial_trailer_state": False}

    first = scheduler.select_for_agent_budget(3, inspect)
    second = scheduler.select_for_agent_budget(3, inspect)

    assert {entry.dataset_id for entry, _ in first} == {0, 1, 2}
    assert {entry.dataset_id for entry, _ in second} == {0, 1, 2}


def test_scheduler_select_all_valid_preserves_order_and_skips_invalid():
    entries = [
        MapDatasetEntry(dataset_id=0, map_path="/tmp/a.bin", relative_path="a.bin"),
        MapDatasetEntry(dataset_id=1, map_path="/tmp/b.bin", relative_path="b.bin"),
        MapDatasetEntry(dataset_id=2, map_path="/tmp/c.bin", relative_path="c.bin"),
    ]
    scheduler = MapScheduler(entries, schedule="sequential", seed=7, allow_live_duplicates=False)

    def inspect(entry):
        if entry.dataset_id == 1:
            return {"active_agent_count": 0, "valid_for_sampling": False, "invalid_initial_trailer_state": False}
        return {"active_agent_count": 1, "valid_for_sampling": True, "invalid_initial_trailer_state": False}

    selected = scheduler.select_all_valid(inspect)

    assert [entry.dataset_id for entry, _ in selected] == [0, 2]


def test_drive_startup_and_resample_do_not_use_binding_shared(tmp_path, monkeypatch):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_simple_map(map_dir / "map_000.bin", unique_map_id=1)

    def fail_shared(*args, **kwargs):
        raise AssertionError("binding.shared should not be called")

    monkeypatch.setattr(drive_module.binding, "shared", fail_shared)

    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        resample_frequency=1,
        episode_length=10,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
    )
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape[0] == env.num_agents

        actions = np.zeros_like(env.actions)
        obs, rewards, terminals, truncations, _ = env.step(actions)
        assert obs.shape[0] == env.num_agents
        assert rewards.shape[0] == env.num_agents
        assert terminals.shape[0] == env.num_agents
        assert truncations.shape[0] == env.num_agents
    finally:
        env.close()


def test_drive_sequential_sampling_preserves_valid_dataset_order_without_shared(tmp_path, monkeypatch):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_simple_map(map_dir / "map_000.bin", unique_map_id=1)
    _write_simple_map(map_dir / "map_001.bin", unique_map_id=2)
    _write_simple_map(map_dir / "map_002.bin", unique_map_id=3)

    def fail_shared(*args, **kwargs):
        raise AssertionError("binding.shared should not be called")

    original_inspect_map = drive_module.binding.inspect_map

    def inspect_with_one_invalid(*args, **kwargs):
        metadata = original_inspect_map(*args, **kwargs)
        if kwargs.get("map_path", "").endswith("map_001.bin"):
            metadata["active_agent_count"] = 0
            metadata["valid_for_sampling"] = False
        return metadata

    monkeypatch.setattr(drive_module.binding, "shared", fail_shared)
    monkeypatch.setattr(drive_module.binding, "inspect_map", inspect_with_one_invalid)

    env = Drive(
        num_agents=1,
        num_maps=3,
        map_dir=str(map_dir),
        resample_frequency=0,
        episode_length=10,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        sequential_map_sampling=True,
        map_schedule="shuffle_once_per_epoch",
    )
    try:
        assert env.map_schedule == "sequential"
        assert env.map_ids == [0, 2]
        assert env.num_envs == 2
        assert env.num_agents == env.agent_offsets[-1]
    finally:
        env.close()


def test_pufferl_eval_configures_sequential_map_pipeline(monkeypatch):
    captured = {}

    def fake_load_env(env_name, args):
        captured["env_name"] = env_name
        captured["env"] = dict(args["env"])
        captured["vec"] = dict(args["vec"])
        return SimpleNamespace(driver_env=SimpleNamespace(agent_offsets=[0, 1]))

    def fake_load_policy(args, vecenv, env_name=""):
        return object()

    class DummyHumanReplayEvaluator:
        def __init__(self, args):
            self.args = args

        def rollout(self, args, vecenv, policy):
            return {"ok": True}

    monkeypatch.setattr(pufferl_module, "load_env", fake_load_env)
    monkeypatch.setattr(pufferl_module, "load_policy", fake_load_policy)
    import pufferlib.ocean.benchmark.evaluator as evaluator_module

    monkeypatch.setattr(evaluator_module, "HumanReplayEvaluator", DummyHumanReplayEvaluator)

    args = {
        "package": "ocean",
        "env": {"map_dir": "unused", "num_maps": 1},
        "vec": {"backend": "Multiprocessing", "num_envs": 8},
        "train": {"device": "cpu"},
        "eval": {
            "map_dir": "resources/drive/binaries/validation",
            "wosac_num_maps": 5,
            "wosac_realism_eval": False,
            "human_replay_eval": True,
            "human_replay_control_mode": "control_sdc_only",
            "backend": "PufferEnv",
        },
    }

    result = pufferl_module.eval("puffer_drive", args=args)

    assert result == {"ok": True}
    assert captured["env_name"] == "puffer_drive"
    assert captured["env"]["map_dir"] == "resources/drive/binaries/validation"
    assert captured["env"]["num_maps"] == 5
    assert captured["env"]["sequential_map_sampling"] is True
    assert captured["env"]["map_schedule"] == "sequential"
    assert captured["env"]["map_allow_live_duplicates"] is False
    assert captured["vec"] == {"backend": "PufferEnv", "num_envs": 1}
