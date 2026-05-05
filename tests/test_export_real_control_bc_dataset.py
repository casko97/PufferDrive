from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.export_real_control_bc_dataset import (
    EGO_DYNAMICS_OBS_KEY,
    MOTION_PREV_CONTROL_OBS_KEY,
    MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY,
    VELOCITY_XY_OBS_KEY,
    build_real_control_bc_payload,
    export_real_control_bc_dataset,
    load_source_scenarios,
)
from tests.test_drive_offline_fitting_bindings import _write_offline_fit_map


def test_build_real_control_bc_payload_smoke(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_offline_fit_map(map_dir)

    payload = build_real_control_bc_payload(map_dir / "map_000.bin")

    assert tuple(payload["obs"].shape)[0] == int(payload["action"].shape[0])
    assert tuple(payload["obs"].shape)[1] > 0
    assert payload["obs_default"].shape == payload["obs"].shape
    assert payload[EGO_DYNAMICS_OBS_KEY].shape[0] == payload["obs"].shape[0]
    assert payload[EGO_DYNAMICS_OBS_KEY].shape[1] == payload["obs"].shape[1] + 5
    assert payload[VELOCITY_XY_OBS_KEY].shape[0] == payload["obs"].shape[0]
    assert payload[VELOCITY_XY_OBS_KEY].shape[1] > payload["obs"].shape[1]
    assert payload[MOTION_PREV_CONTROL_OBS_KEY].shape[0] == payload["obs"].shape[0]
    assert payload[MOTION_PREV_CONTROL_OBS_KEY].shape[1] == payload["obs"].shape[1] + 6
    assert payload[MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY].shape[0] == payload["obs"].shape[0]
    assert payload[MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY].shape[1] == payload["obs"].shape[1] + 18
    assert payload["obs_sdc_only_with_trailer"].shape[0] == payload["obs"].shape[0]
    assert payload["obs_sdc_only_with_trailer"].shape[1] > payload["obs"].shape[1]
    assert payload["timestep"].shape == payload["action"].shape
    assert payload["sequence_row_index"].shape == payload["action"].shape
    assert payload["sequence_length"].shape == payload["action"].shape
    assert payload["trajectory_ref_accel"].shape == payload["action"].shape
    assert payload["trajectory_ref_steer"].shape == payload["action"].shape
    ego_dynamics = payload[EGO_DYNAMICS_OBS_KEY][:, -5:]
    assert torch.isfinite(ego_dynamics).all()
    assert torch.all(ego_dynamics[:, 0].abs() <= 1.0)
    assert torch.all(ego_dynamics[:, 1].abs() <= 1.0)
    assert torch.all(ego_dynamics[:, 2].abs() <= 1.0)
    assert torch.all(ego_dynamics[:, 3].abs() <= 1.0)
    assert torch.all(ego_dynamics[:, 4].abs() <= 1.0)
    velocity_xy = payload[VELOCITY_XY_OBS_KEY]
    assert torch.isfinite(velocity_xy).all()
    motion_prev_control = payload[MOTION_PREV_CONTROL_OBS_KEY][:, -6:]
    assert torch.isfinite(motion_prev_control).all()
    assert torch.all(motion_prev_control[:, 0].abs() <= 1.0)
    assert torch.all(motion_prev_control[:, 1].abs() <= 1.0)
    assert torch.all(motion_prev_control[:, 2].abs() <= 1.0)
    assert torch.all(motion_prev_control[:, 3].abs() <= 1.0)
    assert torch.all(motion_prev_control[:, 4].abs() <= 1.0)
    assert torch.all(motion_prev_control[:, 5].abs() <= 1.0)
    assert torch.allclose(motion_prev_control[0, 4:], torch.zeros(2))
    road_controls = payload[MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY][:, -12:]
    assert torch.isfinite(road_controls).all()
    assert torch.all((road_controls[:, 0::3] == 0.0) | (road_controls[:, 0::3] == 1.0))
    assert torch.all(road_controls[:, 1::3].abs() <= 1.0)
    assert torch.all(road_controls[:, 2::3].abs() <= 1.0)


def test_export_real_control_bc_dataset_writes_bc_shards(tmp_path):
    source_dir = tmp_path / "source_split"
    source_dir.mkdir()
    _write_offline_fit_map(source_dir)
    manifest = {
        "count": 1,
        "scenarios": [
            {
                "map_name": "map_000.bin",
                "split_index": 0,
                "source_dataset_map_path": str(source_dir / "map_000.bin"),
                "scenario_id": "synthetic",
                "scenario_type": "synthetic_test",
            }
        ],
    }
    (source_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))

    scenarios = load_source_scenarios(source_dir)
    assert len(scenarios) == 1
    assert scenarios[0].shard_name == "map_000.pt"

    output_dir = tmp_path / "bc_real_controls"
    export_real_control_bc_dataset(source_dir=source_dir, output_dir=output_dir, max_maps=1, log_every=1)

    shard_path = output_dir / "map_000.pt"
    assert shard_path.is_file()
    payload = torch.load(shard_path, map_location="cpu")
    required = {
        "obs",
        "obs_default",
        EGO_DYNAMICS_OBS_KEY,
        VELOCITY_XY_OBS_KEY,
        MOTION_PREV_CONTROL_OBS_KEY,
        MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY,
        "obs_sdc_only_with_trailer",
        "action",
        "map_id",
        "timestep",
        "sequence_id",
        "sequence_row_index",
        "sequence_length",
        "trajectory_ref_accel",
        "trajectory_ref_steer",
    }
    assert required.issubset(payload.keys())
    assert int(payload["obs"].shape[0]) == int(payload["action"].shape[0])
    assert payload["obs_default"].shape == payload["obs"].shape
    assert payload[EGO_DYNAMICS_OBS_KEY].shape[1] == payload["obs"].shape[1] + 5
    assert payload[VELOCITY_XY_OBS_KEY].shape[1] > payload["obs"].shape[1]
    assert payload[MOTION_PREV_CONTROL_OBS_KEY].shape[1] == payload["obs"].shape[1] + 6
    assert payload[MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY].shape[1] == payload["obs"].shape[1] + 18
    assert int(payload["obs_sdc_only_with_trailer"].shape[0]) == int(payload["action"].shape[0])
    dataset_manifest = json.loads((output_dir / "dataset_manifest.json").read_text())
    assert dataset_manifest["count"] == 1
    assert dataset_manifest["row_count"] == int(payload["action"].shape[0])
    assert dataset_manifest["observation_keys"] == [
        "obs",
        "obs_default",
        EGO_DYNAMICS_OBS_KEY,
        VELOCITY_XY_OBS_KEY,
        MOTION_PREV_CONTROL_OBS_KEY,
        MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY,
        "obs_sdc_only_with_trailer",
    ]
    assert int(dataset_manifest[f"{EGO_DYNAMICS_OBS_KEY}_dim"]) == int(payload[EGO_DYNAMICS_OBS_KEY].shape[1])
    assert int(dataset_manifest[f"{VELOCITY_XY_OBS_KEY}_dim"]) == int(payload[VELOCITY_XY_OBS_KEY].shape[1])
    assert int(dataset_manifest[f"{MOTION_PREV_CONTROL_OBS_KEY}_dim"]) == int(payload[MOTION_PREV_CONTROL_OBS_KEY].shape[1])
    assert int(dataset_manifest[f"{MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY}_dim"]) == int(
        payload[MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_KEY].shape[1]
    )
