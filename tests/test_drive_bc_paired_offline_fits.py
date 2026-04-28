import json
from pathlib import Path

import gymnasium
import numpy as np
import pytest
import torch

from pufferlib.ocean.drive import drive as drive_module


OBS_DIM = 1120
ACTION_SPACE = 91


def _write_source_split(source_dir, fit_map_names):
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for idx, fit_map_name in enumerate(fit_map_names):
        source_map_name = f"map_{idx:03d}.bin"
        (source_dir / source_map_name).touch()
        scenarios.append(
            {
                "map_name": source_map_name,
                "original_map_name": fit_map_name,
                "split_index": idx,
                "source_index": idx,
                "scenario_id": f"scenario-{idx}",
            }
        )
    with open(source_dir / "selection_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(scenarios), "scenarios": scenarios}, f)


def _fit_side(*, steps=6, obs_dim=OBS_DIM, action_offset=0, timesteps=None, actions=None):
    obs = torch.arange(steps * obs_dim, dtype=torch.float32).reshape(steps, obs_dim)
    obs = obs / float(max(steps * obs_dim, 1))
    if actions is None:
        actions = torch.tensor([(action_offset + idx) % ACTION_SPACE for idx in range(steps)], dtype=torch.int64)
    if timesteps is None:
        timesteps = torch.arange(steps, dtype=torch.int64)
    return {
        "status": "ok",
        "num_steps": steps,
        "actions": torch.as_tensor(actions, dtype=torch.int64),
        "logged_obs_default": obs,
        "logged_timestep": torch.as_tensor(timesteps, dtype=torch.int64),
    }


def _write_paired_fit_export(tmp_path, fit_map_names, *, side_mutator=None):
    fit_export = tmp_path / "paired_offline_fits.pt"
    shard_dir = tmp_path / "paired_offline_fits_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "paired_offline_fits.part00001.pt"
    pairs = {}
    for idx, fit_map_name in enumerate(fit_map_names):
        car_side = _fit_side(action_offset=idx * 10)
        if side_mutator is not None:
            car_side = side_mutator(idx, car_side)
        pairs[fit_map_name] = {"car": car_side}
    torch.save(
        {
            "format": "sharded_paired_fits_v1",
            "map_names": list(fit_map_names),
            "pairs": pairs,
        },
        shard_path,
    )
    torch.save(
        {
            "format": "sharded_paired_fits_v1",
            "metadata": {"shared_maps": list(fit_map_names)},
            "shards": [
                {
                    "path": str(shard_path),
                    "map_count": len(fit_map_names),
                    "start_index": 0,
                    "end_index": len(fit_map_names),
                }
            ],
        },
        fit_export,
    )
    return fit_export


def test_paired_offline_fit_sequence_loader_shapes_and_masks(tmp_path):
    fit_map_names = ["map_00000.bin", "map_00001.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)
    fit_export = _write_paired_fit_export(tmp_path, fit_map_names)

    records = drive_module._load_paired_fit_source_records(source_dir)
    dataset = drive_module._PairedOfflineFitSequenceDataset(
        fit_export,
        records,
        OBS_DIM,
        ACTION_SPACE,
        seq_len=4,
        stride=2,
        fit_side="car",
        obs_key="logged_obs_default",
        shuffle=False,
        seed=0,
    )

    assert [record.fit_map_name for record in records] == fit_map_names
    assert len(dataset) == 6
    samples = list(dataset)
    assert len(samples) == 6
    assert sum(int(mask.sum().item()) for _, _, mask in samples) == 20
    for obs, action, mask in samples:
        assert obs.shape == (4, OBS_DIM)
        assert action.shape == (4,)
        assert mask.shape == (4,)
        assert action[mask].min().item() >= 0
        assert action[mask].max().item() < ACTION_SPACE

    loader = drive_module._make_bc_loader(dataset, batch_size=3, shuffle=False, num_workers=0)
    obs_batch, action_batch, mask_batch = next(iter(loader))
    assert obs_batch.shape == (3, 4, OBS_DIM)
    assert action_batch.shape == (3, 4)
    assert mask_batch.shape == (3, 4)


def test_paired_offline_fit_sequence_loader_rejects_nonconsecutive_timesteps(tmp_path):
    fit_map_names = ["map_00000.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)

    def _break_timestep(_idx, side):
        side = dict(side)
        side["logged_timestep"] = torch.tensor([0, 1, 2, 4, 5, 6], dtype=torch.int64)
        return side

    fit_export = _write_paired_fit_export(tmp_path, fit_map_names, side_mutator=_break_timestep)
    records = drive_module._load_paired_fit_source_records(source_dir)
    dataset = drive_module._PairedOfflineFitSequenceDataset(
        fit_export,
        records,
        OBS_DIM,
        ACTION_SPACE,
        seq_len=4,
        stride=2,
        fit_side="car",
        obs_key="logged_obs_default",
        shuffle=False,
        seed=0,
    )

    with pytest.raises(ValueError, match="logged_timestep"):
        len(dataset)


class _DummyBCEnv:
    def __init__(self):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.MultiDiscrete([ACTION_SPACE])

    def close(self):
        pass


class _TinyRecurrentPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(OBS_DIM, ACTION_SPACE)

    def forward(self, obs, state=None):
        logits = self.actor(obs.reshape(-1, obs.shape[-1]))
        return logits, state


def test_bc_trainer_streams_paired_offline_fits_and_writes_checkpoints(tmp_path, monkeypatch):
    fit_map_names = ["map_00000.bin", "map_00001.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)
    fit_export = _write_paired_fit_export(tmp_path, fit_map_names)
    output_dir = tmp_path / "checkpoints"

    monkeypatch.setattr(drive_module, "_make_bc_training_env", lambda _env_cfg: _DummyBCEnv())
    monkeypatch.setattr(
        drive_module,
        "_build_bc_policy",
        lambda _args, _env, device: _TinyRecurrentPolicy().to(device),
    )

    args = {
        "package": "ocean",
        "env_name": "puffer_drive",
        "policy_name": "Drive",
        "rnn_name": "Recurrent",
        "policy": {},
        "rnn": {},
        "train": {
            "device": "cpu",
            "seed": 7,
            "learning_rate": 0.01,
            "bptt_horizon": 4,
        },
        "env": {
            "map_dir": str(source_dir),
            "num_maps": len(fit_map_names),
            "num_agents": 1,
            "action_type": "discrete",
            "dynamics_model": "articulated",
            "extend_classic_action_space": False,
            "observation_mode": "default",
            "init_mode": "create_all_valid",
            "control_mode": "control_sdc_only",
        },
        "bc_train": {
            "source_format": "paired_offline_fits",
            "fit_export": str(fit_export),
            "source_map_dir": str(source_dir),
            "fit_side": "car",
            "obs_key": "logged_obs_default",
            "output_dir": str(output_dir),
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "num_workers": 0,
            "shard_shuffle_buffer": 1,
            "val_fraction": 0.0,
            "seq_len": 4,
            "sequence_stride": 2,
            "max_maps": -1,
            "save_best": True,
            "log_interval": 0,
        },
    }

    result = drive_module.train_bc_policy(args)
    latest_path = Path(result["latest_path"])
    best_path = Path(result["best_path"])
    metrics_path = Path(result["metadata_path"])
    assert latest_path.is_file()
    assert best_path.is_file()
    assert metrics_path.is_file()

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["source_format"] == "paired_offline_fits"
    assert metrics["observation_dim"] == OBS_DIM
    assert metrics["action_space_size"] == ACTION_SPACE
    assert sorted(metrics["train_maps"]) == sorted(fit_map_names)
    assert metrics["history"][0]["train_samples"] == 20
