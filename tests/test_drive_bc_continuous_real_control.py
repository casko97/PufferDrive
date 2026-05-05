import json
from pathlib import Path

import gymnasium
import numpy as np
import pytest
import torch

from pufferlib.ocean.drive import drive as drive_module


OBS_DIM = 4
ACTION_SPACE = 91


def _write_continuous_shard_dataset(dataset_dir, *, num_shards=2, rows_per_shard=6):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for shard_idx in range(num_shards):
        obs = torch.arange(rows_per_shard * OBS_DIM, dtype=torch.float32).reshape(rows_per_shard, OBS_DIM)
        accel = torch.linspace(-4.0, 4.0, rows_per_shard, dtype=torch.float32)
        steer = torch.linspace(-1.0, 1.0, rows_per_shard, dtype=torch.float32)
        payload = {
            "obs": obs + float(10 * shard_idx),
            "action": torch.arange(rows_per_shard, dtype=torch.int64) % ACTION_SPACE,
            "map_id": torch.full((rows_per_shard,), shard_idx, dtype=torch.int64),
            "timestep": torch.arange(rows_per_shard, dtype=torch.int64),
            "sequence_id": torch.full((rows_per_shard,), shard_idx, dtype=torch.int64),
            "sequence_row_index": torch.arange(rows_per_shard, dtype=torch.int64),
            "sequence_length": torch.full((rows_per_shard,), rows_per_shard, dtype=torch.int64),
            "trajectory_ref_accel": accel,
            "trajectory_ref_steer": steer,
        }
        torch.save(payload, dataset_dir / f"map_{shard_idx:03d}.pt")

    manifest = {
        "count": num_shards,
        "observation_keys": ["obs"],
        "obs_dim": OBS_DIM,
        "action_source": "trajectory",
    }
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class _DummyContinuousBCEnv:
    def __init__(self):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def close(self):
        pass


class _TinyContinuousRecurrentPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(OBS_DIM, 2)
        self.logstd = torch.nn.Parameter(torch.zeros(1, 2))

    def forward(self, obs, state=None):
        mean = self.actor(obs.reshape(-1, obs.shape[-1]))
        std = torch.exp(self.logstd.expand_as(mean))
        return torch.distributions.Normal(mean, std), state


def test_continuous_sequence_loader_normalizes_real_control_targets(tmp_path):
    dataset_dir = tmp_path / "bc_dataset"
    _write_continuous_shard_dataset(dataset_dir, num_shards=1)
    dataset = drive_module._ContinuousSequenceBCDataset(
        [str(dataset_dir / "map_000.pt")],
        OBS_DIM,
        ACTION_SPACE,
        seq_len=4,
        stride=2,
        shuffle=False,
        seed=0,
        obs_field="obs",
        require_embedded=False,
        continuous_accel_field="trajectory_ref_accel",
        continuous_steer_field="trajectory_ref_steer",
        accel_scale=4.0,
        steer_scale=1.0,
    )

    samples = list(dataset)
    assert len(samples) == 3
    obs_window, target_window, mask_window = samples[0]
    assert obs_window.shape == (4, OBS_DIM)
    assert target_window.shape == (4, 2)
    assert mask_window.shape == (4,)
    assert torch.all(target_window[mask_window].abs() <= 1.0 + 1e-6)


def test_bc_trainer_supports_continuous_real_control_shards(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "bc_dataset"
    _write_continuous_shard_dataset(dataset_dir, num_shards=2)
    output_dir = tmp_path / "checkpoints"

    monkeypatch.setattr(drive_module, "_make_bc_training_env", lambda _env_cfg: _DummyContinuousBCEnv())
    monkeypatch.setattr(
        drive_module,
        "_build_bc_policy",
        lambda _args, _env, device: _TinyContinuousRecurrentPolicy().to(device),
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
            "use_rnn": True,
        },
        "env": {
            "map_dir": str(dataset_dir),
            "num_maps": 2,
            "num_agents": 1,
            "action_type": "continuous",
            "dynamics_model": "articulated",
            "extend_classic_action_space": False,
            "observation_mode": "default",
            "init_mode": "create_all_valid",
            "control_mode": "control_sdc_only",
        },
        "bc_train": {
            "source_format": "shards",
            "mode": "recurrent",
            "dataset_dir": str(dataset_dir),
            "obs_field": "obs",
            "continuous_accel_field": "trajectory_ref_accel",
            "continuous_steer_field": "trajectory_ref_steer",
            "output_dir": str(output_dir),
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "num_workers": 0,
            "shard_shuffle_buffer": 1,
            "val_fraction": 0.5,
            "seq_len": 4,
            "sequence_stride": 2,
            "save_best": True,
            "log_interval": 0,
            "index_log_interval": 1,
        },
    }

    result = drive_module.train_bc_policy(args)
    metrics_path = Path(result["metadata_path"])
    assert Path(result["latest_path"]).is_file()
    assert Path(result["best_path"]).is_file()
    assert metrics_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["continuous_actions"] is True
    assert metrics["model_action_dim"] == 2
    assert metrics["action_space_size"] == ACTION_SPACE
    assert "train_mae" in metrics["history"][0]
    assert "val_mae" in metrics["history"][0]
