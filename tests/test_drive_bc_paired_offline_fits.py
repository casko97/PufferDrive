import json
from pathlib import Path

import gymnasium
import numpy as np
import pytest
import torch

from pufferlib.ocean.drive import drive as drive_module


OBS_DIM = 1120
ACTION_SPACE = 91
STACK_LEN = 3


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


class _TinyFeedforwardPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(OBS_DIM, ACTION_SPACE)

    def forward(self, obs, state=None):
        logits = self.actor(obs)
        return logits, state


class _TinyTwoHeadFeedforwardPolicy(torch.nn.Module):
    def __init__(self, *, action_horizon=1):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.accel = torch.nn.Linear(OBS_DIM * STACK_LEN, 7 * self.action_horizon)
        self.steer = torch.nn.Linear(OBS_DIM * STACK_LEN, 13 * self.action_horizon)

    def forward(self, obs, state=None):
        accel = self.accel(obs)
        steer = self.steer(obs)
        if self.action_horizon == 1:
            return (accel, steer), state
        batch = obs.shape[0]
        return (
            accel.reshape(batch, self.action_horizon, 7),
            steer.reshape(batch, self.action_horizon, 13),
        ), state


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
    assert metrics["mode"] == "recurrent"


def test_paired_offline_fit_flat_loader_shapes_and_counts(tmp_path):
    fit_map_names = ["map_00000.bin", "map_00001.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)
    fit_export = _write_paired_fit_export(tmp_path, fit_map_names)

    records = drive_module._load_paired_fit_source_records(source_dir)
    dataset = drive_module._PairedOfflineFitFlatDataset(
        fit_export,
        records,
        OBS_DIM,
        ACTION_SPACE,
        fit_side="car",
        obs_key="logged_obs_default",
        shuffle=False,
        seed=0,
    )

    assert len(dataset) == 12
    samples = list(dataset)
    assert len(samples) == 12
    for obs, action in samples:
        assert obs.shape == (OBS_DIM,)
        assert isinstance(int(action.item()), int)
        assert 0 <= int(action.item()) < ACTION_SPACE

    loader = drive_module._make_bc_loader(dataset, batch_size=3, shuffle=False, num_workers=0)
    obs_batch, action_batch = next(iter(loader))
    assert obs_batch.shape == (3, OBS_DIM)
    assert action_batch.shape == (3,)


def test_bc_trainer_streams_paired_offline_fits_iid_and_writes_checkpoints(tmp_path, monkeypatch):
    fit_map_names = ["map_00000.bin", "map_00001.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)
    fit_export = _write_paired_fit_export(tmp_path, fit_map_names)
    output_dir = tmp_path / "checkpoints"

    monkeypatch.setattr(drive_module, "_make_bc_training_env", lambda _env_cfg: _DummyBCEnv())
    monkeypatch.setattr(
        drive_module,
        "_build_bc_policy",
        lambda _args, _env, device: _TinyFeedforwardPolicy().to(device),
    )

    args = {
        "package": "ocean",
        "env_name": "puffer_drive",
        "policy_name": "Drive",
        "rnn_name": None,
        "policy": {},
        "rnn": {},
        "train": {
            "device": "cpu",
            "seed": 7,
            "learning_rate": 0.01,
            "use_rnn": False,
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
            "mode": "iid",
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
    assert metrics["mode"] == "iid"
    assert metrics["recurrent"] is False
    assert metrics["observation_dim"] == OBS_DIM
    assert metrics["action_space_size"] == ACTION_SPACE
    assert sorted(metrics["train_maps"]) == sorted(fit_map_names)
    assert metrics["history"][0]["train_samples"] == 12


def test_paired_offline_fit_stacked_iid_loader_shapes_and_targets(tmp_path):
    fit_map_names = ["map_00000.bin", "map_00001.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)
    fit_export = _write_paired_fit_export(tmp_path, fit_map_names)

    records = drive_module._load_paired_fit_source_records(source_dir)
    dataset = drive_module._PairedOfflineFitStackedIIDDataset(
        fit_export,
        records,
        OBS_DIM,
        ACTION_SPACE,
        stack_len=STACK_LEN,
        action_horizon=1,
        fit_side="car",
        obs_key="logged_obs_default",
        extend_classic_action_space=False,
        shuffle=False,
        seed=0,
    )

    assert len(dataset) == 8
    samples = list(dataset)
    assert len(samples) == 8
    for obs, accel, steer in samples:
        assert obs.shape == (OBS_DIM * STACK_LEN,)
        assert 0 <= int(accel.item()) < 7
        assert 0 <= int(steer.item()) < 13
    pairs = torch.load(fit_export.parent / "paired_offline_fits_shards" / "paired_offline_fits.part00001.pt", map_location="cpu")[
        "pairs"
    ]
    raw_obs = pairs[fit_map_names[0]]["car"]["logged_obs_default"]
    expected_obs = torch.cat([raw_obs[2], raw_obs[1], raw_obs[0]], dim=0)
    matching = [
        (obs, accel, steer)
        for obs, accel, steer in samples
        if torch.allclose(obs, expected_obs)
    ]
    assert len(matching) >= 1
    match_obs, match_accel, match_steer = next(
        (obs, accel, steer)
        for obs, accel, steer in matching
        if int(accel.item()) == 0 and int(steer.item()) == 2
    )
    assert torch.allclose(match_obs[:OBS_DIM], raw_obs[2])
    assert torch.allclose(match_obs[OBS_DIM : 2 * OBS_DIM], raw_obs[1])
    assert torch.allclose(match_obs[2 * OBS_DIM :], raw_obs[0])
    assert int(match_accel.item()) == 0
    assert int(match_steer.item()) == 2

    loader = drive_module._make_bc_loader(dataset, batch_size=2, shuffle=False, num_workers=0)
    obs_batch, accel_batch, steer_batch = next(iter(loader))
    assert obs_batch.shape == (2, OBS_DIM * STACK_LEN)
    assert accel_batch.shape == (2,)
    assert steer_batch.shape == (2,)


def test_paired_offline_fit_stacked_iid_loader_supports_future_action_horizon(tmp_path):
    fit_map_names = ["map_00000.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)
    fit_export = _write_paired_fit_export(tmp_path, fit_map_names)

    records = drive_module._load_paired_fit_source_records(source_dir)
    dataset = drive_module._PairedOfflineFitStackedIIDDataset(
        fit_export,
        records,
        OBS_DIM,
        ACTION_SPACE,
        stack_len=STACK_LEN,
        action_horizon=2,
        fit_side="car",
        obs_key="logged_obs_default",
        extend_classic_action_space=False,
        shuffle=False,
        seed=0,
    )

    stacked_obs, accel_idx, steer_idx = next(iter(dataset))
    assert stacked_obs.shape == (OBS_DIM * STACK_LEN,)
    assert accel_idx.shape == (2,)
    assert steer_idx.shape == (2,)


def test_bc_trainer_streams_paired_offline_fits_stacked_iid_and_writes_checkpoints(tmp_path, monkeypatch):
    fit_map_names = ["map_00000.bin", "map_00001.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)
    fit_export = _write_paired_fit_export(tmp_path, fit_map_names)
    output_dir = tmp_path / "checkpoints"

    monkeypatch.setattr(drive_module, "_make_bc_training_env", lambda _env_cfg: _DummyBCEnv())
    monkeypatch.setattr(
        drive_module,
        "_build_bc_policy",
        lambda _args, _env, device: _TinyTwoHeadFeedforwardPolicy().to(device),
    )

    args = {
        "package": "ocean",
        "env_name": "puffer_drive",
        "policy_name": "Drive",
        "rnn_name": None,
        "policy": {},
        "rnn": {},
        "train": {
            "device": "cpu",
            "seed": 7,
            "learning_rate": 0.01,
            "use_rnn": False,
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
            "mode": "stacked_iid",
            "stack_len": STACK_LEN,
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
            "max_maps": -1,
            "save_best": True,
            "log_interval": 0,
        },
    }

    result = drive_module.train_bc_policy(args)
    metrics_path = Path(result["metadata_path"])
    assert metrics_path.is_file()

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["mode"] == "stacked_iid"
    assert metrics["recurrent"] is False
    assert metrics["model_observation_dim"] == OBS_DIM * STACK_LEN
    assert metrics["history"][0]["train_samples"] == 8
    assert "train_accel_accuracy" in metrics["history"][0]
    assert "train_steer_accuracy" in metrics["history"][0]


def test_bc_trainer_streams_paired_offline_fits_stacked_iid_with_future_action_horizon(tmp_path, monkeypatch):
    fit_map_names = ["map_00000.bin", "map_00001.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)
    fit_export = _write_paired_fit_export(tmp_path, fit_map_names)
    output_dir = tmp_path / "checkpoints_h2"

    monkeypatch.setattr(drive_module, "_make_bc_training_env", lambda _env_cfg: _DummyBCEnv())
    monkeypatch.setattr(
        drive_module,
        "_build_bc_policy",
        lambda _args, _env, device: _TinyTwoHeadFeedforwardPolicy(action_horizon=2).to(device),
    )

    args = {
        "package": "ocean",
        "env_name": "puffer_drive",
        "policy_name": "Drive",
        "rnn_name": None,
        "policy": {},
        "rnn": {},
        "train": {
            "device": "cpu",
            "seed": 7,
            "learning_rate": 0.01,
            "use_rnn": False,
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
            "mode": "stacked_iid",
            "stack_len": STACK_LEN,
            "action_horizon": 2,
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
            "max_maps": -1,
            "save_best": True,
            "log_interval": 0,
        },
    }

    result = drive_module.train_bc_policy(args)
    metrics = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metrics["mode"] == "stacked_iid"
    assert metrics["action_horizon"] == 2
    assert "train_sequence_accuracy" in metrics["history"][0]


def test_bc_trainer_paired_offline_fits_iid_rejects_enabled_rnn(tmp_path):
    fit_map_names = ["map_00000.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)
    fit_export = _write_paired_fit_export(tmp_path, fit_map_names)

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
            "use_rnn": True,
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
            "mode": "iid",
            "fit_export": str(fit_export),
            "source_map_dir": str(source_dir),
            "fit_side": "car",
            "obs_key": "logged_obs_default",
            "output_dir": str(tmp_path / "checkpoints"),
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "num_workers": 0,
            "shard_shuffle_buffer": 1,
            "val_fraction": 0.0,
            "save_best": True,
            "log_interval": 0,
        },
    }

    with pytest.raises(ValueError, match="train.use_rnn=false"):
        drive_module.train_bc_policy(args)


def test_bc_trainer_paired_offline_fits_recurrent_rejects_missing_rnn(tmp_path, monkeypatch):
    fit_map_names = ["map_00000.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)
    fit_export = _write_paired_fit_export(tmp_path, fit_map_names)
    monkeypatch.setattr(drive_module, "_make_bc_training_env", lambda _env_cfg: _DummyBCEnv())

    args = {
        "package": "ocean",
        "env_name": "puffer_drive",
        "policy_name": "Drive",
        "rnn_name": None,
        "policy": {},
        "rnn": {},
        "train": {
            "device": "cpu",
            "seed": 7,
            "learning_rate": 0.01,
            "use_rnn": False,
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
            "mode": "recurrent",
            "fit_export": str(fit_export),
            "source_map_dir": str(source_dir),
            "fit_side": "car",
            "obs_key": "logged_obs_default",
            "output_dir": str(tmp_path / "checkpoints"),
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
            "save_best": True,
            "log_interval": 0,
        },
    }

    with pytest.raises(ValueError, match="requires a non-empty rnn_name"):
        drive_module.train_bc_policy(args)


def test_bc_trainer_paired_offline_fits_stacked_iid_rejects_enabled_rnn(tmp_path):
    fit_map_names = ["map_00000.bin"]
    source_dir = tmp_path / "source"
    _write_source_split(source_dir, fit_map_names)
    fit_export = _write_paired_fit_export(tmp_path, fit_map_names)

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
            "use_rnn": True,
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
            "mode": "stacked_iid",
            "stack_len": STACK_LEN,
            "fit_export": str(fit_export),
            "source_map_dir": str(source_dir),
            "fit_side": "car",
            "obs_key": "logged_obs_default",
            "output_dir": str(tmp_path / "checkpoints"),
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "num_workers": 0,
            "shard_shuffle_buffer": 1,
            "val_fraction": 0.0,
            "save_best": True,
            "log_interval": 0,
        },
    }

    with pytest.raises(ValueError, match="train.use_rnn=false"):
        drive_module.train_bc_policy(args)
