import json
from pathlib import Path

import gymnasium
import numpy as np
import pytest
import torch

from pufferlib.ocean.drive import drive as drive_module


DEFAULT_OBS_DIM = 4
EGO_DYNAMICS_OBS_DIM = 9
VELOCITY_XY_OBS_DIM = 10
MOTION_PREV_CONTROL_OBS_DIM = 10
MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_DIM = 22
TRAILER_OBS_DIM = 6
ACTION_SPACE = 91
STACK_LEN = 3


def _write_shard_dataset(dataset_dir, *, num_shards=2, rows_per_shard=6):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for shard_idx in range(num_shards):
        base = torch.arange(rows_per_shard * DEFAULT_OBS_DIM, dtype=torch.float32).reshape(rows_per_shard, DEFAULT_OBS_DIM)
        trailer = torch.arange(rows_per_shard * TRAILER_OBS_DIM, dtype=torch.float32).reshape(rows_per_shard, TRAILER_OBS_DIM)
        payload = {
            "obs": base + float(10 * shard_idx),
            "obs_default": base + float(100 + 10 * shard_idx),
            "obs_default_plus_ego_dynamics": torch.arange(
                rows_per_shard * EGO_DYNAMICS_OBS_DIM, dtype=torch.float32
            ).reshape(rows_per_shard, EGO_DYNAMICS_OBS_DIM)
            + float(150 + 10 * shard_idx),
            "obs_default_vxy_speed25": torch.arange(
                rows_per_shard * VELOCITY_XY_OBS_DIM, dtype=torch.float32
            ).reshape(rows_per_shard, VELOCITY_XY_OBS_DIM)
            + float(175 + 10 * shard_idx),
            "obs_default_motion_prev_control": torch.arange(
                rows_per_shard * MOTION_PREV_CONTROL_OBS_DIM, dtype=torch.float32
            ).reshape(rows_per_shard, MOTION_PREV_CONTROL_OBS_DIM)
            + float(190 + 10 * shard_idx),
            "obs_default_motion_prev_control_road_controls": torch.arange(
                rows_per_shard * MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_DIM, dtype=torch.float32
            ).reshape(rows_per_shard, MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_DIM)
            + float(220 + 10 * shard_idx),
            "obs_sdc_only_with_trailer": trailer + float(200 + 10 * shard_idx),
            "action": torch.arange(rows_per_shard, dtype=torch.int64) % ACTION_SPACE,
            "map_id": torch.full((rows_per_shard,), shard_idx, dtype=torch.int64),
            "timestep": torch.arange(rows_per_shard, dtype=torch.int64),
            "sequence_id": torch.full((rows_per_shard,), shard_idx, dtype=torch.int64),
            "sequence_row_index": torch.arange(rows_per_shard, dtype=torch.int64),
            "sequence_length": torch.full((rows_per_shard,), rows_per_shard, dtype=torch.int64),
            "trajectory_ref_accel": torch.linspace(-4.0, 4.0, rows_per_shard, dtype=torch.float32),
            "trajectory_ref_steer": torch.linspace(-1.0, 1.0, rows_per_shard, dtype=torch.float32),
        }
        torch.save(payload, dataset_dir / f"map_{shard_idx:03d}.pt")

    manifest = {
        "count": num_shards,
        "observation_keys": [
            "obs",
            "obs_default",
            "obs_default_plus_ego_dynamics",
            "obs_default_vxy_speed25",
            "obs_default_motion_prev_control",
            "obs_default_motion_prev_control_road_controls",
            "obs_sdc_only_with_trailer",
        ],
        "obs_dim": DEFAULT_OBS_DIM,
        "obs_default_dim": DEFAULT_OBS_DIM,
        "obs_default_plus_ego_dynamics_dim": EGO_DYNAMICS_OBS_DIM,
        "obs_default_vxy_speed25_dim": VELOCITY_XY_OBS_DIM,
        "obs_default_motion_prev_control_dim": MOTION_PREV_CONTROL_OBS_DIM,
        "obs_default_motion_prev_control_road_controls_dim": MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_DIM,
        "obs_sdc_only_with_trailer_dim": TRAILER_OBS_DIM,
    }
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class _DummyBCEnv:
    def __init__(self, obs_dim):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.MultiDiscrete([ACTION_SPACE])

    def close(self):
        pass


class _DummyContinuousBCEnv:
    def __init__(self, obs_dim):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def close(self):
        pass


class _TinyRecurrentPolicy(torch.nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.actor = torch.nn.Linear(obs_dim, ACTION_SPACE)

    def forward(self, obs, state=None):
        logits = self.actor(obs.reshape(-1, obs.shape[-1]))
        return logits, state


class _TinyTwoHeadFeedforwardPolicy(torch.nn.Module):
    def __init__(self, obs_dim, *, action_horizon=1):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.accel = torch.nn.Linear(obs_dim * STACK_LEN, 7 * self.action_horizon)
        self.steer = torch.nn.Linear(obs_dim * STACK_LEN, 13 * self.action_horizon)

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


class _TinyContinuousFeedforwardPolicy(torch.nn.Module):
    def __init__(self, obs_dim, *, action_horizon=1):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.loc = torch.nn.Linear(obs_dim * STACK_LEN, 2 * self.action_horizon)
        self.scale = torch.nn.Linear(obs_dim * STACK_LEN, 2 * self.action_horizon)

    def forward(self, obs, state=None):
        loc = self.loc(obs)
        scale = torch.nn.functional.softplus(self.scale(obs)) + 1e-4
        if self.action_horizon == 1:
            return torch.distributions.Normal(loc.reshape(-1, 2), scale.reshape(-1, 2)), state
        batch = obs.shape[0]
        return (
            torch.distributions.Normal(
                loc.reshape(batch, self.action_horizon, 2),
                scale.reshape(batch, self.action_horizon, 2),
            ),
            state,
        )


def _train_args(dataset_dir, *, obs_field="obs", observation_mode="default"):
    return {
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
            "max_controlled_agents": 1,
            "action_type": "discrete",
            "dynamics_model": "articulated",
            "extend_classic_action_space": False,
            "observation_mode": observation_mode,
            "init_mode": "create_all_valid",
            "control_mode": "control_sdc_only",
        },
        "bc_train": {
            "source_format": "shards",
            "mode": "recurrent",
            "dataset_dir": str(dataset_dir),
            "obs_field": obs_field,
            "output_dir": str(Path(dataset_dir) / f"checkpoints_{obs_field}"),
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


def test_shard_loader_defaults_to_obs_field_obs(tmp_path):
    dataset_dir = tmp_path / "bc_dataset"
    _write_shard_dataset(dataset_dir, num_shards=1)
    payload = torch.load(dataset_dir / "map_000.pt", map_location="cpu")

    dataset = drive_module._FlatBCDataset(
        [str(dataset_dir / "map_000.pt")],
        obs_dim=DEFAULT_OBS_DIM,
        action_space_size=ACTION_SPACE,
        shuffle=False,
        seed=0,
    )
    obs, action = next(iter(dataset))
    assert obs.shape == (DEFAULT_OBS_DIM,)
    assert torch.allclose(obs, payload["obs"][-1].float())
    assert int(action.item()) == int(payload["action"][-1].item())


def test_shard_loader_supports_obs_default_and_trailer_fields(tmp_path):
    dataset_dir = tmp_path / "bc_dataset"
    _write_shard_dataset(dataset_dir, num_shards=1)
    shard_path = dataset_dir / "map_000.pt"
    payload = torch.load(shard_path, map_location="cpu")

    default_dataset = drive_module._FlatBCDataset(
        [str(shard_path)],
        obs_dim=DEFAULT_OBS_DIM,
        action_space_size=ACTION_SPACE,
        shuffle=False,
        seed=0,
        obs_field="obs_default",
    )
    default_obs, _ = next(iter(default_dataset))
    assert default_obs.shape == (DEFAULT_OBS_DIM,)
    assert torch.allclose(default_obs, payload["obs_default"][-1].float())

    trailer_dataset = drive_module._SequenceBCDataset(
        [str(shard_path)],
        obs_dim=TRAILER_OBS_DIM,
        action_space_size=ACTION_SPACE,
        seq_len=4,
        stride=2,
        shuffle=False,
        seed=0,
        obs_field="obs_sdc_only_with_trailer",
        require_embedded=False,
    )
    obs_batch, action_batch, mask_batch = next(iter(trailer_dataset))
    assert obs_batch.shape == (4, TRAILER_OBS_DIM)
    expected_rows = payload["obs_sdc_only_with_trailer"][4:6].float()
    assert torch.allclose(obs_batch[:2], expected_rows)
    assert action_batch.shape == (4,)
    assert mask_batch.shape == (4,)

    ego_dynamics_dataset = drive_module._StackedIIDBCDataset(
        [str(shard_path)],
        obs_dim=EGO_DYNAMICS_OBS_DIM,
        action_space_size=ACTION_SPACE,
        stack_len=STACK_LEN,
        action_horizon=1,
        shuffle=False,
        seed=0,
        obs_field="obs_default_plus_ego_dynamics",
        extend_classic_action_space=False,
    )
    stacked_obs, accel_idx, steer_idx = next(iter(ego_dynamics_dataset))
    assert stacked_obs.shape == (EGO_DYNAMICS_OBS_DIM * STACK_LEN,)
    expected_rows = torch.flip(payload["obs_default_plus_ego_dynamics"][3:6].float(), dims=(0,)).reshape(-1)
    assert torch.allclose(stacked_obs, expected_rows)
    assert int(accel_idx.item()) >= 0
    assert int(steer_idx.item()) >= 0

    velocity_xy_dataset = drive_module._StackedIIDBCDataset(
        [str(shard_path)],
        obs_dim=VELOCITY_XY_OBS_DIM,
        action_space_size=ACTION_SPACE,
        stack_len=STACK_LEN,
        action_horizon=1,
        shuffle=False,
        seed=0,
        obs_field="obs_default_vxy_speed25",
        extend_classic_action_space=False,
    )
    stacked_velocity_xy_obs, accel_idx, steer_idx = next(iter(velocity_xy_dataset))
    assert stacked_velocity_xy_obs.shape == (VELOCITY_XY_OBS_DIM * STACK_LEN,)
    expected_rows = torch.flip(payload["obs_default_vxy_speed25"][3:6].float(), dims=(0,)).reshape(-1)
    assert torch.allclose(stacked_velocity_xy_obs, expected_rows)
    assert int(accel_idx.item()) >= 0
    assert int(steer_idx.item()) >= 0

    motion_prev_control_dataset = drive_module._StackedIIDBCDataset(
        [str(shard_path)],
        obs_dim=MOTION_PREV_CONTROL_OBS_DIM,
        action_space_size=ACTION_SPACE,
        stack_len=STACK_LEN,
        action_horizon=1,
        shuffle=False,
        seed=0,
        obs_field="obs_default_motion_prev_control",
        extend_classic_action_space=False,
    )
    stacked_motion_prev_control_obs, accel_idx, steer_idx = next(iter(motion_prev_control_dataset))
    assert stacked_motion_prev_control_obs.shape == (MOTION_PREV_CONTROL_OBS_DIM * STACK_LEN,)
    expected_rows = torch.flip(payload["obs_default_motion_prev_control"][3:6].float(), dims=(0,)).reshape(-1)
    assert torch.allclose(stacked_motion_prev_control_obs, expected_rows)
    assert int(accel_idx.item()) >= 0
    assert int(steer_idx.item()) >= 0

    motion_prev_control_road_controls_dataset = drive_module._StackedIIDBCDataset(
        [str(shard_path)],
        obs_dim=MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_DIM,
        action_space_size=ACTION_SPACE,
        stack_len=STACK_LEN,
        action_horizon=1,
        shuffle=False,
        seed=0,
        obs_field="obs_default_motion_prev_control_road_controls",
        extend_classic_action_space=False,
    )
    stacked_motion_prev_control_road_controls_obs, accel_idx, steer_idx = next(iter(motion_prev_control_road_controls_dataset))
    assert stacked_motion_prev_control_road_controls_obs.shape == (MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_DIM * STACK_LEN,)
    expected_rows = torch.flip(payload["obs_default_motion_prev_control_road_controls"][3:6].float(), dims=(0,)).reshape(-1)
    assert torch.allclose(stacked_motion_prev_control_road_controls_obs, expected_rows)
    assert int(accel_idx.item()) >= 0
    assert int(steer_idx.item()) >= 0


def test_shard_loader_rejects_missing_obs_field(tmp_path):
    dataset_dir = tmp_path / "bc_dataset"
    _write_shard_dataset(dataset_dir, num_shards=1)

    dataset = drive_module._FlatBCDataset(
        [str(dataset_dir / "map_000.pt")],
        obs_dim=DEFAULT_OBS_DIM,
        action_space_size=ACTION_SPACE,
        shuffle=False,
        seed=0,
        obs_field="obs_missing",
    )

    with pytest.raises(ValueError, match="does not contain requested obs_field='obs_missing'"):
        list(dataset)


def test_shard_loader_rejects_selected_obs_width_mismatch(tmp_path):
    dataset_dir = tmp_path / "bc_dataset"
    _write_shard_dataset(dataset_dir, num_shards=1)
    payload = torch.load(dataset_dir / "map_000.pt", map_location="cpu")

    with pytest.raises(ValueError, match="observation width mismatch"):
        drive_module._validate_bc_shard_payload(
            payload,
            DEFAULT_OBS_DIM,
            ACTION_SPACE,
            str(dataset_dir / "map_000.pt"),
            obs_field="obs_sdc_only_with_trailer",
        )


def test_bc_trainer_shards_supports_obs_field_default(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "bc_dataset"
    _write_shard_dataset(dataset_dir, num_shards=2)

    monkeypatch.setattr(
        drive_module,
        "_make_bc_training_env",
        lambda env_cfg: _DummyBCEnv(DEFAULT_OBS_DIM),
    )
    monkeypatch.setattr(
        drive_module,
        "_build_bc_policy",
        lambda _args, _env, device: _TinyRecurrentPolicy(DEFAULT_OBS_DIM).to(device),
    )

    result = drive_module.train_bc_policy(_train_args(dataset_dir, obs_field="obs", observation_mode="default"))
    metrics_path = Path(result["metadata_path"])
    assert Path(result["latest_path"]).is_file()
    assert Path(result["best_path"]).is_file()
    assert metrics_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["source_format"] == "shards"
    assert metrics["bc_train"]["obs_field"] == "obs"
    assert metrics["observation_dim"] == DEFAULT_OBS_DIM
    assert metrics["history"][0]["train_samples"] > 0
    assert metrics["history"][0]["val_samples"] > 0


def test_bc_trainer_shards_supports_obs_field_trailer(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "bc_dataset"
    _write_shard_dataset(dataset_dir, num_shards=2)

    monkeypatch.setattr(
        drive_module,
        "_make_bc_training_env",
        lambda env_cfg: _DummyBCEnv(TRAILER_OBS_DIM),
    )
    monkeypatch.setattr(
        drive_module,
        "_build_bc_policy",
        lambda _args, _env, device: _TinyRecurrentPolicy(TRAILER_OBS_DIM).to(device),
    )

    result = drive_module.train_bc_policy(
        _train_args(
            dataset_dir,
            obs_field="obs_sdc_only_with_trailer",
            observation_mode="sdc_only_with_trailer",
        )
    )
    metrics_path = Path(result["metadata_path"])
    assert Path(result["latest_path"]).is_file()
    assert Path(result["best_path"]).is_file()
    assert metrics_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["source_format"] == "shards"
    assert metrics["bc_train"]["obs_field"] == "obs_sdc_only_with_trailer"
    assert metrics["observation_dim"] == TRAILER_OBS_DIM
    assert metrics["history"][0]["train_samples"] > 0
    assert metrics["history"][0]["val_samples"] > 0


def test_bc_trainer_shards_supports_stacked_iid_enriched_ego_obs(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "bc_dataset"
    _write_shard_dataset(dataset_dir, num_shards=2)

    monkeypatch.setattr(
        drive_module,
        "_make_bc_training_env",
        lambda env_cfg: _DummyBCEnv(DEFAULT_OBS_DIM),
    )
    monkeypatch.setattr(
        drive_module,
        "_build_bc_policy",
        lambda _args, _env, device: _TinyTwoHeadFeedforwardPolicy(EGO_DYNAMICS_OBS_DIM).to(device),
    )

    args = _train_args(dataset_dir, obs_field="obs_default_plus_ego_dynamics", observation_mode="default")
    args["rnn_name"] = "None"
    args["train"]["use_rnn"] = False
    args["bc_train"]["mode"] = "stacked_iid"
    args["bc_train"]["stack_len"] = STACK_LEN

    result = drive_module.train_bc_policy(args)
    metrics = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metrics["mode"] == "stacked_iid"
    assert metrics["bc_train"]["obs_field"] == "obs_default_plus_ego_dynamics"
    assert metrics["observation_dim"] == EGO_DYNAMICS_OBS_DIM
    assert metrics["model_observation_dim"] == EGO_DYNAMICS_OBS_DIM * STACK_LEN
    assert "train_accel_accuracy" in metrics["history"][0]
    assert "val_steer_accuracy" in metrics["history"][0]


def test_stacked_iid_loader_and_trainer_support_future_action_horizon(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "bc_dataset"
    _write_shard_dataset(dataset_dir, num_shards=1, rows_per_shard=8)
    shard_path = dataset_dir / "map_000.pt"
    payload = torch.load(shard_path, map_location="cpu")

    dataset = drive_module._StackedIIDBCDataset(
        [str(shard_path)],
        obs_dim=DEFAULT_OBS_DIM,
        action_space_size=ACTION_SPACE,
        stack_len=STACK_LEN,
        action_horizon=2,
        shuffle=False,
        seed=0,
        obs_field="obs_default",
        extend_classic_action_space=False,
    )
    stacked_obs, accel_idx, steer_idx = next(iter(dataset))
    assert stacked_obs.shape == (DEFAULT_OBS_DIM * STACK_LEN,)
    assert accel_idx.shape == (2,)
    assert steer_idx.shape == (2,)
    expected_rows = torch.flip(payload["obs_default"][4:7].float(), dims=(0,)).reshape(-1)
    assert torch.allclose(stacked_obs, expected_rows)

    monkeypatch.setattr(
        drive_module,
        "_make_bc_training_env",
        lambda env_cfg: _DummyBCEnv(DEFAULT_OBS_DIM),
    )
    monkeypatch.setattr(
        drive_module,
        "_build_bc_policy",
        lambda _args, _env, device: _TinyTwoHeadFeedforwardPolicy(DEFAULT_OBS_DIM, action_horizon=2).to(device),
    )

    args = _train_args(dataset_dir, obs_field="obs_default", observation_mode="default")
    args["rnn_name"] = "None"
    args["train"]["use_rnn"] = False
    args["bc_train"]["mode"] = "stacked_iid"
    args["bc_train"]["stack_len"] = STACK_LEN
    args["bc_train"]["action_horizon"] = 2

    result = drive_module.train_bc_policy(args)
    metrics = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metrics["mode"] == "stacked_iid"
    assert metrics["action_horizon"] == 2
    assert metrics["history"][0]["train_sequence_accuracy"] >= 0.0
    assert metrics["history"][0]["val_sequence_accuracy"] >= 0.0


def test_stacked_iid_loader_and_trainer_support_continuous_targets(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "bc_dataset"
    _write_shard_dataset(dataset_dir, num_shards=1, rows_per_shard=8)
    shard_path = dataset_dir / "map_000.pt"
    payload = torch.load(shard_path, map_location="cpu")

    dataset = drive_module._ContinuousStackedIIDBCDataset(
        [str(shard_path)],
        obs_dim=MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_DIM,
        action_space_size=ACTION_SPACE,
        stack_len=STACK_LEN,
        action_horizon=1,
        shuffle=False,
        seed=0,
        obs_field="obs_default_motion_prev_control_road_controls",
        continuous_accel_field="trajectory_ref_accel",
        continuous_steer_field="trajectory_ref_steer",
        accel_scale=4.0,
        steer_scale=1.0,
    )
    stacked_obs, target = next(iter(dataset))
    assert stacked_obs.shape == (MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_DIM * STACK_LEN,)
    assert target.shape == (2,)
    expected_rows = torch.flip(payload["obs_default_motion_prev_control_road_controls"][5:8].float(), dims=(0,)).reshape(-1)
    assert torch.allclose(stacked_obs, expected_rows)
    assert torch.allclose(
        target,
        torch.tensor([payload["trajectory_ref_accel"][7] / 4.0, payload["trajectory_ref_steer"][7]], dtype=torch.float32),
    )

    monkeypatch.setattr(
        drive_module,
        "_make_bc_training_env",
        lambda env_cfg: _DummyContinuousBCEnv(DEFAULT_OBS_DIM),
    )
    monkeypatch.setattr(
        drive_module,
        "_build_bc_policy",
        lambda _args, _env, device: _TinyContinuousFeedforwardPolicy(MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_DIM).to(device),
    )

    args = _train_args(
        dataset_dir,
        obs_field="obs_default_motion_prev_control_road_controls",
        observation_mode="default",
    )
    args["rnn_name"] = "None"
    args["train"]["use_rnn"] = False
    args["env"]["action_type"] = "continuous"
    args["bc_train"]["mode"] = "stacked_iid"
    args["bc_train"]["stack_len"] = STACK_LEN
    args["bc_train"]["continuous_accel_field"] = "trajectory_ref_accel"
    args["bc_train"]["continuous_steer_field"] = "trajectory_ref_steer"

    result = drive_module.train_bc_policy(args)
    metrics = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metrics["mode"] == "stacked_iid"
    assert metrics["bc_train"]["obs_field"] == "obs_default_motion_prev_control_road_controls"
    assert metrics["observation_dim"] == MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_DIM
    assert metrics["model_observation_dim"] == MOTION_PREV_CONTROL_ROAD_CONTROLS_OBS_DIM * STACK_LEN
    assert "train_mae" in metrics["history"][0]
    assert "val_accel_mae" in metrics["history"][0]
