from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

if "psutil" not in sys.modules:
    sys.modules["psutil"] = types.SimpleNamespace()
if "pyro" not in sys.modules:
    pyro_module = types.ModuleType("pyro")
    pyro_contrib = types.ModuleType("pyro.contrib")
    pyro_gp = types.ModuleType("pyro.contrib.gp")
    pyro_contrib.gp = pyro_gp
    pyro_module.contrib = pyro_contrib
    sys.modules["pyro"] = pyro_module
    sys.modules["pyro.contrib"] = pyro_contrib
    sys.modules["pyro.contrib.gp"] = pyro_gp
if "rich" not in sys.modules:
    rich_module = types.ModuleType("rich")
    rich_traceback = types.ModuleType("rich.traceback")
    rich_traceback.install = lambda **kwargs: None
    rich_table = types.ModuleType("rich.table")
    rich_table.Table = object
    rich_console = types.ModuleType("rich.console")
    rich_console.Console = object
    rich_module.traceback = rich_traceback
    sys.modules["rich"] = rich_module
    sys.modules["rich.traceback"] = rich_traceback
    sys.modules["rich.table"] = rich_table
    sys.modules["rich.console"] = rich_console
if "rich_argparse" not in sys.modules:
    rich_argparse = types.ModuleType("rich_argparse")
    rich_argparse.RichHelpFormatter = object
    sys.modules["rich_argparse"] = rich_argparse

from pufferlib import pufferl
from pufferlib.ocean.drive.drive import Drive, save_map_binary
from pufferlib.ocean.torch import Drive as DrivePolicy


def _linear_traj(x0, y0, dx, dy, heading=0.0, length=91):
    return {
        "position": [{"x": float(x0 + dx * t), "y": float(y0 + dy * t), "z": 0.0} for t in range(length)],
        "velocity": [{"x": float(dx / 0.1), "y": float(dy / 0.1), "z": 0.0} for _ in range(length)],
        "heading": [float(heading) for _ in range(length)],
        "valid": [1 for _ in range(length)],
    }


def _write_eval_map(map_dir: Path):
    scenario = {
        "metadata": {"sdc_track_index": 0, "tracks_to_predict": [{"track_index": 0}]},
        "objects": [
            {
                "id": 100,
                "type": "vehicle",
                "length": 4.5,
                "width": 1.9,
                "height": 1.6,
                **_linear_traj(0.0, 0.0, 1.0, 0.0),
                "goalPosition": {"x": 100.0, "y": 0.0, "z": 0.0},
            }
        ],
        "roads": [
            {
                "id": 200,
                "type": "lane",
                "geometry": [{"x": -10.0, "y": 0.0, "z": 0.0}, {"x": 90.0, "y": 0.0, "z": 0.0}],
                "width": 3.5,
                "length": 100.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            }
        ],
    }
    save_map_binary(scenario, str(map_dir / "map_000.bin"), unique_map_id=11)


def _make_env(
    map_dir: Path,
    *,
    action_type: str = "trajectory",
    observation_mode: str = "trajectory_history_32",
) -> Drive:
    return Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        resample_frequency=0,
        action_type=action_type,
        observation_mode=observation_mode,
    )


def _make_args(load_path, *, action_type="trajectory", observation_mode="trajectory_history_32"):
    return {
        "package": "ocean",
        "policy_name": "Drive",
        "rnn_name": None,
        "policy": {"input_size": 64, "hidden_size": 256},
        "train": {"device": "cpu"},
        "env": {
            "action_type": action_type,
            "observation_mode": observation_mode,
            "dynamics_model": "classic",
        },
        "load_id": None,
        "load_model_path": str(load_path),
        "wandb": False,
        "neptune": False,
    }


def _make_bc_payload(policy: DrivePolicy):
    return {
        "model_state_dict": policy.state_dict(),
        "config": {
            "env": {
                "action_type": "trajectory",
                "observation_mode": "trajectory_history_32",
                "dynamics_model": "classic",
            },
            "policy": {"input_size": 64, "hidden_size": 256},
        },
    }


def test_load_policy_accepts_raw_state_dict_checkpoint(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_eval_map(map_dir)

    env = _make_env(map_dir)
    try:
        checkpoint_policy = DrivePolicy(env, input_size=64, hidden_size=256)
        checkpoint_path = tmp_path / "raw_state_dict.pt"
        torch.save(checkpoint_policy.state_dict(), checkpoint_path)

        vecenv = SimpleNamespace(driver_env=env)
        loaded = pufferl.load_policy(_make_args(checkpoint_path), vecenv, env_name="puffer_drive")

        for key, value in checkpoint_policy.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[key])
    finally:
        env.close()


def test_load_policy_accepts_bc_payload_checkpoint(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_eval_map(map_dir)

    env = _make_env(map_dir)
    try:
        checkpoint_policy = DrivePolicy(env, input_size=64, hidden_size=256)
        checkpoint_path = tmp_path / "bc_payload.pt"
        torch.save(_make_bc_payload(checkpoint_policy), checkpoint_path)

        vecenv = SimpleNamespace(driver_env=env)
        loaded = pufferl.load_policy(_make_args(checkpoint_path), vecenv, env_name="puffer_drive")

        for key, value in checkpoint_policy.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[key])
    finally:
        env.close()


def test_load_policy_rejects_unknown_checkpoint_payload(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_eval_map(map_dir)

    env = _make_env(map_dir)
    try:
        checkpoint_path = tmp_path / "unknown_payload.pt"
        torch.save({"foo": torch.tensor([1.0])}, checkpoint_path)

        vecenv = SimpleNamespace(driver_env=env)
        with pytest.raises(ValueError, match="Unsupported checkpoint payload"):
            pufferl.load_policy(_make_args(checkpoint_path), vecenv, env_name="puffer_drive")
    finally:
        env.close()


def test_load_policy_rejects_bc_payload_with_mismatched_config(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_eval_map(map_dir)

    checkpoint_env = _make_env(map_dir)
    live_env = _make_env(map_dir, action_type="continuous", observation_mode="default")
    try:
        checkpoint_policy = DrivePolicy(checkpoint_env, input_size=64, hidden_size=256)
        checkpoint_path = tmp_path / "bc_payload.pt"
        torch.save(_make_bc_payload(checkpoint_policy), checkpoint_path)

        vecenv = SimpleNamespace(driver_env=live_env)
        with pytest.raises(ValueError, match="BC checkpoint is incompatible"):
            pufferl.load_policy(
                _make_args(checkpoint_path, action_type="continuous", observation_mode="default"),
                vecenv,
                env_name="puffer_drive",
            )
    finally:
        checkpoint_env.close()
        live_env.close()


def test_bc_checkpoint_eval_rollout_smoke(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_eval_map(map_dir)

    env = _make_env(map_dir)
    try:
        checkpoint_policy = DrivePolicy(env, input_size=64, hidden_size=256)
        checkpoint_path = tmp_path / "bc_payload.pt"
        torch.save(_make_bc_payload(checkpoint_policy), checkpoint_path)

        vecenv = SimpleNamespace(driver_env=env)
        policy = pufferl.load_policy(_make_args(checkpoint_path), vecenv, env_name="puffer_drive")
        policy.eval()

        observations, _ = env.reset(seed=0)
        with torch.no_grad():
            action_dist, _value = policy(torch.as_tensor(observations, dtype=torch.float32))
            actions = action_dist.mean.cpu().numpy()

        next_obs, rewards, terminals, truncations, info = env.step(actions)

        assert next_obs.shape == observations.shape
        assert rewards.shape == (1,)
        assert terminals.shape == (1,)
        assert truncations.shape == (1,)
        assert isinstance(info, list)
    finally:
        env.close()
