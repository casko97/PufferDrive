from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pufferlib import pufferl
import pufferlib.pytorch
from pufferlib.ocean.drive.drive import Drive, save_map_binary
from pufferlib.ocean.drive.trajectory_bc_viz import _TRAJECTORY_HISTORY_FEATURES, _TRAJECTORY_HORIZON, _base_obs_dim
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
    save_map_binary(scenario, str(map_dir / "map_000.bin"), unique_map_id=17)


def _make_env(map_dir: Path) -> Drive:
    return Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        resample_frequency=0,
        action_type="trajectory",
        observation_mode="trajectory_history_32",
    )


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


def _make_args(load_path, mode):
    return {
        "package": "ocean",
        "policy_name": "Drive",
        "rnn_name": None,
        "policy": {
            "input_size": 64,
            "hidden_size": 256,
            "initial_std_bias": -2.0,
            "min_action_std": 0.01,
            "max_action_std": 0.5,
        },
        "train": {
            "device": "cpu",
            "trajectory_training_mode": mode,
            "trajectory_reinit_value_head": True,
            "trajectory_freeze_backbone": True,
            "trajectory_freeze_actor_head": mode == "critic_warmstart",
            "trajectory_freeze_value_head": False,
        },
        "env": {
            "action_type": "trajectory",
            "observation_mode": "trajectory_history_32",
            "dynamics_model": "classic",
        },
        "load_id": None,
        "load_model_path": str(load_path),
        "wandb": False,
        "neptune": False,
    }


def _weight_snapshot(module):
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def test_critic_warmstart_reinitializes_value_head_and_freezes_actor(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_eval_map(map_dir)

    env = _make_env(map_dir)
    try:
        checkpoint_policy = DrivePolicy(env, input_size=64, hidden_size=256)
        with torch.no_grad():
            checkpoint_policy.value_fn.weight.fill_(3.0)
            checkpoint_policy.value_fn.bias.fill_(1.5)
        actor_before = _weight_snapshot(checkpoint_policy.actor)
        checkpoint_path = tmp_path / "bc_payload.pt"
        torch.save(_make_bc_payload(checkpoint_policy), checkpoint_path)

        loaded = pufferl.load_policy(_make_args(checkpoint_path, "critic_warmstart"), SimpleNamespace(driver_env=env))
        base_policy = loaded

        assert torch.allclose(base_policy.actor.weight, actor_before["weight"])
        assert torch.allclose(base_policy.actor.bias, actor_before["bias"])
        assert not torch.allclose(base_policy.value_fn.weight, torch.full_like(base_policy.value_fn.weight, 3.0))
        assert not torch.allclose(base_policy.value_fn.bias, torch.full_like(base_policy.value_fn.bias, 1.5))

        for name, parameter in base_policy.named_parameters():
            if name.startswith("actor."):
                assert parameter.requires_grad is False
            elif name.startswith("value_fn."):
                assert parameter.requires_grad is True
            else:
                assert parameter.requires_grad is False
    finally:
        env.close()


def test_ppo_finetune_keeps_actor_trainable_and_backbone_frozen(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_eval_map(map_dir)

    env = _make_env(map_dir)
    try:
        checkpoint_policy = DrivePolicy(env, input_size=64, hidden_size=256)
        checkpoint_path = tmp_path / "bc_payload.pt"
        torch.save(_make_bc_payload(checkpoint_policy), checkpoint_path)

        loaded = pufferl.load_policy(_make_args(checkpoint_path, "ppo_finetune"), SimpleNamespace(driver_env=env))

        for name, parameter in loaded.named_parameters():
            if name.startswith("actor.") or name.startswith("value_fn."):
                assert parameter.requires_grad is True
            else:
                assert parameter.requires_grad is False
    finally:
        env.close()


def test_trajectory_policy_std_controls_are_applied(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_eval_map(map_dir)

    env = _make_env(map_dir)
    try:
        policy = DrivePolicy(
            env,
            input_size=64,
            hidden_size=256,
            initial_std_bias=-8.0,
            min_action_std=0.05,
            max_action_std=0.2,
        )
        hidden = torch.zeros((1, policy.hidden_size), dtype=torch.float32)
        action_dist, _ = policy.decode_actions(hidden)

        assert torch.all(action_dist.scale >= 0.05)
        assert torch.all(action_dist.scale <= 0.2)
    finally:
        env.close()


def test_trajectory_policy_uses_squashed_normal_distribution(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_eval_map(map_dir)

    trajectory_env = _make_env(map_dir)
    continuous_env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        episode_length=91,
        init_steps=0,
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        resample_frequency=0,
        action_type="continuous",
        observation_mode="default",
    )
    try:
        trajectory_policy = DrivePolicy(
            trajectory_env,
            input_size=64,
            hidden_size=256,
            initial_std_bias=-2.0,
            min_action_std=0.01,
            max_action_std=0.2,
        )
        continuous_policy = DrivePolicy(
            continuous_env,
            input_size=64,
            hidden_size=256,
            initial_std_bias=-2.0,
            min_action_std=0.01,
            max_action_std=0.2,
        )

        trajectory_hidden = torch.zeros((1, trajectory_policy.hidden_size), dtype=torch.float32)
        trajectory_dist, _ = trajectory_policy.decode_actions(trajectory_hidden)
        assert getattr(trajectory_dist, "is_squashed_normal", False) is True
        assert isinstance(trajectory_dist, pufferlib.pytorch.SquashedNormal)

        continuous_hidden = torch.zeros((1, continuous_policy.hidden_size), dtype=torch.float32)
        continuous_dist, _ = continuous_policy.decode_actions(continuous_hidden)
        assert isinstance(continuous_dist, torch.distributions.Normal)
    finally:
        trajectory_env.close()
        continuous_env.close()


def test_live_trajectory_rollout_keeps_ego_history_after_first_step(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_eval_map(map_dir)

    env = _make_env(map_dir)
    try:
        checkpoint_policy = DrivePolicy(env, input_size=64, hidden_size=256)
        checkpoint_path = tmp_path / "bc_payload.pt"
        torch.save(_make_bc_payload(checkpoint_policy), checkpoint_path)

        loaded = pufferl.load_policy(_make_args(checkpoint_path, "none"), SimpleNamespace(driver_env=env))
        loaded.eval()

        observations, _ = env.reset(seed=0)
        with torch.no_grad():
            action_dist, _value = loaded(torch.as_tensor(observations, dtype=torch.float32))
            actions = action_dist.mean.cpu().numpy()

        next_obs, _rewards, _terminals, _truncations, _info = env.step(actions)
        history_start = _base_obs_dim()
        history_end = history_start + (_TRAJECTORY_HORIZON * _TRAJECTORY_HISTORY_FEATURES)
        ego_history = next_obs[0, history_start:history_end].reshape(_TRAJECTORY_HORIZON, _TRAJECTORY_HISTORY_FEATURES)

        assert float(ego_history[0, 5]) == 1.0
        assert float(ego_history[1, 5]) == 1.0
    finally:
        env.close()
