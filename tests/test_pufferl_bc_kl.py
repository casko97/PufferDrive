import contextlib

import pytest
import torch

import pufferlib
import pufferlib.pufferl as pufferl


def test_bc_teacher_kl_identical_logits_is_zero():
    logits = torch.tensor([[1.0, -0.5, 0.25], [0.0, 0.0, 2.0]])

    kl, entropy, count = pufferl._bc_teacher_kl_from_logits(logits, logits, temperature=1.0)

    assert count == 2
    assert kl.item() == pytest.approx(0.0, abs=1e-7)
    assert entropy.item() > 0.0


def test_bc_teacher_kl_different_logits_is_positive():
    student = torch.tensor([[2.0, 0.0, -1.0]])
    teacher = torch.tensor([[-1.0, 0.0, 2.0]])

    kl, _entropy, count = pufferl._bc_teacher_kl_from_logits(student, teacher, temperature=1.0)

    assert count == 1
    assert kl.item() > 0.0


def test_bc_teacher_kl_identical_normals_is_zero():
    loc = torch.tensor([[0.0, 0.25], [-0.5, 1.0]])
    scale = torch.tensor([[0.5, 0.75], [1.25, 0.9]])
    dist = torch.distributions.Normal(loc, scale)

    kl, entropy, count = pufferl._bc_teacher_kl_from_normals(dist, dist)

    assert count == 2
    assert kl.item() == pytest.approx(0.0, abs=1e-7)
    assert entropy.item() > 0.0


def test_bc_kl_coef_anneals_linearly():
    config = pufferl._resolve_bc_kl_config(
        {
            "enabled": True,
            "model_path": "teacher.pt",
            "coef": 0.01,
            "final_coef": 0.0,
            "anneal_fraction": 0.5,
            "temperature": 1.0,
        }
    )

    assert pufferl._bc_kl_coef_at_step(config, 0, 1000) == pytest.approx(0.01)
    assert pufferl._bc_kl_coef_at_step(config, 250, 1000) == pytest.approx(0.005)
    assert pufferl._bc_kl_coef_at_step(config, 500, 1000) == pytest.approx(0.0)
    assert pufferl._bc_kl_coef_at_step(config, 1000, 1000) == pytest.approx(0.0)


def test_bc_kl_rejects_unsupported_action_logits():
    multi_head = (torch.zeros(2, 3), torch.zeros(2, 4))
    continuous = torch.distributions.Normal(torch.zeros(2, 1), torch.ones(2, 1))

    with pytest.raises(pufferlib.APIUsageError, match="single-head"):
        pufferl._single_discrete_logits(multi_head, context="bc_kl")
    with pytest.raises(pufferlib.APIUsageError, match="discrete"):
        pufferl._single_discrete_logits(continuous, context="bc_kl")


def test_load_bc_kl_reference_policy_loads_checkpoint_path_and_freezes(monkeypatch):
    teacher = torch.nn.Linear(4, 3)
    captured = {}

    def _fake_load_policy(args, vecenv, env_name):
        captured["load_model_path"] = args["load_model_path"]
        captured["load_id"] = args["load_id"]
        captured["env_name"] = env_name
        return teacher

    monkeypatch.setattr(pufferl, "load_policy", _fake_load_policy)

    ref = pufferl.load_bc_kl_reference_policy(
        {
            "load_model_path": "ppo_init.pt",
            "load_id": "ppo-run",
            "bc_kl": {
                "enabled": True,
                "model_path": "bc_teacher.pt",
                "coef": 0.01,
                "final_coef": 0.0,
                "anneal_fraction": 1.0,
                "temperature": 1.0,
            },
        },
        vecenv=None,
        env_name="puffer_drive",
    )

    assert ref is teacher
    assert captured == {
        "load_model_path": "bc_teacher.pt",
        "load_id": None,
        "env_name": "puffer_drive",
    }
    assert not teacher.training
    assert not any(param.requires_grad for param in teacher.parameters())


class _TinyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(5, 3)

    def forward(self, obs, state=None):
        flat_obs = obs.reshape(-1, obs.shape[-1])
        return (self.actor(flat_obs),), torch.zeros(flat_obs.shape[0], device=flat_obs.device)


class _TinyContinuousTeacher(torch.nn.Module):
    def forward(self, obs, state=None):
        flat_obs = obs.reshape(-1, obs.shape[-1])
        loc = flat_obs[:, :2]
        scale = torch.full_like(loc, 0.5)
        return torch.distributions.Normal(loc, scale), torch.zeros(flat_obs.shape[0], device=flat_obs.device)


class _MaskedContinuousTeacher:
    def __init__(self, valid_mask, teacher_dist):
        self.valid_mask = valid_mask
        self.teacher_dist = teacher_dist

    def forward_bc_kl(self, observations):
        del observations
        return self.teacher_dist, self.valid_mask


def test_pufferl_bc_kl_helper_uses_frozen_teacher_without_teacher_grads():
    teacher = pufferl._freeze_reference_policy(_TinyTeacher())
    trainer = pufferl.PuffeRL.__new__(pufferl.PuffeRL)
    trainer.bc_kl_policy = teacher
    trainer.global_step = 25
    trainer.config = {"total_timesteps": 100}
    trainer.bc_kl_config = pufferl._resolve_bc_kl_config(
        {
            "enabled": True,
            "model_path": "bc_teacher.pt",
            "coef": 0.01,
            "final_coef": 0.0,
            "anneal_fraction": 1.0,
            "temperature": 1.0,
        }
    )
    trainer.amp_context = contextlib.nullcontext()

    mb_obs = torch.randn(2, 4, 5)
    student_logits = torch.zeros(8, 3, requires_grad=True)

    bc_kl_loss, teacher_entropy, extra = trainer._compute_bc_kl_loss((student_logits,), mb_obs)
    metrics = trainer._compute_bc_kl_term((student_logits,), mb_obs)
    metrics["bc_kl_weighted"].backward()

    assert extra == {}
    assert bc_kl_loss.item() >= 0.0
    assert teacher_entropy.item() > 0.0
    assert set(metrics) == {"bc_kl", "bc_kl_weighted", "bc_kl_coef", "bc_teacher_entropy"}
    assert metrics["bc_kl_coef"] == pytest.approx(0.0075)
    assert metrics["bc_kl_weighted"].item() == pytest.approx(
        metrics["bc_kl"].item() * metrics["bc_kl_coef"],
        rel=1e-6,
    )
    assert student_logits.grad is not None
    assert all(param.grad is None for param in teacher.parameters())


def test_pufferl_bc_kl_term_disabled_is_empty():
    trainer = pufferl.PuffeRL.__new__(pufferl.PuffeRL)
    trainer.bc_kl_policy = None

    assert trainer._compute_bc_kl_term(torch.zeros(2, 3), torch.zeros(2, 5)) == {}


def test_pufferl_continuous_bc_kl_helper_uses_matching_normal_teacher():
    teacher = pufferl._freeze_reference_policy(_TinyContinuousTeacher())
    trainer = pufferl.PuffeRL.__new__(pufferl.PuffeRL)
    trainer.bc_kl_policy = teacher
    trainer.global_step = 20
    trainer.config = {"total_timesteps": 100}
    trainer.bc_kl_config = pufferl._resolve_bc_kl_config(
        {
            "enabled": True,
            "model_path": "bc_teacher.pt",
            "teacher_kind": "ppo",
            "coef": 0.01,
            "final_coef": 0.0,
            "anneal_fraction": 1.0,
            "temperature": 1.0,
        }
    )
    trainer.amp_context = contextlib.nullcontext()

    mb_obs = torch.tensor([[[0.1, -0.2, 0.5], [0.4, 0.3, -0.1]]], dtype=torch.float32)
    student = torch.distributions.Normal(
        mb_obs.reshape(-1, 3)[:, :2],
        torch.full((2, 2), 0.5),
    )

    bc_kl_loss, teacher_entropy, extra = trainer._compute_bc_kl_loss(student, mb_obs)
    metrics = trainer._compute_bc_kl_term(student, mb_obs)

    assert extra == {}
    assert bc_kl_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert teacher_entropy.item() > 0.0
    assert metrics["bc_kl"].item() == pytest.approx(0.0, abs=1e-7)
    assert metrics["bc_kl_coef"] == pytest.approx(0.008)


def test_pufferl_continuous_bc_kl_respects_valid_mask():
    valid_mask = torch.tensor([[False, True, True]])
    teacher_dist = torch.distributions.Normal(
        torch.tensor([[0.2, -0.1], [0.4, 0.5]], dtype=torch.float32),
        torch.full((2, 2), 0.7),
    )
    trainer = pufferl.PuffeRL.__new__(pufferl.PuffeRL)
    trainer.bc_kl_policy = _MaskedContinuousTeacher(valid_mask, teacher_dist)
    trainer.global_step = 10
    trainer.config = {"total_timesteps": 100}
    trainer.bc_kl_config = pufferl._resolve_bc_kl_config(
        {
            "enabled": True,
            "model_path": "bc_teacher.pt",
            "teacher_kind": "stacked_bc_continuous",
            "coef": 0.01,
            "final_coef": 0.0,
            "anneal_fraction": 1.0,
            "temperature": 1.0,
        }
    )
    trainer.amp_context = contextlib.nullcontext()

    student_loc = torch.tensor([[[9.0, 9.0], [0.2, -0.1], [0.4, 0.5]]], dtype=torch.float32)
    student_scale = torch.full_like(student_loc, 0.7)
    student = torch.distributions.Normal(student_loc, student_scale)

    bc_kl_loss, teacher_entropy, extra = trainer._compute_bc_kl_loss(student, torch.zeros(1, 3, 5))

    assert extra == {}
    assert bc_kl_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert teacher_entropy.item() > 0.0


def test_load_bc_kl_reference_policy_supports_stacked_bc_continuous(monkeypatch):
    sentinel = object()
    captured = {}

    def _fake_loader(path, *, device, runtime_env_cfg=None):
        captured["path"] = path
        captured["device"] = device
        captured["runtime_env_cfg"] = runtime_env_cfg
        return sentinel

    monkeypatch.setattr("pufferlib.ocean.drive.drive.load_stacked_bc_reference_policy", _fake_loader)

    ref = pufferl.load_bc_kl_reference_policy(
        {
            "train": {"device": "cuda"},
            "env": {"map_dir": "/tmp/new-train", "num_maps": 17},
            "bc_kl": {
                "enabled": True,
                "teacher_kind": "stacked_bc_continuous",
                "model_path": "/tmp/fake-teacher/best.pt",
                "coef": 0.01,
                "final_coef": 0.0,
                "anneal_fraction": 1.0,
                "temperature": 1.0,
            },
        },
        vecenv=None,
        env_name="puffer_drive",
    )

    assert ref is sentinel
    assert captured == {
        "path": "/tmp/fake-teacher/best.pt",
        "device": "cuda",
        "runtime_env_cfg": {"map_dir": "/tmp/new-train", "num_maps": 17},
    }
