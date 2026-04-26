import torch

import pufferlib.pytorch


def test_eval_action_from_logits_uses_mean_for_normal_when_deterministic():
    loc = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    scale = torch.full_like(loc, 0.3)
    logits = torch.distributions.Normal(loc, scale)

    action, logprob, entropy = pufferlib.pytorch.eval_action_from_logits(logits, deterministic=True)

    assert torch.allclose(action, loc)
    assert logprob.shape == torch.Size([1])
    assert entropy.shape == torch.Size([1])


def test_eval_action_from_logits_samples_for_normal_when_not_deterministic():
    loc = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    scale = torch.full_like(loc, 0.3)
    logits = torch.distributions.Normal(loc, scale)

    torch.manual_seed(0)
    action, logprob, entropy = pufferlib.pytorch.eval_action_from_logits(logits, deterministic=False)

    assert action.shape == loc.shape
    assert not torch.allclose(action, loc)
    assert logprob.shape == torch.Size([1])
    assert entropy.shape == torch.Size([1])


def test_eval_action_from_logits_uses_bounded_mean_for_squashed_normal_when_deterministic():
    loc = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    scale = torch.full_like(loc, 0.3)
    logits = pufferlib.pytorch.SquashedNormal(loc, scale)

    action, logprob, entropy = pufferlib.pytorch.eval_action_from_logits(logits, deterministic=True)

    assert torch.allclose(action, torch.tanh(loc))
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
    assert logprob.shape == torch.Size([1])
    assert entropy.shape == torch.Size([1])


def test_sample_logits_samples_bounded_actions_for_squashed_normal():
    loc = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    scale = torch.full_like(loc, 0.3)
    logits = pufferlib.pytorch.SquashedNormal(loc, scale)

    torch.manual_seed(0)
    action, logprob, entropy = pufferlib.pytorch.sample_logits(logits)

    assert action.shape == loc.shape
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
    assert not torch.allclose(action, torch.tanh(loc))
    assert logprob.shape == torch.Size([1])
    assert entropy.shape == torch.Size([1])
