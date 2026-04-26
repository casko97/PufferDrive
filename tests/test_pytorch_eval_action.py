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
