import torch

from pufferlib.pufferl import advantage_filter_mask, masked_mean


def test_advantage_filter_initializes_from_rollout_maximum():
    advantages = torch.tensor([[-10.0, -0.11, 0.10, 0.0]])
    mask, running_max, threshold = advantage_filter_mask(advantages)

    assert running_max == 10.0
    assert threshold == 0.1
    assert torch.equal(mask, torch.tensor([[True, True, True, False]]))


def test_advantage_filter_uses_gigaflow_ewma():
    _, running_max, _ = advantage_filter_mask(torch.tensor([[-8.0]]))
    mask, running_max, threshold = advantage_filter_mask(
        torch.tensor([[-4.0, 0.05]]), running_max, threshold_scale=0.01, ema_decay=0.25
    )

    assert running_max == 7.0  # 0.25 * 4 + 0.75 * 8
    assert threshold == 0.07
    assert torch.equal(mask, torch.tensor([[True, False]]))


def test_masked_mean_excludes_filtered_transitions():
    values = torch.tensor([[2.0, 100.0, 4.0]])
    mask = torch.tensor([[True, False, True]])

    assert masked_mean(values, mask).item() == 3.0


def test_masked_mean_does_not_propagate_filtered_nan():
    values = torch.tensor([[float("nan"), 3.0]])
    mask = torch.tensor([[False, True]])

    assert masked_mean(values, mask).item() == 3.0
