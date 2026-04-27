import torch

from pufferlib.pufferl import _distribution_anchor_loss
from pufferlib.pytorch import SquashedNormal


def test_distribution_anchor_loss_zero_for_matching_squashed_normals():
    current = SquashedNormal(torch.zeros(2, 4), torch.ones(2, 4) * 0.1)
    reference = SquashedNormal(torch.zeros(2, 4), torch.ones(2, 4) * 0.1)

    loss = _distribution_anchor_loss(current, reference)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_distribution_anchor_loss_increases_with_mean_drift():
    current = SquashedNormal(torch.ones(2, 4) * 0.3, torch.ones(2, 4) * 0.1)
    reference = SquashedNormal(torch.zeros(2, 4), torch.ones(2, 4) * 0.1)

    loss = _distribution_anchor_loss(current, reference)

    assert float(loss) > 0.0
