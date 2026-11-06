import pytest
import torch

from pufferlib.pufferl import _adapt_legacy_drive_ego_encoder
from pufferlib.pufferl import _load_policy_state_dict


class EgoEncoderPolicy(torch.nn.Module):
    def __init__(self, input_features=8):
        super().__init__()
        self.ego_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, 2, bias=False)
        )


def test_legacy_seven_feature_drive_encoder_is_zero_padded():
    policy = EgoEncoderPolicy(input_features=8)
    legacy_weight = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    state_dict = {"ego_encoder.0.weight": legacy_weight.clone()}

    with pytest.warns(UserWarning, match="seven-feature PufferDrive checkpoint"):
        _load_policy_state_dict(policy, state_dict, "puffer_drive")

    loaded_weight = policy.ego_encoder[0].weight.detach()
    torch.testing.assert_close(loaded_weight[:, :7], legacy_weight)
    torch.testing.assert_close(loaded_weight[:, 7], torch.zeros(2))


def test_current_eight_feature_drive_encoder_is_unchanged():
    policy = EgoEncoderPolicy(input_features=8)
    current_weight = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    state_dict = {"ego_encoder.0.weight": current_weight.clone()}

    _load_policy_state_dict(policy, state_dict, "puffer_drive")

    torch.testing.assert_close(policy.ego_encoder[0].weight.detach(), current_weight)


def test_adapter_does_not_hide_unrelated_shape_mismatches():
    policy = EgoEncoderPolicy(input_features=8)
    state_dict = {"ego_encoder.0.weight": torch.zeros((2, 6))}

    adapted = _adapt_legacy_drive_ego_encoder(state_dict, policy.state_dict())

    assert adapted == []
    with pytest.raises(RuntimeError, match="size mismatch"):
        _load_policy_state_dict(policy, state_dict, "puffer_drive")
