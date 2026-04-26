from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401


Recurrent = pufferlib.models.LSTMWrapper
EMPTY_PARTNER_EPS = 1e-8
EGO_TRAILER_STATE_FEATURES = 4
TRAJECTORY_HISTORY_FEATURES = 6


class Drive(nn.Module):
    def __init__(
        self,
        env,
        input_size=128,
        hidden_size=128,
        initial_std_bias=0.0,
        min_action_std=1e-4,
        max_action_std=None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.observation_mode = getattr(env, "observation_mode", 0)
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6  # 6 is the number of one-hot encoded categories
        self.is_trajectory_policy = getattr(env, "is_trajectory_action", False)
        self.trajectory_base_obs_dim = getattr(env, "trajectory_base_obs_dim", self.observation_size)
        self.trajectory_ego_history_dim = getattr(env, "trajectory_ego_history_dim", 0)
        self.trajectory_partner_history_dim = getattr(env, "trajectory_partner_history_dim", 0)
        self.trajectory_history_horizon = getattr(env, "trajectory_history_horizon", 0)
        self.trajectory_history_features = getattr(env, "trajectory_history_features", TRAJECTORY_HISTORY_FEATURES)
        self.initial_std_bias = float(initial_std_bias)
        self.min_action_std = float(min_action_std)
        self.max_action_std = None if max_action_std is None else float(max_action_std)

        self.base_ego_dim = 10 if env.dynamics_model == "jerk" else 7
        self.ego_dim = env.ego_features
        self.base_partner_features = 7
        self.type_classes = env.type_classes
        self.real_type_classes = max(1, self.type_classes - 1)
        self.has_augmented_ego = self.ego_dim > self.base_ego_dim
        self.has_partner_type = self.partner_features > self.base_partner_features
        self.ego_trailer_state_features = EGO_TRAILER_STATE_FEATURES if self.has_augmented_ego else 0
        self.ego_encoder_input_dim = self.base_ego_dim + self.ego_trailer_state_features + (
            self.type_classes if self.has_augmented_ego else 0
        )
        self.partner_encoder_input_dim = self.base_partner_features + (
            self.real_type_classes if self.has_partner_type else 0
        )

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_encoder_input_dim, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_encoder_input_dim, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        if self.trajectory_ego_history_dim > 0:
            self.ego_history_encoder = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(self.trajectory_history_horizon * self.trajectory_history_features, input_size)
                ),
                nn.LayerNorm(input_size),
                nn.GELU(),
                pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
            )
            self.partner_history_encoder = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(self.trajectory_history_horizon * self.trajectory_history_features, input_size)
                ),
                nn.LayerNorm(input_size),
                nn.GELU(),
                pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
            )
            shared_input_dim = 5 * input_size
        else:
            self.ego_history_encoder = None
            self.partner_history_encoder = None
            shared_input_dim = 3 * input_size

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(shared_input_dim, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def reinitialize_value_head(self):
        refreshed = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1), std=1)
        device = self.value_fn.weight.device
        dtype = self.value_fn.weight.dtype
        refreshed = refreshed.to(device=device, dtype=dtype)
        with torch.no_grad():
            self.value_fn.weight.copy_(refreshed.weight)
            self.value_fn.bias.copy_(refreshed.bias)

    def configure_finetune_trainability(self, *, freeze_backbone=True, train_actor=True, train_value=True):
        for name, parameter in self.named_parameters():
            if name.startswith("actor."):
                parameter.requires_grad = train_actor
            elif name.startswith("value_fn."):
                parameter.requires_grad = train_value
            else:
                parameter.requires_grad = not freeze_backbone

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        base_obs = observations[:, : self.trajectory_base_obs_dim]
        ego_obs = base_obs[:, :ego_dim]
        partner_obs = base_obs[:, ego_dim : ego_dim + partner_dim]
        road_obs = base_obs[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        if self.has_partner_type:
            partner_continuous = partner_objects[:, :, : self.base_partner_features]
            partner_type = partner_objects[:, :, self.base_partner_features].long().clamp(
                min=0, max=self.type_classes - 1
            )
            occupied_partner_slots = partner_continuous.abs().amax(dim=2) > EMPTY_PARTNER_EPS
            partner_type_idx = (partner_type - 1).clamp(min=0, max=self.real_type_classes - 1)
            partner_type_onehot = F.one_hot(partner_type_idx, num_classes=self.real_type_classes).to(
                partner_continuous.dtype
            )
            partner_type_onehot = partner_type_onehot * occupied_partner_slots.unsqueeze(-1).to(partner_continuous.dtype)
            partner_objects = torch.cat([partner_continuous, partner_type_onehot], dim=2)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, ROAD_MAX_OBJECTS, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        if self.has_augmented_ego:
            ego_core = ego_obs[:, : self.base_ego_dim]
            ego_type = ego_obs[:, self.base_ego_dim].long().clamp(min=0, max=self.type_classes - 1)
            ego_trailer_state = ego_obs[
                :, self.base_ego_dim + 1 : self.base_ego_dim + 1 + self.ego_trailer_state_features
            ]
            ego_type_onehot = F.one_hot(ego_type, num_classes=self.type_classes).to(ego_core.dtype)
            ego_obs = torch.cat([ego_core, ego_trailer_state, ego_type_onehot], dim=1)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        if self.trajectory_ego_history_dim > 0:
            history_start = self.trajectory_base_obs_dim
            history_end = history_start + self.trajectory_ego_history_dim
            ego_history = observations[:, history_start:history_end]
            ego_history_features = self.ego_history_encoder(ego_history)

            partner_history = observations[:, history_end : history_end + self.trajectory_partner_history_dim]
            partner_history = partner_history.view(
                -1,
                self.max_partner_objects,
                self.trajectory_history_horizon * self.trajectory_history_features,
            )
            partner_history_features, _ = self.partner_history_encoder(partner_history).max(dim=1)
            concat_features = torch.cat(
                [ego_features, road_features, partner_features, ego_history_features, partner_history_features], dim=1
            )

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale + self.initial_std_bias)
            if self.max_action_std is None:
                std = torch.clamp(std, min=self.min_action_std)
            else:
                std = torch.clamp(std, min=self.min_action_std, max=self.max_action_std)
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value
