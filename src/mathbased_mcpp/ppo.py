from __future__ import annotations

from dataclasses import dataclass
import math

from .runtime import configure_runtime

configure_runtime()

import torch
from torch import nn
from torch.distributions import Categorical


class GraphAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        edge_dim: int = 0,
        residual: bool = False,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = max(int(num_heads), 1)
        if hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by gat_num_heads={self.num_heads}")
        self.head_dim = hidden_dim // self.num_heads
        self.edge_dim = max(int(edge_dim), 0)
        self.residual = bool(residual)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(self.edge_dim, self.num_heads, bias=False) if self.edge_dim > 0 else None
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        features: torch.Tensor,
        neighbor_mask: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if neighbor_mask.ndim == 2:
            neighbor_mask = neighbor_mask.unsqueeze(0)
        neighbor_mask = neighbor_mask.to(device=features.device, dtype=torch.bool)
        if neighbor_mask.shape[0] == 1 and features.shape[0] > 1:
            neighbor_mask = neighbor_mask.expand(features.shape[0], -1, -1)
        if edge_features is not None:
            if edge_features.ndim == 3:
                edge_features = edge_features.unsqueeze(0)
            edge_features = edge_features.to(device=features.device, dtype=features.dtype)
            if edge_features.shape[0] == 1 and features.shape[0] > 1:
                edge_features = edge_features.expand(features.shape[0], -1, -1, -1)

        batch_size, num_agents, _ = features.shape
        query = self._split_heads(self.query(features), batch_size, num_agents)
        key = self._split_heads(self.key(features), batch_size, num_agents)
        value = self._split_heads(self.value(features), batch_size, num_agents)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.edge_bias is not None and edge_features is not None:
            edge_bias = self.edge_bias(edge_features).permute(0, 3, 1, 2)
            scores = scores + edge_bias
        scores = scores.masked_fill(~neighbor_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        self.last_attention_weights = attention.detach()
        context = torch.matmul(self.attention_dropout(attention), value)
        context = context.transpose(1, 2).contiguous().reshape(batch_size, num_agents, self.hidden_dim)
        output = self.output(torch.cat([features, context], dim=-1))
        return features + output if self.residual else output

    def _split_heads(self, tensor: torch.Tensor, batch_size: int, num_agents: int) -> torch.Tensor:
        return tensor.reshape(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        state_dim: int | None = None,
        state_shape: tuple[int, int] | None = None,
        state_channels: int = 5,
        state_metadata_dim: int = 7,
        use_graph_attention: bool = False,
        gat_num_heads: int = 1,
        gat_edge_dim: int = 0,
        gat_residual: bool = False,
        gat_attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.state_shape = state_shape
        self.state_channels = state_channels
        self.state_metadata_dim = state_metadata_dim
        self.use_graph_attention = use_graph_attention
        self.gat_num_heads = max(int(gat_num_heads), 1)
        self.gat_edge_dim = max(int(gat_edge_dim), 0)
        self.gat_residual = bool(gat_residual)
        self.gat_attention_dropout = float(gat_attention_dropout)
        self._uses_spatial_state = state_shape is not None
        self.actor_body = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.graph_attention = (
            GraphAttentionBlock(
                hidden_dim,
                num_heads=self.gat_num_heads,
                edge_dim=self.gat_edge_dim,
                residual=self.gat_residual,
                attention_dropout=self.gat_attention_dropout,
            )
            if use_graph_attention
            else None
        )
        if self._uses_spatial_state:
            if state_shape is None:
                raise ValueError("state_shape is required when using the spatial critic")
            self.state_dim = state_channels * state_shape[0] * state_shape[1] + state_metadata_dim
            self.state_encoder = nn.Sequential(
                nn.Conv2d(state_channels, hidden_dim, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.critic_body = nn.Sequential(
                nn.Linear(hidden_dim + state_metadata_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
        else:
            if state_dim is None:
                raise ValueError("state_dim is required when state_shape is not provided")
            self.state_dim = state_dim
            self.critic_body = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    @property
    def critic_mode(self) -> str:
        return "spatial" if self._uses_spatial_state else "legacy"

    def _actor_features(
        self,
        observations: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if observations.ndim == 2:
            features = self.actor_body(observations)
            if self.graph_attention is not None and neighbor_mask is not None:
                grouped = features.unsqueeze(0)
                return self.graph_attention(grouped, neighbor_mask, edge_features=edge_features).squeeze(0)
            return features
        if observations.ndim == 3:
            batch_size, num_agents, observation_dim = observations.shape
            features = self.actor_body(observations.reshape(-1, observation_dim)).reshape(batch_size, num_agents, -1)
            if self.graph_attention is not None and neighbor_mask is not None:
                features = self.graph_attention(features, neighbor_mask, edge_features=edge_features)
            return features
        raise ValueError(f"expected observations with 1, 2, or 3 dimensions, got {observations.ndim}")

    def _critic_features(self, states: torch.Tensor) -> torch.Tensor:
        leading_shape = states.shape[:-1] if states.ndim > 1 else torch.Size()
        if not self._uses_spatial_state:
            if states.ndim == 1:
                states = states.unsqueeze(0)
                leading_shape = torch.Size()
            flat_states = states.reshape(-1, states.shape[-1])
            features = self.critic_body(flat_states)
            return features.reshape(*leading_shape, -1) if leading_shape else features

        if self.state_shape is None:
            raise ValueError("state_shape is required when using the spatial critic")
        if states.ndim == 1:
            states = states.unsqueeze(0)
            leading_shape = torch.Size()

        flat_states = states.reshape(states.shape[0], -1)
        if states.ndim > 2:
            flat_states = states.reshape(-1, states.shape[-1])
        height, width = self.state_shape
        map_area = self.state_channels * height * width
        expected_dim = map_area + self.state_metadata_dim
        if flat_states.shape[-1] != expected_dim:
            raise ValueError(
                f"expected state vectors with {expected_dim} values for state_shape={self.state_shape}, "
                f"got {flat_states.shape[-1]}"
            )

        maps = flat_states[:, :map_area].reshape(-1, self.state_channels, height, width)
        metadata = flat_states[:, map_area:]
        encoded = self.state_encoder(maps)
        features = self.critic_body(torch.cat([encoded, metadata], dim=-1))
        return features.reshape(*leading_shape, -1) if leading_shape else features

    def forward(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is None:
            if self._uses_spatial_state:
                raise ValueError("state tensor is required when using the spatial critic")
            states = observations
        actor_features = self._actor_features(observations, neighbor_mask=neighbor_mask, edge_features=edge_features)
        critic_features = self._critic_features(states)
        return self.actor(actor_features), self.critic(critic_features).squeeze(-1)

    def distribution(
        self,
        observations: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> Categorical:
        logits = self.actor(self._actor_features(observations, neighbor_mask=neighbor_mask, edge_features=edge_features))
        return Categorical(logits=logits)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 1:
            states = states.unsqueeze(0)
        return self.critic(self._critic_features(states)).squeeze(-1)

    def act(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        actions, log_probs, values = self.act_batch(
            observation,
            state,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            deterministic=deterministic,
        )
        return int(actions[0].item()), log_probs[0], values[0]

    def act_batch(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if state is not None and state.ndim == 1:
            state = state.unsqueeze(0)
        logits, values = self.forward(observation, state, neighbor_mask=neighbor_mask, edge_features=edge_features)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(observations, states, neighbor_mask=neighbor_mask, edge_features=edge_features)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values

    def latest_attention_weights(self) -> torch.Tensor | None:
        return None if self.graph_attention is None else self.graph_attention.last_attention_weights


@dataclass(slots=True)
class RolloutBatch:
    observations: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    neighbor_masks: torch.Tensor | None = None
    edge_features: torch.Tensor | None = None
