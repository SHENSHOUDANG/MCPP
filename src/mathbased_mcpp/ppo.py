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
        node_message_dim: int = 0,
        use_phase_critics: bool = False,
        use_phase_actors: bool = False,
        phase_metadata_index: int = 7,
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
        self.node_message_dim = max(int(node_message_dim), 0)
        self.use_phase_critics = bool(use_phase_critics)
        self.use_phase_actors = bool(use_phase_actors)
        self.phase_metadata_index = int(phase_metadata_index)
        self._uses_spatial_state = state_shape is not None
        self.actor_body = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.message_body = (
            nn.Sequential(
                nn.Linear(self.node_message_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            if self.node_message_dim > 0
            else None
        )
        self.actor_message_fusion = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
            )
            if self.node_message_dim > 0
            else None
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
        self.actor_communication_fusion = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
            )
            if use_graph_attention and self.node_message_dim > 0
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
            if self.use_phase_critics:
                self.return_state_encoder = nn.Sequential(
                    nn.Conv2d(state_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.Tanh(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.Tanh(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                self.return_critic_body = nn.Sequential(
                    nn.Linear(hidden_dim + state_metadata_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                )
            else:
                self.return_state_encoder = None
                self.return_critic_body = None
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
            self.return_state_encoder = None
            self.return_critic_body = (
                nn.Sequential(
                    nn.Linear(self.state_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                )
                if self.use_phase_critics
                else None
            )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.return_actor = nn.Linear(hidden_dim, action_dim) if self.use_phase_actors else None
        self.critic = nn.Linear(hidden_dim, 1)
        self.return_critic = nn.Linear(hidden_dim, 1) if self.use_phase_critics else None

    @property
    def critic_mode(self) -> str:
        return "spatial" if self._uses_spatial_state else "legacy"

    def _actor_features(
        self,
        observations: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        node_messages: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if observations.ndim == 2:
            features = self.actor_body(observations)
            if self.message_body is not None:
                if node_messages is None:
                    raise ValueError("node_messages are required when node_message_dim is enabled")
                if node_messages.ndim == 1:
                    node_messages = node_messages.unsqueeze(0)
                messages = self.message_body(node_messages)
                assert self.actor_message_fusion is not None
                features = self.actor_message_fusion(torch.cat([features, messages], dim=-1))
                if self.graph_attention is not None and neighbor_mask is not None:
                    communicated = self.graph_attention(messages.unsqueeze(0), neighbor_mask, edge_features=edge_features).squeeze(0)
                    assert self.actor_communication_fusion is not None
                    return self.actor_communication_fusion(torch.cat([features, communicated], dim=-1))
                return features
            if self.graph_attention is not None and neighbor_mask is not None:
                grouped = features.unsqueeze(0)
                return self.graph_attention(grouped, neighbor_mask, edge_features=edge_features).squeeze(0)
            return features
        if observations.ndim == 3:
            batch_size, num_agents, observation_dim = observations.shape
            features = self.actor_body(observations.reshape(-1, observation_dim)).reshape(batch_size, num_agents, -1)
            if self.message_body is not None:
                if node_messages is None:
                    raise ValueError("node_messages are required when node_message_dim is enabled")
                if node_messages.ndim == 2:
                    node_messages = node_messages.unsqueeze(0)
                messages = self.message_body(node_messages.reshape(-1, node_messages.shape[-1])).reshape(batch_size, num_agents, -1)
                assert self.actor_message_fusion is not None
                features = self.actor_message_fusion(torch.cat([features, messages], dim=-1))
                if self.graph_attention is not None and neighbor_mask is not None:
                    communicated = self.graph_attention(messages, neighbor_mask, edge_features=edge_features)
                    assert self.actor_communication_fusion is not None
                    features = self.actor_communication_fusion(torch.cat([features, communicated], dim=-1))
                return features
            if self.graph_attention is not None and neighbor_mask is not None:
                features = self.graph_attention(features, neighbor_mask, edge_features=edge_features)
            return features
        raise ValueError(f"expected observations with 1, 2, or 3 dimensions, got {observations.ndim}")

    def _critic_features(self, states: torch.Tensor) -> torch.Tensor:
        return self._critic_features_from_modules(states, self.state_encoder if self._uses_spatial_state else None, self.critic_body)

    def _return_critic_features(self, states: torch.Tensor) -> torch.Tensor:
        if not self.use_phase_critics or self.return_critic_body is None:
            return self._critic_features(states)
        return self._critic_features_from_modules(
            states,
            self.return_state_encoder if self._uses_spatial_state else None,
            self.return_critic_body,
        )

    def _critic_features_from_modules(
        self,
        states: torch.Tensor,
        state_encoder: nn.Module | None,
        critic_body: nn.Module,
    ) -> torch.Tensor:
        leading_shape = states.shape[:-1] if states.ndim > 1 else torch.Size()
        if not self._uses_spatial_state:
            if states.ndim == 1:
                states = states.unsqueeze(0)
                leading_shape = torch.Size()
            flat_states = states.reshape(-1, states.shape[-1])
            features = critic_body(flat_states)
            return features.reshape(*leading_shape, -1) if leading_shape else features

        if self.state_shape is None:
            raise ValueError("state_shape is required when using the spatial critic")
        if state_encoder is None:
            raise ValueError("state_encoder is required when using the spatial critic")
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
        encoded = state_encoder(maps)
        features = critic_body(torch.cat([encoded, metadata], dim=-1))
        return features.reshape(*leading_shape, -1) if leading_shape else features

    def _critic_values(self, states: torch.Tensor) -> torch.Tensor:
        coverage_values = self.critic(self._critic_features(states)).squeeze(-1)
        if not self.use_phase_critics or self.return_critic is None:
            return coverage_values
        return_values = self.return_critic(self._return_critic_features(states)).squeeze(-1)
        return_mask = self._return_phase_mask(states).to(device=coverage_values.device)
        while return_mask.ndim < coverage_values.ndim:
            return_mask = return_mask.unsqueeze(-1)
        return torch.where(return_mask, return_values, coverage_values)

    def _actor_logits(self, features: torch.Tensor, states: torch.Tensor | None = None) -> torch.Tensor:
        coverage_logits = self.actor(features)
        if not self.use_phase_actors or self.return_actor is None or states is None:
            return coverage_logits
        return_logits = self.return_actor(features)
        return_mask = self._return_phase_mask(states).to(device=coverage_logits.device)
        while return_mask.ndim < coverage_logits.ndim - 1:
            return_mask = return_mask.unsqueeze(-1)
        return_mask = return_mask.unsqueeze(-1)
        return torch.where(return_mask, return_logits, coverage_logits)

    def _return_phase_mask(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 1:
            flat_states = states.unsqueeze(0)
            leading_shape = torch.Size([1])
        else:
            leading_shape = states.shape[:-1]
            flat_states = states.reshape(-1, states.shape[-1])
        phase_offset = self.phase_metadata_index
        if self._uses_spatial_state:
            if self.state_shape is None:
                raise ValueError("state_shape is required when using the spatial critic")
            phase_offset += self.state_channels * self.state_shape[0] * self.state_shape[1]
        if phase_offset >= flat_states.shape[-1]:
            mask = torch.zeros(flat_states.shape[0], dtype=torch.bool, device=states.device)
        else:
            mask = flat_states[:, phase_offset] >= 0.5
        return mask.reshape(leading_shape)

    def forward(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        node_messages: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is None:
            if self._uses_spatial_state:
                raise ValueError("state tensor is required when using the spatial critic")
            states = observations
        actor_features = self._actor_features(
            observations,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            node_messages=node_messages,
        )
        logits = self._actor_logits(actor_features, states)
        values = self._critic_values(states)
        if logits.ndim == 2 and values.shape == (1,) and logits.shape[0] != 1:
            values = values.expand(logits.shape[0])
        elif logits.ndim == 3 and values.shape == logits.shape[:1]:
            values = values.unsqueeze(-1).expand(logits.shape[:2])
        return logits, values

    def distribution(
        self,
        observations: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        node_messages: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        action_prior_logits: torch.Tensor | None = None,
    ) -> Categorical:
        features = self._actor_features(
            observations,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            node_messages=node_messages,
        )
        logits = self._actor_logits(features, states)
        logits = self._add_action_prior_logits(logits, action_prior_logits)
        return Categorical(logits=self._mask_action_logits(logits, action_mask))

    @staticmethod
    def _add_action_prior_logits(logits: torch.Tensor, action_prior_logits: torch.Tensor | None) -> torch.Tensor:
        if action_prior_logits is None:
            return logits
        prior = action_prior_logits.to(device=logits.device, dtype=logits.dtype)
        if prior.ndim + 1 == logits.ndim:
            prior = prior.unsqueeze(0)
        if prior.shape != logits.shape:
            raise ValueError(f"action prior shape {tuple(prior.shape)} does not match logits shape {tuple(logits.shape)}")
        return logits + prior

    @staticmethod
    def _mask_action_logits(logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return logits
        mask = action_mask.to(device=logits.device, dtype=torch.bool)
        if mask.ndim + 1 == logits.ndim:
            mask = mask.unsqueeze(0)
        if mask.shape != logits.shape:
            raise ValueError(f"action mask shape {tuple(mask.shape)} does not match logits shape {tuple(logits.shape)}")
        has_available_action = mask.any(dim=-1, keepdim=True)
        effective_mask = torch.where(has_available_action, mask, torch.ones_like(mask))
        return logits.masked_fill(~effective_mask, torch.finfo(logits.dtype).min)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 1:
            states = states.unsqueeze(0)
        return self._critic_values(states)

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        converted = dict(state_dict)
        if self.use_phase_critics:
            self._copy_missing_phase_weights(converted)
        missing, unexpected = self.load_state_dict(converted, strict=False)
        allowed_missing = (
            "return_state_encoder.",
            "return_critic_body.",
            "return_critic.",
        )
        hard_missing = [key for key in missing if not key.startswith(allowed_missing)]
        if hard_missing or unexpected:
            raise RuntimeError(f"incompatible checkpoint: missing={hard_missing}, unexpected={unexpected}")

    def _copy_missing_phase_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        prefixes = {
            "return_state_encoder.": "state_encoder.",
            "return_critic_body.": "critic_body.",
            "return_critic.": "critic.",
            "return_actor.": "actor.",
        }
        own_state = self.state_dict()
        for target_key in own_state:
            for target_prefix, source_prefix in prefixes.items():
                if not target_key.startswith(target_prefix) or target_key in state_dict:
                    continue
                source_key = source_prefix + target_key[len(target_prefix) :]
                if source_key in state_dict and state_dict[source_key].shape == own_state[target_key].shape:
                    state_dict[target_key] = state_dict[source_key].clone()

    def act(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        node_messages: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        action_prior_logits: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        actions, log_probs, values = self.act_batch(
            observation,
            state,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            node_messages=node_messages,
            action_mask=action_mask,
            action_prior_logits=action_prior_logits,
            deterministic=deterministic,
        )
        return int(actions[0].item()), log_probs[0], values[0]

    def act_batch(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        node_messages: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        action_prior_logits: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if state is not None and state.ndim == 1:
            state = state.unsqueeze(0)
        logits, values = self.forward(
            observation,
            state,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            node_messages=node_messages,
        )
        logits = self._add_action_prior_logits(logits, action_prior_logits)
        logits = self._mask_action_logits(logits, action_mask)
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
        node_messages: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        action_prior_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(
            observations,
            states,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            node_messages=node_messages,
        )
        logits = self._add_action_prior_logits(logits, action_prior_logits)
        dist = Categorical(logits=self._mask_action_logits(logits, action_mask))
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
    node_messages: torch.Tensor | None = None
    action_masks: torch.Tensor | None = None
    action_prior_logits: torch.Tensor | None = None
