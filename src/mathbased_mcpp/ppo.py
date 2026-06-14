from __future__ import annotations

from dataclasses import dataclass
import math

from .runtime import configure_runtime

configure_runtime()

import torch
from torch import nn
from torch.distributions import Categorical

from .intent_relation import intent_overlap_from_node_messages


class GraphAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        edge_dim: int = 0,
        residual: bool = False,
        attention_dropout: float = 0.0,
        use_intent_relation: bool = False,
        intent_relation_beta_max: float = 2.0,
        intent_relation_detach: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = max(int(num_heads), 1)
        if hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by gat_num_heads={self.num_heads}")
        self.head_dim = hidden_dim // self.num_heads
        self.edge_dim = max(int(edge_dim), 0)
        self.residual = bool(residual)
        self.use_intent_relation = bool(use_intent_relation)
        self.intent_relation_beta_max = float(intent_relation_beta_max)
        self.intent_relation_detach = bool(intent_relation_detach)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(self.edge_dim, self.num_heads, bias=False) if self.edge_dim > 0 else None
        self.raw_intent_bias = nn.Parameter(torch.zeros(())) if self.use_intent_relation else None
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )
        self.last_attention_weights: torch.Tensor | None = None
        self.last_attention_entropy: torch.Tensor | None = None
        self.last_intent_overlap: torch.Tensor | None = None
        self.last_intent_mask: torch.Tensor | None = None
        self.last_intent_beta: torch.Tensor | None = None

    def forward(
        self,
        features: torch.Tensor,
        neighbor_mask: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        intent_overlap: torch.Tensor | None = None,
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
        self.last_intent_overlap = None
        self.last_intent_mask = None
        self.last_intent_beta = None
        if self.use_intent_relation and intent_overlap is not None:
            rho = self._align_intent_overlap(intent_overlap, scores)
            if self.intent_relation_detach:
                rho = rho.detach()
            assert self.raw_intent_bias is not None
            beta = self.intent_relation_beta_max * torch.tanh(self.raw_intent_bias)
            scores = scores + beta * rho.unsqueeze(1)
            self.last_intent_overlap = rho.detach()
            self.last_intent_mask = self._relation_mask(neighbor_mask).detach()
            self.last_intent_beta = beta.detach()
        scores = scores.masked_fill(~neighbor_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        self.last_attention_weights = attention.detach()
        attention_for_entropy = attention.clamp_min(1e-8)
        self.last_attention_entropy = (-(attention_for_entropy * attention_for_entropy.log()).sum(dim=-1)).detach()
        context = torch.matmul(self.attention_dropout(attention), value)
        context = context.transpose(1, 2).contiguous().reshape(batch_size, num_agents, self.hidden_dim)
        output = self.output(torch.cat([features, context], dim=-1))
        return features + output if self.residual else output

    def _split_heads(self, tensor: torch.Tensor, batch_size: int, num_agents: int) -> torch.Tensor:
        return tensor.reshape(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _relation_mask(neighbor_mask: torch.Tensor) -> torch.Tensor:
        mask = neighbor_mask.clone()
        num_agents = mask.shape[-1]
        eye = torch.eye(num_agents, dtype=torch.bool, device=mask.device)
        while eye.ndim < mask.ndim:
            eye = eye.unsqueeze(0)
        return mask & ~eye

    @staticmethod
    def _align_intent_overlap(intent_overlap: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        rho = intent_overlap.to(device=scores.device, dtype=scores.dtype)
        if rho.ndim == 2:
            rho = rho.unsqueeze(0)
        expected_shape = (scores.shape[0], scores.shape[-2], scores.shape[-1])
        if rho.shape[0] == 1 and expected_shape[0] > 1:
            rho = rho.expand(expected_shape)
        if rho.shape != expected_shape:
            raise ValueError(f"intent overlap shape {tuple(rho.shape)} does not match expected shape {expected_shape}")
        return rho


class CUAPGate(nn.Module):
    def __init__(
        self,
        actor_feature_dim: int,
        gate_hidden_dim: int = 32,
        init_prob: float = 0.1,
        detach_actor_features: bool = True,
    ) -> None:
        super().__init__()
        self.actor_feature_dim = max(int(actor_feature_dim), 1)
        self.gate_hidden_dim = max(int(gate_hidden_dim), 1)
        self.detach_actor_features = bool(detach_actor_features)
        bounded_init_prob = min(max(float(init_prob), 1e-6), 1.0 - 1e-6)
        self.net = nn.Sequential(
            nn.Linear(self.actor_feature_dim + 2, self.gate_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(bounded_init_prob / (1.0 - bounded_init_prob)))

    def forward(self, actor_features: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        features = actor_features.detach() if self.detach_actor_features else actor_features
        return torch.sigmoid(self.net(torch.cat([features, confidence], dim=-1)))


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
        use_gated_cuap: bool = False,
        cuap_beta: float = 0.5,
        cuap_gate_hidden_dim: int = 32,
        cuap_gate_init_prob: float = 0.1,
        cuap_gate_detach_actor_features: bool = True,
        use_intent_relation: bool = False,
        intent_relation_beta_max: float = 2.0,
        intent_relation_detach: bool = True,
        intent_grid_size: int = 3,
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
        self.use_gated_cuap = bool(use_gated_cuap)
        self.cuap_beta = float(cuap_beta)
        self.cuap_gate_hidden_dim = max(int(cuap_gate_hidden_dim), 1)
        self.cuap_gate_init_prob = float(cuap_gate_init_prob)
        self.cuap_gate_detach_actor_features = bool(cuap_gate_detach_actor_features)
        self.use_intent_relation = bool(use_intent_relation)
        self.intent_relation_beta_max = float(intent_relation_beta_max)
        self.intent_relation_detach = bool(intent_relation_detach)
        self.intent_grid_size = max(int(intent_grid_size), 1)
        self.latest_gate: torch.Tensor | None = None
        self.latest_applied_gate: torch.Tensor | None = None
        self.latest_cuap_confidence: torch.Tensor | None = None
        self.latest_effective_strength: torch.Tensor | None = None
        self.latest_argmax_change: torch.Tensor | None = None
        self.latest_intent_overlap: torch.Tensor | None = None
        self.latest_intent_mask: torch.Tensor | None = None
        self.latest_intent_beta: torch.Tensor | None = None
        self.latest_attention_entropy: torch.Tensor | None = None
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
                use_intent_relation=self.use_intent_relation,
                intent_relation_beta_max=self.intent_relation_beta_max,
                intent_relation_detach=self.intent_relation_detach,
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
        self.cuap_gate = (
            CUAPGate(
                hidden_dim,
                gate_hidden_dim=self.cuap_gate_hidden_dim,
                init_prob=self.cuap_gate_init_prob,
                detach_actor_features=self.cuap_gate_detach_actor_features,
            )
            if self.use_gated_cuap
            else None
        )

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
                    intent_overlap = self._intent_overlap_from_messages(node_messages, neighbor_mask)
                    communicated = self.graph_attention(
                        messages.unsqueeze(0),
                        neighbor_mask,
                        edge_features=edge_features,
                        intent_overlap=intent_overlap,
                    ).squeeze(0)
                    self._record_intent_relation_diagnostics()
                    assert self.actor_communication_fusion is not None
                    return self.actor_communication_fusion(torch.cat([features, communicated], dim=-1))
                return features
            if self.graph_attention is not None and neighbor_mask is not None:
                if self.use_intent_relation:
                    raise ValueError("node_messages are required when use_intent_relation=true")
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
                    intent_overlap = self._intent_overlap_from_messages(node_messages, neighbor_mask)
                    communicated = self.graph_attention(
                        messages,
                        neighbor_mask,
                        edge_features=edge_features,
                        intent_overlap=intent_overlap,
                    )
                    self._record_intent_relation_diagnostics()
                    assert self.actor_communication_fusion is not None
                    features = self.actor_communication_fusion(torch.cat([features, communicated], dim=-1))
                return features
            if self.graph_attention is not None and neighbor_mask is not None:
                if self.use_intent_relation:
                    raise ValueError("node_messages are required when use_intent_relation=true")
                features = self.graph_attention(features, neighbor_mask, edge_features=edge_features)
            return features
        raise ValueError(f"expected observations with 1, 2, or 3 dimensions, got {observations.ndim}")

    def _intent_overlap_from_messages(
        self,
        node_messages: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.use_intent_relation:
            return None
        if node_messages is None:
            raise ValueError("node_messages are required when use_intent_relation=true")
        return intent_overlap_from_node_messages(node_messages, neighbor_mask, self.intent_grid_size)

    def _record_intent_relation_diagnostics(self) -> None:
        if self.graph_attention is None:
            return
        self.latest_intent_overlap = self.graph_attention.last_intent_overlap
        self.latest_intent_mask = self.graph_attention.last_intent_mask
        self.latest_intent_beta = self.graph_attention.last_intent_beta
        self.latest_attention_entropy = self.graph_attention.last_attention_entropy

    def _reset_intent_relation_diagnostics(self) -> None:
        self.latest_intent_overlap = None
        self.latest_intent_mask = None
        self.latest_intent_beta = None
        self.latest_attention_entropy = None

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

    def _policy_logits(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        node_messages: torch.Tensor | None = None,
        action_prior_logits: torch.Tensor | None = None,
        cuap_prior: torch.Tensor | None = None,
        cuap_confidence: torch.Tensor | None = None,
        cuap_phase_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._reset_intent_relation_diagnostics()
        features = self._actor_features(
            observations,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            node_messages=node_messages,
        )
        logits = self._actor_logits(features, states)
        self.latest_gate = None
        self.latest_applied_gate = None
        self.latest_cuap_confidence = None
        self.latest_effective_strength = None
        self.latest_argmax_change = None
        if not self.use_gated_cuap:
            return self._add_action_prior_logits(logits, action_prior_logits)
        if action_prior_logits is not None:
            raise ValueError("action_prior_logits cannot be mixed with gated CUAP inputs")
        if self.cuap_gate is None:
            raise ValueError("cuap_gate is required when use_gated_cuap=true")
        if cuap_prior is None or cuap_confidence is None or cuap_phase_mask is None:
            raise ValueError("cuap_prior, cuap_confidence, and cuap_phase_mask are required for gated CUAP")
        prior = self._align_action_tensor(cuap_prior, logits, "cuap prior")
        confidence = self._align_confidence_tensor(cuap_confidence, features, "cuap confidence")
        gate = self.cuap_gate(features, confidence)
        mask = self._align_gate_tensor(cuap_phase_mask, gate, "cuap phase mask")
        applied_gate = gate * mask
        final_logits = logits + self.cuap_beta * applied_gate * prior
        self.latest_gate = gate
        self.latest_applied_gate = applied_gate
        self.latest_cuap_confidence = confidence.detach()
        self.latest_effective_strength = (self.cuap_beta * applied_gate * prior).abs().detach()
        self.latest_argmax_change = (torch.argmax(logits, dim=-1) != torch.argmax(final_logits, dim=-1)).detach()
        return final_logits

    def distribution(
        self,
        observations: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        node_messages: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        action_prior_logits: torch.Tensor | None = None,
        cuap_prior: torch.Tensor | None = None,
        cuap_confidence: torch.Tensor | None = None,
        cuap_phase_mask: torch.Tensor | None = None,
    ) -> Categorical:
        logits = self._policy_logits(
            observations,
            states=states,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            node_messages=node_messages,
            action_prior_logits=action_prior_logits,
            cuap_prior=cuap_prior,
            cuap_confidence=cuap_confidence,
            cuap_phase_mask=cuap_phase_mask,
        )
        return Categorical(logits=self._mask_action_logits(logits, action_mask))

    @staticmethod
    def _add_action_prior_logits(logits: torch.Tensor, action_prior_logits: torch.Tensor | None) -> torch.Tensor:
        if action_prior_logits is None:
            return logits
        prior = ActorCritic._align_action_tensor(action_prior_logits, logits, "action prior")
        return logits + prior

    @staticmethod
    def _align_action_tensor(tensor: torch.Tensor, logits: torch.Tensor, label: str) -> torch.Tensor:
        aligned = tensor.to(device=logits.device, dtype=logits.dtype)
        if aligned.ndim + 1 == logits.ndim:
            aligned = aligned.unsqueeze(0)
        if aligned.shape != logits.shape:
            raise ValueError(f"{label} shape {tuple(aligned.shape)} does not match logits shape {tuple(logits.shape)}")
        return aligned

    @staticmethod
    def _align_gate_tensor(tensor: torch.Tensor, gate: torch.Tensor, label: str) -> torch.Tensor:
        aligned = tensor.to(device=gate.device, dtype=gate.dtype)
        if aligned.ndim + 1 == gate.ndim:
            aligned = aligned.unsqueeze(-1)
        if aligned.shape != gate.shape:
            raise ValueError(f"{label} shape {tuple(aligned.shape)} does not match gate shape {tuple(gate.shape)}")
        return aligned

    @staticmethod
    def _align_confidence_tensor(tensor: torch.Tensor, features: torch.Tensor, label: str) -> torch.Tensor:
        aligned = tensor.to(device=features.device, dtype=features.dtype)
        if aligned.ndim + 1 == features.ndim:
            aligned = aligned.unsqueeze(0)
        expected_shape = (*features.shape[:-1], 2)
        if aligned.shape != expected_shape:
            raise ValueError(f"{label} shape {tuple(aligned.shape)} does not match expected shape {expected_shape}")
        return aligned

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
        if self.use_gated_cuap:
            allowed_missing = (*allowed_missing, "cuap_gate.")
        if self.use_intent_relation:
            allowed_missing = (*allowed_missing, "graph_attention.raw_intent_bias")
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
        cuap_prior: torch.Tensor | None = None,
        cuap_confidence: torch.Tensor | None = None,
        cuap_phase_mask: torch.Tensor | None = None,
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
            cuap_prior=cuap_prior,
            cuap_confidence=cuap_confidence,
            cuap_phase_mask=cuap_phase_mask,
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
        cuap_prior: torch.Tensor | None = None,
        cuap_confidence: torch.Tensor | None = None,
        cuap_phase_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if state is not None and state.ndim == 1:
            state = state.unsqueeze(0)
        if state is None:
            if self._uses_spatial_state:
                raise ValueError("state tensor is required when using the spatial critic")
            state = observation
        logits = self._policy_logits(
            observation,
            states=state,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            node_messages=node_messages,
            action_prior_logits=action_prior_logits,
            cuap_prior=cuap_prior,
            cuap_confidence=cuap_confidence,
            cuap_phase_mask=cuap_phase_mask,
        )
        values = self._critic_values(state)
        if logits.ndim == 2 and values.shape == (1,) and logits.shape[0] != 1:
            values = values.expand(logits.shape[0])
        elif logits.ndim == 3 and values.shape == logits.shape[:1]:
            values = values.unsqueeze(-1).expand(logits.shape[:2])
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
        cuap_prior: torch.Tensor | None = None,
        cuap_confidence: torch.Tensor | None = None,
        cuap_phase_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self._policy_logits(
            observations,
            states=states,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            node_messages=node_messages,
            action_prior_logits=action_prior_logits,
            cuap_prior=cuap_prior,
            cuap_confidence=cuap_confidence,
            cuap_phase_mask=cuap_phase_mask,
        )
        values = self._critic_values(states)
        if logits.ndim == 2 and values.shape == (1,) and logits.shape[0] != 1:
            values = values.expand(logits.shape[0])
        elif logits.ndim == 3 and values.shape == logits.shape[:1]:
            values = values.unsqueeze(-1).expand(logits.shape[:2])
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
    cuap_priors: torch.Tensor | None = None
    cuap_confidences: torch.Tensor | None = None
    cuap_phase_masks: torch.Tensor | None = None
