from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(slots=True)
class PortMappoBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    action_masks: torch.Tensor
    agent_types: torch.Tensor
    agent_masks: torch.Tensor
    alive_masks: torch.Tensor


class SharedActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(observations))


class DeepSetsCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.agent_encoder = nn.Sequential(
            nn.Linear(observation_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor, agent_types: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        type_features = torch.stack([agent_types == 0, agent_types == 1], dim=-1).to(observations.dtype)
        encoded = self.agent_encoder(torch.cat([observations, type_features], dim=-1))
        mask = agent_mask.to(observations.dtype).unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.value_head(pooled).squeeze(-1)


class HeterogeneousMappo(nn.Module):
    """Two shared decentralized actors plus a centralized set critic."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.uav_actor = SharedActor(observation_dim, action_dim, hidden_dim)
        self.usv_actor = SharedActor(observation_dim, action_dim, hidden_dim)
        self.critic = DeepSetsCritic(observation_dim, hidden_dim)

    def logits(
        self,
        observations: torch.Tensor,
        agent_types: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observations.ndim != 3:
            raise ValueError("observations must have shape [batch, agents, obs_dim]")
        batch, agents, obs_dim = observations.shape
        flat_obs = observations.reshape(batch * agents, obs_dim)
        flat_types = agent_types.reshape(batch * agents)
        logits = torch.empty(batch * agents, self.action_dim, dtype=observations.dtype, device=observations.device)
        uav_mask = flat_types == 0
        usv_mask = flat_types == 1
        if uav_mask.any():
            logits[uav_mask] = self.uav_actor(flat_obs[uav_mask])
        if usv_mask.any():
            logits[usv_mask] = self.usv_actor(flat_obs[usv_mask])
        return logits.reshape(batch, agents, self.action_dim)

    def value(self, observations: torch.Tensor, agent_types: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        return self.critic(observations, agent_types, agent_mask)

    def act(
        self,
        observations: torch.Tensor,
        agent_types: torch.Tensor,
        action_masks: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logits(observations, agent_types, agent_mask)
        logits = _mask_logits(logits, action_masks)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions) * agent_mask.to(logits.dtype)
        values = self.value(observations, agent_types, agent_mask)
        return actions, log_probs, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        agent_types: torch.Tensor,
        action_masks: torch.Tensor,
        agent_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logits(observations, agent_types, agent_mask)
        logits = _mask_logits(logits, action_masks)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions) * agent_mask.to(logits.dtype)
        entropy = dist.entropy() * agent_mask.to(logits.dtype)
        values = self.value(observations, agent_types, agent_mask)
        return log_probs, entropy, values


class SharedPolicyMappo(nn.Module):
    """One decentralized actor shared by UAV and USV agents plus a centralized set critic."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.actor = SharedActor(observation_dim, action_dim, hidden_dim)
        self.critic = DeepSetsCritic(observation_dim, hidden_dim)

    def logits(
        self,
        observations: torch.Tensor,
        agent_types: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observations.ndim != 3:
            raise ValueError("observations must have shape [batch, agents, obs_dim]")
        batch, agents, obs_dim = observations.shape
        flat_logits = self.actor(observations.reshape(batch * agents, obs_dim))
        return flat_logits.reshape(batch, agents, self.action_dim)

    def value(self, observations: torch.Tensor, agent_types: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        return self.critic(observations, agent_types, agent_mask)

    def act(
        self,
        observations: torch.Tensor,
        agent_types: torch.Tensor,
        action_masks: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logits(observations, agent_types, agent_mask)
        logits = _mask_logits(logits, action_masks)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions) * agent_mask.to(logits.dtype)
        values = self.value(observations, agent_types, agent_mask)
        return actions, log_probs, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        agent_types: torch.Tensor,
        action_masks: torch.Tensor,
        agent_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logits(observations, agent_types, agent_mask)
        logits = _mask_logits(logits, action_masks)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions) * agent_mask.to(logits.dtype)
        entropy = dist.entropy() * agent_mask.to(logits.dtype)
        values = self.value(observations, agent_types, agent_mask)
        return log_probs, entropy, values


class PooledContextActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.agent_encoder = nn.Sequential(
            nn.Linear(observation_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        observations: torch.Tensor,
        agent_types: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        type_features = torch.stack([agent_types == 0, agent_types == 1], dim=-1).to(observations.dtype)
        encoded = self.agent_encoder(torch.cat([observations, type_features], dim=-1))
        if agent_mask is None:
            agent_mask = torch.ones(observations.shape[:2], dtype=torch.bool, device=observations.device)
        mask = agent_mask.to(observations.dtype).unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        context = self.context_encoder(pooled).unsqueeze(1).expand(-1, observations.shape[1], -1)
        return self.head(torch.cat([encoded, context], dim=-1))


class CentralizedPpo(nn.Module):
    """A centralized-context PPO candidate for comparing against decentralized actor variants."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.actor = PooledContextActor(observation_dim, action_dim, hidden_dim)
        self.critic = DeepSetsCritic(observation_dim, hidden_dim)

    def logits(
        self,
        observations: torch.Tensor,
        agent_types: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observations.ndim != 3:
            raise ValueError("observations must have shape [batch, agents, obs_dim]")
        return self.actor(observations, agent_types, agent_mask)

    def value(self, observations: torch.Tensor, agent_types: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        return self.critic(observations, agent_types, agent_mask)

    def act(
        self,
        observations: torch.Tensor,
        agent_types: torch.Tensor,
        action_masks: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logits(observations, agent_types, agent_mask)
        logits = _mask_logits(logits, action_masks)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions) * agent_mask.to(logits.dtype)
        values = self.value(observations, agent_types, agent_mask)
        return actions, log_probs, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        agent_types: torch.Tensor,
        action_masks: torch.Tensor,
        agent_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logits(observations, agent_types, agent_mask)
        logits = _mask_logits(logits, action_masks)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions) * agent_mask.to(logits.dtype)
        entropy = dist.entropy() * agent_mask.to(logits.dtype)
        values = self.value(observations, agent_types, agent_mask)
        return log_probs, entropy, values


def _mask_logits(logits: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    if logits.shape != action_masks.shape:
        raise ValueError(f"mask shape {tuple(action_masks.shape)} does not match logits {tuple(logits.shape)}")
    return logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)
