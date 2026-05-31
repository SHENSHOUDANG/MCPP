"""PPO 使用的 actor-critic 神经网络与图注意力通信模块。

这里没有环境规则：网络只接收 ``env.py`` 构造的向量/张量。actor 为每个
agent 输出动作分布；critic 在训练时可接收全局状态估计价值。可选 GAT
只在通信邻接允许的 agent 之间聚合消息。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .runtime import configure_runtime

configure_runtime()

import torch
from torch import nn
from torch.distributions import Categorical


class GraphAttentionBlock(nn.Module):
    """多头 masked 图注意力层。

    输入形状为 ``[batch, agents, hidden]``。``neighbor_mask`` 指定每个
    agent 可以关注谁；因此远离通信范围的 agent 不会通过该层交换特征。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        edge_dim: int = 0,
        residual: bool = False,
        attention_dropout: float = 0.0,
    ) -> None:
        """建立 query/key/value 投影以及可选的相对几何边偏置。"""

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
        """按通信图汇总邻居特征并返回每个 agent 的新隐藏表示。"""

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
        # 每个注意力头分别计算 source 对 target 的相似度。
        query = self._split_heads(self.query(features), batch_size, num_agents)
        key = self._split_heads(self.key(features), batch_size, num_agents)
        value = self._split_heads(self.value(features), batch_size, num_agents)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.edge_bias is not None and edge_features is not None:
            edge_bias = self.edge_bias(edge_features).permute(0, 3, 1, 2)
            scores = scores + edge_bias
        # 被 mask 的连接在 softmax 前压到极小值，从而不贡献消息。
        scores = scores.masked_fill(~neighbor_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        self.last_attention_weights = attention.detach()
        context = torch.matmul(self.attention_dropout(attention), value)
        context = context.transpose(1, 2).contiguous().reshape(batch_size, num_agents, self.hidden_dim)
        output = self.output(torch.cat([features, context], dim=-1))
        return features + output if self.residual else output

    def _split_heads(self, tensor: torch.Tensor, batch_size: int, num_agents: int) -> torch.Tensor:
        """将隐藏维度拆成多头格式 ``[batch, heads, agents, head_dim]``。"""

        return tensor.reshape(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)


class ActorCritic(nn.Module):
    """PPO 的共享 actor 与集中式 critic。

    所有 agent 共用 actor 参数。actor 只能读取个体观测及允许的消息；
    spatial critic 可以在训练时读取环境提供的全局图层，这是 CTDE
    （集中训练、分散执行）设定中的有意设计。
    """

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
        actor_encoder: str = "mlp",
        actor_map_shape: tuple[int, int, int] | None = None,
        actor_metadata_dim: int = 0,
    ) -> None:
        """根据输入形状和通信开关组装 actor、critic 及可选 GAT。"""

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
        self.actor_encoder = str(actor_encoder).lower()
        self.actor_map_shape = actor_map_shape
        self.actor_metadata_dim = max(int(actor_metadata_dim), 0)
        self._uses_spatial_state = state_shape is not None
        if self.actor_encoder == "cnn":
            if actor_map_shape is None:
                raise ValueError("actor_map_shape is required when actor_encoder='cnn'")
            actor_channels, actor_height, actor_width = actor_map_shape
            actor_map_dim = int(actor_channels) * int(actor_height) * int(actor_width)
            if actor_map_dim + self.actor_metadata_dim != observation_dim:
                raise ValueError(
                    "actor_map_shape and actor_metadata_dim must match observation_dim "
                    f"({actor_map_dim} + {self.actor_metadata_dim} != {observation_dim})"
                )
            self.actor_map_dim = actor_map_dim
            self.actor_spatial_encoder = nn.Sequential(
                nn.Conv2d(actor_channels, hidden_dim, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            actor_body_input_dim = hidden_dim + self.actor_metadata_dim
        else:
            self.actor_encoder = "mlp"
            self.actor_map_dim = 0
            self.actor_spatial_encoder = None
            actor_body_input_dim = observation_dim
        # actor_body 独立编码每个 agent 的私有观测。
        self.actor_body = nn.Sequential(
            nn.Linear(actor_body_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # coverage message 与局部观测先各自编码，避免原始尺度混在一起。
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
        # 开启 GAT 时，只对节点隐藏特征做通信范围内的信息聚合。
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
            # 新版 critic 把全局状态还原成地图通道，再通过卷积提取空间特征。
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
            # legacy 分支保留对旧扁平 critic checkpoint 的加载兼容性。
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
        """返回 checkpoint/日志中使用的 critic 表示类型。"""

        return "spatial" if self._uses_spatial_state else "legacy"

    def _actor_features(
        self,
        observations: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        node_messages: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """编码 actor 输入，并按需要融合消息与图注意力结果。

        ``[agents, observation_dim]`` 用于单个环境步；
        ``[batch, agents, observation_dim]`` 用于 PPO 批量回放更新。
        """

        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if observations.ndim == 2:
            # 在线采样路径：同一时刻的一组 agent 共享一张通信图。
            features = self._encode_actor_observations(observations)
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
            # 训练路径：批次中每条 transition 都保留对应的 agent 维度。
            batch_size, num_agents, observation_dim = observations.shape
            features = self._encode_actor_observations(observations.reshape(-1, observation_dim)).reshape(
                batch_size, num_agents, -1
            )
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

    def _encode_actor_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode flattened actor observations with either the MLP or CNN actor path."""

        if self.actor_spatial_encoder is None:
            return self.actor_body(observations)
        if self.actor_map_shape is None:
            raise ValueError("actor_map_shape is required for the CNN actor encoder")
        channels, height, width = self.actor_map_shape
        spatial = observations[:, : self.actor_map_dim].reshape(-1, channels, height, width)
        metadata = observations[:, self.actor_map_dim :]
        encoded = self.actor_spatial_encoder(spatial)
        return self.actor_body(torch.cat([encoded, metadata], dim=-1))

    def _critic_features(self, states: torch.Tensor) -> torch.Tensor:
        """将全局状态编码成 critic 的隐藏特征。"""

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

        # env 将地图通道与少量统计元数据拼在一起；此处拆开再分别编码。
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
        node_messages: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """同时计算 actor 动作 logits 与 critic 的状态价值。"""

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
        logits = self.actor(actor_features)
        values = self.critic(self._critic_features(states)).squeeze(-1)
        # The centralized state is shared at each environment step. When callers
        # provide it once per step, avoid repeating identical critic work per agent.
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
        action_mask: torch.Tensor | None = None,
    ) -> Categorical:
        """只构造 actor 的离散动作概率分布。"""

        logits = self.actor(
            self._actor_features(
                observations,
                neighbor_mask=neighbor_mask,
                edge_features=edge_features,
                node_messages=node_messages,
            )
        )
        return Categorical(logits=self._mask_action_logits(logits, action_mask))

    @staticmethod
    def _mask_action_logits(logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        """Remove only environment-declared infeasible actions before sampling."""

        if action_mask is None:
            return logits
        mask = action_mask.to(device=logits.device, dtype=torch.bool)
        if mask.ndim + 1 == logits.ndim and logits.shape[0] == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != logits.shape:
            raise ValueError(f"action mask shape {tuple(mask.shape)} does not match logits shape {tuple(logits.shape)}")
        has_available_action = mask.any(dim=-1, keepdim=True)
        effective_mask = torch.where(has_available_action, mask, torch.ones_like(mask))
        return logits.masked_fill(~effective_mask, torch.finfo(logits.dtype).min)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        """只计算 critic 价值，常用于 rollout 最后一步的 bootstrap。"""

        if states.ndim == 1:
            states = states.unsqueeze(0)
        return self.critic(self._critic_features(states)).squeeze(-1)

    def act(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        node_messages: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """兼容单 agent 调用的动作采样包装。"""

        actions, log_probs, values = self.act_batch(
            observation,
            state,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            node_messages=node_messages,
            action_mask=action_mask,
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
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为一组 agent 采样动作；评估时可选择确定性最大概率动作。"""

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
        logits = self._mask_action_logits(logits, action_mask)
        dist = Categorical(logits=logits)
        # 训练需要随机采样维持探索，评估则固定选取最高概率动作。
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """重新评估 rollout 中的既有动作，供 PPO 计算 ratio 和 entropy。"""

        logits, values = self.forward(
            observations,
            states,
            neighbor_mask=neighbor_mask,
            edge_features=edge_features,
            node_messages=node_messages,
        )
        dist = Categorical(logits=self._mask_action_logits(logits, action_mask))
        return dist.log_prob(actions), dist.entropy(), values

    def latest_attention_weights(self) -> torch.Tensor | None:
        """返回最近一次前向计算中的注意力权重，用于诊断通信行为。"""

        return None if self.graph_attention is None else self.graph_attention.last_attention_weights


@dataclass(slots=True)
class RolloutBatch:
    """一次 PPO 更新使用的轨迹批次及可选通信输入。"""

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
