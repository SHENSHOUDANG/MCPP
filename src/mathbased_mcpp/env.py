"""网格覆盖任务环境。

本模块同时服务单智能体 smoke 测试和多智能体 MAPPO 训练。理解代码时要
区分两种状态：

* ``covered``、``free_cells`` 等是环境真值，可供奖励、评价和集中式 critic 使用；
* ``known_*_by_agent`` 是每个 agent 通过局部感知或通信得到的知识，只有
  这些内容才能构成新的去中心化 actor 观测。

动作编号对应上下左右移动，坐标统一使用 ``(row, column)``。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any, Sequence

import numpy as np

from .config import GridCoverageConfig, GridPosition


ACTIONS: dict[int, GridPosition] = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

ACTION_BY_DELTA: dict[GridPosition, int] = {delta: action for action, delta in ACTIONS.items()}


@dataclass(slots=True)
class StepResult:
    """环境执行一步后交给训练/评估代码的完整结果。"""

    observation: np.ndarray
    state: np.ndarray
    reward: float | np.ndarray
    done: bool
    info: dict[str, Any]


class GridCoverageEnv:
    """支持单 agent 兼容模式与 MAPPO 多 agent 模式的覆盖环境。

    环境内部仍保存全局地图以判断奖励和完成状态，但 actor 的新观测路径
    通过私有/融合记忆构造，避免直接泄漏团队真实覆盖情况。
    """

    # 观测由若干局部地图通道展平后再拼接固定长度的标量元数据。
    observation_channels = 6
    legacy_observation_channels = 7
    explicit_observation_channels = 9
    observation_metadata_dim = 12
    state_channels = 5
    state_metadata_dim = 7
    neighbor_feature_dim = 4
    coverage_message_base_dim = 15

    def __init__(self, config: GridCoverageConfig) -> None:
        """创建环境容器；真正的回合初值在 ``reset`` 中重新建立。"""

        self.config = config
        self.random = random.Random(config.seed)
        self.start_position: GridPosition = config.start
        self.start_positions: list[GridPosition] = []
        self._row_flip = 1
        self._col_flip = 1
        self.obstacles: set[GridPosition] = set()
        self.free_cells: set[GridPosition] = set()
        self.covered: set[GridPosition] = set()  # 全团队真实已覆盖格，仅供环境侧使用。
        self.positions: list[GridPosition] = [config.start]
        self.position: GridPosition = config.start
        self.teammate_positions: list[GridPosition] = list(config.teammate_positions)
        self.paths: list[list[GridPosition]] = [[config.start]]
        self.path: list[GridPosition] = []
        self.covered_by_agent: list[set[GridPosition]] = [{config.start}]
        # 下列三组 known 集合是 actor 允许看到的知识边界。
        self.known_free_by_agent: list[set[GridPosition]] = [set()]
        self.known_obstacles_by_agent: list[set[GridPosition]] = [set()]
        self.known_team_covered_by_agent: list[set[GridPosition]] = [{config.start}]
        self._node_message_cache: list[np.ndarray | None] = [None]
        self.last_novel_step_by_agent: list[int] = [0]
        self.path_lengths: list[int] = [0]
        self.path_length = 0
        self.step_count = 0
        self.reset_count = 0
        self.done = False
        self.last_blocked_cells: set[GridPosition] = set()
        self.last_new_cells = 0
        self.last_collision_agents = 0
        self._build_map()
        self._sync_legacy_aliases()

    @property
    def num_agents(self) -> int:
        """返回有效 agent 数量，并保证至少有一个 agent。"""

        return max(int(self.config.num_agents), 1)

    @property
    def action_dim(self) -> int:
        """动作空间大小：上下左右四个离散方向。"""

        return len(ACTIONS)

    @property
    def observation_dim(self) -> int:
        """一个 actor 的扁平观测向量长度。"""

        _, window_size, _ = self.actor_map_shape
        channel_dim = window_size * window_size * self.active_observation_channels
        return channel_dim + self.observation_metadata_dim

    @property
    def actor_map_shape(self) -> tuple[int, int, int]:
        """Return ``(channels, height, width)`` for the actor's spatial map block."""

        if self._uses_centered_memory_observation():
            window_size = self._centered_map_size()
        else:
            radius = max(self.config.observation_radius, 0)
            window_size = radius * 2 + 1
        return self.active_observation_channels, window_size, window_size

    @property
    def active_observation_channels(self) -> int:
        """根据兼容/记忆开关选择 actor 观测的地图通道数量。"""

        if self.config.use_explicit_map_memory:
            return self.explicit_observation_channels
        if self.config.use_legacy_truth_coverage_observation:
            return self.legacy_observation_channels
        return self.observation_channels

    @property
    def node_message_dim(self) -> int:
        """每个 agent 发给 GAT 的覆盖意图消息长度。"""

        return self.coverage_message_base_dim + max(self.config.intent_grid_size, 1) ** 2

    @property
    def state_dim(self) -> int:
        """集中式 critic 输入的扁平全局状态长度。"""

        return self.config.height * self.config.width * self.state_channels + self.state_metadata_dim

    def reset(self, seed: int | None = None) -> np.ndarray:
        """开始一个新回合，并返回每个 agent 的初始观测。

        提供 ``seed`` 时会从头复现实验地图序列；无 seed 的多次 reset 可按
        配置的 seed 池轮换地图，以便课程训练接触多种布局。
        """

        if seed is not None:
            self.random.seed(seed)
            self.reset_count = 0
        self._build_map()
        self.reset_count += 1
        self.positions = self._select_start_positions()
        self._configure_orientation(self.positions[0])
        self.teammate_positions = self._valid_teammate_positions(self.config.teammate_positions)
        # 起始所在格在回合开始时就算已覆盖。
        self.covered = set(self.positions)
        self.covered_by_agent = [{position} for position in self.positions]
        self.known_free_by_agent = [set() for _ in self.positions]
        self.known_obstacles_by_agent = [set() for _ in self.positions]
        self.known_team_covered_by_agent = [{position} for position in self.positions]
        self.last_novel_step_by_agent = [0 for _ in self.positions]
        self.paths = [[position] for position in self.positions]
        self.path_lengths = [0 for _ in self.positions]
        self.step_count = 0
        self.done = False
        self.last_blocked_cells = set()
        self.last_new_cells = 0
        self.last_collision_agents = 0
        self._sync_legacy_aliases()
        # 首次局部感知发生在返回 actor 观测之前。
        self._refresh_explicit_map_memory()
        observations = self._observations()
        return observations[0] if self.num_agents == 1 else observations

    def reset_preview(self) -> np.ndarray:
        """预览下一次 reset 观测，但在结束后恢复当前回合的全部状态。"""

        positions = list(self.positions)
        start_position = self.start_position
        start_positions = list(self.start_positions)
        row_flip = self._row_flip
        col_flip = self._col_flip
        covered = set(self.covered)
        paths = [list(path) for path in self.paths]
        teammate_positions = list(self.teammate_positions)
        path_lengths = list(self.path_lengths)
        step_count = self.step_count
        reset_count = self.reset_count
        done = self.done
        last_blocked_cells = set(self.last_blocked_cells)
        last_new_cells = self.last_new_cells
        last_collision_agents = self.last_collision_agents
        covered_by_agent = [set(cells) for cells in self.covered_by_agent]
        known_free_by_agent = [set(cells) for cells in self.known_free_by_agent]
        known_obstacles_by_agent = [set(cells) for cells in self.known_obstacles_by_agent]
        known_team_covered_by_agent = [set(cells) for cells in self.known_team_covered_by_agent]
        node_message_cache = [None if message is None else message.copy() for message in self._node_message_cache]
        last_novel_step_by_agent = list(self.last_novel_step_by_agent)
        obstacles = set(self.obstacles)
        free_cells = set(self.free_cells)
        obs = self.reset()
        self.positions = positions
        self.start_position = start_position
        self.start_positions = start_positions
        self._row_flip = row_flip
        self._col_flip = col_flip
        self.covered = covered
        self.covered_by_agent = covered_by_agent
        self.known_free_by_agent = known_free_by_agent
        self.known_obstacles_by_agent = known_obstacles_by_agent
        self.known_team_covered_by_agent = known_team_covered_by_agent
        self._node_message_cache = node_message_cache
        self.last_novel_step_by_agent = last_novel_step_by_agent
        self.paths = paths
        self.teammate_positions = teammate_positions
        self.path_lengths = path_lengths
        self.step_count = step_count
        self.reset_count = reset_count
        self.done = done
        self.last_blocked_cells = last_blocked_cells
        self.last_new_cells = last_new_cells
        self.last_collision_agents = last_collision_agents
        self.obstacles = obstacles
        self.free_cells = free_cells
        self._sync_legacy_aliases()
        return obs

    def step(self, action: int | Sequence[int]) -> StepResult:
        """执行一个时间步，并在单/多智能体逻辑之间分发。"""

        scalar_action = isinstance(action, (int, np.integer))
        actions = [int(action)] if scalar_action else [int(item) for item in action]
        if len(actions) != self.num_agents:
            raise ValueError(f"expected {self.num_agents} actions, got {len(actions)}")
        if self.done:
            reward: float | np.ndarray = 0.0 if scalar_action else np.zeros(self.num_agents, dtype=np.float32)
            layers = self._canonical_layers()
            observation = self._observations(layers)
            return StepResult(observation[0] if scalar_action else observation, self._global_state_from_layers(layers), reward, True, self._info({}))
        if self.num_agents == 1:
            return self._step_single(actions[0], scalar_action=scalar_action)
        return self._step_multi(actions)

    def global_state(self) -> np.ndarray:
        """返回集中式 critic 使用的全局真值状态。"""

        return self._global_state_from_layers(self._canonical_layers())

    def _global_state_from_layers(self, layers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """将全局地图层和团队统计量拼接成 critic 输入向量。"""

        all_agents, uncovered, team_covered, obstacles, blocked = layers
        metadata = np.array(
            [
                self.coverage_ratio(),
                self.step_count / max(self.config.max_steps, 1),
                self._agent_density(),
                float(np.mean(self.path_lengths)) / max(self.config.max_steps, 1),
                float(max(self.path_lengths, default=0)) / max(self.config.max_steps, 1),
                self.last_new_cells / max(self.num_agents, 1),
                self.last_collision_agents / max(self.num_agents, 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                all_agents.ravel(),
                uncovered.ravel(),
                team_covered.ravel(),
                obstacles.ravel(),
                blocked.ravel(),
                metadata,
            ]
        )

    def set_teammate_positions(self, positions: list[GridPosition]) -> None:
        """设置额外静态队友位置，供兼容旧单 agent 行为使用。"""

        self.teammate_positions = self._valid_teammate_positions(positions)

    def peek(self, action: int, agent_index: int = 0) -> tuple[GridPosition, bool]:
        """只检查一个动作的目标与合法性，不真正修改环境。"""

        actual_action = self._actual_action(action)
        position = self.positions[agent_index]
        if actual_action is None:
            return position, False
        delta = ACTIONS[actual_action]
        target = (position[0] + delta[0], position[1] + delta[1])
        return target, self.is_free(target)

    def is_free(self, position: GridPosition) -> bool:
        """判断格子是否在边界内且不是障碍。"""

        row, col = position
        return 0 <= row < self.config.height and 0 <= col < self.config.width and position not in self.obstacles

    def is_row_complete(self, row: int, extra_cell: GridPosition | None = None) -> bool:
        """判断规范坐标下的一整行可通行格是否已经覆盖。"""

        canonical_row = self._canonical_row(row)
        covered = self._canonical_covered_cells()
        if extra_cell is not None:
            covered.add(self._canonical_position(extra_cell))
        row_cells = {
            cell
            for cell in self._canonical_free_cells()
            if cell[0] == canonical_row
        }
        return bool(row_cells) and row_cells <= covered

    def row_direction(self, row: int) -> int:
        """返回犁式（boustrophedon）扫掠在该行的推荐横向方向。"""

        return 1 if self._canonical_row(row) % 2 == 0 else -1

    def legal_actions(self, agent_index: int = 0) -> list[int]:
        """列出不会越界或撞障碍的动作。"""

        return [action for action in ACTIONS if self.peek(action, agent_index=agent_index)[1]]

    def safe_actions(self, agent_index: int = 0) -> list[int]:
        """在合法动作中进一步排除过于靠近障碍的动作。"""

        actions = []
        for action in self.legal_actions(agent_index=agent_index):
            target, _ = self.peek(action, agent_index=agent_index)
            if not self.is_dangerous(target):
                actions.append(action)
        return actions

    def action_mask(self) -> np.ndarray:
        """Return decentralized feasibility masks for policy action sampling.

        Bounds are always known. Obstacle cells are masked only after an agent
        has observed them locally or received them through explicit map fusion.
        The mask deliberately does not predict simultaneous agent collisions.
        """

        masks = np.ones((self.num_agents, self.action_dim), dtype=bool)
        radius = max(self.config.observation_radius, 0)
        for agent_index, position in enumerate(self.positions):
            if self.config.use_explicit_map_memory:
                known_obstacles = self.known_obstacles_by_agent[agent_index]
            else:
                known_obstacles = {
                    obstacle
                    for obstacle in self.obstacles
                    if abs(obstacle[0] - position[0]) <= radius and abs(obstacle[1] - position[1]) <= radius
                }
            for action in ACTIONS:
                actual_action = self._actual_action(action)
                if actual_action is None:
                    masks[agent_index, action] = False
                    continue
                delta = ACTIONS[actual_action]
                target = (position[0] + delta[0], position[1] + delta[1])
                inside_bounds = 0 <= target[0] < self.config.height and 0 <= target[1] < self.config.width
                masks[agent_index, action] = inside_bounds and target not in known_obstacles
        return masks

    def is_dangerous(self, position: GridPosition) -> bool:
        """按曼哈顿距离判断位置是否落在障碍危险半径内。"""

        if self.config.danger_radius <= 0:
            return False
        for obstacle in self.obstacles:
            if abs(position[0] - obstacle[0]) + abs(position[1] - obstacle[1]) <= self.config.danger_radius:
                return True
        return False

    def coverage_ratio(self) -> float:
        """团队真实覆盖格数占全部可通行格数的比例。"""

        return len(self.covered) / max(len(self.free_cells), 1)

    def neighbor_mask(self) -> np.ndarray:
        """为通信/GAT 生成邻接矩阵；自身节点始终可见。"""

        mask = np.eye(self.num_agents, dtype=bool)
        radius = max(self.config.communication_radius, 0)
        if radius <= 0:
            return mask
        for first in range(self.num_agents):
            for second in range(first + 1, self.num_agents):
                if self._manhattan(self.positions[first], self.positions[second]) <= radius:
                    mask[first, second] = True
                    mask[second, first] = True
        return mask

    def neighbor_features(self) -> np.ndarray:
        """生成 agent 对之间的归一化距离、相对位移与连通标志。"""

        features = np.zeros((self.num_agents, self.num_agents, self.neighbor_feature_dim), dtype=np.float32)
        radius = max(self.config.communication_radius, 0)
        distance_scale = max(radius, 1)
        row_scale = max(self.config.height - 1, 1)
        col_scale = max(self.config.width - 1, 1)
        for source, source_position in enumerate(self.positions):
            for target, target_position in enumerate(self.positions):
                distance = self._manhattan(source_position, target_position)
                connected = source == target or (radius > 0 and distance <= radius)
                features[source, target, 0] = min(distance / distance_scale, 1.0)
                features[source, target, 1] = (target_position[0] - source_position[0]) / row_scale
                features[source, target, 2] = (target_position[1] - source_position[1]) / col_scale
                features[source, target, 3] = 1.0 if connected else 0.0
        return features

    def _step_single(self, action: int, scalar_action: bool) -> StepResult:
        """执行单 agent 动作；奖励是统一团队公式在 ``J = 1`` 时的特例。"""

        previous_position = self.position
        previous_covered = set(self.covered)
        # frontier 权重为零时跳过 BFS 距离搜索，节省训练时的环境开销。
        use_frontier_progress = self.config.reward.team_frontier_weight != 0.0
        before_distance = self._distance_to_nearest_uncovered(previous_position, previous_covered) if use_frontier_progress else None
        target, valid = self.peek(action)

        self.step_count += 1
        repeated = False
        straight_moves = 0
        if valid:
            straight_moves = int(self._continues_previous_direction(0, previous_position, target))
            self_novel = target not in self.covered_by_agent[0]
            self.positions[0] = target
            self.path_lengths[0] += abs(target[0] - previous_position[0]) + abs(target[1] - previous_position[1])
            self.paths[0].append(target)
            self.covered_by_agent[0].add(target)
            if self_novel:
                self.last_novel_step_by_agent[0] = self.step_count
            repeated = target in self.covered
            self.covered.add(target)
            self.last_new_cells = 0 if repeated else 1
            self.last_blocked_cells = set()
        else:
            self.last_new_cells = 0
            self.last_blocked_cells = {target}
            self.paths[0].append(previous_position)

        completed = self.covered >= self.free_cells
        self.done = completed or self.step_count >= self.config.max_steps
        frontier_progress = 0.0
        if use_frontier_progress and valid:
            after_distance = self._distance_to_nearest_uncovered(self.positions[0], self.covered)
        else:
            after_distance = None
        if before_distance is not None and after_distance is not None:
            frontier_progress = float(
                np.clip((before_distance - after_distance) / max(self.config.height + self.config.width, 1), -1.0, 1.0)
            )
        new_cells = int(valid and not repeated)
        avoidable_repeats = int(valid and repeated and self._has_uncovered_neighbor(previous_position, previous_covered))
        reward, reward_terms = self._team_reward(
            new_cells=new_cells,
            straight_moves=straight_moves,
            frontier_progress=frontier_progress,
            avoidable_repeats=avoidable_repeats,
            invalid_moves=int(not valid),
            obstacle_or_boundary_invalid_moves=int(not valid),
            agent_collision_invalid_moves=0,
            completed=completed,
        )
        self.last_collision_agents = 0
        self._sync_legacy_aliases()
        self._refresh_explicit_map_memory()
        layers = self._canonical_layers()
        observation = self._observations(layers)
        step_reward: float | np.ndarray = reward if scalar_action else np.array([reward], dtype=np.float32)
        return StepResult(observation[0] if scalar_action else observation, self._global_state_from_layers(layers), step_reward, self.done, self._info(reward_terms))

    def _step_multi(self, actions: list[int]) -> StepResult:
        """同步执行团队动作，处理冲突后返回共享团队奖励。"""

        previous_positions = list(self.positions)
        previous_covered = set(self.covered)
        use_frontier_progress = self.config.reward.team_frontier_weight != 0.0
        before_distances = (
            self._distances_to_nearest_uncovered(previous_positions, previous_covered)
            if use_frontier_progress
            else [None for _ in previous_positions]
        )
        targets: list[GridPosition] = []
        base_valid: list[bool] = []
        for index, action in enumerate(actions):
            target, valid = self.peek(action, agent_index=index)
            targets.append(target)
            base_valid.append(valid)

        obstacle_or_boundary_invalid_agents = {index for index, valid in enumerate(base_valid) if not valid}
        collision_agents: set[int] = set()
        target_to_agents: dict[GridPosition, list[int]] = {}
        for index, target in enumerate(targets):
            if index in obstacle_or_boundary_invalid_agents:
                continue
            target_to_agents.setdefault(target, []).append(index)
            for other_index, other_position in enumerate(previous_positions):
                if other_index != index and target == other_position:
                    collision_agents.add(index)
                    collision_agents.add(other_index)
        # 同一目标格、移入队友当前位置以及两两交换位置都视为碰撞。
        for agents in target_to_agents.values():
            if len(agents) > 1:
                collision_agents.update(agents)
        for first in range(self.num_agents):
            for second in range(first + 1, self.num_agents):
                if targets[first] == previous_positions[second] and targets[second] == previous_positions[first]:
                    collision_agents.add(first)
                    collision_agents.add(second)

        # 撞边界/障碍以及撞队友均为未执行成功的非法动作，使用同一处罚系数。
        invalid_agents = obstacle_or_boundary_invalid_agents | collision_agents
        blocked_agents = invalid_agents
        self.last_blocked_cells = {targets[index] for index in blocked_agents}
        final_positions = list(previous_positions)
        moved_agents: list[int] = []
        for index, target in enumerate(targets):
            if index in blocked_agents:
                continue
            final_positions[index] = target
            moved_agents.append(index)

        # 多个 agent 同步踏入同一新格会在前面的碰撞处理中被阻止，因此用集合计数。
        new_cells = {final_positions[index] for index in moved_agents if final_positions[index] not in previous_covered}
        repeated_cells = sum(1 for index in moved_agents if final_positions[index] in previous_covered)
        avoidable_repeats = sum(
            1
            for index in moved_agents
            if final_positions[index] in previous_covered and self._has_uncovered_neighbor(previous_positions[index], previous_covered)
        )
        straight_moves = sum(
            int(self._continues_previous_direction(index, previous_positions[index], final_positions[index]))
            for index in moved_agents
        )

        self.step_count += 1
        self.positions = final_positions
        for index in moved_agents:
            previous = previous_positions[index]
            target = final_positions[index]
            self.path_lengths[index] += abs(target[0] - previous[0]) + abs(target[1] - previous[1])
            self.paths[index].append(target)
            if target not in self.covered_by_agent[index]:
                self.last_novel_step_by_agent[index] = self.step_count
            self.covered_by_agent[index].add(target)
        for index in blocked_agents:
            self.paths[index].append(previous_positions[index])
        self.covered.update(new_cells)
        self.last_new_cells = len(new_cells)
        self.last_collision_agents = len(collision_agents)
        frontier_progress = 0.0
        if use_frontier_progress:
            after_distances = self._distances_to_nearest_uncovered(final_positions, self.covered)
            progress_values = [
                (before - after) / max(self.config.height + self.config.width, 1)
                for before, after in zip(before_distances, after_distances)
                if before is not None and after is not None
            ]
            frontier_progress = float(np.clip(np.mean(progress_values), -1.0, 1.0)) if progress_values else 0.0
        completed = self.covered >= self.free_cells
        reward, reward_terms = self._team_reward(
            new_cells=len(new_cells),
            straight_moves=straight_moves,
            frontier_progress=frontier_progress,
            avoidable_repeats=avoidable_repeats,
            invalid_moves=len(invalid_agents),
            obstacle_or_boundary_invalid_moves=len(obstacle_or_boundary_invalid_agents),
            agent_collision_invalid_moves=len(collision_agents),
            completed=completed,
        )
        reward_terms["repeated_cells"] = float(repeated_cells)
        self.done = completed or self.step_count >= self.config.max_steps
        self._sync_legacy_aliases()
        self._refresh_explicit_map_memory()
        layers = self._canonical_layers()
        rewards = np.full(self.num_agents, reward, dtype=np.float32)
        return StepResult(self._observations(layers), self._global_state_from_layers(layers), rewards, self.done, self._info(reward_terms))

    def _team_reward(
        self,
        *,
        new_cells: int,
        straight_moves: int,
        frontier_progress: float,
        avoidable_repeats: int,
        invalid_moves: int,
        obstacle_or_boundary_invalid_moves: int,
        agent_collision_invalid_moves: int,
        completed: bool,
    ) -> tuple[float, dict[str, float]]:
        """根据统一团队公式计算奖励，单 agent 自然对应 ``J = 1``。"""

        uncovered_ratio = 1.0 - self.coverage_ratio()
        time_cost_scale = uncovered_ratio if self.config.reward.scale_time_cost_by_uncovered else 1.0
        finish_team_total = self.config.reward.finish_reward if completed else 0.0
        finish_reward = finish_team_total
        if completed and self.config.reward.normalize_team_finish_reward:
            # 将完成看成一次团队事件，避免 agent 增多时同一终止奖金被重复放大。
            finish_reward = finish_team_total / self.num_agents
        straight_bonus = self.config.reward.team_straight_weight * straight_moves / self.num_agents
        reward_terms = {
            "new_cells": float(new_cells),
            "straight_moves": float(straight_moves),
            "straight_bonus": straight_bonus,
            "frontier_progress": frontier_progress,
            "avoidable_repeats": float(avoidable_repeats),
            "invalid_moves": float(invalid_moves),
            "obstacle_or_boundary_invalid_moves": float(obstacle_or_boundary_invalid_moves),
            "agent_collision_invalid_moves": float(agent_collision_invalid_moves),
            "collision_agents": float(agent_collision_invalid_moves),
            "uncovered_ratio": uncovered_ratio,
            "time_cost_scale": time_cost_scale,
            "time": -self.config.reward.team_time_weight * time_cost_scale,
            "repeat": -self.config.reward.team_repeat_weight * avoidable_repeats / self.num_agents,
            "invalid": -self.config.reward.team_invalid_weight * invalid_moves / self.num_agents,
            "finish": finish_reward,
            "finish_team_total": finish_team_total,
        }
        reward = float(
            self.config.reward.team_new_cell_weight * new_cells / self.num_agents
            + straight_bonus
            + self.config.reward.team_frontier_weight * frontier_progress
            + reward_terms["repeat"]
            + reward_terms["invalid"]
            + reward_terms["time"]
            + finish_reward
        )
        return reward, reward_terms

    def _continues_previous_direction(
        self,
        agent_index: int,
        previous_position: GridPosition,
        target: GridPosition,
    ) -> bool:
        """判断一次成功移动是否延续了该 agent 上一次真实位移方向。

        第一段移动还没有可比较方向，因此不奖励。若之前出现被阻挡导致
        的原地记录，则向前查找最近一次非零位移，以保持该偏好可学习。
        """

        current_vector = (target[0] - previous_position[0], target[1] - previous_position[1])
        if current_vector == (0, 0):
            return False
        path = self.paths[agent_index]
        for index in range(len(path) - 1, 0, -1):
            prior_vector = (
                path[index][0] - path[index - 1][0],
                path[index][1] - path[index - 1][1],
            )
            if prior_vector != (0, 0):
                return current_vector == prior_vector
        return False

    def _uncovered_neighbor_count(self, position: GridPosition) -> int:
        """统计指定位置四邻域中仍未被团队覆盖的自由格。"""

        count = 0
        for delta in ACTIONS.values():
            neighbor = (position[0] + delta[0], position[1] + delta[1])
            if self.is_free(neighbor) and neighbor not in self.covered:
                count += 1
        return count

    def _has_uncovered_neighbor(self, position: GridPosition, covered: set[GridPosition]) -> bool:
        """检查从当前位置是否本可直接前往尚未覆盖的相邻格。"""

        for delta in ACTIONS.values():
            neighbor = (position[0] + delta[0], position[1] + delta[1])
            if self.is_free(neighbor) and neighbor not in covered:
                return True
        return False

    def _distance_to_nearest_uncovered(self, start: GridPosition, covered: set[GridPosition]) -> int | None:
        """计算到最近未覆盖格的最短可通行步数。"""

        return self._distance_to_nearest_uncovered_from_set(start, self.free_cells - covered)

    def _distances_to_nearest_uncovered(self, starts: Sequence[GridPosition], covered: set[GridPosition]) -> list[int | None]:
        """为多个 agent 计算最近未覆盖距离，必要时复用距离场。"""

        uncovered = self.free_cells - covered
        if not uncovered:
            return [0 for _ in starts]
        if len(uncovered) <= 1 and len(starts) > 1:
            distance_field = self._distance_field_to_uncovered(covered)
            return [self._distance_from_field(distance_field, position) for position in starts]
        return [self._distance_to_nearest_uncovered_from_set(position, uncovered) for position in starts]

    def _distance_to_nearest_uncovered_from_set(self, start: GridPosition, uncovered: set[GridPosition]) -> int | None:
        """从一个起点执行 BFS，首次遇到目标未覆盖格即返回距离。"""

        if not uncovered:
            return 0
        if start in uncovered:
            return 0
        queue: deque[tuple[GridPosition, int]] = deque([(start, 0)])
        visited = {start}
        while queue:
            position, distance = queue.popleft()
            for delta in ACTIONS.values():
                neighbor = (position[0] + delta[0], position[1] + delta[1])
                if neighbor in visited or not self.is_free(neighbor):
                    continue
                if neighbor in uncovered:
                    return distance + 1
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return None

    def _distance_field_to_uncovered(self, covered: set[GridPosition]) -> np.ndarray:
        """从全部未覆盖格反向 BFS，一次生成整张最近距离场。"""

        uncovered = self.free_cells - covered
        if not uncovered:
            return np.zeros((self.config.height, self.config.width), dtype=np.int32)
        distances = np.full((self.config.height, self.config.width), -1, dtype=np.int32)
        queue: deque[GridPosition] = deque()
        for row, col in uncovered:
            distances[row, col] = 0
            queue.append((row, col))
        while queue:
            position = queue.popleft()
            distance = int(distances[position[0], position[1]])
            for delta in ACTIONS.values():
                neighbor = (position[0] + delta[0], position[1] + delta[1])
                if not self.is_free(neighbor) or distances[neighbor[0], neighbor[1]] >= 0:
                    continue
                distances[neighbor[0], neighbor[1]] = distance + 1
                queue.append(neighbor)
        return distances

    def _distance_from_field(self, distance_field: np.ndarray, position: GridPosition) -> int | None:
        """从距离场读取位置距离，负值代表不可达。"""

        distance = int(distance_field[position[0], position[1]])
        return None if distance < 0 else distance

    def _manhattan(self, first: GridPosition, second: GridPosition) -> int:
        """网格四连通移动下的曼哈顿距离。"""

        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def node_messages(self) -> np.ndarray:
        """返回 GAT 节点消息；仅显式记忆实验启用有效消息。"""

        if not self.config.use_explicit_map_memory:
            return np.zeros((self.num_agents, self.node_message_dim), dtype=np.float32)
        for index in range(self.num_agents):
            if self._node_message_cache[index] is None:
                self._node_message_cache[index] = self._coverage_message(index)
        return np.stack(self._node_message_cache).astype(np.float32)

    def _observations(self, layers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """批量构建所有 agent 的 actor 观测。"""

        if layers is None:
            layers = self._canonical_layers()
        return np.stack([self._observation(index, layers) for index in range(self.num_agents)]).astype(np.float32)

    def _observation(self, agent_index: int = 0, layers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """按实验兼容开关选择一种 actor 观测构造路径。

        legacy 分支显式保留给旧 checkpoint 重放，其中含全局团队覆盖信息；
        新实验应使用 private 或 explicit-memory 分支。
        """

        if layers is None:
            layers = self._canonical_layers()
        if self.config.use_explicit_map_memory:
            return self._explicit_memory_observation(agent_index, layers)
        if not self.config.use_legacy_truth_coverage_observation:
            return self._private_local_observation(agent_index, layers)
        all_agents, uncovered, team_covered, obstacles, _ = layers
        self_agent = np.zeros_like(all_agents)
        self_agent[self._canonical_position(self.positions[agent_index])] = 1.0
        other_agents = all_agents - self_agent
        self_covered = np.zeros_like(all_agents)
        for cell in self.covered_by_agent[agent_index]:
            self_covered[self._canonical_position(cell)] = 1.0
        recent_path = self._recent_path_layer(agent_index)

        radius = max(self.config.observation_radius, 0)
        center = self._canonical_position(self.positions[agent_index])
        channels = [
            self._local_window(self_agent, radius, center),
            self._local_window(other_agents, radius, center),
            self._local_window(uncovered, radius, center),
            self._local_window(team_covered, radius, center),
            self._local_window(obstacles, radius, center),
            self._local_window(self_covered, radius, center),
            self._local_window(recent_path, radius, center),
        ]

        row = center[0]
        metadata = np.array(
            [
                float(self.row_direction(row)),
                self.step_count / max(self.config.max_steps, 1),
                self.coverage_ratio(),
                self._agent_density(),
                *self._communication_metadata(agent_index),
            ],
            dtype=np.float32,
        )
        return np.concatenate([channel.ravel() for channel in channels] + [metadata])

    def _private_local_observation(
        self,
        agent_index: int,
        layers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """生成无通信记忆时的私有局部观测，不暴露队友覆盖历史。"""

        all_agents, _, _, obstacles, _ = layers
        self_agent = np.zeros_like(all_agents)
        self_agent[self._canonical_position(self.positions[agent_index])] = 1.0
        other_agents = all_agents - self_agent
        self_covered = self._cells_layer(self.covered_by_agent[agent_index])
        observed_free = np.ones_like(obstacles) - obstacles
        self_uncovered = np.clip(observed_free - self_covered, 0.0, 1.0)
        radius = max(self.config.observation_radius, 0)
        center = self._canonical_position(self.positions[agent_index])
        channels = [
            self._local_window(self_agent, radius, center),
            self._local_window(other_agents, radius, center),
            self._local_window(self_uncovered, radius, center),
            self._local_window(obstacles, radius, center),
            self._local_window(self_covered, radius, center),
            self._local_window(self._recent_path_layer(agent_index), radius, center),
        ]
        metadata = np.array(
            [
                float(self.row_direction(center[0])),
                self.step_count / max(self.config.max_steps, 1),
                len(self.covered_by_agent[agent_index]) / max(self.config.width * self.config.height, 1),
                self.num_agents / max(self.config.width * self.config.height, 1),
                *self._last_effective_move_vector(agent_index),
                *([0.0] * 6),
            ],
            dtype=np.float32,
        )
        return np.concatenate([channel.ravel() for channel in channels] + [metadata])

    def _explicit_memory_observation(
        self,
        agent_index: int,
        layers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """从 agent 已知地图生成显式记忆观测。

        ``known_team_covered`` 只包含自身轨迹或已经通信融合的队友信息；
        这里不会从环境真实 ``covered`` 集合读取未知团队覆盖。
        """

        all_agents, _, _, _, _ = layers
        self_agent = np.zeros_like(all_agents)
        self_agent[self._canonical_position(self.positions[agent_index])] = 1.0
        other_agents = all_agents - self_agent
        known_free = self.known_free_by_agent[agent_index]
        known_team_covered = self.known_team_covered_by_agent[agent_index]
        known_uncovered = known_free - known_team_covered
        known_obstacles = self.known_obstacles_by_agent[agent_index]
        unknown = self._memory_unknown(agent_index)
        frontier = self._memory_frontiers(agent_index, unknown)
        self_covered = self.covered_by_agent[agent_index]

        radius = max(self.config.observation_radius, 0)
        center = self._canonical_position(self.positions[agent_index])
        if self._uses_centered_memory_observation():
            teammates = self._known_teammates_layer(agent_index)
            window = self._centered_memory_window
        else:
            teammates = other_agents
            window = lambda grid, current_center: self._local_window(grid, radius, current_center)
        channels = [
            window(self_agent, center),
            window(teammates, center),
            window(self._cells_layer(known_uncovered), center),
            window(self._cells_layer(known_team_covered), center),
            window(self._cells_layer(known_obstacles), center),
            window(self._cells_layer(self_covered), center),
            window(self._recent_path_layer(agent_index), center),
            window(self._cells_layer(unknown), center),
            window(self._cells_layer(frontier), center),
        ]
        message = self._coverage_message(agent_index, unknown=unknown, frontier=frontier)
        self._node_message_cache[agent_index] = message
        metadata = np.array(
            [
                float(self.row_direction(center[0])),
                self.step_count / max(self.config.max_steps, 1),
                self._known_coverage_ratio(agent_index),
                len(known_free) / max(self.config.height * self.config.width, 1),
                *self._last_effective_move_vector(agent_index),
                *message[:6],
            ],
            dtype=np.float32,
        )
        return np.concatenate([channel.ravel() for channel in channels] + [metadata])

    def _refresh_explicit_map_memory(self) -> None:
        """用局部传感更新各自记忆，并在通信可达时融合已知地图。"""

        self._node_message_cache = [None for _ in self.positions]
        if not self.config.use_explicit_map_memory:
            return
        radius = max(self.config.observation_radius, 0)
        for agent_index, position in enumerate(self.positions):
            visible_cells = {
                (row, col)
                for row in range(max(position[0] - radius, 0), min(position[0] + radius + 1, self.config.height))
                for col in range(max(position[1] - radius, 0), min(position[1] + radius + 1, self.config.width))
            }
            # 全局真值只能通过这个局部传感窗口进入某个 agent 的私有记忆。
            self.known_obstacles_by_agent[agent_index].update(visible_cells & self.obstacles)
            self.known_free_by_agent[agent_index].update(visible_cells - self.obstacles)
            self.known_free_by_agent[agent_index].update(self.covered_by_agent[agent_index])
            self.known_team_covered_by_agent[agent_index].update(self.covered_by_agent[agent_index])

        if not self.config.share_map_memory:
            return
        # 使用更新前的快照做成对融合，避免单个时间步内沿通信链无限传播知识。
        mask = self.neighbor_mask()
        source_free = [set(cells) for cells in self.known_free_by_agent]
        source_obstacles = [set(cells) for cells in self.known_obstacles_by_agent]
        source_covered = [set(cells) for cells in self.known_team_covered_by_agent]
        for source in range(self.num_agents):
            for target in range(source + 1, self.num_agents):
                if not mask[source, target]:
                    continue
                merged_free = source_free[source] | source_free[target]
                merged_obstacles = source_obstacles[source] | source_obstacles[target]
                merged_covered = source_covered[source] | source_covered[target]
                self.known_free_by_agent[source].update(merged_free)
                self.known_free_by_agent[target].update(merged_free)
                self.known_obstacles_by_agent[source].update(merged_obstacles)
                self.known_obstacles_by_agent[target].update(merged_obstacles)
                self.known_team_covered_by_agent[source].update(merged_covered)
                self.known_team_covered_by_agent[target].update(merged_covered)

    def _coverage_message(
        self,
        agent_index: int,
        *,
        unknown: set[GridPosition] | None = None,
        frontier: set[GridPosition] | None = None,
    ) -> np.ndarray:
        """将已知覆盖进度、近期行为与探索意图编码为固定长度消息。"""

        known_free = self.known_free_by_agent[agent_index]
        known_team_covered = self.known_team_covered_by_agent[agent_index] & known_free
        self_covered = self.covered_by_agent[agent_index] & known_free
        unknown = self._memory_unknown(agent_index) if unknown is None else unknown
        frontier = self._memory_frontiers(agent_index, unknown) if frontier is None else frontier
        map_area = max(self.config.width * self.config.height, 1)
        known_free_count = max(len(known_free), 1)
        recent_new_ratio, recent_repeat_ratio = self._recent_self_coverage_rates(agent_index)
        stall_ratio = min(
            (self.step_count - self.last_novel_step_by_agent[agent_index]) / max(self.config.recent_path_length, 1),
            1.0,
        )
        target = self._intent_target(agent_index, frontier)
        direction = np.zeros(len(ACTIONS), dtype=np.float32)
        relative_target = np.zeros(2, dtype=np.float32)
        target_distance = 0.0
        regions = np.zeros(max(self.config.intent_grid_size, 1) ** 2, dtype=np.float32)
        intent_valid = 0.0
        if target is not None:
            # 消息只发送目标方向/区域摘要，不直接把整张记忆地图交给邻居。
            source_position = self._canonical_position(self.positions[agent_index])
            target_position = self._canonical_position(target)
            delta_row = target_position[0] - source_position[0]
            delta_col = target_position[1] - source_position[1]
            if abs(delta_row) >= abs(delta_col) and delta_row != 0:
                direction[1 if delta_row > 0 else 0] = 1.0
            elif delta_col != 0:
                direction[3 if delta_col > 0 else 2] = 1.0
            relative_target[:] = [
                delta_row / max(self.config.height - 1, 1),
                delta_col / max(self.config.width - 1, 1),
            ]
            target_distance = self._manhattan(source_position, target_position) / max(
                self.config.height + self.config.width - 2, 1
            )
            region_index = self._intent_region_index(target_position)
            regions[region_index] = 1.0
            intent_valid = 1.0
        return np.concatenate(
            [
                np.asarray(
                    [
                        len(known_team_covered) / known_free_count,
                        len(self_covered) / known_free_count,
                        len(unknown) / map_area,
                        len(frontier) / known_free_count,
                        recent_new_ratio,
                        recent_repeat_ratio,
                        stall_ratio,
                    ],
                    dtype=np.float32,
                ),
                direction,
                relative_target,
                np.asarray([target_distance], dtype=np.float32),
                regions,
                np.asarray([intent_valid], dtype=np.float32),
            ]
        )

    def _recent_self_coverage_rates(self, agent_index: int) -> tuple[float, float]:
        """统计最近路径中新格与重复格比例，作为是否停滞的提示。"""

        path = self.paths[agent_index]
        horizon = max(self.config.recent_path_length, 1)
        start = max(len(path) - horizon, 1)
        recent_moves = path[start:]
        if not recent_moves:
            return 0.0, 0.0
        seen = set(path[:start])
        novel = 0
        for cell in recent_moves:
            if cell not in seen:
                novel += 1
            seen.add(cell)
        repeat = len(recent_moves) - novel
        return novel / len(recent_moves), repeat / len(recent_moves)

    def _intent_target(self, agent_index: int, frontier: set[GridPosition]) -> GridPosition | None:
        """从已知未覆盖格或 frontier 中选择最近的消息目标。"""

        uncovered = self.known_free_by_agent[agent_index] - self.known_team_covered_by_agent[agent_index]
        candidates = uncovered if uncovered else frontier
        if not candidates:
            return None
        position = self.positions[agent_index]
        return min(
            candidates,
            key=lambda cell: (self._manhattan(position, cell), self._canonical_position(cell)),
        )

    def _intent_region_index(self, canonical_position: GridPosition) -> int:
        """将目标位置量化到粗网格区域，以固定维度发送意图。"""

        bins = max(self.config.intent_grid_size, 1)
        row = min(canonical_position[0] * bins // max(self.config.height, 1), bins - 1)
        col = min(canonical_position[1] * bins // max(self.config.width, 1), bins - 1)
        return row * bins + col

    def _known_coverage_ratio(self, agent_index: int) -> float:
        """计算该 agent 已知自由空间中的已知团队覆盖比例。"""

        known_free = self.known_free_by_agent[agent_index]
        if not known_free:
            return 0.0
        return len(self.known_team_covered_by_agent[agent_index] & known_free) / len(known_free)

    def _memory_unknown(self, agent_index: int) -> set[GridPosition]:
        """返回该 agent 尚未感知、也未通过通信获知的格子。"""

        known = self.known_free_by_agent[agent_index] | self.known_obstacles_by_agent[agent_index]
        return {
            (row, col)
            for row in range(self.config.height)
            for col in range(self.config.width)
            if (row, col) not in known
        }

    def _memory_frontiers(
        self,
        agent_index: int,
        unknown: set[GridPosition] | None = None,
    ) -> set[GridPosition]:
        """返回已知自由区与未知区域交界处的探索 frontier。"""

        unknown = self._memory_unknown(agent_index) if unknown is None else unknown
        return {
            cell
            for cell in self.known_free_by_agent[agent_index]
            if any((cell[0] + delta[0], cell[1] + delta[1]) in unknown for delta in ACTIONS.values())
        }

    def _cells_layer(self, cells: set[GridPosition]) -> np.ndarray:
        """将格子集合转换为按规范坐标排列的二值图层。"""

        layer = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        for cell in cells:
            layer[self._canonical_position(cell)] = 1.0
        return layer

    def _known_teammates_layer(self, agent_index: int) -> np.ndarray:
        """Encode only currently sensed or communication-reachable teammates."""

        layer = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        position = self.positions[agent_index]
        visibility_radius = max(self.config.observation_radius, self.config.communication_radius, 0)
        if visibility_radius <= 0:
            return layer
        for other_index, other_position in enumerate(self.positions):
            if other_index == agent_index:
                continue
            if self._manhattan(position, other_position) <= visibility_radius:
                layer[self._canonical_position(other_position)] = 1.0
        return layer

    def _uses_centered_memory_observation(self) -> bool:
        """Whether actor observations should use the map-size-invariant memory tensor."""

        return self.config.use_explicit_map_memory and self.config.observation_mode in {
            "centered_memory",
            "centered_compressed_memory",
        }

    def _centered_map_size(self) -> int:
        """Return a positive odd window size for centered memory observations."""

        size = max(int(self.config.centered_map_size), 3)
        if size % 2 == 0:
            size += 1
        return size

    def _centered_memory_window(self, grid: np.ndarray, center: GridPosition) -> np.ndarray:
        """Build a fixed-size agent-centered tensor with optional compressed borders.

        The high-resolution interior is a local crop around the agent. Cells outside
        that interior are summarized into the outer border, so larger remembered maps
        keep a fixed actor input size without reading hidden environment truth.
        """

        size = self._centered_map_size()
        if not self.config.compressed_border or size <= 3:
            return self._local_window(grid, size // 2, center)

        inner_size = size - 2
        inner_radius = inner_size // 2
        row, col = center
        window = np.zeros((size, size), dtype=grid.dtype)
        window[1:-1, 1:-1] = self._local_window(grid, inner_radius, center)

        row_min = max(row - inner_radius, 0)
        row_max = min(row + inner_radius + 1, grid.shape[0])
        col_min = max(col - inner_radius, 0)
        col_max = min(col + inner_radius + 1, grid.shape[1])

        inner_row_origin = row - inner_radius
        inner_col_origin = col - inner_radius
        for actual_col in range(col_min, col_max):
            target_col = 1 + actual_col - inner_col_origin
            if 1 <= target_col < size - 1:
                window[0, target_col] = self._region_mean(grid, 0, row_min, actual_col, actual_col + 1)
                window[-1, target_col] = self._region_mean(grid, row_max, grid.shape[0], actual_col, actual_col + 1)
        for actual_row in range(row_min, row_max):
            target_row = 1 + actual_row - inner_row_origin
            if 1 <= target_row < size - 1:
                window[target_row, 0] = self._region_mean(grid, actual_row, actual_row + 1, 0, col_min)
                window[target_row, -1] = self._region_mean(grid, actual_row, actual_row + 1, col_max, grid.shape[1])

        window[0, 0] = self._region_mean(grid, 0, row_min, 0, col_min)
        window[0, -1] = self._region_mean(grid, 0, row_min, col_max, grid.shape[1])
        window[-1, 0] = self._region_mean(grid, row_max, grid.shape[0], 0, col_min)
        window[-1, -1] = self._region_mean(grid, row_max, grid.shape[0], col_max, grid.shape[1])
        return window

    @staticmethod
    def _region_mean(grid: np.ndarray, row_start: int, row_stop: int, col_start: int, col_stop: int) -> float:
        """Return the density in a possibly empty rectangular region."""

        if row_stop <= row_start or col_stop <= col_start:
            return 0.0
        region = grid[row_start:row_stop, col_start:col_stop]
        if region.size == 0:
            return 0.0
        return float(region.mean())

    def _recent_path_layer(self, agent_index: int) -> np.ndarray:
        """将近期轨迹编码为越新越亮的衰减图层。"""

        layer = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        memory_length = max(self.config.recent_path_length, 1)
        recent = self.paths[agent_index][-memory_length:]
        for age, position in enumerate(reversed(recent)):
            value = (memory_length - age) / memory_length
            row, col = self._canonical_position(position)
            layer[row, col] = max(layer[row, col], float(value))
        return layer

    def _communication_metadata(self, agent_index: int) -> list[float]:
        """旧局部观测使用的邻居摘要统计量。"""

        radius = max(self.config.communication_radius, 0)
        if radius <= 0 or self.num_agents <= 1:
            return [0.0] * 8

        position = self.positions[agent_index]
        neighbors: list[int] = []
        distances: list[int] = []
        for other_index, other_position in enumerate(self.positions):
            if other_index == agent_index:
                continue
            distance = self._manhattan(position, other_position)
            if distance <= radius:
                neighbors.append(other_index)
                distances.append(distance)
        if not neighbors:
            return [0.0] * 8

        row_scale = max(self.config.height - 1, 1)
        col_scale = max(self.config.width - 1, 1)
        distance_scale = max(self.config.height + self.config.width - 2, 1)
        relative_rows = [(self.positions[index][0] - position[0]) / row_scale for index in neighbors]
        relative_cols = [(self.positions[index][1] - position[1]) / col_scale for index in neighbors]
        path_progress = [self.path_lengths[index] / max(self.config.max_steps, 1) for index in neighbors]
        frontier_density = [self._uncovered_neighbor_count(self.positions[index]) / max(len(ACTIONS), 1) for index in neighbors]
        intents = [self._last_move_vector(index) for index in neighbors]
        return [
            len(neighbors) / max(self.num_agents - 1, 1),
            float(np.mean(relative_rows)),
            float(np.mean(relative_cols)),
            min(distances) / distance_scale,
            float(np.mean(path_progress)),
            float(np.mean(frontier_density)),
            float(np.mean([intent[0] for intent in intents])),
            float(np.mean([intent[1] for intent in intents])),
        ]

    def _last_move_vector(self, agent_index: int) -> tuple[float, float]:
        """读取某 agent 最近一个时间步的二维位移，供旧邻居摘要使用。"""

        path = self.paths[agent_index]
        if len(path) < 2:
            return 0.0, 0.0
        previous = path[-2]
        current = path[-1]
        return float(current[0] - previous[0]), float(current[1] - previous[1])

    def _last_effective_move_vector(self, agent_index: int) -> tuple[float, float]:
        """读取最近一次有效移动的规范方向，供 actor 判断是否继续直行。"""

        path = self.paths[agent_index]
        for index in range(len(path) - 1, 0, -1):
            previous = self._canonical_position(path[index - 1])
            current = self._canonical_position(path[index])
            vector = float(current[0] - previous[0]), float(current[1] - previous[1])
            if vector != (0.0, 0.0):
                return vector
        return 0.0, 0.0

    def _canonical_layers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """构造规范方向下的全局真值图层。

        这些层直接用于 centralized critic 和评估；仅 legacy 分支会把其中
        的团队覆盖层送入 actor，因此新实验不得依赖那个分支。
        """

        all_agents = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        uncovered = np.zeros_like(all_agents)
        team_covered = np.zeros_like(all_agents)
        obstacles = np.zeros_like(all_agents)
        blocked = np.zeros_like(all_agents)

        for position in self.positions:
            if self.is_free(position):
                all_agents[self._canonical_position(position)] = 1.0
        for cell in self.free_cells - self.covered:
            uncovered[self._canonical_position(cell)] = 1.0
        for cell in self.covered:
            team_covered[self._canonical_position(cell)] = 1.0
        for cell in self.obstacles:
            obstacles[self._canonical_position(cell)] = 1.0
        for cell in self.last_blocked_cells:
            row, col = self._canonical_position(cell)
            if 0 <= row < self.config.height and 0 <= col < self.config.width:
                blocked[row, col] = 1.0

        return all_agents, uncovered, team_covered, obstacles, blocked

    def _info(self, reward_terms: dict[str, float]) -> dict[str, Any]:
        """组织日志和评估使用的可读回合信息。"""

        return {
            "position": self.position,
            "positions": list(self.positions),
            "coverage_ratio": self.coverage_ratio(),
            "covered_cells": len(self.covered),
            "free_cells": len(self.free_cells),
            "path_length": self.path_length,
            "path_lengths": list(self.path_lengths),
            "step_count": self.step_count,
            "completed": self.covered >= self.free_cells,
            "teammate_positions": list(self.teammate_positions),
            "reward_terms": reward_terms,
            "last_blocked_cells": list(self.last_blocked_cells),
        }

    def _build_map(self) -> None:
        """合并固定/随机障碍并确保剩余自由区域连通。"""

        obstacles = set(self.config.obstacles)
        random_count = self.config.random_obstacle_count
        if self.config.obstacle_ratio is not None:
            random_count = int(round(self.config.width * self.config.height * self.config.obstacle_ratio))
        if random_count > 0:
            obstacles.update(self._sample_connected_random_obstacles(obstacles, random_count))

        all_cells = {(row, col) for row in range(self.config.height) for col in range(self.config.width)}
        self.obstacles = {cell for cell in obstacles if cell in all_cells}
        self.free_cells = all_cells - self.obstacles
        if not self.free_cells:
            raise ValueError("grid must contain at least one free cell")
        if not self._free_cells_are_connected(self.free_cells):
            raise ValueError("free cells must form a connected region")

    def _sample_connected_random_obstacles(self, base_obstacles: set[GridPosition], random_count: int) -> set[GridPosition]:
        """随机放置障碍，同时拒绝会把自由区切断的候选格。"""

        rng = random.Random(self._current_random_obstacle_seed())
        all_cells = {(row, col) for row in range(self.config.height) for col in range(self.config.width)}
        corner_positions = set(self._corner_positions()) if self.config.random_corner_start else set()
        protected = {self.config.start, *self.config.start_positions, *corner_positions}
        obstacles = {cell for cell in base_obstacles if cell in all_cells}
        candidates = [cell for cell in all_cells if cell not in obstacles and cell not in protected]
        rng.shuffle(candidates)

        selected: set[GridPosition] = set()
        target_count = min(random_count, len(candidates))
        # 逐个试放使生成地图天然可完成，而不是在训练时出现隔离目标格。
        for candidate in candidates:
            if len(selected) >= target_count:
                break
            next_obstacles = obstacles | selected | {candidate}
            free_cells = all_cells - next_obstacles
            if free_cells and self._free_cells_are_connected(free_cells):
                selected.add(candidate)

        if len(selected) < target_count:
            raise ValueError(
                f"could not place {target_count} random obstacles while keeping free cells connected; "
                f"placed {len(selected)}"
            )
        return selected

    def _current_random_obstacle_seed(self) -> int:
        """依据回合序号选择当前随机地图 seed。"""

        seeds = self.config.random_obstacle_seeds
        if not seeds:
            return self.config.random_obstacle_seed
        refresh_episodes = max(self.config.map_refresh_episodes, 1)
        seed_index = (self.reset_count // refresh_episodes) % len(seeds)
        return seeds[seed_index]

    def _free_cells_are_connected(self, free_cells: set[GridPosition]) -> bool:
        """用 BFS 验证所有可通行格是否属于同一连通分量。"""

        if not free_cells:
            return False
        start = next(iter(free_cells))
        visited = {start}
        queue: deque[GridPosition] = deque([start])
        while queue:
            row, col = queue.popleft()
            for delta_row, delta_col in ACTIONS.values():
                neighbor = (row + delta_row, col + delta_col)
                if neighbor in free_cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == len(free_cells)

    def _select_start_position(self) -> GridPosition:
        """返回第一个 agent 的起点，保留给单 agent 兼容调用。"""

        return self._select_start_positions()[0]

    def _select_start_positions(self) -> list[GridPosition]:
        """选择互不重合的起点，并尽可能让额外 agent 分散开。"""

        selected: list[GridPosition] = []
        for position in self.config.start_positions:
            if position in self.free_cells and position not in selected:
                selected.append(position)
            if len(selected) == self.num_agents:
                break
        if not selected and self.config.start in self.free_cells:
            selected.append(self.config.start)
        if not selected:
            selected.append(min(self.free_cells))

        if self.config.random_corner_start:
            corners = [corner for corner in self._corner_positions() if corner in self.free_cells and corner not in selected]
            self.random.shuffle(corners)
            for corner in corners:
                if len(selected) == self.num_agents:
                    break
                selected.append(corner)

        # 起点不足时选择离既有起点最远的格，降低开局路径重叠。
        while len(selected) < self.num_agents:
            candidates = [cell for cell in self.free_cells if cell not in selected]
            if not candidates:
                raise ValueError("num_agents exceeds available free cells")
            selected.append(max(candidates, key=lambda cell: min(self._manhattan(cell, item) for item in selected)))

        self.start_positions = selected
        return selected

    def _valid_teammate_positions(self, positions: list[GridPosition]) -> list[GridPosition]:
        """过滤旧接口提供的静态队友位置，排除无效或重叠格。"""

        valid_positions: list[GridPosition] = []
        seen: set[GridPosition] = set()
        occupied = set(self.positions)
        for position in positions:
            if position in occupied or position in seen or position not in self.free_cells:
                continue
            valid_positions.append(position)
            seen.add(position)
        return valid_positions

    def _configure_orientation(self, start_position: GridPosition) -> None:
        """将随机角落起点翻转为统一的左上角视角。"""

        self.start_position = start_position
        self._row_flip = -1 if start_position[0] == self.config.height - 1 else 1
        self._col_flip = -1 if start_position[1] == self.config.width - 1 else 1
        if not self.config.random_corner_start:
            self._row_flip = 1
            self._col_flip = 1

    def _canonical_position(self, position: GridPosition) -> GridPosition:
        """把真实坐标映射到与起点方向无关的规范坐标。"""

        row, col = position
        if self._row_flip == -1:
            row = self.config.height - 1 - row
        if self._col_flip == -1:
            col = self.config.width - 1 - col
        return row, col

    def _canonical_row(self, row: int) -> int:
        """返回真实行在规范视角下的行号。"""

        return self._canonical_position((row, 0))[0]

    def _canonical_free_cells(self) -> set[GridPosition]:
        """将全部自由格转换到规范坐标。"""

        return {self._canonical_position(cell) for cell in self.free_cells}

    def _canonical_covered_cells(self) -> set[GridPosition]:
        """将真实覆盖集合转换到规范坐标。"""

        return {self._canonical_position(cell) for cell in self.covered}

    def _corner_positions(self) -> list[GridPosition]:
        """列出地图的四个角落，用于随机角落起点。"""

        return [
            (0, 0),
            (0, self.config.width - 1),
            (self.config.height - 1, 0),
            (self.config.height - 1, self.config.width - 1),
        ]

    def _actual_action(self, action: int) -> int | None:
        """将规范视角中的动作翻转回真实地图方向。"""

        delta = ACTIONS.get(action)
        if delta is None:
            return None
        actual_delta = (delta[0] * self._row_flip, delta[1] * self._col_flip)
        return ACTION_BY_DELTA.get(actual_delta)

    def _local_window(self, grid: np.ndarray, radius: int, center: GridPosition | None = None) -> np.ndarray:
        """围绕 agent 裁出固定大小局部窗口；边界外用零填充。"""

        center = self._canonical_position(self.position) if center is None else center
        if radius <= 0:
            row, col = center
            return grid[row : row + 1, col : col + 1]

        row, col = center
        row_start = max(row - radius, 0)
        row_stop = min(row + radius + 1, grid.shape[0])
        col_start = max(col - radius, 0)
        col_stop = min(col + radius + 1, grid.shape[1])
        window = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=grid.dtype)
        target_row = row_start - (row - radius)
        target_col = col_start - (col - radius)
        window[
            target_row : target_row + row_stop - row_start,
            target_col : target_col + col_stop - col_start,
        ] = grid[row_start:row_stop, col_start:col_stop]
        return window

    def _agent_density(self) -> float:
        """返回每个自由格平均承载的 agent 数量。"""

        return self.num_agents / max(len(self.free_cells), 1)

    def _sync_legacy_aliases(self) -> None:
        """同步单 agent 时代保留的 ``position``/``path`` 别名。"""

        self.position = self.positions[0]
        self.path = self.paths[0] if self.paths else []
        self.path_length = int(sum(self.path_lengths))
