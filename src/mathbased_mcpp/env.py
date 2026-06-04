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
NONAVOIDABLE_REPEAT_WEIGHT = 0.1


@dataclass(slots=True)
class StepResult:
    observation: np.ndarray
    state: np.ndarray
    reward: float | np.ndarray
    done: bool
    info: dict[str, Any]


class GridCoverageEnv:
    """Grid coverage environment with single-agent compatibility and MAPPO multi-agent mode."""

    observation_channels = 6
    legacy_observation_channels = 7
    explicit_observation_channels = 9
    observation_metadata_dim = 12
    state_channels = 5
    state_metadata_dim = 7
    neighbor_feature_dim = 4
    coverage_message_base_dim = 15

    def __init__(self, config: GridCoverageConfig) -> None:
        self.config = config
        self.random = random.Random(config.seed)
        self.start_position: GridPosition = config.start
        self.start_positions: list[GridPosition] = []
        self._row_flip = 1
        self._col_flip = 1
        self.obstacles: set[GridPosition] = set()
        self.free_cells: set[GridPosition] = set()
        self.covered: set[GridPosition] = set()
        self.positions: list[GridPosition] = [config.start]
        self.position: GridPosition = config.start
        self.teammate_positions: list[GridPosition] = list(config.teammate_positions)
        self.paths: list[list[GridPosition]] = [[config.start]]
        self.path: list[GridPosition] = []
        self.covered_by_agent: list[set[GridPosition]] = [{config.start}]
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
        return max(int(self.config.num_agents), 1)

    @property
    def action_dim(self) -> int:
        return len(ACTIONS)

    @property
    def observation_dim(self) -> int:
        radius = max(self.config.observation_radius, 0)
        window_size = radius * 2 + 1
        channel_dim = window_size * window_size * self.active_observation_channels
        return channel_dim + self.observation_metadata_dim

    @property
    def active_observation_channels(self) -> int:
        if self.config.use_explicit_map_memory:
            return self.explicit_observation_channels
        if self.config.use_legacy_truth_coverage_observation:
            return self.legacy_observation_channels
        return self.observation_channels

    @property
    def node_message_dim(self) -> int:
        return self.coverage_message_base_dim + max(self.config.intent_grid_size, 1) ** 2

    @property
    def state_dim(self) -> int:
        return self.config.height * self.config.width * self.state_channels + self.state_metadata_dim

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.random.seed(seed)
            self.reset_count = 0
        self._build_map()
        self.reset_count += 1
        self.positions = self._select_start_positions()
        self._configure_orientation(self.positions[0])
        self.teammate_positions = self._valid_teammate_positions(self.config.teammate_positions)
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
        self._refresh_explicit_map_memory()
        observations = self._observations()
        return observations[0] if self.num_agents == 1 else observations

    def reset_preview(self) -> np.ndarray:
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
        return self._global_state_from_layers(self._canonical_layers())

    def _global_state_from_layers(self, layers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
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
        self.teammate_positions = self._valid_teammate_positions(positions)

    def peek(self, action: int, agent_index: int = 0) -> tuple[GridPosition, bool]:
        actual_action = self._actual_action(action)
        position = self.positions[agent_index]
        if actual_action is None:
            return position, False
        delta = ACTIONS[actual_action]
        target = (position[0] + delta[0], position[1] + delta[1])
        return target, self.is_free(target)

    def is_free(self, position: GridPosition) -> bool:
        row, col = position
        return 0 <= row < self.config.height and 0 <= col < self.config.width and position not in self.obstacles

    def is_row_complete(self, row: int, extra_cell: GridPosition | None = None) -> bool:
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
        return 1 if self._canonical_row(row) % 2 == 0 else -1

    def legal_actions(self, agent_index: int = 0) -> list[int]:
        return [action for action in ACTIONS if self.peek(action, agent_index=agent_index)[1]]

    def action_masks(self) -> np.ndarray:
        return np.asarray(
            [
                [self.peek(action, agent_index=agent_index)[1] for action in ACTIONS]
                for agent_index in range(self.num_agents)
            ],
            dtype=bool,
        )

    def safe_actions(self, agent_index: int = 0) -> list[int]:
        actions = []
        for action in self.legal_actions(agent_index=agent_index):
            target, _ = self.peek(action, agent_index=agent_index)
            if not self.is_dangerous(target):
                actions.append(action)
        return actions

    def is_dangerous(self, position: GridPosition) -> bool:
        if self.config.danger_radius <= 0:
            return False
        for obstacle in self.obstacles:
            if abs(position[0] - obstacle[0]) + abs(position[1] - obstacle[1]) <= self.config.danger_radius:
                return True
        return False

    def coverage_ratio(self) -> float:
        return len(self.covered) / max(len(self.free_cells), 1)

    def neighbor_mask(self) -> np.ndarray:
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
        previous_position = self.position
        previous_covered = set(self.covered)
        use_frontier_progress = self.config.reward.team_frontier_weight != 0.0
        before_distance = self._distance_to_nearest_uncovered(previous_position, previous_covered) if use_frontier_progress else None
        target, valid = self.peek(action)
        reward_terms = self._single_reward_terms(previous_position=previous_position, target=target, valid=valid)

        self.step_count += 1
        repeated = False
        if valid:
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

        completed = self.covered >= self.free_cells
        if valid:
            if completed:
                reward_terms["finish"] = self.config.reward.finish_reward
        else:
            reward_terms["invalid"] = self.config.reward.invalid_move_penalty

        self.done = completed or self.step_count >= self.config.max_steps
        if not valid:
            reward = float(self.config.reward.invalid_move_penalty)
        else:
            frontier_progress = 0.0
            if use_frontier_progress:
                after_distance = self._distance_to_nearest_uncovered(self.positions[0], self.covered)
            else:
                after_distance = None
            if before_distance is not None and after_distance is not None:
                frontier_progress = float(
                    np.clip((before_distance - after_distance) / max(self.config.height + self.config.width, 1), -1.0, 1.0)
                )
            new_cell = 0.0 if repeated else 1.0
            repeated_cell = float(repeated)
            avoidable_repeat = float(repeated and self._has_uncovered_neighbor(previous_position, previous_covered))
            nonavoidable_repeat = max(repeated_cell - avoidable_repeat, 0.0)
            uncovered_ratio = 1.0 - self.coverage_ratio()
            time_cost_scale = uncovered_ratio if self.config.reward.scale_time_cost_by_uncovered else 1.0
            finish_reward = self.config.reward.finish_reward if completed else 0.0
            straight_bonus = (
                self.config.reward.team_straight_weight
                * reward_terms["Rs"]
                * new_cell
            )
            repeat_penalty = -self.config.reward.team_repeat_weight * (
                avoidable_repeat + NONAVOIDABLE_REPEAT_WEIGHT * nonavoidable_repeat
            )
            reward_terms.update(
                {
                    "new_cells": new_cell,
                    "frontier_progress": frontier_progress,
                    "avoidable_repeats": avoidable_repeat,
                    "repeated_cells": repeated_cell,
                    "nonavoidable_repeats": nonavoidable_repeat,
                    "uncovered_ratio": uncovered_ratio,
                    "time_cost_scale": time_cost_scale,
                    "straight_bonus": straight_bonus,
                    "time": -self.config.reward.team_time_weight * time_cost_scale,
                    "repeat": repeat_penalty,
                    "finish": finish_reward,
                }
            )
            reward = float(
                self.config.reward.team_new_cell_weight * new_cell
                + self.config.reward.team_frontier_weight * frontier_progress
                + straight_bonus
                + reward_terms["time"]
                + reward_terms["repeat"]
                + reward_terms["finish"]
            )
        self.last_collision_agents = 0
        self._sync_legacy_aliases()
        self._refresh_explicit_map_memory()
        layers = self._canonical_layers()
        observation = self._observations(layers)
        step_reward: float | np.ndarray = reward if scalar_action else np.array([reward], dtype=np.float32)
        return StepResult(observation[0] if scalar_action else observation, self._global_state_from_layers(layers), step_reward, self.done, self._info(reward_terms))

    def _step_multi(self, actions: list[int]) -> StepResult:
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

        invalid_agents = {index for index, valid in enumerate(base_valid) if not valid}
        collision_agents: set[int] = set()
        target_to_agents: dict[GridPosition, list[int]] = {}
        for index, target in enumerate(targets):
            if index in invalid_agents:
                continue
            target_to_agents.setdefault(target, []).append(index)
            for other_index, other_position in enumerate(previous_positions):
                if other_index != index and target == other_position:
                    collision_agents.add(index)
                    collision_agents.add(other_index)
        for agents in target_to_agents.values():
            if len(agents) > 1:
                collision_agents.update(agents)
        for first in range(self.num_agents):
            for second in range(first + 1, self.num_agents):
                if targets[first] == previous_positions[second] and targets[second] == previous_positions[first]:
                    collision_agents.add(first)
                    collision_agents.add(second)

        blocked_agents = invalid_agents | collision_agents
        self.last_blocked_cells = {targets[index] for index in blocked_agents}
        final_positions = list(previous_positions)
        moved_agents: list[int] = []
        for index, target in enumerate(targets):
            if index in blocked_agents:
                continue
            final_positions[index] = target
            moved_agents.append(index)

        new_cells = {final_positions[index] for index in moved_agents if final_positions[index] not in previous_covered}
        repeated_cells = sum(1 for index in moved_agents if final_positions[index] in previous_covered)
        avoidable_repeats = sum(
            1
            for index in moved_agents
            if final_positions[index] in previous_covered and self._has_uncovered_neighbor(previous_positions[index], previous_covered)
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
        uncovered_ratio = 1.0 - self.coverage_ratio()
        time_cost_scale = uncovered_ratio if self.config.reward.scale_time_cost_by_uncovered else 1.0
        finish_team_total = self.config.reward.finish_reward if completed else 0.0
        finish_reward = finish_team_total
        if completed and self.config.reward.normalize_team_finish_reward:
            finish_reward = finish_team_total / self.num_agents
        reward_terms = {
            "new_cells": float(len(new_cells)),
            "frontier_progress": frontier_progress,
            "avoidable_repeats": float(avoidable_repeats),
            "repeated_cells": float(repeated_cells),
            "nonavoidable_repeats": float(max(repeated_cells - avoidable_repeats, 0)),
            "invalid_moves": float(len(invalid_agents)),
            "collision_agents": float(len(collision_agents)),
            "uncovered_ratio": uncovered_ratio,
            "time_cost_scale": time_cost_scale,
            "time": -self.config.reward.team_time_weight * time_cost_scale,
            "finish": finish_reward,
            "finish_team_total": finish_team_total,
        }
        repeat_penalty = -self.config.reward.team_repeat_weight * (
            avoidable_repeats + NONAVOIDABLE_REPEAT_WEIGHT * reward_terms["nonavoidable_repeats"]
        ) / self.num_agents
        reward_terms["repeat"] = repeat_penalty
        reward = float(
            self.config.reward.team_new_cell_weight * len(new_cells) / self.num_agents
            + self.config.reward.team_frontier_weight * frontier_progress
            + repeat_penalty
            - self.config.reward.team_invalid_weight * len(invalid_agents) / self.num_agents
            - self.config.reward.team_collision_weight * len(collision_agents) / self.num_agents
            + reward_terms["time"]
            + finish_reward
        )
        self.done = completed or self.step_count >= self.config.max_steps
        self._sync_legacy_aliases()
        self._refresh_explicit_map_memory()
        layers = self._canonical_layers()
        rewards = np.full(self.num_agents, reward, dtype=np.float32)
        return StepResult(self._observations(layers), self._global_state_from_layers(layers), rewards, self.done, self._info(reward_terms))

    def _single_reward_terms(self, previous_position: GridPosition, target: GridPosition, valid: bool) -> dict[str, float]:
        terms = {
            "Rd": 0.0,
            "Rs": 0.0,
            "Rb": 0.0,
            "time": 0.0,
            "repeat": 0.0,
            "finish": 0.0,
            "invalid": 0.0,
        }
        if not valid:
            return terms

        legal_targets = [self.peek(action)[0] for action in self.legal_actions()]
        terms["Rd"] = self._distance_reward(target, legal_targets)
        terms["Rs"] = self._straight_reward(previous_position, target)
        terms["Rb"] = self._coverage_reward(target, legal_targets)
        return terms

    def _distance_reward(self, target: GridPosition, candidate_targets: list[GridPosition]) -> float:
        if not candidate_targets:
            return 0.0
        start = self.start_position
        target_distance = self._manhattan(target, start)
        distances = [self._manhattan(candidate, start) for candidate in candidate_targets]
        d_min = min(distances)
        d_max = max(distances)
        if d_max == d_min:
            return 1.0
        return (target_distance - d_min) / (d_max - d_min)

    def _straight_reward(self, previous_position: GridPosition, target: GridPosition) -> float:
        if len(self.path) < 2:
            return 1.0
        prior_position = self.path[-2]
        previous_vector = (previous_position[0] - prior_position[0], previous_position[1] - prior_position[1])
        current_vector = (target[0] - previous_position[0], target[1] - previous_position[1])
        return 1.0 if current_vector == previous_vector else 0.5

    def _coverage_reward(self, target: GridPosition, candidate_targets: list[GridPosition]) -> float:
        if not candidate_targets:
            return 0.0
        counts = [self._uncovered_neighbor_count(candidate) for candidate in candidate_targets]
        target_count = self._uncovered_neighbor_count(target)
        n_max = max(counts)
        if n_max <= 0:
            return 0.0
        return (n_max - target_count) / n_max

    def _uncovered_neighbor_count(self, position: GridPosition) -> int:
        count = 0
        for delta in ACTIONS.values():
            neighbor = (position[0] + delta[0], position[1] + delta[1])
            if self.is_free(neighbor) and neighbor not in self.covered:
                count += 1
        return count

    def _has_uncovered_neighbor(self, position: GridPosition, covered: set[GridPosition]) -> bool:
        for delta in ACTIONS.values():
            neighbor = (position[0] + delta[0], position[1] + delta[1])
            if self.is_free(neighbor) and neighbor not in covered:
                return True
        return False

    def _distance_to_nearest_uncovered(self, start: GridPosition, covered: set[GridPosition]) -> int | None:
        return self._distance_to_nearest_uncovered_from_set(start, self.free_cells - covered)

    def _distances_to_nearest_uncovered(self, starts: Sequence[GridPosition], covered: set[GridPosition]) -> list[int | None]:
        uncovered = self.free_cells - covered
        if not uncovered:
            return [0 for _ in starts]
        if len(uncovered) <= 1 and len(starts) > 1:
            distance_field = self._distance_field_to_uncovered(covered)
            return [self._distance_from_field(distance_field, position) for position in starts]
        return [self._distance_to_nearest_uncovered_from_set(position, uncovered) for position in starts]

    def _distance_to_nearest_uncovered_from_set(self, start: GridPosition, uncovered: set[GridPosition]) -> int | None:
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
        distance = int(distance_field[position[0], position[1]])
        return None if distance < 0 else distance

    def _manhattan(self, first: GridPosition, second: GridPosition) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def node_messages(self) -> np.ndarray:
        if not self.config.use_explicit_map_memory:
            return np.zeros((self.num_agents, self.node_message_dim), dtype=np.float32)
        for index in range(self.num_agents):
            if self._node_message_cache[index] is None:
                self._node_message_cache[index] = self._coverage_message(index)
        return np.stack(self._node_message_cache).astype(np.float32)

    def _observations(self, layers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        if layers is None:
            layers = self._canonical_layers()
        return np.stack([self._observation(index, layers) for index in range(self.num_agents)]).astype(np.float32)

    def _observation(self, agent_index: int = 0, layers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
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
                *([0.0] * 8),
            ],
            dtype=np.float32,
        )
        return np.concatenate([channel.ravel() for channel in channels] + [metadata])

    def _explicit_memory_observation(
        self,
        agent_index: int,
        layers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
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
        channels = [
            self._local_window(self_agent, radius, center),
            self._local_window(other_agents, radius, center),
            self._local_window(self._cells_layer(known_uncovered), radius, center),
            self._local_window(self._cells_layer(known_team_covered), radius, center),
            self._local_window(self._cells_layer(known_obstacles), radius, center),
            self._local_window(self._cells_layer(self_covered), radius, center),
            self._local_window(self._recent_path_layer(agent_index), radius, center),
            self._local_window(self._cells_layer(unknown), radius, center),
            self._local_window(self._cells_layer(frontier), radius, center),
        ]
        message = self._coverage_message(agent_index, unknown=unknown, frontier=frontier)
        self._node_message_cache[agent_index] = message
        metadata = np.array(
            [
                float(self.row_direction(center[0])),
                self.step_count / max(self.config.max_steps, 1),
                self._known_coverage_ratio(agent_index),
                len(known_free) / max(self.config.height * self.config.width, 1),
                *message[:8],
            ],
            dtype=np.float32,
        )
        return np.concatenate([channel.ravel() for channel in channels] + [metadata])

    def _refresh_explicit_map_memory(self) -> None:
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
            # Global sets are sampled only through this local sensor window.
            self.known_obstacles_by_agent[agent_index].update(visible_cells & self.obstacles)
            self.known_free_by_agent[agent_index].update(visible_cells - self.obstacles)
            self.known_free_by_agent[agent_index].update(self.covered_by_agent[agent_index])
            self.known_team_covered_by_agent[agent_index].update(self.covered_by_agent[agent_index])

        if not self.config.share_map_memory:
            return
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
        bins = max(self.config.intent_grid_size, 1)
        row = min(canonical_position[0] * bins // max(self.config.height, 1), bins - 1)
        col = min(canonical_position[1] * bins // max(self.config.width, 1), bins - 1)
        return row * bins + col

    def _known_coverage_ratio(self, agent_index: int) -> float:
        known_free = self.known_free_by_agent[agent_index]
        if not known_free:
            return 0.0
        return len(self.known_team_covered_by_agent[agent_index] & known_free) / len(known_free)

    def _memory_unknown(self, agent_index: int) -> set[GridPosition]:
        known = self.known_free_by_agent[agent_index] | self.known_obstacles_by_agent[agent_index]
        return {
            (row, col)
            for row in range(self.config.height)
            for col in range(self.config.width)
            if (row, col) not in known
        }

    def _memory_frontiers(self, agent_index: int, unknown: set[GridPosition] | None = None) -> set[GridPosition]:
        unknown = self._memory_unknown(agent_index) if unknown is None else unknown
        return {
            cell
            for cell in self.known_free_by_agent[agent_index]
            if any((cell[0] + delta[0], cell[1] + delta[1]) in unknown for delta in ACTIONS.values())
        }

    def _cells_layer(self, cells: set[GridPosition]) -> np.ndarray:
        layer = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        for cell in cells:
            layer[self._canonical_position(cell)] = 1.0
        return layer

    def _recent_path_layer(self, agent_index: int) -> np.ndarray:
        layer = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        memory_length = max(self.config.recent_path_length, 1)
        recent = self.paths[agent_index][-memory_length:]
        for age, position in enumerate(reversed(recent)):
            value = (memory_length - age) / memory_length
            row, col = self._canonical_position(position)
            layer[row, col] = max(layer[row, col], float(value))
        return layer

    def _communication_metadata(self, agent_index: int) -> list[float]:
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
        path = self.paths[agent_index]
        if len(path) < 2:
            return 0.0, 0.0
        previous = path[-2]
        current = path[-1]
        return float(current[0] - previous[0]), float(current[1] - previous[1])

    def _canonical_layers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        rng = random.Random(self._current_random_obstacle_seed())
        all_cells = {(row, col) for row in range(self.config.height) for col in range(self.config.width)}
        corner_positions = set(self._corner_positions()) if self.config.random_corner_start else set()
        protected = {self.config.start, *self.config.start_positions, *corner_positions}
        obstacles = {cell for cell in base_obstacles if cell in all_cells}
        candidates = [cell for cell in all_cells if cell not in obstacles and cell not in protected]
        rng.shuffle(candidates)

        selected: set[GridPosition] = set()
        target_count = min(random_count, len(candidates))
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
        seeds = self.config.random_obstacle_seeds
        if not seeds:
            return self.config.random_obstacle_seed
        refresh_episodes = max(self.config.map_refresh_episodes, 1)
        seed_index = (self.reset_count // refresh_episodes) % len(seeds)
        return seeds[seed_index]

    def _free_cells_are_connected(self, free_cells: set[GridPosition]) -> bool:
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
        return self._select_start_positions()[0]

    def _select_start_positions(self) -> list[GridPosition]:
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

        while len(selected) < self.num_agents:
            candidates = [cell for cell in self.free_cells if cell not in selected]
            if not candidates:
                raise ValueError("num_agents exceeds available free cells")
            selected.append(max(candidates, key=lambda cell: min(self._manhattan(cell, item) for item in selected)))

        self.start_positions = selected
        return selected

    def _valid_teammate_positions(self, positions: list[GridPosition]) -> list[GridPosition]:
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
        self.start_position = start_position
        self._row_flip = -1 if start_position[0] == self.config.height - 1 else 1
        self._col_flip = -1 if start_position[1] == self.config.width - 1 else 1
        if not self.config.random_corner_start:
            self._row_flip = 1
            self._col_flip = 1

    def _canonical_position(self, position: GridPosition) -> GridPosition:
        row, col = position
        if self._row_flip == -1:
            row = self.config.height - 1 - row
        if self._col_flip == -1:
            col = self.config.width - 1 - col
        return row, col

    def _canonical_row(self, row: int) -> int:
        return self._canonical_position((row, 0))[0]

    def _canonical_free_cells(self) -> set[GridPosition]:
        return {self._canonical_position(cell) for cell in self.free_cells}

    def _canonical_covered_cells(self) -> set[GridPosition]:
        return {self._canonical_position(cell) for cell in self.covered}

    def _corner_positions(self) -> list[GridPosition]:
        return [
            (0, 0),
            (0, self.config.width - 1),
            (self.config.height - 1, 0),
            (self.config.height - 1, self.config.width - 1),
        ]

    def _actual_action(self, action: int) -> int | None:
        delta = ACTIONS.get(action)
        if delta is None:
            return None
        actual_delta = (delta[0] * self._row_flip, delta[1] * self._col_flip)
        return ACTION_BY_DELTA.get(actual_delta)

    def _local_window(self, grid: np.ndarray, radius: int, center: GridPosition | None = None) -> np.ndarray:
        center = self._canonical_position(self.position) if center is None else center
        if radius <= 0:
            row, col = center
            return grid[row : row + 1, col : col + 1]

        padded = np.pad(grid, radius, mode="constant")
        row, col = center
        row += radius
        col += radius
        return padded[row - radius : row + radius + 1, col - radius : col + radius + 1]

    def _agent_density(self) -> float:
        return self.num_agents / max(len(self.free_cells), 1)

    def _sync_legacy_aliases(self) -> None:
        self.position = self.positions[0]
        self.path = self.paths[0] if self.paths else []
        self.path_length = int(sum(self.path_lengths))
