from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .env import ACTIONS, GridCoverageEnv


class AvoidancePolicy(ABC):
    """Interface reserved for a future learned local avoidance policy."""

    @abstractmethod
    def act(self, observation: np.ndarray, proposed_action: int) -> int:
        raise NotImplementedError


class SafetyLayer:
    """Rule-based local safety layer used until the avoidance policy is learned."""

    def filter_action(self, env: GridCoverageEnv, proposed_action: int) -> int:
        target, valid = env.peek(proposed_action)
        repeats_covered = valid and target in env.covered
        has_uncovered_neighbor = any(env.peek(action)[0] not in env.covered for action in env.legal_actions())
        if valid and not repeats_covered and not self._avoidable_danger(env, proposed_action):
            return proposed_action
        if valid and repeats_covered and not has_uncovered_neighbor and not self._avoidable_danger(env, proposed_action):
            return proposed_action

        candidates = env.safe_actions() or env.legal_actions()
        if not candidates:
            return proposed_action
        return max(candidates, key=lambda action: self._score(env, action))

    def _avoidable_danger(self, env: GridCoverageEnv, action: int) -> bool:
        target, valid = env.peek(action)
        if not valid or not env.is_dangerous(target):
            return False
        return bool(env.safe_actions())

    def _score(self, env: GridCoverageEnv, action: int) -> tuple[float, float, float, float]:
        target, valid = env.peek(action)
        if not valid:
            return (-10.0, -10.0, -10.0, -10.0)
        row_delta = target[0] - env.position[0]
        col_delta = target[1] - env.position[1]
        newly_covered = target not in env.covered
        row_forward = row_delta == 0 and col_delta == env.row_direction(env.position[0])
        row_switch = row_delta == 1 and col_delta == 0 and env.is_row_complete(env.position[0])
        action_order = -float(list(ACTIONS).index(action))
        return (float(newly_covered), float(row_forward), float(row_switch), action_order)
