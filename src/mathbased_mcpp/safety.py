"""基于规则的执行时安全过滤接口。

此模块与训练策略分离：actor 先提出动作，安全层只在明显不合理时替换它。
这样后续可以独立比较“无过滤”“规则过滤”或学习型避障策略。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .env import ACTIONS, GridCoverageEnv


class AvoidancePolicy(ABC):
    """未来可学习局部避障策略需要实现的最小接口。"""

    @abstractmethod
    def act(self, observation: np.ndarray, proposed_action: int) -> int:
        raise NotImplementedError


class SafetyLayer:
    """尚未学习避障策略时使用的规则式局部安全层。"""

    def filter_action(self, env: GridCoverageEnv, proposed_action: int) -> int:
        """尽量保留 actor 提议，必要时选择更安全且少重复的动作。"""

        target, valid = env.peek(proposed_action)
        repeats_covered = valid and target in env.covered
        has_uncovered_neighbor = any(env.peek(action)[0] not in env.covered for action in env.legal_actions())
        if valid and not repeats_covered and not self._avoidable_danger(env, proposed_action):
            return proposed_action
        if valid and repeats_covered and not has_uncovered_neighbor and not self._avoidable_danger(env, proposed_action):
            return proposed_action

        # 只有提议动作确实不合适时，才在可用动作中按覆盖启发式重选。
        candidates = env.safe_actions() or env.legal_actions()
        if not candidates:
            return proposed_action
        return max(candidates, key=lambda action: self._score(env, action))

    def _avoidable_danger(self, env: GridCoverageEnv, action: int) -> bool:
        """判断该危险移动是否存在可替代的安全动作。"""

        target, valid = env.peek(action)
        if not valid or not env.is_dangerous(target):
            return False
        return bool(env.safe_actions())

    def _score(self, env: GridCoverageEnv, action: int) -> tuple[float, float, float, float]:
        """按新覆盖、犁式前进、换行和稳定动作顺序给候选动作排序。"""

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
