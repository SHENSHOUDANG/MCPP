from __future__ import annotations

import numpy as np


class LocalPathPlanningEnv:
    """Placeholder for Stage 2 local path-planning RL.

    Stage 1 intentionally uses deterministic path proxies. This interface reserves
    the future point/line/area local path RL contract without coupling it into
    the upper scheduling trainer yet.
    """

    def __init__(self) -> None:
        self.task = None
        self.platform = None
        self.local_map = None

    def reset(self, task: object, platform: object, local_map: object) -> np.ndarray:
        self.task = task
        self.platform = platform
        self.local_map = local_map
        return self.observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        reward = self.compute_local_reward(action)
        return self.observation(), reward, True, {"stage": "placeholder", "action": int(action)}

    def observation(self) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        return np.ones(6, dtype=bool)

    def compute_local_reward(self, action: int | None = None) -> float:
        return 0.0
