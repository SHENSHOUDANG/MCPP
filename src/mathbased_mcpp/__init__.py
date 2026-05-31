"""多智能体覆盖路径规划实验包的公开接口。

外部脚本通常只需要从这里导入配置对象、环境和安全层；训练与评估入口
则由 :mod:`mathbased_mcpp.cli` 提供。
"""

from .config import (
    CurriculumConfig,
    CurriculumCourseConfig,
    ExperimentConfig,
    GridCoverageConfig,
    PPOConfig,
    RewardConfig,
    build_course_config,
    default_max_steps,
    load_config,
    select_curriculum_course,
)
from .env import ACTIONS, GridCoverageEnv, StepResult
from .safety import AvoidancePolicy, SafetyLayer

__all__ = [
    "ACTIONS",
    "AvoidancePolicy",
    "CurriculumConfig",
    "CurriculumCourseConfig",
    "ExperimentConfig",
    "GridCoverageConfig",
    "GridCoverageEnv",
    "PPOConfig",
    "RewardConfig",
    "SafetyLayer",
    "StepResult",
    "build_course_config",
    "default_max_steps",
    "load_config",
    "select_curriculum_course",
]
