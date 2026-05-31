"""实验配置的类型定义与文件读取逻辑。

配置被拆为环境、奖励、PPO、输出设置和课程学习五部分。训练代码接收的
``ExperimentConfig`` 是这些部分组合后的强类型对象，因此核心算法无需
反复处理 TOML/JSON 字典或默认值。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


GridPosition = tuple[int, int]


@dataclass(slots=True)
class RewardConfig:
    """奖励函数各项系数。

    所有课程都使用同一套团队奖励公式；单智能体课程是 ``num_agents = 1``
    的自然特例。系数本身只描述实验设置，实际奖励组合发生在环境的
    ``step`` 流程中。
    ``normalize_team_finish_reward`` 为新实验提供团队规模一致的完成奖金；
    默认关闭以保持历史 checkpoint 的奖励重放口径。
    """

    finish_reward: float = 80.0
    normalize_team_finish_reward: bool = False
    team_new_cell_weight: float = 1.0
    # Only a tie-breaking path preference: one straight move is worth 1% of a new cell.
    team_straight_weight: float = 0.01
    team_frontier_weight: float = 0.0
    team_repeat_weight: float = 0.3
    team_invalid_weight: float = 1.0
    team_time_weight: float = 0.02
    # Old checkpoints used a cheaper tail search as coverage approached one.
    # New budgeted-coverage experiments disable this for a fixed per-step cost.
    scale_time_cost_by_uncovered: bool = True


@dataclass(slots=True)
class GridCoverageConfig:
    """网格覆盖环境的静态设置与观测信息边界开关。

    ``use_explicit_map_memory`` 与 ``share_map_memory`` 控制 actor 可使用的
    私有/通信融合地图记忆；``use_legacy_truth_coverage_observation`` 只用于
    重放旧 checkpoint，不应在新的去中心化实验中启用。
    """

    width: int = 6
    height: int = 6
    max_steps: int = 200
    seed: int = 0
    start: GridPosition = (0, 0)
    start_positions: list[GridPosition] = field(default_factory=list)
    num_agents: int = 1
    random_corner_start: bool = False
    teammate_positions: list[GridPosition] = field(default_factory=list)
    observation_radius: int = 1
    recent_path_length: int = 8
    communication_radius: int = 0
    use_legacy_truth_coverage_observation: bool = False
    use_explicit_map_memory: bool = False
    share_map_memory: bool = False
    observation_mode: str = "local"
    centered_map_size: int = 7
    compressed_border: bool = True
    intent_grid_size: int = 3
    obstacles: list[GridPosition] = field(default_factory=list)
    obstacle_ratio: float | None = None
    random_obstacle_count: int = 0
    random_obstacle_seed: int = 0
    random_obstacle_seeds: list[int] = field(default_factory=list)
    map_refresh_episodes: int = 0
    danger_radius: int = 1
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass(slots=True)
class CurriculumCourseConfig:
    """课程学习中的单个难度阶段及其独立训练预算。"""

    name: str
    env: GridCoverageConfig
    total_timesteps: int = 500_000
    rollout_steps: int | None = None
    load_previous: bool = True


@dataclass(slots=True)
class CurriculumConfig:
    """按顺序组织的课程阶段列表。"""

    courses: list[CurriculumCourseConfig] = field(default_factory=list)


@dataclass(slots=True)
class PPOConfig:
    """PPO 网络、优化过程以及可选 GAT 通信模块的设置。"""

    total_timesteps: int = 10_000
    rollout_steps: int = 256
    update_epochs: int = 4
    mini_batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    seed: int = 0
    device: str = "auto"
    use_graph_attention: bool = False
    gat_num_heads: int = 1
    gat_use_edge_features: bool = False
    gat_residual: bool = False
    gat_attention_dropout: float = 0.0
    use_coverage_messages: bool = False
    use_action_mask: bool = False


@dataclass(slots=True)
class TrainConfig:
    """日志、checkpoint 和 TensorBoard 输出策略。"""

    run_root: str = "runs"
    log_interval: int = 10
    eval_interval: int = 0
    checkpoint_interval: int = 0
    use_tensorboard: bool = True
    tensorboard_dir: str = "tensorboard"


@dataclass(slots=True)
class ExperimentConfig:
    """一次可运行实验所需的完整配置对象。"""

    env: GridCoverageConfig
    ppo: PPOConfig
    train: TrainConfig
    curriculum: CurriculumConfig | None = None


def load_config(path: str | Path) -> ExperimentConfig:
    """从 TOML 或 JSON 文件加载并规范化一次实验配置。"""

    config_path = Path(path)
    raw = _load_raw_config(config_path)
    return _experiment_config_from_raw(raw)


def _position(value: Any) -> GridPosition:
    """把配置中的 ``[row, col]`` 转为环境内部使用的坐标元组。"""

    if len(value) != 2:
        raise ValueError(f"expected a grid position with two values, got {value!r}")
    return int(value[0]), int(value[1])


def default_max_steps(width: int, height: int) -> int:
    """在配置未指定回合长度时，根据地图面积给出保守步数预算。"""

    tier_steps = {
        (6, 6): 64,
        (8, 8): 100,
        (10, 10): 130,
    }
    if (width, height) in tier_steps:
        return tier_steps[(width, height)]
    area = width * height
    return max(int(round(area * 1.25)), area + 2 * max(width, height))


def build_course_config(base_config: ExperimentConfig, course: CurriculumCourseConfig) -> ExperimentConfig:
    """将总配置与一个课程阶段合并为可以直接训练的单阶段配置。"""

    ppo_raw = asdict(base_config.ppo)
    ppo_raw["total_timesteps"] = int(course.total_timesteps)
    if course.rollout_steps is not None:
        ppo_raw["rollout_steps"] = int(course.rollout_steps)
    return ExperimentConfig(
        env=course.env,
        ppo=PPOConfig(**ppo_raw),
        train=base_config.train,
        curriculum=None,
    )


def select_curriculum_course(
    config: ExperimentConfig,
    course_name: str | None = None,
    course_index: int | None = None,
) -> tuple[int, CurriculumCourseConfig]:
    """按名称或下标查找要训练的课程阶段。"""

    if not config.curriculum or not config.curriculum.courses:
        raise ValueError("curriculum configuration is required")
    if course_name is None and course_index is None:
        raise ValueError("either course_name or course_index is required")

    if course_index is not None:
        if course_index < 0 or course_index >= len(config.curriculum.courses):
            raise IndexError(f"course_index {course_index} is out of range")
        return course_index, config.curriculum.courses[course_index]

    assert course_name is not None
    for index, course in enumerate(config.curriculum.courses):
        if course.name == course_name:
            return index, course
    raise ValueError(f"unknown curriculum course: {course_name}")


def _load_raw_config(path: Path) -> dict[str, Any]:
    """读取原始配置字典；JSON 也用于保存/恢复课程快照。"""

    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _experiment_config_from_raw(raw: dict[str, Any]) -> ExperimentConfig:
    """把松散的文件字典转换成带默认值的配置对象树。"""

    env_raw = dict(raw.get("env", {}))
    reward_source = dict(raw.get("reward", {}))
    if not reward_source:
        reward_source = dict(env_raw.pop("reward", {}))
    else:
        env_raw.pop("reward", None)
    reward = _reward_from_raw(reward_source)
    env = _grid_config_from_raw(env_raw, reward)
    ppo = PPOConfig(**raw.get("ppo", {}))
    train = TrainConfig(**raw.get("train", {}))
    curriculum = _curriculum_from_raw(raw, reward, ppo.total_timesteps, env_raw)
    return ExperimentConfig(env=env, ppo=ppo, train=train, curriculum=curriculum)


def _reward_from_raw(raw_reward: dict[str, Any], fallback: RewardConfig | None = None) -> RewardConfig:
    """读取奖励配置，并忽略不再进入最终回报的早期字段。"""

    reward_raw = asdict(fallback) if fallback is not None else {}
    reward_raw.update(raw_reward)
    if "finish_bonus" in reward_raw and "finish_reward" not in reward_raw:
        reward_raw["finish_reward"] = reward_raw.pop("finish_bonus")
    for legacy_key in (
        "distance_weight",
        "straight_weight",
        "coverage_weight",
        "time_penalty_weight",
        "repeat_penalty_weight",
        "invalid_move_penalty",
        "team_collision_weight",
        "auxiliary_weight",
        "new_cell",
        "legal_move",
        "illegal_move",
        "move_time_penalty",
        "dangerous_move",
        "row_forward",
        "premature_switch",
        "completed_row_switch",
        "repeat_cell",
    ):
        reward_raw.pop(legacy_key, None)
    return RewardConfig(**reward_raw)


def _grid_config_from_raw(env_raw: dict[str, Any], reward: RewardConfig) -> GridCoverageConfig:
    """规范化环境字段的类型并附加已经解析好的奖励对象。"""

    raw = dict(env_raw)
    width = int(raw.get("width", 6))
    height = int(raw.get("height", 6))
    raw["width"] = width
    raw["height"] = height
    raw["max_steps"] = int(raw.get("max_steps", default_max_steps(width, height)))
    raw["reward"] = reward
    raw["start"] = _position(raw.get("start", (0, 0)))
    raw["start_positions"] = [_position(item) for item in raw.get("start_positions", [])]
    raw["num_agents"] = max(int(raw.get("num_agents", 1)), 1)
    raw["random_corner_start"] = bool(raw.get("random_corner_start", False))
    raw["teammate_positions"] = [_position(item) for item in raw.get("teammate_positions", [])]
    raw["obstacles"] = [_position(item) for item in raw.get("obstacles", [])]
    raw["obstacle_ratio"] = None if raw.get("obstacle_ratio") is None else float(raw.get("obstacle_ratio"))
    raw["random_obstacle_count"] = int(raw.get("random_obstacle_count", 0))
    raw["random_obstacle_seed"] = int(raw.get("random_obstacle_seed", 0))
    raw["random_obstacle_seeds"] = [int(item) for item in raw.get("random_obstacle_seeds", [])]
    raw["map_refresh_episodes"] = max(int(raw.get("map_refresh_episodes", 0)), 0)
    raw["observation_radius"] = int(raw.get("observation_radius", 1))
    raw["recent_path_length"] = int(raw.get("recent_path_length", 8))
    raw["communication_radius"] = int(raw.get("communication_radius", 0))
    raw["use_legacy_truth_coverage_observation"] = bool(raw.get("use_legacy_truth_coverage_observation", False))
    raw["use_explicit_map_memory"] = bool(raw.get("use_explicit_map_memory", False))
    raw["share_map_memory"] = bool(raw.get("share_map_memory", False))
    raw["observation_mode"] = str(raw.get("observation_mode", "local")).strip().lower()
    raw["centered_map_size"] = _odd_window_size(raw.get("centered_map_size", 7))
    raw["compressed_border"] = bool(raw.get("compressed_border", True))
    raw["intent_grid_size"] = max(int(raw.get("intent_grid_size", 3)), 1)
    raw["danger_radius"] = int(raw.get("danger_radius", 1))
    raw["seed"] = int(raw.get("seed", 0))
    return GridCoverageConfig(**raw)


def _odd_window_size(value: Any) -> int:
    """Return a positive odd map window size for centered spatial observations."""

    size = max(int(value), 3)
    if size % 2 == 0:
        size += 1
    return size


def _curriculum_from_raw(
    raw: dict[str, Any],
    reward: RewardConfig,
    default_total_timesteps: int,
    base_env_raw: dict[str, Any] | None = None,
) -> CurriculumConfig | None:
    """解析课程列表，让各课程继承基础环境并只覆盖变化字段。"""

    curriculum_raw = dict(raw.get("curriculum") or {})
    course_rows = curriculum_raw.get("courses", [])
    if not course_rows:
        return None

    courses: list[CurriculumCourseConfig] = []
    for index, course_raw in enumerate(course_rows):
        course_data = dict(base_env_raw or {})
        course_data.update(course_raw)
        name = str(course_data.pop("name", f"tier-{index + 1}"))
        course_reward = _reward_from_raw(dict(course_data.pop("reward", {})), reward)
        total_timesteps = int(course_data.pop("total_timesteps", default_total_timesteps))
        rollout_steps_raw = course_data.pop("rollout_steps", None)
        rollout_steps = None if rollout_steps_raw is None else int(rollout_steps_raw)
        load_previous = bool(course_data.pop("load_previous", index > 0))
        env = _grid_config_from_raw(course_data, course_reward)
        courses.append(
            CurriculumCourseConfig(
                name=name,
                env=env,
                total_timesteps=total_timesteps,
                rollout_steps=rollout_steps,
                load_previous=load_previous,
            )
        )
    return CurriculumConfig(courses=courses)
