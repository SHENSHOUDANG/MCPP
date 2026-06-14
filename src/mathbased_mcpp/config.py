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
    distance_weight: float = 1.0
    straight_weight: float = 1.0
    coverage_weight: float = 1.0
    time_penalty_weight: float = 0.3
    repeat_penalty_weight: float = 0.1
    finish_reward: float = 80.0
    normalize_team_finish_reward: bool = False
    invalid_move_penalty: float = -1.0
    team_new_cell_weight: float = 1.0
    team_straight_weight: float = 0.01
    team_frontier_weight: float = 0.25
    team_repeat_weight: float = 0.3
    team_invalid_weight: float = 0.8
    team_collision_weight: float = 1.2
    team_time_weight: float = 0.03
    scale_time_cost_by_uncovered: bool = True
    return_progress_weight: float = 1.0
    return_time_weight: float = 0.05
    return_wrong_way_weight: float = 0.2
    return_arrival_reward: float = 2.0
    return_all_arrived_reward: float = 10.0


@dataclass(slots=True)
class GridCoverageConfig:
    width: int = 6
    height: int = 6
    max_steps: int = 200
    seed: int = 0
    start: GridPosition = (0, 0)
    start_positions: list[GridPosition] = field(default_factory=list)
    use_depot: bool = False
    depot: GridPosition | None = None
    depot_dispatch_enabled: bool = True
    require_return_to_depot: bool = False
    initial_return_mode: bool = False
    return_start_positions: list[GridPosition] = field(default_factory=list)
    return_start_strategy: str = "farthest"
    broadcast_coverage_threshold: float = 0.90
    num_agents: int = 1
    random_corner_start: bool = False
    teammate_positions: list[GridPosition] = field(default_factory=list)
    observation_radius: int = 1
    recent_path_length: int = 8
    communication_radius: int = 0
    use_legacy_truth_coverage_observation: bool = False
    use_explicit_map_memory: bool = False
    share_map_memory: bool = False
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
    name: str
    env: GridCoverageConfig
    total_timesteps: int = 500_000
    rollout_steps: int | None = None
    load_previous: bool = True


@dataclass(slots=True)
class CurriculumConfig:
    courses: list[CurriculumCourseConfig] = field(default_factory=list)


@dataclass(slots=True)
class PPOConfig:
    total_timesteps: int = 10_000
    policy_phase: str = "coverage"
    rollout_steps: int = 256
    num_envs: int = 1
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
    use_intent_relation: bool = False
    intent_relation_beta_max: float = 2.0
    intent_relation_detach: bool = True


@dataclass(slots=True)
class TrainConfig:
    run_root: str = "runs"
    log_interval: int = 10
    eval_interval: int = 0
    checkpoint_interval: int = 0
    use_tensorboard: bool = True
    tensorboard_dir: str = "tensorboard"
    cpu_threads: int = 4
    float32_matmul_precision: str = "high"
    compile_model: bool = False


@dataclass(slots=True)
class CUAPConfig:
    enabled: bool = False
    gated: bool = False
    beta: float = 0.3
    disable_in_return_phase: bool = True
    w_novelty: float = 1.0
    w_frontier: float = 0.5
    w_repeat: float = 0.7
    w_conflict: float = 0.5
    eval_radius: int = 1
    normalize: bool = True
    clip: float = 2.0
    use_density_features: bool = False
    tau: float = 1.0
    confidence_tau: float = 1.0
    gate_hidden_dim: int = 32
    gate_init_prob: float = 0.1
    gate_detach_actor_features: bool = True
    gate_input_dim: int = 9
    recent_window: int = 10
    stagnation_normalizer: float = 20.0
    score_scale: float = 2.0
    gate_regularization: float = 0.0


@dataclass(slots=True)
class ExperimentConfig:
    env: GridCoverageConfig
    ppo: PPOConfig
    train: TrainConfig
    cuap: CUAPConfig = field(default_factory=CUAPConfig)
    curriculum: CurriculumConfig | None = None


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = _load_raw_config(config_path)
    return _experiment_config_from_raw(raw)


def _position(value: Any) -> GridPosition:
    if len(value) != 2:
        raise ValueError(f"expected a grid position with two values, got {value!r}")
    return int(value[0]), int(value[1])


def default_max_steps(width: int, height: int) -> int:
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
    ppo_raw = asdict(base_config.ppo)
    ppo_raw["total_timesteps"] = int(course.total_timesteps)
    if course.rollout_steps is not None:
        ppo_raw["rollout_steps"] = int(course.rollout_steps)
    return ExperimentConfig(
        env=course.env,
        ppo=PPOConfig(**ppo_raw),
        train=base_config.train,
        cuap=base_config.cuap,
        curriculum=None,
    )


def select_curriculum_course(
    config: ExperimentConfig,
    course_name: str | None = None,
    course_index: int | None = None,
) -> tuple[int, CurriculumCourseConfig]:
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
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _experiment_config_from_raw(raw: dict[str, Any]) -> ExperimentConfig:
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
    cuap = CUAPConfig(**raw.get("cuap", {}))
    curriculum = _curriculum_from_raw(raw, reward, ppo.total_timesteps, env_raw)
    return ExperimentConfig(env=env, ppo=ppo, train=train, cuap=cuap, curriculum=curriculum)


def _reward_from_raw(raw_reward: dict[str, Any], fallback: RewardConfig | None = None) -> RewardConfig:
    reward_raw = asdict(fallback) if fallback is not None else {}
    reward_raw.update(raw_reward)
    if "finish_bonus" in reward_raw and "finish_reward" not in reward_raw:
        reward_raw["finish_reward"] = reward_raw.pop("finish_bonus")
    if "auxiliary_weight" in reward_raw:
        auxiliary_weight = float(reward_raw.pop("auxiliary_weight"))
        reward_raw.setdefault("time_penalty_weight", auxiliary_weight)
        reward_raw.setdefault("repeat_penalty_weight", auxiliary_weight)
    for legacy_key in (
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
    raw = dict(env_raw)
    width = int(raw.get("width", 6))
    height = int(raw.get("height", 6))
    raw["width"] = width
    raw["height"] = height
    raw["max_steps"] = int(raw.get("max_steps", default_max_steps(width, height)))
    raw["reward"] = reward
    raw["start"] = _position(raw.get("start", (0, 0)))
    raw["start_positions"] = [_position(item) for item in raw.get("start_positions", [])]
    raw["use_depot"] = bool(raw.get("use_depot", False))
    raw["depot"] = None if raw.get("depot") is None else _position(raw.get("depot"))
    raw["depot_dispatch_enabled"] = bool(raw.get("depot_dispatch_enabled", True))
    raw["require_return_to_depot"] = bool(raw.get("require_return_to_depot", False))
    raw["initial_return_mode"] = bool(raw.get("initial_return_mode", False))
    raw["return_start_positions"] = [_position(item) for item in raw.get("return_start_positions", [])]
    raw["return_start_strategy"] = str(raw.get("return_start_strategy", "farthest")).strip().lower()
    raw["broadcast_coverage_threshold"] = float(raw.get("broadcast_coverage_threshold", 0.90))
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
    raw["intent_grid_size"] = max(int(raw.get("intent_grid_size", 3)), 1)
    raw["danger_radius"] = int(raw.get("danger_radius", 1))
    raw["seed"] = int(raw.get("seed", 0))
    return GridCoverageConfig(**raw)


def _curriculum_from_raw(
    raw: dict[str, Any],
    reward: RewardConfig,
    default_total_timesteps: int,
    base_env_raw: dict[str, Any] | None = None,
) -> CurriculumConfig | None:
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
