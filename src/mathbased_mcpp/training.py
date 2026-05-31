"""PPO 训练循环与课程学习调度。

训练的主流程为：

``建立环境/模型 -> 收集一段 rollout -> 用 GAE 计算回报与优势
-> 进行若干轮 PPO 更新 -> 周期性评估和保存 checkpoint``。

在多智能体模式中，每个环境步会产生多个 agent transition，因此总训练
步数按 agent transition 数量累计，而不是仅按环境调用次数累计。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from .runtime import configure_runtime

configure_runtime()

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig, build_course_config, select_curriculum_course
from .env import GridCoverageEnv
from .ppo import ActorCritic, RolloutBatch
from .utils import append_metrics, make_run_dir, make_tensorboard_writer, set_seed, write_tensorboard_rows


def train_ppo(
    config: ExperimentConfig,
    env: GridCoverageEnv | None = None,
    run_dir: str | Path | None = None,
    course: str | None = None,
    previous_checkpoint: str | Path | None = None,
) -> Path:
    """训练入口：根据配置选择单课程、完整课程链或普通训练。"""

    if config.curriculum and config.curriculum.courses:
        if course is not None:
            return _train_single_curriculum_course(
                config=config,
                course_name=course,
                run_dir=run_dir,
                previous_checkpoint=previous_checkpoint,
            )
        return _train_curriculum(config, run_dir=run_dir)
    return _train_single_course(config, env=env, run_dir=run_dir)


def _train_curriculum(config: ExperimentConfig, run_dir: str | Path | None = None) -> Path:
    """依次训练全部课程阶段，并把上一阶段最优模型传给下一阶段。"""

    assert config.curriculum is not None
    master_run_path = Path(run_dir) if run_dir is not None else make_run_dir(config.train.run_root)
    master_run_path.mkdir(parents=True, exist_ok=True)

    previous_checkpoint: Path | None = None
    final_checkpoint = master_run_path / "policy.pt"
    for index, course in enumerate(config.curriculum.courses):
        # 课程难度逐级提升，允许新阶段从上一个阶段的最优权重继续学习。
        course_config = build_course_config(config, course)
        course_dir = master_run_path / f"{index + 1:02d}-{_slugify(course.name)}"
        checkpoint = _train_single_course(
            course_config,
            run_dir=course_dir,
            checkpoint_path=previous_checkpoint if index > 0 and course.load_previous else None,
        )
        previous_checkpoint = course_dir / "best_policy.pt"
        _update_curriculum_state(config, course.name, previous_checkpoint, course_dir, state_root=master_run_path)
        final_checkpoint = checkpoint

    return final_checkpoint


def _train_single_curriculum_course(
    config: ExperimentConfig,
    course_name: str,
    run_dir: str | Path | None = None,
    previous_checkpoint: str | Path | None = None,
) -> Path:
    """单独运行指定课程，适合耗时较长的正式训练逐阶段执行。"""

    assert config.curriculum is not None
    course_index, course = select_curriculum_course(config, course_name=course_name)
    course_config = build_course_config(config, course)
    checkpoint_path = _resolve_previous_checkpoint(
        config=config,
        course_index=course_index,
        explicit_checkpoint=previous_checkpoint,
    )
    if course_index > 0 and course.load_previous and checkpoint_path is None:
        previous_course = config.curriculum.courses[course_index - 1]
        raise FileNotFoundError(
            f"{course.name} requires a checkpoint from {previous_course.name}; "
            "train the previous course first or pass --previous-checkpoint"
        )
    run_base = Path(run_dir) if run_dir is not None else make_run_dir(config.train.run_root)
    course_run_dir = run_base / f"{course_index + 1:02d}-{_slugify(course.name)}"
    course_run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _train_single_course(
        course_config,
        run_dir=course_run_dir,
        checkpoint_path=checkpoint_path,
    )
    _update_curriculum_state(config, course.name, course_run_dir / "best_policy.pt", course_run_dir, state_root=run_base)
    return checkpoint


def _train_single_course(
    config: ExperimentConfig,
    env: GridCoverageEnv | None = None,
    run_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """训练一个固定环境/课程配置并产出可评估的模型文件。"""

    set_seed(config.ppo.seed)
    env = env or GridCoverageEnv(config.env)
    if config.ppo.use_coverage_messages and not env.config.use_explicit_map_memory:
        raise ValueError("use_coverage_messages requires use_explicit_map_memory=true")
    run_path = Path(run_dir) if run_dir is not None else make_run_dir(config.train.run_root)
    run_path.mkdir(parents=True, exist_ok=True)
    _write_config_snapshot(config, run_path)
    writer = make_tensorboard_writer(run_path, config.train.tensorboard_dir) if config.train.use_tensorboard else None

    device = _resolve_device(config.ppo.device)
    # actor 接收每个 agent 的去中心化观测；critic 在训练中接收全局空间状态。
    model = ActorCritic(
        env.observation_dim,
        env.action_dim,
        config.ppo.hidden_dim,
        state_shape=(env.config.height, env.config.width),
        state_channels=env.state_channels,
        state_metadata_dim=env.state_metadata_dim,
        use_graph_attention=config.ppo.use_graph_attention,
        gat_num_heads=config.ppo.gat_num_heads,
        gat_edge_dim=env.neighbor_feature_dim if config.ppo.gat_use_edge_features else 0,
        gat_residual=config.ppo.gat_residual,
        gat_attention_dropout=config.ppo.gat_attention_dropout,
        node_message_dim=env.node_message_dim if config.ppo.use_coverage_messages else 0,
        actor_encoder=config.ppo.actor_encoder,
        actor_map_shape=env.actor_map_shape if config.ppo.actor_encoder == "cnn" else None,
        actor_metadata_dim=env.observation_metadata_dim if config.ppo.actor_encoder == "cnn" else 0,
    ).to(device)
    if checkpoint_path is not None:
        _load_checkpoint_weights(model, checkpoint_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate)

    # observation 带 agent 维度；state 为所有 agent 共享的一份全局 critic 输入。
    observation = _agent_observations(env, env.reset(seed=config.env.seed))
    state = env.global_state()
    timestep = 0
    update_index = 0
    episode_index = 0
    episode_reward = 0.0
    episode_start_length = env.path_length
    metric_rows: list[dict[str, float | int]] = []
    best_score: tuple[float, int, float, float] | None = None
    best_checkpoint_path = run_path / "best_policy.pt"
    last_checkpoint_path = run_path / "last_policy.pt"
    eval_interval = max(config.train.eval_interval or config.train.log_interval, 1)
    checkpoint_interval = max(config.train.checkpoint_interval or config.train.log_interval, 1)

    try:
        while timestep < config.ppo.total_timesteps:
            # 一轮更新先用当前策略采样固定长度轨迹，再在这批旧数据上优化。
            batch, observation, state, rollout_metrics = _collect_rollout(
                config=config,
                env=env,
                model=model,
                observation=observation,
                state=state,
                max_steps=max(1, min(config.ppo.rollout_steps, config.ppo.total_timesteps - timestep) // env.num_agents),
                episode_index=episode_index,
                episode_reward=episode_reward,
                episode_start_length=episode_start_length,
            )
            episode_index = int(rollout_metrics.pop("episode_index"))
            episode_reward = float(rollout_metrics.pop("episode_reward"))
            episode_start_length = int(rollout_metrics.pop("episode_start_length"))
            new_metric_rows = rollout_metrics.pop("metric_rows")
            metric_rows.extend(new_metric_rows)
            # numel = 环境步数 x agent 数量，代表实际学习到的 transition 数。
            timestep += int(batch.actions.numel())
            update_index += 1

            _update_policy(config, model, optimizer, batch)

            if metric_rows:
                append_metrics(run_path / "metrics.csv", metric_rows)
                if writer is not None:
                    write_tensorboard_rows(writer, "train", metric_rows)
                metric_rows.clear()

            if update_index % checkpoint_interval == 0:
                torch.save(_checkpoint_payload(config, model), last_checkpoint_path)

            if update_index % eval_interval == 0:
                # 确定性评估用于挑选 best_policy，避免只按随机训练奖励选模型。
                eval_row = _evaluate_model(config, model, update_index)
                append_metrics(run_path / "eval_metrics.csv", [eval_row])
                if writer is not None:
                    write_tensorboard_rows(writer, "eval", [eval_row])
                    writer.add_scalar("train/timesteps", timestep, update_index)
                    writer.flush()
                score = _metric_score(eval_row)
                if best_score is None or score > best_score:
                    best_score = score
                    torch.save(_checkpoint_payload(config, model), best_checkpoint_path)

        torch.save(_checkpoint_payload(config, model), last_checkpoint_path)
        if not best_checkpoint_path.exists():
            torch.save(_checkpoint_payload(config, model), best_checkpoint_path)
        shutil.copyfile(best_checkpoint_path, run_path / "policy.pt")

        _finalize_course_outputs(config, run_path, best_checkpoint_path)
        return run_path / "policy.pt"
    finally:
        if writer is not None:
            writer.close()


def _finalize_course_outputs(config: ExperimentConfig, run_path: Path, checkpoint_path: Path) -> None:
    """用最优 checkpoint 生成最终评估轨迹与可视化图片。"""

    from .evaluation import evaluate_policy
    from .rendering import render_trajectory

    summary = evaluate_policy(config, checkpoint_path, output_path=run_path / "trajectory.json")
    render_trajectory(config, summary["trajectory"], run_path / "trajectory.png")


def _write_config_snapshot(config: ExperimentConfig, run_path: Path) -> None:
    """把实际生效配置写入结果目录，保证实验可追溯。"""

    run_path.joinpath("course_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def _curriculum_state_path(config: ExperimentConfig, state_root: str | Path | None = None) -> Path:
    """返回记录各课程最优 checkpoint 的状态文件位置。"""

    return Path(state_root) / "_curriculum_state.json" if state_root is not None else Path(config.train.run_root) / "_curriculum_state.json"


def _resolve_previous_checkpoint(
    config: ExperimentConfig,
    course_index: int,
    explicit_checkpoint: str | Path | None = None,
) -> Path | None:
    """优先使用显式指定模型，否则从课程状态文件查找前一阶段模型。"""

    if explicit_checkpoint is not None:
        path = Path(explicit_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"previous checkpoint not found: {path}")
        return path
    if course_index <= 0 or not config.curriculum:
        return None

    previous_course = config.curriculum.courses[course_index - 1]
    state_path = _curriculum_state_path(config)
    if not state_path.exists():
        return None

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    previous_entry = state.get("courses", {}).get(previous_course.name, {})
    candidate = Path(previous_entry.get("best_checkpoint", ""))
    if candidate.exists():
        return candidate
    return None


def _update_curriculum_state(
    config: ExperimentConfig,
    course_name: str,
    checkpoint: Path,
    run_dir: Path,
    state_root: str | Path | None = None,
) -> None:
    """在课程完成后记录其最优模型与输出目录。"""

    if not config.curriculum or not config.curriculum.courses:
        return
    state_path = _curriculum_state_path(config, state_root=state_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"courses": {}}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {"courses": {}}
    state.setdefault("courses", {})
    state["courses"][course_name] = {
        "best_checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _evaluate_model(config: ExperimentConfig, model: ActorCritic, update_index: int) -> dict[str, float | int]:
    """用确定性动作完整执行一个回合，返回选模需要的简明指标。"""

    env = GridCoverageEnv(config.env)
    observation = _agent_observations(env, env.reset(seed=config.env.seed))
    state = env.global_state()
    device = _model_device(model)
    total_reward = 0.0
    done = False
    info = {}
    while not done:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        neighbor_mask = torch.as_tensor(env.neighbor_mask(), dtype=torch.bool, device=device)
        edge_features = _edge_features_tensor(env, model, device)
        node_messages = _node_messages_tensor(env, model, device)
        action_mask = _action_mask_tensor(env, config, device)
        with torch.no_grad():
            actions, _, _ = model.act_batch(
                obs_tensor,
                state_tensor,
                neighbor_mask=neighbor_mask,
                edge_features=edge_features,
                node_messages=node_messages,
                action_mask=action_mask,
                deterministic=True,
            )
        result = env.step(actions.cpu().numpy().tolist())
        rewards = _agent_rewards(env, result.reward)
        total_reward += float(np.mean(rewards))
        observation = _agent_observations(env, result.observation)
        state = result.state
        done = result.done
        info = result.info
    return {
        "episode": update_index,
        "reward": total_reward,
        "coverage_ratio": info.get("coverage_ratio", env.coverage_ratio()),
        "path_length": env.path_length,
        "completed": int(info.get("completed", False)),
        "steps": info.get("step_count", env.step_count),
    }


def _load_checkpoint_weights(model: ActorCritic, checkpoint_path: str | Path) -> None:
    """把前一课程或恢复训练的参数装入结构相同的模型。"""

    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])


def _metric_score(row: dict[str, float | int]) -> tuple[float, int, float, float]:
    """定义 best checkpoint 排序：覆盖优先，其次完成、路径短和奖励高。"""

    coverage = float(row["coverage_ratio"])
    completed = int(row["completed"])
    path_length = float(row["path_length"])
    reward = float(row["reward"])
    return coverage, completed, -path_length, reward


def _collect_rollout(
    config: ExperimentConfig,
    env: GridCoverageEnv,
    model: ActorCritic,
    observation: np.ndarray,
    state: np.ndarray,
    max_steps: int,
    episode_index: int,
    episode_reward: float,
    episode_start_length: int,
) -> tuple[RolloutBatch, np.ndarray, np.ndarray, dict[str, object]]:
    """使用当前策略采样一段轨迹，并构造一次 PPO 更新所需批次。

    列表的第一个维度是环境时间步，第二个维度是 agent。这样 GAT 更新时
    仍能保留每一时刻的完整邻接图，而不是打散 agent 间的同步关系。
    """

    observations: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_probs: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    values: list[np.ndarray] = []
    neighbor_masks: list[np.ndarray] = []
    edge_features: list[np.ndarray] = []
    node_messages: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    metric_rows: list[dict[str, float | int]] = []
    device = _model_device(model)

    for _ in range(max_steps):
        # 采样时缓存策略看到的全部输入，之后 PPO 才能正确重算旧动作概率。
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        neighbor_mask = env.neighbor_mask()
        neighbor_mask_tensor = torch.as_tensor(neighbor_mask, dtype=torch.bool, device=device)
        edge_feature_array = env.neighbor_features() if _uses_edge_features(model) else None
        edge_feature_tensor = (
            torch.as_tensor(edge_feature_array, dtype=torch.float32, device=device) if edge_feature_array is not None else None
        )
        node_message_array = env.node_messages() if model.node_message_dim > 0 else None
        node_message_tensor = (
            torch.as_tensor(node_message_array, dtype=torch.float32, device=device) if node_message_array is not None else None
        )
        action_mask_array = env.action_mask() if config.ppo.use_action_mask else None
        action_mask_tensor = (
            torch.as_tensor(action_mask_array, dtype=torch.bool, device=device) if action_mask_array is not None else None
        )
        with torch.no_grad():
            action_tensor, log_prob_tensor, value_tensor = model.act_batch(
                obs_tensor,
                state_tensor,
                neighbor_mask=neighbor_mask_tensor,
                edge_features=edge_feature_tensor,
                node_messages=node_message_tensor,
                action_mask=action_mask_tensor,
            )

        result = env.step(action_tensor.cpu().numpy().tolist())
        reward_array = _agent_rewards(env, result.reward)
        observations.append(observation)
        states.append(state)
        actions.append(action_tensor.cpu().numpy())
        log_probs.append(log_prob_tensor.cpu().numpy())
        rewards.append(reward_array)
        dones.append(np.full(env.num_agents, float(result.done), dtype=np.float32))
        values.append(value_tensor.cpu().numpy())
        neighbor_masks.append(neighbor_mask)
        if edge_feature_array is not None:
            edge_features.append(edge_feature_array)
        if node_message_array is not None:
            node_messages.append(node_message_array)
        if action_mask_array is not None:
            action_masks.append(action_mask_array)
        episode_reward += float(np.mean(reward_array))
        observation = _agent_observations(env, result.observation)
        state = result.state

        if result.done:
            # rollout 可以跨越多个回合；终止后立即 reset 继续填满训练批次。
            metric_rows.append(
                {
                    "episode": episode_index,
                    "reward": episode_reward,
                    "coverage_ratio": result.info["coverage_ratio"],
                    "path_length": result.info["path_length"] - episode_start_length,
                    "completed": int(result.info["completed"]),
                    "steps": result.info["step_count"],
                }
            )
            episode_index += 1
            episode_reward = 0.0
            observation = _agent_observations(env, env.reset())
            state = env.global_state()
            episode_start_length = env.path_length

    with torch.no_grad():
        # 非终止截断的 rollout 需要用最后状态价值 bootstrap 后续回报。
        if dones and bool(dones[-1][0]):
            next_values = np.zeros(env.num_agents, dtype=np.float32)
        else:
            next_value = model.value(torch.as_tensor(state, dtype=torch.float32, device=device))
            next_values = next_value.expand(env.num_agents).cpu().numpy()

    # GAE 将一步 TD 残差累积为方差较小的策略优势估计。
    returns, advantages = _gae_array(
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        next_values=np.asarray(next_values, dtype=np.float32),
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
    )

    observation_array = np.asarray(observations, dtype=np.float32)
    state_array = np.asarray(states, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.int64)
    log_prob_array = np.asarray(log_probs, dtype=np.float32)
    value_array = np.asarray(values, dtype=np.float32)
    batch = RolloutBatch(
        observations=torch.as_tensor(observation_array, dtype=torch.float32, device=device),
        states=torch.as_tensor(state_array, dtype=torch.float32, device=device),
        actions=torch.as_tensor(action_array, dtype=torch.long, device=device),
        log_probs=torch.as_tensor(log_prob_array, dtype=torch.float32, device=device),
        returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
        values=torch.as_tensor(value_array, dtype=torch.float32, device=device),
        neighbor_masks=torch.as_tensor(np.asarray(neighbor_masks, dtype=bool), dtype=torch.bool, device=device),
        edge_features=(
            torch.as_tensor(np.asarray(edge_features, dtype=np.float32), dtype=torch.float32, device=device)
            if edge_features
            else None
        ),
        node_messages=(
            torch.as_tensor(np.asarray(node_messages, dtype=np.float32), dtype=torch.float32, device=device)
            if node_messages
            else None
        ),
        action_masks=(
            torch.as_tensor(np.asarray(action_masks, dtype=bool), dtype=torch.bool, device=device)
            if action_masks
            else None
        ),
    )
    metrics = {
        "episode_index": episode_index,
        "episode_reward": episode_reward,
        "episode_start_length": episode_start_length,
        "metric_rows": metric_rows,
    }
    return batch, observation, state, metrics


def _agent_observations(env: GridCoverageEnv, observation: np.ndarray) -> np.ndarray:
    """为单 agent 观测补上 agent 维度，使训练循环始终处理统一形状。"""

    observation = np.asarray(observation, dtype=np.float32)
    if observation.ndim == 1:
        return observation.reshape(1, -1)
    return observation


def _agent_rewards(env: GridCoverageEnv, reward: float | np.ndarray) -> np.ndarray:
    """将标量团队奖励转换为每个 agent 对齐的一维数组。"""

    reward_array = np.asarray(reward, dtype=np.float32)
    if reward_array.ndim == 0:
        return np.full(env.num_agents, float(reward_array), dtype=np.float32)
    return reward_array


def _resolve_device(device_name: str) -> torch.device:
    """解析 CPU/GPU 配置；请求不可用 CUDA 时稳妥回落到 CPU。"""

    normalized = device_name.lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


def _model_device(model: ActorCritic) -> torch.device:
    """读取模型参数所在设备，供新建输入张量对齐。"""

    return next(model.parameters()).device


def _gae(
    rewards: list[float],
    dones: list[float],
    values: list[float],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """单序列版 generalized advantage estimation，保留给简单调用/测试。"""

    advantages = [0.0 for _ in rewards]
    last_advantage = 0.0
    for index in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[index]
        next_val = next_value if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + gamma * next_val * next_non_terminal - values[index]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[index] = last_advantage
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return returns, advantages


def _gae_array(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """对 ``[time, agents]`` 奖励批量计算 GAE 与目标回报。"""

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    for index in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[index]
        next_value = next_values if index == rewards.shape[0] - 1 else values[index + 1]
        delta = rewards[index] + gamma * next_value * next_non_terminal - values[index]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[index] = last_advantage
    returns = advantages + values
    return returns, advantages


def _update_policy(config: ExperimentConfig, model: ActorCritic, optimizer: torch.optim.Optimizer, batch: RolloutBatch) -> None:
    """以 PPO clipped objective 对一段 rollout 进行多轮小批量更新。"""

    advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std(unbiased=False) + 1e-8)
    rollout_steps = batch.actions.shape[0]
    num_agents = batch.actions.shape[1] if batch.actions.ndim > 1 else 1
    mini_batch_steps = max(1, config.ppo.mini_batch_size // max(num_agents, 1))
    batch_size = rollout_steps
    indices = np.arange(batch_size)

    for _ in range(config.ppo.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, mini_batch_steps):
            mb = indices[start : start + mini_batch_steps]
            mb_tensor = torch.as_tensor(mb, dtype=torch.long, device=batch.actions.device)
            neighbor_mask = batch.neighbor_masks[mb_tensor] if batch.neighbor_masks is not None else None
            edge_features = batch.edge_features[mb_tensor] if batch.edge_features is not None else None
            node_messages = batch.node_messages[mb_tensor] if batch.node_messages is not None else None
            action_masks = batch.action_masks[mb_tensor] if batch.action_masks is not None else None
            log_probs, entropy, values = model.evaluate_actions(
                batch.observations[mb_tensor],
                batch.states[mb_tensor],
                batch.actions[mb_tensor],
                neighbor_mask=neighbor_mask,
                edge_features=edge_features,
                node_messages=node_messages,
                action_mask=action_masks,
            )
            # ratio 衡量新策略相对采样时旧策略改变了多少；clip 防止步子过大。
            ratio = torch.exp(log_probs - batch.log_probs[mb_tensor])
            clipped = torch.clamp(ratio, 1.0 - config.ppo.clip_ratio, 1.0 + config.ppo.clip_ratio) * advantages[mb_tensor]
            policy_loss = -torch.min(ratio * advantages[mb_tensor], clipped).mean()
            value_loss = nn.functional.mse_loss(values, batch.returns[mb_tensor])
            entropy_loss = entropy.mean()
            # actor 希望提升优势动作，critic 拟合回报，entropy 鼓励适量探索。
            loss = policy_loss + config.ppo.value_coef * value_loss - config.ppo.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.ppo.max_grad_norm)
            optimizer.step()


def _checkpoint_payload(config: ExperimentConfig, model: ActorCritic) -> dict[str, object]:
    """保存权重及重建网络结构所需元信息。"""

    return {
        "model_state_dict": model.state_dict(),
        "observation_dim": model.observation_dim,
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "hidden_dim": config.ppo.hidden_dim,
        "critic_type": model.critic_mode,
        "state_shape": model.state_shape,
        "state_channels": model.state_channels,
        "state_metadata_dim": model.state_metadata_dim,
        "num_agents": config.env.num_agents,
        "use_graph_attention": model.use_graph_attention,
        "gat_num_heads": model.gat_num_heads,
        "gat_edge_dim": model.gat_edge_dim,
        "gat_use_edge_features": model.gat_edge_dim > 0,
        "gat_residual": model.gat_residual,
        "gat_attention_dropout": model.gat_attention_dropout,
        "node_message_dim": model.node_message_dim,
        "use_coverage_messages": model.node_message_dim > 0,
        "actor_encoder": model.actor_encoder,
        "actor_map_shape": model.actor_map_shape,
        "actor_metadata_dim": model.actor_metadata_dim,
    }


def _edge_features_tensor(env: GridCoverageEnv, model: ActorCritic, device: torch.device) -> torch.Tensor | None:
    """仅在模型启用边特征时将环境相对几何转换为张量。"""

    if not _uses_edge_features(model):
        return None
    return torch.as_tensor(env.neighbor_features(), dtype=torch.float32, device=device)


def _uses_edge_features(model: ActorCritic) -> bool:
    """判断当前 GAT 是否需要环境提供 agent 对几何特征。"""

    return model.use_graph_attention and model.gat_edge_dim > 0


def _node_messages_tensor(env: GridCoverageEnv, model: ActorCritic, device: torch.device) -> torch.Tensor | None:
    """仅在消息化 actor 开启时取得 map-intent 节点消息。"""

    if model.node_message_dim <= 0:
        return None
    return torch.as_tensor(env.node_messages(), dtype=torch.float32, device=device)


def _action_mask_tensor(env: GridCoverageEnv, config: ExperimentConfig, device: torch.device) -> torch.Tensor | None:
    """Build the decentralized feasibility mask only for configured experiments."""

    if not config.ppo.use_action_mask:
        return None
    return torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device)


def _slugify(text: str) -> str:
    """将课程名称转换为适合目录名的 ASCII 标识。"""

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "course"
