from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from .runtime import configure_runtime

configure_runtime()

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig, GridCoverageConfig, GridPosition, build_course_config, select_curriculum_course
from .env import ACTIONS, GridCoverageEnv
from .evaluation import coverage_efficiency_metrics, evaluate_policy
from .ppo import ActorCritic
from .rendering import render_trajectory
from .utils import append_metrics, make_run_dir, set_seed


@dataclass(slots=True)
class ExpertDataset:
    observations: np.ndarray
    actions: np.ndarray
    neighbor_masks: np.ndarray
    action_masks: np.ndarray | None
    edge_features: np.ndarray | None
    node_messages: np.ndarray | None
    episodes: int

    @property
    def transitions(self) -> int:
        return int(self.actions.size)

    @property
    def rollout_steps(self) -> int:
        return int(self.actions.shape[0])


@dataclass(slots=True)
class ImitationPretrainResult:
    checkpoint: Path
    run_dir: Path
    transitions: int
    episodes: int
    final_loss: float
    final_accuracy: float
    expert_render: Path
    bc_render: Path


class BoustrophedonExpert:
    """Rule expert that sweeps rows while using shortest paths around obstacles."""

    def actions(self, env: GridCoverageEnv) -> list[int]:
        current_positions = list(env.positions)
        reserved_targets: set[GridPosition] = set()
        actions: list[int] = []
        for agent_index in range(env.num_agents):
            action = self._action_for_agent(env, agent_index, current_positions, reserved_targets)
            target, valid = env.peek(action, agent_index=agent_index)
            if valid:
                reserved_targets.add(target)
            actions.append(action)
        return actions

    def _action_for_agent(
        self,
        env: GridCoverageEnv,
        agent_index: int,
        current_positions: list[GridPosition],
        reserved_targets: set[GridPosition],
    ) -> int:
        start = current_positions[agent_index]
        other_positions = {position for index, position in enumerate(current_positions) if index != agent_index}
        blocked = set(env.obstacles) | other_positions | set(reserved_targets)
        candidates = self._target_candidates(env, agent_index)
        action = self._first_path_action(env, agent_index, start, candidates, blocked)
        if action is not None:
            return action
        return self._fallback_action(env, agent_index, other_positions, reserved_targets)

    def _target_candidates(self, env: GridCoverageEnv, agent_index: int) -> list[GridPosition]:
        uncovered = env.free_cells - env.covered
        if not uncovered:
            return []
        owned = [cell for cell in uncovered if self._owns_cell(env, cell, agent_index)]
        candidates = owned if owned else list(uncovered)
        return sorted(candidates, key=lambda cell: (self._sweep_order(env, cell), self._agent_distance(env, agent_index, cell)))

    def _owns_cell(self, env: GridCoverageEnv, cell: GridPosition, agent_index: int) -> bool:
        if env.num_agents <= 1:
            return True
        canonical_row, _ = env._canonical_position(cell)
        band = min(canonical_row * env.num_agents // max(env.config.height, 1), env.num_agents - 1)
        return band == agent_index

    def _first_path_action(
        self,
        env: GridCoverageEnv,
        agent_index: int,
        start: GridPosition,
        candidates: list[GridPosition],
        blocked: set[GridPosition],
    ) -> int | None:
        goals = [cell for cell in candidates if cell != start and cell not in blocked]
        if not goals:
            return None
        goal_set = set(goals)
        queue: deque[GridPosition] = deque([start])
        parent: dict[GridPosition, GridPosition | None] = {start: None}
        while queue:
            cell = queue.popleft()
            if cell in goal_set:
                first_step = cell
                while parent[first_step] != start:
                    previous = parent[first_step]
                    if previous is None:
                        return None
                    first_step = previous
                return self._action_to_cell(env, agent_index, first_step)
            for neighbor in self._ordered_neighbors(env, cell):
                if neighbor in parent or neighbor in blocked or neighbor not in env.free_cells:
                    continue
                parent[neighbor] = cell
                queue.append(neighbor)
        return None

    def _ordered_neighbors(self, env: GridCoverageEnv, cell: GridPosition) -> list[GridPosition]:
        neighbors = [(cell[0] + delta[0], cell[1] + delta[1]) for delta in ACTIONS.values()]
        return sorted(neighbors, key=lambda item: self._sweep_order(env, item))

    def _sweep_order(self, env: GridCoverageEnv, cell: GridPosition) -> tuple[int, int]:
        row, col = env._canonical_position(cell)
        if row % 2 == 0:
            return row, col
        return row, env.config.width - 1 - col

    def _agent_distance(self, env: GridCoverageEnv, agent_index: int, cell: GridPosition) -> int:
        position = env.positions[agent_index]
        return abs(position[0] - cell[0]) + abs(position[1] - cell[1])

    def _action_to_cell(self, env: GridCoverageEnv, agent_index: int, target: GridPosition) -> int | None:
        for action in ACTIONS:
            peek_target, valid = env.peek(action, agent_index=agent_index)
            if valid and peek_target == target:
                return action
        return None

    def _fallback_action(
        self,
        env: GridCoverageEnv,
        agent_index: int,
        other_positions: set[GridPosition],
        reserved_targets: set[GridPosition],
    ) -> int:
        legal_actions = env.legal_actions(agent_index=agent_index)
        if not legal_actions:
            return min(ACTIONS)

        def score(action: int) -> tuple[int, int, int, tuple[int, int]]:
            target, valid = env.peek(action, agent_index=agent_index)
            blocked = target in other_positions or target in reserved_targets
            repeated = target in env.covered
            return int(blocked or not valid), int(repeated), self._nearest_uncovered_distance(env, target), self._sweep_order(env, target)

        return min(legal_actions, key=score)

    def _nearest_uncovered_distance(self, env: GridCoverageEnv, target: GridPosition) -> int:
        uncovered = env.free_cells - env.covered
        if not uncovered:
            return 0
        return min(abs(target[0] - cell[0]) + abs(target[1] - cell[1]) for cell in uncovered)


def generate_expert_dataset(
    config: ExperimentConfig,
    episodes: int = 8,
    max_steps_per_episode: int | None = None,
    expert: BoustrophedonExpert | None = None,
) -> ExpertDataset:
    if config.ppo.use_coverage_messages and not config.env.use_explicit_map_memory:
        raise ValueError("use_coverage_messages requires use_explicit_map_memory=true")
    expert = expert or BoustrophedonExpert()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    neighbor_masks: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    edge_features: list[np.ndarray] = []
    node_messages: list[np.ndarray] = []
    use_edge_features = config.ppo.use_graph_attention and config.ppo.gat_use_edge_features
    use_node_messages = config.ppo.use_coverage_messages

    for episode_index in range(max(int(episodes), 1)):
        episode_config = _episode_env_config(config.env, episode_index)
        env = GridCoverageEnv(episode_config)
        observation = _agent_observations(env, env.reset(seed=episode_config.seed))
        max_steps = min(max_steps_per_episode or episode_config.max_steps, episode_config.max_steps)
        for _ in range(max_steps):
            observations.append(observation.copy())
            neighbor_masks.append(env.neighbor_mask().copy())
            if config.ppo.use_action_mask:
                action_masks.append(env.action_masks().copy())
            if use_edge_features:
                edge_features.append(env.neighbor_features().copy())
            if use_node_messages:
                node_messages.append(env.node_messages().copy())

            action_list = expert.actions(env)
            actions.append(np.asarray(action_list, dtype=np.int64))
            result = env.step(action_list)
            observation = _agent_observations(env, result.observation)
            if result.done:
                break

    if not observations:
        raise RuntimeError("expert dataset is empty")

    return ExpertDataset(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        neighbor_masks=np.asarray(neighbor_masks, dtype=bool),
        action_masks=np.asarray(action_masks, dtype=bool) if action_masks else None,
        edge_features=np.asarray(edge_features, dtype=np.float32) if edge_features else None,
        node_messages=np.asarray(node_messages, dtype=np.float32) if node_messages else None,
        episodes=max(int(episodes), 1),
    )


def pretrain_imitation(
    config: ExperimentConfig,
    run_dir: str | Path | None = None,
    course: str | None = None,
    episodes: int = 64,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float | None = None,
) -> ImitationPretrainResult:
    course_config = _course_config(config, course)
    set_seed(course_config.ppo.seed)
    dataset = generate_expert_dataset(course_config, episodes=episodes)
    env = GridCoverageEnv(course_config.env)
    model = _build_model(course_config, env).to(_resolve_device(course_config.ppo.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate or course_config.ppo.learning_rate)
    run_path = Path(run_dir) if run_dir is not None else make_run_dir(Path(course_config.train.run_root) / "imitation")
    run_path.mkdir(parents=True, exist_ok=True)
    run_path.joinpath("course_config.json").write_text(json.dumps(asdict(course_config), indent=2), encoding="utf-8")

    final_loss = 0.0
    final_accuracy = 0.0
    device = next(model.parameters()).device
    batch_steps = max(1, int(batch_size) // max(course_config.env.num_agents, 1))
    indices = np.arange(dataset.rollout_steps)
    for epoch in range(max(int(epochs), 1)):
        np.random.shuffle(indices)
        total_loss = 0.0
        correct = 0
        total = 0
        for start in range(0, dataset.rollout_steps, batch_steps):
            batch_indices = indices[start : start + batch_steps]
            observations = torch.as_tensor(dataset.observations[batch_indices], dtype=torch.float32, device=device)
            actions = torch.as_tensor(dataset.actions[batch_indices], dtype=torch.long, device=device)
            neighbor_mask = torch.as_tensor(dataset.neighbor_masks[batch_indices], dtype=torch.bool, device=device)
            action_mask = (
                torch.as_tensor(dataset.action_masks[batch_indices], dtype=torch.bool, device=device)
                if dataset.action_masks is not None
                else None
            )
            edge_features = (
                torch.as_tensor(dataset.edge_features[batch_indices], dtype=torch.float32, device=device)
                if dataset.edge_features is not None
                else None
            )
            node_messages = (
                torch.as_tensor(dataset.node_messages[batch_indices], dtype=torch.float32, device=device)
                if dataset.node_messages is not None
                else None
            )
            logits = model.actor(
                model._actor_features(
                    observations,
                    neighbor_mask=neighbor_mask,
                    edge_features=edge_features,
                    node_messages=node_messages,
                )
            )
            logits = model._mask_action_logits(logits, action_mask)
            loss = nn.functional.cross_entropy(logits.reshape(-1, model.action_dim), actions.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), course_config.ppo.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item()) * int(actions.numel())
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == actions).sum().item())
            total += int(actions.numel())

        final_loss = total_loss / max(total, 1)
        final_accuracy = correct / max(total, 1)
        append_metrics(
            run_path / "imitation_metrics.csv",
            [
                {
                    "episode": epoch,
                    "loss": final_loss,
                    "accuracy": final_accuracy,
                    "transitions": dataset.transitions,
                }
            ],
        )

    checkpoint_path = run_path / "bc_policy.pt"
    torch.save(_checkpoint_payload(course_config, model, dataset, final_loss, final_accuracy), checkpoint_path)
    expert_summary = rollout_expert_policy(course_config, output_path=run_path / "expert_trajectory.json")
    expert_render = render_trajectory(course_config, expert_summary["trajectory"], run_path / "expert_trajectory.png")
    bc_summary = evaluate_policy(course_config, checkpoint_path, output_path=run_path / "bc_trajectory.json")
    bc_render = render_trajectory(course_config, bc_summary["trajectory"], run_path / "bc_trajectory.png")
    run_path.joinpath("imitation_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "episodes": dataset.episodes,
                "transitions": dataset.transitions,
                "final_loss": final_loss,
                "final_accuracy": final_accuracy,
                "expert_coverage_ratio": expert_summary["coverage_ratio"],
                "expert_render": str(expert_render),
                "bc_coverage_ratio": bc_summary["coverage_ratio"],
                "bc_expert_coverage_gap": float(expert_summary["coverage_ratio"]) - float(bc_summary["coverage_ratio"]),
                "bc_render": str(bc_render),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ImitationPretrainResult(
        checkpoint=checkpoint_path,
        run_dir=run_path,
        transitions=dataset.transitions,
        episodes=dataset.episodes,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        expert_render=expert_render,
        bc_render=bc_render,
    )


def rollout_expert_policy(
    config: ExperimentConfig,
    output_path: str | Path | None = None,
    expert: BoustrophedonExpert | None = None,
) -> dict[str, object]:
    expert = expert or BoustrophedonExpert()
    env = GridCoverageEnv(config.env)
    env.reset(seed=config.env.seed)
    trajectories = [[position] for position in env.positions]
    coverage_curve = [env.coverage_ratio()]
    total_reward = 0.0
    done = False
    info: dict[str, object] = {}
    while not done:
        result = env.step(expert.actions(env))
        rewards = np.asarray(result.reward, dtype=np.float32)
        total_reward += float(rewards.mean() if rewards.ndim > 0 else rewards)
        done = result.done
        info = result.info
        for index, position in enumerate(env.positions):
            trajectories[index].append(position)
        coverage_curve.append(env.coverage_ratio())

    summary: dict[str, object] = {
        "total_reward": total_reward,
        "coverage_ratio": info.get("coverage_ratio", env.coverage_ratio()),
        "path_length": env.path_length,
        "path_lengths": list(env.path_lengths),
        "completed": bool(info.get("completed", False)),
        "steps": info.get("step_count", env.step_count),
        "trajectory": trajectories[0] if env.num_agents == 1 else trajectories,
        "trajectories": trajectories,
    }
    summary.update(
        coverage_efficiency_metrics(
            trajectories=trajectories,
            coverage_curve=coverage_curve,
            max_steps=env.config.max_steps,
        )
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = dict(summary)
        serializable["trajectory"] = _serialize_trajectory(summary["trajectory"])
        serializable["trajectories"] = [[list(cell) for cell in trajectory] for trajectory in trajectories]
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return summary


def _course_config(config: ExperimentConfig, course: str | None) -> ExperimentConfig:
    if config.curriculum and config.curriculum.courses:
        if course is None:
            raise ValueError("curriculum configs require --course for imitation pretraining")
        _, selected_course = select_curriculum_course(config, course_name=course)
        return build_course_config(config, selected_course)
    return config


def _episode_env_config(base_config: GridCoverageConfig, episode_index: int) -> GridCoverageConfig:
    env_config = copy.deepcopy(base_config)
    env_config.seed = int(base_config.seed) + episode_index
    if env_config.random_obstacle_seeds:
        env_config.random_obstacle_seed = env_config.random_obstacle_seeds[
            episode_index % len(env_config.random_obstacle_seeds)
        ]
        env_config.random_obstacle_seeds = []
    elif env_config.obstacle_ratio is not None or env_config.random_obstacle_count > 0:
        env_config.random_obstacle_seed = int(env_config.random_obstacle_seed) + episode_index
    return env_config


def _build_model(config: ExperimentConfig, env: GridCoverageEnv) -> ActorCritic:
    return ActorCritic(
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
    )


def _checkpoint_payload(
    config: ExperimentConfig,
    model: ActorCritic,
    dataset: ExpertDataset,
    final_loss: float,
    final_accuracy: float,
) -> dict[str, object]:
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
        "use_action_mask": config.ppo.use_action_mask,
        "pretrain_type": "boustrophedon_behavior_cloning",
        "expert": "boustrophedon_shortest_path",
        "expert_episodes": dataset.episodes,
        "expert_transitions": dataset.transitions,
        "bc_final_loss": final_loss,
        "bc_final_accuracy": final_accuracy,
    }


def _agent_observations(env: GridCoverageEnv, observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    if observation.ndim == 1:
        return observation.reshape(1, -1)
    return observation


def _serialize_trajectory(trajectory: object) -> object:
    if isinstance(trajectory, list) and trajectory and isinstance(trajectory[0], tuple):
        return [list(cell) for cell in trajectory]
    if isinstance(trajectory, list):
        return [[list(cell) for cell in path] for path in trajectory]
    return trajectory


def _resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)
