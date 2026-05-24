from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runtime import configure_runtime

configure_runtime()

import numpy as np
import torch

from .config import ExperimentConfig, load_config
from .env import GridCoverageEnv
from .ppo import ActorCritic


def resolve_runtime_config(config: ExperimentConfig, checkpoint_path: str | Path) -> ExperimentConfig:
    checkpoint_path = Path(checkpoint_path)
    manifest_path = checkpoint_path.parent / "course_config.json"
    if manifest_path.exists():
        return load_config(manifest_path)
    return config


def load_policy(config: ExperimentConfig, checkpoint_path: str | Path) -> ActorCritic:
    config = resolve_runtime_config(config, checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    hidden_dim = int(payload.get("hidden_dim", config.ppo.hidden_dim))
    observation_dim = int(payload["observation_dim"])
    action_dim = int(payload["action_dim"])
    critic_type = payload.get("critic_type")
    use_graph_attention = bool(payload.get("use_graph_attention", False))
    gat_use_edge_features = bool(payload.get("gat_use_edge_features", config.ppo.gat_use_edge_features))
    gat_edge_dim = int(payload.get("gat_edge_dim", GridCoverageEnv(config.env).neighbor_feature_dim if gat_use_edge_features else 0))
    gat_num_heads = int(payload.get("gat_num_heads", config.ppo.gat_num_heads))
    gat_residual = bool(payload.get("gat_residual", config.ppo.gat_residual))
    gat_attention_dropout = float(payload.get("gat_attention_dropout", config.ppo.gat_attention_dropout))
    node_message_dim = int(payload.get("node_message_dim", 0))
    if critic_type == "spatial" or "state_shape" in payload:
        model = ActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            state_shape=(config.env.height, config.env.width),
            state_channels=int(payload.get("state_channels", 5)),
            state_metadata_dim=int(payload.get("state_metadata_dim", 7)),
            use_graph_attention=use_graph_attention,
            gat_num_heads=gat_num_heads,
            gat_edge_dim=gat_edge_dim,
            gat_residual=gat_residual,
            gat_attention_dropout=gat_attention_dropout,
            node_message_dim=node_message_dim,
        )
    else:
        model = ActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            state_dim=int(payload.get("state_dim", observation_dim)),
            use_graph_attention=use_graph_attention,
            gat_num_heads=gat_num_heads,
            gat_edge_dim=gat_edge_dim,
            gat_residual=gat_residual,
            gat_attention_dropout=gat_attention_dropout,
            node_message_dim=node_message_dim,
        )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def evaluate_policy(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    deterministic: bool = True,
    budgets: list[int] | None = None,
    stall_steps: int = 50,
) -> dict[str, Any]:
    config = resolve_runtime_config(config, checkpoint_path)
    env = GridCoverageEnv(config.env)
    model = load_policy(config, checkpoint_path)
    observation = _agent_observations(env, env.reset(seed=config.env.seed))
    state = env.global_state()
    trajectories = [[position] for position in env.positions]
    coverage_curve = [env.coverage_ratio()]
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}

    while not done:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        state_tensor = torch.as_tensor(np.repeat(state[None, :], env.num_agents, axis=0), dtype=torch.float32)
        neighbor_mask = torch.as_tensor(env.neighbor_mask(), dtype=torch.bool)
        edge_features = (
            torch.as_tensor(env.neighbor_features(), dtype=torch.float32)
            if model.use_graph_attention and model.gat_edge_dim > 0
            else None
        )
        node_messages = (
            torch.as_tensor(env.node_messages(), dtype=torch.float32)
            if model.node_message_dim > 0
            else None
        )
        with torch.no_grad():
            actions, _, _ = model.act_batch(
                obs_tensor,
                state_tensor,
                neighbor_mask=neighbor_mask,
                edge_features=edge_features,
                node_messages=node_messages,
                deterministic=deterministic,
            )
        result = env.step(actions.cpu().numpy().tolist())
        rewards = np.asarray(result.reward, dtype=np.float32)
        total_reward += float(rewards.mean() if rewards.ndim > 0 else rewards)
        observation = _agent_observations(env, result.observation)
        state = result.state
        done = result.done
        info = result.info
        for index, position in enumerate(env.positions):
            trajectories[index].append(position)
        coverage_curve.append(env.coverage_ratio())

    summary = {
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
            budgets=budgets,
            stall_steps=stall_steps,
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


def coverage_efficiency_metrics(
    trajectories: list[list[tuple[int, int]]],
    coverage_curve: list[float],
    max_steps: int,
    budgets: list[int] | None = None,
    stall_steps: int = 50,
) -> dict[str, Any]:
    if not coverage_curve:
        raise ValueError("coverage_curve must contain at least the initial coverage ratio")
    horizon = max(int(max_steps), 1)
    metric_budgets = _normalize_budgets(budgets, horizon)
    actual_steps = len(coverage_curve) - 1
    terminal_coverage = float(coverage_curve[-1])
    padded_curve = list(coverage_curve[1 : horizon + 1])
    if len(padded_curve) < horizon:
        padded_curve.extend([terminal_coverage] * (horizon - len(padded_curve)))

    metrics: dict[str, Any] = {
        "metric_budgets": metric_budgets,
        "coverage_auc": float(np.mean(padded_curve)),
    }
    for budget in metric_budgets:
        metrics[f"coverage_at_{budget}"] = float(padded_curve[budget - 1])

    threshold_steps: dict[int, int | None] = {}
    for percentage in (90, 95, 99):
        threshold = percentage / 100.0
        threshold_steps[percentage] = next(
            (step for step, coverage in enumerate(coverage_curve) if coverage >= threshold),
            None,
        )
        metrics[f"t{percentage}"] = threshold_steps[percentage]

    total_visits = sum(len(path) for path in trajectories)
    unique_visits = len({cell for path in trajectories for cell in path})
    metrics["repeat_ratio"] = float(max(total_visits - unique_visits, 0) / max(total_visits, 1))
    metrics["repeat_ratio_after_90"] = _repeat_ratio_after_threshold(trajectories, threshold_steps[90])
    metrics["inter_agent_overlap_ratio"] = _inter_agent_overlap_ratio(trajectories)

    stall_coverage = _stall_coverage(coverage_curve[: actual_steps + 1], max(int(stall_steps), 1))
    metrics["stall_steps"] = max(int(stall_steps), 1)
    metrics["stalled"] = int(stall_coverage is not None)
    metrics["stall_coverage"] = stall_coverage
    metrics["stall_termination_coverage"] = float(stall_coverage if stall_coverage is not None else terminal_coverage)
    return metrics


def _agent_observations(env: GridCoverageEnv, observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    if observation.ndim == 1:
        return observation.reshape(1, -1)
    return observation


def _serialize_trajectory(trajectory: Any) -> Any:
    if not trajectory:
        return []
    first = trajectory[0]
    if isinstance(first, tuple):
        return [list(cell) for cell in trajectory]
    return [[list(cell) for cell in path] for path in trajectory]


def _normalize_budgets(budgets: list[int] | None, max_steps: int) -> list[int]:
    if budgets is None:
        budgets = list(range(100, max_steps + 1, 100))
        if not budgets or budgets[-1] != max_steps:
            budgets.append(max_steps)
    normalized = sorted({min(max(int(budget), 1), max_steps) for budget in budgets})
    return normalized or [max_steps]


def _repeat_ratio_after_threshold(trajectories: list[list[tuple[int, int]]], threshold_step: int | None) -> float:
    if threshold_step is None:
        return 0.0
    visited = {path[0] for path in trajectories if path}
    repeats = 0
    transitions = 0
    actual_steps = max((len(path) for path in trajectories), default=1) - 1
    for step in range(1, actual_steps + 1):
        positions = [path[step] for path in trajectories if step < len(path)]
        if step > threshold_step:
            transitions += len(positions)
            repeats += sum(position in visited for position in positions)
        visited.update(positions)
    return float(repeats / max(transitions, 1))


def _inter_agent_overlap_ratio(trajectories: list[list[tuple[int, int]]]) -> float:
    visits_by_agent = [set(path) for path in trajectories]
    union = set().union(*visits_by_agent) if visits_by_agent else set()
    overlapped = {
        cell
        for cell in union
        if sum(cell in visited for visited in visits_by_agent) > 1
    }
    return float(len(overlapped) / max(len(union), 1))


def _stall_coverage(coverage_curve: list[float], stall_steps: int) -> float | None:
    stalled_for = 0
    for before, after in zip(coverage_curve, coverage_curve[1:]):
        stalled_for = stalled_for + 1 if after <= before + 1e-12 else 0
        if stalled_for >= stall_steps and after < 1.0:
            return float(after)
    return None
