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
    if critic_type == "spatial" or "state_shape" in payload:
        model = ActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            state_shape=(config.env.height, config.env.width),
            state_channels=int(payload.get("state_channels", 5)),
            state_metadata_dim=int(payload.get("state_metadata_dim", 7)),
            use_graph_attention=bool(payload.get("use_graph_attention", False)),
        )
    else:
        model = ActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            state_dim=int(payload.get("state_dim", observation_dim)),
            use_graph_attention=bool(payload.get("use_graph_attention", False)),
        )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def evaluate_policy(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    config = resolve_runtime_config(config, checkpoint_path)
    env = GridCoverageEnv(config.env)
    model = load_policy(config, checkpoint_path)
    observation = _agent_observations(env, env.reset(seed=config.env.seed))
    state = env.global_state()
    trajectories = [[position] for position in env.positions]
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}

    while not done:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        state_tensor = torch.as_tensor(np.repeat(state[None, :], env.num_agents, axis=0), dtype=torch.float32)
        neighbor_mask = torch.as_tensor(env.neighbor_mask(), dtype=torch.bool)
        with torch.no_grad():
            actions, _, _ = model.act_batch(obs_tensor, state_tensor, neighbor_mask=neighbor_mask, deterministic=deterministic)
        result = env.step(actions.cpu().numpy().tolist())
        rewards = np.asarray(result.reward, dtype=np.float32)
        total_reward += float(rewards.mean() if rewards.ndim > 0 else rewards)
        observation = _agent_observations(env, result.observation)
        state = result.state
        done = result.done
        info = result.info
        for index, position in enumerate(env.positions):
            trajectories[index].append(position)

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
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = dict(summary)
        serializable["trajectory"] = _serialize_trajectory(summary["trajectory"])
        serializable["trajectories"] = [[list(cell) for cell in trajectory] for trajectory in trajectories]
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return summary


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
