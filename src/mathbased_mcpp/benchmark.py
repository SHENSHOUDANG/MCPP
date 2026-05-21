from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

from .runtime import configure_runtime

configure_runtime()

import numpy as np
import torch

from .config import ExperimentConfig
from .env import GridCoverageEnv
from .evaluation import load_policy, resolve_runtime_config


def benchmark_policy(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    seeds: list[int],
    obstacle_ratios: list[float | None] | None = None,
    output_path: str | Path | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    runtime_config = resolve_runtime_config(config, checkpoint)
    model = load_policy(runtime_config, checkpoint)
    ratios = obstacle_ratios or [runtime_config.env.obstacle_ratio]

    rows: list[dict[str, float | int | str]] = []
    for ratio in ratios:
        for seed in seeds:
            trial_config = copy.deepcopy(runtime_config)
            trial_config.env.seed = int(seed)
            trial_config.env.random_obstacle_seed = int(seed)
            trial_config.env.random_obstacle_seeds = []
            trial_config.env.map_refresh_episodes = 0
            trial_config.env.obstacle_ratio = None if ratio is None else float(ratio)
            rows.append(_evaluate_trial(trial_config, model, seed=int(seed), obstacle_ratio=ratio, deterministic=deterministic))

    summary = _summarize_rows(rows)
    if output_path is not None:
        _write_rows(Path(output_path), rows)
    summary["rows"] = rows
    return summary


def _evaluate_trial(
    config: ExperimentConfig,
    model: torch.nn.Module,
    seed: int,
    obstacle_ratio: float | None,
    deterministic: bool,
) -> dict[str, float | int | str]:
    env = GridCoverageEnv(config.env)
    observation = _agent_observations(env, env.reset(seed=seed))
    state = env.global_state()
    trajectories = [[position] for position in env.positions]
    device = next(model.parameters()).device
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}

    while not done:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(np.repeat(state[None, :], env.num_agents, axis=0), dtype=torch.float32, device=device)
        neighbor_mask = torch.as_tensor(env.neighbor_mask(), dtype=torch.bool, device=device)
        edge_features = (
            torch.as_tensor(env.neighbor_features(), dtype=torch.float32, device=device)
            if getattr(model, "use_graph_attention", False) and getattr(model, "gat_edge_dim", 0) > 0
            else None
        )
        with torch.no_grad():
            actions, _, _ = model.act_batch(
                obs_tensor,
                state_tensor,
                neighbor_mask=neighbor_mask,
                edge_features=edge_features,
                deterministic=deterministic,
            )
        result = env.step(actions.cpu().numpy().tolist())
        rewards = _agent_rewards(env, result.reward)
        total_reward += float(np.mean(rewards))
        observation = _agent_observations(env, result.observation)
        state = result.state
        done = result.done
        info = result.info
        for index, position in enumerate(env.positions):
            trajectories[index].append(position)

    total_visits = sum(len(path) for path in trajectories)
    unique_visits = len({cell for path in trajectories for cell in path})
    repeat_ratio = max(total_visits - unique_visits, 0) / max(total_visits, 1)
    return {
        "seed": seed,
        "obstacle_ratio": "" if obstacle_ratio is None else float(obstacle_ratio),
        "obstacles": len(env.obstacles),
        "free_cells": len(env.free_cells),
        "total_reward": total_reward,
        "coverage_ratio": float(info.get("coverage_ratio", env.coverage_ratio())),
        "completed": int(info.get("completed", False)),
        "steps": int(info.get("step_count", env.step_count)),
        "path_length": int(env.path_length),
        "repeat_ratio": float(repeat_ratio),
    }


def _summarize_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    if not rows:
        return {
            "episodes": 0,
            "coverage_ratio_mean": 0.0,
            "coverage_ratio_min": 0.0,
            "completion_rate": 0.0,
            "path_length_mean": 0.0,
            "steps_mean": 0.0,
            "repeat_ratio_mean": 0.0,
            "total_reward_mean": 0.0,
        }
    return {
        "episodes": len(rows),
        "coverage_ratio_mean": _mean(rows, "coverage_ratio"),
        "coverage_ratio_min": _min(rows, "coverage_ratio"),
        "completion_rate": _mean(rows, "completed"),
        "path_length_mean": _mean(rows, "path_length"),
        "steps_mean": _mean(rows, "steps"),
        "repeat_ratio_mean": _mean(rows, "repeat_ratio"),
        "total_reward_mean": _mean(rows, "total_reward"),
    }


def _mean(rows: list[dict[str, float | int | str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _min(rows: list[dict[str, float | int | str]], key: str) -> float:
    return float(np.min([float(row[key]) for row in rows]))


def _write_rows(output_path: Path, rows: list[dict[str, float | int | str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "obstacle_ratio",
        "obstacles",
        "free_cells",
        "total_reward",
        "coverage_ratio",
        "completed",
        "steps",
        "path_length",
        "repeat_ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _agent_observations(env: GridCoverageEnv, observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    if observation.ndim == 1:
        return observation.reshape(1, -1)
    return observation


def _agent_rewards(env: GridCoverageEnv, reward: float | np.ndarray) -> np.ndarray:
    reward_array = np.asarray(reward, dtype=np.float32)
    if reward_array.ndim == 0:
        return np.full(env.num_agents, float(reward_array), dtype=np.float32)
    return reward_array
