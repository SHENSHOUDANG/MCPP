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
from .evaluation import coverage_efficiency_metrics, load_policy, resolve_runtime_config


def benchmark_policy(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    seeds: list[int],
    obstacle_ratios: list[float | None] | None = None,
    output_path: str | Path | None = None,
    deterministic: bool = True,
    budgets: list[int] | None = None,
    stall_steps: int = 50,
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
            rows.append(
                _evaluate_trial(
                    trial_config,
                    model,
                    seed=int(seed),
                    obstacle_ratio=ratio,
                    deterministic=deterministic,
                    budgets=budgets,
                    stall_steps=stall_steps,
                )
            )

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
    budgets: list[int] | None,
    stall_steps: int,
) -> dict[str, Any]:
    env = GridCoverageEnv(config.env)
    observation = _agent_observations(env, env.reset(seed=seed))
    state = env.global_state()
    trajectories = [[position] for position in env.positions]
    coverage_curve = [env.coverage_ratio()]
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
        coverage_curve.append(env.coverage_ratio())

    row: dict[str, Any] = {
        "seed": seed,
        "obstacle_ratio": "" if obstacle_ratio is None else float(obstacle_ratio),
        "obstacles": len(env.obstacles),
        "free_cells": len(env.free_cells),
        "total_reward": total_reward,
        "coverage_ratio": float(info.get("coverage_ratio", env.coverage_ratio())),
        "completed": int(info.get("completed", False)),
        "steps": int(info.get("step_count", env.step_count)),
        "path_length": int(env.path_length),
    }
    row.update(
        coverage_efficiency_metrics(
            trajectories=trajectories,
            coverage_curve=coverage_curve,
            max_steps=env.config.max_steps,
            budgets=budgets,
            stall_steps=stall_steps,
        )
    )
    row.pop("metric_budgets")
    row.pop("stall_steps")
    return row


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "episodes": 0,
            "coverage_ratio_mean": 0.0,
            "coverage_ratio_min": 0.0,
            "completion_rate": 0.0,
            "path_length_mean": 0.0,
            "steps_mean": 0.0,
            "repeat_ratio_mean": 0.0,
            "coverage_auc_mean": 0.0,
            "repeat_ratio_after_90_mean": 0.0,
            "inter_agent_overlap_ratio_mean": 0.0,
            "stall_rate": 0.0,
            "stall_termination_coverage_mean": 0.0,
            "total_reward_mean": 0.0,
        }
    summary: dict[str, float | int] = {
        "episodes": len(rows),
        "coverage_ratio_mean": _mean(rows, "coverage_ratio"),
        "coverage_ratio_min": _min(rows, "coverage_ratio"),
        "completion_rate": _mean(rows, "completed"),
        "path_length_mean": _mean(rows, "path_length"),
        "steps_mean": _mean(rows, "steps"),
        "repeat_ratio_mean": _mean(rows, "repeat_ratio"),
        "coverage_auc_mean": _mean(rows, "coverage_auc"),
        "repeat_ratio_after_90_mean": _mean(rows, "repeat_ratio_after_90"),
        "inter_agent_overlap_ratio_mean": _mean(rows, "inter_agent_overlap_ratio"),
        "stall_rate": _mean(rows, "stalled"),
        "stall_termination_coverage_mean": _mean(rows, "stall_termination_coverage"),
        "total_reward_mean": _mean(rows, "total_reward"),
    }
    for field in ("t90", "t95", "t99"):
        values = _numeric_values(rows, field)
        summary[f"{field}_mean_reached"] = float(np.mean(values)) if values else 0.0
        summary[f"{field}_reach_rate"] = float(len(values) / len(rows))
    for key in rows[0]:
        if key.startswith("coverage_at_"):
            summary[f"{key}_mean"] = _mean(rows, key)
    return summary


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _min(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.min([float(row[key]) for row in rows]))


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None and row.get(key) != ""]


def _write_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["seed", "obstacle_ratio", "coverage_ratio", "completed"]
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
