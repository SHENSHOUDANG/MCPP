from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch import nn

from check_port_inspection_env import build_env
from mathbased_mcpp.port_inspection.mappo import HeterogeneousMappo, PortMappoBatch


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the frozen UAV-USV inspection scheduler with CTDE MAPPO.")
    parser.add_argument("--config", default="configs/port_yangshan_task_initial_v1.toml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--checkpoint-interval", type=int, default=100000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--env-workers", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--interop-threads", type=int, default=None)
    parser.add_argument("--gpu-memory-fraction", type=float, default=None)
    parser.add_argument("--process-priority", choices=("normal", "below_normal", "idle"), default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = _load_config(args.config)
    rl_config = dict(config.get("scheduler_rl", {}))
    _configure_cpu_threads(args, rl_config)
    _configure_process_priority(str(args.process_priority or rl_config.get("process_priority", "below_normal")))
    device = _select_device(str(args.device or rl_config.get("device", "auto")))
    _configure_gpu_budget(device, args, rl_config)
    num_envs = max(int(args.num_envs if args.num_envs is not None else rl_config.get("num_envs", 2)), 1)
    env_workers = max(int(args.env_workers if args.env_workers is not None else rl_config.get("env_workers", num_envs)), 1)
    env_workers = min(env_workers, num_envs)
    envs = [build_env(config) for _ in range(num_envs)]
    env = envs[0]
    output_root = Path(str(args.output_dir or config.get("output_dir", "outputs/port_inspection/scheduler")))
    output_dir = output_root / "scheduler_rl"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = HeterogeneousMappo(
        observation_dim=env.local_observation_dim,
        action_dim=env.action_choices,
        hidden_dim=int(rl_config.get("hidden_dim", 128)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(rl_config.get("learning_rate", 3e-4)))
    gamma = float(rl_config.get("gamma", 0.98))
    gae_lambda = float(rl_config.get("gae_lambda", 0.95))
    clip_ratio = float(rl_config.get("clip_ratio", 0.2))
    rollout_steps = int(rl_config.get("rollout_steps", 128))
    update_epochs = int(rl_config.get("update_epochs", 4))
    entropy_coef = float(rl_config.get("entropy_coef", 0.01))
    value_coef = float(rl_config.get("value_coef", 0.5))

    agent_types = _agent_types(env)
    rows: list[dict[str, float | int]] = []
    total_steps = 0
    episode_index = 0
    resume_state: dict[str, Any] = {}
    next_checkpoint_step = max(int(args.checkpoint_interval), 0)
    if args.resume:
        checkpoint_path = _resolve_resume_path(output_dir, args.resume)
        if checkpoint_path is not None:
            resume_state = _load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                device=device,
            )
            total_steps = int(resume_state.get("total_steps", _checkpoint_step_from_name(checkpoint_path)))
            episode_index = int(resume_state.get("episode_index", 0))
            rows = _load_existing_metrics(output_dir / "scheduler_metrics.csv", max_steps=total_steps)
            if episode_index <= 0:
                episode_index = len(rows)
            next_checkpoint_step = _next_checkpoint_step(total_steps, int(args.checkpoint_interval))
            restored_envs = resume_state.get("envs")
            if isinstance(restored_envs, list) and restored_envs:
                envs = restored_envs
                env = envs[0]
                num_envs = len(envs)
                env_workers = min(env_workers, num_envs)
                agent_types = _agent_types(env)
                _clear_env_view_caches(envs)
            _restore_rng_state(resume_state.get("rng_state"))
            print(f"resumed={checkpoint_path} total_steps={total_steps} episodes={episode_index}", flush=True)

    restored_obs = resume_state.get("obs_by_env")
    restored_rewards = resume_state.get("episode_rewards")
    if isinstance(restored_obs, list) and len(restored_obs) == len(envs):
        obs_by_env = [np.asarray(item, dtype=np.float32) for item in restored_obs]
    else:
        obs_by_env = []
        for env_index, rollout_env in enumerate(envs):
            reset = rollout_env.reset_model(seed=args.seed + episode_index + env_index)
            obs_by_env.append(_obs_matrix(rollout_env, reset.obs_dict))
    if isinstance(restored_rewards, list) and len(restored_rewards) == len(envs):
        episode_rewards = [float(item) for item in restored_rewards]
    else:
        episode_rewards = [0.0 for _ in envs]
    print(
        f"device={device} torch={torch.__version__} "
        f"cuda_available={torch.cuda.is_available()} cpu_threads={torch.get_num_threads()} "
        f"num_envs={num_envs} env_workers={env_workers}",
        flush=True,
    )

    while total_steps < args.steps:
        remaining_steps = max(int(args.steps) - total_steps, 1)
        per_env_steps = max(1, math.ceil(min(rollout_steps * num_envs, remaining_steps) / num_envs))
        batch, obs_by_env, episode_events = _collect_multi_env_rollout(
            envs=envs,
            model=model,
            obs_by_env=obs_by_env,
            agent_types=agent_types,
            rollout_steps=per_env_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
            seed_base=args.seed,
            episode_index=episode_index,
            episode_rewards=episode_rewards,
            env_workers=env_workers,
        )
        total_steps += int(batch.observations.shape[0])
        _mappo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            clip_ratio=clip_ratio,
            update_epochs=update_epochs,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
        )
        for event in episode_events:
            episode_index += 1
            row = dict(event)
            row["episode"] = episode_index
            row["steps"] = min(int(total_steps), int(args.steps))
            rows.append(row)
            print(
                f"episode={episode_index} steps={total_steps} "
                f"reward={row['episode_reward']:.3f} completed={row['completed_tasks']} env={row['env_index']}",
                flush=True,
            )
        if episode_events:
            _write_metrics(output_dir / "scheduler_metrics.csv", rows)
            (output_dir / "scheduler_summary.json").write_text(
                json.dumps(rows[-1], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if next_checkpoint_step and total_steps >= next_checkpoint_step:
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                env=env,
                agent_types=agent_types,
                config_path=args.config,
                path=output_dir / f"scheduler_mappo_step{total_steps}.pt",
                total_steps=total_steps,
                episode_index=episode_index,
                rows=rows,
                args=vars(args),
                envs=envs,
                obs_by_env=obs_by_env,
                episode_rewards=episode_rewards,
            )
            next_checkpoint_step += max(int(args.checkpoint_interval), 1)

    _save_checkpoint(
        model=model,
        optimizer=optimizer,
        env=env,
        agent_types=agent_types,
        config_path=args.config,
        path=output_dir / "scheduler_mappo.pt",
        total_steps=total_steps,
        episode_index=episode_index,
        rows=rows,
        args=vars(args),
        envs=envs,
        obs_by_env=obs_by_env,
        episode_rewards=episode_rewards,
    )
    _write_metrics(output_dir / "scheduler_metrics.csv", rows)
    summary = rows[-1] if rows else {"steps": total_steps, "episode_reward": 0.0}
    (output_dir / "scheduler_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"steps={total_steps}")
    print(f"episodes={episode_index}")
    print(f"checkpoint={output_dir / 'scheduler_mappo.pt'}")
    print(f"metrics={output_dir / 'scheduler_metrics.csv'}")
    print(f"summary={output_dir / 'scheduler_summary.json'}")


def _collect_rollout(
    env,
    model: HeterogeneousMappo,
    obs: np.ndarray,
    agent_types: np.ndarray,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device | None = None,
) -> tuple[PortMappoBatch, np.ndarray, bool, float]:
    if device is None:
        device = next(model.parameters()).device
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_probs: list[np.ndarray] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []
    action_masks: list[np.ndarray] = []
    agent_masks: list[np.ndarray] = []
    alive_masks: list[np.ndarray] = []
    reward_sum = 0.0
    done = False

    for _ in range(rollout_steps):
        info = env.info()
        mask = env.action_masks()
        agent_mask = np.asarray(info["agent_mask"], dtype=bool)
        alive_mask = np.asarray(info["alive_mask"], dtype=bool)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        type_tensor = torch.as_tensor(agent_types, dtype=torch.long, device=device).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        agent_mask_tensor = torch.as_tensor(agent_mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            action_tensor, log_prob_tensor, value_tensor = model.act(obs_tensor, type_tensor, mask_tensor, agent_mask_tensor)

        action = action_tensor.squeeze(0).cpu().numpy().astype(np.int64)
        result = env.step(action.tolist())
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob_tensor.squeeze(0).cpu().numpy().astype(np.float32))
        values.append(float(value_tensor.item()))
        rewards.append(float(result.reward))
        dones.append(float(result.done))
        action_masks.append(mask)
        agent_masks.append(agent_mask)
        alive_masks.append(alive_mask)
        reward_sum += float(result.reward)
        obs = _obs_matrix(env, env.local_observations())
        done = result.done
        if done:
            break

    with torch.no_grad():
        next_value = model.value(
            torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(agent_types, dtype=torch.long, device=device).unsqueeze(0),
            torch.as_tensor(alive_masks[-1], dtype=torch.bool, device=device).unsqueeze(0),
        )
    returns, advantages = _gae(rewards, dones, values, float(next_value.item()), gamma, gae_lambda)

    repeated_types = np.repeat(agent_types[None, :], len(observations), axis=0)
    return (
        PortMappoBatch(
            observations=torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=device),
            actions=torch.as_tensor(np.asarray(actions, dtype=np.int64), dtype=torch.long, device=device),
            old_log_probs=torch.as_tensor(np.asarray(log_probs, dtype=np.float32), dtype=torch.float32, device=device),
            returns=torch.as_tensor(np.asarray(returns, dtype=np.float32), dtype=torch.float32, device=device),
            advantages=torch.as_tensor(np.asarray(advantages, dtype=np.float32), dtype=torch.float32, device=device),
            values=torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32, device=device),
            action_masks=torch.as_tensor(np.asarray(action_masks, dtype=bool), dtype=torch.bool, device=device),
            agent_types=torch.as_tensor(repeated_types, dtype=torch.long, device=device),
            agent_masks=torch.as_tensor(np.asarray(agent_masks, dtype=bool), dtype=torch.bool, device=device),
            alive_masks=torch.as_tensor(np.asarray(alive_masks, dtype=bool), dtype=torch.bool, device=device),
        ),
        obs,
        done,
        reward_sum,
    )


def _collect_multi_env_rollout(
    envs: list[Any],
    model: HeterogeneousMappo,
    obs_by_env: list[np.ndarray],
    agent_types: np.ndarray,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    seed_base: int,
    episode_index: int,
    episode_rewards: list[float],
    env_workers: int = 1,
) -> tuple[PortMappoBatch, list[np.ndarray], list[dict[str, float | int]]]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_probs: list[np.ndarray] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []
    action_masks: list[np.ndarray] = []
    agent_masks: list[np.ndarray] = []
    alive_masks: list[np.ndarray] = []
    env_indices: list[int] = []
    episode_events: list[dict[str, float | int]] = []
    num_envs = len(envs)
    if num_envs == 0:
        raise ValueError("at least one environment is required")

    executor = ThreadPoolExecutor(max_workers=env_workers) if env_workers > 1 else None
    try:
        for _ in range(rollout_steps):
            mask_batch = np.asarray([env.action_masks() for env in envs], dtype=bool)
            agent_mask_batch = np.asarray([_agent_mask(env) for env in envs], dtype=bool)
            alive_mask_batch = np.asarray([_alive_mask(env) for env in envs], dtype=bool)
            obs_tensor = torch.as_tensor(np.asarray(obs_by_env, dtype=np.float32), dtype=torch.float32, device=device)
            type_tensor = torch.as_tensor(
                np.repeat(agent_types[None, :], num_envs, axis=0),
                dtype=torch.long,
                device=device,
            )
            mask_tensor = torch.as_tensor(mask_batch, dtype=torch.bool, device=device)
            agent_mask_tensor = torch.as_tensor(agent_mask_batch, dtype=torch.bool, device=device)
            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = model.act(
                    obs_tensor,
                    type_tensor,
                    mask_tensor,
                    agent_mask_tensor,
                )

            action_batch = action_tensor.cpu().numpy().astype(np.int64)
            log_prob_batch = log_prob_tensor.cpu().numpy().astype(np.float32)
            value_batch = value_tensor.cpu().numpy().astype(np.float32)
            step_results = _step_envs(envs, action_batch, executor)

            for env_index, result in enumerate(step_results):
                observations.append(obs_by_env[env_index])
                actions.append(action_batch[env_index])
                log_probs.append(log_prob_batch[env_index])
                values.append(float(value_batch[env_index]))
                rewards.append(float(result.reward))
                dones.append(float(result.done))
                action_masks.append(mask_batch[env_index])
                agent_masks.append(agent_mask_batch[env_index])
                alive_masks.append(alive_mask_batch[env_index])
                env_indices.append(env_index)
                episode_rewards[env_index] += float(result.reward)

                if result.done:
                    episode_events.append(_episode_row(envs[env_index], result.info, episode_rewards[env_index], env_index))
                    next_seed = seed_base + episode_index + len(episode_events) + num_envs
                    reset = envs[env_index].reset_model(seed=next_seed)
                    obs_by_env[env_index] = _obs_matrix(envs[env_index], reset.obs_dict)
                    episode_rewards[env_index] = 0.0
                else:
                    obs_by_env[env_index] = _obs_matrix(envs[env_index], envs[env_index].local_observations())
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    final_next_values = _bootstrap_values(model, obs_by_env, agent_types, envs, device)
    returns, advantages = _multi_env_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        env_indices=env_indices,
        final_next_values=final_next_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    repeated_types = np.repeat(agent_types[None, :], len(observations), axis=0)
    return (
        PortMappoBatch(
            observations=torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=device),
            actions=torch.as_tensor(np.asarray(actions, dtype=np.int64), dtype=torch.long, device=device),
            old_log_probs=torch.as_tensor(np.asarray(log_probs, dtype=np.float32), dtype=torch.float32, device=device),
            returns=torch.as_tensor(np.asarray(returns, dtype=np.float32), dtype=torch.float32, device=device),
            advantages=torch.as_tensor(np.asarray(advantages, dtype=np.float32), dtype=torch.float32, device=device),
            values=torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32, device=device),
            action_masks=torch.as_tensor(np.asarray(action_masks, dtype=bool), dtype=torch.bool, device=device),
            agent_types=torch.as_tensor(repeated_types, dtype=torch.long, device=device),
            agent_masks=torch.as_tensor(np.asarray(agent_masks, dtype=bool), dtype=torch.bool, device=device),
            alive_masks=torch.as_tensor(np.asarray(alive_masks, dtype=bool), dtype=torch.bool, device=device),
        ),
        obs_by_env,
        episode_events,
    )


def _step_envs(envs: list[Any], action_batch: np.ndarray, executor: ThreadPoolExecutor | None) -> list[Any]:
    if executor is None:
        return [env.step(action_batch[index].tolist()) for index, env in enumerate(envs)]
    futures = [executor.submit(env.step, action_batch[index].tolist()) for index, env in enumerate(envs)]
    return [future.result() for future in futures]


def _bootstrap_values(
    model: HeterogeneousMappo,
    obs_by_env: list[np.ndarray],
    agent_types: np.ndarray,
    envs: list[Any],
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(np.asarray(obs_by_env, dtype=np.float32), dtype=torch.float32, device=device)
        type_tensor = torch.as_tensor(
            np.repeat(agent_types[None, :], len(envs), axis=0),
            dtype=torch.long,
            device=device,
        )
        alive_tensor = torch.as_tensor(np.asarray([_alive_mask(env) for env in envs], dtype=bool), dtype=torch.bool, device=device)
        return model.value(obs_tensor, type_tensor, alive_tensor).cpu().numpy().astype(np.float32)


def _multi_env_gae(
    rewards: list[float],
    dones: list[float],
    values: list[float],
    env_indices: list[int],
    final_next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    returns = [0.0 for _ in rewards]
    advantages = [0.0 for _ in rewards]
    next_value_by_env = {index: float(value) for index, value in enumerate(final_next_values)}
    next_advantage_by_env = {index: 0.0 for index in range(len(final_next_values))}
    for index in reversed(range(len(rewards))):
        env_index = env_indices[index]
        next_non_terminal = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * next_value_by_env[env_index] * next_non_terminal - values[index]
        advantage = delta + gamma * gae_lambda * next_non_terminal * next_advantage_by_env[env_index]
        advantages[index] = advantage
        returns[index] = advantage + values[index]
        next_value_by_env[env_index] = values[index]
        next_advantage_by_env[env_index] = advantage
    return returns, advantages


def _agent_mask(env) -> np.ndarray:
    return np.asarray([bool(platform.alive) for platform in env.platforms], dtype=bool)


def _alive_mask(env) -> np.ndarray:
    return _agent_mask(env)


def _episode_row(env, info: dict[str, Any], episode_reward: float, env_index: int) -> dict[str, float | int]:
    return {
        "env_index": int(env_index),
        "episode_reward": float(episode_reward),
        "completed_tasks": len(env.completed_tasks),
        "risk_exposure_sum": float(info["risk_exposure_sum"]),
        "late_tasks": len(info["late_tasks"]),
        "total_path_length": int(info["total_path_length"]),
        "total_energy": float(info["total_energy"]),
        "total_conflicts": int(info["metrics"]["total_conflicts"]),
        "total_invalid_actions": int(info["metrics"]["total_invalid_actions"]),
        "total_replenishments": int(info["metrics"]["total_replenishments"]),
        "total_returns": int(info["metrics"]["total_returns"]),
    }


def _mappo_update(
    model: HeterogeneousMappo,
    optimizer: torch.optim.Optimizer,
    batch: PortMappoBatch,
    clip_ratio: float,
    update_epochs: int,
    entropy_coef: float,
    value_coef: float,
) -> None:
    advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std(unbiased=False) + 1e-8)
    for _ in range(update_epochs):
        log_probs, entropy, values = model.evaluate_actions(
            observations=batch.observations,
            agent_types=batch.agent_types,
            action_masks=batch.action_masks,
            agent_mask=batch.agent_masks,
            actions=batch.actions,
        )
        ratio = torch.exp(log_probs - batch.old_log_probs)
        expanded_advantages = advantages.unsqueeze(-1)
        mask = batch.agent_masks.to(torch.float32)
        policy_terms = torch.min(
            ratio * expanded_advantages,
            torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * expanded_advantages,
        )
        policy_loss = -(policy_terms * mask).sum() / mask.sum().clamp_min(1.0)
        value_loss = 0.5 * (batch.returns - values).pow(2).mean()
        entropy_loss = -(entropy * mask).sum() / mask.sum().clamp_min(1.0)
        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


def _gae(
    rewards: list[float],
    dones: list[float],
    values: list[float],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    advantages = [0.0 for _ in rewards]
    last_gae = 0.0
    for step in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[step]
        next_val = next_value if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_val * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    returns = [adv + value for adv, value in zip(advantages, values)]
    return returns, advantages


def _obs_matrix(env, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([obs_dict[platform.platform_id] for platform in env.platforms], dtype=np.float32)


def _agent_types(env) -> np.ndarray:
    return np.asarray([0 if platform.platform_type == "UAV" else 1 for platform in env.platforms], dtype=np.int64)


def _write_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "episode",
        "env_index",
        "steps",
        "episode_reward",
        "completed_tasks",
        "risk_exposure_sum",
        "late_tasks",
        "total_path_length",
        "total_energy",
        "total_conflicts",
        "total_invalid_actions",
        "total_replenishments",
        "total_returns",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(
    model: HeterogeneousMappo,
    optimizer: torch.optim.Optimizer | None,
    env,
    agent_types: np.ndarray,
    config_path: str,
    path: Path,
    total_steps: int = 0,
    episode_index: int = 0,
    rows: list[dict[str, float | int]] | None = None,
    args: dict[str, Any] | None = None,
    envs: list[Any] | None = None,
    obs_by_env: list[np.ndarray] | None = None,
    episode_rewards: list[float] | None = None,
) -> None:
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "observation_dim": env.local_observation_dim,
            "action_dim": env.action_choices,
            "agent_types": agent_types.tolist(),
            "config": config_path,
            "model": "heterogeneous_mappo_ctde",
            "total_steps": int(total_steps),
            "episode_index": int(episode_index),
            "rows": rows or [],
            "args": args or {},
            "envs": copy.deepcopy(envs) if envs is not None else None,
            "obs_by_env": [np.asarray(item, dtype=np.float32) for item in obs_by_env] if obs_by_env is not None else None,
            "episode_rewards": [float(item) for item in episode_rewards] if episode_rewards is not None else None,
            "rng_state": _rng_state_payload(),
        },
        path,
    )


def _load_checkpoint(
    checkpoint_path: Path,
    model: HeterogeneousMappo,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    return checkpoint


def _rng_state_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng_state(state: object) -> None:
    if not isinstance(state, dict):
        return
    if state.get("python") is not None:
        random.setstate(state["python"])  # type: ignore[arg-type]
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch_state = state.get("torch")
    if torch.is_tensor(torch_state):
        torch.set_rng_state(torch_state.cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        if isinstance(cuda_state, list):
            cuda_state = [item.cpu() if torch.is_tensor(item) else item for item in cuda_state]
        torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]


def _clear_env_view_caches(envs: list[Any]) -> None:
    for env in envs:
        clear = getattr(env, "_clear_view_cache", None)
        if callable(clear):
            clear()


def _resolve_resume_path(output_dir: Path, requested: str) -> Path | None:
    if requested != "auto":
        path = Path(requested)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    candidates = list(output_dir.glob("scheduler_mappo_step*.pt"))
    final_checkpoint = output_dir / "scheduler_mappo.pt"
    if final_checkpoint.exists():
        candidates.append(final_checkpoint)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (_checkpoint_step_from_name(path), path.stat().st_mtime))


def _checkpoint_step_from_name(path: Path) -> int:
    stem = path.stem
    marker = "_step"
    if marker not in stem:
        return 0
    suffix = stem.rsplit(marker, 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 0


def _next_checkpoint_step(total_steps: int, checkpoint_interval: int) -> int:
    if checkpoint_interval <= 0:
        return 0
    return ((total_steps // checkpoint_interval) + 1) * checkpoint_interval


def _load_existing_metrics(path: Path, max_steps: int) -> list[dict[str, float | int]]:
    if not path.exists():
        return []
    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = _parse_metric_row(row)
            if int(parsed.get("steps", 0)) <= max_steps:
                rows.append(parsed)
    return rows


def _parse_metric_row(row: dict[str, str]) -> dict[str, float | int]:
    parsed: dict[str, float | int] = {}
    int_fields = {
        "episode",
        "env_index",
        "steps",
        "completed_tasks",
        "late_tasks",
        "total_path_length",
        "total_conflicts",
        "total_invalid_actions",
        "total_replenishments",
        "total_returns",
    }
    for key, value in row.items():
        if value in ("", None):
            parsed[key] = -1 if key == "env_index" else 0
            continue
        parsed[key] = int(float(value)) if key in int_fields else float(value)
    parsed.setdefault("env_index", -1)
    return parsed


def _configure_cpu_threads(args, rl_config: dict[str, object]) -> None:
    configured_threads = args.cpu_threads if args.cpu_threads is not None else rl_config.get("cpu_threads", 6)
    configured_interop = args.interop_threads if args.interop_threads is not None else rl_config.get("interop_threads", 2)
    if configured_threads is not None:
        torch.set_num_threads(max(int(configured_threads), 1))
    if configured_interop is not None:
        try:
            torch.set_num_interop_threads(max(int(configured_interop), 1))
        except RuntimeError:
            pass


def _configure_process_priority(priority: str) -> None:
    normalized = priority.lower()
    if normalized == "normal":
        return
    if os.name == "nt":
        classes = {
            "below_normal": 0x00004000,
            "idle": 0x00000040,
        }
        priority_class = classes.get(normalized)
        if priority_class is None:
            return
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), priority_class)
        except Exception:
            pass
        return
    if normalized == "idle":
        nice_value = 15
    elif normalized == "below_normal":
        nice_value = 5
    else:
        return
    try:
        os.nice(nice_value)
    except OSError:
        pass


def _select_device(name: str) -> torch.device:
    normalized = name.lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            print("requested cuda but this PyTorch build cannot access CUDA; falling back to cpu", flush=True)
            return torch.device("cpu")
        return torch.device("cuda")
    if normalized != "auto":
        raise ValueError(f"unsupported device: {name}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_gpu_budget(device: torch.device, args, rl_config: dict[str, object]) -> None:
    if device.type != "cuda":
        return
    fraction = args.gpu_memory_fraction
    if fraction is None:
        fraction = rl_config.get("gpu_memory_fraction", 0.35)
    if fraction is not None:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        try:
            torch.cuda.set_per_process_memory_fraction(min(max(float(fraction), 0.05), 0.95), device=device_index)
        except (RuntimeError, ValueError):
            pass
    torch.backends.cudnn.benchmark = False


def _load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


if __name__ == "__main__":
    main()
