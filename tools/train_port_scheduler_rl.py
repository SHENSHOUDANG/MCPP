from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

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
    parser.add_argument("--config", default="configs/port_shanghai_yangshan_v1.toml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = _load_config(args.config)
    env = build_env(config)
    rl_config = dict(config.get("scheduler_rl", {}))
    output_dir = Path(str(config.get("output_dir", "outputs/port_inspection/scheduler"))) / "scheduler_rl"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = HeterogeneousMappo(
        observation_dim=env.local_observation_dim,
        action_dim=env.action_choices,
        hidden_dim=int(rl_config.get("hidden_dim", 128)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(rl_config.get("learning_rate", 3e-4)))
    gamma = float(rl_config.get("gamma", 0.98))
    gae_lambda = float(rl_config.get("gae_lambda", 0.95))
    clip_ratio = float(rl_config.get("clip_ratio", 0.2))
    rollout_steps = int(rl_config.get("rollout_steps", 128))
    update_epochs = int(rl_config.get("update_epochs", 4))
    entropy_coef = float(rl_config.get("entropy_coef", 0.01))
    value_coef = float(rl_config.get("value_coef", 0.5))

    reset = env.reset_model(seed=args.seed)
    obs = _obs_matrix(env, reset.obs_dict)
    agent_types = _agent_types(env)
    total_steps = 0
    episode_reward = 0.0
    episode_index = 0
    rows: list[dict[str, float | int]] = []

    while total_steps < args.steps:
        batch, obs, done, reward_sum = _collect_rollout(
            env=env,
            model=model,
            obs=obs,
            agent_types=agent_types,
            rollout_steps=min(rollout_steps, args.steps - total_steps),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        episode_reward += reward_sum
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
        if done:
            episode_index += 1
            info = env.info()
            rows.append(
                {
                    "episode": episode_index,
                    "steps": total_steps,
                    "episode_reward": episode_reward,
                    "completed_tasks": len(env.completed_tasks),
                    "risk_exposure_sum": float(info["risk_exposure_sum"]),
                    "late_tasks": len(info["late_tasks"]),
                    "total_path_length": int(info["total_path_length"]),
                    "total_energy": float(info["total_energy"]),
                    "total_conflicts": int(info["metrics"]["total_conflicts"]),
                    "total_invalid_actions": int(info["metrics"]["total_invalid_actions"]),
                }
            )
            reset = env.reset_model(seed=args.seed + episode_index)
            obs = _obs_matrix(env, reset.obs_dict)
            episode_reward = 0.0

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "observation_dim": env.local_observation_dim,
            "action_dim": env.action_choices,
            "agent_types": agent_types.tolist(),
            "config": args.config,
            "model": "heterogeneous_mappo_ctde",
        },
        output_dir / "scheduler_mappo.pt",
    )
    _write_metrics(output_dir / "scheduler_metrics.csv", rows)
    summary = rows[-1] if rows else {"steps": total_steps, "episode_reward": episode_reward}
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
) -> tuple[PortMappoBatch, np.ndarray, bool, float]:
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
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        type_tensor = torch.as_tensor(agent_types, dtype=torch.long).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        agent_mask_tensor = torch.as_tensor(agent_mask, dtype=torch.bool).unsqueeze(0)
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
            torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(agent_types, dtype=torch.long).unsqueeze(0),
            torch.as_tensor(alive_masks[-1], dtype=torch.bool).unsqueeze(0),
        )
    returns, advantages = _gae(rewards, dones, values, float(next_value.item()), gamma, gae_lambda)

    repeated_types = np.repeat(agent_types[None, :], len(observations), axis=0)
    return (
        PortMappoBatch(
            observations=torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32),
            actions=torch.as_tensor(np.asarray(actions, dtype=np.int64), dtype=torch.long),
            old_log_probs=torch.as_tensor(np.asarray(log_probs, dtype=np.float32), dtype=torch.float32),
            returns=torch.as_tensor(np.asarray(returns, dtype=np.float32), dtype=torch.float32),
            advantages=torch.as_tensor(np.asarray(advantages, dtype=np.float32), dtype=torch.float32),
            values=torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32),
            action_masks=torch.as_tensor(np.asarray(action_masks, dtype=bool), dtype=torch.bool),
            agent_types=torch.as_tensor(repeated_types, dtype=torch.long),
            agent_masks=torch.as_tensor(np.asarray(agent_masks, dtype=bool), dtype=torch.bool),
            alive_masks=torch.as_tensor(np.asarray(alive_masks, dtype=bool), dtype=torch.bool),
        ),
        obs,
        done,
        reward_sum,
    )


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
        "steps",
        "episode_reward",
        "completed_tasks",
        "risk_exposure_sum",
        "late_tasks",
        "total_path_length",
        "total_energy",
        "total_conflicts",
        "total_invalid_actions",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


if __name__ == "__main__":
    main()
