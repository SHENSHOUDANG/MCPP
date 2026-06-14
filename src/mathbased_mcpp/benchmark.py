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
from .cuap import build_cuap_step_inputs, scaled_cuap_prior
from .env import GridCoverageEnv
from .evaluation import coverage_efficiency_metrics, load_policy, resolve_runtime_config
from .utils import agent_observations, agent_rewards


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
    observation = agent_observations(env.reset(seed=seed))
    state = env.global_state()
    trajectories = [[position] for position in env.positions]
    coverage_curve = [env.coverage_ratio()]
    device = next(model.parameters()).device
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    cuap_diagnostics = _empty_cuap_diagnostics()
    cir_diagnostics = _empty_cir_diagnostics()

    while not done:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        neighbor_mask = torch.as_tensor(env.neighbor_mask(), dtype=torch.bool, device=device)
        edge_features = (
            torch.as_tensor(env.neighbor_features(), dtype=torch.float32, device=device)
            if getattr(model, "use_graph_attention", False) and getattr(model, "gat_edge_dim", 0) > 0
            else None
        )
        node_messages = (
            torch.as_tensor(env.node_messages(), dtype=torch.float32, device=device)
            if getattr(model, "node_message_dim", 0) > 0
            else None
        )
        action_mask = torch.as_tensor(env.action_masks(), dtype=torch.bool, device=device) if config.ppo.use_action_mask else None
        action_prior = _action_prior_tensor(config, env, device)
        cuap_tensors = _cuap_step_tensors(config, env, device)
        with torch.no_grad():
            actions, _, _ = model.act_batch(
                obs_tensor,
                state_tensor,
                neighbor_mask=neighbor_mask,
                edge_features=edge_features,
                node_messages=node_messages,
                action_mask=action_mask,
                action_prior_logits=action_prior,
                **cuap_tensors,
                deterministic=deterministic,
            )
        _record_cuap_diagnostics(model, cuap_diagnostics)
        _record_cir_diagnostics(model, cir_diagnostics)
        result = env.step(actions.cpu().numpy().tolist())
        rewards = agent_rewards(env.num_agents, result.reward)
        total_reward += float(np.mean(rewards))
        observation = agent_observations(result.observation)
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
    row.update(_summarize_cuap_diagnostics(cuap_diagnostics))
    row.update(_summarize_cir_diagnostics(cir_diagnostics))
    row.pop("metric_budgets")
    row.pop("stall_steps")
    return row


def _action_prior_tensor(config: ExperimentConfig, env: GridCoverageEnv, device: torch.device) -> torch.Tensor | None:
    if not config.cuap.enabled or config.cuap.gated:
        return None
    prior = scaled_cuap_prior(env, config.cuap, phase="return" if env.return_mode else "coverage")
    if prior is None:
        return None
    return torch.as_tensor(prior, dtype=torch.float32, device=device)


def _cuap_step_tensors(
    config: ExperimentConfig,
    env: GridCoverageEnv,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not config.cuap.enabled or not config.cuap.gated:
        return {}
    inputs = build_cuap_step_inputs(env, config.cuap, phase="return" if env.return_mode else "coverage")
    return {
        "cuap_prior": torch.as_tensor(inputs.prior, dtype=torch.float32, device=device),
        "cuap_confidence": torch.as_tensor(inputs.confidence, dtype=torch.float32, device=device),
        "cuap_phase_mask": torch.as_tensor(inputs.phase_mask, dtype=torch.float32, device=device),
    }


def _empty_cuap_diagnostics() -> dict[str, list[float]]:
    return {
        "gate": [],
        "effective_strength": [],
        "argmax_change": [],
        "prior_margin": [],
        "prior_spread": [],
    }


def _record_cuap_diagnostics(model: torch.nn.Module, diagnostics: dict[str, list[float]]) -> None:
    applied_gate = getattr(model, "latest_applied_gate", None)
    effective_strength = getattr(model, "latest_effective_strength", None)
    argmax_change = getattr(model, "latest_argmax_change", None)
    confidence_tensor = getattr(model, "latest_cuap_confidence", None)
    if applied_gate is not None:
        diagnostics["gate"].extend(_flatten_tensor_values(applied_gate))
    if effective_strength is not None:
        diagnostics["effective_strength"].extend(_flatten_tensor_values(effective_strength))
    if argmax_change is not None:
        diagnostics["argmax_change"].extend(_flatten_tensor_values(argmax_change.float()))
    if confidence_tensor is not None:
        confidence = confidence_tensor.detach().cpu().numpy().reshape(-1, confidence_tensor.shape[-1])
        if confidence.shape[-1] >= 2:
            diagnostics["prior_margin"].extend(float(item) for item in confidence[:, 0])
            diagnostics["prior_spread"].extend(float(item) for item in confidence[:, 1])


def _summarize_cuap_diagnostics(diagnostics: dict[str, list[float]]) -> dict[str, float]:
    gate = np.asarray(diagnostics["gate"], dtype=np.float32)
    if gate.size == 0:
        return {}
    effective_strength = np.asarray(diagnostics["effective_strength"], dtype=np.float32)
    argmax_change = np.asarray(diagnostics["argmax_change"], dtype=np.float32)
    prior_margin = np.asarray(diagnostics["prior_margin"], dtype=np.float32)
    prior_spread = np.asarray(diagnostics["prior_spread"], dtype=np.float32)
    return {
        "gate_mean": float(gate.mean()),
        "gate_std": float(gate.std()),
        "gate_p10": float(np.percentile(gate, 10)),
        "gate_p50": float(np.percentile(gate, 50)),
        "gate_p90": float(np.percentile(gate, 90)),
        "effective_strength": float(effective_strength.mean()) if effective_strength.size else 0.0,
        "argmax_change_rate": float(argmax_change.mean()) if argmax_change.size else 0.0,
        "prior_margin": float(prior_margin.mean()) if prior_margin.size else 0.0,
        "prior_spread": float(prior_spread.mean()) if prior_spread.size else 0.0,
    }


def _empty_cir_diagnostics() -> dict[str, list[float]]:
    return {
        "beta": [],
        "overlap": [],
        "overlap_nonzero": [],
        "attention_entropy": [],
    }


def _record_cir_diagnostics(model: torch.nn.Module, diagnostics: dict[str, list[float]]) -> None:
    beta = getattr(model, "latest_intent_beta", None)
    overlap = getattr(model, "latest_intent_overlap", None)
    attention_entropy = getattr(model, "latest_attention_entropy", None)
    if beta is not None:
        diagnostics["beta"].extend(_flatten_tensor_values(beta))
    if overlap is not None:
        overlap_tensor = overlap.detach()
        mask = getattr(model, "latest_intent_mask", None)
        if mask is not None:
            overlap_tensor = overlap_tensor[mask.to(device=overlap_tensor.device, dtype=torch.bool)]
        diagnostics["overlap"].extend(_flatten_tensor_values(overlap_tensor))
        diagnostics["overlap_nonzero"].extend(_flatten_tensor_values((overlap_tensor > 1e-6).float()))
    if attention_entropy is not None:
        diagnostics["attention_entropy"].extend(_flatten_tensor_values(attention_entropy))


def _summarize_cir_diagnostics(diagnostics: dict[str, list[float]]) -> dict[str, float]:
    overlap = np.asarray(diagnostics["overlap"], dtype=np.float32)
    if overlap.size == 0:
        return {}
    beta = np.asarray(diagnostics["beta"], dtype=np.float32)
    overlap_nonzero = np.asarray(diagnostics["overlap_nonzero"], dtype=np.float32)
    attention_entropy = np.asarray(diagnostics["attention_entropy"], dtype=np.float32)
    return {
        "cir_beta": float(beta.mean()) if beta.size else 0.0,
        "intent_conflict_rate": float(overlap.mean()),
        "intent_conflict_nonzero": float(overlap_nonzero.mean()) if overlap_nonzero.size else 0.0,
        "attention_entropy": float(attention_entropy.mean()) if attention_entropy.size else 0.0,
    }


def _flatten_tensor_values(tensor: torch.Tensor) -> list[float]:
    return [float(item) for item in tensor.detach().cpu().reshape(-1)]


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
    for field in (
        "gate_mean",
        "gate_std",
        "gate_p10",
        "gate_p50",
        "gate_p90",
        "effective_strength",
        "argmax_change_rate",
        "prior_margin",
        "prior_spread",
    ):
        values = _numeric_values(rows, field)
        if values:
            summary[f"{field}_mean"] = float(np.mean(values))
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
