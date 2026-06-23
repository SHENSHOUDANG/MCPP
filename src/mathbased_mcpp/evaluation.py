from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any

from .runtime import configure_runtime

configure_runtime()

import numpy as np
import torch

from .config import ExperimentConfig, load_config
from .cuap import build_cuap_step_inputs, scaled_cuap_prior
from .env import GridCoverageEnv
from .ppo import ActorCritic
from .utils import agent_observations, serialize_trajectory


def resolve_runtime_config(config: ExperimentConfig, checkpoint_path: str | Path) -> ExperimentConfig:
    checkpoint_path = Path(checkpoint_path)
    manifest_path = checkpoint_path.parent / "course_config.json"
    if manifest_path.exists():
        runtime_config = load_config(manifest_path)
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        env_manifest = raw_manifest.get("env", {})
        if (
            "use_legacy_truth_coverage_observation" not in env_manifest
            and not runtime_config.env.use_explicit_map_memory
        ):
            runtime_config.env.use_legacy_truth_coverage_observation = True
        return runtime_config
    return config


def load_policy(config: ExperimentConfig, checkpoint_path: str | Path) -> ActorCritic:
    config = resolve_runtime_config(config, checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    use_phase_critics = bool(payload.get("use_phase_critics", False))
    use_phase_actors = bool(payload.get("use_phase_actors", False))
    phase_metadata_index = int(payload.get("phase_metadata_index", GridCoverageEnv.base_state_metadata_dim))
    cuap_meta = payload.get("cuap", {})
    if not isinstance(cuap_meta, dict):
        cuap_meta = {}
    use_gated_cuap = bool(cuap_meta.get("gated", payload.get("use_gated_cuap", False)))
    cuap_beta = float(cuap_meta.get("beta", payload.get("cuap_beta", config.cuap.beta)))
    cuap_gate_hidden_dim = int(
        cuap_meta.get("gate_hidden_dim", payload.get("cuap_gate_hidden_dim", config.cuap.gate_hidden_dim))
    )
    cuap_gate_init_prob = float(
        cuap_meta.get("gate_init_prob", payload.get("cuap_gate_init_prob", config.cuap.gate_init_prob))
    )
    cuap_gate_detach_actor_features = bool(
        cuap_meta.get(
            "gate_detach_actor_features",
            payload.get("cuap_gate_detach_actor_features", config.cuap.gate_detach_actor_features),
        )
    )
    intent_meta = payload.get("intent_relation", {})
    if not isinstance(intent_meta, dict):
        intent_meta = {}
    use_intent_relation = bool(intent_meta.get("enabled", payload.get("use_intent_relation", False)))
    intent_relation_beta_max = float(
        intent_meta.get("beta_max", payload.get("intent_relation_beta_max", config.ppo.intent_relation_beta_max))
    )
    intent_relation_detach = bool(
        intent_meta.get("detach", payload.get("intent_relation_detach", config.ppo.intent_relation_detach))
    )
    intent_grid_size = int(intent_meta.get("intent_grid_size", payload.get("intent_grid_size", config.env.intent_grid_size)))
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
            use_phase_critics=use_phase_critics,
            use_phase_actors=use_phase_actors,
            phase_metadata_index=phase_metadata_index,
            use_gated_cuap=use_gated_cuap,
            cuap_beta=cuap_beta,
            cuap_gate_hidden_dim=cuap_gate_hidden_dim,
            cuap_gate_init_prob=cuap_gate_init_prob,
            cuap_gate_detach_actor_features=cuap_gate_detach_actor_features,
            use_intent_relation=use_intent_relation,
            intent_relation_beta_max=intent_relation_beta_max,
            intent_relation_detach=intent_relation_detach,
            intent_grid_size=intent_grid_size,
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
            use_phase_critics=use_phase_critics,
            use_phase_actors=use_phase_actors,
            phase_metadata_index=phase_metadata_index,
            use_gated_cuap=use_gated_cuap,
            cuap_beta=cuap_beta,
            cuap_gate_hidden_dim=cuap_gate_hidden_dim,
            cuap_gate_init_prob=cuap_gate_init_prob,
            cuap_gate_detach_actor_features=cuap_gate_detach_actor_features,
            use_intent_relation=use_intent_relation,
            intent_relation_beta_max=intent_relation_beta_max,
            intent_relation_detach=intent_relation_detach,
            intent_grid_size=intent_grid_size,
        )
    model.load_compatible_state_dict(payload["model_state_dict"])
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
    _validate_cuap_config(config, env)
    model = load_policy(config, checkpoint_path)
    observation = agent_observations(env.reset(seed=config.env.seed))
    state = env.global_state()
    trajectories = [[position] for position in env.positions]
    coverage_curve = [env.coverage_ratio()]
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    cuap_diagnostics = _empty_cuap_diagnostics()
    cir_diagnostics = _empty_cir_diagnostics()

    while not done:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
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
        action_mask = torch.as_tensor(env.action_masks(), dtype=torch.bool) if config.ppo.use_action_mask else None
        action_prior = _action_prior_tensor(config, env)
        cuap_tensors = _cuap_step_tensors(config, env)
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
        rewards = np.asarray(result.reward, dtype=np.float32)
        total_reward += float(rewards.mean() if rewards.ndim > 0 else rewards)
        observation = agent_observations(result.observation)
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
    summary.update(_summarize_cuap_diagnostics(cuap_diagnostics))
    summary.update(_summarize_cir_diagnostics(cir_diagnostics))
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = dict(summary)
        serializable["trajectory"] = serialize_trajectory(summary["trajectory"])
        serializable["trajectories"] = [[list(cell) for cell in trajectory] for trajectory in trajectories]
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return summary


def evaluate_two_phase_policy(
    config: ExperimentConfig,
    coverage_checkpoint_path: str | Path,
    return_checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    deterministic: bool = True,
    budgets: list[int] | None = None,
    stall_steps: int = 50,
) -> dict[str, Any]:
    config = resolve_runtime_config(copy.deepcopy(config), coverage_checkpoint_path)
    config.env.use_depot = True
    config.env.require_return_to_depot = True
    config.env.initial_return_mode = False
    env = GridCoverageEnv(config.env)
    _validate_cuap_config(config, env)
    coverage_model = load_policy(config, coverage_checkpoint_path)
    return_model = load_policy(config, return_checkpoint_path)
    observation = agent_observations(env.reset(seed=config.env.seed))
    state = env.global_state()
    trajectories = [[position] for position in env.positions]
    coverage_curve = [env.coverage_ratio()]
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    cuap_diagnostics = _empty_cuap_diagnostics()
    cir_diagnostics = _empty_cir_diagnostics()
    phase_steps = {"coverage": 0, "return": 0}

    while not done:
        model = return_model if env.return_mode else coverage_model
        phase_steps["return" if env.return_mode else "coverage"] += 1
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
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
        action_mask = torch.as_tensor(env.action_masks(), dtype=torch.bool) if config.ppo.use_action_mask else None
        action_prior = _action_prior_tensor(config, env)
        cuap_tensors = _cuap_step_tensors(config, env)
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
        rewards = np.asarray(result.reward, dtype=np.float32)
        total_reward += float(rewards.mean() if rewards.ndim > 0 else rewards)
        observation = agent_observations(result.observation)
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
        "coverage_completed": bool(info.get("coverage_completed", False)),
        "returned_to_depot": bool(info.get("all_at_depot", False)),
        "phase_steps": phase_steps,
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
    summary.update(_summarize_cuap_diagnostics(cuap_diagnostics))
    summary.update(_summarize_cir_diagnostics(cir_diagnostics))
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = dict(summary)
        serializable["trajectory"] = serialize_trajectory(summary["trajectory"])
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


def _validate_cuap_config(config: ExperimentConfig, env: GridCoverageEnv) -> None:
    if not config.cuap.enabled:
        return
    if not config.ppo.use_action_mask:
        raise ValueError("CUAP requires ppo.use_action_mask=true so the action mask is applied after the prior")
    if not env.config.use_explicit_map_memory:
        raise ValueError("CUAP requires env.use_explicit_map_memory=true to avoid global coverage truth leakage")


def _action_prior_tensor(config: ExperimentConfig, env: GridCoverageEnv) -> torch.Tensor | None:
    if config.cuap.gated:
        return None
    prior = scaled_cuap_prior(env, config.cuap, phase="return" if env.return_mode else "coverage")
    if prior is None:
        return None
    return torch.as_tensor(prior, dtype=torch.float32)


def _cuap_step_tensors(config: ExperimentConfig, env: GridCoverageEnv) -> dict[str, torch.Tensor]:
    if not config.cuap.enabled or not config.cuap.gated:
        return {}
    inputs = build_cuap_step_inputs(env, config.cuap, phase="return" if env.return_mode else "coverage")
    return {
        "cuap_prior": torch.as_tensor(inputs.prior, dtype=torch.float32),
        "cuap_confidence": torch.as_tensor(inputs.confidence, dtype=torch.float32),
        "cuap_phase_mask": torch.as_tensor(inputs.phase_mask, dtype=torch.float32),
    }


def _empty_cuap_diagnostics() -> dict[str, list[float]]:
    return {
        "gate": [],
        "effective_strength": [],
        "argmax_change": [],
        "prior_margin": [],
        "prior_spread": [],
    }


def _record_cuap_diagnostics(model: ActorCritic, diagnostics: dict[str, list[float]]) -> None:
    if model.latest_applied_gate is not None:
        diagnostics["gate"].extend(_flatten_tensor_values(model.latest_applied_gate))
    if model.latest_effective_strength is not None:
        diagnostics["effective_strength"].extend(_flatten_tensor_values(model.latest_effective_strength))
    if model.latest_argmax_change is not None:
        diagnostics["argmax_change"].extend(_flatten_tensor_values(model.latest_argmax_change.float()))
    if model.latest_cuap_confidence is not None:
        confidence = model.latest_cuap_confidence.detach().cpu().numpy().reshape(-1, model.latest_cuap_confidence.shape[-1])
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


def _flatten_tensor_values(tensor: torch.Tensor) -> list[float]:
    return [float(item) for item in tensor.detach().cpu().reshape(-1)]


def _empty_cir_diagnostics() -> dict[str, list[float]]:
    return {
        "beta": [],
        "overlap": [],
        "overlap_nonzero": [],
        "attention_entropy": [],
    }


def _record_cir_diagnostics(model: ActorCritic, diagnostics: dict[str, list[float]]) -> None:
    if model.latest_intent_beta is not None:
        diagnostics["beta"].extend(_flatten_tensor_values(model.latest_intent_beta))
    if model.latest_intent_overlap is not None:
        overlap = model.latest_intent_overlap.detach()
        mask = model.latest_intent_mask
        if mask is not None:
            overlap = overlap[mask.to(device=overlap.device, dtype=torch.bool)]
        diagnostics["overlap"].extend(_flatten_tensor_values(overlap))
        diagnostics["overlap_nonzero"].extend(_flatten_tensor_values((overlap > 1e-6).float()))
    if model.latest_attention_entropy is not None:
        diagnostics["attention_entropy"].extend(_flatten_tensor_values(model.latest_attention_entropy))


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
