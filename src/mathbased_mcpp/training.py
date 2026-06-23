from __future__ import annotations

import json
import copy
import math
import os
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .runtime import configure_runtime

configure_runtime()

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig, build_course_config, select_curriculum_course
from .cuap import build_cuap_step_inputs, scaled_cuap_prior
from .env import GridCoverageEnv
from .ppo import ActorCritic, RolloutBatch
from .utils import (
    agent_observations,
    agent_rewards,
    append_metrics,
    checkpoint_model_metadata,
    make_run_dir,
    make_tensorboard_writer,
    resolve_device,
    set_seed,
    write_tensorboard_rows,
)


def train_ppo(
    config: ExperimentConfig,
    env: GridCoverageEnv | None = None,
    run_dir: str | Path | None = None,
    course: str | None = None,
    previous_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    if config.curriculum and config.curriculum.courses:
        if course is not None:
            return _train_single_curriculum_course(
                config=config,
                course_name=course,
                run_dir=run_dir,
                previous_checkpoint=previous_checkpoint,
                resume_checkpoint=resume_checkpoint,
            )
        return _train_curriculum(config, run_dir=run_dir)
    return _train_single_course(config, env=env, run_dir=run_dir, resume_checkpoint_path=resume_checkpoint)


def _train_curriculum(config: ExperimentConfig, run_dir: str | Path | None = None) -> Path:
    assert config.curriculum is not None
    master_run_path = Path(run_dir) if run_dir is not None else make_run_dir(config.train.run_root)
    master_run_path.mkdir(parents=True, exist_ok=True)

    previous_checkpoint: Path | None = None
    final_checkpoint = master_run_path / "policy.pt"
    for index, course in enumerate(config.curriculum.courses):
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
    resume_checkpoint: str | Path | None = None,
) -> Path:
    assert config.curriculum is not None
    course_index, course = select_curriculum_course(config, course_name=course_name)
    course_config = build_course_config(config, course)
    resume_checkpoint_path = _resolve_existing_path(resume_checkpoint, "resume checkpoint")
    checkpoint_path = None
    if resume_checkpoint_path is None:
        checkpoint_path = _resolve_previous_checkpoint(
            config=config,
            course_index=course_index,
            explicit_checkpoint=previous_checkpoint,
        )
    if resume_checkpoint_path is None and course_index > 0 and course.load_previous and checkpoint_path is None:
        previous_course = config.curriculum.courses[course_index - 1]
        raise FileNotFoundError(
            f"{course.name} requires a checkpoint from {previous_course.name}; "
            "train the previous course first or pass --previous-checkpoint"
        )
    if run_dir is not None:
        run_base = Path(run_dir)
    elif resume_checkpoint_path is not None:
        run_base = resume_checkpoint_path.parent.parent
    else:
        run_base = make_run_dir(config.train.run_root)
    course_run_dir = run_base / f"{course_index + 1:02d}-{_slugify(course.name)}"
    course_run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _train_single_course(
        course_config,
        run_dir=course_run_dir,
        checkpoint_path=checkpoint_path,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    _update_curriculum_state(config, course.name, course_run_dir / "best_policy.pt", course_run_dir, state_root=run_base)
    return checkpoint


def _train_single_course(
    config: ExperimentConfig,
    env: GridCoverageEnv | None = None,
    run_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    resume_checkpoint_path: str | Path | None = None,
) -> Path:
    config = _prepare_policy_phase_config(config)
    _configure_training_runtime(config)
    set_seed(config.ppo.seed)
    rollout_envs = [env] if env is not None else _make_rollout_envs(config)
    reference_env = rollout_envs[0]
    if config.ppo.use_coverage_messages and not reference_env.config.use_explicit_map_memory:
        raise ValueError("use_coverage_messages requires use_explicit_map_memory=true")
    _validate_cuap_config(config, reference_env)
    _validate_intent_relation_config(config, reference_env)
    if run_dir is not None:
        run_path = Path(run_dir)
    elif resume_checkpoint_path is not None:
        run_path = Path(resume_checkpoint_path).parent
    else:
        run_path = make_run_dir(config.train.run_root)
    run_path.mkdir(parents=True, exist_ok=True)
    _write_config_snapshot(config, run_path)
    writer = make_tensorboard_writer(run_path, config.train.tensorboard_dir) if config.train.use_tensorboard else None

    device = resolve_device(config.ppo.device)
    model = ActorCritic(
        reference_env.observation_dim,
        reference_env.action_dim,
        config.ppo.hidden_dim,
        state_shape=(reference_env.config.height, reference_env.config.width),
        state_channels=reference_env.state_channels,
        state_metadata_dim=reference_env.state_metadata_dim,
        use_graph_attention=config.ppo.use_graph_attention,
        gat_num_heads=config.ppo.gat_num_heads,
        gat_edge_dim=reference_env.neighbor_feature_dim if config.ppo.gat_use_edge_features else 0,
        gat_residual=config.ppo.gat_residual,
        gat_attention_dropout=config.ppo.gat_attention_dropout,
        node_message_dim=reference_env.node_message_dim if config.ppo.use_coverage_messages else 0,
        use_phase_critics=_uses_joint_phase_model(config),
        use_phase_actors=_uses_joint_phase_model(config),
        phase_metadata_index=reference_env.base_state_metadata_dim,
        use_gated_cuap=config.cuap.enabled and config.cuap.gated,
        cuap_beta=config.cuap.beta,
        cuap_gate_hidden_dim=config.cuap.gate_hidden_dim,
        cuap_gate_init_prob=config.cuap.gate_init_prob,
        cuap_gate_detach_actor_features=config.cuap.gate_detach_actor_features,
        use_intent_relation=config.ppo.use_intent_relation,
        intent_relation_beta_max=config.ppo.intent_relation_beta_max,
        intent_relation_detach=config.ppo.intent_relation_detach,
        intent_grid_size=reference_env.config.intent_grid_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate)
    resume_state: dict[str, object] = {}
    if resume_checkpoint_path is not None:
        resume_state = _load_training_checkpoint(model, optimizer, resume_checkpoint_path, device)
        _restore_rng_state(resume_state.get("rng_state"))
    elif checkpoint_path is not None:
        _load_checkpoint_weights(model, checkpoint_path)

    observation, state = _initialize_rollout_state(rollout_envs, resume_state)
    timestep = int(resume_state.get("timestep", 0))
    update_index = int(resume_state.get("update_index", 0))
    episode_index = int(resume_state.get("episode_index", 0))
    episode_rewards = np.asarray(
        resume_state.get("episode_rewards", np.zeros(len(rollout_envs), dtype=np.float32)),
        dtype=np.float32,
    )
    if episode_rewards.shape != (len(rollout_envs),):
        episode_rewards = np.zeros(len(rollout_envs), dtype=np.float32)
    episode_start_lengths = [
        int(item)
        for item in resume_state.get("episode_start_lengths", [worker.path_length for worker in rollout_envs])
    ]
    if len(episode_start_lengths) != len(rollout_envs):
        episode_start_lengths = [worker.path_length for worker in rollout_envs]
    metric_rows: list[dict[str, float | int]] = []
    best_score = _restore_best_score(resume_state.get("best_score"))
    best_checkpoint_path = run_path / "best_policy.pt"
    last_checkpoint_path = run_path / "last_policy.pt"
    eval_interval = max(config.train.eval_interval or config.train.log_interval, 1)
    checkpoint_interval = max(config.train.checkpoint_interval or config.train.log_interval, 1)
    training_complete_marker = run_path / "training_complete.json"
    if training_complete_marker.exists():
        training_complete_marker.unlink()

    env_executor = _make_rollout_executor(config, len(rollout_envs))
    try:
        while timestep < config.ppo.total_timesteps:
            batch, observation, state, rollout_metrics = _collect_rollout(
                config=config,
                envs=rollout_envs,
                model=model,
                observation=observation,
                state=state,
                max_steps=_rollout_env_steps(config, reference_env, len(rollout_envs), timestep),
                episode_index=episode_index,
                episode_rewards=episode_rewards,
                episode_start_lengths=episode_start_lengths,
                env_executor=env_executor,
            )
            episode_index = int(rollout_metrics.pop("episode_index"))
            episode_rewards = np.asarray(rollout_metrics.pop("episode_rewards"), dtype=np.float32)
            episode_start_lengths = [int(item) for item in rollout_metrics.pop("episode_start_lengths")]
            new_metric_rows = rollout_metrics.pop("metric_rows")
            metric_rows.extend(new_metric_rows)
            batch_transition_count = int(batch.actions.numel())
            timestep += batch_transition_count
            update_index += 1

            _update_policy(config, model, optimizer, batch)
            del batch

            if metric_rows:
                append_metrics(run_path / "metrics.csv", metric_rows)
                if writer is not None:
                    write_tensorboard_rows(writer, "train", metric_rows)
                metric_rows.clear()

            if update_index % checkpoint_interval == 0:
                torch.save(
                    _checkpoint_payload(
                        config,
                        model,
                        optimizer=optimizer,
                        training_state=_training_state_payload(
                            rollout_envs,
                            observation,
                            state,
                            episode_rewards,
                            episode_start_lengths,
                            timestep=timestep,
                            update_index=update_index,
                            episode_index=episode_index,
                            best_score=best_score,
                        ),
                    ),
                    last_checkpoint_path,
                )

            if update_index % eval_interval == 0:
                eval_row = _evaluate_model(config, model, update_index)
                append_metrics(run_path / "eval_metrics.csv", [eval_row])
                if writer is not None:
                    write_tensorboard_rows(writer, "eval", [eval_row])
                    writer.add_scalar("train/timesteps", timestep, update_index)
                    writer.flush()
                score = _metric_score(eval_row)
                if best_score is None or score > best_score:
                    best_score = score
                    torch.save(
                        _checkpoint_payload(
                            config,
                            model,
                            optimizer=optimizer,
                            training_state=_training_state_payload(
                                rollout_envs,
                                observation,
                                state,
                                episode_rewards,
                                episode_start_lengths,
                                timestep=timestep,
                                update_index=update_index,
                                episode_index=episode_index,
                                best_score=best_score,
                            ),
                        ),
                        best_checkpoint_path,
                    )

        torch.save(
            _checkpoint_payload(
                config,
                model,
                optimizer=optimizer,
                training_state=_training_state_payload(
                    rollout_envs,
                    observation,
                    state,
                    episode_rewards,
                    episode_start_lengths,
                    timestep=timestep,
                    update_index=update_index,
                    episode_index=episode_index,
                    best_score=best_score,
                ),
            ),
            last_checkpoint_path,
        )
        if not best_checkpoint_path.exists():
            torch.save(
                _checkpoint_payload(
                    config,
                    model,
                    optimizer=optimizer,
                    training_state=_training_state_payload(
                        rollout_envs,
                        observation,
                        state,
                        episode_rewards,
                        episode_start_lengths,
                        timestep=timestep,
                        update_index=update_index,
                        episode_index=episode_index,
                        best_score=best_score,
                    ),
                ),
                best_checkpoint_path,
            )
        shutil.copyfile(best_checkpoint_path, run_path / "policy.pt")

        _finalize_course_outputs(config, run_path, best_checkpoint_path)
        training_complete_marker.write_text(
            json.dumps(
                {
                    "completed": True,
                    "timestep": timestep,
                    "update_index": update_index,
                    "checkpoint": str(run_path / "policy.pt"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return run_path / "policy.pt"
    finally:
        if env_executor is not None:
            env_executor.shutdown(wait=True)
        if writer is not None:
            writer.close()


def _finalize_course_outputs(config: ExperimentConfig, run_path: Path, checkpoint_path: Path) -> None:
    from .evaluation import evaluate_policy
    from .rendering import render_trajectory

    summary = evaluate_policy(config, checkpoint_path, output_path=run_path / "trajectory.json")
    render_trajectory(config, summary["trajectory"], run_path / "trajectory.png")


def _write_config_snapshot(config: ExperimentConfig, run_path: Path) -> None:
    run_path.joinpath("course_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def _curriculum_state_path(config: ExperimentConfig, state_root: str | Path | None = None) -> Path:
    phase = _policy_phase(config)
    filename = "_curriculum_state.json" if phase == "coverage" else f"_curriculum_state_{phase}.json"
    return Path(state_root) / filename if state_root is not None else Path(config.train.run_root) / filename


def _resolve_previous_checkpoint(
    config: ExperimentConfig,
    course_index: int,
    explicit_checkpoint: str | Path | None = None,
) -> Path | None:
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
    env = GridCoverageEnv(config.env)
    observation = agent_observations(env.reset(seed=config.env.seed))
    state = env.global_state()
    device = _model_device(model)
    total_reward = 0.0
    done = False
    info = {}
    cuap_diagnostics = _empty_cuap_diagnostics()
    cir_diagnostics = _empty_cir_diagnostics()
    while not done:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        neighbor_mask = torch.as_tensor(env.neighbor_mask(), dtype=torch.bool, device=device)
        edge_features = _edge_features_tensor(env, model, device)
        node_messages = _node_messages_tensor(env, model, device)
        action_mask = _action_mask_tensor(config, env, device)
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
                deterministic=True,
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
    row = {
        "episode": update_index,
        "reward": total_reward,
        "coverage_ratio": info.get("coverage_ratio", env.coverage_ratio()),
        "path_length": env.path_length,
        "completed": int(info.get("completed", False)),
        "steps": info.get("step_count", env.step_count),
    }
    row.update(_summarize_cuap_diagnostics(cuap_diagnostics))
    row.update(_summarize_cir_diagnostics(cir_diagnostics))
    return row


def _load_checkpoint_weights(model: ActorCritic, checkpoint_path: str | Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_compatible_state_dict(payload["model_state_dict"])


def _load_training_checkpoint(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_compatible_state_dict(payload["model_state_dict"])
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state(optimizer, device)
    training_state = payload.get("training_state", {})
    return training_state if isinstance(training_state, dict) else {}


def _initialize_rollout_state(
    rollout_envs: list[GridCoverageEnv],
    resume_state: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    env_states = resume_state.get("env_states")
    observation = resume_state.get("observation")
    state = resume_state.get("state")
    if env_states is not None or observation is not None or state is not None:
        if not isinstance(env_states, list) or len(env_states) != len(rollout_envs):
            raise ValueError("resume checkpoint env state count does not match configured ppo.num_envs")
        for worker, worker_state in zip(rollout_envs, env_states):
            if not isinstance(worker_state, dict):
                raise ValueError("resume checkpoint contains an invalid environment state")
            worker.load_state_dict(worker_state)
        if observation is None or state is None:
            raise ValueError("resume checkpoint is missing rollout observation/state arrays")
        return np.asarray(observation, dtype=np.float32), np.asarray(state, dtype=np.float32)

    return (
        np.asarray([agent_observations(worker.reset(seed=worker.config.seed)) for worker in rollout_envs], dtype=np.float32),
        np.asarray([worker.global_state() for worker in rollout_envs], dtype=np.float32),
    )


def _training_state_payload(
    envs: list[GridCoverageEnv],
    observation: np.ndarray,
    state: np.ndarray,
    episode_rewards: np.ndarray,
    episode_start_lengths: list[int],
    *,
    timestep: int,
    update_index: int,
    episode_index: int,
    best_score: tuple[float, int, float, float] | None,
) -> dict[str, object]:
    return {
        "timestep": int(timestep),
        "update_index": int(update_index),
        "episode_index": int(episode_index),
        "episode_rewards": np.asarray(episode_rewards, dtype=np.float32).tolist(),
        "episode_start_lengths": [int(item) for item in episode_start_lengths],
        "observation": np.asarray(observation, dtype=np.float32),
        "state": np.asarray(state, dtype=np.float32),
        "env_states": [worker.state_dict() for worker in envs],
        "rng_state": _rng_state_payload(),
        "best_score": None if best_score is None else list(best_score),
    }


def _rng_state_payload() -> dict[str, object]:
    payload: dict[str, object] = {
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
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_state = state.get("torch")
    cuda_state = state.get("torch_cuda")
    if python_state is not None:
        random.setstate(python_state)  # type: ignore[arg-type]
    if numpy_state is not None:
        np.random.set_state(numpy_state)  # type: ignore[arg-type]
    if torch_state is not None and torch.is_tensor(torch_state):
        torch.set_rng_state(torch_state.cpu())
    if cuda_state is not None and torch.cuda.is_available():
        if isinstance(cuda_state, list):
            cuda_state = [item.cpu() if torch.is_tensor(item) else item for item in cuda_state]
        torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]


def _restore_best_score(score: object) -> tuple[float, int, float, float] | None:
    if not isinstance(score, (list, tuple)) or len(score) != 4:
        return None
    return float(score[0]), int(score[1]), float(score[2]), float(score[3])


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _resolve_existing_path(path: str | Path | None, label: str) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _metric_score(row: dict[str, float | int]) -> tuple[float, int, float, float]:
    coverage = float(row["coverage_ratio"])
    completed = int(row["completed"])
    path_length = float(row["path_length"])
    reward = float(row["reward"])
    return coverage, completed, -path_length, reward


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


def _prepare_policy_phase_config(config: ExperimentConfig) -> ExperimentConfig:
    phase = _policy_phase(config)
    prepared = copy.deepcopy(config)
    prepared.ppo.policy_phase = phase
    if phase == "coverage":
        prepared.env.initial_return_mode = False
        prepared.env.require_return_to_depot = False
    elif phase == "return":
        prepared.env.use_depot = True
        prepared.env.require_return_to_depot = True
        prepared.env.initial_return_mode = True
    elif phase == "joint":
        prepared.env.initial_return_mode = False
    else:
        raise ValueError(f"unknown policy_phase: {phase}")
    return prepared


def _policy_phase(config: ExperimentConfig) -> str:
    return str(config.ppo.policy_phase).strip().lower()


def _uses_joint_phase_model(config: ExperimentConfig) -> bool:
    return _policy_phase(config) == "joint" and config.env.use_depot and config.env.require_return_to_depot


def _configure_training_runtime(config: ExperimentConfig) -> None:
    threads = configure_runtime(config.train.cpu_threads)
    torch.set_num_threads(threads)
    try:
        interop_threads = max(int(getattr(config.train, "interop_threads", 1)), 1)
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass
    _configure_process_priority(config.train.process_priority)
    _configure_gpu_budget(config.train.gpu_memory_fraction)
    if hasattr(torch, "set_float32_matmul_precision"):
        precision = config.train.float32_matmul_precision
        if precision in {"highest", "high", "medium"}:
            torch.set_float32_matmul_precision(precision)


def _configure_gpu_budget(memory_fraction: float | None) -> None:
    if not torch.cuda.is_available() or memory_fraction is None:
        return
    fraction = min(max(float(memory_fraction), 0.05), 0.95)
    try:
        torch.cuda.set_per_process_memory_fraction(fraction)
    except RuntimeError:
        pass
    torch.backends.cudnn.benchmark = False


def _configure_process_priority(priority: str) -> None:
    normalized = str(priority).strip().lower()
    if normalized in {"", "normal"}:
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
    nice_value = 15 if normalized == "idle" else 5 if normalized == "below_normal" else None
    if nice_value is None:
        return
    try:
        os.nice(nice_value)
    except OSError:
        pass


def _validate_cuap_config(config: ExperimentConfig, env: GridCoverageEnv) -> None:
    if not config.cuap.enabled:
        return
    if not config.ppo.use_action_mask:
        raise ValueError("CUAP requires ppo.use_action_mask=true so the action mask is applied after the prior")
    if not env.config.use_explicit_map_memory:
        raise ValueError("CUAP requires env.use_explicit_map_memory=true to avoid global coverage truth leakage")


def _validate_intent_relation_config(config: ExperimentConfig, env: GridCoverageEnv) -> None:
    if not config.ppo.use_intent_relation:
        return
    if not config.ppo.use_graph_attention:
        raise ValueError("use_intent_relation requires ppo.use_graph_attention=true")
    if not config.ppo.use_coverage_messages:
        raise ValueError("use_intent_relation requires ppo.use_coverage_messages=true")
    if not env.config.use_explicit_map_memory:
        raise ValueError("use_intent_relation requires env.use_explicit_map_memory=true")
    if env.config.intent_grid_size <= 0:
        raise ValueError("use_intent_relation requires env.intent_grid_size > 0")


def _make_rollout_envs(config: ExperimentConfig) -> list[GridCoverageEnv]:
    num_envs = max(int(config.ppo.num_envs), 1)
    envs: list[GridCoverageEnv] = []
    for index in range(num_envs):
        env_config = copy.deepcopy(config.env)
        if index > 0:
            offset = 1009 * index
            env_config.seed += offset
            env_config.random_obstacle_seed += offset
            env_config.random_obstacle_seeds = [seed + offset for seed in env_config.random_obstacle_seeds]
        envs.append(GridCoverageEnv(env_config))
    return envs


def _make_rollout_executor(config: ExperimentConfig, num_envs: int) -> ThreadPoolExecutor | None:
    worker_count = min(max(int(getattr(config.train, "rollout_workers", 1)), 1), max(num_envs, 1))
    if worker_count <= 1 or num_envs <= 1:
        return None
    return ThreadPoolExecutor(max_workers=worker_count)


def _rollout_env_steps(
    config: ExperimentConfig,
    env: GridCoverageEnv,
    num_envs: int,
    timestep: int,
) -> int:
    per_env_steps = max(1, config.ppo.rollout_steps // max(env.num_agents, 1))
    remaining = max(config.ppo.total_timesteps - timestep, 1)
    remaining_steps = max(1, math.ceil(remaining / max(num_envs * env.num_agents, 1)))
    return min(per_env_steps, remaining_steps)


def _collect_rollout(
    config: ExperimentConfig,
    envs: list[GridCoverageEnv],
    model: ActorCritic,
    observation: np.ndarray,
    state: np.ndarray,
    max_steps: int,
    episode_index: int,
    episode_rewards: np.ndarray,
    episode_start_lengths: list[int],
    env_executor: ThreadPoolExecutor | None = None,
) -> tuple[RolloutBatch, np.ndarray, np.ndarray, dict[str, object]]:
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
    action_prior_logits: list[np.ndarray] = []
    cuap_priors: list[np.ndarray] = []
    cuap_confidences: list[np.ndarray] = []
    cuap_phase_masks: list[np.ndarray] = []
    metric_rows: list[dict[str, float | int]] = []
    device = _model_device(model)
    num_envs = len(envs)
    num_agents = envs[0].num_agents

    for _ in range(max_steps):
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        neighbor_mask = np.asarray([worker.neighbor_mask() for worker in envs], dtype=bool)
        neighbor_mask_tensor = torch.as_tensor(neighbor_mask, dtype=torch.bool, device=device)
        action_mask = np.asarray([worker.action_masks() for worker in envs], dtype=bool) if config.ppo.use_action_mask else None
        action_mask_tensor = (
            torch.as_tensor(action_mask, dtype=torch.bool, device=device) if action_mask is not None else None
        )
        action_prior_array = _action_prior_array(config, envs)
        action_prior_tensor = (
            torch.as_tensor(action_prior_array, dtype=torch.float32, device=device) if action_prior_array is not None else None
        )
        cuap_step_arrays = _cuap_step_arrays(config, envs)
        cuap_prior_tensor = (
            torch.as_tensor(cuap_step_arrays["cuap_prior"], dtype=torch.float32, device=device)
            if cuap_step_arrays
            else None
        )
        cuap_confidence_tensor = (
            torch.as_tensor(cuap_step_arrays["cuap_confidence"], dtype=torch.float32, device=device)
            if cuap_step_arrays
            else None
        )
        cuap_phase_mask_tensor = (
            torch.as_tensor(cuap_step_arrays["cuap_phase_mask"], dtype=torch.float32, device=device)
            if cuap_step_arrays
            else None
        )
        edge_feature_array = (
            np.asarray([worker.neighbor_features() for worker in envs], dtype=np.float32)
            if _uses_edge_features(model)
            else None
        )
        edge_feature_tensor = (
            torch.as_tensor(edge_feature_array, dtype=torch.float32, device=device) if edge_feature_array is not None else None
        )
        node_message_array = (
            np.asarray([worker.node_messages() for worker in envs], dtype=np.float32)
            if model.node_message_dim > 0
            else None
        )
        node_message_tensor = (
            torch.as_tensor(node_message_array, dtype=torch.float32, device=device) if node_message_array is not None else None
        )
        with torch.no_grad():
            action_tensor, log_prob_tensor, value_tensor = model.act_batch(
                obs_tensor,
                state_tensor,
                neighbor_mask=neighbor_mask_tensor,
                edge_features=edge_feature_tensor,
                node_messages=node_message_tensor,
                action_mask=action_mask_tensor,
                action_prior_logits=action_prior_tensor,
                cuap_prior=cuap_prior_tensor,
                cuap_confidence=cuap_confidence_tensor,
                cuap_phase_mask=cuap_phase_mask_tensor,
            )

        observations.append(observation)
        states.append(state)
        action_array = action_tensor.cpu().numpy()
        actions.append(action_array)
        log_probs.append(log_prob_tensor.cpu().numpy())
        values.append(value_tensor.cpu().numpy())
        neighbor_masks.append(neighbor_mask)
        if action_mask is not None:
            action_masks.append(action_mask)
        if action_prior_array is not None:
            action_prior_logits.append(action_prior_array)
        if cuap_step_arrays:
            cuap_priors.append(cuap_step_arrays["cuap_prior"])
            cuap_confidences.append(cuap_step_arrays["cuap_confidence"])
            cuap_phase_masks.append(cuap_step_arrays["cuap_phase_mask"])
        if edge_feature_array is not None:
            edge_features.append(edge_feature_array)
        if node_message_array is not None:
            node_messages.append(node_message_array)

        step_rewards = np.zeros((num_envs, num_agents), dtype=np.float32)
        step_dones = np.zeros((num_envs, num_agents), dtype=np.float32)
        next_observation = np.empty_like(observation)
        next_state = np.empty_like(state)
        step_results = _step_rollout_envs(envs, action_array, env_executor)
        for env_index, (worker, result) in enumerate(zip(envs, step_results)):
            reward_array = agent_rewards(worker.num_agents, result.reward)
            step_rewards[env_index] = reward_array
            step_dones[env_index] = float(result.done)
            episode_rewards[env_index] += float(np.mean(reward_array))
            if result.done:
                metric_rows.append(
                    {
                        "episode": episode_index,
                        "env_index": env_index,
                        "reward": float(episode_rewards[env_index]),
                        "coverage_ratio": result.info["coverage_ratio"],
                        "path_length": result.info["path_length"] - episode_start_lengths[env_index],
                        "completed": int(result.info["completed"]),
                        "coverage_completed": int(result.info.get("coverage_completed", result.info["completed"])),
                        "returned_to_depot": int(result.info.get("all_at_depot", False)),
                        "steps": result.info["step_count"],
                    }
                )
                episode_index += 1
                episode_rewards[env_index] = 0.0
                next_observation[env_index] = agent_observations(worker.reset())
                next_state[env_index] = worker.global_state()
                episode_start_lengths[env_index] = worker.path_length
            else:
                next_observation[env_index] = agent_observations(result.observation)
                next_state[env_index] = result.state
        rewards.append(step_rewards)
        dones.append(step_dones)
        observation = next_observation
        state = next_state

    with torch.no_grad():
        next_value = model.value(torch.as_tensor(state, dtype=torch.float32, device=device))
        next_values = next_value.unsqueeze(-1).expand(num_envs, num_agents).cpu().numpy()
        if dones:
            last_done = np.asarray(dones[-1], dtype=bool)
            next_values = np.where(last_done, np.zeros_like(next_values), next_values)

    returns, advantages = _gae_array(
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        next_values=np.asarray(next_values, dtype=np.float32),
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
    )

    observation_array = _flatten_time_env(np.asarray(observations, dtype=np.float32))
    state_array = _flatten_time_env(np.asarray(states, dtype=np.float32))
    action_array = _flatten_time_env(np.asarray(actions, dtype=np.int64))
    log_prob_array = _flatten_time_env(np.asarray(log_probs, dtype=np.float32))
    value_array = _flatten_time_env(np.asarray(values, dtype=np.float32))
    return_array = _flatten_time_env(returns)
    advantage_array = _flatten_time_env(advantages)
    batch = RolloutBatch(
        observations=torch.as_tensor(observation_array, dtype=torch.float32, device=device),
        states=torch.as_tensor(state_array, dtype=torch.float32, device=device),
        actions=torch.as_tensor(action_array, dtype=torch.long, device=device),
        log_probs=torch.as_tensor(log_prob_array, dtype=torch.float32, device=device),
        returns=torch.as_tensor(return_array, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantage_array, dtype=torch.float32, device=device),
        values=torch.as_tensor(value_array, dtype=torch.float32, device=device),
        neighbor_masks=torch.as_tensor(_flatten_time_env(np.asarray(neighbor_masks, dtype=bool)), dtype=torch.bool, device=device),
        action_masks=(
            torch.as_tensor(_flatten_time_env(np.asarray(action_masks, dtype=bool)), dtype=torch.bool, device=device)
            if action_masks
            else None
        ),
        edge_features=(
            torch.as_tensor(_flatten_time_env(np.asarray(edge_features, dtype=np.float32)), dtype=torch.float32, device=device)
            if edge_features
            else None
        ),
        node_messages=(
            torch.as_tensor(_flatten_time_env(np.asarray(node_messages, dtype=np.float32)), dtype=torch.float32, device=device)
            if node_messages
            else None
        ),
        action_prior_logits=(
            torch.as_tensor(_flatten_time_env(np.asarray(action_prior_logits, dtype=np.float32)), dtype=torch.float32, device=device)
            if action_prior_logits
            else None
        ),
        cuap_priors=(
            torch.as_tensor(_flatten_time_env(np.asarray(cuap_priors, dtype=np.float32)), dtype=torch.float32, device=device)
            if cuap_priors
            else None
        ),
        cuap_confidences=(
            torch.as_tensor(_flatten_time_env(np.asarray(cuap_confidences, dtype=np.float32)), dtype=torch.float32, device=device)
            if cuap_confidences
            else None
        ),
        cuap_phase_masks=(
            torch.as_tensor(_flatten_time_env(np.asarray(cuap_phase_masks, dtype=np.float32)), dtype=torch.float32, device=device)
            if cuap_phase_masks
            else None
        ),
    )
    metrics = {
        "episode_index": episode_index,
        "episode_rewards": episode_rewards.tolist(),
        "episode_start_lengths": list(episode_start_lengths),
        "metric_rows": metric_rows,
    }
    return batch, observation, state, metrics


def _step_rollout_envs(
    envs: list[GridCoverageEnv],
    action_array: np.ndarray,
    executor: ThreadPoolExecutor | None,
) -> list[object]:
    if executor is None:
        return [worker.step(action_array[index].tolist()) for index, worker in enumerate(envs)]
    futures = [executor.submit(worker.step, action_array[index].tolist()) for index, worker in enumerate(envs)]
    return [future.result() for future in futures]


def _flatten_time_env(array: np.ndarray) -> np.ndarray:
    if array.ndim < 2:
        return array
    return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])


def _model_device(model: ActorCritic) -> torch.device:
    return next(model.parameters()).device


def _gae(
    rewards: list[float],
    dones: list[float],
    values: list[float],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
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
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1:], dtype=np.float32)
    for index in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[index]
        next_value = next_values if index == rewards.shape[0] - 1 else values[index + 1]
        delta = rewards[index] + gamma * next_value * next_non_terminal - values[index]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[index] = last_advantage
    returns = advantages + values
    return returns, advantages


def _update_policy(config: ExperimentConfig, model: ActorCritic, optimizer: torch.optim.Optimizer, batch: RolloutBatch) -> None:
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
            action_mask = batch.action_masks[mb_tensor] if batch.action_masks is not None else None
            action_prior = batch.action_prior_logits[mb_tensor] if batch.action_prior_logits is not None else None
            cuap_prior = batch.cuap_priors[mb_tensor] if batch.cuap_priors is not None else None
            cuap_confidence = batch.cuap_confidences[mb_tensor] if batch.cuap_confidences is not None else None
            cuap_phase_mask = batch.cuap_phase_masks[mb_tensor] if batch.cuap_phase_masks is not None else None
            log_probs, entropy, values = model.evaluate_actions(
                batch.observations[mb_tensor],
                batch.states[mb_tensor],
                batch.actions[mb_tensor],
                neighbor_mask=neighbor_mask,
                edge_features=edge_features,
                node_messages=node_messages,
                action_mask=action_mask,
                action_prior_logits=action_prior,
                cuap_prior=cuap_prior,
                cuap_confidence=cuap_confidence,
                cuap_phase_mask=cuap_phase_mask,
            )
            ratio = torch.exp(log_probs - batch.log_probs[mb_tensor])
            clipped = torch.clamp(ratio, 1.0 - config.ppo.clip_ratio, 1.0 + config.ppo.clip_ratio) * advantages[mb_tensor]
            policy_loss = -torch.min(ratio * advantages[mb_tensor], clipped).mean()
            value_loss = nn.functional.mse_loss(values, batch.returns[mb_tensor])
            entropy_loss = entropy.mean()
            loss = policy_loss + config.ppo.value_coef * value_loss - config.ppo.entropy_coef * entropy_loss
            if config.cuap.enabled and config.cuap.gated and config.cuap.gate_regularization > 0:
                if model.latest_applied_gate is not None:
                    loss = loss + float(config.cuap.gate_regularization) * model.latest_applied_gate.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.ppo.max_grad_norm)
            optimizer.step()


def _checkpoint_payload(
    config: ExperimentConfig,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer | None = None,
    training_state: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = checkpoint_model_metadata(config, model)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if training_state is not None:
        payload["training_state"] = dict(training_state)
    return payload


def _edge_features_tensor(env: GridCoverageEnv, model: ActorCritic, device: torch.device) -> torch.Tensor | None:
    if not _uses_edge_features(model):
        return None
    return torch.as_tensor(env.neighbor_features(), dtype=torch.float32, device=device)


def _uses_edge_features(model: ActorCritic) -> bool:
    return model.use_graph_attention and model.gat_edge_dim > 0


def _node_messages_tensor(env: GridCoverageEnv, model: ActorCritic, device: torch.device) -> torch.Tensor | None:
    if model.node_message_dim <= 0:
        return None
    return torch.as_tensor(env.node_messages(), dtype=torch.float32, device=device)


def _action_mask_tensor(config: ExperimentConfig, env: GridCoverageEnv, device: torch.device) -> torch.Tensor | None:
    if not config.ppo.use_action_mask:
        return None
    return torch.as_tensor(env.action_masks(), dtype=torch.bool, device=device)


def _action_prior_tensor(config: ExperimentConfig, env: GridCoverageEnv, device: torch.device) -> torch.Tensor | None:
    if config.cuap.gated:
        return None
    prior = scaled_cuap_prior(env, config.cuap, phase=_env_phase(env))
    if prior is None:
        return None
    return torch.as_tensor(prior, dtype=torch.float32, device=device)


def _action_prior_array(config: ExperimentConfig, envs: list[GridCoverageEnv]) -> np.ndarray | None:
    if not config.cuap.enabled or config.cuap.gated:
        return None
    return np.asarray(
        [scaled_cuap_prior(worker, config.cuap, phase=_env_phase(worker)) for worker in envs],
        dtype=np.float32,
    )


def _cuap_step_tensors(
    config: ExperimentConfig,
    env: GridCoverageEnv,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not config.cuap.enabled or not config.cuap.gated:
        return {}
    inputs = build_cuap_step_inputs(env, config.cuap, phase=_env_phase(env))
    return {
        "cuap_prior": torch.as_tensor(inputs.prior, dtype=torch.float32, device=device),
        "cuap_confidence": torch.as_tensor(inputs.confidence, dtype=torch.float32, device=device),
        "cuap_phase_mask": torch.as_tensor(inputs.phase_mask, dtype=torch.float32, device=device),
    }


def _cuap_step_arrays(config: ExperimentConfig, envs: list[GridCoverageEnv]) -> dict[str, np.ndarray] | None:
    if not config.cuap.enabled or not config.cuap.gated:
        return None
    inputs = [build_cuap_step_inputs(worker, config.cuap, phase=_env_phase(worker)) for worker in envs]
    return {
        "cuap_prior": np.asarray([item.prior for item in inputs], dtype=np.float32),
        "cuap_confidence": np.asarray([item.confidence for item in inputs], dtype=np.float32),
        "cuap_phase_mask": np.asarray([item.phase_mask for item in inputs], dtype=np.float32),
    }


def _env_phase(env: GridCoverageEnv) -> str:
    return "return" if env.return_mode else "coverage"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "course"
