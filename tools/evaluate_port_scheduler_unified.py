from __future__ import annotations

import csv
import hashlib
import json
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
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import torch

from check_port_inspection_env import build_env
from mathbased_mcpp.port_inspection.mappo import HeterogeneousMappo
from train_port_scheduler_rl import _agent_types, _obs_matrix


OBSERVED_GREEDY_POLICIES = {"greedy_observed_score"}
MAPPO_POLICIES = {"mappo_argmax", "mappo_sample"}
POLICIES = tuple(sorted(OBSERVED_GREEDY_POLICIES | MAPPO_POLICIES))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MAPPO and greedy policies through the same agent observation API.")
    parser.add_argument("--config", default="configs/port_yangshan_task_initial_v1.toml")
    parser.add_argument("--checkpoint", default="data/ports/yangshan_task_initial_v1/scheduler_rl/scheduler_mappo.pt")
    parser.add_argument("--output-dir", default="data/ports/yangshan_task_initial_v1/unified_eval")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260622, 20260623, 20260624, 20260625, 20260626])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    args = parser.parse_args()

    config = _load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cache: HeterogeneousMappo | None = None
    rows: list[dict[str, Any]] = []
    for policy in args.policies:
        for seed in args.seeds:
            env = build_env(config)
            model = None
            if policy in MAPPO_POLICIES:
                if model_cache is None:
                    model_cache = _load_mappo(Path(args.checkpoint), config)
                model = model_cache
            summary, trace = run_unified_rollout(env, seed=seed, policy=policy, model=model)
            rows.append(summary)
            trace_path = output_dir / f"{policy}_seed{seed}_trace.csv"
            _write_trace(trace_path, trace)

    aggregate = _aggregate(rows)
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seeds": args.seeds,
        "environment_control": _environment_control(rows),
        "aggregate": aggregate,
        "runs": rows,
    }
    summary_path = output_dir / "unified_comparison_summary.json"
    runs_path = output_dir / "unified_comparison_runs.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_trace(runs_path, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary={summary_path}")
    print(f"runs={runs_path}")


def run_unified_rollout(env, seed: int, policy: str, model: HeterogeneousMappo | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    reset = env.reset_model(seed=seed)
    obs_dict = reset.obs_dict
    available_actions = reset.available_actions
    info = reset.info
    initial_environment = _environment_snapshot(env)
    agent_types = _agent_types(env)
    total_reward = 0.0
    trace: list[dict[str, Any]] = []
    done = False

    while not done:
        action_by_id = _choose_actions(env, policy, obs_dict, available_actions, info, agent_types, model)
        step = env.step_model(action_by_id)
        total_reward += sum(step.rewards.values()) / max(len(step.rewards), 1)
        info = step.info
        trace.append(
            {
                "step": env.current_step,
                "policy": policy,
                "seed": seed,
                "actions": json.dumps(action_by_id, ensure_ascii=False),
                "reward": sum(step.rewards.values()) / max(len(step.rewards), 1),
                "completed_tasks": len(info["completed_tasks"]),
                "late_tasks": len(info["late_tasks"]),
                "risk_exposure_sum": float(info["risk_exposure_sum"]),
                "total_path_length": int(info["total_path_length"]),
                "total_energy": float(info["total_energy"]),
                "total_conflicts": int(info["metrics"]["total_conflicts"]),
                "total_invalid_actions": int(info["metrics"]["total_invalid_actions"]),
                "total_replenishments": int(info["metrics"]["total_replenishments"]),
                "total_returns": int(info["metrics"]["total_returns"]),
                "platform_resources": json.dumps(_platform_resources(env), ensure_ascii=False),
                "resource_hash": _hash_json(_platform_resources(env)),
                "accepted_actions": json.dumps(info["accepted_actions"], ensure_ascii=False),
                "conflicts": json.dumps(info["conflicts"], ensure_ascii=False),
            }
        )
        obs_dict = step.obs_dict
        available_actions = step.available_actions
        done = step.terminated or step.truncated

    final_info = env.info()
    summary = {
        "policy": policy,
        "seed": seed,
        "initial_environment_hash": str(initial_environment["environment_hash"]),
        "initial_task_truth_hash": str(initial_environment["task_truth_hash"]),
        "initial_platform_resources": json.dumps(initial_environment["platform_resources"], ensure_ascii=False),
        "steps": env.current_step,
        "episode_reward": total_reward,
        "completed_tasks": len(env.completed_tasks),
        "task_count": env.num_tasks,
        "completion_rate": len(env.completed_tasks) / max(env.num_tasks, 1),
        "late_tasks": len(final_info["late_tasks"]),
        "risk_exposure_sum": float(final_info["risk_exposure_sum"]),
        "total_path_length": int(final_info["total_path_length"]),
        "total_energy": float(final_info["total_energy"]),
        "total_conflicts": int(final_info["metrics"]["total_conflicts"]),
        "total_invalid_actions": int(final_info["metrics"]["total_invalid_actions"]),
        "total_replenishments": int(final_info["metrics"]["total_replenishments"]),
        "total_returns": int(final_info["metrics"]["total_returns"]),
    }
    return summary, trace


def _choose_actions(
    env,
    policy: str,
    obs_dict: dict[str, np.ndarray],
    available_actions: dict[str, np.ndarray],
    info: dict[str, Any],
    agent_types: np.ndarray,
    model: HeterogeneousMappo | None,
) -> dict[str, int]:
    if policy == "mappo_argmax":
        if model is None:
            raise ValueError("mappo_argmax requires a loaded model")
        return _mappo_argmax_actions(env, model, obs_dict, available_actions, agent_types)
    if policy == "mappo_sample":
        if model is None:
            raise ValueError("mappo_sample requires a loaded model")
        return _mappo_sample_actions(env, model, obs_dict, available_actions, agent_types)
    if policy == "greedy_observed_score":
        return _greedy_observed_actions(env, available_actions, info)
    raise ValueError(f"unknown policy: {policy}")


def _mappo_argmax_actions(
    env,
    model: HeterogeneousMappo,
    obs_dict: dict[str, np.ndarray],
    available_actions: dict[str, np.ndarray],
    agent_types: np.ndarray,
) -> dict[str, int]:
    obs = _obs_matrix(env, obs_dict)
    mask = np.asarray([available_actions[platform.platform_id].astype(bool) for platform in env.platforms], dtype=bool)
    with torch.no_grad():
        logits = model.logits(
            torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(agent_types, dtype=torch.long).unsqueeze(0),
        ).squeeze(0)
        masked_logits = logits.masked_fill(~torch.as_tensor(mask, dtype=torch.bool), torch.finfo(logits.dtype).min)
        actions = masked_logits.argmax(dim=-1).cpu().numpy().astype(int)
    return {platform.platform_id: int(actions[index]) for index, platform in enumerate(env.platforms)}


def _mappo_sample_actions(
    env,
    model: HeterogeneousMappo,
    obs_dict: dict[str, np.ndarray],
    available_actions: dict[str, np.ndarray],
    agent_types: np.ndarray,
) -> dict[str, int]:
    obs = _obs_matrix(env, obs_dict)
    mask = np.asarray([available_actions[platform.platform_id].astype(bool) for platform in env.platforms], dtype=bool)
    with torch.no_grad():
        logits = model.logits(
            torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(agent_types, dtype=torch.long).unsqueeze(0),
        ).squeeze(0)
        masked_logits = logits.masked_fill(~torch.as_tensor(mask, dtype=torch.bool), torch.finfo(logits.dtype).min)
        actions = torch.distributions.Categorical(logits=masked_logits).sample().cpu().numpy().astype(int)
    return {platform.platform_id: int(actions[index]) for index, platform in enumerate(env.platforms)}


def _greedy_observed_actions(env, available_actions: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, int]:
    actions: dict[str, int] = {}
    candidate_details = list(info.get("candidate_details", []))
    for platform_index, platform in enumerate(env.platforms):
        mask = np.asarray(available_actions[platform.platform_id], dtype=bool)
        if mask[env.continue_action]:
            actions[platform.platform_id] = env.continue_action
            continue

        scored: list[tuple[float, int]] = []
        platform_candidates = candidate_details[platform_index] if platform_index < len(candidate_details) else []
        for detail in platform_candidates:
            action = int(detail.get("relative_position", -1))
            if 0 <= action < env.candidate_k and mask[action]:
                scored.append((_observed_candidate_score(detail), action))
        if scored:
            _, action = max(scored, key=lambda item: (item[0], -item[1]))
            actions[platform.platform_id] = action
        elif mask[env.return_action]:
            actions[platform.platform_id] = env.return_action
        else:
            actions[platform.platform_id] = env.wait_action
    return actions


def _observed_candidate_score(detail: dict[str, Any]) -> float:
    risk = float(detail.get("risk", 0.0))
    urgency = float(detail.get("urgency", 0.0))
    confidence = float(detail.get("confidence", 0.0))
    review_wait = float(detail.get("review_waiting_time", 0.0))
    arrival = float(detail.get("estimated_arrival_time", 0.0))
    stage_bonus = 2.0 if str(detail.get("task_stage", "")) == "review" else 0.0
    return 14.0 * risk + 6.0 * urgency + 2.0 * confidence + 0.5 * review_wait - 0.07 * arrival + stage_bonus


def _load_mappo(checkpoint_path: Path, config: dict[str, Any]) -> HeterogeneousMappo:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden_dim = int(dict(config.get("scheduler_rl", {})).get("hidden_dim", 128))
    model = HeterogeneousMappo(
        observation_dim=int(checkpoint["observation_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy"]), []).append(row)
    metrics = [
        "episode_reward",
        "completed_tasks",
        "completion_rate",
        "late_tasks",
        "risk_exposure_sum",
        "total_path_length",
        "total_energy",
        "total_conflicts",
        "total_invalid_actions",
        "total_replenishments",
        "total_returns",
    ]
    aggregate: dict[str, dict[str, float]] = {}
    for policy, policy_rows in grouped.items():
        aggregate[policy] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in policy_rows]
            aggregate[policy][f"{metric}_mean"] = sum(values) / max(len(values), 1)
            aggregate[policy][f"{metric}_min"] = min(values)
            aggregate[policy][f"{metric}_max"] = max(values)
    return aggregate


def _environment_control(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(row)
    control: dict[str, dict[str, Any]] = {}
    for seed, seed_rows in grouped.items():
        environment_hashes = sorted({str(row["initial_environment_hash"]) for row in seed_rows})
        truth_hashes = sorted({str(row["initial_task_truth_hash"]) for row in seed_rows})
        control[str(seed)] = {
            "matched_initial_environment": len(environment_hashes) == 1,
            "matched_task_truth": len(truth_hashes) == 1,
            "initial_environment_hashes": environment_hashes,
            "task_truth_hashes": truth_hashes,
        }
    return control


def _environment_snapshot(env) -> dict[str, Any]:
    task_truth = [
        {
            "task_id": task.task_id,
            "risk": int(task.risk),
            "true_anomaly": bool(task.true_anomaly),
            "screening_workload": float(task.screening_workload),
            "review_workload": float(task.review_workload),
            "max_interval": float(task.max_interval),
        }
        for task in env.tasks
    ]
    payload = {
        "scenario_seed": getattr(env, "_scenario_seed", None),
        "current_step": int(env.current_step),
        "platform_resources": _platform_resources(env),
        "task_truth": task_truth,
    }
    return {
        "environment_hash": _hash_json(payload),
        "task_truth_hash": _hash_json(task_truth),
        "platform_resources": payload["platform_resources"],
    }


def _platform_resources(env) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    for platform in env.platforms:
        energy_capacity = max(float(platform.energy_capacity), 1e-6)
        resources.append(
            {
                "platform_id": platform.platform_id,
                "platform_type": platform.platform_type,
                "cell": list(platform.current_cell),
                "depot": list(env._platform_depot(platform)),
                "mode": platform.mode,
                "energy": round(float(platform.energy), 8),
                "energy_capacity": round(float(platform.energy_capacity), 8),
                "energy_ratio": round(float(platform.energy) / energy_capacity, 8),
                "remaining_travel_time": round(float(platform.remaining_travel_time), 8),
                "remaining_service_time": round(float(platform.remaining_service_time), 8),
                "remaining_replenish_time": round(float(platform.remaining_replenish_time), 8),
            }
        )
    return resources


def _hash_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


def _write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


if __name__ == "__main__":
    main()
