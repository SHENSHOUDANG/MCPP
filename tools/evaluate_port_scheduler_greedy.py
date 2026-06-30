from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

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

from check_port_inspection_env import build_env
from mathbased_mcpp.port_inspection.v12_contract import classify_config_boundary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate greedy schedulers inside the RL scheduling environment.")
    parser.add_argument("--config", default="configs/port_los_angeles_training_v1.toml")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--strategy",
        choices=("legacy_order", "global_score"),
        default="legacy_order",
        help="legacy_order mirrors the original sorted-task greedy baseline; global_score picks the best legal pair each step.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    contract_boundary = classify_config_boundary(config).as_dict()
    env = build_env(config)
    output_dir = Path(args.output_dir or str(config.get("output_dir", "outputs/port_inspection/scheduler")))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, trace = run_greedy_env_rollout(env, config, seed=args.seed, strategy=args.strategy)
    summary["contract_boundary"] = contract_boundary
    summary_path = output_dir / f"greedy_env_{args.strategy}_summary.json"
    trace_path = output_dir / f"greedy_env_{args.strategy}_trace.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_trace(trace_path, trace)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")
    print(f"trace={trace_path}")


def run_greedy_env_rollout(env, config: dict[str, Any], seed: int, strategy: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env.reset(seed=seed)
    scheduling_config = dict(config.get("scheduling", {}))
    weights = {
        "risk_weight": float(scheduling_config.get("risk_weight", 10.0)),
        "distance_weight": float(scheduling_config.get("distance_weight", 0.05)),
        "load_weight": float(scheduling_config.get("load_weight", 0.8)),
        "compatibility_bonus": float(scheduling_config.get("compatibility_bonus", 3.0)),
    }
    trace: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False

    while not done:
        action, decision = _choose_action(env, weights, strategy)
        result = env.step(action)
        total_reward += float(result.reward)
        info = result.info
        row = {
            "step": env.current_step,
            "task_lifecycle": getattr(env, "task_lifecycle", "legacy_screen_review"),
            "action": json.dumps(action, ensure_ascii=False),
            "platform_id": decision["platform_id"],
            "task_id": decision["task_id"],
            "task_geometry": decision["task_geometry"],
            "task_stage": decision["task_stage"],
            "risk": decision["risk"],
            "score": decision["score"],
            "path_length": decision["path_length"],
            "energy_cost": decision["energy_cost"],
            "reward": float(result.reward),
            "completed_count": len(info["completed_tasks"]),
            "late_count": len(info["late_tasks"]),
            "risk_exposure_sum": float(info["risk_exposure_sum"]),
            "total_path_length": int(info["total_path_length"]),
            "total_energy": float(info["total_energy"]),
            "accepted_actions": json.dumps(info["accepted_actions"], ensure_ascii=False),
            "conflicts": json.dumps(info["conflicts"], ensure_ascii=False),
            "open_task_count": int(env.open_task_count()),
            "valid_action_count": int(env.action_masks().sum()),
        }
        if getattr(env, "task_lifecycle", "") != "v1_2_direct_service":
            row["review_queue_length"] = int(info["review_queue_length"])
        trace.append(row)
        done = result.done

    info = env.info()
    summary = {
        "strategy": strategy,
        "task_lifecycle": getattr(env, "task_lifecycle", "legacy_screen_review"),
        "steps": env.current_step,
        "episode_reward": total_reward,
        "completed_tasks": len(env.completed_tasks),
        "task_count": env.num_tasks,
        "completion_rate": len(env.completed_tasks) / max(env.num_tasks, 1),
        "late_tasks": len(info["late_tasks"]),
        "late_task_ids": info["late_tasks"],
        "risk_exposure_sum": float(info["risk_exposure_sum"]),
        "total_path_length": int(info["total_path_length"]),
        "total_energy": float(info["total_energy"]),
        "platform_loads": info["platform_loads"],
        "open_task_count": int(env.open_task_count()),
    }
    if getattr(env, "task_lifecycle", "") != "v1_2_direct_service":
        summary["review_queue_length"] = int(info["review_queue_length"])
    return summary, trace


def _choose_action(env, weights: dict[str, float], strategy: str) -> tuple[list[int], dict[str, Any]]:
    if strategy == "legacy_order":
        return _choose_legacy_order_action(env, weights)
    return _choose_global_score_action(env, weights)


def _choose_legacy_order_action(env, weights: dict[str, float]) -> tuple[list[int], dict[str, Any]]:
    candidate_sets = env.candidate_lists()
    ordered_indices = [task_index for task_index, _ in _ordered_task_indices(env)]
    actions = [0 for _ in env.platforms]
    decisions: list[dict[str, Any]] = []
    for platform_index, platform_candidates in enumerate(candidate_sets):
        ranked = []
        for candidate in platform_candidates:
            task = env.tasks[candidate.task_index]
            cost = env._estimate_cost(platform_index, candidate.task_index, candidate.task_stage)
            score = _score(env.platforms[platform_index], task, candidate, cost, weights)
            order_rank = ordered_indices.index(candidate.task_index) if candidate.task_index in ordered_indices else len(ordered_indices)
            ranked.append((order_rank, -score, platform_index, candidate, cost, score))
        if not ranked:
            continue
        _, _, platform_index, candidate, cost, score = min(ranked, key=lambda item: (item[0], item[1]))
        actions[platform_index] = candidate.relative_position
        decisions.append(_decision(env, platform_index, candidate, cost, score))
    return actions, _merge_decisions(env, decisions)


def _choose_global_score_action(env, weights: dict[str, float]) -> tuple[list[int], dict[str, Any]]:
    actions = [0 for _ in env.platforms]
    decisions: list[dict[str, Any]] = []
    for platform_index, platform_candidates in enumerate(env.candidate_lists()):
        candidates = []
        for candidate in platform_candidates:
            task = env.tasks[candidate.task_index]
            cost = env._estimate_cost(platform_index, candidate.task_index, candidate.task_stage)
            score = _score(env.platforms[platform_index], task, candidate, cost, weights)
            candidates.append((score, platform_index, candidate, cost))
        if not candidates:
            continue
        score, platform_index, candidate, cost = max(candidates, key=lambda item: item[0])
        actions[platform_index] = candidate.relative_position
        decisions.append(_decision(env, platform_index, candidate, cost, score))
    return actions, _merge_decisions(env, decisions)


def _ordered_task_indices(env) -> list[tuple[int, Any]]:
    indexed = [(index, task) for index, task in enumerate(env.tasks) if task.active_stage is not None]
    return sorted(indexed, key=lambda item: (-item[1].risk, _geometry_priority(item[1].geometry), item[1].task_id))


def _score(platform, task, candidate, cost, weights: dict[str, float]) -> float:
    bonus = weights["compatibility_bonus"] + _platform_task_bonus(platform.platform_type, task.geometry)
    if task.geometry in platform.preferred_task_types:
        bonus += weights["compatibility_bonus"] * 0.6
    return (
        weights["risk_weight"] * task.risk
        + 2.0 * candidate.urgency
        + 2.0 * candidate.confidence
        + 0.2 * candidate.review_waiting_time
        - weights["distance_weight"] * cost.path_length
        - weights["load_weight"] * platform.current_load
        + bonus
    )


def _decision(env, platform_index: int, candidate, cost, score: float) -> dict[str, Any]:
    platform = env.platforms[platform_index]
    task = env.tasks[candidate.task_index]
    return {
        "platform_id": platform.platform_id,
        "task_id": task.task_id,
        "task_geometry": task.geometry,
        "task_stage": candidate.task_stage,
        "risk": task.risk,
        "score": score,
        "path_length": cost.path_length,
        "energy_cost": cost.energy_cost,
    }


def _merge_decisions(env, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    if not decisions:
        return {
            "platform_id": env.platforms[0].platform_id if env.platforms else "NA",
            "task_id": "WAIT",
            "task_geometry": "wait",
            "task_stage": "wait",
            "risk": 0,
            "score": 0.0,
            "path_length": 0,
            "energy_cost": 0.0,
        }
    return {
        "platform_id": ";".join(str(item["platform_id"]) for item in decisions),
        "task_id": ";".join(str(item["task_id"]) for item in decisions),
        "task_geometry": ";".join(str(item["task_geometry"]) for item in decisions),
        "task_stage": ";".join(str(item["task_stage"]) for item in decisions),
        "risk": sum(int(item["risk"]) for item in decisions),
        "score": sum(float(item["score"]) for item in decisions),
        "path_length": sum(int(item["path_length"]) for item in decisions),
        "energy_cost": sum(float(item["energy_cost"]) for item in decisions),
    }


def _geometry_priority(geometry: str) -> int:
    return {"point": 0, "line": 1, "area": 2}.get(geometry, 3)


def _platform_task_bonus(platform_type: str, geometry: str) -> float:
    if platform_type == "UAV" and geometry in {"point", "line"}:
        return 1.5
    if platform_type == "USV" and geometry == "area":
        return 1.0
    return 0.0


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
