from __future__ import annotations

from .schema import InspectionTask, Platform


def compute_reward_terms(
    tasks: list[InspectionTask],
    platforms: list[Platform],
    completed_task: InspectionTask | None,
    path_length: int,
    energy_cost: float,
    invalid: bool,
    weights: dict[str, float] | None = None,
    screened_tasks: list[InspectionTask] | None = None,
    reviewed_tasks: list[InspectionTask] | None = None,
    service_tasks: list[InspectionTask] | None = None,
    closed_tasks: list[InspectionTask] | None = None,
    conflict_count: int = 0,
    review_queue_length: int = 0,
) -> dict[str, float]:
    cfg = {
        "team_close_reward": 8.0,
        "screen_progress_reward": 1.0,
        "review_progress_reward": 1.5,
        "service_progress_reward": 1.0,
        "energy_cost": 0.5,
        "time_cost": 0.01,
        "invalid_penalty": 5.0,
        "conflict_penalty": 1.0,
    }
    if weights:
        cfg.update({key: float(value) for key, value in weights.items()})

    if closed_tasks is None:
        closed_tasks = [completed_task] if completed_task is not None else []
    screened_tasks = screened_tasks or []
    reviewed_tasks = reviewed_tasks or []
    service_tasks = service_tasks or []

    complete = 0.0
    for task in closed_tasks:
        complete += cfg["team_close_reward"] * task.risk * task.priority

    screen_progress = cfg["screen_progress_reward"] * sum(task.risk * task.priority for task in screened_tasks)
    review_progress = cfg["review_progress_reward"] * sum(task.risk * task.priority for task in reviewed_tasks)
    service_progress = cfg["service_progress_reward"] * sum(task.risk * task.priority for task in service_tasks)
    alive_count = sum(1 for platform in platforms if platform.alive)

    terms = {
        "team_close_reward": complete,
        "screen_progress_reward": screen_progress,
        "review_progress_reward": review_progress,
        "service_progress_reward": service_progress,
        "energy_cost": -cfg["energy_cost"] * energy_cost,
        "time_cost": -cfg["time_cost"] * alive_count,
        "invalid_penalty": -cfg["invalid_penalty"] if invalid else 0.0,
        "conflict_penalty": -cfg["conflict_penalty"] * max(int(conflict_count), 0),
    }
    terms["total"] = sum(terms.values())
    return terms
