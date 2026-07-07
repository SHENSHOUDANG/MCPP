from __future__ import annotations

from .schema import InspectionTask, Platform


def compute_reward_terms(
    tasks: list[InspectionTask],
    platforms: list[Platform],
    completed_task: InspectionTask | None,
    path_length: int,
    energy_cost: float,
    invalid: bool,
    weights: dict[str, float | str] | None = None,
    screened_tasks: list[InspectionTask] | None = None,
    reviewed_tasks: list[InspectionTask] | None = None,
    service_tasks: list[InspectionTask] | None = None,
    closed_tasks: list[InspectionTask] | None = None,
    newly_assigned_tasks: list[InspectionTask] | None = None,
    terminal_unassigned_tasks: list[InspectionTask] | None = None,
    current_time: float | None = None,
    horizon_end: float | None = None,
    conflict_count: int = 0,
    review_queue_length: int = 0,
) -> dict[str, float]:
    cfg = {
        "team_close_reward": 5.0,
        "screen_progress_reward": 0.4,
        "review_progress_reward": 0.6,
        "service_progress_reward": 0.4,
        "energy_cost": 3.0,
        "time_cost": 0.08,
        "wait_time_cost": 0.0,
        "wait_time_scale": 60.0,
        "wait_time_aggregation": "mean_open",
        "invalid_penalty": 3.0,
        "conflict_penalty": 0.5,
    }
    if weights:
        for key, value in weights.items():
            cfg[key] = str(value) if key == "wait_time_aggregation" else float(value)

    if closed_tasks is None:
        closed_tasks = [completed_task] if completed_task is not None else []
    screened_tasks = screened_tasks or []
    reviewed_tasks = reviewed_tasks or []
    service_tasks = service_tasks or []
    newly_assigned_tasks = newly_assigned_tasks or []
    terminal_unassigned_tasks = terminal_unassigned_tasks or []

    complete = 0.0
    for task in closed_tasks:
        complete += cfg["team_close_reward"] * task.risk * task.priority

    screen_progress = cfg["screen_progress_reward"] * sum(task.risk * task.priority for task in screened_tasks)
    review_progress = cfg["review_progress_reward"] * sum(task.risk * task.priority for task in reviewed_tasks)
    service_progress = cfg["service_progress_reward"] * sum(task.risk * task.priority for task in service_tasks)
    alive_count = sum(1 for platform in platforms if platform.alive)
    current_wait_load = _current_wait_load(
        tasks=tasks,
        newly_assigned_tasks=newly_assigned_tasks,
        current_time=current_time,
        wait_time_scale=float(cfg["wait_time_scale"]),
        aggregation=str(cfg["wait_time_aggregation"]),
    )

    terms = {
        "team_close_reward": complete,
        "screen_progress_reward": screen_progress,
        "review_progress_reward": review_progress,
        "service_progress_reward": service_progress,
        "energy_cost": -cfg["energy_cost"] * energy_cost,
        "time_cost": -cfg["time_cost"] * alive_count,
        "wait_time_cost": -cfg["wait_time_cost"] * current_wait_load,
        "invalid_penalty": -cfg["invalid_penalty"] if invalid else 0.0,
        "conflict_penalty": -cfg["conflict_penalty"] * max(int(conflict_count), 0),
    }
    terms["total"] = sum(terms.values())
    return terms


def _assignment_wait(task: InspectionTask, current_time: float | None) -> float:
    if task.release_time is None:
        return 0.0
    assignment_time = task.first_valid_assignment_time
    if assignment_time is None:
        assignment_time = current_time
    if assignment_time is None:
        return 0.0
    return max(0.0, float(assignment_time) - float(task.release_time))


def _current_wait_load(
    tasks: list[InspectionTask],
    newly_assigned_tasks: list[InspectionTask],
    current_time: float | None,
    wait_time_scale: float,
    aggregation: str,
) -> float:
    if current_time is None:
        return 0.0
    waits = [
        _open_wait(task, current_time)
        for task in tasks
        if task.release_time is not None and task.first_valid_assignment_time is None and not task.completed
    ]
    waits.extend(_assignment_wait(task, current_time) for task in newly_assigned_tasks)
    waits = [wait for wait in waits if wait > 0.0]
    if not waits:
        return 0.0
    if aggregation == "sum":
        raw_wait = sum(waits)
    elif aggregation == "mean_released":
        released_count = sum(1 for task in tasks if task.release_time is not None)
        raw_wait = sum(waits) / max(released_count, 1)
    elif aggregation == "mean_open":
        raw_wait = sum(waits) / len(waits)
    else:
        raise ValueError(f"unsupported wait_time_aggregation: {aggregation!r}")
    return raw_wait / max(float(wait_time_scale), 1e-6)


def _open_wait(task: InspectionTask, current_time: float) -> float:
    if task.release_time is None:
        return 0.0
    return max(0.0, float(current_time) - float(task.release_time))
