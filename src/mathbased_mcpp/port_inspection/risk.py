from __future__ import annotations

from .schema import InspectionTask


def update_uninspected_time(tasks: list[InspectionTask], completed_task_ids: set[str], delta_t: float = 1.0) -> None:
    for task in tasks:
        if task.task_id in completed_task_ids:
            task.uninspected_time = 0.0
            task.completed = True
        elif not task.completed:
            task.uninspected_time += float(delta_t)


def risk_exposure(task: InspectionTask) -> float:
    return float(task.risk) * float(task.uninspected_time)


def lateness(task: InspectionTask) -> float:
    return max(0.0, float(task.uninspected_time) - float(task.max_interval))


def total_risk_exposure(tasks: list[InspectionTask]) -> float:
    return sum(risk_exposure(task) for task in tasks if not task.completed)


def late_task_count(tasks: list[InspectionTask]) -> int:
    return sum(1 for task in tasks if not task.completed and lateness(task) > 0.0)
