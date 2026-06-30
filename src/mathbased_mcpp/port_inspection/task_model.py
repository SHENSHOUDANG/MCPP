from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import GridCell, InspectionTask, PortGridMap, TASK_CLOSED, TASK_COMPLETED, TASK_UNSCREENED


def load_inspection_tasks(path: str | Path, grid: PortGridMap | None = None) -> list[InspectionTask]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks: list[InspectionTask] = []
    for item in raw.get("point_tasks", []):
        cells = item.get("cells", [item["cell"]])
        tasks.append(_task_from_raw(item, geometry="point", cells=cells))
    for item in raw.get("line_tasks", []):
        tasks.append(_task_from_raw(item, geometry="line", cells=list(item["cells"])))
    for item in raw.get("area_tasks", []):
        tasks.append(_task_from_raw(item, geometry="area", cells=list(item["cells"])))
    if grid is not None:
        _validate_tasks(tasks, grid)
    return tasks


def _task_from_raw(raw: dict[str, Any], geometry: str, cells: list[Any]) -> InspectionTask:
    cell_tuple = tuple(_cell(item) for item in cells)
    if not cell_tuple:
        raise ValueError(f"task {raw.get('id', '<unknown>')} has no cells")
    metadata = dict(raw.get("metadata", {}))
    default_service = max(1, len(cell_tuple) // (8 if geometry == "area" else 1))
    service_time = int(raw.get("service_time", default_service))
    screening_workload = float(raw.get("screening_workload", max(1.0, service_time * 0.6)))
    review_workload = float(raw.get("review_workload", max(1.0, service_time * 1.2)))
    state = str(raw.get("state", TASK_UNSCREENED))
    completed = bool(raw.get("completed", state in {TASK_CLOSED, TASK_COMPLETED}))
    required_work = float(raw.get("required_work", metadata.get("required_work", max(1.0, service_time))))
    completed_work = float(raw.get("completed_work", metadata.get("completed_work", 0.0)))
    remaining_work = float(raw.get("remaining_work", metadata.get("remaining_work", max(required_work - completed_work, 0.0))))
    deadline = _optional_float(raw.get("deadline", metadata.get("deadline", raw.get("max_interval"))))
    return InspectionTask(
        task_id=str(raw["id"]),
        task_type=str(raw["type"]),
        geometry=geometry,
        cells=cell_tuple,
        risk=int(raw["risk"]),
        service_time=service_time,
        allowed_platforms=tuple(str(item).upper() for item in raw.get("allowed_platforms", ("UAV", "USV"))),
        max_interval=int(raw.get("max_interval", _default_max_interval(int(raw["risk"])))),
        coverage_threshold=float(raw.get("coverage_threshold", 1.0 if geometry != "area" else 0.9)),
        priority=float(raw.get("priority", int(raw["risk"]))),
        completed=completed,
        task_family=str(metadata.get("task_family", raw.get("task_family", raw["type"]))),
        geometry_mode=str(metadata.get("geometry_mode", raw.get("geometry_mode", _default_geometry_mode(geometry)))),
        release_mode=str(metadata.get("release_mode", raw.get("release_mode", "PERIODIC"))),
        required_work=required_work,
        completed_work=completed_work,
        remaining_work=remaining_work,
        work_threshold=float(raw.get("work_threshold", metadata.get("work_threshold", 1.0))),
        quality_pass=bool(raw.get("quality_pass", metadata.get("quality_pass", True))),
        quality_requirement=dict(raw.get("quality_requirement", metadata.get("quality_requirement", {}))),
        quality_acceptance_ref=str(raw.get("quality_acceptance_ref", metadata.get("quality_acceptance_ref", ""))),
        executor=str(raw.get("executor", "rule_based")),
        parent_task_id=str(raw["parent_task_id"]) if raw.get("parent_task_id") is not None else None,
        state=TASK_CLOSED if completed and state != TASK_COMPLETED else state,
        screening_workload=screening_workload,
        review_workload=review_workload,
        screening_workload_remaining=float(raw.get("screening_workload_remaining", screening_workload)),
        review_workload_remaining=float(raw.get("review_workload_remaining", review_workload)),
        deadline=deadline,
        review_deadline=float(raw.get("review_deadline", 0.0)),
        generation_time=float(raw.get("generation_time", 0.0)),
        true_anomaly=bool(raw.get("true_anomaly", False)),
        metadata=metadata,
    )


def _cell(value: Any) -> GridCell:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"expected [row, col], got {value!r}")
    return int(value[0]), int(value[1])


def _validate_tasks(tasks: list[InspectionTask], grid: PortGridMap) -> None:
    free = grid.free_cell_set
    for task in tasks:
        if task.geometry not in {"point", "line", "area"}:
            raise ValueError(f"unsupported task geometry: {task.geometry}")
        if task.geometry in {"line", "area"} and len(task.cells) < 2:
            raise ValueError(f"{task.geometry} task must reference at least two cells: {task.task_id}")
        if task.risk < 1 or task.risk > 3:
            raise ValueError(f"task risk must be in [1, 3]: {task.task_id}")
        for cell in task.cells:
            if cell not in free:
                raise ValueError(f"task {task.task_id} references non-water cell {cell}")


def _default_max_interval(risk: int) -> int:
    if risk >= 3:
        return 10
    if risk == 2:
        return 24
    return 36


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _default_geometry_mode(geometry: str) -> str:
    return {
        "point": "TARGET",
        "line": "CORRIDOR",
        "area": "AREA",
    }.get(geometry, "TARGET")
