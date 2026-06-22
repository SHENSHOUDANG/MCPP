from __future__ import annotations

from dataclasses import dataclass

from .schema import GridCell, InspectionTask, Platform, PortGridMap, STAGE_REVIEW, STAGE_SCREENING
from .simple_planner import shortest_path


_USV_DISTANCE_CACHE: dict[tuple[str, GridCell, GridCell], int] = {}


@dataclass(slots=True)
class TaskCost:
    path_length: int
    travel_time: float
    energy_cost: float
    return_cost: float
    completion_time: float
    feasible: bool
    entry_cell: GridCell
    exit_cell: GridCell
    coverage_ratio: float = 1.0


def estimate_task_cost(platform: Platform, task: InspectionTask, grid: PortGridMap, stage: str | None = None) -> TaskCost:
    if not platform.can_execute(task, stage):
        return _infeasible(task)
    try:
        if task.geometry == "point":
            cost = point_task_path_proxy(platform, task, grid, stage=stage)
        elif task.geometry == "line":
            cost = line_task_path_proxy(platform, task, grid, stage=stage)
        else:
            cost = area_task_execute_stub(task, platform, grid, stage=stage)
    except ValueError:
        return _infeasible(task)
    reserve = min(max(platform.return_reserve_ratio, 0.0), 0.95)
    feasible = cost.energy_cost + cost.return_cost <= max(platform.energy - reserve, 0.0)
    return TaskCost(
        path_length=cost.path_length,
        travel_time=cost.travel_time,
        energy_cost=cost.energy_cost,
        return_cost=cost.return_cost,
        completion_time=cost.completion_time,
        feasible=feasible,
        entry_cell=cost.entry_cell,
        exit_cell=cost.exit_cell,
        coverage_ratio=cost.coverage_ratio,
    )


def point_task_path_proxy(platform: Platform, task: InspectionTask, grid: PortGridMap, stage: str | None = None) -> TaskCost:
    entry = task.entry_cell
    travel_len = _travel_distance(platform, grid, platform.current_cell, entry)
    return_len = _travel_distance(platform, grid, entry, grid.depot)
    return _cost_from_lengths(platform, grid, task, travel_len, return_len, entry, entry, coverage_ratio=1.0, stage=stage)


def line_task_path_proxy(platform: Platform, task: InspectionTask, grid: PortGridMap, stage: str | None = None) -> TaskCost:
    cells = list(task.cells)
    forward = _line_length(platform, grid, platform.current_cell, cells)
    backward = _line_length(platform, grid, platform.current_cell, list(reversed(cells)))
    if backward[0] < forward[0]:
        path_len, entry, exit_cell = backward
    else:
        path_len, entry, exit_cell = forward
    return_len = _travel_distance(platform, grid, exit_cell, grid.depot)
    return _cost_from_lengths(platform, grid, task, path_len, return_len, entry, exit_cell, coverage_ratio=1.0, stage=stage)


def area_task_execute_stub(task: InspectionTask, platform: Platform, grid: PortGridMap, stage: str | None = None) -> TaskCost:
    entry = task.entry_cell
    exit_cell = task.exit_cell
    travel_len = _travel_distance(platform, grid, platform.current_cell, entry)
    coverage_cells = max(1, int(len(task.cells) * task.coverage_threshold))
    sweep_len = max(1, coverage_cells // max(platform.coverage_width_cells, 1))
    transfer_len = _travel_distance(platform, grid, entry, exit_cell)
    path_len = travel_len + sweep_len + transfer_len
    return_len = _travel_distance(platform, grid, exit_cell, grid.depot)
    return _cost_from_lengths(platform, grid, task, path_len, return_len, entry, exit_cell, coverage_ratio=task.coverage_threshold, stage=stage)


def _line_length(platform: Platform, grid: PortGridMap, start: GridCell, cells: list[GridCell]) -> tuple[int, GridCell, GridCell]:
    if not cells:
        return 0, start, start
    total = _travel_distance(platform, grid, start, cells[0])
    for first, second in zip(cells, cells[1:]):
        total += _travel_distance(platform, grid, first, second)
    return total, cells[0], cells[-1]


def _travel_distance(platform: Platform, grid: PortGridMap, start: GridCell, goal: GridCell) -> int:
    if platform.platform_type == "UAV":
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    key = (grid.name, start, goal)
    if key not in _USV_DISTANCE_CACHE:
        _USV_DISTANCE_CACHE[key] = max(len(shortest_path(grid, start, goal)) - 1, 0)
    return _USV_DISTANCE_CACHE[key]


def _cost_from_lengths(
    platform: Platform,
    grid: PortGridMap,
    task: InspectionTask,
    path_len: int,
    return_len: int,
    entry: GridCell,
    exit_cell: GridCell,
    coverage_ratio: float,
    stage: str | None = None,
) -> TaskCost:
    cell_m = max(float(grid.cell_size_m), 1.0)
    speed = max(float(platform.speed_mps), 0.1)
    travel_time = path_len * cell_m / speed / 60.0
    service_time = _stage_service_time(task, stage)
    completion_time = travel_time + service_time
    return_time = return_len * cell_m / speed / 60.0
    energy_base = max(float(platform.endurance_minutes), 1.0)
    energy_cost = completion_time / energy_base * max(platform.energy_rate_per_cell, 0.1)
    return_cost = return_time / energy_base * max(platform.energy_rate_per_cell, 0.1)
    return TaskCost(
        path_length=path_len,
        travel_time=travel_time,
        energy_cost=energy_cost,
        return_cost=return_cost,
        completion_time=completion_time,
        feasible=True,
        entry_cell=entry,
        exit_cell=exit_cell,
        coverage_ratio=coverage_ratio,
    )


def _stage_service_time(task: InspectionTask, stage: str | None) -> float:
    if stage == STAGE_SCREENING:
        return max(float(task.screening_workload), 1.0)
    if stage == STAGE_REVIEW:
        return max(float(task.review_workload), 1.0)
    return float(task.service_time)


def _infeasible(task: InspectionTask) -> TaskCost:
    return TaskCost(
        path_length=0,
        travel_time=0.0,
        energy_cost=0.0,
        return_cost=0.0,
        completion_time=0.0,
        feasible=False,
        entry_cell=task.entry_cell,
        exit_cell=task.exit_cell,
    )
