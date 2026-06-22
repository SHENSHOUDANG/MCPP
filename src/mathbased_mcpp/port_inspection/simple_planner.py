from __future__ import annotations

from collections import deque
from dataclasses import replace
import heapq
from typing import Any

from .schema import AssignmentResult, GridCell, InspectionTask, Platform, PortGridMap


NEIGHBORS: tuple[GridCell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def create_platforms(
    depot: GridCell,
    uav_count: int,
    usv_count: int,
    uav_config: dict[str, Any],
    usv_config: dict[str, Any],
) -> list[Platform]:
    platforms: list[Platform] = []
    for index in range(max(int(uav_count), 0)):
        platforms.append(_platform(f"UAV-{index + 1}", "UAV", depot, uav_config))
    for index in range(max(int(usv_count), 0)):
        platforms.append(_platform(f"USV-{index + 1}", "USV", depot, usv_config))
    if not platforms:
        raise ValueError("at least one UAV or USV platform is required")
    return platforms


def assign_tasks(
    grid: PortGridMap,
    tasks: list[InspectionTask],
    platforms: list[Platform],
    risk_weight: float = 10.0,
    distance_weight: float = 0.05,
    load_weight: float = 0.8,
    compatibility_bonus: float = 3.0,
) -> list[AssignmentResult]:
    ordered_tasks = sorted(tasks, key=lambda task: (-task.risk, _geometry_priority(task.geometry), task.task_id))
    working_platforms = [replace(platform, route=list(platform.route)) for platform in platforms]
    results: list[AssignmentResult] = []
    for order, task in enumerate(ordered_tasks, start=1):
        scored = [
            _score_platform(
                grid,
                platform,
                task,
                risk_weight=risk_weight,
                distance_weight=distance_weight,
                load_weight=load_weight,
                compatibility_bonus=compatibility_bonus,
            )
            for platform in working_platforms
            if platform.can_execute(task)
        ]
        if not scored:
            raise ValueError(f"no compatible platform for task {task.task_id}")
        score, platform, travel_path = max(scored, key=lambda item: item[0])
        execution_path = _execution_path(grid, task)
        path = _join_paths(travel_path, execution_path)
        path_length = max(len(path) - 1, 0)
        result = AssignmentResult(
            task_id=task.task_id,
            task_type=task.task_type,
            task_geometry=task.geometry,
            risk=task.risk,
            assigned_platform=platform.platform_id,
            platform_type=platform.platform_type,
            start_cell=platform.current_cell,
            entry_cell=task.entry_cell,
            exit_cell=task.exit_cell,
            path_length=path_length,
            service_time=task.service_time,
            completion_order=order,
            executor=task.executor,
            score=score,
            path=tuple(path),
        )
        results.append(result)
        platform.current_cell = task.exit_cell
        platform.current_load += path_length + task.service_time
        platform.route.extend(path[1:] if platform.route else path)
    return results


def shortest_path(grid: PortGridMap, start: GridCell, goal: GridCell) -> list[GridCell]:
    if start == goal:
        return [start]
    free = grid.free_cell_set
    if start not in free or goal not in free:
        raise ValueError(f"start and goal must be free cells: {start}, {goal}")
    frontier: list[tuple[int, int, GridCell]] = [(0, 0, start)]
    came_from: dict[GridCell, GridCell | None] = {start: None}
    cost_so_far: dict[GridCell, int] = {start: 0}
    sequence = 0
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in _free_neighbors(current, free):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                sequence += 1
                priority = new_cost + _manhattan(neighbor, goal)
                heapq.heappush(frontier, (priority, sequence, neighbor))
                came_from[neighbor] = current
    if goal not in came_from:
        raise ValueError(f"no water path from {start} to {goal}")
    return _reconstruct_path(came_from, goal)


def _platform(platform_id: str, platform_type: str, depot: GridCell, config: dict[str, Any]) -> Platform:
    reserved = {
        "speed_mps",
        "endurance_minutes",
        "allowed_task_types",
        "preferred_task_types",
        "payload_kg",
        "coverage_width_cells",
        "energy_rate_per_cell",
        "max_speed_mps",
        "max_speed_mps_reference",
        "nominal_endurance_minutes",
        "nominal_endurance_min",
        "max_endurance_minutes_reference",
        "return_reserve_ratio",
        "sensor_radius_m",
        "energy_capacity",
        "energy",
    }
    return Platform(
        platform_id=platform_id,
        platform_type=platform_type,
        current_cell=depot,
        speed_mps=float(config.get("speed_mps", 10.0)),
        endurance_minutes=float(config.get("endurance_minutes", 30.0)),
        allowed_task_types=tuple(str(item) for item in config.get("allowed_task_types", ("point", "line", "area"))),
        preferred_task_types=tuple(str(item) for item in config.get("preferred_task_types", ())),
        max_speed_mps=float(config.get("max_speed_mps", config.get("max_speed_mps_reference", 0.0))),
        nominal_endurance_minutes=float(
            config.get("nominal_endurance_minutes", config.get("nominal_endurance_min", config.get("max_endurance_minutes_reference", 0.0)))
        ),
        return_reserve_ratio=float(config.get("return_reserve_ratio", 0.15)),
        sensor_radius_m=float(config.get("sensor_radius_m", 0.0)),
        energy_capacity=float(config.get("energy_capacity", 1.0)),
        energy=float(config.get("energy", config.get("energy_capacity", 1.0))),
        payload_kg=float(config.get("payload_kg", 0.0)),
        coverage_width_cells=int(config.get("coverage_width_cells", 1)),
        energy_rate_per_cell=float(config.get("energy_rate_per_cell", 1.0)),
        metadata={key: value for key, value in config.items() if key not in reserved},
        route=[depot],
    )


def _score_platform(
    grid: PortGridMap,
    platform: Platform,
    task: InspectionTask,
    risk_weight: float,
    distance_weight: float,
    load_weight: float,
    compatibility_bonus: float,
) -> tuple[float, Platform, list[GridCell]]:
    travel_path = shortest_path(grid, platform.current_cell, task.entry_cell)
    travel_distance = max(len(travel_path) - 1, 0)
    bonus = compatibility_bonus + _platform_task_bonus(platform.platform_type, task.geometry)
    if task.geometry in platform.preferred_task_types:
        bonus += compatibility_bonus * 0.6
    score = risk_weight * task.risk - distance_weight * travel_distance - load_weight * platform.current_load + bonus
    return score, platform, travel_path


def _execution_path(grid: PortGridMap, task: InspectionTask) -> list[GridCell]:
    if task.geometry == "point":
        return [task.entry_cell]
    if task.geometry == "line":
        return _connect_sequence(grid, list(task.cells))
    return _boustrophedon_area_path(grid, task.cells)


def _connect_sequence(grid: PortGridMap, cells: list[GridCell]) -> list[GridCell]:
    if not cells:
        return []
    path = [cells[0]]
    for start, goal in zip(cells, cells[1:]):
        path = _join_paths(path, shortest_path(grid, start, goal))
    return path


def _boustrophedon_area_path(grid: PortGridMap, cells: tuple[GridCell, ...]) -> list[GridCell]:
    remaining = set(cells)
    if not remaining:
        return []
    path: list[GridCell] = []
    rows = sorted({row for row, _ in remaining})
    for index, row in enumerate(rows):
        row_cells = sorted((cell for cell in remaining if cell[0] == row), key=lambda cell: cell[1])
        if index % 2 == 1:
            row_cells.reverse()
        for cell in row_cells:
            if not path:
                path.append(cell)
            elif cell != path[-1]:
                path = _join_paths(path, shortest_path(grid, path[-1], cell))
    return path


def _join_paths(first: list[GridCell], second: list[GridCell]) -> list[GridCell]:
    if not first:
        return list(second)
    if not second:
        return list(first)
    return first + second[1:] if first[-1] == second[0] else first + second


def _free_neighbors(cell: GridCell, free: set[GridCell]) -> list[GridCell]:
    row, col = cell
    return [(row + dr, col + dc) for dr, dc in NEIGHBORS if (row + dr, col + dc) in free]


def _reconstruct_path(came_from: dict[GridCell, GridCell | None], goal: GridCell) -> list[GridCell]:
    path = deque([goal])
    current = goal
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.appendleft(current)
    return list(path)


def _geometry_priority(geometry: str) -> int:
    return {"point": 0, "line": 1, "area": 2}.get(geometry, 3)


def _platform_task_bonus(platform_type: str, geometry: str) -> float:
    if platform_type == "UAV" and geometry in {"point", "line"}:
        return 1.5
    if platform_type == "USV" and geometry == "area":
        return 1.0
    return 0.0


def _manhattan(first: GridCell, second: GridCell) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])
