from __future__ import annotations

import json
from pathlib import Path


WIDTH = 50
HEIGHT = 50


def main() -> None:
    root = Path("data/ports/dalian_port_v0")
    root.mkdir(parents=True, exist_ok=True)
    obstacles = _obstacles()
    risk_grid = _risk_grid(obstacles)
    free_cells = [
        [row, col]
        for row in range(HEIGHT)
        for col in range(WIDTH)
        if (row, col) not in obstacles
    ]
    grid = {
        "name": "dalian_port_v0",
        "description": "Prototype grid for port water-surface inspection. Land and terminals are treated only as boundaries/obstacles.",
        "width": WIDTH,
        "height": HEIGHT,
        "cell_size_m": 100,
        "depot": [45, 5],
        "free_cells": free_cells,
        "obstacles": sorted([list(cell) for cell in obstacles]),
        "risk_grid": risk_grid,
        "metadata": {
            "scenario_type": "port_water_surface_inspection",
            "prototype": True,
            "note": "Research prototype, not a nautical chart; ground-side inspection is excluded.",
            "water_only_scope": True,
        },
    }
    tasks = {
        "point_tasks": [
            {"id": "P01", "type": "buoy_or_marker_check", "cell": [32, 18], "risk": 3, "service_time": 3, "allowed_platforms": ["UAV", "USV"]},
            {"id": "P02", "type": "berth_front_key_point", "cell": [19, 34], "risk": 3, "service_time": 4, "allowed_platforms": ["UAV", "USV"]},
            {"id": "P03", "type": "suspected_floating_object", "cell": [25, 25], "risk": 3, "service_time": 3, "allowed_platforms": ["UAV", "USV"]},
            {"id": "P04", "type": "suspected_pollution_spot", "cell": [37, 31], "risk": 2, "service_time": 4, "allowed_platforms": ["UAV", "USV"]},
            {"id": "P05", "type": "abnormal_vessel_stay_point", "cell": [12, 22], "risk": 2, "service_time": 3, "allowed_platforms": ["UAV", "USV"]},
            {"id": "P06", "type": "anchorage_watch_point", "cell": [42, 38], "risk": 2, "service_time": 3, "allowed_platforms": ["UAV", "USV"]},
        ],
        "line_tasks": [
            {"id": "L01", "type": "main_channel_patrol", "cells": [[44, 8], [40, 12], [36, 16], [32, 20], [28, 24], [24, 28], [20, 32], [16, 36], [12, 40]], "risk": 3, "service_time": 12, "allowed_platforms": ["UAV", "USV"]},
            {"id": "L02", "type": "berth_front_line_patrol", "cells": [[17, 30], [18, 32], [19, 34], [20, 36], [21, 38], [22, 40]], "risk": 3, "service_time": 8, "allowed_platforms": ["UAV", "USV"]},
            {"id": "L03", "type": "breakwater_nearshore_patrol", "cells": [[8, 12], [9, 15], [10, 18], [11, 21], [12, 24]], "risk": 2, "service_time": 7, "allowed_platforms": ["UAV", "USV"]},
        ],
        "area_tasks": [
            {"id": "A01", "type": "harbor_basin_coverage", "cells": _rect_cells(25, 34, 11, 20, obstacles), "risk": 2, "service_time": 16, "allowed_platforms": ["UAV", "USV"], "executor": "mcpp_or_boustrophedon"},
            {"id": "A02", "type": "berth_front_high_risk_water_coverage", "cells": _rect_cells(16, 24, 31, 41, obstacles), "risk": 3, "service_time": 18, "allowed_platforms": ["UAV", "USV"], "executor": "mcpp_or_boustrophedon"},
            {"id": "A03", "type": "anchorage_waiting_area_coverage", "cells": _rect_cells(38, 45, 34, 44, obstacles), "risk": 2, "service_time": 14, "allowed_platforms": ["UAV", "USV"], "executor": "mcpp_or_boustrophedon"},
        ],
    }
    (root / "dalian_port_v0_grid.json").write_text(json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "dalian_port_v0_tasks.json").write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")


def _obstacles() -> set[tuple[int, int]]:
    obstacles: set[tuple[int, int]] = set()
    for row in range(HEIGHT):
        for col in range(WIDTH):
            if row in {0, HEIGHT - 1} or col in {0, WIDTH - 1}:
                obstacles.add((row, col))
            if row < 8 and col < 28:
                obstacles.add((row, col))
            if row < 14 and col > 43:
                obstacles.add((row, col))
            if 15 <= row <= 24 and 42 <= col <= 48:
                obstacles.add((row, col))
            if 1 <= row <= 12 and 1 <= col <= 6:
                obstacles.add((row, col))
            if 8 <= row <= 20 and 1 <= col <= 4:
                obstacles.add((row, col))
            if 21 <= row <= 29 and 1 <= col <= 3:
                obstacles.add((row, col))
            if 13 <= row <= 16 and 27 <= col <= 31:
                obstacles.add((row, col))
            if 24 <= row <= 29 and 36 <= col <= 39:
                obstacles.add((row, col))
    for row in range(8, 13):
        for col in range(28, 36):
            if col - row > 20:
                obstacles.add((row, col))
    return obstacles


def _risk_grid(obstacles: set[tuple[int, int]]) -> list[list[int]]:
    grid: list[list[int]] = []
    for row in range(HEIGHT):
        values: list[int] = []
        for col in range(WIDTH):
            risk = 0
            if (row, col) not in obstacles:
                risk = 1
                if _near_line(row, col, [(44, 8), (32, 20), (20, 32), (12, 40)], tolerance=3):
                    risk = 3
                elif 15 <= row <= 25 and 29 <= col <= 42:
                    risk = 3
                elif 36 <= row <= 46 and 32 <= col <= 45:
                    risk = 2
                elif 23 <= row <= 35 and 10 <= col <= 24:
                    risk = 2
            values.append(risk)
        grid.append(values)
    return grid


def _near_line(row: int, col: int, points: list[tuple[int, int]], tolerance: int) -> bool:
    return any(abs(row - pr) + abs(col - pc) <= tolerance for pr, pc in points)


def _rect_cells(row0: int, row1: int, col0: int, col1: int, obstacles: set[tuple[int, int]]) -> list[list[int]]:
    return [
        [row, col]
        for row in range(row0, row1 + 1)
        for col in range(col0, col1 + 1)
        if (row, col) not in obstacles
    ]


if __name__ == "__main__":
    main()
