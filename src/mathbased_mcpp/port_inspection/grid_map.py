from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import GridCell, PortGridMap


def load_port_grid(path: str | Path) -> PortGridMap:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    grid = PortGridMap(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        width=int(data["width"]),
        height=int(data["height"]),
        cell_size_m=float(data["cell_size_m"]),
        depot=_cell(data["depot"]),
        free_cells=tuple(_cell(item) for item in data["free_cells"]),
        obstacles=tuple(_cell(item) for item in data["obstacles"]),
        risk_grid=tuple(tuple(int(value) for value in row) for row in data["risk_grid"]),
        metadata=dict(data.get("metadata", {})),
    )
    _validate_grid(grid)
    return grid


def _cell(value: Any) -> GridCell:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"expected [row, col], got {value!r}")
    return int(value[0]), int(value[1])


def _validate_grid(grid: PortGridMap) -> None:
    if grid.width <= 0 or grid.height <= 0:
        raise ValueError("port grid width and height must be positive")
    if len(grid.risk_grid) != grid.height:
        raise ValueError("risk_grid row count does not match height")
    for row in grid.risk_grid:
        if len(row) != grid.width:
            raise ValueError("risk_grid column count does not match width")
        if any(value < 0 or value > 3 for value in row):
            raise ValueError("risk_grid values must be in [0, 3]")

    free = set(grid.free_cells)
    obstacles = set(grid.obstacles)
    if free & obstacles:
        raise ValueError("free_cells and obstacles overlap")
    for cell in free | obstacles:
        if not grid.in_bounds(cell):
            raise ValueError(f"cell out of bounds: {cell}")
    if grid.depot not in free:
        raise ValueError(f"depot must be a free water cell: {grid.depot}")
