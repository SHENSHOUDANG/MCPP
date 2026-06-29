from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCENARIO_NAME = "los_angeles_training_v1"
DEFAULT_OUTPUT_DIR = Path("data/ports") / SCENARIO_NAME
CELL_SIZE_M = 250.0
LAT0 = 33.735
LON0 = -118.255


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact Los Angeles port scheduler training scenario.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = _management_objects()
    bounds = _bounds(objects)
    grid, tasks = _build_scenario(objects, bounds)

    grid_path = output_dir / f"{SCENARIO_NAME}_grid.json"
    tasks_path = output_dir / f"{SCENARIO_NAME}_tasks.json"
    readme_path = output_dir / "README.md"
    grid_path.write_text(json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8")
    tasks_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(_readme(len(tasks["point_tasks"]), len(tasks["line_tasks"]), len(tasks["area_tasks"])), encoding="utf-8")
    print(f"grid={grid_path}")
    print(f"tasks={tasks_path}")
    print(f"readme={readme_path}")


def _management_objects() -> list[dict[str, Any]]:
    return [
        _line(
            "LA-CH-MAIN",
            "Main Channel hydrographic survey",
            "HYDROGRAPHIC_SURVEY",
            "channel",
            [(-118.246, 33.706), (-118.248, 33.723), (-118.252, 33.740), (-118.262, 33.754)],
            risk=3,
            deadline=300,
            service_time=8,
        ),
        _line(
            "LA-CH-CABRILLO",
            "Cabrillo approach corridor survey",
            "HYDROGRAPHIC_SURVEY",
            "approach_channel",
            [(-118.292, 33.710), (-118.278, 33.719), (-118.260, 33.728)],
            risk=3,
            deadline=300,
            service_time=7,
        ),
        _line(
            "LA-BERTH-PIER400-W",
            "Pier 400 west berth face inspection",
            "WATERSIDE_ASSET_INSPECTION",
            "berth_face",
            [(-118.271, 33.731), (-118.262, 33.735), (-118.253, 33.738)],
            risk=2,
            deadline=480,
            service_time=6,
        ),
        _line(
            "LA-BERTH-TERMINAL-ISLAND",
            "Terminal Island waterside berth inspection",
            "WATERSIDE_ASSET_INSPECTION",
            "berth_face",
            [(-118.276, 33.746), (-118.266, 33.751), (-118.255, 33.756)],
            risk=2,
            deadline=480,
            service_time=6,
        ),
        _line(
            "LA-BREAKWATER-SAN-PEDRO",
            "San Pedro breakwater waterside inspection",
            "WATERSIDE_ASSET_INSPECTION",
            "breakwater",
            [(-118.286, 33.708), (-118.273, 33.711), (-118.257, 33.713)],
            risk=2,
            deadline=540,
            service_time=7,
        ),
        _line(
            "LA-BRIDGE-VINCENT-THOMAS",
            "Vincent Thomas Bridge pier waterline inspection",
            "WATERSIDE_ASSET_INSPECTION",
            "bridge_pier_waterline",
            [(-118.277, 33.748), (-118.268, 33.750), (-118.258, 33.752)],
            risk=3,
            deadline=360,
            service_time=5,
        ),
        _area(
            "LA-AREA-PIER400-BASIN",
            "Pier 400 basin surface safety patrol",
            "SURFACE_SAFETY_PATROL",
            "basin",
            [(-118.274, 33.725), (-118.250, 33.725), (-118.250, 33.742), (-118.274, 33.742)],
            risk=2,
            deadline=420,
            service_time=9,
        ),
        _area(
            "LA-AREA-WEST-BASIN",
            "West Basin visible obstruction patrol",
            "SURFACE_SAFETY_PATROL",
            "basin",
            [(-118.285, 33.735), (-118.266, 33.735), (-118.266, 33.753), (-118.285, 33.753)],
            risk=2,
            deadline=420,
            service_time=8,
        ),
        _area(
            "LA-AREA-TURNING-BASIN",
            "Turning basin hydrographic check",
            "HYDROGRAPHIC_SURVEY",
            "turning_basin",
            [(-118.263, 33.739), (-118.244, 33.739), (-118.244, 33.756), (-118.263, 33.756)],
            risk=3,
            deadline=360,
            service_time=10,
        ),
        _area(
            "LA-AREA-FISH-HARBOR",
            "Fish Harbor surface condition patrol",
            "SURFACE_SAFETY_PATROL",
            "harbor_basin",
            [(-118.280, 33.716), (-118.260, 33.716), (-118.260, 33.728), (-118.280, 33.728)],
            risk=1,
            deadline=600,
            service_time=6,
        ),
        _point(
            "LA-ATON-ANGELS-GATE",
            "Angels Gate navigation aid inspection",
            "WATERSIDE_ASSET_INSPECTION",
            "navigation_aid",
            -118.248,
            33.706,
            risk=3,
            deadline=300,
            service_time=2,
        ),
        _point(
            "LA-ATON-QUEENS-GATE",
            "Queens Gate navigation aid inspection",
            "WATERSIDE_ASSET_INSPECTION",
            "navigation_aid",
            -118.286,
            33.711,
            risk=3,
            deadline=300,
            service_time=2,
        ),
        _point(
            "LA-ATON-MAIN-CHANNEL-1",
            "Main Channel buoy inspection 1",
            "WATERSIDE_ASSET_INSPECTION",
            "navigation_aid",
            -118.253,
            33.727,
            risk=2,
            deadline=480,
            service_time=2,
        ),
        _point(
            "LA-ATON-MAIN-CHANNEL-2",
            "Main Channel buoy inspection 2",
            "WATERSIDE_ASSET_INSPECTION",
            "navigation_aid",
            -118.258,
            33.742,
            risk=2,
            deadline=480,
            service_time=2,
        ),
        _point(
            "LA-ASSET-PIER400-FENDER",
            "Pier 400 fender cluster inspection",
            "WATERSIDE_ASSET_INSPECTION",
            "fender_cluster",
            -118.257,
            33.732,
            risk=2,
            deadline=480,
            service_time=3,
        ),
        _point(
            "LA-ASSET-VTS-WATERLINE",
            "Inner harbor waterline asset inspection",
            "WATERSIDE_ASSET_INSPECTION",
            "waterside_asset",
            -118.268,
            33.758,
            risk=2,
            deadline=540,
            service_time=3,
        ),
        _point(
            "LA-EVENT-FLOATING-DEBRIS-A",
            "Scenario floating debris response A",
            "SURFACE_SAFETY_PATROL",
            "floating_debris",
            -118.262,
            33.734,
            risk=3,
            deadline=180,
            service_time=2,
            release_mode="EVENT",
        ),
        _point(
            "LA-EVENT-SURFACE-SHEEN-A",
            "Scenario visible surface sheen response A",
            "SURFACE_SAFETY_PATROL",
            "surface_sheen",
            -118.274,
            33.744,
            risk=3,
            deadline=180,
            service_time=2,
            release_mode="EVENT",
        ),
    ]


def _build_scenario(objects: list[dict[str, Any]], bounds: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    width = int(math.ceil(bounds["width_m"] / CELL_SIZE_M)) + 1
    height = int(math.ceil(bounds["height_m"] / CELL_SIZE_M)) + 1
    depot = _to_cell(-118.270, 33.742, bounds)
    free_cells = [[row, col] for row in range(height) for col in range(width)]
    risk_grid = [[0 for _ in range(width)] for _ in range(height)]
    tasks = {"metadata": _scenario_metadata(bounds, width, height), "point_tasks": [], "line_tasks": [], "area_tasks": []}

    for obj in objects:
        if obj["geometry"] == "point":
            cells = [_to_cell(obj["lon"], obj["lat"], bounds)]
        elif obj["geometry"] == "line":
            cells = _line_cells(obj["coordinates"], bounds)
        else:
            cells = _area_cells(obj["coordinates"], bounds)
        for row, col in cells:
            risk_grid[row][col] = max(risk_grid[row][col], int(obj["risk"]))
        task = _task(obj, cells)
        if obj["geometry"] == "point":
            tasks["point_tasks"].append(task)
        elif obj["geometry"] == "line":
            tasks["line_tasks"].append(task)
        else:
            tasks["area_tasks"].append(task)

    grid = {
        "name": SCENARIO_NAME,
        "description": "Compact coordinate-grid Los Angeles port scheduler training scenario.",
        "width": width,
        "height": height,
        "cell_size_m": CELL_SIZE_M,
        "depot": list(depot),
        "free_cells": free_cells,
        "obstacles": [],
        "risk_grid": risk_grid,
        "metadata": {
            **tasks["metadata"],
            "distance_mode": "utm_euclidean",
            "depot_note": "Engineering shoreline depot near Terminal Island; replace with approved recovery points before final experiments.",
        },
    }
    return grid, tasks


def _task(obj: dict[str, Any], cells: list[tuple[int, int]]) -> dict[str, Any]:
    service_time = int(obj["service_time"])
    geometry = obj["geometry"]
    task = {
        "id": obj["id"],
        "type": obj["object_type"],
        "risk": int(obj["risk"]),
        "service_time": service_time,
        "screening_workload": max(1.0, round(service_time * 0.6, 2)),
        "review_workload": max(1.0, round(service_time * 1.25, 2)),
        "max_interval": int(obj["deadline"]),
        "deadline": int(obj["deadline"]),
        "allowed_platforms": ["UAV", "USV"],
        "coverage_threshold": 0.85 if geometry == "area" else 1.0,
        "priority": float(obj["risk"]),
        "metadata": {
            "task_family": obj["task_family"],
            "geometry_mode": {"point": "TARGET", "line": "CORRIDOR", "area": "AREA"}[geometry],
            "release_mode": obj["release_mode"],
            "scenario_generated": True,
            "parameter_status": "engineering_training_scenario",
            "object_name": obj["name"],
            "source_dataset": "Los Angeles port engineering seed objects",
            "source_kind": "scenario_seed_not_final_official_dataset",
            "verification_level": "engineering_seed",
            "longitude": obj.get("lon"),
            "latitude": obj.get("lat"),
        },
    }
    if geometry == "point":
        task["cell"] = list(cells[0])
    else:
        task["cells"] = [list(cell) for cell in cells]
    return task


def _point(
    object_id: str,
    name: str,
    task_family: str,
    object_type: str,
    lon: float,
    lat: float,
    *,
    risk: int,
    deadline: int,
    service_time: int,
    release_mode: str = "SCHEDULED",
) -> dict[str, Any]:
    return {
        "id": object_id,
        "name": name,
        "task_family": task_family,
        "object_type": object_type,
        "geometry": "point",
        "lon": lon,
        "lat": lat,
        "risk": risk,
        "deadline": deadline,
        "service_time": service_time,
        "release_mode": release_mode,
    }


def _line(
    object_id: str,
    name: str,
    task_family: str,
    object_type: str,
    coordinates: list[tuple[float, float]],
    *,
    risk: int,
    deadline: int,
    service_time: int,
    release_mode: str = "SCHEDULED",
) -> dict[str, Any]:
    return {
        "id": object_id,
        "name": name,
        "task_family": task_family,
        "object_type": object_type,
        "geometry": "line",
        "coordinates": coordinates,
        "risk": risk,
        "deadline": deadline,
        "service_time": service_time,
        "release_mode": release_mode,
    }


def _area(
    object_id: str,
    name: str,
    task_family: str,
    object_type: str,
    coordinates: list[tuple[float, float]],
    *,
    risk: int,
    deadline: int,
    service_time: int,
    release_mode: str = "SCHEDULED",
) -> dict[str, Any]:
    return {
        "id": object_id,
        "name": name,
        "task_family": task_family,
        "object_type": object_type,
        "geometry": "area",
        "coordinates": coordinates,
        "risk": risk,
        "deadline": deadline,
        "service_time": service_time,
        "release_mode": release_mode,
    }


def _bounds(objects: list[dict[str, Any]]) -> dict[str, float]:
    coords: list[tuple[float, float]] = [(-118.270, 33.742)]
    for obj in objects:
        if obj["geometry"] == "point":
            coords.append((obj["lon"], obj["lat"]))
        else:
            coords.extend(obj["coordinates"])
    lon_values = [lon for lon, _ in coords]
    lat_values = [lat for _, lat in coords]
    lon_min = min(lon_values) - 0.01
    lon_max = max(lon_values) + 0.01
    lat_min = min(lat_values) - 0.01
    lat_max = max(lat_values) + 0.01
    return {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "width_m": _lon_to_m(lon_max - lon_min),
        "height_m": _lat_to_m(lat_max - lat_min),
    }


def _to_cell(lon: float, lat: float, bounds: dict[str, float]) -> tuple[int, int]:
    col = int(round(_lon_to_m(lon - bounds["lon_min"]) / CELL_SIZE_M))
    row = int(round(_lat_to_m(bounds["lat_max"] - lat) / CELL_SIZE_M))
    return max(row, 0), max(col, 0)


def _line_cells(coordinates: list[tuple[float, float]], bounds: dict[str, float]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    anchors = [_to_cell(lon, lat, bounds) for lon, lat in coordinates]
    for start, end in zip(anchors, anchors[1:]):
        cells.extend(_interpolate_cells(start, end))
    return _dedupe(cells or anchors)


def _area_cells(coordinates: list[tuple[float, float]], bounds: dict[str, float]) -> list[tuple[int, int]]:
    anchors = [_to_cell(lon, lat, bounds) for lon, lat in coordinates]
    rows = [row for row, _ in anchors]
    cols = [col for _, col in anchors]
    row_min, row_max = min(rows), max(rows)
    col_min, col_max = min(cols), max(cols)
    cells = [
        (row, col)
        for row in range(row_min, row_max + 1, 2)
        for col in range(col_min, col_max + 1, 2)
    ]
    return cells or anchors


def _interpolate_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    row0, col0 = start
    row1, col1 = end
    steps = max(abs(row1 - row0), abs(col1 - col0), 1)
    return [
        (round(row0 + (row1 - row0) * step / steps), round(col0 + (col1 - col0) * step / steps))
        for step in range(steps + 1)
    ]


def _dedupe(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for cell in cells:
        if cell in seen:
            continue
        seen.add(cell)
        result.append(cell)
    return result


def _lon_to_m(delta_lon: float) -> float:
    return delta_lon * 111_320.0 * math.cos(math.radians(LAT0))


def _lat_to_m(delta_lat: float) -> float:
    return delta_lat * 110_540.0


def _scenario_metadata(bounds: dict[str, float], width: int, height: int) -> dict[str, Any]:
    return {
        "scenario_name": SCENARIO_NAME,
        "port": "Los Angeles",
        "contract_status": "PENDING_ENGINEERING_TRAINING",
        "scenario_generated": True,
        "cell_size_m": CELL_SIZE_M,
        "bounds_lon_lat": {
            "lon_min": bounds["lon_min"],
            "lon_max": bounds["lon_max"],
            "lat_min": bounds["lat_min"],
            "lat_max": bounds["lat_max"],
        },
        "grid_shape": [height, width],
        "note": "Compact scenario seed for making LA-port scheduler training runnable; replace with verified official GIS layers before final experiments.",
    }


def _readme(point_count: int, line_count: int, area_count: int) -> str:
    return f"""# {SCENARIO_NAME}

This is a compact Los Angeles port scheduler training scenario.

Status: `PENDING_ENGINEERING_TRAINING`.

It is designed to make the LA-port training command runnable today while V1.2 item 9 and the final official GIS workflow remain unfrozen. The task objects are named after plausible LA port management objects and are marked as scenario-generated engineering seeds. They are not final official work orders and must not be reported as final experiment evidence.

- Point tasks: {point_count}
- Corridor tasks: {line_count}
- Area tasks: {area_count}
- Coordinate mode: local equirectangular approximation, `distance_mode=utm_euclidean`
- Cell size: {CELL_SIZE_M:.0f} m

Run a smoke check:

```powershell
.\\.venv\\Scripts\\python.exe tools\\check_port_inspection_env.py --config configs\\port_los_angeles_training_v1.toml --steps 2
```

Run scheduler training:

```powershell
.\\.venv\\Scripts\\python.exe tools\\train_port_scheduler_rl.py --config configs\\port_los_angeles_training_v1.toml --steps 10000
```
"""


if __name__ == "__main__":
    main()
