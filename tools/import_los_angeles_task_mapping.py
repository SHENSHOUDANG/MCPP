from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCENARIO_NAME = "los_angeles_training_v1"
DEFAULT_SOURCE_DIR = Path("D:/地图/洛杉矶")
DEFAULT_OUTPUT_DIR = Path("data/ports") / SCENARIO_NAME
CELL_SIZE_M = 250.0
LAT0 = 33.735
LON0 = -118.255
SCRIPT_VERSION = "chart-aligned-task-mapping-v1"
ACCESS_DATE = "2026-06-30"
COORD_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import user-provided Los Angeles task mapping CSVs.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    task_rows = _read_csv(source_dir / "task_catalog_v2_0.csv")
    reinspection_rows = _read_csv(source_dir / "reinspection_catalog_v2_0.csv")
    qa_rows = _read_csv(source_dir / "geometry_alignment_qa_v2_0.csv")
    source_files = _source_file_metadata(source_dir)

    objects = [_object_from_row(row, source_files["task_catalog_v2_0.csv"]["sha256"]) for row in task_rows]
    bounds = _bounds(objects)
    grid, tasks = _build_outputs(objects, bounds, source_files, reinspection_rows, qa_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / f"{SCENARIO_NAME}_grid.json"
    tasks_path = output_dir / f"{SCENARIO_NAME}_tasks.json"
    readme_path = output_dir / "README.md"
    grid_path.write_text(json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8")
    tasks_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(
        _readme(
            point_count=len(tasks["point_tasks"]),
            line_count=len(tasks["line_tasks"]),
            area_count=len(tasks["area_tasks"]),
            reinspection_count=len(tasks["reinspection_tasks"]),
            task_type_counts=dict(Counter(obj["task_type"] for obj in objects)),
        ),
        encoding="utf-8",
    )
    print(f"grid={grid_path}")
    print(f"tasks={tasks_path}")
    print(f"tasks_point={len(tasks['point_tasks'])} tasks_line={len(tasks['line_tasks'])} tasks_area={len(tasks['area_tasks'])}")
    print(f"reinspection_tasks={len(tasks['reinspection_tasks'])}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _source_file_metadata(source_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in (
        "task_catalog_v2_0.csv",
        "reinspection_catalog_v2_0.csv",
        "geometry_alignment_qa_v2_0.csv",
        "port_of_los_angeles_task_mapping_v2_0.gpkg",
    ):
        path = source_dir / name
        if not path.exists():
            continue
        result[name] = {
            "filename": name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return result


def _object_from_row(row: dict[str, str], task_catalog_sha256: str) -> dict[str, Any]:
    coordinates = _parse_wkt_coordinates(row["geometry_wkt"])
    geometry = _geometry_from_row(row)
    quality_requirement = _parse_json_field(row.get("quality_requirement", "{}"))
    hard_capability = _parse_json_field(row.get("hard_capability_vector", "{}"))
    return {
        "id": row["task_id"],
        "object_id": row["object_id"],
        "name": row["object_name"],
        "task_type": row["task_type"],
        "task_name_zh": row["task_name_zh"],
        "geometry": geometry,
        "geometry_role": row["geometry_role"],
        "wkt_type": row["geometry_wkt"].split(" ", 1)[0],
        "coordinates": coordinates,
        "importance_class": row["importance_class"],
        "risk": _risk_from_importance(row["importance_class"]),
        "allowed_platforms": _allowed_platforms(row.get("allowed_platform_types", "")),
        "max_interval": _int_or_none(row.get("max_revisit_interval")) or _int_or_none(row.get("period_interval_min")) or 300,
        "period_interval": _int_or_none(row.get("period_interval_min")),
        "deadline": _float_or_none(row.get("deadline")),
        "quality_requirement": quality_requirement,
        "hard_capability_vector": hard_capability,
        "metadata": {
            "task_family": _task_family(row["task_type"]),
            "geometry_mode": {"point": "TARGET", "line": "CORRIDOR", "area": "AREA"}[geometry],
            "release_mode": "PERIODIC",
            "source_release_mode": row["release_mode"],
            "scenario_generated": True,
            "geometry_source_status": "chart_aligned_research_geometry",
            "work_order_status": "training_task_from_user_provided_catalog_not_official_work_order",
            "parameter_status": row["parameter_status"],
            "object_id": row["object_id"],
            "object_name": row["object_name"],
            "task_name_zh": row["task_name_zh"],
            "importance_class": row["importance_class"],
            "obligation_level": row["obligation_level"],
            "period_interval_min": _int_or_none(row.get("period_interval_min")),
            "calendar_anchor": row.get("calendar_anchor") or None,
            "calendar_update_mode": row.get("calendar_update_mode") or None,
            "deadline": _float_or_none(row.get("deadline")),
            "max_revisit_interval": _int_or_none(row.get("max_revisit_interval")),
            "last_completion_time": _float_or_none(row.get("last_completion_time")),
            "next_due_time": _float_or_none(row.get("next_due_time")),
            "hard_capability_vector": hard_capability,
            "inspection_scope": row["inspection_scope"],
            "data_product": row["data_product"],
            "management_use": row["management_use"],
            "quality_requirement": quality_requirement,
            "quality_acceptance_ref": row["acceptance_function_id"],
            "completion_rule": row["completion_rule"],
            "source_ids": row["source_ids"].split("|") if row.get("source_ids") else [],
            "source_dataset": "Port of Los Angeles Task Mapping V2.0",
            "source_agency": "User-provided chart-aligned package based on NOAA and Port of Los Angeles public data",
            "source_date": ACCESS_DATE,
            "source_url": "local:D:/地图/洛杉矶",
            "source_urls": [
                "https://www.charts.noaa.gov/ENCs/CA_ENCs.zip",
                "https://data.lacity.org/resource/9r7y-tdse.geojson?$limit=5000",
            ],
            "source_version_or_edition": row["version"],
            "access_date": ACCESS_DATE,
            "license_or_usage_terms": "Research task mapping package derived from public chart/port data; not for navigation.",
            "original_id": row["task_id"],
            "original_crs": row["crs"],
            "file_checksum": task_catalog_sha256,
            "processing_script_version": SCRIPT_VERSION,
            "processing_note": (
                "Imported from user-provided chart-aligned Los Angeles task catalog. Geometry was converted from WKT "
                "to the existing 250 m local training grid. Schedule parameters remain research/training inputs."
            ),
            "geometry_source_type": row["geometry_source_type"],
            "geometry_accuracy_class": row["geometry_accuracy_class"],
            "geometry_wkt": row["geometry_wkt"],
            "centroid_lon_lat": list(_centroid(coordinates)),
        },
    }


def _build_outputs(
    objects: list[dict[str, Any]],
    bounds: dict[str, float],
    source_files: dict[str, dict[str, Any]],
    reinspection_rows: list[dict[str, str]],
    qa_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    width = int(math.ceil(bounds["width_m"] / CELL_SIZE_M)) + 1
    height = int(math.ceil(bounds["height_m"] / CELL_SIZE_M)) + 1
    free_cells = [[row, col] for row in range(height) for col in range(width)]
    risk_grid = [[0 for _ in range(width)] for _ in range(height)]
    tasks: dict[str, Any] = {
        "metadata": _scenario_metadata(bounds, width, height, source_files, objects, qa_rows),
        "point_tasks": [],
        "line_tasks": [],
        "area_tasks": [],
        "reinspection_tasks": [_reinspection_metadata(row) for row in reinspection_rows],
    }
    for obj in objects:
        cells = _cells_for_object(obj, bounds)
        if obj["geometry"] in {"line", "area"} and len(cells) < 2:
            raise ValueError(f"{obj['id']} converted to too few cells: {cells}")
        for row, col in cells:
            risk_grid[row][col] = max(risk_grid[row][col], int(obj["risk"]))
        task = _task_payload(obj, cells)
        if obj["geometry"] == "point":
            tasks["point_tasks"].append(task)
        elif obj["geometry"] == "line":
            tasks["line_tasks"].append(task)
        else:
            tasks["area_tasks"].append(task)
    depot = _derived_depot_cell(objects, bounds)
    grid = {
        "name": SCENARIO_NAME,
        "description": "Los Angeles port scheduler training scenario from user-provided chart-aligned task mapping V2.0.",
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
            "depot_note": "Training depot is derived from the provided task geometry centroids; final experiments still require approved recovery points.",
        },
    }
    return grid, tasks


def _scenario_metadata(
    bounds: dict[str, float],
    width: int,
    height: int,
    source_files: dict[str, dict[str, Any]],
    objects: list[dict[str, Any]],
    qa_rows: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "scenario_name": SCENARIO_NAME,
        "port": "Los Angeles",
        "contract_status": "PENDING_CHART_ALIGNED_TASK_MAPPING_TRAINING",
        "scenario_generated": True,
        "official_geometry": False,
        "cell_size_m": CELL_SIZE_M,
        "access_date": ACCESS_DATE,
        "source_package": "D:/地图/洛杉矶",
        "source_files": source_files,
        "bounds_lon_lat": {
            "lon_min": bounds["lon_min"],
            "lon_max": bounds["lon_max"],
            "lat_min": bounds["lat_min"],
            "lat_max": bounds["lat_max"],
        },
        "grid_shape": [height, width],
        "task_type_counts": dict(Counter(obj["task_type"] for obj in objects)),
        "geometry_counts": dict(Counter(obj["geometry"] for obj in objects)),
        "qa_summary": dict(Counter(row.get("qa_status", "") for row in qa_rows)),
        "note": (
            "Task geometry is imported from the user-provided Port of Los Angeles Task Mapping V2.0 package. "
            "The package is chart-aligned research geometry, not native ENC vector geometry, and remains PENDING training data."
        ),
    }


def _task_payload(obj: dict[str, Any], cells: list[tuple[int, int]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": obj["id"],
        "type": obj["task_type"],
        "risk": int(obj["risk"]),
        "max_interval": int(obj["max_interval"]),
        "allowed_platforms": list(obj["allowed_platforms"]),
        "coverage_threshold": 0.9 if obj["geometry"] == "point" else 0.95,
        "priority": float(obj["risk"]),
        "metadata": dict(obj["metadata"]),
    }
    if obj["geometry"] == "point":
        payload["cell"] = list(cells[0])
    else:
        payload["cells"] = [list(cell) for cell in cells]
    return payload


def _reinspection_metadata(row: dict[str, str]) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "task_role": row["task_role"],
        "parent_task_id": row["parent_task_id"],
        "base_task_type": row["base_task_type"],
        "task_name_zh": row["task_name_zh"],
        "trigger_type": row["trigger_type"],
        "release_mode": row["release_mode"],
        "obligation_level": row["obligation_level"],
        "status": row["status"],
        "is_static_baseline": row["is_static_baseline"] == "1",
        "scenario_note": row["scenario_note"],
        "version": row["version"],
        "geometry_wkt": row["geometry_wkt"],
        "note": "Stored as metadata only; not released into the current scheduler training task set.",
    }


def _geometry_from_row(row: dict[str, str]) -> str:
    wkt_type = row["geometry_wkt"].split(" ", 1)[0].upper()
    if wkt_type == "POINT":
        return "point"
    if wkt_type == "LINESTRING":
        return "line"
    if wkt_type in {"POLYGON", "MULTIPOLYGON"}:
        return "area"
    raise ValueError(f"unsupported WKT type for {row['task_id']}: {wkt_type}")


def _parse_wkt_coordinates(wkt: str) -> list[tuple[float, float]]:
    coords = [(float(lon), float(lat)) for lon, lat in COORD_RE.findall(wkt)]
    if not coords:
        raise ValueError(f"no coordinates found in WKT: {wkt[:80]}")
    return _dedupe_coords(coords)


def _risk_from_importance(value: str) -> int:
    return {"A": 3, "B": 2, "C": 1}.get(value.upper(), 1)


def _allowed_platforms(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip().upper() for item in value.split("|") if item.strip())
    return parsed or ("UAV", "USV")


def _task_family(task_type: str) -> str:
    if task_type == "CHANNEL_INSPECTION":
        return "HYDROGRAPHIC_SURVEY"
    if task_type == "ANCHORAGE_INSPECTION":
        return "SURFACE_SAFETY_PATROL"
    if task_type in {"BUOY_INSPECTION", "BERTH_AREA_INSPECTION"}:
        return "WATERSIDE_ASSET_INSPECTION"
    raise ValueError(f"unsupported task type: {task_type}")


def _cells_for_object(obj: dict[str, Any], bounds: dict[str, float]) -> list[tuple[int, int]]:
    coordinates = obj["coordinates"]
    if obj["geometry"] == "point":
        return [_to_cell(*coordinates[0], bounds)]
    if obj["geometry"] == "line":
        return _line_cells(coordinates, bounds)
    return _area_cells(coordinates, bounds)


def _bounds(objects: list[dict[str, Any]]) -> dict[str, float]:
    coords: list[tuple[float, float]] = []
    for obj in objects:
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


def _derived_depot_cell(objects: list[dict[str, Any]], bounds: dict[str, float]) -> tuple[int, int]:
    berth_objects = [obj for obj in objects if obj["task_type"] == "BERTH_AREA_INSPECTION"]
    source = berth_objects[0] if berth_objects else min(objects, key=lambda obj: _distance_m(_centroid(obj["coordinates"]), (LON0, LAT0)))
    lon, lat = _centroid(source["coordinates"])
    return _to_cell(lon, lat, bounds)


def _to_cell(lon: float, lat: float, bounds: dict[str, float]) -> tuple[int, int]:
    col = int(round(_lon_to_m(lon - bounds["lon_min"]) / CELL_SIZE_M))
    row = int(round(_lat_to_m(bounds["lat_max"] - lat) / CELL_SIZE_M))
    return max(row, 0), max(col, 0)


def _line_cells(coordinates: list[tuple[float, float]], bounds: dict[str, float]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    anchors = [_to_cell(lon, lat, bounds) for lon, lat in coordinates]
    for start, end in zip(anchors, anchors[1:]):
        cells.extend(_interpolate_cells(start, end))
    return _dedupe_cells(cells or anchors)


def _area_cells(coordinates: list[tuple[float, float]], bounds: dict[str, float]) -> list[tuple[int, int]]:
    anchors = [_to_cell(lon, lat, bounds) for lon, lat in coordinates]
    rows = [row for row, _ in anchors]
    cols = [col for _, col in anchors]
    row_min, row_max = min(rows), max(rows)
    col_min, col_max = min(cols), max(cols)
    cells = [(row, col) for row in range(row_min, row_max + 1, 2) for col in range(col_min, col_max + 1, 2)]
    return cells or anchors


def _interpolate_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    row0, col0 = start
    row1, col1 = end
    steps = max(abs(row1 - row0), abs(col1 - col0), 1)
    return [
        (round(row0 + (row1 - row0) * step / steps), round(col0 + (col1 - col0) * step / steps))
        for step in range(steps + 1)
    ]


def _dedupe_cells(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for cell in cells:
        if cell in seen:
            continue
        seen.add(cell)
        result.append(cell)
    return result


def _dedupe_coords(coordinates: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    seen: set[tuple[float, float]] = set()
    result: list[tuple[float, float]] = []
    for lon, lat in coordinates:
        key = (round(lon, 7), round(lat, 7))
        if key in seen:
            continue
        seen.add(key)
        result.append((lon, lat))
    return result


def _centroid(coordinates: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(lon for lon, _ in coordinates) / len(coordinates),
        sum(lat for _, lat in coordinates) / len(coordinates),
    )


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(_lon_to_m(a[0] - b[0]), _lat_to_m(a[1] - b[1]))


def _lon_to_m(delta_lon: float) -> float:
    return delta_lon * 111_320.0 * math.cos(math.radians(LAT0))


def _lat_to_m(delta_lat: float) -> float:
    return delta_lat * 110_540.0


def _int_or_none(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def _float_or_none(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _parse_json_field(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readme(
    point_count: int,
    line_count: int,
    area_count: int,
    reinspection_count: int,
    task_type_counts: dict[str, int],
) -> str:
    task_type_lines = "\n".join(f"- {name}: {count}" for name, count in sorted(task_type_counts.items()))
    return f"""# {SCENARIO_NAME}

This is a Los Angeles port scheduler training scenario imported from the user-provided
`Port_of_Los_Angeles_Task_Mapping_V2.0_Chart_Aligned` package.

Status: `PENDING_CHART_ALIGNED_TASK_MAPPING_TRAINING`.

The checked-in grid and task JSON were generated from `D:/地图/洛杉矶/task_catalog_v2_0.csv`.
The source package describes chart-aligned research geometry validated against NOAA Chart 18751 and
supporting public NOAA/Port of Los Angeles datasets. It is not native ENC vector geometry and must not be
reported as final experiment evidence until the V1.2 algorithm contract and official experiment workflow
are frozen.

- Point tasks: {point_count}
- Corridor tasks: {line_count}
- Area tasks: {area_count}
- Stored reinspection metadata tasks: {reinspection_count}
- Coordinate mode: local equirectangular approximation, `distance_mode=utm_euclidean`
- Cell size: {CELL_SIZE_M:.0f} m

Task type counts:

{task_type_lines}

Regenerate from the provided local task mapping directory:

```powershell
.\\.venv\\Scripts\\python.exe tools\\import_los_angeles_task_mapping.py
```

Run a smoke check:

```powershell
.\\.venv\\Scripts\\python.exe tools\\check_port_inspection_env.py --config configs\\port_los_angeles_training_v1.toml --steps 2
```

Run scheduler training:

```powershell
.\\.venv\\Scripts\\python.exe tools\\run_port_algorithm_comparison.py --config configs\\port_los_angeles_training_v1.toml --steps 50000 --device auto --num-envs 1 --env-workers 1 --checkpoint-interval 10000
```
"""


if __name__ == "__main__":
    main()
