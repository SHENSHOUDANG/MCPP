from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import struct
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCENARIO_NAME = "yangshan_training_v133"
DEFAULT_SOURCE_DIR = Path("D:/map/yangshan2")
DEFAULT_OUTPUT_DIR = Path("data/ports") / SCENARIO_NAME
CELL_SIZE_M = 250.0
SCRIPT_VERSION = "yangshan-v133-training-import-v1"
ACCESS_DATE = "2026-06-30"
DEPOT_LAT = 30.6045
DEPOT_LON = 122.095
DEPOT_TEXT = "30 deg 36.27 min N, 122 deg 5.70 min E"

SOURCE_FILENAMES = (
    "task_catalog_v1_3_3.csv",
    "task_exclusion_policy_v1_3_3.csv",
    "task_asset_point_v1_3_3.geojson",
    "task_surface_area_v1_3_3.geojson",
    "yangshan_stage2.gpkg",
    "yangshan_manual_edit_V1.3.3_NoVirtualAtoN_NoSouthBerthing.qgz",
    "Yangshan_QGIS_Project_V1.3.3_NoVirtualAtoN_NoSouthBerthing.zip",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the user-provided Yangshan V1.3.3 tasks for scheduler training.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cell-size-m", type=float, default=CELL_SIZE_M)
    parser.add_argument("--depot-lat", type=float, default=DEPOT_LAT)
    parser.add_argument("--depot-lon", type=float, default=DEPOT_LON)
    args = parser.parse_args()

    source_dir = args.source_dir
    catalog_path = source_dir / "task_catalog_v1_3_3.csv"
    gpkg_path = source_dir / "yangshan_stage2.gpkg"
    rows = _read_csv(catalog_path)
    active_rows = [row for row in rows if _is_active_training_row(row)]
    geometries = _load_task_geometries(gpkg_path)
    source_files = _source_file_metadata(source_dir)

    objects = [_object_from_row(row, geometries.get(row["task_id"], []), catalog_path) for row in active_rows]
    depot_utm = _wgs84_to_utm_zone51(lon=float(args.depot_lon), lat=float(args.depot_lat))
    bounds = _bounds(objects, depot_utm, float(args.cell_size_m))
    grid, tasks = _build_outputs(objects, bounds, source_files, source_dir, depot_utm, args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / f"{SCENARIO_NAME}_grid.json"
    tasks_path = output_dir / f"{SCENARIO_NAME}_tasks.json"
    summary_path = output_dir / "import_summary.json"
    readme_path = output_dir / "README.md"
    grid_path.write_text(json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8")
    tasks_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = _summary(grid, tasks, source_dir, grid_path, tasks_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(_readme(summary), encoding="utf-8")

    print(f"grid={grid_path}")
    print(f"tasks={tasks_path}")
    print(f"summary={summary_path}")
    print(
        "tasks_point={point} tasks_line={line} tasks_area={area} total={total}".format(
            point=len(tasks["point_tasks"]),
            line=len(tasks["line_tasks"]),
            area=len(tasks["area_tasks"]),
            total=sum(len(tasks[key]) for key in ("point_tasks", "line_tasks", "area_tasks")),
        )
    )
    print(f"grid_shape={grid['height']}x{grid['width']} cell_size_m={grid['cell_size_m']:g}")
    print(f"depot={grid['depot']}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _is_active_training_row(row: dict[str, str]) -> bool:
    if str(row.get("task_eligible", "")).strip() not in {"1", "1.0", "true", "True"}:
        return False
    marker = str(row.get("active_in_v1_3", "")).strip().lower()
    return marker not in {"0", "0.0", "false", "no"}


def _source_file_metadata(source_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in SOURCE_FILENAMES:
        path = source_dir / name
        if not path.exists():
            continue
        result[name] = {
            "filename": name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return result


def _load_task_geometries(gpkg_path: Path) -> dict[str, list[list[tuple[float, float]]]]:
    result: dict[str, list[list[tuple[float, float]]]] = {}
    con = _connect_sqlite_readonly(gpkg_path)
    try:
        tables = [
            str(row[0])
            for row in con.execute("select table_name from gpkg_contents where data_type = 'features' order by table_name")
        ]
        for table in tables:
            columns = {str(row[1]) for row in con.execute(f'pragma table_info("{table}")')}
            if "task_id" not in columns or "geom" not in columns:
                continue
            for task_id, geom_blob in con.execute(f'select task_id, geom from "{table}" where task_id is not null'):
                if geom_blob is None:
                    continue
                paths = _geometry_paths(_parse_gpkg_geometry(geom_blob))
                if paths:
                    result[str(task_id)] = paths
    finally:
        con.close()
    return result


def _connect_sqlite_readonly(path: Path) -> sqlite3.Connection:
    tmp_path = _sqlite_readable_path(path)
    tmp_uri = tmp_path.resolve().as_posix()
    return sqlite3.connect(f"file:{tmp_uri}?mode=ro", uri=True)


def _sqlite_readable_path(path: Path) -> Path:
    tmp_dir = Path(tempfile.gettempdir()) / "codex_yangshan_v133_import"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / path.name
    if not tmp_path.exists() or tmp_path.stat().st_size != path.stat().st_size:
        shutil.copy2(path, tmp_path)
    return tmp_path


def _object_from_row(
    row: dict[str, str],
    lon_lat_paths: list[list[tuple[float, float]]],
    catalog_path: Path,
) -> dict[str, Any]:
    if not lon_lat_paths:
        lon = _float_or_none(row.get("centroid_lon"))
        lat = _float_or_none(row.get("centroid_lat"))
        if lon is None or lat is None:
            raise ValueError(f"task {row.get('task_id', '<unknown>')} has no geometry or centroid")
        lon_lat_paths = [[(lon, lat)]]
    utm_paths = [[_wgs84_to_utm_zone51(lon=lon, lat=lat) for lon, lat in path] for path in lon_lat_paths]
    geometry_mode = str(row["geometry_mode"]).upper()
    geometry = _geometry_from_mode(geometry_mode)
    required_work = _float_or_none(row.get("required_work")) or 1.0
    max_interval = _int_or_none(row.get("max_revisit_interval_min")) or _int_or_none(row.get("period_interval_min")) or 10080
    deadline = _float_or_none(row.get("deadline_min"))
    quality_requirement = _parse_json_field(row.get("quality_requirement", "{}"))
    provenance = _parse_json_field(row.get("provenance", "{}"))
    return {
        "id": row["task_id"],
        "task_type": row.get("task_object_class") or row.get("object_type") or row["task_family"],
        "geometry": geometry,
        "geometry_mode": geometry_mode,
        "lon_lat_paths": lon_lat_paths,
        "utm_paths": utm_paths,
        "risk": _risk_from_importance(row.get("importance_class", "")),
        "importance_class": row.get("importance_class", ""),
        "allowed_platforms": _allowed_platforms(row.get("allowed_platform_types", "")),
        "required_work": float(required_work),
        "completed_work": _float_or_none(row.get("completed_work")) or 0.0,
        "remaining_work": _float_or_none(row.get("remaining_work")) or float(required_work),
        "work_threshold": _float_or_none(row.get("work_threshold")) or 1.0,
        "max_interval": int(max_interval),
        "deadline": deadline,
        "quality_requirement": quality_requirement,
        "quality_acceptance_ref": row.get("quality_acceptance_ref", ""),
        "metadata": {
            "task_family": row["task_family"],
            "geometry_mode": geometry_mode,
            "release_mode": row.get("release_mode") or "PERIODIC",
            "release_time": _float_or_none(row.get("release_time")) or 0.0,
            "importance_class": row.get("importance_class", ""),
            "obligation_level": row.get("obligation_level") or "MANDATORY",
            "period_interval_min": _int_or_none(row.get("period_interval_min")),
            "calendar_anchor": row.get("calendar_anchor") or None,
            "calendar_update_mode": row.get("calendar_update_mode") or None,
            "deadline": deadline,
            "max_revisit_interval": _int_or_none(row.get("max_revisit_interval_min")),
            "last_completion_time": _float_or_none(row.get("last_completion_time")),
            "next_due_time": _float_or_none(row.get("next_due_time")),
            "revisit_initialization_mode": row.get("revisit_initialization_mode") or None,
            "revisit_initialization_time": _float_or_none(row.get("revisit_initialization_time")),
            "parameter_status": row.get("parameter_status", ""),
            "scope_status": row.get("scope_status", ""),
            "task_role": row.get("task_role", ""),
            "parent_object_id": row.get("parent_object_id", ""),
            "object_id": row.get("object_id", ""),
            "object_name": row.get("object_name", ""),
            "object_type": row.get("object_type", ""),
            "source_layer": row.get("source_layer", ""),
            "source_fid": _int_or_none(row.get("source_fid")),
            "source_feature_id": row.get("source_feature_id", ""),
            "source_chart": row.get("source_chart", ""),
            "geometry_ref": row.get("geometry_ref", ""),
            "execution_template_ref": row.get("execution_template_ref", ""),
            "hard_capability_requirement": _parse_json_field(row.get("hard_capability_requirement", "{}")),
            "inspection_scope": row.get("inspection_scope", ""),
            "data_product": row.get("data_product", ""),
            "management_use": row.get("management_use", ""),
            "quality_requirement": quality_requirement,
            "quality_acceptance_ref": row.get("quality_acceptance_ref", ""),
            "trigger_rule": _parse_json_field(row.get("trigger_rule", "{}")),
            "reinspection_enabled": _truthy(row.get("reinspection_enabled", "")),
            "reinspection_template_id": row.get("reinspection_template_id", ""),
            "legacy_task_id": row.get("legacy_task_id", ""),
            "default_depot_id": row.get("default_depot_id", ""),
            "legal_recovery_point_ids_by_platform": _parse_json_field(row.get("legal_recovery_point_ids_by_platform", "{}")),
            "depot_distance_lower_bound_m": _float_or_none(row.get("depot_distance_lower_bound_m")),
            "source_dataset": provenance.get("source_dataset", "Yangshan user-provided manual QGIS task package"),
            "source_agency": provenance.get("source_agency", "User-provided Yangshan manual task package"),
            "source_date": ACCESS_DATE,
            "source_url": f"local:{catalog_path.parent.as_posix()}",
            "source_version_or_edition": "V1.3.3 NoVirtualAtoN NoSouthBerthing",
            "access_date": ACCESS_DATE,
            "license_or_usage_terms": "Research use only; historical Yangshan baseline, not final V1.2 evidence.",
            "original_id": row["task_id"],
            "original_crs": "EPSG:4326",
            "file_checksum": _sha256(catalog_path),
            "processing_script_version": SCRIPT_VERSION,
            "processing_note": (
                "Imported from the user-provided Yangshan V1.3.3 task catalog and GeoPackage. "
                "Geometry is preserved from task layers and snapped to a 250 m scheduler feature grid; "
                "the scenario remains a historical engineering training baseline."
            ),
            "centroid_lon_lat": list(_centroid(_flatten_paths(lon_lat_paths))),
            "source_geometry_point_count": sum(len(path) for path in lon_lat_paths),
        },
    }


def _build_outputs(
    objects: list[dict[str, Any]],
    bounds: dict[str, float],
    source_files: dict[str, dict[str, Any]],
    source_dir: Path,
    depot_utm: tuple[float, float],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    width = int(math.ceil((bounds["max_x"] - bounds["origin_min_x"]) / bounds["cell_size_m"])) + 1
    height = int(math.ceil((bounds["origin_max_y"] - bounds["min_y"]) / bounds["cell_size_m"])) + 1
    free_cells = [[row, col] for row in range(height) for col in range(width)]
    risk_grid = [[0 for _ in range(width)] for _ in range(height)]
    tasks: dict[str, Any] = {
        "metadata": _scenario_metadata(objects, bounds, source_files, source_dir),
        "point_tasks": [],
        "line_tasks": [],
        "area_tasks": [],
    }
    for obj in objects:
        cells = _cells_for_object(obj, bounds)
        if obj["geometry"] in {"line", "area"} and len(cells) < 2:
            cells = _expand_single_cell(cells[0], height, width)
        for row, col in cells:
            risk_grid[row][col] = max(risk_grid[row][col], int(obj["risk"]))
        task = _task_payload(obj, cells)
        if obj["geometry"] == "point":
            tasks["point_tasks"].append(task)
        elif obj["geometry"] == "line":
            tasks["line_tasks"].append(task)
        else:
            tasks["area_tasks"].append(task)

    depot_cell = _utm_to_cell(depot_utm, bounds)
    grid = {
        "name": SCENARIO_NAME,
        "description": "Yangshan V1.3.3 historical scheduler training scenario from user-provided manual task data.",
        "width": width,
        "height": height,
        "cell_size_m": float(args.cell_size_m),
        "depot": list(depot_cell),
        "free_cells": free_cells,
        "obstacles": [],
        "risk_grid": risk_grid,
        "metadata": {
            **tasks["metadata"],
            "coordinate_native": True,
            "distance_mode": "utm_euclidean",
            "crs": "EPSG:4326 converted to EPSG:32651 scheduler grid",
            "coordinate_resolution_m": float(args.cell_size_m),
            "coordinate_origin": {
                "min_x": bounds["origin_min_x"],
                "max_y": bounds["origin_max_y"],
                "resolution_m": bounds["cell_size_m"],
            },
            "depot_wgs84": {
                "lat": float(args.depot_lat),
                "lon": float(args.depot_lon),
                "coordinate_text": DEPOT_TEXT,
            },
            "depot_utm_epsg_32651": [depot_utm[0], depot_utm[1]],
            "platform_depots": {"uav": list(depot_cell), "usv": list(depot_cell)},
        },
    }
    return grid, tasks


def _scenario_metadata(
    objects: list[dict[str, Any]],
    bounds: dict[str, float],
    source_files: dict[str, dict[str, Any]],
    source_dir: Path,
) -> dict[str, Any]:
    return {
        "scenario_name": SCENARIO_NAME,
        "port": "Yangshan",
        "contract_status": "HISTORICAL_ENGINEERING_TRAINING",
        "scenario_generated": True,
        "historical_only": True,
        "final_experiment_eligible": False,
        "source_package": source_dir.as_posix(),
        "source_version_or_edition": "V1.3.3 NoVirtualAtoN NoSouthBerthing",
        "source_files": source_files,
        "cell_size_m": bounds["cell_size_m"],
        "grid_shape": [bounds["height"], bounds["width"]],
        "task_type_counts": dict(Counter(obj["task_type"] for obj in objects)),
        "geometry_counts": dict(Counter(obj["geometry"] for obj in objects)),
        "task_family_counts": dict(Counter(obj["metadata"]["task_family"] for obj in objects)),
        "importance_counts": dict(Counter(obj["importance_class"] for obj in objects)),
        "allowed_platform_counts": dict(Counter("|".join(obj["allowed_platforms"]) for obj in objects)),
        "note": (
            "Yangshan is retained as a historical/manual engineering training scenario. "
            "This does not replace Los Angeles as the primary V1.2 empirical scenario and is not final evidence."
        ),
    }


def _task_payload(obj: dict[str, Any], cells: list[tuple[int, int]]) -> dict[str, Any]:
    metadata = dict(obj["metadata"])
    payload: dict[str, Any] = {
        "id": obj["id"],
        "type": obj["task_type"],
        "risk": int(obj["risk"]),
        "max_interval": int(obj["max_interval"]),
        "deadline": obj["deadline"],
        "allowed_platforms": list(obj["allowed_platforms"]),
        "required_work": float(obj["required_work"]),
        "completed_work": float(obj["completed_work"]),
        "remaining_work": float(obj["remaining_work"]),
        "work_threshold": float(obj["work_threshold"]),
        "quality_requirement": dict(obj["quality_requirement"]),
        "quality_acceptance_ref": obj["quality_acceptance_ref"],
        "coverage_threshold": 0.9 if obj["geometry"] == "point" else 0.95,
        "priority": float(obj["risk"]),
        "metadata": metadata,
    }
    if obj["geometry"] == "point":
        payload["cell"] = list(cells[0])
        payload["service_time"] = max(1, int(math.ceil(float(obj["required_work"]))))
    else:
        payload["cells"] = [list(cell) for cell in cells]
        payload["service_time"] = max(1, int(math.ceil(float(obj["required_work"]))))
    return payload


def _cells_for_object(obj: dict[str, Any], bounds: dict[str, float]) -> list[tuple[int, int]]:
    if obj["geometry"] == "point":
        points = _flatten_paths(obj["utm_paths"])
        return [_utm_to_cell(_centroid(points), bounds)]
    cells: list[tuple[int, int]] = []
    for path in obj["utm_paths"]:
        anchors = [_utm_to_cell(point, bounds) for point in path]
        if len(anchors) == 1:
            cells.extend(anchors)
            continue
        for start, end in zip(anchors, anchors[1:]):
            cells.extend(_interpolate_cells(start, end))
    return _dedupe_cells(cells)


def _bounds(objects: list[dict[str, Any]], depot_utm: tuple[float, float], cell_size_m: float) -> dict[str, float]:
    points = [depot_utm]
    for obj in objects:
        points.extend(_flatten_paths(obj["utm_paths"]))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    margin = max(cell_size_m * 2.0, 500.0)
    origin_min_x = math.floor((min(xs) - margin) / cell_size_m) * cell_size_m
    origin_max_y = math.ceil((max(ys) + margin) / cell_size_m) * cell_size_m
    max_x = math.ceil((max(xs) + margin) / cell_size_m) * cell_size_m
    min_y = math.floor((min(ys) - margin) / cell_size_m) * cell_size_m
    width = int(math.ceil((max_x - origin_min_x) / cell_size_m)) + 1
    height = int(math.ceil((origin_max_y - min_y) / cell_size_m)) + 1
    return {
        "origin_min_x": origin_min_x,
        "origin_max_y": origin_max_y,
        "max_x": max_x,
        "min_y": min_y,
        "cell_size_m": cell_size_m,
        "width": width,
        "height": height,
    }


def _utm_to_cell(point: tuple[float, float], bounds: dict[str, float]) -> tuple[int, int]:
    x, y = point
    resolution = float(bounds["cell_size_m"])
    row = int(round((float(bounds["origin_max_y"]) - y) / resolution))
    col = int(round((x - float(bounds["origin_min_x"])) / resolution))
    row = min(max(row, 0), int(bounds["height"]) - 1)
    col = min(max(col, 0), int(bounds["width"]) - 1)
    return row, col


def _interpolate_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    row0, col0 = start
    row1, col1 = end
    steps = max(abs(row1 - row0), abs(col1 - col0), 1)
    return [
        (round(row0 + (row1 - row0) * step / steps), round(col0 + (col1 - col0) * step / steps))
        for step in range(steps + 1)
    ]


def _expand_single_cell(cell: tuple[int, int], height: int, width: int) -> list[tuple[int, int]]:
    row, col = cell
    candidates = [(row, col), (min(row + 1, height - 1), col), (row, min(col + 1, width - 1))]
    return _dedupe_cells(candidates)


def _geometry_from_mode(geometry_mode: str) -> str:
    if geometry_mode == "TARGET":
        return "point"
    if geometry_mode == "CORRIDOR":
        return "line"
    if geometry_mode == "AREA":
        return "area"
    raise ValueError(f"unsupported geometry_mode: {geometry_mode}")


def _risk_from_importance(value: str) -> int:
    return {"A": 3, "B": 2, "C": 1}.get(str(value).strip().upper(), 1)


def _allowed_platforms(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip().upper() for item in str(value).replace(",", "|").split("|") if item.strip())
    return parsed or ("UAV", "USV")


def _parse_json_field(raw: str | None) -> dict[str, Any]:
    if raw is None or not str(raw).strip():
        return {}
    return json.loads(raw)


def _int_or_none(value: str | None) -> int | None:
    numeric = _float_or_none(value)
    if numeric is None:
        return None
    return int(numeric)


def _float_or_none(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _dedupe_cells(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for cell in cells:
        if cell in seen:
            continue
        seen.add(cell)
        result.append(cell)
    return result


def _flatten_paths(paths: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    return [point for path in paths for point in path]


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / max(len(points), 1),
        sum(point[1] for point in points) / max(len(points), 1),
    )


def _parse_gpkg_geometry(blob: bytes) -> tuple[str, Any]:
    data = bytes(blob)
    if data[:2] != b"GP":
        raise ValueError("not a GeoPackage binary geometry")
    flags = data[3]
    envelope_code = (flags >> 1) & 7
    offset = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope_code, 0)
    return _parse_wkb(data[offset:])


def _parse_wkb(data: bytes, offset: int = 0) -> tuple[str, Any]:
    endian = "<" if data[offset] == 1 else ">"
    geometry_type = struct.unpack(endian + "I", data[offset + 1 : offset + 5])[0]
    base_type = geometry_type % 1000
    cursor = offset + 5
    if base_type == 1:
        x, y = struct.unpack(endian + "dd", data[cursor : cursor + 16])
        return "Point", [(x, y)]
    if base_type == 2:
        count = struct.unpack(endian + "I", data[cursor : cursor + 4])[0]
        cursor += 4
        points = []
        for _ in range(count):
            x, y = struct.unpack(endian + "dd", data[cursor : cursor + 16])
            cursor += 16
            points.append((x, y))
        return "LineString", points
    if base_type == 3:
        ring_count = struct.unpack(endian + "I", data[cursor : cursor + 4])[0]
        cursor += 4
        rings = []
        for _ in range(ring_count):
            count = struct.unpack(endian + "I", data[cursor : cursor + 4])[0]
            cursor += 4
            ring = []
            for _ in range(count):
                x, y = struct.unpack(endian + "dd", data[cursor : cursor + 16])
                cursor += 16
                ring.append((x, y))
            rings.append(ring)
        return "Polygon", rings
    if base_type in {4, 5, 6, 7}:
        count = struct.unpack(endian + "I", data[cursor : cursor + 4])[0]
        cursor += 4
        geometries = []
        for _ in range(count):
            geom = _parse_wkb(data, cursor)
            geometries.append(geom)
            cursor += _wkb_size(data, cursor)
        return "GeometryCollection", geometries
    raise NotImplementedError(f"unsupported WKB type: {geometry_type}")


def _wkb_size(data: bytes, offset: int = 0) -> int:
    endian = "<" if data[offset] == 1 else ">"
    geometry_type = struct.unpack(endian + "I", data[offset + 1 : offset + 5])[0]
    base_type = geometry_type % 1000
    cursor = offset + 5
    if base_type == 1:
        return 21
    if base_type == 2:
        count = struct.unpack(endian + "I", data[cursor : cursor + 4])[0]
        return 9 + count * 16
    if base_type == 3:
        ring_count = struct.unpack(endian + "I", data[cursor : cursor + 4])[0]
        cursor += 4
        for _ in range(ring_count):
            count = struct.unpack(endian + "I", data[cursor : cursor + 4])[0]
            cursor += 4 + count * 16
        return cursor - offset
    count = struct.unpack(endian + "I", data[cursor : cursor + 4])[0]
    cursor += 4
    for _ in range(count):
        cursor += _wkb_size(data, cursor)
    return cursor - offset


def _geometry_paths(geometry: tuple[str, Any]) -> list[list[tuple[float, float]]]:
    kind, value = geometry
    if kind in {"Point", "LineString"}:
        return [list(value)]
    if kind == "Polygon":
        return [list(ring) for ring in value]
    paths: list[list[tuple[float, float]]] = []
    for subgeometry in value:
        paths.extend(_geometry_paths(subgeometry))
    return paths


def _wgs84_to_utm_zone51(*, lon: float, lat: float) -> tuple[float, float]:
    semi_major = 6378137.0
    flattening = 1 / 298.257223563
    eccentricity_sq = flattening * (2 - flattening)
    second_eccentricity_sq = eccentricity_sq / (1 - eccentricity_sq)
    scale = 0.9996
    phi = math.radians(lat)
    lam = math.radians(lon)
    central_meridian = math.radians(123.0)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    tan_phi = math.tan(phi)
    n_radius = semi_major / math.sqrt(1 - eccentricity_sq * sin_phi * sin_phi)
    tan_sq = tan_phi * tan_phi
    c_term = second_eccentricity_sq * cos_phi * cos_phi
    a_term = cos_phi * (lam - central_meridian)
    meridian_arc = semi_major * (
        (1 - eccentricity_sq / 4 - 3 * eccentricity_sq**2 / 64 - 5 * eccentricity_sq**3 / 256) * phi
        - (3 * eccentricity_sq / 8 + 3 * eccentricity_sq**2 / 32 + 45 * eccentricity_sq**3 / 1024)
        * math.sin(2 * phi)
        + (15 * eccentricity_sq**2 / 256 + 45 * eccentricity_sq**3 / 1024) * math.sin(4 * phi)
        - (35 * eccentricity_sq**3 / 3072) * math.sin(6 * phi)
    )

    easting = 500000 + scale * n_radius * (
        a_term
        + (1 - tan_sq + c_term) * a_term**3 / 6
        + (5 - 18 * tan_sq + tan_sq**2 + 72 * c_term - 58 * second_eccentricity_sq) * a_term**5 / 120
    )
    northing = scale * (
        meridian_arc
        + n_radius
        * tan_phi
        * (
            a_term**2 / 2
            + (5 - tan_sq + 9 * c_term + 4 * c_term**2) * a_term**4 / 24
            + (61 - 58 * tan_sq + tan_sq**2 + 600 * c_term - 330 * second_eccentricity_sq)
            * a_term**6
            / 720
        )
    )
    return easting, northing


def _summary(
    grid: dict[str, Any],
    tasks: dict[str, Any],
    source_dir: Path,
    grid_path: Path,
    tasks_path: Path,
) -> dict[str, Any]:
    return {
        "scenario_name": SCENARIO_NAME,
        "source_dir": source_dir.as_posix(),
        "grid_path": grid_path.as_posix(),
        "tasks_path": tasks_path.as_posix(),
        "task_counts": {
            "point": len(tasks["point_tasks"]),
            "line": len(tasks["line_tasks"]),
            "area": len(tasks["area_tasks"]),
            "total": len(tasks["point_tasks"]) + len(tasks["line_tasks"]) + len(tasks["area_tasks"]),
        },
        "grid_shape": [grid["height"], grid["width"]],
        "depot_cell": grid["depot"],
        "contract_status": grid["metadata"]["contract_status"],
        "historical_only": grid["metadata"]["historical_only"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _readme(summary: dict[str, Any]) -> str:
    return f"""# {SCENARIO_NAME}

This is a Yangshan V1.3.3 scheduler training scenario imported from the
user-provided manual task package.

Status: `HISTORICAL_ENGINEERING_TRAINING`.

It is intended for engineering training and cross-port comparison only. It does
not replace Los Angeles as the primary V1.2 empirical scenario and must not be
reported as final experiment evidence.

- Source directory: `{summary["source_dir"]}`
- Point tasks: {summary["task_counts"]["point"]}
- Corridor tasks: {summary["task_counts"]["line"]}
- Area tasks: {summary["task_counts"]["area"]}
- Total active tasks: {summary["task_counts"]["total"]}
- Grid shape: {summary["grid_shape"]}
- Depot cell: {summary["depot_cell"]}
- Cell size: {CELL_SIZE_M:.0f} m
- Lifecycle: `v1_2_direct_service`

Regenerate from the local source package:

```powershell
.\\.venv\\Scripts\\python.exe tools\\import_yangshan_v133_training.py
```

Smoke-check the scheduler environment:

```powershell
.\\.venv\\Scripts\\python.exe tools\\check_port_inspection_env.py --config configs\\port_yangshan_training_v133.toml --steps 2 --seed 7
```

Training requires explicit historical-baseline acknowledgement:

```powershell
.\\.venv\\Scripts\\python.exe tools\\train_port_scheduler_rl.py --config configs\\port_yangshan_training_v133.toml --allow-historical-baseline --algorithm heterogeneous_mappo
```
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
