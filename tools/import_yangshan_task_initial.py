from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sqlite3
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path("D:/地图/任务初版")
DEFAULT_OUTPUT_DIR = Path("data/ports/yangshan_task_initial_v1")
DEFAULT_CONFIG = Path("configs/port_yangshan_task_initial_v1.toml")

SOURCE_FILES = {
    "report": ("洋山港任务信息补齐报告.md", "source_report.md"),
    "project": ("洋山港任务工程_补齐版_修复.qgz", "source_project.qgz"),
    "package": ("洋山港任务工程_补齐版_完整包.zip", "source_package.zip"),
    "gpkg": ("洋山港全港区正式GIS训练任务数据_补齐版.gpkg", "source_tasks.gpkg"),
    "dynamic": ("洋山港动态任务种子_补齐版.csv", "source_dynamic_seeds.csv"),
    "fixed": ("洋山港固定巡检任务节点_补齐版.csv", "source_fixed_tasks.csv"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the initial Yangshan QGIS/GPKG task map for scheduler training.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset-name", default="yangshan_task_initial_v1")
    parser.add_argument("--coordinate-resolution-m", type=float, default=100.0)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    source_out = output_dir / "source"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_out.mkdir(parents=True, exist_ok=True)

    copied = copy_source_package(source_dir, source_out)
    gpkg = inspect_gpkg(copied["gpkg"])
    origin = make_coordinate_origin(gpkg["extent"], args.coordinate_resolution_m)
    fixed_rows = read_csv(copied["fixed"])
    dynamic_rows = read_csv(copied["dynamic"])
    fixed_tasks = [fixed_task(row, origin) for row in fixed_rows]
    dynamic_tasks = [dynamic_task(row, origin) for row in dynamic_rows]
    depots = build_depots(gpkg["coastline_points"], fixed_tasks + dynamic_tasks, origin)
    all_cells = {tuple(depot["cell"]) for depot in depots}
    all_cells.update(tuple(task["cell"]) for task in fixed_tasks + dynamic_tasks)
    width = max(cell[1] for cell in all_cells) + 1
    height = max(cell[0] for cell in all_cells) + 1
    risk_grid = build_risk_grid(width, height, fixed_tasks + dynamic_tasks)
    free_cells = [[row, col] for row in range(height) for col in range(width)]

    primary_depot = next((depot for depot in depots if depot["platform_type"] == "UAV"), depots[0])
    platform_depots = {}
    for depot in depots:
        platform_depots[depot["platform_type"]] = depot["cell"]
    grid = {
        "name": args.dataset_name,
        "description": "Coordinate-native Yangshan port scheduler map imported from the supplied QGIS/GPKG task package.",
        "width": width,
        "height": height,
        "cell_size_m": args.coordinate_resolution_m,
        "depot": primary_depot["cell"],
        "free_cells": free_cells,
        "obstacles": [],
        "risk_grid": risk_grid,
        "metadata": {
            "scenario_type": "port_water_surface_inspection",
            "map_source": "D:/地图/任务初版/洋山港全港区正式GIS训练任务数据_补齐版.gpkg",
            "qgis_project_source": "D:/地图/任务初版/洋山港任务工程_补齐版.qgz",
            "coordinate_native": True,
            "distance_mode": "utm_euclidean",
            "crs": "EPSG:32651",
            "coordinate_resolution_m": args.coordinate_resolution_m,
            "coordinate_origin": origin,
            "source_files": {key: str(path.as_posix()) for key, path in copied.items()},
            "source_layer_counts": gpkg["layer_counts"],
            "source_extent_utm": gpkg["extent"],
            "generated_depots": depots,
            "platform_depots": platform_depots,
            "fixed_task_count": len(fixed_tasks),
            "dynamic_seed_count": len(dynamic_tasks),
            "task_count": len(fixed_tasks) + len(dynamic_tasks),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "No legacy project map is used. The current scheduler treats UTM coordinate bins as model features only; no low-level path actions are emitted.",
        },
    }
    tasks = {"point_tasks": fixed_tasks + dynamic_tasks, "line_tasks": [], "area_tasks": []}

    grid_path = output_dir / f"{args.dataset_name}_grid.json"
    tasks_path = output_dir / f"{args.dataset_name}_tasks.json"
    summary_path = output_dir / "import_summary.json"
    readme_path = output_dir / "README.md"
    config_path = Path(args.config)

    grid_path.write_text(json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8")
    tasks_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    write_config(config_path, grid_path, tasks_path, output_dir, platform_depots)

    summary = {
        "dataset_name": args.dataset_name,
        "source_dir": str(source_dir),
        "grid_path": str(grid_path),
        "tasks_path": str(tasks_path),
        "config_path": str(config_path),
        "coordinate_resolution_m": args.coordinate_resolution_m,
        "grid_shape": [height, width],
        "fixed_task_count": len(fixed_tasks),
        "dynamic_seed_count": len(dynamic_tasks),
        "task_count": len(tasks["point_tasks"]),
        "task_class_counts": dict(Counter(task["metadata"]["source_task_class"] for task in tasks["point_tasks"])),
        "risk_counts": dict(Counter(task["risk"] for task in tasks["point_tasks"])),
        "platform_depots": platform_depots,
        "generated_depots": depots,
        "source_layer_counts": gpkg["layer_counts"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(readme(summary), encoding="utf-8")

    print(f"grid={grid_path}")
    print(f"tasks={tasks_path}")
    print(f"config={config_path}")
    print(f"summary={summary_path}")
    print(f"tasks={len(tasks['point_tasks'])} fixed={len(fixed_tasks)} dynamic={len(dynamic_tasks)}")
    print(f"grid_shape={height}x{width} resolution_m={args.coordinate_resolution_m:g}")


def copy_source_package(source_dir: Path, target_dir: Path) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for key, (source_name, target_name) in SOURCE_FILES.items():
        source_path = source_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"missing source file: {source_path}")
        target_path = target_dir / target_name
        shutil.copy2(source_path, target_path)
        copied[key] = target_path
    return copied


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def inspect_gpkg(path: Path) -> dict[str, Any]:
    con = sqlite3.connect(path)
    try:
        tables = [row[0] for row in con.execute("select table_name from gpkg_contents order by table_name").fetchall()]
        layer_counts: dict[str, int] = {}
        all_points: list[tuple[float, float]] = []
        for table in tables:
            layer_counts[table] = int(con.execute(f'select count(*) from "{table}"').fetchone()[0])
            for (blob,) in con.execute(f'select geom from "{table}"').fetchall():
                all_points.extend(geometry_points(parse_gpkg_geometry(blob)))
        source_depots = []
        for row in con.execute(
            'select depot_id, platform_type, source_kind, initial_visible, longitude, latitude, x_utm, y_utm, geom from depots order by fid'
        ).fetchall():
            geom = parse_gpkg_geometry(row[-1])
            point = geometry_points(geom)[0]
            source_depots.append(
                {
                    "depot_id": row[0],
                    "platform_type": str(row[1]).upper(),
                    "source_kind": row[2],
                    "initial_visible": bool(row[3]),
                    "longitude": float(row[4]),
                    "latitude": float(row[5]),
                    "x_utm": float(row[6]),
                    "y_utm": float(row[7]),
                    "geometry_x": point[0],
                    "geometry_y": point[1],
                }
            )
        if not all_points:
            raise ValueError("GeoPackage contains no geometry points")
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        return {
            "layer_counts": layer_counts,
            "extent": {
                "min_x": min(xs),
                "min_y": min(ys),
                "max_x": max(xs),
                "max_y": max(ys),
            },
            "source_depots": source_depots,
            "coastline_points": collect_layer_points(con, "source_port_coastline"),
        }
    finally:
        con.close()


def collect_layer_points(con: sqlite3.Connection, table: str) -> list[tuple[float, float]]:
    exists = con.execute("select 1 from gpkg_contents where table_name = ?", (table,)).fetchone()
    if not exists:
        return []
    points: list[tuple[float, float]] = []
    for (blob,) in con.execute(f'select geom from "{table}"').fetchall():
        points.extend(geometry_points(parse_gpkg_geometry(blob)))
    return points


def parse_gpkg_geometry(blob: bytes) -> tuple[str, Any]:
    data = bytes(blob)
    if data[:2] != b"GP":
        raise ValueError("not a GeoPackage binary geometry")
    flags = data[3]
    envelope_code = (flags >> 1) & 7
    offset = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope_code, 0)
    return parse_wkb(data[offset:])


def parse_wkb(data: bytes, offset: int = 0) -> tuple[str, Any]:
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
            geom = parse_wkb(data, cursor)
            geometries.append(geom)
            cursor += wkb_size(data, cursor)
        return "GeometryCollection", geometries
    raise NotImplementedError(f"unsupported WKB type: {geometry_type}")


def wkb_size(data: bytes, offset: int = 0) -> int:
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
        cursor += wkb_size(data, cursor)
    return cursor - offset


def geometry_points(geometry: tuple[str, Any]) -> list[tuple[float, float]]:
    kind, value = geometry
    if kind in {"Point", "LineString"}:
        return list(value)
    if kind == "Polygon":
        return [point for ring in value for point in ring]
    return [point for subgeometry in value for point in geometry_points(subgeometry)]


def make_coordinate_origin(extent: dict[str, float], resolution_m: float) -> dict[str, float]:
    margin = max(resolution_m * 2.0, 200.0)
    return {
        "min_x": math.floor((extent["min_x"] - margin) / resolution_m) * resolution_m,
        "max_y": math.ceil((extent["max_y"] + margin) / resolution_m) * resolution_m,
        "resolution_m": resolution_m,
    }


def build_depots(
    coastline_points: list[tuple[float, float]],
    tasks: list[dict[str, Any]],
    origin: dict[str, float],
) -> list[dict[str, Any]]:
    if not coastline_points:
        raise ValueError("the GeoPackage must provide source_port_coastline geometry for depot placement")
    if not tasks:
        raise ValueError("at least one task is required before placing a depot")

    task_points = [(float(task["metadata"]["x_utm"]), float(task["metadata"]["y_utm"])) for task in tasks]
    centroid_x = sum(point[0] for point in task_points) / len(task_points)
    centroid_y = sum(point[1] for point in task_points) / len(task_points)
    x_utm, y_utm = min(
        coastline_points,
        key=lambda point: (point[0] - centroid_x) ** 2 + (point[1] - centroid_y) ** 2,
    )
    cell = utm_to_cell(x_utm, y_utm, origin)
    depot = {
        "depot_id": "COAST_AUTO_01",
        "source_kind": "user_defined_coastline",
        "placement_rule": "nearest source_port_coastline vertex to task centroid",
        "x_utm": x_utm,
        "y_utm": y_utm,
        "cell": cell,
    }
    return [
        {**depot, "platform_type": "UAV"},
        {**depot, "platform_type": "USV"},
    ]


def fixed_task(row: dict[str, str], origin: dict[str, float]) -> dict[str, Any]:
    task_class = row["task_class"]
    facility_type = row["facility_type"]
    risk = fixed_risk(task_class, facility_type)
    cell = utm_to_cell(float(row["x_utm"]), float(row["y_utm"]), origin)
    return {
        "id": row["task_id"],
        "type": f"{task_class}_{facility_type}".strip("_"),
        "cell": cell,
        "risk": risk,
        "service_time": service_time(task_class, facility_type, risk),
        "max_interval": max_interval(risk),
        "deadline": max_interval(risk),
        "allowed_platforms": ["UAV", "USV"],
        "metadata": {
            "source_task_class": task_class,
            "facility_type": facility_type,
            "parent_id": row.get("parent_id", ""),
            "side_zone": row.get("side_zone", ""),
            "source_dataset": row.get("source_dataset", ""),
            "source_feature_id": row.get("source_feature_id", ""),
            "verification_level": row.get("verification_level", ""),
            "source_kind": row.get("source_kind", ""),
            "generation_method": row.get("generation_method", ""),
            "manual_added": parse_bool(row.get("manual_added", "False")),
            "attribute_status": row.get("attribute_status", ""),
            "longitude": float(row["longitude"]),
            "latitude": float(row["latitude"]),
            "x_utm": float(row["x_utm"]),
            "y_utm": float(row["y_utm"]),
            "initial_visible": parse_bool(row.get("initial_visible", "True")),
            "remark": row.get("remark", ""),
        },
    }


def dynamic_task(row: dict[str, str], origin: dict[str, float]) -> dict[str, Any]:
    event_family = row["event_family"]
    risk = 3 if event_family == "floating_object" else 2
    cell = utm_to_cell(float(row["x_utm"]), float(row["y_utm"]), origin)
    return {
        "id": row["event_id"],
        "type": f"dynamic_{event_family}",
        "cell": cell,
        "risk": risk,
        "service_time": 4,
        "max_interval": max_interval(risk),
        "deadline": max_interval(risk),
        "allowed_platforms": ["UAV", "USV"],
        "metadata": {
            "source_task_class": event_family,
            "parent_zone_id": row.get("parent_zone_id", ""),
            "source_kind": row.get("source_kind", ""),
            "longitude": float(row["longitude"]),
            "latitude": float(row["latitude"]),
            "x_utm": float(row["x_utm"]),
            "y_utm": float(row["y_utm"]),
            "initial_visible": parse_bool(row.get("initial_visible", "False")),
        },
    }


def utm_to_cell(x: float, y: float, origin: dict[str, float]) -> list[int]:
    resolution = float(origin["resolution_m"])
    row = int(round((float(origin["max_y"]) - y) / resolution))
    col = int(round((x - float(origin["min_x"])) / resolution))
    return [max(row, 0), max(col, 0)]


def fixed_risk(task_class: str, facility_type: str) -> int:
    risk = 3 if task_class == "navigation_aid" else 2
    if facility_type in {"light_major", "beacon_special_purpose", "breakwater", "bridge_pier_or_bridge_water_structure"}:
        risk += 1
    if "coastline" in facility_type:
        risk -= 1
    return max(1, min(3, risk))


def service_time(task_class: str, facility_type: str, risk: int) -> int:
    if task_class == "navigation_aid":
        return 4 if risk >= 3 else 3
    if "breakwater" in facility_type or "bridge" in facility_type:
        return 5
    return 4 if risk >= 2 else 3


def max_interval(risk: int) -> int:
    if risk >= 3:
        return 12
    if risk == 2:
        return 24
    return 36


def build_risk_grid(width: int, height: int, tasks: list[dict[str, Any]]) -> list[list[int]]:
    grid = [[1 for _ in range(width)] for _ in range(height)]
    for task in tasks:
        row, col = task["cell"]
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = max(grid[row][col], int(task["risk"]))
    return grid


def write_config(
    config_path: Path,
    grid_path: Path,
    tasks_path: Path,
    output_dir: Path,
    platform_depots: dict[str, list[int]],
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    content = f'''grid_path = "{grid_path.as_posix()}"
tasks_path = "{tasks_path.as_posix()}"
output_dir = "{output_dir.as_posix()}"
map_source = "yangshan_task_initial_qgis_gpkg"
platform_profiles_path = "configs/platform_profiles_cn_common.toml"
uav_count = 4
usv_count = 4

[platform_depots]
uav = {platform_depots.get("UAV", platform_depots.get("USV", [0, 0]))}
usv = {platform_depots.get("USV", platform_depots.get("UAV", [0, 0]))}

platform_profile_sequence = [
  "UAV_H_M350",
  "UAV_H_M350",
  "UAV_M_M30",
  "UAV_M_M30",
  "USV_P_M75",
  "USV_P_M75",
  "USV_S_SURVEY",
  "USV_S_SURVEY",
]

[scheduling]
risk_weight = 14.0
distance_weight = 0.07
load_weight = 0.28
compatibility_bonus = 5.0

[scheduler_rl]
max_steps = 96
candidate_k = 12
learning_rate = 0.0003
gamma = 0.98
gae_lambda = 0.95
clip_ratio = 0.2
update_epochs = 4
rollout_steps = 32
hidden_dim = 128

[scheduler_rl.reward]
team_close_reward = 8.0
screen_progress_reward = 1.0
review_progress_reward = 1.5
energy_cost = 0.5
time_cost = 0.01
invalid_penalty = 5.0
conflict_penalty = 1.0

[review_trigger]
confidence_threshold = 0.65
mandatory_review_risk = 3
base_review_deadline = 36.0
risk_deadline_scale = 4.0
confidence_deadline_scale = 3.0
sensitivity = 0.85
specificity = 0.80
confidence_noise = 0.08
uncertainty_base = 0.35

[review_trigger.anomaly_probability_by_risk]
1 = 0.10
2 = 0.25
3 = 0.45
'''
    config_path.write_text(content, encoding="utf-8")


def readme(summary: dict[str, Any]) -> str:
    return f"""# yangshan_task_initial_v1

This scenario uses only the QGIS/GeoPackage task map from `D:/地图/任务初版`.
No legacy project map is reused.

- Coordinate mode: EPSG:32651 UTM, Euclidean travel proxy
- Coordinate feature resolution: {summary["coordinate_resolution_m"]} m
- Model grid shape used only as coordinate-feature envelope: {summary["grid_shape"]}
- Fixed inspection tasks: {summary["fixed_task_count"]}
- Dynamic seed tasks: {summary["dynamic_seed_count"]}
- Total point tasks: {summary["task_count"]}
- Risk counts: {summary["risk_counts"]}
- Platform depots: {summary["platform_depots"]}
- Depot placement: user-defined shoreline depot on `source_port_coastline`;
  the supplied QGIS map has no explicit depot marker, so UAV and USV share
  this coast-edge base instead of using a water-surface point.

The scheduler still consumes the existing `PortInspectionSchedulingEnv` JSON
schema, so coordinates are encoded as UTM-derived feature bins. These bins are
not a rasterized path-planning map and no low-level path action is emitted.
"""


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
