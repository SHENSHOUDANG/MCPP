from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from http.client import IncompleteRead, RemoteDisconnected
import json
from math import cos, floor, radians
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request, urlopen

from matplotlib.path import Path as MplPath


OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
)

DEFAULT_BBOX = {
    "south": 30.585,
    "west": 121.985,
    "north": 30.675,
    "east": 122.165,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real OSM-derived port inspection grid.")
    parser.add_argument("--name", default="shanghai_yangshan_osm_v1")
    parser.add_argument("--output-dir", default="data/ports/shanghai_yangshan_osm_v1")
    parser.add_argument("--cell-size-m", type=float, default=150.0)
    parser.add_argument("--south", type=float, default=DEFAULT_BBOX["south"])
    parser.add_argument("--west", type=float, default=DEFAULT_BBOX["west"])
    parser.add_argument("--north", type=float, default=DEFAULT_BBOX["north"])
    parser.add_argument("--east", type=float, default=DEFAULT_BBOX["east"])
    parser.add_argument("--depot-lat", type=float, default=30.596)
    parser.add_argument("--depot-lon", type=float, default=122.000)
    parser.add_argument("--raw-osm", default=None, help="Use an existing Overpass JSON file instead of downloading.")
    parser.add_argument("--skip-download", action="store_true", help="Require --raw-osm and do not call Overpass.")
    parser.add_argument("--reuse-raw-cache", action="store_true", help="Reuse the default raw OSM file if it already exists.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(args.raw_osm) if args.raw_osm else output_dir / "osm_overpass_raw.json"
    bbox = (args.south, args.west, args.north, args.east)
    if args.raw_osm or args.skip_download or (args.reuse_raw_cache and raw_path.exists()):
        if not raw_path.exists():
            raise FileNotFoundError(f"raw OSM file not found: {raw_path}")
        osm = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        query = build_overpass_query(*bbox)
        osm = download_overpass(query)
        raw_path.write_text(json.dumps(osm, ensure_ascii=False), encoding="utf-8")

    projector = LocalProjector(*bbox, cell_size_m=float(args.cell_size_m))
    layers = extract_layers(osm, projector)
    grid = build_grid(args.name, projector, layers, depot_lat=args.depot_lat, depot_lon=args.depot_lon, raw_path=raw_path)
    tasks = build_tasks(grid, layers)

    grid_path = output_dir / f"{args.name}_grid.json"
    tasks_path = output_dir / f"{args.name}_tasks.json"
    grid_path.write_text(json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8")
    tasks_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "README.md").write_text(readme(args.name, bbox, args.cell_size_m), encoding="utf-8")
    print(f"raw_osm={raw_path}")
    print(f"grid={grid_path}")
    print(f"tasks={tasks_path}")
    print(f"free_cells={len(grid['free_cells'])} obstacles={len(grid['obstacles'])}")
    print(
        "tasks="
        f"{len(tasks['point_tasks'])} point, "
        f"{len(tasks['line_tasks'])} line, "
        f"{len(tasks['area_tasks'])} area"
    )


def build_overpass_query(south: float, west: float, north: float, east: float) -> str:
    bbox = f"{south},{west},{north},{east}"
    return f"""
[out:json][timeout:120];
(
  node["seamark:type"]({bbox});
  node["man_made"~"beacon|lighthouse"]({bbox});
  way["landuse"~"industrial|commercial|port"]({bbox});
  relation["landuse"~"industrial|commercial|port"]({bbox});
  way["man_made"~"pier|breakwater|quay|jetty|groyne"]({bbox});
  relation["man_made"~"pier|breakwater|quay|jetty|groyne"]({bbox});
  way["harbour"]({bbox});
  relation["harbour"]({bbox});
  way["natural"~"water|coastline|bay|strait"]({bbox});
  relation["natural"~"water|bay|strait"]({bbox});
  way["waterway"]({bbox});
  relation["waterway"]({bbox});
  way["seamark:type"~"fairway|harbour|anchorage|buoy|beacon"]({bbox});
  relation["seamark:type"~"fairway|harbour|anchorage|buoy|beacon"]({bbox});
  way["bridge"]({bbox});
);
out body geom;
"""


def download_overpass(query: str) -> dict[str, Any]:
    body = urlencode({"data": query}).encode("utf-8")
    errors: list[str] = []
    for url in OVERPASS_URLS:
        for attempt in range(2):
            request = Request(
                url,
                data=body,
                headers={
                    "User-Agent": "mathbased-mcpp-port-inspection/0.1",
                    "Accept-Encoding": "identity",
                },
            )
            try:
                with urlopen(request, timeout=45) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (IncompleteRead, RemoteDisconnected, URLError, TimeoutError, json.JSONDecodeError) as exc:
                errors.append(f"{url} attempt {attempt + 1}: {exc}")
    raise RuntimeError("Overpass download failed:\n" + "\n".join(errors))


class LocalProjector:
    def __init__(self, south: float, west: float, north: float, east: float, cell_size_m: float) -> None:
        if south >= north or west >= east:
            raise ValueError("invalid bbox")
        self.south = south
        self.west = west
        self.north = north
        self.east = east
        self.cell_size_m = cell_size_m
        self.lat0 = (south + north) / 2.0
        self.m_per_lat = 110_540.0
        self.m_per_lon = 111_320.0 * cos(radians(self.lat0))
        self.width = max(1, int(floor(((east - west) * self.m_per_lon) / cell_size_m)) + 1)
        self.height = max(1, int(floor(((north - south) * self.m_per_lat) / cell_size_m)) + 1)

    def xy(self, lat: float, lon: float) -> tuple[float, float]:
        x = (lon - self.west) * self.m_per_lon
        y = (self.north - lat) * self.m_per_lat
        return x, y

    def cell(self, lat: float, lon: float) -> tuple[int, int]:
        x, y = self.xy(lat, lon)
        row = int(floor(y / self.cell_size_m))
        col = int(floor(x / self.cell_size_m))
        return clamp(row, 0, self.height - 1), clamp(col, 0, self.width - 1)

    def cell_center_xy(self, row: int, col: int) -> tuple[float, float]:
        return (col + 0.5) * self.cell_size_m, (row + 0.5) * self.cell_size_m


def extract_layers(osm: dict[str, Any], projector: LocalProjector) -> dict[str, Any]:
    land_polygons: list[list[tuple[float, float]]] = []
    risk_lines: list[list[tuple[int, int]]] = []
    seamark_points: list[tuple[int, int]] = []
    visual_lines: list[dict[str, Any]] = []
    visual_features: list[dict[str, Any]] = []
    feature_counts: dict[str, int] = {}
    for element in osm.get("elements", []):
        if not isinstance(element, dict):
            continue
        tags = dict(element.get("tags", {}))
        if not tags:
            continue
        key = classify_feature(tags)
        if key:
            feature_counts[key] = feature_counts.get(key, 0) + 1
        if element.get("type") == "node":
            if is_seamark(tags) and "lat" in element and "lon" in element:
                cell = projector.cell(float(element["lat"]), float(element["lon"]))
                seamark_points.append(cell)
                visual_features.append(
                    {
                        "kind": "point",
                        "label": line_label(tags),
                        "cells": [list(cell)],
                        "anchor": list(cell),
                        "source": "osm_seamark",
                    }
                )
            continue
        geometries = element_geometries(element)
        if is_land_polygon(tags):
            for geometry in geometries:
                polygon = geometry_to_polygon_xy(geometry, projector)
                if polygon:
                    land_polygons.append(polygon)
                    cells = sorted(rasterize_polygons([polygon], projector))
                    if cells:
                        visual_features.append(
                            {
                                "kind": "land_polygon",
                                "label": line_label(tags),
                                "cells": [list(cell) for cell in cells],
                                "bbox": list(cell_bbox(cells)),
                                "anchor": list(cell_anchor(cells)),
                                "source": "osm_land_or_port_polygon",
                            }
                        )
        if is_risk_line(tags):
            for geometry in geometries:
                cells = geometry_to_cells(geometry, projector)
                if len(cells) >= 2:
                    line_cells = densify_cells(cells)
                    risk_lines.append(line_cells)
                    feature = {
                        "kind": "line",
                        "label": line_label(tags),
                        "cells": [list(cell) for cell in line_cells],
                        "anchor": list(cell_anchor(line_cells)),
                        "source": "osm_waterway_harbour_or_structure_line",
                    }
                    visual_lines.append({**feature, "cells": [list(cell) for cell in sparsify(line_cells, 12)]})
                    visual_features.append(feature)
        if is_seamark(tags):
            for geometry in geometries:
                cells = geometry_to_cells(geometry, projector)
                if cells:
                    cell = cells[len(cells) // 2]
                    seamark_points.append(cell)
                    visual_features.append(
                        {
                            "kind": "point",
                            "label": line_label(tags),
                            "cells": [list(cell)],
                            "anchor": list(cell),
                            "source": "osm_seamark",
                        }
                    )
    return {
        "land_polygons": land_polygons,
        "risk_lines": risk_lines,
        "seamark_points": dedupe_cells(seamark_points),
        "visual_lines": visual_lines,
        "visual_features": visual_features,
        "feature_counts": feature_counts,
    }


def element_geometries(element: dict[str, Any]) -> list[list[dict[str, float]]]:
    if isinstance(element.get("geometry"), list):
        return [element["geometry"]]
    geometries: list[list[dict[str, float]]] = []
    for member in element.get("members", []):
        if isinstance(member, dict) and isinstance(member.get("geometry"), list):
            geometries.append(member["geometry"])
    return geometries


def geometry_to_polygon_xy(geometry: list[dict[str, float]], projector: LocalProjector) -> list[tuple[float, float]] | None:
    if len(geometry) < 4:
        return None
    first = geometry[0]
    last = geometry[-1]
    if abs(float(first["lat"]) - float(last["lat"])) > 1e-7 or abs(float(first["lon"]) - float(last["lon"])) > 1e-7:
        return None
    return [projector.xy(float(point["lat"]), float(point["lon"])) for point in geometry]


def geometry_to_cells(geometry: list[dict[str, float]], projector: LocalProjector) -> list[tuple[int, int]]:
    return dedupe_cells([projector.cell(float(point["lat"]), float(point["lon"])) for point in geometry])


def build_grid(
    name: str,
    projector: LocalProjector,
    layers: dict[str, Any],
    depot_lat: float,
    depot_lon: float,
    raw_path: Path,
) -> dict[str, Any]:
    obstacles = rasterize_polygons(layers["land_polygons"], projector)
    risk_grid = build_risk_grid(projector, obstacles, layers["risk_lines"])
    free_cells = [(row, col) for row in range(projector.height) for col in range(projector.width) if (row, col) not in obstacles]
    depot = nearest_free(projector.cell(depot_lat, depot_lon), set(free_cells), projector)
    return {
        "name": name,
        "description": "OSM-derived Yangshan port water-surface inspection grid; real GIS features are rasterized from OpenStreetMap.",
        "width": projector.width,
        "height": projector.height,
        "cell_size_m": projector.cell_size_m,
        "depot": list(depot),
        "free_cells": [list(cell) for cell in free_cells],
        "obstacles": [list(cell) for cell in sorted(obstacles)],
        "risk_grid": risk_grid,
        "metadata": {
            "scenario_type": "port_water_surface_inspection",
            "source": "OpenStreetMap Overpass API",
            "source_license": "Open Database License (ODbL)",
            "source_port": "Yangshan Deep-Water Port",
            "real_gis_basis": True,
            "prototype": False,
            "not_for_navigation": True,
            "bbox": {
                "south": projector.south,
                "west": projector.west,
                "north": projector.north,
                "east": projector.east,
            },
            "cell_size_m": projector.cell_size_m,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_osm_path": str(raw_path),
            "feature_counts": layers["feature_counts"],
            "feature_layer_counts": {
                "visual_features": len(layers["visual_features"]),
                "visual_lines": len(layers["visual_lines"]),
                "land_polygons": len(layers["land_polygons"]),
                "risk_lines": len(layers["risk_lines"]),
                "seamark_points": len(layers["seamark_points"]),
            },
            "visual_features": layers["visual_features"],
            "visual_lines_preview": layers["visual_lines"][:12],
        },
    }


def rasterize_polygons(polygons: list[list[tuple[float, float]]], projector: LocalProjector) -> set[tuple[int, int]]:
    obstacles: set[tuple[int, int]] = set()
    for polygon in polygons:
        rows_cols = polygon_bbox_cells(polygon, projector)
        if not rows_cols:
            continue
        min_row, max_row, min_col, max_col = rows_cols
        points: list[tuple[float, float]] = []
        cells: list[tuple[int, int]] = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                points.append(projector.cell_center_xy(row, col))
                cells.append((row, col))
        path = MplPath(polygon)
        mask = path.contains_points(points, radius=projector.cell_size_m * 0.15)
        for cell, inside in zip(cells, mask):
            if inside:
                obstacles.add(cell)
    return obstacles


def polygon_bbox_cells(polygon: list[tuple[float, float]], projector: LocalProjector) -> tuple[int, int, int, int] | None:
    if not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    min_col = clamp(int(floor(min(xs) / projector.cell_size_m)) - 1, 0, projector.width - 1)
    max_col = clamp(int(floor(max(xs) / projector.cell_size_m)) + 1, 0, projector.width - 1)
    min_row = clamp(int(floor(min(ys) / projector.cell_size_m)) - 1, 0, projector.height - 1)
    max_row = clamp(int(floor(max(ys) / projector.cell_size_m)) + 1, 0, projector.height - 1)
    return min_row, max_row, min_col, max_col


def cell_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return min(rows), min(cols), max(rows), max(cols)


def cell_anchor(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return int(round(sum(rows) / len(rows))), int(round(sum(cols) / len(cols)))


def build_risk_grid(projector: LocalProjector, obstacles: set[tuple[int, int]], risk_lines: list[list[tuple[int, int]]]) -> list[list[int]]:
    grid = [[0 if (row, col) in obstacles else 1 for col in range(projector.width)] for row in range(projector.height)]
    for row, col in obstacles:
        for neighbor, dist in cells_within((row, col), radius=4, width=projector.width, height=projector.height):
            if neighbor in obstacles:
                continue
            nr, nc = neighbor
            grid[nr][nc] = max(grid[nr][nc], 3 if dist <= 2 else 2)
    for line in risk_lines:
        for cell in line:
            for neighbor, dist in cells_within(cell, radius=3, width=projector.width, height=projector.height):
                if neighbor in obstacles:
                    continue
                nr, nc = neighbor
                grid[nr][nc] = max(grid[nr][nc], 3 if dist <= 1 else 2)
    return grid


def build_tasks(grid: dict[str, Any], layers: dict[str, Any]) -> dict[str, list[dict[str, object]]]:
    width = int(grid["width"])
    height = int(grid["height"])
    obstacles = {tuple(cell) for cell in grid["obstacles"]}
    free = {tuple(cell) for cell in grid["free_cells"]}
    risk_grid = grid["risk_grid"]
    points = select_point_tasks(layers["seamark_points"], free, risk_grid, width, height)
    lines = select_line_tasks(layers["risk_lines"], free, risk_grid)
    areas = select_area_tasks(free, risk_grid, width, height)
    if len(points) < 6:
        points.extend(fallback_points(points, free, risk_grid, target=10))
    if len(lines) < 3:
        lines.extend(fallback_lines(lines, free, obstacles, width, height))
    if len(areas) < 3:
        areas.extend(fallback_areas(areas, free, risk_grid, width, height))
    return {
        "point_tasks": [
            point_task(index + 1, cell, risk_grid[cell[0]][cell[1]], source="osm_or_risk_edge")
            for index, cell in enumerate(points[:14])
        ],
        "line_tasks": [
            line_task(index + 1, cells, max(risk_grid[row][col] for row, col in cells), source="osm_or_water_corridor")
            for index, cells in enumerate(lines[:8])
            if len(cells) >= 2
        ],
        "area_tasks": [
            area_task(index + 1, cells, max(risk_grid[row][col] for row, col in cells), source="osm_risk_component")
            for index, cells in enumerate(areas[:8])
            if len(cells) >= 4
        ],
    }


def select_point_tasks(
    seamark_points: list[tuple[int, int]],
    free: set[tuple[int, int]],
    risk_grid: list[list[int]],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    points = [nearest_free(cell, free, None, width=width, height=height) for cell in seamark_points]
    points = [cell for cell in points if cell in free]
    edge_candidates = sorted(
        (cell for cell in free if risk_grid[cell[0]][cell[1]] >= 2),
        key=lambda cell: (-risk_grid[cell[0]][cell[1]], cell[0], cell[1]),
    )
    return select_spaced(dedupe_cells(points) + edge_candidates, count=14, min_distance=5)


def select_line_tasks(
    risk_lines: list[list[tuple[int, int]]],
    free: set[tuple[int, int]],
    risk_grid: list[list[int]],
) -> list[list[tuple[int, int]]]:
    lines: list[list[tuple[int, int]]] = []
    for raw in risk_lines:
        cells = [cell for cell in dedupe_cells(raw) if cell in free]
        if len(cells) >= 5:
            score = sum(risk_grid[row][col] for row, col in cells) / len(cells)
            if score >= 1.2:
                lines.append(sparsify(cells, max_points=18))
    lines.sort(key=lambda cells: (-sum(risk_grid[row][col] for row, col in cells), len(cells)))
    return lines


def select_area_tasks(
    free: set[tuple[int, int]],
    risk_grid: list[list[int]],
    width: int,
    height: int,
) -> list[list[tuple[int, int]]]:
    high = {cell for cell in free if risk_grid[cell[0]][cell[1]] >= 2}
    components = connected_components(high, width, height)
    components.sort(key=lambda comp: (-max(risk_grid[row][col] for row, col in comp), -len(comp)))
    areas: list[list[tuple[int, int]]] = []
    for component in components:
        if len(component) < 12:
            continue
        areas.append(sorted(component)[:140])
    return areas


def fallback_points(
    existing: list[tuple[int, int]],
    free: set[tuple[int, int]],
    risk_grid: list[list[int]],
    target: int,
) -> list[tuple[int, int]]:
    used = set(existing)
    candidates = sorted((cell for cell in free if cell not in used), key=lambda cell: (-risk_grid[cell[0]][cell[1]], cell[0], cell[1]))
    return select_spaced(candidates, count=max(target - len(existing), 0), min_distance=6)


def fallback_lines(
    existing: list[list[tuple[int, int]]],
    free: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
    width: int,
    height: int,
) -> list[list[tuple[int, int]]]:
    del existing
    rows = [height // 4, height // 2, (height * 3) // 4]
    lines: list[list[tuple[int, int]]] = []
    for row in rows:
        cells = [(row, col) for col in range(1, width - 1) if (row, col) in free and (row, col) not in obstacles]
        if len(cells) >= 5:
            lines.append(sparsify(cells, 18))
    return lines


def fallback_areas(
    existing: list[list[tuple[int, int]]],
    free: set[tuple[int, int]],
    risk_grid: list[list[int]],
    width: int,
    height: int,
) -> list[list[tuple[int, int]]]:
    del existing
    candidates = sorted((cell for cell in free if risk_grid[cell[0]][cell[1]] >= 1), key=lambda cell: (-risk_grid[cell[0]][cell[1]], cell[0], cell[1]))
    areas: list[list[tuple[int, int]]] = []
    used: set[tuple[int, int]] = set()
    for center in candidates:
        if center in used:
            continue
        cells = [cell for cell, _ in cells_within(center, radius=4, width=width, height=height) if cell in free]
        if len(cells) >= 8:
            areas.append(sorted(cells))
            used.update(cells)
        if len(areas) >= 4:
            break
    return areas


def point_task(index: int, cell: tuple[int, int], risk: int, source: str) -> dict[str, object]:
    service_time = 3 if risk <= 2 else 4
    deadline = task_deadline(risk, service_time, geometry="point")
    return {
        "id": f"P{index:02d}",
        "type": "osm_buoy_beacon_or_risk_edge_check",
        "cell": list(cell),
        "risk": int(max(1, risk)),
        "service_time": service_time,
        "max_interval": deadline,
        "deadline": deadline,
        "allowed_platforms": ["UAV", "USV"],
        "metadata": {"source": source},
    }


def line_task(index: int, cells: list[tuple[int, int]], risk: int, source: str) -> dict[str, object]:
    service_time = max(5, min(20, len(cells)))
    deadline = task_deadline(risk, service_time, geometry="line")
    return {
        "id": f"L{index:02d}",
        "type": "osm_waterway_or_berth_front_patrol",
        "cells": [list(cell) for cell in cells],
        "risk": int(max(1, risk)),
        "service_time": service_time,
        "max_interval": deadline,
        "deadline": deadline,
        "allowed_platforms": ["UAV", "USV"],
        "metadata": {"source": source, "line_representation": "OSM-derived or water-corridor grid sequence"},
    }


def area_task(index: int, cells: list[tuple[int, int]], risk: int, source: str) -> dict[str, object]:
    service_time = max(8, min(28, len(cells) // 5))
    deadline = task_deadline(risk, service_time, geometry="area")
    return {
        "id": f"A{index:02d}",
        "type": "osm_risk_water_area_coverage",
        "cells": [list(cell) for cell in cells],
        "risk": int(max(1, risk)),
        "service_time": service_time,
        "max_interval": deadline,
        "deadline": deadline,
        "allowed_platforms": ["UAV", "USV"],
        "executor": "mcpp_or_boustrophedon",
        "metadata": {"source": source},
    }


def task_deadline(risk: int, service_time: int, geometry: str) -> int:
    base_by_geometry = {"point": 42, "line": 56, "area": 72}
    risk_tightening = {1: 12, 2: 6, 3: 0}.get(int(risk), 0)
    return max(int(service_time + 24), int(base_by_geometry.get(geometry, 48) + risk_tightening))


def classify_feature(tags: dict[str, str]) -> str | None:
    if is_land_polygon(tags):
        return "land_or_port_polygon"
    if is_risk_line(tags):
        return "waterway_or_harbour_line"
    if is_seamark(tags):
        return "seamark"
    return None


def is_land_polygon(tags: dict[str, str]) -> bool:
    landuse = tags.get("landuse", "")
    man_made = tags.get("man_made", "")
    natural = tags.get("natural", "")
    return (
        landuse in {"industrial", "commercial", "port"}
        or man_made in {"pier", "breakwater", "quay", "jetty", "groyne"}
        or tags.get("harbour") in {"yes", "port", "dock"}
        or natural in {"bare_rock", "island", "land"}
        or "building" in tags
    )


def is_risk_line(tags: dict[str, str]) -> bool:
    return (
        "waterway" in tags
        or tags.get("seamark:type") in {"fairway", "harbour", "anchorage"}
        or tags.get("man_made") in {"breakwater", "pier", "quay", "jetty"}
        or tags.get("harbour") in {"yes", "port", "dock"}
        or "bridge" in tags
    )


def is_seamark(tags: dict[str, str]) -> bool:
    seamark = tags.get("seamark:type", "")
    return seamark in {"buoy", "beacon", "light", "lighthouse", "fairway"} or tags.get("man_made") in {"beacon", "lighthouse"}


def line_label(tags: dict[str, str]) -> str:
    return tags.get("name") or tags.get("seamark:name") or tags.get("waterway") or tags.get("man_made") or "OSM line"


def densify_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(cells) < 2:
        return cells
    result: list[tuple[int, int]] = []
    for first, second in zip(cells, cells[1:]):
        segment = bresenham(first, second)
        if not result:
            result.extend(segment)
        else:
            result.extend(segment[1:])
    return dedupe_cells(result)


def bresenham(first: tuple[int, int], second: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = first
    r1, c1 = second
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    row, col = r0, c0
    while True:
        points.append((row, col))
        if row == r1 and col == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            row += sr
        if e2 < dr:
            err += dr
            col += sc
    return points


def connected_components(cells: set[tuple[int, int]], width: int, height: int) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        component = [start]
        while queue:
            row, col = queue.popleft()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    return components


def cells_within(
    center: tuple[int, int],
    radius: int,
    width: int,
    height: int,
) -> list[tuple[tuple[int, int], int]]:
    row, col = center
    cells: list[tuple[tuple[int, int], int]] = []
    for rr in range(max(0, row - radius), min(height - 1, row + radius) + 1):
        for cc in range(max(0, col - radius), min(width - 1, col + radius) + 1):
            dist = abs(rr - row) + abs(cc - col)
            if dist <= radius:
                cells.append(((rr, cc), dist))
    return cells


def nearest_free(
    cell: tuple[int, int],
    free: set[tuple[int, int]],
    projector: LocalProjector | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[int, int]:
    if cell in free:
        return cell
    if projector is not None:
        width = projector.width
        height = projector.height
    if width is None or height is None:
        raise ValueError("width/height required when projector is not supplied")
    queue = deque([cell])
    seen = {cell}
    while queue:
        current = queue.popleft()
        row, col = current
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if neighbor in seen or not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width):
                continue
            if neighbor in free:
                return neighbor
            seen.add(neighbor)
            queue.append(neighbor)
    raise ValueError("no free cell found")


def select_spaced(candidates: list[tuple[int, int]], count: int, min_distance: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for cell in candidates:
        if cell in selected:
            continue
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= min_distance for other in selected):
            selected.append(cell)
        if len(selected) >= count:
            break
    return selected


def sparsify(cells: list[tuple[int, int]], max_points: int) -> list[tuple[int, int]]:
    if len(cells) <= max_points:
        return cells
    step = max(1, len(cells) // (max_points - 1))
    sampled = cells[::step]
    if sampled[-1] != cells[-1]:
        sampled.append(cells[-1])
    return sampled[:max_points]


def dedupe_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for cell in cells:
        if cell in seen:
            continue
        seen.add(cell)
        result.append(cell)
    return result


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def readme(name: str, bbox: tuple[float, float, float, float], cell_size_m: float) -> str:
    south, west, north, east = bbox
    return f"""# {name}

This dataset is generated from OpenStreetMap data downloaded through the Overpass API.

- BBox: south={south}, west={west}, north={north}, east={east}
- Cell size: {cell_size_m:g} m
- Scope: expanded Yangshan Deep-Water Port water-surface inspection research grid
- License note: OSM data is available under the Open Database License (ODbL).
- Safety note: this is not a nautical chart and must not be used for navigation.

The grid is no longer a hand-drawn random/prototype layout. Port land, pier,
breakwater, harbour, waterway, and seamark-related OSM features are rasterized
or used as task-generation anchors. OSM source features are also retained in
grid metadata under `visual_features` so the extracted linear and polygon
layers can be audited and rendered explicitly.
"""


if __name__ == "__main__":
    main()
