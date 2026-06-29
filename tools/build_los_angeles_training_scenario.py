from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCENARIO_NAME = "los_angeles_training_v1"
DEFAULT_OUTPUT_DIR = Path("data/ports") / SCENARIO_NAME
CELL_SIZE_M = 250.0
LAT0 = 33.735
LON0 = -118.255
SCRIPT_VERSION = "official-noaa-rest-v2"
USER_AGENT = "Codex LA-port official NOAA geometry scenario builder"
LA_BBOX = {
    "xmin": -118.33,
    "ymin": 33.67,
    "xmax": -118.18,
    "ymax": 33.80,
}
NOAA_LICENSE_NOTE = (
    "NOAA ENC Direct to GIS public service; not intended for navigation. "
    "Geometry is from NOAA Office of Coast Survey REST services."
)
OFFICIAL_SNAPSHOT_ACCESS_DATE = "2026-06-29"


@dataclass(frozen=True)
class LayerSpec:
    service_name: str
    service_url: str
    layer_id: int
    layer_name: str
    geometry: str
    task_family: str
    object_type: str
    risk: int
    deadline: int
    service_time: int
    limit: int
    name_filter: str | None = None

    @property
    def layer_url(self) -> str:
        return f"{self.service_url}/{self.layer_id}"


NOAA_HARBOUR = "https://encdirect.noaa.gov/arcgis/rest/services/encdirect/enc_harbour/MapServer"
NOAA_APPROACH = "https://encdirect.noaa.gov/arcgis/rest/services/encdirect/enc_approach/MapServer"

OFFICIAL_LAYERS: list[LayerSpec] = [
    LayerSpec(
        "NOAA ENC Direct Harbour",
        NOAA_HARBOUR,
        6,
        "Harbor.Buoy_Lateral_point",
        "point",
        "WATERSIDE_ASSET_INSPECTION",
        "navigation_aid",
        3,
        300,
        2,
        6,
    ),
    LayerSpec(
        "NOAA ENC Direct Harbour",
        NOAA_HARBOUR,
        11,
        "Harbor.Light_point",
        "point",
        "WATERSIDE_ASSET_INSPECTION",
        "navigation_aid",
        2,
        420,
        2,
        6,
    ),
    LayerSpec(
        "NOAA ENC Direct Harbour",
        NOAA_HARBOUR,
        87,
        "Harbor.Bridge_line",
        "line",
        "WATERSIDE_ASSET_INSPECTION",
        "bridge_waterline",
        3,
        360,
        5,
        4,
    ),
    LayerSpec(
        "NOAA ENC Direct Harbour",
        NOAA_HARBOUR,
        134,
        "Harbor.Recommended_Track_line",
        "line",
        "HYDROGRAPHIC_SURVEY",
        "recommended_track",
        3,
        300,
        7,
        4,
    ),
    LayerSpec(
        "NOAA ENC Direct Approach",
        NOAA_APPROACH,
        139,
        "Approach.Recommended_Track_line",
        "line",
        "HYDROGRAPHIC_SURVEY",
        "approach_recommended_track",
        3,
        300,
        7,
        4,
    ),
    LayerSpec(
        "NOAA ENC Direct Harbour",
        NOAA_HARBOUR,
        104,
        "Harbor.Depth_Contour_line",
        "line",
        "HYDROGRAPHIC_SURVEY",
        "depth_contour",
        2,
        480,
        6,
        4,
    ),
    LayerSpec(
        "NOAA ENC Direct Approach",
        NOAA_APPROACH,
        242,
        "Approach.Sea_Area_Named_Water_Area",
        "area",
        "SURFACE_SAFETY_PATROL",
        "named_water_area",
        2,
        420,
        8,
        6,
        name_filter="Harbor|Channel|Basin",
    ),
    LayerSpec(
        "NOAA ENC Direct Harbour",
        NOAA_HARBOUR,
        237,
        "Harbor.Sea_Area_Named_Water_Area",
        "area",
        "SURFACE_SAFETY_PATROL",
        "harbor_named_water_area",
        2,
        420,
        8,
        4,
        name_filter="Harbor|Channel|Basin",
    ),
    LayerSpec(
        "NOAA ENC Direct Harbour",
        NOAA_HARBOUR,
        208,
        "Harbor.Fairway_area",
        "area",
        "HYDROGRAPHIC_SURVEY",
        "fairway",
        3,
        360,
        9,
        4,
    ),
    LayerSpec(
        "NOAA ENC Direct Approach",
        NOAA_APPROACH,
        233,
        "Approach.Dredged_Area",
        "area",
        "HYDROGRAPHIC_SURVEY",
        "dredged_area",
        3,
        360,
        10,
        4,
    ),
]

EMBEDDED_OFFICIAL_SNAPSHOT: dict[tuple[str, int], list[dict[str, Any]]] = {
    (NOAA_HARBOUR, 6): [
        {
            "attributes": {
                "OBJECTID": 2845058,
                "OBJL": 17,
                "BOYSHP": "pillar",
                "CATLAM": 2,
                "OBJNAM": "Los Angeles Approach Channel Lighted Buoy 2",
                "SCAMIN": 29999,
                "SORDAT": "20000829",
                "SORIND": "US,US,reprt,11thCGD,LNM 35/00",
                "DSNM": "US5LGBBD.000",
            },
            "geometry": {"x": -118.2289833, "y": 33.673325},
        },
        {
            "attributes": {
                "OBJECTID": 2845060,
                "OBJL": 17,
                "BOYSHP": "pillar",
                "CATLAM": 2,
                "OBJNAM": "Point Fermin Lighted Whistle Buoy 6PF",
                "SCAMIN": 29999,
                "SORDAT": "20041208",
                "SORIND": "US,US,reprt,11thCGD,ATONIS",
                "DSNM": "US5LGBCC.000",
            },
            "geometry": {"x": -118.2916822, "y": 33.6988253},
        },
        {
            "attributes": {
                "OBJECTID": 2845061,
                "OBJL": 17,
                "BOYSHP": "pillar",
                "CATLAM": 1,
                "OBJNAM": "Terminal Island Channel Lighted Buoy 5",
                "SCAMIN": 29999,
                "SORDAT": "20030325",
                "SORIND": "US,US,reprt,11thCGD,LNM 12/03",
                "DSNM": "US5LGBCD.000",
            },
            "geometry": {"x": -118.2656596, "y": 33.7293432},
        },
    ],
    (NOAA_HARBOUR, 87): [
        {
            "attributes": {
                "OBJECTID": 409663,
                "OBJL": 11,
                "CATBRG": "footbridge",
                "INFORM": "Catwalk",
                "SCAMIN": 44999,
                "SORDAT": "20130416",
                "SORIND": "US,US,graph,GC-11100",
                "DSNM": "US5LGBCD.000",
                "SHAPE.LEN": 0.00023125661071632102,
            },
            "geometry": {
                "paths": [
                    [
                        [-118.2613683, 33.7314073],
                        [-118.2614562, 33.7316212],
                    ]
                ]
            },
        }
    ],
    (NOAA_HARBOUR, 134): [
        {
            "attributes": {
                "OBJECTID": 363104,
                "OBJL": 109,
                "CATTRK": "based on a system of fixed marks",
                "ORIENT": 295.8,
                "TRAFIC": 3,
                "SCAMIN": 29999,
                "SORDAT": "20120925",
                "SORIND": "US,US,reprt,11thCGD,LNM 39/12",
                "DSNM": "US5LGBCD.000",
                "SHAPE.LEN": 0.021522463676122199,
            },
            "geometry": {
                "paths": [
                    [
                        [-118.2482006, 33.7102109],
                        [-118.2682243, 33.7181019],
                    ]
                ]
            },
        }
    ],
    (NOAA_APPROACH, 139): [
        {
            "attributes": {
                "OBJL": 109,
                "CATTRK": "based on a system of fixed marks",
                "ORIENT": 355.3,
                "TRAFIC": 3,
                "SCAMIN": 179999,
                "SORDAT": "20120925",
                "SORIND": "US,US,reprt,11thCGD,LNM 39/12",
                "DSNM": "US4CA60M.000",
                "APPROACH.RECTRC_LINE.FID": 223275,
                "SHAPE.LEN": 0.030189325176293701,
            },
            "geometry": {
                "paths": [
                    [
                        [-118.1806531, 33.6924563],
                        [-118.1836259, 33.7224989],
                    ]
                ]
            },
        }
    ],
    (NOAA_APPROACH, 242): [
        {
            "attributes": {
                "OBJL": 119,
                "CATSEA": " ",
                "OBJNAM": "Fish Harbor",
                "SCAMIN": 89999,
                "SORDAT": "20090410",
                "SORIND": "US,US,graph,Chart 18746",
                "DSNM": "US4CA60M.000",
                "APPROACH.SEAARE_POLYGON.FID": 2476641,
                "SHAPE.AREA": 1.5639973375e-05,
                "SHAPE.LEN": 0.016865859421287901,
            },
            "geometry": {
                "rings": [
                    [
                        [-118.2655495, 33.7353564],
                        [-118.2674783, 33.7338497],
                        [-118.2684003, 33.7340868],
                        [-118.2701792, 33.7362104],
                        [-118.270755, 33.7367559],
                        [-118.2685157, 33.7377018],
                        [-118.2662955, 33.7386072],
                        [-118.2652561, 33.73684],
                        [-118.2655495, 33.7353564],
                    ]
                ]
            },
        },
        {
            "attributes": {
                "OBJL": 119,
                "CATSEA": "sea channel",
                "OBJNAM": "Flood Control Channel",
                "SCAMIN": 89999,
                "SORDAT": "20090410",
                "SORIND": "US,US,graph,Chart 18746",
                "DSNM": "US4CA60M.000",
                "APPROACH.SEAARE_POLYGON.FID": 2476663,
                "SHAPE.AREA": 2.475201237e-05,
                "SHAPE.LEN": 0.033121862834592702,
            },
            "geometry": {
                "rings": [
                    [
                        [-118.206222, 33.7671409],
                        [-118.2062109, 33.7687868],
                        [-118.2061957, 33.7720501],
                        [-118.2061944, 33.7772701],
                        [-118.2061305, 33.7820062],
                        [-118.2045622, 33.7820985],
                        [-118.2045297, 33.7741134],
                        [-118.2045235, 33.7674311],
                        [-118.206222, 33.7671409],
                    ]
                ]
            },
        },
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Los Angeles port scheduler training scenario from official NOAA data.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bbox", default=None, help="Optional xmin,ymin,xmax,ymax override in EPSG:4326.")
    parser.add_argument(
        "--use-embedded-official-snapshot",
        action="store_true",
        help="Use the checked-in NOAA official feature sample captured on 2026-06-29 instead of live REST queries.",
    )
    args = parser.parse_args()

    bbox = _parse_bbox(args.bbox) if args.bbox else dict(LA_BBOX)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    access_date = OFFICIAL_SNAPSHOT_ACCESS_DATE if args.use_embedded_official_snapshot else dt.date.today().isoformat()
    objects, source_summary = _fetch_management_objects(bbox, access_date, args.use_embedded_official_snapshot)
    _require_mixed_official_geometry(objects)
    bounds = _bounds(objects)
    grid, tasks = _build_scenario(objects, bounds, bbox, source_summary, access_date)

    grid_path = output_dir / f"{SCENARIO_NAME}_grid.json"
    tasks_path = output_dir / f"{SCENARIO_NAME}_tasks.json"
    readme_path = output_dir / "README.md"
    grid_path.write_text(json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8")
    tasks_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(
        _readme(len(tasks["point_tasks"]), len(tasks["line_tasks"]), len(tasks["area_tasks"]), access_date),
        encoding="utf-8",
    )
    print(f"grid={grid_path}")
    print(f"tasks={tasks_path}")
    print(f"readme={readme_path}")
    print(f"official_objects={len(objects)}")


def _fetch_management_objects(
    bbox: dict[str, float],
    access_date: str,
    use_embedded_snapshot: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    objects: list[dict[str, Any]] = []
    source_summary: list[dict[str, Any]] = []
    for spec in OFFICIAL_LAYERS:
        features = _embedded_features(spec) if use_embedded_snapshot else _query_layer(spec, bbox)
        selected = _select_features(spec, features, bbox)
        source_summary.append(
            {
                "service_name": spec.service_name,
                "service_url": spec.service_url,
                "layer_id": spec.layer_id,
                "layer_name": spec.layer_name,
                "source_url": spec.layer_url,
                "queried_feature_count": len(features),
                "selected_feature_count": len(selected),
                "access_date": access_date,
                "original_crs": "EPSG:4326",
                "usage_terms": NOAA_LICENSE_NOTE,
                "retrieval_mode": "embedded_official_snapshot" if use_embedded_snapshot else "live_rest_query",
            }
        )
        for feature in selected:
            obj = _feature_to_object(spec, feature, bbox, access_date)
            if obj is not None:
                objects.append(obj)
    return _dedupe_objects(objects), source_summary


def _embedded_features(spec: LayerSpec) -> list[dict[str, Any]]:
    return list(EMBEDDED_OFFICIAL_SNAPSHOT.get((spec.service_url, spec.layer_id), []))


def _query_layer(spec: LayerSpec, bbox: dict[str, float]) -> list[dict[str, Any]]:
    params = {
        "f": "json",
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "geometry": json.dumps(
            {
                "xmin": bbox["xmin"],
                "ymin": bbox["ymin"],
                "xmax": bbox["xmax"],
                "ymax": bbox["ymax"],
                "spatialReference": {"wkid": 4326},
            },
            separators=(",", ":"),
        ),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outSR": "4326",
        "resultRecordCount": str(max(spec.limit * 4, 12)),
    }
    data = _post_json(f"{spec.layer_url}/query", params)
    if "error" in data:
        details = data["error"].get("details") or []
        detail_text = "; ".join(str(detail) for detail in details)
        raise RuntimeError(f"NOAA query failed for {spec.layer_name}: {data['error'].get('message')} {detail_text}")
    return list(data.get("features", []))


def _post_json(url: str, params: dict[str, str]) -> dict[str, Any]:
    # ArcGIS MapServer query endpoints are more reliable for NOAA's public
    # service as simple GET requests than as form POSTs from Python.
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(5):
        request = urllib.request.Request(
            full_url,
            headers={
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network resilience path
            last_error = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"NOAA request failed after retries: {full_url}") from last_error


def _select_features(spec: LayerSpec, features: list[dict[str, Any]], bbox: dict[str, float]) -> list[dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for feature in features:
        coordinates = _feature_coordinates(spec.geometry, feature.get("geometry", {}), bbox)
        if spec.geometry == "point":
            point = coordinates[0] if coordinates else None
        else:
            point = _centroid(coordinates)
        if point is None or not _inside_bbox(point[0], point[1], bbox):
            continue
        name = _feature_name(feature)
        if spec.name_filter and not _matches_name_filter(name, spec.name_filter):
            continue
        distance = _distance_m(point, (LON0, LAT0))
        candidates.append((distance, feature))
    candidates.sort(key=lambda item: (_feature_sort_name(item[1]), item[0]))
    return [feature for _, feature in candidates[: spec.limit]]


def _feature_to_object(
    spec: LayerSpec,
    feature: dict[str, Any],
    bbox: dict[str, float],
    access_date: str,
) -> dict[str, Any] | None:
    coordinates = _feature_coordinates(spec.geometry, feature.get("geometry", {}), bbox)
    if spec.geometry == "point" and len(coordinates) != 1:
        return None
    if spec.geometry in {"line", "area"} and len(coordinates) < 2:
        return None
    attributes = dict(feature.get("attributes", {}))
    source_hash = _feature_checksum(feature)
    object_id = _official_object_id(spec, attributes, source_hash)
    source_name = _feature_name(feature) or spec.layer_name
    return {
        "id": object_id,
        "name": source_name,
        "task_family": spec.task_family,
        "object_type": spec.object_type,
        "geometry": spec.geometry,
        "coordinates": coordinates,
        "risk": spec.risk,
        "deadline": spec.deadline,
        "service_time": spec.service_time,
        "release_mode": "SCHEDULED",
        "provenance": _provenance(spec, attributes, source_hash, access_date),
    }


def _feature_coordinates(geometry: str, raw_geometry: dict[str, Any], bbox: dict[str, float]) -> list[tuple[float, float]]:
    if geometry == "point":
        lon = raw_geometry.get("x")
        lat = raw_geometry.get("y")
        if isinstance(lon, (int, float)) and isinstance(lat, (int, float)) and _inside_bbox(lon, lat, bbox):
            return [(float(lon), float(lat))]
        return []
    if geometry == "line":
        coordinates = []
        for path in raw_geometry.get("paths", []):
            coordinates.extend(_clip_sequence_to_bbox(path, bbox))
        return _simplify_coordinates(_dedupe_coords(coordinates), max_points=12)
    coordinates = []
    for ring in raw_geometry.get("rings", []):
        coordinates.extend(_clip_sequence_to_bbox(ring, bbox))
    return _simplify_coordinates(_dedupe_coords(coordinates), max_points=16)


def _clip_sequence_to_bbox(sequence: Iterable[Iterable[float]], bbox: dict[str, float]) -> list[tuple[float, float]]:
    clipped: list[tuple[float, float]] = []
    for point in sequence:
        try:
            lon = float(point[0])
            lat = float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if _inside_bbox(lon, lat, bbox):
            clipped.append((lon, lat))
    return clipped


def _inside_bbox(lon: float, lat: float, bbox: dict[str, float]) -> bool:
    return bbox["xmin"] <= lon <= bbox["xmax"] and bbox["ymin"] <= lat <= bbox["ymax"]


def _simplify_coordinates(coordinates: list[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    if len(coordinates) <= max_points:
        return coordinates
    step = (len(coordinates) - 1) / float(max_points - 1)
    return [coordinates[round(index * step)] for index in range(max_points)]


def _feature_name(feature: dict[str, Any]) -> str:
    attributes = feature.get("attributes", {})
    for key in ("OBJNAM", "INFORM", "CATSEA", "CATTRK", "CATBRG"):
        value = attributes.get(key)
        if value and str(value).strip():
            return str(value).strip()
    object_id = attributes.get("OBJECTID") or _first_fid(attributes)
    return f"NOAA object {object_id}" if object_id is not None else ""


def _feature_sort_name(feature: dict[str, Any]) -> str:
    return _feature_name(feature).lower()


def _matches_name_filter(name: str, pattern: str) -> bool:
    options = [part.lower() for part in pattern.split("|")]
    lowered = name.lower()
    return any(option in lowered for option in options)


def _first_fid(attributes: dict[str, Any]) -> Any:
    for key, value in attributes.items():
        if key.endswith(".FID"):
            return value
    return None


def _official_object_id(spec: LayerSpec, attributes: dict[str, Any], source_hash: str) -> str:
    raw_id = attributes.get("OBJECTID") or _first_fid(attributes) or source_hash[:10]
    safe_layer = spec.layer_name.replace(".", "-").replace("_", "-").upper()
    return f"NOAA-{safe_layer}-{raw_id}"


def _feature_checksum(feature: dict[str, Any]) -> str:
    payload = json.dumps(feature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provenance(spec: LayerSpec, attributes: dict[str, Any], source_hash: str, access_date: str) -> dict[str, Any]:
    return {
        "source_dataset": spec.service_name,
        "source_agency": "NOAA Office of Coast Survey",
        "source_date": attributes.get("SORDAT"),
        "source_url": spec.layer_url,
        "source_version_or_edition": attributes.get("DSNM") or "ENC Direct weekly REST service",
        "access_date": access_date,
        "license_or_usage_terms": NOAA_LICENSE_NOTE,
        "original_id": attributes.get("OBJECTID") or _first_fid(attributes),
        "original_crs": "EPSG:4326",
        "file_checksum": source_hash,
        "processing_script_version": SCRIPT_VERSION,
        "processing_note": (
            "Official NOAA geometry was clipped to the Los Angeles training bbox and converted to a "
            "250 m local grid. Scheduling workload, deadlines, risk, and release settings are training "
            "parameters, not official port work orders."
        ),
        "source_layer_id": spec.layer_id,
        "source_layer_name": spec.layer_name,
        "source_attributes": {
            "OBJNAM": attributes.get("OBJNAM"),
            "INFORM": attributes.get("INFORM"),
            "SORDAT": attributes.get("SORDAT"),
            "SORIND": attributes.get("SORIND"),
            "DSNM": attributes.get("DSNM"),
        },
    }


def _build_scenario(
    objects: list[dict[str, Any]],
    bounds: dict[str, float],
    bbox: dict[str, float],
    source_summary: list[dict[str, Any]],
    access_date: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    width = int(math.ceil(bounds["width_m"] / CELL_SIZE_M)) + 1
    height = int(math.ceil(bounds["height_m"] / CELL_SIZE_M)) + 1
    depot = _derived_depot_cell(objects, bounds)
    free_cells = [[row, col] for row in range(height) for col in range(width)]
    risk_grid = [[0 for _ in range(width)] for _ in range(height)]
    tasks = {"metadata": _scenario_metadata(bounds, width, height, bbox, source_summary, access_date), "point_tasks": [], "line_tasks": [], "area_tasks": []}

    for obj in objects:
        if obj["geometry"] == "point":
            cells = [_to_cell(*obj["coordinates"][0], bounds)]
        elif obj["geometry"] == "line":
            cells = _line_cells(obj["coordinates"], bounds)
        else:
            cells = _area_cells(obj["coordinates"], bounds)
        if not cells:
            continue
        if obj["geometry"] in {"line", "area"} and len(cells) < 2:
            continue
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
        "description": "Los Angeles port scheduler training scenario from official NOAA ENC Direct geometry.",
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
            "depot_note": "Training depot is derived from official NOAA management-object centroids; final experiments still require approved recovery points.",
        },
    }
    return grid, tasks


def _task(obj: dict[str, Any], cells: list[tuple[int, int]]) -> dict[str, Any]:
    service_time = int(obj["service_time"])
    geometry = obj["geometry"]
    provenance = obj["provenance"]
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
            "geometry_source_status": "official_noaa_geometry",
            "work_order_status": "training_task_derived_not_official_work_order",
            "parameter_status": "training_parameters_not_official_operational_schedule",
            "object_name": obj["name"],
            "longitude": _centroid(obj["coordinates"])[0] if obj["coordinates"] else None,
            "latitude": _centroid(obj["coordinates"])[1] if obj["coordinates"] else None,
            **provenance,
        },
    }
    if geometry == "point":
        task["cell"] = list(cells[0])
    else:
        task["cells"] = [list(cell) for cell in cells]
    return task


def _bounds(objects: list[dict[str, Any]]) -> dict[str, float]:
    coords: list[tuple[float, float]] = []
    for obj in objects:
        coords.extend(obj["coordinates"])
    if not coords:
        raise RuntimeError("No official NOAA geometry was selected for the Los Angeles scenario.")
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
    area_objects = [obj for obj in objects if obj["geometry"] == "area" and "fish harbor" in obj["name"].lower()]
    source = area_objects[0] if area_objects else min(objects, key=lambda obj: _distance_m(_centroid(obj["coordinates"]), (LON0, LAT0)))
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


def _dedupe_objects(objects: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for obj in objects:
        if obj["id"] in seen:
            continue
        seen.add(obj["id"])
        result.append(obj)
    return result


def _centroid(coordinates: list[tuple[float, float]]) -> tuple[float, float]:
    if not coordinates:
        return (LON0, LAT0)
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


def _scenario_metadata(
    bounds: dict[str, float],
    width: int,
    height: int,
    bbox: dict[str, float],
    source_summary: list[dict[str, Any]],
    access_date: str,
) -> dict[str, Any]:
    return {
        "scenario_name": SCENARIO_NAME,
        "port": "Los Angeles",
        "contract_status": "PENDING_OFFICIAL_GEOMETRY_TRAINING",
        "scenario_generated": True,
        "official_geometry": True,
        "cell_size_m": CELL_SIZE_M,
        "access_date": access_date,
        "official_query_bbox_epsg4326": bbox,
        "bounds_lon_lat": {
            "lon_min": bounds["lon_min"],
            "lon_max": bounds["lon_max"],
            "lat_min": bounds["lat_min"],
            "lat_max": bounds["lat_max"],
        },
        "grid_shape": [height, width],
        "source_layers": source_summary,
        "note": (
            "All task geometry is derived from official NOAA ENC Direct REST layers. "
            "Training workload/deadline/risk parameters are scenario parameters and are not official work orders."
        ),
    }


def _require_mixed_official_geometry(objects: list[dict[str, Any]]) -> None:
    geometries = {obj["geometry"] for obj in objects}
    missing = {"point", "line", "area"} - geometries
    if missing:
        raise RuntimeError(f"Official NOAA query did not return required geometry classes: {sorted(missing)}")
    for geometry in ("point", "line", "area"):
        count = sum(1 for obj in objects if obj["geometry"] == geometry)
        if count < 2:
            raise RuntimeError(f"Official NOAA query returned too few {geometry} objects: {count}")


def _parse_bbox(raw: str) -> dict[str, float]:
    parts = [float(part.strip()) for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be xmin,ymin,xmax,ymax")
    return {"xmin": parts[0], "ymin": parts[1], "xmax": parts[2], "ymax": parts[3]}


def _readme(point_count: int, line_count: int, area_count: int, access_date: str) -> str:
    return f"""# {SCENARIO_NAME}

This is a Los Angeles port scheduler training scenario built from official NOAA ENC Direct geometry.

Status: `PENDING_OFFICIAL_GEOMETRY_TRAINING`.

The geometry is sourced from NOAA Office of Coast Survey ENC Direct REST services for the Los Angeles port
area. This checked-in scenario was regenerated from the embedded official sample snapshot captured on
2026-06-29 because live network execution was unavailable during the update. The generated scheduler tasks
are derived from official chart objects, but the workload, deadlines, risk, and release settings remain
training parameters rather than official Port of Los Angeles work orders. Do not report this as final
experiment evidence until the V1.2 algorithm contract and official experiment workflow are frozen.

- Point tasks: {point_count}
- Corridor tasks: {line_count}
- Area tasks: {area_count}
- Geometry source: NOAA ENC Direct Harbour and Approach REST services
- Access date: {access_date}
- Coordinate mode: local equirectangular approximation, `distance_mode=utm_euclidean`
- Cell size: {CELL_SIZE_M:.0f} m

Regenerate from official NOAA services:

```powershell
.\\.venv\\Scripts\\python.exe tools\\build_los_angeles_training_scenario.py
```

Regenerate from the embedded official NOAA sample snapshot:

```powershell
.\\.venv\\Scripts\\python.exe tools\\build_los_angeles_training_scenario.py --use-embedded-official-snapshot
```

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
