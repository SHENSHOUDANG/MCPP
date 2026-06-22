from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import struct
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties, fontManager
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OSM_RAW = ROOT / "data/ports/shanghai_yangshan_osm_v1/osm_overpass_raw.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/port_inspection/shanghai_yangshan_osm_v1/stage2_gis"
DEFAULT_BACKGROUND = ROOT / "outputs/real_map_tiles/yangshan_arcgis_imagery_z14.png"
STUDY_BBOX = {
    "west": 121.985,
    "south": 30.585,
    "east": 122.165,
    "north": 30.675,
}
IMAGERY_BBOX = {
    "west": 122.010,
    "south": 30.600,
    "east": 122.100,
    "north": 30.655,
}
WGS84 = 4326
UTM51N = 32651


@dataclass(slots=True)
class Feature:
    source_dataset: str
    source_feature_type: str
    source_feature_id: str
    osm_tags_json: str
    geometry_type: str
    verification_level: str
    extraction_rule: str
    layer_class: str
    seamark_type: str
    seamark_name: str
    geom: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract stage-2 Yangshan GIS audit layers into a GeoPackage.")
    parser.add_argument("--osm-raw", default=str(DEFAULT_OSM_RAW))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--background", default=str(DEFAULT_BACKGROUND))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    osm = json.loads(Path(args.osm_raw).read_text(encoding="utf-8"))

    extracted = extract_layers(osm, STUDY_BBOX)
    geopackage = output_dir / "yangshan_stage2.gpkg"
    write_geopackage(geopackage, extracted)

    imagery_metadata = write_imagery_metadata(output_dir, Path(args.background))
    registration_png = output_dir / "01_registration_check.png"
    render_registration_check(
        background=Path(args.background),
        output=registration_png,
        shoreline=extracted["shoreline_osm_raw"],
        quay=extracted["quay_front_raw"],
        breakwater=extracted["breakwater_raw"],
    )

    checks = run_quality_checks(geopackage, extracted)
    write_qgis_layer_guide(output_dir / "qgis_layer_guide.md", imagery_metadata)
    write_extraction_report(output_dir / "osm_extraction_report.md", extracted, checks, imagery_metadata)
    print(geopackage.resolve())
    print((output_dir / "qgis_layer_guide.md").resolve())
    print((output_dir / "osm_extraction_report.md").resolve())
    print(registration_png.resolve())


def extract_layers(osm: dict[str, Any], bbox: dict[str, float]) -> dict[str, list[Feature] | dict[str, Any]]:
    raw_features: list[Feature] = []
    shoreline: list[Feature] = []
    quay: list[Feature] = []
    breakwater: list[Feature] = []
    seamark: list[Feature] = []

    for element in osm.get("elements", []):
        if not isinstance(element, dict):
            continue
        tags = dict(element.get("tags") or {})
        if not tags or not is_possible_related(tags):
            continue
        geom = element_geometry(element, tags)
        if geom is None:
            continue
        clipped = clip_geometry(geom, bbox)
        if clipped is None:
            continue

        base = make_feature(element, tags, clipped, "osm_raw_features", "all_related_osm_tags")
        raw_features.append(base)
        if is_shoreline_candidate(tags):
            shoreline.append(make_feature(element, tags, clipped, "shoreline_osm_raw", "natural_or_water_boundary_tag"))
        if is_quay_candidate(tags):
            quay.append(make_feature(element, tags, clipped, "quay_front_raw", "quay_pier_jetty_harbour_or_port_tag"))
        if is_breakwater_candidate(tags):
            breakwater.append(make_feature(element, tags, clipped, "breakwater_raw", "breakwater_or_groyne_tag"))
        if is_seamark_candidate(tags):
            seamark.append(make_feature(element, tags, clipped, "seamark_raw", "seamark_tag"))

    study_geom = bbox_polygon(bbox)
    return {
        "study_area": {
            "area_id": "yangshan_stage2_study_area",
            "source": "stage1_adopted_osm_bbox",
            "bbox_west": bbox["west"],
            "bbox_south": bbox["south"],
            "bbox_east": bbox["east"],
            "bbox_north": bbox["north"],
            "crs": "EPSG:4326",
            "geom": study_geom,
        },
        "osm_raw_features": raw_features,
        "shoreline_osm_raw": shoreline,
        "quay_front_raw": quay,
        "breakwater_raw": breakwater,
        "seamark_raw": seamark,
        "shoreline_osm_utm": [transform_feature(feature, UTM51N) for feature in shoreline],
        "quay_front_utm": [transform_feature(feature, UTM51N) for feature in quay],
        "breakwater_utm": [transform_feature(feature, UTM51N) for feature in breakwater],
        "seamark_utm": [transform_feature(feature, UTM51N) for feature in seamark],
    }


def make_feature(element: dict[str, Any], tags: dict[str, str], geom: dict[str, Any], layer_class: str, rule: str) -> Feature:
    seamark_type = str(tags.get("seamark:type", ""))
    seamark_name = str(tags.get("seamark:name", tags.get("name", "")))
    return Feature(
        source_dataset="OpenStreetMap Overpass API",
        source_feature_type=str(element.get("type", "")),
        source_feature_id=str(element.get("id", "")),
        osm_tags_json=json.dumps(tags, ensure_ascii=False, sort_keys=True),
        geometry_type=str(geom["type"]),
        verification_level="osm_only" if is_seamark_candidate(tags) else "unverified",
        extraction_rule=rule,
        layer_class=layer_class,
        seamark_type=seamark_type,
        seamark_name=seamark_name,
        geom=geom,
    )


def transform_feature(feature: Feature, target_srs: int) -> Feature:
    if target_srs != UTM51N:
        raise ValueError("only EPSG:32651 is supported by the stage-2 extractor")
    return Feature(
        source_dataset=feature.source_dataset,
        source_feature_type=feature.source_feature_type,
        source_feature_id=feature.source_feature_id,
        osm_tags_json=feature.osm_tags_json,
        geometry_type=feature.geometry_type,
        verification_level=feature.verification_level,
        extraction_rule=feature.extraction_rule + "_projected_to_epsg_32651",
        layer_class=feature.layer_class.replace("_raw", "_utm"),
        seamark_type=feature.seamark_type,
        seamark_name=feature.seamark_name,
        geom=transform_geometry(feature.geom, lonlat_to_utm51),
    )


def is_possible_related(tags: dict[str, str]) -> bool:
    return (
        is_seamark_candidate(tags)
        or tags.get("man_made") in {"quay", "pier", "jetty", "breakwater", "groyne", "lighthouse", "beacon"}
        or "harbour" in tags
        or "port" in tags
        or tags.get("landuse") in {"industrial", "commercial", "port", "harbour"}
        or tags.get("natural") in {"coastline", "water", "bay", "strait", "bare_rock", "island", "land"}
        or "waterway" in tags
        or "bridge" in tags
        or "building" in tags
        or "highway" in tags
    )


def is_shoreline_candidate(tags: dict[str, str]) -> bool:
    return (
        tags.get("natural") in {"coastline", "water", "bay", "strait", "bare_rock", "island", "land"}
        or tags.get("place") in {"island", "islet"}
    )


def is_quay_candidate(tags: dict[str, str]) -> bool:
    return (
        tags.get("man_made") in {"quay", "pier", "jetty"}
        or "harbour" in tags
        or "port" in tags
        or tags.get("landuse") in {"harbour", "port"}
    )


def is_breakwater_candidate(tags: dict[str, str]) -> bool:
    return tags.get("man_made") in {"breakwater", "groyne"}


def is_seamark_candidate(tags: dict[str, str]) -> bool:
    return any(str(key).startswith("seamark:") or key == "seamark:type" for key in tags)


def element_geometry(element: dict[str, Any], tags: dict[str, str]) -> dict[str, Any] | None:
    if element.get("type") == "node" and "lat" in element and "lon" in element:
        return {"type": "Point", "coordinates": (float(element["lon"]), float(element["lat"]))}
    if isinstance(element.get("geometry"), list):
        coords = coords_from_osm_geometry(element["geometry"])
        if len(coords) == 1:
            return {"type": "Point", "coordinates": coords[0]}
        if len(coords) >= 4 and is_closed(coords) and is_area_tags(tags):
            return {"type": "Polygon", "coordinates": [ensure_closed(coords)]}
        if len(coords) >= 2:
            return {"type": "LineString", "coordinates": coords}
    member_geoms: list[dict[str, Any]] = []
    for index, member in enumerate(element.get("members", []) or []):
        if not isinstance(member, dict) or not isinstance(member.get("geometry"), list):
            continue
        coords = coords_from_osm_geometry(member["geometry"])
        if len(coords) >= 4 and is_closed(coords) and is_area_tags(tags):
            member_geoms.append({"type": "Polygon", "coordinates": [ensure_closed(coords)]})
        elif len(coords) >= 2:
            member_geoms.append({"type": "LineString", "coordinates": coords})
        elif len(coords) == 1:
            member_geoms.append({"type": "Point", "coordinates": coords[0]})
        del index
    if not member_geoms:
        return None
    return {"type": "GeometryCollection", "geometries": member_geoms}


def coords_from_osm_geometry(raw: list[dict[str, Any]]) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for point in raw:
        if "lat" in point and "lon" in point:
            coords.append((float(point["lon"]), float(point["lat"])))
    return coords


def is_area_tags(tags: dict[str, str]) -> bool:
    if tags.get("area") == "yes":
        return True
    if "landuse" in tags or "building" in tags:
        return True
    if tags.get("natural") in {"water", "bay", "strait", "bare_rock", "island", "land"}:
        return True
    if tags.get("man_made") in {"pier", "breakwater", "quay", "jetty", "groyne"}:
        return True
    if "harbour" in tags or "port" in tags:
        return True
    return False


def is_closed(coords: list[tuple[float, float]]) -> bool:
    return len(coords) >= 2 and coords[0] == coords[-1]


def ensure_closed(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return coords if is_closed(coords) else coords + [coords[0]]


def bbox_polygon(bbox: dict[str, float]) -> dict[str, Any]:
    ring = [
        (bbox["west"], bbox["south"]),
        (bbox["east"], bbox["south"]),
        (bbox["east"], bbox["north"]),
        (bbox["west"], bbox["north"]),
        (bbox["west"], bbox["south"]),
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def clip_geometry(geom: dict[str, Any], bbox: dict[str, float]) -> dict[str, Any] | None:
    gtype = geom["type"]
    if gtype == "Point":
        return geom if point_in_bbox(geom["coordinates"], bbox) else None
    if gtype == "LineString":
        return clip_linestring(geom["coordinates"], bbox)
    if gtype == "Polygon":
        return clip_polygon(geom["coordinates"], bbox)
    if gtype == "GeometryCollection":
        clipped = [clip_geometry(item, bbox) for item in geom.get("geometries", [])]
        geometries = [item for item in clipped if item is not None]
        if not geometries:
            return None
        return {"type": "GeometryCollection", "geometries": geometries}
    if gtype == "MultiLineString":
        lines = []
        for line in geom["coordinates"]:
            clipped = clip_linestring(line, bbox)
            if clipped is not None:
                if clipped["type"] == "LineString":
                    lines.append(clipped["coordinates"])
                else:
                    lines.extend(clipped["coordinates"])
        return {"type": "MultiLineString", "coordinates": lines} if lines else None
    return None


def point_in_bbox(point: tuple[float, float], bbox: dict[str, float]) -> bool:
    x, y = point
    return bbox["west"] <= x <= bbox["east"] and bbox["south"] <= y <= bbox["north"]


def clip_linestring(coords: list[tuple[float, float]], bbox: dict[str, float]) -> dict[str, Any] | None:
    pieces: list[list[tuple[float, float]]] = []
    for first, second in zip(coords, coords[1:]):
        segment = clip_segment(first, second, bbox)
        if segment is None:
            continue
        if pieces and points_close(pieces[-1][-1], segment[0]):
            pieces[-1].append(segment[1])
        else:
            pieces.append([segment[0], segment[1]])
    pieces = [dedupe_adjacent(line) for line in pieces if len(dedupe_adjacent(line)) >= 2]
    if not pieces:
        return None
    if len(pieces) == 1:
        return {"type": "LineString", "coordinates": pieces[0]}
    return {"type": "MultiLineString", "coordinates": pieces}


def clip_segment(
    first: tuple[float, float],
    second: tuple[float, float],
    bbox: dict[str, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    x0, y0 = first
    x1, y1 = second
    dx = x1 - x0
    dy = y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - bbox["west"], bbox["east"] - x0, y0 - bbox["south"], bbox["north"] - y0]
    u1, u2 = 0.0, 1.0
    for pp, qq in zip(p, q):
        if abs(pp) < 1e-15:
            if qq < 0:
                return None
            continue
        r = qq / pp
        if pp < 0:
            if r > u2:
                return None
            u1 = max(u1, r)
        else:
            if r < u1:
                return None
            u2 = min(u2, r)
    return (x0 + u1 * dx, y0 + u1 * dy), (x0 + u2 * dx, y0 + u2 * dy)


def clip_polygon(rings: list[list[tuple[float, float]]], bbox: dict[str, float]) -> dict[str, Any] | None:
    if not rings:
        return None
    ring = rings[0]
    clipped = sutherland_hodgman(ring, bbox)
    if len(clipped) < 3:
        return None
    clipped = ensure_closed(dedupe_adjacent(clipped))
    if len(clipped) < 4:
        return None
    return {"type": "Polygon", "coordinates": [clipped]}


def sutherland_hodgman(ring: list[tuple[float, float]], bbox: dict[str, float]) -> list[tuple[float, float]]:
    output = ring[:-1] if is_closed(ring) else list(ring)
    for edge in ("left", "right", "bottom", "top"):
        if not output:
            return []
        input_points = output
        output = []
        previous = input_points[-1]
        for current in input_points:
            if inside_edge(current, bbox, edge):
                if not inside_edge(previous, bbox, edge):
                    output.append(edge_intersection(previous, current, bbox, edge))
                output.append(current)
            elif inside_edge(previous, bbox, edge):
                output.append(edge_intersection(previous, current, bbox, edge))
            previous = current
    return output


def inside_edge(point: tuple[float, float], bbox: dict[str, float], edge: str) -> bool:
    x, y = point
    if edge == "left":
        return x >= bbox["west"]
    if edge == "right":
        return x <= bbox["east"]
    if edge == "bottom":
        return y >= bbox["south"]
    if edge == "top":
        return y <= bbox["north"]
    raise ValueError(edge)


def edge_intersection(
    first: tuple[float, float],
    second: tuple[float, float],
    bbox: dict[str, float],
    edge: str,
) -> tuple[float, float]:
    x1, y1 = first
    x2, y2 = second
    if edge in {"left", "right"}:
        x = bbox["west"] if edge == "left" else bbox["east"]
        if abs(x2 - x1) < 1e-15:
            return x, y1
        t = (x - x1) / (x2 - x1)
        return x, y1 + t * (y2 - y1)
    y = bbox["south"] if edge == "bottom" else bbox["north"]
    if abs(y2 - y1) < 1e-15:
        return x1, y
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1), y


def points_close(left: tuple[float, float], right: tuple[float, float], tolerance: float = 1e-12) -> bool:
    return abs(left[0] - right[0]) <= tolerance and abs(left[1] - right[1]) <= tolerance


def dedupe_adjacent(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for point in coords:
        if result and points_close(result[-1], point):
            continue
        result.append(point)
    return result


def transform_geometry(geom: dict[str, Any], transform) -> dict[str, Any]:
    gtype = geom["type"]
    if gtype == "Point":
        x, y = geom["coordinates"]
        return {"type": "Point", "coordinates": transform(x, y)}
    if gtype == "LineString":
        return {"type": "LineString", "coordinates": [transform(x, y) for x, y in geom["coordinates"]]}
    if gtype == "Polygon":
        return {"type": "Polygon", "coordinates": [[transform(x, y) for x, y in ring] for ring in geom["coordinates"]]}
    if gtype == "MultiLineString":
        return {"type": "MultiLineString", "coordinates": [[transform(x, y) for x, y in line] for line in geom["coordinates"]]}
    if gtype == "GeometryCollection":
        return {"type": "GeometryCollection", "geometries": [transform_geometry(item, transform) for item in geom["geometries"]]}
    raise ValueError(f"unsupported geometry type: {gtype}")


def lonlat_to_utm51(lon: float, lat: float) -> tuple[float, float]:
    return lonlat_to_utm(lon, lat, zone=51)


def lonlat_to_utm(lon: float, lat: float, zone: int) -> tuple[float, float]:
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)
    k0 = 0.9996
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    tan_lat = math.tan(lat_rad)
    n = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = ep2 * cos_lat * cos_lat
    aa = cos_lat * (lon_rad - lon0)
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat_rad)
        - (35 * e2**3 / 3072) * math.sin(6 * lat_rad)
    )
    easting = 500000 + k0 * n * (
        aa
        + (1 - t + c) * aa**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * aa**5 / 120
    )
    northing = k0 * (
        m
        + n
        * tan_lat
        * (
            aa**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720
        )
    )
    return easting, northing


def utm51_to_lonlat(easting: float, northing: float) -> tuple[float, float]:
    return utm_to_lonlat(easting, northing, zone=51)


def utm_to_lonlat(easting: float, northing: float, zone: int) -> tuple[float, float]:
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)
    k0 = 0.9996
    x = easting - 500000.0
    y = northing
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    m = y / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512
    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)
    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)
    c1 = ep2 * cos_fp**2
    t1 = tan_fp**2
    n1 = a / math.sqrt(1 - e2 * sin_fp**2)
    r1 = a * (1 - e2) / (1 - e2 * sin_fp**2) ** 1.5
    d = x / (n1 * k0)
    lat = fp - (n1 * tan_fp / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ep2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ep2 - 3 * c1**2) * d**6 / 720
    )
    lon = lon0 + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ep2 + 24 * t1**2) * d**5 / 120
    ) / cos_fp
    return math.degrees(lon), math.degrees(lat)


def write_geopackage(path: Path, extracted: dict[str, list[Feature] | dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA application_id = 1196437808")
        conn.execute("PRAGMA user_version = 10400")
        create_gpkg_core(conn)
        insert_study_area(conn, extracted["study_area"])  # type: ignore[arg-type]
        layer_specs = [
            ("osm_raw_features", WGS84, extracted["osm_raw_features"]),
            ("shoreline_osm_raw", WGS84, extracted["shoreline_osm_raw"]),
            ("quay_front_raw", WGS84, extracted["quay_front_raw"]),
            ("breakwater_raw", WGS84, extracted["breakwater_raw"]),
            ("seamark_raw", WGS84, extracted["seamark_raw"]),
            ("shoreline_osm_utm", UTM51N, extracted["shoreline_osm_utm"]),
            ("quay_front_utm", UTM51N, extracted["quay_front_utm"]),
            ("breakwater_utm", UTM51N, extracted["breakwater_utm"]),
            ("seamark_utm", UTM51N, extracted["seamark_utm"]),
        ]
        for name, srs_id, features in layer_specs:
            create_feature_layer(conn, name, srs_id, features)  # type: ignore[arg-type]
        conn.commit()
    finally:
        conn.close()


def create_gpkg_core(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE gpkg_spatial_ref_sys (
            srs_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL PRIMARY KEY,
            organization TEXT NOT NULL,
            organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL,
            description TEXT
        );
        CREATE TABLE gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY,
            data_type TEXT NOT NULL,
            identifier TEXT UNIQUE,
            description TEXT DEFAULT '',
            last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            min_x DOUBLE,
            min_y DOUBLE,
            max_x DOUBLE,
            max_y DOUBLE,
            srs_id INTEGER,
            CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
        );
        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL,
            m TINYINT NOT NULL,
            CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
            CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
            CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
        );
        """
    )
    conn.executemany(
        "INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("Undefined Cartesian", -1, "NONE", -1, "undefined", "undefined Cartesian coordinate reference system"),
            ("Undefined Geographic", 0, "NONE", 0, "undefined", "undefined geographic coordinate reference system"),
            (
                "WGS 84 geodetic",
                WGS84,
                "EPSG",
                WGS84,
                'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
                "longitude/latitude WGS84",
            ),
            (
                "WGS 84 / UTM zone 51N",
                UTM51N,
                "EPSG",
                UTM51N,
                'PROJCS["WGS 84 / UTM zone 51N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",123],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]',
                "WGS84 UTM zone 51N for Yangshan distance checks",
            ),
        ],
    )


def insert_study_area(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    conn.execute(
        """
        CREATE TABLE study_area (
            fid INTEGER PRIMARY KEY AUTOINCREMENT,
            area_id TEXT,
            source TEXT,
            bbox_west REAL,
            bbox_south REAL,
            bbox_east REAL,
            bbox_north REAL,
            crs TEXT,
            geom BLOB NOT NULL
        )
        """
    )
    geom = payload["geom"]
    conn.execute(
        "INSERT INTO study_area (area_id, source, bbox_west, bbox_south, bbox_east, bbox_north, crs, geom) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            payload["area_id"],
            payload["source"],
            payload["bbox_west"],
            payload["bbox_south"],
            payload["bbox_east"],
            payload["bbox_north"],
            payload["crs"],
            gpkg_geometry_blob(geom, WGS84),
        ),
    )
    register_layer(conn, "study_area", WGS84, "POLYGON", [geom])


def create_feature_layer(conn: sqlite3.Connection, name: str, srs_id: int, features: list[Feature]) -> None:
    conn.execute(
        f"""
        CREATE TABLE {name} (
            fid INTEGER PRIMARY KEY AUTOINCREMENT,
            source_dataset TEXT,
            source_feature_type TEXT,
            source_feature_id TEXT,
            osm_tags_json TEXT,
            geometry_type TEXT,
            verification_level TEXT,
            extraction_rule TEXT,
            layer_class TEXT,
            seamark_type TEXT,
            seamark_name TEXT,
            geom BLOB NOT NULL
        )
        """
    )
    for feature in features:
        conn.execute(
            f"""
            INSERT INTO {name}
            (source_dataset, source_feature_type, source_feature_id, osm_tags_json, geometry_type,
             verification_level, extraction_rule, layer_class, seamark_type, seamark_name, geom)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feature.source_dataset,
                feature.source_feature_type,
                feature.source_feature_id,
                feature.osm_tags_json,
                feature.geometry_type,
                feature.verification_level,
                feature.extraction_rule,
                feature.layer_class,
                feature.seamark_type,
                feature.seamark_name,
                gpkg_geometry_blob(feature.geom, srs_id),
            ),
        )
    register_layer(conn, name, srs_id, "GEOMETRY", [feature.geom for feature in features])


def register_layer(conn: sqlite3.Connection, table_name: str, srs_id: int, geometry_type: str, geometries: list[dict[str, Any]]) -> None:
    bbox = layer_bbox(geometries)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    conn.execute(
        "INSERT INTO gpkg_contents (table_name, data_type, identifier, description, last_change, min_x, min_y, max_x, max_y, srs_id) VALUES (?, 'features', ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            table_name,
            table_name,
            "Yangshan stage-2 GIS audit layer",
            now,
            bbox[0] if bbox else None,
            bbox[1] if bbox else None,
            bbox[2] if bbox else None,
            bbox[3] if bbox else None,
            srs_id,
        ),
    )
    conn.execute(
        "INSERT INTO gpkg_geometry_columns VALUES (?, 'geom', ?, ?, 0, 0)",
        (table_name, geometry_type, srs_id),
    )


def layer_bbox(geometries: Iterable[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    coords: list[tuple[float, float]] = []
    for geom in geometries:
        coords.extend(flat_coords(geom))
    if not coords:
        return None
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    return min(xs), min(ys), max(xs), max(ys)


def flat_coords(geom: dict[str, Any]) -> list[tuple[float, float]]:
    gtype = geom["type"]
    if gtype == "Point":
        return [geom["coordinates"]]
    if gtype == "LineString":
        return list(geom["coordinates"])
    if gtype == "Polygon":
        return [point for ring in geom["coordinates"] for point in ring]
    if gtype == "MultiLineString":
        return [point for line in geom["coordinates"] for point in line]
    if gtype == "GeometryCollection":
        return [point for item in geom["geometries"] for point in flat_coords(item)]
    return []


def gpkg_geometry_blob(geom: dict[str, Any], srs_id: int) -> bytes:
    flags = 1
    return b"GP" + bytes([0, flags]) + struct.pack("<i", srs_id) + wkb(geom)


def wkb(geom: dict[str, Any]) -> bytes:
    gtype = geom["type"]
    if gtype == "Point":
        x, y = geom["coordinates"]
        return struct.pack("<BI2d", 1, 1, x, y)
    if gtype == "LineString":
        coords = geom["coordinates"]
        return struct.pack("<BI", 1, 2) + struct.pack("<I", len(coords)) + b"".join(struct.pack("<2d", x, y) for x, y in coords)
    if gtype == "Polygon":
        rings = geom["coordinates"]
        data = struct.pack("<BI", 1, 3) + struct.pack("<I", len(rings))
        for ring in rings:
            data += struct.pack("<I", len(ring)) + b"".join(struct.pack("<2d", x, y) for x, y in ring)
        return data
    if gtype == "MultiLineString":
        lines = geom["coordinates"]
        return struct.pack("<BI", 1, 5) + struct.pack("<I", len(lines)) + b"".join(
            wkb({"type": "LineString", "coordinates": line}) for line in lines
        )
    if gtype == "GeometryCollection":
        geometries = geom["geometries"]
        return struct.pack("<BI", 1, 7) + struct.pack("<I", len(geometries)) + b"".join(wkb(item) for item in geometries)
    raise ValueError(f"unsupported WKB geometry: {gtype}")


def write_imagery_metadata(output_dir: Path, background: Path) -> dict[str, Any]:
    image = Image.open(background)
    metadata = {
        "provider": "ArcGIS World Imagery",
        "zoom": 14,
        "requested_bbox": dict(IMAGERY_BBOX),
        "image_width": image.size[0],
        "image_height": image.size[1],
        "generation_time": datetime.fromtimestamp(background.stat().st_mtime, timezone.utc).isoformat(),
        "source_file": str(background),
        "georeference_note": "Bbox metadata is reconstructed from the local tile-fetch script request; the PNG itself has no embedded authoritative georeference.",
        "geotiff_output": None,
        "geotiff_note": "GeoTIFF was not generated because rasterio/GDAL are not available in this runtime and the existing PNG has no embedded official georeference.",
    }
    (output_dir / "imagery_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def render_registration_check(
    background: Path,
    output: Path,
    shoreline: list[Feature],
    quay: list[Feature],
    breakwater: list[Feature],
) -> None:
    image = Image.open(background).convert("RGB")
    width, height = image.size
    fig, ax = plt.subplots(figsize=(16, 9.5), constrained_layout=True)
    ax.imshow(image)
    ax.set_axis_off()
    draw_feature_lines(ax, shoreline, width, height, "#22c55e", 1.25, "shoreline_osm_raw")
    draw_feature_lines(ax, quay, width, height, "#f97316", 1.35, "quay_front_raw")
    draw_feature_lines(ax, breakwater, width, height, "#ef4444", 1.65, "breakwater_raw")
    ax.text(
        0.015,
        0.025,
        "洋山港 Stage 2 视觉空间核验图\n仅含卫星影像、shoreline_osm_raw、quay_front_raw、breakwater_raw；未绘制任务节点/航标/服务点/航道。\n无真实控制点坐标，未计算RMSE。",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#f8fafc",
        fontproperties=FontProperties(family=pick_chinese_font()),
        bbox={"facecolor": "#020617", "edgecolor": "#e2e8f0", "alpha": 0.72, "boxstyle": "round,pad=0.42"},
        zorder=30,
    )
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.86)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_feature_lines(ax, features: list[Feature], width: int, height: int, color: str, linewidth: float, label: str) -> None:
    labeled = False
    for feature in features:
        for line in geometry_lines(feature.geom):
            pixel_line = [lonlat_to_pixel(lon, lat, width, height, IMAGERY_BBOX) for lon, lat in line if point_in_bbox((lon, lat), IMAGERY_BBOX)]
            if len(pixel_line) < 2:
                continue
            xs = [point[0] for point in pixel_line]
            ys = [point[1] for point in pixel_line]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=linewidth,
                alpha=0.88,
                label=label if not labeled else None,
                zorder=12,
                path_effects=[patheffects.withStroke(linewidth=linewidth + 1.4, foreground="#020617", alpha=0.42)],
            )
            labeled = True


def geometry_lines(geom: dict[str, Any]) -> list[list[tuple[float, float]]]:
    gtype = geom["type"]
    if gtype == "LineString":
        return [geom["coordinates"]]
    if gtype == "Polygon":
        return list(geom["coordinates"])
    if gtype == "MultiLineString":
        return list(geom["coordinates"])
    if gtype == "GeometryCollection":
        lines: list[list[tuple[float, float]]] = []
        for item in geom["geometries"]:
            lines.extend(geometry_lines(item))
        return lines
    return []


def lonlat_to_pixel(lon: float, lat: float, width: int, height: int, bbox: dict[str, float]) -> tuple[float, float]:
    x = (lon - bbox["west"]) / (bbox["east"] - bbox["west"]) * width
    y = (bbox["north"] - lat) / (bbox["north"] - bbox["south"]) * height
    return x, y


def pick_chinese_font() -> str:
    available = {font.name for font in fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "STSong", "FangSong"):
        if candidate in available:
            return candidate
    return "sans-serif"


def run_quality_checks(path: Path, extracted: dict[str, list[Feature] | dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    conn = sqlite3.connect(path)
    try:
        checks["sqlite_integrity"] = conn.execute("PRAGMA integrity_check").fetchone()[0]
        layer_rows = conn.execute("SELECT table_name, geometry_type_name, srs_id FROM gpkg_geometry_columns ORDER BY table_name").fetchall()
        checks["gpkg_geometry_columns"] = layer_rows
        checks["all_layers_have_crs"] = all(row[2] in {WGS84, UTM51N} for row in layer_rows)
        checks["gpkg_feature_counts"] = {
            row[0]: conn.execute(f"SELECT COUNT(*) FROM {row[0]}").fetchone()[0]
            for row in layer_rows
        }
    finally:
        conn.close()
    checks["empty_layers"] = [name for name, value in extracted.items() if isinstance(value, list) and len(value) == 0]
    checks["lonlat_out_of_reasonable_range"] = lonlat_range_errors(extracted)
    checks["missing_source_feature_id"] = missing_source_ids(extracted)
    checks["seamark_tags_preserved"] = all('"seamark:' in feature.osm_tags_json or '"seamark:type"' in feature.osm_tags_json for feature in extracted["seamark_raw"])  # type: ignore[index]
    checks["fallback_objects"] = fallback_objects(extracted)
    checks["epsg_4326_32651_roundtrip"] = roundtrip_check()
    checks["qgis_command_available"] = False
    checks["qgis_open_check"] = "not_executed_qgis_process_not_found; sqlite_geopackage_structure_checked"
    return checks


def lonlat_range_errors(extracted: dict[str, list[Feature] | dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for name, value in extracted.items():
        if not isinstance(value, list) or name.endswith("_utm"):
            continue
        for feature in value:
            for lon, lat in flat_coords(feature.geom):
                if not (120.0 <= lon <= 123.0 and 29.0 <= lat <= 32.0):
                    errors.append(f"{name}:{feature.source_feature_id}:{lon},{lat}")
                    break
    return errors


def missing_source_ids(extracted: dict[str, list[Feature] | dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for name, value in extracted.items():
        if not isinstance(value, list):
            continue
        for feature in value:
            if not feature.source_feature_id:
                missing.append(f"{name}:{feature.source_feature_type}")
    return missing


def fallback_objects(extracted: dict[str, list[Feature] | dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for name, value in extracted.items():
        if not isinstance(value, list):
            continue
        for feature in value:
            combined = f"{feature.source_dataset} {feature.source_feature_id} {feature.extraction_rule}".lower()
            if "fallback" in combined:
                found.append(f"{name}:{feature.source_feature_id}")
    return found


def roundtrip_check() -> dict[str, Any]:
    samples = [
        (STUDY_BBOX["west"], STUDY_BBOX["south"]),
        (STUDY_BBOX["east"], STUDY_BBOX["north"]),
        ((STUDY_BBOX["west"] + STUDY_BBOX["east"]) / 2, (STUDY_BBOX["south"] + STUDY_BBOX["north"]) / 2),
    ]
    max_error_deg = 0.0
    rows = []
    for lon, lat in samples:
        east, north = lonlat_to_utm51(lon, lat)
        lon2, lat2 = utm51_to_lonlat(east, north)
        err = max(abs(lon - lon2), abs(lat - lat2))
        max_error_deg = max(max_error_deg, err)
        rows.append({"lon": lon, "lat": lat, "easting": east, "northing": north, "lon2": lon2, "lat2": lat2, "error_deg": err})
    return {"max_error_deg": max_error_deg, "passed": max_error_deg < 1e-8, "samples": rows}


def write_qgis_layer_guide(path: Path, imagery_metadata: dict[str, Any]) -> None:
    text = f"""# QGIS图层核验指南

本文件对应 `yangshan_stage2.gpkg`。本阶段仅用于真实几何抽取与空间配准核验，不包含任务节点、USV服务点、动态异常、depot或可航网络。

## 建议加载顺序

1. 在线XYZ/WMTS卫星底图，优先使用QGIS中配置的 ArcGIS World Imagery 或其他可信影像底图。
2. `study_area`，EPSG:4326，研究区边界。
3. `osm_raw_features`，EPSG:4326，裁剪后的相关OSM原始候选要素。
4. `shoreline_osm_raw`，EPSG:4326，可能表达水陆边界的OSM原始候选几何。
5. `quay_front_raw`，EPSG:4326，码头、岸壁、泊位前沿相关OSM原始候选几何。
6. `breakwater_raw`，EPSG:4326，防波堤或丁坝相关OSM原始候选几何。
7. `seamark_raw`，EPSG:4326，所有具有 `seamark:*` 标签的OSM原始航标候选要素。
8. `shoreline_osm_utm`、`quay_front_utm`、`breakwater_utm`、`seamark_utm`，EPSG:32651，仅用于测距、缓冲和人工检查。

## 图层含义

| 图层 | CRS | 来源 | 含义 | 核验状态 |
| --- | --- | --- | --- | --- |
| `study_area` | EPSG:4326 | 第一阶段采纳的OSM研究区bbox | 研究区边界面 | 可作为范围参考 |
| `osm_raw_features` | EPSG:4326 | OpenStreetMap Overpass API | 所有裁剪后可能相关OSM要素，保留完整tags | 原始候选，需要人工筛选 |
| `shoreline_osm_raw` | EPSG:4326 | OSM natural/place 标签 | 可能表达水陆边界的原始几何 | 原始候选，需要配准核验 |
| `quay_front_raw` | EPSG:4326 | OSM man_made/harbour/port 标签 | 码头、岸壁、泊位前沿相关候选几何 | 原始候选，需要剔除非水侧线 |
| `breakwater_raw` | EPSG:4326 | OSM `man_made=breakwater/groyne` | 防波结构候选几何 | 原始候选，需要影像核验 |
| `seamark_raw` | EPSG:4326 | OSM `seamark:*` 标签 | 航标/虚拟航标/灯标候选 | `verification_level=osm_only`，正式A类仍需官方或交叉核验 |
| `*_utm` | EPSG:32651 | 由同名EPSG:4326图层投影得到 | QGIS中测距、缓冲、偏移检查 | 不作为原始存储坐标 |

## 建议符号

| 图层 | 符号建议 |
| --- | --- |
| `study_area` | 无填充，黑色边线，线宽0.7 |
| `osm_raw_features` | 灰色细线/小点，透明度70%，默认隐藏 |
| `shoreline_osm_raw` | 绿色线，线宽1.0，透明度20%-40% |
| `quay_front_raw` | 橙色线，线宽1.2，透明度20%-40% |
| `breakwater_raw` | 红色线，线宽1.4，透明度10%-30% |
| `seamark_raw` | 蓝色或紫色点/线，默认隐藏；只用于来源核验，不参与本阶段配准图 |
| `*_utm` | 默认隐藏，仅在测距或缓冲检查时打开 |

## 需要人工核验

- `shoreline_osm_raw` 是否稳定贴合影像中的真实水陆边界。
- `quay_front_raw` 是否混入道路、堆场边界、桥梁或非水侧设施线。
- `breakwater_raw` 是否真实对应防波堤/丁坝，不得将普通围堰或堆场边界混入。
- `seamark_raw` 中哪些要素可以由官方资料或第二公开源交叉核验。
- 若没有控制点实测坐标，本阶段只能做视觉空间核验，不得填写RMSE。

## 影像元数据

```json
{json.dumps(imagery_metadata, ensure_ascii=False, indent=2)}
```

说明：已有PNG没有嵌入官方地理参考。正式核验优先使用QGIS在线底图；本地PNG只作为脚本生成的视觉核验背景。
"""
    path.write_text(text, encoding="utf-8")


def write_extraction_report(path: Path, extracted: dict[str, list[Feature] | dict[str, Any]], checks: dict[str, Any], imagery_metadata: dict[str, Any]) -> None:
    counts = {
        name: len(value)
        for name, value in extracted.items()
        if isinstance(value, list)
    }
    seamark_types = {}
    for feature in extracted["seamark_raw"]:  # type: ignore[index]
        seamark_types[feature.seamark_type or "(missing)"] = seamark_types.get(feature.seamark_type or "(missing)", 0) + 1
    lines = [
        "# OSM真实几何抽取报告",
        "",
        "阶段：第二阶段，真实几何抽取和空间配准核验",
        "",
        "## 输出文件",
        "",
        "- `yangshan_stage2.gpkg`",
        "- `qgis_layer_guide.md`",
        "- `osm_extraction_report.md`",
        "- `01_registration_check.png`",
        "- `imagery_metadata.json`",
        "",
        "## 抽取原则",
        "",
        "- 原始抽取图层保存为 EPSG:4326。",
        "- UTM副本保存为 EPSG:32651。",
        "- 未使用旧 `LocalProjector`。",
        "- 未读取旧 tasks JSON 作为正式几何来源。",
        "- 未生成A/B/C任务节点、设施锚点、USV服务线、USV服务点、动态异常、depot或可航网络。",
        "- 未使用fallback补充数据。",
        "",
        "## 图层要素数量",
        "",
        "| 图层 | 要素数 |",
        "| --- | ---: |",
    ]
    for name in [
        "osm_raw_features",
        "shoreline_osm_raw",
        "quay_front_raw",
        "breakwater_raw",
        "seamark_raw",
        "shoreline_osm_utm",
        "quay_front_utm",
        "breakwater_utm",
        "seamark_utm",
    ]:
        lines.append(f"| `{name}` | {counts.get(name, 0)} |")
    lines.extend(
        [
            "",
            "## seamark:type统计",
            "",
            "| seamark:type | 数量 |",
            "| --- | ---: |",
        ]
    )
    for key, value in sorted(seamark_types.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## 影像元数据",
            "",
            "```json",
            json.dumps(imagery_metadata, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 自动检查",
            "",
            "```json",
            json.dumps(checks, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            "## 检查结论",
            "",
        ]
    )
    if checks["empty_layers"]:
        lines.append(f"- 存在空图层：{checks['empty_layers']}。")
    else:
        lines.append("- 所有输出候选图层均非空。")
    if checks["lonlat_out_of_reasonable_range"]:
        lines.append("- 存在超出洋山研究区合理范围的经纬度坐标，需要复查。")
    else:
        lines.append("- EPSG:4326图层坐标均位于合理经纬度范围内。")
    if checks["missing_source_feature_id"]:
        lines.append("- 存在缺失 `source_feature_id` 的要素，需要复查。")
    else:
        lines.append("- 所有OSM要素均保留 `source_feature_id`。")
    if checks["fallback_objects"]:
        lines.append("- 检测到fallback对象，不合格。")
    else:
        lines.append("- 未检测到fallback对象。")
    lines.append("- QGIS命令行未在当前环境中找到；已完成SQLite/GeoPackage结构检查，仍需人工在QGIS中打开核验。")
    lines.append("- `01_registration_check.png` 为视觉空间核验图，未计算也未伪造RMSE。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
