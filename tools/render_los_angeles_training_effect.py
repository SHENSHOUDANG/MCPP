from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_GRID = Path("data/ports/los_angeles_training_v1/los_angeles_training_v1_grid.json")
DEFAULT_TASKS = Path("data/ports/los_angeles_training_v1/los_angeles_training_v1_tasks.json")
DEFAULT_OUTPUT = Path("reports/los_angeles_training_effect.png")
DEFAULT_BASEMAP = Path("reports/los_angeles_noaa_enc_harbour_basemap.png")
NOAA_HARBOUR_EXPORT = "https://encdirect.noaa.gov/arcgis/rest/services/encdirect/enc_harbour/MapServer/export"

GEOMETRY_STYLE = {
    "point": {"color": "#0ea5e9", "marker": "o", "label": "Point / navigation aid"},
    "line": {"color": "#ef4444", "marker": None, "label": "Line / route or survey corridor"},
    "area": {"color": "#f59e0b", "marker": "s", "label": "Area / water-area patrol"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Los Angeles official-data training scenario effect image.")
    parser.add_argument("--grid", default=str(DEFAULT_GRID))
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--basemap", default=str(DEFAULT_BASEMAP), help="NOAA ENC chart export PNG to use as the map background.")
    parser.add_argument("--fetch-noaa-basemap", action="store_true", help="Download a NOAA ENC Harbour export PNG before rendering.")
    args = parser.parse_args()

    grid = _load_json(Path(args.grid))
    tasks = _load_json(Path(args.tasks))
    basemap = Path(args.basemap) if args.basemap else None
    if args.fetch_noaa_basemap and basemap is not None:
        fetch_noaa_basemap(grid, basemap)
    output = render_effect(grid, tasks, Path(args.output), basemap)
    print(output.resolve())


def render_effect(grid: dict[str, Any], tasks: dict[str, Any], output: Path, basemap: Path | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18.2, 11.2), facecolor="#f8fafc")
    layout = fig.add_gridspec(1, 2, width_ratios=[4.95, 1.8], wspace=0.045)
    ax = fig.add_subplot(layout[0, 0])
    info_ax = fig.add_subplot(layout[0, 1])

    _draw_map(ax, grid, tasks, basemap)
    _draw_info_panel(info_ax, grid, tasks)

    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output


def fetch_noaa_basemap(grid: dict[str, Any], output: Path, *, size: tuple[int, int] = (1800, 1300)) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    bbox = _plot_bbox(grid)
    params = {
        "bbox": f"{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{size[0]},{size[1]}",
        "format": "png32",
        "transparent": "false",
        "dpi": "160",
        "f": "image",
    }
    url = f"{NOAA_HARBOUR_EXPORT}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "Codex LA chart basemap renderer"})
    with urllib.request.urlopen(request, timeout=120) as response:
        output.write_bytes(response.read())
    return output


def _draw_map(ax, grid: dict[str, Any], tasks: dict[str, Any], basemap: Path | None) -> None:
    bbox = _plot_bbox(grid)
    if basemap and basemap.exists():
        image = Image.open(basemap).convert("RGB")
        ax.imshow(image, extent=[bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"]], origin="upper", zorder=0)
    else:
        ax.set_facecolor("#dbeafe")
        _draw_fallback_grid(ax, grid, bbox)

    _draw_task_risk_cells(ax, grid, bbox)
    _draw_tasks(ax, grid, tasks)
    _draw_depot(ax, grid)
    _draw_title(ax, grid, tasks)

    ax.set_xlim(bbox["xmin"], bbox["xmax"])
    ax.set_ylim(bbox["ymin"], bbox["ymax"])
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=8, colors="#334155")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(0.8)


def _draw_fallback_grid(ax, grid: dict[str, Any], bbox: dict[str, float]) -> None:
    width = int(grid["width"])
    height = int(grid["height"])
    for col in range(0, width + 1, max(1, width // 8)):
        lon, _ = _cell_to_lon_lat(grid, (0, col))
        ax.axvline(lon, color="#94a3b8", linewidth=0.55, alpha=0.26, zorder=1)
    for row in range(0, height + 1, max(1, height // 8)):
        _, lat = _cell_to_lon_lat(grid, (row, 0))
        ax.axhline(lat, color="#94a3b8", linewidth=0.55, alpha=0.26, zorder=1)
    ax.set_xlim(bbox["xmin"], bbox["xmax"])
    ax.set_ylim(bbox["ymin"], bbox["ymax"])


def _draw_task_risk_cells(ax, grid: dict[str, Any], bbox: dict[str, float]) -> None:
    risk = np.array(grid["risk_grid"], dtype=float)
    if risk.max() <= 0:
        return
    rows, cols = np.where(risk > 0)
    lons = []
    lats = []
    values = []
    for row, col in zip(rows, cols):
        lon, lat = _cell_to_lon_lat(grid, (int(row), int(col)))
        if bbox["xmin"] <= lon <= bbox["xmax"] and bbox["ymin"] <= lat <= bbox["ymax"]:
            lons.append(lon)
            lats.append(lat)
            values.append(risk[row, col])
    if not lons:
        return
    ax.scatter(lons, lats, c=values, cmap="YlOrRd", s=70, marker="s", alpha=0.26, linewidths=0, zorder=4)


def _draw_tasks(ax, grid: dict[str, Any], tasks: dict[str, Any]) -> None:
    for geometry, records in _task_groups(tasks):
        style = GEOMETRY_STYLE[geometry]
        for index, task in enumerate(records, start=1):
            cells = _task_cells(task)
            if not cells:
                continue
            lon_lat = [_cell_to_lon_lat(grid, cell) for cell in cells]
            xs = [lon for lon, _ in lon_lat]
            ys = [lat for _, lat in lon_lat]
            if geometry == "line":
                ax.plot(xs, ys, color="#111827", linewidth=6.2, alpha=0.55, solid_capstyle="round", zorder=9)
                ax.plot(xs, ys, color=style["color"], linewidth=3.1, alpha=0.98, solid_capstyle="round", zorder=10)
                label_x, label_y = xs[len(xs) // 2], ys[len(ys) // 2]
            elif geometry == "area":
                ax.scatter(xs, ys, marker="s", s=104, facecolor=style["color"], edgecolor="#111827", linewidth=0.75, alpha=0.78, zorder=11)
                label_x, label_y = float(np.mean(xs)), float(np.mean(ys))
            else:
                ax.scatter(xs, ys, marker="o", s=145, facecolor=style["color"], edgecolor="#ffffff", linewidth=1.5, alpha=0.96, zorder=12)
                ax.scatter(xs, ys, marker="o", s=175, facecolor="none", edgecolor="#0f172a", linewidth=1.0, alpha=0.9, zorder=11)
                label_x, label_y = xs[0], ys[0]
            _label_task(ax, task, geometry, index, label_x, label_y)

    handles = []
    labels = []
    for geometry, style in GEOMETRY_STYLE.items():
        if geometry == "line":
            handle = plt.Line2D([0], [0], color=style["color"], linewidth=2.4)
        else:
            handle = plt.Line2D([0], [0], marker=style["marker"], color="w", markerfacecolor=style["color"], markeredgecolor="#0f172a", markersize=8)
        handles.append(handle)
        labels.append(style["label"])
    ax.legend(handles, labels, loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#111827", fontsize=8.2)


def _label_task(ax, task: dict[str, Any], geometry: str, index: int, x: float, y: float) -> None:
    prefix = {"point": "P", "line": "L", "area": "A"}[geometry]
    source_name = str(task.get("metadata", {}).get("object_name") or task["id"])
    label = f"{prefix}{index}"
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = 0.0014
    text_ha = "left"
    if x > xmax - (xmax - xmin) * 0.09:
        dx = -0.0014
        text_ha = "right"
    label_dy = 0.0015
    name_dy = -0.0025
    if y < ymin + (ymax - ymin) * 0.08:
        label_dy = 0.004
        name_dy = 0.001
    ax.text(
        x + dx,
        y + label_dy,
        label,
        fontsize=9.2,
        color="#0f172a",
        weight="bold",
        ha=text_ha,
        bbox={"facecolor": "#ffffff", "edgecolor": "#111827", "linewidth": 0.5, "alpha": 0.9, "boxstyle": "round,pad=0.16"},
        zorder=15,
    )
    ax.text(
        x + dx,
        y + name_dy,
        _shorten(source_name, 30),
        fontsize=7.2,
        color="#0f172a",
        ha=text_ha,
        bbox={"facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "linewidth": 0.25, "alpha": 0.78, "boxstyle": "round,pad=0.12"},
        zorder=14,
    )


def _draw_depot(ax, grid: dict[str, Any]) -> None:
    row, col = grid["depot"]
    lon, lat = _cell_to_lon_lat(grid, (row, col))
    ax.scatter([lon], [lat], marker="*", s=340, facecolor="#facc15", edgecolor="#111827", linewidth=1.2, zorder=16)
    ax.text(
        lon + 0.0018,
        lat + 0.0015,
        "DEPOT",
        fontsize=9,
        color="#111827",
        weight="bold",
        bbox={"facecolor": "#fef9c3", "edgecolor": "#111827", "linewidth": 0.5, "alpha": 0.92, "boxstyle": "round,pad=0.18"},
        zorder=17,
    )


def _draw_title(ax, grid: dict[str, Any], tasks: dict[str, Any]) -> None:
    metadata = grid.get("metadata", {})
    point_count = len(tasks.get("point_tasks", []))
    line_count = len(tasks.get("line_tasks", []))
    area_count = len(tasks.get("area_tasks", []))
    title = "Los Angeles Port UAV-USV Training Scenario"
    subtitle = (
        f"Chart-aligned LA task mapping | PENDING training assumptions | "
        f"NOAA ENC chart basemap | {point_count} point, {line_count} line, {area_count} area tasks | "
        f"{metadata.get('cell_size_m', grid.get('cell_size_m'))} m cells"
    )
    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=14, color="#0f172a", pad=12, weight="bold")


def _draw_info_panel(ax, grid: dict[str, Any], tasks: dict[str, Any]) -> None:
    ax.set_axis_off()
    metadata = grid.get("metadata", {})
    task_records = [task for _, records in _task_groups(tasks) for task in records]
    chart_aligned_count = sum(
        1
        for task in task_records
        if task.get("metadata", {}).get("geometry_source_status") == "chart_aligned_research_geometry"
    )
    families = _count_by(task_records, lambda task: task.get("metadata", {}).get("task_family", "UNKNOWN"))
    bbox = metadata.get("official_query_bbox_epsg4326", {})

    ax.text(0.02, 0.97, "Effect Figure Notes", fontsize=14, weight="bold", color="#0f172a", va="top")
    body = [
        ("Status", str(metadata.get("contract_status", "PENDING"))),
        ("Geometry", f"{chart_aligned_count}/{len(task_records)} tasks chart-aligned"),
        ("Basemap", "NOAA ENC Direct Harbour export"),
        ("Access date", str(metadata.get("access_date", "unknown"))),
        ("Grid", f"{grid['width']} x {grid['height']} cells"),
        ("BBox", _format_bbox(bbox)),
    ]
    y = 0.89
    for key, value in body:
        ax.text(0.02, y, key, fontsize=8.2, color="#475569", va="top")
        ax.text(0.36, y, value, fontsize=8.2, color="#0f172a", va="top", wrap=True)
        y -= 0.048

    ax.text(0.02, y - 0.015, "Task Families", fontsize=10.5, weight="bold", color="#0f172a", va="top")
    y -= 0.058
    family_text = " | ".join(f"{family}: {count}" for family, count in families.items())
    ax.text(0.04, y, family_text, fontsize=7.4, color="#334155", va="top", wrap=True)
    y -= 0.082

    ax.text(0.02, y - 0.015, "Task Labels", fontsize=10.5, weight="bold", color="#0f172a", va="top")
    y -= 0.058
    for label, task in _labelled_tasks(tasks):
        name = task.get("metadata", {}).get("object_name") or task["id"]
        ax.text(0.04, y, f"{label}: {_shorten(str(name), 38)}", fontsize=7.25, color="#334155", va="top")
        y -= 0.031

    ax.text(0.02, y - 0.018, "Data Boundary", fontsize=10.5, weight="bold", color="#0f172a", va="top")
    y -= 0.06
    ax.text(
        0.04,
        y,
        "NOAA ENC Direct Harbour chart export is used as the basemap. Task cells are imported from the V2.0 chart-aligned LA task mapping package.",
        fontsize=7.4,
        color="#334155",
        va="top",
        wrap=True,
    )

    ax.text(
        0.02,
        0.03,
        "Chart basemap is NOAA-derived; task geometries are chart-aligned research inputs. Workload, risk, deadlines, and depot are training assumptions.",
        fontsize=7.3,
        color="#64748b",
        va="bottom",
        wrap=True,
    )


def _task_groups(tasks: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        ("point", list(tasks.get("point_tasks", []))),
        ("line", list(tasks.get("line_tasks", []))),
        ("area", list(tasks.get("area_tasks", []))),
    ]


def _labelled_tasks(tasks: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    labelled: list[tuple[str, dict[str, Any]]] = []
    prefixes = {"point": "P", "line": "L", "area": "A"}
    for geometry, records in _task_groups(tasks):
        for index, task in enumerate(records, start=1):
            labelled.append((f"{prefixes[geometry]}{index}", task))
    return labelled


def _task_cells(task: dict[str, Any]) -> list[tuple[int, int]]:
    if "cell" in task:
        row, col = task["cell"]
        return [(int(row), int(col))]
    return [(int(row), int(col)) for row, col in task.get("cells", [])]


def _plot_bbox(grid: dict[str, Any]) -> dict[str, float]:
    metadata = grid.get("metadata", {})
    bbox = metadata.get("official_query_bbox_epsg4326") or {}
    if {"xmin", "ymin", "xmax", "ymax"}.issubset(bbox):
        return {key: float(bbox[key]) for key in ("xmin", "ymin", "xmax", "ymax")}
    bounds = metadata.get("bounds_lon_lat", {})
    return {
        "xmin": float(bounds["lon_min"]),
        "xmax": float(bounds["lon_max"]),
        "ymin": float(bounds["lat_min"]),
        "ymax": float(bounds["lat_max"]),
    }


def _cell_to_lon_lat(grid: dict[str, Any], cell: tuple[int, int]) -> tuple[float, float]:
    bounds = grid.get("metadata", {}).get("bounds_lon_lat", {})
    lon_min = float(bounds["lon_min"])
    lon_max = float(bounds["lon_max"])
    lat_min = float(bounds["lat_min"])
    lat_max = float(bounds["lat_max"])
    row, col = cell
    lon = lon_min + (float(col) + 0.5) / float(grid["width"]) * (lon_max - lon_min)
    lat = lat_max - (float(row) + 0.5) / float(grid["height"]) * (lat_max - lat_min)
    return lon, lat


def _source_layers(metadata: dict[str, Any]) -> list[str]:
    result = []
    for source in metadata.get("source_layers", []):
        count = int(source.get("selected_feature_count", 0))
        if count <= 0:
            continue
        result.append(f"{source.get('layer_name')} ({count})")
    return result


def _count_by(records: list[dict[str, Any]], key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(key_fn(record))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _format_bbox(bbox: dict[str, Any]) -> str:
    try:
        return f"{float(bbox['xmin']):.3f}, {float(bbox['ymin']):.3f} to {float(bbox['xmax']):.3f}, {float(bbox['ymax']):.3f}"
    except (KeyError, TypeError, ValueError):
        return "not recorded"


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
