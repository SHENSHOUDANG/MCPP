from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_GRID = Path("data/ports/los_angeles_training_v1/los_angeles_training_v1_grid.json")
DEFAULT_TASKS = Path("data/ports/los_angeles_training_v1/los_angeles_training_v1_tasks.json")
DEFAULT_OUTPUT = Path("reports/los_angeles_training_effect.png")

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
    args = parser.parse_args()

    grid = _load_json(Path(args.grid))
    tasks = _load_json(Path(args.tasks))
    output = render_effect(grid, tasks, Path(args.output))
    print(output.resolve())


def render_effect(grid: dict[str, Any], tasks: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10), facecolor="#f8fafc")
    layout = fig.add_gridspec(1, 2, width_ratios=[4.7, 1.55], wspace=0.045)
    ax = fig.add_subplot(layout[0, 0])
    info_ax = fig.add_subplot(layout[0, 1])

    _draw_map(ax, grid, tasks)
    _draw_info_panel(info_ax, grid, tasks)

    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output


def _draw_map(ax, grid: dict[str, Any], tasks: dict[str, Any]) -> None:
    risk = np.array(grid["risk_grid"], dtype=float)
    height = int(grid["height"])
    width = int(grid["width"])

    water = np.zeros((height, width, 3), dtype=float)
    water[:, :, 0] = 0.86
    water[:, :, 1] = 0.94
    water[:, :, 2] = 0.98
    ax.imshow(water, extent=[0, width, height, 0], interpolation="nearest", zorder=0)
    if risk.max() > 0:
        masked = np.ma.masked_where(risk <= 0, risk)
        ax.imshow(masked, cmap="YlOrRd", alpha=0.34, extent=[0, width, height, 0], interpolation="nearest", zorder=1)

    _draw_grid(ax, width, height)
    _draw_tasks(ax, tasks)
    _draw_depot(ax, grid)
    _draw_title(ax, grid, tasks)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.tick_params(labelsize=8, colors="#334155")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(0.8)


def _draw_grid(ax, width: int, height: int) -> None:
    major_x = max(1, width // 8)
    major_y = max(1, height // 8)
    ax.set_xticks(np.arange(0, width + 1, major_x))
    ax.set_yticks(np.arange(0, height + 1, major_y))
    ax.grid(which="major", color="#94a3b8", linewidth=0.55, alpha=0.32, zorder=2)


def _draw_tasks(ax, tasks: dict[str, Any]) -> None:
    for geometry, records in _task_groups(tasks):
        style = GEOMETRY_STYLE[geometry]
        for index, task in enumerate(records, start=1):
            cells = _task_cells(task)
            if not cells:
                continue
            xs = [col + 0.5 for row, col in cells]
            ys = [row + 0.5 for row, col in cells]
            if geometry == "line":
                ax.plot(xs, ys, color="#7f1d1d", linewidth=4.8, alpha=0.34, solid_capstyle="round", zorder=5)
                ax.plot(xs, ys, color=style["color"], linewidth=2.2, alpha=0.95, solid_capstyle="round", zorder=6)
                label_x, label_y = xs[len(xs) // 2], ys[len(ys) // 2]
            elif geometry == "area":
                ax.scatter(xs, ys, marker="s", s=78, facecolor=style["color"], edgecolor="#92400e", linewidth=0.6, alpha=0.66, zorder=7)
                label_x, label_y = float(np.mean(xs)), float(np.mean(ys))
            else:
                ax.scatter(xs, ys, marker="o", s=110, facecolor=style["color"], edgecolor="#0f172a", linewidth=0.85, alpha=0.92, zorder=8)
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
    ax.legend(handles, labels, loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)


def _label_task(ax, task: dict[str, Any], geometry: str, index: int, x: float, y: float) -> None:
    prefix = {"point": "P", "line": "L", "area": "A"}[geometry]
    source_name = str(task.get("metadata", {}).get("object_name") or task["id"])
    label = f"{prefix}{index}"
    ax.text(
        x + 0.35,
        y - 0.35,
        label,
        fontsize=8.5,
        color="#0f172a",
        weight="bold",
        bbox={"facecolor": "#ffffff", "edgecolor": "#cbd5e1", "linewidth": 0.45, "alpha": 0.82, "boxstyle": "round,pad=0.16"},
        zorder=12,
    )
    ax.text(
        x + 0.35,
        y + 0.45,
        _shorten(source_name, 26),
        fontsize=6.7,
        color="#334155",
        bbox={"facecolor": "#f8fafc", "edgecolor": "none", "alpha": 0.7, "boxstyle": "round,pad=0.12"},
        zorder=11,
    )


def _draw_depot(ax, grid: dict[str, Any]) -> None:
    row, col = grid["depot"]
    ax.scatter([col + 0.5], [row + 0.5], marker="*", s=260, facecolor="#facc15", edgecolor="#713f12", linewidth=1.0, zorder=14)
    ax.text(
        col + 0.8,
        row - 0.4,
        "DEPOT",
        fontsize=9,
        color="#713f12",
        weight="bold",
        bbox={"facecolor": "#fef9c3", "edgecolor": "#facc15", "linewidth": 0.5, "alpha": 0.9, "boxstyle": "round,pad=0.18"},
        zorder=15,
    )


def _draw_title(ax, grid: dict[str, Any], tasks: dict[str, Any]) -> None:
    metadata = grid.get("metadata", {})
    point_count = len(tasks.get("point_tasks", []))
    line_count = len(tasks.get("line_tasks", []))
    area_count = len(tasks.get("area_tasks", []))
    title = "Los Angeles Port UAV-USV Training Scenario"
    subtitle = (
        f"Official NOAA ENC Direct geometry | PENDING training assumptions | "
        f"{point_count} point, {line_count} line, {area_count} area tasks | "
        f"{metadata.get('cell_size_m', grid.get('cell_size_m'))} m cells"
    )
    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=14, color="#0f172a", pad=12, weight="bold")


def _draw_info_panel(ax, grid: dict[str, Any], tasks: dict[str, Any]) -> None:
    ax.set_axis_off()
    metadata = grid.get("metadata", {})
    task_records = [task for _, records in _task_groups(tasks) for task in records]
    official_count = sum(1 for task in task_records if task.get("metadata", {}).get("geometry_source_status") == "official_noaa_geometry")
    families = _count_by(task_records, lambda task: task.get("metadata", {}).get("task_family", "UNKNOWN"))
    sources = _source_layers(metadata)
    bbox = metadata.get("official_query_bbox_epsg4326", {})

    ax.text(0.02, 0.97, "Effect Figure Notes", fontsize=14, weight="bold", color="#0f172a", va="top")
    body = [
        ("Status", str(metadata.get("contract_status", "PENDING"))),
        ("Geometry", f"{official_count}/{len(task_records)} tasks official NOAA"),
        ("Access date", str(metadata.get("access_date", "unknown"))),
        ("Grid", f"{grid['width']} x {grid['height']} cells"),
        ("BBox", _format_bbox(bbox)),
    ]
    y = 0.89
    for key, value in body:
        ax.text(0.02, y, key, fontsize=8.5, color="#475569", va="top")
        ax.text(0.37, y, value, fontsize=8.5, color="#0f172a", va="top", wrap=True)
        y -= 0.062

    ax.text(0.02, y - 0.015, "Task Families", fontsize=10.5, weight="bold", color="#0f172a", va="top")
    y -= 0.07
    for family, count in families.items():
        ax.text(0.04, y, f"{family}: {count}", fontsize=8.2, color="#334155", va="top")
        y -= 0.045

    ax.text(0.02, y - 0.015, "NOAA Layers Used", fontsize=10.5, weight="bold", color="#0f172a", va="top")
    y -= 0.07
    for source in sources[:8]:
        ax.text(0.04, y, _shorten(source, 38), fontsize=7.7, color="#334155", va="top")
        y -= 0.04

    ax.text(
        0.02,
        0.035,
        "Rendered from repository JSON only. Workload, risk, deadlines, and depot are training assumptions, not official work orders.",
        fontsize=8,
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


def _task_cells(task: dict[str, Any]) -> list[tuple[int, int]]:
    if "cell" in task:
        row, col = task["cell"]
        return [(int(row), int(col))]
    return [(int(row), int(col)) for row, col in task.get("cells", [])]


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
