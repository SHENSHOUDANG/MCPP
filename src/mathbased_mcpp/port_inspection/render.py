from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle

from .schema import AssignmentResult, GridCell, InspectionTask, PortGridMap


def _pick_chinese_font() -> str:
    available = {font.name for font in fontManager.ttflist}
    for candidate in ("SimSun", "KaiTi", "STSong", "STKaiti", "FangSong", "Microsoft YaHei", "SimHei"):
        if candidate in available:
            return candidate
    return "sans-serif"


_CN_FONT = FontProperties(family=_pick_chinese_font())


def render_port_inspection_map(
    grid: PortGridMap,
    tasks: list[InspectionTask],
    output_path: str | Path,
    assignments: list[AssignmentResult] | None = None,
    title: str | None = None,
    show_risk: bool = False,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    mask = np.zeros((grid.height, grid.width), dtype=np.float32)
    for row, col in grid.obstacles:
        mask[row, col] = 1.0

    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    if show_risk:
        risk = np.asarray(grid.risk_grid, dtype=np.float32)
        cmap = ListedColormap(["#d9eef8", "#b9e4cc", "#f5de8b", "#ef9a8a"])
        ax.imshow(risk, cmap=cmap, vmin=0, vmax=3, origin="upper")
        obstacle_overlay = np.ma.masked_where(mask == 0, mask)
        ax.imshow(obstacle_overlay, cmap=ListedColormap(["#4d4d4d"]), origin="upper", alpha=0.95)
    else:
        base = np.zeros((grid.height, grid.width), dtype=np.float32)
        for row, col in grid.free_cells:
            base[row, col] = 1.0
        for row, col in grid.obstacles:
            base[row, col] = 2.0
        ax.imshow(base, cmap=ListedColormap(["#edf7fb", "#cfeaf2", "#39434a"]), vmin=0, vmax=2, origin="upper")

    _draw_visual_features(ax, grid)
    _draw_tasks(ax, tasks)
    _draw_assignments(ax, assignments or [])

    depot_row, depot_col = grid.depot
    ax.scatter(depot_col, depot_row, marker="*", s=220, color="#f6f1d1", edgecolors="#111111", linewidths=1.2, label="基地/母港")
    source_name = grid.metadata.get("source_port_zh", grid.metadata.get("source_port", grid.name))
    actual_title = title or f"{source_name}水域巡检任务场景"
    ax.set_title(actual_title, fontsize=14, pad=12, fontproperties=_CN_FONT)
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(grid.height - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, grid.width, 5), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 5), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.35, alpha=0.35)
    ax.set_xlabel(f"列号（栅格边长 {grid.cell_size_m:g} m）", fontproperties=_CN_FONT)
    ax.set_ylabel("行号", fontproperties=_CN_FONT)
    legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    for text in legend.get_texts():
        text.set_fontproperties(_CN_FONT)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _draw_visual_features(ax: plt.Axes, grid: PortGridMap) -> None:
    raw_features = grid.metadata.get("visual_features", [])
    if not raw_features:
        raw_features = grid.metadata.get("visual_lines_preview", [])
    if not isinstance(raw_features, list):
        return
    for raw in raw_features:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", ""))
        label = str(raw.get("label", ""))
        if kind in {"land_box", "water_box"}:
            _draw_feature_box(ax, raw, label, kind)
        elif kind == "land_polygon":
            _draw_feature_cells(ax, raw, label, "#111827")
        elif kind == "line":
            _draw_feature_line(ax, raw, label)
        elif kind == "point":
            _draw_feature_point(ax, raw, label)


def _draw_feature_box(ax: plt.Axes, raw: dict[str, object], label: str, kind: str) -> None:
    bbox = raw.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return
    row0, col0, row1, col1 = [int(value) for value in bbox]
    edge = "#1f2933" if kind == "land_box" else "#2166ac"
    patch = Rectangle(
        (col0 - 0.5, row0 - 0.5),
        col1 - col0 + 1,
        row1 - row0 + 1,
        fill=False,
        edgecolor=edge,
        linewidth=1.3,
        linestyle="--",
        alpha=0.72,
    )
    ax.add_patch(patch)
    _draw_feature_label(ax, raw, label, edge)


def _draw_feature_line(ax: plt.Axes, raw: dict[str, object], label: str) -> None:
    cells = raw.get("cells")
    if not isinstance(cells, list) or len(cells) < 2:
        return
    rows: list[int] = []
    cols: list[int] = []
    for item in cells:
        if not isinstance(item, list | tuple) or len(item) != 2:
            return
        rows.append(int(item[0]))
        cols.append(int(item[1]))
    ax.plot(cols, rows, color="#0f766e", linewidth=1.2, linestyle=":", alpha=0.85)
    _draw_feature_label(ax, raw, label, "#0f766e")


def _draw_feature_cells(ax: plt.Axes, raw: dict[str, object], label: str, color: str) -> None:
    bbox = raw.get("bbox")
    if isinstance(bbox, list | tuple) and len(bbox) == 4:
        row0, col0, row1, col1 = [int(value) for value in bbox]
        patch = Rectangle(
            (col0 - 0.5, row0 - 0.5),
            col1 - col0 + 1,
            row1 - row0 + 1,
            fill=False,
            edgecolor=color,
            linewidth=0.9,
            linestyle="--",
            alpha=0.55,
            zorder=6,
        )
        ax.add_patch(patch)
    _draw_feature_label(ax, raw, label, color)


def _draw_feature_point(ax: plt.Axes, raw: dict[str, object], label: str) -> None:
    cells = raw.get("cells")
    if not isinstance(cells, list) or not cells:
        return
    first = cells[0]
    if not isinstance(first, list | tuple) or len(first) != 2:
        return
    row, col = int(first[0]), int(first[1])
    ax.scatter([col], [row], marker="x", s=26, color="#dc2626", linewidths=0.9, zorder=8)
    _draw_feature_label(ax, raw, label, "#dc2626")


def _draw_feature_label(ax: plt.Axes, raw: dict[str, object], label: str, color: str) -> None:
    anchor = raw.get("anchor")
    if not label or not isinstance(anchor, list | tuple) or len(anchor) != 2:
        return
    row, col = int(anchor[0]), int(anchor[1])
    ax.text(
        col,
        row,
        label,
        fontsize=6.5,
        color=color,
        fontproperties=_CN_FONT,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#ffffff", "edgecolor": color, "linewidth": 0.45, "alpha": 0.72},
        zorder=7,
    )


def _draw_tasks(ax: plt.Axes, tasks: list[InspectionTask]) -> None:
    labels_used: set[str] = set()
    ordered_tasks = (
        [task for task in tasks if task.geometry == "area"]
        + [task for task in tasks if task.geometry == "line"]
        + [task for task in tasks if task.geometry == "point"]
    )
    for task in ordered_tasks:
        rows = [cell[0] for cell in task.cells]
        cols = [cell[1] for cell in task.cells]
        if task.geometry == "point":
            _scatter_once(ax, cols, rows, label="点任务", labels_used=labels_used, marker="o", color="#1f77b4")
            ax.text(cols[0] + 0.35, rows[0] + 0.35, task.task_id, fontsize=7, color="#0b2d4d")
        elif task.geometry == "line":
            visual_rows, visual_cols = _line_visual_points(task, rows, cols)
            _draw_line_task(ax, visual_cols, visual_rows, task, labels_used)
        elif task.geometry == "area":
            _scatter_once(
                ax,
                cols,
                rows,
                label="面任务",
                labels_used=labels_used,
                marker="s",
                color="#ffb000",
                alpha=0.34,
                size=30,
            )
            center_row = sum(rows) / len(rows)
            center_col = sum(cols) / len(cols)
            ax.text(center_col, center_row, task.task_id, fontsize=8, weight="bold", color="#5a3b00")


def _draw_line_task(ax: plt.Axes, cols: list[int], rows: list[int], task: InspectionTask, labels_used: set[str]) -> None:
    label = "线任务" if "线任务" not in labels_used else None
    labels_used.add("线任务")
    ax.plot(cols, rows, color="#6f2c91", linewidth=4.2, alpha=0.95, solid_capstyle="round", label=label)
    ax.plot(cols, rows, color="#f4d7ff", linewidth=1.15, alpha=0.9, solid_capstyle="round")
    ax.scatter([cols[0]], [rows[0]], marker=">", s=42, color="#6f2c91", edgecolors="#ffffff", linewidths=0.4, zorder=5)
    ax.scatter([cols[-1]], [rows[-1]], marker="|", s=64, color="#6f2c91", linewidths=1.2, zorder=5)
    _draw_line_arrow(ax, cols, rows)
    label_index = min(max(len(cols) // 2, 0), len(cols) - 1)
    ax.text(cols[label_index] + 0.25, rows[label_index] - 0.25, task.task_id, fontsize=7, weight="bold", color="#4a235a")


def _draw_line_arrow(ax: plt.Axes, cols: list[int], rows: list[int]) -> None:
    if len(cols) < 3:
        return
    start = max(len(cols) // 2 - 1, 0)
    end = min(start + 1, len(cols) - 1)
    if start == end:
        return
    ax.annotate(
        "",
        xy=(cols[end], rows[end]),
        xytext=(cols[start], rows[start]),
        arrowprops={"arrowstyle": "->", "color": "#4a235a", "lw": 1.6, "shrinkA": 0, "shrinkB": 0},
        zorder=6,
    )


def _line_visual_points(task: InspectionTask, rows: list[int], cols: list[int]) -> tuple[list[int], list[int]]:
    waypoints = task.metadata.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        return rows, cols
    parsed_rows: list[int] = []
    parsed_cols: list[int] = []
    for item in waypoints:
        if not isinstance(item, list | tuple) or len(item) != 2:
            return rows, cols
        parsed_rows.append(int(item[0]))
        parsed_cols.append(int(item[1]))
    return parsed_rows, parsed_cols


def _draw_assignments(ax: plt.Axes, assignments: list[AssignmentResult]) -> None:
    colors = {
        "UAV": "#005f73",
        "USV": "#ae2012",
    }
    labeled: set[str] = set()
    for result in assignments:
        if len(result.path) < 2:
            continue
        rows = [cell[0] for cell in result.path]
        cols = [cell[1] for cell in result.path]
        label = f"{result.platform_type}路径" if result.platform_type not in labeled else None
        labeled.add(result.platform_type)
        ax.plot(cols, rows, color=colors.get(result.platform_type, "#222222"), linewidth=1.1, alpha=0.62, label=label)


def _scatter_once(
    ax: plt.Axes,
    cols: list[int],
    rows: list[int],
    label: str,
    labels_used: set[str],
    marker: str,
    color: str,
    alpha: float = 0.92,
    size: int = 54,
) -> None:
    actual_label = label if label not in labels_used else None
    labels_used.add(label)
    ax.scatter(cols, rows, marker=marker, s=size, color=color, edgecolors="#111111", linewidths=0.35, alpha=alpha, label=actual_label)
