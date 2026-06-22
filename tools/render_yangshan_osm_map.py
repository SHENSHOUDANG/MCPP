from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.port_inspection import load_inspection_tasks, load_port_grid


def _pick_chinese_font() -> str:
    available = {font.name for font in fontManager.ttflist}
    for candidate in ("SimSun", "Microsoft YaHei", "SimHei", "KaiTi", "STSong", "FangSong"):
        if candidate in available:
            return candidate
    return "sans-serif"


_CN_FONT = FontProperties(family=_pick_chinese_font())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render the OSM-based Yangshan Port map used by the inspection model.")
    parser.add_argument(
        "--grid",
        default="data/ports/shanghai_yangshan_osm_v1/shanghai_yangshan_osm_v1_grid.json",
    )
    parser.add_argument(
        "--tasks",
        default="data/ports/shanghai_yangshan_osm_v1/shanghai_yangshan_osm_v1_tasks.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/port_inspection/shanghai_yangshan_osm_v1/yangshan_osm_real_map.png",
    )
    parser.add_argument("--hide-tasks", action="store_true", help="Render only the OSM rasterized map without tasks.")
    parser.add_argument("--show-risk", action="store_true", help="Overlay the model risk raster for diagnostics.")
    args = parser.parse_args()

    grid = load_port_grid(args.grid)
    tasks = [] if args.hide_tasks else load_inspection_tasks(args.tasks, grid)
    output = render_map(grid, tasks, Path(args.output), show_risk=args.show_risk)
    print(output.resolve())


def render_map(grid, tasks, output: Path, show_risk: bool = False) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 9.2), constrained_layout=True)
    if show_risk:
        canvas = np.zeros((grid.height, grid.width), dtype=float)
        risk = np.asarray(grid.risk_grid, dtype=float)
        for cell in grid.free_cells:
            canvas[cell] = 1.0 + risk[cell]
        for cell in grid.obstacles:
            canvas[cell] = 5.0
        cmap = ListedColormap(["#f7fbff", "#d8eff8", "#c8e8ce", "#f3d675", "#ea8e79", "#3e454b"])
        ax.imshow(canvas, origin="upper", cmap=cmap, vmin=0, vmax=5)
    else:
        canvas = np.zeros((grid.height, grid.width), dtype=float)
        for cell in grid.free_cells:
            canvas[cell] = 1.0
        for cell in grid.obstacles:
            canvas[cell] = 2.0
        cmap = ListedColormap(["#edf7fb", "#cfeaf2", "#39434a"])
        ax.imshow(canvas, origin="upper", cmap=cmap, vmin=0, vmax=2)

    _draw_visual_features(ax, grid)
    _draw_tasks(ax, tasks)
    _draw_depot(ax, grid.depot)
    _style_axes(ax, grid, show_risk=show_risk)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _draw_visual_features(ax, grid) -> None:
    features = grid.metadata.get("visual_features", [])
    if not features:
        features = grid.metadata.get("visual_lines_preview", [])
    if not isinstance(features, list):
        return
    labeled_bridge = False
    labeled_osm_line = False
    labeled_land_feature = False
    labeled_point_feature = False
    for feature in features:
        if not isinstance(feature, dict):
            continue
        kind = str(feature.get("kind", ""))
        if kind == "land_polygon":
            label = None if labeled_land_feature else "OSM land/port polygon extent"
            labeled_land_feature = True
            _draw_polygon_feature(ax, feature, label)
            continue
        if kind == "point":
            label = None if labeled_point_feature else "OSM seamark/beacon point"
            labeled_point_feature = True
            _draw_point_feature(ax, feature, label)
            continue
        if kind != "line":
            continue
        cells = feature.get("cells")
        if not isinstance(cells, list) or len(cells) < 2:
            continue
        rows = [int(cell[0]) for cell in cells if isinstance(cell, list | tuple) and len(cell) == 2]
        cols = [int(cell[1]) for cell in cells if isinstance(cell, list | tuple) and len(cell) == 2]
        if len(rows) < 2 or len(rows) != len(cols):
            continue
        label_text = str(feature.get("label", "OSM line"))
        is_bridge = "bridge" in label_text.lower() or "大桥" in label_text
        color = "#0f766e" if is_bridge else "#334155"
        linestyle = "-" if is_bridge else ":"
        label = None
        if is_bridge and not labeled_bridge:
            label = "OSM bridge/structure"
            labeled_bridge = True
        elif not is_bridge and not labeled_osm_line:
            label = "OSM linear feature"
            labeled_osm_line = True
        ax.plot(cols, rows, color=color, linewidth=1.3, linestyle=linestyle, alpha=0.9, label=label, zorder=6)


def _draw_polygon_feature(ax, feature: dict[str, object], label: str | None) -> None:
    bbox = feature.get("bbox")
    if isinstance(bbox, list | tuple) and len(bbox) == 4:
        row0, col0, row1, col1 = [int(value) for value in bbox]
        patch = Rectangle(
            (col0 - 0.5, row0 - 0.5),
            col1 - col0 + 1,
            row1 - row0 + 1,
            fill=False,
            edgecolor="#111827",
            linewidth=0.85,
            linestyle="--",
            alpha=0.58,
            label=label,
            zorder=6,
        )
        ax.add_patch(patch)
        _draw_feature_label(ax, feature, "#111827")


def _draw_point_feature(ax, feature: dict[str, object], label: str | None) -> None:
    cells = feature.get("cells")
    if not isinstance(cells, list) or not cells:
        return
    first = cells[0]
    if not isinstance(first, list | tuple) or len(first) != 2:
        return
    row, col = int(first[0]), int(first[1])
    ax.scatter([col], [row], marker="x", s=36, color="#dc2626", linewidths=1.0, label=label, zorder=12)
    _draw_feature_label(ax, feature, "#dc2626")


def _draw_feature_label(ax, feature: dict[str, object], color: str) -> None:
    label = str(feature.get("label", ""))
    anchor = feature.get("anchor")
    if not label or not isinstance(anchor, list | tuple) or len(anchor) != 2:
        return
    row, col = int(anchor[0]), int(anchor[1])
    if label == "OSM line":
        return
        ax.text(
            col,
            row,
            label,
            fontsize=5.4,
            color=color,
            fontproperties=_CN_FONT,
            ha="center",
            va="center",
        bbox={"facecolor": "#ffffff", "edgecolor": color, "linewidth": 0.25, "alpha": 0.56, "boxstyle": "round,pad=0.12"},
        zorder=14,
    )


def _draw_tasks(ax, tasks) -> None:
    for task in tasks:
        rows = [cell[0] for cell in task.cells]
        cols = [cell[1] for cell in task.cells]
        if task.geometry == "area":
            ax.scatter(
                cols,
                rows,
                marker="s",
                s=18,
                color="#f5b33b",
                edgecolors="#9a6700",
                linewidths=0.18,
                alpha=0.52,
                zorder=7,
            )
            ax.text(np.mean(cols), np.mean(rows), task.task_id, fontsize=6.5, weight="bold", color="#654000", zorder=9)
        elif task.geometry == "line":
            ax.plot(cols, rows, color="#7c2d92", linewidth=3.2, alpha=0.86, solid_capstyle="round", zorder=8)
            ax.plot(cols, rows, color="#f5dbff", linewidth=0.9, alpha=0.96, solid_capstyle="round", zorder=9)
            mid = len(cols) // 2
            ax.text(cols[mid] + 0.15, rows[mid] - 0.15, task.task_id, fontsize=6.2, weight="bold", color="#4a1d5f", zorder=10)
        else:
            ax.scatter(
                cols,
                rows,
                marker="o",
                s=38,
                color="#5da2ff",
                edgecolors="#111111",
                linewidths=0.3,
                alpha=0.94,
                zorder=10,
            )
            ax.text(cols[0] + 0.25, rows[0] + 0.25, task.task_id, fontsize=6.1, color="#102a43", zorder=11)


def _draw_depot(ax, depot: tuple[int, int]) -> None:
    row, col = depot
    ax.scatter(col, row, marker="*", s=260, color="#ffe066", edgecolors="#111111", linewidths=1.0, zorder=12)
    ax.text(col + 0.45, row - 0.45, "DEPOT", fontsize=8, weight="bold", color="#111111", zorder=13)


def _style_axes(ax, grid, show_risk: bool = False) -> None:
    bbox = grid.metadata.get("bbox", {})
    title = "Yangshan Port OSM morphology and facility layers"
    if show_risk:
        title += " | risk diagnostic overlay"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"Grid column, {grid.cell_size_m:g} m/cell")
    ax.set_ylabel("Grid row")
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(grid.height - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, grid.width, 5), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 5), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.35, alpha=0.38)

    if show_risk:
        legend_items = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#3e454b", label="OSM land / port structure"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#d8eff8", label="Water, low risk"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#f3d675", label="Medium risk water"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#ea8e79", label="High risk water"),
        ]
    else:
        legend_items = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#cfeaf2", label="Navigable water / model free cell"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#39434a", label="OSM land / port structure"),
        ]
    legend_items.extend(
        [
            Line2D([0], [0], color="#111827", lw=0.85, linestyle="--", label="OSM polygon extent"),
            Line2D([0], [0], color="#0f766e", lw=1.3, label="OSM bridge/structure"),
            Line2D([0], [0], marker="x", color="#dc2626", lw=0, label="OSM seamark/beacon point"),
            Line2D([0], [0], color="#7c2d92", lw=3.2, label="Line task"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#5da2ff", markeredgecolor="#111111", label="Point task"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffe066", markeredgecolor="#111111", markersize=13, label="Depot"),
        ]
    )
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.92)

    caption = (
        f"OSM bbox: S {bbox.get('south')} / W {bbox.get('west')} / N {bbox.get('north')} / E {bbox.get('east')}\n"
        f"Grid: {grid.height} x {grid.width}, free water {len(grid.free_cells)}, obstacles {len(grid.obstacles)}\n"
        f"OSM feature counts: {grid.metadata.get('feature_counts', {})}\n"
        "Note: OSM-derived model map, not a nautical chart."
    )
    ax.text(
        0.01,
        0.02,
        caption,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontproperties=_CN_FONT,
        bbox={"facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=.35"},
        zorder=20,
    )


if __name__ == "__main__":
    main()
