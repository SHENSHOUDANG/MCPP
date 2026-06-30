from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from check_port_inspection_env import build_env
from evaluate_port_scheduler_greedy import _choose_action
from mathbased_mcpp.port_inspection.schema import STAGE_REVIEW, STAGE_SCREENING, STAGE_SERVICE
from mathbased_mcpp.port_inspection.simple_planner import shortest_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render a QGIS task-map port inspection scheduling result.")
    parser.add_argument("--config", default="configs/port_yangshan_task_initial_v1.toml")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--strategy", choices=("legacy_order", "global_score"), default="global_score")
    parser.add_argument(
        "--output",
        default="outputs/port_inspection/yangshan_task_initial_v1/schedule_result_seed11.png",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    env = build_env(config)
    history = _run_schedule(env, config, seed=args.seed, steps=args.steps, strategy=args.strategy)
    output = render_schedule_result(env, history, Path(args.output), seed=args.seed, strategy=args.strategy)

    summary = {
        "output": str(output),
        "task_lifecycle": getattr(env, "task_lifecycle", "legacy_screen_review"),
        "steps": env.current_step,
        "completed_tasks": len(env.completed_tasks),
        "task_count": env.num_tasks,
        "total_path_length": env.total_path_length,
        "total_energy": env.total_energy,
        "open_task_count": env.open_task_count(),
    }
    if getattr(env, "task_lifecycle", "") != "v1_2_direct_service":
        summary["review_queue_length"] = env.review_queue_length()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _run_schedule(env, config: dict[str, Any], seed: int, steps: int, strategy: str) -> list[dict[str, Any]]:
    env.reset(seed=seed)
    scheduling_config = dict(config.get("scheduling", {}))
    weights = {
        "risk_weight": float(scheduling_config.get("risk_weight", 10.0)),
        "distance_weight": float(scheduling_config.get("distance_weight", 0.05)),
        "load_weight": float(scheduling_config.get("load_weight", 0.8)),
        "compatibility_bonus": float(scheduling_config.get("compatibility_bonus", 3.0)),
    }
    history: list[dict[str, Any]] = []
    done = False
    while not done and env.current_step < max(1, steps):
        action, decision = _choose_action(env, weights, strategy)
        result = env.step(action)
        history.append(
            {
                "step": env.current_step,
                "action": action,
                "decision": decision,
                "accepted": list(result.info.get("accepted_actions", [])),
                "completed": list(result.info.get("completed_tasks", [])),
                "reward": float(result.reward),
            }
        )
        done = result.done
    return history


def render_schedule_result(env, history: list[dict[str, Any]], output_path: Path, seed: int, strategy: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = env.grid
    tasks = env.tasks

    risk = np.asarray(grid.risk_grid, dtype=float)
    base = np.zeros((grid.height, grid.width), dtype=float)
    for row, col in grid.free_cells:
        base[row, col] = 1.0 + risk[row, col]
    for row, col in grid.obstacles:
        base[row, col] = 5.0

    fig, (ax, panel) = plt.subplots(
        1,
        2,
        figsize=(15.5, 8.8),
        gridspec_kw={"width_ratios": [4.8, 1.65]},
        constrained_layout=True,
    )
    cmap = ListedColormap(["#f6fbff", "#d7eef7", "#c6e7cf", "#f2d47b", "#eb8f7a", "#3d4349"])
    ax.imshow(base, origin="upper", cmap=cmap, vmin=0, vmax=5)

    _draw_tasks(ax, tasks)
    _draw_routes(ax, env)
    _draw_depot(ax, grid.depot)

    source = grid.metadata.get("source", "OpenStreetMap")
    bbox = grid.metadata.get("bbox", {})
    title = f"Yangshan Port UAV-USV schedule result | {source}"
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(grid.height - 0.5, -0.5)
    ax.set_xlabel(f"Grid column, {grid.cell_size_m:g} m/cell")
    ax.set_ylabel("Grid row")
    ax.set_xticks(np.arange(-0.5, grid.width, 5), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 5), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.35, alpha=0.38)

    if getattr(env, "task_lifecycle", "") == "v1_2_direct_service":
        uav_label = "UAV service path"
        usv_label = "USV service path"
    else:
        uav_label = "UAV screening path"
        usv_label = "USV review path"
    legend_items = [
        Line2D([0], [0], color="#175ddc", lw=1.8, linestyle="-", label=uav_label),
        Line2D([0], [0], color="#c0342b", lw=2.1, linestyle="-", label=usv_label),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#175ddc", markeredgecolor="#111", label="Point task"),
        Line2D([0], [0], color="#7c2d92", lw=3.5, label="Line task"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#f6b23d", markeredgecolor="#8a5a00", label="Area task"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffe066", markeredgecolor="#111", markersize=13, label="Depot"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.92)

    _draw_side_panel(panel, env, history, bbox, seed, strategy)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)
    return output_path


def _draw_tasks(ax: plt.Axes, tasks) -> None:
    for task in tasks:
        rows = [cell[0] for cell in task.cells]
        cols = [cell[1] for cell in task.cells]
        if task.geometry == "area":
            color = "#f6b23d" if task.task_id in {item.task_id for item in tasks if item.completed} else "#ffd88a"
            ax.scatter(cols, rows, marker="s", s=22, color=color, edgecolors="#8a5a00", linewidths=0.25, alpha=0.78, zorder=4)
            ax.text(np.mean(cols), np.mean(rows), task.task_id, fontsize=6.5, weight="bold", color="#5c3a00", zorder=8)
        elif task.geometry == "line":
            ax.plot(cols, rows, color="#7c2d92", linewidth=3.6, alpha=0.82, solid_capstyle="round", zorder=5)
            ax.plot(cols, rows, color="#f5dbff", linewidth=1.0, alpha=0.95, solid_capstyle="round", zorder=6)
            mid = len(cols) // 2
            ax.text(cols[mid] + 0.18, rows[mid] - 0.18, task.task_id, fontsize=6.4, weight="bold", color="#4a1d5f", zorder=8)
        else:
            marker_color = "#1d63dd" if task.completed else "#88b7ff"
            ax.scatter(cols, rows, marker="o", s=42, color=marker_color, edgecolors="#111111", linewidths=0.35, alpha=0.94, zorder=7)
            ax.text(cols[0] + 0.28, rows[0] + 0.28, task.task_id, fontsize=6.2, color="#102a43", zorder=8)


def _draw_routes(ax: plt.Axes, env) -> None:
    palette = {
        "UAV": ["#175ddc", "#00a6a6", "#4c78a8", "#2f80ed"],
        "USV": ["#c0342b", "#f97316", "#b91c1c", "#9a3412"],
    }
    for index, platform in enumerate(env.platforms):
        if len(platform.route) < 2:
            continue
        color = palette.get(platform.platform_type, ["#333333"])[index % len(palette.get(platform.platform_type, ["#333333"]))]
        route = _expanded_route(env, platform)
        rows = [cell[0] for cell in route]
        cols = [cell[1] for cell in route]
        linestyle = "-" if platform.platform_type == "USV" else "--"
        linewidth = 2.15 if platform.platform_type == "USV" else 1.7
        ax.plot(cols, rows, color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.82, zorder=9)
        ax.scatter(cols[-1], rows[-1], marker="X", s=58, color=color, edgecolors="#ffffff", linewidths=0.45, zorder=10)
        ax.text(cols[-1] + 0.28, rows[-1] - 0.28, platform.platform_id, fontsize=7.2, color=color, weight="bold", zorder=11)
        _draw_direction_arrows(ax, cols, rows, color)


def _expanded_route(env, platform) -> list[tuple[int, int]]:
    expanded: list[tuple[int, int]] = []
    for start, goal in zip(platform.route, platform.route[1:]):
        if platform.platform_type == "USV":
            try:
                segment = shortest_path(env.grid, start, goal)
            except ValueError:
                segment = [start, goal]
        else:
            segment = [start, goal]
        if not expanded:
            expanded.extend(segment)
        else:
            expanded.extend(segment[1:] if expanded[-1] == segment[0] else segment)
    return expanded or list(platform.route)


def _draw_direction_arrows(ax: plt.Axes, cols: list[int], rows: list[int], color: str) -> None:
    if len(cols) < 4:
        return
    for fraction in (0.35, 0.7):
        index = min(max(int(len(cols) * fraction), 1), len(cols) - 1)
        ax.annotate(
            "",
            xy=(cols[index], rows[index]),
            xytext=(cols[index - 1], rows[index - 1]),
            arrowprops={"arrowstyle": "->", "lw": 1.1, "color": color, "shrinkA": 0, "shrinkB": 0},
            zorder=12,
        )


def _draw_depot(ax: plt.Axes, depot: tuple[int, int]) -> None:
    row, col = depot
    ax.scatter(col, row, marker="*", s=260, color="#ffe066", edgecolors="#111111", linewidths=1.0, zorder=13)
    ax.text(col + 0.45, row - 0.45, "DEPOT", fontsize=7.5, weight="bold", color="#111111", zorder=14)


def _draw_side_panel(panel: plt.Axes, env, history: list[dict[str, Any]], bbox: dict[str, Any], seed: int, strategy: str) -> None:
    panel.axis("off")
    accepted = [item for row in history for item in row.get("accepted", [])]
    screening = sum(1 for item in accepted if item.get("stage") == STAGE_SCREENING)
    review = sum(1 for item in accepted if item.get("stage") == STAGE_REVIEW)
    service = sum(1 for item in accepted if item.get("stage") == STAGE_SERVICE)
    if getattr(env, "task_lifecycle", "") == "v1_2_direct_service":
        action_lines = [
            f"service actions: {service}",
            f"open tasks: {env.open_task_count()}",
        ]
    else:
        action_lines = [
            f"screening actions: {screening}",
            f"review actions: {review}",
            f"review queue: {env.review_queue_length()}",
        ]
    lines = [
        "Run summary",
        "",
        f"strategy: {strategy}",
        f"seed: {seed}",
        f"steps: {env.current_step}",
        f"completed: {len(env.completed_tasks)}/{env.num_tasks}",
        *action_lines,
        f"path length: {env.total_path_length} cells",
        f"energy: {env.total_energy:.2f}",
        "",
        "OSM bbox",
        f"S {bbox.get('south', 'NA')}",
        f"W {bbox.get('west', 'NA')}",
        f"N {bbox.get('north', 'NA')}",
        f"E {bbox.get('east', 'NA')}",
        "",
        "Recent accepted actions",
    ]
    for item in accepted[-10:]:
        lines.append(f"{item.get('platform_id')} -> {item.get('task_id')} ({item.get('stage')})")

    panel.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=9.2,
        family="DejaVu Sans Mono",
        bbox={"facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.55"},
    )


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


if __name__ == "__main__":
    main()
