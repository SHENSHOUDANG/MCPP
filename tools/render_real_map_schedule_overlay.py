from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
from mathbased_mcpp.port_inspection.simple_planner import shortest_path


DEFAULT_BBOX = {
    "south": 30.600,
    "west": 122.010,
    "north": 30.655,
    "east": 122.100,
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Overlay UAV-USV scheduling on a map image.")
    parser.add_argument("--config", default="configs/port_yangshan_task_initial_v1.toml")
    parser.add_argument(
        "--background",
        default="outputs/real_map_tiles/yangshan_arcgis_world_imagery_export_hires.png",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--strategy", choices=("legacy_order", "global_score"), default="global_score")
    parser.add_argument("--show-all-tasks", action="store_true", help="Show every model task instead of only scheduled tasks.")
    parser.add_argument(
        "--output",
        default="outputs/real_map_tiles/yangshan_task_initial_schedule_overlay_seed11.png",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    env = build_env(config)
    history = _run_schedule(env, config, seed=args.seed, steps=args.steps, strategy=args.strategy)
    output = render_overlay(
        env=env,
        history=history,
        background=Path(args.background),
        output=Path(args.output),
        seed=args.seed,
        strategy=args.strategy,
        show_all_tasks=args.show_all_tasks,
    )
    print(output.resolve())


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


def render_overlay(
    env,
    history: list[dict[str, Any]],
    background: Path,
    output: Path,
    seed: int,
    strategy: str,
    show_all_tasks: bool,
) -> Path:
    image = Image.open(background).convert("RGB")
    width, height = image.size
    bbox = _bbox(env)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 11), constrained_layout=True)
    ax.imshow(image)
    ax.set_axis_off()

    _draw_task_layer(ax, env, history, bbox, width, height, show_all_tasks=show_all_tasks)
    _draw_route_layer(ax, env, bbox, width, height)
    _draw_depot(ax, env.grid.depot, env.grid, bbox, width, height)
    _draw_summary(ax, env, history, seed, strategy)

    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output


def _draw_task_layer(ax, env, history: list[dict[str, Any]], bbox: dict[str, float], width: int, height: int, show_all_tasks: bool) -> None:
    scheduled = {str(item.get("task_id")) for row in history for item in row.get("accepted", [])}
    visible = set(env.completed_tasks) | scheduled
    for task in env.tasks:
        if not show_all_tasks and task.task_id not in visible:
            continue
        xy = [_cell_to_pixel(cell, env.grid, bbox, width, height) for cell in task.cells]
        xs = [item[0] for item in xy]
        ys = [item[1] for item in xy]
        is_done = task.task_id in env.completed_tasks
        if task.geometry == "line":
            ax.plot(xs, ys, color="#8b5cf6", linewidth=2.4, alpha=0.78, solid_capstyle="round", zorder=5)
            ax.plot(xs, ys, color="#ffffff", linewidth=0.8, alpha=0.72, solid_capstyle="round", zorder=6)
            label_x, label_y = xs[len(xs) // 2], ys[len(ys) // 2]
        elif task.geometry == "area":
            label_x, label_y = float(np.mean(xs)), float(np.mean(ys))
            ax.scatter(
                [label_x],
                [label_y],
                marker="D",
                s=54,
                color="#fbbf24",
                edgecolors="#78350f",
                linewidths=0.5,
                alpha=0.86,
                zorder=8,
            )
        else:
            color = "#22c55e" if is_done else "#38bdf8"
            ax.scatter(xs, ys, marker="o", s=42, color=color, edgecolors="#0f172a", linewidths=0.5, alpha=0.86, zorder=8)
            label_x, label_y = xs[0], ys[0]
        if is_done or task.risk >= 3:
            ax.text(
                label_x + 7,
                label_y - 7,
                task.task_id,
                fontsize=8,
                color="#ffffff",
                weight="bold",
                path_effects=[],
                bbox={"facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.58, "boxstyle": "round,pad=0.18"},
                zorder=12,
            )


def _draw_route_layer(ax, env, bbox: dict[str, float], width: int, height: int) -> None:
    colors = {
        "UAV": ["#60a5fa", "#22d3ee", "#93c5fd", "#38bdf8"],
        "USV": ["#ef4444", "#f97316", "#dc2626", "#fb7185"],
    }
    for index, platform in enumerate(env.platforms):
        if len(platform.route) < 2:
            continue
        route = _expanded_route(env, platform)
        xy = [_cell_to_pixel(cell, env.grid, bbox, width, height) for cell in route]
        xs = [item[0] for item in xy]
        ys = [item[1] for item in xy]
        palette = colors.get(platform.platform_type, ["#ffffff"])
        color = palette[index % len(palette)]
        linestyle = "--" if platform.platform_type == "UAV" else "-"
        linewidth = 2.0 if platform.platform_type == "UAV" else 2.4
        ax.plot(xs, ys, color="#0f172a", linewidth=linewidth + 1.4, linestyle=linestyle, alpha=0.42, zorder=14)
        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.92, zorder=15)
        ax.scatter(xs[-1], ys[-1], marker="X", s=72, color=color, edgecolors="#ffffff", linewidths=0.55, zorder=16)
        ax.text(
            xs[-1] + 8,
            ys[-1] - 8,
            platform.platform_id,
            fontsize=8.5,
            color="#ffffff",
            weight="bold",
            bbox={"facecolor": "#111827", "edgecolor": "none", "alpha": 0.62, "boxstyle": "round,pad=0.18"},
            zorder=17,
        )


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


def _draw_depot(ax, depot: tuple[int, int], grid, bbox: dict[str, float], width: int, height: int) -> None:
    x, y = _cell_to_pixel(depot, grid, bbox, width, height)
    ax.scatter(x, y, marker="*", s=210, color="#fde047", edgecolors="#111827", linewidths=1.0, zorder=18)
    ax.text(
        x + 9,
        y - 9,
        "DEPOT",
        fontsize=9,
        color="#111827",
        weight="bold",
        bbox={"facecolor": "#fefce8", "edgecolor": "none", "alpha": 0.7, "boxstyle": "round,pad=0.18"},
        zorder=19,
    )


def _draw_summary(ax, env, history: list[dict[str, Any]], seed: int, strategy: str) -> None:
    accepted = [item for row in history for item in row.get("accepted", [])]
    if getattr(env, "task_lifecycle", "") == "v1_2_direct_service":
        service = sum(1 for item in accepted if item.get("stage") == "service")
        text = (
            f"UAV-USV service schedule on chart imagery\n"
            f"strategy: {strategy} | seed: {seed} | steps: {env.current_step}\n"
            f"completed: {len(env.completed_tasks)}/{env.num_tasks} | service actions: {service}\n"
            f"path length: {env.total_path_length} cells | open tasks: {env.open_task_count()}"
        )
    else:
        screening = sum(1 for item in accepted if item.get("stage") == "screening")
        review = sum(1 for item in accepted if item.get("stage") == "review")
        text = (
            f"UAV-USV schedule on ArcGIS imagery\n"
            f"strategy: {strategy} | seed: {seed} | steps: {env.current_step}\n"
            f"completed: {len(env.completed_tasks)}/{env.num_tasks} | screening: {screening} | review: {review}\n"
            f"path length: {env.total_path_length} cells | review queue: {env.review_queue_length()}"
        )
    ax.text(
        0.015,
        0.025,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#f8fafc",
        bbox={"facecolor": "#020617", "edgecolor": "#e2e8f0", "linewidth": 0.4, "alpha": 0.58, "boxstyle": "round,pad=0.42"},
        zorder=30,
    )


def _cell_to_pixel(cell: tuple[int, int], grid, bbox: dict[str, float], width: int, height: int) -> tuple[float, float]:
    if grid is None:
        grid_height = 41
        grid_width = 58
    else:
        grid_height = grid.height
        grid_width = grid.width
    row, col = cell
    lon = bbox["west"] + (col + 0.5) / grid_width * (bbox["east"] - bbox["west"])
    lat = bbox["north"] - (row + 0.5) / grid_height * (bbox["north"] - bbox["south"])
    x = (lon - bbox["west"]) / (bbox["east"] - bbox["west"]) * width
    y = (bbox["north"] - lat) / (bbox["north"] - bbox["south"]) * height
    return x, y


def _bbox(env) -> dict[str, float]:
    raw = env.grid.metadata.get("bbox", {})
    return {key: float(raw.get(key, DEFAULT_BBOX[key])) for key in DEFAULT_BBOX}


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


if __name__ == "__main__":
    main()
