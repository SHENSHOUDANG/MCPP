from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .runtime import configure_runtime

configure_runtime()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .config import ExperimentConfig, GridPosition
from .env import GridCoverageEnv


def render_trajectory(
    config: ExperimentConfig,
    trajectory: Iterable[GridPosition] | Sequence[Iterable[GridPosition]],
    output_path: str | Path,
) -> Path:
    env = GridCoverageEnv(config.env)
    trajectories = _normalize_trajectories(trajectory)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig_width = min(max(config.env.width / 2, 5), 14)
    fig_height = min(max(config.env.height / 2, 5), 14)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.5, config.env.width - 0.5)
    ax.set_ylim(config.env.height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(config.env.width))
    ax.set_yticks(range(config.env.height))
    ax.grid(color="#d0d7de", linewidth=0.8)

    for row, col in env.free_cells:
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor="#f6f8fa", edgecolor="none"))
    for row, col in env.obstacles:
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor="#24292f", edgecolor="none"))

    colors = ["#0969da", "#cf222e", "#2da44e", "#8250df", "#bf8700", "#1b7f83", "#d1242f", "#57606a"]
    for index, path in enumerate(trajectories):
        if not path:
            continue
        ys = [row for row, _ in path]
        xs = [col for _, col in path]
        color = colors[index % len(colors)]
        label = "path" if len(trajectories) == 1 else f"agent {index}"
        ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=3, label=label)
        ax.scatter([xs[0]], [ys[0]], color=color, s=70, marker="s", zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], color=color, s=70, marker="X", zorder=3)
    if any(trajectories):
        ax.legend(loc="upper right")

    ax.set_title("PPO Grid Coverage Trajectory")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _normalize_trajectories(
    trajectory: Iterable[GridPosition] | Sequence[Iterable[GridPosition]],
) -> list[list[GridPosition]]:
    paths = list(trajectory)
    if not paths:
        return []
    first = paths[0]
    if isinstance(first, tuple) and len(first) == 2:
        return [paths]  # type: ignore[list-item]
    return [list(path) for path in paths]  # type: ignore[arg-type]
