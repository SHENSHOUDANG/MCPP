"""Evaluate the trained 20x20 policy on larger zero-shot map sizes.

This script intentionally reuses the learned actor weights while replacing the
environment shape. It writes per-trial trajectories, path plots, CSV summaries,
and a short Markdown report under reports/large_map_transfer_<date>.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.config import ExperimentConfig, load_config
from mathbased_mcpp.env import GridCoverageEnv
from mathbased_mcpp.evaluation import coverage_efficiency_metrics
from mathbased_mcpp.ppo import ActorCritic


DEFAULT_CHECKPOINT = Path(
    r"E:\test plot\ablation_mapmsg_gat_on\20260526-113831\04-tier-4-20x20-4agents\best_policy.pt"
)


@dataclass(frozen=True)
class Scenario:
    size: int
    max_steps: int

    @property
    def name(self) -> str:
        return f"{self.size}x{self.size}"


SCENARIOS = (
    Scenario(30, 1125),
    Scenario(40, 2000),
    Scenario(50, 3125),
    Scenario(60, 4500),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "large_map_transfer_2026-05-29"))
    parser.add_argument("--seeds", default="20260530,20260531,20260532")
    parser.add_argument("--obstacle-ratio", type=float, default=0.05)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"
    figures_dir = output_dir / "figures"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")

    rows_path = output_dir / "detail_rows.csv"
    if args.skip_existing and rows_path.exists():
        rows = read_csv(rows_path)
    else:
        rows = run_trials(
            checkpoint=checkpoint,
            scenarios=SCENARIOS,
            seeds=seeds,
            obstacle_ratio=args.obstacle_ratio,
            trajectories_dir=trajectories_dir,
            figures_dir=figures_dir,
        )
        write_csv(rows_path, rows)

    summary_rows = summarize(rows)
    summary_path = output_dir / "summary_by_size.csv"
    write_csv(summary_path, summary_rows)

    summary_figure = plot_summary(summary_rows, figures_dir / "summary_by_size.png")
    report_path = output_dir / "large_map_transfer_report.md"
    report_path.write_text(build_report(checkpoint, rows, summary_rows, summary_figure, output_dir), encoding="utf-8")

    print(f"detail={rows_path}")
    print(f"summary={summary_path}")
    print(f"report={report_path}")
    print(f"figures={figures_dir}")
    print(f"trajectories={trajectories_dir}")


def run_trials(
    checkpoint: Path,
    scenarios: tuple[Scenario, ...],
    seeds: list[int],
    obstacle_ratio: float,
    trajectories_dir: Path,
    figures_dir: Path,
) -> list[dict[str, Any]]:
    base_config = load_config(checkpoint.parent / "course_config.json")
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        config_template = scenario_config(base_config, scenario, obstacle_ratio)
        model = load_policy_for_shape(checkpoint, config_template)
        for seed in seeds:
            trial_config = copy.deepcopy(config_template)
            trial_config.env.seed = seed
            trial_config.env.random_obstacle_seed = seed
            row = evaluate_trial(
                config=trial_config,
                model=model,
                scenario=scenario,
                seed=seed,
                trajectories_dir=trajectories_dir,
                figures_dir=figures_dir,
            )
            rows.append(row)
            print(
                f"{scenario.name} seed={seed} "
                f"coverage={row['coverage_ratio']:.4f} "
                f"auc={row['coverage_auc']:.4f} "
                f"steps={row['steps']} completed={row['completed']}"
            )
    return rows


def scenario_config(base_config: ExperimentConfig, scenario: Scenario, obstacle_ratio: float) -> ExperimentConfig:
    config = copy.deepcopy(base_config)
    config.env.width = scenario.size
    config.env.height = scenario.size
    config.env.max_steps = scenario.max_steps
    config.env.num_agents = 4
    config.env.random_corner_start = True
    config.env.start_positions = []
    config.env.teammate_positions = []
    config.env.obstacles = []
    config.env.obstacle_ratio = obstacle_ratio
    config.env.random_obstacle_count = 0
    config.env.random_obstacle_seeds = []
    config.env.map_refresh_episodes = 0
    return config


def load_policy_for_shape(checkpoint_path: Path, config: ExperimentConfig) -> ActorCritic:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = ActorCritic(
        observation_dim=int(payload["observation_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_dim=int(payload.get("hidden_dim", config.ppo.hidden_dim)),
        state_shape=(config.env.height, config.env.width),
        state_channels=int(payload.get("state_channels", 5)),
        state_metadata_dim=int(payload.get("state_metadata_dim", 7)),
        use_graph_attention=bool(payload.get("use_graph_attention", False)),
        gat_num_heads=int(payload.get("gat_num_heads", config.ppo.gat_num_heads)),
        gat_edge_dim=int(payload.get("gat_edge_dim", GridCoverageEnv(config.env).neighbor_feature_dim)),
        gat_residual=bool(payload.get("gat_residual", config.ppo.gat_residual)),
        gat_attention_dropout=float(payload.get("gat_attention_dropout", config.ppo.gat_attention_dropout)),
        node_message_dim=int(payload.get("node_message_dim", 0)),
        actor_encoder=str(payload.get("actor_encoder", "mlp")),
        actor_map_shape=tuple(payload["actor_map_shape"]) if payload.get("actor_encoder") == "cnn" else None,
        actor_metadata_dim=int(payload.get("actor_metadata_dim", 0)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def evaluate_trial(
    config: ExperimentConfig,
    model: ActorCritic,
    scenario: Scenario,
    seed: int,
    trajectories_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    env = GridCoverageEnv(config.env)
    observation = agent_observations(env.reset(seed=seed))
    state = env.global_state()
    trajectories = [[position] for position in env.positions]
    coverage_curve = [env.coverage_ratio()]
    device = next(model.parameters()).device
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}

    while not done:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        neighbor_mask = torch.as_tensor(env.neighbor_mask(), dtype=torch.bool, device=device)
        edge_features = (
            torch.as_tensor(env.neighbor_features(), dtype=torch.float32, device=device)
            if model.use_graph_attention and model.gat_edge_dim > 0
            else None
        )
        node_messages = (
            torch.as_tensor(env.node_messages(), dtype=torch.float32, device=device)
            if model.node_message_dim > 0
            else None
        )
        action_mask = (
            torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device)
            if config.ppo.use_action_mask
            else None
        )
        with torch.no_grad():
            actions, _, _ = model.act_batch(
                obs_tensor,
                state_tensor,
                neighbor_mask=neighbor_mask,
                edge_features=edge_features,
                node_messages=node_messages,
                action_mask=action_mask,
                deterministic=True,
            )
        result = env.step(actions.cpu().numpy().tolist())
        rewards = np.asarray(result.reward, dtype=np.float32)
        total_reward += float(rewards.mean() if rewards.ndim > 0 else rewards)
        observation = agent_observations(result.observation)
        state = result.state
        done = result.done
        info = result.info
        for index, position in enumerate(env.positions):
            trajectories[index].append(position)
        coverage_curve.append(env.coverage_ratio())

    budgets = metric_budgets(scenario.max_steps)
    metrics = coverage_efficiency_metrics(
        trajectories=trajectories,
        coverage_curve=coverage_curve,
        max_steps=env.config.max_steps,
        budgets=budgets,
        stall_steps=max(50, scenario.max_steps // 20),
    )
    stem = f"{scenario.name}_seed_{seed}"
    trajectory_path = trajectories_dir / f"{stem}.json"
    figure_path = figures_dir / f"{stem}_paths.png"
    write_trajectory_json(trajectory_path, env, trajectories, coverage_curve, info, metrics)
    render_paths(figure_path, env, trajectories, coverage_curve, scenario, seed)

    row: dict[str, Any] = {
        "scenario": scenario.name,
        "size": scenario.size,
        "seed": seed,
        "width": scenario.size,
        "height": scenario.size,
        "num_agents": env.num_agents,
        "max_steps": scenario.max_steps,
        "obstacle_ratio": config.env.obstacle_ratio,
        "obstacles": len(env.obstacles),
        "free_cells": len(env.free_cells),
        "coverage_ratio": float(info.get("coverage_ratio", env.coverage_ratio())),
        "completed": int(info.get("completed", False)),
        "steps": int(info.get("step_count", env.step_count)),
        "path_length": int(env.path_length),
        "avg_agent_path_length": float(np.mean(env.path_lengths)),
        "total_reward": total_reward,
        "trajectory_json": str(trajectory_path),
        "path_plot": str(figure_path),
    }
    row.update(metrics)
    row["metric_budgets"] = json.dumps(metrics["metric_budgets"])
    return row


def metric_budgets(max_steps: int) -> list[int]:
    budgets = [500, 1000, 1500, 2000, 3000, 4000, max_steps]
    return sorted({min(max_steps, item) for item in budgets if item > 0})


def write_trajectory_json(
    path: Path,
    env: GridCoverageEnv,
    trajectories: list[list[tuple[int, int]]],
    coverage_curve: list[float],
    info: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    payload = {
        "width": env.config.width,
        "height": env.config.height,
        "max_steps": env.config.max_steps,
        "num_agents": env.num_agents,
        "seed": env.config.seed,
        "obstacles": [list(cell) for cell in sorted(env.obstacles)],
        "free_cells": len(env.free_cells),
        "coverage_ratio": float(info.get("coverage_ratio", env.coverage_ratio())),
        "completed": bool(info.get("completed", False)),
        "steps": int(info.get("step_count", env.step_count)),
        "path_lengths": list(env.path_lengths),
        "coverage_curve": coverage_curve,
        "metrics": metrics,
        "trajectories": [[[int(row), int(col)] for row, col in path] for path in trajectories],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_paths(
    path: Path,
    env: GridCoverageEnv,
    trajectories: list[list[tuple[int, int]]],
    coverage_curve: list[float],
    scenario: Scenario,
    seed: int,
) -> None:
    grid = np.zeros((env.config.height, env.config.width), dtype=np.float32)
    for row, col in env.obstacles:
        grid[row, col] = 1.0

    fig_size = min(max(env.config.width / 5, 7), 13)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(grid, cmap="Greys", vmin=0.0, vmax=1.0, origin="upper", alpha=0.35)
    colors = ["#0969da", "#cf222e", "#2da44e", "#8250df"]
    for index, agent_path in enumerate(trajectories):
        rows = [cell[0] for cell in agent_path]
        cols = [cell[1] for cell in agent_path]
        color = colors[index % len(colors)]
        ax.plot(cols, rows, color=color, linewidth=1.35, alpha=0.86, label=f"agent {index}")
        ax.scatter(cols[0], rows[0], color=color, marker="s", s=48, edgecolor="white", linewidth=0.7, zorder=4)
        ax.scatter(cols[-1], rows[-1], color=color, marker="X", s=54, edgecolor="white", linewidth=0.7, zorder=4)
    ax.set_xlim(-0.5, env.config.width - 0.5)
    ax.set_ylim(env.config.height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-0.5, env.config.width, 5), minor=True)
    ax.set_yticks(np.arange(-0.5, env.config.height, 5), minor=True)
    ax.grid(which="minor", color="#d0d7de", linewidth=0.45, alpha=0.45)
    ax.set_title(
        f"{scenario.name}, seed {seed}: coverage {coverage_curve[-1] * 100:.1f}%, "
        f"steps {env.step_count}/{scenario.max_steps}"
    )
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["size"])].append(row)

    summaries: list[dict[str, Any]] = []
    for size in sorted(groups):
        group = groups[size]
        summary: dict[str, Any] = {
            "scenario": f"{size}x{size}",
            "size": size,
            "episodes": len(group),
            "num_agents": int(group[0]["num_agents"]),
            "max_steps": int(group[0]["max_steps"]),
            "free_cells_mean": mean(group, "free_cells"),
            "coverage_ratio_mean": mean(group, "coverage_ratio"),
            "coverage_ratio_min": min_value(group, "coverage_ratio"),
            "coverage_ratio_max": max_value(group, "coverage_ratio"),
            "completion_rate": mean(group, "completed"),
            "coverage_auc_mean": mean(group, "coverage_auc"),
            "steps_mean": mean(group, "steps"),
            "path_length_mean": mean(group, "path_length"),
            "avg_agent_path_length_mean": mean(group, "avg_agent_path_length"),
            "repeat_ratio_mean": mean(group, "repeat_ratio"),
            "repeat_ratio_after_90_mean": mean(group, "repeat_ratio_after_90"),
            "inter_agent_overlap_ratio_mean": mean(group, "inter_agent_overlap_ratio"),
            "stall_rate": mean(group, "stalled"),
            "stall_termination_coverage_mean": mean(group, "stall_termination_coverage"),
            "t90_reach_rate": reach_rate(group, "t90"),
            "t95_reach_rate": reach_rate(group, "t95"),
            "t99_reach_rate": reach_rate(group, "t99"),
            "t90_mean_reached": reached_mean(group, "t90"),
            "t95_mean_reached": reached_mean(group, "t95"),
            "t99_mean_reached": reached_mean(group, "t99"),
        }
        for key in sorted(k for k in group[0] if k.startswith("coverage_at_")):
            summary[f"{key}_mean"] = mean(group, key)
        summaries.append(summary)
    return summaries


def plot_summary(summary_rows: list[dict[str, Any]], output_path: Path) -> Path:
    sizes = [str(row["scenario"]) for row in summary_rows]
    x = np.arange(len(sizes))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    plot_bar(axes[0], x, sizes, [float(row["coverage_ratio_mean"]) for row in summary_rows], "Final coverage", ratio=True)
    plot_bar(axes[1], x, sizes, [float(row["coverage_auc_mean"]) for row in summary_rows], "Coverage-AUC", ratio=True)
    plot_bar(axes[2], x, sizes, [float(row["completion_rate"]) for row in summary_rows], "Completion rate", ratio=True)
    plot_bar(axes[3], x, sizes, [float(row["repeat_ratio_after_90_mean"]) for row in summary_rows], "Repeat after 90%", ratio=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_bar(ax: Any, x: np.ndarray, labels: list[str], values: list[float], title: str, ratio: bool = False) -> None:
    ax.bar(x, values, color="#2f6f8f")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ymax = max(values + [0.05])
    ax.set_ylim(0, min(1.05, ymax * 1.18) if ratio else ymax * 1.18)
    for xpos, value in zip(x, values):
        label = f"{value * 100:.1f}%" if ratio else f"{value:.2f}"
        ax.text(xpos, value, label, ha="center", va="bottom", fontsize=9)


def build_report(
    checkpoint: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    summary_figure: Path,
    output_dir: Path,
) -> str:
    lines = [
        "# Large-map zero-shot transfer report",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Training reference: 20x20, 4 agents, max_steps=500.",
        "- Evaluation maps: 30x30, 40x40, 50x50, 60x60.",
        "- Step budgets scale with map area: 1125, 2000, 3125, 4500.",
        f"- Episodes: {len(rows)} total, {len(rows) // max(len(summary_rows), 1)} seeds per size, 5% random obstacles.",
        "",
        f"![summary]({summary_figure.relative_to(output_dir).as_posix()})",
        "",
        "## Summary",
        "",
        "| Map | Episodes | Max steps | Final coverage | Min coverage | AUC | Completion | Steps | Avg path/agent | Repeat after 90% | Overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['scenario']} | {int(row['episodes'])} | {int(row['max_steps'])} | "
            f"{pct(row['coverage_ratio_mean'])} | {pct(row['coverage_ratio_min'])} | "
            f"{pct(row['coverage_auc_mean'])} | {pct(row['completion_rate'])} | "
            f"{float(row['steps_mean']):.1f} | {float(row['avg_agent_path_length_mean']):.1f} | "
            f"{pct(row['repeat_ratio_after_90_mean'])} | {pct(row['inter_agent_overlap_ratio_mean'])} |"
        )
    lines.extend(["", "## Path Plots", ""])
    for row in rows:
        figure = Path(str(row["path_plot"]))
        try:
            figure_text = figure.relative_to(output_dir).as_posix()
        except ValueError:
            figure_text = str(figure)
        lines.append(f"- {row['scenario']} seed {row['seed']}: [{figure.name}]({figure_text})")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These are deterministic rollouts; no additional training was performed.",
            "- Completion means all free cells were covered before the step limit.",
            "- Coverage-AUC is the mean coverage over the full step budget, so it rewards faster coverage as well as final coverage.",
        ]
    )
    return "\n".join(lines) + "\n"


def agent_observations(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    if observation.ndim == 1:
        return observation.reshape(1, -1)
    return observation


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    return float(np.mean(values)) if values else 0.0


def min_value(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.min([float(row[key]) for row in rows]))


def max_value(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.max([float(row[key]) for row in rows]))


def reached_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in ("", None)]


def reach_rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(len(reached_values(rows, key)) / max(len(rows), 1))


def reached_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = reached_values(rows, key)
    return float(np.mean(values)) if values else 0.0


def pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
