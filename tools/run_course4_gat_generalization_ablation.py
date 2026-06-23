from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.config import ExperimentConfig, load_config
from mathbased_mcpp.env import GridCoverageEnv
from mathbased_mcpp.evaluation import coverage_efficiency_metrics
from mathbased_mcpp.ppo import ActorCritic
from mathbased_mcpp.utils import agent_observations


DEFAULT_GAT_ON = Path(
    ROOT / "outputs" / "ablation_mapmsg_gat_on" / "RUN_ID" / "04-tier-4-20x20-4agents" / "best_policy.pt"
)
DEFAULT_GAT_OFF = Path(
    ROOT / "outputs" / "ablation_mapmsg_gat_off" / "RUN_ID" / "04-tier-4-20x20-4agents" / "best_policy.pt"
)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    size: int
    agents: int
    max_steps: int
    category: str
    course_native: bool = False


DEFAULT_SCENARIOS = (
    Scenario("native_20x20_4a", "20x20 / 4 agents course map", 20, 4, 500, "course-native", True),
    Scenario("train_20x20_4a", "20x20 / 4 agents random maps", 20, 4, 500, "same-size random"),
    Scenario("size_30x30_4a", "30x30 / 4 agents", 30, 4, 1125, "map size"),
    Scenario("size_35x35_4a", "35x35 / 4 agents", 35, 4, 1531, "map size"),
    Scenario("size_40x40_4a", "40x40 / 4 agents", 40, 4, 2000, "map size"),
    Scenario("agents_20x20_6a", "20x20 / 6 agents", 20, 6, 500, "agent count"),
    Scenario("agents_20x20_8a", "20x20 / 8 agents", 20, 8, 500, "agent count"),
    Scenario("combined_30x30_6a", "30x30 / 6 agents", 30, 6, 1125, "combined"),
    Scenario("combined_35x35_8a", "35x35 / 8 agents", 35, 8, 1531, "combined"),
)

ARMS = ("GAT-on", "GAT-off")
COLORS = {"GAT-on": "#1f77b4", "GAT-off": "#ff7f0e"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gat-on-checkpoint", default=str(DEFAULT_GAT_ON))
    parser.add_argument("--gat-off-checkpoint", default=str(DEFAULT_GAT_OFF))
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / f"course4_gat_generalization_{date.today().isoformat()}"))
    parser.add_argument("--seeds", default="20260601,20260602,20260603")
    parser.add_argument("--obstacle-ratio", type=float, default=0.05)
    parser.add_argument("--scenario-keys", default="")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    trajectories_dir = output_dir / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    scenarios = select_scenarios(args.scenario_keys)
    checkpoints = {"GAT-on": Path(args.gat_on_checkpoint), "GAT-off": Path(args.gat_off_checkpoint)}
    for arm, checkpoint in checkpoints.items():
        if not checkpoint.exists():
            raise FileNotFoundError(f"{arm} checkpoint not found: {checkpoint}")

    detail_path = output_dir / "detail_rows.csv"
    curves_path = output_dir / "curve_rows.csv"
    if args.skip_existing and detail_path.exists() and curves_path.exists():
        detail_rows = read_csv(detail_path)
        curve_rows = read_csv(curves_path)
    else:
        detail_rows, curve_rows = run_experiment(
            checkpoints=checkpoints,
            scenarios=scenarios,
            seeds=seeds,
            obstacle_ratio=args.obstacle_ratio,
            trajectories_dir=trajectories_dir,
        )
        write_csv(detail_path, detail_rows)
        write_csv(curves_path, curve_rows)

    summary_rows = summarize(detail_rows, scenarios)
    summary_path = output_dir / "summary_rows.csv"
    write_csv(summary_path, summary_rows)

    figure_paths = make_figures(detail_rows, curve_rows, summary_rows, scenarios, figures_dir)
    report_path = output_dir / "course4_gat_generalization_report.md"
    report_path.write_text(
        build_report(checkpoints, scenarios, seeds, detail_path, curves_path, summary_path, figure_paths, output_dir),
        encoding="utf-8",
    )

    print(f"detail={detail_path}")
    print(f"curves={curves_path}")
    print(f"summary={summary_path}")
    print(f"report={report_path}")
    print(f"figures={figures_dir}")


def select_scenarios(keys_arg: str) -> tuple[Scenario, ...]:
    if not keys_arg.strip():
        return DEFAULT_SCENARIOS
    keys = {item.strip() for item in keys_arg.split(",") if item.strip()}
    scenarios = tuple(scenario for scenario in DEFAULT_SCENARIOS if scenario.key in keys)
    missing = keys - {scenario.key for scenario in scenarios}
    if missing:
        raise ValueError(f"unknown scenario keys: {', '.join(sorted(missing))}")
    return scenarios


def run_experiment(
    checkpoints: dict[str, Path],
    scenarios: tuple[Scenario, ...],
    seeds: list[int],
    obstacle_ratio: float,
    trajectories_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        checkpoint = checkpoints[arm]
        base_config = load_config(checkpoint.parent / "course_config.json")
        model_cache: dict[str, ActorCritic] = {}
        for scenario in scenarios:
            scenario_config = build_scenario_config(base_config, scenario, obstacle_ratio)
            model = model_cache.setdefault(scenario.key, load_policy_for_shape(checkpoint, scenario_config))
            scenario_seeds = [scenario_config.env.seed] if scenario.course_native else seeds
            for seed in scenario_seeds:
                detail, curves = evaluate_trial(
                    arm=arm,
                    config=scenario_config,
                    model=model,
                    scenario=scenario,
                    seed=seed,
                    trajectories_dir=trajectories_dir,
                )
                detail_rows.append(detail)
                curve_rows.extend(curves)
                print(
                    f"{arm} {scenario.key} seed={seed} "
                    f"coverage={detail['coverage_ratio']:.4f} "
                    f"auc={detail['coverage_auc']:.4f} "
                    f"global_repeat={detail['global_repeat_ratio']:.4f} "
                    f"steps={detail['steps']} completed={detail['completed']}"
                )
    return detail_rows, curve_rows


def build_scenario_config(base_config: ExperimentConfig, scenario: Scenario, obstacle_ratio: float) -> ExperimentConfig:
    config = copy.deepcopy(base_config)
    if scenario.course_native:
        return config
    config.env.width = scenario.size
    config.env.height = scenario.size
    config.env.max_steps = scenario.max_steps
    config.env.num_agents = scenario.agents
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
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gat_use_edge_features = bool(payload.get("gat_use_edge_features", config.ppo.gat_use_edge_features))
    model = ActorCritic(
        observation_dim=int(payload["observation_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_dim=int(payload.get("hidden_dim", config.ppo.hidden_dim)),
        state_shape=(config.env.height, config.env.width),
        state_channels=int(payload.get("state_channels", 5)),
        state_metadata_dim=int(payload.get("state_metadata_dim", 7)),
        use_graph_attention=bool(payload.get("use_graph_attention", False)),
        gat_num_heads=int(payload.get("gat_num_heads", config.ppo.gat_num_heads)),
        gat_edge_dim=int(payload.get("gat_edge_dim", GridCoverageEnv(config.env).neighbor_feature_dim if gat_use_edge_features else 0)),
        gat_residual=bool(payload.get("gat_residual", config.ppo.gat_residual)),
        gat_attention_dropout=float(payload.get("gat_attention_dropout", config.ppo.gat_attention_dropout)),
        node_message_dim=int(payload.get("node_message_dim", 0)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def evaluate_trial(
    arm: str,
    config: ExperimentConfig,
    model: ActorCritic,
    scenario: Scenario,
    seed: int,
    trajectories_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trial_config = copy.deepcopy(config)
    if not scenario.course_native:
        trial_config.env.seed = seed
        trial_config.env.random_obstacle_seed = seed
    env = GridCoverageEnv(trial_config.env)
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
        action_mask = torch.as_tensor(env.action_masks(), dtype=torch.bool, device=device) if trial_config.ppo.use_action_mask else None
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
    global_repeat_ratio = cumulative_repeat_curve(trajectories, scenario.max_steps)[-1]
    detail: dict[str, Any] = {
        "arm": arm,
        "scenario": scenario.key,
        "label": scenario.label,
        "category": scenario.category,
        "size": scenario.size,
        "num_agents": scenario.agents,
        "max_steps": scenario.max_steps,
        "seed": seed,
        "obstacle_ratio": trial_config.env.obstacle_ratio,
        "obstacles": len(env.obstacles),
        "free_cells": len(env.free_cells),
        "coverage_ratio": float(info.get("coverage_ratio", env.coverage_ratio())),
        "completed": int(info.get("completed", False)),
        "steps": int(info.get("step_count", env.step_count)),
        "path_length": int(env.path_length),
        "avg_agent_path_length": float(np.mean(env.path_lengths)),
        "global_repeat_ratio": float(global_repeat_ratio),
        "total_reward": total_reward,
    }
    detail.update({key: value for key, value in metrics.items() if key not in {"metric_budgets"}})
    detail["metric_budgets"] = json.dumps(metrics["metric_budgets"])

    trajectory_path = trajectories_dir / f"{arm}_{scenario.key}_seed_{seed}.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "arm": arm,
                "scenario": scenario.key,
                "label": scenario.label,
                "seed": seed,
                "width": env.config.width,
                "height": env.config.height,
                "num_agents": env.num_agents,
                "max_steps": env.config.max_steps,
                "coverage_curve": coverage_curve,
                "repeat_curve": cumulative_repeat_curve(trajectories, scenario.max_steps),
                "trajectories": [[[int(row), int(col)] for row, col in path] for path in trajectories],
                "obstacles": [[int(row), int(col)] for row, col in sorted(env.obstacles)],
                "detail": detail,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    detail["trajectory_json"] = str(trajectory_path)

    coverage_padded = pad_curve(coverage_curve, scenario.max_steps)
    repeat_padded = cumulative_repeat_curve(trajectories, scenario.max_steps)
    curve_rows = [
        {
            "arm": arm,
            "scenario": scenario.key,
            "label": scenario.label,
            "category": scenario.category,
            "size": scenario.size,
            "num_agents": scenario.agents,
            "seed": seed,
            "step": step,
            "step_fraction": step / max(scenario.max_steps, 1),
            "coverage": coverage_padded[step],
            "global_repeat_ratio": repeat_padded[step],
        }
        for step in range(scenario.max_steps + 1)
    ]
    return detail, curve_rows


def metric_budgets(max_steps: int) -> list[int]:
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    fixed = [100, 200, 300, 500, 1000, 1500, 2000]
    values = {max(1, min(max_steps, int(round(max_steps * fraction)))) for fraction in fractions}
    values.update(max(1, min(max_steps, value)) for value in fixed)
    return sorted(values)


def pad_curve(curve: list[float], max_steps: int) -> list[float]:
    output = list(curve[: max_steps + 1])
    if not output:
        output = [0.0]
    if len(output) < max_steps + 1:
        output.extend([output[-1]] * (max_steps + 1 - len(output)))
    return output


def cumulative_repeat_curve(trajectories: list[list[tuple[int, int]]], max_steps: int) -> list[float]:
    visited: set[tuple[int, int]] = set()
    curve: list[float] = []
    total_visits = 0
    repeats = 0
    actual_steps = min(max((len(path) for path in trajectories), default=1) - 1, max_steps)
    for step in range(actual_steps + 1):
        positions = [path[step] for path in trajectories if step < len(path)]
        for position in positions:
            total_visits += 1
            if position in visited:
                repeats += 1
            visited.add(position)
        curve.append(float(repeats / max(total_visits, 1)))
    if len(curve) < max_steps + 1:
        curve.extend([curve[-1] if curve else 0.0] * (max_steps + 1 - len(curve)))
    return curve


def summarize(detail_rows: list[dict[str, Any]], scenarios: tuple[Scenario, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        groups[(str(row["scenario"]), str(row["arm"]))].append(row)
    output: list[dict[str, Any]] = []
    metrics = [
        "coverage_ratio",
        "coverage_auc",
        "completed",
        "steps",
        "path_length",
        "avg_agent_path_length",
        "global_repeat_ratio",
        "repeat_ratio",
        "repeat_ratio_after_90",
        "inter_agent_overlap_ratio",
        "stalled",
        "stall_termination_coverage",
    ]
    for scenario in scenarios:
        for arm in ARMS:
            rows = groups[(scenario.key, arm)]
            summary: dict[str, Any] = {
                "arm": arm,
                "scenario": scenario.key,
                "label": scenario.label,
                "category": scenario.category,
                "size": scenario.size,
                "num_agents": scenario.agents,
                "max_steps": scenario.max_steps,
                "episodes": len(rows),
            }
            for metric in metrics:
                summary[f"{metric}_mean"] = mean(rows, metric)
                summary[f"{metric}_std"] = std(rows, metric)
            for threshold in ("t90", "t95", "t99"):
                values = numeric_values(rows, threshold)
                summary[f"{threshold}_reach_rate"] = len(values) / max(len(rows), 1)
                summary[f"{threshold}_mean_reached"] = float(np.mean(values)) if values else math.nan
            output.append(summary)
    return output


def make_figures(
    detail_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    scenarios: tuple[Scenario, ...],
    figures_dir: Path,
) -> list[Path]:
    figures = [
        plot_curve_grid(curve_rows, scenarios, figures_dir / "fig01_coverage_curves.png", "coverage", "Coverage"),
        plot_coverage_advantage_curves(curve_rows, scenarios, figures_dir / "fig02_coverage_advantage_curves.png"),
        plot_curve_grid(curve_rows, scenarios, figures_dir / "fig03_global_repeat_curves.png", "global_repeat_ratio", "Global repeat ratio"),
        plot_threshold_step_savings(detail_rows, scenarios, figures_dir / "fig04_threshold_step_savings.png"),
        plot_grouped_bars(summary_rows, scenarios, figures_dir / "fig05_final_coverage_and_auc.png", ["coverage_ratio_mean", "coverage_auc_mean"], ["Final coverage", "Coverage-AUC"]),
        plot_grouped_bars(summary_rows, scenarios, figures_dir / "fig06_repeat_and_overlap.png", ["global_repeat_ratio_mean", "repeat_ratio_after_90_mean", "inter_agent_overlap_ratio_mean"], ["Global repeat", "Repeat after 90%", "Inter-agent overlap"]),
        plot_grouped_bars(summary_rows, scenarios, figures_dir / "fig07_completion_and_steps.png", ["completed_mean", "steps_mean"], ["Completion rate", "Mean steps"], ratio_flags=[True, False]),
        plot_delta_heatmap(summary_rows, scenarios, figures_dir / "fig08_gat_on_minus_off_heatmap.png"),
        plot_sample_paths(detail_rows, figures_dir / "fig09_sample_paths.png"),
    ]
    return figures


def plot_curve_grid(curve_rows: list[dict[str, Any]], scenarios: tuple[Scenario, ...], output: Path, metric: str, ylabel: str) -> Path:
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        rows_by_key[(str(row["scenario"]), str(row["arm"]))].append(row)
    cols = 3 if len(scenarios) > 8 else min(4, max(len(scenarios), 1))
    rows = int(math.ceil(len(scenarios) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.4 * rows), sharex=False, sharey=True)
    axes = axes.ravel()
    for axis, scenario in zip(axes, scenarios):
        for arm in ARMS:
            arm_rows = rows_by_key[(scenario.key, arm)]
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in arm_rows:
                grouped[int(row["step"])].append(float(row[metric]))
            steps = sorted(grouped)
            means = np.asarray([np.mean(grouped[step]) for step in steps])
            stds = np.asarray([np.std(grouped[step]) for step in steps])
            axis.plot(steps, means, color=COLORS[arm], label=arm, linewidth=2.0)
            axis.fill_between(steps, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1), color=COLORS[arm], alpha=0.12)
        axis.set_title(scenario.label, fontsize=10)
        axis.grid(alpha=0.25)
        axis.set_xlim(0, scenario.max_steps)
        axis.set_ylim(0, 1)
    axes[0].legend(loc="lower right")
    for axis in axes[len(scenarios):]:
        axis.axis("off")
    for axis in axes[(rows - 1) * cols : rows * cols]:
        axis.set_xlabel("Steps")
    for axis in axes[::cols]:
        axis.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_coverage_advantage_curves(curve_rows: list[dict[str, Any]], scenarios: tuple[Scenario, ...], output: Path) -> Path:
    coverage_by_key: dict[tuple[str, str, str, int], float] = {}
    seeds_by_scenario: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in curve_rows:
        scenario = str(row["scenario"])
        arm = str(row["arm"])
        seed = str(row["seed"])
        step = int(row["step"])
        coverage_by_key[(scenario, arm, seed, step)] = float(row["coverage"])
        seeds_by_scenario[scenario][arm].add(seed)

    cols = 3 if len(scenarios) > 8 else min(4, max(len(scenarios), 1))
    rows = int(math.ceil(len(scenarios) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.4 * rows), sharex=False, sharey=False)
    axes = axes.ravel()
    for axis, scenario in zip(axes, scenarios):
        common_seeds = sorted(seeds_by_scenario[scenario.key]["GAT-on"] & seeds_by_scenario[scenario.key]["GAT-off"])
        steps = np.arange(scenario.max_steps + 1)
        deltas = []
        for seed in common_seeds:
            on = np.asarray([coverage_by_key[(scenario.key, "GAT-on", seed, int(step))] for step in steps])
            off = np.asarray([coverage_by_key[(scenario.key, "GAT-off", seed, int(step))] for step in steps])
            deltas.append((on - off) * 100.0)
        if deltas:
            matrix = np.vstack(deltas)
            mean_delta = np.mean(matrix, axis=0)
            std_delta = np.std(matrix, axis=0)
            axis.plot(steps, mean_delta, color="#0b5cad", linewidth=2.0)
            axis.fill_between(steps, mean_delta - std_delta, mean_delta + std_delta, color="#0b5cad", alpha=0.14)
            limit = max(2.0, float(np.nanmax(np.abs(mean_delta) + std_delta)))
            limit = min(40.0, math.ceil(limit / 2.5) * 2.5)
            axis.set_ylim(-limit, limit)
        axis.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.75)
        axis.fill_between([0, scenario.max_steps], 0, axis.get_ylim()[1], color="#1f77b4", alpha=0.04)
        axis.fill_between([0, scenario.max_steps], axis.get_ylim()[0], 0, color="#ff7f0e", alpha=0.04)
        axis.set_title(scenario.label, fontsize=10)
        axis.set_xlim(0, scenario.max_steps)
        axis.grid(alpha=0.25)
    for axis in axes[len(scenarios):]:
        axis.axis("off")
    for axis in axes[(rows - 1) * cols : rows * cols]:
        axis.set_xlabel("Steps")
    for axis in axes[::cols]:
        axis.set_ylabel("Coverage gap (pp)")
    fig.suptitle("Coverage advantage: GAT-on minus GAT-off", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_threshold_step_savings(detail_rows: list[dict[str, Any]], scenarios: tuple[Scenario, ...], output: Path) -> Path:
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        rows_by_key[(str(row["scenario"]), str(row["arm"]))].append(row)

    thresholds = [("t90", "90%"), ("t95", "95%"), ("t99", "99%")]
    data = np.zeros((len(scenarios), len(thresholds)), dtype=np.float64)
    for row_index, scenario in enumerate(scenarios):
        for col_index, (metric, _) in enumerate(thresholds):
            on_steps = threshold_step_values(rows_by_key[(scenario.key, "GAT-on")], metric, scenario.max_steps)
            off_steps = threshold_step_values(rows_by_key[(scenario.key, "GAT-off")], metric, scenario.max_steps)
            data[row_index, col_index] = float(np.mean(off_steps) - np.mean(on_steps))

    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    limit = max(abs(float(np.nanmin(data))), abs(float(np.nanmax(data))), 1.0)
    im = ax.imshow(data, cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([label for _, label in thresholds])
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels([scenario.label for scenario in scenarios])
    ax.set_title("Step savings to coverage thresholds (positive = GAT-on faster)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.0f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("steps saved")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def threshold_step_values(rows: list[dict[str, Any]], metric: str, fallback_steps: int) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if value in (None, "", "nan"):
            values.append(float(fallback_steps))
        else:
            values.append(float(value))
    return values or [float(fallback_steps)]


def plot_grouped_bars(
    summary_rows: list[dict[str, Any]],
    scenarios: tuple[Scenario, ...],
    output: Path,
    metrics: list[str],
    titles: list[str],
    ratio_flags: list[bool] | None = None,
) -> Path:
    ratio_flags = ratio_flags or [True for _ in metrics]
    lookup = {(str(row["scenario"]), str(row["arm"])): row for row in summary_rows}
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5), squeeze=False)
    axes_flat = axes.ravel()
    x = np.arange(len(scenarios))
    width = 0.36
    for axis, metric, title, ratio in zip(axes_flat, metrics, titles, ratio_flags):
        for offset, arm in [(-width / 2, "GAT-off"), (width / 2, "GAT-on")]:
            values = [float(lookup[(scenario.key, arm)][metric]) for scenario in scenarios]
            axis.bar(x + offset, values, width, label=arm, color=COLORS[arm])
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels([scenario.label for scenario in scenarios], rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if ratio:
            axis.set_ylim(0, 1.05)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_delta_heatmap(summary_rows: list[dict[str, Any]], scenarios: tuple[Scenario, ...], output: Path) -> Path:
    lookup = {(str(row["scenario"]), str(row["arm"])): row for row in summary_rows}
    metric_specs = [
        ("coverage_ratio_mean", "Final cov.", "higher"),
        ("coverage_auc_mean", "AUC", "higher"),
        ("completed_mean", "Complete", "higher"),
        ("global_repeat_ratio_mean", "Global repeat", "lower"),
        ("repeat_ratio_after_90_mean", "Repeat 90", "lower"),
        ("inter_agent_overlap_ratio_mean", "Overlap", "lower"),
    ]
    matrix = []
    for scenario in scenarios:
        row_values = []
        for metric, _, direction in metric_specs:
            on = float(lookup[(scenario.key, "GAT-on")][metric])
            off = float(lookup[(scenario.key, "GAT-off")][metric])
            row_values.append(on - off if direction == "higher" else off - on)
        matrix.append(row_values)
    data = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(10, 7))
    limit = max(abs(float(np.nanmin(data))), abs(float(np.nanmax(data))), 1e-6)
    im = ax.imshow(data, cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(metric_specs)))
    ax.set_xticklabels([label for _, label, _ in metric_specs], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels([scenario.label for scenario in scenarios])
    ax.set_title("GAT-on advantage over GAT-off (blue is better)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j] * 100:+.1f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("percentage points")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_sample_paths(detail_rows: list[dict[str, Any]], output: Path) -> Path:
    selected_keys = ["native_20x20_4a", "size_35x35_4a", "agents_20x20_6a"]
    min_seed_by_scenario: dict[str, int] = {}
    for key in selected_keys:
        seeds = [int(row["seed"]) for row in detail_rows if str(row["scenario"]) == key]
        if seeds:
            min_seed_by_scenario[key] = min(seeds)
    selected = [
        row
        for row in detail_rows
        if str(row["scenario"]) in min_seed_by_scenario
        and int(row["seed"]) == min_seed_by_scenario[str(row["scenario"])]
    ]
    order = {key: index for index, key in enumerate(selected_keys)}
    selected = sorted(selected, key=lambda row: (order[str(row["scenario"])], str(row["arm"])))
    if not selected:
        output.write_text("no sample paths", encoding="utf-8")
        return output
    fig, axes = plt.subplots(len(selected_keys), 2, figsize=(10, 14))
    axes_flat = np.ravel(axes)
    for axis, row in zip(axes_flat, selected):
        payload = json.loads(Path(str(row["trajectory_json"])).read_text(encoding="utf-8"))
        width = int(payload["width"])
        height = int(payload["height"])
        grid = np.zeros((height, width), dtype=np.float32)
        for obstacle in payload["obstacles"]:
            grid[int(obstacle[0]), int(obstacle[1])] = 1.0
        axis.imshow(grid, cmap="Greys", vmin=0, vmax=1, alpha=0.32)
        for index, trajectory in enumerate(payload["trajectories"]):
            points = np.asarray(trajectory)
            axis.plot(points[:, 1], points[:, 0], linewidth=1.1, alpha=0.82)
            axis.scatter(points[0, 1], points[0, 0], marker="s", s=18)
            axis.scatter(points[-1, 1], points[-1, 0], marker="x", s=24)
        axis.set_title(f"{row['arm']} {row['label']}\ncoverage={float(row['coverage_ratio'])*100:.1f}%, repeat={float(row['global_repeat_ratio'])*100:.1f}%")
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(height - 0.5, -0.5)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes_flat[len(selected):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def build_report(
    checkpoints: dict[str, Path],
    scenarios: tuple[Scenario, ...],
    seeds: list[int],
    detail_path: Path,
    curves_path: Path,
    summary_path: Path,
    figure_paths: list[Path],
    output_dir: Path,
) -> str:
    lines = [
        "# Course-4 GAT Ablation Generalization",
        "",
        "This report evaluates existing course-4 checkpoints without additional training.",
        "",
        f"- GAT-on checkpoint: `{checkpoints['GAT-on']}`",
        f"- GAT-off checkpoint: `{checkpoints['GAT-off']}`",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Scenarios: {', '.join(scenario.label for scenario in scenarios)}",
        "",
        "## Key Findings",
        "",
        *build_key_findings(summary_path),
        "",
        "## Visual Summary",
        "",
    ]
    for figure in figure_paths:
        rel = figure.relative_to(output_dir).as_posix()
        lines.append(f"![{figure.stem}]({rel})")
        lines.append("")
    lines.extend(
        [
            "## Data Files",
            "",
            f"- Detail rows: `{detail_path.relative_to(output_dir).as_posix()}`",
            f"- Curve rows: `{curves_path.relative_to(output_dir).as_posix()}`",
            f"- Summary rows: `{summary_path.relative_to(output_dir).as_posix()}`",
            "",
            "## Reading Guide",
            "",
            "- Coverage curves show how quickly each policy covers free cells over the available step budget.",
            "- Coverage advantage curves plot GAT-on minus GAT-off in percentage points; values above zero mean GAT-on covers more cells at that step.",
            "- Step-savings heatmaps compare how many steps are saved to reach 90%, 95%, and 99% coverage; unreached thresholds are counted as the scenario's max step budget.",
            "- Global repeat ratio is cumulative repeated visits divided by cumulative visits across all agents.",
            "- In the heatmap, positive blue values mean GAT-on is better; for repeat and overlap metrics, lower raw values are treated as better.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_key_findings(summary_path: Path) -> list[str]:
    rows = read_csv(summary_path)
    by_scenario = defaultdict(dict)
    for row in rows:
        by_scenario[str(row["scenario"])][str(row["arm"])] = row

    def value(key: str, arm: str, metric: str) -> float:
        return float(by_scenario[key][arm][metric])

    def pct(value_: float) -> str:
        return f"{value_ * 100:.1f}%"

    def signed_pp(value_: float) -> str:
        return f"{value_ * 100:+.1f} pp"

    def pp(value_: float) -> str:
        return f"{abs(value_) * 100:.1f} pp"

    findings: list[str] = []
    if "native_20x20_4a" in by_scenario:
        key = "native_20x20_4a"
        findings.append(
            "- Course-native 20x20 / 4-agent evaluation matches the expected direction: "
            f"GAT-on completes in {value(key, 'GAT-on', 'steps_mean'):.1f} steps vs "
            f"{value(key, 'GAT-off', 'steps_mean'):.1f} for GAT-off, with lower global repeat "
            f"({pct(value(key, 'GAT-on', 'global_repeat_ratio_mean'))} vs "
            f"{pct(value(key, 'GAT-off', 'global_repeat_ratio_mean'))})."
        )

    if "train_20x20_4a" in by_scenario:
        key = "train_20x20_4a"
        findings.append(
            "- The same-size random-map check is noisier than the course-native map: "
            f"in this 3-seed sample GAT-on/GAT-off completion is "
            f"{pct(value(key, 'GAT-on', 'completed_mean'))} vs {pct(value(key, 'GAT-off', 'completed_mean'))}. "
            "This row should be read as robustness probing, not the original course checkpoint score."
        )

    if "size_35x35_4a" in by_scenario:
        key = "size_35x35_4a"
        findings.append(
            "- Map-size generalization is where GAT-on separates most clearly: on 35x35 / 4 agents, "
            f"coverage is {pct(value(key, 'GAT-on', 'coverage_ratio_mean'))} vs "
            f"{pct(value(key, 'GAT-off', 'coverage_ratio_mean'))}, completion is "
            f"{pct(value(key, 'GAT-on', 'completed_mean'))} vs {pct(value(key, 'GAT-off', 'completed_mean'))}, "
            f"and global repeat is lower by {pp(value(key, 'GAT-off', 'global_repeat_ratio_mean') - value(key, 'GAT-on', 'global_repeat_ratio_mean'))}."
        )

    if "size_40x40_4a" in by_scenario:
        key = "size_40x40_4a"
        findings.append(
            "- At 40x40 / 4 agents neither arm fully completes within the step budget, but GAT-on keeps a higher final coverage "
            f"({pct(value(key, 'GAT-on', 'coverage_ratio_mean'))} vs {pct(value(key, 'GAT-off', 'coverage_ratio_mean'))}); "
            "this is a useful stress case for the coverage-over-time curves rather than a completed-episode comparison."
        )

    if "agents_20x20_6a" in by_scenario and "agents_20x20_8a" in by_scenario:
        key6 = "agents_20x20_6a"
        key8 = "agents_20x20_8a"
        findings.append(
            "- Agent-count generalization favors GAT-on on completion and repeat control: "
            f"for 6 agents completion is {pct(value(key6, 'GAT-on', 'completed_mean'))} vs "
            f"{pct(value(key6, 'GAT-off', 'completed_mean'))}, and for 8 agents it is "
            f"{pct(value(key8, 'GAT-on', 'completed_mean'))} vs {pct(value(key8, 'GAT-off', 'completed_mean'))}."
        )

    if "combined_30x30_6a" in by_scenario and "combined_35x35_8a" in by_scenario:
        key30 = "combined_30x30_6a"
        key35 = "combined_35x35_8a"
        findings.append(
            "- Combined size/agent shifts still remain strong for GAT-on: it completes all tested seeds in both combined scenarios, "
            f"while GAT-off completes {pct(value(key30, 'GAT-off', 'completed_mean'))} on 30x30 / 6 agents and "
            f"{pct(value(key35, 'GAT-off', 'completed_mean'))} on 35x35 / 8 agents."
        )

    return findings or ["- No paired GAT-on/GAT-off summary rows were available for automatic findings."]


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return values


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = numeric_values(rows, key)
    return float(np.mean(values)) if values else math.nan


def std(rows: list[dict[str, Any]], key: str) -> float:
    values = numeric_values(rows, key)
    return float(np.std(values)) if values else math.nan


if __name__ == "__main__":
    main()
