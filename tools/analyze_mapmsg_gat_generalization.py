"""Offline comparison report for the trained map-message GAT ablation.

The regular evaluation helpers intentionally replay the checkpoint's saved
course_config.json. For zero-shot map-size and agent-count tests we need to
reuse only the learned weights while replacing the environment shape.
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


GAT_ON_CHECKPOINT = Path(r"E:\test plot\ablation_mapmsg_gat_on\20260526-113831\04-tier-4-20x20-4agents\best_policy.pt")
GAT_OFF_CHECKPOINT = Path(r"E:\test plot\ablation_mapmsg_gat_off\20260527-210103\04-tier-4-20x20-4agents\best_policy.pt")


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    width: int
    height: int
    num_agents: int
    max_steps: int
    obstacle_ratios: tuple[float, ...]
    seeds: tuple[int, ...]


SCENARIOS = (
    Scenario(
        "reference_20x20_4agents",
        "20x20 / 4 agents",
        20,
        20,
        4,
        500,
        (0.05, 0.10, 0.15, 0.20),
        (20260601, 20260602, 20260603, 20260604, 20260605),
    ),
    Scenario(
        "larger_24x24_4agents",
        "24x24 / 4 agents",
        24,
        24,
        4,
        720,
        (0.05, 0.10, 0.15, 0.20),
        (20260701, 20260702, 20260703),
    ),
    Scenario(
        "larger_28x28_4agents",
        "28x28 / 4 agents",
        28,
        28,
        4,
        980,
        (0.05, 0.10, 0.15),
        (20260801, 20260802, 20260803),
    ),
    Scenario(
        "more_agents_20x20_6agents",
        "20x20 / 6 agents",
        20,
        20,
        6,
        500,
        (0.05, 0.10, 0.15, 0.20),
        (20260901, 20260902, 20260903),
    ),
    Scenario(
        "more_agents_20x20_8agents",
        "20x20 / 8 agents",
        20,
        20,
        8,
        500,
        (0.05, 0.10, 0.15),
        (20261001, 20261002, 20261003),
    ),
    Scenario(
        "combined_24x24_6agents",
        "24x24 / 6 agents",
        24,
        24,
        6,
        720,
        (0.05, 0.10, 0.15),
        (20261101, 20261102, 20261103),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "mapmsg_gat_generalization_2026-05-28"))
    parser.add_argument("--skip-eval", action="store_true", help="reuse existing detail CSV in output-dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "detail_rows.csv"
    summary_path = output_dir / "scenario_summary.csv"
    ratio_summary_path = output_dir / "ratio_summary.csv"

    if args.skip_eval and detail_path.exists():
        rows = read_csv(detail_path)
    else:
        rows = run_all_trials()
        write_csv(detail_path, rows)

    scenario_summary = summarize(rows, ["scenario", "arm"])
    ratio_summary = summarize(rows, ["scenario", "obstacle_ratio", "arm"])
    write_csv(summary_path, scenario_summary)
    write_csv(ratio_summary_path, ratio_summary)

    figures = make_figures(scenario_summary, ratio_summary, figure_dir)
    report_md = output_dir / "mapmsg_gat_generalization_report.md"
    report_html = output_dir / "mapmsg_gat_generalization_report.html"
    report_text = build_report(rows, scenario_summary, ratio_summary, figures, output_dir)
    report_md.write_text(report_text, encoding="utf-8")
    report_html.write_text(markdown_to_simple_html(report_text), encoding="utf-8")

    print(f"detail={detail_path}")
    print(f"summary={summary_path}")
    print(f"ratio_summary={ratio_summary_path}")
    print(f"report_md={report_md}")
    print(f"report_html={report_html}")


def run_all_trials() -> list[dict[str, Any]]:
    checkpoints = {
        "GAT-on": GAT_ON_CHECKPOINT,
        "GAT-off": GAT_OFF_CHECKPOINT,
    }
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for arm, checkpoint in checkpoints.items():
            config_template = scenario_config(load_config(checkpoint.parent / "course_config.json"), scenario)
            model = load_policy_for_shape(checkpoint, config_template)
            for ratio in scenario.obstacle_ratios:
                for seed in scenario.seeds:
                    trial_config = copy.deepcopy(config_template)
                    trial_config.env.seed = seed
                    trial_config.env.random_obstacle_seed = seed
                    trial_config.env.random_obstacle_seeds = []
                    trial_config.env.map_refresh_episodes = 0
                    trial_config.env.obstacle_ratio = ratio
                    row = evaluate_trial(trial_config, model, scenario, arm, seed, ratio)
                    rows.append(row)
                    print(
                        f"{scenario.name} {arm} ratio={ratio:.2f} seed={seed} "
                        f"coverage={row['coverage_ratio']:.3f} auc={row['coverage_auc']:.3f} "
                        f"completed={row['completed']}"
                    )
    return rows


def scenario_config(base_config: ExperimentConfig, scenario: Scenario) -> ExperimentConfig:
    config = copy.deepcopy(base_config)
    config.env.width = scenario.width
    config.env.height = scenario.height
    config.env.max_steps = scenario.max_steps
    config.env.num_agents = scenario.num_agents
    config.env.random_corner_start = True
    config.env.start_positions = []
    config.env.teammate_positions = []
    config.env.obstacles = []
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
        gat_edge_dim=int(payload.get("gat_edge_dim", GridCoverageEnv(config.env).neighbor_feature_dim if config.ppo.gat_use_edge_features else 0)),
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
    arm: str,
    seed: int,
    obstacle_ratio: float,
) -> dict[str, Any]:
    env = GridCoverageEnv(config.env)
    observation = agent_observations(env.reset(seed=seed))
    state = env.global_state()
    trajectories = [[position] for position in env.positions]
    coverage_curve = [env.coverage_ratio()]
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}

    while not done:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        neighbor_mask = torch.as_tensor(env.neighbor_mask(), dtype=torch.bool)
        edge_features = (
            torch.as_tensor(env.neighbor_features(), dtype=torch.float32)
            if model.use_graph_attention and model.gat_edge_dim > 0
            else None
        )
        node_messages = (
            torch.as_tensor(env.node_messages(), dtype=torch.float32)
            if model.node_message_dim > 0
            else None
        )
        action_mask = (
            torch.as_tensor(env.action_mask(), dtype=torch.bool)
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

    metrics = coverage_efficiency_metrics(
        trajectories=trajectories,
        coverage_curve=coverage_curve,
        max_steps=env.config.max_steps,
        budgets=None,
        stall_steps=50,
    )
    row: dict[str, Any] = {
        "scenario": scenario.name,
        "scenario_label": scenario.label,
        "arm": arm,
        "seed": seed,
        "width": scenario.width,
        "height": scenario.height,
        "num_agents": scenario.num_agents,
        "max_steps": scenario.max_steps,
        "obstacle_ratio": obstacle_ratio,
        "obstacles": len(env.obstacles),
        "free_cells": len(env.free_cells),
        "coverage_ratio": float(info.get("coverage_ratio", env.coverage_ratio())),
        "completed": int(info.get("completed", False)),
        "steps": int(info.get("step_count", env.step_count)),
        "path_length": int(env.path_length),
        "total_reward": total_reward,
    }
    row.update(metrics)
    row["metric_budgets"] = json.dumps(metrics["metric_budgets"])
    return row


def agent_observations(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    if observation.ndim == 1:
        return observation.reshape(1, -1)
    return observation


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    summaries: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        summary = {key: value for key, value in zip(keys, group_key)}
        first = group_rows[0]
        for field in ("scenario_label", "width", "height", "num_agents", "max_steps"):
            if field in first and field not in summary:
                summary[field] = first[field]
        summary["episodes"] = len(group_rows)
        for field in numeric_metric_fields(group_rows):
            values = [float(row[field]) for row in group_rows if row.get(field) not in ("", None)]
            if values:
                summary[f"{field}_mean"] = float(np.mean(values))
                summary[f"{field}_min"] = float(np.min(values))
                summary[f"{field}_max"] = float(np.max(values))
        if "path_length_mean" in summary and float(summary.get("num_agents", 0) or 0) > 0:
            num_agents = float(summary["num_agents"])
            summary["avg_agent_path_length_mean"] = float(summary["path_length_mean"]) / num_agents
            summary["avg_agent_path_length_min"] = float(summary["path_length_min"]) / num_agents
            summary["avg_agent_path_length_max"] = float(summary["path_length_max"]) / num_agents
        summaries.append(summary)
    return summaries


def numeric_metric_fields(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "coverage_ratio",
        "completed",
        "coverage_auc",
        "steps",
        "path_length",
        "repeat_ratio",
        "repeat_ratio_after_90",
        "inter_agent_overlap_ratio",
        "stalled",
        "stall_termination_coverage",
        "total_reward",
        "t90",
        "t95",
        "t99",
    ]
    coverage_fields = sorted(key for key in rows[0] if key.startswith("coverage_at_"))
    return preferred + coverage_fields


def make_figures(
    scenario_summary: list[dict[str, Any]],
    ratio_summary: list[dict[str, Any]],
    figure_dir: Path,
) -> list[Path]:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figures = [
        plot_scenario_bars(scenario_summary, figure_dir, "coverage_at_100_mean", "Coverage@100", "fig01_coverage_at_100.png", ratio=True, zoom_floor=0.5),
        plot_scenario_bars(scenario_summary, figure_dir, "coverage_at_200_mean", "Coverage@200", "fig02_coverage_at_200.png", ratio=True, zoom_floor=0.5),
        plot_scenario_bars(scenario_summary, figure_dir, "coverage_at_300_mean", "Coverage@300", "fig03_coverage_at_300.png", ratio=True, zoom_floor=0.5),
        plot_scenario_bars(scenario_summary, figure_dir, "coverage_at_500_mean", "Coverage@500", "fig04_coverage_at_500.png", ratio=True, zoom_floor=0.5),
        plot_scenario_bars(scenario_summary, figure_dir, "coverage_ratio_mean", "最终覆盖率", "fig05_final_coverage.png", ratio=True, zoom_floor=0.5),
        plot_scenario_bars(scenario_summary, figure_dir, "coverage_auc_mean", "Coverage-AUC", "fig06_coverage_auc.png", ratio=True, zoom_floor=0.5),
        plot_scenario_bars(scenario_summary, figure_dir, "completed_mean", "完成率", "fig07_completion_rate.png", ratio=True, zoom_floor=0.3),
        plot_scenario_bars(scenario_summary, figure_dir, "repeat_ratio_after_90_mean", "90% 后重复率", "fig08_repeat_after90.png", ratio=True, zoom_floor=0.4),
        plot_scenario_bars(scenario_summary, figure_dir, "steps_mean", "平均步数", "fig09_steps_mean.png"),
        plot_scenario_bars(scenario_summary, figure_dir, "avg_agent_path_length_mean", "单 agent 平均路径长度", "fig10_avg_agent_path_length.png"),
        plot_ratio_lines(ratio_summary, figure_dir),
        plot_delta_heatmap(scenario_summary, figure_dir),
    ]
    return figures


def plot_scenario_bars(
    summary: list[dict[str, Any]],
    figure_dir: Path,
    field: str,
    title: str,
    filename: str,
    ratio: bool = False,
    zoom_floor: float | None = None,
) -> Path:
    scenarios = scenario_order(summary)
    x = np.arange(len(scenarios))
    width = 0.36
    on_values = [value_for(summary, scenario, "GAT-on", field) for scenario in scenarios]
    off_values = [value_for(summary, scenario, "GAT-off", field) for scenario in scenarios]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(x - width / 2, on_values, width, label="GAT-on", color="#2f6f8f")
    ax.bar(x + width / 2, off_values, width, label="GAT-off", color="#c45a3c")
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(summary, scenario) for scenario in scenarios], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    if ratio:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    if zoom_floor is not None:
        all_values = [value for value in on_values + off_values if value > 0]
        min_value = min(all_values) if all_values else zoom_floor
        floor = min(zoom_floor, max(0.0, min_value - 0.04))
        if min_value >= zoom_floor:
            floor = zoom_floor
        ceiling = min(1.02, max(max(on_values + off_values) + 0.03, floor + 0.08))
        ax.set_ylim(floor, ceiling)
        ax.axhline(floor, color="#666666", linewidth=0.8)
        ax.text(
            0.01,
            0.02,
            f"纵轴从 {floor * 100:.0f}% 放大显示",
            transform=ax.transAxes,
            fontsize=9,
            color="#555555",
            va="bottom",
        )
    else:
        max_value = max(on_values + off_values) if on_values or off_values else 1.0
        ax.set_ylim(0, max_value * 1.12 if max_value > 0 else 1.0)
    for offset, values in ((-width / 2, on_values), (width / 2, off_values)):
        for xpos, value in zip(x + offset, values):
            label = f"{value * 100:.1f}%" if ratio else f"{value:.1f}"
            ax.text(xpos, value, label, ha="center", va="bottom", fontsize=8, rotation=90)
    ax.legend()
    fig.tight_layout()
    path = figure_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_ratio_lines(summary: list[dict[str, Any]], figure_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
    axes = axes.ravel()
    for ax, scenario in zip(axes, scenario_order(summary)):
        rows = [row for row in summary if row["scenario"] == scenario]
        for arm, color, marker in (("GAT-on", "#2f6f8f", "o"), ("GAT-off", "#c45a3c", "s")):
            arm_rows = sorted((row for row in rows if row["arm"] == arm), key=lambda item: float(item["obstacle_ratio"]))
            ax.plot(
                [float(row["obstacle_ratio"]) * 100 for row in arm_rows],
                [float(row["coverage_auc_mean"]) for row in arm_rows],
                marker=marker,
                color=color,
                label=arm,
            )
        ax.set_title(label_for(summary, scenario))
        ax.set_xlabel("障碍比例 (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Coverage-AUC")
    axes[3].set_ylabel("Coverage-AUC")
    axes[0].legend()
    fig.suptitle("不同障碍比例下的 Coverage-AUC")
    fig.tight_layout()
    path = figure_dir / "fig05_auc_by_obstacle_ratio.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_delta_heatmap(summary: list[dict[str, Any]], figure_dir: Path) -> Path:
    scenarios = scenario_order(summary)
    fields = [
        ("coverage_ratio_mean", "Final"),
        ("coverage_auc_mean", "AUC"),
        ("completed_mean", "Completion"),
        ("repeat_ratio_after_90_mean", "RepeatAfter90"),
        ("inter_agent_overlap_ratio_mean", "Overlap"),
    ]
    data = []
    for scenario in scenarios:
        row = []
        for field, _ in fields:
            row.append(value_for(summary, scenario, "GAT-on", field) - value_for(summary, scenario, "GAT-off", field))
        data.append(row)
    array = np.asarray(data)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    vmax = max(float(np.max(np.abs(array))), 0.01)
    image = ax.imshow(array, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(fields)))
    ax.set_xticklabels([label for _, label in fields], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels([label_for(summary, scenario) for scenario in scenarios])
    ax.set_title("GAT-on minus GAT-off")
    for row_idx in range(array.shape[0]):
        for col_idx in range(array.shape[1]):
            ax.text(col_idx, row_idx, f"{array[row_idx, col_idx]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = figure_dir / "fig06_delta_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def scenario_order(rows: list[dict[str, Any]]) -> list[str]:
    configured = [scenario.name for scenario in SCENARIOS]
    present = {row["scenario"] for row in rows}
    return [scenario for scenario in configured if scenario in present]


def value_for(rows: list[dict[str, Any]], scenario: str, arm: str, field: str) -> float:
    for row in rows:
        if row["scenario"] == scenario and row["arm"] == arm:
            return float(row.get(field, 0.0) or 0.0)
    return 0.0


def label_for(rows: list[dict[str, Any]], scenario: str) -> str:
    for row in rows:
        if row["scenario"] == scenario:
            return str(row["scenario_label"])
    return scenario


def build_report(
    detail_rows: list[dict[str, Any]],
    scenario_summary: list[dict[str, Any]],
    ratio_summary: list[dict[str, Any]],
    figures: list[Path],
    output_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# GAT-mapmsg 课程四模型对比与泛化实验报告")
    lines.append("")
    lines.append("生成日期：2026-05-28")
    lines.append("")
    lines.append("## 1. 实验目的")
    lines.append("")
    lines.append(
        "本报告比较已经训练完成的 `mapmsg_gat_on` 与 `mapmsg_gat_off` 课程四模型，并进一步测试它们在更大地图、不同障碍比例、以及更多智能体数量下的零样本泛化表现。"
    )
    lines.append("所有评估均为离线确定性评估，不继续训练模型。")
    lines.append("")
    lines.append("## 2. 模型与评估设置")
    lines.append("")
    lines.append(f"- GAT-on checkpoint: `{GAT_ON_CHECKPOINT}`")
    lines.append(f"- GAT-off checkpoint: `{GAT_OFF_CHECKPOINT}`")
    lines.append("- 训练课程四配置：20x20，4 agents，显式地图记忆，coverage message，已知可行性 action mask。")
    lines.append("- 对比差异：GAT-on 启用 range-masked multi-head GAT；GAT-off 使用相同 map-message 输入但不做 GAT 聚合。")
    lines.append("- 泛化评估：保留 actor 输入半径、地图记忆与通信机制，替换环境尺寸、障碍比例和 agent 数量。")
    lines.append("- 指标：最终覆盖率、Coverage-AUC、完成率、Coverage@H、T90/T95/T99、重复率、90% 覆盖后的重复率、agent 间覆盖重叠率、stall 覆盖率。")
    lines.append("")
    lines.append("## 3. 场景设计")
    lines.append("")
    lines.append("| 场景 | 尺寸 | agents | 最大步数 | 障碍比例 | seeds |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |")
    for scenario in SCENARIOS:
        ratio_text = ", ".join(f"{ratio:.0%}" for ratio in scenario.obstacle_ratios)
        seed_text = f"{scenario.seeds[0]}-{scenario.seeds[-1]} ({len(scenario.seeds)} seeds)"
        lines.append(f"| {scenario.label} | {scenario.width}x{scenario.height} | {scenario.num_agents} | {scenario.max_steps} | {ratio_text} | {seed_text} |")
    lines.append("")
    lines.append("## 4. 总体结果图")
    lines.append("")
    for figure in figures:
        rel = figure.relative_to(output_dir).as_posix()
        lines.append(f"![{figure.stem}]({rel})")
        lines.append("")
    lines.append("## 5. 场景聚合结果")
    lines.append("")
    lines.append("| 场景 | Arm | Episodes | Final Coverage | Coverage-AUC | Completion | Steps | Avg Path/Agent | RepeatAfter90 | InterAgentOverlap | StallCoverage |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for scenario in scenario_order(scenario_summary):
        for arm in ("GAT-on", "GAT-off"):
            row = next(item for item in scenario_summary if item["scenario"] == scenario and item["arm"] == arm)
            lines.append(
                f"| {row['scenario_label']} | {arm} | {int(row['episodes'])} | "
                f"{pct(row['coverage_ratio_mean'])} | {float(row['coverage_auc_mean']):.3f} | "
                f"{pct(row['completed_mean'])} | {float(row['steps_mean']):.1f} | "
                f"{float(row['avg_agent_path_length_mean']):.1f} | {pct(row['repeat_ratio_after_90_mean'])} | "
                f"{pct(row['inter_agent_overlap_ratio_mean'])} | {pct(row['stall_termination_coverage_mean'])} |"
            )
    lines.append("")
    lines.append("## 6. 固定预算覆盖率 Coverage@H")
    lines.append("")
    lines.append(
        "`Coverage@H` 表示在固定环境步数预算 H 内达到的覆盖比例，是本项目判断在线覆盖效率的核心指标之一。下面四个预算点分别对应早期推进、中期推进和课程四训练图上的完整预算参考。"
    )
    lines.append("")
    lines.append("| 场景 | Arm | Coverage@100 | Coverage@200 | Coverage@300 | Coverage@500 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for scenario in scenario_order(scenario_summary):
        for arm in ("GAT-on", "GAT-off"):
            row = next(item for item in scenario_summary if item["scenario"] == scenario and item["arm"] == arm)
            lines.append(
                f"| {row['scenario_label']} | {arm} | "
                f"{pct(row.get('coverage_at_100_mean', 0.0))} | {pct(row.get('coverage_at_200_mean', 0.0))} | "
                f"{pct(row.get('coverage_at_300_mean', 0.0))} | {pct(row.get('coverage_at_500_mean', 0.0))} |"
            )
    lines.append("")
    lines.append("## 7. GAT-on 相对 GAT-off 的差值")
    lines.append("")
    lines.append("| 场景 | Final Δ | AUC Δ | Completion Δ | Steps Δ | Avg Path/Agent Δ | RepeatAfter90 Δ | Overlap Δ | 结论摘要 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for scenario in scenario_order(scenario_summary):
        final_delta = value_for(scenario_summary, scenario, "GAT-on", "coverage_ratio_mean") - value_for(scenario_summary, scenario, "GAT-off", "coverage_ratio_mean")
        auc_delta = value_for(scenario_summary, scenario, "GAT-on", "coverage_auc_mean") - value_for(scenario_summary, scenario, "GAT-off", "coverage_auc_mean")
        completion_delta = value_for(scenario_summary, scenario, "GAT-on", "completed_mean") - value_for(scenario_summary, scenario, "GAT-off", "completed_mean")
        steps_delta = value_for(scenario_summary, scenario, "GAT-on", "steps_mean") - value_for(scenario_summary, scenario, "GAT-off", "steps_mean")
        avg_path_delta = value_for(scenario_summary, scenario, "GAT-on", "avg_agent_path_length_mean") - value_for(scenario_summary, scenario, "GAT-off", "avg_agent_path_length_mean")
        repeat_delta = value_for(scenario_summary, scenario, "GAT-on", "repeat_ratio_after_90_mean") - value_for(scenario_summary, scenario, "GAT-off", "repeat_ratio_after_90_mean")
        overlap_delta = value_for(scenario_summary, scenario, "GAT-on", "inter_agent_overlap_ratio_mean") - value_for(scenario_summary, scenario, "GAT-off", "inter_agent_overlap_ratio_mean")
        conclusion = automatic_conclusion(final_delta, auc_delta, repeat_delta, overlap_delta)
        lines.append(
            f"| {label_for(scenario_summary, scenario)} | {final_delta:+.3f} | {auc_delta:+.3f} | "
            f"{completion_delta:+.3f} | {steps_delta:+.1f} | {avg_path_delta:+.1f} | "
            f"{repeat_delta:+.3f} | {overlap_delta:+.3f} | {conclusion} |"
        )
    lines.append("")
    lines.append("## 8. 障碍比例敏感性")
    lines.append("")
    lines.append(
        "从障碍比例分组结果看，GAT-on 与 GAT-off 的胜负并不只由最终覆盖率决定。若 GAT-on 提高最终覆盖率但同时显著提高 `RepeatAfter90` 或 `InterAgentOverlap`，说明注意力通信可能帮助维持高覆盖推进，但也可能导致多个 agent 在高覆盖阶段聚集或重复搜索。"
    )
    lines.append("")
    lines.append("| 场景 | Ratio | GAT-on AUC | GAT-off AUC | AUC Δ | GAT-on Final | GAT-off Final | Final Δ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for scenario in scenario_order(ratio_summary):
        ratios = sorted({float(row["obstacle_ratio"]) for row in ratio_summary if row["scenario"] == scenario})
        for ratio in ratios:
            on = next(row for row in ratio_summary if row["scenario"] == scenario and row["arm"] == "GAT-on" and float(row["obstacle_ratio"]) == ratio)
            off = next(row for row in ratio_summary if row["scenario"] == scenario and row["arm"] == "GAT-off" and float(row["obstacle_ratio"]) == ratio)
            lines.append(
                f"| {on['scenario_label']} | {ratio:.0%} | {float(on['coverage_auc_mean']):.3f} | "
                f"{float(off['coverage_auc_mean']):.3f} | {float(on['coverage_auc_mean']) - float(off['coverage_auc_mean']):+.3f} | "
                f"{pct(on['coverage_ratio_mean'])} | {pct(off['coverage_ratio_mean'])} | "
                f"{float(on['coverage_ratio_mean']) - float(off['coverage_ratio_mean']):+.3f} |"
            )
    lines.append("")
    lines.append("## 9. 主要结论")
    lines.append("")
    lines.extend(conclusion_bullets(scenario_summary))
    lines.append("")
    lines.append("## 10. 局限性")
    lines.append("")
    lines.append("- 泛化评估没有重新训练，只检验课程四权重的零样本迁移行为。")
    lines.append("- 更多 agent 的测试沿用了同一个共享 actor 和同一套 GAT/message 机制，未对更大队伍重新调参。")
    lines.append("- `max_steps` 按地图面积放大，但不同尺寸间仍应优先比较 Coverage-AUC、Coverage@H 和相对趋势，而不是只看最终是否 100% 完成。")
    lines.append("- 测试 seeds 数量有限，适合作为阶段性实验报告；正式论文表格建议再扩展 seed 数并报告置信区间。")
    lines.append("")
    lines.append("## 11. 产物路径")
    lines.append("")
    lines.append(f"- 明细结果：`{output_dir / 'detail_rows.csv'}`")
    lines.append(f"- 场景聚合：`{output_dir / 'scenario_summary.csv'}`")
    lines.append(f"- 障碍比例聚合：`{output_dir / 'ratio_summary.csv'}`")
    return "\n".join(lines) + "\n"


def automatic_conclusion(final_delta: float, auc_delta: float, repeat_delta: float, overlap_delta: float) -> str:
    if auc_delta > 0.01 and final_delta >= -0.01:
        return "GAT-on 覆盖效率占优"
    if auc_delta < -0.01 and final_delta <= 0.01:
        return "GAT-off 覆盖效率占优"
    if final_delta > 0.02 and repeat_delta > 0.02:
        return "GAT-on 覆盖更高但重复代价更大"
    if abs(auc_delta) <= 0.01 and abs(final_delta) <= 0.01:
        return "两者接近，需看重复与重叠"
    if repeat_delta > 0.05 or overlap_delta > 0.05:
        return "GAT-on 协作冗余偏高"
    return "差异较小"


def conclusion_bullets(summary: list[dict[str, Any]]) -> list[str]:
    scenarios = scenario_order(summary)
    auc_wins = sum(value_for(summary, scenario, "GAT-on", "coverage_auc_mean") > value_for(summary, scenario, "GAT-off", "coverage_auc_mean") for scenario in scenarios)
    final_wins = sum(value_for(summary, scenario, "GAT-on", "coverage_ratio_mean") > value_for(summary, scenario, "GAT-off", "coverage_ratio_mean") for scenario in scenarios)
    completion_wins = sum(value_for(summary, scenario, "GAT-on", "completed_mean") > value_for(summary, scenario, "GAT-off", "completed_mean") for scenario in scenarios)
    repeat_lower = sum(value_for(summary, scenario, "GAT-on", "repeat_ratio_after_90_mean") < value_for(summary, scenario, "GAT-off", "repeat_ratio_after_90_mean") for scenario in scenarios)
    lines = [
        f"- 在 {len(scenarios)} 个聚合场景中，GAT-on 的 Coverage-AUC 高于 GAT-off 的场景数为 {auc_wins}，最终覆盖率更高的场景数为 {final_wins}，完成率更高的场景数为 {completion_wins}。",
        f"- GAT-on 的 `RepeatAfter90` 低于 GAT-off 的场景数为 {repeat_lower}；如果该数偏低，说明 GAT 通信虽然可能提高早期或最终覆盖，但高覆盖阶段的重复搜索仍需改进。",
        "- 当前结果应被理解为 map-message 基础上的 GAT 消融，而不是完整通信机制的终局结论。后续可以继续比较无通信、地图共享无注意力、任务语义 GAT、以及更强地图融合策略。",
    ]
    return lines


def pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%"


def markdown_to_simple_html(markdown: str) -> str:
    body = []
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("!["):
            alt, rest = line[2:].split("](", 1)
            src = rest.rstrip(")")
            body.append(f'<img src="{src}" alt="{alt}">')
        elif line.startswith("| "):
            if set(line.replace("|", "").replace(" ", "").replace("-", "").replace(":", "")) == set():
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not in_table:
                body.append("<table>")
                in_table = True
            tag = "th" if not body[-1].startswith("<tr>") and cells[0] in {"场景", "指标", "Arm"} else "td"
            body.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
        else:
            if in_table:
                body.append("</table>")
                in_table = False
            if line.startswith("- "):
                body.append(f"<p>{line}</p>")
            elif line:
                body.append(f"<p>{line}</p>")
    if in_table:
        body.append("</table>")
    style = """
body { font-family: "Microsoft YaHei", Arial, sans-serif; max-width: 1180px; margin: 32px auto; line-height: 1.55; color: #222; }
h1, h2 { margin-top: 28px; }
img { max-width: 100%; margin: 12px 0 24px; border: 1px solid #ddd; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 22px; font-size: 14px; }
th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }
th { background: #eef3f6; }
code { background: #f5f5f5; padding: 1px 4px; }
"""
    return f"<!doctype html><meta charset=\"utf-8\"><style>{style}</style>" + "\n".join(body)


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
