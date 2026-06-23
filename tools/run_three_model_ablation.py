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
from mathbased_mcpp.cuap import build_cuap_step_inputs, scaled_cuap_prior
from mathbased_mcpp.env import GridCoverageEnv
from mathbased_mcpp.evaluation import coverage_efficiency_metrics
from mathbased_mcpp.ppo import ActorCritic
from mathbased_mcpp.utils import agent_observations


DEFAULT_OFF_COVERAGE = ROOT / "outputs" / "ablation_mapmsg_gat_off_nocomm" / "depot_return_pipeline" / "coverage" / "04-tier-4-20x20-4agents" / "best_policy.pt"
DEFAULT_ON_COVERAGE = ROOT / "outputs" / "ablation_mapmsg_gat_on" / "depot_return_pipeline" / "coverage" / "04-tier-4-20x20-4agents" / "best_policy.pt"
DEFAULT_CUAP_COVERAGE = ROOT / "outputs" / "ablation_mapmsg_gat_on_cuap" / "depot_return_pipeline" / "coverage" / "04-tier-4-20x20-4agents" / "best_policy.pt"
DEFAULT_RETURN = ROOT / "outputs" / "ablation_mapmsg_gat_off_nocomm" / "depot_return_pipeline" / "return_diverse_scale60" / "04-tier-4-20x20-4agents" / "policy.pt"

ARM_ORDER = ("GAT-OFF", "GAT-ON", "GAT-CUAP")
COLORS = {
    "GAT-OFF": "#cc5a2d",
    "GAT-ON": "#1f77b4",
    "GAT-CUAP": "#2ca02c",
}


@dataclass(frozen=True)
class Arm:
    name: str
    coverage_checkpoint: Path
    return_checkpoint: Path


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    width: int
    height: int
    agents: int
    max_steps: int
    obstacle_ratio: float
    seeds: tuple[int, ...]
    category: str
    course_native: bool = False


DEFAULT_SCENARIOS = (
    Scenario(
        key="native_20x20_4a",
        label="Course-4 native 20x20 / 4 agents",
        width=20,
        height=20,
        agents=4,
        max_steps=500,
        obstacle_ratio=0.05,
        seeds=(),
        category="course-native",
        course_native=True,
    ),
    Scenario(
        key="unseen_20x20_4a_r05",
        label="Unseen 20x20 / 4 agents / 5% obstacles",
        width=20,
        height=20,
        agents=4,
        max_steps=500,
        obstacle_ratio=0.05,
        seeds=(20260601, 20260602, 20260603, 20260604, 20260605),
        category="same-size",
    ),
    Scenario(
        key="stress_20x20_4a_r10",
        label="Obstacle stress 20x20 / 4 agents / 10%",
        width=20,
        height=20,
        agents=4,
        max_steps=500,
        obstacle_ratio=0.10,
        seeds=(20260611, 20260612, 20260613),
        category="obstacle-stress",
    ),
    Scenario(
        key="stress_20x20_4a_r15",
        label="Obstacle stress 20x20 / 4 agents / 15%",
        width=20,
        height=20,
        agents=4,
        max_steps=500,
        obstacle_ratio=0.15,
        seeds=(20260621, 20260622, 20260623),
        category="obstacle-stress",
    ),
    Scenario(
        key="stress_20x20_4a_r20",
        label="Obstacle stress 20x20 / 4 agents / 20%",
        width=20,
        height=20,
        agents=4,
        max_steps=500,
        obstacle_ratio=0.20,
        seeds=(20260631, 20260632, 20260633),
        category="obstacle-stress",
    ),
    Scenario(
        key="transfer_30x30_4a_r05",
        label="Transfer 30x30 / 4 agents / 5%",
        width=30,
        height=30,
        agents=4,
        max_steps=1125,
        obstacle_ratio=0.05,
        seeds=(20260701, 20260702, 20260703),
        category="map-size",
    ),
    Scenario(
        key="agents_20x20_6a_r05",
        label="Transfer 20x20 / 6 agents / 5%",
        width=20,
        height=20,
        agents=6,
        max_steps=500,
        obstacle_ratio=0.05,
        seeds=(20260801, 20260802, 20260803),
        category="agent-count",
    ),
    Scenario(
        key="combined_30x30_6a_r05",
        label="Transfer 30x30 / 6 agents / 5%",
        width=30,
        height=30,
        agents=6,
        max_steps=1125,
        obstacle_ratio=0.05,
        seeds=(20260901, 20260902, 20260903),
        category="combined",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline three-arm ablation for GAT-OFF, GAT-ON, and GAT-CUAP.")
    parser.add_argument("--gat-off-coverage", default=str(DEFAULT_OFF_COVERAGE))
    parser.add_argument("--gat-on-coverage", default=str(DEFAULT_ON_COVERAGE))
    parser.add_argument("--gat-cuap-coverage", default=str(DEFAULT_CUAP_COVERAGE))
    parser.add_argument("--return-checkpoint", default=str(DEFAULT_RETURN))
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / f"three_model_ablation_{date.today().isoformat()}"))
    parser.add_argument("--scenario-keys", default="", help="Comma-separated subset of scenario keys.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--coverage-only", action="store_true", help="evaluate only coverage policies; do not enter depot-return mode")
    args = parser.parse_args()

    return_checkpoint = resolve_path(args.return_checkpoint)
    arms = (
        Arm("GAT-OFF", resolve_path(args.gat_off_coverage), return_checkpoint),
        Arm("GAT-ON", resolve_path(args.gat_on_coverage), return_checkpoint),
        Arm("GAT-CUAP", resolve_path(args.gat_cuap_coverage), return_checkpoint),
    )
    for arm in arms:
        if not arm.coverage_checkpoint.exists():
            raise FileNotFoundError(f"{arm.name} coverage checkpoint not found: {arm.coverage_checkpoint}")
        if not arm.return_checkpoint.exists():
            raise FileNotFoundError(f"{arm.name} return checkpoint not found: {arm.return_checkpoint}")

    scenarios = select_scenarios(args.scenario_keys)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    trajectories_dir = output_dir / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "detail_rows.csv"
    curves_path = output_dir / "curve_rows.csv"
    summary_path = output_dir / "summary_rows.csv"

    if args.skip_existing and detail_path.exists() and curves_path.exists():
        detail_rows = read_csv(detail_path)
        curve_rows = read_csv(curves_path)
    else:
        detail_rows, curve_rows = run_experiment(arms, scenarios, trajectories_dir, coverage_only=args.coverage_only)
        write_csv(detail_path, detail_rows)
        write_csv(curves_path, curve_rows)

    summary_rows = summarize(detail_rows, scenarios)
    write_csv(summary_path, summary_rows)

    figure_paths = make_figures(detail_rows, curve_rows, summary_rows, scenarios, figures_dir)
    report_path = output_dir / "three_model_ablation_report.md"
    report_path.write_text(
        build_report(arms, scenarios, detail_path, curves_path, summary_path, figure_paths, output_dir),
        encoding="utf-8",
    )

    print(f"detail={detail_path}")
    print(f"curves={curves_path}")
    print(f"summary={summary_path}")
    print(f"report={report_path}")
    print(f"figures={figures_dir}")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


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
    arms: tuple[Arm, ...],
    scenarios: tuple[Scenario, ...],
    trajectories_dir: Path,
    coverage_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for arm in arms:
        base_config = load_config(arm.coverage_checkpoint.parent / "course_config.json")
        return_base_config = load_config(arm.return_checkpoint.parent / "course_config.json")
        model_cache: dict[str, tuple[ActorCritic, ActorCritic]] = {}
        for scenario in scenarios:
            config = build_scenario_config(base_config, scenario, coverage_only=coverage_only)
            return_config = build_scenario_config(return_base_config, scenario, coverage_only=coverage_only)
            model_pair = model_cache.setdefault(
                scenario.key,
                (
                    load_policy_for_shape(arm.coverage_checkpoint, config),
                    load_policy_for_shape(arm.return_checkpoint, return_config),
                ),
            )
            seeds = (config.env.seed,) if scenario.course_native else scenario.seeds
            for seed in seeds:
                detail, curves = evaluate_trial(
                    arm=arm,
                    config=config,
                    coverage_model=model_pair[0],
                    return_model=model_pair[1],
                    scenario=scenario,
                    seed=seed,
                    trajectories_dir=trajectories_dir,
                    coverage_only=coverage_only,
                )
                detail_rows.append(detail)
                curve_rows.extend(curves)
                print(
                    f"{arm.name} {scenario.key} seed={seed} "
                    f"coverage={detail['coverage_ratio']:.4f} "
                    f"coverage_done={detail['coverage_completed']} "
                    f"mission_done={detail['mission_completed']} "
                    f"auc={detail['coverage_auc']:.4f} "
                    f"repeat90={detail['repeat_ratio_after_90']:.4f} "
                    f"steps={detail['steps']}"
                )
    return detail_rows, curve_rows


def build_scenario_config(base_config: ExperimentConfig, scenario: Scenario, coverage_only: bool = False) -> ExperimentConfig:
    config = copy.deepcopy(base_config)
    config.env.use_depot = True
    config.env.require_return_to_depot = not coverage_only
    config.env.initial_return_mode = False
    if scenario.course_native:
        return config
    config.env.width = scenario.width
    config.env.height = scenario.height
    config.env.max_steps = scenario.max_steps
    config.env.num_agents = scenario.agents
    config.env.random_corner_start = True
    config.env.start_positions = []
    config.env.teammate_positions = []
    config.env.return_start_positions = []
    config.env.obstacles = []
    config.env.obstacle_ratio = scenario.obstacle_ratio
    config.env.random_obstacle_count = 0
    config.env.random_obstacle_seeds = []
    config.env.map_refresh_episodes = 0
    return config


def load_policy_for_shape(checkpoint_path: Path, config: ExperimentConfig) -> ActorCritic:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gat_use_edge_features = bool(payload.get("gat_use_edge_features", config.ppo.gat_use_edge_features))
    cuap_meta = payload.get("cuap", {})
    if not isinstance(cuap_meta, dict):
        cuap_meta = {}
    intent_meta = payload.get("intent_relation", {})
    if not isinstance(intent_meta, dict):
        intent_meta = {}
    model = ActorCritic(
        observation_dim=int(payload["observation_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_dim=int(payload.get("hidden_dim", config.ppo.hidden_dim)),
        state_shape=(config.env.height, config.env.width),
        state_channels=int(payload.get("state_channels", 5)),
        state_metadata_dim=int(payload.get("state_metadata_dim", 7)),
        use_graph_attention=bool(payload.get("use_graph_attention", False)),
        gat_num_heads=int(payload.get("gat_num_heads", config.ppo.gat_num_heads)),
        gat_edge_dim=int(
            payload.get(
                "gat_edge_dim",
                GridCoverageEnv(config.env).neighbor_feature_dim if gat_use_edge_features else 0,
            )
        ),
        gat_residual=bool(payload.get("gat_residual", config.ppo.gat_residual)),
        gat_attention_dropout=float(payload.get("gat_attention_dropout", config.ppo.gat_attention_dropout)),
        node_message_dim=int(payload.get("node_message_dim", 0)),
        use_phase_critics=bool(payload.get("use_phase_critics", False)),
        use_phase_actors=bool(payload.get("use_phase_actors", False)),
        phase_metadata_index=int(payload.get("phase_metadata_index", GridCoverageEnv.base_state_metadata_dim)),
        use_gated_cuap=bool(cuap_meta.get("gated", False)),
        cuap_beta=float(cuap_meta.get("beta", getattr(config.cuap, "beta", 0.0))),
        cuap_gate_hidden_dim=int(cuap_meta.get("gate_hidden_dim", getattr(config.cuap, "gate_hidden_dim", 32))),
        cuap_gate_init_prob=float(cuap_meta.get("gate_init_prob", getattr(config.cuap, "gate_init_prob", 0.1))),
        cuap_gate_detach_actor_features=bool(
            cuap_meta.get("gate_detach_actor_features", getattr(config.cuap, "gate_detach_actor_features", True))
        ),
        use_intent_relation=bool(intent_meta.get("enabled", getattr(config.ppo, "use_intent_relation", False))),
        intent_relation_beta_max=float(
            intent_meta.get("beta_max", getattr(config.ppo, "intent_relation_beta_max", 2.0))
        ),
        intent_relation_detach=bool(intent_meta.get("detach", getattr(config.ppo, "intent_relation_detach", True))),
        intent_grid_size=int(intent_meta.get("intent_grid_size", getattr(config.env, "intent_grid_size", 3))),
    )
    model.load_compatible_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def evaluate_trial(
    arm: Arm,
    config: ExperimentConfig,
    coverage_model: ActorCritic,
    return_model: ActorCritic,
    scenario: Scenario,
    seed: int,
    trajectories_dir: Path,
    coverage_only: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trial_config = copy.deepcopy(config)
    if not scenario.course_native:
        trial_config.env.seed = int(seed)
        trial_config.env.random_obstacle_seed = int(seed)
    env = GridCoverageEnv(trial_config.env)
    observation = agent_observations(env.reset(seed=int(seed)))
    state = env.global_state()
    trajectories = [[position] for position in env.positions]
    coverage_curve = [env.coverage_ratio()]
    phase_by_step = ["coverage"]
    phase_steps = {"coverage": 0, "return": 0}
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}

    while not done:
        phase = "coverage" if coverage_only else ("return" if env.return_mode else "coverage")
        model = coverage_model if coverage_only else (return_model if env.return_mode else coverage_model)
        phase_steps[phase] += 1
        device = next(model.parameters()).device
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
        action_prior = action_prior_tensor(trial_config, env, phase, device)
        cuap_prior = cuap_confidence = cuap_phase_mask = None
        if model.use_gated_cuap:
            cuap_inputs = build_cuap_step_inputs(env, trial_config.cuap, phase=phase)
            cuap_prior = torch.as_tensor(cuap_inputs.prior, dtype=torch.float32, device=device)
            cuap_confidence = torch.as_tensor(cuap_inputs.confidence, dtype=torch.float32, device=device)
            cuap_phase_mask = torch.as_tensor(cuap_inputs.phase_mask, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, _, _ = model.act_batch(
                obs_tensor,
                state_tensor,
                neighbor_mask=neighbor_mask,
                edge_features=edge_features,
                node_messages=node_messages,
                action_mask=action_mask,
                action_prior_logits=action_prior,
                cuap_prior=cuap_prior,
                cuap_confidence=cuap_confidence,
                cuap_phase_mask=cuap_phase_mask,
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
        phase_by_step.append("return" if env.return_mode else "coverage")

    budgets = metric_budgets(scenario.max_steps)
    metrics = coverage_efficiency_metrics(
        trajectories=trajectories,
        coverage_curve=coverage_curve,
        max_steps=env.config.max_steps,
        budgets=budgets,
        stall_steps=max(50, scenario.max_steps // 20),
    )
    global_repeat = cumulative_repeat_curve(trajectories, scenario.max_steps)[-1]
    trajectory_path = trajectories_dir / f"{arm.name}_{scenario.key}_seed_{seed}.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "arm": arm.name,
                "scenario": scenario.key,
                "label": scenario.label,
                "seed": int(seed),
                "width": env.config.width,
                "height": env.config.height,
                "num_agents": env.num_agents,
                "obstacle_ratio": env.config.obstacle_ratio,
                "obstacles": [[int(row), int(col)] for row, col in sorted(env.obstacles)],
                "coverage_curve": coverage_curve,
                "repeat_curve": cumulative_repeat_curve(trajectories, scenario.max_steps),
                "phase_by_step": phase_by_step,
                "trajectories": [[[int(row), int(col)] for row, col in path] for path in trajectories],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    coverage_completed = bool(info.get("coverage_completed", False))
    mission_completed = bool(info.get("completed", False))
    returned_to_depot = bool(info.get("all_at_depot", False))
    detail: dict[str, Any] = {
        "arm": arm.name,
        "scenario": scenario.key,
        "label": scenario.label,
        "category": scenario.category,
        "seed": int(seed),
        "width": env.config.width,
        "height": env.config.height,
        "num_agents": env.num_agents,
        "max_steps": env.config.max_steps,
        "obstacle_ratio": "" if env.config.obstacle_ratio is None else float(env.config.obstacle_ratio),
        "obstacles": len(env.obstacles),
        "free_cells": len(env.free_cells),
        "coverage_ratio": float(info.get("coverage_ratio", env.coverage_ratio())),
        "coverage_completed": int(coverage_completed),
        "mission_completed": int(mission_completed),
        "returned_to_depot": int(returned_to_depot and mission_completed),
        "steps": int(info.get("step_count", env.step_count)),
        "coverage_steps": int(phase_steps["coverage"]),
        "return_steps": int(phase_steps["return"]),
        "path_length": int(env.path_length),
        "avg_agent_path_length": float(np.mean(env.path_lengths)),
        "global_repeat_ratio": float(global_repeat),
        "total_reward": total_reward,
        "trajectory_json": str(trajectory_path),
    }
    detail.update(metrics)
    detail["metric_budgets"] = json.dumps(metrics["metric_budgets"])

    coverage_padded = pad_curve(coverage_curve, scenario.max_steps)
    repeat_padded = cumulative_repeat_curve(trajectories, scenario.max_steps)
    phase_padded = pad_phase(phase_by_step, scenario.max_steps)
    curve_rows = [
        {
            "arm": arm.name,
            "scenario": scenario.key,
            "label": scenario.label,
            "category": scenario.category,
            "seed": int(seed),
            "step": step,
            "coverage": coverage_padded[step],
            "global_repeat_ratio": repeat_padded[step],
            "phase": phase_padded[step],
        }
        for step in range(scenario.max_steps + 1)
    ]
    return detail, curve_rows


def action_prior_tensor(config: ExperimentConfig, env: GridCoverageEnv, phase: str, device: torch.device) -> torch.Tensor | None:
    prior = scaled_cuap_prior(env, config.cuap, phase=phase)
    if prior is None:
        return None
    return torch.as_tensor(prior, dtype=torch.float32, device=device)


def metric_budgets(max_steps: int) -> list[int]:
    fixed = [100, 200, 300, 500, 750, 1000, 1500, 2000]
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
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


def pad_phase(phases: list[str], max_steps: int) -> list[str]:
    output = list(phases[: max_steps + 1])
    if not output:
        output = ["coverage"]
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

    metrics = [
        "coverage_ratio",
        "coverage_auc",
        "coverage_completed",
        "mission_completed",
        "returned_to_depot",
        "steps",
        "coverage_steps",
        "return_steps",
        "path_length",
        "avg_agent_path_length",
        "global_repeat_ratio",
        "repeat_ratio",
        "repeat_ratio_after_90",
        "inter_agent_overlap_ratio",
        "stalled",
        "stall_termination_coverage",
        "total_reward",
    ]
    output: list[dict[str, Any]] = []
    for scenario in scenarios:
        for arm in ARM_ORDER:
            rows = groups.get((scenario.key, arm), [])
            summary: dict[str, Any] = {
                "arm": arm,
                "scenario": scenario.key,
                "label": scenario.label,
                "category": scenario.category,
                "width": scenario.width,
                "height": scenario.height,
                "num_agents": scenario.agents,
                "max_steps": scenario.max_steps,
                "obstacle_ratio": scenario.obstacle_ratio,
                "episodes": len(rows),
            }
            for metric in metrics:
                summary[f"{metric}_mean"] = mean(rows, metric)
                summary[f"{metric}_std"] = std(rows, metric)
            for threshold in ("t90", "t95", "t99"):
                values = numeric_values(rows, threshold)
                summary[f"{threshold}_reach_rate"] = len(values) / max(len(rows), 1)
                summary[f"{threshold}_mean_reached"] = float(np.mean(values)) if values else math.nan
            for key in sorted(k for k in rows[0] if k.startswith("coverage_at_")) if rows else []:
                summary[f"{key}_mean"] = mean(rows, key)
            output.append(summary)
    return output


def make_figures(
    detail_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    scenarios: tuple[Scenario, ...],
    figures_dir: Path,
) -> list[Path]:
    return [
        plot_curve_grid(curve_rows, scenarios, figures_dir / "fig01_coverage_curves.png", "coverage", "Coverage"),
        plot_curve_grid(curve_rows, scenarios, figures_dir / "fig02_global_repeat_curves.png", "global_repeat_ratio", "Global repeat ratio"),
        plot_grouped_bars(
            summary_rows,
            scenarios,
            figures_dir / "fig03_final_coverage_auc.png",
            ["coverage_ratio_mean", "coverage_auc_mean"],
            ["Final coverage", "Coverage-AUC"],
        ),
        plot_grouped_bars(
            summary_rows,
            scenarios,
            figures_dir / "fig04_completion_return.png",
            ["coverage_completed_mean", "mission_completed_mean", "returned_to_depot_mean"],
            ["Coverage completed", "Mission completed", "Returned to depot"],
        ),
        plot_grouped_bars(
            summary_rows,
            scenarios,
            figures_dir / "fig05_repeat_overlap.png",
            ["global_repeat_ratio_mean", "repeat_ratio_after_90_mean", "inter_agent_overlap_ratio_mean"],
            ["Global repeat", "Repeat after 90%", "Inter-agent overlap"],
        ),
        plot_grouped_bars(
            summary_rows,
            scenarios,
            figures_dir / "fig06_phase_steps.png",
            ["coverage_steps_mean", "return_steps_mean"],
            ["Coverage steps", "Return steps"],
            ratio_flags=[False, False],
        ),
        plot_advantage_heatmap(summary_rows, scenarios, figures_dir / "fig07_advantage_heatmap.png"),
        plot_sample_paths(detail_rows, figures_dir / "fig08_sample_paths.png"),
    ]


def plot_curve_grid(
    curve_rows: list[dict[str, Any]],
    scenarios: tuple[Scenario, ...],
    output: Path,
    metric: str,
    ylabel: str,
) -> Path:
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        rows_by_key[(str(row["scenario"]), str(row["arm"]))].append(row)
    cols = 2
    rows_count = int(math.ceil(len(scenarios) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(7.2 * cols, 3.4 * rows_count), sharey=True)
    axes_flat = np.ravel(axes)
    for axis, scenario in zip(axes_flat, scenarios):
        for arm in ARM_ORDER:
            arm_rows = rows_by_key.get((scenario.key, arm), [])
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in arm_rows:
                grouped[int(row["step"])].append(float(row[metric]))
            if not grouped:
                continue
            steps = sorted(grouped)
            means = np.asarray([np.mean(grouped[step]) for step in steps])
            stds = np.asarray([np.std(grouped[step]) for step in steps])
            axis.plot(steps, means, color=COLORS[arm], label=arm, linewidth=1.9)
            axis.fill_between(steps, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1), color=COLORS[arm], alpha=0.10)
        axis.set_title(scenario.label, fontsize=9.5)
        axis.set_xlim(0, scenario.max_steps)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    for axis in axes_flat[len(scenarios):]:
        axis.axis("off")
    for axis in axes_flat[::cols]:
        axis.set_ylabel(ylabel)
    for axis in axes_flat[-cols:]:
        axis.set_xlabel("Steps")
    axes_flat[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


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
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.6 * len(metrics), 5.2), squeeze=False)
    axes_flat = axes.ravel()
    x = np.arange(len(scenarios))
    width = 0.24
    offsets = {"GAT-OFF": -width, "GAT-ON": 0.0, "GAT-CUAP": width}
    for axis, metric, title, ratio in zip(axes_flat, metrics, titles, ratio_flags):
        for arm in ARM_ORDER:
            values = [float(lookup[(scenario.key, arm)].get(metric, math.nan)) for scenario in scenarios]
            axis.bar(x + offsets[arm], values, width, label=arm, color=COLORS[arm])
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels([scenario.label for scenario in scenarios], rotation=35, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        if ratio:
            axis.set_ylim(0, 1.05)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_advantage_heatmap(summary_rows: list[dict[str, Any]], scenarios: tuple[Scenario, ...], output: Path) -> Path:
    lookup = {(str(row["scenario"]), str(row["arm"])): row for row in summary_rows}
    metric_specs = [
        ("coverage_auc_mean", "AUC", "higher"),
        ("coverage_ratio_mean", "Final", "higher"),
        ("mission_completed_mean", "Mission", "higher"),
        ("coverage_steps_mean", "Cov steps", "lower"),
        ("global_repeat_ratio_mean", "Repeat", "lower"),
        ("inter_agent_overlap_ratio_mean", "Overlap", "lower"),
    ]
    row_labels: list[str] = []
    data: list[list[float]] = []
    baseline = "GAT-OFF"
    for scenario in scenarios:
        for arm in ("GAT-ON", "GAT-CUAP"):
            row_labels.append(f"{scenario.label}\n{arm} vs {baseline}")
            values: list[float] = []
            for metric, _, direction in metric_specs:
                arm_value = float(lookup[(scenario.key, arm)][metric])
                base_value = float(lookup[(scenario.key, baseline)][metric])
                delta = arm_value - base_value if direction == "higher" else base_value - arm_value
                if metric == "coverage_steps_mean":
                    delta /= max(float(scenario.max_steps), 1.0)
                values.append(delta)
            data.append(values)
    matrix = np.asarray(data, dtype=np.float64)
    fig_height = max(7.0, 0.42 * len(row_labels))
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    limit = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1e-6)
    im = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(metric_specs)))
    ax.set_xticklabels([label for _, label, _ in metric_specs], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title("Advantage over GAT-OFF (positive is better)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j] * 100:+.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("percentage points; coverage steps normalized by step budget")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_sample_paths(detail_rows: list[dict[str, Any]], output: Path) -> Path:
    selected_scenarios = ["unseen_20x20_4a_r05", "transfer_30x30_4a_r05", "combined_30x30_6a_r05"]
    selected: list[dict[str, Any]] = []
    for scenario in selected_scenarios:
        seeds = sorted({int(row["seed"]) for row in detail_rows if str(row["scenario"]) == scenario})
        if not seeds:
            continue
        seed = seeds[0]
        for arm in ARM_ORDER:
            match = next((row for row in detail_rows if row["scenario"] == scenario and row["arm"] == arm and int(row["seed"]) == seed), None)
            if match is not None:
                selected.append(match)
    if not selected:
        output.write_text("no sample paths", encoding="utf-8")
        return output

    rows_count = len({row["scenario"] for row in selected})
    cols = len(ARM_ORDER)
    fig, axes = plt.subplots(rows_count, cols, figsize=(4.3 * cols, 4.1 * rows_count), squeeze=False)
    axis_iter = iter(np.ravel(axes))
    for row in selected:
        axis = next(axis_iter)
        payload = json.loads(Path(str(row["trajectory_json"])).read_text(encoding="utf-8"))
        width = int(payload["width"])
        height = int(payload["height"])
        grid = np.zeros((height, width), dtype=np.float32)
        for obstacle in payload["obstacles"]:
            grid[int(obstacle[0]), int(obstacle[1])] = 1.0
        axis.imshow(grid, cmap="Greys", vmin=0, vmax=1, alpha=0.30)
        for index, trajectory in enumerate(payload["trajectories"]):
            points = np.asarray(trajectory)
            axis.plot(points[:, 1], points[:, 0], linewidth=1.0, alpha=0.82)
            axis.scatter(points[0, 1], points[0, 0], marker="s", s=15)
            axis.scatter(points[-1, 1], points[-1, 0], marker="x", s=22)
        axis.set_title(
            f"{row['arm']}\n{row['label']}\n"
            f"cov={float(row['coverage_ratio']) * 100:.1f}%, done={int(row['mission_completed'])}",
            fontsize=8,
        )
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(height - 0.5, -0.5)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axis_iter:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def build_report(
    arms: tuple[Arm, ...],
    scenarios: tuple[Scenario, ...],
    detail_path: Path,
    curves_path: Path,
    summary_path: Path,
    figure_paths: list[Path],
    output_dir: Path,
) -> str:
    summary_rows = read_csv(summary_path)
    lines = [
        "# Three-model ablation: GAT-OFF vs GAT-ON vs GAT-CUAP",
        "",
        "This report evaluates trained checkpoints with deterministic offline rollouts. No additional PPO training is performed.",
        "",
        "## Checkpoints",
        "",
    ]
    for arm in arms:
        lines.append(f"- {arm.name} coverage: `{arm.coverage_checkpoint}`")
    lines.append(f"- Shared return policy: `{arms[0].return_checkpoint}`")
    lines.extend(
        [
            "",
            "## Experimental Setup",
            "",
            "- Task: depot-return coverage. The coverage policy acts until the environment enters return mode; all arms then use the same return policy.",
            "- GAT-OFF is the no-communication explicit-memory baseline from `ablation_mapmsg_gat_off_nocomm`.",
            "- GAT-ON enables shared map memory, coverage messages, and range-limited multi-head GAT.",
            "- GAT-CUAP keeps the GAT-ON architecture and adds the CUAP action-prior logits during coverage.",
            "- Main metrics: Coverage-AUC, final coverage, coverage completion, mission completion, coverage steps, repeated visits, and inter-agent overlap.",
            "",
            "## Key Findings",
            "",
            *key_findings(summary_rows),
            "",
            "## Scenario Summary",
            "",
            "| Scenario | Arm | Ep. | Final cov. | AUC | Cov done | Mission done | Returned | Steps | Cov steps | Return steps | Repeat90 | Overlap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scenario in scenarios:
        for arm in ARM_ORDER:
            row = find_summary(summary_rows, scenario.key, arm)
            lines.append(
                f"| {scenario.label} | {arm} | {int(float(row['episodes']))} | "
                f"{pct(row['coverage_ratio_mean'])} | {float(row['coverage_auc_mean']):.3f} | "
                f"{pct(row['coverage_completed_mean'])} | {pct(row['mission_completed_mean'])} | "
                f"{pct(row['returned_to_depot_mean'])} | {float(row['steps_mean']):.1f} | "
                f"{float(row['coverage_steps_mean']):.1f} | {float(row['return_steps_mean']):.1f} | "
                f"{pct(row['repeat_ratio_after_90_mean'])} | {pct(row['inter_agent_overlap_ratio_mean'])} |"
            )
    lines.extend(["", "## Visual Summary", ""])
    for figure in figure_paths:
        lines.append(f"![{figure.stem}]({figure.relative_to(output_dir).as_posix()})")
        lines.append("")
    lines.extend(
        [
            "## Data Files",
            "",
            f"- Detail rows: `{detail_path.relative_to(output_dir).as_posix()}`",
            f"- Curve rows: `{curves_path.relative_to(output_dir).as_posix()}`",
            f"- Summary rows: `{summary_path.relative_to(output_dir).as_posix()}`",
            "",
            "## Notes",
            "",
            "- `Mission done` requires both full coverage and all agents returning to the depot before the step limit.",
            "- `Coverage-AUC` is averaged over the full episode budget, so it rewards early coverage as well as final coverage.",
            "- `Repeat90` is only meaningful after a trial reaches 90% coverage; trials that never reach 90% report zero for that field by the existing metric convention.",
        ]
    )
    return "\n".join(lines) + "\n"


def key_findings(rows: list[dict[str, str]]) -> list[str]:
    non_native = [row for row in rows if row["category"] != "course-native"]
    aggregate_rows = non_native or rows
    lines: list[str] = []
    for metric, label, higher_is_better in (
        ("coverage_auc_mean", "overall Coverage-AUC", True),
        ("coverage_ratio_mean", "overall final coverage", True),
        ("mission_completed_mean", "overall mission completion", True),
        ("global_repeat_ratio_mean", "overall global repeat", False),
        ("inter_agent_overlap_ratio_mean", "overall inter-agent overlap", False),
    ):
        means = {
            arm: float(np.mean([float(row[metric]) for row in aggregate_rows if row["arm"] == arm]))
            for arm in ARM_ORDER
        }
        best = max(means, key=means.get) if higher_is_better else min(means, key=means.get)
        value = pct(means[best]) if "steps" not in metric else f"{means[best]:.1f}"
        lines.append(f"- Best {label} across non-native scenarios: {best} ({value}).")

    cuap_delta_auc = paired_delta(rows, "GAT-CUAP", "GAT-ON", "coverage_auc_mean")
    cuap_delta_repeat = paired_delta(rows, "GAT-CUAP", "GAT-ON", "global_repeat_ratio_mean")
    on_delta_auc = paired_delta(rows, "GAT-ON", "GAT-OFF", "coverage_auc_mean")
    lines.append(f"- GAT-CUAP vs GAT-ON average Coverage-AUC delta: {cuap_delta_auc:+.3f}; global repeat delta: {cuap_delta_repeat:+.3f}.")
    lines.append(f"- GAT-ON vs GAT-OFF average Coverage-AUC delta: {on_delta_auc:+.3f}.")

    scenario_winners: dict[str, str] = defaultdict(str)
    for scenario in sorted({row["scenario"] for row in rows}):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        best = max(scenario_rows, key=lambda row: float(row["coverage_auc_mean"]))
        scenario_winners[best["arm"]] += "x"
    winner_text = ", ".join(f"{arm}: {len(scenario_winners[arm])}" for arm in ARM_ORDER)
    lines.append(f"- Per-scenario Coverage-AUC wins: {winner_text}.")
    return lines


def paired_delta(rows: list[dict[str, str]], arm: str, baseline: str, metric: str) -> float:
    deltas: list[float] = []
    scenarios = sorted({row["scenario"] for row in rows if row["category"] != "course-native"})
    for scenario in scenarios:
        arm_row = find_summary(rows, scenario, arm)
        base_row = find_summary(rows, scenario, baseline)
        deltas.append(float(arm_row[metric]) - float(base_row[metric]))
    return float(np.mean(deltas)) if deltas else math.nan


def find_summary(rows: list[dict[str, str]], scenario: str, arm: str) -> dict[str, str]:
    for row in rows:
        if row["scenario"] == scenario and row["arm"] == arm:
            return row
    raise KeyError(f"summary row not found: {scenario} {arm}")


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, "", "nan"):
            continue
        try:
            if math.isnan(float(value)):
                continue
        except (TypeError, ValueError):
            continue
        values.append(float(value))
    return values


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = numeric_values(rows, key)
    return float(np.mean(values)) if values else math.nan


def std(rows: list[dict[str, Any]], key: str) -> float:
    values = numeric_values(rows, key)
    return float(np.std(values)) if values else math.nan


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
