from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .benchmark import benchmark_policy
from .config import load_config
from .env import GridCoverageEnv
from .evaluation import evaluate_policy, evaluate_two_phase_policy, resolve_runtime_config
from .imitation import pretrain_imitation
from .rendering import render_trajectory
from .training import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(prog="mathbased_mcpp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("doctor", "train", "pretrain", "evaluate", "render", "benchmark"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", default="configs/smoke.toml")
        if name in {"evaluate", "render", "benchmark"}:
            subparser.add_argument("--checkpoint", required=True)
            if name in {"evaluate", "render"}:
                subparser.add_argument("--return-checkpoint", default=None)
        if name in {"train", "pretrain"}:
            subparser.add_argument("--course", default=None)
        if name == "train":
            subparser.add_argument("--previous-checkpoint", default=None)
            subparser.add_argument("--resume-checkpoint", default=None)
            subparser.add_argument("--policy-phase", choices=("coverage", "return", "joint"), default=None)
        if name == "pretrain":
            subparser.add_argument("--episodes", type=int, default=64)
            subparser.add_argument("--epochs", type=int, default=40)
            subparser.add_argument("--batch-size", type=int, default=256)
            subparser.add_argument("--learning-rate", type=float, default=None)
            subparser.add_argument("--run-dir", default=None)
        if name == "benchmark":
            subparser.add_argument("--seeds", default="20260501,20260502,20260503,20260504,20260505")
            subparser.add_argument("--obstacle-ratios", default=None)
            subparser.add_argument("--budgets", default=None)
            subparser.add_argument("--stall-steps", type=int, default=50)
            subparser.add_argument("--output", default=None)

    ablation = subparsers.add_parser("gat-ablation")
    ablation.add_argument("--gat-on-config", default="configs/ablation_gat_on.toml")
    ablation.add_argument("--gat-on-checkpoint", required=True)
    ablation.add_argument("--gat-off-config", default="configs/ablation_gat_off.toml")
    ablation.add_argument("--gat-off-checkpoint", required=True)
    ablation.add_argument("--seeds", default="20260501,20260502,20260503,20260504,20260505")
    ablation.add_argument("--obstacle-ratios", default="0.05,0.10,0.15,0.20")
    ablation.add_argument("--budgets", default=None)
    ablation.add_argument("--stall-steps", type=int, default=50)
    ablation.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.command == "gat-ablation":
        seeds = _parse_int_csv(args.seeds)
        if not seeds:
            parser.error("--seeds must contain at least one integer seed")
        obstacle_ratios = _parse_float_csv(args.obstacle_ratios)
        budgets = _parse_int_csv(args.budgets) if args.budgets else None
        output_path = Path(args.output) if args.output else Path("outputs") / "gat_ablation" / "summary.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        on_detail = output_path.with_name(f"{output_path.stem}_gat_on.csv")
        off_detail = output_path.with_name(f"{output_path.stem}_gat_off.csv")
        gat_on = benchmark_policy(
            load_config(args.gat_on_config),
            args.gat_on_checkpoint,
            seeds=seeds,
            obstacle_ratios=obstacle_ratios,
            output_path=on_detail,
            budgets=budgets,
            stall_steps=args.stall_steps,
        )
        gat_off = benchmark_policy(
            load_config(args.gat_off_config),
            args.gat_off_checkpoint,
            seeds=seeds,
            obstacle_ratios=obstacle_ratios,
            output_path=off_detail,
            budgets=budgets,
            stall_steps=args.stall_steps,
        )
        rows = _ablation_summary_rows(gat_on, gat_off)
        _write_ablation_summary(output_path, rows)
        for row in rows:
            print(
                f"{row['arm']}: coverage_mean={row['coverage_ratio_mean']:.4f}, "
                f"auc={row['coverage_auc_mean']:.4f}, completion_rate={row['completion_rate']:.4f}, "
                f"repeat_after90={row['repeat_ratio_after_90_mean']:.4f}, "
                f"path_length_mean={row['path_length_mean']:.1f}"
            )
        print(f"summary={output_path}")
        print(f"gat_on_rows={on_detail}")
        print(f"gat_off_rows={off_detail}")
        return

    config = load_config(args.config)

    if args.command == "doctor":
        env = GridCoverageEnv(config.env)
        env.reset(seed=config.env.seed)
        print(f"grid={config.env.height}x{config.env.width}")
        print(f"num_agents={config.env.num_agents}")
        print(f"free_cells={len(env.free_cells)}")
        print(f"obstacles={len(env.obstacles)}")
        print(f"random_corner_start={str(config.env.random_corner_start).lower()}")
        print(f"start_position={env.position}")
        print(f"observation_dim={env.observation_dim}")
        print(f"state_dim={env.state_dim}")
        print(f"critic_mode=spatial")
        use_joint_phase_model = (
            config.ppo.policy_phase == "joint"
            and config.env.use_depot
            and config.env.require_return_to_depot
        )
        print(f"use_phase_critics={str(use_joint_phase_model).lower()}")
        print(f"use_phase_actors={str(use_joint_phase_model).lower()}")
        print(f"state_shape={config.env.height}x{config.env.width}")
        print(f"action_dim={env.action_dim}")
        print(f"total_timesteps={config.ppo.total_timesteps}")
        print(f"policy_phase={config.ppo.policy_phase}")
        print(f"num_envs={config.ppo.num_envs}")
        print(f"cpu_threads={config.train.cpu_threads}")
        print(f"use_graph_attention={str(config.ppo.use_graph_attention).lower()}")
        print(f"gat_num_heads={config.ppo.gat_num_heads}")
        print(f"gat_use_edge_features={str(config.ppo.gat_use_edge_features).lower()}")
        print(f"gat_residual={str(config.ppo.gat_residual).lower()}")
        print(f"use_legacy_truth_coverage_observation={str(config.env.use_legacy_truth_coverage_observation).lower()}")
        print(f"use_explicit_map_memory={str(config.env.use_explicit_map_memory).lower()}")
        print(f"share_map_memory={str(config.env.share_map_memory).lower()}")
        print(f"use_coverage_messages={str(config.ppo.use_coverage_messages).lower()}")
        print(f"node_message_dim={env.node_message_dim if config.ppo.use_coverage_messages else 0}")
        print(f"cuap_enabled={str(config.cuap.enabled).lower()}")
        print(f"cuap_beta={config.cuap.beta}")
        print(f"cuap_disable_in_return_phase={str(config.cuap.disable_in_return_phase).lower()}")
        print(f"use_depot={str(config.env.use_depot).lower()}")
        print(f"depot={env.depot_position}")
        print(f"require_return_to_depot={str(config.env.require_return_to_depot).lower()}")
        if config.curriculum and config.curriculum.courses:
            print(f"curriculum_courses={len(config.curriculum.courses)}")
            for index, course in enumerate(config.curriculum.courses, start=1):
                print(
                    f"course_{index}={course.name}:{course.env.height}x{course.env.width}:"
                    f"agents={course.env.num_agents}:max_steps={course.env.max_steps}:timesteps={course.total_timesteps}"
                )
        return

    if args.command == "train":
        if config.curriculum and config.curriculum.courses and not args.course:
            parser.error("curriculum configs require --course so each course can be trained separately")
        if args.policy_phase is not None:
            config.ppo.policy_phase = args.policy_phase
        checkpoint = train_ppo(
            config,
            course=args.course,
            previous_checkpoint=args.previous_checkpoint,
            resume_checkpoint=args.resume_checkpoint,
        )
        print(f"checkpoint={checkpoint}")
        return

    if args.command == "pretrain":
        if config.curriculum and config.curriculum.courses and not args.course:
            parser.error("curriculum configs require --course for imitation pretraining")
        result = pretrain_imitation(
            config,
            run_dir=args.run_dir,
            course=args.course,
            episodes=args.episodes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        print(f"checkpoint={result.checkpoint}")
        print(f"run_dir={result.run_dir}")
        print(f"episodes={result.episodes}")
        print(f"transitions={result.transitions}")
        print(f"final_loss={result.final_loss:.6f}")
        print(f"final_accuracy={result.final_accuracy:.4f}")
        print(f"expert_render={result.expert_render}")
        print(f"bc_render={result.bc_render}")
        return

    checkpoint_path = Path(args.checkpoint)

    if args.command == "benchmark":
        seeds = _parse_int_csv(args.seeds)
        if not seeds:
            parser.error("--seeds must contain at least one integer seed")
        obstacle_ratios = _parse_float_csv(args.obstacle_ratios) if args.obstacle_ratios else None
        budgets = _parse_int_csv(args.budgets) if args.budgets else None
        output_path = Path(args.output) if args.output else checkpoint_path.parent / "benchmark.csv"
        summary = benchmark_policy(
            config,
            checkpoint_path,
            seeds=seeds,
            obstacle_ratios=obstacle_ratios,
            output_path=output_path,
            budgets=budgets,
            stall_steps=args.stall_steps,
        )
        print(f"episodes={summary['episodes']}")
        print(f"coverage_ratio_mean={summary['coverage_ratio_mean']:.4f}")
        print(f"coverage_ratio_min={summary['coverage_ratio_min']:.4f}")
        print(f"completion_rate={summary['completion_rate']:.4f}")
        print(f"path_length_mean={summary['path_length_mean']:.1f}")
        print(f"steps_mean={summary['steps_mean']:.1f}")
        print(f"repeat_ratio_mean={summary['repeat_ratio_mean']:.4f}")
        print(f"repeat_ratio_after_90_mean={summary['repeat_ratio_after_90_mean']:.4f}")
        print(f"coverage_auc_mean={summary['coverage_auc_mean']:.4f}")
        print(f"t90_mean_reached={summary['t90_mean_reached']:.1f}")
        print(f"t95_mean_reached={summary['t95_mean_reached']:.1f}")
        print(f"t99_mean_reached={summary['t99_mean_reached']:.1f}")
        print(f"stall_termination_coverage_mean={summary['stall_termination_coverage_mean']:.4f}")
        for key in sorted(field for field in summary if field.startswith("coverage_at_")):
            print(f"{key}={summary[key]:.4f}")
        print(f"total_reward_mean={summary['total_reward_mean']:.4f}")
        print(f"benchmark={output_path}")
        return

    run_dir = checkpoint_path.parent
    trajectory_path = run_dir / "trajectory.json"
    runtime_config = resolve_runtime_config(config, checkpoint_path)
    if args.return_checkpoint:
        trajectory_path = run_dir / "two_phase_trajectory.json"
        summary = evaluate_two_phase_policy(
            runtime_config,
            checkpoint_path,
            args.return_checkpoint,
            output_path=trajectory_path,
        )
    else:
        summary = evaluate_policy(runtime_config, checkpoint_path, output_path=trajectory_path)

    if args.command == "evaluate":
        print(f"coverage_ratio={summary['coverage_ratio']:.3f}")
        print(f"path_length={summary['path_length']}")
        print(f"completed={str(summary['completed']).lower()}")
        if "coverage_completed" in summary:
            print(f"coverage_completed={str(summary['coverage_completed']).lower()}")
        if "returned_to_depot" in summary:
            print(f"returned_to_depot={str(summary['returned_to_depot']).lower()}")
        if "phase_steps" in summary:
            print(f"phase_steps={summary['phase_steps']}")
        print(f"coverage_auc={summary['coverage_auc']:.4f}")
        print(f"t90={summary['t90']}")
        print(f"t95={summary['t95']}")
        print(f"t99={summary['t99']}")
        print(f"repeat_ratio_after_90={summary['repeat_ratio_after_90']:.4f}")
        print(f"stall_termination_coverage={summary['stall_termination_coverage']:.4f}")
        print(f"trajectory={trajectory_path}")
        return

    if args.command == "render":
        render_name = "two_phase_trajectory.png" if args.return_checkpoint else "trajectory.png"
        output = render_trajectory(runtime_config, summary["trajectory"], run_dir / render_name)
        print(f"trajectory={trajectory_path}")
        print(f"render={output}")
        return


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _ablation_summary_rows(gat_on: dict[str, object], gat_off: dict[str, object]) -> list[dict[str, float | int | str]]:
    fields = [
        "episodes",
        "coverage_ratio_mean",
        "coverage_ratio_min",
        "completion_rate",
        "path_length_mean",
        "steps_mean",
        "repeat_ratio_mean",
        "coverage_auc_mean",
        "repeat_ratio_after_90_mean",
        "inter_agent_overlap_ratio_mean",
        "stall_rate",
        "stall_termination_coverage_mean",
        "t90_mean_reached",
        "t90_reach_rate",
        "t95_mean_reached",
        "t95_reach_rate",
        "t99_mean_reached",
        "t99_reach_rate",
        "total_reward_mean",
    ]
    fields.extend(sorted(key for key in gat_on if key.startswith("coverage_at_")))
    rows: list[dict[str, float | int | str]] = []
    for arm, summary in (("gat_on", gat_on), ("gat_off", gat_off)):
        row: dict[str, float | int | str] = {"arm": arm}
        for field in fields:
            row[field] = summary[field]  # type: ignore[index]
        rows.append(row)
    delta: dict[str, float | int | str] = {"arm": "delta_on_minus_off"}
    for field in fields:
        delta[field] = float(gat_on[field]) - float(gat_off[field])  # type: ignore[index]
    rows.append(delta)
    return rows


def _write_ablation_summary(output_path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = list(rows[0]) if rows else ["arm"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
