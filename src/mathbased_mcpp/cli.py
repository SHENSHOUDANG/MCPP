from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .env import GridCoverageEnv
from .evaluation import evaluate_policy, resolve_runtime_config
from .rendering import render_trajectory
from .training import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(prog="mathbased_mcpp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("doctor", "train", "evaluate", "render"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", default="configs/smoke.toml")
        if name in {"evaluate", "render"}:
            subparser.add_argument("--checkpoint", required=True)
        if name == "train":
            subparser.add_argument("--course", default=None)
            subparser.add_argument("--previous-checkpoint", default=None)

    args = parser.parse_args()
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
        print(f"state_shape={config.env.height}x{config.env.width}")
        print(f"action_dim={env.action_dim}")
        print(f"total_timesteps={config.ppo.total_timesteps}")
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
        checkpoint = train_ppo(config, course=args.course, previous_checkpoint=args.previous_checkpoint)
        print(f"checkpoint={checkpoint}")
        return

    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent
    trajectory_path = run_dir / "trajectory.json"
    runtime_config = resolve_runtime_config(config, checkpoint_path)
    summary = evaluate_policy(runtime_config, checkpoint_path, output_path=trajectory_path)

    if args.command == "evaluate":
        print(f"coverage_ratio={summary['coverage_ratio']:.3f}")
        print(f"path_length={summary['path_length']}")
        print(f"completed={str(summary['completed']).lower()}")
        print(f"trajectory={trajectory_path}")
        return

    if args.command == "render":
        output = render_trajectory(runtime_config, summary["trajectory"], run_dir / "trajectory.png")
        print(f"trajectory={trajectory_path}")
        print(f"render={output}")
        return
