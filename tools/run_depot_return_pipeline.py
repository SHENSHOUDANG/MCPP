from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.config import ExperimentConfig, load_config
from mathbased_mcpp.imitation import pretrain_imitation
from mathbased_mcpp.training import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full depot-return training pipeline: imitation warm start, "
            "coverage-only curriculum, then depot-return curriculum."
        )
    )
    parser.add_argument("--config", default="configs/formal_v1.toml")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--pretrain-episodes", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument(
        "--return-timesteps-scale",
        type=float,
        default=0.6,
        help="Scale PPO total_timesteps for the depot-return curriculum; coverage timesteps are unchanged.",
    )
    parser.add_argument(
        "--return-min-timesteps",
        type=int,
        default=0,
        help="Optional lower bound for each return course after scaling. Defaults to 0 so smoke configs stay tiny.",
    )
    parser.add_argument("--return-num-envs", type=int, default=8)
    parser.add_argument("--return-mini-batch-size", type=int, default=1024)
    parser.add_argument("--return-cpu-threads", type=int, default=6)
    parser.add_argument("--return-start-strategy", default="diverse", choices=("farthest", "diverse"))
    parser.add_argument(
        "--return-run-name",
        default="return",
        help="Directory name under the pipeline root for return-phase runs.",
    )
    parser.add_argument(
        "--retrain-return",
        action="store_true",
        help="Ignore completed return checkpoints and train the return phase again.",
    )
    parser.add_argument(
        "--shared-return-checkpoint",
        default=None,
        help="Existing return policy checkpoint to reuse instead of training a return phase for this arm.",
    )
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--skip-return", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_config = load_config(ROOT / args.config)
    if not base_config.curriculum or not base_config.curriculum.courses:
        raise ValueError("the pipeline requires a curriculum config")
    shared_return_checkpoint = _resolve_shared_return_checkpoint(args.shared_return_checkpoint)
    if shared_return_checkpoint is not None and args.retrain_return:
        parser.error("--shared-return-checkpoint cannot be combined with --retrain-return")

    pipeline_root = Path(args.run_root) if args.run_root else Path(base_config.train.run_root) / "depot_return_pipeline"
    coverage_run_root = pipeline_root / "coverage"
    return_run_root = pipeline_root / args.return_run_name
    coverage_config = _phase_config(base_config, coverage_run_root, phase="coverage")
    return_config = _phase_config(
        base_config,
        return_run_root,
        phase="return",
        timestep_scale=args.return_timesteps_scale,
        min_timesteps=args.return_min_timesteps,
        return_num_envs=args.return_num_envs,
        return_mini_batch_size=args.return_mini_batch_size,
        return_cpu_threads=args.return_cpu_threads,
        return_start_strategy=args.return_start_strategy,
    )
    course_names = [course.name for course in base_config.curriculum.courses]

    if args.dry_run:
        print(f"pipeline_root={pipeline_root}")
        print(f"courses={','.join(course_names)}")
        print(f"pretrain={'no' if args.skip_pretrain else 'yes'}")
        print(f"coverage_phase={'no' if args.skip_coverage else 'yes'}")
        print(f"return_phase={'no' if args.skip_return else 'yes'}")
        print(f"resume={str(args.resume).lower()}")
        print(f"coverage_run_root={coverage_config.train.run_root}")
        print(f"return_run_root={return_config.train.run_root}")
        print(f"return_run_name={args.return_run_name}")
        print(f"retrain_return={str(args.retrain_return).lower()}")
        print(f"shared_return_checkpoint={shared_return_checkpoint}")
        print(f"return_training={'no' if shared_return_checkpoint is not None or args.skip_return else 'yes'}")
        print(f"return_timesteps_scale={args.return_timesteps_scale}")
        print(f"return_min_timesteps={args.return_min_timesteps}")
        print(f"return_num_envs={return_config.ppo.num_envs}")
        print(f"return_mini_batch_size={return_config.ppo.mini_batch_size}")
        print(f"return_cpu_threads={return_config.train.cpu_threads}")
        print(f"return_start_strategy={return_config.env.return_start_strategy}")
        if return_config.curriculum is not None:
            budgets = ",".join(f"{course.name}:{course.total_timesteps}" for course in return_config.curriculum.courses)
            print(f"return_total_timesteps={budgets}")
        print("dry_run=true")
        return

    pipeline_root.mkdir(parents=True, exist_ok=True)
    _write_config_snapshot(pipeline_root / "coverage_config.json", coverage_config)
    _write_config_snapshot(pipeline_root / "return_config.json", return_config)
    if shared_return_checkpoint is not None:
        (pipeline_root / "shared_return_checkpoint.txt").write_text(str(shared_return_checkpoint), encoding="utf-8")

    pretrain_checkpoint: Path | None = None
    if not args.skip_pretrain:
        first_course = course_names[0]
        pretrain_dir = pipeline_root / "imitation" / first_course
        existing_pretrain = pretrain_dir / "bc_policy.pt"
        if args.resume and existing_pretrain.exists():
            pretrain_checkpoint = existing_pretrain
            print(f"[pretrain] reuse checkpoint={pretrain_checkpoint}")
        else:
            print(f"[pretrain] course={first_course}")
            pretrain_result = pretrain_imitation(
                coverage_config,
                run_dir=pretrain_dir,
                course=first_course,
                episodes=args.pretrain_episodes,
                epochs=args.pretrain_epochs,
                batch_size=args.pretrain_batch_size,
            )
            pretrain_checkpoint = pretrain_result.checkpoint
            print(f"[pretrain] checkpoint={pretrain_checkpoint}")

    coverage_checkpoints: dict[str, Path] = {}
    if not args.skip_coverage:
        previous = pretrain_checkpoint
        for index, course_name in enumerate(course_names):
            course_dir = _course_run_dir(coverage_run_root, index, course_name)
            completed_checkpoint = _completed_checkpoint(course_dir) if args.resume else None
            if completed_checkpoint is not None:
                coverage_checkpoints[course_name] = completed_checkpoint
                previous = completed_checkpoint
                print(f"[coverage] skip completed course={course_name} checkpoint={completed_checkpoint}")
                continue
            resume_checkpoint = _resume_checkpoint(course_dir) if args.resume else None
            print(f"[coverage] course={course_name} previous={previous} resume={resume_checkpoint}")
            checkpoint = train_ppo(
                coverage_config,
                run_dir=coverage_run_root,
                course=course_name,
                previous_checkpoint=previous,
                resume_checkpoint=resume_checkpoint,
            )
            coverage_checkpoints[course_name] = checkpoint
            previous = checkpoint
            print(f"[coverage] checkpoint={checkpoint}")
    elif args.resume:
        coverage_checkpoints = _load_completed_phase_checkpoints(coverage_run_root, course_names)

    final_return_checkpoint: Path | None = None
    if shared_return_checkpoint is not None:
        final_return_checkpoint = shared_return_checkpoint
        print(f"[return] use shared checkpoint={shared_return_checkpoint}")
    elif not args.skip_return:
        previous = coverage_checkpoints.get(course_names[0], pretrain_checkpoint)
        for index, course_name in enumerate(course_names):
            course_dir = _course_run_dir(return_run_root, index, course_name)
            completed_checkpoint = None if args.retrain_return else (_completed_checkpoint(course_dir) if args.resume else None)
            if completed_checkpoint is not None:
                previous = completed_checkpoint
                final_return_checkpoint = completed_checkpoint
                print(f"[return] skip completed course={course_name} checkpoint={completed_checkpoint}")
                continue
            resume_checkpoint = _resume_checkpoint(course_dir) if args.resume else None
            if index == 0:
                initial = coverage_checkpoints.get(course_name, previous)
            else:
                initial = previous
            print(f"[return] course={course_name} previous={initial} resume={resume_checkpoint}")
            checkpoint = train_ppo(
                return_config,
                run_dir=return_run_root,
                course=course_name,
                previous_checkpoint=initial,
                resume_checkpoint=resume_checkpoint,
            )
            previous = checkpoint
            print(f"[return] checkpoint={checkpoint}")
        final_return_checkpoint = previous

    final_coverage = coverage_checkpoints.get(course_names[-1])
    print(f"[done] pipeline_root={pipeline_root}")
    if final_coverage is not None:
        print(f"[done] final_coverage_checkpoint={final_coverage}")
    if final_return_checkpoint is not None:
        print(f"[done] final_return_checkpoint={final_return_checkpoint}")


def _phase_config(
    base_config: ExperimentConfig,
    run_root: Path,
    *,
    phase: str,
    timestep_scale: float = 1.0,
    min_timesteps: int = 0,
    return_num_envs: int | None = None,
    return_mini_batch_size: int | None = None,
    return_cpu_threads: int | None = None,
    return_start_strategy: str | None = None,
) -> ExperimentConfig:
    config = copy.deepcopy(base_config)
    config.train.run_root = str(run_root)
    config.ppo.policy_phase = phase
    _apply_depot_phase(config, require_return=phase == "return")
    if phase == "return":
        _scale_return_timesteps(config, timestep_scale=timestep_scale, min_timesteps=min_timesteps)
        _apply_return_training_overrides(
            config,
            num_envs=return_num_envs,
            mini_batch_size=return_mini_batch_size,
            cpu_threads=return_cpu_threads,
            start_strategy=return_start_strategy,
        )
    return config


def _scale_return_timesteps(config: ExperimentConfig, *, timestep_scale: float, min_timesteps: int = 0) -> None:
    if timestep_scale <= 0:
        raise ValueError("return timestep scale must be positive")
    lower_bound = max(int(min_timesteps), 0)
    config.ppo.total_timesteps = _scaled_timesteps(config.ppo.total_timesteps, timestep_scale, lower_bound)
    if config.curriculum is None:
        return
    for course in config.curriculum.courses:
        course.total_timesteps = _scaled_timesteps(course.total_timesteps, timestep_scale, lower_bound)


def _scaled_timesteps(total_timesteps: int, timestep_scale: float, lower_bound: int) -> int:
    scaled = max(1, int(round(int(total_timesteps) * float(timestep_scale))))
    return max(scaled, lower_bound)


def _apply_return_training_overrides(
    config: ExperimentConfig,
    *,
    num_envs: int | None,
    mini_batch_size: int | None,
    cpu_threads: int | None,
    start_strategy: str | None,
) -> None:
    if num_envs is not None:
        config.ppo.num_envs = max(int(num_envs), 1)
    if mini_batch_size is not None:
        config.ppo.mini_batch_size = max(int(mini_batch_size), 1)
    if cpu_threads is not None:
        config.train.cpu_threads = max(int(cpu_threads), 1)
    if start_strategy is not None:
        strategy = start_strategy.strip().lower()
        if strategy not in {"farthest", "diverse"}:
            raise ValueError(f"unknown return start strategy: {start_strategy}")
        config.env.return_start_strategy = strategy
        if config.curriculum is not None:
            for course in config.curriculum.courses:
                course.env.return_start_strategy = strategy


def _resolve_shared_return_checkpoint(path: str | None) -> Path | None:
    if not path:
        return None
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"shared return checkpoint not found: {checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"shared return checkpoint is not a file: {checkpoint}")
    return checkpoint


def _apply_depot_phase(config: ExperimentConfig, *, require_return: bool) -> None:
    config.env.use_depot = True
    config.env.require_return_to_depot = require_return
    if config.env.depot is None:
        config.env.depot = config.env.start
    if config.curriculum is None:
        return
    for course in config.curriculum.courses:
        course.env.use_depot = True
        course.env.require_return_to_depot = require_return
        if course.env.depot is None:
            course.env.depot = config.env.depot


def _write_config_snapshot(path: Path, config: ExperimentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def _course_run_dir(run_root: Path, index: int, course_name: str) -> Path:
    return run_root / f"{index + 1:02d}-{_slugify(course_name)}"


def _slugify(text: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "course"


def _completed_checkpoint(course_dir: Path) -> Path | None:
    checkpoint = course_dir / "policy.pt"
    if checkpoint.exists() and (course_dir / "training_complete.json").exists():
        return checkpoint
    return None


def _resume_checkpoint(course_dir: Path) -> Path | None:
    checkpoint = course_dir / "last_policy.pt"
    if checkpoint.exists() and not (course_dir / "training_complete.json").exists():
        return checkpoint
    return None


def _load_completed_phase_checkpoints(run_root: Path, course_names: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for index, course_name in enumerate(course_names):
        checkpoint = _completed_checkpoint(_course_run_dir(run_root, index, course_name))
        if checkpoint is not None:
            checkpoints[course_name] = checkpoint
    return checkpoints


if __name__ == "__main__":
    main()
