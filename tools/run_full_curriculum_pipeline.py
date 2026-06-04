from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.config import ExperimentConfig, load_config
from mathbased_mcpp.imitation import pretrain_imitation
from mathbased_mcpp.training import train_ppo


DEFAULT_CONFIGS = (
    ROOT / "configs" / "ablation_mapmsg_gat_on.toml",
    ROOT / "configs" / "ablation_mapmsg_gat_off.toml",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full imitation-pretrain + curriculum PPO pipeline for one or more configs."
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Config path. Can be passed multiple times. Defaults to latest map-message GAT-on and GAT-off configs.",
    )
    parser.add_argument("--output-root", default=None, help="Optional parent directory for all generated run folders.")
    parser.add_argument("--run-name", default=None, help="Optional shared run name. Defaults to full_pipeline_<timestamp>.")
    parser.add_argument("--courses", default="", help="Comma-separated curriculum course names. Defaults to all courses.")
    parser.add_argument("--pretrain-course", default="", help="Course used for behavior-cloning pretraining. Defaults to the first selected course.")
    parser.add_argument("--skip-pretrain", action="store_true", help="Start PPO without behavior-cloning pretraining.")
    parser.add_argument("--initial-checkpoint", default=None, help="Optional checkpoint used as the first PPO initialization.")
    parser.add_argument("--pretrain-episodes", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-learning-rate", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned pipeline without running training.")
    args = parser.parse_args()

    config_paths = [Path(item) for item in args.config] if args.config else list(DEFAULT_CONFIGS)
    run_name = args.run_name or f"full_pipeline_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    all_summaries: list[dict[str, Any]] = []

    for config_path in config_paths:
        config = load_config(config_path)
        if not config.curriculum or not config.curriculum.courses:
            raise ValueError(f"{config_path} does not define curriculum courses")
        selected_courses = select_courses(config, args.courses)
        pretrain_course = args.pretrain_course.strip() or selected_courses[0]
        master_run_dir = resolve_master_run_dir(config, config_path, args.output_root, run_name)
        summary = run_pipeline_for_config(
            config_path=config_path,
            config=config,
            selected_courses=selected_courses,
            pretrain_course=pretrain_course,
            master_run_dir=master_run_dir,
            skip_pretrain=args.skip_pretrain,
            initial_checkpoint=Path(args.initial_checkpoint) if args.initial_checkpoint else None,
            pretrain_episodes=args.pretrain_episodes,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_batch_size=args.pretrain_batch_size,
            pretrain_learning_rate=args.pretrain_learning_rate,
            dry_run=args.dry_run,
        )
        all_summaries.append(summary)

    print(json.dumps(all_summaries, indent=2, ensure_ascii=False))


def select_courses(config: ExperimentConfig, courses_arg: str) -> list[str]:
    assert config.curriculum is not None
    available = [course.name for course in config.curriculum.courses]
    if not courses_arg.strip():
        return available
    requested = [item.strip() for item in courses_arg.split(",") if item.strip()]
    missing = [course for course in requested if course not in available]
    if missing:
        raise ValueError(f"unknown courses: {', '.join(missing)}; available: {', '.join(available)}")
    return requested


def resolve_master_run_dir(config: ExperimentConfig, config_path: Path, output_root: str | None, run_name: str) -> Path:
    root = Path(output_root) if output_root else Path(config.train.run_root)
    arm_name = config_path.stem
    return root / run_name / arm_name


def run_pipeline_for_config(
    *,
    config_path: Path,
    config: ExperimentConfig,
    selected_courses: list[str],
    pretrain_course: str,
    master_run_dir: Path,
    skip_pretrain: bool,
    initial_checkpoint: Path | None,
    pretrain_episodes: int,
    pretrain_epochs: int,
    pretrain_batch_size: int,
    pretrain_learning_rate: float | None,
    dry_run: bool,
) -> dict[str, Any]:
    master_run_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "config": str(config_path),
        "master_run_dir": str(master_run_dir),
        "pretrain_course": None if skip_pretrain else pretrain_course,
        "courses": [],
    }
    plan_path = master_run_dir / "pipeline_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "config": str(config_path),
                "config_snapshot": asdict(config),
                "selected_courses": selected_courses,
                "pretrain_course": None if skip_pretrain else pretrain_course,
                "pretrain": {
                    "episodes": pretrain_episodes,
                    "epochs": pretrain_epochs,
                    "batch_size": pretrain_batch_size,
                    "learning_rate": pretrain_learning_rate,
                },
                "dry_run": dry_run,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    previous_checkpoint = initial_checkpoint
    if not skip_pretrain:
        pretrain_dir = master_run_dir / f"00-pretrain-{slugify(pretrain_course)}"
        summary["pretrain_run_dir"] = str(pretrain_dir)
        if dry_run:
            previous_checkpoint = pretrain_dir / "bc_policy.pt"
            summary["pretrain_checkpoint"] = str(previous_checkpoint)
        else:
            result = pretrain_imitation(
                config,
                run_dir=pretrain_dir,
                course=pretrain_course,
                episodes=pretrain_episodes,
                epochs=pretrain_epochs,
                batch_size=pretrain_batch_size,
                learning_rate=pretrain_learning_rate,
            )
            previous_checkpoint = result.checkpoint
            summary["pretrain_checkpoint"] = str(result.checkpoint)
            summary["pretrain_transitions"] = result.transitions
            summary["pretrain_final_loss"] = result.final_loss
            summary["pretrain_final_accuracy"] = result.final_accuracy

    for course_name in selected_courses:
        course_entry: dict[str, Any] = {
            "course": course_name,
            "previous_checkpoint": str(previous_checkpoint) if previous_checkpoint else None,
        }
        if dry_run:
            course_dir = master_run_dir / f"{course_index(config, course_name):02d}-{slugify(course_name)}"
            policy_checkpoint = course_dir / "policy.pt"
            best_checkpoint = course_dir / "best_policy.pt"
            previous_checkpoint = best_checkpoint
            course_entry["policy_checkpoint"] = str(policy_checkpoint)
            course_entry["best_checkpoint"] = str(best_checkpoint)
        else:
            policy_checkpoint = train_ppo(
                config,
                course=course_name,
                run_dir=master_run_dir,
                previous_checkpoint=previous_checkpoint,
            )
            best_checkpoint = policy_checkpoint.parent / "best_policy.pt"
            previous_checkpoint = best_checkpoint if best_checkpoint.exists() else policy_checkpoint
            course_entry["policy_checkpoint"] = str(policy_checkpoint)
            course_entry["best_checkpoint"] = str(previous_checkpoint)
        summary["courses"].append(course_entry)

    summary_path = master_run_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["plan_path"] = str(plan_path)
    return summary


def course_index(config: ExperimentConfig, course_name: str) -> int:
    assert config.curriculum is not None
    for index, course in enumerate(config.curriculum.courses, start=1):
        if course.name == course_name:
            return index
    raise ValueError(f"unknown course: {course_name}")


def slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "course"


if __name__ == "__main__":
    main()
