from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_port_scheduler_rl import SUPPORTED_ALGORITHMS, _normalize_algorithm


DEFAULT_COMPARISON_ALGORITHMS = ("shared_mappo", "centralized_ppo", "happo")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run comparable scheduler training candidates and collect summaries.")
    parser.add_argument("--config", default="configs/port_los_angeles_training_v1.toml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--algorithms", default=",".join(DEFAULT_COMPARISON_ALGORITHMS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--env-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume each candidate from its own latest checkpoint.")
    parser.add_argument(
        "--allow-historical-baseline",
        action="store_true",
        help="Forward historical-baseline acknowledgement when using a HISTORICAL config.",
    )
    args = parser.parse_args()

    config_path = _resolve_workspace_path(args.config)
    config = _load_config(config_path)
    output_root = (
        _resolve_workspace_path(args.output_dir)
        if args.output_dir is not None
        else _resolve_workspace_path(str(config.get("output_dir", "outputs/port_inspection/scheduler"))) / "algorithm_comparison"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    algorithms = _parse_algorithms(args.algorithms)
    rows: list[dict[str, Any]] = []
    for algorithm in algorithms:
        candidate_output = output_root / algorithm
        command = [
            sys.executable,
            str(TOOLS / "train_port_scheduler_rl.py"),
            "--config",
            str(config_path),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--algorithm",
            algorithm,
            "--output-dir",
            str(candidate_output),
            "--checkpoint-interval",
            str(args.checkpoint_interval),
        ]
        if args.device is not None:
            command.extend(["--device", args.device])
        if args.num_envs is not None:
            command.extend(["--num-envs", str(args.num_envs)])
        if args.env_workers is not None:
            command.extend(["--env-workers", str(args.env_workers)])
        if args.resume:
            command.extend(["--resume", "auto"])
        if args.allow_historical_baseline:
            command.append("--allow-historical-baseline")

        print(f"running algorithm={algorithm} output={candidate_output}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        rows.append(
            _comparison_row_from_summary(
                algorithm=algorithm,
                seed=args.seed,
                requested_steps=args.steps,
                candidate_output=candidate_output,
            )
        )

    payload = {
        "config": str(config_path),
        "seed": int(args.seed),
        "requested_steps": int(args.steps),
        "algorithms": algorithms,
        "status": "PENDING_ENGINEERING_COMPARISON",
        "note": "These runs compare training candidates only; they do not freeze the final upper-level algorithm.",
        "rows": rows,
    }
    json_path = output_root / "algorithm_comparison_summary.json"
    csv_path = output_root / "algorithm_comparison_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_comparison_csv(csv_path, rows)
    print(f"comparison_json={json_path}")
    print(f"comparison_csv={csv_path}")


def _parse_algorithms(raw: str) -> list[str]:
    algorithms: list[str] = []
    for token in raw.replace(",", " ").split():
        algorithm = _normalize_algorithm(token)
        if algorithm not in algorithms:
            algorithms.append(algorithm)
    if not algorithms:
        raise ValueError("at least one algorithm is required")
    return algorithms


def _comparison_row_from_summary(
    algorithm: str,
    seed: int,
    requested_steps: int,
    candidate_output: Path,
) -> dict[str, Any]:
    summary_path = candidate_output / "scheduler_rl" / "scheduler_summary.json"
    checkpoint_dir = candidate_output / "scheduler_rl"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing scheduler summary for {algorithm}: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row: dict[str, Any] = {
        "algorithm": _normalize_algorithm(algorithm),
        "seed": int(seed),
        "requested_steps": int(requested_steps),
        "steps": int(summary.get("steps", 0)),
        "episode": int(summary.get("episode", 0)),
        "episode_reward": float(summary.get("episode_reward", 0.0)),
        "completed_tasks": int(summary.get("completed_tasks", 0)),
        "late_tasks": int(summary.get("late_tasks", 0)),
        "total_energy": float(summary.get("total_energy", 0.0)),
        "total_conflicts": int(summary.get("total_conflicts", 0)),
        "total_invalid_actions": int(summary.get("total_invalid_actions", 0)),
        "summary_path": str(summary_path),
        "checkpoint_dir": str(checkpoint_dir),
    }
    return row


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "algorithm",
        "seed",
        "requested_steps",
        "steps",
        "episode",
        "episode_reward",
        "completed_tasks",
        "late_tasks",
        "total_energy",
        "total_conflicts",
        "total_invalid_actions",
        "summary_path",
        "checkpoint_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_workspace_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return ROOT / resolved


def _load_config(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


if __name__ == "__main__":
    main()
