from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_three_model_ablation import (  # noqa: E402
    Arm,
    DEFAULT_CUAP_COVERAGE,
    DEFAULT_OFF_COVERAGE,
    DEFAULT_ON_COVERAGE,
    DEFAULT_RETURN,
    Scenario,
    read_csv,
    resolve_path,
    run_experiment,
    summarize,
    write_csv,
)


FOCUS_FIELDS = [
    "scenario",
    "label",
    "arm",
    "seed",
    "coverage_auc",
    "t90",
    "t95",
    "coverage_at_100",
    "coverage_at_200",
    "coverage_at_300",
    "repeat_ratio",
    "repeat_ratio_after_90",
    "global_repeat_ratio",
    "coverage_ratio",
    "coverage_completed",
    "mission_completed",
    "coverage_steps",
    "return_steps",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select CUAP's best seed, then compare three models on focused seeds.")
    parser.add_argument("--source-detail", default=str(ROOT / "reports" / "three_model_ablation_2026-06-07" / "detail_rows.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / f"seed_focus_ablation_{date.today().isoformat()}"))
    parser.add_argument("--gat-off-coverage", default=str(DEFAULT_OFF_COVERAGE))
    parser.add_argument("--gat-on-coverage", default=str(DEFAULT_ON_COVERAGE))
    parser.add_argument("--gat-cuap-coverage", default=str(DEFAULT_CUAP_COVERAGE))
    parser.add_argument("--return-checkpoint", default=str(DEFAULT_RETURN))
    args = parser.parse_args()

    source_detail = resolve_path(args.source_detail)
    best = select_best_cuap_seed(source_detail)
    training_seeds = load_training_obstacle_seeds(resolve_path(args.gat_cuap_coverage))
    output_dir = Path(args.output_dir)
    trajectories_dir = output_dir / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    return_checkpoint = resolve_path(args.return_checkpoint)
    arms = (
        Arm("GAT-OFF", resolve_path(args.gat_off_coverage), return_checkpoint),
        Arm("GAT-ON", resolve_path(args.gat_on_coverage), return_checkpoint),
        Arm("GAT-CUAP", resolve_path(args.gat_cuap_coverage), return_checkpoint),
    )
    scenarios = (
        scenario_from_best_row(best),
        Scenario(
            key="course4_native_config",
            label="Course-4 native config seed 20260431",
            width=20,
            height=20,
            agents=4,
            max_steps=500,
            obstacle_ratio=0.05,
            seeds=(),
            category="training-native",
            course_native=True,
        ),
        Scenario(
            key="course4_training_obstacle_seeds",
            label="Course-4 training obstacle seeds 20260440-20260447",
            width=20,
            height=20,
            agents=4,
            max_steps=500,
            obstacle_ratio=0.05,
            seeds=tuple(training_seeds),
            category="training-seeds",
        ),
    )

    detail_rows, curve_rows = run_experiment(arms, scenarios, trajectories_dir, coverage_only=True)
    summary_rows = summarize(detail_rows, scenarios)
    focused_rows = [{field: row.get(field, "") for field in FOCUS_FIELDS} for row in detail_rows]
    focused_summary_rows = build_focused_summary(summary_rows)

    detail_path = output_dir / "detail_rows.csv"
    curves_path = output_dir / "curve_rows.csv"
    summary_path = output_dir / "summary_rows.csv"
    focused_path = output_dir / "focused_metrics.csv"
    focused_summary_path = output_dir / "focused_summary.csv"
    report_path = output_dir / "seed_focus_report.md"
    write_csv(detail_path, detail_rows)
    write_csv(curves_path, curve_rows)
    write_csv(summary_path, summary_rows)
    write_csv(focused_path, focused_rows)
    write_csv(focused_summary_path, focused_summary_rows)
    report_path.write_text(build_report(best, training_seeds, focused_rows, focused_summary_rows), encoding="utf-8")

    print(f"best_cuap_source_scenario={best['scenario']}")
    print(f"best_cuap_seed={best['seed']}")
    print(f"best_cuap_coverage_auc={best['coverage_auc']}")
    print(f"training_seeds={','.join(str(seed) for seed in training_seeds)}")
    print(f"focused={focused_path}")
    print(f"summary={focused_summary_path}")
    print(f"report={report_path}")


def select_best_cuap_seed(source_detail: Path) -> dict[str, str]:
    rows = read_csv(source_detail)
    cuap_rows = [row for row in rows if row.get("arm") == "GAT-CUAP"]
    if not cuap_rows:
        raise ValueError(f"no GAT-CUAP rows found in {source_detail}")
    return max(cuap_rows, key=lambda row: float(row["coverage_auc"]))


def scenario_from_best_row(row: dict[str, str]) -> Scenario:
    seed = int(row["seed"])
    obstacle_raw = row.get("obstacle_ratio", "")
    obstacle_ratio = 0.05 if obstacle_raw == "" else float(obstacle_raw)
    return Scenario(
        key=f"cuap_best_seed_{seed}",
        label=f"CUAP best prior-eval seed {seed} ({row['label']})",
        width=int(float(row["width"])),
        height=int(float(row["height"])),
        agents=int(float(row["num_agents"])),
        max_steps=int(float(row["max_steps"])),
        obstacle_ratio=obstacle_ratio,
        seeds=(seed,),
        category="cuap-best-seed",
    )


def load_training_obstacle_seeds(cuap_checkpoint: Path) -> list[int]:
    config_path = cuap_checkpoint.parent / "course_config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in raw["env"].get("random_obstacle_seeds", [])]
    if not seeds:
        seeds = [int(raw["env"]["random_obstacle_seed"])]
    return seeds


def build_focused_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "scenario",
        "label",
        "arm",
        "episodes",
        "coverage_auc_mean",
        "t90_mean_reached",
        "t90_reach_rate",
        "t95_mean_reached",
        "t95_reach_rate",
        "coverage_at_100_mean",
        "coverage_at_200_mean",
        "coverage_at_300_mean",
        "repeat_ratio_mean",
        "repeat_ratio_after_90_mean",
        "global_repeat_ratio_mean",
        "coverage_ratio_mean",
        "coverage_completed_mean",
        "mission_completed_mean",
        "coverage_steps_mean",
        "return_steps_mean",
    ]
    return [{field: row.get(field, "") for field in fields} for row in summary_rows]


def build_report(
    best: dict[str, str],
    training_seeds: list[int],
    focused_rows: list[dict[str, Any]],
    focused_summary_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Seed-focused coverage-only three-model ablation",
        "",
        "## Selected Seed",
        "",
        f"- CUAP best seed from prior detail rows: `{best['seed']}`",
        f"- Source scenario: `{best['scenario']}` / {best['label']}",
        f"- CUAP source Coverage-AUC: `{float(best['coverage_auc']):.4f}`",
        f"- Course-4 training obstacle seeds: `{', '.join(str(seed) for seed in training_seeds)}`",
        "- Evaluation mode: coverage-only. Return policy is not used; coverage completion terminates the rollout.",
        "",
        "## Focused Summary",
        "",
        "| Scenario | Arm | Ep. | AUC | T90 | T95 | C@100 | C@200 | C@300 | Repeat | Repeat90 | Final cov. | Cov done | Mission done |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in focused_summary_rows:
        lines.append(
            f"| {row['label']} | {row['arm']} | {int(float(row['episodes']))} | "
            f"{float(row['coverage_auc_mean']):.4f} | {step_value(row['t90_mean_reached'], row['t90_reach_rate'])} | "
            f"{step_value(row['t95_mean_reached'], row['t95_reach_rate'])} | "
            f"{pct(row['coverage_at_100_mean'])} | {pct(row['coverage_at_200_mean'])} | {pct(row['coverage_at_300_mean'])} | "
            f"{pct(row['repeat_ratio_mean'])} | {pct(row['repeat_ratio_after_90_mean'])} | {pct(row['coverage_ratio_mean'])} | "
            f"{pct(row['coverage_completed_mean'])} | {pct(row['mission_completed_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed Focused Rows",
            "",
            "| Scenario | Arm | Seed | AUC | T90 | T95 | C@100 | C@200 | C@300 | Repeat | Repeat90 | Final cov. |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in focused_rows:
        lines.append(
            f"| {row['label']} | {row['arm']} | {row['seed']} | {float(row['coverage_auc']):.4f} | "
            f"{plain_step(row['t90'])} | {plain_step(row['t95'])} | "
            f"{pct(row['coverage_at_100'])} | {pct(row['coverage_at_200'])} | {pct(row['coverage_at_300'])} | "
            f"{pct(row['repeat_ratio'])} | {pct(row['repeat_ratio_after_90'])} | {pct(row['coverage_ratio'])} |"
        )
    return "\n".join(lines) + "\n"


def pct(value: Any) -> str:
    if value in ("", None):
        return ""
    numeric = float(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric * 100:.1f}%"


def plain_step(value: Any) -> str:
    if value in ("", None):
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(numeric):
        return "-"
    return f"{numeric:.0f}"


def step_value(value: Any, reach_rate: Any) -> str:
    step = plain_step(value)
    if step == "-":
        return "-"
    return f"{step} ({pct(reach_rate)})"


if __name__ == "__main__":
    main()
