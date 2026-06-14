from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path
from typing import Any

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
    resolve_path,
    run_experiment,
    summarize,
    write_csv,
)


FOCUS_FIELDS = [
    "scenario",
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
    "steps",
]


SUMMARY_FIELDS = [
    "scenario",
    "arm",
    "episodes",
    "coverage_auc_mean",
    "t90_mean_reached",
    "t95_mean_reached",
    "coverage_at_100_mean",
    "coverage_at_200_mean",
    "coverage_at_300_mean",
    "repeat_ratio_mean",
    "repeat_ratio_after_90_mean",
    "global_repeat_ratio_mean",
    "coverage_ratio_mean",
    "coverage_completed_mean",
    "steps_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three coverage policies on the exact same random maps.")
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / f"same_random_maps_coverage_only_{date.today().isoformat()}"))
    parser.add_argument("--seeds", default="20261001-20261020")
    parser.add_argument("--obstacle-ratio", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gat-off-coverage", default=str(DEFAULT_OFF_COVERAGE))
    parser.add_argument("--gat-on-coverage", default=str(DEFAULT_ON_COVERAGE))
    parser.add_argument("--gat-cuap-coverage", default=str(DEFAULT_CUAP_COVERAGE))
    parser.add_argument("--return-checkpoint", default=str(DEFAULT_RETURN))
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
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
    scenario = Scenario(
        key=f"same_random_{args.width}x{args.height}_{args.agents}a_r{int(args.obstacle_ratio * 100):02d}",
        label=f"Same random maps {args.width}x{args.height} / {args.agents} agents / {args.obstacle_ratio:.0%}",
        width=args.width,
        height=args.height,
        agents=args.agents,
        max_steps=args.max_steps,
        obstacle_ratio=args.obstacle_ratio,
        seeds=tuple(seeds),
        category="same-random-maps",
    )

    detail_rows, curve_rows = run_experiment(arms, (scenario,), trajectories_dir, coverage_only=True)
    summary_rows = summarize(detail_rows, (scenario,))
    focused_rows = [{field: row.get(field, "") for field in FOCUS_FIELDS} for row in detail_rows]
    focused_summary_rows = [{field: row.get(field, "") for field in SUMMARY_FIELDS} for row in summary_rows]
    winners = per_seed_winners(focused_rows)

    detail_path = output_dir / "detail_rows.csv"
    curves_path = output_dir / "curve_rows.csv"
    focused_path = output_dir / "focused_metrics.csv"
    summary_path = output_dir / "focused_summary.csv"
    winners_path = output_dir / "per_seed_winners.csv"
    report_path = output_dir / "same_random_maps_report.md"
    write_csv(detail_path, detail_rows)
    write_csv(curves_path, curve_rows)
    write_csv(focused_path, focused_rows)
    write_csv(summary_path, focused_summary_rows)
    write_csv(winners_path, winners)
    report_path.write_text(build_report(scenario, focused_summary_rows, winners), encoding="utf-8")

    print(f"seeds={','.join(str(seed) for seed in seeds)}")
    print(f"focused={focused_path}")
    print(f"summary={summary_path}")
    print(f"winners={winners_path}")
    print(f"report={report_path}")


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, stop_text = item.split("-", 1)
            start = int(start_text)
            stop = int(stop_text)
            step = 1 if stop >= start else -1
            seeds.extend(range(start, stop + step, step))
        else:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    return seeds


def per_seed_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    winners: list[dict[str, Any]] = []
    for seed, seed_rows in sorted(by_seed.items()):
        auc_winner = max(seed_rows, key=lambda row: float(row["coverage_auc"]))
        repeat_winner = min(seed_rows, key=lambda row: float(row["repeat_ratio"]))
        c100_winner = max(seed_rows, key=lambda row: float(row["coverage_at_100"]))
        winners.append(
            {
                "seed": seed,
                "auc_winner": auc_winner["arm"],
                "auc": auc_winner["coverage_auc"],
                "c100_winner": c100_winner["arm"],
                "coverage_at_100": c100_winner["coverage_at_100"],
                "repeat_winner": repeat_winner["arm"],
                "repeat_ratio": repeat_winner["repeat_ratio"],
            }
        )
    return winners


def build_report(
    scenario: Scenario,
    summary_rows: list[dict[str, Any]],
    winners: list[dict[str, Any]],
) -> str:
    lines = [
        "# Same-random-maps coverage-only ablation",
        "",
        f"- Scenario: {scenario.label}",
        f"- Seeds: {scenario.seeds[0]}-{scenario.seeds[-1]} ({len(scenario.seeds)} maps)",
        "- Evaluation mode: coverage-only. Return policy is not used.",
        "",
        "## Summary",
        "",
        "| Arm | Ep. | AUC | T90 | T95 | C@100 | C@200 | C@300 | Repeat | Repeat90 | Final cov. | Cov done | Steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['arm']} | {int(float(row['episodes']))} | {float(row['coverage_auc_mean']):.4f} | "
            f"{float(row['t90_mean_reached']):.1f} | {float(row['t95_mean_reached']):.1f} | "
            f"{pct(row['coverage_at_100_mean'])} | {pct(row['coverage_at_200_mean'])} | {pct(row['coverage_at_300_mean'])} | "
            f"{pct(row['repeat_ratio_mean'])} | {pct(row['repeat_ratio_after_90_mean'])} | "
            f"{pct(row['coverage_ratio_mean'])} | {pct(row['coverage_completed_mean'])} | {float(row['steps_mean']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed Wins",
            "",
            f"- AUC wins: {win_counts(winners, 'auc_winner')}",
            f"- Coverage@100 wins: {win_counts(winners, 'c100_winner')}",
            f"- Lowest RepeatRatio wins: {win_counts(winners, 'repeat_winner')}",
        ]
    )
    return "\n".join(lines) + "\n"


def win_counts(rows: list[dict[str, Any]], field: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row[field])] = counts.get(str(row[field]), 0) + 1
    return ", ".join(f"{arm}: {counts.get(arm, 0)}" for arm in ("GAT-OFF", "GAT-ON", "GAT-CUAP"))


def pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%"


if __name__ == "__main__":
    main()
