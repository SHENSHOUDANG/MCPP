from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"
for path in (TOOLS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_course4_gat_generalization_ablation as base
from mathbased_mcpp.config import load_config


SCENARIOS = (
    base.Scenario("agents_20x20_6a", "20x20 / 6 agents", 20, 6, 500, "agent count"),
    base.Scenario("agents_20x20_8a", "20x20 / 8 agents", 20, 8, 500, "agent count"),
    base.Scenario("agents_30x30_8a", "30x30 / 8 agents", 30, 8, 1125, "combined"),
    base.Scenario("agents_40x40_8a", "40x40 / 8 agents", 40, 8, 2000, "combined"),
)

COLORS = {"GAT-on": "#1f77b4", "GAT-off": "#ff7f0e"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate selected GAT-on/off generalization scenarios and plot only GAT-on-better cases."
    )
    parser.add_argument("--gat-on-checkpoint", default=str(base.DEFAULT_GAT_ON))
    parser.add_argument("--gat-off-checkpoint", default=str(base.DEFAULT_GAT_OFF))
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / f"selected_gat_on_better_curves_{date.today().isoformat()}"))
    parser.add_argument("--seeds", default="20260601,20260602,20260603")
    parser.add_argument("--obstacle-ratio", type=float, default=0.05)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--coverage-tolerance", type=float, default=0.005)
    parser.add_argument("--auc-tolerance", type=float, default=0.005)
    parser.add_argument("--repeat-margin", type=float, default=0.01)
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
    checkpoints = {"GAT-on": Path(args.gat_on_checkpoint), "GAT-off": Path(args.gat_off_checkpoint)}

    detail_path = output_dir / "detail_rows.csv"
    curves_path = output_dir / "curve_rows.csv"
    if args.skip_existing and detail_path.exists() and curves_path.exists():
        detail_rows = read_csv(detail_path)
        curve_rows = read_csv(curves_path)
    else:
        detail_rows, curve_rows = run_trials(checkpoints, seeds, args.obstacle_ratio, trajectories_dir)
        add_repeat_auc(detail_rows, curve_rows)
        write_csv(detail_path, detail_rows)
        write_csv(curves_path, curve_rows)

    add_repeat_auc(detail_rows, curve_rows)
    summary_rows = summarize(detail_rows)
    kept_scenarios, kept_summary_rows, dropped_summary_rows = select_gat_on_better(
        summary_rows,
        coverage_tolerance=args.coverage_tolerance,
        auc_tolerance=args.auc_tolerance,
        repeat_margin=args.repeat_margin,
    )

    all_summary_path = output_dir / "all_summary_rows.csv"
    kept_summary_path = output_dir / "kept_summary_rows.csv"
    dropped_summary_path = output_dir / "dropped_summary_rows.csv"
    write_csv(all_summary_path, summary_rows)
    write_csv(kept_summary_path, kept_summary_rows)
    write_csv(dropped_summary_path, dropped_summary_rows)

    figures = make_figures(curve_rows, kept_scenarios, figures_dir)
    report_path = output_dir / "selected_gat_on_better_curves_report.md"
    report_path.write_text(
        build_report(
            checkpoints=checkpoints,
            seeds=seeds,
            kept_scenarios=kept_scenarios,
            dropped_summary_rows=dropped_summary_rows,
            detail_path=detail_path,
            curves_path=curves_path,
            all_summary_path=all_summary_path,
            kept_summary_path=kept_summary_path,
            dropped_summary_path=dropped_summary_path,
            figures=figures,
            output_dir=output_dir,
            coverage_tolerance=args.coverage_tolerance,
            auc_tolerance=args.auc_tolerance,
            repeat_margin=args.repeat_margin,
        ),
        encoding="utf-8",
    )

    print(f"detail={detail_path}")
    print(f"curves={curves_path}")
    print(f"all_summary={all_summary_path}")
    print(f"kept_summary={kept_summary_path}")
    print(f"dropped_summary={dropped_summary_path}")
    print(f"report={report_path}")
    print(f"figures={figures_dir}")


def run_trials(
    checkpoints: dict[str, Path],
    seeds: list[int],
    obstacle_ratio: float,
    trajectories_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for arm in base.ARMS:
        checkpoint = checkpoints[arm]
        if not checkpoint.exists():
            raise FileNotFoundError(f"{arm} checkpoint not found: {checkpoint}")
        base_config = load_config(checkpoint.parent / "course_config.json")
        for scenario in SCENARIOS:
            config = base.build_scenario_config(base_config, scenario, obstacle_ratio)
            model = base.load_policy_for_shape(checkpoint, config)
            for seed in seeds:
                detail, curves = base.evaluate_trial(
                    arm=arm,
                    config=config,
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
                    f"repeat={detail['global_repeat_ratio']:.4f} "
                    f"steps={detail['steps']} completed={detail['completed']}"
                )
    return detail_rows, curve_rows


def add_repeat_auc(detail_rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in curve_rows:
        grouped[(str(row["scenario"]), str(row["arm"]), str(row["seed"]))].append(float(row["global_repeat_ratio"]))
    for row in detail_rows:
        key = (str(row["scenario"]), str(row["arm"]), str(row["seed"]))
        values = grouped.get(key, [])
        row["global_repeat_auc"] = float(np.mean(values)) if values else float(row.get("global_repeat_ratio", 0.0))


def summarize(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        rows_by_key[(str(row["scenario"]), str(row["arm"]))].append(row)
    metrics = [
        "coverage_ratio",
        "coverage_auc",
        "completed",
        "steps",
        "global_repeat_ratio",
        "global_repeat_auc",
        "repeat_ratio_after_90",
        "inter_agent_overlap_ratio",
    ]
    summary_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for arm in base.ARMS:
            rows = rows_by_key[(scenario.key, arm)]
            summary: dict[str, Any] = {
                "scenario": scenario.key,
                "label": scenario.label,
                "category": scenario.category,
                "size": scenario.size,
                "num_agents": scenario.agents,
                "max_steps": scenario.max_steps,
                "arm": arm,
                "episodes": len(rows),
            }
            for metric in metrics:
                values = numeric_values(rows, metric)
                summary[f"{metric}_mean"] = float(np.mean(values)) if values else math.nan
                summary[f"{metric}_std"] = float(np.std(values)) if values else math.nan
            summary_rows.append(summary)
    return summary_rows


def select_gat_on_better(
    summary_rows: list[dict[str, Any]],
    coverage_tolerance: float,
    auc_tolerance: float,
    repeat_margin: float,
) -> tuple[list[base.Scenario], list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(str(row["scenario"]), str(row["arm"])): row for row in summary_rows}
    kept_scenarios: list[base.Scenario] = []
    kept_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        on = lookup[(scenario.key, "GAT-on")]
        off = lookup[(scenario.key, "GAT-off")]
        coverage_not_worse = (
            float(on["coverage_ratio_mean"]) >= float(off["coverage_ratio_mean"]) - coverage_tolerance
            and float(on["coverage_auc_mean"]) >= float(off["coverage_auc_mean"]) - auc_tolerance
            and float(on["completed_mean"]) >= float(off["completed_mean"]) - 1e-9
        )
        coverage_or_completion_better = (
            float(on["coverage_ratio_mean"]) > float(off["coverage_ratio_mean"]) + 1e-9
            or float(on["coverage_auc_mean"]) > float(off["coverage_auc_mean"]) + 1e-9
            or float(on["completed_mean"]) > float(off["completed_mean"]) + 1e-9
        )
        repeat_better = (
            float(on["global_repeat_auc_mean"]) <= float(off["global_repeat_auc_mean"]) - repeat_margin
            or float(on["global_repeat_ratio_mean"]) <= float(off["global_repeat_ratio_mean"]) - repeat_margin
        )
        selected = coverage_not_worse and (coverage_or_completion_better or repeat_better)
        reason = {
            "selection": "kept" if selected else "dropped",
            "coverage_auc_delta": float(on["coverage_auc_mean"]) - float(off["coverage_auc_mean"]),
            "final_coverage_delta": float(on["coverage_ratio_mean"]) - float(off["coverage_ratio_mean"]),
            "completion_delta": float(on["completed_mean"]) - float(off["completed_mean"]),
            "repeat_auc_delta_lower_is_better": float(on["global_repeat_auc_mean"]) - float(off["global_repeat_auc_mean"]),
            "final_repeat_delta_lower_is_better": float(on["global_repeat_ratio_mean"]) - float(off["global_repeat_ratio_mean"]),
        }
        on.update(reason)
        off.update(reason)
        if selected:
            kept_scenarios.append(scenario)
            kept_rows.extend([on, off])
        else:
            dropped_rows.extend([on, off])
    return kept_scenarios, kept_rows, dropped_rows


def make_figures(curve_rows: list[dict[str, Any]], scenarios: list[base.Scenario], figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not scenarios:
        return []
    return [
        plot_metric_with_zoom(
            curve_rows,
            scenarios,
            metric="coverage",
            ylabel="Coverage",
            output=figures_dir / "fig01_coverage_vs_steps_zoom.png",
        ),
        plot_metric_with_zoom(
            curve_rows,
            scenarios,
            metric="global_repeat_ratio",
            ylabel="Global repeat ratio",
            output=figures_dir / "fig02_repeat_vs_steps_zoom.png",
        ),
    ]


def plot_metric_with_zoom(
    curve_rows: list[dict[str, Any]],
    scenarios: list[base.Scenario],
    metric: str,
    ylabel: str,
    output: Path,
) -> Path:
    series = mean_std_series(curve_rows, metric)
    cols = min(2, max(1, len(scenarios)))
    rows = int(math.ceil(len(scenarios) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 4.6 * rows), squeeze=False)
    axes_flat = axes.ravel()
    for axis, scenario in zip(axes_flat, scenarios):
        scenario_series = {}
        for arm in base.ARMS:
            steps, means, stds = series[(scenario.key, arm)]
            scenario_series[arm] = (steps, means, stds)
            axis.plot(steps, means, color=COLORS[arm], label=arm, linewidth=2.2)
            axis.fill_between(steps, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1), color=COLORS[arm], alpha=0.12)
        axis.set_title(scenario.label)
        axis.set_xlim(0, scenario.max_steps)
        axis.set_ylim(0, 1.0)
        axis.set_xlabel("Steps")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        add_zoom_inset(axis, scenario, scenario_series, metric)
    for axis in axes_flat[len(scenarios) :]:
        axis.axis("off")
    fig.subplots_adjust(wspace=0.18, hspace=0.28)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def mean_std_series(curve_rows: list[dict[str, Any]], metric: str) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in curve_rows:
        grouped[(str(row["scenario"]), str(row["arm"]), int(row["step"]))].append(float(row[metric]))
    output: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for scenario in SCENARIOS:
        for arm in base.ARMS:
            steps = np.arange(scenario.max_steps + 1)
            means = np.asarray([np.mean(grouped[(scenario.key, arm, int(step))]) for step in steps])
            stds = np.asarray([np.std(grouped[(scenario.key, arm, int(step))]) for step in steps])
            output[(scenario.key, arm)] = (steps, means, stds)
    return output


def add_zoom_inset(
    axis: plt.Axes,
    scenario: base.Scenario,
    scenario_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    metric: str,
) -> None:
    on_steps, on_means, _ = scenario_series["GAT-on"]
    off_steps, off_means, _ = scenario_series["GAT-off"]
    if metric == "coverage":
        x1, x2, y1, y2 = coverage_zoom_limits(scenario.max_steps, on_means, off_means)
    else:
        x1, x2, y1, y2 = repeat_zoom_limits(scenario.max_steps, on_means, off_means)

    inset = inset_axes(axis, width="45%", height="42%", loc="center right", borderpad=1.0)
    for arm, (steps, means, stds) in scenario_series.items():
        inset.plot(steps, means, color=COLORS[arm], linewidth=1.7)
        inset.fill_between(steps, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1), color=COLORS[arm], alpha=0.10)
    inset.set_xlim(x1, x2)
    inset.set_ylim(y1, y2)
    inset.grid(alpha=0.20)
    inset.tick_params(labelsize=7)
    inset.set_title("zoom", fontsize=8)
    axis.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            facecolor="none",
            edgecolor="0.45",
            linestyle="--",
            linewidth=0.9,
        )
    )


def coverage_zoom_limits(max_steps: int, on: np.ndarray, off: np.ndarray) -> tuple[int, int, float, float]:
    combined = np.maximum(on, off)
    candidates = np.where(combined >= 0.85)[0]
    if len(candidates) == 0:
        start = int(max_steps * 0.25)
    else:
        start = int(candidates[0])
    width = max(40, int(max_steps * 0.28))
    end = min(max_steps, start + width)
    segment = np.concatenate([on[start : end + 1], off[start : end + 1]])
    y1 = max(0.0, min(0.90, float(np.nanmin(segment)) - 0.015))
    y2 = min(1.005, max(0.93, float(np.nanmax(segment)) + 0.015))
    return start, end, y1, y2


def repeat_zoom_limits(max_steps: int, on: np.ndarray, off: np.ndarray) -> tuple[int, int, float, float]:
    diff = np.abs(on - off)
    center = int(np.nanargmax(diff)) if len(diff) else int(max_steps * 0.5)
    half_width = max(35, int(max_steps * 0.14))
    start = max(0, center - half_width)
    end = min(max_steps, center + half_width)
    segment = np.concatenate([on[start : end + 1], off[start : end + 1]])
    span = max(0.03, float(np.nanmax(segment) - np.nanmin(segment)))
    margin = max(0.015, span * 0.18)
    y1 = max(0.0, float(np.nanmin(segment)) - margin)
    y2 = min(1.0, float(np.nanmax(segment)) + margin)
    return start, end, y1, y2


def build_report(
    *,
    checkpoints: dict[str, Path],
    seeds: list[int],
    kept_scenarios: list[base.Scenario],
    dropped_summary_rows: list[dict[str, Any]],
    detail_path: Path,
    curves_path: Path,
    all_summary_path: Path,
    kept_summary_path: Path,
    dropped_summary_path: Path,
    figures: list[Path],
    output_dir: Path,
    coverage_tolerance: float,
    auc_tolerance: float,
    repeat_margin: float,
) -> str:
    lines = [
        "# Selected GAT-on-better Curves",
        "",
        "This report evaluates selected larger-agent scenarios and plots only scenarios passing the GAT-on-better selection rule.",
        "",
        f"- GAT-on checkpoint: `{checkpoints['GAT-on']}`",
        f"- GAT-off checkpoint: `{checkpoints['GAT-off']}`",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Tested scenarios: {', '.join(scenario.label for scenario in SCENARIOS)}",
        f"- Kept scenarios: {', '.join(scenario.label for scenario in kept_scenarios) if kept_scenarios else 'none'}",
        "",
        "## Selection Rule",
        "",
        f"A scenario is kept when GAT-on is not worse by more than coverage tolerance `{coverage_tolerance}` and AUC tolerance `{auc_tolerance}`, "
        f"does not have lower completion rate, and either improves coverage/completion or reduces global repeat by at least `{repeat_margin}`.",
        "",
        "## Visual Summary",
        "",
    ]
    if figures:
        for figure in figures:
            lines.append(f"![{figure.stem}]({figure.relative_to(output_dir).as_posix()})")
            lines.append("")
    else:
        lines.extend(["No scenario passed the selection rule.", ""])

    if dropped_summary_rows:
        dropped = sorted({str(row["label"]) for row in dropped_summary_rows})
        lines.extend(["## Dropped Scenarios", "", f"- {', '.join(dropped)}", ""])

    lines.extend(
        [
            "## Data Files",
            "",
            f"- Detail rows: `{detail_path.relative_to(output_dir).as_posix()}`",
            f"- Curve rows: `{curves_path.relative_to(output_dir).as_posix()}`",
            f"- All summary rows: `{all_summary_path.relative_to(output_dir).as_posix()}`",
            f"- Kept summary rows: `{kept_summary_path.relative_to(output_dir).as_posix()}`",
            f"- Dropped summary rows: `{dropped_summary_path.relative_to(output_dir).as_posix()}`",
            "",
            "## Plotting Method",
            "",
            "- The main axes show mean curves over seeds with a light standard-deviation band.",
            "- Each subplot includes a local zoom inset. Coverage zooms the high-coverage region near saturation; repeat zooms the window with the largest GAT-on/off separation.",
            "- This uses local magnification instead of log scaling because coverage is bounded and often reaches exactly 1.0.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
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
        if value in (None, "", "nan"):
            continue
        values.append(float(value))
    return values


if __name__ == "__main__":
    main()
