from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.port_inspection import assign_tasks, create_platforms, load_inspection_tasks, load_port_grid
from mathbased_mcpp.port_inspection.render import render_port_inspection_map
from mathbased_mcpp.port_inspection.schema import AssignmentResult


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the rule-based UAV-USV water inspection baseline.")
    parser.add_argument("--config", default="configs/port_dalian_water_v0.toml")
    args = parser.parse_args()

    config = _load_config(args.config)
    grid = load_port_grid(config["grid_path"])
    tasks = load_inspection_tasks(config["tasks_path"], grid)
    platform_config = dict(config.get("platform", {}))
    platforms = create_platforms(
        depot=tuple(config.get("depot", grid.depot)),  # type: ignore[arg-type]
        uav_count=int(config.get("uav_count", 1)),
        usv_count=int(config.get("usv_count", 1)),
        uav_config=dict(platform_config.get("uav", {})),
        usv_config=dict(platform_config.get("usv", {})),
    )
    scheduling = dict(config.get("scheduling", {}))
    assignments = assign_tasks(
        grid,
        tasks,
        platforms,
        risk_weight=float(scheduling.get("risk_weight", 12.0)),
        distance_weight=float(scheduling.get("distance_weight", 0.08)),
        load_weight=float(scheduling.get("load_weight", 0.35)),
        compatibility_bonus=float(scheduling.get("compatibility_bonus", 4.0)),
    )

    output_dir = Path(config.get("output_dir", "outputs/port_inspection/dalian_port_v0"))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline_assignments.csv"
    v2_csv_path = output_dir / "baseline_result.csv"
    summary_path = output_dir / "baseline_summary.json"
    preview_path = output_dir / "baseline_preview.png"
    _write_assignments(csv_path, assignments)
    _write_assignments(v2_csv_path, assignments)
    summary_path.write_text(json.dumps(_summary(assignments), indent=2, ensure_ascii=False), encoding="utf-8")
    render_port_inspection_map(grid, tasks, preview_path, assignments=assignments, title="Rule-based UAV-USV inspection baseline")
    print(f"assignments={csv_path}")
    print(f"baseline_result={v2_csv_path}")
    print(f"summary={summary_path}")
    print(f"preview={preview_path}")


def _load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _write_assignments(path: Path, assignments: list[AssignmentResult]) -> None:
    fieldnames = [
        "task_id",
        "task_type",
        "task_geometry",
        "risk",
        "assigned_platform",
        "platform_type",
        "start_cell",
        "entry_cell",
        "path_length",
        "service_time",
        "completion_order",
        "executor",
        "score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in assignments:
            row = asdict(result)
            row["start_cell"] = list(result.start_cell)
            row["entry_cell"] = list(result.entry_cell)
            writer.writerow({field: row[field] for field in fieldnames})


def _summary(assignments: list[AssignmentResult]) -> dict[str, object]:
    platform_loads: dict[str, int] = {}
    platform_task_counts: dict[str, int] = {}
    risk_weighted_path = 0
    for result in assignments:
        platform_loads[result.assigned_platform] = platform_loads.get(result.assigned_platform, 0) + result.path_length + result.service_time
        platform_task_counts[result.assigned_platform] = platform_task_counts.get(result.assigned_platform, 0) + 1
        risk_weighted_path += result.risk * result.path_length
    return {
        "task_count": len(assignments),
        "high_risk_task_count": sum(1 for result in assignments if result.risk >= 3),
        "total_path_length": sum(result.path_length for result in assignments),
        "total_service_time": sum(result.service_time for result in assignments),
        "risk_weighted_path_length": risk_weighted_path,
        "platform_loads": platform_loads,
        "platform_task_counts": platform_task_counts,
    }


if __name__ == "__main__":
    main()
