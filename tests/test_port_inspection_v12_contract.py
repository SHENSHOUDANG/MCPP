from __future__ import annotations

from math import isinf
from pathlib import Path
import sys
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.port_inspection.v12_contract import (
    ContractValidationError,
    best_case_slack,
    classify_config_boundary,
    deadline_metrics,
    require_historical_baseline_ack,
    revisit_metrics,
    transition_allowed,
    validate_v12_task_record,
)


class V12ContractTests(unittest.TestCase):
    def test_deadline_null_metrics_remain_null(self) -> None:
        metrics = deadline_metrics(
            current_time=30.0,
            deadline=None,
            estimated_travel_time=5.0,
            estimated_remaining_service_time=10.0,
            completion_time=45.0,
        )
        self.assertIsNone(metrics.slack)
        self.assertIsNone(metrics.overdue)
        self.assertIsNone(metrics.lateness)

    def test_revisit_null_metrics_remain_null(self) -> None:
        metrics = revisit_metrics(
            current_time=30.0,
            max_revisit_interval=None,
            last_completion_time=None,
        )
        self.assertIsNone(metrics.revisit_age)
        self.assertIsNone(metrics.revisit_violation)

    def test_revisit_requires_initialized_history(self) -> None:
        with self.assertRaises(ContractValidationError):
            revisit_metrics(
                current_time=30.0,
                max_revisit_interval=12.0,
                last_completion_time=None,
            )

    def test_best_case_slack_respects_no_feasible_platforms(self) -> None:
        self.assertEqual(best_case_slack([2.0, -1.0], has_deadline=True), 2.0)
        self.assertTrue(isinf(best_case_slack([], has_deadline=True)))
        self.assertIsNone(best_case_slack([], has_deadline=False))

    def test_v12_state_transitions_reject_overdue_state(self) -> None:
        self.assertTrue(transition_allowed("ACTIVE", "ASSIGNED"))
        with self.assertRaises(ContractValidationError):
            transition_allowed("ACTIVE", "OVERDUE")

    def test_yangshan_config_is_historical_only(self) -> None:
        config = _load_config(ROOT / "configs" / "port_yangshan_task_initial_v1.toml")
        boundary = classify_config_boundary(config)
        self.assertEqual(boundary.scenario_status, "HISTORICAL")
        self.assertTrue(boundary.historical_only)
        self.assertFalse(boundary.final_experiment_eligible)
        with self.assertRaises(ContractValidationError):
            require_historical_baseline_ack(boundary, False, purpose="test training")

    def test_los_angeles_config_is_pending_training_prototype(self) -> None:
        config = _load_config(ROOT / "configs" / "port_los_angeles_training_v1.toml")
        boundary = classify_config_boundary(config)
        self.assertEqual(boundary.scenario_status, "PENDING")
        self.assertFalse(boundary.historical_only)
        self.assertFalse(boundary.final_experiment_eligible)
        require_historical_baseline_ack(boundary, False, purpose="test training")

    def test_validate_periodic_task_record(self) -> None:
        validate_v12_task_record(_task_record())

    def test_validate_rejects_overdue_as_task_status(self) -> None:
        record = _task_record()
        record["status"] = "OVERDUE"
        with self.assertRaises(ContractValidationError):
            validate_v12_task_record(record)

    def test_validate_rejects_nonperiodic_period_fields(self) -> None:
        record = _task_record()
        record["release_mode"] = "EVENT"
        with self.assertRaises(ContractValidationError):
            validate_v12_task_record(record)


def _task_record() -> dict[str, object]:
    return {
        "task_id": "T-001",
        "parent_object_id": "OBJ-001",
        "task_family": "HYDROGRAPHIC_SURVEY",
        "object_type": "channel",
        "geometry_mode": "CORRIDOR",
        "geometry_ref": "geometries/channel-001",
        "execution_template_ref": "templates/hydrographic_corridor_v1",
        "release_mode": "PERIODIC",
        "release_time": 0.0,
        "importance_class": "scenario_medium",
        "hard_capability_requirement": {"sonar": 1},
        "required_work": 100.0,
        "completed_work": 0.0,
        "remaining_work": 100.0,
        "estimated_remaining_service_time_by_platform": {"USV": 30.0},
        "work_threshold": 1.0,
        "quality_requirement": {"coverage_rate": 0.95},
        "quality_acceptance_ref": "hydrographic_quality_v1",
        "deadline": None,
        "service_window_start": None,
        "service_window_end": None,
        "max_revisit_interval": 1440.0,
        "last_completion_time": None,
        "next_due_time": 0.0,
        "period_interval": 1440.0,
        "calendar_anchor": 0.0,
        "calendar_update_mode": "FIXED_CALENDAR",
        "revisit_initialization_mode": "INITIAL_INSPECTION_REQUIRED",
        "revisit_initialization_time": None,
        "obligation_level": "PENALIZED",
        "parent_task_id": None,
        "predecessor_ids": [],
        "trigger_rule": None,
        "substitution_set_id": None,
        "status": "ACTIVE",
        "status_history": [],
        "provenance": {},
        "scenario_generated": True,
    }


def _load_config(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


if __name__ == "__main__":
    unittest.main()
