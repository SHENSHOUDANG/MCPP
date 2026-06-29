from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Mapping, Sequence


V12_CONTRACT_VERSION = "V1.2"

TASK_FAMILIES = frozenset(
    {
        "HYDROGRAPHIC_SURVEY",
        "SURFACE_SAFETY_PATROL",
        "WATERSIDE_ASSET_INSPECTION",
    }
)
GEOMETRY_MODES = frozenset({"TARGET", "CORRIDOR", "AREA"})
RELEASE_MODES = frozenset({"PERIODIC", "SCHEDULED", "EVENT"})
OBLIGATION_LEVELS = frozenset({"MANDATORY", "PENALIZED", "OPTIONAL"})
TASK_STATES = frozenset(
    {
        "UNRELEASED",
        "ACTIVE",
        "ASSIGNED",
        "IN_SERVICE",
        "INTERRUPTED",
        "COMPLETED",
        "CANCELLED",
        "SUBSTITUTED",
    }
)
LEGAL_TASK_TRANSITIONS = {
    "UNRELEASED": frozenset({"ACTIVE"}),
    "ACTIVE": frozenset({"ASSIGNED", "CANCELLED", "SUBSTITUTED"}),
    "ASSIGNED": frozenset({"IN_SERVICE", "ACTIVE", "CANCELLED"}),
    "IN_SERVICE": frozenset({"COMPLETED", "INTERRUPTED"}),
    "INTERRUPTED": frozenset({"ASSIGNED", "ACTIVE", "CANCELLED", "SUBSTITUTED"}),
    "COMPLETED": frozenset(),
    "CANCELLED": frozenset(),
    "SUBSTITUTED": frozenset(),
}
PLATFORM_STATUSES = frozenset(
    {
        "AVAILABLE",
        "TRAVELING",
        "SETUP",
        "IN_SERVICE",
        "WAITING",
        "RETURNING",
        "REPLENISHING",
        "UNAVAILABLE",
    }
)
CALENDAR_UPDATE_MODES = frozenset({"ACTUAL_COMPLETION", "FIXED_CALENDAR"})
REVISIT_INITIALIZATION_MODES = frozenset(
    {
        "TRUSTED_HISTORY",
        "COMMISSIONING_TIME",
        "STUDY_START",
        "INITIAL_INSPECTION_REQUIRED",
    }
)
REQUIRED_PROVENANCE_FIELDS = (
    "source_dataset",
    "source_agency",
    "source_date",
    "source_url",
    "source_version_or_edition",
    "access_date",
    "license_or_usage_terms",
    "original_id",
    "original_crs",
    "file_checksum",
    "processing_script_version",
    "processing_note",
)
REQUIRED_TASK_FIELDS = (
    "task_id",
    "parent_object_id",
    "task_family",
    "object_type",
    "geometry_mode",
    "geometry_ref",
    "execution_template_ref",
    "release_mode",
    "release_time",
    "importance_class",
    "hard_capability_requirement",
    "required_work",
    "completed_work",
    "remaining_work",
    "estimated_remaining_service_time_by_platform",
    "work_threshold",
    "quality_requirement",
    "quality_acceptance_ref",
    "deadline",
    "service_window_start",
    "service_window_end",
    "max_revisit_interval",
    "last_completion_time",
    "next_due_time",
    "period_interval",
    "calendar_anchor",
    "calendar_update_mode",
    "revisit_initialization_mode",
    "revisit_initialization_time",
    "obligation_level",
    "parent_task_id",
    "predecessor_ids",
    "trigger_rule",
    "substitution_set_id",
    "status",
    "status_history",
    "provenance",
    "scenario_generated",
)

HISTORICAL_BASELINE_NOTICE = (
    "This scenario is retained only as a historical engineering baseline under "
    "the V1.2 contract. Do not use it as final real-port evidence."
)


class ContractValidationError(ValueError):
    """Raised when a record or run request violates the frozen V1.2 contract."""


@dataclass(frozen=True, slots=True)
class ContractBoundary:
    model_contract_version: str
    scenario_status: str
    historical_only: bool
    final_experiment_eligible: bool
    notice: str
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_contract_version": self.model_contract_version,
            "scenario_status": self.scenario_status,
            "historical_only": self.historical_only,
            "final_experiment_eligible": self.final_experiment_eligible,
            "notice": self.notice,
            "blocking_reasons": list(self.blocking_reasons),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class DeadlineMetrics:
    slack: float | None
    overdue: float | None
    lateness: float | None


@dataclass(frozen=True, slots=True)
class RevisitMetrics:
    revisit_age: float | None
    revisit_violation: float | None


def classify_config_boundary(config: Mapping[str, Any]) -> ContractBoundary:
    contract = _mapping(config.get("contract", {}))
    version = str(contract.get("model_contract_version", "unspecified"))
    status = str(contract.get("scenario_status", "PENDING")).upper()
    looks_legacy = _looks_like_legacy_yangshan_config(config)
    historical_only = _as_bool(contract.get("historical_only", status == "HISTORICAL" or looks_legacy))
    final_eligible = _as_bool(
        contract.get("final_experiment_eligible", version == V12_CONTRACT_VERSION and not historical_only)
    )

    blocking: list[str] = []
    warnings: list[str] = []
    if historical_only:
        final_eligible = False
        if status != "HISTORICAL":
            warnings.append("historical_only=true coerces scenario_status to HISTORICAL")
            status = "HISTORICAL"
        blocking.append("scenario is marked as historical-only under V1.2")
    if version != V12_CONTRACT_VERSION:
        warnings.append(f"model_contract_version is {version!r}, not {V12_CONTRACT_VERSION}")
        if final_eligible:
            final_eligible = False
            blocking.append("final experiments require a V1.2 model contract")
    if final_eligible and status != "ACTIVE":
        final_eligible = False
        blocking.append("final experiments require scenario_status=ACTIVE")

    notice = str(contract.get("notice", ""))
    if not notice and historical_only:
        notice = HISTORICAL_BASELINE_NOTICE

    return ContractBoundary(
        model_contract_version=version,
        scenario_status=status,
        historical_only=historical_only,
        final_experiment_eligible=final_eligible,
        notice=notice,
        blocking_reasons=tuple(blocking),
        warnings=tuple(warnings),
    )


def require_historical_baseline_ack(
    boundary_or_config: ContractBoundary | Mapping[str, Any],
    acknowledged: bool,
    *,
    purpose: str,
) -> None:
    boundary = (
        boundary_or_config
        if isinstance(boundary_or_config, ContractBoundary)
        else classify_config_boundary(boundary_or_config)
    )
    if boundary.historical_only and not acknowledged:
        raise ContractValidationError(
            f"{purpose} targets a historical pre-V1.2 scenario. "
            "Pass --allow-historical-baseline only for engineering baselines; "
            "do not report the result as a final V1.2 experiment."
        )


def deadline_metrics(
    *,
    current_time: float,
    deadline: float | None,
    estimated_travel_time: float,
    estimated_remaining_service_time: float,
    completion_time: float | None = None,
) -> DeadlineMetrics:
    if deadline is None:
        return DeadlineMetrics(slack=None, overdue=None, lateness=None)
    deadline_value = float(deadline)
    current = float(current_time)
    slack = deadline_value - current - float(estimated_travel_time) - float(estimated_remaining_service_time)
    overdue = max(0.0, current - deadline_value)
    lateness = None if completion_time is None else max(0.0, float(completion_time) - deadline_value)
    return DeadlineMetrics(slack=slack, overdue=overdue, lateness=lateness)


def best_case_slack(slacks: Sequence[float], *, has_deadline: bool) -> float | None:
    if not has_deadline:
        return None
    if not slacks:
        return -inf
    return max(float(value) for value in slacks)


def revisit_metrics(
    *,
    current_time: float,
    max_revisit_interval: float | None,
    last_completion_time: float | None,
) -> RevisitMetrics:
    if max_revisit_interval is None:
        return RevisitMetrics(revisit_age=None, revisit_violation=None)
    if last_completion_time is None:
        raise ContractValidationError(
            "last_completion_time=null requires an explicit revisit initialization mode before "
            "revisit_age can be computed"
        )
    age = float(current_time) - float(last_completion_time)
    return RevisitMetrics(
        revisit_age=age,
        revisit_violation=max(0.0, age - float(max_revisit_interval)),
    )


def transition_allowed(current_state: str, next_state: str) -> bool:
    if current_state not in LEGAL_TASK_TRANSITIONS:
        raise ContractValidationError(f"unknown V1.2 task state: {current_state}")
    if next_state not in TASK_STATES:
        raise ContractValidationError(f"unknown V1.2 task state: {next_state}")
    return next_state == current_state or next_state in LEGAL_TASK_TRANSITIONS[current_state]


def validate_v12_task_record(record: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_TASK_FIELDS if field not in record]
    if missing:
        raise ContractValidationError(f"task record missing required fields: {', '.join(missing)}")

    _require_member(record, "task_family", TASK_FAMILIES)
    _require_member(record, "geometry_mode", GEOMETRY_MODES)
    _require_member(record, "release_mode", RELEASE_MODES)
    _require_member(record, "obligation_level", OBLIGATION_LEVELS)
    _require_member(record, "status", TASK_STATES)

    for field in (
        "release_time",
        "required_work",
        "completed_work",
        "remaining_work",
        "work_threshold",
        "deadline",
        "service_window_start",
        "service_window_end",
        "max_revisit_interval",
        "last_completion_time",
        "next_due_time",
        "period_interval",
        "calendar_anchor",
        "revisit_initialization_time",
    ):
        _require_number_or_none(record, field)

    if float(record["required_work"]) <= 0.0:
        raise ContractValidationError("required_work must be positive")
    if not isinstance(record["quality_requirement"], Mapping):
        raise ContractValidationError("quality_requirement must be an object")
    if not str(record["quality_acceptance_ref"]).strip():
        raise ContractValidationError("quality_acceptance_ref is required")
    if not str(record["geometry_ref"]).strip():
        raise ContractValidationError("geometry_ref is required and cannot be replaced by a centroid feature")

    release_mode = str(record["release_mode"])
    if release_mode == "PERIODIC":
        for field in ("period_interval", "next_due_time", "calendar_update_mode"):
            if record.get(field) is None:
                raise ContractValidationError(f"periodic task requires {field}")
        _require_member(record, "calendar_update_mode", CALENDAR_UPDATE_MODES)
        if record["calendar_update_mode"] == "FIXED_CALENDAR" and record.get("calendar_anchor") is None:
            raise ContractValidationError("FIXED_CALENDAR periodic tasks require calendar_anchor")
    else:
        for field in ("period_interval", "next_due_time", "calendar_anchor", "calendar_update_mode"):
            if record.get(field) is not None:
                raise ContractValidationError(f"non-periodic task must keep {field}=null")

    if record.get("max_revisit_interval") is not None and record.get("last_completion_time") is None:
        _require_member(record, "revisit_initialization_mode", REVISIT_INITIALIZATION_MODES)
        if (
            record["revisit_initialization_mode"] != "INITIAL_INSPECTION_REQUIRED"
            and record.get("revisit_initialization_time") is None
        ):
            raise ContractValidationError(
                "revisit initialization modes other than INITIAL_INSPECTION_REQUIRED "
                "require revisit_initialization_time"
            )

    if not isinstance(record["scenario_generated"], bool):
        raise ContractValidationError("scenario_generated must be a bool")
    provenance = record["provenance"]
    if not isinstance(provenance, Mapping):
        raise ContractValidationError("provenance must be an object")
    if not record["scenario_generated"]:
        missing_provenance = [field for field in REQUIRED_PROVENANCE_FIELDS if field not in provenance]
        if missing_provenance:
            raise ContractValidationError(
                "formal task provenance missing fields: " + ", ".join(missing_provenance)
            )


def _require_member(record: Mapping[str, Any], field: str, allowed: frozenset[str]) -> None:
    value = record.get(field)
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ContractValidationError(f"{field} must be one of: {allowed_text}; got {value!r}")


def _require_number_or_none(record: Mapping[str, Any], field: str) -> None:
    value = record[field]
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field} must be numeric or null")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _looks_like_legacy_yangshan_config(config: Mapping[str, Any]) -> bool:
    paths = " ".join(str(config.get(key, "")) for key in ("grid_path", "tasks_path", "output_dir"))
    return "yangshan_task_initial_v1" in paths or "review_trigger" in config
