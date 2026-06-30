from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


GridCell = tuple[int, int]

STAGE_SCREENING = "screening"
STAGE_REVIEW = "review"
STAGE_SERVICE = "service"

TASK_UNSCREENED = "unscreened"
TASK_RESERVED_SCREENING = "reserved_screening"
TASK_SCREENING = "screening"
TASK_SCREENED_PENDING = "screened_pending"
TASK_AWAITING_REVIEW = "awaiting_review"
TASK_RESERVED_REVIEW = "reserved_review"
TASK_REVIEWING = "reviewing"
TASK_CLOSED = "closed"

TASK_UNRELEASED = "UNRELEASED"
TASK_ACTIVE = "ACTIVE"
TASK_ASSIGNED = "ASSIGNED"
TASK_IN_SERVICE = "IN_SERVICE"
TASK_COMPLETED = "COMPLETED"
TASK_INTERRUPTED = "INTERRUPTED"
TASK_CANCELLED = "CANCELLED"
TASK_SUBSTITUTED = "SUBSTITUTED"

TASK_STATE_TO_CODE = {
    TASK_UNSCREENED: 0,
    TASK_RESERVED_SCREENING: 1,
    TASK_SCREENING: 2,
    TASK_SCREENED_PENDING: 3,
    TASK_AWAITING_REVIEW: 4,
    TASK_RESERVED_REVIEW: 5,
    TASK_REVIEWING: 6,
    TASK_CLOSED: 7,
    TASK_UNRELEASED: 10,
    TASK_ACTIVE: 11,
    TASK_ASSIGNED: 12,
    TASK_IN_SERVICE: 13,
    TASK_COMPLETED: 14,
    TASK_INTERRUPTED: 15,
    TASK_CANCELLED: 16,
    TASK_SUBSTITUTED: 17,
}
TASK_CODE_TO_STATE = {value: key for key, value in TASK_STATE_TO_CODE.items()}
LEGAL_TASK_TRANSITIONS = {
    TASK_UNSCREENED: {TASK_RESERVED_SCREENING},
    TASK_RESERVED_SCREENING: {TASK_SCREENING},
    TASK_SCREENING: {TASK_SCREENED_PENDING},
    TASK_SCREENED_PENDING: {TASK_AWAITING_REVIEW, TASK_CLOSED},
    TASK_AWAITING_REVIEW: {TASK_RESERVED_REVIEW},
    TASK_RESERVED_REVIEW: {TASK_REVIEWING},
    TASK_REVIEWING: {TASK_CLOSED},
    TASK_CLOSED: set(),
    TASK_UNRELEASED: {TASK_ACTIVE, TASK_CANCELLED},
    TASK_ACTIVE: {TASK_ASSIGNED, TASK_CANCELLED, TASK_SUBSTITUTED},
    TASK_ASSIGNED: {TASK_IN_SERVICE, TASK_ACTIVE, TASK_INTERRUPTED, TASK_CANCELLED},
    TASK_IN_SERVICE: {TASK_COMPLETED, TASK_INTERRUPTED, TASK_CANCELLED},
    TASK_COMPLETED: set(),
    TASK_INTERRUPTED: {TASK_ACTIVE, TASK_CANCELLED},
    TASK_CANCELLED: set(),
    TASK_SUBSTITUTED: set(),
}

MODE_IDLE = "idle"
MODE_TRAVEL = "travel"
MODE_SCREEN = "screen"
MODE_REVIEW = "review"
MODE_SERVICE = "service"
MODE_RETURN = "return"
MODE_REPLENISH = "replenish"


class TaskStateTransitionError(ValueError):
    pass


def task_state_code(state: str) -> int:
    if state not in TASK_STATE_TO_CODE:
        raise TaskStateTransitionError(f"unknown task state: {state}")
    return TASK_STATE_TO_CODE[state]


def transition_task_state(task: "InspectionTask", new_state: str) -> None:
    if new_state == task.state:
        return
    if new_state not in LEGAL_TASK_TRANSITIONS.get(task.state, set()):
        old_code = TASK_STATE_TO_CODE.get(task.state, "?")
        new_code = TASK_STATE_TO_CODE.get(new_state, "?")
        raise TaskStateTransitionError(
            f"illegal task transition for {task.task_id}: {task.state}({old_code}) -> {new_state}({new_code})"
        )
    task.state = new_state


@dataclass(slots=True)
class PortGridMap:
    name: str
    description: str
    width: int
    height: int
    cell_size_m: float
    depot: GridCell
    free_cells: tuple[GridCell, ...]
    obstacles: tuple[GridCell, ...]
    risk_grid: tuple[tuple[int, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def free_cell_set(self) -> set[GridCell]:
        return set(self.free_cells)

    @property
    def obstacle_set(self) -> set[GridCell]:
        return set(self.obstacles)

    def in_bounds(self, cell: GridCell) -> bool:
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, cell: GridCell) -> bool:
        return cell in self.free_cell_set

    def risk_at(self, cell: GridCell) -> int:
        row, col = cell
        return int(self.risk_grid[row][col])


@dataclass(slots=True)
class InspectionTask:
    task_id: str
    task_type: str
    geometry: str
    cells: tuple[GridCell, ...]
    risk: int
    service_time: int
    allowed_platforms: tuple[str, ...]
    max_interval: int = 20
    coverage_threshold: float = 1.0
    priority: float = 1.0
    uninspected_time: float = 0.0
    completed: bool = False
    task_family: str = ""
    geometry_mode: str = ""
    release_mode: str = "PERIODIC"
    required_work: float = 1.0
    completed_work: float = 0.0
    remaining_work: float = 1.0
    work_threshold: float = 1.0
    quality_pass: bool = True
    quality_requirement: dict[str, Any] = field(default_factory=dict)
    quality_acceptance_ref: str = ""
    executor: str = "rule_based"
    parent_task_id: str | None = None
    state: str = TASK_UNSCREENED
    screening_workload: float = 1.0
    review_workload: float = 1.0
    screening_workload_remaining: float = 1.0
    review_workload_remaining: float = 1.0
    screening_confidence: float = 0.0
    screening_uncertainty: float = 1.0
    screening_result: int | None = None
    review_required: bool = False
    review_result: int | None = None
    deadline: float | None = 0.0
    review_deadline: float = 0.0
    generation_time: float = 0.0
    screened_by: str | None = None
    screening_finish_time: float | None = None
    reviewed_by: str | None = None
    review_finish_time: float | None = None
    reserved_by: str | None = None
    reservation_time: float | None = None
    release_time: float | None = None
    close_time: float | None = None
    true_anomaly: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def entry_cell(self) -> GridCell:
        return self.cells[0]

    @property
    def exit_cell(self) -> GridCell:
        return self.cells[-1]

    @property
    def active_stage(self) -> str | None:
        if self.state in {TASK_ACTIVE, TASK_ASSIGNED, TASK_IN_SERVICE}:
            return STAGE_SERVICE
        if self.state in {TASK_UNSCREENED, TASK_RESERVED_SCREENING, TASK_SCREENING}:
            return STAGE_SCREENING
        if self.state in {TASK_AWAITING_REVIEW, TASK_RESERVED_REVIEW, TASK_REVIEWING}:
            return STAGE_REVIEW
        return None

    @property
    def is_closed(self) -> bool:
        return self.completed or self.state in {TASK_CLOSED, TASK_COMPLETED}

    @property
    def state_code(self) -> int:
        return task_state_code(self.state)


@dataclass(slots=True)
class Platform:
    platform_id: str
    platform_type: str
    current_cell: GridCell
    speed_mps: float
    endurance_minutes: float
    allowed_task_types: tuple[str, ...]
    preferred_task_types: tuple[str, ...] = ()
    max_speed_mps: float = 0.0
    nominal_endurance_minutes: float = 0.0
    return_reserve_ratio: float = 0.15
    sensor_radius_m: float = 0.0
    energy_capacity: float = 1.0
    energy: float = 1.0
    payload_kg: float = 0.0
    coverage_width_cells: int = 1
    energy_rate_per_cell: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    current_load: float = 0.0
    route: list[GridCell] = field(default_factory=list)
    mode: str = MODE_IDLE
    current_task_id: str | None = None
    current_stage: str | None = None
    target_cell: GridCell | None = None
    remaining_travel_time: float = 0.0
    remaining_service_time: float = 0.0
    remaining_replenish_time: float = 0.0
    depot_id: str = "DEPOT-0"
    alive: bool = True
    can_decide: bool = True

    def can_execute(self, task: InspectionTask, stage: str | None = None) -> bool:
        if task.geometry not in self.allowed_task_types:
            return False
        if stage == STAGE_SERVICE:
            return self.platform_type in task.allowed_platforms
        if stage == STAGE_SCREENING:
            return self.platform_type == "UAV"
        if stage == STAGE_REVIEW:
            return self.platform_type == "USV"
        return self.platform_type in task.allowed_platforms


@dataclass(slots=True)
class AssignmentResult:
    task_id: str
    task_type: str
    task_geometry: str
    risk: int
    assigned_platform: str
    platform_type: str
    start_cell: GridCell
    entry_cell: GridCell
    exit_cell: GridCell
    path_length: int
    service_time: int
    completion_order: int
    executor: str
    score: float
    path: tuple[GridCell, ...]


@dataclass(slots=True)
class CandidateEntry:
    task_id: str
    task_index: int
    task_stage: str
    relative_position: int
    risk: float
    urgency: float
    confidence: float
    review_waiting_time: float
    estimated_arrival_time: float
    estimated_energy: float
    capability_match: float
    reservation_status: float
    feasible: bool
    score: float
    relative_row: float = 0.0
    relative_col: float = 0.0
    task_geometry_code: float = 0.0
    estimated_finish_time: float = 0.0
