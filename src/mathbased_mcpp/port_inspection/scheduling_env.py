from __future__ import annotations

import copy
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

from .path_proxy import TaskCost, estimate_task_cost
from .reward import compute_reward_terms
from .risk import late_task_count, total_risk_exposure, update_uninspected_time
from .schema import (
    CandidateEntry,
    GridCell,
    InspectionTask,
    MODE_IDLE,
    MODE_REPLENISH,
    MODE_RETURN,
    MODE_REVIEW,
    MODE_TRAVEL,
    MODE_SCREEN,
    Platform,
    PortGridMap,
    STAGE_REVIEW,
    STAGE_SCREENING,
    TASK_AWAITING_REVIEW,
    TASK_CLOSED,
    TASK_RESERVED_REVIEW,
    TASK_RESERVED_SCREENING,
    TASK_REVIEWING,
    TASK_SCREENED_PENDING,
    TASK_SCREENING,
    TASK_UNSCREENED,
    transition_task_state,
)
from .simple_planner import shortest_path


@dataclass(slots=True)
class SchedulingStepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass(slots=True)
class SchedulingModelReset:
    obs_dict: dict[str, np.ndarray]
    state: np.ndarray
    available_actions: dict[str, np.ndarray]
    info: dict[str, Any]


@dataclass(slots=True)
class SchedulingModelStep:
    obs_dict: dict[str, np.ndarray]
    state: np.ndarray
    rewards: dict[str, float]
    terminated: bool
    truncated: bool
    available_actions: dict[str, np.ndarray]
    info: dict[str, Any]


@dataclass(slots=True)
class _Proposal:
    platform_index: int
    choice: int
    candidate: CandidateEntry
    cost: TaskCost


CANDIDATE_FEATURE_DIM = 8


class PortInspectionSchedulingEnv:
    """High-level UAV screening / USV review scheduling environment.

    The environment keeps movement and service completion event-based for the
    first coupled baseline: one accepted high-level action completes one task
    stage. The business state machine, candidate masks, dynamic review queue,
    conflict arbitration, and reward accounting follow the V1.0 model.
    """

    def __init__(
        self,
        grid: PortGridMap,
        tasks: list[InspectionTask],
        platforms: list[Platform],
        max_steps: int = 64,
        reward_weights: dict[str, float] | None = None,
        candidate_k: int = 8,
        candidate_weights: dict[str, float] | None = None,
        review_trigger: dict[str, Any] | None = None,
    ) -> None:
        self.grid = grid
        self.base_tasks = copy.deepcopy(tasks)
        self.base_platforms = copy.deepcopy(platforms)
        self.max_steps = max(int(max_steps), 1)
        self.reward_weights = reward_weights or {}
        self.candidate_k = max(int(candidate_k), 1)
        self.candidate_weights = {
            "risk_weight": 10.0,
            "urgency_weight": 6.0,
            "review_wait_weight": 0.5,
            "distance_weight": 0.05,
        }
        if candidate_weights:
            self.candidate_weights.update({key: float(value) for key, value in candidate_weights.items()})
        self.review_trigger = {
            "confidence_threshold": 0.65,
            "mandatory_review_risk": 3,
            "base_review_deadline": 36.0,
            "risk_deadline_scale": 4.0,
            "confidence_deadline_scale": 3.0,
            "sensitivity": 0.85,
            "specificity": 0.80,
            "confidence_noise": 0.08,
            "anomaly_probability_by_risk": {1: 0.10, 2: 0.25, 3: 0.45},
            "replenish_steps": 2,
            "idle_energy_cost": 0.0,
        }
        if review_trigger:
            self.review_trigger.update(review_trigger)
        self.depot_capacity = max(len(platforms), 1)
        self.tasks: list[InspectionTask] = []
        self.platforms: list[Platform] = []
        self.current_step = 0
        self.total_path_length = 0
        self.total_energy = 0.0
        self.total_conflicts = 0
        self.total_invalid_actions = 0
        self.total_replenishments = 0
        self.total_returns = 0
        self.total_screened = 0
        self.total_reviewed = 0
        self.total_anomaly_closed = 0
        self.total_anomaly_missed = 0
        self.completed_tasks: set[str] = set()
        self.last_reward_terms: dict[str, float] = {}
        self.last_conflicts: list[dict[str, Any]] = []
        self.last_accepted: list[dict[str, Any]] = []
        self.rng = np.random.default_rng()
        self._cost_cache: dict[tuple[Any, ...], TaskCost] = {}
        self.reset()

    @property
    def num_platforms(self) -> int:
        return len(self.platforms)

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def action_choices(self) -> int:
        return self.candidate_k + 3

    @property
    def return_action(self) -> int:
        return self.candidate_k + 2

    @property
    def continue_action(self) -> int:
        return self.candidate_k

    @property
    def wait_action(self) -> int:
        return self.candidate_k + 1

    @property
    def action_dim(self) -> int:
        return self.num_platforms * self.action_choices

    @property
    def observation_dim(self) -> int:
        return int(self.observation().shape[0])

    @property
    def local_observation_dim(self) -> int:
        observations = self.local_observations()
        if not observations:
            return 0
        return int(next(iter(observations.values())).shape[0])

    @property
    def global_state_dim(self) -> int:
        return int(self.global_state().shape[0])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            np.random.seed(seed)
        self.tasks = copy.deepcopy(self.base_tasks)
        self.platforms = copy.deepcopy(self.base_platforms)
        for platform in self.platforms:
            platform.current_cell = self._platform_depot(platform)
            platform.energy = platform.energy_capacity
            platform.current_load = 0.0
            platform.route = [platform.current_cell]
            platform.mode = MODE_IDLE
            platform.current_task_id = None
            platform.current_stage = None
            platform.target_cell = None
            platform.remaining_travel_time = 0.0
            platform.remaining_service_time = 0.0
            platform.remaining_replenish_time = 0.0
            platform.alive = True
            platform.can_decide = True
            platform.metadata.pop("pending_cost", None)
        for task in self.tasks:
            task.completed = False
            task.state = TASK_UNSCREENED
            task.uninspected_time = float(task.metadata.get("initial_uninspected_time", 0.0))
            task.screening_workload_remaining = float(task.screening_workload)
            task.review_workload_remaining = float(task.review_workload)
            task.screening_confidence = 0.0
            task.screening_uncertainty = 1.0
            task.screening_result = None
            task.review_required = False
            task.review_result = None
            task.review_deadline = 0.0
            task.generation_time = 0.0
            task.screened_by = None
            task.screening_finish_time = None
            task.reviewed_by = None
            task.review_finish_time = None
            task.reserved_by = None
            task.reservation_time = None
            task.release_time = None
            task.close_time = None
            task.true_anomaly = self._sample_true_anomaly(task)
        self.current_step = 0
        self.total_path_length = 0
        self.total_energy = 0.0
        self.total_conflicts = 0
        self.total_invalid_actions = 0
        self.total_replenishments = 0
        self.total_returns = 0
        self.total_screened = 0
        self.total_reviewed = 0
        self.total_anomaly_closed = 0
        self.total_anomaly_missed = 0
        self.completed_tasks = set()
        self.last_reward_terms = {}
        self.last_conflicts = []
        self.last_accepted = []
        self._cost_cache = {}
        return self.observation()

    def reset_model(self, seed: int | None = None) -> SchedulingModelReset:
        self.reset(seed=seed)
        return SchedulingModelReset(
            obs_dict=self.local_observations(),
            state=self.global_state(),
            available_actions=self.available_actions(),
            info=self.info(),
        )

    def step(self, action: int | list[int] | tuple[int, ...] | np.ndarray) -> SchedulingStepResult:
        self.current_step += 1
        self.last_conflicts = []
        self.last_accepted = []
        invalid = False
        invalid_count = 0

        candidate_sets = self.candidate_lists()
        masks = self.action_masks(candidate_sets)
        proposals: list[_Proposal] = []
        chosen_pairs = self._decode_actions(action)
        choices_by_platform = {platform_index: choice for platform_index, choice in chosen_pairs}

        for platform_index, platform in enumerate(self.platforms):
            choice = choices_by_platform.get(platform_index, self.continue_action if platform.mode != MODE_IDLE else self.wait_action)
            if platform_index < 0 or platform_index >= self.num_platforms:
                invalid = True
                invalid_count += 1
                continue
            if choice < 0 or choice >= self.action_choices:
                invalid = True
                invalid_count += 1
                continue

            if platform.mode != MODE_IDLE:
                if choice != self.continue_action:
                    invalid = True
                    invalid_count += 1
                continue

            if not masks[platform_index, choice]:
                invalid = True
                invalid_count += 1
                continue

            if choice == self.wait_action:
                continue
            if choice == self.return_action:
                self._start_return(platform_index)
                continue
            if choice == self.continue_action:
                invalid = True
                invalid_count += 1
                continue

            candidate_index = choice
            candidates = candidate_sets[platform_index]
            if candidate_index >= len(candidates):
                invalid = True
                invalid_count += 1
                continue
            candidate = candidates[candidate_index]
            cost = self._estimate_cost(platform_index, candidate.task_index, candidate.task_stage)
            proposals.append(_Proposal(platform_index, choice, candidate, cost))

        accepted = self._resolve_conflicts(proposals)
        conflict_count = max(len(proposals) - len(accepted), 0)
        self.total_conflicts += conflict_count
        self.total_invalid_actions += invalid_count

        for proposal in accepted:
            platform = self.platforms[proposal.platform_index]
            task = self.tasks[proposal.candidate.task_index]
            self._start_task_stage(task, platform, proposal.candidate.task_stage, proposal.cost)
            self.last_accepted.append(
                {
                    "platform_id": platform.platform_id,
                    "task_id": task.task_id,
                    "stage": proposal.candidate.task_stage,
                    "score": proposal.candidate.score,
                }
            )

        progress = self._advance_platforms()
        update_uninspected_time(self.tasks, {task.task_id for task in progress["closed_tasks"]}, delta_t=1.0)
        review_queue_length = self.review_queue_length()
        terms = compute_reward_terms(
            self.tasks,
            self.platforms,
            completed_task=None,
            path_length=int(progress["path_length"]),
            energy_cost=float(progress["energy_cost"]),
            invalid=invalid,
            weights=self.reward_weights,
            screened_tasks=progress["screened_tasks"],
            reviewed_tasks=progress["reviewed_tasks"],
            closed_tasks=progress["closed_tasks"],
            conflict_count=conflict_count + invalid_count,
            review_queue_length=review_queue_length,
        )
        self.last_reward_terms = terms
        self._cost_cache = {}
        done = len(self.completed_tasks) == len(self.tasks) or self.current_step >= self.max_steps
        return SchedulingStepResult(self.observation(), float(terms["total"]), done, self.info())

    def step_model(self, actions: dict[str, int] | list[int] | tuple[int, ...] | np.ndarray | int) -> SchedulingModelStep:
        result = self.step(self._actions_from_mapping(actions))
        terminated = len(self.completed_tasks) == len(self.tasks)
        truncated = self.current_step >= self.max_steps and not terminated
        return SchedulingModelStep(
            obs_dict=self.local_observations(),
            state=self.global_state(),
            rewards={platform.platform_id: float(result.reward) for platform in self.platforms},
            terminated=terminated,
            truncated=truncated,
            available_actions=self.available_actions(),
            info=result.info,
        )

    def candidate_lists(self) -> list[list[CandidateEntry]]:
        return [self._build_candidates(index) for index in range(self.num_platforms)]

    def action_masks(self, candidate_sets: list[list[CandidateEntry]] | None = None) -> np.ndarray:
        candidate_sets = candidate_sets if candidate_sets is not None else self.candidate_lists()
        masks = np.zeros((self.num_platforms, self.action_choices), dtype=bool)
        for platform_index, candidates in enumerate(candidate_sets):
            platform = self.platforms[platform_index]
            if platform.mode != MODE_IDLE:
                masks[platform_index, self.continue_action] = True
                continue
            for candidate in candidates[: self.candidate_k]:
                masks[platform_index, candidate.relative_position] = candidate.feasible
            masks[platform_index, self.wait_action] = True
            depot = self._platform_depot(platform)
            masks[platform_index, self.return_action] = platform.current_cell != depot
            if not any(candidate.feasible for candidate in candidates) and self._must_return(platform):
                masks[platform_index, self.return_action] = platform.current_cell != depot
        return masks

    def flat_action_mask(self) -> np.ndarray:
        return self.action_masks().reshape(-1)

    def available_actions(self, candidate_sets: list[list[CandidateEntry]] | None = None) -> dict[str, np.ndarray]:
        masks = self.action_masks(candidate_sets)
        return {
            platform.platform_id: masks[index].astype(np.float32)
            for index, platform in enumerate(self.platforms)
        }

    def local_observations(self) -> dict[str, np.ndarray]:
        candidate_sets = self.candidate_lists()
        masks = self.action_masks(candidate_sets)
        broadcast = self.aggregate_broadcast()
        observations: dict[str, np.ndarray] = {}
        for platform_index, platform in enumerate(self.platforms):
            values = []
            values.extend(self._platform_features(platform))
            values.extend(broadcast.tolist())
            candidates = candidate_sets[platform_index]
            for slot in range(self.candidate_k):
                if slot < len(candidates):
                    values.extend(self._candidate_features(candidates[slot]))
                else:
                    values.extend([0.0] * CANDIDATE_FEATURE_DIM)
            values.extend(masks[platform_index].astype(np.float32).tolist())
            observations[platform.platform_id] = np.asarray(values, dtype=np.float32)
        return observations

    def global_state(self) -> np.ndarray:
        values: list[float] = []
        values.extend(self.aggregate_broadcast().tolist())
        for platform in self.platforms:
            values.extend(self._platform_features(platform))
        for task in self.tasks:
            values.extend(self._task_features(task))
        metrics = self.metrics()
        values.extend(
            [
                float(metrics["completion_ratio"]),
                float(metrics["late_task_fraction"]),
                float(metrics["risk_exposure_sum"]) / max(len(self.tasks), 1) / 40.0,
                float(metrics["total_conflicts"]) / max(self.current_step, 1),
                float(metrics["total_invalid_actions"]) / max(self.current_step, 1),
                float(metrics["total_replenishments"]) / max(self.current_step, 1),
                float(metrics["total_returns"]) / max(self.current_step, 1),
                float(metrics["screened_tasks"]) / max(len(self.tasks), 1),
                float(metrics["reviewed_tasks"]) / max(len(self.tasks), 1),
                float(metrics["anomaly_closed"]) / max(len(self.tasks), 1),
                float(metrics["anomaly_missed"]) / max(len(self.tasks), 1),
            ]
        )
        return np.asarray(values, dtype=np.float32)

    def observation(self) -> np.ndarray:
        candidate_sets = self.candidate_lists()
        masks = self.action_masks(candidate_sets)
        values: list[float] = []
        height = max(self.grid.height - 1, 1)
        width = max(self.grid.width - 1, 1)
        for platform_index, platform in enumerate(self.platforms):
            values.extend(
                [
                    1.0 if platform.platform_type == "UAV" else -1.0,
                    platform.current_cell[0] / height,
                    platform.current_cell[1] / width,
                    platform.energy / max(platform.energy_capacity, 1e-6),
                    platform.current_load / 500.0,
                    platform.speed_mps / 25.0,
                    platform.endurance_minutes / 360.0,
                    platform.return_reserve_ratio,
                    1.0 if platform.can_decide else 0.0,
                    self._mode_code(platform.mode),
                ]
            )
            candidates = candidate_sets[platform_index]
            for slot in range(self.candidate_k):
                if slot < len(candidates):
                    values.extend(self._candidate_features(candidates[slot]))
                else:
                    values.extend([0.0] * CANDIDATE_FEATURE_DIM)
        values.extend(masks.astype(np.float32).ravel().tolist())
        values.extend(
            [
                self.current_step / self.max_steps,
                len(self.completed_tasks) / max(len(self.tasks), 1),
                self.review_queue_length() / max(len(self.tasks), 1),
                self.screening_open_count() / max(len(self.tasks), 1),
                total_risk_exposure(self.tasks) / max(len(self.tasks), 1) / 40.0,
                late_task_count(self.tasks) / max(len(self.tasks), 1),
                self.total_path_length / 2000.0,
                self.total_energy,
            ]
        )
        return np.asarray(values, dtype=np.float32)

    def info(self) -> dict[str, Any]:
        candidate_sets = self.candidate_lists()
        return {
            "reward_terms": dict(self.last_reward_terms),
            "completed_tasks": sorted(self.completed_tasks),
            "closed_tasks": sorted(self.completed_tasks),
            "screening_open_tasks": [task.task_id for task in self.tasks if task.state == TASK_UNSCREENED],
            "review_queue_tasks": [task.task_id for task in self.tasks if task.state == TASK_AWAITING_REVIEW],
            "review_queue_length": self.review_queue_length(),
            "late_tasks": [task.task_id for task in self.tasks if not task.completed and task.uninspected_time > task.max_interval],
            "task_states": {task.task_id: task.state for task in self.tasks},
            "task_confidence": {task.task_id: task.screening_confidence for task in self.tasks if task.screened_by},
            "total_path_length": self.total_path_length,
            "total_energy": self.total_energy,
            "risk_exposure_sum": total_risk_exposure(self.tasks),
            "metrics": self.metrics(),
            "platform_loads": {platform.platform_id: platform.current_load for platform in self.platforms},
            "action_masks": self.action_masks(candidate_sets).astype(int).tolist(),
            "candidate_mask": self.action_masks(candidate_sets)[:, : self.candidate_k].astype(int).tolist(),
            "agent_mask": [1 if platform.alive else 0 for platform in self.platforms],
            "alive_mask": [1 if platform.alive else 0 for platform in self.platforms],
            "available_actions": {
                platform_id: mask.astype(int).tolist()
                for platform_id, mask in self.available_actions(candidate_sets).items()
            },
            "aggregate_broadcast": self.aggregate_broadcast().tolist(),
            "local_observation_dim": self.local_observation_dim,
            "global_state_dim": self.global_state_dim,
            "candidate_task_ids": [[candidate.task_id for candidate in candidates] for candidates in candidate_sets],
            "candidate_details": [[self._candidate_info(candidate) for candidate in candidates] for candidates in candidate_sets],
            "accepted_actions": list(self.last_accepted),
            "conflicts": list(self.last_conflicts),
        }

    def review_queue_length(self) -> int:
        return sum(1 for task in self.tasks if task.state == TASK_AWAITING_REVIEW)

    def screening_open_count(self) -> int:
        return sum(1 for task in self.tasks if task.state == TASK_UNSCREENED)

    def review_wait_stats(self) -> tuple[float, float]:
        waits = [self._review_waiting_time(task) for task in self.tasks if task.state == TASK_AWAITING_REVIEW]
        if not waits:
            return 0.0, 0.0
        return float(np.mean(waits)), float(max(waits))

    def aggregate_broadcast(self) -> np.ndarray:
        uav_idle = sum(1 for platform in self.platforms if platform.platform_type == "UAV" and platform.mode == MODE_IDLE)
        usv_idle = sum(1 for platform in self.platforms if platform.platform_type == "USV" and platform.mode == MODE_IDLE)
        uav_count = max(sum(1 for platform in self.platforms if platform.platform_type == "UAV"), 1)
        usv_count = max(sum(1 for platform in self.platforms if platform.platform_type == "USV"), 1)
        values = [
            self.screening_open_count() / max(len(self.tasks), 1),
            self.review_queue_length() / max(len(self.tasks), 1),
            uav_idle / uav_count,
            usv_idle / usv_count,
        ]
        return np.asarray(values, dtype=np.float32)

    def metrics(self) -> dict[str, float | int]:
        mean_wait, max_wait = self.review_wait_stats()
        task_count = max(len(self.tasks), 1)
        return {
            "completion_ratio": len(self.completed_tasks) / task_count,
            "review_queue_length": self.review_queue_length(),
            "mean_review_wait": mean_wait,
            "max_review_wait": max_wait,
            "screening_open_count": self.screening_open_count(),
            "late_task_count": late_task_count(self.tasks),
            "late_task_fraction": late_task_count(self.tasks) / task_count,
            "risk_exposure_sum": total_risk_exposure(self.tasks),
            "total_conflicts": self.total_conflicts,
            "total_invalid_actions": self.total_invalid_actions,
            "total_replenishments": self.total_replenishments,
            "total_returns": self.total_returns,
            "screened_tasks": self.total_screened,
            "reviewed_tasks": self.total_reviewed,
            "anomaly_closed": self.total_anomaly_closed,
            "anomaly_missed": self.total_anomaly_missed,
            "total_path_length": self.total_path_length,
            "total_energy": self.total_energy,
        }

    def _decode_actions(self, action: int | list[int] | tuple[int, ...] | np.ndarray) -> list[tuple[int, int]]:
        if isinstance(action, (list, tuple, np.ndarray)):
            return [(index, int(choice)) for index, choice in enumerate(action) if index < self.num_platforms]
        flat = int(action)
        return [(flat // self.action_choices, flat % self.action_choices)]

    def _actions_from_mapping(self, actions: dict[str, int] | list[int] | tuple[int, ...] | np.ndarray | int) -> list[int] | int:
        if not isinstance(actions, dict):
            return actions
        return [int(actions.get(platform.platform_id, 0)) for platform in self.platforms]

    def _build_candidates(self, platform_index: int) -> list[CandidateEntry]:
        platform = self.platforms[platform_index]
        entries: list[CandidateEntry] = []
        height = max(self.grid.height - 1, 1)
        width = max(self.grid.width - 1, 1)
        for task_index, task in enumerate(self.tasks):
            stage = self._task_stage(task)
            if stage is None:
                continue
            if not platform.can_execute(task, stage):
                continue
            cost = self._estimate_cost(platform_index, task_index, stage)
            feasible = cost.feasible and self._deadline_feasible(task, stage, cost)
            if not feasible:
                continue
            score = self._candidate_score(platform, task, stage, cost)
            entries.append(
                CandidateEntry(
                    task_id=task.task_id,
                    task_index=task_index,
                    task_stage=stage,
                    relative_position=0,
                    risk=float(task.risk),
                    urgency=self._urgency(task, stage, cost),
                    confidence=float(task.screening_confidence if stage == STAGE_REVIEW else 0.0),
                    review_waiting_time=self._review_waiting_time(task),
                    estimated_arrival_time=float(cost.travel_time),
                    estimated_energy=float(cost.energy_cost + cost.return_cost),
                    capability_match=1.0,
                    reservation_status=1.0 if task.reserved_by else 0.0,
                    feasible=feasible,
                    score=score,
                    relative_row=(task.entry_cell[0] - platform.current_cell[0]) / height,
                    relative_col=(task.entry_cell[1] - platform.current_cell[1]) / width,
                    task_geometry_code=self._geometry_code(task.geometry),
                    estimated_finish_time=float(self.current_step + ceil(cost.completion_time)),
                )
            )
        entries.sort(key=lambda item: (-item.score, item.task_stage, item.task_id))
        for index, entry in enumerate(entries[: self.candidate_k]):
            entry.relative_position = index
        return entries[: self.candidate_k]

    def _task_stage(self, task: InspectionTask) -> str | None:
        if task.state == TASK_UNSCREENED:
            return STAGE_SCREENING
        if task.state == TASK_AWAITING_REVIEW:
            return STAGE_REVIEW
        return None

    def _estimate_cost(self, platform_index: int, task_index: int, stage: str) -> TaskCost:
        platform = self.platforms[platform_index]
        task = self.tasks[task_index]
        key = (
            platform_index,
            task_index,
            stage,
            platform.current_cell,
            int(platform.energy * 1000),
            task.state,
            task.reserved_by,
        )
        if key not in self._cost_cache:
            self._cost_cache[key] = estimate_task_cost(platform, task, self.grid, stage=stage)
        return self._cost_cache[key]

    def _candidate_score(self, platform: Platform, task: InspectionTask, stage: str, cost: TaskCost) -> float:
        weights = self.candidate_weights
        return (
            weights["risk_weight"] * task.risk
            + weights["urgency_weight"] * self._urgency(task, stage, cost)
            + weights["review_wait_weight"] * self._review_waiting_time(task)
            - weights["distance_weight"] * cost.travel_time
        )

    def _urgency(self, task: InspectionTask, stage: str, cost: TaskCost) -> float:
        deadline = self._deadline(task, stage)
        if deadline <= 0:
            return min(task.uninspected_time / max(task.max_interval, 1), 2.0)
        projected_finish = self.current_step + ceil(cost.completion_time)
        slack = max(deadline - projected_finish, 0.0)
        return 1.0 / (slack + 1.0)

    def _review_waiting_time(self, task: InspectionTask) -> float:
        if task.state != TASK_AWAITING_REVIEW:
            return 0.0
        return max(0.0, float(self.current_step) - float(task.generation_time))

    def _deadline(self, task: InspectionTask, stage: str) -> float:
        if stage == STAGE_REVIEW and task.review_deadline > 0:
            return float(task.review_deadline)
        return float(task.deadline or task.max_interval)

    def _deadline_feasible(self, task: InspectionTask, stage: str, cost: TaskCost) -> bool:
        deadline = self._deadline(task, stage)
        if deadline <= 0:
            return True
        return self.current_step + ceil(cost.completion_time) <= deadline

    def _must_return(self, platform: Platform) -> bool:
        return_energy = self._energy_for_travel(
            platform,
            self._travel_distance(platform, platform.current_cell, self._platform_depot(platform)),
        )
        return platform.energy <= return_energy + platform.return_reserve_ratio

    def _start_task_stage(self, task: InspectionTask, platform: Platform, stage: str, cost: TaskCost) -> None:
        self._reserve(task, platform, stage)
        platform.current_stage = stage
        platform.target_cell = cost.exit_cell
        travel_steps = max(1 if cost.path_length > 0 else 0, int(ceil(cost.travel_time)))
        service_steps = max(1, int(ceil(self._stage_service_time(task, stage))))
        platform.remaining_travel_time = float(travel_steps)
        platform.remaining_service_time = float(service_steps)
        platform.metadata["pending_cost"] = {
            "path_length": int(cost.path_length),
            "energy_cost": float(cost.energy_cost),
            "energy_steps": max(travel_steps + service_steps, 1),
        }
        if travel_steps > 0:
            platform.mode = MODE_TRAVEL
        else:
            transition_task_state(task, TASK_SCREENING if stage == STAGE_SCREENING else TASK_REVIEWING)
            platform.mode = MODE_SCREEN if stage == STAGE_SCREENING else MODE_REVIEW

    def _advance_platforms(self) -> dict[str, Any]:
        screened_tasks: list[InspectionTask] = []
        reviewed_tasks: list[InspectionTask] = []
        closed_tasks: list[InspectionTask] = []
        path_length = 0
        energy_cost = 0.0

        for platform in self.platforms:
            if platform.mode == MODE_IDLE:
                idle_cost = float(self.review_trigger.get("idle_energy_cost", 0.0))
                if idle_cost > 0:
                    platform.energy = max(0.0, platform.energy - idle_cost)
                    self.total_energy += idle_cost
                    energy_cost += idle_cost
                continue

            pending = dict(platform.metadata.get("pending_cost", {}))
            step_energy = float(pending.get("energy_cost", 0.0)) / max(float(pending.get("energy_steps", 1.0)), 1.0)

            if platform.mode == MODE_TRAVEL:
                platform.remaining_travel_time = max(0.0, platform.remaining_travel_time - 1.0)
                platform.energy = max(0.0, platform.energy - step_energy)
                self.total_energy += step_energy
                energy_cost += step_energy
                if platform.remaining_travel_time <= 0:
                    if platform.target_cell is not None:
                        platform.current_cell = platform.target_cell
                    segment_length = int(pending.get("path_length", 0))
                    path_length += segment_length
                    self.total_path_length += segment_length
                    task = self._current_task(platform)
                    if task is not None and platform.current_stage is not None:
                        transition_task_state(task, TASK_SCREENING if platform.current_stage == STAGE_SCREENING else TASK_REVIEWING)
                    platform.mode = MODE_SCREEN if platform.current_stage == STAGE_SCREENING else MODE_REVIEW
                continue

            if platform.mode in {MODE_SCREEN, MODE_REVIEW}:
                platform.remaining_service_time = max(0.0, platform.remaining_service_time - 1.0)
                platform.energy = max(0.0, platform.energy - step_energy)
                self.total_energy += step_energy
                energy_cost += step_energy
                if platform.remaining_service_time <= 0:
                    task = self._current_task(platform)
                    if task is None:
                        self._clear_platform_assignment(platform)
                        continue
                    if platform.mode == MODE_SCREEN:
                        screened_tasks.append(task)
                        self.total_screened += 1
                        if self._complete_screening(task, platform):
                            closed_tasks.append(task)
                    else:
                        reviewed_tasks.append(task)
                        self.total_reviewed += 1
                        self._complete_review(task, platform)
                        closed_tasks.append(task)
                continue

            if platform.mode == MODE_RETURN:
                platform.remaining_travel_time = max(0.0, platform.remaining_travel_time - 1.0)
                platform.energy = max(0.0, platform.energy - step_energy)
                self.total_energy += step_energy
                energy_cost += step_energy
                if platform.remaining_travel_time <= 0:
                    platform.current_cell = self._platform_depot(platform)
                    segment_length = int(pending.get("path_length", 0))
                    path_length += segment_length
                    self.total_path_length += segment_length
                    platform.mode = MODE_REPLENISH
                    platform.remaining_replenish_time = float(max(int(self.review_trigger.get("replenish_steps", 2)), 1))
                    platform.metadata.pop("pending_cost", None)
                    self.total_replenishments += 1
                continue

            if platform.mode == MODE_REPLENISH:
                platform.remaining_replenish_time = max(0.0, platform.remaining_replenish_time - 1.0)
                if platform.remaining_replenish_time <= 0:
                    platform.energy = platform.energy_capacity
                    platform.mode = MODE_IDLE

        return {
            "screened_tasks": screened_tasks,
            "reviewed_tasks": reviewed_tasks,
            "closed_tasks": closed_tasks,
            "path_length": path_length,
            "energy_cost": energy_cost,
        }

    def _resolve_conflicts(self, proposals: list[_Proposal]) -> list[_Proposal]:
        grouped: dict[tuple[int, str], list[_Proposal]] = {}
        for proposal in proposals:
            key = (proposal.candidate.task_index, proposal.candidate.task_stage)
            grouped.setdefault(key, []).append(proposal)
        accepted: list[_Proposal] = []
        for group in grouped.values():
            if len(group) == 1:
                accepted.append(group[0])
                continue
            winner = self._arbitrate(group)
            accepted.append(winner)
            for loser in group:
                if loser is winner:
                    continue
                self.last_conflicts.append(
                    {
                        "task_id": loser.candidate.task_id,
                        "stage": loser.candidate.task_stage,
                        "loser": self.platforms[loser.platform_index].platform_id,
                        "winner": self.platforms[winner.platform_index].platform_id,
                    }
                )
        return accepted

    def _arbitrate(self, proposals: list[_Proposal]) -> _Proposal:
        ranked: list[tuple[float, float, float, _Proposal]] = []
        for proposal in proposals:
            platform = self.platforms[proposal.platform_index]
            energy_ratio = platform.energy / max(platform.energy_capacity, 1e-6)
            ranked.append((proposal.cost.travel_time, -energy_ratio, float(self.rng.random()), proposal))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return ranked[0][3]

    def _reserve(self, task: InspectionTask, platform: Platform, stage: str) -> None:
        task.reserved_by = platform.platform_id
        task.reservation_time = float(self.current_step)
        if stage == STAGE_SCREENING:
            transition_task_state(task, TASK_RESERVED_SCREENING)
        else:
            transition_task_state(task, TASK_RESERVED_REVIEW)
        platform.current_task_id = task.task_id

    def _complete_screening(self, task: InspectionTask, platform: Platform) -> bool:
        task.screening_workload_remaining = 0.0
        task.screened_by = platform.platform_id
        task.screening_finish_time = float(self.current_step)
        result, confidence = self._screening_observation(task)
        task.screening_result = int(result)
        task.screening_confidence = confidence
        task.screening_uncertainty = 1.0 - confidence
        task.review_required = self._should_trigger_review(task)
        task.reserved_by = None
        task.reservation_time = None
        task.release_time = float(self.current_step)
        transition_task_state(task, TASK_SCREENED_PENDING)
        self._clear_platform_assignment(platform)
        if task.review_required:
            transition_task_state(task, TASK_AWAITING_REVIEW)
            task.generation_time = float(self.current_step)
            task.review_deadline = float(self.current_step) + self._review_deadline(task)
            return False
        if task.true_anomaly:
            self.total_anomaly_missed += 1
        self._close_task(task)
        return True

    def _complete_review(self, task: InspectionTask, platform: Platform) -> None:
        task.review_workload_remaining = 0.0
        task.reviewed_by = platform.platform_id
        task.review_finish_time = float(self.current_step)
        task.review_result = int(task.true_anomaly)
        task.reserved_by = None
        task.reservation_time = None
        task.release_time = float(self.current_step)
        self._clear_platform_assignment(platform)
        if task.true_anomaly:
            self.total_anomaly_closed += 1
        self._close_task(task)

    def _close_task(self, task: InspectionTask) -> None:
        transition_task_state(task, TASK_CLOSED)
        task.completed = True
        task.close_time = float(self.current_step)
        task.reserved_by = None
        self.completed_tasks.add(task.task_id)

    def _screening_observation(self, task: InspectionTask) -> tuple[int, float]:
        sensitivity = float(self.review_trigger["sensitivity"])
        specificity = float(self.review_trigger["specificity"])
        noise = float(self.review_trigger["confidence_noise"])
        if task.true_anomaly:
            result = int(self.rng.random() < sensitivity)
            mean = sensitivity if result else 1.0 - sensitivity
        else:
            result = int(self.rng.random() < (1.0 - specificity))
            mean = specificity if not result else 1.0 - specificity
        confidence = float(np.clip(self.rng.normal(mean, noise), 0.0, 1.0))
        return result, confidence

    def _should_trigger_review(self, task: InspectionTask) -> bool:
        return (
            int(task.screening_result or 0) == 1
            or task.screening_confidence < float(self.review_trigger["confidence_threshold"])
            or task.risk >= int(self.review_trigger["mandatory_review_risk"])
        )

    def _review_deadline(self, task: InspectionTask) -> float:
        base = float(self.review_trigger["base_review_deadline"])
        risk_scale = float(self.review_trigger["risk_deadline_scale"])
        confidence_scale = float(self.review_trigger["confidence_deadline_scale"])
        risk_adjusted = base - risk_scale * max(task.risk - 1, 0) - confidence_scale * task.screening_confidence
        service_floor = max(float(task.review_workload) + 4.0, 2.0)
        return max(service_floor, risk_adjusted)

    def _sample_true_anomaly(self, task: InspectionTask) -> bool:
        probabilities = self.review_trigger.get("anomaly_probability_by_risk", {})
        probability = 0.1
        if isinstance(probabilities, dict):
            probability = float(probabilities.get(task.risk, probabilities.get(str(task.risk), probability)))
        return bool(self.rng.random() < probability)

    def _stage_service_time(self, task: InspectionTask, stage: str) -> float:
        if stage == STAGE_SCREENING:
            return max(float(task.screening_workload), 1.0)
        return max(float(task.review_workload), 1.0)

    def _current_task(self, platform: Platform) -> InspectionTask | None:
        if platform.current_task_id is None:
            return None
        return next((task for task in self.tasks if task.task_id == platform.current_task_id), None)

    def _clear_platform_assignment(self, platform: Platform) -> None:
        platform.mode = MODE_IDLE
        platform.current_task_id = None
        platform.current_stage = None
        platform.target_cell = None
        platform.remaining_travel_time = 0.0
        platform.remaining_service_time = 0.0
        platform.metadata.pop("pending_cost", None)

    def _start_return(self, platform_index: int) -> None:
        platform = self.platforms[platform_index]
        depot = self._platform_depot(platform)
        travel = self._travel_distance(platform, platform.current_cell, depot)
        energy = self._energy_for_travel(platform, travel)
        platform.mode = MODE_RETURN
        platform.current_task_id = None
        platform.current_stage = None
        platform.target_cell = depot
        platform.remaining_travel_time = float(max(1 if travel > 0 else 0, int(ceil(travel))))
        platform.metadata["pending_cost"] = {
            "path_length": int(travel),
            "energy_cost": float(energy),
            "energy_steps": max(int(platform.remaining_travel_time), 1),
        }
        self.total_returns += 1

    def _travel_distance(self, platform: Platform, start: GridCell, goal: GridCell) -> int:
        if start == goal:
            return 0
        if self._uses_coordinate_distance():
            return self._coordinate_distance(start, goal)
        if platform.platform_type == "UAV":
            return abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        return max(len(shortest_path(self.grid, start, goal)) - 1, 0)

    def _energy_for_travel(self, platform: Platform, travel_len: int) -> float:
        cell_m = max(float(self.grid.cell_size_m), 1.0)
        speed = max(float(platform.speed_mps), 0.1)
        minutes = travel_len * cell_m / speed / 60.0
        return minutes / max(float(platform.endurance_minutes), 1.0) * max(platform.energy_rate_per_cell, 0.1)

    def _cell_distance(self, left: GridCell, right: GridCell) -> int:
        if self._uses_coordinate_distance():
            return self._coordinate_distance(left, right)
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _uses_coordinate_distance(self) -> bool:
        return str(self.grid.metadata.get("distance_mode", "")).lower() == "utm_euclidean"

    def _coordinate_distance(self, left: GridCell, right: GridCell) -> int:
        return int(round(((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5))

    def _platform_depot(self, platform: Platform) -> GridCell:
        depot = platform.metadata.get("depot_cell")
        if isinstance(depot, (list, tuple)) and len(depot) == 2:
            return int(depot[0]), int(depot[1])
        return self.grid.depot

    def _platform_features(self, platform: Platform) -> list[float]:
        height = max(self.grid.height - 1, 1)
        width = max(self.grid.width - 1, 1)
        return [
            1.0 if platform.platform_type == "UAV" else -1.0,
            platform.current_cell[0] / height,
            platform.current_cell[1] / width,
            platform.energy / max(platform.energy_capacity, 1e-6),
            platform.current_load / 500.0,
            platform.speed_mps / 25.0,
            platform.endurance_minutes / 360.0,
            platform.return_reserve_ratio,
            1.0 if platform.can_decide else 0.0,
            self._mode_code(platform.mode),
        ]

    def _neighbor_summary(self, platform_index: int, neighbor_mask: np.ndarray) -> list[float]:
        neighbors = [
            self.platforms[index]
            for index in range(self.num_platforms)
            if index != platform_index and neighbor_mask[platform_index, index]
        ]
        if not neighbors:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        uav_count = sum(1 for platform in neighbors if platform.platform_type == "UAV")
        usv_count = sum(1 for platform in neighbors if platform.platform_type == "USV")
        mean_energy = float(np.mean([platform.energy / max(platform.energy_capacity, 1e-6) for platform in neighbors]))
        mean_load = float(np.mean([platform.current_load / 500.0 for platform in neighbors]))
        return [
            uav_count / max(self.num_platforms, 1),
            usv_count / max(self.num_platforms, 1),
            mean_energy,
            mean_load,
            len(neighbors) / max(self.num_platforms - 1, 1),
        ]

    def _task_features(self, task: InspectionTask) -> list[float]:
        active_stage = task.active_stage or STAGE_SCREENING
        deadline = self._deadline(task, active_stage)
        slack = 0.0 if task.completed else max(deadline - self.current_step, 0.0) / max(self.max_steps, 1)
        return [
            self._state_code(task.state),
            self._stage_code(active_stage) if not task.completed else 0.0,
            task.risk / 5.0,
            min(task.uninspected_time / max(task.max_interval, 1), 2.0),
            slack,
            self._review_waiting_time(task) / max(self.max_steps, 1),
            task.screening_confidence,
            task.screening_uncertainty,
            self._geometry_code(task.geometry),
            1.0 if task.completed else 0.0,
        ]

    def _candidate_features(self, candidate: CandidateEntry) -> list[float]:
        return [
            candidate.relative_row,
            candidate.relative_col,
            candidate.risk / 3.0,
            candidate.urgency,
            min(candidate.review_waiting_time / max(self.max_steps, 1), 1.0),
            candidate.estimated_arrival_time / 60.0,
            self._state_code(self.tasks[candidate.task_index].state),
            candidate.confidence,
        ]

    def _candidate_info(self, candidate: CandidateEntry) -> dict[str, Any]:
        return {
            "task_id": candidate.task_id,
            "task_stage": candidate.task_stage,
            "relative_position": candidate.relative_position,
            "risk": candidate.risk,
            "urgency": candidate.urgency,
            "confidence": candidate.confidence,
            "review_waiting_time": candidate.review_waiting_time,
            "estimated_arrival_time": candidate.estimated_arrival_time,
            "estimated_energy": candidate.estimated_energy,
            "relative_row": candidate.relative_row,
            "relative_col": candidate.relative_col,
            "task_geometry_code": candidate.task_geometry_code,
            "estimated_finish_time": candidate.estimated_finish_time,
            "score": candidate.score,
        }

    def _stage_code(self, stage: str) -> float:
        if stage == STAGE_SCREENING:
            return 1.0
        if stage == STAGE_REVIEW:
            return -1.0
        return 0.0

    def _mode_code(self, mode: str) -> float:
        return {
            MODE_IDLE: 0.0,
            MODE_TRAVEL: 0.1,
            MODE_SCREEN: 0.2,
            MODE_REVIEW: 0.4,
            MODE_RETURN: 0.6,
            MODE_REPLENISH: 0.8,
        }.get(mode, 0.0)

    def _state_code(self, state: str) -> float:
        return {
            TASK_UNSCREENED: 0.0,
            TASK_RESERVED_SCREENING: 0.15,
            TASK_SCREENING: 0.25,
            TASK_AWAITING_REVIEW: 0.45,
            TASK_RESERVED_REVIEW: 0.55,
            TASK_REVIEWING: 0.65,
            TASK_SCREENED_PENDING: 0.8,
            TASK_CLOSED: 1.0,
        }.get(state, 0.0)

    def _geometry_code(self, geometry: str) -> float:
        normalized = geometry.lower()
        if normalized in {"point", "poi", "berth", "facility"}:
            return 0.25
        if normalized in {"line", "waterway", "channel", "航道"}:
            return 0.5
        if normalized in {"polygon", "area", "land", "port_polygon", "陆地港口"}:
            return 0.75
        return 0.0
