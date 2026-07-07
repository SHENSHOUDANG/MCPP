import unittest

from mathbased_mcpp.port_inspection.reward import compute_reward_terms
from mathbased_mcpp.port_inspection.schema import InspectionTask, Platform, TASK_ACTIVE


def _task(
    task_id: str,
    release_time: float,
    first_valid_assignment_time: float | None = None,
    completed: bool = False,
) -> InspectionTask:
    return InspectionTask(
        task_id=task_id,
        task_type="inspection",
        geometry="POINT",
        cells=((0, 0),),
        risk=1,
        service_time=1,
        allowed_platforms=("UAV",),
        state=TASK_ACTIVE,
        release_time=release_time,
        first_valid_assignment_time=first_valid_assignment_time,
        completed=completed,
    )


def _platform() -> Platform:
    return Platform(
        platform_id="uav-0",
        platform_type="UAV",
        current_cell=(0, 0),
        speed_mps=1.0,
        endurance_minutes=60.0,
        allowed_task_types=("POINT",),
    )


class PortRewardTests(unittest.TestCase):
    def test_wait_time_cost_uses_scaled_current_open_wait(self) -> None:
        newly_assigned = _task("assigned-now", release_time=20.0, first_valid_assignment_time=80.0)
        terms = compute_reward_terms(
            tasks=[
                _task("open-early", release_time=0.0),
                _task("open-late", release_time=60.0),
                newly_assigned,
                _task("done", release_time=0.0, first_valid_assignment_time=10.0, completed=True),
            ],
            platforms=[_platform()],
            completed_task=None,
            path_length=0,
            energy_cost=0.0,
            invalid=False,
            newly_assigned_tasks=[newly_assigned],
            current_time=120.0,
            weights={
                "time_cost": 0.0,
                "wait_time_cost": 0.05,
                "wait_time_scale": 60.0,
            },
        )

        self.assertAlmostEqual(terms["wait_time_cost"], -0.05 * ((120.0 + 60.0 + 60.0) / 3.0 / 60.0))

    def test_wait_time_cost_zero_keeps_wait_reward_disabled(self) -> None:
        terms = compute_reward_terms(
            tasks=[_task("open", release_time=0.0)],
            platforms=[_platform()],
            completed_task=None,
            path_length=0,
            energy_cost=0.0,
            invalid=False,
            current_time=120.0,
            weights={"time_cost": 0.0, "wait_time_cost": 0.0},
        )

        self.assertEqual(terms["wait_time_cost"], -0.0)

    def test_wait_time_sum_aggregation_remains_available_for_ablation(self) -> None:
        terms = compute_reward_terms(
            tasks=[_task("open-0", release_time=0.0), _task("open-60", release_time=60.0)],
            platforms=[_platform()],
            completed_task=None,
            path_length=0,
            energy_cost=0.0,
            invalid=False,
            current_time=120.0,
            weights={
                "time_cost": 0.0,
                "wait_time_cost": 0.05,
                "wait_time_scale": 60.0,
                "wait_time_aggregation": "sum",
            },
        )

        self.assertAlmostEqual(terms["wait_time_cost"], -0.05 * ((120.0 + 60.0) / 60.0))


if __name__ == "__main__":
    unittest.main()
