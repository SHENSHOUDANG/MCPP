from pathlib import Path
import sys
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from check_port_inspection_env import build_env
from mathbased_mcpp.port_inspection.mappo import HeterogeneousMappo
from mathbased_mcpp.port_inspection.schema import STAGE_SCREENING, STAGE_REVIEW, TASK_AWAITING_REVIEW, TASK_CLOSED
from train_port_scheduler_rl import _agent_types, _collect_rollout, _mappo_update, _obs_matrix

import torch


class PortInspectionCoupledEnvTests(unittest.TestCase):
    def test_screening_triggers_review_and_usv_closes_task(self) -> None:
        config = _load_config(ROOT / "configs" / "port_dalian_water_v0.toml")
        env = build_env(config)
        env.reset(seed=7)

        self.assertEqual(env.action_dim, env.num_platforms * env.action_choices)
        self.assertEqual(env.action_choices, env.candidate_k + 3)
        self.assertEqual(env.continue_action, env.candidate_k)
        self.assertEqual(env.wait_action, env.candidate_k + 1)
        self.assertEqual(env.return_action, env.candidate_k + 2)

        uav_index = next(index for index, platform in enumerate(env.platforms) if platform.platform_type == "UAV")
        screening = next(
            candidate
            for candidate in env.candidate_lists()[uav_index]
            if candidate.task_stage == STAGE_SCREENING and env.tasks[candidate.task_index].risk >= 3
        )

        actions = [env.wait_action for _ in env.platforms]
        actions[uav_index] = screening.relative_position
        env.step(actions)
        self.assertTrue(env.action_masks()[uav_index, env.continue_action])
        _continue_until(env, lambda: env.tasks[screening.task_index].state == TASK_AWAITING_REVIEW)
        task = env.tasks[screening.task_index]
        self.assertEqual(task.state, TASK_AWAITING_REVIEW)
        self.assertTrue(task.review_required)
        self.assertIsNotNone(task.screening_result)
        self.assertEqual(task.screened_by, env.platforms[uav_index].platform_id)

        usv_index, review = next(
            (platform_index, candidate)
            for platform_index, candidates in enumerate(env.candidate_lists())
            if env.platforms[platform_index].platform_type == "USV"
            for candidate in candidates
            if candidate.task_stage == STAGE_REVIEW and candidate.task_id == task.task_id
        )
        actions = [env.wait_action for _ in env.platforms]
        actions[usv_index] = review.relative_position
        env.step(actions)
        _continue_until(env, lambda: task.state == TASK_CLOSED)

        self.assertEqual(task.state, TASK_CLOSED)
        self.assertTrue(task.completed)
        self.assertIn(task.task_id, env.completed_tasks)
        self.assertEqual(task.reviewed_by, env.platforms[usv_index].platform_id)
        self.assertEqual(task.review_result, int(task.true_anomaly))

    def test_model_interface_exposes_local_state_masks_and_metrics(self) -> None:
        config = _load_config(ROOT / "configs" / "port_dalian_water_v0.toml")
        env = build_env(config)

        reset = env.reset_model(seed=11)
        platform_ids = {platform.platform_id for platform in env.platforms}
        self.assertEqual(set(reset.obs_dict), platform_ids)
        self.assertEqual(set(reset.available_actions), platform_ids)
        self.assertEqual(reset.state.ndim, 1)
        self.assertGreater(reset.state.size, 0)
        self.assertEqual(reset.info["local_observation_dim"], env.local_observation_dim)
        self.assertEqual(reset.info["global_state_dim"], env.global_state_dim)
        self.assertIn("metrics", reset.info)
        self.assertIn("aggregate_broadcast", reset.info)

        for platform_id in platform_ids:
            self.assertEqual(reset.obs_dict[platform_id].shape, (env.local_observation_dim,))
            self.assertEqual(reset.available_actions[platform_id].shape, (env.action_choices,))
            self.assertEqual(float(reset.available_actions[platform_id][env.wait_action]), 1.0)

        candidate_details = [
            detail
            for platform_candidates in reset.info["candidate_details"]
            for detail in platform_candidates
        ]
        self.assertTrue(candidate_details)
        self.assertIn("relative_row", candidate_details[0])
        self.assertIn("relative_col", candidate_details[0])
        self.assertIn("task_geometry_code", candidate_details[0])
        self.assertIn("estimated_finish_time", candidate_details[0])

        step = env.step_model({platform_id: env.wait_action for platform_id in platform_ids})
        self.assertEqual(set(step.obs_dict), platform_ids)
        self.assertEqual(set(step.rewards), platform_ids)
        self.assertEqual(set(step.available_actions), platform_ids)
        self.assertFalse(step.terminated)
        self.assertFalse(step.truncated)
        self.assertIn("total_invalid_actions", step.info["metrics"])

    def test_heterogeneous_mappo_rollout_update_smoke(self) -> None:
        config = _load_config(ROOT / "configs" / "port_dalian_water_v0.toml")
        env = build_env(config)
        reset = env.reset_model(seed=19)
        model = HeterogeneousMappo(env.local_observation_dim, env.action_choices, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        obs = _obs_matrix(env, reset.obs_dict)
        agent_types = _agent_types(env)

        batch, _, _, _ = _collect_rollout(env, model, obs, agent_types, rollout_steps=4, gamma=0.98, gae_lambda=0.95)
        self.assertEqual(batch.observations.ndim, 3)
        self.assertEqual(batch.observations.shape[1], env.num_platforms)
        self.assertEqual(batch.action_masks.shape[-1], env.action_choices)
        self.assertTrue(hasattr(model, "uav_actor"))
        self.assertTrue(hasattr(model, "usv_actor"))

        _mappo_update(model, optimizer, batch, clip_ratio=0.2, update_epochs=1, entropy_coef=0.01, value_coef=0.5)

    def test_centralized_critic_accepts_variable_agent_counts(self) -> None:
        model = HeterogeneousMappo(observation_dim=12, action_dim=11, hidden_dim=16)
        for agent_count in (2, 5):
            observations = torch.zeros((1, agent_count, 12), dtype=torch.float32)
            agent_types = torch.tensor([[0, 1, 0, 1, 1][:agent_count]], dtype=torch.long)
            agent_mask = torch.ones((1, agent_count), dtype=torch.bool)
            values = model.value(observations, agent_types, agent_mask)
            logits = model.logits(observations, agent_types)
            self.assertEqual(values.shape, (1,))
            self.assertEqual(logits.shape, (1, agent_count, 11))


def _load_config(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _continue_until(env, predicate, limit: int = 64) -> None:
    for _ in range(limit):
        if predicate():
            return
        env.step([env.continue_action if platform.mode != "idle" else env.wait_action for platform in env.platforms])
    raise AssertionError("environment did not reach expected state")


if __name__ == "__main__":
    unittest.main()
