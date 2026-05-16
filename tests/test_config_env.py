from pathlib import Path
import sys
import unittest
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json
from dataclasses import asdict

import numpy as np

from mathbased_mcpp.config import (
    GridCoverageConfig,
    RewardConfig,
    build_course_config,
    load_config,
    select_curriculum_course,
)
from mathbased_mcpp.evaluation import load_policy
from mathbased_mcpp.env import GridCoverageEnv
from mathbased_mcpp.ppo import ActorCritic

import torch


class ConfigEnvTests(unittest.TestCase):
    def test_load_smoke_config(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        self.assertEqual(config.env.width, 6)
        self.assertEqual(config.env.height, 6)
        self.assertEqual(config.env.reward.finish_reward, 80.0)
        self.assertEqual(config.env.reward.time_penalty_weight, 0.3)
        self.assertEqual(config.env.reward.repeat_penalty_weight, 0.1)
        self.assertEqual(config.ppo.rollout_steps, 64)

    def test_load_curriculum_config(self) -> None:
        config = load_config(ROOT / "configs" / "formal_v1.toml")
        self.assertIsNotNone(config.curriculum)
        self.assertEqual(len(config.curriculum.courses), 4)
        self.assertEqual(config.curriculum.courses[0].name, "tier-1-8x8-1agent")
        self.assertEqual(config.curriculum.courses[0].env.max_steps, 100)
        self.assertEqual(config.curriculum.courses[0].env.observation_radius, 2)
        self.assertEqual(config.curriculum.courses[1].env.width, 13)
        self.assertEqual(config.curriculum.courses[1].env.height, 13)
        self.assertEqual(config.curriculum.courses[1].env.num_agents, 2)
        self.assertEqual(config.curriculum.courses[2].env.width, 18)
        self.assertEqual(config.curriculum.courses[2].env.height, 18)
        self.assertEqual(config.curriculum.courses[2].env.num_agents, 3)
        self.assertEqual(config.curriculum.courses[3].env.width, 30)
        self.assertEqual(config.curriculum.courses[3].env.height, 30)
        self.assertEqual(config.curriculum.courses[3].env.num_agents, 4)
        self.assertEqual(config.curriculum.courses[3].env.obstacle_ratio, 0.0625)
        self.assertEqual(config.curriculum.courses[3].env.recent_path_length, 8)
        self.assertEqual(config.curriculum.courses[3].env.communication_radius, 4)
        self.assertTrue(config.ppo.use_graph_attention)
        self.assertEqual(config.curriculum.courses[3].total_timesteps, 4000000)
        self.assertFalse(config.curriculum.courses[0].load_previous)

    def test_select_curriculum_course_by_name(self) -> None:
        config = load_config(ROOT / "configs" / "formal_v1.toml")
        index, course = select_curriculum_course(config, course_name="tier-2-13x13-2agents")
        self.assertEqual(index, 1)
        self.assertEqual(course.name, "tier-2-13x13-2agents")

    def test_course_config_snapshot_roundtrip(self) -> None:
        config = load_config(ROOT / "configs" / "formal_v1.toml")
        _, course = select_curriculum_course(config, course_name="tier-1-8x8-1agent")
        course_config = build_course_config(config, course)
        run_dir = ROOT / ".tmp_tests" / "course-config-roundtrip"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot = run_dir / "course_config.json"
            snapshot.write_text(json.dumps(asdict(course_config)), encoding="utf-8")
            loaded = load_config(snapshot)
            self.assertEqual(loaded.env.width, course_config.env.width)
            self.assertEqual(loaded.env.reward.time_penalty_weight, course_config.env.reward.time_penalty_weight)
            self.assertEqual(loaded.ppo.total_timesteps, course_config.ppo.total_timesteps)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_local_observation_shape(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=6, height=6, start=(0, 0)))
        observation = env.reset()
        self.assertEqual(observation.shape[0], env.observation_dim)
        self.assertEqual(env.observation_dim, 75)

    def test_global_state_shape_and_map_layers(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=3,
                height=2,
                start=(0, 0),
                obstacles=[(1, 1)],
            )
        )
        env.reset()
        state = env.global_state()
        cells = env.config.width * env.config.height
        agent = state[0:cells].reshape(env.config.height, env.config.width)
        uncovered = state[cells : cells * 2].reshape(env.config.height, env.config.width)
        covered = state[cells * 2 : cells * 3].reshape(env.config.height, env.config.width)
        obstacles = state[cells * 3 : cells * 4].reshape(env.config.height, env.config.width)

        self.assertEqual(state.shape[0], env.state_dim)
        self.assertEqual(env.state_dim, 37)
        self.assertEqual(agent[0, 0], 1.0)
        self.assertEqual(uncovered[0, 0], 0.0)
        self.assertEqual(uncovered[0, 1], 1.0)
        self.assertEqual(covered[0, 0], 1.0)
        self.assertEqual(obstacles[1, 1], 1.0)

    def test_random_corner_start_is_canonicalized(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=4, height=3, start=(0, 0), random_corner_start=True, seed=7))
        observation = env.reset()
        corners = {(0, 0), (0, 3), (2, 0), (2, 3)}
        self.assertIn(env.position, corners)
        self.assertEqual(observation.shape[0], env.observation_dim)

        cells = env.config.width * env.config.height
        state = env.global_state()
        agent = state[:cells].reshape(env.config.height, env.config.width)
        self.assertEqual(agent[0, 0], 1.0)

        result = env.step(3)
        next_agent = result.state[:cells].reshape(env.config.height, env.config.width)
        self.assertEqual(next_agent[0, 1], 1.0)

    def test_step_result_carries_next_global_state(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=3, height=1, start=(0, 0)))
        env.reset()
        result = env.step(3)
        self.assertEqual(result.observation.shape[0], env.observation_dim)
        self.assertEqual(result.state.shape[0], env.state_dim)

    def test_obstacle_ratio_generates_scaled_obstacles(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=20, height=20, obstacle_ratio=0.09375, random_obstacle_seed=3))
        env.reset()
        self.assertEqual(len(env.obstacles), 38)

    def test_multi_agent_reset_shapes_and_state_channels_are_fixed(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=5,
                num_agents=2,
                start_positions=[(0, 0), (4, 4)],
                observation_radius=1,
            )
        )
        observation = env.reset()
        self.assertEqual(observation.shape, (2, env.observation_dim))
        self.assertEqual(env.observation_dim, 75)
        self.assertEqual(env.state_dim, 5 * 5 * 5 + 7)

    def test_observation_includes_self_memory_channels(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(width=3, height=3, start=(1, 1), observation_radius=1, recent_path_length=4)
        )
        env.reset()
        result = env.step(3)
        window_area = (env.config.observation_radius * 2 + 1) ** 2
        self_covered = result.observation[5 * window_area : 6 * window_area].reshape(3, 3)
        recent_path = result.observation[6 * window_area : 7 * window_area].reshape(3, 3)

        self.assertEqual(float(self_covered.sum()), 2.0)
        self.assertEqual(self_covered[1, 1], 1.0)
        self.assertEqual(self_covered[1, 0], 1.0)
        self.assertEqual(recent_path[1, 1], 1.0)
        self.assertGreater(recent_path[1, 1], recent_path[1, 0])

    def test_neighbor_mask_uses_communication_radius(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=5,
                num_agents=3,
                start_positions=[(0, 0), (0, 2), (4, 4)],
                communication_radius=2,
            )
        )
        env.reset()
        mask = env.neighbor_mask()

        self.assertTrue(np.array_equal(np.diag(mask), np.ones(3, dtype=bool)))
        self.assertTrue(mask[0, 1])
        self.assertTrue(mask[1, 0])
        self.assertFalse(mask[0, 2])
        self.assertFalse(mask[1, 2])

    def test_graph_attention_policy_preserves_agent_batch_shape(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=5,
                num_agents=3,
                start_positions=[(0, 0), (0, 2), (4, 4)],
                communication_radius=3,
            )
        )
        observation = env.reset()
        state = np.repeat(env.global_state()[None, :], env.num_agents, axis=0)
        model = ActorCritic(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=16,
            state_shape=(env.config.height, env.config.width),
            use_graph_attention=True,
        )
        actions, log_probs, values = model.act_batch(
            torch.as_tensor(observation, dtype=torch.float32),
            torch.as_tensor(state, dtype=torch.float32),
            neighbor_mask=torch.as_tensor(env.neighbor_mask(), dtype=torch.bool),
        )

        self.assertEqual(actions.shape, (3,))
        self.assertEqual(log_probs.shape, (3,))
        self.assertEqual(values.shape, (3,))

    def test_multi_agent_rewards_are_shared(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(width=4, height=1, num_agents=2, start_positions=[(0, 0), (0, 3)], max_steps=5)
        )
        env.reset()
        result = env.step([3, 2])
        self.assertEqual(result.reward.shape, (2,))
        self.assertAlmostEqual(float(result.reward[0]), float(result.reward[1]))
        self.assertEqual(result.info["reward_terms"]["new_cells"], 2.0)

    def test_multi_agent_same_target_collision_blocks_both(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(width=3, height=1, num_agents=2, start_positions=[(0, 0), (0, 2)], max_steps=5)
        )
        env.reset()
        result = env.step([3, 2])
        self.assertEqual(env.positions, [(0, 0), (0, 2)])
        self.assertEqual(result.info["reward_terms"]["collision_agents"], 2.0)

    def test_multi_agent_swap_collision_blocks_both(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(width=3, height=1, num_agents=2, start_positions=[(0, 0), (0, 1)], max_steps=5)
        )
        env.reset()
        result = env.step([3, 2])
        self.assertEqual(env.positions, [(0, 0), (0, 1)])
        self.assertEqual(result.info["reward_terms"]["collision_agents"], 2.0)

    def test_spatial_policy_can_load_on_larger_map(self) -> None:
        source_env = GridCoverageEnv(GridCoverageConfig(width=6, height=6, start=(0, 0)))
        source_model = ActorCritic(
            observation_dim=source_env.observation_dim,
            action_dim=source_env.action_dim,
            hidden_dim=64,
            state_shape=(source_env.config.height, source_env.config.width),
        )

        run_dir = ROOT / ".tmp_tests" / "spatial-policy-load"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint_path = run_dir / "policy.pt"
            torch.save(
                {
                    "model_state_dict": source_model.state_dict(),
                    "observation_dim": source_model.observation_dim,
                    "state_dim": source_model.state_dim,
                    "action_dim": source_model.action_dim,
                    "hidden_dim": source_model.hidden_dim,
                    "critic_type": source_model.critic_mode,
                    "state_shape": source_model.state_shape,
                    "state_channels": source_model.state_channels,
                    "state_metadata_dim": source_model.state_metadata_dim,
                },
                checkpoint_path,
            )

            config = load_config(ROOT / "configs" / "smoke.toml")
            config.env.width = 8
            config.env.height = 8
            config.env.max_steps = 2
            model = load_policy(config, checkpoint_path)

            target_env = GridCoverageEnv(config.env)
            observation = target_env.reset()
            state = target_env.global_state()
            action, _, value = model.act(
                torch.as_tensor(observation, dtype=torch.float32),
                torch.as_tensor(state, dtype=torch.float32),
            )

            self.assertEqual(model.state_shape, (8, 8))
            self.assertIn(action, range(target_env.action_dim))
            self.assertTrue(torch.isfinite(value).item())
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_reset_marks_start_covered(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=6, height=6, start=(0, 0)))
        env.reset()
        self.assertIn((0, 0), env.covered)
        self.assertEqual(env.path_length, 0)

    def test_legal_move_updates_position_and_path_length(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=6, height=6, start=(0, 0)))
        env.reset()
        result = env.step(3)
        self.assertEqual(env.position, (0, 1))
        self.assertEqual(result.info["path_length"], 1)
        self.assertIn((0, 1), env.covered)

    def test_illegal_move_keeps_state_and_penalizes(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=6, height=6, start=(0, 0)))
        env.reset()
        result = env.step(0)
        self.assertEqual(env.position, (0, 0))
        self.assertLess(result.reward, 0)
        self.assertEqual(result.info["path_length"], 0)

    def test_finish_reward_uses_terminal_value(self) -> None:
        reward = RewardConfig(distance_weight=0.0, straight_weight=0.0, coverage_weight=0.0, finish_reward=10.0)
        env = GridCoverageEnv(GridCoverageConfig(width=2, height=1, start=(0, 0), reward=reward))
        env.reset()
        result = env.step(3)
        self.assertTrue(result.done)
        self.assertEqual(result.info["reward_terms"]["finish"], 10.0)
        self.assertGreater(result.reward, 10.0)


if __name__ == "__main__":
    unittest.main()
