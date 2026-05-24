from pathlib import Path
from collections import deque
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
        self.assertEqual(config.curriculum.courses[2].env.random_obstacle_seeds, [20260430, 20260431, 20260432, 20260433])
        self.assertEqual(config.curriculum.courses[2].env.map_refresh_episodes, 5)
        self.assertEqual(config.curriculum.courses[3].name, "tier-4-20x20-4agents")
        self.assertEqual(config.curriculum.courses[3].env.width, 20)
        self.assertEqual(config.curriculum.courses[3].env.height, 20)
        self.assertEqual(config.curriculum.courses[3].env.max_steps, 500)
        self.assertEqual(config.curriculum.courses[3].env.num_agents, 4)
        self.assertEqual(config.env.obstacle_ratio, 0.05)
        self.assertEqual([course.env.obstacle_ratio for course in config.curriculum.courses], [0.05, 0.05, 0.05, 0.05])
        self.assertEqual(len(config.curriculum.courses[3].env.random_obstacle_seeds), 8)
        self.assertEqual(config.curriculum.courses[3].env.map_refresh_episodes, 3)
        self.assertEqual(config.curriculum.courses[3].env.recent_path_length, 8)
        self.assertEqual(config.curriculum.courses[3].env.communication_radius, 4)
        self.assertTrue(config.ppo.use_graph_attention)
        self.assertEqual(config.ppo.gat_num_heads, 4)
        self.assertTrue(config.ppo.gat_use_edge_features)
        self.assertTrue(config.ppo.gat_residual)
        self.assertEqual(config.ppo.gat_attention_dropout, 0.0)
        self.assertEqual(config.curriculum.courses[0].rollout_steps, 256)
        self.assertEqual(config.curriculum.courses[1].rollout_steps, 640)
        self.assertEqual(config.curriculum.courses[2].rollout_steps, 1152)
        self.assertEqual(config.curriculum.courses[3].rollout_steps, 2048)
        self.assertEqual(config.curriculum.courses[2].total_timesteps, 1800000)
        self.assertEqual(config.curriculum.courses[3].total_timesteps, 4400000)
        self.assertEqual(config.ppo.mini_batch_size, 256)
        self.assertEqual(config.train.eval_interval, 10)
        self.assertEqual(config.train.checkpoint_interval, 10)
        self.assertFalse(config.curriculum.courses[0].load_previous)

    def test_course_obstacle_ratios_are_independent_overrides(self) -> None:
        config_path = ROOT / ".tmp_tests" / "course-obstacle-overrides.toml"
        shutil.rmtree(config_path.parent, ignore_errors=True)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            config_path.write_text(
                "\n".join(
                    [
                        "[env]",
                        "obstacle_ratio = 0.01",
                        "",
                        "[[curriculum.courses]]",
                        'name = "c1"',
                        "width = 4",
                        "height = 4",
                        "obstacle_ratio = 0.10",
                        "",
                        "[[curriculum.courses]]",
                        'name = "c2"',
                        "width = 5",
                        "height = 5",
                        "obstacle_ratio = 0.20",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.env.obstacle_ratio, 0.01)
            self.assertIsNotNone(config.curriculum)
            assert config.curriculum is not None
            self.assertEqual([course.env.obstacle_ratio for course in config.curriculum.courses], [0.10, 0.20])
        finally:
            shutil.rmtree(config_path.parent, ignore_errors=True)

    def test_gat_ablation_configs_only_differ_by_attention_arm(self) -> None:
        gat_on = load_config(ROOT / "configs" / "ablation_gat_on.toml")
        gat_off = load_config(ROOT / "configs" / "ablation_gat_off.toml")

        self.assertTrue(gat_on.ppo.use_graph_attention)
        self.assertFalse(gat_off.ppo.use_graph_attention)
        self.assertEqual(gat_on.ppo.gat_num_heads, gat_off.ppo.gat_num_heads)
        self.assertEqual(gat_on.ppo.gat_use_edge_features, gat_off.ppo.gat_use_edge_features)
        self.assertEqual(gat_on.ppo.gat_residual, gat_off.ppo.gat_residual)
        self.assertIsNotNone(gat_on.curriculum)
        self.assertIsNotNone(gat_off.curriculum)
        self.assertEqual(gat_on.train.run_root, "E:\\test plot\\ablation_gat_on")
        self.assertEqual(gat_off.train.run_root, "E:\\test plot\\ablation_gat_off")
        self.assertEqual(
            [(course.env.width, course.env.height, course.env.num_agents, course.env.obstacle_ratio) for course in gat_on.curriculum.courses],
            [(course.env.width, course.env.height, course.env.num_agents, course.env.obstacle_ratio) for course in gat_off.curriculum.courses],
        )

    def test_mapmsg_ablation_configs_share_memory_and_message_design(self) -> None:
        gat_on = load_config(ROOT / "configs" / "ablation_mapmsg_gat_on.toml")
        gat_off = load_config(ROOT / "configs" / "ablation_mapmsg_gat_off.toml")

        self.assertTrue(gat_on.env.use_explicit_map_memory)
        self.assertTrue(gat_on.env.share_map_memory)
        self.assertTrue(gat_on.ppo.use_coverage_messages)
        self.assertTrue(gat_on.ppo.use_graph_attention)
        self.assertFalse(gat_off.ppo.use_graph_attention)
        self.assertEqual(gat_on.env.intent_grid_size, gat_off.env.intent_grid_size)
        self.assertEqual(gat_on.ppo.use_coverage_messages, gat_off.ppo.use_coverage_messages)
        self.assertEqual(
            [(course.env.width, course.env.height, course.env.num_agents, course.total_timesteps) for course in gat_on.curriculum.courses],
            [(course.env.width, course.env.height, course.env.num_agents, course.total_timesteps) for course in gat_off.curriculum.courses],
        )

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
            self.assertEqual(loaded.ppo.rollout_steps, course_config.ppo.rollout_steps)
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

    def test_distance_field_matches_nearest_uncovered_bfs(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=5,
                start=(0, 0),
                obstacles=[(1, 2), (2, 2), (3, 2)],
            )
        )
        env.reset()
        uncovered = {(0, 4), (4, 4)}
        covered = set(env.free_cells) - uncovered
        distance_field = env._distance_field_to_uncovered(covered)

        def brute_distance(start: tuple[int, int]) -> int | None:
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                position, distance = queue.popleft()
                if position in uncovered:
                    return distance
                for delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    neighbor = (position[0] + delta[0], position[1] + delta[1])
                    if neighbor in visited or neighbor not in env.free_cells:
                        continue
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
            return None

        for position in env.free_cells:
            self.assertEqual(env._distance_from_field(distance_field, position), brute_distance(position))
            self.assertEqual(env._distance_to_nearest_uncovered(position, covered), brute_distance(position))

        complete_field = env._distance_field_to_uncovered(set(env.free_cells))
        self.assertEqual(env._distance_from_field(complete_field, (0, 0)), 0)

    def test_frontier_progress_distance_is_skipped_when_weight_is_zero(self) -> None:
        reward = RewardConfig(team_frontier_weight=0.0)
        env = GridCoverageEnv(GridCoverageConfig(width=4, height=4, start=(0, 0), reward=reward))
        env.reset()

        def fail_distance(*_args: object, **_kwargs: object) -> int:
            raise AssertionError("frontier distance should be skipped")

        env._distance_to_nearest_uncovered = fail_distance  # type: ignore[method-assign]
        result = env.step(3)

        self.assertEqual(result.info["reward_terms"]["frontier_progress"], 0.0)

    def test_multi_agent_frontier_progress_distance_is_skipped_when_weight_is_zero(self) -> None:
        reward = RewardConfig(team_frontier_weight=0.0)
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=4,
                height=4,
                num_agents=2,
                start_positions=[(0, 0), (3, 3)],
                reward=reward,
            )
        )
        env.reset()

        def fail_distances(*_args: object, **_kwargs: object) -> list[int]:
            raise AssertionError("frontier distances should be skipped")

        env._distances_to_nearest_uncovered = fail_distances  # type: ignore[method-assign]
        result = env.step([3, 2])

        self.assertEqual(result.info["reward_terms"]["frontier_progress"], 0.0)

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

    def test_obstacle_seed_pool_rotates_by_refresh_episodes(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=8,
                height=8,
                obstacle_ratio=0.0625,
                random_obstacle_seeds=[101, 202],
                map_refresh_episodes=2,
            )
        )

        env.reset(seed=7)
        first_map = set(env.obstacles)
        env.reset()
        second_map = set(env.obstacles)
        env.reset()
        third_map = set(env.obstacles)
        env.reset()
        fourth_map = set(env.obstacles)

        self.assertEqual(first_map, second_map)
        self.assertEqual(third_map, fourth_map)
        self.assertNotEqual(first_map, third_map)
        self.assertEqual(len(first_map), 4)
        self.assertEqual(len(third_map), 4)

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

    def test_explicit_map_memory_stays_private_until_agents_communicate(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=4,
                height=1,
                num_agents=2,
                start_positions=[(0, 0), (0, 3)],
                observation_radius=0,
                communication_radius=1,
                use_explicit_map_memory=True,
                share_map_memory=True,
            )
        )
        observation = env.reset()

        self.assertEqual(observation.shape, (2, env.observation_dim))
        self.assertEqual(env.observation_dim, 21)
        self.assertEqual(env.known_team_covered_by_agent[0], {(0, 0)})
        self.assertEqual(env.known_team_covered_by_agent[1], {(0, 3)})

        env.step([3, 2])
        expected_covered = {(0, 0), (0, 1), (0, 2), (0, 3)}
        self.assertEqual(env.known_team_covered_by_agent[0], expected_covered)
        self.assertEqual(env.known_team_covered_by_agent[1], expected_covered)

    def test_coverage_message_contains_memory_derived_intent(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=3,
                height=3,
                start=(1, 1),
                observation_radius=1,
                use_explicit_map_memory=True,
                intent_grid_size=3,
            )
        )
        env.reset()
        messages = env.node_messages()

        self.assertEqual(messages.shape, (1, 24))
        self.assertEqual(messages[0, -1], 1.0)
        self.assertTrue(np.array_equal(messages[0, 7:11], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))
        self.assertEqual(messages[0, 15], 1.0)

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

    def test_neighbor_features_include_relative_geometry_and_connectivity(self) -> None:
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
        features = env.neighbor_features()

        self.assertEqual(features.shape, (3, 3, env.neighbor_feature_dim))
        self.assertEqual(features[0, 0, 0], 0.0)
        self.assertEqual(features[0, 0, 3], 1.0)
        self.assertAlmostEqual(float(features[0, 1, 0]), 1.0)
        self.assertAlmostEqual(float(features[0, 1, 2]), 0.5)
        self.assertEqual(features[0, 1, 3], 1.0)
        self.assertEqual(features[0, 2, 3], 0.0)

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
            gat_num_heads=4,
            gat_edge_dim=env.neighbor_feature_dim,
            gat_residual=True,
        )
        actions, log_probs, values = model.act_batch(
            torch.as_tensor(observation, dtype=torch.float32),
            torch.as_tensor(state, dtype=torch.float32),
            neighbor_mask=torch.as_tensor(env.neighbor_mask(), dtype=torch.bool),
            edge_features=torch.as_tensor(env.neighbor_features(), dtype=torch.float32),
        )

        self.assertEqual(actions.shape, (3,))
        self.assertEqual(log_probs.shape, (3,))
        self.assertEqual(values.shape, (3,))
        attention = model.latest_attention_weights()
        self.assertIsNotNone(attention)
        assert attention is not None
        self.assertEqual(attention.shape, (1, 4, 3, 3))
        mask = torch.as_tensor(env.neighbor_mask(), dtype=torch.bool)
        self.assertTrue(torch.all(attention[:, :, ~mask] == 0.0).item())

    def test_graph_attention_policy_consumes_coverage_messages(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=5,
                num_agents=2,
                start_positions=[(0, 0), (0, 2)],
                observation_radius=1,
                communication_radius=3,
                use_explicit_map_memory=True,
                share_map_memory=True,
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
            gat_num_heads=4,
            gat_edge_dim=env.neighbor_feature_dim,
            gat_residual=True,
            node_message_dim=env.node_message_dim,
        )
        actions, log_probs, values = model.act_batch(
            torch.as_tensor(observation, dtype=torch.float32),
            torch.as_tensor(state, dtype=torch.float32),
            neighbor_mask=torch.as_tensor(env.neighbor_mask(), dtype=torch.bool),
            edge_features=torch.as_tensor(env.neighbor_features(), dtype=torch.float32),
            node_messages=torch.as_tensor(env.node_messages(), dtype=torch.float32),
        )

        self.assertEqual(actions.shape, (2,))
        self.assertEqual(log_probs.shape, (2,))
        self.assertEqual(values.shape, (2,))
        self.assertEqual(model.node_message_dim, 24)
        self.assertIsNotNone(model.latest_attention_weights())

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
