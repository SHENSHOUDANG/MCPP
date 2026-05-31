"""配置、环境观测与通信信息边界的回归测试。

阅读核心环境代码后，可以用本文件确认设计约束是否真的成立，尤其是
“未通信时 actor 不能知道队友覆盖历史”这一去中心化要求。
"""

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
from mathbased_mcpp.evaluation import load_policy, resolve_runtime_config
from mathbased_mcpp.env import GridCoverageEnv
from mathbased_mcpp.ppo import ActorCritic

import torch


class ConfigEnvTests(unittest.TestCase):
    """验证配置解析、地图生成、观测隐私、GAT 输入与兼容加载路径。"""

    def test_load_smoke_config(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        self.assertEqual(config.env.width, 6)
        self.assertEqual(config.env.height, 6)
        self.assertEqual(config.env.reward.finish_reward, 10.0)
        self.assertFalse(config.env.reward.scale_time_cost_by_uncovered)
        self.assertEqual(config.env.reward.team_straight_weight, 0.01)
        self.assertEqual(config.env.reward.team_repeat_weight, 0.3)
        self.assertEqual(config.env.reward.team_invalid_weight, 1.0)
        self.assertEqual(config.ppo.rollout_steps, 64)
        self.assertTrue(config.ppo.use_action_mask)

    def test_load_current_curriculum_config(self) -> None:
        config = load_config(ROOT / "configs" / "ablation_mapmsg_gat_on.toml")
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
        self.assertFalse(config.env.use_legacy_truth_coverage_observation)
        self.assertTrue(config.env.use_explicit_map_memory)
        self.assertTrue(config.env.share_map_memory)
        self.assertTrue(config.ppo.use_graph_attention)
        self.assertTrue(config.ppo.use_coverage_messages)
        self.assertTrue(config.ppo.use_action_mask)
        self.assertEqual(config.ppo.gat_num_heads, 4)
        self.assertTrue(config.ppo.gat_use_edge_features)
        self.assertTrue(config.ppo.gat_residual)
        self.assertEqual(config.ppo.gat_attention_dropout, 0.0)
        self.assertEqual(config.curriculum.courses[0].rollout_steps, 256)
        self.assertEqual(config.curriculum.courses[1].rollout_steps, 640)
        self.assertEqual(config.curriculum.courses[2].rollout_steps, 1152)
        self.assertEqual(config.curriculum.courses[3].rollout_steps, 2048)
        self.assertEqual(config.curriculum.courses[2].total_timesteps, 1800000)
        self.assertEqual(config.curriculum.courses[3].total_timesteps, 3200000)
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

    def test_archived_gat_configs_only_differ_by_attention_arm(self) -> None:
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
        self.assertEqual(gat_on.env.reward.finish_reward, 10.0)
        self.assertTrue(gat_on.env.reward.normalize_team_finish_reward)
        self.assertFalse(gat_on.env.reward.scale_time_cost_by_uncovered)
        self.assertTrue(gat_on.ppo.use_action_mask)
        self.assertEqual(gat_on.env.reward.finish_reward, gat_off.env.reward.finish_reward)
        self.assertEqual(
            gat_on.env.reward.normalize_team_finish_reward,
            gat_off.env.reward.normalize_team_finish_reward,
        )
        self.assertEqual(
            gat_on.env.reward.scale_time_cost_by_uncovered,
            gat_off.env.reward.scale_time_cost_by_uncovered,
        )
        self.assertEqual(gat_on.ppo.use_action_mask, gat_off.ppo.use_action_mask)
        self.assertEqual(gat_on.env.intent_grid_size, gat_off.env.intent_grid_size)
        self.assertEqual(gat_on.ppo.use_coverage_messages, gat_off.ppo.use_coverage_messages)
        self.assertEqual(
            [(course.env.width, course.env.height, course.env.num_agents, course.total_timesteps) for course in gat_on.curriculum.courses],
            [(course.env.width, course.env.height, course.env.num_agents, course.total_timesteps) for course in gat_off.curriculum.courses],
        )

    def test_centered_cnn_ablation_configs_use_fixed_actor_map(self) -> None:
        gat_on = load_config(ROOT / "configs" / "ablation_centered_cnn_gat_on.toml")
        gat_off = load_config(ROOT / "configs" / "ablation_centered_cnn_gat_off.toml")

        self.assertEqual(gat_on.env.observation_mode, "centered_compressed_memory")
        self.assertEqual(gat_on.env.centered_map_size, 15)
        self.assertEqual(gat_on.ppo.actor_encoder, "cnn")
        self.assertTrue(gat_on.ppo.use_graph_attention)
        self.assertFalse(gat_off.ppo.use_graph_attention)
        self.assertEqual(gat_on.ppo.actor_encoder, gat_off.ppo.actor_encoder)
        self.assertEqual(gat_on.env.centered_map_size, gat_off.env.centered_map_size)
        self.assertEqual(gat_on.env.observation_mode, gat_off.env.observation_mode)
        assert gat_on.curriculum is not None
        for course in gat_on.curriculum.courses:
            env = GridCoverageEnv(course.env)
            self.assertEqual(course.env.observation_mode, "centered_compressed_memory")
            self.assertEqual(env.actor_map_shape, (9, 15, 15))
            self.assertEqual(env.observation_dim, 9 * 15 * 15 + env.observation_metadata_dim)

    def test_select_curriculum_course_by_name(self) -> None:
        config = load_config(ROOT / "configs" / "ablation_mapmsg_gat_on.toml")
        index, course = select_curriculum_course(config, course_name="tier-2-13x13-2agents")
        self.assertEqual(index, 1)
        self.assertEqual(course.name, "tier-2-13x13-2agents")

    def test_course_config_snapshot_roundtrip(self) -> None:
        config = load_config(ROOT / "configs" / "ablation_mapmsg_gat_on.toml")
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
            self.assertEqual(loaded.env.reward.team_straight_weight, course_config.env.reward.team_straight_weight)
            self.assertEqual(loaded.ppo.total_timesteps, course_config.ppo.total_timesteps)
            self.assertEqual(loaded.ppo.rollout_steps, course_config.ppo.rollout_steps)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_local_observation_shape(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=6, height=6, start=(0, 0)))
        observation = env.reset()
        self.assertEqual(observation.shape[0], env.observation_dim)
        self.assertEqual(env.observation_dim, 66)

    def test_local_window_matches_zero_padded_slice(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=4, height=3, start=(0, 0)))
        grid = np.arange(12, dtype=np.float32).reshape(3, 4)
        for radius in (0, 1, 2, 4):
            for center in ((0, 0), (1, 2), (2, 3)):
                with self.subTest(radius=radius, center=center):
                    expected = np.pad(grid, radius, mode="constant")[
                        center[0] : center[0] + radius * 2 + 1,
                        center[1] : center[1] + radius * 2 + 1,
                    ]
                    self.assertTrue(np.array_equal(env._local_window(grid, radius, center), expected))

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
        self.assertEqual(env.observation_dim, 66)
        self.assertEqual(env.state_dim, 5 * 5 * 5 + 7)

    def test_private_observation_includes_self_memory_channels(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(width=3, height=3, start=(1, 1), observation_radius=1, recent_path_length=4)
        )
        env.reset()
        result = env.step(3)
        window_area = (env.config.observation_radius * 2 + 1) ** 2
        self_covered = result.observation[4 * window_area : 5 * window_area].reshape(3, 3)
        recent_path = result.observation[5 * window_area : 6 * window_area].reshape(3, 3)

        self.assertEqual(float(self_covered.sum()), 2.0)
        self.assertEqual(self_covered[1, 1], 1.0)
        self.assertEqual(self_covered[1, 0], 1.0)
        self.assertEqual(recent_path[1, 1], 1.0)
        self.assertGreater(recent_path[1, 1], recent_path[1, 0])

    def test_new_actor_observations_expose_last_effective_move_direction(self) -> None:
        for use_explicit_map_memory in (False, True):
            with self.subTest(use_explicit_map_memory=use_explicit_map_memory):
                env = GridCoverageEnv(
                    GridCoverageConfig(
                        width=3,
                        height=2,
                        start=(0, 0),
                        observation_radius=1,
                        use_explicit_map_memory=use_explicit_map_memory,
                    )
                )
                env.reset()
                env.step(3)
                moved = env.step(3)
                blocked = env.step(3)
                moved_metadata = moved.observation[-env.observation_metadata_dim :]
                blocked_metadata = blocked.observation[-env.observation_metadata_dim :]

                self.assertTrue(np.array_equal(moved_metadata[4:6], np.array([0.0, 1.0], dtype=np.float32)))
                self.assertTrue(np.array_equal(blocked_metadata[4:6], np.array([0.0, 1.0], dtype=np.float32)))

    def test_private_observation_does_not_reveal_teammate_coverage(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=4,
                height=1,
                num_agents=2,
                start_positions=[(0, 0), (0, 3)],
                observation_radius=3,
                communication_radius=0,
            )
        )
        env.reset()
        result = env.step([0, 2])
        window_area = (env.config.observation_radius * 2 + 1) ** 2
        first_agent = result.observation[0]
        self_uncovered = first_agent[2 * window_area : 3 * window_area].reshape(7, 7)
        metadata = first_agent[6 * window_area :]

        self.assertEqual(self_uncovered[3, 5], 1.0)
        self.assertAlmostEqual(float(metadata[2]), 0.25)
        self.assertAlmostEqual(env.coverage_ratio(), 0.75)

    def test_private_observation_is_invariant_to_teammate_coverage_history(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=1,
                num_agents=2,
                start_positions=[(0, 0), (0, 4)],
                observation_radius=4,
                communication_radius=0,
            )
        )
        env.reset()
        before = env._observations()[0].copy()

        hidden_teammate_history = {(0, 2), (0, 3)}
        env.covered.update(hidden_teammate_history)
        env.covered_by_agent[1].update(hidden_teammate_history)
        after = env._observations()[0]

        self.assertTrue(np.array_equal(before, after))
        self.assertNotAlmostEqual(env.coverage_ratio(), 2 / 5)

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

    def test_centered_memory_observation_shape_is_map_size_invariant(self) -> None:
        base_kwargs = dict(
            start=(3, 3),
            observation_radius=1,
            use_explicit_map_memory=True,
            observation_mode="centered_compressed_memory",
            centered_map_size=7,
        )
        small_env = GridCoverageEnv(GridCoverageConfig(width=8, height=8, **base_kwargs))
        large_env = GridCoverageEnv(GridCoverageConfig(width=30, height=30, **base_kwargs))

        self.assertEqual(small_env.observation_dim, large_env.observation_dim)
        self.assertEqual(small_env.observation_dim, 7 * 7 * small_env.explicit_observation_channels + 12)
        self.assertEqual(small_env.reset().shape[0], small_env.observation_dim)
        self.assertEqual(large_env.reset().shape[0], large_env.observation_dim)

    def test_centered_memory_compresses_known_remote_cells_into_border(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=5,
                start=(2, 2),
                observation_radius=0,
                use_explicit_map_memory=True,
                observation_mode="centered_compressed_memory",
                centered_map_size=5,
            )
        )
        env.reset()
        env.known_obstacles_by_agent[0].add((0, 2))
        observation = env._observations()[0]
        area = env.config.centered_map_size**2
        obstacle_channel = observation[4 * area : 5 * area].reshape(5, 5)

        self.assertGreater(obstacle_channel[0, 2], 0.0)
        self.assertEqual(obstacle_channel[2, 2], 0.0)

    def test_centered_memory_observation_ignores_uncommunicated_team_coverage(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=7,
                height=1,
                num_agents=2,
                start_positions=[(0, 0), (0, 6)],
                observation_radius=0,
                communication_radius=0,
                use_explicit_map_memory=True,
                share_map_memory=True,
                observation_mode="centered_compressed_memory",
                centered_map_size=5,
            )
        )
        env.reset()
        before = env._observations()[0].copy()
        hidden_teammate_history = {(0, 4), (0, 5)}
        env.covered.update(hidden_teammate_history)
        env.covered_by_agent[1].update(hidden_teammate_history)
        env._refresh_explicit_map_memory()
        after = env._observations()[0]

        self.assertTrue(np.array_equal(before, after))
        self.assertEqual(env.known_team_covered_by_agent[0], {(0, 0)})

    def test_explicit_memory_does_not_read_visible_teammate_coverage_without_communication(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=4,
                height=1,
                num_agents=2,
                start_positions=[(0, 0), (0, 3)],
                observation_radius=3,
                communication_radius=0,
                use_explicit_map_memory=True,
                share_map_memory=True,
            )
        )
        env.reset()
        env.step([0, 2])

        self.assertEqual(env.known_team_covered_by_agent[0], {(0, 0)})
        self.assertEqual(env.known_team_covered_by_agent[1], {(0, 2), (0, 3)})

    def test_explicit_actor_inputs_are_invariant_to_uncommunicated_teammate_history(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=1,
                num_agents=2,
                start_positions=[(0, 0), (0, 4)],
                observation_radius=4,
                communication_radius=0,
                use_explicit_map_memory=True,
                share_map_memory=True,
            )
        )
        env.reset()
        before_observation = env._observations()[0].copy()
        before_message = env.node_messages()[0].copy()

        hidden_teammate_history = {(0, 2), (0, 3)}
        env.covered.update(hidden_teammate_history)
        env.covered_by_agent[1].update(hidden_teammate_history)
        env._refresh_explicit_map_memory()
        after_observation = env._observations()[0]
        after_message = env.node_messages()[0]

        self.assertTrue(np.array_equal(before_observation, after_observation))
        self.assertTrue(np.array_equal(before_message, after_message))
        self.assertEqual(env.known_team_covered_by_agent[0], {(0, 0)})
        self.assertNotAlmostEqual(env.coverage_ratio(), 2 / 5)

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

    def test_explicit_observation_caches_same_step_coverage_messages(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=4,
                height=2,
                num_agents=2,
                start_positions=[(0, 0), (1, 3)],
                observation_radius=1,
                communication_radius=4,
                use_explicit_map_memory=True,
                share_map_memory=True,
            )
        )
        env.reset()
        original = env._coverage_message
        message_calls = 0

        def tracked_message(*args: object, **kwargs: object) -> np.ndarray:
            nonlocal message_calls
            message_calls += 1
            return original(*args, **kwargs)

        env._coverage_message = tracked_message  # type: ignore[method-assign]
        env.step([3, 2])
        first = env.node_messages()
        second = env.node_messages()

        self.assertEqual(message_calls, env.num_agents)
        self.assertTrue(np.array_equal(first, second))

        before_preview = first.copy()
        env.reset_preview()
        self.assertTrue(np.array_equal(env.node_messages(), before_preview))

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

    def test_action_mask_uses_known_obstacles_without_oracle_lookahead(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=3,
                height=3,
                start=(1, 1),
                observation_radius=0,
                obstacles=[(1, 2)],
                use_explicit_map_memory=True,
            )
        )
        env.reset()
        self.assertTrue(env.action_mask()[0, 3])

        env.known_obstacles_by_agent[0].add((1, 2))
        self.assertFalse(env.action_mask()[0, 3])

    def test_policy_action_mask_removes_infeasible_argmax_action(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=3, height=3, start=(1, 1)))
        observation = torch.as_tensor(env.reset(), dtype=torch.float32)
        state = torch.as_tensor(env.global_state(), dtype=torch.float32)
        model = ActorCritic(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=16,
            state_shape=(env.config.height, env.config.width),
        )
        with torch.no_grad():
            model.actor.weight.zero_()
            model.actor.bias[:] = torch.tensor([100.0, 0.0, 0.0, 0.0])
        action, _, _ = model.act(
            observation,
            state,
            action_mask=torch.tensor([False, True, False, False]),
            deterministic=True,
        )
        self.assertEqual(action, 1)

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

    def test_policy_broadcasts_one_central_state_per_environment_step(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=5,
                num_agents=3,
                start_positions=[(0, 0), (0, 2), (4, 4)],
                communication_radius=3,
            )
        )
        observation = torch.as_tensor(env.reset(), dtype=torch.float32)
        state = torch.as_tensor(env.global_state(), dtype=torch.float32)
        model = ActorCritic(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=16,
            state_shape=(env.config.height, env.config.width),
            use_graph_attention=True,
            gat_num_heads=4,
        )
        mask = torch.as_tensor(env.neighbor_mask(), dtype=torch.bool)
        repeated_state = state.unsqueeze(0).expand(env.num_agents, -1)
        batch_observation = observation.unsqueeze(0).expand(2, -1, -1)
        batch_state = state.unsqueeze(0).expand(2, -1)
        repeated_batch_state = batch_state.unsqueeze(1).expand(-1, env.num_agents, -1)

        with torch.no_grad():
            shared_logits, shared_values = model(observation, state, neighbor_mask=mask)
            repeated_logits, repeated_values = model(observation, repeated_state, neighbor_mask=mask)
            batch_logits, batch_values = model(batch_observation, batch_state, neighbor_mask=mask)
            repeated_batch_logits, repeated_batch_values = model(batch_observation, repeated_batch_state, neighbor_mask=mask)

        self.assertTrue(torch.allclose(shared_logits, repeated_logits))
        self.assertTrue(torch.allclose(shared_values, repeated_values))
        self.assertTrue(torch.allclose(batch_logits, repeated_batch_logits))
        self.assertTrue(torch.allclose(batch_values, repeated_batch_values))

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

    def test_cnn_actor_consumes_centered_memory_observation(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=10,
                height=10,
                num_agents=2,
                start_positions=[(3, 3), (3, 5)],
                observation_radius=1,
                communication_radius=3,
                use_explicit_map_memory=True,
                observation_mode="centered_compressed_memory",
                centered_map_size=7,
            )
        )
        observation = env.reset()
        state = np.repeat(env.global_state()[None, :], env.num_agents, axis=0)
        model = ActorCritic(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=16,
            state_shape=(env.config.height, env.config.width),
            actor_encoder="cnn",
            actor_map_shape=env.actor_map_shape,
            actor_metadata_dim=env.observation_metadata_dim,
        )
        actions, log_probs, values = model.act_batch(
            torch.as_tensor(observation, dtype=torch.float32),
            torch.as_tensor(state, dtype=torch.float32),
        )

        self.assertEqual(model.actor_encoder, "cnn")
        self.assertEqual(model.actor_map_shape, env.actor_map_shape)
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(log_probs.shape, (2,))
        self.assertEqual(values.shape, (2,))

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
        self.assertEqual(result.info["reward_terms"]["invalid_moves"], 2.0)
        self.assertEqual(result.info["reward_terms"]["agent_collision_invalid_moves"], 2.0)

    def test_multi_agent_swap_collision_blocks_both(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(width=3, height=1, num_agents=2, start_positions=[(0, 0), (0, 1)], max_steps=5)
        )
        env.reset()
        result = env.step([3, 2])
        self.assertEqual(env.positions, [(0, 0), (0, 1)])
        self.assertEqual(result.info["reward_terms"]["collision_agents"], 2.0)
        self.assertEqual(result.info["reward_terms"]["invalid_moves"], 2.0)

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

    def test_cnn_actor_checkpoint_can_load_on_larger_map(self) -> None:
        source_env = GridCoverageEnv(
            GridCoverageConfig(
                width=8,
                height=8,
                start=(3, 3),
                observation_radius=1,
                use_explicit_map_memory=True,
                observation_mode="centered_compressed_memory",
                centered_map_size=7,
            )
        )
        source_model = ActorCritic(
            observation_dim=source_env.observation_dim,
            action_dim=source_env.action_dim,
            hidden_dim=16,
            state_shape=(source_env.config.height, source_env.config.width),
            actor_encoder="cnn",
            actor_map_shape=source_env.actor_map_shape,
            actor_metadata_dim=source_env.observation_metadata_dim,
        )

        run_dir = ROOT / ".tmp_tests" / "cnn-actor-policy-load"
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
                    "actor_encoder": source_model.actor_encoder,
                    "actor_map_shape": source_model.actor_map_shape,
                    "actor_metadata_dim": source_model.actor_metadata_dim,
                },
                checkpoint_path,
            )

            config = load_config(ROOT / "configs" / "smoke.toml")
            config.env.width = 24
            config.env.height = 24
            config.env.start = (3, 3)
            config.env.use_explicit_map_memory = True
            config.env.observation_mode = "centered_compressed_memory"
            config.env.centered_map_size = 7
            model = load_policy(config, checkpoint_path)

            target_env = GridCoverageEnv(config.env)
            observation = target_env.reset()
            state = target_env.global_state()
            action, _, value = model.act(
                torch.as_tensor(observation, dtype=torch.float32),
                torch.as_tensor(state, dtype=torch.float32),
            )

            self.assertEqual(model.actor_encoder, "cnn")
            self.assertEqual(model.actor_map_shape, source_env.actor_map_shape)
            self.assertEqual(target_env.observation_dim, source_env.observation_dim)
            self.assertIn(action, range(target_env.action_dim))
            self.assertTrue(torch.isfinite(value).item())
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_legacy_snapshot_replays_truth_coverage_observation_explicitly(self) -> None:
        run_dir = ROOT / ".tmp_tests" / "legacy-runtime-config"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint = run_dir / "policy.pt"
            checkpoint.write_bytes(b"placeholder")
            (run_dir / "course_config.json").write_text(
                json.dumps({"env": {"width": 4, "height": 4}, "ppo": {}, "train": {}}),
                encoding="utf-8",
            )
            runtime = resolve_runtime_config(load_config(ROOT / "configs" / "smoke.toml"), checkpoint)
            self.assertTrue(runtime.env.use_legacy_truth_coverage_observation)
            self.assertEqual(GridCoverageEnv(runtime.env).observation_dim, 75)
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
        reward = RewardConfig(
            finish_reward=10.0,
            team_straight_weight=0.0,
            team_frontier_weight=0.0,
            team_repeat_weight=0.0,
            team_invalid_weight=0.0,
            team_time_weight=0.0,
        )
        env = GridCoverageEnv(GridCoverageConfig(width=2, height=1, start=(0, 0), reward=reward))
        env.reset()
        result = env.step(3)
        self.assertTrue(result.done)
        self.assertEqual(result.info["reward_terms"]["finish"], 10.0)
        self.assertGreater(result.reward, 10.0)


if __name__ == "__main__":
    unittest.main()
