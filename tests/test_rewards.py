"""奖励组成与规则式安全层的行为测试。"""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.config import GridCoverageConfig, RewardConfig
from mathbased_mcpp.env import GridCoverageEnv
from mathbased_mcpp.safety import SafetyLayer


class RewardStructureTests(unittest.TestCase):
    """验证新覆盖、重复/非法移动处罚以及安全过滤的基本倾向。"""

    def test_single_agent_straight_reward_is_a_tiny_path_preference(self) -> None:
        reward = RewardConfig(
            finish_reward=0.0,
            team_new_cell_weight=0.0,
            team_straight_weight=0.01,
            team_frontier_weight=0.0,
            team_repeat_weight=0.0,
            team_invalid_weight=0.0,
            team_time_weight=0.0,
        )
        env = GridCoverageEnv(GridCoverageConfig(width=4, height=3, start=(1, 1), reward=reward))
        env.reset()
        result = env.step(3)
        result = env.step(3)
        self.assertEqual(result.info["reward_terms"]["straight_moves"], 1.0)
        self.assertAlmostEqual(result.info["reward_terms"]["straight_bonus"], 0.01)
        self.assertAlmostEqual(result.reward, 0.01)

    def test_new_cell_reward_is_reported_by_active_team_formula(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=3, height=3, start=(1, 1)))
        env.reset()
        result = env.step(3)
        self.assertAlmostEqual(result.info["reward_terms"]["new_cells"], 1.0)

    def test_repeat_penalty_is_light_and_negative(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=3, height=1, start=(0, 0)))
        env.reset()
        env.step(3)
        result = env.step(2)
        self.assertLess(result.info["reward_terms"]["repeat"], 0.0)
        self.assertLess(result.info["reward_terms"]["time"], 0.0)
        self.assertEqual(result.info["reward_terms"]["avoidable_repeats"], 1.0)

    def test_invalid_move_is_penalized(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=3, height=1, start=(0, 0)))
        env.reset()
        result = env.step(0)
        self.assertLess(result.reward, 0)

    def test_fixed_time_cost_does_not_discount_tail_search(self) -> None:
        reward = RewardConfig(
            finish_reward=0.0,
            team_new_cell_weight=0.0,
            team_straight_weight=0.0,
            team_frontier_weight=0.0,
            team_repeat_weight=0.0,
            team_invalid_weight=0.0,
            team_time_weight=0.02,
            scale_time_cost_by_uncovered=False,
        )
        env = GridCoverageEnv(GridCoverageConfig(width=3, height=1, start=(0, 0), reward=reward))
        env.reset()
        first = env.step(3)
        second = env.step(3)
        self.assertAlmostEqual(first.info["reward_terms"]["time"], -0.02)
        self.assertAlmostEqual(second.info["reward_terms"]["time"], -0.02)

    def test_legacy_time_cost_can_still_scale_with_uncovered_ratio(self) -> None:
        reward = RewardConfig(
            finish_reward=0.0,
            team_new_cell_weight=0.0,
            team_straight_weight=0.0,
            team_frontier_weight=0.0,
            team_repeat_weight=0.0,
            team_invalid_weight=0.0,
            team_time_weight=0.02,
        )
        env = GridCoverageEnv(GridCoverageConfig(width=3, height=1, start=(0, 0), reward=reward))
        env.reset()
        first = env.step(3)
        second = env.step(3)
        self.assertAlmostEqual(first.info["reward_terms"]["time"], -0.02 / 3.0)
        self.assertAlmostEqual(second.info["reward_terms"]["time"], 0.0)

    def test_multi_agent_avoidable_repeat_only_penalizes_waste(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(width=4, height=2, num_agents=2, start_positions=[(0, 0), (1, 3)], max_steps=8)
        )
        env.reset()
        env.step([3, 2])
        result = env.step([2, 2])
        self.assertEqual(result.info["reward_terms"]["avoidable_repeats"], 1.0)
        self.assertAlmostEqual(float(result.reward[0]), float(result.reward[1]))

    def test_multi_agent_straight_reward_uses_same_small_weight(self) -> None:
        reward = RewardConfig(
            finish_reward=0.0,
            team_new_cell_weight=0.0,
            team_straight_weight=0.01,
            team_frontier_weight=0.0,
            team_repeat_weight=0.0,
            team_invalid_weight=0.0,
            team_time_weight=0.0,
        )
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=2,
                num_agents=2,
                start_positions=[(0, 0), (1, 4)],
                max_steps=4,
                reward=reward,
            )
        )
        env.reset()
        env.step([3, 2])
        result = env.step([3, 2])
        self.assertEqual(result.info["reward_terms"]["straight_moves"], 2.0)
        self.assertAlmostEqual(result.info["reward_terms"]["straight_bonus"], 0.01)
        self.assertAlmostEqual(float(result.reward[0]), 0.01)
        self.assertAlmostEqual(float(result.reward[0]), float(result.reward[1]))

    def test_agent_collision_uses_the_same_invalid_penalty_as_obstacle_collision(self) -> None:
        reward = RewardConfig(
            finish_reward=0.0,
            team_new_cell_weight=0.0,
            team_straight_weight=0.0,
            team_frontier_weight=0.0,
            team_repeat_weight=0.0,
            team_invalid_weight=1.0,
            team_time_weight=0.0,
        )
        obstacle_env = GridCoverageEnv(
            GridCoverageConfig(width=2, height=1, start=(0, 0), obstacles=[(0, 1)], reward=reward)
        )
        obstacle_env.reset()
        obstacle_result = obstacle_env.step(3)

        collision_env = GridCoverageEnv(
            GridCoverageConfig(
                width=3,
                height=1,
                num_agents=2,
                start_positions=[(0, 0), (0, 2)],
                reward=reward,
            )
        )
        collision_env.reset()
        collision_result = collision_env.step([3, 2])

        self.assertEqual(obstacle_result.info["reward_terms"]["invalid_moves"], 1.0)
        self.assertEqual(obstacle_result.info["reward_terms"]["obstacle_or_boundary_invalid_moves"], 1.0)
        self.assertAlmostEqual(obstacle_result.reward, -1.0)
        self.assertEqual(collision_result.info["reward_terms"]["invalid_moves"], 2.0)
        self.assertEqual(collision_result.info["reward_terms"]["agent_collision_invalid_moves"], 2.0)
        self.assertAlmostEqual(float(collision_result.reward[0]), -1.0)

    def test_multi_agent_finish_reward_keeps_legacy_unscaled_default(self) -> None:
        reward = RewardConfig(finish_reward=20.0)
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=2,
                height=1,
                num_agents=2,
                start_positions=[(0, 0), (0, 1)],
                max_steps=1,
                reward=reward,
            )
        )
        env.reset()
        result = env.step([0, 1])
        self.assertTrue(result.done)
        self.assertEqual(result.info["reward_terms"]["finish"], env.config.reward.finish_reward)
        self.assertAlmostEqual(float(result.reward[0]), float(result.reward[1]))

    def test_multi_agent_normalized_finish_reward_distributes_team_bonus(self) -> None:
        reward = RewardConfig(finish_reward=20.0, normalize_team_finish_reward=True)
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=4,
                height=1,
                num_agents=2,
                start_positions=[(0, 0), (0, 2)],
                max_steps=1,
                reward=reward,
            )
        )
        env.reset()
        result = env.step([3, 3])
        self.assertTrue(result.done)
        self.assertEqual(result.info["reward_terms"]["finish_team_total"], 20.0)
        self.assertEqual(result.info["reward_terms"]["finish"], 10.0)
        self.assertAlmostEqual(float(result.reward[0]), float(result.reward[1]))

    def test_safety_layer_prefers_uncovered_neighbor_over_repeat(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=3, height=1, start=(0, 0)))
        env.reset()
        env.step(3)
        action = SafetyLayer().filter_action(env, 2)
        self.assertEqual(action, 3)


if __name__ == "__main__":
    unittest.main()
