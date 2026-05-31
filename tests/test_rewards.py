from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.config import GridCoverageConfig
from mathbased_mcpp.env import GridCoverageEnv
from mathbased_mcpp.safety import SafetyLayer


class RewardStructureTests(unittest.TestCase):
    def test_distance_and_straight_rewards(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=4, height=3, start=(1, 1)))
        env.reset()
        result = env.step(3)
        result = env.step(3)
        self.assertAlmostEqual(result.info["reward_terms"]["Rd"], 1.0)
        self.assertAlmostEqual(result.info["reward_terms"]["Rs"], 1.0)
        self.assertLess(result.info["reward_terms"]["time"], 0.0)

    def test_coverage_reward_prefers_boundary_move(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=3,
                height=3,
                start=(1, 1),
                obstacles=[(0, 2), (2, 2)],
            )
        )
        env.reset()
        result = env.step(3)
        self.assertAlmostEqual(result.info["reward_terms"]["Rb"], 1.0)

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

    def test_multi_agent_avoidable_repeat_only_penalizes_waste(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(width=4, height=2, num_agents=2, start_positions=[(0, 0), (1, 3)], max_steps=8)
        )
        env.reset()
        env.step([3, 2])
        result = env.step([2, 2])
        self.assertEqual(result.info["reward_terms"]["avoidable_repeats"], 1.0)
        self.assertAlmostEqual(float(result.reward[0]), float(result.reward[1]))

    def test_multi_agent_finish_reward_is_shared(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(width=2, height=1, num_agents=2, start_positions=[(0, 0), (0, 1)], max_steps=1)
        )
        env.reset()
        result = env.step([0, 1])
        self.assertTrue(result.done)
        self.assertGreaterEqual(result.info["reward_terms"]["finish"], env.config.reward.finish_reward)
        self.assertAlmostEqual(float(result.reward[0]), float(result.reward[1]))

    def test_safety_layer_prefers_uncovered_neighbor_over_repeat(self) -> None:
        env = GridCoverageEnv(GridCoverageConfig(width=3, height=1, start=(0, 0)))
        env.reset()
        env.step(3)
        action = SafetyLayer().filter_action(env, 2)
        self.assertEqual(action, 3)


if __name__ == "__main__":
    unittest.main()
