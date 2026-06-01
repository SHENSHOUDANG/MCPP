from pathlib import Path
import shutil
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.config import ExperimentConfig, GridCoverageConfig, PPOConfig, TrainConfig
from mathbased_mcpp.env import GridCoverageEnv
from mathbased_mcpp.evaluation import evaluate_policy, load_policy
from mathbased_mcpp.imitation import BoustrophedonExpert, generate_expert_dataset, pretrain_imitation


class ImitationTests(unittest.TestCase):
    def test_boustrophedon_expert_avoids_obstacles_and_covers_cells(self) -> None:
        env = GridCoverageEnv(
            GridCoverageConfig(
                width=5,
                height=4,
                max_steps=20,
                start=(0, 0),
                obstacles=[(1, 1), (2, 1)],
                random_obstacle_count=0,
            )
        )
        env.reset(seed=3)
        expert = BoustrophedonExpert()
        for _ in range(8):
            action = expert.actions(env)
            target, valid = env.peek(action[0])
            self.assertTrue(valid)
            self.assertNotIn(target, env.obstacles)
            result = env.step(action)
            if result.done:
                break

        self.assertGreater(env.coverage_ratio(), 0.4)
        self.assertNotIn((1, 1), env.path)
        self.assertNotIn((2, 1), env.path)

    def test_generate_expert_dataset_keeps_mapmsg_gat_inputs(self) -> None:
        config = ExperimentConfig(
            env=GridCoverageConfig(
                width=4,
                height=4,
                max_steps=6,
                seed=11,
                num_agents=2,
                start_positions=[(0, 0), (3, 3)],
                observation_radius=1,
                communication_radius=4,
                use_explicit_map_memory=True,
                share_map_memory=True,
            ),
            ppo=PPOConfig(
                hidden_dim=32,
                use_graph_attention=True,
                gat_num_heads=4,
                gat_use_edge_features=True,
                gat_residual=True,
                use_coverage_messages=True,
            ),
            train=TrainConfig(use_tensorboard=False),
        )

        dataset = generate_expert_dataset(config, episodes=1)

        self.assertGreater(dataset.transitions, 0)
        self.assertEqual(dataset.observations.shape[1], 2)
        self.assertEqual(dataset.actions.shape[1], 2)
        self.assertIsNotNone(dataset.edge_features)
        self.assertIsNotNone(dataset.node_messages)
        assert dataset.edge_features is not None
        assert dataset.node_messages is not None
        self.assertEqual(dataset.edge_features.shape[1:3], (2, 2))
        self.assertEqual(dataset.node_messages.shape[1], 2)

    def test_pretrain_writes_loadable_checkpoint(self) -> None:
        config = ExperimentConfig(
            env=GridCoverageConfig(width=4, height=4, max_steps=8, seed=19, start=(0, 0)),
            ppo=PPOConfig(hidden_dim=32, learning_rate=0.001, seed=19),
            train=TrainConfig(use_tensorboard=False),
        )
        run_dir = ROOT / ".tmp_tests" / "imitation-pretrain"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = pretrain_imitation(config, run_dir=run_dir, episodes=2, epochs=1, batch_size=8)

            self.assertTrue(result.checkpoint.exists())
            self.assertTrue((run_dir / "imitation_metrics.csv").exists())
            self.assertTrue((run_dir / "imitation_summary.json").exists())
            self.assertGreater(result.transitions, 0)
            self.assertGreaterEqual(result.final_accuracy, 0.0)

            model = load_policy(config, result.checkpoint)
            env = GridCoverageEnv(config.env)
            observation = env.reset(seed=config.env.seed)
            state = env.global_state()
            action, _, value = model.act(
                torch.as_tensor(observation, dtype=torch.float32),
                torch.as_tensor(state, dtype=torch.float32),
            )
            self.assertIn(action, range(env.action_dim))
            self.assertTrue(torch.isfinite(value).item())

            summary = evaluate_policy(config, result.checkpoint, output_path=run_dir / "trajectory.json")
            self.assertIn("coverage_ratio", summary)
            self.assertTrue((run_dir / "trajectory.json").exists())
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
