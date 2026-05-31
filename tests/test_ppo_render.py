"""从训练到评估/渲染的端到端 smoke 测试。

这些测试只使用很小训练预算，目标是验证整条流水线能运行和写出结果，
而不是证明策略已经收敛。
"""

from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.benchmark import benchmark_policy
from mathbased_mcpp.config import GridCoverageConfig, PPOConfig, TrainConfig, ExperimentConfig, load_config
from mathbased_mcpp.evaluation import coverage_efficiency_metrics, evaluate_policy
from mathbased_mcpp.rendering import render_trajectory
from mathbased_mcpp.training import train_ppo


class PpoRenderTests(unittest.TestCase):
    """验证 PPO 输出、评价指标、图片渲染和 map-intent GAT 训练路径。"""

    def test_train_evaluate_and_render_smoke(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        run_dir = ROOT / ".tmp_tests" / "ppo-render"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint = train_ppo(config, run_dir=run_dir)
            self.assertTrue(checkpoint.exists())
            self.assertEqual(checkpoint.name, "policy.pt")
            self.assertTrue((run_dir / "best_policy.pt").exists())
            self.assertTrue((run_dir / "last_policy.pt").exists())
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "eval_metrics.csv").exists())
            self.assertTrue(any((run_dir / "tensorboard").glob("events.out.tfevents.*")))
            self.assertTrue((run_dir / "trajectory.json").exists())
            self.assertTrue((run_dir / "trajectory.png").exists())
            self.assertTrue((run_dir / "course_config.json").exists())

            summary = evaluate_policy(config, checkpoint, output_path=run_dir / "trajectory.json")
            self.assertIn("coverage_ratio", summary)
            self.assertIn("coverage_auc", summary)
            self.assertIn("repeat_ratio_after_90", summary)
            self.assertTrue((run_dir / "trajectory.json").exists())

            image = render_trajectory(config, summary["trajectory"], run_dir / "trajectory.png")
            self.assertTrue(image.exists())
            self.assertGreater(image.stat().st_size, 0)

            benchmark = benchmark_policy(config, checkpoint, seeds=[101, 102], output_path=run_dir / "benchmark.csv")
            self.assertEqual(benchmark["episodes"], 2)
            self.assertTrue((run_dir / "benchmark.csv").exists())
            self.assertIn("coverage_ratio_mean", benchmark)
            self.assertIn("coverage_auc_mean", benchmark)
            self.assertIn("stall_termination_coverage_mean", benchmark)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_coverage_efficiency_metrics(self) -> None:
        metrics = coverage_efficiency_metrics(
            trajectories=[
                [(0, 0), (0, 1), (0, 1), (0, 2)],
                [(1, 0), (1, 1), (1, 1), (0, 2)],
            ],
            coverage_curve=[0.2, 0.5, 0.5, 0.9],
            max_steps=4,
            budgets=[1, 3, 4],
            stall_steps=1,
        )
        self.assertAlmostEqual(metrics["coverage_at_1"], 0.5)
        self.assertAlmostEqual(metrics["coverage_at_3"], 0.9)
        self.assertAlmostEqual(metrics["coverage_at_4"], 0.9)
        self.assertAlmostEqual(metrics["coverage_auc"], 0.7)
        self.assertEqual(metrics["t90"], 3)
        self.assertIsNone(metrics["t95"])
        self.assertEqual(metrics["stalled"], 1)
        self.assertAlmostEqual(metrics["stall_termination_coverage"], 0.5)
        self.assertAlmostEqual(metrics["inter_agent_overlap_ratio"], 0.2)

    def test_train_two_agent_smoke(self) -> None:
        config = ExperimentConfig(
            env=GridCoverageConfig(
                width=4,
                height=4,
                max_steps=8,
                seed=13,
                num_agents=2,
                start_positions=[(0, 0), (3, 3)],
                random_obstacle_count=0,
            ),
            ppo=PPOConfig(
                total_timesteps=32,
                rollout_steps=8,
                update_epochs=1,
                mini_batch_size=8,
                hidden_dim=32,
                seed=13,
            ),
            train=TrainConfig(run_root="runs", log_interval=1),
        )
        run_dir = ROOT / ".tmp_tests" / "ppo-two-agent"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint = train_ppo(config, run_dir=run_dir)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(any((run_dir / "tensorboard").glob("events.out.tfevents.*")))
            summary = evaluate_policy(config, checkpoint, output_path=run_dir / "trajectory.json")
            self.assertEqual(len(summary["trajectories"]), 2)
            image = render_trajectory(config, summary["trajectory"], run_dir / "trajectory.png")
            self.assertTrue(image.exists())
            self.assertGreater(image.stat().st_size, 0)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_train_mapmsg_gat_smoke(self) -> None:
        config = ExperimentConfig(
            env=GridCoverageConfig(
                width=4,
                height=4,
                max_steps=8,
                seed=17,
                num_agents=2,
                start_positions=[(0, 0), (3, 3)],
                observation_radius=1,
                communication_radius=4,
                use_explicit_map_memory=True,
                share_map_memory=True,
            ),
            ppo=PPOConfig(
                total_timesteps=32,
                rollout_steps=8,
                update_epochs=1,
                mini_batch_size=8,
                hidden_dim=32,
                seed=17,
                use_graph_attention=True,
                gat_num_heads=4,
                gat_use_edge_features=True,
                gat_residual=True,
                use_coverage_messages=True,
                use_action_mask=True,
            ),
            train=TrainConfig(run_root="runs", log_interval=1),
        )
        run_dir = ROOT / ".tmp_tests" / "ppo-mapmsg-gat"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint = train_ppo(config, run_dir=run_dir)
            self.assertTrue(checkpoint.exists())
            summary = evaluate_policy(config, checkpoint, output_path=run_dir / "trajectory.json")
            self.assertEqual(len(summary["trajectories"]), 2)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
