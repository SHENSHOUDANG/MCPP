from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.config import GridCoverageConfig, PPOConfig, TrainConfig, ExperimentConfig, load_config
from mathbased_mcpp.evaluation import evaluate_policy
from mathbased_mcpp.rendering import render_trajectory
from mathbased_mcpp.training import train_ppo


class PpoRenderTests(unittest.TestCase):
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
            self.assertTrue((run_dir / "trajectory.json").exists())

            image = render_trajectory(config, summary["trajectory"], run_dir / "trajectory.png")
            self.assertTrue(image.exists())
            self.assertGreater(image.stat().st_size, 0)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

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


if __name__ == "__main__":
    unittest.main()
