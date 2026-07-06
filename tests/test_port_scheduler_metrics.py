from pathlib import Path
import tempfile
import unittest

import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_port_scheduler_rl import _write_metrics, _write_text_atomic


class PortSchedulerMetricsTests(unittest.TestCase):
    def test_metric_and_summary_writes_replace_existing_files(self) -> None:
        row = {
            "algorithm": "shared_mappo",
            "episode": 1,
            "env_index": 0,
            "steps": 128,
            "episode_reward": 1.5,
            "completed_tasks": 2,
            "risk_exposure_sum": 0.0,
            "late_tasks": 0,
            "total_path_length": 12,
            "total_energy": 3.25,
            "total_conflicts": 0,
            "total_invalid_actions": 0,
            "total_replenishments": 1,
            "total_returns": 1,
            "assigned_task_count": 2,
            "unassigned_task_count": 0,
            "mean_assigned_scheduling_wait": 4.0,
            "mean_all_scheduling_wait_truncated": 4.0,
            "p50_scheduling_wait": 4.0,
            "p90_scheduling_wait": 5.0,
            "p95_scheduling_wait": 5.5,
            "max_open_scheduling_wait": 0.0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "nested" / "scheduler_metrics.csv"
            summary_path = Path(tmpdir) / "nested" / "scheduler_summary.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text("stale", encoding="utf-8")
            summary_path.write_text("stale", encoding="utf-8")

            _write_metrics(metrics_path, [row])
            _write_text_atomic(summary_path, '{"episode": 1}')

            metrics_text = metrics_path.read_text(encoding="utf-8")
            self.assertIn("algorithm,episode,env_index,steps", metrics_text)
            self.assertIn("shared_mappo,1,0,128", metrics_text)
            self.assertEqual(summary_path.read_text(encoding="utf-8"), '{"episode": 1}')
            self.assertFalse((metrics_path.parent / "scheduler_metrics.csv.tmp").exists())
            self.assertFalse((summary_path.parent / "scheduler_summary.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
