import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_port_algorithm_comparison import DEFAULT_COMPARISON_ALGORITHMS, _comparison_row_from_summary, _parse_algorithms


class PortAlgorithmComparisonTests(unittest.TestCase):
    def test_parse_algorithms_normalizes_aliases_and_removes_duplicates(self) -> None:
        self.assertEqual(
            _parse_algorithms("mappo, shared-policy-mappo centralized_context_ppo happo mappo"),
            ["heterogeneous_mappo", "shared_mappo", "centralized_ppo", "happo"],
        )

    def test_default_comparison_algorithms_exclude_heterogeneous_mappo(self) -> None:
        self.assertEqual(DEFAULT_COMPARISON_ALGORITHMS, ("shared_mappo", "centralized_ppo", "happo"))
        self.assertNotIn("heterogeneous_mappo", DEFAULT_COMPARISON_ALGORITHMS)

    def test_comparison_row_reads_scheduler_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_output = Path(tmpdir) / "centralized_ppo"
            summary_dir = candidate_output / "scheduler_rl"
            summary_dir.mkdir(parents=True)
            (summary_dir / "scheduler_summary.json").write_text(
                json.dumps(
                    {
                        "algorithm": "centralized_ppo",
                        "episode": 2,
                        "steps": 16,
                        "episode_reward": 3.5,
                        "completed_tasks": 4,
                        "late_tasks": 1,
                        "total_energy": 9.25,
                        "total_conflicts": 0,
                        "total_invalid_actions": 2,
                    }
                ),
                encoding="utf-8",
            )

            row = _comparison_row_from_summary(
                algorithm="centralized-context-ppo",
                seed=20260630,
                requested_steps=16,
                candidate_output=candidate_output,
            )

        self.assertEqual(row["algorithm"], "centralized_ppo")
        self.assertEqual(row["seed"], 20260630)
        self.assertEqual(row["steps"], 16)
        self.assertEqual(row["completed_tasks"], 4)
        self.assertTrue(str(row["summary_path"]).endswith("scheduler_summary.json"))


if __name__ == "__main__":
    unittest.main()
