from pathlib import Path
import sys
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from render_los_angeles_training_effect import _load_json, render_effect


class LosAngelesEffectRenderTests(unittest.TestCase):
    def test_render_effect_writes_png_from_official_training_json(self) -> None:
        grid = _load_json(ROOT / "data" / "ports" / "los_angeles_training_v1" / "los_angeles_training_v1_grid.json")
        tasks = _load_json(ROOT / "data" / "ports" / "los_angeles_training_v1" / "los_angeles_training_v1_tasks.json")
        output = ROOT / ".tmp_tests" / "los_angeles_training_effect_test.png"

        result = render_effect(grid, tasks, output)

        self.assertEqual(result, output)
        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 10_000)

    def test_render_effect_accepts_chart_basemap(self) -> None:
        grid = _load_json(ROOT / "data" / "ports" / "los_angeles_training_v1" / "los_angeles_training_v1_grid.json")
        tasks = _load_json(ROOT / "data" / "ports" / "los_angeles_training_v1" / "los_angeles_training_v1_tasks.json")
        basemap = ROOT / ".tmp_tests" / "synthetic_noaa_chart_basemap.png"
        output = ROOT / ".tmp_tests" / "los_angeles_training_effect_chart_test.png"
        basemap.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 420), color=(222, 232, 240)).save(basemap)

        result = render_effect(grid, tasks, output, basemap)

        self.assertEqual(result, output)
        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
