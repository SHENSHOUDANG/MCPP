from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from import_yangshan_v133_training import (
    _allowed_platforms,
    _float_or_none,
    _geometry_from_mode,
    _is_active_training_row,
    _risk_from_importance,
)


class YangshanV133ImportTests(unittest.TestCase):
    def test_inactive_rows_are_excluded_without_treating_blank_as_false(self) -> None:
        self.assertTrue(_is_active_training_row({"task_eligible": "1", "active_in_v1_3": ""}))
        self.assertFalse(_is_active_training_row({"task_eligible": "1", "active_in_v1_3": "0.0"}))
        self.assertFalse(_is_active_training_row({"task_eligible": "0", "active_in_v1_3": ""}))

    def test_geometry_and_capability_mappings_are_explicit(self) -> None:
        self.assertEqual(_geometry_from_mode("TARGET"), "point")
        self.assertEqual(_geometry_from_mode("CORRIDOR"), "line")
        self.assertEqual(_geometry_from_mode("AREA"), "area")
        self.assertEqual(_allowed_platforms("UAV|USV"), ("UAV", "USV"))
        self.assertEqual(_allowed_platforms("USV"), ("USV",))

    def test_null_deadline_stays_none(self) -> None:
        self.assertIsNone(_float_or_none(""))
        self.assertIsNone(_float_or_none(None))
        self.assertEqual(_float_or_none("0"), 0.0)

    def test_importance_to_training_risk(self) -> None:
        self.assertEqual(_risk_from_importance("A"), 3)
        self.assertEqual(_risk_from_importance("B"), 2)
        self.assertEqual(_risk_from_importance("C"), 1)


if __name__ == "__main__":
    unittest.main()
