from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from import_los_angeles_task_mapping import _geometry_from_row, _parse_wkt_coordinates, _risk_from_importance, _task_family


class LosAngelesTaskMappingImportTests(unittest.TestCase):
    def test_wkt_parser_handles_supported_geometry_types(self) -> None:
        self.assertEqual(_parse_wkt_coordinates("POINT (-118.25 33.7)"), [(-118.25, 33.7)])
        self.assertEqual(
            _parse_wkt_coordinates("LINESTRING (-118.25 33.7, -118.24 33.71)"),
            [(-118.25, 33.7), (-118.24, 33.71)],
        )
        self.assertEqual(
            _parse_wkt_coordinates("POLYGON ((-118.25 33.7, -118.24 33.7, -118.25 33.7))"),
            [(-118.25, 33.7), (-118.24, 33.7)],
        )

    def test_catalog_mappings_are_explicit(self) -> None:
        self.assertEqual(_risk_from_importance("A"), 3)
        self.assertEqual(_risk_from_importance("B"), 2)
        self.assertEqual(_task_family("CHANNEL_INSPECTION"), "HYDROGRAPHIC_SURVEY")
        self.assertEqual(_task_family("ANCHORAGE_INSPECTION"), "SURFACE_SAFETY_PATROL")
        self.assertEqual(_task_family("BUOY_INSPECTION"), "WATERSIDE_ASSET_INSPECTION")
        self.assertEqual(_task_family("BERTH_AREA_INSPECTION"), "WATERSIDE_ASSET_INSPECTION")

    def test_geometry_role_follows_wkt_type(self) -> None:
        self.assertEqual(_geometry_from_row({"task_id": "P", "geometry_wkt": "POINT (-118.25 33.7)"}), "point")
        self.assertEqual(
            _geometry_from_row({"task_id": "L", "geometry_wkt": "LINESTRING (-118.25 33.7, -118.24 33.71)"}),
            "line",
        )
        self.assertEqual(
            _geometry_from_row({"task_id": "A", "geometry_wkt": "MULTIPOLYGON (((-118.25 33.7, -118.24 33.7)))"}),
            "area",
        )


if __name__ == "__main__":
    unittest.main()
