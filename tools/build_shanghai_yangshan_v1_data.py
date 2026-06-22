from __future__ import annotations

import sys

from build_osm_port_real_map import main as build_osm_real_map


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--name",
                "shanghai_yangshan_osm_v1",
                "--output-dir",
                "data/ports/shanghai_yangshan_osm_v1",
            ]
        )
    build_osm_real_map()
