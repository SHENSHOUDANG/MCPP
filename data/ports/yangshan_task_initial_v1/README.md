# yangshan_task_initial_v1

This scenario is the compact runtime import of the Yangshan QGIS/GeoPackage task map.
Under the V1.2 contract it is retained only as a HISTORICAL engineering baseline.
It is not the final real-port task set and must not be used as final V1.2 evidence.
Raw QGIS/GeoPackage/CSV source packages are not stored in Git; regenerate them with the import tooling when needed.

- Coordinate mode: EPSG:32651 UTM, Euclidean travel proxy
- Coordinate feature resolution: 100.0 m
- Model grid shape used only as coordinate-feature envelope: [110, 159]
- Fixed inspection tasks: 219
- Dynamic seed tasks: 3
- Total point tasks: 222
- Risk counts: {3: 66, 1: 120, 2: 36}
- Platform depots: {'UAV': [73, 110], 'USV': [73, 110]}
- Depot placement: user-defined shoreline depot on `source_port_coastline`; the supplied QGIS map has no explicit depot marker, so UAV and USV share this coast-edge base instead of using a water-surface point.

The scheduler consumes the existing `PortInspectionSchedulingEnv` JSON schema, so coordinates are encoded as UTM-derived feature bins. These bins are not a rasterized path-planning map and no low-level path action is emitted.
