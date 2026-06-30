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
- Platform depots: {'UAV': [82, 108], 'USV': [82, 108]}
- Depot placement: user-provided WGS84 coordinate `30 deg 36.27 min N, 122 deg 5.70 min E`, transformed to EPSG:32651 as `(413246.952064, 3386120.770780)` and snapped to the Yangshan 100 m grid cell `[82, 108]`.

The scheduler consumes the existing `PortInspectionSchedulingEnv` JSON schema, so coordinates are encoded as UTM-derived feature bins. These bins are not a rasterized path-planning map and no low-level path action is emitted.
