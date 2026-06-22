# shanghai_yangshan_osm_v1

This dataset is generated from OpenStreetMap data downloaded through the Overpass API.

- BBox: south=30.585, west=121.985, north=30.675, east=122.165
- Cell size: 150 m
- Scope: expanded Yangshan Deep-Water Port water-surface inspection research grid
- License note: OSM data is available under the Open Database License (ODbL).
- Safety note: this is not a nautical chart and must not be used for navigation.

The grid is no longer a hand-drawn random/prototype layout. Port land, pier,
breakwater, harbour, waterway, and seamark-related OSM features are rasterized
or used as task-generation anchors. OSM source features are also retained in
grid metadata under `visual_features` so the extracted linear and polygon
layers can be audited and rendered explicitly.
