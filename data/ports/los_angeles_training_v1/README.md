# los_angeles_training_v1

This is a Los Angeles port scheduler training scenario imported from the user-provided
`Port_of_Los_Angeles_Task_Mapping_V2.0_Chart_Aligned` package.

Status: `PENDING_CHART_ALIGNED_TASK_MAPPING_TRAINING`.

The checked-in grid and task JSON were generated from `D:/地图/洛杉矶/task_catalog_v2_0.csv`.
The source package describes chart-aligned research geometry validated against NOAA Chart 18751 and
supporting public NOAA/Port of Los Angeles datasets. It is not native ENC vector geometry and must not be
reported as final experiment evidence until the V1.2 algorithm contract and official experiment workflow
are frozen.

- Point tasks: 3
- Corridor tasks: 10
- Area tasks: 13
- Stored reinspection metadata tasks: 4
- Coordinate mode: local equirectangular approximation, `distance_mode=utm_euclidean`
- Cell size: 250 m

Task type counts:

- ANCHORAGE_INSPECTION: 2
- BERTH_AREA_INSPECTION: 11
- BUOY_INSPECTION: 3
- CHANNEL_INSPECTION: 10

Regenerate from the provided local task mapping directory:

```powershell
.\.venv\Scripts\python.exe tools\import_los_angeles_task_mapping.py
```

Run a smoke check:

```powershell
.\.venv\Scripts\python.exe tools\check_port_inspection_env.py --config configs\port_los_angeles_training_v1.toml --steps 2
```

Run scheduler training:

```powershell
.\.venv\Scripts\python.exe tools\run_port_algorithm_comparison.py --config configs\port_los_angeles_training_v1.toml --steps 50000 --device auto --num-envs 1 --env-workers 1 --checkpoint-interval 10000
```
