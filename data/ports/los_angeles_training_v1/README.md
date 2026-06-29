# los_angeles_training_v1

This is a Los Angeles port scheduler training scenario built from official NOAA ENC Direct geometry.

Status: `PENDING_OFFICIAL_GEOMETRY_TRAINING`.

The geometry is sourced from NOAA Office of Coast Survey ENC Direct REST services for the Los Angeles port
area. This checked-in scenario was regenerated from the embedded official sample snapshot captured on
2026-06-29 because live network execution was unavailable during the update. The generated scheduler tasks
are derived from official chart objects, but the workload, deadlines, risk, and release settings remain
training parameters rather than official Port of Los Angeles work orders. Do not report this as final
experiment evidence until the V1.2 algorithm contract and official experiment workflow are frozen.

- Point tasks: 3
- Corridor tasks: 2
- Area tasks: 2
- Geometry source: NOAA ENC Direct Harbour and Approach REST services
- Access date: 2026-06-29
- Coordinate mode: local equirectangular approximation, `distance_mode=utm_euclidean`
- Cell size: 250 m

Regenerate from official NOAA services:

```powershell
.\.venv\Scripts\python.exe tools\build_los_angeles_training_scenario.py
```

Regenerate from the embedded official NOAA sample snapshot:

```powershell
.\.venv\Scripts\python.exe tools\build_los_angeles_training_scenario.py --use-embedded-official-snapshot
```

Run a smoke check:

```powershell
.\.venv\Scripts\python.exe tools\check_port_inspection_env.py --config configs\port_los_angeles_training_v1.toml --steps 2
```

Run scheduler training:

```powershell
.\.venv\Scripts\python.exe tools\train_port_scheduler_rl.py --config configs\port_los_angeles_training_v1.toml --steps 10000
```
