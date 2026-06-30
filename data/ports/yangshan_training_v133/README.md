# yangshan_training_v133

This is a Yangshan V1.3.3 scheduler training scenario imported from the
user-provided manual task package.

Status: `HISTORICAL_ENGINEERING_TRAINING`.

It is intended for engineering training and cross-port comparison only. It does
not replace Los Angeles as the primary V1.2 empirical scenario and must not be
reported as final experiment evidence.

- Source directory: `D:/map/yangshan2`
- Point tasks: 22
- Corridor tasks: 46
- Area tasks: 3
- Total active tasks: 71
- Grid shape: [40, 63]
- Depot cell: [26, 41]
- Cell size: 250 m
- Lifecycle: `v1_2_direct_service`

Regenerate from the local source package:

```powershell
.\.venv\Scripts\python.exe tools\import_yangshan_v133_training.py
```

Smoke-check the scheduler environment:

```powershell
.\.venv\Scripts\python.exe tools\check_port_inspection_env.py --config configs\port_yangshan_training_v133.toml --steps 2 --seed 7
```

Training requires explicit historical-baseline acknowledgement:

```powershell
.\.venv\Scripts\python.exe tools\train_port_scheduler_rl.py --config configs\port_yangshan_training_v133.toml --allow-historical-baseline --algorithm heterogeneous_mappo
```
