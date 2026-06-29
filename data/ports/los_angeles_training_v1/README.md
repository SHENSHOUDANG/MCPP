# los_angeles_training_v1

This is a compact Los Angeles port scheduler training scenario.

Status: `PENDING_ENGINEERING_TRAINING`.

It is designed to make the LA-port training command runnable today while V1.2 item 9 and the final official GIS workflow remain unfrozen. The task objects are named after plausible LA port management objects and are marked as scenario-generated engineering seeds. They are not final official work orders and must not be reported as final experiment evidence.

- Point tasks: 8
- Corridor tasks: 6
- Area tasks: 4
- Coordinate mode: local equirectangular approximation, `distance_mode=utm_euclidean`
- Cell size: 250 m

Run a smoke check:

```powershell
.\.venv\Scripts\python.exe tools\check_port_inspection_env.py --config configs\port_los_angeles_training_v1.toml --steps 2
```

Run scheduler training:

```powershell
.\.venv\Scripts\python.exe tools\train_port_scheduler_rl.py --config configs\port_los_angeles_training_v1.toml --steps 10000
```
