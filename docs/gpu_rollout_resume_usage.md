# GPU Rollout Resume Usage

## Main MAPPO Training

Check the runtime settings:

```powershell
.\.venv\Scripts\python.exe -m mathbased_mcpp doctor --config configs\smoke.toml
```

Run training:

```powershell
.\.venv\Scripts\python.exe -m mathbased_mcpp train --config configs\formal_v1.toml --course tier-1-8x8-1agent
```

Resume from the latest course checkpoint:

```powershell
.\.venv\Scripts\python.exe -m mathbased_mcpp train --config configs\formal_v1.toml --course tier-1-8x8-1agent --resume-checkpoint outputs\formal_v1\<run>\01-tier-1-8x8-1agent\last_policy.pt
```

The resume checkpoint now stores model weights, optimizer state, Python/NumPy/PyTorch RNG state, per-environment state, current rollout observations, current global states, episode rewards, and episode path offsets.

## Port Scheduler Training

Run a GPU-backed, resource-limited scheduler training job:

```powershell
.\.venv\Scripts\python.exe tools\train_port_scheduler_rl.py --config configs\port_yangshan_task_initial_v1.toml --steps 100000 --checkpoint-interval 10000 --num-envs 2 --env-workers 2 --cpu-threads 4 --gpu-memory-fraction 0.35 --process-priority below_normal
```

Resume automatically from the newest scheduler checkpoint:

```powershell
.\.venv\Scripts\python.exe tools\train_port_scheduler_rl.py --config configs\port_yangshan_task_initial_v1.toml --steps 200000 --checkpoint-interval 10000 --resume auto --num-envs 2 --env-workers 2 --cpu-threads 4 --gpu-memory-fraction 0.35 --process-priority below_normal
```

## Resource Headroom

The default training runtime is intentionally conservative:

- `cpu_threads = 4`
- `interop_threads = 1`
- `rollout_workers = 2`
- `gpu_memory_fraction = 0.35`
- `process_priority = "below_normal"`

For office work or gaming while training, keep `gpu_memory_fraction` between `0.25` and `0.35`, and keep `num_envs`/`rollout_workers` at `2`. For overnight training, raise `num_envs` and `rollout_workers` gradually after checking GPU memory and system responsiveness.
