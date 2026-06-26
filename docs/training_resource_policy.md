# Port Scheduler Training Resource Policy

The scheduler training script is configured for low-impact local training by default:

- `device = "auto"` uses CUDA when a CUDA-enabled PyTorch build is installed, otherwise CPU.
- `num_envs = 2` and `env_workers = 2` keep rollout parallelism modest for a 16 GB laptop.
- `cpu_threads = 6` and `interop_threads = 2` avoid occupying all Ryzen CPU threads.
- `gpu_memory_fraction = 0.35` caps PyTorch's CUDA allocator to roughly one third of GPU memory.
- `process_priority = "below_normal"` lets foreground work and games win OS scheduling.

The GPU memory fraction does not throttle GPU compute. If games or office apps feel sluggish, reduce `num_envs`, `env_workers`, and `cpu_threads` before increasing the memory cap.

## Artifact Policy

Training products are local-only and should not be committed:

- model checkpoints (`*.pt`);
- scheduler metrics and summaries;
- training logs and TensorBoard event files;
- evaluation traces, generated figures, and report folders.

Keep source code, configs, tests, docs, and compact scenario definitions in Git. Regenerate training products from the commands below when needed.

Example training command:

```powershell
.\.venv\Scripts\python.exe tools\train_port_scheduler_rl.py `
  --config configs\port_yangshan_task_initial_v1.toml `
  --steps 1000000 `
  --seed 20260622 `
  --resume auto
```

When gaming or doing heavy office work, use a lighter profile:

```powershell
.\.venv\Scripts\python.exe tools\train_port_scheduler_rl.py `
  --config configs\port_yangshan_task_initial_v1.toml `
  --steps 1000000 `
  --seed 20260622 `
  --resume auto `
  --num-envs 1 `
  --env-workers 1 `
  --cpu-threads 4 `
  --gpu-memory-fraction 0.20 `
  --process-priority idle
```

The current `environment.yml` is CPU-only. Use `environment.cuda.yml` or install a CUDA-enabled PyTorch build before expecting `device=auto` to select the RTX GPU.
