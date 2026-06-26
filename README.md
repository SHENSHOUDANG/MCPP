# mathbased-mcpp

`mathbased-mcpp` is a Python/PyTorch project for multi-agent coverage path planning and port inspection scheduling. It includes:

- a grid-coverage PPO/MAPPO baseline;
- optional explicit-memory, GAT, CUAP, and related ablation configs;
- a port-inspection scheduling environment for the Yangshan task scenario;
- command-line tools for training, evaluation, rendering, and scenario import.

The repository should contain source code, configs, tests, documentation, and compact runtime scenario definitions. Training products are local artifacts and should not be committed.

## Repository Layout

```text
configs/                         Experiment and scheduler configs
data/ports/yangshan_task_initial_v1/
  yangshan_task_initial_v1_grid.json
  yangshan_task_initial_v1_tasks.json
  README.md                      Compact runtime scenario definition
docs/                            Runtime and resource notes
src/mathbased_mcpp/              Main package
src/mathbased_mcpp/port_inspection/
tools/                           Training, evaluation, import, and render scripts
tests/                           Unit and smoke tests
```

Generated folders such as `runs/`, `outputs/`, `reports/`, `.tmp_tests/`, scheduler checkpoint folders, training logs, and evaluation traces are ignored by Git.

## Project Memory

Long-lived project knowledge is kept in small Markdown files instead of a single large prompt:

- `AGENTS.md`: durable agent rules and artifact policy.
- `docs/model_specification.md`: theoretical task and model contracts.
- `docs/current_task.md`: current objective, decisions, and next steps.
- `docs/experiment_log.md`: experiment summaries and cleanup history.

After each repository change, update the relevant project memory file before finishing:

- long-term rules -> `AGENTS.md`;
- theoretical definitions -> `docs/model_specification.md`;
- current task state -> `docs/current_task.md`;
- experiment results -> `docs/experiment_log.md`;
- concrete one-off instructions -> the active Codex conversation.

Every completed code, config, data-cleanup, or documentation change should be committed after review. Keep commits focused and do not include generated training artifacts.

## Setup

Use Python 3.10+ and install the package dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

For CUDA training, install a CUDA-enabled PyTorch build or use `environment.cuda.yml` if you manage the environment with Conda. The default config uses `device = "auto"`, so CUDA is selected only when the active PyTorch installation supports it.

## Quick Checks

Validate the basic grid-coverage configuration:

```powershell
.\.venv\Scripts\python.exe -m mathbased_mcpp doctor --config configs\smoke.toml
```

Run the unit/smoke test set:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

## Grid-Coverage Training

Run a small smoke training job:

```powershell
.\.venv\Scripts\python.exe -m mathbased_mcpp train --config configs\smoke.toml
```

Run the formal curriculum one course at a time:

```powershell
.\.venv\Scripts\python.exe -m mathbased_mcpp train --config configs\formal_v1.toml --course tier-1-8x8-1agent
```

Depot-return training can be driven through the pipeline script:

```powershell
.\.venv\Scripts\python.exe tools\run_depot_return_pipeline.py --config configs\formal_v1.toml --dry-run
.\.venv\Scripts\python.exe tools\run_depot_return_pipeline.py --config configs\formal_v1.toml
```

Training writes checkpoints, metrics, TensorBoard logs, trajectories, and rendered images under the configured run root. These outputs are not versioned.

## Port Scheduler Workflow

The current Yangshan scheduler config uses compact runtime inputs:

```text
configs\port_yangshan_task_initial_v1.toml
data\ports\yangshan_task_initial_v1\yangshan_task_initial_v1_grid.json
data\ports\yangshan_task_initial_v1\yangshan_task_initial_v1_tasks.json
```

Check the scenario/environment:

```powershell
.\.venv\Scripts\python.exe tools\check_port_inspection_env.py --config configs\port_yangshan_task_initial_v1.toml
```

Run a conservative local scheduler training job:

```powershell
.\.venv\Scripts\python.exe tools\train_port_scheduler_rl.py `
  --config configs\port_yangshan_task_initial_v1.toml `
  --steps 100000 `
  --checkpoint-interval 10000 `
  --num-envs 2 `
  --env-workers 2 `
  --cpu-threads 4 `
  --gpu-memory-fraction 0.35 `
  --process-priority below_normal
```

Resume from locally generated scheduler checkpoints:

```powershell
.\.venv\Scripts\python.exe tools\train_port_scheduler_rl.py `
  --config configs\port_yangshan_task_initial_v1.toml `
  --steps 200000 `
  --checkpoint-interval 10000 `
  --resume auto
```

After cleanup, `--resume auto` only finds checkpoints created by a later local run.

## Scenario Data Policy

The repository keeps only compact scenario files needed by the current config. Raw QGIS/GeoPackage/CSV source packages, import summaries, generated tile caches, model checkpoints, metrics, and report folders are excluded.

If the Yangshan scenario must be rebuilt from raw source material, use:

```powershell
.\.venv\Scripts\python.exe tools\import_yangshan_task_initial.py
```

The raw source package must exist locally; it is not stored in Git.

## Artifact Policy

Do not commit:

- `*.pt` model checkpoints;
- scheduler output folders such as `data/ports/*/scheduler_rl/`;
- `runs/`, `outputs/`, `reports/`, and `.tmp_tests/`;
- metrics CSV/JSON summaries produced by training or evaluation;
- TensorBoard event files, rendered trajectories, map tile caches, and temporary diagnostics;
- raw GIS/QGIS/GeoPackage/CSV source packages under `data/ports/*/source/`.

Keep commits focused on source code, configs, tests, docs, and compact runtime scenario definitions.

## Common Commands

Render a port scenario:

```powershell
.\.venv\Scripts\python.exe tools\render_port_scenario.py --config configs\port_yangshan_task_initial_v1.toml
```

Evaluate a port scheduler checkpoint generated locally:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_port_scheduler_unified.py `
  --config configs\port_yangshan_task_initial_v1.toml `
  --checkpoint <local-checkpoint.pt>
```

Run a greedy scheduler baseline:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_port_scheduler_greedy.py --config configs\port_yangshan_task_initial_v1.toml
```

## Development Notes

- Historical checkpoints and generated reports were removed from the working tree and are no longer part of the repository.
- Old training results should be treated as local artifacts only, not as project source.
- If new experiments produce results worth preserving, summarize the method and conclusion in Markdown and keep the raw artifacts outside Git.
