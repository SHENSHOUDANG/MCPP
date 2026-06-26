# Experiment Log

Use this file for durable experiment summaries. Do not store raw checkpoints, generated plots, training logs, or bulk metric traces in Git.

## 2026-06-26 Repository Cleanup

Repository cleanup removed tracked and untracked training products from the working tree:

- Yangshan scheduler checkpoints and checkpoint backups;
- scheduler metrics, summaries, and logs;
- baseline and unified-evaluation traces;
- raw source package files under `data/ports/yangshan_task_initial_v1/source/`;
- generated `reports/`, `outputs/`, `.tmp_tests/`, and Python cache folders.

The cleanup keeps compact runtime scenario files:

- `data/ports/yangshan_task_initial_v1/yangshan_task_initial_v1_grid.json`
- `data/ports/yangshan_task_initial_v1/yangshan_task_initial_v1_tasks.json`
- `configs/port_yangshan_task_initial_v1.toml`

No model-quality conclusion should be inferred from deleted artifacts. Regenerate training/evaluation outputs locally when needed and summarize only the relevant conclusions here.
