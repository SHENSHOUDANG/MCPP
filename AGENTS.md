# Agent Instructions

Keep this file short. It defines long-lived working rules for Codex and other coding agents in this repository.

## Working Style

- Do not use one long prompt as the project memory.
- Keep durable project knowledge in repository Markdown files.
- After each completed repository change, update the relevant project memory file before finishing.
- Commit completed code, config, data-cleanup, or documentation changes after review; keep commits focused.
- Prefer small, scoped changes that match the existing codebase.
- Do not commit model checkpoints, training outputs, raw source data packages, generated reports, or cache folders.
- Do not delete or rewrite source code unless the current task explicitly asks for it.
- Before broad cleanup, inspect `git status` and distinguish code/config/docs from generated artifacts.

## Project Memory Files

- `docs/model_specification.md`: theoretical definitions, task semantics, observation/action/reward contracts.
- `docs/current_task.md`: current active objective, decisions, constraints, and next steps.
- `docs/experiment_log.md`: experiment history, results, and artifact locations if they exist outside Git.
- `README.md`: concise entry point for setup, commands, repository layout, and artifact policy.

Use the active Codex conversation only for concrete one-off instructions. If an instruction should survive a new thread, move it into one of the Markdown files above.

## Artifact Policy

Tracked files should be source code, configs, tests, docs, and compact runtime scenario definitions.

Keep these out of Git:

- `*.pt` checkpoints;
- `runs/`, `outputs/`, `reports/`, `.tmp_tests/`;
- scheduler output folders under `data/ports/*/`;
- raw QGIS/GIS/GeoPackage/CSV source packages under `data/ports/*/source/`;
- metrics, logs, TensorBoard events, rendered trajectories, and temporary diagnostics.
