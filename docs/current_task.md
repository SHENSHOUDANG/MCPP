# Current Task

Last updated: 2026-06-26

## Active Objective

Clean the repository so it tracks code, configs, tests, docs, and compact runtime scenario definitions only. Remove training products, training data packages, stale generated artifacts, and outdated README content.

## Decisions

- Do not rely on one long Codex prompt for project memory.
- Keep long-lived rules in `AGENTS.md`.
- Keep theoretical definitions in `docs/model_specification.md`.
- Keep current task state here.
- Keep experiment history in `docs/experiment_log.md`.
- Keep raw training artifacts and generated reports out of Git.
- After each completed repository change, update the relevant project memory file.
- Completed code, config, cleanup, or documentation changes should be committed after review.

## Current Repository State

- Code-only update was committed and pushed as `df418c6 chore: save code updates 2026-06-26`.
- Generated training products under the Yangshan scheduler scenario have been removed from the working tree.
- Raw `source/` data packages and `import_summary.json` were removed from the working tree.
- `README.md` has been shortened to current setup, workflow, and artifact policy.
- `docs/model_specification.md` now defines the port scheduler model contract, including UAV/USV sets, screening/review tasks, review triggers, USV backlog backpressure, state/action spaces, candidate generation, conflict resolution, rewards, completion rules, depot/replenishment behavior, constraints, and thesis metrics.

## Remaining Local Changes

The working tree currently contains cleanup/documentation changes that have not been committed yet:

- updated artifact ignore rules;
- deleted generated training and evaluation artifacts;
- updated repository and scenario README files;
- updated runtime/resource docs;
- added project memory documents.

## Next Steps

1. Review this cleanup diff.
2. Commit the cleanup as a documentation/artifact-pruning change.
3. Push the cleanup commit after review.
