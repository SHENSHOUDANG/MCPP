# mathbased-mcpp

`mathbased-mcpp` is a PPO baseline for multi-agent grid coverage path planning.

The formal curriculum is run one course at a time. Each course writes its own checkpoints, metrics, TensorBoard logs, evaluation trajectory, rendered path, and config snapshot.

## Quick Start

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp doctor --config configs/smoke.toml
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp train --config configs/smoke.toml
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp train --config configs/formal_v1.toml --course tier-1-8x8-1agent
```

## Formal Curriculum

`configs/formal_v1.toml` defines four tiers:

- `tier-1-8x8-1agent`: 8x8 map, 100 episode steps, 1 agent, 500000 PPO timesteps, no previous model.
- `tier-2-13x13-2agents`: 13x13 map, 180 episode steps, 2 agents, 1000000 PPO timesteps, initialized from tier 1.
- `tier-3-18x18-3agents`: 18x18 map, 300 episode steps, 3 agents, 2000000 PPO timesteps, initialized from tier 2.
- `tier-4-30x30-4agents`: 30x30 map, 550 episode steps, 4 agents, 4000000 PPO timesteps, initialized from tier 3.

All formal tiers use `observation_radius = 2`, so the actor receives a 5x5 local observation window. The obstacle density is `obstacle_ratio = 0.0625`, which gives 4 obstacles on 8x8, 11 on 13x13, 20 on 18x18, and 56 on 30x30.

Each course writes:

- `best_policy.pt`
- `last_policy.pt`
- `policy.pt`
- `metrics.csv`
- `eval_metrics.csv`
- `tensorboard/`
- `trajectory.json`
- `trajectory.png`
- `course_config.json`

The formal curriculum output root defaults to `E:\test plot`.

## Memory And Communication

The actor uses explicit observation memory. It does not use an RNN, GRU, or LSTM.

The local actor observation has seven map channels:

- `self_agent`
- `other_agents`
- `uncovered`
- `team_covered`
- `obstacles`
- `self_covered`
- `recent_path`

`self_covered` records cells visited by the current agent. `recent_path` records the current agent's latest path cells, with larger values for more recent positions.

Graph attention communication is optional in code and enabled in the formal PPO config. Each agent is one graph node. For the current homogeneous-agent setting, two agents are neighbors when:

```text
manhattan_distance(agent_i, agent_j) <= communication_radius
```

Every agent has a self-edge. During PPO updates, attention is applied only within the same rollout step, using observations shaped as `[time, agent, dim]` and neighbor masks shaped as `[time, agent, agent]`.

## Configuration Fields

The formal config uses:

```toml
[env]
observation_radius = 2
recent_path_length = 8
communication_radius = 4

[ppo]
use_graph_attention = true
```

Each curriculum course also sets `recent_path_length = 8` and `communication_radius = 4`. Keep the course-level values in sync with the top-level `[env]` values.

For future heterogeneous agents, the GAT path already accepts a general `[agent, agent]` mask. The current config is homogeneous and uses one shared `communication_radius`.

## TensorBoard

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -m tensorboard.main --logdir "E:\test plot\<run>\01-tier-1-8x8-1agent\tensorboard"
```

Point TensorBoard at the course-specific `tensorboard` directory to inspect train/eval reward, coverage, path length, completion, and step metrics.

## PyCharm Configuration

Use the same project settings for all four courses:

```text
Project root: <this repository root>
Interpreter:   E:\miniconda3\envs\two-stage-mcpp\python.exe
Working dir:   <this repository root>
Run target:    module `mathbased_mcpp`
```

If you prefer a script path instead of a module, use:

```text
<this repository root>\mathbased_mcpp\__main__.py
```

For every run configuration below, keep the working directory and interpreter the same. Only the script parameters change.

### Course 1

Training:

```text
train --config configs/formal_v1.toml --course tier-1-8x8-1agent
```

Testing:

```text
evaluate --config configs/formal_v1.toml --checkpoint "E:\test plot\<run>\01-tier-1-8x8-1agent\policy.pt"
```

Final path drawing:

```text
render --config configs/formal_v1.toml --checkpoint "E:\test plot\<run>\01-tier-1-8x8-1agent\policy.pt"
```

### Course 2

Training:

```text
train --config configs/formal_v1.toml --course tier-2-13x13-2agents --previous-checkpoint "E:\test plot\<run>\01-tier-1-8x8-1agent\best_policy.pt"
```

Testing:

```text
evaluate --config configs/formal_v1.toml --checkpoint "E:\test plot\<run>\02-tier-2-13x13-2agents\policy.pt"
```

Final path drawing:

```text
render --config configs/formal_v1.toml --checkpoint "E:\test plot\<run>\02-tier-2-13x13-2agents\policy.pt"
```

### Course 3

Training:

```text
train --config configs/formal_v1.toml --course tier-3-18x18-3agents --previous-checkpoint "E:\test plot\<run>\02-tier-2-13x13-2agents\best_policy.pt"
```

Testing:

```text
evaluate --config configs/formal_v1.toml --checkpoint "E:\test plot\<run>\03-tier-3-18x18-3agents\policy.pt"
```

Final path drawing:

```text
render --config configs/formal_v1.toml --checkpoint "E:\test plot\<run>\03-tier-3-18x18-3agents\policy.pt"
```

### Course 4

Training:

```text
train --config configs/formal_v1.toml --course tier-4-30x30-4agents --previous-checkpoint "E:\test plot\<run>\03-tier-3-18x18-3agents\best_policy.pt"
```

Testing:

```text
evaluate --config configs/formal_v1.toml --checkpoint "E:\test plot\<run>\04-tier-4-30x30-4agents\policy.pt"
```

Final path drawing:

```text
render --config configs/formal_v1.toml --checkpoint "E:\test plot\<run>\04-tier-4-30x30-4agents\policy.pt"
```

## Notes

- Explicit memory channels and graph attention change the actor input/model structure, so old checkpoints from before this change cannot be reused. Start again from course 1.
- The `--previous-checkpoint` path is optional for later courses if you want the command to reuse the shared curriculum state file automatically, but in PyCharm it is clearer to point directly at the previous course's `best_policy.pt`.
- The `checkpoint=` line printed after training is the path you should paste into the corresponding `evaluate` and `render` configurations.
- `evaluate` writes `trajectory.json`, and `render` writes `trajectory.png` next to that checkpoint.
- `best_policy.pt` is selected by deterministic evaluation during training, using the same policy mode as the test command.

## Tests

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -m unittest tests.test_config_env tests.test_rewards tests.test_ppo_render
```
