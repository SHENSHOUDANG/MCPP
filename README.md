# mathbased-mcpp

`mathbased-mcpp` is a PPO baseline for multi-agent grid coverage path planning.

The formal curriculum is run one course at a time. Each course writes its own checkpoints, metrics, TensorBoard logs, evaluation trajectory, rendered path, and config snapshot.

## Quick Start

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp doctor --config configs/smoke.toml
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp train --config configs/smoke.toml
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp train --config configs/formal_v1.toml --course tier-1-8x8-1agent
```

## Recent Change Recap

The recent training line now includes these changes:

- Random obstacle maps are generated with connectivity preserved, so agents should not be isolated in a disconnected region.
- The actor has explicit local memory channels for self coverage and recent path history, without RNN/GRU/LSTM state.
- PPO can optionally use range-limited multi-head graph attention communication between neighboring agents.
- `team_frontier_weight = 0.0` disables frontier-distance shaping, and the environment skips the associated distance-search work.
- Course-level PPO rollout sizes increase with task difficulty.
- The formal curriculum is capped at 20x20 maps. Obstacle density is a per-course experiment knob, not a curriculum difficulty axis.

## Formal Curriculum

`configs/formal_v1.toml` defines four tiers:

- `tier-1-8x8-1agent`: 8x8 map, 100 episode steps, 1 agent, 500000 PPO timesteps, no previous model.
- `tier-2-13x13-2agents`: 13x13 map, 180 episode steps, 2 agents, 1000000 PPO timesteps, initialized from tier 1.
- `tier-3-18x18-3agents`: 18x18 map, 300 episode steps, 3 agents, 1800000 PPO timesteps, initialized from tier 2.
- `tier-4-20x20-4agents`: 20x20 map, 500 episode steps, 4 agents, 4400000 PPO timesteps, initialized from tier 3.

All formal tiers use `observation_radius = 2`, so the actor receives a 5x5 local observation window. The largest formal course is fixed at 20x20 to keep training and ablation experiments manageable. Each course has its own explicit `obstacle_ratio`, so you can change course 2, 3, or 4 independently without changing the others.

## Obstacle Density Rationale

Obstacle density is controlled per course:

```toml
[[curriculum.courses]]
name = "tier-3-18x18-3agents"
obstacle_ratio = 0.05
```

The 5%, 10%, 15%, and 20% obstacle levels should be treated as optional experimental stress settings, not as automatic curriculum stages. If you want course 3 to use 10% obstacles, change only course 3's `obstacle_ratio` to `0.10`. If you want all courses to use the same density for a matched experiment, set the same value in each course block.

Maritime autonomy literature usually frames ASV/MASS navigation around safe path planning, dynamic traffic objects, COLREGs-compliant collision avoidance, and environmental factors rather than dense static obstacle fields:

- Wu et al. review ASV path planning as safe route generation under constraints such as obstacles, dynamic obstacles, vehicle capability, and marine environmental factors: [A Review of Path Planning Methods for Marine Autonomous Surface Vehicles](https://www.mdpi.com/2077-1312/12/5/833).
- A survey on autonomous collision avoidance at sea emphasizes COLREGs compliance and a perception set containing traffic objects, own-ship state, and environmental information: [Autonomous Collision Avoidance at Sea: A Survey](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.739013/full).
- VORRT-COLREGs evaluates open-ocean and traffic-separation scenarios through collision encounters with other vessels, supporting the view that open-sea difficulty is often dynamic-traffic driven: [VORRT-COLREGs](https://arxiv.org/abs/2109.00862).
- AIS traffic-density work treats maritime density as vessel occupancy over grid cells and distinguishes open-sea, coastal, inland, and access routes, so open-water scenarios should not be modeled like cluttered static obstacle maps: [Maritime Traffic Evaluation Using Spatial-Temporal Density Analysis Based on Big AIS Data](https://www.mdpi.com/2076-3417/12/21/11246).

For this project, obstacle density should stay explicit in each course block. Later open-water deployment tests can use lower obstacle ratios again, but each experiment should avoid silently sampling obstacle difficulty randomly.

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

Important current limitation: these channels are still exposed to the policy as a local crop around the agent. The environment maintains the ground-truth coverage state for reward, evaluation, and rendering, then the actor receives only the local observation window. In the current implementation, `team_covered` should therefore be read as a local team-coverage observation channel, not as a decentralized map that each robot has built and fused by itself.

The intended next memory design is decentralized explicit map memory:

- Each agent owns a full-map belief/coverage memory, such as `agent_known_covered[i]`, `agent_self_covered[i]`, `agent_known_obstacles[i]`, `agent_known_free[i]`, and `agent_unknown[i]`.
- At every step, an agent updates its own memory only from what it can locally observe and from its own movement history.
- When two agents are within communication or observation range, they exchange map memory or a compressed coverage summary.
- Map fusion is local to the communicating agents. For coverage, fusion can be a union of known covered cells. For obstacles and free cells, unknown cells remain unknown until observed or received from a neighbor.
- The environment's global truth remains internal and is used only for reward, termination, metrics, and visualization. It should not be directly leaked to every policy as a global `team_covered` map.
- Actor observations should then be generated from each agent's own memory, usually as a local crop plus optional compact global summaries such as known coverage ratio, nearest known uncovered direction, or known frontier count.

This planned design matches the robotics assumption more closely: local sensing, persistent per-robot mapping, and conditional map sharing when agents become neighbors. It is not implemented in the current training run. Existing runs can still serve as the baseline for local explicit memory plus GAT communication; a later switch to decentralized full-map memory will likely change observation semantics and may require a new curriculum run or a dedicated fine-tuning curriculum.

Graph attention communication is optional in code and enabled in the formal PPO config. Each agent is one graph node. For the current homogeneous-agent setting, two agents are neighbors when:

```text
manhattan_distance(agent_i, agent_j) <= communication_radius
```

Every agent has a self-edge. During PPO updates, attention is applied only within the same rollout step, using observations shaped as `[time, agent, dim]`, neighbor masks shaped as `[time, agent, agent]`, and optional edge features shaped as `[time, agent, agent, edge_dim]`.

The current GAT implementation is a lightweight multi-head masked attention layer over actor features:

- Q/K/V projections are applied to each agent's actor embedding.
- The dynamic communication mask blocks non-neighbor attention.
- Edge features add a learned bias per attention head. The current edge feature vector contains normalized Manhattan distance, relative row offset, relative column offset, and a connectivity flag.
- The formal config uses residual GAT output, so the actor keeps its own feature and adds neighbor-aggregated context.
- The latest attention tensor is cached inside the policy as `[batch, head, source_agent, target_agent]`, which can be used later for visualization or debugging.

## Configuration Fields

The formal config uses:

```toml
[env]
observation_radius = 2
recent_path_length = 8
communication_radius = 4

[ppo]
use_graph_attention = true
gat_num_heads = 4
gat_use_edge_features = true
gat_residual = true
gat_attention_dropout = 0.0
```

The formal reward config globally disables distance-based frontier shaping:

```toml
[reward]
team_frontier_weight = 0.0
```

When this weight is zero, the environment skips the distance-search work for `frontier_progress` and records the term as `0.0`. Raising the weight again re-enables the diagnostic/reward calculation.

Each curriculum course also sets `recent_path_length = 8`, `communication_radius = 4`, and its own `rollout_steps`. Keep the memory/communication values in sync with the top-level `[env]` values; let `rollout_steps` grow with course difficulty and agent count.

For future heterogeneous agents, the GAT path already accepts a general `[agent, agent]` mask. The current config is homogeneous and uses one shared `communication_radius`.

## Late-Stage Map Randomization

Courses 1 and 2 use fixed obstacle seeds so the model first learns basic coverage, explicit memory, and two-agent communication on stable maps.

Courses 3 and 4 use controlled seed pools:

```toml
# Course 3
random_obstacle_seeds = [20260430, 20260431, 20260432, 20260433]
map_refresh_episodes = 5

# Course 4
random_obstacle_seeds = [
  20260440,
  20260441,
  20260442,
  20260443,
  20260444,
  20260445,
  20260446,
  20260447,
]
map_refresh_episodes = 3
```

`random_obstacle_seeds` is a finite, reproducible map pool. `map_refresh_episodes` controls how many episodes stay on one map before rotating to the next seed. This keeps randomization gradual instead of making every episode a brand-new map.

Future courses can extend this idea with mixed fixed-pool and unbounded random seeds after the first four courses are stable. Keep the maximum map size at 20x20 and keep the obstacle ratio fixed inside each course. The randomization should change map layout, not silently change obstacle difficulty:

```text
Course 5: 20x20, 75% fixed seed pool + 25% unbounded random seeds, obstacle_ratio fixed by the chosen stress level
Course 6: 20x20, 50% fixed seed pool + 50% unbounded random seeds, obstacle_ratio fixed by the chosen stress level
Course 7: 20x20, 25% fixed seed pool + 75% unbounded random seeds, obstacle_ratio fixed by the chosen stress level
```

This should randomize map layouts without increasing obstacle difficulty. For the GAT ablation, use the same per-course obstacle ratios on both arms. To compare 5%, 10%, 15%, and 20%, run separate matched experiments by editing the relevant course-level `obstacle_ratio` values in both configs. For later open-water deployment tests, lower ratios can still be used as a separate scenario family.

Suggested future config field names:

```toml
fixed_seed_pool_probability = 0.75
use_unbounded_random_obstacle_seed = true
```

The fixed seed pool probability would decrease over later courses, while the unbounded random seed probability would increase. `obstacle_ratio` should stay explicitly configured per course and should not be sampled randomly.

## Thesis Simulation Scope

The simulation plan should balance thesis novelty with graduation risk. The project should not try to become a full ocean-world simulator. The main research scope is:

```text
Multi-agent coverage under local observation,
with explicit map memory and range-limited neighbor communication.
```

Recommended thesis-level contributions:

- Decentralized explicit coverage memory: each agent maintains its own belief/coverage map instead of relying on a recurrent hidden state or global truth.
- Neighbor communication and fusion: agents share coverage information only when they are within communication range, with GAT or lightweight attention used to aggregate neighbor information.
- Curriculum training: maps, agent count, and seed randomization are increased gradually to improve training stability and generalization.

Keep the simulation at the grid planning level for the main thesis experiments. This is enough to study cooperative coverage, repeated coverage, communication, and generalization while keeping implementation risk manageable. More realistic ship dynamics, COLREGs-compliant collision avoidance, AIS-driven traffic, and continuous control can be framed as future work.

Useful baselines:

- No memory and no communication.
- Local explicit memory without communication.
- Local explicit memory with neighbor/GAT communication.
- Decentralized full-map memory with range-limited map sharing.

Primary metrics:

- `coverage_ratio`
- `completion_rate`
- `repeat_ratio`
- `path_length`
- `steps_to_complete`
- `communication_count`
- generalization to unseen obstacle seeds

The thesis story should emphasize decision-level multi-agent coverage rather than low-level vessel control. In an engineering system, the learned grid action can be treated as a planning-layer command that would be executed by a separate path-following and safety controller.

## TensorBoard

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -m tensorboard.main --logdir "E:\test plot\<run>\01-tier-1-8x8-1agent\tensorboard"
```

Point TensorBoard at the course-specific `tensorboard` directory to inspect train/eval reward, coverage, path length, completion, and step metrics.

The formal config uses sparse deterministic evaluation/checkpointing to reduce overhead:

```toml
[train]
log_interval = 1
eval_interval = 10
checkpoint_interval = 10
```

Training metrics are still written as completed episodes arrive. Eval metrics and intermediate `last_policy.pt` saves happen every ten PPO updates. This reduces deterministic eval and checkpoint I/O cost without reducing training samples.

## Convergence Tuning

The first stability pass should focus on PPO scale rather than reward redesign.

Main metrics to judge convergence:

- `eval/coverage_ratio`
- `eval/completed`
- `eval/reward`
- `eval/path_length`

Reward alone can be misleading because `finish_reward` is large. A run can learn high coverage while still failing to converge on stable full completion.

The top-level PPO rollout is still the course-1 default:

```toml
rollout_steps = 256
```

Each curriculum course can override it. The implementation counts agent transitions, so the approximate continuous environment horizon per rollout is `rollout_steps / num_agents`.

Current course-level rollout settings:

```text
course 1: 256 / 1 = 256 env steps
course 2: 640 / 2 = 320 env steps
course 3: 1152 / 3 = 384 env steps
course 4: 2048 / 4 = 512 env steps
```

This keeps both the raw rollout size and the effective environment horizon increasing with course difficulty. It also lets course 3 cover a full `max_steps = 300` episode and course 4 cover a full `max_steps = 500` episode inside one rollout.

Use this rule when changing future courses:

```text
rollout_steps ~= num_agents * desired_environment_horizon
```

Current rollout overrides:

```toml
# Course 1
rollout_steps = 256

# Course 2
rollout_steps = 640

# Course 3
rollout_steps = 1152

# Course 4
rollout_steps = 2048
```

If course 4 still has unstable reward and low completion rate, the next low-risk lever is learning rate: test `learning_rate = 0.0001` for courses 3 and 4, or `learning_rate = 0.00005` for course 4. Larger rollouts increase compute time and memory use, but they are the most direct stability lever before changing the reward function.

The formal config also uses a larger PPO mini-batch:

```toml
mini_batch_size = 256
```

This reduces optimizer overhead on CPU. For course 4, it lowers the approximate number of backward passes per PPO update from `36 minibatches * 4 epochs = 144` to `9 minibatches * 4 epochs = 36`, while keeping the same rollout samples.

GPU note: the config currently uses:

```toml
device = "auto"
```

The trainer resolves this to CUDA when `torch.cuda.is_available()` is true, otherwise CPU. To verify the active PyTorch environment:

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

GPU acceleration may be modest for small maps because environment stepping is still CPU-side, but it should help more as `rollout_steps`, batch size, hidden size, and GAT computation grow. If the installed PyTorch build supports CUDA, PyCharm does not need a different training command; keep the same interpreter and leave `device = "auto"`, or set `device = "cuda"` explicitly for later experiments.

## GAT Ablation

The GAT ablation uses two matched configs:

- `configs/ablation_gat_on.toml`: same curriculum, `use_graph_attention = true`.
- `configs/ablation_gat_off.toml`: same curriculum, `use_graph_attention = false`.

Both configs keep the same GAT hyperparameters, but the off arm does not instantiate the attention module. The two arms should be trained separately from course 1. Do not initialize the GAT-off arm from a GAT-on checkpoint, because the actor architecture and checkpoint keys differ.

Train the GAT-on arm:

```text
train --config configs/ablation_gat_on.toml --course tier-1-8x8-1agent
train --config configs/ablation_gat_on.toml --course tier-2-13x13-2agents --previous-checkpoint "E:\test plot\ablation_gat_on\<run>\01-tier-1-8x8-1agent\best_policy.pt"
train --config configs/ablation_gat_on.toml --course tier-3-18x18-3agents --previous-checkpoint "E:\test plot\ablation_gat_on\<run>\02-tier-2-13x13-2agents\best_policy.pt"
train --config configs/ablation_gat_on.toml --course tier-4-20x20-4agents --previous-checkpoint "E:\test plot\ablation_gat_on\<run>\03-tier-3-18x18-3agents\best_policy.pt"
```

Train the GAT-off arm:

```text
train --config configs/ablation_gat_off.toml --course tier-1-8x8-1agent
train --config configs/ablation_gat_off.toml --course tier-2-13x13-2agents --previous-checkpoint "E:\test plot\ablation_gat_off\<run>\01-tier-1-8x8-1agent\best_policy.pt"
train --config configs/ablation_gat_off.toml --course tier-3-18x18-3agents --previous-checkpoint "E:\test plot\ablation_gat_off\<run>\02-tier-2-13x13-2agents\best_policy.pt"
train --config configs/ablation_gat_off.toml --course tier-4-20x20-4agents --previous-checkpoint "E:\test plot\ablation_gat_off\<run>\03-tier-3-18x18-3agents\best_policy.pt"
```

After both arms finish, compare them on the same held-out seeds and obstacle ratios:

```text
gat-ablation --gat-on-config configs/ablation_gat_on.toml --gat-on-checkpoint "E:\test plot\ablation_gat_on\<run>\04-tier-4-20x20-4agents\best_policy.pt" --gat-off-config configs/ablation_gat_off.toml --gat-off-checkpoint "E:\test plot\ablation_gat_off\<run>\04-tier-4-20x20-4agents\best_policy.pt" --seeds 20260601,20260602,20260603,20260604,20260605 --obstacle-ratios 0.05,0.10,0.15,0.20 --output "E:\test plot\gat_ablation_summary.csv"
```

The summary CSV contains `gat_on`, `gat_off`, and `delta_on_minus_off`. The most important fields are `coverage_ratio_mean`, `coverage_ratio_min`, `completion_rate`, `repeat_ratio_mean`, and `path_length_mean`. A useful GAT result would usually show higher completion/coverage and lower repeat ratio on the same seeds, especially in courses with 3 or 4 agents where communication can actually matter.

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
train --config configs/formal_v1.toml --course tier-4-20x20-4agents --previous-checkpoint "E:\test plot\<run>\03-tier-3-18x18-3agents\best_policy.pt"
```

Testing:

```text
evaluate --config configs/formal_v1.toml --checkpoint "E:\test plot\<run>\04-tier-4-20x20-4agents\policy.pt"
```

Final path drawing:

```text
render --config configs/formal_v1.toml --checkpoint "E:\test plot\<run>\04-tier-4-20x20-4agents\policy.pt"
```

## Notes

- Explicit memory channels and graph attention change the actor input/model structure, so old checkpoints from before this change cannot be reused. Start again from course 1.
- The enhanced GAT adds multi-head attention and edge-bias parameters. Old preliminary GAT checkpoints should be treated as a separate baseline; for the current GAT ablation, start both arms again from course 1.
- The current formal course 4 is `tier-4-20x20-4agents`; old `30x30` course-4 checkpoints belong to the previous curriculum and should not be mixed into the current ablation.
- The `--previous-checkpoint` path is optional for later courses if you want the command to reuse the shared curriculum state file automatically, but in PyCharm it is clearer to point directly at the previous course's `best_policy.pt`.
- The `checkpoint=` line printed after training is the path you should paste into the corresponding `evaluate` and `render` configurations.
- `evaluate` writes `trajectory.json`, and `render` writes `trajectory.png` next to that checkpoint.
- `best_policy.pt` is selected by deterministic evaluation during training, using the same policy mode as the test command.

## Tests

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -m unittest tests.test_config_env tests.test_rewards tests.test_ppo_render
```
