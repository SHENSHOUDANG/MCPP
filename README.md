# mathbased-mcpp

`mathbased-mcpp` is a PPO baseline for multi-agent grid coverage path planning.

The formal curriculum is run one course at a time. Each course writes its own checkpoints, metrics, TensorBoard logs, evaluation trajectory, rendered path, and config snapshot.

## Quick Start

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp doctor --config configs/smoke.toml
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp train --config configs/smoke.toml
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp pretrain --config configs/formal_v1.toml --course tier-1-8x8-1agent --episodes 16 --epochs 4
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
- A low-cost imitation warm start is available through a boustrophedon obstacle-avoidance expert and behavior cloning pretraining.

## Formal Curriculum

`configs/formal_v1.toml` defines four tiers:

- `tier-1-8x8-1agent`: 8x8 map, 100 episode steps, 1 agent, 500000 PPO timesteps, no previous model.
- `tier-2-13x13-2agents`: 13x13 map, 180 episode steps, 2 agents, 1000000 PPO timesteps, initialized from tier 1.
- `tier-3-18x18-3agents`: 18x18 map, 300 episode steps, 3 agents, 1800000 PPO timesteps, initialized from tier 2.
- `tier-4-20x20-4agents`: 20x20 map, 500 episode steps, 4 agents, 4400000 PPO timesteps, initialized from tier 3.

All formal tiers use `observation_radius = 2`, so the actor receives a 5x5 local observation window. The largest formal course is fixed at 20x20 to keep training and ablation experiments manageable. Each course has its own explicit `obstacle_ratio`, so you can change course 2, 3, or 4 independently without changing the others.

## Imitation Warm Start

The large-map zero-shot representation from the literature is intentionally not part of the current implementation. It is too expensive for the current training budget because it enlarges the actor input and becomes especially costly on courses 3 and 4. The cheaper alternative now implemented is behavior cloning from a rule expert before PPO.

The expert is a boustrophedon-style coverage policy with obstacle avoidance:

- it orders free cells in alternating row sweeps;
- it uses shortest paths through connected free cells to route around obstacles;
- in multi-agent maps, it assigns row bands to agents and avoids immediate same-cell and current-position conflicts;
- it generates labels offline, while the learned actor still receives only the normal local observation and optional GAT/message inputs.

Run behavior cloning for one curriculum course:

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp pretrain --config configs/formal_v1.toml --course tier-1-8x8-1agent --episodes 16 --epochs 4
```

The command writes:

- `bc_policy.pt`
- `imitation_metrics.csv`
- `imitation_summary.json`
- `expert_trajectory.json` and `expert_trajectory.png`
- `bc_trajectory.json` and `bc_trajectory.png`
- `course_config.json`

Use `expert_trajectory.png` to inspect whether the rule expert generated a sensible boustrophedon path around obstacles. Use `bc_trajectory.png` to inspect the cloned actor after pretraining. The expert plot validates the labels; the BC plot validates what the network actually learned from those labels.

Use the produced checkpoint as a PPO warm start:

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp train --config configs/formal_v1.toml --course tier-1-8x8-1agent --previous-checkpoint "E:\test plot\imitation\<run>\bc_policy.pt"
```

After course 1, the existing curriculum transfer remains unchanged: train course 2 from course 1, course 3 from course 2, and course 4 from course 3. The imitation checkpoint is meant to reduce early PPO exploration cost, not to replace PPO or claim large-map zero-shot generalization.

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

The actor does not use an RNN, GRU, or LSTM. New training paths enforce decentralized actor information: global coverage truth belongs to the centralized critic, reward, metrics, and rendering only.

The private local-memory actor observation has six map channels:

- `self_agent`
- `other_agents`
- `self_uncovered`
- `obstacles`
- `self_covered`
- `recent_path`

`self_uncovered` means locally visible free cells not yet covered by that agent itself. It does not claim to know whether a teammate visited them. `self_covered` records cells visited by the current agent. `recent_path` records the current agent's latest path cells, with larger values for more recent positions.

The completed May 2026 legacy GAT-on/GAT-off run used a different seven-channel observation containing locally cropped environment-truth `team_covered` and a global-team-derived `uncovered` signal. That run is retained for audit and historical comparison, but it is not a valid decentralized-actor baseline. Loading its saved `course_config.json` enables an explicit legacy replay mode so its artifacts remain reproducible; new training does not enable that mode.

The current private local baseline has:

- local explicit memory channels, mainly `self_covered` and `recent_path`;
- range-limited GAT feature communication between neighboring agents.

The private local baseline does not have:

- a full-map memory owned by each individual robot;
- per-agent map updates based only on that robot's own local observation and movement history;
- conditional map exchange when robots become neighbors;
- map fusion between neighboring robots;
- actor observations generated from each robot's own fused memory map.

The map-intent baseline implements decentralized explicit map memory:

- Each agent owns a full-map belief/coverage memory, such as `agent_known_covered[i]`, `agent_self_covered[i]`, `agent_known_obstacles[i]`, `agent_known_free[i]`, and `agent_unknown[i]`.
- At every step, an agent updates its own memory only from what it can locally observe and from its own movement history.
- When two agents are within communication or observation range, they exchange map memory or a compressed coverage summary.
- Map fusion is local to the communicating agents. For coverage, fusion can be a union of known covered cells. For obstacles and free cells, unknown cells remain unknown until observed or received from a neighbor.
- The environment's global truth remains internal and is used only for reward, termination, metrics, and visualization. It should not be directly leaked to every policy as a global `team_covered` map.
- Actor observations should then be generated from each agent's own memory, usually as a local crop plus optional compact global summaries such as known coverage ratio, nearest known uncovered direction, or known frontier count.

The implementation deliberately does not add a pheromone channel. A pheromone/heuristic coverage field would be a distinct communication mechanism and must be introduced only as its own explicit ablation, not as a hidden replacement for team-coverage truth.

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

### Map-Intent Communication Baseline

A new opt-in ablation baseline implements this memory design without changing the completed legacy GAT experiment artifacts:

- `configs/ablation_mapmsg_gat_on.toml`
- `configs/ablation_mapmsg_gat_off.toml`

In these configs, every agent maintains its own full-map memory of known free cells, known obstacles, self-covered cells, and communicated known-team-covered cells. A teammate's historical coverage is not visible merely because that cell lies inside the local sensor window. Memory is updated from the agent's own movement and local map sensing; teammate coverage enters only through map exchange within `communication_radius`. The actor reads a fixed-size local crop from this memory, so curriculum transfer from 8x8 to 20x20 keeps a fixed actor input size.

Explicit-memory observations add `unknown` and `frontier` map channels. The policy also receives a fixed-size coverage message generated from the agent's own memory:

- known team coverage, self coverage, unknown-space, and frontier summaries;
- recent new-cell, repeat, and stall summaries;
- a memory-derived proposed exploration direction;
- a normalized target direction and distance;
- a one-hot `3 x 3` target exploration region and an intent-valid flag.

The target region is a compact coverage intent proposal derived only from remembered free/uncovered/frontier cells. It is not an oracle path plan and does not read unseen environment truth.

The two new ablation arms share the same explicit memory, map-fusion rule, coverage message, curriculum, seeds, rewards, and evaluation settings. Their substantive model difference remains:

```toml
use_graph_attention = true   # mapmsg_gat_on
use_graph_attention = false  # mapmsg_gat_off
```

`mapmsg_gat_off` encodes each agent's own coverage message. `mapmsg_gat_on` additionally applies the existing range-masked multi-head attention module to neighboring coverage messages. This isolates whether attention over coverage intent helps after both policies receive the same decentralized map-memory foundation.

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

Current `straight_weight` behavior:

- In single-agent mode, `straight_weight` controls a small straight-motion bonus. Continuing in the same direction receives the full small bonus, while turning receives a smaller bonus.
- In multi-agent mode, the current team reward does not use `straight_weight`. Courses 2, 3, and 4 therefore do not receive a straight-motion reward from this config field.
- Do not add this reward during the current GAT ablation. If it is introduced later, keep it small and make it a separate ablation, because too much straight-motion preference can reduce turning into local missed cells and may hurt final coverage.

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

Primary metrics should reflect budgeted online coverage rather than treating exact 100% completion as the only success condition:

- `Coverage@H`: coverage ratio reached within fixed environment-step budgets, such as `H = 100, 200, 300, 500` for the largest course.
- `Coverage-AUC`: area under the coverage-ratio-versus-time curve, rewarding policies that cover useful area early instead of wandering before reaching a similar final ratio.
- `T90`, `T95`, and `T99`: steps required to reach 90%, 95%, and 99% coverage.
- `StallCoverage@K`: achieved coverage when the policy has produced no new covered cell for `K` consecutive environment steps.
- generalization of the same metrics to unseen obstacle seeds.

Secondary diagnostic metrics:

- `RepeatRatio` and `RepeatRatioAfter90`, especially to explain inefficient motion after most of the environment has already been covered.
- `InterAgentOverlapRatio`, to identify repeated work caused by poor multi-agent coordination.
- `communication_count` or communication-use statistics for communication ablations.
- `completion_rate` and `T100`, retained as supplementary evidence that a policy can reach strict full coverage, but not used as the only judgment of quality in an unknown environment.
- `path_length`, interpreted as a resource/efficiency measure rather than a demand for visually tidy trajectories.

Path regularity is not a primary objective. In an unknown environment, requiring both high online coverage efficiency and predominantly straight, visually neat paths can impose a conflicting objective on the policy. Straight-motion preference or visual path neatness should therefore remain secondary observations unless a later application introduces an explicit vehicle-dynamics or energy-cost requirement.

The thesis story should emphasize decision-level multi-agent coverage rather than low-level vessel control. In an engineering system, the learned grid action can be treated as a planning-layer command that would be executed by a separate path-following and safety controller.

## Innovation Direction Notes

Do not frame common neural-network components as the main innovation. MAPPO, GAT, multi-head attention, edge features, communication gates, auxiliary losses, and communication budgets are established tools. In this project they should be treated as implementation mechanisms, not as the core thesis contribution by themselves.

A stronger thesis direction is to focus on a coverage-specific problem:

```text
Under team rewards and local observations, agents can achieve high coverage while still producing high repeated coverage because individual coverage contribution and coverage conflict are not clearly assigned.
```

The next method discussion should therefore prioritize coverage-task credit assignment and repeat-conflict modeling:

- explicit per-agent coverage contribution: how many new cells each agent contributes, and when that contribution occurs;
- repeated coverage conflict: whether an agent is revisiting cells already covered by itself or by teammates when useful alternatives exist;
- counterfactual or difference-style coverage contribution: how much team coverage would decrease if one agent's recent trajectory or new-cell contribution were removed;
- coverage conflict graph: graph edges should eventually encode potential repeat-coverage conflict or overlapping intent, not only communication distance;
- advantage or reward shaping that separates team coverage gain from individual repeat/conflict cost without turning the system into a hand-written planner.

This direction should be described carefully. It is not claiming to invent counterfactual credit assignment or graph attention. The possible contribution is to instantiate these ideas for unknown-environment multi-agent coverage, where the important failure mode is repeated coverage under weak individual credit. GAT-MAPPO remains the learning framework, while the thesis contribution should be the coverage-specific contribution/conflict formulation and its ablation evidence.

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

The main task interpretation is budgeted online coverage in an unknown environment:

```text
Within a limited time or energy budget, agents should cover as much useful area as possible
while avoiding clearly wasteful repeated work.
```

Exact full coverage remains useful evidence, but it should not dominate the evaluation. The final few unobserved or uncovered cells can create a large tail cost in an unknown environment, so a policy that quickly reaches high coverage should not be judged only by whether it finds every final isolated cell.

Main metrics to judge future convergence and model quality:

- `Coverage@H`, evaluated at fixed environment-step budgets.
- `Coverage-AUC`.
- `T90`, `T95`, and `T99`.
- `StallCoverage@K`, where no new cell for `K` consecutive steps indicates ineffective continued motion rather than task success.
- the same metrics on held-out obstacle seeds.

Secondary diagnostics:

- `RepeatRatioAfter90` and `InterAgentOverlapRatio`, for diagnosing late-stage repeated work and coordination failure.
- `completion_rate` and `T100`, for strict-full-coverage reference only.
- `eval/reward`, because reward magnitude can be distorted by `finish_reward` and should not be read alone.
- path length and rendered trajectory appearance, for resource interpretation and qualitative inspection rather than as primary measures of path beauty or straight-line regularity.

These expanded metrics are produced by the offline `evaluate`, `benchmark`, and `gat-ablation` measurement flow. They do not change policy training or checkpoint compatibility. Use them to re-evaluate existing checkpoints after a training run is complete.

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

## Official MAPPO Reference Points

Do not change the training algorithm again until the current GAT-on/GAT-off ablation has produced comparable course results. After that ablation, use the official MAPPO implementation as an engineering reference and absorb one stabilizing mechanism at a time.

Reference implementation:

- Official repository: [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)
- Rollout buffer and masks: [shared_buffer.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/utils/shared_buffer.py)
- PPO update and value loss: [r_mappo.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py)
- Actor/critic interfaces: [r_actor_critic.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py)
- Discrete action masking: [distributions.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/utils/distributions.py)

Useful ideas to borrow later, in priority order:

- `masks` and `bad_masks`: separate true task termination from time-limit truncation. This is important for course 3 and 4 because many episodes can hit `max_steps`; GAE should not treat every timeout exactly like a natural terminal state.
- `ValueNorm` or `PopArt`: normalize value targets so the critic is less sensitive to reward-scale changes across courses, finish rewards, and failed long episodes.
- Clipped value loss: constrain critic updates in the same spirit as PPO policy clipping, reducing large jumps in value prediction.
- Huber value loss: reduce the effect of rare extreme value errors compared with plain MSE.
- Separate actor and critic optimizers: allow different learning rates or update behavior for policy and value function. This should be tested only after the simpler value-loss changes.
- `active_masks`: useful if future environments include dead/inactive agents, turn-based behavior, or temporarily disabled robots. It is not urgent for the current simultaneous grid coverage setting.
- `available_actions`: official code supports discrete action masking by setting unavailable action logits to a very small value. For this project it should remain a later optional ablation, because masking illegal moves changes the learning problem and may hide whether the policy learned boundary/obstacle avoidance.
- Parallel rollout threads: official MAPPO uses vectorized rollout shapes, but this should be copied carefully. First add timing logs and a synchronous vector environment; multiprocessing comes only after shape, seed, reset, and GAT-mask tests pass.

Mechanisms not to copy by default:

- Recurrent policy state. The current thesis direction favors explicit map memory over RNN/GRU/LSTM memory, so recurrent MAPPO should remain a separate future baseline rather than the main path.
- Whole-framework replacement. The project should keep its task-specific `GridCoverageEnv`, GAT communication path, curriculum, rendering, and ablation tooling, while selectively importing stability ideas from official MAPPO.

Recommended order after the GAT ablation:

1. Add timing logs to identify environment/update/evaluation bottlenecks.
2. Fix termination semantics with `masks` and `bad_masks`.
3. Add `ValueNorm`, clipped value loss, and Huber value loss behind config flags.
4. Compare against the current baseline on the same seeds and obstacle ratios.
5. Only then consider separate actor/critic optimizers, action masking, or vectorized rollout.

## Parallel Environment Cautions

Parallel environment rollout is a possible future optimization, but it is risky because it can silently change MAPPO training semantics. Do not move directly from the current single-environment rollout to a full multiprocessing trainer without first preserving the rollout definitions and adding tests.

Main risks:

- Randomness can become uncontrolled if workers reuse the same seed, reset order changes the map sequence, or the fixed obstacle seed pool rotates differently in each worker.
- `rollout_steps` can become ambiguous. It must mean total agent transitions across all workers, not per-worker environment steps.
- Episode boundaries become asynchronous. Some workers may finish and reset while others continue, so GAE must use a per-environment done mask and must not leak advantage values across episode boundaries.
- MAPPO tensors can become misaligned. `obs`, centralized `state`, `action`, `reward`, `done`, `neighbor_mask`, and optional GAT `edge_features` must all refer to the same worker and time step.
- Inter-process communication can erase the speedup if too many small arrays are copied every step.
- Windows uses `spawn` multiprocessing, so workers must be pickle-safe and must not rely on forked global state.
- Metrics, CSV, TensorBoard, and checkpoints must remain main-process responsibilities. Workers should only return environment transition data and episode summaries.

Safety rules for a later implementation:

```text
rollout_steps = total agent transitions across all workers
env_steps_per_worker ~= rollout_steps / (num_envs * num_agents)
seed(worker, episode) = base_seed + worker_id * large_offset + episode_id
batch shape before flattening = [time, env, agent, ...]
```

Recommended implementation path:

1. Add a synchronous vector environment first, still in one process. This validates shapes, reset behavior, metrics, and GAE while keeping debugging simple.
2. Add multiprocessing only after the synchronous vector environment matches single-environment behavior on smoke tests.
3. Keep the PyTorch model, PPO update, GAE, checkpointing, and logging in the main process. Worker processes should own only `GridCoverageEnv` instances and execute `reset` / `step`.
4. Start with `num_envs = 2`, then try `4`. More workers are not automatically faster on small grid environments.
5. Keep `num_envs = 1` as the default until the vectorized path has stable training curves and regression tests.

Minimum tests before enabling multiprocessing by default:

- single-env and vector-env smoke training both produce valid checkpoints and trajectories;
- different workers produce different obstacle maps when their seeds differ;
- `rollout_steps` produces the expected total number of agent transitions;
- done/reset in one worker does not reset other workers;
- GAE uses per-worker done masks;
- `neighbor_mask` and GAT edge features keep shape `[time, env, agent, agent]` and `[time, env, agent, agent, edge_dim]` before flattening.

## GAT Ablation

The archived May 2026 GAT comparison used two matched arms but also used the now-rejected legacy actor coverage observation. Its checkpoints and reported metrics remain a historical diagnostic only.

The source configs below now explicitly disable legacy truth-coverage observation for any new run:

- `configs/ablation_gat_on.toml`: same curriculum, `use_graph_attention = true`.
- `configs/ablation_gat_off.toml`: same curriculum, `use_graph_attention = false`.

Both configs keep the same GAT hyperparameters, but the off arm does not instantiate the attention module. Any new run from these configs starts a corrected private-local baseline and is not directly the same model family as the archived trained checkpoints. Do not initialize a corrected run from an archived legacy checkpoint.

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

The summary CSV contains `gat_on`, `gat_off`, and `delta_on_minus_off`. The primary fields are `coverage_at_<budget>_mean`, `coverage_auc_mean`, `t90/t95/t99_mean_reached`, `t90/t95/t99_reach_rate`, and `stall_termination_coverage_mean`. Use `repeat_ratio_after_90_mean`, `inter_agent_overlap_ratio_mean`, `completion_rate`, and `path_length_mean` as supporting diagnostics. Results from archived legacy checkpoints must be labeled as legacy truth-observation results, not decentralized baseline results.

The map-intent ablation follows the same workflow with new configs and separate output roots:

```text
train --config configs/ablation_mapmsg_gat_on.toml --course tier-1-8x8-1agent
train --config configs/ablation_mapmsg_gat_off.toml --course tier-1-8x8-1agent
```

Continue courses 2 through 4 using the matching arm's preceding `best_policy.pt`, then invoke `gat-ablation` with `--gat-on-config configs/ablation_mapmsg_gat_on.toml` and `--gat-off-config configs/ablation_mapmsg_gat_off.toml`. Do not compare a legacy GAT checkpoint directly against a map-intent checkpoint as though only attention differed; their observation and message semantics are different.

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
