# mathbased-mcpp

`mathbased-mcpp` is a PPO baseline for multi-agent grid coverage path planning.

The current map-intent curriculum is run one course at a time. Each course writes its own checkpoints, metrics, TensorBoard logs, evaluation trajectory, rendered path, and config snapshot.

## Quick Start

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp doctor --config configs/smoke.toml
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp train --config configs/smoke.toml
E:\miniconda3\envs\two-stage-mcpp\python.exe -m mathbased_mcpp train --config configs/ablation_mapmsg_gat_on.toml --course tier-1-8x8-1agent
```

## Recent Change Recap

The recent training line now includes these changes:

- Random obstacle maps are generated with connectivity preserved, so agents should not be isolated in a disconnected region.
- The private-local actor has self-history channels for self coverage and recent path history, without RNN/GRU/LSTM state.
- PPO can optionally use range-limited multi-head graph attention communication between neighboring agents.
- `team_frontier_weight = 0.0` disables frontier-distance shaping, and the environment skips the associated distance-search work.
- Course-level PPO rollout sizes increase with task difficulty.
- The current map-intent curriculum is capped at 20x20 maps. Obstacle density is a per-course experiment knob, not a curriculum difficulty axis.

## Current Curriculum

Both `configs/ablation_mapmsg_gat_on.toml` and `configs/ablation_mapmsg_gat_off.toml` define the current four-tier curriculum:

- `tier-1-8x8-1agent`: 8x8 map, 100 episode steps, 1 agent, 500000 PPO timesteps, no previous model.
- `tier-2-13x13-2agents`: 13x13 map, 180 episode steps, 2 agents, 1000000 PPO timesteps, initialized from tier 1.
- `tier-3-18x18-3agents`: 18x18 map, 300 episode steps, 3 agents, 1800000 PPO timesteps, initialized from tier 2.
- `tier-4-20x20-4agents`: 20x20 map, 500 episode steps, 4 agents, 3200000 PPO timesteps, initialized from tier 3.

All current tiers use `observation_radius = 2`, so the actor receives a 5x5 local observation window. The largest current course is fixed at 20x20 to keep training and ablation experiments manageable. Each course has its own explicit `obstacle_ratio`, so you can change course 2, 3, or 4 independently without changing the others.

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

The current output roots are `E:\test plot\ablation_mapmsg_gat_on` and `E:\test plot\ablation_mapmsg_gat_off`.

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

- local self-history channels, mainly `self_covered` and `recent_path`;
- range-limited GAT feature communication between neighboring agents.

The private local baseline does not have:

- a full-map memory owned by each individual robot;
- per-agent map updates based only on that robot's own local observation and movement history;
- conditional map exchange when robots become neighbors;
- map fusion between neighboring robots;
- actor observations generated from each robot's own fused memory map.

The implemented map-intent configs add decentralized explicit map memory:

- Each agent owns full-map memory sets named `covered_by_agent[i]`, `known_team_covered_by_agent[i]`, `known_obstacles_by_agent[i]`, and `known_free_by_agent[i]`; unknown cells are derived from the absence of known free/obstacle information.
- At every step, an agent updates its own memory only from what it can locally observe and from its own movement history.
- When two agents are within `communication_radius`, `share_map_memory` fuses their known free, known obstacle, and known-team-covered sets. Separately, `node_messages()` produces compact coverage/intent summaries for the actor and optional GAT.
- Map fusion is local to the communicating agents. For coverage, fusion is a union of known covered cells. For obstacles and free cells, unknown cells remain unknown until observed or received from a neighbor.
- The environment's global truth remains internal and is used only for reward, termination, metrics, and visualization. It should not be directly leaked to every policy as a global `team_covered` map.
- Explicit-memory actor observations are generated from each agent's own memory as a local crop plus compact summaries such as known coverage ratio and memory-derived coverage intent.

The implementation deliberately does not add a pheromone channel. A pheromone/heuristic coverage field would be a distinct communication mechanism and must be introduced only as its own explicit ablation, not as a hidden replacement for team-coverage truth.

Graph attention communication is optional in code and enabled in the current `mapmsg_gat_on` config. Each agent is one graph node. For the current homogeneous-agent setting, two agents are neighbors when:

```text
manhattan_distance(agent_i, agent_j) <= communication_radius
```

Every agent has a self-edge. During PPO updates, attention is applied only within the same rollout step, using observations shaped as `[time, agent, dim]`, neighbor masks shaped as `[time, agent, agent]`, and optional edge features shaped as `[time, agent, agent, edge_dim]`.

The current GAT implementation is a lightweight multi-head masked attention layer over the selected node representation:

- In the private-local GAT path, Q/K/V projections are applied to each agent's actor embedding.
- In the map-intent GAT path, Q/K/V projections are applied to encoded coverage-intent messages, then the communicated result is fused back into the actor feature.
- The dynamic communication mask blocks non-neighbor attention.
- Edge features add a learned bias per attention head. The current edge feature vector contains normalized Manhattan distance, relative row offset, relative column offset, and a connectivity flag.
- The current GAT-on config uses residual GAT output, so the actor keeps its own feature and adds neighbor-aggregated context.
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

The current GAT-on config uses:

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

The current map-intent configs use one shared reward formula for single-agent and multi-agent courses:

```toml
[reward]
team_new_cell_weight = 1.0
team_straight_weight = 0.01
team_repeat_weight = 0.5
team_invalid_weight = 0.8
team_time_weight = 0.05
scale_time_cost_by_uncovered = false
```

`team_time_weight` is a fixed cost for every environment step in the current map-intent training line. It no longer shrinks with remaining uncovered area, so searching inefficiently for tail cells is not made artificially cheap. Archived snapshots that do not set `scale_time_cost_by_uncovered` retain the previous coverage-scaled reward semantics for replay.

Distance-based frontier shaping is disabled by default and is omitted from active training configs. Setting the optional `team_frontier_weight` above zero in a dedicated ablation re-enables its diagnostic/reward calculation.

Current straight-motion behavior:

- `team_straight_weight` applies in both single-agent and multi-agent mode. It is deliberately `0.01`, so a continued move is only a path tie-breaker relative to a newly covered cell.
- New actor observations expose the agent's most recent effective move direction. The direction survives a blocked move, matching how the straight bonus is evaluated and making the preference observable during training.
- Obstacle/boundary impacts and agent-agent collisions are both failed actions and enter the same `invalid` penalty. Collision counts remain in `reward_terms` as diagnostics, not as an additional reward.
- The legacy reward keys previously left in TOML files (`straight_weight`, `distance_weight`, `coverage_weight`, `invalid_move_penalty`, and `team_collision_weight`) are not active reward inputs and have been removed from training configs.

Current map-intent terminal-reward behavior:

```toml
[reward]
finish_reward = 10.0
normalize_team_finish_reward = true
```

- The new `ablation_mapmsg_gat_on.toml` and `ablation_mapmsg_gat_off.toml` arms treat `finish_reward` as one team-level completion bonus and divide it by `num_agents` before writing the shared per-agent transition reward.
- The smaller terminal bonus records strict completion without making the last isolated cell the main training driver.
- For example, a four-agent map-intent course receives `10 / 4 = 2.5` completion reward per agent transition instead of giving every transition the former `120` bonus.
- Archived legacy configurations retain their historical terminal-reward semantics for replay; do not compare their total rewards directly with the new map-intent training line.

The current map-intent arms also set `use_action_mask = true`. PPO masks only moves that an executing agent can already prove infeasible: boundary exits and obstacles present in its local or communication-fused memory. Unknown obstacles, repeat coverage and simultaneous agent-agent collision outcomes remain part of the learned problem.

Each curriculum course also sets `recent_path_length = 8`, `communication_radius = 4`, and its own `rollout_steps`. Keep the memory/communication values in sync with the top-level `[env]` values; let `rollout_steps` grow with course difficulty and agent count.

For future heterogeneous agents, the GAT path already accepts a general `[agent, agent]` mask. The current config is homogeneous and uses one shared `communication_radius`.

## Late-Stage Map Randomization

Courses 1 and 2 use fixed obstacle seeds so the selected model first learns basic coverage and two-agent interaction on stable maps. In the map-intent configs, that interaction also includes explicit memory and map exchange.

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

The task objective is:

```text
Within a fixed task budget, maximize effective coverage progress while minimizing
repeat coverage, infeasible motion and inter-agent conflict.
```

Exact 100% coverage is a supplementary outcome rather than the main training driver. Primary metrics are ordered accordingly:

- `Coverage@H`: coverage ratio reached within fixed environment-step budgets, such as `H = 100, 200, 300, 500` for the largest course.
- `Coverage-AUC`: area under the coverage-ratio-versus-time curve, rewarding policies that cover useful area early instead of wandering before reaching a similar final ratio.
- `T90` and `T95`: steps required to reach useful high coverage without making tail search the central target.
- `RepeatRatioAfter90`: repeated movement after high coverage has already been reached.
- `InterAgentOverlapRatio`: redundant coverage caused by multi-agent coordination failure.
- generalization of the same metrics to unseen obstacle seeds.

Secondary diagnostic metrics:

- `T99` and `StallCoverage@K`, especially to diagnose inefficient late-stage search.
- `RepeatRatio`, to explain total repeated work.
- `communication_count` or communication-use statistics for communication ablations.
- `completion_rate` and `T100`, retained only as supplementary strict-full-coverage evidence.
- `path_length`, interpreted as a resource/efficiency measure rather than a demand for visually tidy trajectories.

Path regularity is not a primary objective. In an unknown environment, requiring both high online coverage efficiency and predominantly straight, visually neat paths can impose a conflicting objective on the policy. Straight-motion preference or visual path neatness should therefore remain secondary observations unless a later application introduces an explicit vehicle-dynamics or energy-cost requirement.

The thesis story should emphasize decision-level multi-agent coverage rather than low-level vessel control. In an engineering system, the learned grid action can be treated as a planning-layer command that would be executed by a separate path-following and safety controller.

## Literature Transfer Plan For Zero-Shot Generalization

Two papers by Carvalho and Aguiar provide useful reference designs for the next experimental stages:

- J. P. Carvalho and A. P. Aguiar, "Deep Reinforcement Learning for Zero-Shot Coverage Path Planning With Mobile Robots," *IEEE/CAA Journal of Automatica Sinica*, 2025. Official page: [DOI 10.1109/JAS.2024.125064](https://www.ieee-jas.net/article/doi/10.1109/JAS.2024.125064?pageType=en).
- J. P. Carvalho and A. P. Aguiar, "Multi-Agent Reinforcement Learning for Zero-Shot Coverage Path Planning With Dynamic UAV Networks," *Robotics and Autonomous Systems*, vol. 195, 2026. Official page: [ScienceDirect article S092188902500260X](https://www.sciencedirect.com/science/article/pii/S092188902500260X).

These papers should be treated as method references, not as direct baselines already reproduced by this repository. Their learning framework is Rainbow DQN / VDN with safety filtering, while this repository uses PPO/MAPPO-style actor-critic training with optional GAT communication and decentralized map-intent memory. The transferable parts are representation, reward normalization ideas, action feasibility mechanisms, and evaluation design. Any adopted component must be re-tested under this project's information boundary and metrics.

### What Each Paper Contributes

The mobile-robot paper addresses zero-shot transfer across workspace sizes for a single robot. Its most relevant ideas for this repository are:

- an egocentric map representation that keeps the policy input shape fixed while deployment maps can be larger than training maps;
- map centering and border-region compression for observations larger than the trained input size;
- a size-invariant value/reward design intended to reduce dependence on map dimensions;
- action masking combined with curriculum learning and a robustness rule to improve feasibility and coverage behavior.

The dynamic-UAV-network paper extends the zero-shot argument to multiple agents. Its most relevant ideas are:

- centralized training with decentralized execution and parameter sharing;
- an input representation intended to tolerate a varying number of UAVs;
- an event-driven, size-invariant reward formulation with a cooperation coefficient `K`;
- a safety and robustness filter applied after policy output;
- evaluation on configurations outside training distributions, including larger maps, more obstacles, and more agents.

The multi-agent paper is the stronger reference for future reward and agent-count experiments. The mobile-robot paper is the stronger reference for map resizing and action-feasibility experiments.

### Non-Negotiable Information Boundary

The papers use full-information map representations in parts of their formulation. This repository must not reintroduce environment-truth leakage into a decentralized actor merely to copy their zero-shot mechanisms.

For this project:

- The centralized critic may use environment truth, including global coverage and obstacle state, during training.
- Rewards, termination checks, metrics, and rendering may use environment truth.
- Each actor must obtain coverage/map features only from its own local sensing, movement history, and explicitly received neighbor messages.
- Larger-map compression must operate on each agent's private or fused memory map, not on the environment's true global `team_covered` map.
- Action feasibility masks must only use facts available to the executing agent unless a separate oracle-mask ablation is explicitly labeled as such.

This restriction is essential because the current thesis claim concerns cooperative coverage under local observation and communication, rather than centralized access to the complete map.

### Primary Representation Direction: Agent-Centered Memory Observation

The first implementation priority suggested by the two papers is an observation redesign, not an immediate reward redesign. The mobile-robot paper motivates the spatial operation: re-center a map around the acting robot and preserve a fixed actor input shape. The dynamic-UAV-network paper motivates the multi-agent content: the observation should distinguish ego state, coverage state, obstacles, and neighboring-agent information in a form that can be shared across agents.

For this repository, the combined design must be derived from each agent's knowledge rather than from a centralized map:

```text
local sensing + self trajectory + received map-intent messages
    -> private or communication-fused memory map for agent i
    -> re-center around agent i
    -> crop/compress to a fixed-size spatial tensor
    -> shared actor spatial encoder

neighbor message summaries and relative geometry
    -> optional GAT aggregation
    -> coordination feature fused with actor spatial feature
```

This design gives the actor a spatially structured view of what it actually knows. It should help distinguish a locally convenient move from a move that advances exploration of a remembered frontier, while GAT remains responsible for coordinating with nearby agents rather than substituting for spatial memory.

A candidate actor tensor should keep knowledge sources distinguishable:

| Channel | Meaning | Permitted source |
| --- | --- | --- |
| `known_free` | Cells known to be traversable | Local sensing or received map memory |
| `known_obstacle` | Cells known to be blocked | Local sensing or received map memory |
| `unknown` | Cells for which the agent has no knowledge | Derived from its memory mask only |
| `self_covered` | Coverage generated by this agent | Own trajectory memory |
| `communicated_team_covered` | Teammate coverage that this agent has learned | Received/fused memory only |
| `recent_path` | Recent ego motion footprint | Own trajectory memory |
| `known_teammates` | Currently sensed or communicated teammate positions | Observation/message availability only |
| `frontier_or_candidate` | Reachable exploration candidates in known memory | Derived from permitted memory layers |
| `current_intent` | The agent's selected or proposed exploration region | Derived from its own memory/policy state |

The current small local window plus vector-style map-intent message is evaluated together with the selected fixed-cost reward and conservative feasibility mask. Further representation, reward, mask, or energy variants must be introduced as separately identified ablations:

| Arm | Spatial actor input | Map sharing | GAT | Question |
| --- | --- | --- | --- | --- |
| Existing baseline | Current local observation | Existing map-intent design | off/on | What does the corrected current baseline achieve? |
| Private centered memory | Agent-centered private memory crop | None | off | Does structured ego memory reduce local wandering by itself? |
| Fused centered memory | Agent-centered fused-memory crop | Enabled | off | Does shared map knowledge help without attention aggregation? |
| Fused centered memory + GAT | Agent-centered fused-memory crop | Enabled | on | Does message attention add useful coordination once spatial memory is visible? |

These arms should use matched courses, seeds, training budgets, rewards, and feasibility rules. Improvement should be judged first on `Coverage@H`, `Coverage-AUC`, `T90 / T95 / T99`, `RepeatRatioAfter90`, and `InterAgentOverlapRatio`, then on zero-shot larger-map and changed-agent-count evaluations.

### Map Expansion Transfer

The current curriculum intentionally caps training at `20 x 20` to keep the principal ablations manageable. The mobile-robot paper suggests a way to evaluate larger maps later without redesigning the actor input dimension.

A compatible future implementation should build an egocentric fixed-size tensor from each agent's explicit memory:

```text
agent private/fused memory map
    -> center around current agent
    -> preserve a fixed local high-resolution window
    -> compress out-of-window remembered regions into border summaries
    -> feed a fixed observation shape to the shared actor
```

The high-resolution center should preserve locally actionable information:

- current position and locally visible teammates;
- known obstacles and known free cells;
- self-covered and communicated known-team-covered cells;
- unknown and frontier cells;
- recent path and current map-derived intent.

Compressed border summaries can later encode directional long-range information from remembered cells:

- known uncovered or frontier mass by direction;
- known coverage density by direction;
- nearest remembered candidate distance in each direction;
- intended target region relative to the current agent.

Compression must be derived from remembered information only. An agent that has not sensed or received information about a remote region must continue to represent it as unknown rather than receiving the true remote map.

Initial implementation support has been added behind an opt-in observation mode:

```toml
[env]
use_explicit_map_memory = true
observation_mode = "centered_compressed_memory"
centered_map_size = 7
compressed_border = true
```

This mode keeps the actor map tensor size fixed with respect to environment width and height. The high-resolution interior is centered on the acting agent, while remembered cells outside that interior are summarized into the outer border. The compression reads only per-agent memory layers such as known free cells, known obstacles, self coverage, communicated known-team coverage, unknown cells, frontier cells, recent path, and currently sensed or communication-reachable teammates. It does not compress global environment truth into the actor observation.

Recommended map-size experiment stages:

| Stage | Training maps | Test maps | Purpose |
| --- | --- | --- | --- |
| Baseline | Up to `20 x 20` | Held-out `20 x 20` | Finish map-intent/GAT comparison without changing observation semantics. |
| Representation ablation | Up to `20 x 20` | `24 x 24`, `30 x 30` | Test whether memory-derived egocentric compression permits zero-shot size transfer. |
| Stress evaluation | Up to `20 x 20` | Larger maps at fixed obstacle ratios | Separate map-size generalization from obstacle-density difficulty. |

Do not silently mix larger maps with higher obstacle ratios. Map extent and obstacle density must remain separate experimental factors.

### Agent-Count Expansion Transfer

The dynamic-UAV-network paper motivates evaluating policies outside the trained number of agents. This repository is already structurally close to that idea:

- the actor parameters are shared across agents;
- map-intent messages have a fixed dimension per agent;
- GAT takes a dynamic neighbor mask and can aggregate a variable number of neighbors;
- the centralized state represents all current positions on a map layer rather than concatenating one fixed block per agent.

The remaining question is empirical: a policy trained through four agents may not coordinate well with additional neighbors, especially when communication creates conflicting exploration intents.

A future agent-count experiment should hold the map family and reward fixed while varying only `num_agents`:

| Train range | Zero-shot test range | Main question |
| --- | --- | --- |
| `1` to `4` agents | `5`, `6`, and possibly `8` agents | Does the shared policy exploit extra agents or merely increase overlap? |
| `1` to `4` agents | Fewer-than-expected agents | Does the policy remain effective when communication opportunities disappear? |

Important metrics for agent-count transfer:

- `Coverage@H` and `Coverage-AUC`;
- `T90 / T95 / T99`;
- `RepeatRatioAfter90`;
- `InterAgentOverlapRatio`;
- communication neighbor counts and attention statistics;
- per-agent new-cell contribution distributions.

The desired outcome is not simply increased final coverage with more agents. Additional agents should improve budgeted coverage without causing overlap to rise so quickly that cooperation becomes inefficient.

### UAV Energy-Budget Extension

Battery capacity is a useful UAV-specific constraint, but it should enter as a separate experimental factor after the observation baseline is established. It is not needed to justify the agent-centered representation, and it should not be introduced at the same time as a new observation design or reward redesign.

A first energy-aware environment extension can maintain an individual battery state for each UAV:

```text
battery_i(t + 1) = battery_i(t) - movement_cost - optional_step_cost
normalized_remaining_battery_i = battery_i(t) / battery_capacity_i
```

The initial version should keep the accounting simple and auditable: fixed movement/step costs, no wind model, no charging behavior, and no communication-energy term unless it is introduced later as its own ablation. The actor may receive its own normalized remaining battery, while team battery summaries should only be delivered through explicit communication if they are part of the decentralized execution design.

The single-UAV condition must be a non-binding battery control. Its battery capacity should be far greater than the energy required to complete the task under the evaluation horizon, for example at least several times a conservative full-episode energy bound. A single-UAV run that fails because the battery is depleted would confound path-planning quality with an unnecessarily restrictive resource setting.

After the observation design is validated, multi-UAV energy experiments can introduce binding budgets deliberately:

| Experiment | Battery setting | Purpose |
| --- | --- | --- |
| Single-UAV control | Capacity far above task requirement | Confirm the baseline path-planning behavior without energy pressure. |
| Multi-UAV non-binding control | Per-agent capacity comfortably above horizon cost | Compare cooperation behavior without depletion confounds. |
| Multi-UAV constrained energy | Matched finite per-agent capacity | Test whether coordination improves coverage under realistic endurance limits. |
| Agent-count energy fairness | Fixed per-agent and fixed-team-energy variants reported separately | Separate the benefit of more agents from simply adding more total battery. |

Energy-aware metrics should include `coverage_per_energy`, remaining battery at termination, depletion/interruption rate, and per-agent energy contribution alongside the existing coverage and overlap metrics. If a return-to-base or reserve-safety requirement is later introduced, it must be treated as a new task definition rather than silently folded into the existing coverage benchmark.

### Reward Function Transfer

The first paper can support the general motivation for map-size-invariant reward/value scaling, but it should not be cited as proof that this project's reward function is effective. It does not provide the same multi-agent fixed-budget evaluation required here.

The second paper supplies a more relevant multi-agent reward reference. Its event-driven formulation distinguishes:

- new cells of interest covered by the acting agent;
- cells previously covered by another agent;
- time-step cost;
- collision cost;
- a cooperation coefficient `K` that trades individual progress against team interaction.

Conceptually, the paper weights an agent's reward using an ego term scaled by `(1 - K)` and contributions from other agents scaled by a normalized `K` share. This idea is relevant to the current failure mode: agents can obtain good team coverage while still duplicating work because individual contribution and conflict are not clearly represented.

The current map-intent baseline now adopts the first conservative reward revision motivated by this framing:

```text
shared_reward =
    useful team new coverage / num_agents
  - avoidable repeat cost / num_agents
  - invalid or collision cost / num_agents
  - fixed time-step cost
  + small normalized team completion bonus
```

The fixed step cost is the essential change: it charges late tail search at the same rate as early motion. The completion bonus is reduced to `10.0` total and normalized by agent count. This is this repository's reward choice; the referenced paper motivates the direction but does not prove this exact MAPPO formula.

Design constraints for this repository:

- Normalize new-coverage terms by available free area so reward scale does not grow automatically with map size.
- Decide explicitly whether team terms are divided by agent count so reward scale does not grow automatically with the number of robots.
- Treat `K` as an experimental cooperation parameter, not as a fixed truth; compare several values under matched seeds.
- Keep completion reward, if retained, as this repository's own independently ablated design decision. It should not be attributed to either reference paper unless a specific cited formula supports that statement.
- Use global truth for reward computation only; never expose the corresponding global team-coverage signal directly to the actor.

Suggested follow-up reward ablations after this fixed-cost baseline:

| Ablation | Fixed components | Variable | Interpretation |
| --- | --- | --- | --- |
| Reward scale | Memory and mask fixed | absolute vs free-area-normalized coverage gain | Tests map-size stability of critic targets. |
| Cooperation weight | Normalized reward and mask fixed | `K` values | Tests individual/team contribution balance. |
| Conflict penalty | Best `K` and mask fixed | repeat/overlap penalty on/off | Tests whether coordination improves beyond team reward. |
| Completion bonus | All above fixed | none/small/current | Tests whether terminal incentive helps or distorts budgeted coverage. |

Reward conclusions must be based on this project's metrics rather than reward magnitude alone:

- primary: `Coverage@H`, `Coverage-AUC`, `T90 / T95`, `RepeatRatioAfter90`, and `InterAgentOverlapRatio`;
- diagnostic: `T99`, `StallCoverage@K`, per-agent contribution, and collisions;
- supplementary: `completion_rate`, `T100`, and total reward.

### Action Mask And Safety Filter Transfer

The mobile-robot paper uses an action masking scheme associated with feasibility and robustness; the multi-agent paper also includes a safety and robustness filtering stage. These mechanisms are relevant but must be separated conceptually in this repository:

- **Logit-level action mask:** remove known infeasible actions before PPO samples an action.
- **Post-policy safety filter:** accept or replace the selected action after inference.
- **Heuristic policy switch:** hand control to a non-learned policy under selected conditions.

These three mechanisms change the learning problem differently and should not be introduced together under one label.

The enabled map-intent action mask is conservative and decentralized:

```text
allow action if the agent cannot prove it is infeasible
mask action only if it leads out of bounds or into a locally known obstacle
```

Rules enforced by the current implementation:

- Do not mask an unknown obstacle using environment truth that the actor has not observed or received.
- Do not mask moves into cells covered only by teammates unless that coverage is present in the agent's communicated memory and the experiment explicitly studies repeat-avoidance masking.
- Do not mask simultaneous multi-agent collision outcomes using other agents' future selected actions unless a separate intention-sharing or joint-planning mechanism is introduced.

Recommended action-mask ablations:

| Arm | Mask knowledge | Purpose |
| --- | --- | --- |
| No mask | None | Existing learned-avoidance baseline. |
| Local feasibility mask | Bounds plus locally known obstacles | Tests safe sample-efficiency improvement without privacy leakage. |
| Fused-memory feasibility mask | Bounds plus obstacles received through map fusion | Tests whether communication improves feasible action selection. |
| Repeat-avoidance mask, optional | Known covered information only | Must be treated as a separate behavioral intervention, not basic safety. |

Both map-intent GAT arms enable the same fixed-cost reward and the same feasibility mask, so their GAT-on/GAT-off comparison remains matched. A separate no-mask run is required before attributing any improvement to masking itself.

### Implementation Order And Claim Boundary

Recommended order for integrating ideas from the two papers:

1. Train and evaluate matched `ablation_mapmsg_gat_on` / `ablation_mapmsg_gat_off` runs using fixed time cost, small completion bonus and the shared decentralized feasibility mask.
2. Compare against a matched no-mask arm before claiming sample-efficiency gains from action masking.
3. With the selected observation and feasibility behavior fixed, evaluate zero-shot map-size and agent-count transfer.
4. Ablate free-area reward normalization, cooperation weighting and explicit overlap penalties.
5. Add UAV battery state as a separately controlled constraint: begin with deliberately non-binding single-UAV capacity, then evaluate bounded multi-UAV endurance.

Claims that this repository may reasonably make after successful experiments:

- the representation or action mask is inspired by the zero-shot mobile-robot paper;
- the multi-agent observation channels are adapted from the dynamic-UAV-network paper under this repository's decentralized knowledge boundary;
- reward normalization and cooperation-weight ablations are motivated by the dynamic-UAV-network paper;
- the proposed map-intent/GAT or coverage-conflict mechanism is evaluated for decentralized multi-agent coverage in this repository.

Claims to avoid unless directly reproduced:

- that the referenced papers prove this MAPPO reward is optimal;
- that larger-map generalization is established before the compression implementation is tested;
- that dynamic-agent generalization is established before held-out agent-count evaluation;
- that an energy-aware UAV result follows from the referenced papers unless the energy constraint and fairness controls are explicitly implemented and evaluated;
- that a policy is decentralized if its actor receives environment-truth global coverage or oracle feasibility masks.

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

The current map-intent configs use sparse deterministic evaluation/checkpointing to reduce overhead:

```toml
[train]
log_interval = 1
eval_interval = 10
checkpoint_interval = 10
```

Training metrics are still written as completed episodes arrive. Eval metrics and intermediate `last_policy.pt` saves happen every ten PPO updates. This reduces deterministic eval and checkpoint I/O cost without reducing training samples.

## Convergence Tuning

The current stability pass fixes the training objective around budgeted coverage by using a constant time price, a smaller completion bonus and a conservative known-feasibility action mask.

The main task interpretation is budgeted online coverage in an unknown environment:

```text
Within a fixed task budget, maximize effective coverage progress while minimizing
repeat coverage, infeasible motion and inter-agent conflict.
```

Exact full coverage remains useful evidence, but it should not dominate the evaluation. The final few unobserved or uncovered cells can create a large tail cost in an unknown environment, so a policy that quickly reaches high coverage should not be judged only by whether it finds every final isolated cell.

Main metrics to judge future convergence and model quality:

- `Coverage@H`, evaluated at fixed environment-step budgets.
- `Coverage-AUC`.
- `T90` and `T95`.
- `RepeatRatioAfter90` and `InterAgentOverlapRatio`.
- the same metrics on held-out obstacle seeds.

Secondary diagnostics:

- `T99` and `StallCoverage@K`, for diagnosing late-stage search and stalls.
- `completion_rate` and `T100`, for strict-full-coverage reference only.
- `eval/reward`, because reward magnitude can be distorted by `finish_reward` and should not be read alone.
- path length and rendered trajectory appearance, for resource interpretation and qualitative inspection rather than as primary measures of path beauty or straight-line regularity.

These expanded metrics are produced by the offline `evaluate`, `benchmark`, and `gat-ablation` measurement flow. Reward and action-mask changes apply only to new configurations/snapshots that enable them; archived snapshots retain their legacy reward and feasibility semantics.

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

If course 4 still has unstable coverage-efficiency metrics, the next low-risk lever is learning rate: test `learning_rate = 0.0001` for courses 3 and 4, or `learning_rate = 0.00005` for course 4. Larger rollouts increase compute time and memory use.

The current map-intent configs also use a larger PPO mini-batch:

```toml
mini_batch_size = 256
```

This reduces optimizer overhead on CPU. With the current course-4 values, one rollout contains `2048 / 4 = 512` synchronized environment steps and each mini-batch contains `256 / 4 = 64` synchronized steps, so one PPO update runs `8 minibatches * 4 epochs = 32` backward passes.

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

The archived GAT-on/GAT-off comparison has already produced comparable course results, but it used the legacy truth-observation actor input. Use the official MAPPO implementation as an engineering reference for later corrected-baseline stabilization work, absorbing one mechanism at a time.

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

Recommended order for later corrected-baseline stabilization work:

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

The archived source config filenames and report artifacts are retained for audit:

- `configs/ablation_gat_on.toml`: same curriculum, `use_graph_attention = true`.
- `configs/ablation_gat_off.toml`: same curriculum, `use_graph_attention = false`.

Do not use these archived configs to start the current training line or initialize a map-intent checkpoint. Historical checkpoint replay should use the saved configuration alongside its artifact.

The map-intent ablation is the current training workflow with separate output roots:

```text
train --config configs/ablation_mapmsg_gat_on.toml --course tier-1-8x8-1agent
train --config configs/ablation_mapmsg_gat_off.toml --course tier-1-8x8-1agent
```

Continue courses 2 through 4 using the matching arm's preceding `best_policy.pt`, then invoke `gat-ablation`; its default configs are now `ablation_mapmsg_gat_on.toml` and `ablation_mapmsg_gat_off.toml`. Do not compare a legacy GAT checkpoint directly against a map-intent checkpoint as though only attention differed; their observation and message semantics are different.

After both current arms finish, compare them on the same held-out seeds and obstacle ratios:

```text
gat-ablation --gat-on-checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\04-tier-4-20x20-4agents\best_policy.pt" --gat-off-checkpoint "E:\test plot\ablation_mapmsg_gat_off\<run>\04-tier-4-20x20-4agents\best_policy.pt" --seeds 20260601,20260602,20260603,20260604,20260605 --obstacle-ratios 0.05,0.10,0.15,0.20 --output "E:\test plot\mapmsg_gat_ablation_summary.csv"
```

The summary CSV contains `gat_on`, `gat_off`, and `delta_on_minus_off`. The primary fields are `coverage_at_<budget>_mean`, `coverage_auc_mean`, `t90/t95/t99_mean_reached`, `t90/t95/t99_reach_rate`, and `stall_termination_coverage_mean`.

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

For every run configuration below, keep the working directory and interpreter the same. The commands show the GAT-on arm; for GAT-off, replace `ablation_mapmsg_gat_on` with `ablation_mapmsg_gat_off` in both the config path and checkpoint output root.

### Course 1

Training:

```text
train --config configs/ablation_mapmsg_gat_on.toml --course tier-1-8x8-1agent
```

Testing:

```text
evaluate --config configs/ablation_mapmsg_gat_on.toml --checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\01-tier-1-8x8-1agent\policy.pt"
```

Final path drawing:

```text
render --config configs/ablation_mapmsg_gat_on.toml --checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\01-tier-1-8x8-1agent\policy.pt"
```

### Course 2

Training:

```text
train --config configs/ablation_mapmsg_gat_on.toml --course tier-2-13x13-2agents --previous-checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\01-tier-1-8x8-1agent\best_policy.pt"
```

Testing:

```text
evaluate --config configs/ablation_mapmsg_gat_on.toml --checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\02-tier-2-13x13-2agents\policy.pt"
```

Final path drawing:

```text
render --config configs/ablation_mapmsg_gat_on.toml --checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\02-tier-2-13x13-2agents\policy.pt"
```

### Course 3

Training:

```text
train --config configs/ablation_mapmsg_gat_on.toml --course tier-3-18x18-3agents --previous-checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\02-tier-2-13x13-2agents\best_policy.pt"
```

Testing:

```text
evaluate --config configs/ablation_mapmsg_gat_on.toml --checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\03-tier-3-18x18-3agents\policy.pt"
```

Final path drawing:

```text
render --config configs/ablation_mapmsg_gat_on.toml --checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\03-tier-3-18x18-3agents\policy.pt"
```

### Course 4

Training:

```text
train --config configs/ablation_mapmsg_gat_on.toml --course tier-4-20x20-4agents --previous-checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\03-tier-3-18x18-3agents\best_policy.pt"
```

Testing:

```text
evaluate --config configs/ablation_mapmsg_gat_on.toml --checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\04-tier-4-20x20-4agents\policy.pt"
```

Final path drawing:

```text
render --config configs/ablation_mapmsg_gat_on.toml --checkpoint "E:\test plot\ablation_mapmsg_gat_on\<run>\04-tier-4-20x20-4agents\policy.pt"
```

## Notes

- Legacy truth-observation checkpoints can be replayed with their saved configuration for historical evaluation, but they must not initialize corrected private-local or map-intent training runs. Start corrected runs again from course 1.
- The enhanced GAT adds multi-head attention and edge-bias parameters. Older preliminary GAT checkpoints should be treated as a separate historical baseline, not mixed into corrected communication comparisons.
- The current map-intent course 4 is `tier-4-20x20-4agents`; old `30x30` course-4 checkpoints belong to the previous curriculum and should not be mixed into the current ablation.
- The `--previous-checkpoint` path is optional for later courses if you want the command to reuse the shared curriculum state file automatically, but in PyCharm it is clearer to point directly at the previous course's `best_policy.pt`.
- The `checkpoint=` line printed after training is the path you should paste into the corresponding `evaluate` and `render` configurations.
- `evaluate` writes `trajectory.json`, and `render` writes `trajectory.png` next to that checkpoint.
- `best_policy.pt` is selected by deterministic evaluation during training, using the same policy mode as the test command.

## Tests

```powershell
E:\miniconda3\envs\two-stage-mcpp\python.exe -m unittest tests.test_config_env tests.test_rewards tests.test_ppo_render
```
