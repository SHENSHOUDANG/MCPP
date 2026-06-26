# Model Specification

This document is the durable theoretical contract for the project. It should be specific enough that a new Codex thread can implement or review scheduler logic without relying on chat history.

Implementation details belong in code. Current work status belongs in `docs/current_task.md`. Experiment outcomes belong in `docs/experiment_log.md`.

## Scope

The repository contains two related model families:

- grid-coverage PPO/MAPPO baselines;
- port inspection scheduling for a coupled UAV/USV workflow.

The current port scheduler scenario is represented by:

- `configs/port_yangshan_task_initial_v1.toml`;
- `data/ports/yangshan_task_initial_v1/yangshan_task_initial_v1_grid.json`;
- `data/ports/yangshan_task_initial_v1/yangshan_task_initial_v1_tasks.json`.

The port scheduler is the current thesis-facing model. It schedules inspection tasks at an abstract operational level. It does not emit low-level flight, vessel, or collision-avoidance control commands.

## Sets And Entities

Let:

- `U = {u_1, ..., u_m}` be the UAV set.
- `V = {v_1, ..., v_n}` be the USV set.
- `A = U union V` be all mobile inspection agents.
- `T = {1, ..., N}` be all inspection task nodes.
- `D_U` and `D_V` be UAV and USV depot locations. They may be the same grid coordinate in the current Yangshan scenario.

Each agent `a in A` has:

- platform type: UAV or USV;
- current location `x_a`;
- remaining energy or endurance `e_a`;
- availability state: idle, assigned, screening, reviewing, returning, replenishing, or inactive;
- current task id, if any;
- elapsed service time on the current task;
- platform profile parameters, including speed, service capability, endurance, and compatible task classes.

Each task `i in T` has:

- location `p_i`;
- task class or facility type;
- risk level `r_i`;
- screening state;
- review state;
- visibility/confidence state;
- deadline or soft due time;
- service-time estimates for compatible UAV/USV platforms;
- compatibility mask over platform types.

## Screening And Review Tasks

The workflow is two-layer:

1. UAV screening performs fast initial inspection or anomaly detection.
2. USV review performs slower close-range verification for tasks that require follow-up.

A screening task is complete when a compatible UAV finishes the screening service at task `i`.

A review task is complete when a compatible USV finishes the review service at task `i`.

A task may be in one of these coupled states:

- `unseen`: no screening has been performed.
- `screening_assigned`: a UAV has been assigned but has not completed screening.
- `screened_clear`: screening finished and no review is required.
- `review_pending`: screening finished and review has been triggered.
- `review_assigned`: a USV has been assigned but has not completed review.
- `reviewed`: review finished.

Only `screened_clear` and `reviewed` count as terminal task states. For high-risk tasks, `screened_clear` is allowed only if the trigger model says review is not required.

## Review Trigger

Review is triggered after UAV screening using a configurable trigger model.

Inputs:

- task risk level `r_i`;
- UAV screening confidence `c_i`;
- anomaly probability by risk;
- mandatory review risk threshold;
- stochastic or deterministic trigger setting;
- optional false-positive/false-negative sensitivity and specificity parameters.

Baseline rule:

```text
review_required(i) =
  (r_i >= mandatory_review_risk)
  OR (c_i < confidence_threshold)
  OR sampled_anomaly(i)
```

If review is triggered, the task enters `review_pending` and receives a review deadline.

Review deadline:

```text
deadline_i = screening_finish_time_i
             + base_review_deadline
             - risk_deadline_scale * r_i
             + confidence_deadline_scale * c_i
```

The deadline may be clamped by config to avoid negative or unrealistically short windows.

## Backpressure From USV Review Queue

UAV screening should not greedily create more review work than USVs can handle. Downstream USV backlog feeds back into UAV scheduling through state features, candidate scoring, action masking only when explicitly configured, and reward.

Backlog variables:

- `B_count`: number of pending or assigned-but-unfinished review tasks.
- `B_risk`: risk-weighted review backlog.
- `B_deadline`: urgency-weighted backlog, increasing as review deadlines approach.
- `B_capacity`: estimated USV review capacity over a lookahead horizon.
- `B_ratio = backlog / max(capacity, epsilon)`.

Backpressure effects:

- UAV observations include review backlog summaries.
- UAV candidate scores penalize screening tasks likely to trigger review when `B_ratio` is high.
- Rewards penalize avoidable review backlog growth and missed review deadlines.
- Hard action masks for UAV screening are allowed only as an explicit ablation, because they change the learning problem.

The intended default is soft backpressure through state and reward, not silent hard blocking.

## State Space

The scheduler state at decision step `k` contains global simulator state and per-agent observations.

Global state may include:

- current decision time `t`;
- all agent locations, energy, availability, current assignments, and remaining service times;
- all task coupled states;
- screen/review completion flags;
- review queue size, risk mix, deadlines, and capacity summaries;
- depot locations;
- platform compatibility data;
- remaining episode budget.

Per-agent observation may include:

- own platform type, location, energy, and availability;
- current assignment summary;
- candidate task features for that agent;
- local or global task progress summaries;
- review backlog and urgency summaries;
- normalized time remaining;
- distance-to-depot and estimated energy-to-depot;
- compatibility indicators.

Centralized critic state may include full global state. Decentralized actor observations must not include hidden future outcomes such as whether a not-yet-screened task will definitely trigger review, unless that variable is explicitly modeled as known prior risk.

## Action Space

At each decision step, each agent chooses one abstract scheduling action.

Allowed action types:

- assign a candidate screening task;
- assign a candidate review task;
- wait;
- return to depot;
- replenish at depot, when at depot and eligible.

For UAVs:

- default productive action is screening assignment;
- UAVs do not perform USV-only review unless platform compatibility explicitly allows it.

For USVs:

- default productive action is review assignment;
- USVs may wait, return, or replenish when no review task is suitable.

Joint decision:

- MAPPO may sample one action per agent simultaneously.
- The environment resolves conflicts after all agent actions are proposed.
- A centralized planner baseline may produce a joint assignment directly, but the RL environment still needs deterministic conflict rules.

## Candidate Task Generation

Each agent receives a bounded candidate set to keep the action space stable.

Candidate generation steps:

1. Filter by task state:
   - UAV screening candidates: `unseen` tasks and optionally stale screening candidates.
   - USV review candidates: `review_pending` tasks.
2. Filter by platform compatibility.
3. Filter by reachability and energy feasibility:
   - enough energy to reach task, perform service, and reach depot or safe replenishment point.
4. Score candidates using distance, risk, deadline urgency, load balance, expected review creation, and compatibility bonus.
5. Keep top `candidate_k` candidates.
6. Fill unused slots with invalid/no-task placeholders.

Candidate features should be normalized and include at least:

- distance from agent to task;
- distance from task to depot;
- task risk;
- task deadline slack if review-related;
- expected service time;
- compatibility;
- current task state;
- estimated downstream review burden for screening candidates.

## Conflict Resolution

When multiple agents choose the same task, the environment resolves conflict deterministically.

Default rule:

1. Reject agents incompatible with the task.
2. Prefer the agent with the lower estimated arrival-plus-service completion time.
3. Break ties by higher remaining energy.
4. Break remaining ties by stable agent id.

Accepted assignment changes the task state to assigned. Rejected agents receive the configured conflict penalty and either wait for the step or keep their previous assignment, depending on environment mode. The default should be wait-on-reject for clarity.

For simultaneous UAV screening and USV review choices on the same task:

- a USV can review only if the task is already `review_pending` at the start of the decision step;
- screening completion and review trigger take effect after service completion, not at assignment time.

## Reward Components

The total reward is a weighted sum. Weights live in config.

Positive terms:

- screening progress reward for completed UAV screening;
- review progress reward for completed USV review;
- high-risk completion bonus;
- team close reward when all required work is done;
- deadline-safe completion bonus for finishing review before due time.

Negative terms:

- travel or distance cost;
- energy cost;
- time cost;
- invalid action penalty;
- conflict penalty;
- missed review deadline penalty;
- backlog penalty when pending review work exceeds estimated USV capacity;
- unnecessary waiting penalty when feasible useful work exists;
- failed-return or out-of-energy penalty.

Backpressure reward should penalize avoidable review queue growth without making UAVs ignore genuinely high-risk tasks. A useful default is risk- and urgency-weighted backlog cost rather than a flat count penalty.

## Task Completion And Episode Termination

Task-level completion:

- screening-only task: complete when UAV screening finishes and review is not triggered;
- review-required task: complete when USV review finishes;
- incompatible or unreachable tasks should be reported as infeasible, not silently counted complete.

Episode success:

- all tasks reach terminal task states;
- no mandatory review remains pending;
- optional depot-return requirement is satisfied if enabled.

Episode terminates when:

- all required tasks are complete and depot-return condition is satisfied;
- max scheduler steps or time horizon is reached;
- all agents become inactive or no feasible productive action remains;
- hard safety failure occurs, such as unrecoverable energy exhaustion.

Timeout is not equivalent to success. Evaluation must report partial completion metrics.

## Depot Return And Replenishment

Depot return is configurable.

If `require_return_to_depot = true`, agents must return to their depot after completing assigned work before team success is declared.

Even when final return is not required, agents must preserve enough energy to reach a depot unless the experiment explicitly disables energy safety.

Replenishment is triggered when:

- an agent is at depot;
- remaining energy is below a configured threshold or the selected action is replenish;
- replenishment is not already complete.

Replenishment consumes time steps and restores energy according to the platform profile or config. Replenishment should not erase task state or assignment history.

## Deadlines, Risk, And Energy

Deadline slack:

```text
slack_i = deadline_i - current_time - estimated_finish_time(agent, i)
```

Urgency can be computed as:

```text
urgency_i = max(0, 1 - slack_i / deadline_window_i)
```

Risk-weighted urgency:

```text
risk_urgency_i = risk_weight * r_i * urgency_i
```

Energy feasibility:

```text
energy_needed = energy(agent_location, task_location)
              + service_energy(agent, task)
              + energy(task_location, depot(agent))
```

A task is energy-feasible if:

```text
remaining_energy_agent >= energy_needed + reserve_energy
```

Distance may be Euclidean over UTM-derived feature bins or another configured travel proxy. If a low-level route planner is introduced later, it must be documented as a separate modeling change.

## Evaluation Metrics For The Thesis

Report aggregate metrics over matched seeds/scenarios.

Primary metrics:

- total task completion rate;
- screening completion rate;
- review completion rate;
- high-risk task completion rate;
- deadline miss rate for review tasks;
- mean and percentile review delay;
- risk-weighted unfinished workload;
- total mission time or makespan;
- total travel distance or energy cost;
- review backlog AUC over time;
- UAV idle time caused by downstream review backpressure;
- USV utilization and idle time.

Coordination metrics:

- assignment conflict rate;
- duplicate assignment rate;
- invalid action rate;
- average candidate acceptance rate;
- workload balance across UAVs and USVs.

Learning metrics:

- episodic return;
- policy entropy;
- value loss and policy loss;
- success rate under deterministic evaluation;
- generalization across held-out task seeds or risk distributions.

Baseline comparisons should include at least:

- greedy nearest-feasible assignment;
- risk-priority greedy assignment;
- no-backpressure scheduler;
- learned scheduler with backpressure features/reward.

Raw metrics, traces, checkpoints, figures, and reports are local artifacts. Summaries and conclusions belong in `docs/experiment_log.md`.

## Grid-Coverage Contract

For the grid-coverage PPO/MAPPO side of the repository:

- agents operate on grid maps with bounded episode length;
- policy observations should not silently include global truth that would make a decentralized actor unrealistic;
- centralized information may be used for reward, critic state, metrics, rendering, and diagnostics;
- optional communication mechanisms such as GAT, map messages, CUAP, CIR, or gated CUAP must remain explicit config choices.

## Training Output Contract

Training outputs are reproducible local artifacts, not source:

- checkpoints;
- metrics and summaries;
- evaluation traces;
- rendered plans and figures;
- TensorBoard events and logs.

When an experiment result matters, summarize the setup and conclusion in `docs/experiment_log.md` and keep raw artifacts outside Git.
