# Development Log

## 2026-05-16

- Reworked the formal MAPPO curriculum after the 20x20/2-agent run showed near-complete training coverage but poor deterministic completion and late-stage local oscillation.
- Replaced the previous 6x6 -> 8x8 -> 20x20 schedule with a smoother four-tier curriculum:
  - `tier-1-8x8-1agent`: 8x8, 1 agent, 100 max steps, 500000 PPO timesteps.
  - `tier-2-13x13-2agents`: 13x13, 2 agents, 180 max steps, 1000000 PPO timesteps.
  - `tier-3-18x18-3agents`: 18x18, 3 agents, 300 max steps, 2000000 PPO timesteps.
  - `tier-4-30x30-4agents`: 30x30, 4 agents, 550 max steps, 4000000 PPO timesteps.
- Standardized formal courses on `observation_radius = 2`, giving each actor a 5x5 local observation window while preserving decentralized actor inputs.
- Standardized obstacle generation with `obstacle_ratio = 0.09375`, matching the original 8x8 density of 6 obstacles over 64 cells.
- Kept map size and agent count fully config-driven through `width`, `height`, `num_agents`, `observation_radius`, and `obstacle_ratio`; no environment or model logic was specialized to the new course dimensions.
- Updated README training/evaluation/rendering instructions for the four-tier curriculum and TensorBoard log paths.
