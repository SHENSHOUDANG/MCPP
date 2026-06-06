from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CUAPConfig, GridPosition
from .env import ACTIONS, GridCoverageEnv


@dataclass(frozen=True, slots=True)
class _MemoryView:
    known_free: set[GridPosition]
    known_team_covered: set[GridPosition]
    known_obstacles: set[GridPosition]
    frontier: set[GridPosition]


def compute_cuap_logits(env: GridCoverageEnv, cfg: CUAPConfig, phase: str = "coverage") -> np.ndarray:
    """Compute agent-agnostic CUAP action-prior logits with shape ``[agent, action]``."""
    scores = np.zeros((env.num_agents, env.action_dim), dtype=np.float32)
    if not cfg.enabled:
        return scores
    if cfg.disable_in_return_phase and phase == "return":
        return scores

    action_masks = env.action_masks()
    neighbor_mask = env.neighbor_mask()
    messages = env.node_messages()
    for agent_index, position in enumerate(env.positions):
        memory = _memory_view(env, agent_index)
        neighbor_regions = _neighbor_intent_regions(env, agent_index, neighbor_mask, messages)
        for action, delta in ACTIONS.items():
            if not action_masks[agent_index, action]:
                continue
            candidate = (position[0] + delta[0], position[1] + delta[1])
            novelty = _novelty(candidate, memory)
            frontier = _frontier(candidate, memory)
            repeat = _repeat(candidate, memory)
            conflict = _conflict(candidate, neighbor_regions)
            scores[agent_index, action] = (
                cfg.w_novelty * novelty
                + cfg.w_frontier * frontier
                - cfg.w_repeat * repeat
                - cfg.w_conflict * conflict
            )

    if cfg.normalize:
        mean = scores.mean(axis=-1, keepdims=True)
        std = scores.std(axis=-1, keepdims=True)
        scores = (scores - mean) / (std + 1e-6)
    if cfg.clip > 0:
        scores = np.clip(scores, -cfg.clip, cfg.clip)
    return scores.astype(np.float32)


def scaled_cuap_prior(env: GridCoverageEnv, cfg: CUAPConfig, phase: str = "coverage") -> np.ndarray | None:
    if not cfg.enabled:
        return None
    return (float(cfg.beta) * compute_cuap_logits(env, cfg, phase=phase)).astype(np.float32)


def _memory_view(env: GridCoverageEnv, agent_index: int) -> _MemoryView:
    known_free = set(env.known_free_by_agent[agent_index])
    known_team_covered = set(env.known_team_covered_by_agent[agent_index])
    known_obstacles = set(env.known_obstacles_by_agent[agent_index])
    unknown = _memory_unknown(env, known_free, known_obstacles)
    frontier = _memory_frontiers(known_free, unknown)
    return _MemoryView(
        known_free=known_free,
        known_team_covered=known_team_covered,
        known_obstacles=known_obstacles,
        frontier=frontier,
    )


def _memory_unknown(
    env: GridCoverageEnv,
    known_free: set[GridPosition],
    known_obstacles: set[GridPosition],
) -> set[GridPosition]:
    known = known_free | known_obstacles
    return {
        (row, col)
        for row in range(env.config.height)
        for col in range(env.config.width)
        if (row, col) not in known
    }


def _memory_frontiers(known_free: set[GridPosition], unknown: set[GridPosition]) -> set[GridPosition]:
    return {
        cell
        for cell in known_free
        if any((cell[0] + delta[0], cell[1] + delta[1]) in unknown for delta in ACTIONS.values())
    }


def _novelty(candidate: GridPosition, memory: _MemoryView) -> float:
    return float(candidate in memory.known_free and candidate not in memory.known_team_covered)


def _frontier(candidate: GridPosition, memory: _MemoryView) -> float:
    return float(candidate in memory.frontier)


def _repeat(candidate: GridPosition, memory: _MemoryView) -> float:
    return float(candidate in memory.known_team_covered)


def _conflict(candidate: GridPosition, neighbor_regions: list[set[GridPosition]]) -> float:
    if not neighbor_regions:
        return 0.0
    hits = sum(1.0 for region in neighbor_regions if candidate in region)
    return float(hits / max(len(neighbor_regions), 1))


def _neighbor_intent_regions(
    env: GridCoverageEnv,
    agent_index: int,
    neighbor_mask: np.ndarray,
    messages: np.ndarray,
) -> list[set[GridPosition]]:
    bins = max(int(env.config.intent_grid_size), 1)
    region_start = env.coverage_message_base_dim - 1
    region_stop = region_start + bins * bins
    intent_valid_index = region_stop
    regions: list[set[GridPosition]] = []
    if messages.shape[-1] <= intent_valid_index:
        return regions
    for neighbor_index in range(env.num_agents):
        if neighbor_index == agent_index or not neighbor_mask[agent_index, neighbor_index]:
            continue
        if messages[neighbor_index, intent_valid_index] < 0.5:
            continue
        region_values = messages[neighbor_index, region_start:region_stop]
        if not np.any(region_values > 0.0):
            continue
        region = int(np.argmax(region_values))
        regions.append(_cells_in_region(env, region, bins))
    return regions


def _cells_in_region(env: GridCoverageEnv, region_index: int, bins: int) -> set[GridPosition]:
    region_row = region_index // bins
    region_col = region_index % bins
    cells: set[GridPosition] = set()
    for row in range(env.config.height):
        if min(row * bins // max(env.config.height, 1), bins - 1) != region_row:
            continue
        for col in range(env.config.width):
            if min(col * bins // max(env.config.width, 1), bins - 1) == region_col:
                cells.add((row, col))
    return cells
