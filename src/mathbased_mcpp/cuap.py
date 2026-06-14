from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CUAPConfig, GridPosition
from .env import ACTIONS, GridCoverageEnv


CUAP_CONFIDENCE_DIM = 2


@dataclass(frozen=True, slots=True)
class CUAPStepInputs:
    prior: np.ndarray
    confidence: np.ndarray
    phase_mask: np.ndarray
    raw_scores: np.ndarray


@dataclass(frozen=True, slots=True)
class _MemoryView:
    known_free: set[GridPosition]
    known_team_covered: set[GridPosition]
    known_obstacles: set[GridPosition]
    frontier: set[GridPosition]


def compute_cuap_raw_scores(env: GridCoverageEnv, cfg: CUAPConfig, phase: str = "coverage") -> np.ndarray:
    """Compute unnormalized CUAP utility scores with shape ``[agent, action]``."""
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
    return scores.astype(np.float32)


def compute_cuap_logits(env: GridCoverageEnv, cfg: CUAPConfig, phase: str = "coverage") -> np.ndarray:
    """Compute fixed-CUAP logits or gated-CUAP bounded priors for compatibility."""
    raw_scores = compute_cuap_raw_scores(env, cfg, phase=phase)
    if not cfg.enabled:
        return raw_scores
    if cfg.gated:
        return build_bounded_prior(raw_scores, env.action_masks(), tau=cfg.tau)

    scores = raw_scores
    if cfg.normalize:
        scores = _z_score_legal_actions(scores, env.action_masks())
    if cfg.clip > 0:
        scores = np.clip(scores, -cfg.clip, cfg.clip)
    return scores.astype(np.float32)


def build_bounded_prior(raw_scores: np.ndarray, action_masks: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Mean-center legal CUAP scores and bound them without per-state z-score amplification."""
    scores = np.asarray(raw_scores, dtype=np.float32)
    masks = np.asarray(action_masks, dtype=bool)
    prior = np.zeros_like(scores, dtype=np.float32)
    if scores.shape != masks.shape:
        raise ValueError(f"raw score shape {scores.shape} does not match action mask shape {masks.shape}")
    scale = max(float(tau), 1e-6)
    for agent_index in range(scores.shape[0]):
        valid = masks[agent_index]
        if int(valid.sum()) <= 1:
            continue
        valid_scores = scores[agent_index, valid]
        centered = (scores[agent_index] - float(valid_scores.mean())) / scale
        prior[agent_index] = np.where(valid, np.tanh(centered), 0.0)
    return prior.astype(np.float32)


def build_cuap_confidence(
    raw_scores: np.ndarray,
    action_masks: np.ndarray,
    confidence_tau: float = 1.0,
) -> np.ndarray:
    """Build the two CUAP confidence scalars: legal top1-top2 margin and legal max-min spread."""
    scores = np.asarray(raw_scores, dtype=np.float32)
    masks = np.asarray(action_masks, dtype=bool)
    confidence = np.zeros((scores.shape[0], CUAP_CONFIDENCE_DIM), dtype=np.float32)
    if scores.shape != masks.shape:
        raise ValueError(f"raw score shape {scores.shape} does not match action mask shape {masks.shape}")
    scale = max(float(confidence_tau), 1e-6)
    for agent_index in range(scores.shape[0]):
        valid_scores = scores[agent_index, masks[agent_index]]
        if valid_scores.size <= 1:
            continue
        sorted_scores = np.sort(valid_scores)
        margin = max(float(sorted_scores[-1] - sorted_scores[-2]), 0.0)
        spread = max(float(sorted_scores[-1] - sorted_scores[0]), 0.0)
        confidence[agent_index] = np.tanh(np.asarray([margin, spread], dtype=np.float32) / scale)
    return confidence.astype(np.float32)


def build_cuap_step_inputs(env: GridCoverageEnv, cfg: CUAPConfig, phase: str = "coverage") -> CUAPStepInputs:
    raw_scores = compute_cuap_raw_scores(env, cfg, phase=phase)
    action_masks = env.action_masks()
    prior = build_bounded_prior(raw_scores, action_masks, tau=cfg.tau) if cfg.enabled else np.zeros_like(raw_scores)
    confidence = (
        build_cuap_confidence(raw_scores, action_masks, confidence_tau=cfg.confidence_tau)
        if cfg.enabled
        else np.zeros((env.num_agents, CUAP_CONFIDENCE_DIM), dtype=np.float32)
    )
    valid_action_counts = action_masks.sum(axis=-1, keepdims=True).astype(np.float32)
    coverage_phase = 1.0
    if cfg.disable_in_return_phase and phase == "return":
        coverage_phase = 0.0
    phase_mask = (valid_action_counts > 1).astype(np.float32) * coverage_phase
    if not cfg.enabled:
        phase_mask = np.zeros_like(phase_mask, dtype=np.float32)
    return CUAPStepInputs(
        prior=prior.astype(np.float32),
        confidence=confidence.astype(np.float32),
        phase_mask=phase_mask.astype(np.float32),
        raw_scores=raw_scores.astype(np.float32),
    )


def scaled_cuap_prior(env: GridCoverageEnv, cfg: CUAPConfig, phase: str = "coverage") -> np.ndarray | None:
    if not cfg.enabled or cfg.gated:
        return None
    return (float(cfg.beta) * compute_cuap_logits(env, cfg, phase=phase)).astype(np.float32)


def _z_score_legal_actions(scores: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(scores, dtype=np.float32)
    masks = np.asarray(action_masks, dtype=bool)
    for agent_index in range(scores.shape[0]):
        valid = masks[agent_index]
        if not np.any(valid):
            continue
        valid_scores = scores[agent_index, valid]
        mean = float(valid_scores.mean())
        std = float(valid_scores.std())
        normalized[agent_index] = np.where(valid, (scores[agent_index] - mean) / (std + 1e-6), 0.0)
    return normalized.astype(np.float32)


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
