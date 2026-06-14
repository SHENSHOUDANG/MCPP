from __future__ import annotations

import torch


def intent_regions_from_node_messages(node_messages: torch.Tensor, intent_grid_size: int) -> torch.Tensor:
    """Extract the existing coverage-intent region vector from node messages."""
    grid_size = max(int(intent_grid_size), 1)
    region_dim = grid_size * grid_size
    if node_messages.shape[-1] < region_dim + 1:
        raise ValueError(
            f"node message dim {node_messages.shape[-1]} is too small for intent_grid_size={grid_size}"
        )
    region_start = node_messages.shape[-1] - region_dim - 1
    region_end = region_start + region_dim
    regions = node_messages[..., region_start:region_end]
    valid = node_messages[..., -1:].clamp(0.0, 1.0)
    return regions.clamp_min(0.0) * valid


def pairwise_soft_iou(intent_regions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute pairwise Soft-IoU for aligned intent vectors with shape ``[..., N, L]``."""
    if intent_regions.ndim < 2:
        raise ValueError(f"expected intent regions with at least 2 dims, got {intent_regions.ndim}")
    regions = intent_regions.clamp_min(0.0)
    inter = torch.matmul(regions, regions.transpose(-2, -1))
    area = regions.sum(dim=-1)
    union = area.unsqueeze(-1) + area.unsqueeze(-2) - inter
    overlap = torch.where(union > eps, inter / union.clamp_min(eps), torch.zeros_like(union))
    return overlap.clamp(0.0, 1.0)


def intent_overlap_from_node_messages(
    node_messages: torch.Tensor,
    neighbor_mask: torch.Tensor | None,
    intent_grid_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return ``rho`` with shape ``[..., N, N]`` from the existing coverage messages."""
    regions = intent_regions_from_node_messages(node_messages, intent_grid_size)
    overlap = pairwise_soft_iou(regions, eps=eps)
    num_agents = overlap.shape[-1]
    eye = torch.eye(num_agents, dtype=torch.bool, device=overlap.device)
    while eye.ndim < overlap.ndim:
        eye = eye.unsqueeze(0)
    overlap = overlap.masked_fill(eye, 0.0)
    if neighbor_mask is not None:
        mask = neighbor_mask.to(device=overlap.device, dtype=torch.bool)
        while mask.ndim < overlap.ndim:
            mask = mask.unsqueeze(0)
        if mask.shape != overlap.shape:
            mask = torch.broadcast_to(mask, overlap.shape)
        overlap = overlap.masked_fill(~mask, 0.0)
    return overlap
