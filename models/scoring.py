"""Nearest-neighbour anomaly scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Prediction:
    image_score: float
    anomaly_map: np.ndarray
    patch_scores: np.ndarray


def min_l2(
    queries: torch.Tensor,
    bank: torch.Tensor,
    query_chunk: int,
    bank_chunk: int,
) -> torch.Tensor:
    """Minimum L2 distance from each query row to the memory bank."""
    if queries.ndim != 2 or bank.ndim != 2:
        raise ValueError("queries and bank must be 2D.")
    if queries.shape[1] != bank.shape[1]:
        raise ValueError("query and bank feature dimensions differ.")
    if bank.shape[0] == 0:
        raise ValueError("memory bank is empty.")
    distances = torch.empty(queries.shape[0], dtype=torch.float32, device=queries.device)
    for start in range(0, queries.shape[0], query_chunk):
        query = queries[start : start + query_chunk]
        best = None
        for bank_start in range(0, bank.shape[0], bank_chunk):
            block = bank[bank_start : bank_start + bank_chunk]
            local = torch.cdist(query, block, p=2).min(dim=1).values
            best = local if best is None else torch.minimum(best, local)
        distances[start : start + query.shape[0]] = best
    return distances


def reweighted_image_score(
    min_distances: torch.Tensor,
    winner_distances: torch.Tensor,
    neighbors: int,
) -> float:
    """Image score from the worst patch.

    With ``neighbors == 1`` this is the maximum nearest-neighbour distance.
    With a larger neighbour count it applies the PatchCore reweighting:
    the worst-patch distance is reduced when that patch's nearby memory
    entries are similarly close.
    """
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1.")
    if min_distances.numel() == 0:
        raise ValueError("min_distances is empty.")
    star = float(min_distances.max().item())
    if neighbors == 1:
        return star
    count = min(neighbors, int(winner_distances.numel()))
    nearest = torch.topk(winner_distances, k=count, largest=False).values
    shifted = nearest - nearest.max()
    exponential = torch.exp(shifted)
    fraction = exponential[0] / exponential.sum()
    return float((1.0 - fraction) * nearest[0])


def score_embedding(
    embedding: torch.Tensor,
    bank: torch.Tensor,
    query_chunk: int,
    bank_chunk: int,
    output_size: tuple[int, int],
    score_neighbors: int,
    smooth_sigma: float,
) -> Prediction:
    """Score one (C, H, W) embedding against a (N, C) memory bank."""
    if embedding.ndim != 3:
        raise ValueError("embedding must have shape (C, H, W).")
    channels, height, width = embedding.shape
    queries = embedding.permute(1, 2, 0).reshape(-1, channels).contiguous()
    patch_distances = min_l2(queries, bank, query_chunk, bank_chunk)
    winner = int(torch.argmax(patch_distances).item())
    if score_neighbors == 1:
        image_score = float(patch_distances[winner].item())
    else:
        winner_distances = _distances_to_bank(queries[winner], bank, bank_chunk)
        image_score = reweighted_image_score(
            patch_distances, winner_distances, score_neighbors
        )
    patch_scores = patch_distances.reshape(height, width)
    anomaly_map = F.interpolate(
        patch_scores[None, None],
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    if smooth_sigma > 0:
        anomaly_map = _gaussian_smooth(anomaly_map, smooth_sigma)
    return Prediction(
        image_score=image_score,
        anomaly_map=anomaly_map.detach().cpu().numpy().astype(np.float32),
        patch_scores=patch_scores.detach().cpu().numpy().astype(np.float32),
    )


def _distances_to_bank(query: torch.Tensor, bank: torch.Tensor, bank_chunk: int) -> torch.Tensor:
    parts = []
    for start in range(0, bank.shape[0], bank_chunk):
        block = bank[start : start + bank_chunk]
        parts.append(torch.linalg.vector_norm(block - query, ord=2, dim=1))
    return torch.cat(parts)


def _gaussian_smooth(values: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=values.device)
    kernel = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    image = values[None, None]
    image = F.pad(image, (radius, radius, radius, radius), mode="reflect")
    image = F.conv2d(image, horizontal)
    image = F.conv2d(image, vertical)
    return image[0, 0]
