"""Local patch embeddings from intermediate ResNet maps."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def embed_patches(
    feature_maps: dict[str, torch.Tensor],
    layers: tuple[str, ...],
    neighborhood: int,
) -> torch.Tensor:
    """Average each local neighbourhood and concatenate selected layers.

    Deeper maps are bilinearly aligned to the finest selected resolution.
    The result has shape (batch, channels, height, width).
    """
    if neighborhood < 1 or neighborhood % 2 == 0:
        raise ValueError("neighborhood must be a positive odd integer.")
    pooled = []
    for name in layers:
        if name not in feature_maps:
            raise KeyError(f"Feature map {name!r} was not extracted.")
        pooled.append(_local_average(feature_maps[name], neighborhood))
    height = max(tensor.shape[-2] for tensor in pooled)
    width = max(tensor.shape[-1] for tensor in pooled)
    aligned = []
    for tensor in pooled:
        if tensor.shape[-2:] != (height, width):
            tensor = F.interpolate(
                tensor, size=(height, width), mode="bilinear", align_corners=False
            )
        aligned.append(tensor)
    return torch.cat(aligned, dim=1)


def sample_patch_rows(embedding: torch.Tensor, count: int, rng) -> "np.ndarray":
    """Return up to ``count`` patch vectors from one embedding map."""
    import numpy as np

    if embedding.ndim != 3:
        raise ValueError("embedding must have shape (C, H, W).")
    channels, height, width = embedding.shape
    flat = embedding.detach().permute(1, 2, 0).reshape(-1, channels).contiguous()
    flat_np = flat.cpu().numpy().astype(np.float32, copy=False)
    if flat_np.shape[0] <= count:
        return np.ascontiguousarray(flat_np)
    indices = rng.choice(flat_np.shape[0], size=count, replace=False)
    return np.ascontiguousarray(flat_np[indices])


def _local_average(features: torch.Tensor, neighborhood: int) -> torch.Tensor:
    padding = neighborhood // 2
    return F.avg_pool2d(features, kernel_size=neighborhood, stride=1, padding=padding)
