"""Frozen extractor plus a nominal memory bank."""

from __future__ import annotations

import numpy as np
import torch

from models.embedding import embed_patches
from models.extractor import ResNet18Extractor
from models.scoring import Prediction, score_embedding


class PatchCoreDetector:
    def __init__(
        self,
        weights: str,
        layers: tuple[str, ...],
        neighborhood: int,
        memory_features: np.ndarray,
        query_chunk: int,
        bank_chunk: int,
        score_neighbors: int,
        smooth_sigma: float,
        device: torch.device,
    ) -> None:
        self.layers = layers
        self.neighborhood = neighborhood
        self.query_chunk = query_chunk
        self.bank_chunk = bank_chunk
        self.score_neighbors = score_neighbors
        self.smooth_sigma = smooth_sigma
        self.device = device
        self.extractor = ResNet18Extractor(weights).to(device)
        bank = np.ascontiguousarray(memory_features, dtype=np.float32)
        if bank.ndim != 2 or bank.shape[0] == 0:
            raise ValueError("memory features must have shape (N, D) with N > 0.")
        self.bank = torch.from_numpy(bank).to(device)

    def score_batch(self, images: torch.Tensor) -> list[Prediction]:
        images = images.to(self.device)
        with torch.inference_mode():
            embedding = embed_patches(
                self.extractor(images), self.layers, self.neighborhood
            )
            output_size = (int(images.shape[-2]), int(images.shape[-1]))
            return [
                score_embedding(
                    embedding[index],
                    self.bank,
                    self.query_chunk,
                    self.bank_chunk,
                    output_size,
                    self.score_neighbors,
                    self.smooth_sigma,
                )
                for index in range(embedding.shape[0])
            ]
