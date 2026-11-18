"""Load a saved memory bank and score images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from models.detector import PatchCoreDetector
from utils.config import ProjectConfig
from utils.device import resolve_device


def memory_dir(config: ProjectConfig, category: str) -> Path:
    return config.output.directory / "memory" / category


def load_memory(path: Path) -> tuple[np.ndarray, dict]:
    feature_file = path / "memory_bank.npz"
    meta_file = path / "metadata.json"
    if not feature_file.is_file() or not meta_file.is_file():
        raise FileNotFoundError(
            f"No memory bank in {path}. Build one with train.py after the dataset is available."
        )
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    with np.load(feature_file) as payload:
        features = np.array(payload["features"], dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Unexpected memory feature shape: {features.shape}")
    return features, metadata


def assert_compatible(config: ProjectConfig, metadata: dict) -> None:
    expected = {
        "backbone": config.model.backbone,
        "weights": config.model.weights,
        "layers": list(config.model.layers),
        "neighborhood": config.model.neighborhood,
        "image_size": config.data.image_size,
    }
    mismatches = [
        key for key, value in expected.items() if metadata.get(key) != value
    ]
    if _rounded(metadata.get("mean")) != _rounded(config.data.mean):
        mismatches.append("mean")
    if _rounded(metadata.get("std")) != _rounded(config.data.std):
        mismatches.append("std")
    if mismatches:
        joined = ", ".join(mismatches)
        raise RuntimeError(
            "The saved memory bank does not match the current config: " + joined
        )


def load_detector(config: ProjectConfig, category: str) -> tuple[PatchCoreDetector, dict]:
    features, metadata = load_memory(memory_dir(config, category))
    if metadata.get("category") != category:
        raise RuntimeError(
            f"Memory bank category is {metadata.get('category')!r}, expected {category!r}."
        )
    assert_compatible(config, metadata)
    detector = PatchCoreDetector(
        weights=str(metadata["weights"]),
        layers=tuple(metadata["layers"]),
        neighborhood=int(metadata["neighborhood"]),
        memory_features=features,
        query_chunk=config.memory.query_chunk,
        bank_chunk=config.memory.bank_chunk,
        score_neighbors=config.scoring.score_neighbors,
        smooth_sigma=config.scoring.heatmap_smooth_sigma,
        device=resolve_device(config.runtime.device),
    )
    return detector, metadata


def _rounded(values) -> tuple[float, ...] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    return tuple(round(float(value), 6) for value in values)


def tensor_to_uint8(image: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> np.ndarray:
    array = image.detach().cpu().float().clone()
    for channel, (shift, scale) in enumerate(zip(mean, std)):
        array[channel] = array[channel] * scale + shift
    array = array.clamp(0, 1).permute(1, 2, 0).numpy()
    return np.rint(array * 255).astype(np.uint8)
