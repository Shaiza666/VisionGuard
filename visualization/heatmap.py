"""Display-only anomaly overlays. Colors are not calibrated probabilities."""

from __future__ import annotations

import numpy as np
from PIL import Image


def overlay_anomaly(rgb: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.55):
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3).")
    normalized, raw_min, raw_max = _normalize(anomaly_map, rgb.shape[:2])
    strength = (alpha * normalized)[..., None]
    red = np.zeros_like(rgb, dtype=np.float32)
    red[..., 0] = 255.0
    blended = (1.0 - strength) * rgb.astype(np.float32) + strength * red
    image = Image.fromarray(np.rint(np.clip(blended, 0, 255)).astype(np.uint8))
    return image, raw_min, raw_max


def heatmap_image(anomaly_map: np.ndarray) -> Image.Image:
    """Blue-yellow-red display of one map. Colors are min-max scaled per image."""
    normalized, _, _ = _normalize(anomaly_map, anomaly_map.shape)
    stops = np.array(
        [
            [37, 52, 148],
            [65, 182, 196],
            [254, 224, 139],
            [215, 48, 39],
        ],
        dtype=np.float32,
    )
    position = normalized * (len(stops) - 1)
    left = np.floor(position).astype(np.int32)
    right = np.clip(left + 1, 0, len(stops) - 1)
    weight = (position - left)[..., None]
    color = (1.0 - weight) * stops[left] + weight * stops[right]
    return Image.fromarray(np.rint(np.clip(color, 0, 255)).astype(np.uint8))


def _normalize(anomaly_map: np.ndarray, expected_shape: tuple[int, int]):
    if anomaly_map.shape != expected_shape:
        raise ValueError("anomaly map shape does not match the image.")
    raw_min = float(np.min(anomaly_map))
    raw_max = float(np.max(anomaly_map))
    span = raw_max - raw_min
    if span < 1e-8:
        normalized = np.zeros(anomaly_map.shape, dtype=np.float32)
    else:
        normalized = ((anomaly_map - raw_min) / span).astype(np.float32)
    return normalized, raw_min, raw_max
