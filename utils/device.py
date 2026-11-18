"""CPU device guard."""

from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    if name != "cpu":
        raise RuntimeError(
            "VisionGuard is CPU-only. "
            f"Refusing device {name!r}. Set runtime.device to cpu."
        )
    return torch.device("cpu")
