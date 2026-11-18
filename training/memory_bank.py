"""Build one category memory bank from defect-free images."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import MVTecDataset
from data.transforms import build_image_transform
from models.embedding import embed_patches, sample_patch_rows
from models.extractor import ResNet18Extractor
from models.memory import PatchReservoir, coreset_target_count, greedy_coreset
from training.log import RunLog
from utils.config import ProjectConfig, config_to_dict
from utils.device import resolve_device


def build_category_memory(config: ProjectConfig, category: str, progress=None) -> tuple[object, dict]:
    device = resolve_device(config.runtime.device)
    transform = build_image_transform(
        config.data.image_size, config.data.mean, config.data.std
    )
    dataset = MVTecDataset(
        root=config.data.root,
        category=category,
        split="train",
        transform=transform,
        image_size=config.data.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        pin_memory=False,
    )
    extractor = ResNet18Extractor(config.model.weights).to(device).eval()
    reservoir = PatchReservoir(config.memory.preselect_max, config.seed)
    rng = np.random.default_rng(config.seed)
    patch_grid: tuple[int, int] | None = None
    seen_images = 0

    log = RunLog(config.output.directory / "logs", category, "memory")
    snapshot = config.output.directory / "memory" / category / "config.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(config_to_dict(config), indent=2), encoding="utf-8")

    def report(seen: int, total: int, message: str) -> None:
        print(message, flush=True)
        log.write(message)
        if progress is not None:
            progress(seen, total, message)

    report(
        0,
        len(dataset),
        (
            f"[{category}] CPU memory-bank build: {len(dataset)} nominal images, "
            f"batch {config.runtime.batch_size}, workers {config.runtime.num_workers}, "
            f"image {config.data.image_size}px, weights {config.model.weights}."
        ),
    )
    if config.model.weights == "none":
        report(
            0,
            len(dataset),
            f"[{category}] Random ResNet-18 weights are a code check only. They are not a detection model.",
        )

    for batch in loader:
        images = batch.to(device)
        with torch.inference_mode():
            embedding = embed_patches(
                extractor(images), config.model.layers, config.model.neighborhood
            )
        grid = (int(embedding.shape[-2]), int(embedding.shape[-1]))
        if patch_grid is None:
            patch_grid = grid
        elif patch_grid != grid:
            raise RuntimeError(f"Patch grid changed from {patch_grid} to {grid}.")
        for index in range(embedding.shape[0]):
            reservoir.add(
                sample_patch_rows(embedding[index], config.memory.patches_per_image, rng)
            )
        seen_images += int(images.shape[0])
        if seen_images % 10 == 0 or seen_images == len(dataset):
            report(
                seen_images,
                len(dataset),
                f"[{category}] embedded {seen_images}/{len(dataset)} images",
            )

    if patch_grid is None:
        raise RuntimeError("No images were embedded.")
    pooled = reservoir.values()
    target = coreset_target_count(
        pooled.shape[0], config.memory.coreset_ratio, config.memory.max_bank_size
    )
    report(
        seen_images,
        len(dataset),
        (
            f"[{category}] patches seen={reservoir.seen}, "
            f"retained={pooled.shape[0]}, coreset target={target}."
        ),
    )
    bank = greedy_coreset(pooled, target, config.seed)
    metadata = _metadata(
        config,
        category,
        extractor.weights_name,
        patch_grid,
        seen_images,
        reservoir.seen,
        pooled.shape[0],
        bank,
    )
    destination = config.output.directory / "memory" / category
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination / "memory_bank.npz", features=bank)
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[{category}] saved {bank.shape[0]} patches to {destination}", flush=True)
    log.write(f"[{category}] saved {bank.shape[0]} patches to {destination}")
    log.write(f"config snapshot {snapshot}")
    return destination, metadata


def _metadata(
    config,
    category,
    weights_name,
    patch_grid,
    train_images,
    patches_seen,
    retained,
    bank,
) -> dict:
    import torch
    import torchvision

    return {
        "project": "VisionGuard",
        "method": "patchcore-inspired",
        "category": category,
        "backbone": config.model.backbone,
        "weights": config.model.weights,
        "weights_name": weights_name,
        "layers": list(config.model.layers),
        "neighborhood": config.model.neighborhood,
        "image_size": config.data.image_size,
        "mean": list(config.data.mean),
        "std": list(config.data.std),
        "patch_grid": list(patch_grid),
        "embedding_dim": int(bank.shape[1]),
        "patches_seen": int(patches_seen),
        "patches_before_coreset": int(retained),
        "patches_after_coreset": int(bank.shape[0]),
        "coreset_ratio": config.memory.coreset_ratio,
        "max_bank_size": config.memory.max_bank_size,
        "preselect_max": config.memory.preselect_max,
        "patches_per_image": config.memory.patches_per_image,
        "train_images": int(train_images),
        "seed": config.seed,
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "metrics": None,
        "note": "This file records a memory bank. It does not contain detection metrics.",
    }
