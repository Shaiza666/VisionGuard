"""End-to-end memory build on two synthetic images and random weights."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inference.predict import load_detector
from training.memory_bank import build_category_memory
from utils.config import (
    DataConfig,
    EvaluationConfig,
    MemoryConfig,
    ModelConfig,
    OutputConfig,
    ProjectConfig,
    RuntimeConfig,
    ScoringConfig,
)


def _png(path: Path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), color).save(path)


def test_memory_bank_builds_and_scores_on_cpu(tmp_path):
    root = tmp_path / "mvtec"
    _png(root / "bottle" / "train" / "good" / "000.png", (20, 40, 60))
    _png(root / "bottle" / "train" / "good" / "001.png", (25, 45, 65))
    _png(root / "bottle" / "test" / "good" / "000.png", (20, 40, 60))
    _png(root / "bottle" / "test" / "broken" / "000.png", (200, 20, 20))
    mask = Image.new("L", (128, 128), 0)
    mask_path = root / "bottle" / "ground_truth" / "broken" / "000_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path)

    config = ProjectConfig(
        seed=1,
        runtime=RuntimeConfig(device="cpu", num_workers=0, batch_size=1),
        data=DataConfig(
            root=root,
            categories=("bottle",),
            image_size=128,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
        model=ModelConfig(
            backbone="resnet18",
            weights="none",
            layers=("layer2", "layer3"),
            neighborhood=3,
        ),
        memory=MemoryConfig(
            coreset_ratio=0.5,
            max_bank_size=32,
            preselect_max=64,
            patches_per_image=8,
            query_chunk=8,
            bank_chunk=8,
        ),
        scoring=ScoringConfig(score_neighbors=1, heatmap_smooth_sigma=0),
        evaluation=EvaluationConfig(max_pixels=1000, save_examples=0),
        output=OutputConfig(directory=tmp_path / "reports"),
    )
    destination, metadata = build_category_memory(config, "bottle")
    assert (destination / "memory_bank.npz").is_file()
    assert metadata["metrics"] is None
    assert metadata["weights"] == "none"
    assert metadata["train_images"] == 2
    assert 1 <= metadata["patches_after_coreset"] <= 32

    detector, _ = load_detector(config, "bottle")
    image = torch.zeros(1, 3, 128, 128)
    prediction = detector.score_batch(image)[0]
    assert prediction.anomaly_map.shape == (128, 128)
    assert np.isfinite(prediction.image_score)
