"""Dataset items load one image and one mask."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.dataset import MVTecDataset


def _write_tree(root: Path) -> None:
    def png(path, color):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), color).save(path)

    png(root / "hazelnut" / "train" / "good" / "000.png", (1, 2, 3))
    png(root / "hazelnut" / "test" / "good" / "000.png", (4, 5, 6))
    png(root / "hazelnut" / "test" / "crack" / "000.png", (7, 8, 9))
    mask = Image.new("L", (20, 20), 0)
    mask.putpixel((5, 5), 255)
    mask_path = root / "hazelnut" / "ground_truth" / "crack" / "000_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path)


def _transform(image):
    array = np.asarray(image.resize((32, 32)), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def test_test_split_returns_label_and_mask(tmp_path):
    _write_tree(tmp_path)
    dataset = MVTecDataset(tmp_path, "hazelnut", "test", _transform, image_size=32)
    items = [dataset[index] for index in range(len(dataset))]
    good = next(item for item in items if item["label"] == 0)
    anomaly = next(item for item in items if item["label"] == 1)
    assert torch.count_nonzero(good["mask"]) == 0
    assert anomaly["defect_type"] == "crack"
    assert anomaly["mask"].shape == (1, 32, 32)
    assert torch.count_nonzero(anomaly["mask"]) > 0
    assert good["image"].shape == (3, 32, 32)
