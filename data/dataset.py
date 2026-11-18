"""One-image-at-a-time MVTec AD dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.layout import Sample, test_samples, train_samples


class MVTecDataset(Dataset):
    """Load a single resized image per index.

    Training items are image tensors. Test items are dictionaries with the
    image, a binary label, a binary mask, the file path, and the defect type.
    """

    def __init__(
        self,
        root: Path,
        category: str,
        split: str,
        transform,
        image_size: int,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'.")
        self.split = split
        self.transform = transform
        self.image_size = image_size
        if split == "train":
            self.samples = train_samples(root, category)
        else:
            self.samples = test_samples(root, category)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        if self.split == "train":
            return tensor
        return {
            "image": tensor,
            "label": sample.label,
            "mask": _load_mask(sample, self.image_size),
            "path": str(sample.path),
            "defect_type": sample.defect_type,
        }


def collate_test_batch(items: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in items]),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.int64),
        "mask": torch.stack([item["mask"] for item in items]),
        "path": [item["path"] for item in items],
        "defect_type": [item["defect_type"] for item in items],
    }


def _load_mask(sample: Sample, image_size: int) -> torch.Tensor:
    if sample.label == 0:
        mask = np.zeros((image_size, image_size), dtype=np.float32)
        return torch.from_numpy(mask).unsqueeze(0)
    if sample.mask_path is None or not sample.mask_path.is_file():
        raise FileNotFoundError(
            "Missing MVTec ground-truth mask for "
            f"{sample.path}. Expected {sample.mask_path}."
        )
    with Image.open(sample.mask_path) as mask_image:
        resized = mask_image.convert("L").resize(
            (image_size, image_size), Image.Resampling.NEAREST
        )
        values = np.asarray(resized, dtype=np.float32)
    binary = (values > 0).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0)
