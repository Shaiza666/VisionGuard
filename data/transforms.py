"""Image transforms used by training and inference."""

from __future__ import annotations


def build_image_transform(image_size: int, mean: tuple[float, ...], std: tuple[float, ...]):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )
