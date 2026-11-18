"""Scan a local MVTec AD category without opening image pixels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_CATEGORIES = ("bottle", "hazelnut")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
OFFICIAL_MVTEC_URL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"


class DatasetLayoutError(FileNotFoundError):
    """The local MVTec AD folder is missing or incomplete."""


@dataclass(frozen=True)
class Sample:
    path: Path
    defect_type: str
    label: int
    mask_path: Path | None


@dataclass(frozen=True)
class CategorySummary:
    category: str
    train_good: int
    test_good: int
    test_anomaly: int
    defect_types: dict[str, int]
    missing_masks: int

    def render(self) -> str:
        defects = ", ".join(
            f"{name}={count}" for name, count in sorted(self.defect_types.items())
        )
        return (
            f"{self.category}: train_good={self.train_good}, "
            f"test_good={self.test_good}, test_anomaly={self.test_anomaly}, "
            f"missing_masks={self.missing_masks}, defects=[{defects}]"
        )


def summarize_category(root: Path, category: str) -> CategorySummary:
    samples = scan_category(root, category)
    train = [sample for sample in samples if sample.defect_type == "train/good"]
    test_good = [
        sample
        for sample in samples
        if sample.label == 0 and sample.defect_type != "train/good"
    ]
    test_anomaly = [sample for sample in samples if sample.label == 1]
    counts: dict[str, int] = {}
    missing = 0
    for sample in test_anomaly:
        counts[sample.defect_type] = counts.get(sample.defect_type, 0) + 1
        if sample.mask_path is None or not sample.mask_path.is_file():
            missing += 1
    return CategorySummary(
        category=category,
        train_good=len(train),
        test_good=len(test_good),
        test_anomaly=len(test_anomaly),
        defect_types=counts,
        missing_masks=missing,
    )


def scan_category(root: Path, category: str) -> list[Sample]:
    validate_category(category)
    category_dir = root / category
    if not category_dir.is_dir():
        raise DatasetLayoutError(_missing_message(root, category))

    train_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise DatasetLayoutError(
            f"{category_dir} is not an MVTec AD category folder. "
            "Expected train/good and test subfolders. "
            "See docs/dataset_instructions.md."
        )

    samples = [
        Sample(path=path, defect_type="train/good", label=0, mask_path=None)
        for path in _images(train_dir)
    ]
    if not samples:
        raise DatasetLayoutError(f"No defect-free training images in {train_dir}.")

    for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        label = 0 if defect_dir.name == "good" else 1
        for image_path in _images(defect_dir):
            mask_path = None
            if label == 1:
                mask_path = (
                    category_dir
                    / "ground_truth"
                    / defect_dir.name
                    / f"{image_path.stem}_mask.png"
                )
            samples.append(
                Sample(
                    path=image_path,
                    defect_type=defect_dir.name,
                    label=label,
                    mask_path=mask_path,
                )
            )
    test_samples = [sample for sample in samples if sample.defect_type != "train/good"]
    if not any(sample.label == 0 for sample in test_samples):
        raise DatasetLayoutError(f"No good test images in {test_dir / 'good'}.")
    if not any(sample.label == 1 for sample in test_samples):
        raise DatasetLayoutError(f"No anomalous test images in {test_dir}.")
    return samples


def train_samples(root: Path, category: str) -> list[Sample]:
    return [
        sample
        for sample in scan_category(root, category)
        if sample.defect_type == "train/good"
    ]


def test_samples(root: Path, category: str) -> list[Sample]:
    return [
        sample
        for sample in scan_category(root, category)
        if sample.defect_type != "train/good"
    ]


def validate_category(category: str) -> None:
    if category not in SUPPORTED_CATEGORIES:
        allowed = ", ".join(SUPPORTED_CATEGORIES)
        raise ValueError(
            f"Category {category!r} is not enabled. Phase 1 supports: {allowed}."
        )


def _images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _missing_message(root: Path, category: str) -> str:
    return (
        f"Could not find MVTec AD category folder: {root / category}\n"
        "VisionGuard does not download datasets.\n"
        f"Download MVTec AD from {OFFICIAL_MVTEC_URL}\n"
        "and follow docs/dataset_instructions.md. "
        "The dataset license is CC BY-NC-SA 4.0 and commercial use is not allowed."
    )
