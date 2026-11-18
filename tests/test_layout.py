"""MVTec folder scanning without reading pixels into a model."""

from pathlib import Path

import pytest
from PIL import Image

from data.layout import DatasetLayoutError, scan_category, summarize_category


def _png(path: Path, color, size=16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)


def _category(root: Path) -> None:
    _png(root / "bottle" / "train" / "good" / "000.png", (10, 20, 30))
    _png(root / "bottle" / "train" / "good" / "001.png", (20, 30, 40))
    _png(root / "bottle" / "test" / "good" / "000.png", (30, 40, 50))
    _png(root / "bottle" / "test" / "broken_large" / "000.png", (40, 50, 60))
    mask_path = root / "bottle" / "ground_truth" / "broken_large" / "000_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), 0).save(mask_path)


def test_scan_counts_nominal_and_anomalous_images(tmp_path):
    _category(tmp_path)
    summary = summarize_category(tmp_path, "bottle")
    assert summary.train_good == 2
    assert summary.test_good == 1
    assert summary.test_anomaly == 1
    assert summary.missing_masks == 0
    assert summary.defect_types == {"broken_large": 1}


def test_missing_category_explains_manual_download(tmp_path):
    with pytest.raises(DatasetLayoutError, match="does not download"):
        scan_category(tmp_path, "hazelnut")


def test_phase1_rejects_other_mvtec_categories(tmp_path):
    with pytest.raises(ValueError, match="Phase 1"):
        scan_category(tmp_path, "cable")
