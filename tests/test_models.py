"""Embedding shapes and nearest-neighbour scores."""

import numpy as np
import pytest
import torch

from models.embedding import embed_patches
from models.extractor import ResNet18Extractor
from models.scoring import min_l2, reweighted_image_score, score_embedding
from utils.device import resolve_device
from visualization.heatmap import overlay_anomaly


def test_layer_maps_align_to_the_finer_grid():
    features = {
        "layer2": torch.zeros(1, 128, 16, 16),
        "layer3": torch.zeros(1, 256, 8, 8),
    }
    embedding = embed_patches(features, ("layer2", "layer3"), neighborhood=3)
    assert embedding.shape == (1, 384, 16, 16)


def test_resnet18_cpu_forward_is_frozen():
    extractor = ResNet18Extractor(weights="none")
    outputs = extractor(torch.zeros(1, 3, 128, 128))
    assert outputs["layer2"].shape == (1, 128, 16, 16)
    assert outputs["layer3"].shape == (1, 256, 8, 8)
    assert all(not parameter.requires_grad for parameter in extractor.parameters())


def test_min_distance_is_zero_for_a_bank_row():
    bank = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    queries = torch.tensor([[3.0, 0.0], [1.0, 0.0]])
    distances = min_l2(queries, bank, query_chunk=1, bank_chunk=2)
    assert distances[0].item() == pytest_approx(0.0)
    assert distances[1].item() == pytest_approx(1.0)


def test_image_score_matches_max_distance_without_reweighting():
    embedding = torch.zeros(2, 2, 2)
    bank = torch.tensor([[0.0, 0.0], [5.0, 0.0]])
    embedding[:, 1, 1] = torch.tensor([10.0, 0.0])
    prediction = score_embedding(
        embedding,
        bank,
        query_chunk=2,
        bank_chunk=1,
        output_size=(8, 8),
        score_neighbors=1,
        smooth_sigma=0,
    )
    assert prediction.image_score == pytest_approx(5.0)
    assert prediction.patch_scores[1, 1] == pytest_approx(5.0)
    assert prediction.anomaly_map.shape == (8, 8)
    assert np.isfinite(prediction.anomaly_map).all()


def test_reweighting_does_not_increase_the_nearest_distance():
    distances = torch.tensor([0.2, 1.5])
    winner = torch.tensor([1.5, 1.5, 8.0])
    score = reweighted_image_score(distances, winner, neighbors=3)
    assert 0 <= score <= 1.5


def test_cpu_device_guard():
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError, match="CPU-only"):
        resolve_device("cuda")


def test_overlay_preserves_image_size():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    anomaly = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    image, raw_min, raw_max = overlay_anomaly(rgb, anomaly)
    assert image.size == (8, 8)
    assert raw_min == pytest_approx(0.0)
    assert raw_max == pytest_approx(1.0)


def pytest_approx(value):
    return pytest.approx(value)
