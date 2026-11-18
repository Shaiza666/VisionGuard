"""AUROC implementation."""

import numpy as np
import pytest

from evaluation.metrics import auroc, stratified_subsample


def test_auroc_perfect_reversed_and_tied():
    labels = np.array([0, 0, 1, 1])
    assert auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert auroc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_auroc_requires_both_classes():
    with pytest.raises(ValueError):
        auroc(np.array([1, 1]), np.array([0.2, 0.4]))


def test_pixel_subsample_keeps_both_classes_and_reports_it():
    labels = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    sampled_labels, sampled_scores, used_sample = stratified_subsample(labels, scores, 2, seed=0)
    assert used_sample is True
    assert set(sampled_labels.tolist()) == {0, 1}
    assert sampled_scores.shape == sampled_labels.shape
