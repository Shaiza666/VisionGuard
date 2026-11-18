"""Thresholded precision, recall, F1, and confusion counts."""

import numpy as np
import pytest

from evaluation.metrics import confusion_counts, decision_summary, precision_recall_f1
from evaluation.report import inspection_report, report_markdown
from visualization.heatmap import heatmap_image


def test_perfect_separation_has_unit_f1_and_a_clean_matrix():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    summary = decision_summary(labels, scores)
    assert summary["auroc"] == pytest.approx(1.0)
    assert summary["precision"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(1.0)
    assert summary["f1"] == pytest.approx(1.0)
    assert summary["threshold_source"] == "f1_max_on_this_split"
    assert summary["confusion_matrix"]["matrix"] == [[2, 0], [0, 2]]


def test_configured_threshold_is_labeled():
    labels = np.array([0, 1])
    scores = np.array([0.2, 0.9])
    summary = decision_summary(labels, scores, threshold=0.5)
    assert summary["threshold"] == pytest.approx(0.5)
    assert summary["threshold_source"] == "configured_threshold"
    assert summary["confusion_matrix"]["counts"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}


def test_metrics_explain_a_missing_class():
    summary = decision_summary(np.array([0, 0]), np.array([0.1, 0.2]))
    assert summary["auroc"] is None
    assert summary["f1"] is None
    assert summary["confusion_matrix"] is None
    assert "both normal and anomalous" in summary["unavailable_reason"]


def test_precision_is_undefined_when_nothing_is_flagged():
    rates = precision_recall_f1(confusion_counts([0, 1], [0, 0]))
    assert rates["precision"] is None
    assert rates["recall"] == pytest.approx(0.0)
    assert rates["f1"] is None


def test_heatmap_is_an_image_not_a_metric():
    anomaly = np.linspace(0, 3, 16, dtype=np.float32).reshape(4, 4)
    image = heatmap_image(anomaly)
    assert image.size == (4, 4)
    assert image.mode == "RGB"


def test_inspection_report_refuses_to_invent_metrics():
    report = inspection_report(
        "bottle",
        {"backbone": "resnet18", "weights": "imagenet", "patches_after_coreset": 10},
        [{"file": "a.png", "image_score": 0.2, "map_min": 0.0, "map_max": 0.2}],
    )
    assert report["metrics"] is None
    assert "cannot be calculated" in report["metrics_unavailable_reason"]
    markdown = report_markdown(report)
    assert "0.200000" in markdown
    assert "AUROC" not in markdown or "cannot" in markdown.lower()
