"""Ranking metrics. No published scores are stored here."""

from __future__ import annotations

import numpy as np


def auroc(y_true, y_score) -> float:
    """Area under the ROC curve by the Mann-Whitney statistic, with tie ranks."""
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=np.float64)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same length.")
    if labels.size == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 or 1.")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite.")
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive == 0 or negative == 0:
        raise ValueError("AUROC requires at least one positive and one negative label.")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.shape[0], dtype=np.float64)
    start = 0
    while start < labels.shape[0]:
        end = start + 1
        while end < labels.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def confusion_counts(labels, predictions) -> dict[str, int]:
    """Count true and false decisions. Predictions use 1 for anomaly."""
    labels = np.asarray(labels).astype(np.int64)
    predictions = np.asarray(predictions).astype(np.int64)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape.")
    if labels.size == 0 or not np.isin(labels, (0, 1)).all() or not np.isin(predictions, (0, 1)).all():
        raise ValueError("labels and predictions must be 0 or 1.")
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    return {
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def precision_recall_f1(counts: dict[str, int]) -> dict[str, float | None]:
    """Precision is undefined when nothing is predicted anomalous."""
    predicted_positive = counts["tp"] + counts["fp"]
    actual_positive = counts["tp"] + counts["fn"]
    precision = None if predicted_positive == 0 else counts["tp"] / predicted_positive
    recall = None if actual_positive == 0 else counts["tp"] / actual_positive
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def best_f1_threshold(labels, scores) -> float:
    """Lowest score at which predicting score >= threshold maximizes F1.

    The search uses the same labels it reports, so the threshold is optimistic
    and is not a held-out operating point.
    """
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isin(labels, (0, 1)).all() or int(np.sum(labels == 1)) == 0:
        raise ValueError("F1 threshold search needs anomalous labels.")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    positive_total = int(np.sum(labels == 1))
    true_positive = 0
    false_positive = 0
    best_f1 = -1.0
    best_threshold = float(scores[0])
    index = 0
    while index < labels.shape[0]:
        end = index + 1
        while end < labels.shape[0] and scores[end] == scores[index]:
            end += 1
        true_positive += int(np.sum(labels[index:end] == 1))
        false_positive += int(np.sum(labels[index:end] == 0))
        rates = precision_recall_f1(
            {"tp": true_positive, "fp": false_positive, "tn": 0, "fn": positive_total - true_positive}
        )
        f1 = rates["f1"] if rates["f1"] is not None else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(scores[index])
        index = end
    return best_threshold


def decision_summary(labels, scores, threshold: float | None = None) -> dict:
    """AUROC plus thresholded precision, recall, F1, and a confusion matrix.

    AUROC does not use a threshold. Precision, recall, F1, and the confusion
    matrix do. When ``threshold`` is omitted, the threshold is the one that
    maximizes F1 on these same labels. That selection is reported in
    ``threshold_source`` and must not be treated as a deployment threshold.
    """
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(np.sum(labels == 1)) if labels.size else 0
    negative = int(np.sum(labels == 0)) if labels.size else 0
    if positive == 0 or negative == 0:
        return {
            "auroc": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "threshold": None,
            "threshold_source": None,
            "confusion_matrix": None,
            "unavailable_reason": (
                "Precision, recall, F1, AUROC, and the confusion matrix need "
                "both normal and anomalous ground-truth labels. "
                f"This split has {positive} anomalous and {negative} normal labels."
            ),
        }
    if threshold is None:
        chosen = best_f1_threshold(labels, scores)
        source = "f1_max_on_this_split"
        note = (
            "The threshold maximizes F1 on the same labels used for the report. "
            "It is an optimistic summary, not a threshold chosen on held-out data."
        )
    else:
        chosen = float(threshold)
        source = "configured_threshold"
        note = "The threshold comes from evaluation.image_threshold in the config."
    predictions = (scores >= chosen).astype(np.int64)
    counts = confusion_counts(labels, predictions)
    rates = precision_recall_f1(counts)
    reason = None
    if rates["precision"] is None:
        reason = "Precision and F1 are undefined because no sample was predicted anomalous at this threshold."
    return {
        "auroc": auroc(labels, scores),
        "precision": rates["precision"],
        "recall": rates["recall"],
        "f1": rates["f1"],
        "threshold": chosen,
        "threshold_source": source,
        "threshold_note": note,
        "confusion_matrix": {
            "labels": ["normal", "anomaly"],
            "matrix": [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]],
            "counts": counts,
        },
        "unavailable_reason": reason,
    }


def stratified_subsample(labels, scores, max_count: int, seed: int):
    """Keep class balance when a pixel sample would exceed ``max_count``."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.shape[0] <= max_count:
        return labels, scores, False
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Cannot subsample pixel scores without both classes.")
    rng = np.random.default_rng(seed)
    positive_count = int(round(max_count * positive.size / labels.shape[0]))
    positive_count = min(max(positive_count, 1), positive.size, max_count - 1)
    negative_count = min(max_count - positive_count, negative.size)
    chosen = np.concatenate(
        [
            rng.choice(positive, size=positive_count, replace=False),
            rng.choice(negative, size=negative_count, replace=False),
        ]
    )
    return labels[chosen], scores[chosen], True
