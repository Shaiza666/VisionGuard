"""Evaluate one saved memory bank on the local MVTec test split."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from data.dataset import MVTecDataset, collate_test_batch
from data.transforms import build_image_transform
from evaluation.metrics import decision_summary, stratified_subsample
from inference.predict import load_detector, tensor_to_uint8
from training.log import RunLog
from utils.config import ProjectConfig
from visualization.heatmap import heatmap_image, overlay_anomaly


DISCLAIMER = (
    "These numbers were measured by this local run. They are not published "
    "PatchCore results and they are not calibrated probabilities."
)


def evaluate_category(config: ProjectConfig, category: str) -> dict:
    detector, metadata = load_detector(config, category)
    transform = build_image_transform(
        config.data.image_size, config.data.mean, config.data.std
    )
    dataset = MVTecDataset(
        root=config.data.root,
        category=category,
        split="test",
        transform=transform,
        image_size=config.data.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        pin_memory=False,
        collate_fn=collate_test_batch,
    )
    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    by_defect: dict[str, list[float]] = defaultdict(list)
    defect_names: list[str] = []
    saved = 0
    figure_dir = config.output.directory / "figures" / category
    log = RunLog(config.output.directory / "logs", category, "evaluation")
    seen = 0

    for batch in loader:
        predictions = detector.score_batch(batch["image"])
        labels = batch["label"].tolist()
        masks = batch["mask"]
        for index, prediction in enumerate(predictions):
            label = int(labels[index])
            image_labels.append(label)
            image_scores.append(prediction.image_score)
            by_defect[batch["defect_type"][index]].append(prediction.image_score)
            defect_names.append(batch["defect_type"][index])
            pixel_labels.append(masks[index, 0].numpy().astype(np.uint8).reshape(-1))
            pixel_scores.append(prediction.anomaly_map.reshape(-1))
            if saved < config.evaluation.save_examples:
                _save_example(
                    figure_dir,
                    seen,
                    batch["image"][index],
                    prediction.anomaly_map,
                    config.data.mean,
                    config.data.std,
                    batch["defect_type"][index],
                    prediction.image_score,
                )
                saved += 1
            seen += 1
            if seen % 10 == 0 or seen == len(dataset):
                message = f"[{category}] scored {seen}/{len(dataset)} test images"
                print(message, flush=True)
                log.write(message)

    label_array = np.asarray(image_labels, dtype=np.int64)
    score_array = np.asarray(image_scores, dtype=np.float64)
    flat_labels = np.concatenate(pixel_labels)
    flat_scores = np.concatenate(pixel_scores)
    sampled_labels, sampled_scores, subsampled = stratified_subsample(
        flat_labels, flat_scores, config.evaluation.max_pixels, config.seed
    )
    image_metrics = decision_summary(label_array, score_array, config.evaluation.image_threshold)
    pixel_metrics = decision_summary(sampled_labels, sampled_scores, config.evaluation.image_threshold)
    report = {
        "project": "VisionGuard",
        "category": category,
        "disclaimer": DISCLAIMER,
        "test_images": int(label_array.size),
        "test_anomalies": int(np.sum(label_array == 1)),
        "test_good": int(np.sum(label_array == 0)),
        "image_metrics": image_metrics,
        "pixel_metrics": pixel_metrics,
        "pixel_auroc_subsampled": subsampled,
        "pixel_count_used": int(sampled_labels.size),
        "pixel_metrics_note": (
            "Pixel metrics use the upsampled anomaly map and the resized ground-truth mask. "
            "If pixel_auroc_subsampled is true, pixel metrics were computed on a stratified "
            "subset and are not a full-resolution measurement."
        ),
        "per_defect": _per_defect(defect_names, label_array, score_array),
        "memory_patches": int(metadata["patches_after_coreset"]),
        "image_size": int(metadata["image_size"]),
        "backbone": metadata["backbone"],
        "weights": metadata["weights"],
        "mean_image_score_by_defect": {
            name: float(np.mean(values)) for name, values in sorted(by_defect.items())
        },
        "heatmap_smooth_sigma": config.scoring.heatmap_smooth_sigma,
        "score_neighbors": config.scoring.score_neighbors,
        "example_figures": saved,
        "figure_note": (
            "PNG heatmaps and overlays are min-max colored per image for display. "
            "The accompanying JSON stores the raw image score."
        ),
    }
    destination = config.output.directory / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / f"{category}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[{category}] wrote {report_path}", flush=True)
    log.write(f"[{category}] wrote {report_path}")
    return report


def _per_defect(defect_names: list[str], labels: np.ndarray, scores: np.ndarray) -> dict:
    """Compare each defect folder with the normal test images.

    A defect folder contains only anomalous images, so AUROC inside that folder
    alone is impossible. The comparison below uses the category's normal test
    images as the negative class.
    """
    names = np.asarray(defect_names)
    good = labels == 0
    if int(np.sum(good)) == 0:
        return {
            "unavailable_reason": "No normal test images were available, so per-defect metrics cannot be calculated."
        }
    results = {}
    for name in sorted(set(names.tolist())):
        if name == "good":
            continue
        chosen = names == name
        subset_labels = np.concatenate([np.zeros(int(np.sum(good)), dtype=np.int64), np.ones(int(np.sum(chosen)), dtype=np.int64)])
        subset_scores = np.concatenate([scores[good], scores[chosen]])
        summary = decision_summary(subset_labels, subset_scores, threshold=None)
        summary["images"] = int(np.sum(chosen))
        results[name] = summary
    return results


def _save_example(folder, index, image, anomaly_map, mean, std, defect_type, score) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rgb = tensor_to_uint8(image, mean, std)
    overlay, raw_min, raw_max = overlay_anomaly(rgb, anomaly_map)
    heat = heatmap_image(anomaly_map)
    stem = f"{index:03d}_{defect_type}"
    Image.fromarray(rgb).save(folder / f"{stem}_image.png")
    heat.save(folder / f"{stem}_heatmap.png")
    overlay.save(folder / f"{stem}_overlay.png")
    (folder / f"{stem}.json").write_text(
        json.dumps(
            {
                "defect_type": defect_type,
                "image_score": score,
                "map_min": raw_min,
                "map_max": raw_max,
                "display": "Overlay colors are min-max normalized for this image only.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
