"""Inspection reports for images that may not have ground truth."""

from __future__ import annotations

import json


UNAVAILABLE = (
    "Uploaded images have no MVTec ground-truth masks in this session, so image AUROC, "
    "pixel AUROC, precision, recall, F1, and the confusion matrix cannot be calculated. "
    "Run evaluate.py on a local MVTec test split to measure those metrics."
)


def inspection_report(category: str, metadata: dict, images: list[dict]) -> dict:
    return {
        "project": "VisionGuard",
        "kind": "inspection",
        "category": category,
        "synthetic": False,
        "ground_truth": None,
        "metrics": None,
        "metrics_unavailable_reason": UNAVAILABLE,
        "model": {
            "backbone": metadata.get("backbone"),
            "weights": metadata.get("weights"),
            "layers": metadata.get("layers"),
            "image_size": metadata.get("image_size"),
            "memory_patches": metadata.get("patches_after_coreset"),
            "train_images": metadata.get("train_images"),
        },
        "images": images,
        "disclaimer": (
            "Scores are nearest-neighbour distances to the nominal memory bank. "
            "Heatmap colors are min-max scaled per image. "
            "This report is not an industrial certification and it contains no invented metrics."
        ),
    }


def report_markdown(report: dict) -> str:
    lines = [
        f"# VisionGuard inspection — {report['category']}",
        "",
        report["disclaimer"],
        "",
        "## Metrics",
        "",
        report["metrics_unavailable_reason"],
        "",
        "## Model",
        "",
        "```json",
        json.dumps(report["model"], indent=2),
        "```",
        "",
        "## Images",
        "",
        "| File | Anomaly score | Map min | Map max |",
        "| --- | ---: | ---: | ---: |",
    ]
    for image in report["images"]:
        lines.append(
            f"| {image['file']} | {image['image_score']:.6f} | "
            f"{image['map_min']:.6f} | {image['map_max']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)
