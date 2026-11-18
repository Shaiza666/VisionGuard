"""VisionGuard inspection dashboard. Every score comes from the saved memory bank."""

from __future__ import annotations

import json
from PIL import Image

from data.layout import DatasetLayoutError, summarize_category
from data.transforms import build_image_transform
from evaluation.report import inspection_report, report_markdown
from inference.predict import load_detector, memory_dir, tensor_to_uint8
from utils.config import load_config, project_root
from visualization.heatmap import heatmap_image, overlay_anomaly


def main() -> None:
    import streamlit as st

    config = load_config(project_root() / "configs" / "default.yaml")
    st.set_page_config(page_title="VisionGuard", layout="wide")
    section = st.sidebar.radio(
        "Section",
        ["Overview", "Dataset", "Model", "Build memory", "Inspect", "Evaluation"],
    )
    category = st.sidebar.selectbox("Category", config.data.categories)
    st.title("VisionGuard")
    st.caption("CPU-only PatchCore-inspired inspection. Scores are distances, not probabilities.")

    if section == "Overview":
        _overview()
    elif section == "Dataset":
        _dataset(config, category)
    elif section == "Model":
        _model(config, category)
    elif section == "Build memory":
        _build(config, category)
    elif section == "Inspect":
        _inspect(config, category)
    else:
        _evaluation(config, category)


def _overview() -> None:
    import streamlit as st

    st.subheader("Project overview")
    st.markdown(
        """
VisionGuard fits a normal-appearance memory from defect-free images and scores
later images by nearest-neighbour distance in ResNet-18 feature space.

- Backbone: frozen ImageNet ResNet-18, layers 2 and 3
- Input: 128×128, batch size 1 or 2, CPU only
- Categories in this phase: bottle and hazelnut
- Test images are not used while the memory bank is built

No accuracy number is shown until `evaluate.py` has written a report from real
MVTec ground truth. Synthetic tests are not industrial results.
        """
    )


def _dataset(config, category: str) -> None:
    import streamlit as st

    st.subheader("Dataset")
    st.write(f"Expected root: `{config.data.root}`")
    try:
        summary = summarize_category(config.data.root, category)
    except DatasetLayoutError as exc:
        st.error(str(exc))
        st.info("The dashboard does not download MVTec AD.")
        return
    st.text(summary.render())
    if summary.missing_masks:
        st.warning(
            f"{summary.missing_masks} anomalous test images have no mask. "
            "Pixel metrics cannot be calculated for those images."
        )


def _model(config, category: str) -> None:
    import streamlit as st

    st.subheader("Model")
    metadata = _read_metadata(config, category)
    if metadata is None:
        st.warning("No memory bank has been saved for this category.")
        return
    st.json(
        {
            "backbone": metadata.get("backbone"),
            "weights": metadata.get("weights"),
            "layers": metadata.get("layers"),
            "neighborhood": metadata.get("neighborhood"),
            "image_size": metadata.get("image_size"),
            "memory_patches": metadata.get("patches_after_coreset"),
            "patches_before_coreset": metadata.get("patches_before_coreset"),
            "train_images": metadata.get("train_images"),
            "coreset_ratio": metadata.get("coreset_ratio"),
            "device": metadata.get("device"),
            "metrics": metadata.get("metrics"),
        }
    )
    st.caption(metadata.get("note", ""))


def _build(config, category: str) -> None:
    import streamlit as st

    st.subheader("Build the normal memory bank")
    st.write(
        "This uses only the defect-free training split. It runs on CPU and can take "
        "a long time for a full MVTec category. ImageNet weights download once if they "
        "are not cached. The dataset itself is not downloaded."
    )
    if not st.button("Build memory bank", type="primary"):
        return
    from training.memory_bank import build_category_memory

    bar = st.progress(0.0)
    status = st.empty()

    def progress(seen: int, total: int, message: str) -> None:
        status.write(message)
        if total > 0:
            bar.progress(min(seen / total, 1.0))

    try:
        destination, metadata = build_category_memory(config, category, progress=progress)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return
    st.success(f"Saved {metadata['patches_after_coreset']} patches to {destination}.")
    st.caption("This save does not contain detection metrics.")


def _inspect(config, category: str) -> None:
    import streamlit as st

    st.subheader("Inspection")
    metadata = _read_metadata(config, category)
    if metadata is None:
        st.warning("Build a memory bank before inspection. No score will be invented.")
        return
    detector, metadata = load_detector(config, category)
    transform = build_image_transform(
        int(metadata["image_size"]), tuple(metadata["mean"]), tuple(metadata["std"])
    )
    single = st.file_uploader(
        "Single image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="single",
    )
    if single is not None:
        _show_prediction(detector, transform, metadata, single)

    uploads = st.file_uploader(
        "Batch of images, scored one at a time",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="batch",
    )
    if not uploads:
        return
    rows = []
    for upload in uploads:
        prediction, _display, _overlay, _heat, raw_min, raw_max = _predict_upload(
            detector, transform, metadata, upload
        )
        rows.append(
            {
                "file": upload.name,
                "image_score": prediction.image_score,
                "map_min": raw_min,
                "map_max": raw_max,
            }
        )
    st.dataframe(rows, use_container_width=True)
    report = inspection_report(category, metadata, rows)
    st.download_button(
        "Download inspection report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"{category}_inspection.json",
        mime="application/json",
    )
    st.download_button(
        "Download inspection report (Markdown)",
        data=report_markdown(report),
        file_name=f"{category}_inspection.md",
        mime="text/markdown",
    )


def _evaluation(config, category: str) -> None:
    import streamlit as st

    st.subheader("Evaluation")
    path = config.output.directory / "evaluation" / f"{category}.json"
    if not path.is_file():
        st.info(
            "No evaluation report exists for this category. Metrics cannot be calculated "
            "until evaluate.py scores the local MVTec test split against its ground truth. "
            "Nothing is filled in with a placeholder."
        )
        st.code(f"python evaluate.py --config configs/default.yaml --category {category}")
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    st.caption(report.get("disclaimer", ""))
    image_metrics = report.get("image_metrics", {})
    pixel_metrics = report.get("pixel_metrics", {})
    columns = st.columns(4)
    for column, name in zip(columns, ("auroc", "precision", "recall", "f1")):
        value = image_metrics.get(name)
        column.metric(f"Image {name}", "n/a" if value is None else f"{value:.4f}")
    if image_metrics.get("unavailable_reason"):
        st.warning(image_metrics["unavailable_reason"])
    if image_metrics.get("threshold_note"):
        st.caption(image_metrics["threshold_note"])
    matrix = image_metrics.get("confusion_matrix")
    if matrix:
        st.write("Image confusion matrix, rows are normal then anomaly, columns are predicted normal then anomaly.")
        st.dataframe(matrix["matrix"], use_container_width=True)
    st.write("Pixel metrics")
    st.json(
        {
            "auroc": pixel_metrics.get("auroc"),
            "precision": pixel_metrics.get("precision"),
            "recall": pixel_metrics.get("recall"),
            "f1": pixel_metrics.get("f1"),
            "unavailable_reason": pixel_metrics.get("unavailable_reason"),
            "subsampled": report.get("pixel_auroc_subsampled"),
            "note": report.get("pixel_metrics_note"),
        }
    )
    st.write("Per defect type, each compared with the normal test images")
    st.json(report.get("per_defect", {}))
    figure_dir = config.output.directory / "figures" / category
    images = sorted(figure_dir.glob("*_overlay.png")) if figure_dir.is_dir() else []
    if images:
        st.write("Localization examples saved by the evaluation run")
        for image_path in images[:4]:
            st.image(str(image_path), caption=image_path.name, use_container_width=True)


def _show_prediction(detector, transform, metadata, upload) -> None:
    import streamlit as st

    prediction, display, overlay, heat, raw_min, raw_max = _predict_upload(
        detector, transform, metadata, upload
    )
    original, heatmap, blended = st.columns(3)
    original.image(display, caption="Preprocessed original", use_container_width=True)
    heatmap.image(heat, caption="Anomaly heatmap, min-max per image", use_container_width=True)
    blended.image(overlay, caption="Anomaly overlay", use_container_width=True)
    st.metric("Anomaly score", f"{prediction.image_score:.6f}")
    st.caption(f"Raw map range {raw_min:.6f} to {raw_max:.6f}. Not a defect probability.")


def _predict_upload(detector, transform, metadata, upload):
    with Image.open(upload) as source:
        rgb = source.convert("RGB")
    tensor = transform(rgb).unsqueeze(0)
    prediction = detector.score_batch(tensor)[0]
    display = tensor_to_uint8(tensor[0], tuple(metadata["mean"]), tuple(metadata["std"]))
    overlay, raw_min, raw_max = overlay_anomaly(display, prediction.anomaly_map)
    heat = heatmap_image(prediction.anomaly_map)
    return prediction, display, overlay, heat, raw_min, raw_max


def _read_metadata(config, category: str) -> dict | None:
    path = memory_dir(config, category) / "metadata.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
