# Project plan

## Phase 1 — structure and a CPU implementation

Status: done in this repository, without a dataset download and without a training run on MVTec.

- Package layout, CPU configuration, and hardware limits.
- PatchCore-inspired memory bank for `bottle` and `hazelnut` only.
- Documentation that separates this code from published PatchCore numbers.
- Synthetic tests for layout, coreset size, distances, AUROC, and a two-image random-weight bank.

Exit condition: `pytest` passes, and `train.py --check` fails with a clear message until MVTec is present. No accuracy is claimed.

## Phase 2 — dashboard, metrics, and logs

Status: code is in the repository. Real MVTec measurements are not.

Implemented:

- Image and pixel AUROC, precision, recall, F1, and a confusion matrix.
- The F1 cutoff is labeled `f1_max_on_this_split` unless you set `evaluation.image_threshold`.
- Per-defect metrics compare each defect folder with the normal test images. A defect folder alone cannot support AUROC.
- Memory-build and evaluation logs, plus a saved config snapshot beside the bank.
- Streamlit sections for overview, dataset, model, memory build, single-image and batch inspection, and evaluation.
- Inspection downloads that keep metrics null when the upload has no ground truth.
- Heatmap and overlay images.

Not done:

- MVTec AD is not on disk, so no bottle or hazelnut bank has been built with ImageNet weights.
- No measured AUROC, precision, recall, F1, or confusion matrix exists for real products.
- The dashboard build button is wired to the real CPU routine, but it was not used to start a full category.

## Phase 3 — publication

Status: the local project is prepared for the public repository. Real MVTec measurements are still not done.

- README, architecture diagram, hardware notes, and methodology limitations describe the implemented system.
- Datasets, virtual environments, credentials, and generated banks stay out of git.
- No screenshot of a detection result is published, because no real inspection has been run.

## Next — local MVTec measurements

Status: not started. Requires MVTec AD under CC BY-NC-SA 4.0.

- Place `bottle` and `hazelnut` under `datasets/mvtec`.
- Build one memory bank per category with ImageNet weights.
- Run `evaluate.py` and keep the JSON it writes.
- Record runtime and peak memory on this CPU. Do not copy paper tables into the report.

Exit condition: each category has a sidecar, an evaluation JSON produced by the code, and a short note of what was changed relative to Roth et al.

## Later — controlled ablations

Status: planned.

- Reservoir size, coreset ratio, and `score_neighbors` on the same two categories.
- Compare only against other VisionGuard runs at 128×128.
- Decide whether sigma smoothing helps at this resolution. The paper's sigma of 4 stays off until that comparison exists.

## Phase 4 — broader data

Status: planned.

- A separate VisA reader after the license and folder layout are checked.
- Additional MVTec categories only if RAM still stays within the 8 GB budget.
- PRO localization, using MVTec's metric definition rather than a substitute.

## Out of scope

- CUDA, Intel GPU offload, and transformer backbones.
- Automatic dataset downloads.
- Defect classification by type.
- Treating display colors as probabilities.
- Listing tools or assistants as contributors.
