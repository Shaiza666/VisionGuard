# Methodology

The detector is fit on defect-free images only. Test defect labels and masks are reserved for `evaluate.py`.

## Nominal memory

For each supported category separately:

1. Resize the image to 128×128 and apply the configured normalization. The default mean and standard deviation are the ImageNet values used by torchvision. The paper instead resizes to 256 and center-crops to 224 (Roth et al., CVPR 2022). Direct 128×128 resizing is a hardware choice. It changes aspect ratio when an image is not square.
2. Run a frozen ResNet-18. Keep the outputs of `layer2` and `layer3`. On a 128×128 input those maps are 16×16×128 and 8×8×256.
3. Average each map with a 3×3 kernel, stride 1, and padding 1. Roth et al. use a 3×3 neighbourhood and describe it as adaptive average pooling. The official PatchCore code unfolds that neighbourhood and then projects the channels. This project keeps the averaged channels and does not add that projection, so the patch vector has 384 dimensions after the next step.
4. Bilinearly upsample the coarser map to 16×16 and concatenate it with the finer map.
5. Keep at most `patches_per_image` patch vectors from each image, then a reservoir of at most `preselect_max` vectors across the category. Images themselves are not retained. This cap is a deviation from PatchCore, which can store every nominal patch before coreset selection. It exists so an 8 GB machine does not hold the full feature matrix.
6. Draw a greedy minimax coreset down to `min(max_bank_size, round(ratio × retained))`. Selected rows are original patch vectors, not new averages. If the farthest remaining distance is already zero, selection stops early.

The bank is saved as `reports/memory/<category>/memory_bank.npz` with a JSON sidecar. The sidecar records shapes, the seed, and library versions. Its `metrics` field is null.

## Scoring

For a new image, the same resize, normalization, and embedding are used.

Each patch is assigned its minimum L2 distance to the memory bank. Distances are exact and are computed in chunks so the full distance matrix is never allocated. Faiss is not used.

The pixel map is that distance grid, bilinearly upsampled to 128×128. Gaussian smoothing is implemented but off by default (`heatmap_smooth_sigma: 0`). Roth et al. used a Gaussian with sigma 4 at their higher resolution and did not tune it. Sigma 4 at 128×128 would blur a large fraction of the image, so it is not the default.

The image score is the maximum patch distance when `score_neighbors` is 1. That matches the default nearest-neighbour count in the official PatchCore code. If `score_neighbors` is greater than 1, the worst patch is reweighted by the softmax of its distances to that many nearest memory entries, as in equation 7 of Roth et al. The reweighted score is never larger than the raw maximum distance. The heatmap stays the raw nearest-neighbour map.

A displayed overlay and a separate blue-yellow-red heatmap min-max scale each image independently. The color is a viewing aid. It is not a probability and it is not comparable across images.

## Evaluation

`evaluate.py` scores the category test split once and writes `reports/evaluation/<category>.json` plus a log under `reports/logs/`.

Image AUROC uses every test image and does not need a threshold. Pixel AUROC compares the upsampled map with the mask resized by nearest neighbour. If the pixel count would exceed `evaluation.max_pixels`, both classes are subsampled and the JSON records `pixel_auroc_subsampled: true`.

Precision, recall, F1, and the confusion matrix need a cutoff on the distance score. If `evaluation.image_threshold` is null, the report uses the cutoff that maximizes F1 on the same split and sets `threshold_source` to `f1_max_on_this_split`. That cutoff saw the evaluation labels, so it is an optimistic summary, not a threshold you can deploy. If a split has only one class, those fields are null and `unavailable_reason` says why.

Each defect folder contains only anomalous images, so a metric cannot be computed inside that folder alone. Per-defect numbers compare that folder with the category's normal test images. Uploaded dashboard images have no masks, so the inspection report leaves metrics null.

AUROC is undefined when a split contains only one class, and the report says so instead of printing a placeholder.

## Reproducibility

`runtime.device` must be `cpu`, `num_workers` must be 0, and `batch_size` must be 1 or 2. The seed fixes Python, NumPy, and PyTorch, and it fixes reservoir and coreset draws. Training order follows sorted file names. Shuffle is off.
