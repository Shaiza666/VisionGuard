# Research gap

PatchCore already provides a strong cold-start detector on MVTec AD when a WideResNet-50, a 224×224 input, and a GPU nearest-neighbour index are available (Roth et al., CVPR 2022). That is not a gap in the published method. The gap this project addresses is practical and narrower.

The target machine has 8 GB of RAM, an Intel HD Graphics 520, and no NVIDIA GPU. CUDA is not available. The published PatchCore setup is a poor fit for that machine: the backbone is wide, the working resolution is 224×224 after a resize to 256, and the reference implementation uses Faiss. A researcher on this laptop needs a smaller, explicit, CPU-only pipeline whose deviations from the paper are written down before any accuracy number is produced.

VisionGuard therefore studies one question:

> How much of the PatchCore procedure can be run, without downloading data automatically and without claiming the paper's accuracy, when the backbone is ResNet-18, the image is 128×128, the memory bank is capped, and every distance is an exact CPU search?

The project does not claim a new architecture, a better AUROC, or a calibrated accept/reject threshold. Those would require measurements that have not been made.

## Questions left for later runs

Once `bottle` and `hazelnut` are available locally, the useful measurements are:

- image-level and pixel-level AUROC at 128×128 with ResNet-18, compared only against this project's own ablations
- the effect of the reservoir cap and the coreset size on those two categories
- whether the neighbour reweighting in Roth et al. changes image scores enough to matter at this resolution

VisA, the remaining thirteen MVTec categories, the PRO localization curve, and self-supervised backbone pretraining are out of scope until those measurements exist. No baseline number for them is invented here.
