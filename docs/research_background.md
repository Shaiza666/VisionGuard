# Research background

VisionGuard is a one-class industrial inspection project. It learns the appearance of defect-free products and then scores unseen images by how far their local patches are from that nominal appearance. Defect labels are not used to fit the detector. Pixel masks, when they exist, are used only to measure localization after the memory bank has been built.

The project title uses the phrase "self-supervised" in the same loose sense as much of the industrial anomaly-detection literature: the detector is not trained on labeled defects. The implemented method is more precise than that phrase. It is a PatchCore-inspired memory-bank detector on a frozen ImageNet ResNet-18. ImageNet pretraining is supervised classification, not a self-supervised objective, and phase 1 does not update backbone weights. A genuine self-supervised pretraining method, SPot-the-Difference, is cited as planned future work and is not implemented.

## Cold-start inspection

Manufacturers can usually photograph acceptable parts. They often cannot list every future defect. Bergmann et al. designed MVTec AD around that setting: each category has defect-free training images and a test set that mixes defect-free images with several defect types, plus pixel masks (Bergmann et al., CVPR 2019; Bergmann et al., IJCV 2021).

Roth et al. call this the cold-start problem and argue for keeping a memory of nominal patch features rather than training a defect classifier (Roth et al., CVPR 2022). VisionGuard follows that problem statement for two MVTec categories, `bottle` and `hazelnut`.

## What this repository is not

Phase 1 does not report detection accuracy. The PatchCore paper reports image-level AUROC up to 99.6% on MVTec AD for its strongest published setting (Roth et al., CVPR 2022). That number belongs to their WideResNet-50 / higher-resolution experiments. It is not a VisionGuard result. This repository will write metrics only when `evaluate.py` is run on a local test split, and those files are not part of the initial project.
