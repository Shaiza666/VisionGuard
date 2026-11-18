# Literature review

This review covers the papers the implementation is based on. Numbers quoted below are the authors' published results, not measurements from VisionGuard.

## Representations

He et al. introduced residual networks, including ResNet-18, for image classification (He et al., CVPR 2016). VisionGuard uses a frozen ResNet-18 because a laptop CPU with 8 GB of RAM cannot host the WideResNet-50 backbone that PatchCore uses by default. Wide residual networks are described by Zagoruyko and Komodakis (BMVC 2016). The ImageNet features themselves come from the ImageNet challenge dataset (Russakovsky et al., IJCV 2015), through the torchvision `IMAGENET1K_V1` weights.

## Industrial data

MVTec AD contains fifteen object and texture categories, high-resolution photographs, defect-free training images, and pixel-precise test masks (Bergmann et al., CVPR 2019). The journal version expands the benchmark discussion (Bergmann et al., IJCV 2021). The official release is licensed CC BY-NC-SA 4.0. See `docs/dataset_instructions.md`.

VisA contains 10,821 images across twelve objects, with image and pixel labels (Zou et al., ECCV 2022). It is the planned second dataset. No VisA loader is implemented.

## Reconstruction methods

Early unsupervised detectors learn to reconstruct normal images and treat a large reconstruction error as an anomaly. Schlegl et al. trained a generative adversarial model on healthy examples and searched for a latent code at test time (Schlegl et al., IPMI 2017). Bergmann et al. showed that an autoencoder trained with a structural-similarity objective localizes industrial defects better than a pixel-wise reconstruction loss on MVTec-style data (Bergmann et al., VISIGRAPP 2019). These methods train a generator or decoder. VisionGuard does not. It stores features from a frozen network.

## Embedding methods that lead to PatchCore

Cohen and Hoshen align multi-scale deep features of a test image to nominal images with nearest neighbours (SPADE; Cohen and Hoshen, 2020). The correspondence pyramid is expressive and expensive. Defard et al. fit a Gaussian at each spatial location of a pretrained feature map (PaDiM; Defard et al., ICPR 2021). That model assumes parts stay roughly aligned, because each image position has its own distribution.

Roth et al. keep a single memory of locally aggregated mid-level patches, subsample it with a greedy coreset, and score a test patch by its distance to the nearest nominal patch (PatchCore; Roth et al., CVPR 2022). The memory is not tied to an absolute image coordinate, which is the main reason VisionGuard follows PatchCore rather than PaDiM. On MVTec AD, Roth et al. report image-level AUROC of 99.1%, 99.0%, and 99.0% for their 25%, 10%, and 1% coreset settings with the default backbone and resolution, and up to 99.6% in a stronger resolution and ensemble setting. Those figures are cited only as the published reference. This project has not reproduced them.

The coreset step is a minimax facility-location selection, the same family of subset selection Sener and Savarese used for active learning (Sener and Savarese, ICLR 2018).

Zou et al. study a different question: how the pretraining objective itself can be changed so that ImageNet-style features are more useful for anomaly detection. Their SPot-the-Difference regularizer is self-supervised (Zou et al., ECCV 2022). VisionGuard does not train that objective.

## References

- Bergmann, P., Fauser, M., Sattlegger, D., and Steger, C. (2019). MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. CVPR.
- Bergmann, P., Löwe, S., Fauser, M., Sattlegger, D., and Steger, C. (2019). Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders. VISIGRAPP.
- Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., and Steger, C. (2021). The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. International Journal of Computer Vision, 129, 1038–1059. https://doi.org/10.1007/s11263-020-01400-4
- Cohen, N., and Hoshen, Y. (2020). Sub-Image Anomaly Detection with Deep Pyramid Correspondences. arXiv:2005.02357.
- Defard, T., Setkov, A., Loesch, A., and Audigier, R. (2021). PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization. ICPR. arXiv:2011.08785.
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., and Gehler, P. (2022). Towards Total Recall in Industrial Anomaly Detection. CVPR, 14318–14328. arXiv:2106.08265.
- Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., and Fei-Fei, L. (2015). ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision, 115, 211–252.
- Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., and Langs, G. (2017). Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery. IPMI.
- Sener, O., and Savarese, S. (2018). Active Learning for Convolutional Neural Networks: A Core-Set Approach. ICLR.
- Zagoruyko, S., and Komodakis, N. (2016). Wide Residual Networks. BMVC.
- Zou, Y., Jeong, J., Pemula, L., Zhang, D., and Dabeer, O. (2022). SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation. ECCV. arXiv:2207.14315.
