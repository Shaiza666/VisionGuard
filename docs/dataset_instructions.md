# Dataset instructions

VisionGuard does not download datasets. Place files on disk yourself and keep them out of git. The default location is `datasets/mvtec`, relative to this repository.

## MVTec AD

Official page: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

https://creativecommons.org/licenses/by-nc-sa/4.0/

Commercial use is not allowed. If a use might be commercial, ask MVTec before using the data. Do not commit the images, and do not redistribute them from this repository.

Cite, when the data is used:

- Bergmann, Fauser, Sattlegger, and Steger, CVPR 2019.
- Bergmann, Batzner, Fauser, Sattlegger, and Steger, IJCV 2021. https://doi.org/10.1007/s11263-020-01400-4

The full archive is several gigabytes. Check the size shown on the official page before downloading. Phase 1 reads only `bottle` and `hazelnut`. Other categories can be deleted after extraction if disk space is tight. There is no separate official zip, that this project knows of, for those two categories alone.

### Expected layout

```text
datasets/mvtec/
  bottle/
    train/good/
    test/good/
    test/<defect type>/
    ground_truth/<defect type>/*_mask.png
  hazelnut/
    train/good/
    test/good/
    test/<defect type>/
    ground_truth/<defect type>/*_mask.png
```

Masks for anomalous test images must be named `<image stem>_mask.png`. Good images have no mask. The loader treats every test folder other than `good` as anomalous.

Check the folders without loading the network:

```text
python train.py --config configs/default.yaml --check
```

A missing folder raises an error that points back to the official page. It does not start a download.

## VisA

Not supported in phase 1. Do not add the files yet expecting the loader to read them.

Official description and download pointers:

- Paper: Zou, Jeong, Pemula, Zhang, and Dabeer, ECCV 2022. arXiv:2207.14315.
- Code and data notes: https://github.com/amazon-science/spot-diff
- AWS Open Data registry: https://registry.opendata.aws/visa/

The dataset description there states that VisA is released under CC BY 4.0. Confirm the license text at the registry before downloading. VisA uses a different folder layout from MVTec AD, which is why a future loader has to be separate.

## Weights are not the dataset

`model.weights: imagenet` makes torchvision download ResNet-18 ImageNet weights the first time `train.py` builds a bank. That is a model-weight download, not an MVTec or VisA download. `model.weights: none` skips it and produces features that must not be reported as detection results.
