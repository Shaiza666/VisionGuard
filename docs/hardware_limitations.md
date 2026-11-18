# Hardware limitations

Verified machine for this project:

| Item | Value |
| --- | --- |
| Operating system | Windows |
| RAM | 8 GB |
| GPU | Intel HD Graphics 520 |
| NVIDIA GPU | None |
| CUDA | Not supported |
| Training device | CPU |
| Editor | Cursor |

PyTorch will not be pointed at the Intel GPU. DirectML and other GPU backends are out of scope. `runtime.device` must stay `cpu`.

## Settings that protect memory

| Setting | Phase-1 value | Reason |
| --- | --- | --- |
| Image size | 128×128 | Feature maps stay 16×16 at `layer2`. |
| Batch size | 1, or at most 2 | One forward pass at a time. |
| DataLoader workers | 0 | Extra worker processes duplicate memory on Windows. |
| Backbone | ResNet-18 | The PatchCore default, WideResNet-50, is not used. |
| Patch reservoir | 8192 vectors | The full nominal patch matrix is not kept. |
| Coreset cap | 2048 vectors | Bounds the exact nearest-neighbour search. |
| Categories at once | One | `train.py` finishes a category before starting the next. |

The dataset class opens one image per index. The memory builder may hold the reservoir, which at 8192 vectors of 384 float32 values is about 13 MB, plus the current batch. That is the intended peak feature storage. Other applications should still be closed during a real build. This document does not claim a measured runtime, because no full MVTec category has been built in this setup.

## Environment inspected at setup

The setup session found Python 3.12.10 at `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe`. A project virtual environment was then created at `.venv` with CPU wheels `torch==2.15.0+cpu` and `torchvision==0.29.2+cpu`, plus NumPy 2.5.5, Pillow 13.0.0, PyYAML 6.0.3, Streamlit 1.67.0, and pytest 9.2.0. Those pins are in `requirements.txt`. No CUDA wheel was installed.

```text
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` points pip at the CPU PyTorch index. After installation, a memory-bank sidecar records the `torch` and `torchvision` versions actually used. Those versions are not known until that install or that run happens.

## What not to change on this machine

- Do not set the image size to 224 or 256.
- Do not switch the backbone to WideResNet-50 or a transformer.
- Do not raise DataLoader workers above 0.
- Do not load every category's images into a list.
