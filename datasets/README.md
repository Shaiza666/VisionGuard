# Local datasets

Put MVTec AD here, not inside the `data/` Python package. A folder named `data/mvtec` would shadow the import path.

```text
datasets/mvtec/bottle/...
datasets/mvtec/hazelnut/...
```

The directory is gitignored. Download and license steps are in `docs/dataset_instructions.md`. Nothing in this folder is fetched by the code.
