# Reports

This directory is empty of results on purpose.

`evaluate.py` is the only writer of detection numbers. Until it has been run on a local MVTec split, there is no image AUROC, pixel AUROC, or example heatmap to show. Published scores from Roth et al., CVPR 2022, are not copied here.

Generated files, when a run happens:

- `memory/<category>/` — nominal patch bank and metadata. `metrics` in the metadata stays null.
- `evaluation/<category>.json` — measurements from that run.
- `figures/<category>/` — display overlays. Colors are min-max scaled per image.

Those generated paths are gitignored.
