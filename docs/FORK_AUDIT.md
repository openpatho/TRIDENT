# Fork audit notes (2026-06-24)

Baseline comparison: `openpatho/TRIDENT@main` vs mahmoodlab pin `adf3b7e8fdd7c815f33bec79da0d8d265200c2d6`.

## Pre-batch fork state

- Fork includes docs/CI/skills and full `trident/` tree from upstream.
- GeoPandas `union_all` fix applied in `WSIPatcher.py` (was Dockerfile sed in worker).

## Batch API additions (this change)

- `trident/batch/` — in-process `run_slide_batch` with progress callbacks and model cache.
- `trident/weights.py` — S3/local checkpoint resolution; HF opt-in via `TRIDENT_ALLOW_HF_WEIGHTS_DOWNLOAD`.
- WSI progress hooks: `tile_progress_callback`, `progress_callback`, `features_h5_path`.

## Worker integration

- `trident_batch_adapter.py` maps callbacks to OpenPatho job progress (app layer, not in this repo).
