# TRIDENT in-process batch runner

`trident.batch.run_slide_batch` executes segment → patch → extract with **global phasing**:
all slides complete segmentation before any patch work, then all patch before extract.

## Progress callbacks

| Callback | Signature | When |
|----------|-----------|------|
| `on_phase_start` | `(phase, total_slides)` | Once before each global phase loop |
| `on_progress` | `(phase, completed_in_phase, total_slides)` | After each slide finishes a phase |
| `on_item_done` | `dict` (see below) | After each slide finishes a phase (success or failure) |
| `on_extract_progress` | `(slide_idx, patches_done, patches_total, total_slides)` | Intra-slide extract granularity |
| `on_segment_tile_progress` | `(done, total)` | Optional intra-segment tile granularity |

### `on_item_done` payload

```python
{
    "idx": int,
    "slide_name": str,
    "stem": str,
    "ok": bool,
    "error": str | None,
    "phase": "segment" | "patch" | "extract",
    "artifacts": {
        "segmentation_geojson": str | None,  # segment phase
        "patches_h5": str | None,            # patch phase
        "features_h5": str | None,           # extract phase
    },
}
```

Paths are absolute under `config.output_dir/<slide_stem>/`.
