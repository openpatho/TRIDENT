from __future__ import annotations

import logging
import os
import threading
import time

from trident.batch.cache import (
    _resolve_device,
    cached_patch_encoder,
    cached_segmenter,
)
from trident.batch.region import attach_region_to_slide, is_region_entry
from trident.batch.types import (
    ExtractProgressCallback,
    ItemDoneCallback,
    PipelineConfig,
    ProgressCallback,
    SlideEntry,
    SlideResult,
    StageName,
    TileProgressCallback,
)

logger = logging.getLogger(__name__)

SEGMENT_TARGET_MAG = 1.25


def _slide_step_timeout_seconds() -> float:
    """Per-slide hard timeout (0 = disabled). Default 1200s for hard WSIs."""
    raw = os.environ.get("SLIDE_STEP_TIMEOUT_SECONDS", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass
    return 1200.0


def _run_with_slide_timeout(fn, *, timeout_s: float, label: str):
    """Run ``fn`` in a daemon thread; raise TimeoutError if it exceeds ``timeout_s``.

    The worker thread cannot be killed in-process (CUDA). Callers that hit a
    timeout should recycle the container after marking the slide failed.
    """
    if timeout_s <= 0:
        return fn()

    box: dict = {}

    def _target() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 — re-raised on join path
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True, name=f"slide-step:{label[:40]}")
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(
            f"slide_step_timeout after {timeout_s:.0f}s ({label}). "
            "Worker thread leaked; container should recycle."
        )
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _slide_stem(slide_name: str) -> str:
    base = os.path.basename(str(slide_name or "").strip())
    lower = base.lower()
    for suffix in (".vslide.json", ".vslide"):
        if lower.endswith(suffix):
            return base[: -len(suffix)] or base
    if "__virtual-" in lower:
        if lower.endswith(".json"):
            return base[: -len(".json")] or base
        return base
    stem, _ext = os.path.splitext(base)
    return stem or base


def _job_dir(output_dir: str, slide_name: str) -> str:
    return os.path.join(output_dir, _slide_stem(slide_name))


def _coords_dir(job_dir: str, mag: int, patch_size: int, overlap: int) -> str:
    return os.path.join(job_dir, f"{mag}x_{patch_size}px_{overlap}px_overlap")


def _load_slide(entry: SlideEntry, config: PipelineConfig, job_dir: str):
    from trident.wsi_objects.WSIFactory import load_wsi

    slide_path = os.path.join(config.input_dir, entry.local_input_name)
    tissue_geojson = os.path.join(job_dir, "contours_geojson", f"{_slide_stem(entry.slide_name)}.geojson")
    tissue_seg_path = tissue_geojson if os.path.isfile(tissue_geojson) else None
    if is_region_entry(entry):
        tissue_seg_path = None
    kwargs: dict = {"lazy_init": False}
    if config.custom_mpp is not None:
        kwargs["mpp"] = float(config.custom_mpp)
    if config.seg_num_workers is not None:
        kwargs["max_workers"] = int(config.seg_num_workers)
    slide = load_wsi(
        slide_path,
        tissue_seg_path=tissue_seg_path,
        name=_slide_stem(entry.slide_name),
        **kwargs,
    )
    if is_region_entry(entry):
        attach_region_to_slide(slide, entry)
    return slide


def _effective_mag(slide, config: PipelineConfig) -> float:
    try:
        return float(slide._fetch_magnification())
    except Exception:
        pass
    if config.target_mag:
        return float(config.target_mag)
    return float(getattr(slide, "mag", 20) or 20)


def _apply_grandqc(slide, config: PipelineConfig, device: str, artifact_model=None) -> None:
    if not config.remove_artifacts and not config.remove_penmarks:
        return
    if artifact_model is None:
        try:
            from trident.segmentation_models.load import segmentation_model_factory

            remove_penmarks_only = bool(config.remove_penmarks) and not config.remove_artifacts
            artifact_model = segmentation_model_factory(
                "grandqc_artifact",
                remove_penmarks_only=remove_penmarks_only,
            )
        except Exception as exc:
            logger.warning("GrandQC artifact model unavailable: %s", exc)
            return
    job_dir = _job_dir(config.output_dir, slide.name)
    seg_kwargs = dict(
        segmentation_model=artifact_model,
        target_mag=getattr(artifact_model, "target_mag", 10),
        job_dir=job_dir,
        device=device,
        holes_are_tissue=False,
        batch_size=max(1, int(config.seg_batch_size or 32)),
    )
    if config.seg_num_workers is not None:
        seg_kwargs["num_workers"] = int(config.seg_num_workers)
    slide.segment_tissue(**seg_kwargs)


def _run_segment(
    entry: SlideEntry,
    config: PipelineConfig,
    device: str,
    segmenter,
    *,
    on_segment_tile_progress: TileProgressCallback | None = None,
    artifact_model=None,
) -> None:
    job_dir = _job_dir(config.output_dir, entry.slide_name)
    os.makedirs(job_dir, exist_ok=True)
    slide = _load_slide(entry, config, job_dir)
    try:
        seg_kwargs = dict(
            segmentation_model=segmenter,
            target_mag=SEGMENT_TARGET_MAG,
            job_dir=job_dir,
            device=device,
            holes_are_tissue=True,
            batch_size=max(1, int(config.seg_batch_size or 32)),
            tile_progress_callback=on_segment_tile_progress,
        )
        if config.seg_num_workers is not None:
            seg_kwargs["num_workers"] = int(config.seg_num_workers)
        slide.segment_tissue(**seg_kwargs)
        _apply_grandqc(slide, config, device, artifact_model=artifact_model)
    finally:
        if hasattr(slide, "release"):
            slide.release()


def _run_patch(entry: SlideEntry, config: PipelineConfig) -> str:
    job_dir = _job_dir(config.output_dir, entry.slide_name)
    save_coords = _coords_dir(job_dir, config.target_mag, config.patch_size, config.overlap)
    slide = _load_slide(entry, config, job_dir)
    try:
        effective_mag = _effective_mag(slide, config)
        coords_path = slide.extract_tissue_coords(
            target_mag=effective_mag,
            patch_size=config.patch_size,
            save_coords=save_coords,
            overlap=config.overlap,
        )
        return coords_path
    finally:
        if hasattr(slide, "release"):
            slide.release()


def _resolve_coords_path(save_coords: str, slide_stem: str) -> str:
    """Locate ``*_patches.h5`` for extract after hydrate / stem mismatch.

    Prefer ``<stem>_patches.h5``, then any coords file whose name contains the
    stem, then the sole ``*_patches.h5`` in the patches dir.
    """
    patches_dir = os.path.join(save_coords, "patches")
    expected = os.path.join(patches_dir, f"{slide_stem}_patches.h5")
    if os.path.isfile(expected):
        return expected
    if not os.path.isdir(patches_dir):
        return expected
    try:
        names = sorted(
            name
            for name in os.listdir(patches_dir)
            if name.endswith("_patches.h5")
            and os.path.isfile(os.path.join(patches_dir, name))
        )
    except OSError:
        return expected
    if not names:
        return expected
    stem_lower = str(slide_stem or "").lower()
    for name in names:
        if stem_lower and stem_lower in name.lower():
            return os.path.join(patches_dir, name)
    if len(names) == 1:
        return os.path.join(patches_dir, names[0])
    return expected


def _run_extract(
    entry: SlideEntry,
    config: PipelineConfig,
    device: str,
    encoder,
    *,
    progress_callback=None,
) -> str:
    job_dir = _job_dir(config.output_dir, entry.slide_name)
    save_coords = _coords_dir(job_dir, config.target_mag, config.patch_size, config.overlap)
    slide = _load_slide(entry, config, job_dir)
    try:
        stem = _slide_stem(entry.slide_name)
        coords_path = _resolve_coords_path(save_coords, slide.name or stem)
        if not os.path.isfile(coords_path):
            coords_path = _resolve_coords_path(save_coords, stem)
        if not os.path.isfile(coords_path):
            raise FileNotFoundError(f"Coords not found for extract: {coords_path}")
        features_dir = os.path.join(save_coords, f"features_{config.patch_encoder}")
        direct_h5 = config.features_h5_by_slide.get(entry.slide_name) or config.features_h5_by_slide.get(stem)
        return slide.extract_patch_features(
            patch_encoder=encoder,
            coords_path=coords_path,
            save_features=features_dir,
            device=device,
            batch_limit=max(1, int(config.batch_size or 32)),
            progress_callback=progress_callback,
            features_h5_path=direct_h5,
        )
    finally:
        if hasattr(slide, "release"):
            slide.release()


def _expand_stages(stage: StageName) -> list[str]:
    if stage == "all":
        return ["segment", "patch", "extract"]
    if stage == "segment_patch":
        return ["segment", "patch"]
    return [stage]


def run_slide_batch(
    entries: list[SlideEntry],
    config: PipelineConfig,
    *,
    stage: StageName = "all",
    on_item_done: ItemDoneCallback | None = None,
    on_progress: ProgressCallback | None = None,
    on_extract_progress: ExtractProgressCallback | None = None,
    on_segment_tile_progress: TileProgressCallback | None = None,
) -> tuple[int, list[SlideResult]]:
    """Run segment / patch / extract in-process for a batch of slides."""
    device = _resolve_device(config.gpu_id)
    results: list[SlideResult] = []
    failed_errors: dict[int, str] = {}
    total = len(entries)
    stages = _expand_stages(stage)

    segmenter = None
    encoder = None
    artifact_model = None
    if "segment" in stages:
        if str(config.segment_backend or "hest").lower() == "hest":
            if not device.startswith("cuda"):
                logger.error("HEST segmentation requires CUDA")
                return 1, []
            segmenter = cached_segmenter(
                config.segment_backend,
                config.seg_conf_thresh,
                device,
                ckpt_path=config.segmenter_ckpt_path,
            )
        else:
            segmenter = cached_segmenter(
                config.segment_backend,
                config.seg_conf_thresh,
                device,
                ckpt_path=config.segmenter_ckpt_path,
            )
        if config.remove_artifacts or config.remove_penmarks:
            try:
                from trident.segmentation_models.load import segmentation_model_factory

                remove_penmarks_only = bool(config.remove_penmarks) and not config.remove_artifacts
                artifact_model = segmentation_model_factory(
                    "grandqc_artifact",
                    remove_penmarks_only=remove_penmarks_only,
                )
            except Exception as exc:
                logger.warning("GrandQC preload skipped: %s", exc)
    if "extract" in stages:
        encoder = cached_patch_encoder(
            config.patch_encoder,
            device,
            ckpt_path=config.encoder_ckpt_path,
        )

    for step in stages:
        step_done = 0
        # Prefer parent/wave idxs when unique (multi-GPU shards). If callers left
        # the default idx=0 on every entry, fall back to local enumerate.
        parent_idxs: list[int] = []
        for local_i, entry in enumerate(entries):
            raw_idx = getattr(entry, "idx", None)
            try:
                parent_idxs.append(int(raw_idx) if raw_idx is not None else int(local_i))
            except (TypeError, ValueError):
                parent_idxs.append(int(local_i))
        use_parent_idx = len(parent_idxs) == len(set(parent_idxs))
        step_timeout_s = _slide_step_timeout_seconds()

        for local_i, entry in enumerate(entries):
            emit_idx = parent_idxs[local_i] if use_parent_idx else int(local_i)
            if emit_idx in failed_errors:
                continue
            try:
                if step == "segment":
                    tile_cb = on_segment_tile_progress

                    def _tile_cb(done: int, tot: int, _cb=tile_cb) -> None:
                        if _cb is not None:
                            _cb(done, tot)

                    _run_with_slide_timeout(
                        lambda: _run_segment(
                            entry,
                            config,
                            device,
                            segmenter,
                            on_segment_tile_progress=_tile_cb if tile_cb else None,
                            artifact_model=artifact_model,
                        ),
                        timeout_s=step_timeout_s,
                        label=f"segment:{entry.slide_name}",
                    )
                elif step == "patch":
                    _run_with_slide_timeout(
                        lambda: _run_patch(entry, config),
                        timeout_s=step_timeout_s,
                        label=f"patch:{entry.slide_name}",
                    )
                elif step == "extract":
                    extract_cb = None
                    if on_extract_progress:

                        def extract_cb(done: int, patch_total: int, _idx: int = emit_idx) -> None:
                            on_extract_progress(_idx, done, patch_total, total)

                    _run_with_slide_timeout(
                        lambda: _run_extract(
                            entry,
                            config,
                            device,
                            encoder,
                            progress_callback=extract_cb,
                        ),
                        timeout_s=step_timeout_s,
                        label=f"extract:{entry.slide_name}",
                    )
                else:
                    raise ValueError(f"Unsupported stage: {step}")
            except TimeoutError as exc:
                failed_errors[emit_idx] = str(exc)
                logger.error("Batch timed out for %s at stage %s: %s", entry.slide_name, step, exc)
                if on_item_done:
                    on_item_done(
                        {
                            "idx": emit_idx,
                            "slide_name": entry.slide_name,
                            "stem": _slide_stem(entry.slide_name),
                            "ok": False,
                            "error": (str(exc) or "")[:500],
                            "phase": step,
                        }
                    )
                # Leaked CUDA thread — stop further slides on this GPU.
                break
            except Exception as exc:
                failed_errors[emit_idx] = str(exc)
                logger.exception("Batch failed for %s at stage %s", entry.slide_name, step)
                if on_item_done:
                    on_item_done(
                        {
                            "idx": emit_idx,
                            "slide_name": entry.slide_name,
                            "stem": _slide_stem(entry.slide_name),
                            "ok": False,
                            "error": (str(exc) or "")[:500],
                            "phase": step,
                        }
                    )
                continue

            step_done += 1
            if on_progress:
                on_progress(step, step_done, total)
            if on_item_done:
                on_item_done(
                    {
                        "idx": emit_idx,
                        "slide_name": entry.slide_name,
                        "stem": _slide_stem(entry.slide_name),
                        "ok": True,
                        "error": None,
                        "phase": step,
                    }
                )

    parent_idxs = []
    for local_i, entry in enumerate(entries):
        raw_idx = getattr(entry, "idx", None)
        try:
            parent_idxs.append(int(raw_idx) if raw_idx is not None else int(local_i))
        except (TypeError, ValueError):
            parent_idxs.append(int(local_i))
    use_parent_idx = len(parent_idxs) == len(set(parent_idxs))
    for local_i, entry in enumerate(entries):
        emit_idx = parent_idxs[local_i] if use_parent_idx else int(local_i)
        if emit_idx in failed_errors:
            results.append(
                {
                    "slide_name": entry.slide_name,
                    "slide_key": entry.slide_key,
                    "slide_id": entry.slide_id,
                    "status": "failed",
                    "error": failed_errors[emit_idx],
                }
            )
        else:
            results.append(
                {
                    "slide_name": entry.slide_name,
                    "slide_key": entry.slide_key,
                    "slide_id": entry.slide_id,
                    "status": "ok",
                }
            )

    failed = len(failed_errors)
    if failed and failed >= total:
        exit_code = 1
    elif failed and stage != "all":
        exit_code = 1
    elif failed:
        exit_code = 0
    else:
        exit_code = 0
    logger.info("trident batch complete: %s/%s ok stage=%s", total - failed, total, stage)
    return exit_code, results


# Re-export for adapter prewarm
from trident.batch.cache import prewarm_models  # noqa: E402,F401
