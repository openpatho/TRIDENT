from __future__ import annotations

import logging

from trident.batch.types import PipelineConfig, StageName

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple, object] = {}


def _resolve_device(gpu_id: int) -> str:
    if gpu_id is not None and int(gpu_id) >= 0:
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda:{int(gpu_id)}"
        except ImportError:
            pass
    return "cpu"


def cached_patch_encoder(name: str, device: str, *, ckpt_path: str | None = None) -> object:
    from trident.patch_encoder_models.load import encoder_factory

    key = ("encoder", str(name or "hoptimus0").strip().lower(), str(device), ckpt_path or "")
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    kwargs = {}
    if ckpt_path:
        kwargs["weights_path"] = ckpt_path
    encoder = encoder_factory(name, **kwargs)
    _MODEL_CACHE[key] = encoder
    return encoder


def cached_segmenter(
    backend: str,
    conf_thresh: float,
    device: str,
    *,
    ckpt_path: str | None = None,
) -> object:
    from trident.segmentation_models.load import segmentation_model_factory

    key = (
        "segmenter",
        str(backend or "hest").lower(),
        round(float(conf_thresh), 4),
        str(device),
        ckpt_path or "",
    )
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model = segmentation_model_factory(
        model_name=str(backend or "hest"),
        confidence_thresh=float(conf_thresh),
    )
    _MODEL_CACHE[key] = model
    return model


def prewarm_models(config: PipelineConfig, stages: StageName | list[str] = "all") -> None:
    """Load models into the process-wide cache (segmenter and/or encoder)."""
    device = _resolve_device(config.gpu_id)
    stage_list = (
        ["segment", "patch", "extract"]
        if stages == "all"
        else ([stages] if isinstance(stages, str) else list(stages))
    )
    if "segment" in stage_list or "segment_patch" in stage_list:
        if str(config.segment_backend or "hest").lower() == "hest" and device.startswith("cuda"):
            cached_segmenter(
                config.segment_backend,
                config.seg_conf_thresh,
                device,
                ckpt_path=config.segmenter_ckpt_path,
            )
    if "extract" in stage_list:
        cached_patch_encoder(
            config.patch_encoder,
            device,
            ckpt_path=config.encoder_ckpt_path,
        )
