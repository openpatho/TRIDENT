"""Resolve model checkpoint paths from env, config, or registry (HF opt-in only)."""

from __future__ import annotations

import os
from typing import Optional

from trident.IO import get_weights_path

HEST_CKPT_FILENAME = "deeplabv3_seg_v4.ckpt"
HOPTIMUS0_WEIGHTS_FILENAME = "pytorch_model.bin"

_ENV_SEG = {
    "hest": ("TRIDENT_HEST_CKPT",),
    "grandqc": ("TRIDENT_GRANDQC_CKPT",),
    "grandqc_artifact": ("TRIDENT_GRANDQC_ARTIFACT_CKPT", "TRIDENT_GRANDQC_CKPT"),
}

_ENV_PATCH = {
    "hoptimus0": ("TRIDENT_HOPTIMUS0_CKPT", "TRIDENT_ENCODER_CKPT"),
    "hoptimus1": ("TRIDENT_HOPTIMUS1_CKPT", "TRIDENT_ENCODER_CKPT"),
}


def hf_weights_download_allowed() -> bool:
    raw = os.environ.get("TRIDENT_ALLOW_HF_WEIGHTS_DOWNLOAD", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return False


def _first_existing_file(*candidates: str | None) -> str:
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return ""


def _export_dir_weights(encoder_name: str, filename: str) -> str:
    hf_home = os.environ.get("HF_HOME", "").strip()
    if not hf_home:
        return ""
    direct = os.path.join(hf_home, "export", encoder_name, filename)
    if os.path.isfile(direct):
        return direct
    export_dir = os.path.join(hf_home, "export", encoder_name)
    if not os.path.isdir(export_dir):
        return ""
    for root, _dirs, files in os.walk(export_dir):
        if filename in files:
            return os.path.join(root, filename)
    return ""


def resolve_segmentation_weights_path(
    model_key: str,
    *,
    explicit_path: str | None = None,
) -> str:
    """Resolve a segmentation checkpoint path without HuggingFace unless opted in."""
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f"Segmentation checkpoint not found: {explicit_path}")
        return explicit_path

    for env_key in _ENV_SEG.get(model_key, ()):
        found = _first_existing_file(os.environ.get(env_key, "").strip() or None)
        if found:
            return found

    if model_key == "hest":
        found = _first_existing_file(_export_dir_weights("hest", HEST_CKPT_FILENAME))
        if found:
            return found

    registry_path = get_weights_path("seg", model_key)
    if registry_path and os.path.isfile(registry_path):
        return registry_path

    if hf_weights_download_allowed():
        return ""

    raise FileNotFoundError(
        f"Segmentation checkpoint for '{model_key}' not found. "
        f"Set TRIDENT_{model_key.upper()}_CKPT or populate segmentation_models/local_ckpts.json. "
        "Platform workers should sync weights from S3 at boot."
    )


def resolve_patch_encoder_weights_path(
    encoder_name: str,
    *,
    explicit_path: str | None = None,
) -> str:
    """Resolve patch encoder weights; empty string means caller may use HF hub."""
    if explicit_path:
        if os.path.isfile(explicit_path) or os.path.isdir(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Patch encoder checkpoint not found: {explicit_path}")

    enc = str(encoder_name or "").strip().lower()
    for env_key in _ENV_PATCH.get(enc, ()) + ("TRIDENT_ENCODER_CKPT",):
        found = _first_existing_file(os.environ.get(env_key, "").strip() or None)
        if found:
            return found

    if enc == "hoptimus0":
        found = _first_existing_file(_export_dir_weights("hoptimus0", HOPTIMUS0_WEIGHTS_FILENAME))
        if found:
            return found

    registry_path = get_weights_path("patch", enc)
    if registry_path and (os.path.isfile(registry_path) or os.path.isdir(registry_path)):
        return registry_path

    if hf_weights_download_allowed():
        return ""

    raise FileNotFoundError(
        f"Patch encoder weights for '{enc}' not found. "
        f"Set TRIDENT_{enc.upper()}_CKPT or TRIDENT_ENCODER_CKPT after S3 sync."
    )
