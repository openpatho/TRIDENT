"""Tests for weight path resolution (no GPU / torch required)."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from trident.weights import (
    hf_weights_download_allowed,
    resolve_segmentation_weights_path,
)


def test_hf_download_disabled_by_default():
    env = os.environ.copy()
    env.pop("TRIDENT_ALLOW_HF_WEIGHTS_DOWNLOAD", None)
    with mock.patch.dict(os.environ, env, clear=True):
        assert hf_weights_download_allowed() is False


def test_hest_resolves_env_ckpt(tmp_path):
    ckpt = tmp_path / "deeplabv3_seg_v4.ckpt"
    ckpt.write_bytes(b"x" * 64)
    with mock.patch.dict(os.environ, {"TRIDENT_HEST_CKPT": str(ckpt)}, clear=False):
        assert resolve_segmentation_weights_path("hest") == str(ckpt)


def test_hest_fails_without_ckpt_when_hf_disabled():
    env = {
        "TRIDENT_ALLOW_HF_WEIGHTS_DOWNLOAD": "0",
        "TRIDENT_HEST_CKPT": "",
        "HF_HOME": "",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        with mock.patch("trident.weights.get_weights_path", return_value=""):
            with pytest.raises(FileNotFoundError):
                resolve_segmentation_weights_path("hest")


def test_hest_hf_fallback_opt_in():
    with mock.patch.dict(os.environ, {"TRIDENT_ALLOW_HF_WEIGHTS_DOWNLOAD": "1"}, clear=False):
        assert hf_weights_download_allowed() is True
        with mock.patch("trident.weights.get_weights_path", return_value=""):
            assert resolve_segmentation_weights_path("hest") == ""
