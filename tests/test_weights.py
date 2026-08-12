"""Tests for weight path resolution (no GPU required)."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from trident.weights import (
    hf_weights_download_allowed,
    resolve_patch_encoder_weights_path,
    resolve_segmentation_weights_path,
)


class TestWeightsResolution(unittest.TestCase):
    def test_hf_download_disabled_by_default(self):
        env = os.environ.copy()
        env.pop("TRIDENT_ALLOW_HF_WEIGHTS_DOWNLOAD", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(hf_weights_download_allowed())

    def test_hest_resolves_env_ckpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "deeplabv3_seg_v4.ckpt")
            with open(ckpt, "wb") as fh:
                fh.write(b"x" * 64)
            with mock.patch.dict(os.environ, {"TRIDENT_HEST_CKPT": ckpt}, clear=False):
                self.assertEqual(resolve_segmentation_weights_path("hest"), ckpt)

    def test_hest_fails_without_ckpt_when_hf_disabled(self):
        env = {
            "TRIDENT_ALLOW_HF_WEIGHTS_DOWNLOAD": "0",
            "TRIDENT_HEST_CKPT": "",
            "HF_HOME": "",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("trident.weights.get_weights_path", return_value=""):
                with self.assertRaises(FileNotFoundError):
                    resolve_segmentation_weights_path("hest")

    def test_hest_hf_fallback_opt_in(self):
        with mock.patch.dict(os.environ, {"TRIDENT_ALLOW_HF_WEIGHTS_DOWNLOAD": "1"}, clear=False):
            self.assertTrue(hf_weights_download_allowed())
            with mock.patch("trident.weights.get_weights_path", return_value=""):
                self.assertEqual(resolve_segmentation_weights_path("hest"), "")

    def test_h0_mini_ignores_generic_encoder_ckpt(self):
        """Boot-default TRIDENT_ENCODER_CKPT must not poison h0-mini resolution."""
        with tempfile.TemporaryDirectory() as tmp:
            hopt = os.path.join(tmp, "pytorch_model.bin")
            with open(hopt, "wb") as fh:
                fh.write(b"x" * 64)
            env = {
                "TRIDENT_ENCODER_CKPT": hopt,
                "TRIDENT_H0_MINI_CKPT": "",
                "TRIDENT_ALLOW_HF_WEIGHTS_DOWNLOAD": "1",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("trident.weights.get_weights_path", return_value=""):
                    self.assertEqual(resolve_patch_encoder_weights_path("h0-mini"), "")

    def test_h0_mini_resolves_snapshot_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "config.json")
            with open(cfg, "w", encoding="utf-8") as fh:
                fh.write("{}")
            with mock.patch.dict(os.environ, {"TRIDENT_H0_MINI_CKPT": tmp}, clear=False):
                self.assertEqual(resolve_patch_encoder_weights_path("h0-mini"), tmp)


if __name__ == "__main__":
    unittest.main()