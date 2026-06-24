"""Tests for weight path resolution (no GPU required)."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from trident.weights import (
    hf_weights_download_allowed,
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


if __name__ == "__main__":
    unittest.main()
