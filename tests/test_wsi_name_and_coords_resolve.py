"""Stem names with dots must not be truncated; coords resolve finds hydrate aliases."""

from __future__ import annotations

import os
from os.path import splitext

from trident.batch.runner import _resolve_coords_path


def test_explicit_stem_must_not_use_splitext():
    """Regression: splitext('2025-01-01 12.10.32') wrongly yields '.32' extension."""
    stem = "2025-01-01 12.10.32"
    buggy_name, buggy_ext = splitext(stem)
    assert buggy_name == "2025-01-01 12.10"
    assert buggy_ext == ".32"
    # Fixed WSI.__init__ keeps the caller stem intact when name= is provided.
    kept_name = stem
    kept_ext = splitext(os.path.basename("/data/2025-01-01 12.10.32.ndpi"))[1]
    assert kept_name == "2025-01-01 12.10.32"
    assert kept_ext == ".ndpi"


def test_resolve_coords_path_falls_back_to_alternate_stem(tmp_path):
    save_coords = tmp_path / "20x_256px_0px_overlap"
    patches = save_coords / "patches"
    patches.mkdir(parents=True)
    alt = patches / "2025-01-01 12.10_patches.h5"
    alt.write_bytes(b"h5")
    resolved = _resolve_coords_path(str(save_coords), "2025-01-01 12.10.32")
    assert os.path.isfile(resolved)
    assert resolved.endswith("2025-01-01 12.10_patches.h5")


def test_resolve_coords_path_prefers_exact_stem(tmp_path):
    save_coords = tmp_path / "20x_256px_0px_overlap"
    patches = save_coords / "patches"
    patches.mkdir(parents=True)
    (patches / "slide_a_patches.h5").write_bytes(b"a")
    (patches / "slide_b_patches.h5").write_bytes(b"b")
    resolved = _resolve_coords_path(str(save_coords), "slide_b")
    assert resolved.endswith("slide_b_patches.h5")
