"""DataLoader start-method preference (spawn-first after CUDA).

Imports helpers without loading full WSI torch stack.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest import mock

# Stub heavy deps before loading WSI.py helpers via a filtered module.
_ROOT = Path(__file__).resolve().parents[1]


def _load_helpers():
    # Minimal stubs so we can exec just the helper functions.
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    fake_torch.inference_mode = lambda *a, **k: (lambda f: f)
    fake_torch.float32 = "float32"
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Module = object
    fake_torch.nn = fake_nn
    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn
    sys.modules.setdefault("torch.utils", types.ModuleType("torch.utils"))
    fake_data = types.ModuleType("torch.utils.data")
    fake_data.DataLoader = object
    sys.modules["torch.utils.data"] = fake_data

    for name in (
        "numpy",
        "tqdm",
        "geopandas",
        "shapely",
        "PIL",
        "cv2",
        "h5py",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    # Import the real helpers by reading source and extracting isn't ideal;
    # instead import WSI with more stubs for trident internals.
    seg_mod = types.ModuleType("trident.segmentation_models")
    seg_load = types.ModuleType("trident.segmentation_models.load")
    seg_load.SegmentationModel = object
    sys.modules.setdefault("trident.segmentation_models", seg_mod)
    sys.modules.setdefault("trident.segmentation_models.load", seg_load)

    patcher_mod = types.ModuleType("trident.wsi_objects.WSIPatcher")
    patcher_mod.WSIPatcher = object
    sys.modules.setdefault("trident.wsi_objects.WSIPatcher", patcher_mod)
    # star-import in WSI.py: from trident.wsi_objects.WSIPatcher import *
    patcher_mod.__all__ = []

    ds_mod = types.ModuleType("trident.wsi_objects.WSIPatcherDataset")
    ds_mod.WSIPatcherDataset = object
    sys.modules.setdefault("trident.wsi_objects.WSIPatcherDataset", ds_mod)

    io_mod = types.ModuleType("trident.IO")
    for fn in (
        "save_h5",
        "read_coords",
        "mask_to_gdf",
        "overlay_gdf_on_thumbnail",
        "get_num_workers",
        "coords_to_h5",
        "splitext",
    ):
        setattr(io_mod, fn, lambda *a, **k: None)
    sys.modules.setdefault("trident.IO", io_mod)

    if "trident.wsi_objects.WSI" in sys.modules:
        del sys.modules["trident.wsi_objects.WSI"]

    from trident.wsi_objects.WSI import (  # noqa: WPS433
        _dataloader_context_candidates,
        _preferred_dataloader_start_methods,
    )

    return _preferred_dataloader_start_methods, _dataloader_context_candidates


_preferred_dataloader_start_methods, _dataloader_context_candidates = _load_helpers()


def test_preferred_methods_auto_prefers_spawn_when_cuda():
    with mock.patch.dict(
        os.environ,
        {
            "TRIDENT_DATALOADER_START_METHOD": "auto",
            "TRIDENT_DATALOADER_ALLOW_FORK": "0",
        },
        clear=False,
    ), mock.patch("trident.wsi_objects.WSI.torch.cuda.is_available", return_value=True):
        assert _preferred_dataloader_start_methods() == ["spawn"]


def test_preferred_methods_allow_fork_when_opt_in():
    with mock.patch.dict(
        os.environ,
        {
            "TRIDENT_DATALOADER_START_METHOD": "auto",
            "TRIDENT_DATALOADER_ALLOW_FORK": "1",
        },
        clear=False,
    ), mock.patch("trident.wsi_objects.WSI.torch.cuda.is_available", return_value=True):
        assert _preferred_dataloader_start_methods() == ["spawn", "fork"]


def test_preferred_methods_explicit_spawn_keeps_fork_fallback():
    with mock.patch.dict(
        os.environ,
        {
            "TRIDENT_DATALOADER_START_METHOD": "spawn",
            "TRIDENT_DATALOADER_ALLOW_FORK": "1",
        },
        clear=False,
    ):
        assert _preferred_dataloader_start_methods() == ["spawn", "fork"]


def test_preferred_methods_explicit_fork_requires_allow():
    with mock.patch.dict(
        os.environ,
        {
            "TRIDENT_DATALOADER_START_METHOD": "fork",
            "TRIDENT_DATALOADER_ALLOW_FORK": "0",
        },
        clear=False,
    ):
        assert _preferred_dataloader_start_methods() == []


def test_candidates_end_with_none_fallback():
    with mock.patch.dict(
        os.environ,
        {
            "TRIDENT_DATALOADER_START_METHOD": "spawn",
            "TRIDENT_DATALOADER_ALLOW_FORK": "0",
        },
        clear=False,
    ):
        cands = _dataloader_context_candidates(4)
        assert cands[-1] is None
