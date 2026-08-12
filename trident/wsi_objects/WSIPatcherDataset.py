"""Spawn-safe tile dataset for WSIPatcher-backed DataLoaders.

Holds only picklable state (slide path, coords, geometry). Each DataLoader
worker reopens the WSI backend so we never pickle live OpenSlide/ctypes handles
or fall back to unsafe fork-after-CUDA.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

try:
    from torch.utils.data import Dataset as _TorchDataset

    _DATASET_BASE = _TorchDataset
except Exception:  # pragma: no cover - unit tests without torch
    _DATASET_BASE = object


class WSIPatcherDataset(_DATASET_BASE):
    """Dataset from a WSI patcher to directly read tiles on a slide.

    Construction snapshots coords + geometry from ``patcher``; the live
    ``patcher.wsi`` handle is not retained across spawn boundaries.
    """

    def __init__(self, patcher, transform):
        wsi = patcher.wsi
        self.slide_path = str(getattr(wsi, "slide_path", "") or "")
        if not self.slide_path:
            raise ValueError("WSIPatcherDataset requires patcher.wsi.slide_path")

        # Keep the concrete WSI class so workers reopen the same backend.
        self._wsi_cls = type(wsi)
        self._mpp = getattr(wsi, "mpp", None)
        self._custom_mpp_keys = getattr(wsi, "custom_mpp_keys", None)
        self._wsi_name = getattr(wsi, "name", None)

        self.coords = np.asarray(patcher.valid_coords, dtype=np.int64).copy()
        self.level = int(patcher.level)
        self.patch_size_level = int(patcher.patch_size_level)
        self.patch_size_target = (
            int(patcher.patch_size_target)
            if getattr(patcher, "patch_size_target", None) is not None
            else None
        )
        self.pil = bool(getattr(patcher, "pil", False))
        self.transform = transform

        # Lazily opened per process — never pickled.
        self._wsi: Optional[Any] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_wsi"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._wsi = None

    def __len__(self):
        return int(len(self.coords))

    def _ensure_wsi(self):
        if self._wsi is not None:
            return
        kwargs = {
            "slide_path": self.slide_path,
            "lazy_init": False,
        }
        if self._wsi_name is not None:
            kwargs["name"] = self._wsi_name
        if self._mpp is not None:
            kwargs["mpp"] = self._mpp
        if self._custom_mpp_keys is not None:
            kwargs["custom_mpp_keys"] = self._custom_mpp_keys
        try:
            self._wsi = self._wsi_cls(**kwargs)
        except TypeError:
            # Some backends only accept slide_path + lazy_init.
            self._wsi = self._wsi_cls(self.slide_path, lazy_init=False)

    def __getitem__(self, index):
        self._ensure_wsi()
        x = int(self.coords[index, 0])
        y = int(self.coords[index, 1])
        tile = self._wsi.read_region(
            location=(x, y),
            level=self.level,
            size=(self.patch_size_level, self.patch_size_level),
            read_as="pil" if self.pil else "numpy",
        )
        if self.patch_size_target is not None:
            if self.pil:
                tile = tile.resize((self.patch_size_target, self.patch_size_target))
            else:
                tile = cv2.resize(tile, (self.patch_size_target, self.patch_size_target))[:, :, :3]

        if self.transform:
            tile = self.transform(tile)

        return tile, (x, y)
