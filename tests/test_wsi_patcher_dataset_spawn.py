"""WSIPatcherDataset must pickle cleanly for DataLoader spawn workers."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np

from trident.wsi_objects.WSIPatcherDataset import WSIPatcherDataset


class _FakeWSI:
    def __init__(self, slide_path="slide.svs", **kwargs):
        self.slide_path = slide_path
        self.mpp = kwargs.get("mpp", 0.5)
        self.custom_mpp_keys = kwargs.get("custom_mpp_keys")
        self.name = kwargs.get("name", "slide")
        self._tiles = {}

    def read_region(self, location, level, size, read_as="numpy"):
        key = (location, level, size, read_as)
        if key not in self._tiles:
            h, w = size[1], size[0]
            if read_as == "pil":
                from PIL import Image

                self._tiles[key] = Image.fromarray(
                    np.zeros((h, w, 3), dtype=np.uint8)
                )
            else:
                self._tiles[key] = np.zeros((h, w, 3), dtype=np.uint8)
        return self._tiles[key]


def _fake_patcher(*, n=3, pil=False):
    coords = np.array([[0, 0], [256, 0], [0, 256]], dtype=np.int64)[:n]
    return SimpleNamespace(
        wsi=_FakeWSI(),
        valid_coords=coords,
        level=0,
        patch_size_level=64,
        patch_size_target=32,
        pil=pil,
    )


def test_wsi_patcher_dataset_pickles_without_live_handle():
    ds = WSIPatcherDataset(_fake_patcher(), transform=None)
    ds._wsi = _FakeWSI()  # simulate parent-process handle
    blob = pickle.dumps(ds)
    restored = pickle.loads(blob)
    assert restored._wsi is None
    assert restored.slide_path == "slide.svs"
    assert len(restored) == 3
    assert restored.coords.shape == (3, 2)


def test_wsi_patcher_dataset_getitem_reopens_wsi(monkeypatch):
    opened = []

    class TrackingWSI(_FakeWSI):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            opened.append(self.slide_path)

        def read_region(self, location, level, size, read_as="numpy"):
            h, w = size[1], size[0]
            return np.zeros((h, w, 3), dtype=np.uint8)

    patcher = _fake_patcher(n=1, pil=False)
    # Parent-side WSI is a plain fake; worker reopen uses TrackingWSI.
    ds = WSIPatcherDataset(patcher, transform=None)
    ds._wsi_cls = TrackingWSI

    class _FakeCv2:
        @staticmethod
        def resize(tile, size):
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    monkeypatch.setattr("trident.wsi_objects.WSIPatcherDataset.cv2", _FakeCv2)
    tile, (x, y) = ds[0]
    assert opened == ["slide.svs"]
    assert (x, y) == (0, 0)
    assert tile.shape == (32, 32, 3)
