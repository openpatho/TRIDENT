from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

StageName = Literal["segment", "patch", "extract", "all", "segment_patch"]

ItemDoneCallback = Callable[[dict], None]
ProgressCallback = Callable[[str, int, int], None]
ExtractProgressCallback = Callable[[int, int, int, int], None]
TileProgressCallback = Callable[[int, int], None]
SlideResult = dict[str, Any]


@dataclass
class PipelineConfig:
    input_dir: str
    output_dir: str
    target_mag: int = 20
    patch_size: int = 256
    overlap: int = 0
    patch_encoder: str = "hoptimus0"
    segment_backend: str = "hest"
    seg_conf_thresh: float = 0.5
    seg_batch_size: int = 32
    seg_num_workers: int | None = None
    batch_size: int = 32
    gpu_id: int = -1
    custom_mpp: float | None = None
    remove_artifacts: bool = False
    remove_penmarks: bool = True
    features_h5_by_slide: dict[str, str] = field(default_factory=dict)
    segmenter_ckpt_path: str | None = None
    encoder_ckpt_path: str | None = None


@dataclass
class SlideEntry:
    slide_name: str
    local_input_name: str
    slide_key: str | None = None
    slide_id: str | None = None
    source_type: str | None = None
    include_polygons: list | None = None
    exclude_polygons: list | None = None
    bbox: dict | list | None = None
    idx: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_region_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "include_polygons": self.include_polygons,
            "exclude_polygons": self.exclude_polygons,
            "bbox": self.bbox,
            "slide_name": self.slide_name,
        }

    @classmethod
    def from_manifest_dict(cls, entry: dict, idx: int = 0) -> SlideEntry:
        name = str(entry.get("slide_name") or "").strip()
        local = str(entry.get("local_input_name") or name).strip() or name
        return cls(
            slide_name=name,
            local_input_name=local,
            slide_key=str(entry.get("slide_key") or "").strip() or None,
            slide_id=str(entry.get("slide_id") or "").strip() or None,
            source_type=str(entry.get("source_type") or "").strip() or None,
            include_polygons=entry.get("include_polygons"),
            exclude_polygons=entry.get("exclude_polygons"),
            bbox=entry.get("bbox"),
            idx=idx,
            extra={
                k: v
                for k, v in entry.items()
                if k
                not in {
                    "slide_name",
                    "local_input_name",
                    "slide_key",
                    "slide_id",
                    "source_type",
                    "include_polygons",
                    "exclude_polygons",
                    "bbox",
                }
            },
        )
