"""Virtual-region polygon helpers (generic manifest metadata, no app names)."""

from __future__ import annotations

import logging
from typing import Any

from trident.batch.types import SlideEntry

logger = logging.getLogger(__name__)

_REGION_SOURCE_TYPES = frozenset(
    {
        "derived_region_dataset",
        "mixed_collection_virtual_slide",
        "virtual_slide",
        "virtual_region",
    }
)


def is_region_entry(entry: SlideEntry | dict | None) -> bool:
    if entry is None:
        return False
    if isinstance(entry, SlideEntry):
        source_type = str(entry.source_type or "").strip().lower()
        has_polys = bool(entry.include_polygons or entry.exclude_polygons)
    else:
        source_type = str(entry.get("source_type") or "").strip().lower()
        has_polys = bool(entry.get("include_polygons") or entry.get("exclude_polygons"))
    if source_type in _REGION_SOURCE_TYPES:
        return True
    return has_polys and source_type not in {"", "slide_collection", "slide"}


def _rings_from_polygons(polygons: list | None) -> list[list[tuple[float, float]]]:
    rings: list[list[tuple[float, float]]] = []
    for poly in polygons or []:
        if not isinstance(poly, dict):
            continue
        coords = poly.get("coordinates") or poly.get("ring") or poly.get("points")
        if not coords:
            continue
        ring: list[tuple[float, float]] = []
        for pt in coords:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                ring.append((float(pt[0]), float(pt[1])))
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def _project_ring(
    ring: list[tuple[float, float]],
    slide_width: int,
    slide_height: int,
) -> list[tuple[float, float]]:
    if not ring:
        return ring
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    if max(xs) <= 1.0 and max(ys) <= 1.0 and min(xs) >= 0.0 and min(ys) >= 0.0:
        return [(x * slide_width, y * slide_height) for x, y in ring]
    return ring


def build_region_mask(entry: SlideEntry, slide_width: int, slide_height: int):
    if not is_region_entry(entry):
        return None
    include = _rings_from_polygons(entry.include_polygons)
    exclude = _rings_from_polygons(entry.exclude_polygons)
    if not include and not exclude:
        return None
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        logger.warning("shapely unavailable; skipping region mask")
        return None

    include_polys = []
    for ring in include:
        projected = _project_ring(ring, slide_width, slide_height)
        if len(projected) >= 3:
            include_polys.append(Polygon(projected))
    if not include_polys:
        return None
    mask = unary_union(include_polys)
    if exclude:
        exclude_polys = []
        for ring in exclude:
            projected = _project_ring(ring, slide_width, slide_height)
            if len(projected) >= 3:
                exclude_polys.append(Polygon(projected))
        if exclude_polys:
            mask = mask.difference(unary_union(exclude_polys))
    if mask is None or mask.is_empty:
        return None
    return mask


def attach_region_to_slide(slide: Any, entry: SlideEntry) -> int:
    """Inject region polygon as ``slide.gdf_contours`` (no Fiona read_file)."""
    slide._lazy_initialize()
    width = int(slide.width or 0)
    height = int(slide.height or 0)
    if width <= 0 or height <= 0:
        return 0
    region_mask = build_region_mask(entry, width, height)
    if region_mask is None:
        return 0
    try:
        import geopandas as gpd
    except ImportError as exc:
        logger.warning("geopandas unavailable for region injection: %s", exc)
        return 0
    geometries = list(region_mask.geoms) if hasattr(region_mask, "geoms") else [region_mask]
    geometries = [g for g in geometries if g is not None and not getattr(g, "is_empty", False)]
    if not geometries:
        return 0
    gdf = gpd.GeoDataFrame({"geometry": geometries})
    try:
        gdf = gdf.set_crs("EPSG:3857")
    except Exception:
        pass
    slide.gdf_contours = gdf
    slide.tissue_seg_path = None
    return len(geometries)
