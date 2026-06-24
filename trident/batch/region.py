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


def _ring_from_flat_points(coords) -> list[tuple[float, float]]:
    ring: list[tuple[float, float]] = []
    for pt in coords or []:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            if isinstance(pt[0], (list, tuple)):
                continue
            ring.append((float(pt[0]), float(pt[1])))
    return ring


def _extract_polygon_rings(value) -> list[list[tuple[float, float]]]:
    """Match worker manifest polygon shapes (GeoJSON Polygon/MultiPolygon, flat rings)."""
    if not value:
        return []
    if isinstance(value, list):
        rings: list[list[tuple[float, float]]] = []
        for item in value:
            rings.extend(_extract_polygon_rings(item))
        return rings
    if not isinstance(value, dict):
        return []
    if value.get("type") == "FeatureCollection":
        return _extract_polygon_rings(value.get("features") or [])
    if value.get("type") == "Feature":
        return _extract_polygon_rings(value.get("geometry"))
    geom_type = str(value.get("type") or "").lower()
    coords = value.get("coordinates")
    if geom_type == "polygon" and isinstance(coords, list):
        outer = coords[0] if coords else []
        ring = _ring_from_flat_points(outer)
        return [ring] if len(ring) >= 3 else []
    if geom_type == "multipolygon" and isinstance(coords, list):
        rings = []
        for polygon in coords:
            outer = polygon[0] if polygon else []
            ring = _ring_from_flat_points(outer)
            if len(ring) >= 3:
                rings.append(ring)
        return rings
    points = value.get("points")
    if isinstance(points, list):
        ring = [
            (float(p.get("x")), float(p.get("y")))
            for p in points
            if isinstance(p, dict) and p.get("x") is not None and p.get("y") is not None
        ]
        return [ring] if len(ring) >= 3 else []
    ring_coords = coords or value.get("ring")
    if isinstance(ring_coords, list):
        ring = _ring_from_flat_points(ring_coords)
        return [ring] if len(ring) >= 3 else []
    return []


def _rings_from_polygons(polygons: list | None) -> list[list[tuple[float, float]]]:
    return _extract_polygon_rings(polygons)


def _parse_bbox(bbox_value) -> tuple[float, float, float, float] | None:
    if isinstance(bbox_value, dict):
        try:
            x = float(bbox_value.get("x", 0))
            y = float(bbox_value.get("y", 0))
            w = float(bbox_value.get("width", 0))
            h = float(bbox_value.get("height", 0))
            if w > 0 and h > 0:
                return (x, y, w, h)
        except (TypeError, ValueError):
            return None
    if isinstance(bbox_value, (list, tuple)) and len(bbox_value) >= 4:
        try:
            x = float(bbox_value[0])
            y = float(bbox_value[1])
            w = float(bbox_value[2])
            h = float(bbox_value[3])
            if w > 0 and h > 0:
                return (x, y, w, h)
        except (TypeError, ValueError):
            return None
    return None


def _union_ring_bounds(rings: list[list[tuple[float, float]]]):
    bounds = []
    for ring in rings or []:
        for x, y in ring:
            bounds.append((float(x), float(y)))
    if not bounds:
        return None
    xs = [b[0] for b in bounds]
    ys = [b[1] for b in bounds]
    return (min(xs), min(ys), max(xs), max(ys))


def _looks_normalized_01(bounds: tuple[float, float, float, float]) -> bool:
    tol = 1e-3
    minx, miny, maxx, maxy = bounds
    return minx >= -tol and miny >= -tol and maxx <= 1.0 + tol and maxy <= 1.0 + tol


def _looks_bbox_local(bounds: tuple[float, float, float, float], bbox: tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bounds
    _bx, _by, bw, bh = bbox
    tolerance = max(2.0, max(bw, bh) * 0.01)
    return minx >= -tolerance and miny >= -tolerance and maxx <= bw + tolerance and maxy <= bh + tolerance


def _fits_image_space(bounds: tuple[float, float, float, float], width: int, height: int) -> bool:
    minx, miny, maxx, maxy = bounds
    tolerance = max(4.0, max(width, height) * 0.001)
    return minx >= -tolerance and miny >= -tolerance and maxx <= width + tolerance and maxy <= height + tolerance


def _same_point(left: tuple[float, float], right: tuple[float, float], tolerance: float = 1e-9) -> bool:
    return abs(left[0] - right[0]) <= tolerance and abs(left[1] - right[1]) <= tolerance


def _sanitize_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    cleaned: list[tuple[float, float]] = []
    for point in ring or []:
        try:
            next_point = (float(point[0]), float(point[1]))
        except (TypeError, ValueError, IndexError):
            continue
        if cleaned and _same_point(cleaned[-1], next_point):
            continue
        cleaned.append(next_point)

    if len(cleaned) > 1 and _same_point(cleaned[0], cleaned[-1]):
        unique_points = cleaned[:-1]
    else:
        unique_points = cleaned

    if len(set(unique_points)) < 3:
        return None

    if not _same_point(cleaned[0], cleaned[-1]):
        cleaned.append(cleaned[0])
    return cleaned


def _project_rings_to_slide_pixels(
    rings: list[list[tuple[float, float]]],
    bbox: tuple[float, float, float, float] | None,
    slide_dimensions: tuple[int, int] | None,
) -> list[list[tuple[float, float]]]:
    if not rings:
        return rings
    bounds = _union_ring_bounds(rings)
    if bounds is None:
        return rings

    if _looks_normalized_01(bounds):
        if slide_dimensions is not None:
            sw, sh = slide_dimensions
            return [[(x * sw, y * sh) for (x, y) in ring] for ring in rings]
        logger.warning(
            "Region polygon looks normalized 0-1 but slide dimensions unavailable; using raw coords"
        )
        return rings

    if bbox is not None:
        bx, by, _bw, _bh = bbox
        if _looks_bbox_local(bounds, bbox):
            return [[(x + bx, y + by) for (x, y) in ring] for ring in rings]

    if slide_dimensions is not None and _fits_image_space(bounds, slide_dimensions[0], slide_dimensions[1]):
        return rings

    return rings


def _polygonal_geometry(geometry):
    if geometry is None or getattr(geometry, "is_empty", False):
        return None
    try:
        geom_type = geometry.geom_type
    except Exception:
        return None
    if geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    parts = [
        part
        for part in getattr(geometry, "geoms", [])
        if getattr(part, "geom_type", None) in ("Polygon", "MultiPolygon")
        and not getattr(part, "is_empty", True)
    ]
    if not parts:
        return None
    try:
        from shapely.ops import unary_union
        return unary_union(parts)
    except Exception:
        return parts[0]


def _make_valid_polygonal(geometry):
    if geometry is None or getattr(geometry, "is_empty", False):
        return None
    try:
        is_valid = bool(getattr(geometry, "is_valid", True))
    except Exception:
        is_valid = True
    if not is_valid:
        try:
            from shapely.validation import make_valid
            geometry = make_valid(geometry)
        except Exception:
            try:
                geometry = geometry.buffer(0)
            except Exception as exc:
                logger.warning("Failed to repair region polygon geometry: %s", exc)
                return None
    return _polygonal_geometry(geometry)


def _safe_polygon_union(rings: list[list[tuple[float, float]]]):
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union
    except Exception as exc:
        logger.warning("Region-mask geometry unavailable (missing shapely): %s", exc)
        return None

    polygons = []
    for ring in rings or []:
        sanitized = _sanitize_ring(ring)
        if not sanitized:
            continue
        try:
            polygon = _make_valid_polygonal(ShapelyPolygon(sanitized))
        except Exception as exc:
            logger.warning("Skipping invalid region polygon ring: %s", exc)
            continue
        if polygon is not None and not polygon.is_empty:
            polygons.append(polygon)

    if not polygons:
        return None

    try:
        return _make_valid_polygonal(unary_union(polygons))
    except Exception as exc:
        logger.warning("Failed to union region polygon rings: %s", exc)
        return None


def build_region_mask(entry: SlideEntry, slide_width: int, slide_height: int):
    if not is_region_entry(entry):
        return None
    include_raw = _rings_from_polygons(entry.include_polygons)
    exclude_raw = _rings_from_polygons(entry.exclude_polygons)
    if not include_raw and not exclude_raw:
        return None

    bbox = _parse_bbox(entry.bbox)
    slide_dims = (int(slide_width), int(slide_height))
    include = _project_rings_to_slide_pixels(include_raw, bbox, slide_dims)
    exclude = _project_rings_to_slide_pixels(exclude_raw, bbox, slide_dims)

    region_mask = _safe_polygon_union(include)
    if region_mask is None:
        return None

    if exclude:
        exclude_mask = _safe_polygon_union(exclude)
        if exclude_mask is not None:
            try:
                region_mask = _make_valid_polygonal(region_mask.difference(exclude_mask))
            except Exception as exc:
                logger.warning("Failed to subtract region exclude polygons: %s", exc)
                return None

    if region_mask is None or region_mask.is_empty:
        return None
    return region_mask


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
