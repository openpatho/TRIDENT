"""Fiona-free GeoJSON loaders for segmentation contours (Fiona >= 1.10 compat)."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _features_from_payload(payload: Any) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    payload_type = str(payload.get("type") or "").lower()
    if payload_type == "featurecollection":
        raw = payload.get("features") or []
        return [f for f in raw if isinstance(f, dict)]
    if payload_type == "feature" and payload.get("geometry"):
        return [payload]
    geom_types = {
        "polygon",
        "multipolygon",
        "linestring",
        "multilinestring",
        "point",
        "multipoint",
        "geometrycollection",
    }
    if payload_type in geom_types:
        return [{"type": "Feature", "properties": {}, "geometry": payload}]
    return []


def load_segmentation_geojson_gdf(path: str):
    """Load a segmentation GeoJSON file without ``geopandas.read_file`` / Fiona."""
    import geopandas as gpd

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    features = _features_from_payload(payload)
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    return gpd.GeoDataFrame.from_features(features, crs=None)
