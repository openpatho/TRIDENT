"""Tests for Fiona-free GeoJSON loading."""

from __future__ import annotations

import json
from pathlib import Path

from trident.batch.geojson_io import _features_from_payload, load_segmentation_geojson_gdf


def test_features_from_payload_feature_collection() -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"label": "tissue"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                },
            }
        ],
    }
    features = _features_from_payload(payload)
    assert len(features) == 1
    assert features[0]["properties"]["label"] == "tissue"


def test_load_segmentation_geojson_gdf_invokes_from_features(tmp_path: Path, monkeypatch) -> None:
    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"label": "tissue"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                },
            }
        ],
    }
    path = tmp_path / "tissue_contours.geojson"
    path.write_text(json.dumps(geo), encoding="utf-8")

    captured: dict = {}

    class FakeGdf:
        @classmethod
        def from_features(cls, features, crs=None):
            captured["features"] = features
            captured["crs"] = crs
            return "gdf"

    import geopandas as gpd

    monkeypatch.setattr(gpd, "GeoDataFrame", FakeGdf)

    result = load_segmentation_geojson_gdf(str(path))
    assert result == "gdf"
    assert len(captured["features"]) == 1
    assert captured["crs"] is None
