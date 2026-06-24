"""Region polygon injection tests."""

from __future__ import annotations

from unittest import mock

from trident.batch.region import attach_region_to_slide, is_region_entry
from trident.batch.types import SlideEntry


def test_is_region_entry_virtual_slide():
    entry = SlideEntry(
        slide_name="region.json",
        local_input_name="region.json",
        source_type="virtual_slide",
        include_polygons=[{"coordinates": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}],
    )
    assert is_region_entry(entry) is True


def test_attach_region_injects_gdf_contours():
    entry = SlideEntry(
        slide_name="region.json",
        local_input_name="region.json",
        source_type="virtual_slide",
        include_polygons=[{"coordinates": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}],
    )
    slide = mock.Mock()
    slide.width = 1000
    slide.height = 800
    slide.tissue_seg_path = "/tmp/mask.geojson"

    gdf = mock.Mock()
    gdf.set_crs.return_value = gdf
    gdf_mock = mock.Mock(return_value=gdf)
    with (
        mock.patch("trident.batch.region.build_region_mask") as build_mask,
        mock.patch("geopandas.GeoDataFrame", gdf_mock),
    ):
        poly = type("Poly", (), {"is_empty": False})()
        build_mask.return_value = poly
        count = attach_region_to_slide(slide, entry)

    assert count == 1
    assert slide.gdf_contours is gdf
    assert slide.tissue_seg_path is None
