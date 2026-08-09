"""Tests for src/data/preprocessing/coordinates.py (stage 5: per-scene geographic coordinates)."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

import coordinates


def test_scene_centroid_latlon_matches_raster_bounds_centroid(tmp_path, tiny_geotiff_factory):
    """Returned (lat, lon) is the midpoint of the raster's own bounds, not an arbitrary corner."""
    band_path = tiny_geotiff_factory(tmp_path / "bandA.tif", np.zeros((4, 4), dtype="float32"))

    lat, lon = coordinates.scene_centroid_latlon(band_path)

    # tiny_geotiff_factory pins transform=from_origin(0, 0, 1, 1) -- a 4x4
    # raster therefore spans left=0/right=4/top=0/bottom=-4 in EPSG:4326.
    assert lat == pytest.approx(-2.0)
    assert lon == pytest.approx(2.0)


def test_scene_centroid_scales_with_raster_size(tmp_path, tiny_geotiff_factory):
    """A different raster shape moves the centroid accordingly -- not hardcoded to the 4x4 case above."""
    band_path = tiny_geotiff_factory(tmp_path / "bandA.tif", np.zeros((10, 6), dtype="float32"))

    lat, lon = coordinates.scene_centroid_latlon(band_path)

    assert lat == pytest.approx(-5.0)
    assert lon == pytest.approx(3.0)


def test_collect_scene_coordinates_one_row_per_scene(tmp_path, tiny_geotiff_factory):
    """Centroid correctness itself is covered by the scene_centroid_latlon tests above --
    this test is only about scene discovery: one row per scene folder, nothing dropped
    or duplicated."""
    selected_root = tmp_path / "selected"
    tiny_geotiff_factory(selected_root / "sceneA" / "mag1c.tif", np.zeros((4, 4), dtype="float32"))
    tiny_geotiff_factory(selected_root / "sceneB" / "mag1c.tif", np.zeros((4, 4), dtype="float32"))

    rows = coordinates.collect_scene_coordinates(selected_root, reference_band="mag1c")

    assert sorted(row["scene_id"] for row in rows) == ["sceneA", "sceneB"]


def test_collect_scene_coordinates_ignores_non_directory_entries(tmp_path, tiny_geotiff_factory):
    """selected_root also holds normalize.py's flat range_check.json/missing_scenes.json."""
    selected_root = tmp_path / "selected"
    tiny_geotiff_factory(selected_root / "sceneA" / "mag1c.tif", np.zeros((4, 4), dtype="float32"))
    (selected_root / "range_check.json").write_text("{}")

    rows = coordinates.collect_scene_coordinates(selected_root, reference_band="mag1c")

    assert [row["scene_id"] for row in rows] == ["sceneA"]


def test_run_writes_scene_coordinates_csv(tmp_path, tiny_geotiff_factory):
    """End-to-end: run() writes a real, parseable scene_coordinates.csv with correct values."""
    processed_root = tmp_path / "processed"
    tiny_geotiff_factory(
        processed_root / "selected" / "sceneA" / "mag1c.tif", np.zeros((4, 4), dtype="float32")
    )

    cfg = SimpleNamespace(
        paths=SimpleNamespace(processed_root=str(processed_root)),
        dataset_cfg=SimpleNamespace(input_products=["mag1c", "TOA_AVIRIS_640nm"]),
    )

    coordinates.run(cfg)

    csv_path = processed_root / "coordinates" / "scene_coordinates.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["scene_id"] == "sceneA"
    assert float(rows[0]["lat"]) == pytest.approx(-2.0)
    assert float(rows[0]["lon"]) == pytest.approx(2.0)
