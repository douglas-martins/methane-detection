"""Tests for src/data/preprocessing/normalize.py (stage 1: select + validate).

STARCOP folders already hold corrected, per-band TOA GeoTIFFs -- nothing to
atmospherically correct here. Normalization (BAND_NORMALIZATION) is a
train-time concern (see _vendor_starcop); this stage only selects the
configured bands, rejects NaN/Inf outright, and flags (without rejecting)
scenes whose values would fall outside their clip range once normalized.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import normalize


def _cfg(raw_root, processed_root, input_products, output_products):
    return SimpleNamespace(
        paths=SimpleNamespace(raw_root=str(raw_root), processed_root=str(processed_root)),
        dataset_cfg=SimpleNamespace(input_products=input_products, output_products=output_products),
    )


def test_copies_only_configured_bands_into_selected_folder(tmp_path, tiny_geotiff_factory):
    scene = tmp_path / "raw" / "scene1"
    band = np.array([[1.0, 1.0], [1.0, 1.0]], dtype="float32")
    tiny_geotiff_factory(scene / "TOA_AVIRIS_640nm.tif", band)
    tiny_geotiff_factory(scene / "TOA_AVIRIS_550nm.tif", band)
    tiny_geotiff_factory(scene / "extra_band.tif", band)  # not configured -- must be excluded
    tiny_geotiff_factory(scene / "labelbinary.tif", band)

    output = tmp_path / "selected" / "scene1"
    flagged = normalize.select_scene(
        scene,
        output,
        input_products=["TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm"],
        output_products=["labelbinary"],
    )

    assert sorted(p.name for p in output.iterdir()) == [
        "TOA_AVIRIS_550nm.tif",
        "TOA_AVIRIS_640nm.tif",
        "labelbinary.tif",
    ]
    assert flagged == []


def test_copied_bands_are_pixel_identical_to_source(tmp_path, tiny_geotiff_factory):
    import rasterio

    scene = tmp_path / "raw" / "scene1"
    band = np.array([[1.0, 2.5], [3.0, 4.25]], dtype="float32")
    tiny_geotiff_factory(scene / "TOA_AVIRIS_640nm.tif", band)

    output = tmp_path / "selected" / "scene1"
    normalize.select_scene(scene, output, input_products=["TOA_AVIRIS_640nm"], output_products=[])

    with rasterio.open(output / "TOA_AVIRIS_640nm.tif") as src:
        np.testing.assert_array_equal(src.read(1), band)


def test_raises_on_nan_in_a_configured_band(tmp_path, tiny_geotiff_factory):
    scene = tmp_path / "raw" / "scene1"
    band_with_nan = np.array([[1.0, np.nan]], dtype="float32")
    tiny_geotiff_factory(scene / "TOA_AVIRIS_640nm.tif", band_with_nan)

    with pytest.raises(ValueError, match="TOA_AVIRIS_640nm"):
        normalize.select_scene(
            scene,
            tmp_path / "selected" / "scene1",
            input_products=["TOA_AVIRIS_640nm"],
            output_products=[],
        )


def test_flags_scene_exceeding_band_normalization_clip_range(tmp_path, tiny_geotiff_factory):
    scene = tmp_path / "raw" / "scene1"
    # TOA_AVIRIS_640nm: offset=0, factor=60, clip=(0, 2) -> normalized 200/60=3.33 > 2
    over_range_band = np.array([[200.0, 1.0]], dtype="float32")
    tiny_geotiff_factory(scene / "TOA_AVIRIS_640nm.tif", over_range_band)

    output = tmp_path / "selected" / "scene1"
    flagged = normalize.select_scene(
        scene,
        output,
        input_products=["TOA_AVIRIS_640nm"],
        output_products=[],
    )

    assert flagged == ["TOA_AVIRIS_640nm"]
    assert (output / "TOA_AVIRIS_640nm.tif").exists()  # flagged, not rejected


def test_run_writes_range_check_json_only_for_flagged_scenes(tmp_path, tiny_geotiff_factory):
    raw_root = tmp_path / "raw"
    tiny_geotiff_factory(
        raw_root / "scene_ok" / "TOA_AVIRIS_640nm.tif", np.array([[1.0]], dtype="float32")
    )
    tiny_geotiff_factory(
        raw_root / "scene_flagged" / "TOA_AVIRIS_640nm.tif", np.array([[200.0]], dtype="float32")
    )
    processed_root = tmp_path / "processed"

    normalize.run(_cfg(raw_root, processed_root, ["TOA_AVIRIS_640nm"], []))

    report = json.loads((processed_root / "selected" / "range_check.json").read_text())
    assert report == {"scene_flagged": ["TOA_AVIRIS_640nm"]}
    assert (processed_root / "selected" / "scene_ok" / "TOA_AVIRIS_640nm.tif").exists()
    assert (processed_root / "selected" / "scene_flagged" / "TOA_AVIRIS_640nm.tif").exists()
