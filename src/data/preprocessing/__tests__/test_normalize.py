"""Tests for src/data/preprocessing/normalize.py (stage 1: select + validate).

STARCOP folders already hold corrected, per-band TOA GeoTIFFs -- nothing to
atmospherically correct here. Normalization (BAND_NORMALIZATION) is a
train-time concern (see _vendor_starcop); this stage only selects the
configured bands, rejects NaN/Inf outright, and flags (without rejecting)
scenes whose values would fall outside their clip range once normalized.
"""

import json
from types import SimpleNamespace

import normalize
import numpy as np
import pandas as pd
import pytest


def _cfg(
    raw_root,
    processed_root,
    input_products,
    output_products,
    train_csv="train.csv",
    test_csv="test.csv",
):
    """Build a minimal Hydra-like config exposing only what normalize.run() reads."""
    return SimpleNamespace(
        paths=SimpleNamespace(raw_root=str(raw_root), processed_root=str(processed_root)),
        dataset_cfg=SimpleNamespace(
            input_products=input_products,
            output_products=output_products,
            train_csv=train_csv,
            test_csv=test_csv,
        ),
    )


def _write_manifest_csvs(raw_root, train_ids, test_ids=()):
    """Write train.csv/test.csv with just the `id` column normalize.run() reads."""
    pd.DataFrame({"id": list(train_ids)}).to_csv(raw_root / "train.csv", index=False)
    pd.DataFrame({"id": list(test_ids)}).to_csv(raw_root / "test.csv", index=False)


def test_copies_only_configured_bands_into_selected_folder(tmp_path, tiny_geotiff_factory):
    """Only the configured input_products/output_products bands are copied; extras are excluded."""
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
    """select_scene() copies bands byte-for-byte -- no resampling or dtype conversion."""
    import rasterio

    scene = tmp_path / "raw" / "scene1"
    band = np.array([[1.0, 2.5], [3.0, 4.25]], dtype="float32")
    tiny_geotiff_factory(scene / "TOA_AVIRIS_640nm.tif", band)

    output = tmp_path / "selected" / "scene1"
    normalize.select_scene(scene, output, input_products=["TOA_AVIRIS_640nm"], output_products=[])

    with rasterio.open(output / "TOA_AVIRIS_640nm.tif") as src:
        np.testing.assert_array_equal(src.read(1), band)


def test_raises_on_nan_in_a_configured_band(tmp_path, tiny_geotiff_factory):
    """A NaN in any configured band is a hard failure, not a flag-and-continue."""
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
    """A band exceeding its BAND_NORMALIZATION clip range is flagged but still copied,
    not rejected."""
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


def test_find_scene_folder_locates_flat_scene(tmp_path):
    """Mini-style layout: scene folder directly under raw_root."""
    raw_root = tmp_path / "raw"
    scene_folder = raw_root / "scene1"
    scene_folder.mkdir(parents=True)

    assert normalize.find_scene_folder(raw_root, "scene1") == scene_folder


def test_find_scene_folder_locates_nested_scene(tmp_path):
    """Raw-style layout: scene folder one subfolder level deep under raw_root."""
    raw_root = tmp_path / "raw"
    scene_folder = raw_root / "STARCOP_train_easy" / "scene1"
    scene_folder.mkdir(parents=True)

    assert normalize.find_scene_folder(raw_root, "scene1") == scene_folder


def test_find_scene_folder_ignores_non_directory_nested_match(tmp_path):
    """A stray file named exactly scene_id under a subfolder isn't a valid scene folder --
    glob('*/scene_id') matches files too, only the flat `direct` candidate filtered on is_dir()."""
    raw_root = tmp_path / "raw"
    subfolder = raw_root / "subfolderA"
    subfolder.mkdir(parents=True)
    (subfolder / "scene1").write_text("not a scene folder")

    with pytest.raises(FileNotFoundError, match="scene1"):
        normalize.find_scene_folder(raw_root, "scene1")


def test_find_scene_folder_raises_file_not_found_when_missing(tmp_path):
    """No folder anywhere under raw_root matches the scene id."""
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    with pytest.raises(FileNotFoundError, match="scene1"):
        normalize.find_scene_folder(raw_root, "scene1")


def test_find_scene_folder_raises_value_error_when_ambiguous(tmp_path):
    """The same scene id under two different nested subfolders is ambiguous, not a silent pick."""
    raw_root = tmp_path / "raw"
    (raw_root / "subfolderA" / "scene1").mkdir(parents=True)
    (raw_root / "subfolderB" / "scene1").mkdir(parents=True)

    with pytest.raises(ValueError, match="scene1"):
        normalize.find_scene_folder(raw_root, "scene1")


def test_find_scene_folder_raises_value_error_when_direct_and_nested_both_match(tmp_path):
    """A flat match short-circuiting before nested folders are even checked
    would silently hide a real duplicate instead of raising -- not just a
    nested-vs-nested case, see test above."""
    raw_root = tmp_path / "raw"
    (raw_root / "scene1").mkdir(parents=True)
    (raw_root / "subfolderA" / "scene1").mkdir(parents=True)

    with pytest.raises(ValueError, match="scene1"):
        normalize.find_scene_folder(raw_root, "scene1")


@pytest.mark.parametrize(
    "scene_id",
    [
        "",
        "/etc/passwd",
        "/",
        "foo/bar",
        "../../etc/passwd",
        "..",
        "a/../b",
    ],
)
def test_find_scene_folder_rejects_unsafe_scene_ids(tmp_path, scene_id):
    """scene_id comes straight from the manifest CSV and is joined into raw_root paths --
    absolute, multi-component, or traversal values must never reach path construction."""
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    with pytest.raises(ValueError, match="[Ss]cene id"):
        normalize.find_scene_folder(raw_root, scene_id)


def test_run_rejects_unsafe_scene_id_before_writing_to_selected_root(tmp_path):
    """A malicious/malformed manifest id must be rejected before selected_root/scene_id
    is ever constructed, not just before the raw_root lookup."""
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_manifest_csvs(raw_root, train_ids=["../../etc/passwd"])
    processed_root = tmp_path / "processed"

    with pytest.raises(ValueError, match="[Ss]cene id"):
        normalize.run(_cfg(raw_root, processed_root, ["TOA_AVIRIS_640nm"], []))


def test_run_writes_range_check_json_only_for_flagged_scenes(tmp_path, tiny_geotiff_factory):
    """End-to-end: run() writes range_check.json listing only the scenes/bands that were flagged."""
    raw_root = tmp_path / "raw"
    tiny_geotiff_factory(
        raw_root / "scene_ok" / "TOA_AVIRIS_640nm.tif", np.array([[1.0]], dtype="float32")
    )
    tiny_geotiff_factory(
        raw_root / "scene_flagged" / "TOA_AVIRIS_640nm.tif", np.array([[200.0]], dtype="float32")
    )
    _write_manifest_csvs(raw_root, train_ids=["scene_ok", "scene_flagged"])
    processed_root = tmp_path / "processed"

    normalize.run(_cfg(raw_root, processed_root, ["TOA_AVIRIS_640nm"], []))

    # exact filename/case and iterdir() (not .exists(), which is case-insensitive on macOS)
    # catch a mutated "SELECTED"/"RANGE_CHECK.JSON" literal that .exists() would miss.
    assert sorted(p.name for p in processed_root.iterdir()) == ["selected"]
    selected_root = processed_root / "selected"
    assert "range_check.json" in {p.name for p in selected_root.iterdir()}
    assert (selected_root / "range_check.json").read_text() == json.dumps(
        {"scene_flagged": ["TOA_AVIRIS_640nm"]}, indent=2
    )
    assert (selected_root / "scene_ok" / "TOA_AVIRIS_640nm.tif").exists()
    assert (selected_root / "scene_flagged" / "TOA_AVIRIS_640nm.tif").exists()


def test_run_discovers_scenes_from_nested_subfolders(tmp_path, tiny_geotiff_factory):
    """End-to-end: run() finds and processes a scene nested under a raw-style subfolder."""
    raw_root = tmp_path / "raw"
    tiny_geotiff_factory(
        raw_root / "subfolderA" / "scene1" / "TOA_AVIRIS_640nm.tif",
        np.array([[1.0]], dtype="float32"),
    )
    _write_manifest_csvs(raw_root, train_ids=["scene1"])
    processed_root = tmp_path / "processed"

    normalize.run(_cfg(raw_root, processed_root, ["TOA_AVIRIS_640nm"], []))

    assert (processed_root / "selected" / "scene1" / "TOA_AVIRIS_640nm.tif").exists()


def test_run_logs_missing_scenes_instead_of_crashing(tmp_path, tiny_geotiff_factory):
    """A manifest id with no matching folder anywhere is logged to missing_scenes.json,
    not fatal."""
    raw_root = tmp_path / "raw"
    tiny_geotiff_factory(
        raw_root / "scene_present" / "TOA_AVIRIS_640nm.tif", np.array([[1.0]], dtype="float32")
    )
    _write_manifest_csvs(raw_root, train_ids=["scene_present", "scene_missing"])
    processed_root = tmp_path / "processed"

    normalize.run(_cfg(raw_root, processed_root, ["TOA_AVIRIS_640nm"], []))

    selected_root = processed_root / "selected"
    # exact filename/case via iterdir() -- a mutated "MISSING_SCENES.JSON" literal would
    # still resolve through .read_text() on macOS's case-insensitive filesystem.
    assert "missing_scenes.json" in {p.name for p in selected_root.iterdir()}
    assert (selected_root / "missing_scenes.json").read_text() == json.dumps(
        ["scene_missing"], indent=2
    )
    assert (selected_root / "scene_present" / "TOA_AVIRIS_640nm.tif").exists()


def test_run_produces_byte_identical_output_across_two_runs(
    tmp_path, tiny_geotiff_factory, assert_trees_identical
):
    """DVC reproducibility: the same raw input run() twice (into separate output
    roots) must produce byte-identical selected/ trees -- not just equal JSON."""
    raw_root = tmp_path / "raw"
    tiny_geotiff_factory(
        raw_root / "scene1" / "TOA_AVIRIS_640nm.tif", np.array([[1.0, 2.5]], dtype="float32")
    )
    _write_manifest_csvs(raw_root, train_ids=["scene1"])

    processed_root_a = tmp_path / "processed_a"
    processed_root_b = tmp_path / "processed_b"
    normalize.run(_cfg(raw_root, processed_root_a, ["TOA_AVIRIS_640nm"], []))
    normalize.run(_cfg(raw_root, processed_root_b, ["TOA_AVIRIS_640nm"], []))

    assert_trees_identical(processed_root_a / "selected", processed_root_b / "selected")


def test_run_creates_deeply_nested_processed_root_when_all_scenes_missing(tmp_path):
    """When no scene is ever selected, selected_root's own parents must not exist yet either --
    the final mkdir(parents=True) is exercised end to end, not made a no-op by an
    already-existing directory from an earlier select_scene() call."""
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_manifest_csvs(raw_root, train_ids=["scene_missing"])
    processed_root = tmp_path / "not_yet_created" / "processed"
    assert not processed_root.parent.exists()

    normalize.run(_cfg(raw_root, processed_root, ["TOA_AVIRIS_640nm"], []))

    missing = json.loads((processed_root / "selected" / "missing_scenes.json").read_text())
    assert missing == ["scene_missing"]


def test_select_scene_is_idempotent_on_an_already_existing_output_folder(
    tmp_path, tiny_geotiff_factory
):
    """A second call into the same output_folder must not raise -- select_scene() may be
    invoked again for a scene whose folder a previous run already created."""
    scene = tmp_path / "raw" / "scene1"
    band = np.array([[1.0, 1.0]], dtype="float32")
    tiny_geotiff_factory(scene / "TOA_AVIRIS_640nm.tif", band)
    output = tmp_path / "selected" / "scene1"
    output.mkdir(parents=True)

    normalize.select_scene(scene, output, input_products=["TOA_AVIRIS_640nm"], output_products=[])


def test_select_scene_uses_the_configured_bands_offset_sign_correctly(
    tmp_path, tiny_geotiff_factory
):
    """offset must be subtracted, not added -- the only BAND_NORMALIZATION entry with a nonzero
    offset (ratio_wv3_B8_B8MLR_SanchezGarcia22_simplediv: offset=-0.5, clip=(-2.0, 2.0)) is the
    one place a sign flip is observable: 2.3 - (-0.5) = 2.8 is flagged, 2.3 + (-0.5) = 1.8 isn't.
    """
    band_name = "ratio_wv3_B8_B8MLR_SanchezGarcia22_simplediv"
    scene = tmp_path / "raw" / "scene1"
    tiny_geotiff_factory(scene / f"{band_name}.tif", np.array([[2.3]], dtype="float32"))

    flagged = normalize.select_scene(
        scene, tmp_path / "selected" / "scene1", input_products=[band_name], output_products=[]
    )

    assert flagged == [band_name]


def test_select_scene_does_not_flag_value_exactly_at_the_clip_min_boundary(
    tmp_path, tiny_geotiff_factory
):
    """normalized == clip_min is inside the accepted range -- the check is strict '<', not '<='."""
    scene = tmp_path / "raw" / "scene1"
    # TOA_AVIRIS_640nm: offset=0, factor=60, clip=(0, 2) -> normalized 0/60 == clip_min exactly.
    tiny_geotiff_factory(scene / "TOA_AVIRIS_640nm.tif", np.array([[0.0]], dtype="float32"))

    flagged = normalize.select_scene(
        scene,
        tmp_path / "selected" / "scene1",
        input_products=["TOA_AVIRIS_640nm"],
        output_products=[],
    )

    assert flagged == []


def test_select_scene_does_not_flag_value_exactly_at_the_clip_max_boundary(
    tmp_path, tiny_geotiff_factory
):
    """normalized == clip_max is inside the accepted range -- the check is strict '>', not '>='."""
    scene = tmp_path / "raw" / "scene1"
    # TOA_AVIRIS_640nm: offset=0, factor=60, clip=(0, 2) -> normalized 120/60 == clip_max exactly.
    tiny_geotiff_factory(scene / "TOA_AVIRIS_640nm.tif", np.array([[120.0]], dtype="float32"))

    flagged = normalize.select_scene(
        scene,
        tmp_path / "selected" / "scene1",
        input_products=["TOA_AVIRIS_640nm"],
        output_products=[],
    )

    assert flagged == []


def test_select_scene_validates_only_band_1_of_a_multi_band_tif(tmp_path):
    """select_scene() reads band 1 only -- a wildly out-of-range value in a second band must
    not affect the range check, matching STARCOP's one-variable-per-file convention."""
    import rasterio
    from rasterio.transform import from_origin

    scene = tmp_path / "raw" / "scene1"
    scene.mkdir(parents=True)
    path = scene / "TOA_AVIRIS_640nm.tif"
    # band 1 in range (1.0 / 60 << 2), band 2 wildly out of range if it were ever read.
    data = np.array([[[1.0]], [[1.0e6]]], dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=2,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data)

    flagged = normalize.select_scene(
        scene,
        tmp_path / "selected" / "scene1",
        input_products=["TOA_AVIRIS_640nm"],
        output_products=[],
    )

    assert flagged == []


def test_run_removes_stale_missing_scenes_json_when_no_longer_missing(
    tmp_path, tiny_geotiff_factory
):
    """A prior run's missing_scenes.json shouldn't linger and misreport once
    the underlying data is fixed and re-run -- e.g. run() invoked directly
    (not via `dvc repro`, which clears stage outputs itself)."""
    raw_root = tmp_path / "raw"
    tiny_geotiff_factory(
        raw_root / "scene_present" / "TOA_AVIRIS_640nm.tif", np.array([[1.0]], dtype="float32")
    )
    _write_manifest_csvs(raw_root, train_ids=["scene_present"])
    processed_root = tmp_path / "processed"
    stale_report = processed_root / "selected" / "missing_scenes.json"
    stale_report.parent.mkdir(parents=True)
    stale_report.write_text(json.dumps(["scene_that_used_to_be_missing"]))

    normalize.run(_cfg(raw_root, processed_root, ["TOA_AVIRIS_640nm"], []))

    assert not stale_report.exists()
