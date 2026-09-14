"""Tests for src/data/preprocessing/stats.py (stage 4: per-band mean/std/min/max).

Not used for model normalization -- STARCOP normalizes via the fixed
BAND_NORMALIZATION table at train time (see normalize.py). This feeds the
TASK-1.3 dataset report and the TASK-6.2 drift-detection baseline instead.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import stats


def _dataframe_for_scenes(scene_folders: list) -> pd.DataFrame:
    """Build a minimal STARCOPDataset-compatible dataframe, one row per scene folder."""
    # has_plume is read unconditionally by STARCOPDataset.__getitem__, even
    # when it isn't among the requested products.
    return pd.DataFrame(
        {"folder": [str(f) for f in scene_folders], "has_plume": [False] * len(scene_folders)}
    )


def test_computes_exact_mean_and_std_for_constant_band(tmp_path, tiny_geotiff_factory):
    """A constant-valued band has mean equal to that value and std exactly 0."""
    scene = tmp_path / "scene1"
    tiny_geotiff_factory(scene / "bandA.tif", np.full((4, 4), 5.0, dtype="float32"))
    dataframe = _dataframe_for_scenes([scene])

    result = stats.compute_band_stats(dataframe, bands=["bandA"])

    assert result["bandA"]["mean"] == 5.0
    assert result["bandA"]["std"] == 0.0


def test_compute_band_stats_stays_finite_for_values_whose_square_overflows_float32(
    tmp_path, tiny_geotiff_factory
):
    """sum_sq must accumulate in float64, not the array's native float32 -- float32
    overflows to inf for values whose square exceeds ~3.4e38, corrupting std into inf."""
    scene = tmp_path / "scene1"
    tiny_geotiff_factory(scene / "bandA.tif", np.full((2, 2), 2e19, dtype="float32"))
    dataframe = _dataframe_for_scenes([scene])

    result = stats.compute_band_stats(dataframe, bands=["bandA"])

    assert np.isfinite(result["bandA"]["std"])
    assert result["bandA"]["std"] == 0.0  # constant band -> zero variance despite the huge scale


def test_computes_correct_min_max_for_known_band(tmp_path, tiny_geotiff_factory):
    """min/max match the known extremes of a small, hand-picked pixel array."""
    scene = tmp_path / "scene1"
    array = np.array([[0.0, 10.0], [5.0, 3.0]], dtype="float32")
    tiny_geotiff_factory(scene / "bandA.tif", array)
    dataframe = _dataframe_for_scenes([scene])

    result = stats.compute_band_stats(dataframe, bands=["bandA"])

    assert result["bandA"]["min"] == 0.0
    assert result["bandA"]["max"] == 10.0


def test_only_reports_configured_stats_bands(tmp_path, tiny_geotiff_factory):
    """A band present on disk but not passed in `bands` is excluded from the result."""
    scene = tmp_path / "scene1"
    tiny_geotiff_factory(scene / "bandA.tif", np.full((2, 2), 1.0, dtype="float32"))
    tiny_geotiff_factory(scene / "bandB.tif", np.full((2, 2), 2.0, dtype="float32"))
    dataframe = _dataframe_for_scenes([scene])

    result = stats.compute_band_stats(dataframe, bands=["bandA"])

    assert list(result.keys()) == ["bandA"]


def test_aggregates_across_multiple_scenes(tmp_path, tiny_geotiff_factory):
    """Stats are pooled across all scenes' pixels, not computed per-scene and averaged."""
    scene1, scene2 = tmp_path / "scene1", tmp_path / "scene2"
    tiny_geotiff_factory(scene1 / "bandA.tif", np.full((2, 2), 0.0, dtype="float32"))
    tiny_geotiff_factory(scene2 / "bandA.tif", np.full((2, 2), 10.0, dtype="float32"))
    dataframe = _dataframe_for_scenes([scene1, scene2])

    result = stats.compute_band_stats(dataframe, bands=["bandA"])

    assert result["bandA"]["mean"] == 5.0
    assert result["bandA"]["min"] == 0.0
    assert result["bandA"]["max"] == 10.0


def test_run_defaults_bands_to_input_products_when_stats_bands_unset(
    tmp_path, tiny_geotiff_factory
):
    """End-to-end: with cfg.stats.bands=None, run() falls back to dataset_cfg.input_products."""
    processed_root = tmp_path / "processed"
    patches_root = processed_root / "patches"
    scene = tmp_path / "raw_scene"
    tiny_geotiff_factory(scene / "bandA.tif", np.full((4, 4), 1.0, dtype="float32"))
    # run() computes class distribution unconditionally alongside band stats, so
    # a labelbinary band must exist even though this test isn't about it --
    # see test_run_writes_class_distribution_json below for that behavior.
    tiny_geotiff_factory(scene / "labelbinary.tif", np.zeros((4, 4), dtype="float32"))

    patches_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "folder": [str(scene)],
            "window_col_off": [0],
            "window_row_off": [0],
            "window_width": [4],
            "window_height": [4],
            "has_plume": [False],
        }
    ).to_csv(patches_root / "train_tiled_4_4.csv", index=False)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(processed_root=str(processed_root)),
        patch=SimpleNamespace(size=[4, 4]),
        dataset_cfg=SimpleNamespace(input_products=["bandA"], output_products=["labelbinary"]),
        stats=SimpleNamespace(bands=None),
    )

    stats.run(cfg)

    band_result = json.loads((processed_root / "stats" / "band_stats.json").read_text())
    assert band_result["bandA"]["mean"] == 1.0


def test_run_writes_class_distribution_json(tmp_path, tiny_geotiff_factory):
    """End-to-end: run() writes class_distribution.json alongside band_stats.json."""
    processed_root = tmp_path / "processed"
    patches_root = processed_root / "patches"
    scene = tmp_path / "raw_scene"
    tiny_geotiff_factory(scene / "bandA.tif", np.full((4, 4), 1.0, dtype="float32"))
    label_array = np.zeros((4, 4), dtype="float32")
    label_array[0, 0] = 1.0
    label_array[1, 1] = 1.0
    tiny_geotiff_factory(scene / "labelbinary.tif", label_array)

    patches_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "folder": [str(scene)],
            "window_col_off": [0],
            "window_row_off": [0],
            "window_width": [4],
            "window_height": [4],
            "has_plume": [False],
        }
    ).to_csv(patches_root / "train_tiled_4_4.csv", index=False)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(processed_root=str(processed_root)),
        patch=SimpleNamespace(size=[4, 4]),
        dataset_cfg=SimpleNamespace(input_products=["bandA"], output_products=["labelbinary"]),
        stats=SimpleNamespace(bands=None),
    )

    stats.run(cfg)

    class_result = json.loads((processed_root / "stats" / "class_distribution.json").read_text())
    assert class_result["labelbinary"]["positive_pixels"] == 2
    assert class_result["labelbinary"]["total_pixels"] == 16


def test_class_distribution_counts_positive_and_background_pixels(tmp_path, tiny_geotiff_factory):
    """Positive/background/total pixel counts and derived fractions match a known label array."""
    scene = tmp_path / "scene1"
    array = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    tiny_geotiff_factory(scene / "labelbinary.tif", array)
    dataframe = _dataframe_for_scenes([scene])

    result = stats.compute_class_distribution(dataframe, bands=["labelbinary"])

    assert result["labelbinary"]["positive_pixels"] == 2
    assert result["labelbinary"]["background_pixels"] == 2
    assert result["labelbinary"]["total_pixels"] == 4
    assert result["labelbinary"]["positive_fraction"] == 0.5
    assert result["labelbinary"]["imbalance_ratio"] == 1.0


def test_class_distribution_aggregates_across_scenes(tmp_path, tiny_geotiff_factory):
    """Positive/total pixel counts are pooled (summed) across scenes, not overwritten by the
    last one -- both scenes contribute a nonzero positive count, so a stray assignment instead
    of an accumulation would report only the last scene's count (4), not the true total (6)."""
    scene1, scene2 = tmp_path / "scene1", tmp_path / "scene2"
    scene1_label = np.zeros((2, 2), dtype="float32")
    scene1_label[0, 0] = 1.0
    scene1_label[0, 1] = 1.0
    tiny_geotiff_factory(scene1 / "labelbinary.tif", scene1_label)
    tiny_geotiff_factory(scene2 / "labelbinary.tif", np.ones((2, 2), dtype="float32"))
    dataframe = _dataframe_for_scenes([scene1, scene2])

    result = stats.compute_class_distribution(dataframe, bands=["labelbinary"])

    assert result["labelbinary"]["positive_pixels"] == 6
    assert result["labelbinary"]["total_pixels"] == 8


def test_class_distribution_imbalance_ratio_is_none_when_no_positives(
    tmp_path, tiny_geotiff_factory
):
    """A band with zero positive pixels reports imbalance_ratio=None instead of dividing by zero."""
    scene = tmp_path / "scene1"
    tiny_geotiff_factory(scene / "labelbinary.tif", np.zeros((2, 2), dtype="float32"))
    dataframe = _dataframe_for_scenes([scene])

    result = stats.compute_class_distribution(dataframe, bands=["labelbinary"])

    assert result["labelbinary"]["positive_pixels"] == 0
    assert result["labelbinary"]["imbalance_ratio"] is None


def test_compute_band_stats_matches_two_pass_result_for_larger_dataset(
    tmp_path, tiny_geotiff_factory
):
    """Pinning test for the incremental (running-sum) rewrite of
    compute_band_stats(): at starcop_raw's scale, materializing every
    patch's every band array before reducing costs ~35GB of peak RAM (see
    starcop-raw-pipeline-plan.md), so the implementation switches to a
    running per-band total. This test proves that rewrite stays numerically
    equivalent to computing mean/std/min/max directly over all the raw
    arrays, for a dataset too large to plausibly get right by eyeballing a
    single constant-value scene."""
    rng = np.random.default_rng(42)
    scene_folders = []
    raw_arrays = []
    for i in range(60):
        array = rng.uniform(0.0, 100.0, size=(4, 4)).astype("float32")
        scene = tmp_path / f"scene{i}"
        tiny_geotiff_factory(scene / "bandA.tif", array)
        scene_folders.append(scene)
        raw_arrays.append(array)
    dataframe = _dataframe_for_scenes(scene_folders)
    all_values = np.concatenate([a.ravel() for a in raw_arrays])

    result = stats.compute_band_stats(dataframe, bands=["bandA"])

    assert result["bandA"]["mean"] == pytest.approx(float(np.mean(all_values)), rel=1e-6)
    assert result["bandA"]["std"] == pytest.approx(float(np.std(all_values)), rel=1e-6)
    assert result["bandA"]["min"] == pytest.approx(float(np.min(all_values)))
    assert result["bandA"]["max"] == pytest.approx(float(np.max(all_values)))


def test_run_only_reads_the_windowed_patch_region_not_the_full_scene(
    tmp_path, tiny_geotiff_factory
):
    """_load_patches_dataframe() must rebuild a real per-row rasterio Window -- a None window
    (or a differently-named column, which STARCOPDataset silently treats the same as missing
    and defaults to None) would read each band's full raster instead of just the configured
    patch, blending in pixels no patch was ever supposed to cover."""
    processed_root = tmp_path / "processed"
    patches_root = processed_root / "patches"
    scene = tmp_path / "raw_scene"

    # 4x4 top-left quadrant is 1.0; everything else (reachable only without windowing) is 100.0.
    band = np.full((8, 8), 100.0, dtype="float32")
    band[0:4, 0:4] = 1.0
    tiny_geotiff_factory(scene / "bandA.tif", band)
    tiny_geotiff_factory(scene / "labelbinary.tif", np.zeros((8, 8), dtype="float32"))

    patches_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "folder": [str(scene)],
            "window_col_off": [0],
            "window_row_off": [0],
            "window_width": [4],
            "window_height": [4],
            "has_plume": [False],
        }
    ).to_csv(patches_root / "train_tiled_4_4.csv", index=False)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(processed_root=str(processed_root)),
        patch=SimpleNamespace(size=[4, 4]),
        dataset_cfg=SimpleNamespace(input_products=["bandA"], output_products=["labelbinary"]),
        stats=SimpleNamespace(bands=None),
    )

    stats.run(cfg)

    band_result = json.loads((processed_root / "stats" / "band_stats.json").read_text())
    assert band_result["bandA"]["mean"] == 1.0
    assert band_result["bandA"]["max"] == 1.0


def test_run_reads_and_writes_exact_expected_paths_and_json_content(tmp_path, monkeypatch):
    """run() reads patches/train_tiled_{h}_{w}.csv (exact lowercase names) and writes
    stats/{band_stats,class_distribution}.json with indent=2 -- verified via the literal
    argument values captured at the call sites (not through the filesystem, which resolves
    a wrong-case path to the same file on a case-insensitive filesystem and would mask every
    one of these mutations, the same pattern already seen in normalize.py/patch_extract.py)."""
    processed_root = tmp_path / "processed"
    processed_root.mkdir()

    read_paths = []

    def fake_read_csv(path, *args, **kwargs):
        """Stand in for pd.read_csv(): record the exact path, skip touching disk."""
        read_paths.append(Path(path))
        return pd.DataFrame({"id": []})

    monkeypatch.setattr(stats.pd, "read_csv", fake_read_csv)

    band_calls = []

    def fake_compute_band_stats(dataframe, bands):
        """Stand in for compute_band_stats(): record the bands, return a fixed result."""
        band_calls.append(bands)
        return {"bandA": {"mean": 1.0}}

    monkeypatch.setattr(stats, "compute_band_stats", fake_compute_band_stats)

    def fake_compute_class_distribution(dataframe, bands):
        """Stand in for compute_class_distribution(): return a fixed result."""
        return {"labelbinary": {"positive_pixels": 1}}

    monkeypatch.setattr(stats, "compute_class_distribution", fake_compute_class_distribution)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(processed_root=str(processed_root)),
        patch=SimpleNamespace(size=[4, 8]),
        dataset_cfg=SimpleNamespace(input_products=["bandA"], output_products=["labelbinary"]),
        stats=SimpleNamespace(bands=None),
    )

    stats.run(cfg)

    assert [p.parent.name for p in read_paths] == ["patches"]
    assert [p.name for p in read_paths] == ["train_tiled_4_8.csv"]
    assert band_calls == [["bandA"]]

    stats_dir = processed_root / "stats"
    assert sorted(p.name for p in processed_root.iterdir()) == ["stats"]
    assert sorted(p.name for p in stats_dir.iterdir()) == [
        "band_stats.json",
        "class_distribution.json",
    ]
    assert (stats_dir / "band_stats.json").read_text() == json.dumps(
        {"bandA": {"mean": 1.0}}, indent=2
    )
    assert (stats_dir / "class_distribution.json").read_text() == json.dumps(
        {"labelbinary": {"positive_pixels": 1}}, indent=2
    )


def test_run_uses_explicit_stats_bands_instead_of_input_products_when_configured(
    tmp_path, monkeypatch
):
    """cfg.stats.bands, when set, must actually override dataset_cfg.input_products -- not be
    silently ignored in favor of always falling back to it."""
    processed_root = tmp_path / "processed"
    processed_root.mkdir()
    monkeypatch.setattr(stats.pd, "read_csv", lambda *a, **kw: pd.DataFrame({"id": []}))
    monkeypatch.setattr(
        stats, "compute_class_distribution", lambda dataframe, bands: {"labelbinary": {}}
    )
    band_calls = []
    monkeypatch.setattr(
        stats,
        "compute_band_stats",
        lambda dataframe, bands: band_calls.append(bands) or {"explicit_band": {}},
    )

    cfg = SimpleNamespace(
        paths=SimpleNamespace(processed_root=str(processed_root)),
        patch=SimpleNamespace(size=[4, 4]),
        dataset_cfg=SimpleNamespace(
            input_products=["different_band"], output_products=["labelbinary"]
        ),
        stats=SimpleNamespace(bands=["explicit_band"]),
    )

    stats.run(cfg)

    assert band_calls == [["explicit_band"]]


def test_run_creates_deeply_nested_stats_root_when_its_parent_is_missing(tmp_path, monkeypatch):
    """The final mkdir(parents=True) must create every missing intermediate directory, not
    just stats_root itself -- exercised with a processed_root whose own parent doesn't exist
    yet, unlike every test above where tmp_path already provides it."""
    processed_root = tmp_path / "not_yet_created" / "processed"
    assert not processed_root.parent.exists()
    monkeypatch.setattr(stats.pd, "read_csv", lambda *a, **kw: pd.DataFrame({"id": []}))
    monkeypatch.setattr(stats, "compute_band_stats", lambda dataframe, bands: {})
    monkeypatch.setattr(stats, "compute_class_distribution", lambda dataframe, bands: {})

    stats.run(
        SimpleNamespace(
            paths=SimpleNamespace(processed_root=str(processed_root)),
            patch=SimpleNamespace(size=[4, 4]),
            dataset_cfg=SimpleNamespace(input_products=["bandA"], output_products=["labelbinary"]),
            stats=SimpleNamespace(bands=None),
        )
    )

    assert (processed_root / "stats").is_dir()


def test_run_is_idempotent_when_stats_root_already_exists(tmp_path, monkeypatch):
    """A second run() over the same processed_root must not raise on the pre-existing
    stats_root -- the mkdir needs exist_ok=True, not just parents=True."""
    processed_root = tmp_path / "processed"
    (processed_root / "stats").mkdir(parents=True)
    monkeypatch.setattr(stats.pd, "read_csv", lambda *a, **kw: pd.DataFrame({"id": []}))
    monkeypatch.setattr(stats, "compute_band_stats", lambda dataframe, bands: {})
    monkeypatch.setattr(stats, "compute_class_distribution", lambda dataframe, bands: {})

    stats.run(
        SimpleNamespace(
            paths=SimpleNamespace(processed_root=str(processed_root)),
            patch=SimpleNamespace(size=[4, 4]),
            dataset_cfg=SimpleNamespace(input_products=["bandA"], output_products=["labelbinary"]),
            stats=SimpleNamespace(bands=None),
        )
    )


def test_run_produces_byte_identical_stats_across_two_runs(
    tmp_path, tiny_geotiff_factory, assert_trees_identical
):
    """DVC reproducibility: run() twice against the same patches (into separate
    output roots) must produce byte-identical stats/ trees."""
    scene = tmp_path / "raw_scene"
    tiny_geotiff_factory(scene / "bandA.tif", np.full((4, 4), 1.0, dtype="float32"))
    label_array = np.zeros((4, 4), dtype="float32")
    label_array[0, 0] = 1.0
    tiny_geotiff_factory(scene / "labelbinary.tif", label_array)

    def _cfg(processed_root):
        patches_root = processed_root / "patches"
        patches_root.mkdir(parents=True)
        pd.DataFrame(
            {
                "folder": [str(scene)],
                "window_col_off": [0],
                "window_row_off": [0],
                "window_width": [4],
                "window_height": [4],
                "has_plume": [False],
            }
        ).to_csv(patches_root / "train_tiled_4_4.csv", index=False)
        return SimpleNamespace(
            paths=SimpleNamespace(processed_root=str(processed_root)),
            patch=SimpleNamespace(size=[4, 4]),
            dataset_cfg=SimpleNamespace(input_products=["bandA"], output_products=["labelbinary"]),
            stats=SimpleNamespace(bands=None),
        )

    processed_root_a = tmp_path / "processed_a"
    processed_root_b = tmp_path / "processed_b"
    stats.run(_cfg(processed_root_a))
    stats.run(_cfg(processed_root_b))

    assert_trees_identical(processed_root_a / "stats", processed_root_b / "stats")
