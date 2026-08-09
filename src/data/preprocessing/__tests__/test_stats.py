"""Tests for src/data/preprocessing/stats.py (stage 4: per-band mean/std/min/max).

Not used for model normalization -- STARCOP normalizes via the fixed
BAND_NORMALIZATION table at train time (see normalize.py). This feeds the
TASK-1.3 dataset report and the TASK-6.2 drift-detection baseline instead.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stats


def _dataframe_for_scenes(scene_folders: list) -> pd.DataFrame:
    # has_plume is read unconditionally by STARCOPDataset.__getitem__, even
    # when it isn't among the requested products.
    return pd.DataFrame(
        {"folder": [str(f) for f in scene_folders], "has_plume": [False] * len(scene_folders)}
    )


def test_computes_exact_mean_and_std_for_constant_band(tmp_path, tiny_geotiff_factory):
    scene = tmp_path / "scene1"
    tiny_geotiff_factory(scene / "bandA.tif", np.full((4, 4), 5.0, dtype="float32"))
    dataframe = _dataframe_for_scenes([scene])

    result = stats.compute_band_stats(dataframe, bands=["bandA"])

    assert result["bandA"]["mean"] == 5.0
    assert result["bandA"]["std"] == 0.0


def test_computes_correct_min_max_for_known_band(tmp_path, tiny_geotiff_factory):
    scene = tmp_path / "scene1"
    array = np.array([[0.0, 10.0], [5.0, 3.0]], dtype="float32")
    tiny_geotiff_factory(scene / "bandA.tif", array)
    dataframe = _dataframe_for_scenes([scene])

    result = stats.compute_band_stats(dataframe, bands=["bandA"])

    assert result["bandA"]["min"] == 0.0
    assert result["bandA"]["max"] == 10.0


def test_only_reports_configured_stats_bands(tmp_path, tiny_geotiff_factory):
    scene = tmp_path / "scene1"
    tiny_geotiff_factory(scene / "bandA.tif", np.full((2, 2), 1.0, dtype="float32"))
    tiny_geotiff_factory(scene / "bandB.tif", np.full((2, 2), 2.0, dtype="float32"))
    dataframe = _dataframe_for_scenes([scene])

    result = stats.compute_band_stats(dataframe, bands=["bandA"])

    assert list(result.keys()) == ["bandA"]


def test_aggregates_across_multiple_scenes(tmp_path, tiny_geotiff_factory):
    scene1, scene2 = tmp_path / "scene1", tmp_path / "scene2"
    tiny_geotiff_factory(scene1 / "bandA.tif", np.full((2, 2), 0.0, dtype="float32"))
    tiny_geotiff_factory(scene2 / "bandA.tif", np.full((2, 2), 10.0, dtype="float32"))
    dataframe = _dataframe_for_scenes([scene1, scene2])

    result = stats.compute_band_stats(dataframe, bands=["bandA"])

    assert result["bandA"]["mean"] == 5.0
    assert result["bandA"]["min"] == 0.0
    assert result["bandA"]["max"] == 10.0


def test_run_defaults_bands_to_input_products_when_stats_bands_unset(tmp_path, tiny_geotiff_factory):
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
    scene1, scene2 = tmp_path / "scene1", tmp_path / "scene2"
    tiny_geotiff_factory(scene1 / "labelbinary.tif", np.zeros((2, 2), dtype="float32"))
    tiny_geotiff_factory(scene2 / "labelbinary.tif", np.ones((2, 2), dtype="float32"))
    dataframe = _dataframe_for_scenes([scene1, scene2])

    result = stats.compute_class_distribution(dataframe, bands=["labelbinary"])

    assert result["labelbinary"]["positive_pixels"] == 4
    assert result["labelbinary"]["total_pixels"] == 8


def test_class_distribution_imbalance_ratio_is_none_when_no_positives(tmp_path, tiny_geotiff_factory):
    scene = tmp_path / "scene1"
    tiny_geotiff_factory(scene / "labelbinary.tif", np.zeros((2, 2), dtype="float32"))
    dataframe = _dataframe_for_scenes([scene])

    result = stats.compute_class_distribution(dataframe, bands=["labelbinary"])

    assert result["labelbinary"]["positive_pixels"] == 0
    assert result["labelbinary"]["imbalance_ratio"] is None


def test_compute_band_stats_matches_two_pass_result_for_larger_dataset(tmp_path, tiny_geotiff_factory):
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
