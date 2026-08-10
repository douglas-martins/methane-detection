"""Tests for src/data/preprocessing/patch_extract.py (stage 3: tiling).

vendor/starcop's tiled_dataframe (reused via _vendor_starcop, not
reimplemented) hardcodes a 512x512 base scene shape -- STARCOP's own raw
scenes are always exactly that size -- so these tests build a full 512x512
synthetic label raster rather than an arbitrarily small one. Confirmed
independently: create_windows((512, 512), window_size=(128, 128),
overlap=(64, 64), include_incomplete=False) yields exactly 49 windows, all
128x128, the first at (row_off=0, col_off=0).

tiled_dataframe itself hardcodes its has_plume rule to frac_positives >
10/64**2 -- not configurable. patch_scenes() reuses tiled_dataframe for the
expensive part (window creation + reading real label data to compute
frac_positives) and then recomputes has_plume itself against the
configured has_plume_threshold, so that value is actually configurable
end-to-end.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

import patch_extract

PATCH_SIZE = [128, 128]
OVERLAP = [64, 64]
DEFAULT_THRESHOLD = 10 / 64**2  # STARCOP's own default


def _scene_row(scene_id: str, folder) -> dict:
    """Build one full-scene (512x512, window covering it entirely) input row for patch_scenes()."""
    return {
        "id": scene_id,
        "name": scene_id,
        "folder": str(folder),
        "window_col_off": 0,
        "window_row_off": 0,
        "window_width": 512,
        "window_height": 512,
        # STARCOP's own CSVs always carry this column -- STARCOPDataset.__getitem__
        # reads it unconditionally, even before tiled_dataframe recomputes it below.
        "has_plume": False,
    }


def _first_patch(tiled: pd.DataFrame) -> pd.Series:
    """Return the top-left (0, 0) patch row from a tiled dataframe."""
    return tiled[(tiled["window_col_off"] == 0) & (tiled["window_row_off"] == 0)].iloc[0]


def test_patches_have_configured_size(tmp_path, tiny_geotiff_factory):
    """Tiling a 512x512 scene at size 128/overlap 64 yields 49 windows, all exactly 128x128."""
    scene_folder = tmp_path / "scene1"
    tiny_geotiff_factory(scene_folder / "labelbinary.tif", np.zeros((512, 512), dtype="float32"))
    dataframe = pd.DataFrame([_scene_row("scene1", scene_folder)])

    tiled = patch_extract.patch_scenes(
        dataframe,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        output_products=["labelbinary"],
        has_plume_threshold=DEFAULT_THRESHOLD,
    )

    assert (tiled["window_width"] == 128).all()
    assert (tiled["window_height"] == 128).all()
    assert len(tiled) == 49  # 7x7 windows tiling a 512x512 scene at size 128 / overlap 64


def test_has_plume_true_when_positive_fraction_exceeds_threshold(tmp_path, tiny_geotiff_factory):
    """A patch with enough positive label pixels is marked has_plume=True."""
    scene_folder = tmp_path / "scene1"
    label = np.zeros((512, 512), dtype="float32")
    label[0:20, 0:20] = 1.0  # 400/16384 = 0.0244 in the first patch, well above threshold
    tiny_geotiff_factory(scene_folder / "labelbinary.tif", label)
    dataframe = pd.DataFrame([_scene_row("scene1", scene_folder)])

    tiled = patch_extract.patch_scenes(
        dataframe,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        output_products=["labelbinary"],
        has_plume_threshold=DEFAULT_THRESHOLD,
    )

    assert bool(_first_patch(tiled)["has_plume"]) is True


def test_has_plume_false_when_positive_fraction_below_threshold(tmp_path, tiny_geotiff_factory):
    """A patch with too few positive label pixels is marked has_plume=False."""
    scene_folder = tmp_path / "scene1"
    label = np.zeros((512, 512), dtype="float32")
    label[0, 0] = 1.0  # 1/16384 = 0.00006 in the first patch, well below threshold
    tiny_geotiff_factory(scene_folder / "labelbinary.tif", label)
    dataframe = pd.DataFrame([_scene_row("scene1", scene_folder)])

    tiled = patch_extract.patch_scenes(
        dataframe,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        output_products=["labelbinary"],
        has_plume_threshold=DEFAULT_THRESHOLD,
    )

    assert bool(_first_patch(tiled)["has_plume"]) is False


def test_has_plume_threshold_is_actually_configurable(tmp_path, tiny_geotiff_factory):
    """A fraction that clears a low threshold but not a high one proves the
    configured has_plume_threshold overrides tiled_dataframe's hardcoded
    10/64**2 default, rather than being silently ignored."""
    scene_folder = tmp_path / "scene1"
    label = np.zeros((512, 512), dtype="float32")
    label[0:10, 0:10] = 1.0  # 100/16384 = 0.0061 in the first patch
    tiny_geotiff_factory(scene_folder / "labelbinary.tif", label)
    dataframe = pd.DataFrame([_scene_row("scene1", scene_folder)])

    lenient = patch_extract.patch_scenes(
        dataframe, patch_size=PATCH_SIZE, overlap=OVERLAP,
        output_products=["labelbinary"], has_plume_threshold=0.001,
    )
    strict = patch_extract.patch_scenes(
        dataframe, patch_size=PATCH_SIZE, overlap=OVERLAP,
        output_products=["labelbinary"], has_plume_threshold=0.5,
    )

    assert bool(_first_patch(lenient)["has_plume"]) is True
    assert bool(_first_patch(strict)["has_plume"]) is False


def test_run_passes_configured_num_workers_to_patch_scenes(tmp_path, monkeypatch):
    """num_workers only changes multiprocessing pool size, not patch_scenes's
    output -- there's no observable state difference to assert on instead."""
    processed_root = tmp_path / "processed"
    splits_root = processed_root / "splits"
    splits_root.mkdir(parents=True)
    for name in ["train", "val", "test"]:
        pd.DataFrame({"id": []}).to_csv(splits_root / f"{name}.csv", index=False)

    captured_num_workers = []

    def fake_patch_scenes(dataframe, patch_size, overlap, output_products, has_plume_threshold, num_workers=1):
        """Stand in for patch_scenes(): record num_workers instead of actually tiling."""
        captured_num_workers.append(num_workers)
        return pd.DataFrame(columns=["window"])

    monkeypatch.setattr(patch_extract, "patch_scenes", fake_patch_scenes)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(processed_root=str(processed_root)),
        patch=SimpleNamespace(
            size=PATCH_SIZE, overlap=OVERLAP, has_plume_threshold=DEFAULT_THRESHOLD, num_workers=4
        ),
        dataset_cfg=SimpleNamespace(output_products=["labelbinary"]),
    )

    patch_extract.run(cfg)

    assert captured_num_workers == [4, 4, 4]
