from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from dataset import PatchDataset
from kornia.augmentation import AugmentationSequential


def _make_scene(tiny_geotiff_factory, tmp_path, name="scene", size=8):
    """Write a tiny 4-input-band + label scene, all-zero except where overridden."""
    folder = tmp_path / name
    bands = {
        "mag1c": np.zeros((size, size), dtype="float32"),
        "TOA_AVIRIS_640nm": np.zeros((size, size), dtype="float32"),
        "TOA_AVIRIS_550nm": np.zeros((size, size), dtype="float32"),
        "TOA_AVIRIS_460nm": np.zeros((size, size), dtype="float32"),
        "labelbinary": np.zeros((size, size), dtype="float32"),
    }
    return folder, bands


def _write_scene(tiny_geotiff_factory, folder, bands):
    for band_name, array in bands.items():
        tiny_geotiff_factory(folder / f"{band_name}.tif", array)


def _patch_row(folder: Path, size: int = 8) -> dict:
    return {
        "id": "patch_0",
        "name": "scene",
        "folder": str(folder),
        "window_col_off": 0,
        "window_row_off": 0,
        "window_width": size,
        "window_height": size,
        "has_plume": True,
    }


class TestPatchDatasetShapesAndDtypes:
    def test_input_and_output_have_the_documented_shapes(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        item = PatchDataset(df, dataset="starcop_mini", augment=False)[0]

        assert item["input"].shape == (4, 8, 8)
        assert item["input"].dtype == torch.float32
        assert item["output"].shape == (1, 8, 8)


class TestPatchDatasetBandOrder:
    def test_channel_order_matches_the_dataset_config(self, tmp_path, tiny_geotiff_factory):
        # Distinct normalized signature per band so order is unambiguous:
        # mag1c raw=175->0.1, 640nm raw=6->0.1 would collide, so vary raw
        # values per band to get a distinct mean per channel instead.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        bands["mag1c"][:] = 175.0  # -> 0.1 (offset 0, factor 1750)
        bands["TOA_AVIRIS_640nm"][:] = 12.0  # -> 0.2 (offset 0, factor 60)
        bands["TOA_AVIRIS_550nm"][:] = 18.0  # -> 0.3
        bands["TOA_AVIRIS_460nm"][:] = 24.0  # -> 0.4
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        item = PatchDataset(df, dataset="starcop_mini", augment=False)[0]

        expected_config = __import__("omegaconf").OmegaConf.load(
            Path(__file__).resolve().parents[3] / "configs" / "dataset" / "starcop_mini.yaml"
        )
        input_products = list(expected_config.dataset_cfg.input_products)
        expected_means = {
            "mag1c": 0.1,
            "TOA_AVIRIS_640nm": 0.2,
            "TOA_AVIRIS_550nm": 0.3,
            "TOA_AVIRIS_460nm": 0.4,
        }
        for channel_idx, product in enumerate(input_products):
            np.testing.assert_allclose(
                item["input"][channel_idx].mean().item(), expected_means[product], atol=1e-5
            )


class TestPatchDatasetLabelPassthrough:
    def test_label_is_returned_unnormalized(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        bands["labelbinary"][0, 0] = 1.0
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        item = PatchDataset(df, dataset="starcop_mini", augment=False)[0]

        assert set(item["output"].unique().tolist()) <= {0.0, 1.0}
        assert item["output"][0, 0, 0].item() == 1.0


class TestPatchDatasetAugmentation:
    def test_val_test_construction_has_no_augmenter(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        assert dataset.augmenter is None

    def test_train_construction_has_a_kornia_augmenter(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        dataset = PatchDataset(df, dataset="starcop_mini", augment=True)

        assert isinstance(dataset.augmenter, AugmentationSequential)

    def test_augmentation_moves_the_image_marker_with_the_mask(
        self, tmp_path, tiny_geotiff_factory
    ):
        # A single marked mask pixel and a matching extreme mag1c marker at
        # the same location: after any flip/90-degree rotation, the two
        # must still coincide -- proving image and mask transform together,
        # not independently.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path, size=8)
        bands["mag1c"][2, 5] = 1e9  # clips to 2.0 regardless of exact value
        bands["labelbinary"][2, 5] = 1.0
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder, size=8)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=True)

        torch.manual_seed(0)
        for _ in range(20):
            item = dataset[0]
            mask = item["output"][0]
            assert mask.sum().item() == 1.0  # exactly one positive pixel survives
            marker_row, marker_col = (mask == 1).nonzero(as_tuple=True)
            mag1c_channel = item["input"][0]
            assert mag1c_channel[marker_row, marker_col].item() == pytest.approx(2.0)
