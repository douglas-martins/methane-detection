from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from dataset import PatchDataset, _load_dataset_config
from kornia.augmentation import AugmentationSequential
from omegaconf import OmegaConf


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

    def test_reads_only_the_declared_window_not_the_whole_file(
        self, tmp_path, tiny_geotiff_factory
    ):
        # Scene is larger than the declared patch window, with a distinctive
        # marker placed outside it -- reading the whole file instead of the
        # declared window (e.g. a dropped/None window) would leak it in.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path, size=16)
        bands["mag1c"][12, 12] = 1e9  # outside the 0:8, 0:8 window below
        _write_scene(tiny_geotiff_factory, folder, bands)
        row = _patch_row(folder, size=16)
        row["window_width"] = 8
        row["window_height"] = 8
        df = pd.DataFrame([row])

        item = PatchDataset(df, dataset="starcop_mini", augment=False)[0]

        assert item["input"].shape == (4, 8, 8)
        assert item["input"][0].max().item() < 1.0


class TestLoadDatasetConfig:
    def test_reads_from_the_exact_lowercase_configs_dataset_path(self, monkeypatch):
        # Assert the literal path components, not just that some file was
        # found -- a case-mismatched path (e.g. "CONFIGS") still resolves on
        # macOS's default case-insensitive filesystem, silently masking a bug
        # that would fail on a case-sensitive one.
        captured = {}

        def fake_load(path):
            captured["path"] = path
            return OmegaConf.create({"dataset_cfg": {"input_products": [], "output_products": []}})

        monkeypatch.setattr("dataset.OmegaConf.load", fake_load)

        _load_dataset_config("starcop_mini")

        assert captured["path"].parts[-3:] == ("configs", "dataset", "starcop_mini.yaml")


class TestPatchDatasetInit:
    def test_patches_df_index_is_reset(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)], index=[7])

        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        assert list(dataset.patches_df.index) == [0]
        assert "index" not in dataset.patches_df.columns


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

        from conftest import _find_repo_root_containing_configs

        repo_root = _find_repo_root_containing_configs(Path(__file__).resolve().parent)
        expected_config = __import__("omegaconf").OmegaConf.load(
            repo_root / "configs" / "dataset" / "starcop_mini.yaml"
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


class TestPatchDatasetCaching:
    def test_reads_the_same_index_from_disk_only_once(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        # PatchDataset otherwise re-reads and re-decodes the same patch from
        # disk on every access -- measured as the dominant per-epoch cost
        # (train.py's fit() docstring: ~8s/epoch on starcop_mini with no
        # caching). Three accesses to the same index must hit disk once.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[0]
        dataset[0]

        assert len(calls) == 1

    def test_different_indices_are_each_read_from_disk_once(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder), _patch_row(folder)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[1]
        dataset[0]
        dataset[1]

        assert len(calls) == 2

    def test_mutating_a_returned_tensor_does_not_leak_into_the_next_access(
        self, tmp_path, tiny_geotiff_factory
    ):
        # A cache that returns the same tensor object on every hit would let
        # one caller's in-place mutation corrupt every later read of that
        # index -- the cache must hand back an independent copy.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        first = dataset[0]
        first["input"].fill_(999.0)
        first["output"].fill_(999.0)

        second = dataset[0]

        assert second["input"].max().item() != 999.0
        assert second["output"].max().item() != 999.0


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

    def test_augmenter_includes_90_degree_rotation_not_only_flips(
        self, tmp_path, tiny_geotiff_factory
    ):
        # Horizontal + vertical flip alone can only reach 4 marker positions
        # (identity, hflip, vflip, hflip+vflip); a real 90-degree rotation
        # reaches 8. Distinguishes an augmenter missing RandomRotation90 from
        # one that has it -- the marker-stays-aligned test above can't, since
        # alignment holds regardless of which transforms are included.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path, size=8)
        bands["mag1c"][1, 5] = 1e9
        bands["labelbinary"][1, 5] = 1.0
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder, size=8)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=True)

        torch.manual_seed(0)
        positions = set()
        for _ in range(200):
            item = dataset[0]
            row, col = (item["output"][0] == 1).nonzero(as_tuple=True)
            positions.add((row.item(), col.item()))

        assert len(positions) > 4
