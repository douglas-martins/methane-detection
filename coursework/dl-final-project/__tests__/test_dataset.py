from pathlib import Path

import numpy as np
import pandas as pd
import patch_cache
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


class TestEntryBytes:
    def test_computes_the_exact_combined_byte_size_of_input_and_output(self):
        input_tensor = torch.zeros((4, 8, 8), dtype=torch.float32)
        output_tensor = torch.zeros((1, 8, 8), dtype=torch.float32)

        assert PatchDataset._entry_bytes(input_tensor, output_tensor) == 4 * 8 * 8 * 4 + 8 * 8 * 4

    def test_scales_with_input_channel_count_not_just_output(self):
        # A +/- or */÷ mutant in the sum of the two tensors' byte counts
        # would still pass a same-shape check -- vary channel count between
        # the two tensors so input and output contribute different, and
        # individually verifiable, amounts.
        input_tensor = torch.zeros((2, 4, 4), dtype=torch.float32)
        output_tensor = torch.zeros((1, 4, 4), dtype=torch.float32)

        assert PatchDataset._entry_bytes(input_tensor, output_tensor) == 2 * 4 * 4 * 4 + 4 * 4 * 4


class TestPatchDatasetCacheEviction:
    # An 8x8, 4-input-channel + 1-output-channel float32 patch is exactly
    # 4*8*8*4 + 1*8*8*4 = 1280 bytes -- used below to size a budget that
    # fits exactly one cached entry.
    _ONE_PATCH_BYTES = 1280

    def test_starts_with_an_empty_cache_and_zero_bytes_used(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        assert dataset._cache == {}
        assert dataset._cache_bytes_used == 0

    def test_a_budget_exactly_equal_to_one_entry_still_caches_it(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        # Guards the store step's `entry_bytes <= max_cache_bytes` boundary:
        # a `<=`->`<` mutant would refuse to cache an entry exactly the size
        # of the budget, turning every access into a disk read.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])
        dataset = PatchDataset(
            df, dataset="starcop_mini", augment=False, max_cache_bytes=self._ONE_PATCH_BYTES
        )

        import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[0]

        assert len(calls) == 1

    def test_a_budget_exactly_equal_to_two_entries_fits_both_without_evicting(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        # Guards the eviction while-loop's `used + entry_bytes > budget`
        # boundary: a `>`->`>=` mutant would evict on an exact fit, which a
        # loose "some eviction happened" assertion wouldn't catch -- this
        # pins the exact read count for a budget sized to fit precisely two
        # entries and no more.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder), _patch_row(folder)])
        dataset = PatchDataset(
            df, dataset="starcop_mini", augment=False, max_cache_bytes=2 * self._ONE_PATCH_BYTES
        )

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

    def test_byte_accounting_stays_correct_across_repeated_evictions(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        # Guards the +=/-= accounting in both the eviction loop and the
        # store step: a wrong running total would drift after enough
        # evictions and either hold onto entries it should have dropped or
        # drop ones it should have kept. Uniform-size entries under-exercise
        # this: with every entry the same byte size, a wrong running total
        # still happens to trigger the same number of evictions per step
        # (the miscount and the correct count are both exact multiples of
        # one entry's size). A later entry *larger* than one slot forces a
        # single access to evict two differently-tracked amounts in a row,
        # which is where a dropped/duplicated accumulation actually shows.
        small_folder, small_bands = _make_scene(tiny_geotiff_factory, tmp_path, name="small")
        _write_scene(tiny_geotiff_factory, small_folder, small_bands)
        big_folder, big_bands = _make_scene(tiny_geotiff_factory, tmp_path, name="big", size=16)
        _write_scene(tiny_geotiff_factory, big_folder, big_bands)
        # 20*H*W bytes/entry (input: 4 channels, output: 1 -- see _entry_bytes);
        # a 10x10 window is 2000 bytes, bigger than one 8x8 (1280 byte) slot.
        medium_row = _patch_row(big_folder, size=16)
        medium_row["window_width"] = 10
        medium_row["window_height"] = 10
        df = pd.DataFrame(
            [_patch_row(small_folder), _patch_row(small_folder), _patch_row(small_folder)]
            + [medium_row]
        )
        # Budget fits exactly the three 8x8 entries (3 * 1280 = 3840); fitting
        # the 2000-byte medium entry afterwards needs two of them evicted,
        # not one -- correct running-total tracking keeps index 2 cached
        # through that, an `-=`-> `=`/`+=` accounting bug does not.
        dataset = PatchDataset(
            df, dataset="starcop_mini", augment=False, max_cache_bytes=3 * self._ONE_PATCH_BYTES
        )

        import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[1]
        dataset[2]
        dataset[3]  # medium entry -- must evict indices 0 and 1, but not 2
        # Probe the still-cached one first: re-probing an evicted index also
        # triggers a fresh store/evict cycle as a side effect (it's a miss),
        # so checking hit-before-miss avoids that clobbering the next probe.
        dataset[2]  # still cached -- must not re-read
        dataset[1]  # evicted -- must re-read

        assert len(calls) == 5

    def test_exceeding_the_byte_budget_evicts_the_least_recently_used_entry(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        # Unbounded caching is fine for starcop_mini's ~125 MB but grows
        # without limit on starcop_raw's 141k-patch train split (~45 GB/
        # worker) -- a budget with LRU eviction is what makes caching safe
        # to keep turned on at that scale. A budget that only fits one
        # entry means caching index 1 must evict index 0, so re-reading
        # index 0 afterwards must hit disk again.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder), _patch_row(folder)])
        dataset = PatchDataset(
            df, dataset="starcop_mini", augment=False, max_cache_bytes=self._ONE_PATCH_BYTES
        )

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

        assert len(calls) == 3

    def test_a_budget_that_fits_every_entry_never_evicts(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder), _patch_row(folder)])
        dataset = PatchDataset(
            df, dataset="starcop_mini", augment=False, max_cache_bytes=10 * self._ONE_PATCH_BYTES
        )

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

    def test_an_entry_larger_than_the_whole_budget_is_never_cached_but_still_returned(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=False, max_cache_bytes=1)

        import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        item = dataset[0]
        dataset[0]

        assert item["input"].shape == (4, 8, 8)
        assert len(calls) == 2

    def test_default_budget_keeps_starcop_minis_whole_dataset_cached(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        # Regression guard: the default budget must stay generous enough
        # that starcop_mini (392 train patches, ~125 MB total) still gets
        # today's zero-eviction, fully-cached behavior with no explicit
        # `max_cache_bytes` override.
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


class TestPatchDatasetOnDiskCache:
    def test_reads_from_the_cache_without_touching_disk(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        # starcop_raw's LRU RAM cache barely helps (see patch_cache.py's own
        # docstring) -- the on-disk cache is what actually removes disk I/O
        # at that scale. A hit must never call read_patch_bands at all.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder), _patch_row(folder)])
        inputs = np.stack([np.full((4, 8, 8), 0.25), np.full((4, 8, 8), 0.75)]).astype(np.float16)
        outputs = np.stack([np.zeros((1, 8, 8)), np.ones((1, 8, 8))]).astype(np.uint8)
        cache_dir = tmp_path / "cache"
        patch_cache.write(cache_dir, df, inputs, outputs)

        import dataset as dataset_module

        calls = []
        monkeypatch.setattr(
            dataset_module, "read_patch_bands", lambda *a, **k: calls.append(a) or {}
        )

        dataset = PatchDataset(df, dataset="starcop_mini", augment=False, cache_dir=cache_dir)
        first = dataset[0]
        second = dataset[1]

        assert calls == []
        assert first["input"].shape == (4, 8, 8)
        assert torch.allclose(first["input"], torch.full((4, 8, 8), 0.25))
        assert torch.allclose(second["output"], torch.ones(1, 8, 8))

    def test_raises_when_the_cache_does_not_match_the_given_dataframe(
        self, tmp_path, tiny_geotiff_factory
    ):
        # An explicitly-passed cache_dir is a contract, not a hint -- a
        # mismatched cache (built for different rows) must fail loudly
        # rather than silently serve wrong patches or silently fall back.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        built_for = pd.DataFrame([_patch_row(folder)])
        inputs = np.zeros((1, 4, 8, 8), dtype=np.float16)
        outputs = np.zeros((1, 1, 8, 8), dtype=np.uint8)
        cache_dir = tmp_path / "cache"
        patch_cache.write(cache_dir, built_for, inputs, outputs)

        mismatched = pd.DataFrame([_patch_row(folder), _patch_row(folder)])

        with pytest.raises(ValueError, match="cache"):
            PatchDataset(mismatched, dataset="starcop_mini", augment=False, cache_dir=cache_dir)

    def test_cache_dir_none_keeps_using_live_reads(self, tmp_path, tiny_geotiff_factory):
        # Regression guard: the default (no cache_dir) path must be
        # untouched by this feature -- every prior test in this file relies
        # on it, this just makes the guarantee explicit.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        item = PatchDataset(df, dataset="starcop_mini", augment=False, cache_dir=None)[0]

        assert item["input"].shape == (4, 8, 8)

    def test_augmentation_still_applies_fresh_on_a_cache_hit(self, tmp_path, tiny_geotiff_factory):
        # The on-disk cache stores pre-augmentation data (same contract as
        # the RAM cache) -- augmentation must still run per access, not get
        # baked into the cached arrays.
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path, size=8)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder, size=8)])
        inputs = np.zeros((1, 4, 8, 8), dtype=np.float16)
        outputs = np.zeros((1, 1, 8, 8), dtype=np.uint8)
        outputs[0, 0, 1, 5] = 1
        cache_dir = tmp_path / "cache"
        patch_cache.write(cache_dir, df, inputs, outputs)

        dataset = PatchDataset(df, dataset="starcop_mini", augment=True, cache_dir=cache_dir)

        torch.manual_seed(0)
        positions = set()
        for _ in range(200):
            item = dataset[0]
            row, col = (item["output"][0] == 1).nonzero(as_tuple=True)
            positions.add((row.item(), col.item()))

        assert len(positions) > 1
