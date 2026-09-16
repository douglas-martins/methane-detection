import numpy as np
import pandas as pd
import pytest
import rasterio
from eda import (
    compute_patch_level_balance,
    read_patch_bands,
    rgb_composite,
    select_example_patches,
)
from rasterio.windows import Window


def _patches_df(has_plume: list[bool], frac_positives: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"patch_{i}" for i in range(len(has_plume))],
            "has_plume": has_plume,
            "frac_positives": frac_positives,
        }
    )


class TestComputePatchLevelBalance:
    def test_counts_positive_and_total(self):
        df = _patches_df([True, False, True, False, False], [0.1, 0.0, 0.02, 0.0, 0.0])
        result = compute_patch_level_balance(df)
        assert result == {"positive": 2, "total": 5, "positive_fraction": pytest.approx(0.4)}

    def test_all_negative_gives_zero_fraction(self):
        df = _patches_df([False, False], [0.0, 0.0])
        result = compute_patch_level_balance(df)
        assert result["positive"] == 0
        assert result["positive_fraction"] == 0.0

    def test_empty_dataframe_does_not_raise(self):
        df = _patches_df([], [])
        result = compute_patch_level_balance(df)
        assert result == {"positive": 0, "total": 0, "positive_fraction": 0.0}


class TestSelectExamplePatches:
    def test_returns_requested_counts(self):
        df = _patches_df(
            has_plume=[True, True, True, False, False, False],
            frac_positives=[0.05, 0.003, 0.2, 0.0, 0.0, 0.0],
        )
        selected = select_example_patches(df, n_positive=2, n_negative=2, seed=42)
        assert len(selected) == 4
        assert selected["has_plume"].sum() == 2
        assert (~selected["has_plume"]).sum() == 2

    def test_includes_the_faintest_plume_as_the_hard_example(self):
        df = _patches_df(
            has_plume=[True, True, True, False, False],
            frac_positives=[0.05, 0.003, 0.2, 0.0, 0.0],
        )
        selected = select_example_patches(df, n_positive=2, n_negative=1, seed=42)
        assert "patch_1" in selected["id"].to_numpy()  # lowest frac_positives among positives

    def test_deterministic_given_same_seed(self):
        df = _patches_df(
            has_plume=[True, True, True, True, False, False, False, False],
            frac_positives=[0.05, 0.003, 0.2, 0.09, 0.0, 0.0, 0.0, 0.0],
        )
        first = select_example_patches(df, n_positive=2, n_negative=2, seed=7)
        second = select_example_patches(df, n_positive=2, n_negative=2, seed=7)
        assert list(first["id"]) == list(second["id"])

    def test_raises_when_not_enough_positive_patches(self):
        df = _patches_df(has_plume=[True, False, False], frac_positives=[0.1, 0.0, 0.0])
        with pytest.raises(ValueError, match="positive"):
            select_example_patches(df, n_positive=2, n_negative=1, seed=42)

    def test_raises_when_not_enough_negative_patches(self):
        df = _patches_df(has_plume=[True, True, False], frac_positives=[0.1, 0.2, 0.0])
        with pytest.raises(ValueError, match="negative"):
            select_example_patches(df, n_positive=1, n_negative=2, seed=42)

    def test_does_not_raise_when_exactly_enough_positive_patches(self):
        df = _patches_df(has_plume=[True, True, False], frac_positives=[0.1, 0.2, 0.0])
        selected = select_example_patches(df, n_positive=2, n_negative=1, seed=42)
        assert selected["has_plume"].sum() == 2

    def test_does_not_raise_when_exactly_enough_negative_patches(self):
        df = _patches_df(has_plume=[True, False, False], frac_positives=[0.1, 0.0, 0.0])
        selected = select_example_patches(df, n_positive=1, n_negative=2, seed=42)
        assert (~selected["has_plume"]).sum() == 2

    def test_result_has_a_clean_contiguous_index(self):
        # hardest/extra_positive/chosen_negative each keep their original
        # (non-contiguous) index from patches_df; the concat must reset it.
        df = _patches_df(
            has_plume=[True, True, True, False, False, False],
            frac_positives=[0.05, 0.003, 0.2, 0.0, 0.0, 0.0],
        )
        selected = select_example_patches(df, n_positive=2, n_negative=2, seed=42)
        assert list(selected.index) == list(range(len(selected)))

    def test_sampling_uses_the_correct_count_and_seed_for_each_group(self, monkeypatch):
        # A spy on DataFrame.sample, rather than relying on output randomness,
        # deterministically verifies n_positive-1/n_negative and seed are
        # actually threaded through to both sample() calls.
        df = _patches_df(
            has_plume=[True, True, True, False, False, False],
            frac_positives=[0.05, 0.003, 0.2, 0.0, 0.0, 0.0],
        )
        captured = []
        original_sample = pd.DataFrame.sample

        def spy_sample(self, *args, **kwargs):
            captured.append(
                {
                    "n_rows": len(self),
                    "n": kwargs.get("n"),
                    "random_state": kwargs.get("random_state"),
                }
            )
            return original_sample(self, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "sample", spy_sample)

        select_example_patches(df, n_positive=3, n_negative=2, seed=99)

        # remaining_positive: 3 positives - 1 hardest = 2 rows, request n=2 (n_positive-1).
        assert {"n_rows": 2, "n": 2, "random_state": 99} in captured
        # negatives: 3 available, request n=2 (n_negative).
        assert {"n_rows": 3, "n": 2, "random_state": 99} in captured


class TestReadPatchBands:
    def test_reads_the_requested_window_from_each_band(self, tmp_path, tiny_geotiff_factory):
        scene = tmp_path / "scene"
        band_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
        band_b = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype="float32")
        tiny_geotiff_factory(scene / "band_a.tif", band_a)
        tiny_geotiff_factory(scene / "band_b.tif", band_b)

        window = Window(col_off=1, row_off=0, width=2, height=2)
        result = read_patch_bands(scene, window, ["band_a", "band_b"])

        assert set(result.keys()) == {"band_a", "band_b"}
        np.testing.assert_array_equal(result["band_a"], [[2.0, 3.0], [5.0, 6.0]])
        np.testing.assert_array_equal(result["band_b"], [[20.0, 30.0], [50.0, 60.0]])

    def test_missing_band_file_raises(self, tmp_path):
        scene = tmp_path / "scene"
        scene.mkdir()
        window = Window(col_off=0, row_off=0, width=1, height=1)
        with pytest.raises(rasterio.errors.RasterioIOError):
            read_patch_bands(scene, window, ["nonexistent_band"])


class TestRgbComposite:
    def test_stacks_three_bands_in_order(self):
        bands = {
            "r": np.array([[0.0, 10.0]], dtype="float32"),
            "g": np.array([[0.0, 20.0]], dtype="float32"),
            "b": np.array([[0.0, 30.0]], dtype="float32"),
        }
        composite = rgb_composite(bands, "r", "g", "b")
        assert composite.shape == (1, 2, 3)
        # each channel independently scaled to [0, 1] by its own min/max
        np.testing.assert_allclose(composite[0, 1], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(composite[0, 0], [0.0, 0.0, 0.0])

    def test_constant_band_does_not_produce_nan(self):
        bands = {
            "r": np.array([[5.0, 5.0]], dtype="float32"),
            "g": np.array([[0.0, 1.0]], dtype="float32"),
            "b": np.array([[0.0, 1.0]], dtype="float32"),
        }
        composite = rgb_composite(bands, "r", "g", "b")
        assert not np.isnan(composite).any()

    def test_scales_relative_to_the_bands_own_min_not_from_zero(self):
        # min=2, max=6 -> span=4: a min-blind formula (e.g. summing instead of
        # subtracting the min) would only be caught by a nonzero band_min.
        bands = {
            "r": np.array([[2.0, 6.0]], dtype="float32"),
            "g": np.array([[0.0, 1.0]], dtype="float32"),
            "b": np.array([[0.0, 1.0]], dtype="float32"),
        }
        composite = rgb_composite(bands, "r", "g", "b")
        np.testing.assert_allclose(composite[0, :, 0], [0.0, 1.0])

    def test_output_is_always_float32_regardless_of_input_dtype(self):
        # r is constant (zeros_like branch), g/b are not (scale branch) --
        # covers both branches' dtype cast in one composite array, since
        # np.stack would upcast the whole result if either returned float64.
        bands = {
            "r": np.array([[5.0, 5.0]], dtype="float64"),
            "g": np.array([[0.0, 1.0]], dtype="float64"),
            "b": np.array([[0.0, 1.0]], dtype="float64"),
        }
        composite = rgb_composite(bands, "r", "g", "b")
        assert composite.dtype == np.float32
