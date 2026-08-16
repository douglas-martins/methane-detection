"""Tests for src/serving/band_baseline.py -- checked-in per-band training
baseline stats (docs/dataset_report.md section 5, starcop_raw) plus the
band-name lookup that fills the gap left by ModelModule only exposing a
channel *count* at runtime, not band identity (see
vendor/starcop/starcop/models/model_module.py's __init__).
"""

import band_baseline


class TestBandNamesForModel:
    def test_returns_known_band_list_for_mag1c_rgb(self):
        names = band_baseline.band_names_for_model("starcop-baseline-mag1c-rgb", 4)

        assert names == ["mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]

    def test_returns_known_band_list_for_mag1c_only(self):
        names = band_baseline.band_names_for_model("starcop-baseline-mag1c-only", 1)

        assert names == ["mag1c"]

    def test_falls_back_to_generic_names_for_an_unknown_model(self):
        names = band_baseline.band_names_for_model("some-future-model", 3)

        assert names == ["band_0", "band_1", "band_2"]

    def test_falls_back_to_generic_names_when_channel_count_mismatches_the_known_model(self):
        # A known model name but an unexpected channel count (e.g. the
        # registry now points MODEL_NAME at a retrained version with a
        # different band layout) -- trusting the stale known list would
        # silently mislabel bands, so fall back instead of guessing.
        names = band_baseline.band_names_for_model("starcop-baseline-mag1c-rgb", 2)

        assert names == ["band_0", "band_1"]


class TestBaselineForBand:
    def test_returns_stats_for_a_known_band(self):
        stats = band_baseline.baseline_for_band("mag1c")

        assert stats.mean == 34.40
        assert stats.std == 37.72

    def test_returns_none_for_an_unknown_band(self):
        assert band_baseline.baseline_for_band("band_0") is None
