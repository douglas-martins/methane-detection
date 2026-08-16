"""Tests for src/serving/drift.py -- rolling per-band statistics and
Gaussian KL divergence against the training baseline (TASK-6.2). Pure
math, no I/O, no mocking -- real deques and real float arithmetic
throughout, matching this repo's established inference.py/metrics_ext.py
pattern.
"""

import math
from collections import deque

import drift
from band_baseline import BandStats


class TestUpdateRollingStats:
    def test_returns_mean_and_std_of_window_contents(self):
        window = deque([1.0, 2.0, 3.0], maxlen=100)

        stats = drift.update_rolling_stats(window, 4.0)

        assert stats.mean == 2.5
        assert math.isclose(stats.std, 1.118033988749895)

    def test_appends_the_new_value_to_the_window(self):
        window = deque([1.0, 2.0], maxlen=100)

        drift.update_rolling_stats(window, 3.0)

        assert list(window) == [1.0, 2.0, 3.0]

    def test_window_maxlen_evicts_the_oldest_value(self):
        window = deque([1.0, 2.0], maxlen=2)

        drift.update_rolling_stats(window, 3.0)

        assert list(window) == [2.0, 3.0]

    def test_a_single_value_window_has_zero_std(self):
        window = deque(maxlen=100)

        stats = drift.update_rolling_stats(window, 5.0)

        assert stats.mean == 5.0
        assert stats.std == 0.0


class TestKlDivergenceGaussian:
    def test_zero_when_distributions_are_identical(self):
        result = drift.kl_divergence_gaussian(mu_p=0.0, sigma_p=1.0, mu_q=0.0, sigma_q=1.0)

        assert math.isclose(result, 0.0, abs_tol=1e-12)

    def test_matches_the_closed_form_value_for_a_mean_shift(self):
        # KL(N(1,1) || N(0,1)) = ln(1/1) + (1 + 1)/(2*1) - 0.5 = 0.5
        result = drift.kl_divergence_gaussian(mu_p=1.0, sigma_p=1.0, mu_q=0.0, sigma_q=1.0)

        assert math.isclose(result, 0.5)

    def test_matches_the_closed_form_value_for_a_variance_shift(self):
        # KL(N(0,4) || N(0,1)) = ln(1/2) + (4 + 0)/2 - 0.5 = -ln(2) + 1.5
        result = drift.kl_divergence_gaussian(mu_p=0.0, sigma_p=2.0, mu_q=0.0, sigma_q=1.0)

        assert math.isclose(result, -math.log(2) + 1.5)

    def test_does_not_raise_or_return_inf_when_rolling_sigma_is_zero(self):
        # A short rolling window of literally identical values (e.g. a
        # handful of duplicate synthetic requests) computes sigma_p=0.0 --
        # the naive closed form's ln(sigma_q/sigma_p) would divide by
        # zero. Should clamp instead of crashing, and still report a
        # large (clearly anomalous), finite divergence.
        result = drift.kl_divergence_gaussian(mu_p=5.0, sigma_p=0.0, mu_q=0.0, sigma_q=1.0)

        assert math.isfinite(result)
        assert result > 10.0


class TestCheckDrift:
    def test_true_when_kl_divergence_exceeds_the_threshold(self):
        rolling = BandStats(mean=5.0, std=1.0)
        baseline = BandStats(mean=0.0, std=1.0)

        assert drift.check_drift(rolling, baseline, threshold=0.5) is True

    def test_false_when_kl_divergence_is_within_the_threshold(self):
        rolling = BandStats(mean=0.01, std=1.0)
        baseline = BandStats(mean=0.0, std=1.0)

        assert drift.check_drift(rolling, baseline, threshold=0.5) is False

    def test_uses_a_default_threshold_of_one_half(self):
        # Same case as test_true_when_kl_divergence_exceeds_the_threshold,
        # relying on the default rather than passing threshold explicitly.
        rolling = BandStats(mean=5.0, std=1.0)
        baseline = BandStats(mean=0.0, std=1.0)

        assert drift.check_drift(rolling, baseline) is True
