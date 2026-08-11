"""Tests for src/training/metrics_ext.py (Test Size: Small -- pure tensor
math, no I/O). f1score_background is new; compute_all wraps STARCOP's own
unmodified metrics.METRICS_CONFUSION_MATRIX list plus it.
"""

import pytest
import torch

import metrics_ext
from _vendor_starcop_training import starcop_metrics


def _cm(tn, fp, fn, tp) -> torch.Tensor:
    """Binary confusion matrix in starcop.metrics's own [[TN,FP],[FN,TP]] layout."""
    return torch.tensor([[tn, fp], [fn, tp]])


class TestF1ScoreBackground:
    def test_matches_hand_computed_value_for_known_confusion_matrix(self):
        # TN=90, FP=10, FN=5, TP=5 -> background precision=90/95, recall=90/100
        cm = _cm(tn=90, fp=10, fn=5, tp=5)

        result = metrics_ext.f1score_background(cm)

        background_precision = 90 / 95
        background_recall = 90 / 100
        expected = 2 * (background_precision * background_recall) / (
            background_precision + background_recall
        )
        assert result == pytest.approx(expected, rel=1e-6)

    def test_is_one_for_a_fully_perfect_confusion_matrix(self):
        # FP=0 and FN=0 -- perfect recall AND precision for both classes,
        # not just perfect background recall (TN=100,FP=0 alone isn't
        # enough: 5 methane-as-background FNs would still cost precision).
        cm = _cm(tn=100, fp=0, fn=0, tp=5)

        assert metrics_ext.f1score_background(cm) == pytest.approx(1.0)

    def test_differs_from_f1score_on_an_imbalanced_confusion_matrix(self):
        cm = _cm(tn=1000, fp=10, fn=50, tp=5)

        assert metrics_ext.f1score_background(cm) != starcop_metrics.f1score(cm)


class TestComputeAll:
    def test_includes_every_metric_name_from_metrics_confusion_matrix_plus_background_f1(self):
        cm = _cm(tn=90, fp=10, fn=5, tp=5)

        result = metrics_ext.compute_all(cm)

        expected_names = {fn.__name__ for fn in starcop_metrics.METRICS_CONFUSION_MATRIX}
        expected_names.add("f1score_background")
        assert set(result.keys()) == expected_names

    def test_f1score_background_value_matches_direct_call(self):
        cm = _cm(tn=90, fp=10, fn=5, tp=5)

        result = metrics_ext.compute_all(cm)

        assert result["f1score_background"] == pytest.approx(
            metrics_ext.f1score_background(cm)
        )
