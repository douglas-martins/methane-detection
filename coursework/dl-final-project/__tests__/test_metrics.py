import pytest
import torch
from metrics import (
    add_counts,
    average_precision_from_sweep,
    confusion_matrix_counts,
    confusion_matrix_from_counts,
    detection_rate_from_counts,
    f1_from_counts,
    patch_detection_counts,
    pixel_f1,
    precision_from_counts,
    precision_recall_points_from_sweep,
    recall_from_counts,
    sort_points_by_recall,
    sweep_confusion_counts,
)


class TestPixelF1:
    def test_perfect_prediction_scores_one(self):
        predictions = torch.tensor([1.0, 0.0, 1.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert pixel_f1(predictions, targets) == 1.0

    def test_all_wrong_scores_zero(self):
        predictions = torch.tensor([1.0, 1.0])
        targets = torch.tensor([0.0, 0.0])
        assert pixel_f1(predictions, targets) == 0.0

    def test_no_predicted_or_true_positives_scores_zero_not_nan(self):
        # The degenerate all-negative case this section explicitly worries
        # about must not silently produce a NaN F1.
        predictions = torch.zeros(4)
        targets = torch.zeros(4)
        result = pixel_f1(predictions, targets)
        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_matches_hand_computed_precision_recall(self):
        # tp=1, fp=1, fn=1 -> precision=0.5, recall=0.5, f1=0.5
        predictions = torch.tensor([1.0, 1.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0])
        assert pixel_f1(predictions, targets) == 0.5


class TestConfusionMatrixCounts:
    def test_perfect_prediction_counts_only_tp_and_tn(self):
        predictions = torch.tensor([1.0, 0.0, 1.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert confusion_matrix_counts(predictions, targets) == {
            "tp": 2.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 2.0,
        }

    def test_matches_hand_computed_case(self):
        # Section 8's own worked example: tp=1, fp=1, fn=1, tn=0.
        predictions = torch.tensor([1.0, 1.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0])
        assert confusion_matrix_counts(predictions, targets) == {
            "tp": 1.0,
            "fp": 1.0,
            "fn": 1.0,
            "tn": 0.0,
        }


class TestAddCounts:
    def test_sums_every_key(self):
        first = {"tp": 1.0, "fp": 2.0, "fn": 0.0, "tn": 3.0}
        second = {"tp": 4.0, "fp": 0.0, "fn": 1.0, "tn": 2.0}
        assert add_counts(first, second) == {"tp": 5.0, "fp": 2.0, "fn": 1.0, "tn": 5.0}


class TestPrecisionRecallF1FromCounts:
    def test_matches_hand_computed_case(self):
        counts = {"tp": 1.0, "fp": 1.0, "fn": 1.0, "tn": 0.0}
        assert precision_from_counts(counts) == 0.5
        assert recall_from_counts(counts) == 0.5
        assert f1_from_counts(counts) == 0.5

    def test_perfect_prediction_scores_one(self):
        counts = {"tp": 4.0, "fp": 0.0, "fn": 0.0, "tn": 6.0}
        assert precision_from_counts(counts) == 1.0
        assert recall_from_counts(counts) == 1.0
        assert f1_from_counts(counts) == 1.0

    def test_no_predicted_positives_precision_is_zero_not_nan(self):
        counts = {"tp": 0.0, "fp": 0.0, "fn": 5.0, "tn": 10.0}
        result = precision_from_counts(counts)
        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_no_true_positives_recall_is_zero_not_nan(self):
        counts = {"tp": 0.0, "fp": 3.0, "fn": 0.0, "tn": 10.0}
        result = recall_from_counts(counts)
        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_all_zero_counts_f1_is_zero_not_nan(self):
        counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 10.0}
        result = f1_from_counts(counts)
        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_denominator_of_exactly_one_is_still_a_valid_denominator(self):
        # Regression for a `> 0` vs `> 1` boundary slip: denominator == 1 must
        # still take the division branch, not fall through to the 0.0 default.
        assert precision_from_counts({"tp": 1.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}) == 1.0
        assert recall_from_counts({"tp": 1.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}) == 1.0
        # f1's denominator is `2*tp + fp + fn`, always even when tp>0 is an
        # integer count, so it can only equal exactly 1 with a fractional tp
        # -- exercise that directly rather than only through pixel counts.
        assert f1_from_counts({"tp": 0.5, "fp": 0.0, "fn": 0.0, "tn": 0.0}) == 1.0


class TestConfusionMatrixFromCounts:
    def test_layout_is_tn_fp_fn_tp(self):
        # Same [[TN, FP], [FN, TP]] layout as vendor/starcop's own
        # starcop.metrics confusion-matrix convention (never imported here --
        # see metrics.py's module docstring for the reuse-vs-reimplement call).
        counts = {"tp": 1.0, "fp": 2.0, "fn": 3.0, "tn": 4.0}
        matrix = confusion_matrix_from_counts(counts)
        assert torch.equal(matrix, torch.tensor([[4.0, 2.0], [3.0, 1.0]]))


class TestSweepConfusionCounts:
    def test_matches_hand_computed_counts_per_threshold(self):
        # probs: [0.9, 0.4, 0.1, 0.6], targets: [1, 1, 0, 0].
        probs = torch.tensor([0.9, 0.4, 0.1, 0.6])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        thresholds = torch.tensor([0.0, 0.5, 1.0])

        counts = sweep_confusion_counts(probs, targets, thresholds)

        # threshold=0.0: all predicted positive -> tp=2, fp=2, fn=0, tn=0.
        # threshold=0.5: predicted=[T,F,F,T] -> tp=1, fp=1, fn=1, tn=1.
        # threshold=1.0: none predicted positive -> tp=0, fp=0, fn=2, tn=2.
        assert torch.equal(
            counts, torch.tensor([[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
        )

    def test_accepts_any_matching_shape_not_only_1d(self):
        probs = torch.tensor([[0.9, 0.1], [0.4, 0.6]])
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        thresholds = torch.tensor([0.5])

        counts = sweep_confusion_counts(probs, targets, thresholds)

        # predicted (>0.5): [[T,F],[F,T]]. targets: [[T,F],[T,F]].
        # tp=1 (0,0), fp=1 (1,1), fn=1 (1,0), tn=1 (0,1).
        assert torch.equal(counts, torch.tensor([[1.0, 1.0, 1.0, 1.0]]))

    def test_probability_exactly_at_threshold_is_not_predicted_positive(self):
        # Strict `>` -- a probability exactly equal to the threshold does not
        # count as a positive prediction.
        probs = torch.tensor([0.5])
        targets = torch.tensor([1.0])
        thresholds = torch.tensor([0.5])

        counts = sweep_confusion_counts(probs, targets, thresholds)

        # Not predicted positive -> tp=0, fn=1 (the one true positive was missed).
        assert torch.equal(counts, torch.tensor([[0.0, 0.0, 1.0, 0.0]]))


class TestPrecisionRecallPointsFromSweep:
    def test_matches_hand_computed_points(self):
        sweep_counts = torch.tensor(
            [[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]]
        )

        points = precision_recall_points_from_sweep(sweep_counts)

        assert points == pytest.approx([(1.0, 0.5), (0.5, 0.5), (0.0, 0.0)])

    def test_all_zero_row_is_zero_not_nan(self):
        sweep_counts = torch.tensor([[0.0, 0.0, 0.0, 5.0]])

        points = precision_recall_points_from_sweep(sweep_counts)

        assert points[0] == (0.0, 0.0)

    def test_denominator_of_exactly_one_is_still_a_valid_denominator(self):
        # tp=1, fp=0, fn=0 -> precision denom = recall denom = 1. Regression
        # for a `> 0` vs `> 1` boundary slip in either torch.where condition.
        sweep_counts = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        points = precision_recall_points_from_sweep(sweep_counts)

        assert points == [(1.0, 1.0)]


class TestSortPointsByRecall:
    def test_sorts_ascending_by_recall(self):
        # Threshold-sweep order: high threshold (low recall) first.
        points = [(1.0, 0.5), (0.5, 0.5), (0.0, 0.0)]

        assert sort_points_by_recall(points) == [(0.0, 0.0), (0.5, 0.5), (1.0, 0.5)]

    def test_already_sorted_input_is_unchanged(self):
        points = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.5)]

        assert sort_points_by_recall(points) == points

    def test_sorts_only_by_recall_ties_keep_original_relative_order(self):
        # Two points share the same recall but differ in precision, in an
        # order that a full-tuple sort (not a recall-only key) would flip.
        points = [(0.5, 0.9), (0.5, 0.1)]

        assert sort_points_by_recall(points) == [(0.5, 0.9), (0.5, 0.1)]


class TestAveragePrecisionFromSweep:
    def test_matches_hand_computed_non_interpolated_average_precision(self):
        # Same convention as src/baselines/starcop/evaluation/paper_metrics.py's
        # non_interpolated_average_precision -- sort ascending by recall,
        # sum (recall_n - recall_{n-1}) * precision_n, recall_0 = 0.
        # Points sorted by recall: (0.0, 0.0), (0.5, 0.5), (1.0, 0.5).
        # AP = (0.5-0)*0.5 + (1.0-0.5)*0.5 = 0.5.
        sweep_counts = torch.tensor(
            [[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]]
        )

        assert average_precision_from_sweep(sweep_counts) == pytest.approx(0.5)

    def test_all_zero_counts_is_zero_not_nan(self):
        sweep_counts = torch.tensor([[0.0, 0.0, 0.0, 5.0]])

        result = average_precision_from_sweep(sweep_counts)

        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_first_recall_step_is_measured_from_zero(self):
        # A single point at (recall=0.5, precision=1.0): AP must be
        # (0.5 - 0) * 1.0 = 0.5, not (0.5 - 1.0) * 1.0 = -0.5 -- regression
        # for previous_recall starting anywhere but 0.0.
        sweep_counts = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        assert average_precision_from_sweep(sweep_counts) == pytest.approx(0.5)


class TestPatchDetectionCounts:
    def test_counts_a_correctly_detected_positive_patch(self):
        predictions = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
        targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])

        counts = patch_detection_counts(predictions, targets)

        assert counts == {"positive_patches": 1, "detected_patches": 1}

    def test_a_missed_positive_patch_is_not_counted_as_detected(self):
        predictions = torch.zeros(1, 1, 2, 2)
        targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

        counts = patch_detection_counts(predictions, targets)

        assert counts == {"positive_patches": 1, "detected_patches": 0}

    def test_a_false_positive_on_a_negative_patch_is_not_counted(self):
        # A negative patch predicted positive must not inflate the detection
        # rate -- this metric is specifically about positive patches.
        predictions = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        targets = torch.zeros(1, 1, 2, 2)

        counts = patch_detection_counts(predictions, targets)

        assert counts == {"positive_patches": 0, "detected_patches": 0}

    def test_counts_per_patch_across_a_multi_channel_batch(self):
        # 4 patches, 2 channels each: patch 0 positive (detected), patch 1
        # positive in a different channel (missed), patch 2 positive across
        # both channels (detected via one channel), patch 3 negative.
        # Regression for reducing over the wrong dims (batch, or only some
        # of channel+spatial) instead of every per-patch dim.
        predictions = torch.zeros(4, 2, 3, 3)
        targets = torch.zeros(4, 2, 3, 3)
        targets[0, 0, 0, 0] = 1.0
        predictions[0, 0, 0, 0] = 1.0
        targets[1, 1, 1, 1] = 1.0
        targets[2, 0, 2, 2] = 1.0
        targets[2, 1, 0, 1] = 1.0
        predictions[2, 0, 2, 2] = 1.0

        counts = patch_detection_counts(predictions, targets)

        assert counts == {"positive_patches": 3, "detected_patches": 2}


class TestDetectionRateFromCounts:
    def test_matches_hand_computed_fraction(self):
        counts = {"positive_patches": 4, "detected_patches": 3}
        assert detection_rate_from_counts(counts) == 0.75

    def test_no_positive_patches_returns_zero_not_nan(self):
        counts = {"positive_patches": 0, "detected_patches": 0}
        result = detection_rate_from_counts(counts)
        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_total_of_exactly_one_is_still_a_valid_denominator(self):
        counts = {"positive_patches": 1, "detected_patches": 1}
        assert detection_rate_from_counts(counts) == 1.0
