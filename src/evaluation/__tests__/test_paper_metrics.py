"""Tests for src/evaluation/paper_metrics.py -- pure functions, no I/O, no
vendor model/dataloader dependency (Test Size: Small).

Covers the two things this module exists for (see track-a-paper-benchmark-
reproduction-plan.md Phase 1): (1) replacing run_validation's own buggy
pixel-count `difficulty` grouping with test.csv's real qplume-based
strong/weak split, with the join validated rather than assumed 1:1, and (2)
deriving AUPRC from run_validation's `metrics["thresholded"]` list using the
non-interpolated (step-function) convention decided in the plan's Context
section, re-sorted by ascending recall rather than just reversed.
"""

import pandas as pd
import paper_metrics
import pytest
import torch


def _out_data_row(**overrides):
    row = {
        "has_plume": True,
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0,
        "pred_pixels_plume": 0,
    }
    row.update(overrides)
    return row


def _out_data(rows: dict) -> pd.DataFrame:
    """rows: {id: {...columns...}}. Mirrors run_validation's own
    `out_data.set_index("id")` shape."""
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "id"
    return df


def _test_csv(qplume_by_id: dict) -> pd.DataFrame:
    return pd.DataFrame({"id": list(qplume_by_id.keys()), "qplume": list(qplume_by_id.values())})


class TestJoinSceneResultsWithTestCsv:
    def test_joins_qplume_column_by_id(self):
        out_data = _out_data({"sceneA": _out_data_row(), "sceneB": _out_data_row()})
        test_csv = _test_csv({"sceneA": 1500.0, "sceneB": 200.0})

        joined = paper_metrics.join_scene_results_with_test_csv(out_data, test_csv)

        assert joined.loc["sceneA", "qplume"] == 1500.0
        assert joined.loc["sceneB", "qplume"] == 200.0

    def test_preserves_row_count_when_ids_match_exactly(self):
        out_data = _out_data({"sceneA": _out_data_row(), "sceneB": _out_data_row()})
        test_csv = _test_csv({"sceneA": 1500.0, "sceneB": 200.0})

        joined = paper_metrics.join_scene_results_with_test_csv(out_data, test_csv)

        assert len(joined) == len(out_data)

    def test_raises_on_duplicate_ids_in_out_data(self):
        out_data = pd.concat(
            [_out_data({"sceneA": _out_data_row()})] * 2
        )  # duplicate "sceneA" index entries
        test_csv = _test_csv({"sceneA": 1500.0})

        with pytest.raises(ValueError, match="duplicate"):
            paper_metrics.join_scene_results_with_test_csv(out_data, test_csv)

    def test_raises_on_duplicate_ids_in_test_csv(self):
        out_data = _out_data({"sceneA": _out_data_row()})
        test_csv = pd.concat([_test_csv({"sceneA": 1500.0})] * 2)

        with pytest.raises(ValueError, match="duplicate"):
            paper_metrics.join_scene_results_with_test_csv(out_data, test_csv)

    def test_raises_when_out_data_has_an_id_missing_from_test_csv(self):
        out_data = _out_data({"sceneA": _out_data_row(), "sceneB": _out_data_row()})
        test_csv = _test_csv({"sceneA": 1500.0})

        with pytest.raises(ValueError, match="id sets"):
            paper_metrics.join_scene_results_with_test_csv(out_data, test_csv)

    def test_raises_when_test_csv_has_an_id_missing_from_out_data(self):
        out_data = _out_data({"sceneA": _out_data_row()})
        test_csv = _test_csv({"sceneA": 1500.0, "sceneB": 200.0})

        with pytest.raises(ValueError, match="id sets"):
            paper_metrics.join_scene_results_with_test_csv(out_data, test_csv)

    def test_raises_on_null_qplume_in_joined_result(self):
        out_data = _out_data({"sceneA": _out_data_row()})
        test_csv = _test_csv({"sceneA": float("nan")})

        with pytest.raises(ValueError, match="qplume"):
            paper_metrics.join_scene_results_with_test_csv(out_data, test_csv)


class TestBucketConfusionMatrices:
    def test_buckets_strong_plume_rows_by_qplume_threshold(self):
        joined = _out_data(
            {
                "strong1": _out_data_row(has_plume=True, TP=10, FN=2),
            }
        )
        joined["qplume"] = [1500.0]

        buckets = paper_metrics.bucket_confusion_matrices(joined)

        assert buckets["strong"][1, 1] == 10  # TP
        assert buckets["strong"][1, 0] == 2  # FN

    def test_buckets_weak_plume_rows_below_threshold(self):
        joined = _out_data({"weak1": _out_data_row(has_plume=True, TP=3, FN=1)})
        joined["qplume"] = [200.0]

        buckets = paper_metrics.bucket_confusion_matrices(joined)

        assert buckets["weak"][1, 1] == 3
        assert buckets["strong"].sum() == 0

    def test_buckets_no_plume_rows_separately_from_plume_rows(self):
        joined = _out_data({"noplume1": _out_data_row(has_plume=False, FP=5, TN=95)})
        joined["qplume"] = [0.0]

        buckets = paper_metrics.bucket_confusion_matrices(joined)

        assert buckets["no_plume"][0, 1] == 5  # FP
        assert buckets["no_plume"][0, 0] == 95  # TN
        assert buckets["strong"].sum() == 0
        assert buckets["weak"].sum() == 0

    def test_sums_confusion_matrix_counts_within_a_bucket_across_scenes(self):
        joined = _out_data(
            {
                "strong1": _out_data_row(has_plume=True, TP=10, FN=2),
                "strong2": _out_data_row(has_plume=True, TP=5, FN=1),
            }
        )
        joined["qplume"] = [1500.0, 2000.0]

        buckets = paper_metrics.bucket_confusion_matrices(joined)

        assert buckets["strong"][1, 1] == 15
        assert buckets["strong"][1, 0] == 3

    def test_qplume_exactly_1000_counts_as_strong(self):
        joined = _out_data({"boundary": _out_data_row(has_plume=True, TP=1)})
        joined["qplume"] = [1000.0]

        buckets = paper_metrics.bucket_confusion_matrices(joined)

        assert buckets["strong"][1, 1] == 1
        assert buckets["weak"].sum() == 0


class TestComputeBucketMetrics:
    def test_computes_known_f1_for_strong_bucket(self):
        # precision = TP/(TP+FP) = 8/10 = 0.8, recall = TP/(TP+FN) = 8/12 ~= 0.6667
        # f1 = 2*p*r/(p+r) = 2*0.8*0.6667/(0.8+0.6667) ~= 0.72727
        buckets = {
            "strong": torch.tensor([[0.0, 2.0], [4.0, 8.0]]),  # [[TN,FP],[FN,TP]]
            "weak": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            "no_plume": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        }

        metrics = paper_metrics.compute_bucket_metrics(buckets)

        assert metrics["strong_f1score"] == pytest.approx(0.72727, abs=1e-4)

    def test_computes_known_fpr_for_no_plume_bucket(self):
        # FPR = FP/(FP+TN) = 5/(5+95) = 0.05
        buckets = {
            "strong": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            "weak": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            "no_plume": torch.tensor([[95.0, 5.0], [0.0, 0.0]]),
        }

        metrics = paper_metrics.compute_bucket_metrics(buckets)

        assert metrics["no_plume_fpr_pixel_level"] == pytest.approx(0.05)

    def test_result_values_are_plain_python_floats(self):
        buckets = {
            "strong": torch.tensor([[0.0, 2.0], [4.0, 8.0]]),
            "weak": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            "no_plume": torch.tensor([[95.0, 5.0], [0.0, 0.0]]),
        }

        metrics = paper_metrics.compute_bucket_metrics(buckets)

        assert type(metrics["strong_f1score"]) is float
        assert type(metrics["no_plume_fpr_pixel_level"]) is float


class TestTileNoPlumeFpr:
    """The paper's FPR is a tile-level classification metric, not
    run_validation's own pixel-summed FPR_no_plume: 'Each tile ... is
    finally marked as containing a plume if the thresholded prediction has
    more than 10 active pixels ... We study the false positive rate (FPR)
    on the subset of the evaluation dataset that does not contain any
    plumes' (paper page 8, "Metrics"). Confirmed empirically during Phase 1
    execution: run_validation's pixel-summed FPR_no_plume came out ~0.5%
    on a real run where the paper reports ~44% for the same variant --
    an ~80x mismatch from comparing two different metrics, not noise."""

    def test_counts_a_no_plume_scene_as_a_false_positive_above_the_pixel_threshold(self):
        joined = _out_data(
            {
                "noplume_fp": _out_data_row(has_plume=False, pred_pixels_plume=11),
                "noplume_clean": _out_data_row(has_plume=False, pred_pixels_plume=10),
            }
        )
        joined["qplume"] = [0.0, 0.0]

        fpr = paper_metrics.tile_no_plume_fpr(joined)

        assert fpr == pytest.approx(0.5)  # 1 of 2 no-plume scenes over threshold

    def test_exactly_ten_pixels_does_not_count_as_a_false_positive(self):
        # ">10 active pixels", not ">=10" -- boundary must not flip.
        joined = _out_data({"noplume1": _out_data_row(has_plume=False, pred_pixels_plume=10)})
        joined["qplume"] = [0.0]

        fpr = paper_metrics.tile_no_plume_fpr(joined)

        assert fpr == pytest.approx(0.0)

    def test_ignores_plume_scenes_entirely(self):
        joined = _out_data(
            {
                "plume_but_huge_pred": _out_data_row(has_plume=True, pred_pixels_plume=5000),
                "noplume_clean": _out_data_row(has_plume=False, pred_pixels_plume=0),
            }
        )
        joined["qplume"] = [1500.0, 0.0]

        fpr = paper_metrics.tile_no_plume_fpr(joined)

        assert fpr == pytest.approx(0.0)

    def test_returns_a_plain_python_float(self):
        joined = _out_data({"noplume1": _out_data_row(has_plume=False, pred_pixels_plume=0)})
        joined["qplume"] = [0.0]

        assert type(paper_metrics.tile_no_plume_fpr(joined)) is float


class TestSortPrecisionRecallByAscendingRecall:
    def test_sorts_by_ascending_recall_not_just_reversed(self):
        # Deliberately non-monotonic in list order (as run_validation's own
        # high-to-low threshold order can produce once TP/FP/FN come from
        # summed per-bucket confusion matrices).
        thresholded = [
            {"threshold": 0.99, "recall": 0.1, "precision": 1.0},
            {"threshold": 0.9, "recall": 0.5, "precision": 0.6},
            {"threshold": 0.5, "recall": 0.4, "precision": 0.9},
            {"threshold": 0.1, "recall": 0.9, "precision": 0.3},
        ]

        sorted_points = paper_metrics.sort_precision_recall_by_ascending_recall(thresholded)

        recalls = [r for r, _ in sorted_points]
        assert recalls == sorted(recalls)
        # A naive reversal of list order would NOT be ascending -- guards
        # against exactly that mistake.
        reversed_recalls = [item["recall"] for item in reversed(thresholded)]
        assert recalls != reversed_recalls

    def test_accepts_torch_tensor_precision_and_recall_values(self):
        # Real run_validation output stores these as 0-d tensors (the return
        # value of starcop.metrics.{precision,recall} applied to a tensor
        # confusion matrix), not plain floats.
        thresholded = [
            {"threshold": 0.9, "recall": torch.tensor(0.5), "precision": torch.tensor(0.6)},
            {"threshold": 0.99, "recall": torch.tensor(0.1), "precision": torch.tensor(1.0)},
        ]

        sorted_points = paper_metrics.sort_precision_recall_by_ascending_recall(thresholded)

        assert sorted_points == [
            (pytest.approx(0.1), pytest.approx(1.0)),
            (pytest.approx(0.5), pytest.approx(0.6)),
        ]
        assert type(sorted_points[0][0]) is float


class TestNonInterpolatedAveragePrecision:
    def test_matches_hand_computed_value_on_a_simple_curve(self):
        # AP = sum (recall_n - recall_{n-1}) * precision_n, recall_0 = 0
        # = (0.5-0)*0.8 + (1.0-0.5)*0.4 = 0.4 + 0.2 = 0.6
        sorted_points = [(0.5, 0.8), (1.0, 0.4)]

        ap = paper_metrics.non_interpolated_average_precision(sorted_points)

        assert ap == pytest.approx(0.6)

    def test_pins_expected_value_on_a_non_monotonic_curve_where_trapezoidal_would_diverge(self):
        # Sorted ascending by recall: (0.1,1.0),(0.4,0.9),(0.5,0.6),(0.9,0.3)
        # Non-interpolated (this function's convention):
        #   0.1*1.0 + 0.3*0.9 + 0.1*0.6 + 0.4*0.3 = 0.55
        # Trapezoidal integration over the same points gives 0.54 instead --
        # if this implementation is ever accidentally swapped to trapezoidal
        # integration, this test fails instead of silently drifting.
        sorted_points = [(0.1, 1.0), (0.4, 0.9), (0.5, 0.6), (0.9, 0.3)]

        ap = paper_metrics.non_interpolated_average_precision(sorted_points)

        assert ap == pytest.approx(0.55)

    def test_handles_a_missing_recall_zero_endpoint_by_treating_it_as_zero(self):
        # No point at recall=0 is provided (not guaranteed by
        # run_validation's threshold list) -- the first point's recall
        # delta is taken from an implicit recall=0, not from an error.
        sorted_points = [(0.2, 1.0)]

        ap = paper_metrics.non_interpolated_average_precision(sorted_points)

        assert ap == pytest.approx(0.2)

    def test_handles_a_missing_recall_one_endpoint_without_extrapolating(self):
        # No point reaches recall=1 (e.g. the highest threshold yields zero
        # predicted positives) -- the curve simply covers what was measured,
        # consistent with sklearn.metrics.average_precision_score.
        sorted_points = [(0.3, 1.0), (0.6, 0.5)]

        ap = paper_metrics.non_interpolated_average_precision(sorted_points)

        assert ap == pytest.approx(0.3 * 1.0 + 0.3 * 0.5)

    def test_returns_zero_for_an_empty_curve(self):
        assert paper_metrics.non_interpolated_average_precision([]) == 0.0


class TestDeriveAuprc:
    def test_sorts_and_integrates_run_validations_thresholded_list(self):
        thresholded = [
            {"threshold": 0.99, "recall": 0.1, "precision": 1.0},
            {"threshold": 0.5, "recall": 0.6, "precision": 0.5},
        ]

        auprc = paper_metrics.derive_auprc(thresholded)

        assert auprc == pytest.approx(0.1 * 1.0 + 0.5 * 0.5)
