"""Tests for src/evaluation/select_docs_examples.py -- pure functions, no
I/O (Test Size: Small).

Selection rule decided for the public docs page (track-a-paper-benchmark-
reproduction-plan.md Phase 1): best strong/weak-plume detection = highest
per-scene IoU in that bucket; false negative = highest-qplume zero-recall
scene in either plume bucket, only if one exists; cleanest no-plume = fewest
predicted-positive pixels.
"""

import pandas as pd
import select_docs_examples


def _row(**overrides):
    row = {
        "has_plume": True,
        "qplume": 1500.0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
        "pred_pixels_plume": 0,
    }
    row.update(overrides)
    return row


def _joined(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "id"
    return df


class TestSelectDocsExamples:
    def test_picks_highest_iou_strong_plume_scene(self):
        joined = _joined(
            {
                # iou = TP/(TP+FP+FN) = 8/10 = 0.8
                "strong_good": _row(qplume=1500.0, TP=8, FP=1, FN=1),
                # iou = 3/10 = 0.3
                "strong_ok": _row(qplume=2000.0, TP=3, FP=3, FN=4),
            }
        )

        picks = select_docs_examples.select_docs_examples(joined)

        assert picks["best_strong_plume"] == "strong_good"

    def test_picks_highest_iou_weak_plume_scene(self):
        joined = _joined(
            {
                "weak_good": _row(qplume=200.0, TP=9, FP=0, FN=1),  # iou=0.9
                "weak_ok": _row(qplume=500.0, TP=2, FP=2, FN=2),  # iou=0.33
            }
        )

        picks = select_docs_examples.select_docs_examples(joined)

        assert picks["best_weak_plume"] == "weak_good"

    def test_picks_highest_qplume_zero_recall_scene_as_false_negative(self):
        joined = _joined(
            {
                "detected": _row(qplume=1200.0, TP=5, FN=0),
                "missed_small": _row(qplume=300.0, TP=0, FN=4),  # recall=0
                "missed_big": _row(qplume=1800.0, TP=0, FN=10),  # recall=0, higher qplume
            }
        )

        picks = select_docs_examples.select_docs_examples(joined)

        assert picks["false_negative"] == "missed_big"

    def test_returns_none_for_false_negative_when_no_misses_exist(self):
        joined = _joined(
            {
                "detected1": _row(qplume=1200.0, TP=5, FN=1),
                "detected2": _row(qplume=300.0, TP=2, FN=1),
            }
        )

        picks = select_docs_examples.select_docs_examples(joined)

        assert picks["false_negative"] is None

    def test_picks_fewest_predicted_pixels_no_plume_scene_as_cleanest(self):
        joined = _joined(
            {
                "clean": _row(has_plume=False, qplume=0.0, pred_pixels_plume=0),
                "noisy": _row(has_plume=False, qplume=0.0, pred_pixels_plume=42),
            }
        )

        picks = select_docs_examples.select_docs_examples(joined)

        assert picks["cleanest_no_plume"] == "clean"

    def test_returns_all_four_expected_keys(self):
        joined = _joined(
            {
                "strong1": _row(qplume=1500.0, TP=8, FP=1, FN=1),
                "weak1": _row(qplume=200.0, TP=9, FP=0, FN=1),
                "noplume1": _row(has_plume=False, qplume=0.0, pred_pixels_plume=0),
            }
        )

        picks = select_docs_examples.select_docs_examples(joined)

        assert set(picks.keys()) == {
            "best_strong_plume",
            "best_weak_plume",
            "false_negative",
            "cleanest_no_plume",
        }

    def test_scene_with_no_prediction_or_label_is_excluded_from_best_pick(self):
        # TP+FP+FN == 0 -- IoU undefined, must not silently win a comparison
        # against a real (nonzero-IoU) scene by an accidental 0-vs-0 tie.
        joined = _joined(
            {
                "empty": _row(qplume=1500.0, TP=0, FP=0, FN=0),
                "real": _row(qplume=1500.0, TP=1, FP=0, FN=0),  # iou=1.0
            }
        )

        picks = select_docs_examples.select_docs_examples(joined)

        assert picks["best_strong_plume"] == "real"

    def test_returns_none_for_a_bucket_with_no_scenes_at_all(self):
        joined = _joined({"strong1": _row(qplume=1500.0, TP=1)})

        picks = select_docs_examples.select_docs_examples(joined)

        assert picks["best_weak_plume"] is None
        assert picks["cleanest_no_plume"] is None
