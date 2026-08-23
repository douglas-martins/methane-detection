"""Corrects run_validation's own strong/weak split and derives AUPRC from
its threshold sweep -- see track-a-paper-benchmark-reproduction-plan.md
Phase 1 for the rationale.

`run_validation` (vendor/starcop/starcop/validation.py, unmodified) computes
its own `difficulty` column from label pixel count (>1000px = "easy"),
silently shadowing test.csv's real `difficulty`/strong-weak definition,
which is keyed off `qplume` (kg/h). This module never edits or calls into
that internal grouping -- it joins run_validation's per-scene `out_data`
back onto test.csv's `qplume` column and buckets from there.
"""

from typing import Optional

import pandas as pd
import torch
from _vendor_starcop_evaluation import starcop_metrics

STRONG_QPLUME_THRESHOLD_KG_H = 1000.0
TILE_CLASSIFICATION_PIXEL_THRESHOLD = 10


def join_scene_results_with_test_csv(
    out_data: pd.DataFrame, test_csv: pd.DataFrame
) -> pd.DataFrame:
    """Joins run_validation's per-scene `out_data` (indexed by scene `id`)
    with `test_csv`'s `qplume` column, by `id`. Raises `ValueError` instead
    of silently duplicating or dropping rows: on a non-unique `id` on either
    side, on a mismatched `id` set between the two, or on a null `qplume` in
    the joined result (which would otherwise fail the strong/weak split
    without erroring)."""
    if not out_data.index.is_unique:
        raise ValueError("out_data has duplicate ids")

    test_csv = test_csv.set_index("id") if "id" in test_csv.columns else test_csv
    if not test_csv.index.is_unique:
        raise ValueError("test_csv has duplicate ids")

    if set(out_data.index) != set(test_csv.index):
        raise ValueError(
            "id sets differ between out_data and test_csv -- "
            f"only in out_data: {sorted(set(out_data.index) - set(test_csv.index))}, "
            f"only in test_csv: {sorted(set(test_csv.index) - set(out_data.index))}"
        )

    joined = out_data.merge(
        test_csv[["qplume"]], left_index=True, right_index=True, validate="one_to_one"
    )

    if joined["qplume"].isna().any():
        raise ValueError("null qplume in joined result")

    return joined


def bucket_confusion_matrices(joined: pd.DataFrame) -> dict[str, torch.Tensor]:
    """Buckets `joined` (the output of `join_scene_results_with_test_csv`)
    into `strong` (`qplume >= 1000`), `weak` (`has_plume` and
    `qplume < 1000`), and `no_plume`; sums each bucket's `TP`/`FP`/`TN`/`FN`
    across scenes into a `[[TN, FP], [FN, TP]]` tensor -- the same layout
    `run_validation` itself builds internally for its per-difficulty
    metrics."""
    has_plume = joined["has_plume"].astype(bool)
    is_strong = has_plume & (joined["qplume"] >= STRONG_QPLUME_THRESHOLD_KG_H)
    is_weak = has_plume & (joined["qplume"] < STRONG_QPLUME_THRESHOLD_KG_H)
    is_no_plume = ~has_plume

    buckets = {}
    for name, mask in [("strong", is_strong), ("weak", is_weak), ("no_plume", is_no_plume)]:
        subset = joined.loc[mask]
        buckets[name] = torch.tensor(
            [
                [subset["TN"].sum(), subset["FP"].sum()],
                [subset["FN"].sum(), subset["TP"].sum()],
            ],
            dtype=torch.float64,
        )
    return buckets


def compute_bucket_metrics(buckets: dict[str, torch.Tensor]) -> dict[str, float]:
    """Applies `starcop.metrics` the same way `run_validation` does
    internally, just off the corrected buckets from `bucket_confusion_matrices`.

    `no_plume_fpr_pixel_level` is exposed for transparency/debugging only --
    it is NOT the paper's reported FPR, which is a tile-level classification
    rate (`tile_no_plume_fpr`, computed separately since it needs per-scene
    `pred_pixels_plume`, not a pixel-summed confusion matrix). Confirmed
    empirically during Phase 1 execution: this pixel-level number came out
    ~0.5% on a real run where the paper reports ~44% for the same variant --
    comparing this key against the paper's Table 2 FPR column would be
    comparing two different metrics."""
    metrics: dict[str, float] = {}
    for bucket in ["strong", "weak"]:
        cm = buckets[bucket]
        metrics[f"{bucket}_precision"] = float(starcop_metrics.precision(cm))
        metrics[f"{bucket}_recall"] = float(starcop_metrics.recall(cm))
        metrics[f"{bucket}_f1score"] = float(starcop_metrics.f1score(cm))
        metrics[f"{bucket}_iou"] = float(starcop_metrics.iou(cm))
    metrics["no_plume_fpr_pixel_level"] = float(starcop_metrics.FPR(buckets["no_plume"]))
    return metrics


def tile_no_plume_fpr(joined: pd.DataFrame) -> float:
    """The paper's own FPR metric (page 8, "Metrics") is tile-level, not
    run_validation's pixel-summed `FPR_no_plume`: 'Each tile ... is finally
    marked as containing a plume if the thresholded prediction has more
    than 10 active pixels ... We study the false positive rate (FPR) on the
    subset of the evaluation dataset that does not contain any plumes.'
    Fraction of no-plume scenes whose `pred_pixels_plume` exceeds
    `TILE_CLASSIFICATION_PIXEL_THRESHOLD`."""
    no_plume = joined.loc[~joined["has_plume"].astype(bool)]
    predicted_positive = (no_plume["pred_pixels_plume"] > TILE_CLASSIFICATION_PIXEL_THRESHOLD).sum()
    return float(predicted_positive / len(no_plume))


def sort_precision_recall_by_ascending_recall(thresholded: list[dict]) -> list[tuple[float, float]]:
    """Returns `(recall, precision)` pairs from `run_validation`'s
    `metrics["thresholded"]` list, sorted ascending by recall.
    `thresholded` is built high-to-low threshold (`validation.py:41-42`),
    which is not guaranteed to be strictly ascending in recall once TP/FP/FN
    come from summed per-bucket confusion matrices -- so this explicitly
    sorts rather than just reversing the input order."""
    pairs = [(float(item["recall"]), float(item["precision"])) for item in thresholded]
    return sorted(pairs, key=lambda pr: pr[0])


def non_interpolated_average_precision(sorted_recall_precision: list[tuple[float, float]]) -> float:
    """Non-interpolated (step-function) average precision: sum
    `(recall_n - recall_{n-1}) * precision_n` over points already sorted
    ascending by recall, with `recall_0 = 0` (matching
    `sklearn.metrics.average_precision_score`'s convention) -- no
    interpolation between points, and no artificial endpoint at recall=1 if
    the curve doesn't reach it."""
    ap = 0.0
    previous_recall = 0.0
    for recall, precision in sorted_recall_precision:
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def derive_auprc(thresholded: list[dict]) -> float:
    """Derives AUPRC from `run_validation`'s `metrics["thresholded"]` list
    using the non-interpolated convention decided in this plan's Context
    section (the paper doesn't specify one, and vendor/starcop never
    computes AUPRC itself)."""
    sorted_points = sort_precision_recall_by_ascending_recall(thresholded)
    return non_interpolated_average_precision(sorted_points)


def scene_iou(row: pd.Series) -> Optional[float]:
    """Per-scene IoU from a joined row's own TP/FP/FN counts. Returns None
    (rather than NaN/raising) when the scene has no positive prediction or
    label at all (TP+FP+FN == 0), so callers can exclude it from a
    best-of selection instead of it winning by an undefined tie."""
    denominator = row["TP"] + row["FP"] + row["FN"]
    if denominator == 0:
        return None
    return float(row["TP"] / denominator)


def scene_recall(row: pd.Series) -> Optional[float]:
    """Per-scene recall from a joined row's own TP/FN counts. Returns None
    when the scene has no labelled positive pixels (TP+FN == 0) -- recall is
    undefined there, not zero."""
    denominator = row["TP"] + row["FN"]
    if denominator == 0:
        return None
    return float(row["TP"] / denominator)
