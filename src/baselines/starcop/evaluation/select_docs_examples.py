"""Deterministic picks of curated scene ids for the public docs page --
see track-a-paper-benchmark-reproduction-plan.md Phase 1 for the decided
selection rule: best strong/weak-plume detection = highest per-scene IoU in
that bucket; false negative = highest-qplume zero-recall scene in either
plume bucket (None if none exist); cleanest no-plume = fewest
predicted-positive pixels.

Takes the same `joined` frame `paper_metrics.bucket_confusion_matrices`
consumes (per-scene TP/FP/FN/TN + qplume + has_plume, indexed by scene id).
"""

from typing import Optional

import pandas as pd
from paper_metrics import STRONG_QPLUME_THRESHOLD_KG_H, scene_iou, scene_recall


def _best_by(subset: pd.DataFrame, score) -> Optional[str]:
    """Returns the index (`id`) of the row in `subset` with the highest
    `score(row)`, skipping rows where `score` returns None (undefined).
    None if no row has a defined score."""
    best_id: Optional[str] = None
    best_score: Optional[float] = None
    for scene_id, row in subset.iterrows():
        value = score(row)
        if value is None:
            continue
        if best_score is None or value > best_score:
            best_score = value
            best_id = scene_id
    return best_id


def select_docs_examples(joined: pd.DataFrame) -> dict[str, Optional[str]]:
    """Returns `{"best_strong_plume", "best_weak_plume", "false_negative",
    "cleanest_no_plume"}` scene ids (None where no qualifying scene exists)."""
    has_plume = joined["has_plume"].astype(bool)
    is_strong = has_plume & (joined["qplume"] >= STRONG_QPLUME_THRESHOLD_KG_H)
    is_weak = has_plume & (joined["qplume"] < STRONG_QPLUME_THRESHOLD_KG_H)
    is_no_plume = ~has_plume

    misses = joined.loc[has_plume & (joined.apply(scene_recall, axis=1) == 0)]

    return {
        "best_strong_plume": _best_by(joined.loc[is_strong], scene_iou),
        "best_weak_plume": _best_by(joined.loc[is_weak], scene_iou),
        "false_negative": _best_by(misses, lambda row: row["qplume"]),
        "cleanest_no_plume": _best_by(
            joined.loc[is_no_plume], lambda row: -row["pred_pixels_plume"]
        ),
    }
