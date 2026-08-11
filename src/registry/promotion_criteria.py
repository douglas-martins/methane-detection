"""Pure promotion-decision logic for the MLflow model registry (TASK-2.3).

No MLflow SDK calls here -- src/registry/mlflow_registry.py fetches the raw
metrics/history from a run and hands them to evaluate_staging/
evaluate_production, which are testable without any tracking server.

Thresholds mirror docs/model_registry_policy.md ("OA" = val_accuracy /
test_accuracy, "F1 (methane class)" = val_f1score / test_f1score -- see
mlops-methane-detection-plan.md TASK-2.3 decisions).
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List

STAGING_THRESHOLDS: Dict[str, float] = {"val_accuracy": 0.85, "val_f1score": 0.70}
PRODUCTION_THRESHOLDS: Dict[str, float] = {"test_accuracy": 0.88, "test_f1score": 0.75}

# Instability proxy for "no training instability (loss curve is smooth)":
# the population stddev of consecutive val_loss deltas over the last `window`
# logged epochs must not exceed `max_delta_stddev`. Picked as a conservative
# starting point (a smoothly converging curve moves by a few hundredths per
# epoch late in training; a spiking one swings by several tenths) -- tune
# against real experiment val_loss histories as more runs accumulate, per
# the plan's "to be updated as experiments run" note on promotion criteria.
DEFAULT_INSTABILITY_WINDOW = 5
DEFAULT_MAX_LOSS_DELTA_STDDEV = 0.1


@dataclass
class PromotionDecision:
    """Outcome of evaluating a run against one stage's promotion criteria."""

    promote: bool
    reasons: List[str] = field(default_factory=list)


def check_thresholds(metrics: Dict[str, float], thresholds: Dict[str, float]) -> List[str]:
    """Returns a rejection reason per threshold not met; empty if all pass.

    A metric absent from `metrics` entirely is treated as a failure (not
    skipped) -- Production must never promote silently just because its
    test-set metrics were never logged (e.g. run_validation failed). A
    non-finite value (NaN, +-inf) is rejected explicitly rather than
    compared directly: `nan < threshold` and `inf < threshold` are both
    False in Python, so without this check a corrupted metric would
    silently pass instead of failing the threshold it can't meaningfully
    satisfy.
    """
    reasons = []
    for name, threshold in thresholds.items():
        if name not in metrics:
            reasons.append(f"missing metric: {name}")
        elif not math.isfinite(metrics[name]):
            reasons.append(f"{name}={metrics[name]} is not a finite value")
        elif metrics[name] < threshold:
            reasons.append(f"{name}={metrics[name]} is below threshold {threshold}")
    return reasons


def is_loss_history_stable(
    loss_history: List[float],
    window: int = DEFAULT_INSTABILITY_WINDOW,
    max_delta_stddev: float = DEFAULT_MAX_LOSS_DELTA_STDDEV,
) -> bool:
    """True if `loss_history` (in logged order) shows no NaN/Inf and the
    stddev of consecutive deltas over its last `window` values stays within
    `max_delta_stddev`. False for an empty history, or one shorter than
    `window` -- too few logged epochs to judge stability from (a single
    delta's population stddev is trivially 0 regardless of its size, so a
    short history must not be treated as automatically stable).
    """
    if not loss_history:
        return False
    if not all(math.isfinite(value) for value in loss_history):
        return False
    if len(loss_history) < window:
        return False

    recent = loss_history[-window:]
    deltas = [b - a for a, b in zip(recent, recent[1:])]
    if not deltas:
        return True
    return statistics.pstdev(deltas) <= max_delta_stddev


def evaluate_staging(metrics: Dict[str, float], val_loss_history: List[float]) -> PromotionDecision:
    """Evaluates a run against Staging's thresholds plus loss-stability."""
    reasons = check_thresholds(metrics, STAGING_THRESHOLDS)
    if not is_loss_history_stable(val_loss_history):
        reasons.append(
            "val_loss history is not stable (see promotion_criteria.is_loss_history_stable)"
        )
    return PromotionDecision(promote=not reasons, reasons=reasons)


def evaluate_production(metrics: Dict[str, float]) -> PromotionDecision:
    """Evaluates a run against Production's (held-out test-set) thresholds."""
    reasons = check_thresholds(metrics, PRODUCTION_THRESHOLDS)
    return PromotionDecision(promote=not reasons, reasons=reasons)
