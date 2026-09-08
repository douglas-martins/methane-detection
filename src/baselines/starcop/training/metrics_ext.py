"""Extra confusion-matrix metrics that compose around starcop.metrics
without editing it (see mlops-methane-detection-plan.md TASK-2.2 decision 0:
vendor/starcop/ is never modified, not even for this).
"""

from typing import Dict

from _vendor_starcop_training import starcop_metrics

Tensor = starcop_metrics.Tensor


def f1score_background(cm: Tensor) -> float:
    """F1 of the background (negative) class -- mirrors starcop.metrics.f1score,
    computed on the row/column-swapped confusion matrix (background treated
    as the positive class instead of methane).
    """
    swapped = cm[[1, 0]][:, [1, 0]]
    return starcop_metrics.f1score(swapped)


def compute_all(cm: Tensor) -> Dict[str, float]:
    """Runs starcop.metrics.METRICS_CONFUSION_MATRIX (unmodified) plus
    f1score_background against cm, returning a {name: value} dict.
    """
    result = {fn.__name__: fn(cm) for fn in starcop_metrics.METRICS_CONFUSION_MATRIX}
    result["f1score_background"] = f1score_background(cm)
    return result
