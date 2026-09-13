"""The training-curve metric for the DL course final project (plan Section 6).

Pixel-level F1 -- the course's own imbalance caveat (PDF Section 8.1)
argues against Accuracy as the headline number even here, at the
"log training curves (loss, chosen metric)" stage, not only in Section 8's
full evaluation. Section 8 owns the full metric suite (Precision/Recall/
F1/PR-AUC/confusion matrix); this is only the one curve Section 6 logs
during training.
"""

import torch


def pixel_f1(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Pixel-level F1 between binary `predictions` and `targets` (any matching shape).

    Returns 0.0, not NaN, when there are no predicted and no true positives
    -- the degenerate all-negative case this section explicitly checks for.
    """
    true_positive = (predictions * targets).sum()
    false_positive = (predictions * (1 - targets)).sum()
    false_negative = ((1 - predictions) * targets).sum()

    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return (2 * true_positive / denominator).item()
