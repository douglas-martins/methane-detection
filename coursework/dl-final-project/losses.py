"""Loss function and non-degeneracy check for the DL course final project (plan Section 6).

`pos_weight` is computed from whichever exact patches a given run actually
trains on (mini's own train split, raw's full train split, or an R2
subsample), not fixed to one tier's number -- `docs/dataset_report.md` §7
explicitly recommends against reusing `starcop_mini`'s imbalance ratio for
`starcop_raw` training, since the two differ by ~3.6x (Section 3). This is
data-dependent normalization, not one of the hyperparameters Section 7
requires held fixed across tiers (optimizer, LR, architecture, ...).
"""

import pandas as pd
import torch


def compute_pos_weight(patches_df: pd.DataFrame) -> float:
    """Return background:positive pixel ratio from `frac_positives`, for `BCEWithLogitsLoss`.

    All patches are the same size, so the plain mean of each patch's own
    positive-pixel fraction equals the overall fraction across the whole
    set -- no need to re-read label rasters; `patch_extract.py` already
    computed `frac_positives` once per patch.
    """
    positive_fraction = patches_df["frac_positives"].mean()
    if positive_fraction == 0:
        raise ValueError("patches_df has no positive pixels at all; pos_weight is undefined")
    return (1 - positive_fraction) / positive_fraction


def build_loss(pos_weight: float) -> torch.nn.BCEWithLogitsLoss:
    """Build `BCEWithLogitsLoss` weighted by `pos_weight` (background:positive ratio)."""
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))


def is_degenerate(sigmoid_predictions: torch.Tensor, epsilon: float = 1e-6) -> bool:
    """True if `sigmoid_predictions` (post-sigmoid, in [0, 1]) collapsed to all-0 or all-1.

    The validation checklist's explicit worry (Section 6): a model that
    gives up and predicts every pixel negative (or, less likely but still
    checked, every pixel positive) given the severe class imbalance.
    """
    return bool((sigmoid_predictions < epsilon).all() or (sigmoid_predictions > 1 - epsilon).all())
