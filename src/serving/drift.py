"""Rolling per-band statistics and Gaussian KL divergence for input data
drift detection (TASK-6.2).

Tracks whether incoming scenes' per-band value distribution has drifted
from the training baseline (band_baseline.py), using a parametric
(Gaussian) comparison rather than a full histogram -- consistent with
tracking mean+std per band rather than raw per-pixel distributions.
"""

import math
from collections import deque

from band_baseline import BandStats

# Floor for the rolling window's std in the KL divergence's ln(sigma_q /
# sigma_p) term -- a short window of near-identical values can compute
# sigma_p == 0.0, which would otherwise divide by zero. Clamping instead
# of raising still reports a large, clearly-anomalous divergence rather
# than crashing the request that triggered it.
_MIN_SIGMA = 1e-6


def update_rolling_stats(window: deque, value: float) -> BandStats:
    """Appends `value` to `window` (mutates it in place -- the caller owns
    the deque's lifecycle and maxlen) and returns the (mean, std) of its
    current contents. std is the population std (ddof=0): this is a
    rolling descriptive statistic, not an estimate of a larger population.
    """
    window.append(value)
    n = len(window)
    mean = sum(window) / n
    variance = sum((x - mean) ** 2 for x in window) / n
    return BandStats(mean=mean, std=math.sqrt(variance))


def kl_divergence_gaussian(mu_p: float, sigma_p: float, mu_q: float, sigma_q: float) -> float:
    """Closed-form KL(P||Q) between two univariate Gaussians N(mu_p,
    sigma_p^2) and N(mu_q, sigma_q^2), where P is the rolling (current)
    distribution and Q is the training baseline.
    """
    sigma_p = max(sigma_p, _MIN_SIGMA)
    return math.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2) - 0.5


def check_drift(rolling: BandStats, baseline: BandStats, threshold: float = 0.5) -> bool:
    """True if the rolling distribution's KL divergence from the baseline
    exceeds `threshold`."""
    divergence = kl_divergence_gaussian(rolling.mean, rolling.std, baseline.mean, baseline.std)
    return divergence > threshold
