"""Model-agnostic value type for per-band distribution statistics."""

from typing import NamedTuple


class BandStats(NamedTuple):
    """Mean and standard deviation for one input band."""

    mean: float
    std: float
