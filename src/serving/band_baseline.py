"""Per-band training-distribution baseline statistics for input drift
detection (TASK-6.2).

Checked-in constants, sourced from docs/dataset_report.md section 5
(starcop_raw, train-split patches) -- not parsed from the markdown at
runtime. starcop_raw is used as the baseline rather than starcop_mini:
mini is curated toward plume-heavy scenes (~87:1 imbalance vs. raw's
~314:1), so using it as "normal" would bias drift detection toward
flagging genuinely typical, plume-sparse traffic as anomalous.

ModelModule only exposes a channel *count* at runtime
(vendor/starcop/starcop/models/model_module.py's __init__ stores
self.num_channels = len(settings.dataset.input_products), never the band
names themselves), so band identity has no other source in the serving
layer -- MODEL_BAND_NAMES fills that gap.
"""

from typing import NamedTuple, Optional


class BandStats(NamedTuple):
    mean: float
    std: float


# Matches vendor/starcop/scripts/configs/config.yaml's dataset.input_products
# order for each known registered model (src/serving/service.py's MODEL_NAME).
MODEL_BAND_NAMES = {
    "starcop-baseline-mag1c-rgb": [
        "mag1c",
        "TOA_AVIRIS_640nm",
        "TOA_AVIRIS_550nm",
        "TOA_AVIRIS_460nm",
    ],
    "starcop-baseline-mag1c-only": ["mag1c"],
}

# docs/dataset_report.md section 5, starcop_raw table.
_STARCOP_RAW_BASELINE = {
    "mag1c": BandStats(mean=36.22, std=310.13),
    "TOA_AVIRIS_640nm": BandStats(mean=28.82, std=15.27),
    "TOA_AVIRIS_550nm": BandStats(mean=26.83, std=14.15),
    "TOA_AVIRIS_460nm": BandStats(mean=24.76, std=12.87),
}


def band_names_for_model(model_name: str, num_channels: int) -> list:
    """Returns the band-name list for `model_name`, in the same channel
    order the model itself uses.

    Falls back to generic band_0..N-1 labels if the model isn't in
    MODEL_BAND_NAMES, or if its known list's length doesn't match
    `num_channels` -- a mismatch means the known list is stale for
    whatever is actually loaded, and guessing would silently mislabel
    bands rather than fail safely.
    """
    names = MODEL_BAND_NAMES.get(model_name)
    if names is not None and len(names) == num_channels:
        return names
    return [f"band_{i}" for i in range(num_channels)]


def baseline_for_band(band_name: str) -> Optional[BandStats]:
    """Returns the starcop_raw training baseline for a band name, or None
    if the band isn't one of the known, named channels (e.g. a generic
    band_i fallback name from an unrecognized model)."""
    return _STARCOP_RAW_BASELINE.get(band_name)
