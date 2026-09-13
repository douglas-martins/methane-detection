"""Input normalization for the DL course final project (plan Section 4).

Reimplements STARCOP's own fixed offset/factor/clip contract
(`vendor/starcop/starcop/data/normalizer_module.py::BAND_NORMALIZATION`,
`DataNormalizer.normalize_x`) as a standalone, coursework-local pure
function rather than importing the vendor module -- this coursework stays
physically isolated from `vendor/starcop`/`src/` (plan Section 0), the
same independence choice Section 5 makes for E2/E3. The canonical version
already lives in `vendor/starcop`; this is not meant to be promoted back
into `src/` (see this section's "Reuse in this project" note).

Deliberately covers *inputs only* -- `labelbinary` has no
`BAND_NORMALIZATION` entry upstream (`DataNormalizer.normalize_y` is a
pass-through when `factors_output` is None), so the label is never run
through this function; callers keep it as raw 0/1.
"""

import numpy as np

BAND_NORMALIZATION = {
    "mag1c": {"offset": 0, "factor": 1750, "clip": (0, 2)},
    "TOA_AVIRIS_640nm": {"offset": 0, "factor": 60, "clip": (0, 2)},
    "TOA_AVIRIS_550nm": {"offset": 0, "factor": 60, "clip": (0, 2)},
    "TOA_AVIRIS_460nm": {"offset": 0, "factor": 60, "clip": (0, 2)},
}


def normalize_band(
    array: np.ndarray, offset: float, factor: float, clip: tuple[float, float]
) -> np.ndarray:
    """Apply STARCOP's fixed (array - offset) / factor, clipped to `clip`.

    Same formula `DataNormalizer.normalize_x` applies via `torch.clamp` --
    this is why the real `mag1c` sentinel value (~100000, confirmed during
    Section 3's raw run) is not a special case: at this factor it already
    lands far outside `clip` and gets clamped like any other out-of-range
    value, regardless of its raw magnitude.
    """
    clip_min, clip_max = clip
    scaled = (array.astype("float32") - offset) / factor
    return np.clip(scaled, clip_min, clip_max).astype("float32")
