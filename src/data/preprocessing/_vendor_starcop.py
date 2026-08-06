"""Single seam for importing vendor/starcop's data-handling modules.

vendor/starcop is pinned to a separate uv environment (Python 3.10 / torch
1.13.1, "Environment A") for exact reproducibility of the original paper's
checkpoints (TASK-0.3). That pin is unrelated to the pure numpy/pandas/
rasterio/torch data-handling modules re-exported below -- their imports
(warnings, numpy, torch, pandas, rasterio, and pytorch_lightning/kornia via
Environment B's `lightning` wheel, which ships a pytorch_lightning compat
shim) are already satisfied by this project's own environment ("Environment
B"). So instead of duplicating STARCOP's normalization table or tiling
logic, this module puts vendor/starcop on sys.path and re-exports the real
objects. Every other file in src/data/preprocessing/ imports from here,
never from vendor.starcop directly, so this sys.path seam lives in exactly
one place.

If a future `lightning` release drops the pytorch_lightning compat shim,
`tiled_dataframe` becomes unimportable here; the fallback is calling
`georeader.slices.create_windows` directly, the primitive it wraps.
"""

import sys
from pathlib import Path

_VENDOR_STARCOP = Path(__file__).resolve().parents[3] / "vendor" / "starcop"
if str(_VENDOR_STARCOP) not in sys.path:
    sys.path.insert(0, str(_VENDOR_STARCOP))

from starcop.data.datamodule import tiled_dataframe  # noqa: E402
from starcop.data.dataset import STARCOPDataset  # noqa: E402
from starcop.data.normalizer_module import BAND_NORMALIZATION  # noqa: E402

__all__ = ["BAND_NORMALIZATION", "STARCOPDataset", "tiled_dataframe"]
