"""Single seam for importing vendor/starcop's evaluation-time modules.

Mirrors src/training/_vendor_starcop_training.py's approach (own file per
consuming package, not a shared import -- pytest's flat/prepend import mode
caches by bare module name, and multiple packages can end up on sys.path
simultaneously under `make test-env-b`). Puts vendor/starcop on sys.path and
re-exports the real, unmodified objects this package composes around (see
mlops-methane-detection-plan.md TASK-2.2 decision 0 / track-a plan's
[[feedback-vendor-starcop-composition-only]]: nothing under vendor/starcop/
is ever edited).
"""

import sys
from pathlib import Path

_VENDOR_STARCOP = Path(__file__).resolve().parents[2] / "vendor" / "starcop"
if str(_VENDOR_STARCOP) not in sys.path:
    sys.path.insert(0, str(_VENDOR_STARCOP))

import starcop.metrics as starcop_metrics  # noqa: E402
import starcop.plot as starcop_plot  # noqa: E402
from starcop.data import feature_extration  # noqa: E402
from starcop.data.dataset import STARCOPDataset  # noqa: E402
from starcop.torch_utils import to_device  # noqa: E402
from starcop.validation import run_validation  # noqa: E402

__all__ = [
    "starcop_metrics",
    "starcop_plot",
    "feature_extration",
    "STARCOPDataset",
    "to_device",
    "run_validation",
]
