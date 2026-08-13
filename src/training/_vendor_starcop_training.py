"""Single seam for importing vendor/starcop's training-time modules.

Mirrors src/data/preprocessing/_vendor_starcop.py's approach (same idea,
separate module -- pytest's flat/prepend import mode caches by bare module
name, and both src/data/preprocessing/ and src/training/ end up on sys.path
simultaneously under `make test-env-b`, so two files literally named
_vendor_starcop.py collide in sys.modules). Puts vendor/starcop on sys.path
and re-exports the real, unmodified objects this package composes around
(see mlops-methane-detection-plan.md TASK-2.2 decision 0: nothing under
vendor/starcop/ is ever edited, so every new module here imports the real
STARCOP code through this one seam instead of duplicating it).
"""

import sys
from pathlib import Path

_VENDOR_STARCOP = Path(__file__).resolve().parents[2] / "vendor" / "starcop"
if str(_VENDOR_STARCOP) not in sys.path:
    sys.path.insert(0, str(_VENDOR_STARCOP))

from starcop import metrics as starcop_metrics  # noqa: E402
from starcop.data import feature_extration  # noqa: E402
from starcop.data.data_logger import ImageLogger  # noqa: E402
from starcop.data.datamodule import Permian2019DataModule, add_sample_weight  # noqa: E402
from starcop.data.dataset import STARCOPDataset  # noqa: E402
from starcop.dataset_setup import get_dataset  # noqa: E402
from starcop.model_setup import get_model  # noqa: E402
from starcop.validation import run_validation  # noqa: E402

__all__ = [
    "starcop_metrics",
    "feature_extration",
    "ImageLogger",
    "STARCOPDataset",
    "Permian2019DataModule",
    "add_sample_weight",
    "get_dataset",
    "get_model",
    "run_validation",
]
