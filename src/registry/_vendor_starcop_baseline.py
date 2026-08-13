"""Single seam for importing vendor/starcop's model-construction classes into
src/registry/.

Mirrors src/training/_vendor_starcop_training.py's approach (own file per
consuming package, not a shared import, since pytest's flat/prepend import
mode caches by bare module name and multiple packages can end up on
sys.path simultaneously -- see that file's docstring). Puts vendor/starcop
on sys.path and re-exports the real, unmodified classes hf_baseline_import.py
composes around (nothing under vendor/starcop/ is ever edited).
"""

import sys
from pathlib import Path

_VENDOR_STARCOP = Path(__file__).resolve().parents[2] / "vendor" / "starcop"
if str(_VENDOR_STARCOP) not in sys.path:
    sys.path.insert(0, str(_VENDOR_STARCOP))

from starcop.models.model_module import ModelModule  # noqa: E402
from starcop.models.model_module_regression import ModelModuleRegression  # noqa: E402

__all__ = ["ModelModule", "ModelModuleRegression"]
