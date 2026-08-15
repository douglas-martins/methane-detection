"""Single seam for putting vendor/starcop on sys.path for src/serving/.

Own file per consuming package (mirrors src/registry/_vendor_starcop_baseline.py
and src/training/_vendor_starcop_training.py) rather than a shared import --
pytest's flat/prepend import mode caches by bare module name, and multiple
packages can end up on sys.path simultaneously.

Importing this module is enough to put vendor/starcop on sys.path. That's
required before mlflow.pytorch.load_model() can unpickle a real STARCOP
checkpoint (ModelModule/ModelModuleRegression), whose class path
(starcop.models.model_module.*) must be importable at unpickle time -- the
same requirement src/registry/hf_baseline_import.py already has before its
own torch.load() call. Nothing under vendor/starcop/ is ever edited.
"""

import sys
from pathlib import Path

_VENDOR_STARCOP = Path(__file__).resolve().parents[2] / "vendor" / "starcop"
if str(_VENDOR_STARCOP) not in sys.path:
    sys.path.insert(0, str(_VENDOR_STARCOP))

from starcop.models.model_module import ModelModule  # noqa: E402
from starcop.models.model_module_regression import ModelModuleRegression  # noqa: E402

__all__ = ["ModelModule", "ModelModuleRegression"]
