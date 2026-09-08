"""Put baseline serving and its shared dependencies on ``sys.path``."""

import sys
from pathlib import Path

_BASELINE_SERVING_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR = _BASELINE_SERVING_DIR.parents[2]

for _dir in (
    _BASELINE_SERVING_DIR,
    _SRC_DIR / "serving",
    _SRC_DIR / "registry",
):
    _dir_str = str(_dir)
    if _dir_str not in sys.path:
        sys.path.insert(0, _dir_str)
