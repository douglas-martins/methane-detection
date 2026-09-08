"""Put baseline and shared registry modules on ``sys.path`` for tests."""

import sys
from pathlib import Path

_BASELINE_REGISTRY_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = Path(__file__).resolve().parents[5]
_SHARED_REGISTRY_DIR = _REPO_ROOT / "src" / "registry"
_SHARED_TRAINING_DIR = _REPO_ROOT / "src" / "training"

for _path in (_BASELINE_REGISTRY_DIR, _SHARED_REGISTRY_DIR, _SHARED_TRAINING_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
