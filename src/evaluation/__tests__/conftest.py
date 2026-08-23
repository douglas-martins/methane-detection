"""Puts src/evaluation/, src/registry/, and src/training/ on sys.path so
tests can import modules directly, regardless of which order a test file
happens to import them in.

Mirrors src/training/__tests__/conftest.py -- pytest's default import mode
only adds the test file's own directory to sys.path, not its parent.

The registry/training paths are here (not left to import-order side
effects inside e.g. paper_eval_mlflow.py, which does its own sys.path
insertion when imported) because relying on "import module X before
module Y so X's sys.path insertion runs first" is fragile: a real
regression (2026-08-22) had `ruff --fix` alphabetize
`test_paper_eval_mlflow.py`'s imports, silently breaking that exact
ordering and making the test file fail when run standalone (it only kept
passing as part of the full suite because a *different* file's conftest
happened to already prime sys.path first). Setting these paths once here,
independent of any test file's import order, removes the whole class of
bug rather than just this one instance.
"""

import sys
from pathlib import Path

_EVALUATION_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR = _EVALUATION_DIR.parent

for _dir in (_EVALUATION_DIR, _SRC_DIR / "registry", _SRC_DIR / "training"):
    _dir_str = str(_dir)
    if _dir_str not in sys.path:
        sys.path.insert(0, _dir_str)
