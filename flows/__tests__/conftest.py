"""Puts flows/ on sys.path so tests can import retrain directly.

Mirrors src/training/__tests__/conftest.py -- pytest's default import mode
only adds the test file's own directory to sys.path, not its parent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
