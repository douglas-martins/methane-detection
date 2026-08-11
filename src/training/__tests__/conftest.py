"""Puts src/training/ on sys.path so tests can import modules directly.

Mirrors src/data/preprocessing/__tests__/conftest.py -- pytest's default
import mode only adds the test file's own directory to sys.path, not its
parent, so this test file needs to add it itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
