"""Puts src/data/ on sys.path so tests can import download_mini_dataset directly.

Needed because this test file lives one directory below its source module
(src/data/__tests__/ vs src/data/) — pytest's default import mode only adds
the test file's own directory to sys.path, not its parent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
