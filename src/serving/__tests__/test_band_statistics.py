"""Tests for the model-agnostic per-band statistics value type."""

import subprocess
import sys
from pathlib import Path

from band_statistics import BandStats

SERVING_DIR = Path(__file__).resolve().parent.parent


def test_band_stats_preserves_named_tuple_behavior():
    stats = BandStats(mean=1.25, std=0.5)

    assert stats.mean == 1.25
    assert stats.std == 0.5
    assert tuple(stats) == (1.25, 0.5)


def test_shared_drift_import_does_not_load_starcop_band_baseline(tmp_path):
    code = f"""
import sys
sys.path.insert(0, {str(SERVING_DIR)!r})
import drift
assert 'band_baseline' not in sys.modules
assert drift.BandStats.__module__ == 'band_statistics'
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
