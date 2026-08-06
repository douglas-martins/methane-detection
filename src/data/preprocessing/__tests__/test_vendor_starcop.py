"""Tests for src/data/preprocessing/_vendor_starcop.py.

The shim's only job is to make vendor/starcop's pure numpy/pandas/rasterio/
torch data-handling modules importable from Environment B without a
sys.path-independent install. This test proves it reaches the *real*
vendored module (not a stub) by asserting on a known constant.
"""

import _vendor_starcop


def test_shim_exposes_real_starcop_symbols():
    assert _vendor_starcop.BAND_NORMALIZATION["mag1c"] == {
        "offset": 0,
        "factor": 1750,
        "clip": (0, 2),
    }
    assert _vendor_starcop.STARCOPDataset.__module__ == "starcop.data.dataset"
    assert callable(_vendor_starcop.tiled_dataframe)
