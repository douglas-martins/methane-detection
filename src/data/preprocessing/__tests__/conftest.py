"""Puts src/data/preprocessing/ on sys.path so tests can import stage modules directly.

Needed because this test file lives one directory below its source modules
(src/data/preprocessing/__tests__/ vs src/data/preprocessing/) -- pytest's
default import mode only adds the test file's own directory to sys.path, not
its parent. Mirrors src/data/download/__tests__/conftest.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def tiny_geotiff_factory():
    """Write a tiny single-band GeoTIFF with a given pixel array.

    Usage:
        tiny_geotiff_factory(path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))
    """

    def _make(path: Path, array: np.ndarray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs="EPSG:4326",
            transform=from_origin(0, 0, 1, 1),
        ) as dst:
            dst.write(array, 1)
        return path

    return _make
