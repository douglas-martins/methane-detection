"""Puts coursework/dl-final-project/ on sys.path so tests can import its modules directly.

Needed because this test file lives one directory below its source module
(coursework/dl-final-project/__tests__/ vs coursework/dl-final-project/) --
pytest's default import mode only adds the test file's own directory to
sys.path, not its parent. Deliberately its own copy rather than importing
from src/data/preprocessing/__tests__/conftest.py: this project stays
physically isolated from src/ (plan Section 0), and pytest's flat import
mode would otherwise risk colliding fixture module names across the two
test trees.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def _find_repo_root_containing_configs(start: Path) -> Path:
    """Climb from `start` until a directory containing `configs/dataset/` is found."""
    candidate = start
    while not (candidate / "configs" / "dataset").is_dir():
        if candidate.parent == candidate:
            raise RuntimeError(f"no configs/dataset/ found above {start}")
        candidate = candidate.parent
    return candidate


@pytest.fixture(autouse=True)
def _fix_dataset_repo_root_for_copied_trees():
    """Repatch `dataset._REPO_ROOT` to the real repo root, independent of copy depth.

    `dataset.py` computes its repo root via a fixed `Path(__file__).parents[2]`
    climb, which assumes it lives exactly two directories below the repo
    root. Any tool that copies this file to a different relative depth (e.g.
    mutmut's `mutants/` copy inserts one extra directory) breaks that fixed
    climb. Repatched here -- a test-only fix, no change to dataset.py's
    runtime behavior -- by walking up from this conftest's own (also-copied)
    location instead, which finds the real root regardless of copy depth.
    """
    if "dataset" not in sys.modules:
        yield
        return
    dataset_module = sys.modules["dataset"]
    original_repo_root = dataset_module._REPO_ROOT
    dataset_module._REPO_ROOT = _find_repo_root_containing_configs(Path(__file__).resolve().parent)
    yield
    dataset_module._REPO_ROOT = original_repo_root


@pytest.fixture
def tiny_geotiff_factory():
    """Write a tiny single-band GeoTIFF with a given pixel array.

    Usage:
        tiny_geotiff_factory(path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))
    """

    def _make(path: Path, array: np.ndarray) -> Path:
        """Write `array` as a single-band GeoTIFF at `path`, creating parent dirs as needed."""
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
