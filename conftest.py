"""Project-wide pytest fixtures, shared by every ``__tests__/`` folder.

Kept at the repo root (rather than under ``tests/``) so fixtures are visible
to both co-located suites (e.g. ``src/data/__tests__/``) and the mirrored
``tests/vendor_starcop/`` suite, which live in separate subtrees.

Fixtures that need the STARCOP package (Environment A only) import it lazily
inside the fixture body via ``pytest.importorskip``, so this file can still
be collected without error by a pytest run under Environment B, which does
not have ``starcop`` installed.
"""

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def fake_zip_factory():
    """Build a zip file with configurable internal layout.

    Usage:
        fake_zip_factory(tmp_path / "asset.zip", {"a.txt": b"hi"})
        fake_zip_factory(tmp_path / "asset.zip", {"a.txt": b"hi"}, nested_under="STARCOP_mini")
    """

    def _make(zip_path: Path, files: dict[str, bytes], nested_under: str | None = None) -> Path:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in files.items():
                arcname = f"{nested_under}/{name}" if nested_under else name
                zf.writestr(arcname, content)
        return zip_path

    return _make


@pytest.fixture
def starcop_module_loader():
    """Load a STARCOP script by absolute file path.

    ``vendor/starcop/scripts/**`` has no ``__init__.py`` (it isn't a
    package), so its scripts can't be imported normally. This loads a file
    directly via importlib, without touching the submodule or sys.path.
    """

    def _load(path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    return _load


@pytest.fixture
def mock_process_aviris(monkeypatch):
    """Stub out every external-boundary call ``starcop_aviris_data_prep.main`` makes.

    Covers network download, mag1c subprocess execution, and GCS/local
    cleanup — real calls that are slow, non-deterministic, or require
    external assets, so they're mocked rather than exercised for real.
    Returns a namespace of call-recording Mocks for assertions.
    """
    process_aviris = pytest.importorskip("starcop.process_aviris")
    starcop_utils = pytest.importorskip("starcop.utils")

    mocks = SimpleNamespace(
        download_aviris=Mock(return_value=("fake.tar.gz", "fake_aviris_folder")),
        save_aviris_cog=Mock(),
        run_mag1c=Mock(),
        aviris_as_sensor=Mock(),
        remove_folder=Mock(),
    )
    monkeypatch.setattr(process_aviris, "download_aviris", mocks.download_aviris)
    monkeypatch.setattr(process_aviris, "save_aviris_cog", mocks.save_aviris_cog)
    monkeypatch.setattr(process_aviris, "run_mag1c", mocks.run_mag1c)
    monkeypatch.setattr(process_aviris, "aviris_as_sensor", mocks.aviris_as_sensor)
    monkeypatch.setattr(starcop_utils, "remove_folder", mocks.remove_folder)
    return mocks
