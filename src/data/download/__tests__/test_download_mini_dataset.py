"""Tests for src/data/download_mini_dataset.py.

Mocks only the true external boundary (gdown.download — a network call);
extraction/flattening is exercised against real temp-directory filesystem
state via tmp_path and the fake_zip_factory fixture (conftest.py, repo root).
"""

from pathlib import Path
from unittest.mock import Mock

import download_mini_dataset


def test_creates_target_dir_if_missing(tmp_path, fake_zip_factory, monkeypatch):
    """target_dir doesn't need to exist beforehand -- download_and_extract creates it."""
    target_dir = tmp_path / "does_not_exist_yet"

    def fake_download(id, output, quiet):
        """Stand in for gdown.download: write a fake zip instead of hitting the network."""
        fake_zip_factory(Path(output), {"file.txt": b"data"})

    monkeypatch.setattr(download_mini_dataset.gdown, "download", fake_download)

    download_mini_dataset.download_and_extract("fake-id", "asset.zip", target_dir)

    assert target_dir.is_dir()


def test_skips_download_if_zip_already_exists(tmp_path, fake_zip_factory, monkeypatch):
    """A zip already sitting in target_dir is reused, not re-downloaded."""
    zip_path = tmp_path / "asset.zip"
    fake_zip_factory(zip_path, {"file.txt": b"data"})
    download_calls = Mock()
    monkeypatch.setattr(download_mini_dataset.gdown, "download", download_calls)

    download_mini_dataset.download_and_extract("fake-id", "asset.zip", tmp_path)

    download_calls.assert_not_called()


def test_downloads_when_zip_missing(tmp_path, fake_zip_factory, monkeypatch):
    """No zip in target_dir -- gdown.download is called with the right id/output path."""
    download_calls = Mock(
        side_effect=lambda id, output, quiet: fake_zip_factory(Path(output), {"file.txt": b"data"})
    )
    monkeypatch.setattr(download_mini_dataset.gdown, "download", download_calls)

    download_mini_dataset.download_and_extract("fake-id-123", "asset.zip", tmp_path)

    download_calls.assert_called_once_with(
        id="fake-id-123", output=str(tmp_path / "asset.zip"), quiet=False
    )


def test_extracts_zip_contents_into_target_dir(tmp_path, fake_zip_factory, monkeypatch):
    """A downloaded zip's contents land in target_dir once extraction completes."""
    zip_path = tmp_path / "asset.zip"
    fake_zip_factory(zip_path, {"scene.csv": b"a,b,c"})
    monkeypatch.setattr(download_mini_dataset.gdown, "download", Mock())

    download_mini_dataset.download_and_extract("fake-id", "asset.zip", tmp_path)

    assert (tmp_path / "scene.csv").read_bytes() == b"a,b,c"


def test_deletes_zip_after_extraction(tmp_path, fake_zip_factory, monkeypatch):
    """The zip itself is removed once its contents are extracted."""
    zip_path = tmp_path / "asset.zip"
    fake_zip_factory(zip_path, {"scene.csv": b"a,b,c"})
    monkeypatch.setattr(download_mini_dataset.gdown, "download", Mock())

    download_mini_dataset.download_and_extract("fake-id", "asset.zip", tmp_path)

    assert not zip_path.exists()


def test_flattens_nested_folder_matching_zip_stem(tmp_path, fake_zip_factory, monkeypatch):
    """Regression test: STARCOP_mini.zip wraps its contents in a top-level
    STARCOP_mini/ folder. Extracting naively used to leave data nested at
    data/starcop_mini/STARCOP_mini/... instead of flat under data/starcop_mini/.
    """
    zip_path = tmp_path / "STARCOP_mini.zip"
    fake_zip_factory(
        zip_path,
        {"train_mini10.csv": b"a,b", "scene1/rgb.tif": b"binarydata"},
        nested_under="STARCOP_mini",
    )
    monkeypatch.setattr(download_mini_dataset.gdown, "download", Mock())

    download_mini_dataset.download_and_extract("fake-id", "STARCOP_mini.zip", tmp_path)

    assert (tmp_path / "train_mini10.csv").read_bytes() == b"a,b"
    assert (tmp_path / "scene1" / "rgb.tif").read_bytes() == b"binarydata"
    assert not (tmp_path / "STARCOP_mini").exists()


def test_no_flatten_when_no_nested_folder_present(tmp_path, fake_zip_factory, monkeypatch):
    """A zip without a top-level folder matching its own stem is left as-is (no flattening applied)."""
    zip_path = tmp_path / "checkpoint.zip"
    fake_zip_factory(zip_path, {"model.pt": b"weights"})
    monkeypatch.setattr(download_mini_dataset.gdown, "download", Mock())

    download_mini_dataset.download_and_extract("fake-id", "checkpoint.zip", tmp_path)

    assert (tmp_path / "model.pt").read_bytes() == b"weights"
    assert not (tmp_path / "checkpoint").exists()


def test_flatten_preserves_preexisting_files_in_target_dir(tmp_path, fake_zip_factory, monkeypatch):
    """Flattening a nested zip doesn't disturb files already present in target_dir."""
    (tmp_path / "already_here.txt").write_text("keep me")
    zip_path = tmp_path / "STARCOP_mini.zip"
    fake_zip_factory(zip_path, {"new_file.csv": b"x"}, nested_under="STARCOP_mini")
    monkeypatch.setattr(download_mini_dataset.gdown, "download", Mock())

    download_mini_dataset.download_and_extract("fake-id", "STARCOP_mini.zip", tmp_path)

    assert (tmp_path / "already_here.txt").read_text() == "keep me"
    assert (tmp_path / "new_file.csv").read_bytes() == b"x"


def test_main_calls_download_and_extract_for_each_asset(monkeypatch):
    """main() dispatches download_and_extract once per entry in ASSETS, in order."""
    calls = []
    monkeypatch.setattr(
        download_mini_dataset,
        "download_and_extract",
        lambda file_id, filename, target_dir: calls.append((file_id, filename, target_dir)),
    )

    download_mini_dataset.main()

    assert calls == download_mini_dataset.ASSETS
