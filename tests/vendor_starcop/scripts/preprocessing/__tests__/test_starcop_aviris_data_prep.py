"""Tests for vendor/starcop/scripts/preprocessing/starcop_aviris_data_prep.py.

The submodule is never modified in place, so this suite lives at a mirrored
path under tests/ instead — see conftest.py (repo root) for how the target
module gets loaded by file path and how process_aviris/utils get mocked.

main() is the only testable logic in the file: the two NAME_FILES_* lists
are static data, and the CLI block is __main__-guarded. All four
process_aviris entry points (download, COG conversion, mag1c, S2 simulation)
do real network/subprocess/GDAL work, so they're mocked (mock_process_aviris)
— the skip/process/cleanup branching around them is exercised for real
against tmp_path.
"""

import json
import re
from pathlib import Path

import pytest

# tests/vendor_starcop/scripts/preprocessing/__tests__/test_x.py
#   parents[0]=__tests__ [1]=preprocessing [2]=scripts [3]=vendor_starcop [4]=tests [5]=repo root
REPO_ROOT = Path(__file__).resolve().parents[5]
MODULE_PATH = REPO_ROOT / "vendor" / "starcop" / "scripts" / "preprocessing" / "starcop_aviris_data_prep.py"

AVIRIS_ID_PATTERN = re.compile(r"^ang\d{8}t\d{6}$")


@pytest.fixture
def module(starcop_module_loader):
    return starcop_module_loader(MODULE_PATH, "starcop_aviris_data_prep_test_target")


def _write_complete_product_folder(folder: Path, bands_sensor: dict, n_wavelengths: int = 3) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "mag1c.tif").write_bytes(b"x")
    (folder / "albedo.tif").write_bytes(b"x")
    (folder / "glt.tif").write_bytes(b"x")
    (folder / "metadata.json").write_text(json.dumps({"wavelengths": list(range(n_wavelengths))}))
    for i in range(n_wavelengths):
        (folder / f"{i}.tif").write_bytes(b"x")
    for sensor, bands in bands_sensor.items():
        for band in bands:
            (folder / f"{sensor}_{band}.tif").write_bytes(b"x")


def test_skips_all_processing_when_every_product_already_exists(tmp_path, module, mock_process_aviris):
    name = "ang20191005t221554"
    folder_dest = tmp_path / name
    _write_complete_product_folder(folder_dest, module.process_aviris.BANDS_SENSOR)
    geotiff_base = tmp_path / "geotiff"

    module.main(
        idx=0,
        name=name,
        n_images=1,
        path_save_images=str(tmp_path),
        path_untar_folder_base=str(tmp_path / "untar"),
        path_geotiff_base=str(geotiff_base),
    )

    mock_process_aviris.download_aviris.assert_not_called()
    mock_process_aviris.save_aviris_cog.assert_not_called()
    mock_process_aviris.run_mag1c.assert_not_called()
    mock_process_aviris.aviris_as_sensor.assert_not_called()
    assert not (geotiff_base / name).exists()


def test_runs_full_pipeline_when_nothing_exists(tmp_path, module, mock_process_aviris):
    name = "ang20191005t221554"
    geotiff_base = tmp_path / "geotiff"

    module.main(
        idx=0,
        name=name,
        n_images=1,
        path_save_images=str(tmp_path),
        path_untar_folder_base=str(tmp_path / "untar"),
        path_geotiff_base=str(geotiff_base),
    )

    folder_dest = tmp_path / name
    mock_process_aviris.download_aviris.assert_called_once()
    mock_process_aviris.save_aviris_cog.assert_called_once()

    run_mag1c_kwargs = mock_process_aviris.run_mag1c.call_args.kwargs
    assert run_mag1c_kwargs["mf_filename"] == str(folder_dest / "mag1c.tif")
    assert run_mag1c_kwargs["albedo_filename"] == str(folder_dest / "albedo.tif")
    assert run_mag1c_kwargs["glt_filename"] == str(folder_dest / "glt.tif")

    mock_process_aviris.aviris_as_sensor.assert_called_once()
    assert (geotiff_base / name).is_dir()


def test_runs_s2_simulation_only_when_mag1c_products_exist_but_bands_missing(
    tmp_path, module, mock_process_aviris
):
    name = "ang20191005t221554"
    folder_dest = tmp_path / name
    folder_dest.mkdir(parents=True)
    (folder_dest / "mag1c.tif").write_bytes(b"x")
    (folder_dest / "albedo.tif").write_bytes(b"x")
    (folder_dest / "glt.tif").write_bytes(b"x")
    (folder_dest / "metadata.json").write_text(json.dumps({"wavelengths": [0, 1, 2]}))
    for i in range(3):
        (folder_dest / f"{i}.tif").write_bytes(b"x")
    # deliberately omit the S2A/S2B/WV3 band tifs

    module.main(
        idx=0,
        name=name,
        n_images=1,
        path_save_images=str(tmp_path),
        path_untar_folder_base=str(tmp_path / "untar"),
        path_geotiff_base=str(tmp_path / "geotiff"),
    )

    mock_process_aviris.download_aviris.assert_not_called()
    mock_process_aviris.save_aviris_cog.assert_not_called()
    mock_process_aviris.run_mag1c.assert_not_called()
    mock_process_aviris.aviris_as_sensor.assert_called_once()

    call_args = mock_process_aviris.aviris_as_sensor.call_args.args
    # aviris_folder was never set by a download (skipped), so it falls back
    # to folder_dest_bucket instead of staying None.
    assert call_args[0] == str(folder_dest)
    assert call_args[1] == str(folder_dest)


def test_removes_temp_folders_by_default(tmp_path, module, mock_process_aviris):
    name = "ang20191005t221554"
    geotiff_base = tmp_path / "geotiff"

    module.main(
        idx=0,
        name=name,
        n_images=1,
        path_save_images=str(tmp_path),
        path_untar_folder_base=str(tmp_path / "untar"),
        path_geotiff_base=str(geotiff_base),
    )

    removed = [call.args[0] for call in mock_process_aviris.remove_folder.call_args_list]
    assert "fake_aviris_folder" in removed
    assert str(geotiff_base / name) in removed


def test_keeps_temp_folders_when_flags_false(tmp_path, module, mock_process_aviris):
    name = "ang20191005t221554"

    module.main(
        idx=0,
        name=name,
        n_images=1,
        path_save_images=str(tmp_path),
        path_untar_folder_base=str(tmp_path / "untar"),
        path_geotiff_base=str(tmp_path / "geotiff"),
        remove_untar_file=False,
        remove_path_tiffs_temp=False,
    )

    mock_process_aviris.remove_folder.assert_not_called()


def test_name_files_permian_has_no_duplicates(module):
    assert len(module.NAME_FILES_PERMIAN) == len(set(module.NAME_FILES_PERMIAN))


def test_name_files_jpl_4corners_has_no_duplicates(module):
    assert len(module.NAME_FILES_JPL_4CORNERS) == len(set(module.NAME_FILES_JPL_4CORNERS))


def test_name_files_entries_match_aviris_id_pattern(module):
    for name in module.NAME_FILES_PERMIAN + module.NAME_FILES_JPL_4CORNERS:
        assert AVIRIS_ID_PATTERN.match(name), name
