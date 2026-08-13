"""Tests for src/training/starcop_datamodule.py.

_load_dataframe is the one testable unit extracted from
ProcessedDatasetDataModule.prepare_data() -- CSV reading + window-column
reconstruction + folder path resolution to absolute (Test Size: Small, real
tmp_path CSV fixtures, no mocking). ProcessedDatasetDataModule itself is thin
glue around it plus STARCOPDataset construction (both unmodified STARCOP
objects), exercised by the real end-to-end training run instead of a unit
test -- see mlops-methane-detection-plan.md TASK-2.2 step 4b.
"""

from pathlib import Path

import pandas as pd
import rasterio.windows
import starcop_datamodule as sdm


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    csv_path = tmp_path / "some_split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _row(**overrides) -> dict:
    row = {
        "id": "sceneA_r0_c0_w128_h128",
        "folder": "data/processed/starcop_mini/selected/sceneA",
        "window_col_off": 0,
        "window_row_off": 0,
        "window_width": 128,
        "window_height": 128,
        "has_plume": False,
    }
    row.update(overrides)
    return row


class TestLoadDataframe:
    def test_reconstructs_window_from_column_bounds(self, tmp_path):
        csv_path = _write_csv(tmp_path, [_row()])

        df = sdm._load_dataframe(csv_path, repo_root=tmp_path)

        window = df.loc["sceneA_r0_c0_w128_h128", "window"]
        assert window == rasterio.windows.Window(col_off=0, row_off=0, width=128, height=128)

    def test_sets_id_column_as_the_index(self, tmp_path):
        csv_path = _write_csv(tmp_path, [_row(id="rowX")])

        df = sdm._load_dataframe(csv_path, repo_root=tmp_path)

        assert "rowX" in df.index
        assert "id" not in df.columns

    def test_resolves_relative_folder_against_repo_root(self, tmp_path):
        csv_path = _write_csv(
            tmp_path, [_row(folder="data/processed/starcop_mini/selected/sceneA")]
        )

        df = sdm._load_dataframe(csv_path, repo_root=tmp_path)

        resolved = df.loc["sceneA_r0_c0_w128_h128", "folder"]
        assert resolved == str(tmp_path / "data/processed/starcop_mini/selected/sceneA")

    def test_leaves_an_already_absolute_folder_unchanged(self, tmp_path):
        absolute_folder = str(tmp_path / "elsewhere" / "sceneA")
        csv_path = _write_csv(tmp_path, [_row(folder=absolute_folder)])

        df = sdm._load_dataframe(csv_path, repo_root=tmp_path)

        assert df.loc["sceneA_r0_c0_w128_h128", "folder"] == absolute_folder
