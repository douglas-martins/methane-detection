"""Tests for src/baselines/starcop/evaluation/dataset_wiring.py.

`_load_dataframe` is the one testable unit extracted from
`build_test_dataloader` -- CSV reading + window-column reconstruction +
`folder = root_folder/id` resolution (Test Size: Small, real tmp_path CSV
fixtures, no mocking). Unlike baseline training's
`starcop_datamodule.py::_load_dataframe` (which reads an existing,
already-absolute `folder` column from this project's processed CSVs), test.csv's `folder` column
is a stale absolute path from the original paper authors' machine and must
be ignored and rebuilt from `id`, per vendor's own
`Permian2019DataModule.load_dataframe` (`datamodule.py:104`) -- verified
against the real data in track-a-paper-benchmark-reproduction-plan.md
Phase 1: `data/starcop_raw/STARCOP_test/<id>` resolves for all 342 test.csv
ids.

`build_test_dataloader` itself (STARCOPDataset + DataLoader construction,
both unmodified STARCOP objects) is covered below with small fakes so its
argument forwarding and defaults remain mutation-testable without requiring
the full multi-GB dataset.
"""

from pathlib import Path

import dataset_wiring
import pandas as pd
import rasterio.windows


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    csv_path = tmp_path / "test.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _row(**overrides) -> dict:
    row = {
        "id": "ang20191018t141549_r19200_c0_w512_h512",
        # A stale absolute path from the original authors' machine -- must
        # be ignored, not read as-is.
        "folder": "/AVIRISNG/Permian2019/ang20191018t141549_r19200_c0_w512_h512",
        "window_col_off": 0,
        "window_row_off": 19200,
        "window_width": 512,
        "window_height": 512,
        "has_plume": True,
        "qplume": 1500.0,
    }
    row.update(overrides)
    return row


class TestLoadDataframe:
    def test_reconstructs_window_from_column_bounds(self, tmp_path):
        csv_path = _write_csv(tmp_path, [_row()])

        df = dataset_wiring._load_dataframe(csv_path, root_folder=tmp_path / "STARCOP_test")

        window = df.loc["ang20191018t141549_r19200_c0_w512_h512", "window"]
        assert window == rasterio.windows.Window(col_off=0, row_off=19200, width=512, height=512)

    def test_sets_id_column_as_the_index(self, tmp_path):
        csv_path = _write_csv(tmp_path, [_row(id="rowX")])

        df = dataset_wiring._load_dataframe(csv_path, root_folder=tmp_path / "STARCOP_test")

        assert "rowX" in df.index
        assert "id" not in df.columns

    def test_rebuilds_folder_from_root_folder_and_id_ignoring_the_csv_column(self, tmp_path):
        root_folder = tmp_path / "STARCOP_test"
        csv_path = _write_csv(
            tmp_path,
            [_row(id="sceneA", folder="/AVIRISNG/Permian2019/some_other_stale_path")],
        )

        df = dataset_wiring._load_dataframe(csv_path, root_folder=root_folder)

        assert df.loc["sceneA", "folder"] == str(root_folder / "sceneA")

    def test_preserves_other_columns_such_as_qplume(self, tmp_path):
        csv_path = _write_csv(tmp_path, [_row(id="sceneA", qplume=1234.5)])

        df = dataset_wiring._load_dataframe(csv_path, root_folder=tmp_path / "STARCOP_test")

        assert df.loc["sceneA", "qplume"] == 1234.5

    def test_builds_unshuffled_loader_with_default_batch_and_worker_counts(
        self, monkeypatch, tmp_path
    ):
        dataframe = pd.DataFrame({"id": ["sceneA"]})
        load_args = []

        def fake_load_dataframe(*args):
            load_args.append(args)
            return dataframe

        class FakeDataset:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class FakeDataLoader:
            def __init__(self, dataset, **kwargs):
                self.dataset = dataset
                self.kwargs = kwargs

        monkeypatch.setattr(dataset_wiring, "_load_dataframe", fake_load_dataframe)
        monkeypatch.setattr(dataset_wiring, "STARCOPDataset", FakeDataset)
        monkeypatch.setattr(dataset_wiring, "DataLoader", FakeDataLoader)

        loader = dataset_wiring.build_test_dataloader(
            tmp_path / "test.csv",
            tmp_path / "STARCOP_test",
            input_products=["mag1c"],
            output_products=["plume"],
            weight_loss="weighted",
        )

        assert load_args == [(tmp_path / "test.csv", tmp_path / "STARCOP_test")]
        assert loader.dataset.args == (dataframe,)
        assert loader.dataset.kwargs == {
            "input_products": ["mag1c"],
            "output_products": ["plume"],
            "weight_loss": "weighted",
            "spatial_augmentations": None,
            "window_size_sample": None,
        }
        assert loader.kwargs == {"batch_size": 1, "shuffle": False, "num_workers": 0}
