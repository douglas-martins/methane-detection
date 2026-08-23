"""Builds a test-only dataloader directly against `STARCOPDataset`, *not*
via `Permian2019DataModule.prepare_data()` (which unconditionally tiles the
full multi-GB training set first -- wasteful for an eval-only pass). See
track-a-paper-benchmark-reproduction-plan.md Phase 1.

Replicates `Permian2019DataModule.load_dataframe`'s few relevant lines
(`vendor/starcop/starcop/data/datamodule.py:104`): read the CSV, rebuild the
`window` column from its four bound columns, rebuild `folder` as
`root_folder/id` (test.csv's own `folder` column is a stale absolute path
from the original paper authors' machine, unusable as-is), and set `id` as
the index.
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import rasterio.windows
from _vendor_starcop_evaluation import STARCOPDataset, feature_extration
from torch.utils.data import DataLoader

TEST_CSV_ROOT_FOLDER = "data/starcop_raw/STARCOP_test"


def _load_dataframe(path: Union[str, Path], root_folder: Union[str, Path]) -> pd.DataFrame:
    """Reads `path` (test.csv's shape) into `STARCOPDataset`'s expected
    frame: rebuilds `window` from its four bound columns, rebuilds `folder`
    as `root_folder/id`, sets `id` as the index."""
    df = pd.read_csv(path)

    df["window"] = df.apply(
        lambda row: rasterio.windows.Window(
            col_off=row.window_col_off,
            row_off=row.window_row_off,
            width=row.window_width,
            height=row.window_height,
        ),
        axis=1,
    )
    df["folder"] = df["id"].apply(lambda scene_id: os.path.join(str(root_folder), scene_id))
    return df.set_index("id")


def build_test_dataloader(
    test_csv_path: Union[str, Path],
    root_folder: Union[str, Path],
    input_products: list[str],
    output_products: list[str],
    weight_loss: Optional[str] = None,
    features_extract: Optional[list[str]] = None,
    scene_ids: Optional[list[str]] = None,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    """Builds a `DataLoader` over the test set, unshuffled, batch_size=1 to
    match `run_validation`'s own requirement. `features_extract` is passed
    for MultiSTARCOP, whose Varon ratio bands are computed features, unlike
    Hyper's raw `mag1c`+RGB bands -- extracted once per scene before the
    dataset is constructed, mirroring
    `ProcessedDatasetDataModule.prepare_data()`'s own `features_extract`
    step. `scene_ids`, when given, restricts the dataloader to just those
    scenes (a `--limit` dry pass or a curated-scene plotting pass) instead
    of the full test set."""
    dataframe = _load_dataframe(test_csv_path, root_folder)
    if scene_ids is not None:
        dataframe = dataframe.loc[scene_ids]

    if features_extract:
        feature_extration.extract_features(features_extract, dataframe)

    dataset = STARCOPDataset(
        dataframe,
        input_products=input_products,
        output_products=output_products,
        weight_loss=weight_loss,
        spatial_augmentations=None,
        window_size_sample=None,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
