"""ProcessedDatasetDataModule -- sources STARCOP training data from this
project's own DVC-processed layout (data/processed/<dataset_name>/{patches,
splits}/) instead of STARCOP's Permian2019 file-discovery/download
convention.

Subclasses starcop.data.datamodule.Permian2019DataModule (imported
unmodified via _vendor_starcop_training.py) and overrides only
prepare_data() -- train_dataloader/val_dataloader/test_dataloader/
train_plot_dataloader/test_plot_dataloader are inherited unchanged, since
reading datamodule.py end to end confirmed they only ever consume
self.train_dataset/self.val_dataset/self.test_dataset/self.train_dataset_plot/
self.test_dataset_plot -- never the file-discovery logic being replaced here.
Same composition pattern as mlflow_image_logger.MultiLoggerImageLogger. See
mlops-methane-detection-plan.md TASK-2.2 step 4b.

Mirrors STARCOP's own original design faithfully: train on small tiles,
evaluate on full 512x512-window scene crops.
  - train_dataset / train_dataset_plot   <- patches/train_tiled_128_128.csv (tiled)
  - train_dataset_non_tiled              <- splits/train.csv (non-tiled)
  - val_dataset                          <- splits/val.csv (non-tiled; this
    project's own real val split -- the one deliberate improvement over
    STARCOP's own convention of reusing test_dataset as val_dataset, since
    STARCOP itself has no val split)
  - test_dataset / test_dataset_plot     <- splits/test.csv (non-tiled)
"""

import logging
import os
from pathlib import Path
from typing import Union

import kornia.augmentation as K
import pandas as pd
import rasterio.windows

from _vendor_starcop_training import Permian2019DataModule, STARCOPDataset, feature_extration


def _load_dataframe(path: Union[str, Path], repo_root: Union[str, Path]) -> pd.DataFrame:
    """Reads one of this project's split/patch CSVs into STARCOPDataset's
    expected shape: reconstructs the `window` column from its four bound
    columns, resolves `folder` to an absolute path (relative paths in these
    CSVs are repo-root-relative, but Hydra's `hydra.job.chdir: True` -- from
    STARCOP's own unmodified config -- moves the process cwd elsewhere
    before this runs), and sets `id` as the index.
    """
    repo_root = Path(repo_root)
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
    df["folder"] = df["folder"].apply(
        lambda p: p if os.path.isabs(p) else str(repo_root / p)
    )
    return df.set_index("id")


class ProcessedDatasetDataModule(Permian2019DataModule):
    def __init__(self, settings, dataset_name: str, repo_root: Union[str, Path]):
        super().__init__(settings)
        self.dataset_name = dataset_name
        self.repo_root = Path(repo_root)
        self.patches_dir = self.repo_root / "data" / "processed" / dataset_name / "patches"
        self.splits_dir = self.repo_root / "data" / "processed" / dataset_name / "splits"

    def prepare_data(self):
        log = logging.getLogger(__name__)

        if self.weight_loss is not None:
            extra_types = ["input"]
            weight_loss_list = [self.weight_loss]
        else:
            extra_types = []
            weight_loss_list = []

        model_output_type = "mask" if self.settings.model.model_mode == "segmentation_output" else "input"

        self.train_augmentations = K.AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            keepdim=True,
            data_keys=["input", model_output_type] + extra_types,
        )

        # Some configured products (e.g. weight_mag1c) are computed features,
        # not raw files -- mirrors Permian2019DataModule.prepare_data()'s own
        # features_extract/raw_bands split (datamodule.py:136-143), which the
        # override below must not silently drop.
        raw_bands_available = feature_extration.raw_bands_available()
        all_products = self.input_products + self.output_products + weight_loss_list
        self.features_extract = [p for p in all_products if p not in raw_bands_available]
        self.raw_bands = [p for p in all_products if p in raw_bands_available]

        train_dataframe = _load_dataframe(
            self.patches_dir / "train_tiled_128_128.csv", self.repo_root
        )
        self.train_dataframe_original = _load_dataframe(
            self.splits_dir / "train.csv", self.repo_root
        )
        val_dataframe = _load_dataframe(self.splits_dir / "val.csv", self.repo_root)
        test_dataframe = _load_dataframe(self.splits_dir / "test.csv", self.repo_root)
        test_dataframe = test_dataframe.sort_values(["has_plume", "qplume"], ascending=False)

        # Extracted once per scene (keyed by the shared `folder` column), so
        # this covers train_dataset/train_dataset_plot (tiled) too -- same as
        # Permian2019DataModule.prepare_data()'s single call on the non-tiled
        # frame (datamodule.py:172-173). val gets its own call since STARCOP
        # itself has no separate val split to extract for.
        if self.features_extract:
            feature_extration.extract_features(self.features_extract, self.train_dataframe_original)
            feature_extration.extract_features(self.features_extract, val_dataframe)
            feature_extration.extract_features(self.features_extract, test_dataframe)

        self.train_dataset = STARCOPDataset(
            train_dataframe,
            input_products=self.input_products,
            output_products=self.output_products,
            weight_loss=self.weight_loss,
            spatial_augmentations=self.train_augmentations,
            window_size_sample=None,
        )
        self.train_dataset_plot = STARCOPDataset(
            train_dataframe,
            input_products=self.input_products,
            output_products=self.output_products,
            weight_loss=self.weight_loss,
            spatial_augmentations=None,
            window_size_sample=None,
        )
        self.train_dataset_non_tiled = STARCOPDataset(
            self.train_dataframe_original,
            input_products=self.input_products,
            output_products=self.output_products,
            weight_loss=self.weight_loss,
            spatial_augmentations=None,
            window_size_sample=None,
        )
        self.val_dataset = STARCOPDataset(
            val_dataframe,
            input_products=self.input_products,
            output_products=self.output_products,
            weight_loss=self.weight_loss,
        )
        self.test_dataset = STARCOPDataset(
            test_dataframe,
            input_products=self.input_products,
            output_products=self.output_products,
            weight_loss=self.weight_loss,
        )
        self.test_dataset_plot = STARCOPDataset(
            test_dataframe,
            input_products=self.input_products,
            output_products=self.output_products,
            weight_loss=self.weight_loss,
        )

        if "rgb_aviris" in self.products_plot and not all(
            b in self.input_products for b in ["TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]
        ):
            self.train_dataset_plot.add_rgb_aviris = True
            self.test_dataset_plot.add_rgb_aviris = True

        if "mag1c" in self.products_plot and "mag1c" not in self.input_products:
            self.train_dataset_plot.add_extra_products(["mag1c"])
            self.test_dataset_plot.add_extra_products(["mag1c"])

        log.info("ProcessedDatasetDataModule ready")
        log.info(f"Train dataset {len(self.train_dataset)} chipsize: {self.training_size}")
        log.info(f"Val dataset {len(self.val_dataset)}")
        log.info(f"Test dataset {len(self.test_dataset)}")
