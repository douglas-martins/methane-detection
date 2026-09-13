"""Dataset-parameterized patch loader for the DL course final project (plan Section 4).

`dataset` (`starcop_mini` | `starcop_raw`) is a constructor parameter from
the very first version, not retrofitted later: which input/output products
to read comes from `configs/dataset/<dataset>.yaml`, the same file the
shared DVC pipeline already uses, so this loader can never silently drift
from that contract. Reuses `eda.py`'s `read_patch_bands` (Section 3)
rather than re-implementing windowed GeoTIFF reads.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from eda import read_patch_bands
from kornia.augmentation import (
    AugmentationSequential,
    RandomHorizontalFlip,
    RandomRotation90,
    RandomVerticalFlip,
)
from omegaconf import OmegaConf
from preprocessing import BAND_NORMALIZATION, normalize_band
from rasterio.windows import Window
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dataset_config(dataset: str) -> tuple[list[str], list[str]]:
    """Return (input_products, output_products) from configs/dataset/<dataset>.yaml."""
    config_path = _REPO_ROOT / "configs" / "dataset" / f"{dataset}.yaml"
    config = OmegaConf.load(config_path)
    return list(config.dataset_cfg.input_products), list(config.dataset_cfg.output_products)


def _build_augmenter() -> AugmentationSequential:
    """Build the train-only augmenter: flips + 90-degree rotations, image and mask synced.

    No color-jitter-style transform is included -- those would distort the
    physically meaningful mag1c/reflectance values (plan Section 4).
    """
    return AugmentationSequential(
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(times=(0, 3), p=0.5),
        data_keys=["image", "mask"],
        same_on_batch=False,
    )


class PatchDataset(Dataset):
    """One 128x128 (or fixture-sized) patch per item: normalized input stack + raw label."""

    def __init__(self, patches_df: pd.DataFrame, dataset: str, augment: bool):
        """Store `patches_df`, resolve `dataset`'s band contract, and build the augmenter."""
        self.patches_df = patches_df.reset_index(drop=True)
        self.input_products, self.output_products = _load_dataset_config(dataset)
        self.augmenter = _build_augmenter() if augment else None

    def __len__(self) -> int:
        """Number of patches."""
        return len(self.patches_df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return {"input": (C,H,W) normalized float32, "output": (1,H,W) raw 0/1 float32}."""
        row = self.patches_df.iloc[index]
        window = Window(
            col_off=row["window_col_off"],
            row_off=row["window_row_off"],
            width=row["window_width"],
            height=row["window_height"],
        )
        output_band = self.output_products[0]
        bands = read_patch_bands(Path(row["folder"]), window, [*self.input_products, output_band])

        normalized = [
            normalize_band(bands[product], **BAND_NORMALIZATION[product])
            for product in self.input_products
        ]
        input_tensor = torch.from_numpy(np.stack(normalized, axis=0))
        output_tensor = torch.from_numpy(bands[output_band]).unsqueeze(0).float()

        if self.augmenter is not None:
            augmented_input, augmented_output = self.augmenter(
                input_tensor.unsqueeze(0), output_tensor.unsqueeze(0)
            )
            input_tensor, output_tensor = augmented_input[0], augmented_output[0]

        return {"input": input_tensor, "output": output_tensor}
