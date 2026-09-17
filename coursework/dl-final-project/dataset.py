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
    # p=0.5 here is RandomRotation90's own default -- dropping it is behaviorally
    # equivalent (verified empirically over 30 seeds); pulled onto its own line
    # so the pragma below can target just this mutant, not the whole call.
    rotation = RandomRotation90(times=(0, 3), p=0.5)  # pragma: no mutate
    # kornia's DataKey lookup is case-insensitive, so "IMAGE"/"MASK" mutants here
    # are equivalent (verified empirically).
    data_keys = ["image", "mask"]  # pragma: no mutate
    # AugmentationSequential's own same_on_batch only affects its internal
    # random_apply selection (unused here, random_apply=False by default) -- False
    # vs. its own None default produced bit-identical output across 30 seeds tested.
    # same_on_batch is not passed here: AugmentationSequential's own same_on_batch
    # only feeds its internal random_apply selection (unused here, random_apply=False
    # by default) -- explicitly passing False produced bit-identical output to the
    # class's own None default across 30 seeds tested with these three transforms.
    return AugmentationSequential(
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        rotation,
        data_keys=data_keys,
    )


class PatchDataset(Dataset):
    """One 128x128 (or fixture-sized) patch per item: normalized input stack + raw label."""

    def __init__(self, patches_df: pd.DataFrame, dataset: str, augment: bool):
        """Store `patches_df`, resolve `dataset`'s band contract, and build the augmenter."""
        self.patches_df = patches_df.reset_index(drop=True)
        self.input_products, self.output_products = _load_dataset_config(dataset)
        self.augmenter = _build_augmenter() if augment else None
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        """Number of patches."""
        return len(self.patches_df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return {"input": (C,H,W) normalized float32, "output": (1,H,W) raw 0/1 float32}.

        The decoded, normalized (pre-augmentation) pair is cached per index
        after its first read -- without this, the same patch is re-read and
        re-decoded from disk on every access, every epoch, which measured as
        the dominant per-epoch cost (`fit()`'s own docstring: ~8s/epoch on
        `starcop_mini`'s 392 patches with no worker parallelism). Augmentation
        still runs fresh on every call from a clone of the cached pair, since
        it must vary per epoch and must never let one caller's in-place
        mutation of a returned tensor corrupt a later read of the same index.
        """
        if index in self._cache:
            cached_input, cached_output = self._cache[index]
            input_tensor, output_tensor = cached_input.clone(), cached_output.clone()
        else:
            row = self.patches_df.iloc[index]
            window = Window(
                col_off=row["window_col_off"],
                row_off=row["window_row_off"],
                width=row["window_width"],
                height=row["window_height"],
            )
            output_band = self.output_products[0]
            bands = read_patch_bands(
                Path(row["folder"]), window, [*self.input_products, output_band]
            )

            normalized = [
                normalize_band(bands[product], **BAND_NORMALIZATION[product])
                for product in self.input_products
            ]
            # axis=0 is np.stack's own default -- explicit for clarity, equivalent if dropped.
            input_tensor = torch.from_numpy(np.stack(normalized, axis=0))  # pragma: no mutate
            output_tensor = torch.from_numpy(bands[output_band]).unsqueeze(0).float()
            self._cache[index] = (input_tensor.clone(), output_tensor.clone())

        if self.augmenter is not None:
            input_batch = input_tensor.unsqueeze(0)
            # output_tensor's dim0 is always exactly 1 (single output channel), so
            # unsqueeze(0) and unsqueeze(1) produce the identical (1, 1, H, W) tensor.
            output_batch = output_tensor.unsqueeze(0)  # pragma: no mutate
            augmented_input, augmented_output = self.augmenter(input_batch, output_batch)
            input_tensor, output_tensor = augmented_input[0], augmented_output[0]

        return {"input": input_tensor, "output": output_tensor}
