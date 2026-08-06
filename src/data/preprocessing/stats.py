"""Stage 4 of the TASK-1.2 DVC pipeline: per-band mean/std/min/max.

Reuses STARCOPDataset (via _vendor_starcop) so stats are computed through
the same per-band read + concatenate path the model will use at train time.
NOT used for model normalization -- STARCOP normalizes via the fixed
BAND_NORMALIZATION table (see normalize.py). This feeds the TASK-1.3
dataset report and the TASK-6.2 drift-detection baseline instead.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio.windows

from _vendor_starcop import STARCOPDataset


def compute_band_stats(dataframe: pd.DataFrame, bands: list[str]) -> dict:
    """Compute exact per-band mean/std/min/max over every row in `dataframe`."""
    dataset = STARCOPDataset(dataframe, input_products=bands, output_products=[])
    per_band_arrays = {band: [] for band in bands}

    for idx in range(len(dataset)):
        input_tensor = dataset[idx]["input"].numpy()
        for band_idx, band in enumerate(bands):
            per_band_arrays[band].append(input_tensor[band_idx])

    return {
        band: {
            "mean": float(np.mean(arrays)),
            "std": float(np.std(arrays)),
            "min": float(np.min(arrays)),
            "max": float(np.max(arrays)),
        }
        for band, arrays in per_band_arrays.items()
    }


def _load_patches_dataframe(path: Path) -> pd.DataFrame:
    """Read a patch_extract.py output CSV, rebuilding the `window` column patch_extract.py dropped."""
    dataframe = pd.read_csv(path)
    dataframe["window"] = dataframe.apply(
        lambda row: rasterio.windows.Window(
            col_off=row.window_col_off,
            row_off=row.window_row_off,
            width=row.window_width,
            height=row.window_height,
        ),
        axis=1,
    )
    return dataframe


def run(cfg) -> None:
    patches_root = Path(cfg.paths.processed_root) / "patches"
    stats_root = Path(cfg.paths.processed_root) / "stats"
    stats_root.mkdir(parents=True, exist_ok=True)

    patch_size = list(cfg.patch.size)
    train_path = patches_root / f"train_tiled_{patch_size[0]}_{patch_size[1]}.csv"
    dataframe = _load_patches_dataframe(train_path)

    bands = list(cfg.stats.bands) if cfg.stats.bands else list(cfg.dataset_cfg.input_products)
    band_stats = compute_band_stats(dataframe, bands)

    (stats_root / "band_stats.json").write_text(json.dumps(band_stats, indent=2))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path="../../../configs", config_name="data")
    def _run(cfg):
        run(cfg)

    _run()


if __name__ == "__main__":
    main()
