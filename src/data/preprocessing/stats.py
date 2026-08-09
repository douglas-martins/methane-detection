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
from tqdm import tqdm

from _vendor_starcop import STARCOPDataset


def compute_band_stats(dataframe: pd.DataFrame, bands: list[str]) -> dict:
    """Compute exact per-band mean/std/min/max over every row in `dataframe`.

    Accumulates a running sum/sum-of-squares/min/max per band instead of
    materializing every patch's every band array before reducing -- at
    starcop_raw's scale that would hold ~35GB of arrays in memory
    simultaneously (see starcop-raw-pipeline-plan.md). Peak memory here is
    O(1) per band, not O(all patches).
    """
    dataset = STARCOPDataset(dataframe, input_products=bands, output_products=[])
    running = {band: {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": np.inf, "max": -np.inf} for band in bands}

    # mininterval=5: redirected (non-tty) output writes one line per refresh
    # instead of overwriting in place, so a low interval would flood a log file.
    for idx in tqdm(range(len(dataset)), total=len(dataset), desc="Computing band stats", mininterval=5.0):
        input_tensor = dataset[idx]["input"].numpy()
        for band_idx, band in enumerate(bands):
            arr = input_tensor[band_idx]
            s = running[band]
            s["count"] += arr.size
            s["sum"] += float(arr.sum())
            s["sum_sq"] += float(np.square(arr, dtype=np.float64).sum())
            s["min"] = min(s["min"], float(arr.min()))
            s["max"] = max(s["max"], float(arr.max()))

    return {
        band: {
            "mean": (mean := s["sum"] / s["count"]),
            "std": float(np.sqrt(s["sum_sq"] / s["count"] - mean**2)),
            "min": s["min"],
            "max": s["max"],
        }
        for band, s in running.items()
    }


def compute_class_distribution(dataframe: pd.DataFrame, bands: list[str]) -> dict:
    """Compute exact positive/background pixel counts per output band (e.g. labelbinary).

    Same running-count accumulation as compute_band_stats(), for the same
    reason: at starcop_raw's patch count, holding every patch's label array
    in memory at once isn't necessary when a positive-pixel tally is O(1)
    per band.
    """
    dataset = STARCOPDataset(dataframe, input_products=[], output_products=bands)
    running = {band: {"positive": 0, "total": 0} for band in bands}

    for idx in tqdm(range(len(dataset)), total=len(dataset), desc="Computing class distribution", mininterval=5.0):
        output_tensor = dataset[idx]["output"].numpy()
        for band_idx, band in enumerate(bands):
            arr = output_tensor[band_idx]
            s = running[band]
            s["positive"] += int((arr > 0).sum())
            s["total"] += arr.size

    return {
        band: {
            "positive_pixels": s["positive"],
            "background_pixels": s["total"] - s["positive"],
            "total_pixels": s["total"],
            "positive_fraction": s["positive"] / s["total"],
            "imbalance_ratio": (s["total"] - s["positive"]) / s["positive"] if s["positive"] else None,
        }
        for band, s in running.items()
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
    """DVC entry point: write band_stats.json and class_distribution.json for the train split."""
    patches_root = Path(cfg.paths.processed_root) / "patches"
    stats_root = Path(cfg.paths.processed_root) / "stats"
    stats_root.mkdir(parents=True, exist_ok=True)

    patch_size = list(cfg.patch.size)
    train_path = patches_root / f"train_tiled_{patch_size[0]}_{patch_size[1]}.csv"
    dataframe = _load_patches_dataframe(train_path)

    bands = list(cfg.stats.bands) if cfg.stats.bands else list(cfg.dataset_cfg.input_products)
    band_stats = compute_band_stats(dataframe, bands)
    (stats_root / "band_stats.json").write_text(json.dumps(band_stats, indent=2))

    output_bands = list(cfg.dataset_cfg.output_products)
    class_distribution = compute_class_distribution(dataframe, output_bands)
    (stats_root / "class_distribution.json").write_text(json.dumps(class_distribution, indent=2))


def main() -> None:
    """CLI entry point: resolve the Hydra config and dispatch to run()."""
    import hydra

    @hydra.main(version_base=None, config_path="../../../configs", config_name="data")
    def _run(cfg):
        """Hydra-decorated wrapper receiving the composed config."""
        run(cfg)

    _run()


if __name__ == "__main__":
    main()
