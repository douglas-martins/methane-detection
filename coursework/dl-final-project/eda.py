"""Exploratory data analysis for the DL course final project (plan Section 3).

Deliberately independent of `src/data/preprocessing/`: this coursework
project stays physically isolated (plan Section 0), so patch-level balance,
example-patch selection, and the RGB/mag1c/mask visualizations used in the
report are implemented here rather than imported from the thesis pipeline.
`stats.py`'s DVC-tracked pixel-level class distribution and per-band
statistics are reused as-is (see `docs/dataset_report.md`) -- this module
only adds what that pipeline doesn't already produce: patch-level balance
and example visualizations.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script writes figures to disk, never shows a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


def compute_patch_level_balance(patches_df: pd.DataFrame) -> dict:
    """Return {positive, total, positive_fraction} counting patches with any plume at all.

    Distinct from stats.py's pixel-level class distribution: a patch with a
    single positive pixel counts fully here, not fractionally.
    """
    total = len(patches_df)
    positive = int(patches_df["has_plume"].sum()) if total else 0
    return {
        "positive": positive,
        "total": total,
        "positive_fraction": (positive / total) if total else 0.0,
    }


def select_example_patches(
    patches_df: pd.DataFrame,
    n_positive: int,
    n_negative: int,
    seed: int,
) -> pd.DataFrame:
    """Pick `n_positive` + `n_negative` example patches for qualitative visualization.

    One positive slot is always the faintest plume available (lowest
    `frac_positives` among `has_plume` rows) -- the hard/ambiguous example
    the report's Validation checklist requires, chosen deterministically
    rather than left to chance. Remaining slots are sampled with `seed` for
    reproducibility across report regenerations.
    """
    positives = patches_df[patches_df["has_plume"]]
    negatives = patches_df[~patches_df["has_plume"]]
    if len(positives) < n_positive:
        raise ValueError(f"Not enough positive patches: need {n_positive}, have {len(positives)}")
    if len(negatives) < n_negative:
        raise ValueError(f"Not enough negative patches: need {n_negative}, have {len(negatives)}")

    hardest = positives.loc[[positives["frac_positives"].idxmin()]]
    remaining_positive = positives.drop(index=hardest.index)
    extra_positive = remaining_positive.sample(n=n_positive - 1, random_state=seed)
    chosen_negative = negatives.sample(n=n_negative, random_state=seed)

    return pd.concat([hardest, extra_positive, chosen_negative], ignore_index=True)


def read_patch_bands(scene_folder: Path, window: Window, bands: list[str]) -> dict[str, np.ndarray]:
    """Read `window` from each of `bands`' GeoTIFFs under `scene_folder`."""
    result = {}
    for band in bands:
        with rasterio.open(scene_folder / f"{band}.tif") as src:
            result[band] = src.read(1, window=window)
    return result


def rgb_composite(bands: dict[str, np.ndarray], r_key: str, g_key: str, b_key: str) -> np.ndarray:
    """Stack three bands into an (H, W, 3) array, each independently scaled to [0, 1].

    Per-band min/max scaling (not a shared scale) so each channel uses its
    own dynamic range -- appropriate here since this is a qualitative
    visualization aid, not the model's actual normalization contract
    (Section 4 uses STARCOP's fixed offset/factor/clip table for that).
    A constant band (max == min) maps to all-zero rather than dividing by
    zero.
    """

    def _scale(band: np.ndarray) -> np.ndarray:
        band_min, band_max = float(band.min()), float(band.max())
        span = band_max - band_min
        if span == 0:
            return np.zeros_like(band, dtype="float32")
        return ((band - band_min) / span).astype("float32")

    return np.stack([_scale(bands[r_key]), _scale(bands[g_key]), _scale(bands[b_key])], axis=-1)


def plot_patch_level_balance(balances: dict[str, dict[str, dict]], out_path: Path) -> None:
    """Save a grouped bar chart of patch-level positive rate per split, per tier.

    `balances` is {tier: {split: {positive, total, positive_fraction}}}.
    """
    splits = ["train", "val", "test"]
    tiers = list(balances.keys())
    x = np.arange(len(splits))
    width = 0.8 / len(tiers)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, tier in enumerate(tiers):
        fractions = [balances[tier][split]["positive_fraction"] * 100 for split in splits]
        ax.bar(x + i * width, fractions, width, label=tier)
    ax.set_xticks(x + width * (len(tiers) - 1) / 2)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Patches with any plume (%)")
    ax.set_title("Patch-level class balance: starcop_mini vs. starcop_raw")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_example_patches(
    examples: pd.DataFrame,
    input_products: list[str],
    rgb_keys: tuple[str, str, str],
    mag1c_key: str,
    output_product: str,
    out_path: Path,
) -> None:
    """Render one row per example: RGB composite | mag1c heatmap | ground-truth mask."""
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    for row_idx, (_, row) in enumerate(examples.iterrows()):
        window = Window(
            col_off=row["window_col_off"],
            row_off=row["window_row_off"],
            width=row["window_width"],
            height=row["window_height"],
        )
        bands = read_patch_bands(Path(row["folder"]), window, [*input_products, output_product])
        composite = rgb_composite(bands, *rgb_keys)

        kind = "positive" if row["has_plume"] else "negative"
        label = f"{kind}\nfrac={row['frac_positives']:.4f}"
        axes[row_idx, 0].imshow(composite)
        axes[row_idx, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
        axes[row_idx, 1].imshow(bands[mag1c_key], cmap="viridis")
        axes[row_idx, 2].imshow(bands[output_product], cmap="gray", vmin=0, vmax=1)
        for col_idx, ax in enumerate(axes[row_idx]):
            if row_idx == 0:
                ax.set_title(["RGB composite", "mag1c", "ground truth"][col_idx], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


INPUT_PRODUCTS = ["mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]
RGB_KEYS = ("TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm")
OUTPUT_PRODUCT = "labelbinary"


def _patches_root(dataset: str) -> Path:
    """Resolve the repo-root-relative patches directory for `dataset`."""
    return Path(f"data/processed/{dataset}/patches")


def main() -> None:
    """Generate this section's figures from the already-processed mini and raw patch CSVs."""
    figures_dir = Path(__file__).parent / "figures"

    balances = {}
    for dataset in ["starcop_mini", "starcop_raw"]:
        balances[dataset] = {}
        for split in ["train", "val", "test"]:
            df = pd.read_csv(_patches_root(dataset) / f"{split}_tiled_128_128.csv")
            balances[dataset][split] = compute_patch_level_balance(df)
    plot_patch_level_balance(balances, figures_dir / "patch_level_balance.png")

    mini_train = pd.read_csv(_patches_root("starcop_mini") / "train_tiled_128_128.csv")
    mini_examples = select_example_patches(mini_train, n_positive=2, n_negative=2, seed=42)
    plot_example_patches(
        mini_examples,
        INPUT_PRODUCTS,
        RGB_KEYS,
        "mag1c",
        OUTPUT_PRODUCT,
        figures_dir / "examples_mini.png",
    )

    raw_train = pd.read_csv(_patches_root("starcop_raw") / "train_tiled_128_128.csv")
    raw_examples = select_example_patches(raw_train, n_positive=1, n_negative=0, seed=42)
    plot_example_patches(
        raw_examples,
        INPUT_PRODUCTS,
        RGB_KEYS,
        "mag1c",
        OUTPUT_PRODUCT,
        figures_dir / "examples_raw.png",
    )


if __name__ == "__main__":
    main()
