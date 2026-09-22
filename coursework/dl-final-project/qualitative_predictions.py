"""Qualitative prediction-vs-ground-truth visualizations for the DL course final project
(plan Section 9).

Reuses `eda.py`'s `select_example_patches`/`read_patch_bands`/`rgb_composite` (Section 3) so
the mini examples shown here are the *same* patches the report's EDA figures already used --
the report's own qualitative continuity, not a new random draw. Adds one `starcop_raw` test
patch that belongs to none of `starcop_mini`'s 16 source scenes (`exclude_scenes`), so the
qualitative evidence for generalization isn't confined to the small slice the models were
trained on.

`predict_probability_map` is the tested logic (a model forward pass + sigmoid, the same
"tested even though it calls model.forward" convention as `evaluate.py`'s
`evaluate_full_metrics` and `scene_inference.py`'s `score_scenes`); `plot_prediction_comparison`
(matplotlib rendering) is thin glue, exercised by running `main()` for real and inspecting the
saved figure, matching `eda.py`'s own `plot_example_patches` split.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script writes figures to disk, never shows a window
import matplotlib.pyplot as plt
import pandas as pd
import torch
from dataset import PatchDataset
from eda import (
    INPUT_PRODUCTS,
    OUTPUT_PRODUCT,
    RGB_KEYS,
    read_patch_bands,
    rgb_composite,
    select_example_patches,
)
from evaluate import load_checkpoint
from metrics import pixel_f1
from rasterio.windows import Window

_COURSEWORK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _COURSEWORK_ROOT.parents[1]
_ARCHITECTURES = ["E1", "E2", "E3"]
_DEFAULT_THRESHOLD = 0.5


def mini_scene_names(*patch_dfs: pd.DataFrame) -> set[str]:
    """Union of the `name` (source-scene) column across any number of `starcop_mini` splits."""
    names: set[str] = set()
    for df in patch_dfs:
        names |= set(df["name"].unique())
    return names


def exclude_scenes(patches_df: pd.DataFrame, excluded_names: set[str]) -> pd.DataFrame:
    """Rows of `patches_df` whose `name` is not in `excluded_names`."""
    return patches_df[~patches_df["name"].isin(excluded_names)]


def predict_probability_map(model: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    """Run `model` on a single patch's `(C, H, W)` input; return its `(H, W)` sigmoid probability
    map.

    Forces eval mode and disables gradients -- a loaded checkpoint here is only ever used for a
    one-off qualitative prediction, never fine-tuned.
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor.unsqueeze(0))
    return torch.sigmoid(logits)[0, 0]


def plot_prediction_comparison(
    examples: pd.DataFrame,
    datasets: list[str],
    models: dict[str, torch.nn.Module],
    threshold: float,
    out_path: Path,
) -> None:
    """Render one row per example: RGB | ground truth | each architecture's binary prediction.

    `datasets[i]` selects `examples.iloc[i]`'s band contract (`starcop_mini` for the EDA
    patches, `starcop_raw` for the cross-tier one) -- the checkpoints in `models` are always
    the `mini`-trained ones, so this is the same "trained on `mini`, evaluated on `raw`"
    checkpoint/dataset split `evaluate.py`'s cross-tier runs already use.
    """
    n = len(examples)
    columns = ["RGB composite", "ground truth", *models.keys()]
    fig, axes = plt.subplots(n, len(columns), figsize=(3 * len(columns), 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (_, row) in enumerate(examples.iterrows()):
        dataset = datasets[row_idx]
        window = Window(
            col_off=row["window_col_off"],
            row_off=row["window_row_off"],
            width=row["window_width"],
            height=row["window_height"],
        )
        bands = read_patch_bands(Path(row["folder"]), window, [*INPUT_PRODUCTS, OUTPUT_PRODUCT])
        composite = rgb_composite(bands, *RGB_KEYS)
        ground_truth = bands[OUTPUT_PRODUCT]

        sample = PatchDataset(examples.iloc[[row_idx]], dataset=dataset, augment=False)[0]
        kind = "positive" if row["has_plume"] else "negative"
        axes[row_idx, 0].set_ylabel(
            f"{kind}\nfrac={row['frac_positives']:.4f}\n({dataset})",
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
        )
        axes[row_idx, 0].imshow(composite)
        axes[row_idx, 1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
        for col_idx, (name, model) in enumerate(models.items(), start=2):
            probs = predict_probability_map(model, sample["input"])
            binary = (probs > threshold).float().numpy()
            axes[row_idx, col_idx].imshow(binary, cmap="gray", vmin=0, vmax=1)

        for col_idx, ax in enumerate(axes[row_idx]):
            if row_idx == 0:
                ax.set_title(columns[col_idx], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def select_largest_gap_row(f1_df: pd.DataFrame, minuend_col: str, subtrahend_col: str) -> pd.Series:
    """Row of `f1_df` where `f1_df[minuend_col] - f1_df[subtrahend_col]` is largest.

    Used to find a patch where one architecture (`minuend_col`) most clearly outperforms
    another (`subtrahend_col`) -- the Validation checklist's requirement that the qualitative
    examples include at least one case where E3 underperforms E2, not only successes.
    """
    gap = f1_df[minuend_col] - f1_df[subtrahend_col]
    return f1_df.loc[gap.idxmax()]


def per_patch_f1_by_architecture(
    examples: pd.DataFrame,
    datasets: list[str],
    models: dict[str, torch.nn.Module],
    threshold: float,
) -> pd.DataFrame:
    """Per-example, per-architecture pixel F1 -- the number behind the qualitative figure.

    Used to confirm the Validation checklist's requirement that at least one shown example
    has the smaller model (E3) underperforming the larger one (E2), not just successes.
    """
    rows = []
    for row_idx, (_, row) in enumerate(examples.iterrows()):
        dataset = datasets[row_idx]
        sample = PatchDataset(
            examples.iloc[[row_idx]],
            dataset=dataset,
            # PatchDataset's own `self.augmenter = _build_augmenter() if augment else None`
            # treats None exactly like False (both falsy, no other branch distinguishes
            # them) -- mutating this literal to None is behaviorally unobservable.
            augment=False,  # pragma: no mutate-line -- None is falsy like False here, see above
        )[0]
        target = sample["output"][0]
        entry = {"id": row["id"], "dataset": dataset, "has_plume": row["has_plume"]}
        for name, model in models.items():
            probs = predict_probability_map(model, sample["input"])
            binary = (probs > threshold).float()
            entry[name] = pixel_f1(binary, target)
        rows.append(entry)
    return pd.DataFrame(rows)


def main() -> None:
    """Generate this section's qualitative figure from the mini EDA examples, one
    `starcop_raw` test patch outside any `starcop_mini` source scene, and one deliberately
    chosen `starcop_mini` test patch where E3 clearly underperforms E2 (Validation checklist:
    the examples must not be only successes) -- all against the three `mini` checkpoints,
    threshold 0.5 (matching the headline table's own basis).
    """
    device = "cpu"
    models = {
        architecture: load_checkpoint(
            architecture, _COURSEWORK_ROOT / "checkpoints" / f"{architecture}-mini.pt", device
        )
        for architecture in _ARCHITECTURES
    }

    mini_train = pd.read_csv(
        _REPO_ROOT / "data/processed/starcop_mini/patches/train_tiled_128_128.csv"
    )
    mini_val = pd.read_csv(_REPO_ROOT / "data/processed/starcop_mini/patches/val_tiled_128_128.csv")
    mini_test = pd.read_csv(
        _REPO_ROOT / "data/processed/starcop_mini/patches/test_tiled_128_128.csv"
    )
    mini_examples = select_example_patches(mini_train, n_positive=2, n_negative=2, seed=42)

    excluded_names = mini_scene_names(mini_train, mini_val, mini_test)
    raw_test = pd.read_csv(
        _REPO_ROOT / "data/processed/starcop_raw/patches/test_tiled_128_128.csv", low_memory=False
    )
    raw_candidates = exclude_scenes(raw_test, excluded_names)
    raw_example = select_example_patches(raw_candidates, n_positive=1, n_negative=0, seed=42)

    # Scan mini's own held-out test positives (never seen by any of the three models during
    # training) for the patch where E2 most clearly beats E3, rather than relying on the
    # EDA/cross-tier picks above to happen to show one -- they didn't (all favored E3).
    test_positives = mini_test[mini_test["has_plume"]].reset_index(drop=True)
    contrast_models = {name: models[name] for name in ("E2", "E3")}
    contrast_f1 = per_patch_f1_by_architecture(
        test_positives, ["starcop_mini"] * len(test_positives), contrast_models, _DEFAULT_THRESHOLD
    )
    contrast_id = select_largest_gap_row(contrast_f1, minuend_col="E2", subtrahend_col="E3")["id"]
    contrast_example = test_positives[test_positives["id"] == contrast_id]

    examples = pd.concat([mini_examples, contrast_example, raw_example], ignore_index=True)
    datasets = (
        ["starcop_mini"] * len(mini_examples)
        + ["starcop_mini"] * len(contrast_example)
        + ["starcop_raw"] * len(raw_example)
    )

    figures_dir = _COURSEWORK_ROOT / "figures"
    plot_prediction_comparison(
        examples, datasets, models, _DEFAULT_THRESHOLD, figures_dir / "qualitative_predictions.png"
    )

    per_patch = per_patch_f1_by_architecture(examples, datasets, models, _DEFAULT_THRESHOLD)
    print(per_patch.to_string(index=False))


if __name__ == "__main__":
    main()
