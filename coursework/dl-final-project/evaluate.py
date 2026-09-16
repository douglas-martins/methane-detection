"""Per-pixel Precision/Recall/F1/confusion-matrix/PR-AUC and per-patch "detected at all"
evaluation for the DL course final project (plan Section 8).

Thin glue over already-tested pieces (`metrics.py`'s confusion-matrix
accumulation, `train.py`'s `build_model`, `dataset.py`'s `PatchDataset`) --
exercised by a real recorded run against each of E1/E2/E3's `mini`-tier
checkpoints (Section 7 Phase C), not deep unit coverage, matching this
project's own established "thin glue vs. tested logic" pattern (see
`train.py`'s own module docstring).

Evaluates against `val` *and* `test` splits, per Section 8's own checklist
item -- val is what training already selected checkpoints on, test is the
number Section 9's headline table actually reports.
"""

import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
from dataset import PatchDataset
from metrics import (
    add_counts,
    average_precision_from_sweep,
    confusion_matrix_counts,
    confusion_matrix_from_counts,
    detection_rate_from_counts,
    f1_from_counts,
    patch_detection_counts,
    precision_from_counts,
    precision_recall_points_from_sweep,
    recall_from_counts,
    sweep_confusion_counts,
)
from torch.utils.data import DataLoader
from train import build_model

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COURSEWORK_ROOT = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _COURSEWORK_ROOT / "checkpoints"
_MLFLOW_TRACKING_URI = f"sqlite:///{_COURSEWORK_ROOT / 'mlflow.db'}"
_MLFLOW_EXPERIMENT = "dl-final-project"

_DEFAULT_THRESHOLD = 0.5
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_NUM_WORKERS = 4
_DEFAULT_PR_THRESHOLDS = torch.linspace(0.0, 1.0, steps=101)


def evaluate_full_metrics(
    model: torch.nn.Module,
    loader,
    device: str,
    threshold: float = _DEFAULT_THRESHOLD,
    max_batches: int | None = None,
    pr_thresholds: torch.Tensor = _DEFAULT_PR_THRESHOLDS,
) -> dict:
    """Run `model` in eval mode over `loader`; return precision/recall/F1/confusion matrix,
    PR-AUC (over `pr_thresholds`), and the per-patch "detected at all" summary -- Section 8's
    full metric suite, computed in one pass over `loader` rather than one pass per metric.

    Accumulates TP/FP/FN/TN counts (fixed-threshold, sweep, and per-patch)
    incrementally per batch via `metrics.add_counts` rather than
    concatenating every batch's predictions -- the same reasoning as
    `train.py`'s own `evaluate()`: `starcop_raw`'s full test split (16,758
    patches) must never require holding every prediction in memory at once.

    `max_batches` (default `None` = every batch) mirrors `train.py`'s own
    escape hatch, for smoke-testing against a slice of a large split.
    `pr_thresholds` (default 101 evenly spaced points in `[0, 1]`) is exposed
    so a future large-split run can trade sweep resolution for speed.
    """
    model.eval()
    totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    patch_totals = {"positive_patches": 0, "detected_patches": 0}
    # `pr_thresholds` defaults to a plain CPU tensor -- must follow `device`,
    # or the sweep crashes comparing it against a CUDA batch's probabilities
    # (a real crash hit running this for real on this machine's GPU).
    pr_thresholds = pr_thresholds.to(device)
    sweep_totals = torch.zeros(len(pr_thresholds), 4, device=device)
    n_batches = 0
    patches_processed = 0

    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and n_batches >= max_batches:
                break
            x, y = batch["input"].to(device), batch["output"].to(device)
            probs = torch.sigmoid(model(x))
            binary = (probs > threshold).float()
            totals = add_counts(totals, confusion_matrix_counts(binary, y))
            patch_totals = add_counts(patch_totals, patch_detection_counts(binary, y))
            sweep_totals += sweep_confusion_counts(probs, y, pr_thresholds)
            patches_processed += x.shape[0]
            n_batches += 1

    return {
        "precision": precision_from_counts(totals),
        "recall": recall_from_counts(totals),
        "f1": f1_from_counts(totals),
        "confusion_matrix": confusion_matrix_from_counts(totals).tolist(),
        "pr_auc": average_precision_from_sweep(sweep_totals),
        "precision_recall_curve": precision_recall_points_from_sweep(sweep_totals),
        "per_patch_positive_patches": patch_totals["positive_patches"],
        "per_patch_detected_patches": patch_totals["detected_patches"],
        "per_patch_detection_rate": detection_rate_from_counts(patch_totals),
        # Explicit, checkable count (plan Section 8's own R1 validation
        # requirement) -- a full-split run must be verified by this number
        # matching the split's manifest row count, not by the run merely
        # exiting 0. Not inferred from `n_batches * batch_size`: a real
        # split's last batch is usually smaller than the configured size.
        "patches_processed": patches_processed,
        **totals,
    }


def load_checkpoint(architecture: str, checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Build `architecture` (E1/E2/E3), load its trained weights, and set it to eval mode.

    Returned already in `eval()` mode -- a loaded checkpoint is only ever
    used for inference, and BatchNorm layers (E1's `_ConvBlock`) reject a
    batch as small as 1x1 spatial resolution in train mode.
    """
    model = build_model(architecture)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def _parse_kv_args(argv: list[str]) -> dict[str, str]:
    """Parse `key=value` CLI args (Hydra-style, matching `train.py`/`confirm_raw.py`)."""
    parsed = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed[key] = value
    return parsed


def _load_eval_splits(dataset: str, tier: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (val_df, test_df) to score `tier`'s checkpoint against.

    `test_df` always comes from `dataset`'s own already-split test CSV --
    plan Section 0.1 states R2's val/test reuse raw's own full splits in
    full, never a redrawn subsample, so there is no tier-specific test
    manifest to read. `val_df` matches whichever split `tier` actually
    trained/selected checkpoints against: `r2`'s persisted subsample
    manifest, or `dataset`'s own val CSV otherwise.
    """
    patches_root = _REPO_ROOT / "data" / "processed" / dataset / "patches"
    test_df = pd.read_csv(patches_root / "test_tiled_128_128.csv", low_memory=False)
    if tier == "r2":
        val_df = pd.read_csv(_COURSEWORK_ROOT / "r2_manifest_val.csv", low_memory=False)
    else:
        val_df = pd.read_csv(patches_root / "val_tiled_128_128.csv", low_memory=False)
    return val_df, test_df


def main() -> None:
    """CLI entry point.

    Usage: python evaluate.py architecture=E1 dataset=starcop_mini tier=mini
           [checkpoint_tier=mini] [splits=val,test] [batch_size=16] [threshold=0.5]
           [num_workers=4]

    Loads `checkpoints/<architecture>-<checkpoint_tier>.pt` (default
    `checkpoint_tier=tier`, the usual same-tier case) and scores it against
    `dataset`'s `splits` (default both val and test; Section 8's cross-tier
    checklist item wants `splits=test` only, scoring a checkpoint trained on
    one tier -- e.g. `checkpoint_tier=mini` -- against a *different* eval
    `dataset`/`tier` -- e.g. `dataset=starcop_raw tier=raw-full` -- to answer
    "does a model trained on 392 patches hold up on the real distribution").
    Prints per-split precision/recall/F1/confusion matrix, and logs them to
    the same MLflow experiment `train.py` writes to -- run name
    `<architecture>-<tier>-eval` in the same-tier case,
    `<architecture>-<checkpoint_tier>-on-<tier>-eval` when cross-tier --
    metrics prefixed `val_`/`test_`, so Section 9 can trace every number to
    a `run_id` the same way the training numbers already are.
    """
    args = _parse_kv_args(sys.argv[1:])
    architecture = args.get("architecture", "E1")
    dataset = args.get("dataset", "starcop_mini")
    tier = args.get("tier", "mini")
    checkpoint_tier = args.get("checkpoint_tier", tier)
    splits = args.get("splits", "val,test").split(",")
    batch_size = int(args.get("batch_size", _DEFAULT_BATCH_SIZE))
    threshold = float(args.get("threshold", _DEFAULT_THRESHOLD))
    num_workers = int(args.get("num_workers", _DEFAULT_NUM_WORKERS))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = _CHECKPOINT_DIR / f"{architecture}-{checkpoint_tier}.pt"
    model = load_checkpoint(architecture, checkpoint_path, device)
    val_df, test_df = _load_eval_splits(dataset, tier)
    available_splits = {"val": val_df, "test": test_df}

    run_name = (
        f"{architecture}-{tier}-eval"
        if checkpoint_tier == tier
        else f"{architecture}-{checkpoint_tier}-on-{tier}-eval"
    )

    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "architecture": architecture,
                "dataset": dataset,
                "tier": tier,
                "checkpoint_tier": checkpoint_tier,
                "stage": "evaluation",
                "device": device,
                "threshold": threshold,
                **{
                    f"{name}_patches": len(df)
                    for name, df in available_splits.items()
                    if name in splits
                },
            }
        )
        for split_name in splits:
            split_df = available_splits[split_name]
            loader = DataLoader(
                PatchDataset(split_df, dataset=dataset, augment=False),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            result = evaluate_full_metrics(model, loader, device, threshold=threshold)
            # Section 8's own R1 validation requirement: verified by the
            # processed-patch count matching the split, not by exiting 0 --
            # checked on every run, not only the dedicated R1 check, since a
            # silent truncation would otherwise misreport every number below.
            assert result["patches_processed"] == len(split_df), (
                f"{split_name}: processed {result['patches_processed']} patches, "
                f"expected {len(split_df)} -- the split was silently truncated"
            )
            mlflow.log_metrics(
                {
                    f"{split_name}_precision": result["precision"],
                    f"{split_name}_recall": result["recall"],
                    f"{split_name}_f1": result["f1"],
                    f"{split_name}_tp": result["tp"],
                    f"{split_name}_fp": result["fp"],
                    f"{split_name}_fn": result["fn"],
                    f"{split_name}_tn": result["tn"],
                    f"{split_name}_pr_auc": result["pr_auc"],
                    f"{split_name}_patch_positives": result["per_patch_positive_patches"],
                    f"{split_name}_patch_detected": result["per_patch_detected_patches"],
                    f"{split_name}_patch_detection_rate": result["per_patch_detection_rate"],
                    f"{split_name}_patches_processed": result["patches_processed"],
                }
            )
            # The full curve isn't a scalar MLflow metric -- logged as a JSON
            # artifact instead, for Section 8's later PR-curve plots.
            mlflow.log_dict(
                {"precision_recall_curve": result["precision_recall_curve"]},
                f"{split_name}_precision_recall_curve.json",
            )
            print(
                f"  {split_name}: precision={result['precision']:.4f} "
                f"recall={result['recall']:.4f} f1={result['f1']:.4f} "
                f"pr_auc={result['pr_auc']:.4f} "
                f"detected_patches={result['per_patch_detected_patches']}/"
                f"{result['per_patch_positive_patches']} "
                f"confusion_matrix={result['confusion_matrix']}"
            )

    print(f"architecture={architecture} tier={tier} dataset={dataset}")


if __name__ == "__main__":
    main()
