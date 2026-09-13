"""Training loop for the DL course final project (plan Section 6/7).

Thin glue over the already-tested pieces (`dataset.py`, `losses.py`,
`early_stopping.py`, `metrics.py`) -- exercised by real recorded runs
(mini baseline training, R1 raw smoke training, R2 subsample training),
not deep unit coverage, matching this repo's own established "thin glue
vs. tested logic" pattern (see `src/baselines/starcop/serving/service.py`,
`train.py`).

MLflow tracking uses a SQLite backend
(`sqlite:///coursework/dl-final-project/mlflow.db`), not the plain
`file:./mlruns` the plan's early sections describe -- the installed
MLflow (3.14) now raises by default on the filesystem backend
(maintenance-mode only), and the tracking URI is explicit and
coursework-scoped so it can never collide with this repo's own
pre-existing root-level `mlruns/` (real thesis experiment data).
"""

import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
from architectures import build_e1, build_e2, build_e3
from dataset import PatchDataset
from early_stopping import EarlyStopper
from losses import build_loss, compute_pos_weight
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COURSEWORK_ROOT = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _COURSEWORK_ROOT / "checkpoints"
_MLFLOW_TRACKING_URI = f"sqlite:///{_COURSEWORK_ROOT / 'mlflow.db'}"
_MLFLOW_EXPERIMENT = "dl-final-project"

_ARCHITECTURE_BUILDERS = {"E1": build_e1, "E2": build_e2, "E3": build_e3}


def build_model(architecture: str) -> torch.nn.Module:
    """Build E1/E2/E3 by name -- E2/E3 pretrained, E1 (no `pretrained` param) is not."""
    build = _ARCHITECTURE_BUILDERS[architecture]
    return build() if architecture == "E1" else build(pretrained=True)


def augment_for(architecture: str) -> bool:
    """Whether `architecture` trains with augmentation -- E1 is the deliberately plain baseline
    (Section 6: no pretraining, no augmentation, minimal regularization); E2/E3 use it."""
    return architecture != "E1"


def evaluate(model: torch.nn.Module, loader, loss_fn: torch.nn.Module, device: str) -> dict:
    """Run `model` in eval mode over `loader`; return loss/F1/degeneracy metrics.

    Accumulates true/false positive/negative counts (and an all-zero/
    all-one flag) incrementally per batch rather than concatenating every
    batch's predictions -- `starcop_raw`'s full val split (26,607 patches)
    OOM'd a first version of this function that did exactly that. Same
    incremental-accumulation reasoning as `stats.py`'s own
    `compute_band_stats`/`compute_class_distribution` (Section 0.1): O(1)
    memory per batch, not O(dataset size).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    true_positive = false_positive = false_negative = 0.0
    positive_predictions = 0.0
    total_pixels = 0
    all_zero = True
    all_one = True
    epsilon = 1e-6

    with torch.no_grad():
        for batch in loader:
            x, y = batch["input"].to(device), batch["output"].to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item()
            n_batches += 1

            probs = torch.sigmoid(logits)
            binary = (probs > 0.5).float()
            true_positive += (binary * y).sum().item()
            false_positive += (binary * (1 - y)).sum().item()
            false_negative += ((1 - binary) * y).sum().item()
            positive_predictions += binary.sum().item()
            total_pixels += binary.numel()
            all_zero = all_zero and bool((probs < epsilon).all())
            all_one = all_one and bool((probs > 1 - epsilon).all())

    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "loss": total_loss / n_batches,
        "f1": (2 * true_positive / f1_denominator) if f1_denominator > 0 else 0.0,
        "degenerate": all_zero or all_one,
        "positive_fraction": positive_predictions / total_pixels,
    }


def fit(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset: str,
    *,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    max_steps: int | None = None,
    augment: bool = True,
    device: str = "cpu",
    checkpoint_path: Path | None = None,
    log_to_mlflow: bool = True,
    num_workers: int = 0,
    verbose: bool = False,
) -> dict:
    """Train `model` on `train_df`, validate on `val_df`; return final metrics + run length.

    `num_workers=0` (the default) is right for tests -- worker-process
    startup is pure overhead against a handful of synthetic patches. Real
    training runs should pass `num_workers>0`: with `PatchDataset`'s
    per-item rasterio reads and no cross-epoch caching, a single-process
    loader re-reads every patch from disk every epoch serially -- measured
    at ~8s/epoch on `starcop_mini`'s 392 patches; `num_workers=4` with
    `persistent_workers=True` cuts that to ~1.5-2s/epoch (real GeoTIFFs,
    not a synthetic fixture) by parallelizing that I/O across worker
    processes that stay alive between epochs instead of respawning every
    time. **`num_workers` applies to both the train and val loader**, each
    with its own persistent pool alive for the whole call -- on this
    machine's 12-thread CPU, `num_workers=8` (16 workers total, both
    pools) measurably thrashes (a 300-step + one-eval smoke run went from
    ~18s isolated to hanging past a 200s timeout); `num_workers=4` (8
    total) was verified fast for the same real workload. Keep
    `num_workers * 2` comfortably under the machine's thread count.
    """
    model.to(device)
    pos_weight = compute_pos_weight(train_df)
    loss_fn = build_loss(pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader_kwargs = (
        {"num_workers": num_workers, "persistent_workers": True} if num_workers > 0 else {}
    )
    train_loader = DataLoader(
        PatchDataset(train_df, dataset=dataset, augment=augment),
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        PatchDataset(val_df, dataset=dataset, augment=False),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    stopper = EarlyStopper(patience=patience, mode="min")
    step_count = 0
    epoch = 0
    val_metrics: dict = {}
    best_metrics: dict = {}
    best_epoch = 0
    stop_early = False

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            x, y = batch["input"].to(device), batch["output"].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            step_count += 1
            if max_steps is not None and step_count >= max_steps:
                break

        val_metrics = evaluate(model, val_loader, loss_fn, device)
        if log_to_mlflow:
            mlflow.log_metrics(
                {"val_loss": val_metrics["loss"], "val_f1": val_metrics["f1"]}, step=epoch
            )
        if verbose:
            print(
                f"  epoch {epoch}/{max_epochs} step {step_count}: "
                f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f}"
            )

        should_stop = stopper.step(val_metrics["loss"])
        if stopper.is_best:
            best_metrics = val_metrics
            best_epoch = epoch
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

        if (max_steps is not None and step_count >= max_steps) or should_stop:
            stop_early = True
            break

    # Report the best epoch's metrics, not whichever epoch training
    # happened to stop on -- early stopping's own patience lets the model
    # wander for up to `patience` epochs after its best result (seen for
    # real: R2's baseline run peaked at epoch 8, stopped at epoch 18 with
    # a visibly worse loss). The saved checkpoint is the best epoch's
    # weights, so the reported number must match what was actually kept.
    return {
        **best_metrics,
        "best_epoch": best_epoch,
        "last_epoch_metrics": val_metrics,
        "epochs_run": epoch,
        "steps_run": step_count,
        "pos_weight": pos_weight,
        "stopped_early": stop_early,
    }


def _parse_kv_args(argv: list[str]) -> dict[str, str]:
    """Parse `key=value` CLI args (Hydra-style, matching `confirm_raw.py`)."""
    parsed = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed[key] = value
    return parsed


def _load_split(dataset: str, tier: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (train_df, val_df) for `tier` -- 'mini'/'raw-full' use the dataset's own splits,
    'r2' uses the persisted subsample manifests (Section 0.1's mechanism, built once by
    `build_r2_manifest.py`, never regenerated per run), 'raw-smoke' uses the full raw
    train split (so `max_steps` samples real, varied patches) but a small seeded slice
    of val -- the smoke run only needs to prove the loop works, not a real val number,
    and evaluating the full 26,607-patch raw val split every epoch is pure waste here."""
    if tier == "r2":
        return (
            pd.read_csv(_COURSEWORK_ROOT / "r2_manifest_train.csv", low_memory=False),
            pd.read_csv(_COURSEWORK_ROOT / "r2_manifest_val.csv", low_memory=False),
        )
    patches_root = _REPO_ROOT / "data" / "processed" / dataset / "patches"
    train_df = pd.read_csv(patches_root / "train_tiled_128_128.csv", low_memory=False)
    val_df = pd.read_csv(patches_root / "val_tiled_128_128.csv", low_memory=False)
    if tier == "raw-smoke":
        val_df = val_df.sample(n=min(200, len(val_df)), random_state=42).reset_index(drop=True)
    return train_df, val_df


def main() -> None:
    """CLI entry point -- see report.md's Metodologia section for the run log this produced.

    Usage: python train.py architecture=E1 dataset=starcop_mini tier=mini [max_steps=300]
    `tier` is one of 'mini' | 'raw-smoke' | 'r2' -- only affects which split is loaded and,
    for 'raw-smoke', how the run is labeled; `max_steps` (optional) caps step count,
    used for the R1 smoke-training run (a few hundred steps, no convergence expected).
    """
    args = _parse_kv_args(sys.argv[1:])
    architecture = args.get("architecture", "E1")
    dataset = args.get("dataset", "starcop_mini")
    tier = args.get("tier", "mini")
    max_steps = int(args["max_steps"]) if "max_steps" in args else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df, val_df = _load_split(dataset, tier)

    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    model = build_model(architecture)
    with mlflow.start_run(run_name=f"{architecture}-{tier}"):
        mlflow.log_params(
            {
                "architecture": architecture,
                "dataset": dataset,
                "tier": tier,
                "device": device,
                "train_patches": len(train_df),
                "val_patches": len(val_df),
            }
        )
        result = fit(
            model,
            train_df,
            val_df,
            dataset=dataset,
            lr=1e-3 if architecture == "E1" else 1e-4,
            batch_size=16,
            max_epochs=50 if max_steps is None else 1,
            patience=10,
            max_steps=max_steps,
            augment=augment_for(architecture),
            device=device,
            checkpoint_path=_CHECKPOINT_DIR / f"{architecture}-{tier}.pt",
            num_workers=4,
            verbose=True,
        )
        mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, int | float)})
        print(f"architecture={architecture} tier={tier} dataset={dataset}")
        print(result)


if __name__ == "__main__":
    main()
