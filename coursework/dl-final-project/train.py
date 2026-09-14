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
import time
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from architectures import build_e1, build_e2, build_e3
from dataset import PatchDataset
from early_stopping import EarlyStopper
from losses import build_loss, compute_pos_weight
from torch.utils.data import DataLoader

_DEFAULT_SEED = 42

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


def set_seed(seed: int) -> None:
    """Seed every RNG `fit()` draws from -- torch (CPU + CUDA) and numpy.

    Plan Section 7.1 Phase A1: `train.py` seeded nothing before this, so
    two runs of the same configuration (weight init via the caller's own
    pre-`fit()` seeding, kornia augmentation, `DataLoader` shuffling --
    all of which read from these same global generators) could not be
    reproduced. Call this before building the model too, not only inside
    `fit()`, so weight initialization is also covered.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total element count across all of `model`'s parameter tensors.

    Plan Section 7.1 Phase A2: logged as the `param_count` MLflow param so
    Section 9's parameter-count column doesn't have to be recomputed or
    retyped from `architectures.py`'s own docstrings later.
    """
    return sum(p.numel() for p in model.parameters())


def library_versions() -> dict[str, str]:
    """Return `{torch_version, smp_version, cuda_version}` for MLflow params.

    Plan Section 7.1 Phase A2: "record exact library versions, not just
    'torch' without a version pin" -- read directly from the imported
    modules so this can't drift out of sync with what actually ran.
    `cuda_version` is `"cpu"` on a build with no CUDA support.
    """
    return {
        "torch_version": torch.__version__,
        "smp_version": smp.__version__,
        "cuda_version": torch.version.cuda or "cpu",
    }


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """`DataLoader` `worker_init_fn`: reseed numpy/torch per worker off `base_seed`.

    Each worker process needs its own deterministic-but-distinct seed --
    without this, workers otherwise fall back to arbitrary per-process RNG
    state, and augmentation draws would no longer be reproducible under
    `num_workers>0` even with the main process's RNGs seeded.
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _epoch_marker(is_best: bool) -> str:
    """One-glyph marker for an epoch's outcome, for `fit(verbose=True)`'s printed line.

    Purely cosmetic terminal readability -- a `raw-full` epoch takes
    ~15 minutes and previously printed nothing until it finished, so at a
    glance it wasn't obvious whether a given epoch had actually improved
    on the run's best result. Carries no information not already in
    `EarlyStopper.is_best`, and plays no role in checkpointing or
    early-stopping decisions themselves.
    """
    return "✅" if is_best else "❌"


def evaluate(
    model: torch.nn.Module,
    loader,
    loss_fn: torch.nn.Module,
    device: str,
    max_batches: int | None = None,
) -> dict:
    """Run `model` in eval mode over `loader`; return loss/F1/degeneracy metrics.

    Accumulates true/false positive/negative counts (and an all-zero/
    all-one flag) incrementally per batch rather than concatenating every
    batch's predictions -- `starcop_raw`'s full val split (26,607 patches)
    OOM'd a first version of this function that did exactly that. Same
    incremental-accumulation reasoning as `stats.py`'s own
    `compute_band_stats`/`compute_class_distribution` (Section 0.1): O(1)
    memory per batch, not O(dataset size).

    `max_batches` (plan Section 7.1 Phase A4, default `None` = every
    batch) stops after that many batches -- for Phase B's dry runs, where
    `max_steps=2` on `tier=raw-full` would otherwise still walk all 26,607
    val patches every "epoch" before the loop even reaches the cheap
    part. Breaking out of the loop, not slicing `loader` beforehand, means
    the un-visited patches are never read off disk at all.
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
            if max_batches is not None and n_batches >= max_batches:
                break
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
    seed: int = _DEFAULT_SEED,
    max_val_batches: int | None = None,
    progress_every: int | None = None,
) -> dict:
    """Train `model` on `train_df`, validate on `val_df`; return final metrics + run length.

    `progress_every` (optional, default `None` = off) prints a mid-epoch
    heartbeat line every that many (cumulative) steps when `verbose=True`
    -- a `raw-full` epoch takes ~15 minutes and otherwise prints nothing
    until it finishes. Harmless to leave off for `mini`/`r2`, whose
    epochs are already short (25-380 steps) and finish before a
    reasonably-sized interval would ever fire.

    `max_val_batches` (plan Section 7.1 Phase A4, default `None` = full
    val split) is forwarded to every `evaluate()` call this makes -- for
    dry-running a cell at `tier=raw-full` with `max_steps=2`, so the val
    pass doesn't still walk all 26,607 patches every "epoch" (`raw-smoke`
    already had its own escape hatch for this via a pre-sliced `val_df` in
    `_load_split`; this is the general version for any tier).

    `seed` (plan Section 7.1 Phase A1) is applied via `set_seed()` before
    the loaders are built and threaded into each `DataLoader`'s `generator=`
    / `worker_init_fn=`, so calling `fit()` twice with the same seed and the
    same starting model weights reproduces the same shuffle order,
    augmentation draws, and therefore the same metrics. It does **not**
    seed the model's own weight initialization -- that already happened
    before `model` was passed in; call `set_seed()` yourself before
    building the model if that also needs to be reproducible (`main()`
    does this).

    `wall_clock_seconds` (total) and `seconds_per_epoch` (average; also
    logged per-epoch to MLflow individually) are returned so Section 7's
    R3 epoch-cap sizing can be based on a number this function actually
    measured, not the `stats.py` I/O-timing estimate carried forward in
    that section's own prose (plan Section 7.1 Phase A3).

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
    set_seed(seed)
    model.to(device)
    pos_weight = compute_pos_weight(train_df)
    loss_fn = build_loss(pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = (
        {
            "num_workers": num_workers,
            "persistent_workers": True,
            "worker_init_fn": partial(_seed_worker, base_seed=seed),
        }
        if num_workers > 0
        else {}
    )
    train_loader = DataLoader(
        PatchDataset(train_df, dataset=dataset, augment=augment),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
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
    fit_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        for batch in train_loader:
            x, y = batch["input"].to(device), batch["output"].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            step_count += 1
            if verbose and progress_every and step_count % progress_every == 0:
                print(
                    f"    ... epoch {epoch}/{max_epochs} step {step_count} "
                    f"({time.perf_counter() - epoch_start:.0f}s elapsed this epoch)"
                )
            if max_steps is not None and step_count >= max_steps:
                break

        val_metrics = evaluate(model, val_loader, loss_fn, device, max_batches=max_val_batches)
        epoch_seconds = time.perf_counter() - epoch_start
        if log_to_mlflow:
            mlflow.log_metrics(
                {
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1"],
                    "seconds_per_epoch": epoch_seconds,
                },
                step=epoch,
            )
        # Computed here, before the verbose print below, so the printed
        # line's marker (`_epoch_marker`) reflects this epoch's own
        # is_best outcome. `step()` mutates EarlyStopper's internal
        # counter/best -- call it exactly once per epoch, never again
        # below, or patience gets silently consumed twice as fast.
        should_stop = stopper.step(val_metrics["loss"])
        if verbose:
            print(
                f"  {_epoch_marker(stopper.is_best)} epoch {epoch}/{max_epochs} "
                f"step {step_count}: val_loss={val_metrics['loss']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} ({epoch_seconds:.1f}s)"
            )

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
    wall_clock_seconds = time.perf_counter() - fit_start
    return {
        **best_metrics,
        "best_epoch": best_epoch,
        "last_epoch_metrics": val_metrics,
        "epochs_run": epoch,
        "steps_run": step_count,
        "pos_weight": pos_weight,
        "stopped_early": stop_early,
        "wall_clock_seconds": wall_clock_seconds,
        # Average over the run, not the last epoch's own time -- this is
        # the number Section 7's R3 epoch-cap sizing is meant to read
        # (plan Section 7.1 Phase A3), and per-epoch times are already
        # visible individually via the `seconds_per_epoch` metric logged
        # inside the loop above.
        "seconds_per_epoch": wall_clock_seconds / epoch,
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
           [seed=42] [max_val_batches=2] [max_epochs=20] [patience=10]
    `tier` is one of 'mini' | 'raw-smoke' | 'r2' -- only affects which split is loaded and,
    for 'raw-smoke', how the run is labeled; `max_steps` (optional) caps step count,
    used for the R1 smoke-training run (a few hundred steps, no convergence expected).
    `seed` (plan Section 7.1 Phase A1, default 42) is applied via `set_seed()` before
    `build_model()` so weight initialization is reproducible too, then passed to `fit()`
    for the loader/augmentation seeding described there.
    `max_val_batches` (plan Section 7.1 Phase A4, optional, default no cap) caps how many
    val batches `evaluate()` reads per epoch -- pair with a small `max_steps` for Phase
    B's dry runs on `tier=raw-full`/`r2`, where a full val pass would otherwise dominate
    the wall-clock time a "smoke test" is supposed to save.
    `max_epochs` (optional, default 50, or 1 when `max_steps` is set) overrides the
    schedule's epoch cap -- added for R3 (plan Section 7 Phase E), whose overnight
    time budget needs a smaller cap than mini/r2's 50 (sized from measured per-step
    cost, not the default). `patience` (optional, default 10) is exposed for
    completeness but R3 keeps it at 10 deliberately -- Section 7's own rule holds
    early-stopping patience fixed across tiers; only the epoch cap is the allowed
    per-tier exception.
    `progress_every` (optional, default no heartbeat) prints a mid-epoch progress
    line every that many cumulative steps -- worth setting for `raw-full`, whose
    ~15-minute epochs otherwise print nothing until they finish.
    """
    args = _parse_kv_args(sys.argv[1:])
    architecture = args.get("architecture", "E1")
    dataset = args.get("dataset", "starcop_mini")
    tier = args.get("tier", "mini")
    max_steps = int(args["max_steps"]) if "max_steps" in args else None
    seed = int(args.get("seed", _DEFAULT_SEED))
    max_val_batches = int(args["max_val_batches"]) if "max_val_batches" in args else None
    progress_every = int(args["progress_every"]) if "progress_every" in args else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Named once here so the values logged as MLflow params are the exact
    # same values `fit()` trains with -- not a second, hand-copied literal
    # that could silently drift from what actually ran (plan Section 7.1
    # Phase A2).
    lr = 1e-3 if architecture == "E1" else 1e-4
    batch_size = 16
    default_max_epochs = 50 if max_steps is None else 1
    max_epochs = int(args.get("max_epochs", default_max_epochs))
    patience = int(args.get("patience", 10))
    num_workers = 4
    augment = augment_for(architecture)
    loss_name = "BCEWithLogitsLoss+pos_weight"
    optimizer_name = "Adam"

    train_df, val_df = _load_split(dataset, tier)

    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    set_seed(seed)
    model = build_model(architecture)
    with mlflow.start_run(run_name=f"{architecture}-{tier}"):
        mlflow.log_params(
            {
                "architecture": architecture,
                "dataset": dataset,
                "tier": tier,
                "device": device,
                "seed": seed,
                "train_patches": len(train_df),
                "val_patches": len(val_df),
                "lr": lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "num_workers": num_workers,
                "augment": augment,
                "loss": loss_name,
                "optimizer": optimizer_name,
                "param_count": count_parameters(model),
                **library_versions(),
            }
        )
        result = fit(
            model,
            train_df,
            val_df,
            dataset=dataset,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            max_steps=max_steps,
            augment=augment,
            device=device,
            checkpoint_path=_CHECKPOINT_DIR / f"{architecture}-{tier}.pt",
            num_workers=num_workers,
            verbose=True,
            seed=seed,
            max_val_batches=max_val_batches,
            progress_every=progress_every,
        )
        mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, int | float)})
        print(f"architecture={architecture} tier={tier} dataset={dataset}")
        print(result)


if __name__ == "__main__":
    main()
