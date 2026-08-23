"""Thin glue that evaluates one paper-baseline checkpoint against the full
STARCOP paper test set -- see track-a-paper-benchmark-reproduction-plan.md
Phases 0 and 1.

The pure/injectable-boundary logic below (`load_and_place_model`,
`run_validation_safely`, `select_limit_scene_ids`,
`assert_known_scene_counts`, `validate_cli_args`) is unit tested
(__tests__/test_run_baseline_eval.py). `evaluate_variant` itself --
checkpoint download, real dataloader, real `run_validation` calls -- is
thin glue, Large-boundary, validated by the real Phase 0/1 run instead of a
unit test, same pattern `src/training/starcop_datamodule.py`'s own untested
glue follows.
"""

import hashlib
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "registry"))

import dataset_wiring  # noqa: E402
import hf_baseline_import  # noqa: E402
import paper_metrics  # noqa: E402
import select_docs_examples  # noqa: E402
from _vendor_starcop_evaluation import (  # noqa: E402
    feature_extration,
    run_validation,
    starcop_plot,
    to_device,
)

log = logging.getLogger(__name__)

KNOWN_SCENE_COUNTS = {"total": 342, "has_plume": 166, "no_plume": 176, "strong": 57, "weak": 109}


class KnownDifficultyBucketGapError(Exception):
    """Raised when run_validation's own internal (buggy) pixel-count
    difficulty groupby can't find one of its required buckets -- expected on
    small --limit samples that don't happen to include both an 'easy' and
    'hard' pixel-count plume scene. Not a sign of an environment/device
    problem; see track-a-paper-benchmark-reproduction-plan.md Phase 0."""


def load_and_place_model(
    checkpoint_path,
    device: torch.device,
    load_model_fn: Callable = hf_baseline_import.load_model,
):
    """Loads a checkpoint via `load_model_fn` (real default:
    `hf_baseline_import.load_model`, which always returns a CPU model) and
    moves it to `device` -- `run_validation` derives everything from
    `model.device`, so this step can't be skipped."""
    model, settings = load_model_fn(checkpoint_path)
    model = model.to(device)
    return model, settings


def run_validation_safely(
    model,
    dataloader,
    run_validation_fn: Callable = run_validation,
    **kwargs,
):
    """Calls `run_validation_fn` and re-raises its own difficulty-groupby
    `KeyError` (the known vendor limitation on small/skewed samples --
    train.py already documents and works around the same one) as
    `KnownDifficultyBucketGapError`, so callers can distinguish it from a
    real environment/device failure instead of getting an opaque KeyError."""
    try:
        return run_validation_fn(model, dataloader, **kwargs)
    except KeyError as exc:
        raise KnownDifficultyBucketGapError(
            f"run_validation's internal difficulty-groupby lookup failed ({exc!r}) -- "
            "expected on small samples that don't include both an 'easy' and 'hard' "
            "pixel-count plume scene; this is the same vendor limitation train.py "
            "already documents (see mlops-methane-detection-plan.md TASK-2.2), not an "
            "environment/device problem."
        ) from exc


def select_limit_scene_ids(test_df: pd.DataFrame, limit: int) -> list[str]:
    """Selects up to `limit` scene ids from `test_df` (indexed by `id`, with
    `has_plume`/`qplume` columns) for a `--limit` diagnostic dry pass.
    Deterministically includes a no-plume scene when available (guaranteed
    to fall into run_validation's own (False, "hard") pixel-count bucket,
    since a no-plume scene always has zero label pixels), then alternates
    between the highest- and lowest-qplume plume scenes. qplume is only a
    proxy for the pixel-count difficulty run_validation actually groups by,
    so this improves the odds of a clean run without guaranteeing one --
    `run_validation_safely` is the fallback when it isn't enough."""
    if limit <= 0:
        return []

    has_plume = test_df["has_plume"].astype(bool)
    no_plume_ids = list(test_df.index[~has_plume])
    plume_ids_low_to_high = list(test_df.loc[has_plume].sort_values("qplume").index)

    selected: list[str] = []
    if no_plume_ids:
        selected.append(no_plume_ids[0])

    take_from_high = True
    while len(selected) < limit and plume_ids_low_to_high:
        selected.append(plume_ids_low_to_high.pop(-1 if take_from_high else 0))
        take_from_high = not take_from_high

    for scene_id in no_plume_ids[1:]:
        if len(selected) >= limit:
            break
        selected.append(scene_id)

    return selected[:limit]


def assert_known_scene_counts(test_df: pd.DataFrame) -> None:
    """Hard-checks test.csv's scene counts against the paper's Figure 4
    test-set table (342 total, 166/176 has_plume split, 57/109
    strong/weak) -- not just eyeballed."""
    has_plume = test_df["has_plume"].astype(bool)
    is_strong = has_plume & (test_df["qplume"] >= paper_metrics.STRONG_QPLUME_THRESHOLD_KG_H)
    counts = {
        "total": len(test_df),
        "has_plume": int(has_plume.sum()),
        "no_plume": int((~has_plume).sum()),
        "strong": int(is_strong.sum()),
        "weak": int((has_plume & ~is_strong).sum()),
    }
    if counts != KNOWN_SCENE_COUNTS:
        raise AssertionError(
            f"test.csv scene counts changed: expected {KNOWN_SCENE_COUNTS}, got {counts}"
        )


def derive_features_extract(input_products: list[str]) -> list[str]:
    """Returns the subset of `input_products` that are computed features
    (e.g. MultiSTARCOP/varon's Varon ratio bands) rather than raw satellite
    bands, keyed off vendor/starcop's own `feature_extration.FEATURES`
    table -- so a variant needing on-the-fly feature extraction is detected
    automatically from its own checkpoint settings, not hand-maintained per
    variant. Empty for HyperSTARCOP's mag1c/RGB variants, whose raw band
    names aren't in FEATURES -- a strict generalization of the previous
    always-None behavior for those two variants."""
    return [product for product in input_products if product in feature_extration.FEATURES]


def validate_cli_args(limit: Optional[int], emit_docs_assets: Optional[str]) -> None:
    """`--limit` is a local smoke-test flag only; `--emit-docs-assets`
    requires the full, unlimited run (assert_known_scene_counts gates it) --
    the two can never be combined."""
    if limit is not None and emit_docs_assets is not None:
        raise ValueError("--limit and --emit-docs-assets cannot be combined")


def _mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def persist_offline_predictions(curated: dict, output_dir: Path, variant: str) -> None:
    """Persists, for each curated scene, the offline predicted mask's sha256
    digest and the confidence array -- the artifact Phase 5's
    `live_verify.py` diffs the live API's response against (not all 342
    scenes, to avoid bloating the artifact store).

    Filenames are prefixed with `variant` (`{variant}_{scene_id}.json`),
    same convention as the curated sample-mask PNGs already use -- all
    three variants share one staging directory within a single
    `eval_baseline` flow run (`run_eval_baseline_cycle`'s single
    `tempfile.TemporaryDirectory()` wraps the whole evaluate loop), so an
    unprefixed filename would let a later variant's MLflow artifact upload
    silently pick up an earlier variant's leftover scene JSONs sitting in
    the same folder -- confirmed live 2026-08-22 (mag1c_only's uploaded
    offline_predictions/ had 8 files, 4 its own + 4 leftover from varon;
    mag1c_rgb's had 12), which made Phase 5's live_verify.py compare the
    wrong variant's stored predictions against the live server and
    correctly report a real mismatch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for scene_id, arrays in curated.items():
        record = {
            "scene_id": scene_id,
            "mask_sha256": _mask_digest(arrays["mask"]),
            "confidence": arrays["confidence"].tolist(),
        }
        with open(output_dir / f"{variant}_{scene_id}.json", "w") as fh:
            json.dump(record, fh)


def evaluate_variant(
    variant: str,
    device: torch.device,
    test_csv_path,
    root_folder,
    limit: Optional[int] = None,
    emit_docs_assets_dir: Optional[Path] = None,
    features_extract: Optional[list[str]] = None,
) -> dict:
    """Evaluates `variant` (a checkpoint already downloadable via
    `hf_baseline_import`) against the full (or `--limit`-ed) test set.
    Returns a dict with the corrected paper metrics, the curated scene
    picks, and a status describing how the run completed."""
    validate_cli_args(limit, emit_docs_assets_dir)

    test_df = dataset_wiring._load_dataframe(test_csv_path, root_folder)
    if limit is None:
        assert_known_scene_counts(test_df)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path, _config_path, checkpoint_provenance = (
            hf_baseline_import.resolve_checkpoint(variant, Path(tmp))
        )
        model, settings = load_and_place_model(checkpoint_path, device)

    input_products = list(settings.dataset.input_products)
    output_products = list(settings.dataset.output_products)
    weight_loss = (
        settings.dataset.get("weight_loss") if settings.dataset.get("use_weight_loss") else None
    )
    if features_extract is None:
        features_extract = derive_features_extract(input_products)

    scene_ids = select_limit_scene_ids(test_df, limit) if limit is not None else None
    eval_df = test_df.loc[scene_ids] if scene_ids is not None else test_df

    dataloader = dataset_wiring.build_test_dataloader(
        test_csv_path,
        root_folder,
        input_products=input_products,
        output_products=output_products,
        weight_loss=weight_loss,
        features_extract=features_extract,
        scene_ids=scene_ids,
        batch_size=1,
    )

    try:
        with tempfile.TemporaryDirectory() as metrics_pass_dir:
            # run_validation's `path_save_results=None` default is broken --
            # it unconditionally calls `.startswith("gs://")` on it
            # (validation.py:72) with no None guard. Every existing caller
            # in this repo (train.py) always passes an explicit path, so
            # nobody hit this before; passing a throwaway tempdir here works
            # around it without touching vendor/starcop.
            out_data, run_validation_metrics = run_validation_safely(
                model,
                dataloader,
                products_plot=None,
                verbose=False,
                show_plots=False,
                path_save_results=metrics_pass_dir,
            )
    except KnownDifficultyBucketGapError as exc:
        log.warning("%s (variant=%s, limit=%s)", exc, variant, limit)
        return {"status": "known_difficulty_bucket_gap", "variant": variant, "error": str(exc)}

    # Both out_data (run_validation's own out_data.set_index("id")) and
    # eval_df (dataset_wiring._load_dataframe's own .set_index("id")) are
    # already id-indexed -- join_scene_results_with_test_csv expects exactly
    # that shape, no reset_index needed.
    joined = paper_metrics.join_scene_results_with_test_csv(out_data, eval_df)
    buckets = paper_metrics.bucket_confusion_matrices(joined)
    metrics = paper_metrics.compute_bucket_metrics(buckets)
    metrics["auprc"] = paper_metrics.derive_auprc(run_validation_metrics["thresholded"])
    # The paper's own reported FPR is a tile-level classification rate, not
    # the pixel-summed no_plume_fpr_pixel_level compute_bucket_metrics also
    # returns -- see paper_metrics.tile_no_plume_fpr's docstring.
    metrics["no_plume_FPR"] = paper_metrics.tile_no_plume_fpr(joined)

    picks = select_docs_examples.select_docs_examples(joined)

    result = {
        "status": "ok",
        "variant": variant,
        "n_scenes": len(out_data),
        "metrics": metrics,
        "docs_examples": picks,
        # Phase 3 (MLflow permanent record) needs these; Phase 1/2 never did --
        # a strict addition, no existing key's shape changes.
        # reset_index first: joined is id-indexed, and to_dict(orient="records")
        # drops the index entirely -- without this, every record silently
        # loses its scene id.
        "joined_scene_results": joined.reset_index().to_dict(orient="records"),
        "run_validation_metrics": run_validation_metrics,
        "checkpoint_provenance": checkpoint_provenance,
        "device": str(device),
    }

    curated_ids = [scene_id for scene_id in picks.values() if scene_id is not None]
    if curated_ids:
        curated_dataloader = dataset_wiring.build_test_dataloader(
            test_csv_path,
            root_folder,
            input_products=input_products,
            output_products=output_products,
            weight_loss=weight_loss,
            features_extract=features_extract,
            scene_ids=curated_ids,
            batch_size=1,
        )

        curated_arrays = {}
        products_plot = list(settings.get("products_plot", []) or [])
        for plume_data in curated_dataloader:
            plume_data = model.batch_with_preds(to_device(plume_data, model.device))
            scene_id = plume_data["id"][0]
            curated_arrays[scene_id] = {
                "mask": plume_data["pred_binary"][0, 0].cpu().numpy(),
                "confidence": plume_data["prediction"][0, 0].detach().cpu().numpy(),
            }
            if emit_docs_assets_dir is not None:
                fig, _ = starcop_plot.plot_batch(
                    to_device(plume_data, "cpu"),
                    input_products=input_products,
                    products_plot=products_plot,
                    figsize_ax=(4, 4),
                )
                emit_docs_assets_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    emit_docs_assets_dir / f"{variant}_{scene_id}.png", dpi=120, bbox_inches="tight"
                )

        if emit_docs_assets_dir is not None:
            persist_offline_predictions(
                curated_arrays, emit_docs_assets_dir / "offline_predictions", variant
            )

    return result
