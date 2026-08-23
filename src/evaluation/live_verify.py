"""Phase 5: verifies the live BentoML `/predict` endpoint agrees with
Phase 1/3's own offline predictions for a variant's curated scenes -- not
just a derived plume-pixel count, pixel-for-pixel mask equality (via a
sha256 digest) plus confidence within an explicit tolerance. See
track-a-paper-benchmark-reproduction-plan.md Phase 5.

Pure, unit-tested logic (`mask_sha256`, `compare_prediction`,
`assert_model_identity`, `assert_variant_is_servable`,
`parse_offline_predictions_dir`, `array_to_npy_bytes`) plus
`resolve_paper_eval_run` (Test Size: Medium, real sqlite MlflowClient, same
convention as paper_eval_mlflow.py::check_registry_version_matches).
`verify_variant` itself is thin HTTP/dataloader glue, Large-boundary,
real-run validated against an actual `bentoml serve` process rather than
unit tested -- same pattern as run_baseline_eval.py::evaluate_variant.
"""

import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
import torch
from mlflow.tracking import MlflowClient

_EVAL_DIR = str(Path(__file__).resolve().parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
_REGISTRY_DIR = str(Path(__file__).resolve().parents[1] / "registry")
if _REGISTRY_DIR not in sys.path:
    sys.path.insert(0, _REGISTRY_DIR)

import dataset_wiring  # noqa: E402
import hf_baseline_import  # noqa: E402
from run_baseline_eval import derive_features_extract, load_and_place_model  # noqa: E402

PAPER_EVAL_EXPERIMENT_NAME = "starcop-paper-eval"
DEFAULT_ATOL = 1e-5
SERVABLE_VARIANTS = ("mag1c_only", "mag1c_rgb")


class ModelIdentityMismatch(Exception):
    """Raised when the served model's identity doesn't match the
    registry_version this run's offline predictions were evaluated
    against -- refuses to compare predictions against a possibly-wrong
    model rather than silently "passing" a check that verified nothing."""


def mask_sha256(mask) -> str:
    """sha256 of a mask -- accepts either a numpy array or the plain nested
    list a JSON response round-trips through (predict_response's "mask"
    field), matching run_baseline_eval.py::_mask_digest's recipe so the two
    are bit-for-bit comparable."""
    array = np.ascontiguousarray(np.asarray(mask, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def compare_prediction(offline: dict, live: dict, atol: float = DEFAULT_ATOL) -> dict:
    """Pure comparison of one scene's offline-recorded prediction
    (`{"scene_id", "mask_sha256", "confidence"}`, from Phase 1's
    offline_predictions/*.json artifact) against the live `/predict`
    response (`{"mask", "confidence"}`). Mask equality is pixel-for-pixel
    via digest match, not a derived plume-pixel count -- two materially
    different masks can share a pixel count. Confidence is compared within
    `atol`, not exact float equality, to tolerate CPU/MPS numeric
    differences."""
    live_digest = mask_sha256(live["mask"])
    mask_match = live_digest == offline["mask_sha256"]
    confidence_match = bool(
        np.allclose(
            np.asarray(live["confidence"]),
            np.asarray(offline["confidence"]),
            atol=atol,
            rtol=0.0,
        )
    )
    return {
        "scene_id": offline["scene_id"],
        "mask_match": mask_match,
        "confidence_match": confidence_match,
        "passed": mask_match and confidence_match,
        "live_mask_sha256": live_digest,
        "offline_mask_sha256": offline["mask_sha256"],
    }


def assert_model_identity(
    health: dict, expected_model_name: str, expected_registry_version
) -> None:
    """Raises ModelIdentityMismatch unless the served model's
    `/health`-reported `model_name`/`model_version` both match
    `expected_model_name`/`expected_registry_version` (the registry model
    name + registry_version tag Phase 3 logged for the run being verified).
    Comparing the version number alone is not enough: each registry model
    name has its own independent version counter, so `mag1c_only` and
    `mag1c_rgb` can legitimately sit at the identical version number (e.g.
    both "2") while being different models entirely -- version-only
    comparison would silently pass a check that's actually comparing
    against the wrong served model. `Staging` can also silently drift to a
    different version between the offline run and the live check (see this
    phase's own "Pin the served model" note)."""
    served_name = health["model_name"]
    served_version = str(health["model_version"])
    expected_version = str(expected_registry_version)
    if served_name != expected_model_name or served_version != expected_version:
        raise ModelIdentityMismatch(
            f"served model {served_name!r} v{served_version!r} does not match the expected "
            f"{expected_model_name!r} v{expected_version!r} for this paper-eval run -- refusing "
            "to compare predictions against a possibly-wrong model"
        )


def assert_variant_is_servable(variant: str) -> None:
    """Raises ValueError for `varon` (MultiSTARCOP) -- only the two Hyper
    variants are servable live today (see this phase's own "Explicit scope
    limit" note): the production service loads exactly one
    MODEL_NAME/MODEL_STAGE at a time, and the serving inference pipeline has
    no Varon ratio-band feature extraction wired in at all."""
    if variant not in SERVABLE_VARIANTS:
        raise ValueError(
            f"variant {variant!r} is not servable live -- only {SERVABLE_VARIANTS} are "
            "(MultiSTARCOP/varon needs a separate local bentoml serve, out of this phase's scope)"
        )


def parse_offline_predictions_dir(directory) -> dict:
    """Pure: parses an already-downloaded offline_predictions/*.json
    directory (Phase 1's persist_offline_predictions output) into a dict
    keyed by scene_id."""
    predictions = {}
    for json_path in sorted(Path(directory).glob("*.json")):
        record = json.loads(json_path.read_text())
        predictions[record["scene_id"]] = record
    return predictions


def array_to_npy_bytes(array: np.ndarray) -> bytes:
    """Serializes `array` to .npy bytes -- the format service.py's
    `/predict` expects as an uploaded file (`np.load(file)`)."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def resolve_paper_eval_run(
    client: MlflowClient, variant: str, experiment_name: str = PAPER_EVAL_EXPERIMENT_NAME
):
    """Finds the most recent active `starcop-paper-eval` run tagged with
    `variant`. Phase 4's flow is explicitly repeatable -- runs accumulate
    over time in this experiment as the permanent audit trail Phase 3 was
    built for, so requiring exactly one run to ever exist (this function's
    original design) broke on every re-run after the first: caught live
    2026-08-22, when a real flow execution's own `evaluate` step created a
    fresh run for a variant alongside runs already there from earlier
    attempts, and the live-check step silently degraded to `"not_run"`
    instead of erroring loudly. Old runs are not stale duplicates to clean
    up, they're history -- pick the newest by `start_time`, don't demand
    there be only one. Still raises if none exist at all -- that's a
    genuine "never logged for this variant" error, not an expected
    accumulation state. `client.search_runs` only sees active runs by
    default, so a soft-deleted run doesn't count."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"no such experiment: {experiment_name!r}")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.variant = '{variant}'",
        order_by=["start_time DESC"],
    )
    if len(runs) == 0:
        raise ValueError(f"no {experiment_name!r} run found for variant={variant!r}")
    return runs[0]


def _check_health(base_url: str, http_post: Callable = requests.post) -> dict:
    response = http_post(f"{base_url}/health", timeout=30)
    response.raise_for_status()
    return response.json()


def _post_predict(base_url: str, array: np.ndarray, http_post: Callable = requests.post) -> dict:
    response = http_post(
        f"{base_url}/predict",
        files={"file": ("scene.npy", array_to_npy_bytes(array), "application/octet-stream")},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _build_scene_input_array(
    test_csv_path,
    root_folder,
    input_products,
    output_products,
    weight_loss,
    features_extract,
    scene_id,
) -> np.ndarray:
    """Rebuilds the raw, pre-normalization input array `/predict` expects
    for `scene_id` -- verified empirically end-to-end against a real
    checkpoint (see this phase's own "Per-scene input-array construction"
    note): `batch["input"]` is exactly what the model's own `forward` would
    normalize internally, so no separate normalization step belongs here."""
    dataloader = dataset_wiring.build_test_dataloader(
        test_csv_path,
        root_folder,
        input_products=input_products,
        output_products=output_products,
        weight_loss=weight_loss,
        features_extract=features_extract,
        scene_ids=[scene_id],
        batch_size=1,
    )
    batch = next(iter(dataloader))
    return batch["input"].squeeze(0).numpy()


def verify_variant(
    variant: str,
    base_url: str,
    tracking_uri: str,
    test_csv_path,
    root_folder,
    atol: float = DEFAULT_ATOL,
    client: Optional[MlflowClient] = None,
) -> dict:
    """Large-boundary, real-run validated (not unit tested) -- same pattern
    as evaluate_variant()/import_variant(). Verifies `variant`'s live
    `/predict` responses against Phase 1/3's offline predictions for its
    curated scenes, one HTTP round-trip per scene, pinning the served model
    to the exact checkpoint being verified first."""
    assert_variant_is_servable(variant)

    if client is None:
        client = MlflowClient(tracking_uri=tracking_uri)

    run = resolve_paper_eval_run(client, variant)
    expected_model_name = hf_baseline_import.registry_model_name(variant)
    expected_registry_version = run.data.tags["registry_version"]

    health = _check_health(base_url)
    assert_model_identity(health, expected_model_name, expected_registry_version)

    with tempfile.TemporaryDirectory() as scratch:
        offline_dir = client.download_artifacts(run.info.run_id, "offline_predictions", scratch)
        offline_predictions = parse_offline_predictions_dir(offline_dir)

        checkpoint_path, _config_path, provenance = hf_baseline_import.resolve_checkpoint(
            variant, Path(scratch) / "checkpoint"
        )
        expected_checkpoint_sha256 = run.data.tags.get("checkpoint_sha256")
        if provenance["checkpoint_sha256"] != expected_checkpoint_sha256:
            raise ModelIdentityMismatch(
                f"resolved checkpoint for variant={variant!r} has "
                f"checkpoint_sha256={provenance['checkpoint_sha256']!r}, but run "
                f"{run.info.run_id!r}'s offline predictions were evaluated with "
                f"checkpoint_sha256={expected_checkpoint_sha256!r}"
            )
        _model, settings = load_and_place_model(checkpoint_path, device=torch.device("cpu"))

    if not offline_predictions:
        raise ValueError(
            f"run {run.info.run_id!r} has no offline_predictions artifacts to verify against"
        )

    input_products = list(settings.dataset.input_products)
    output_products = list(settings.dataset.output_products)
    weight_loss = (
        settings.dataset.get("weight_loss") if settings.dataset.get("use_weight_loss") else None
    )
    features_extract = derive_features_extract(input_products)

    results = []
    for scene_id, offline in offline_predictions.items():
        array = _build_scene_input_array(
            test_csv_path,
            root_folder,
            input_products,
            output_products,
            weight_loss,
            features_extract,
            scene_id,
        )
        live = _post_predict(base_url, array)
        results.append(compare_prediction(offline, live, atol=atol))

    return {
        "variant": variant,
        "run_id": run.info.run_id,
        "served_model_version": health["model_version"],
        "results": results,
        "passed": all(r["passed"] for r in results),
    }
