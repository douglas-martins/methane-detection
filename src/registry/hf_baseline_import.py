"""Imports a pretrained STARCOP checkpoint from HuggingFace (isp-uv-es/starcop)
into MLflow as a comparison baseline -- so models trained by this project's
own pipeline (src/training/train.py) have a reference point already in the
registry, without reproducing STARCOP's original training run.

Reconstructs the real STARCOP LightningModule (ModelModule /
ModelModuleRegression, picked via the same model_mode dispatch
vendor/starcop/starcop/model_setup.get_model uses) from the checkpoint's own
saved hyperparameters, rather than get_model()'s local-checkpoint-path
convention, which doesn't fit a HuggingFace-downloaded checkpoint. Classes
come from _vendor_starcop_baseline.py; no vendor/starcop file is modified.

Pure functions (variant_subfolder, registry_model_name, model_class_for_mode)
are unit tested (Test Size: Small/Medium, no network -- see
__tests__/test_hf_baseline_import.py). download_checkpoint/load_model/
import_variant are SDK glue -- network + a live MLflow server -- validated by
an actual run instead (same Test Size: Large boundary src/training/train.py's
own live path is validated at, see internal-docs/setup/environment-notes.md).
"""

import logging
import tempfile
from pathlib import Path

_HF_REPO = "isp-uv-es/starcop"
_SENSOR = "AVIRIS-NG"  # both HF variants are trained on AVIRIS-NG, same as this project's own data

_VARIANT_SUBFOLDERS = {
    "mag1c_only": "hyperstarcop_mag1c_only",
    "mag1c_rgb": "hyperstarcop_mag1c_rgb",
}

# MultiSTARCOP/varon was never published to _HF_REPO (verified live via
# HfApi().list_repo_files -- only the two hyperstarcop_* subfolders above
# exist there) -- it ships only as a Google-Drive zip
# (src/data/download/download_mini_dataset.py), already pulled and
# DVC-tracked at this path (models/starcop_baseline.dvc). See
# track-a-paper-benchmark-reproduction-plan.md Phase 2.
_LOCAL_CHECKPOINT_PATHS = {
    "varon": Path("models/starcop_baseline/multistarcop_varon"),
}

_MODEL_MODE_CLASS_NAMES = {
    "segmentation_output": "ModelModule",
    "regression_output": "ModelModuleRegression",
}

# Pinned sha256 of each variant's final_checkpoint_model.ckpt, recorded from
# the exact files downloaded, loaded, and verified working (2026-08-12) --
# see HfApi().model_info(_HF_REPO, files_metadata=True)'s per-file
# BlobLfsInfo.sha256. load_model() below must deserialize this file with
# torch.load(weights_only=False) (the checkpoint's hyper_parameters is a real
# OmegaConf DictConfig, which weights_only=True's restricted unpickler
# rejects), so this digest check -- not HTTPS/revision-pinning alone -- is
# what actually stops an unreviewed/compromised upstream file from being
# unpickled. A legitimate upstream update requires updating this table too.
_EXPECTED_CHECKPOINT_SHA256 = {
    "mag1c_only": "2d4391d5b05f90c411fd459db5bbe4e88650e5ff30ec2eb10d36c66ed0a43137",
    "mag1c_rgb": "96e274be943f64e028faded3bac3d1ee325ee7a79d6de2ee7f5deeaea1ef188d",
    # Weaker trust model than the two entries above: those are checked
    # against an independent source (HF's own published blob sha256 via
    # HfApi().model_info(files_metadata=True)). varon has no independent
    # upstream digest to check against -- this is a hash of whatever was
    # already on disk (originally pulled via gdown from Google Drive, now
    # DVC-pinned), so it guards against future silent corruption of the
    # tracked file but does not verify the original download was authentic.
    "varon": "ccc9a7ec3d0bc8acf7e2f6232e798da0744d0890d02c0e2c54d08b3b1702b37e",
}

# Full commit sha of _HF_REPO, reviewed and pinned rather than resolved live
# via HfApi().model_info() -- the same commit both files above (and both
# _EXPECTED_CHECKPOINT_SHA256 entries) were downloaded and verified against
# (2026-08-12). A live lookup would follow main to whatever it currently
# points at, which the digest check alone doesn't fully cover: config.yaml
# has no digest pin, and a live lookup is one more untrusted response in the
# path before verify_checkpoint_digest ever runs. Bumping this is a
# deliberate, reviewed code change, same as updating the digest table above.
_PINNED_REVISION = "b5fd9c0d1028321ab2d6791623e16e910fd45289"


def variant_subfolder(variant: str) -> str:
    """Returns the `models/<subfolder>/` path segment `variant` lives under
    in the isp-uv-es/starcop HF repo. Raises ValueError for an unknown
    variant."""
    try:
        return _VARIANT_SUBFOLDERS[variant]
    except KeyError:
        raise ValueError(
            f"unknown_variant {variant!r}; expected one of {sorted(_VARIANT_SUBFOLDERS)}"
        ) from None


def registry_model_name(variant: str) -> str:
    """Returns the MLflow registered-model name `variant` is imported under,
    e.g. "mag1c_only" -> "starcop-baseline-mag1c-only"."""
    return f"starcop-baseline-{variant.replace('_', '-')}"


def model_class_for_mode(model_mode: str):
    """Returns the STARCOP LightningModule subclass matching `model_mode`,
    mirroring vendor/starcop/starcop/model_setup.get_model's own dispatch.
    Raises ValueError for an unknown mode."""
    import _vendor_starcop_baseline as vendor

    try:
        class_name = _MODEL_MODE_CLASS_NAMES[model_mode]
    except KeyError:
        raise ValueError(
            f"unknown_mode {model_mode!r}; expected one of {sorted(_MODEL_MODE_CLASS_NAMES)}"
        ) from None
    return getattr(vendor, class_name)


def verify_checkpoint_digest(variant: str, checkpoint_path: Path) -> None:
    """Raises ValueError if checkpoint_path's sha256 doesn't match
    _EXPECTED_CHECKPOINT_SHA256[variant]. Reads in chunks via `hashlib.sha256().update()`
    rather than `hashlib.file_digest` (added in Python 3.11) -- this project
    runs a dual-environment setup where Environment A (vendor/starcop/.venv)
    is Python 3.10, so this function must work on both."""
    import hashlib

    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()

    expected = _EXPECTED_CHECKPOINT_SHA256[variant]
    if actual != expected:
        raise ValueError(
            f"checkpoint digest mismatch for variant {variant!r}: expected sha256 "
            f"{expected}, got {actual}. Refusing to unpickle a checkpoint that "
            "doesn't match the pinned, reviewed digest."
        )


def download_checkpoint(variant: str, dest_dir: Path) -> tuple[Path, Path, str]:
    """Downloads config.yaml + checkpoint for `variant` from _HF_REPO at
    _PINNED_REVISION into dest_dir -- a fixed, reviewed commit rather than a
    live lookup of main, so both files always come from the exact same,
    already-reviewed commit. Verifies the checkpoint's digest against
    _EXPECTED_CHECKPOINT_SHA256 before returning (see
    verify_checkpoint_digest). Returns (checkpoint_path, config_path,
    _PINNED_REVISION)."""
    from huggingface_hub import hf_hub_download

    subfolder = variant_subfolder(variant)
    revision = _PINNED_REVISION
    checkpoint_path = Path(
        hf_hub_download(
            _HF_REPO,
            f"models/{subfolder}/final_checkpoint_model.ckpt",
            revision=revision,
            local_dir=str(dest_dir),
        )
    )
    verify_checkpoint_digest(variant, checkpoint_path)
    config_path = Path(
        hf_hub_download(
            _HF_REPO,
            f"models/{subfolder}/config.yaml",
            revision=revision,
            local_dir=str(dest_dir),
        )
    )
    return checkpoint_path, config_path, revision


def local_checkpoint_dir(variant: str) -> Path:
    """Returns the local, DVC-tracked directory `variant` lives in when it's
    not sourced from HuggingFace (e.g. MultiSTARCOP/varon, published only as
    a Google-Drive zip). Raises ValueError for a variant not in this table."""
    try:
        return _LOCAL_CHECKPOINT_PATHS[variant]
    except KeyError:
        raise ValueError(
            f"unknown_local_variant {variant!r}; expected one of {sorted(_LOCAL_CHECKPOINT_PATHS)}"
        ) from None


def resolve_checkpoint(
    variant: str,
    dest_dir: Path,
    download_checkpoint_fn=download_checkpoint,
) -> tuple[Path, Path, dict]:
    """Returns (checkpoint_path, config_path, provenance_tags) for `variant`,
    dispatching on where it actually lives -- HuggingFace (`_VARIANT_SUBFOLDERS`,
    via `download_checkpoint_fn`) or local/DVC-tracked
    (`_LOCAL_CHECKPOINT_PATHS`, read straight off disk, digest-verified, no
    network). This is the single entry point both `import_variant` (registry
    import) and `run_baseline_eval.py::evaluate_variant` (the real eval run)
    call instead of `download_checkpoint` directly, so a variant's source is
    decided in exactly one place. Raises ValueError for a variant in neither
    table."""
    if variant in _VARIANT_SUBFOLDERS:
        checkpoint_path, config_path, revision = download_checkpoint_fn(variant, dest_dir)
        return (
            checkpoint_path,
            config_path,
            {
                "source": "huggingface",
                "hf_repo": _HF_REPO,
                "hf_revision": revision,
                "hf_subfolder": variant_subfolder(variant),
                "checkpoint_sha256": _EXPECTED_CHECKPOINT_SHA256[variant],
            },
        )
    if variant in _LOCAL_CHECKPOINT_PATHS:
        checkpoint_dir = local_checkpoint_dir(variant)
        checkpoint_path = checkpoint_dir / "final_checkpoint_model.ckpt"
        config_path = checkpoint_dir / "config.yaml"
        verify_checkpoint_digest(variant, checkpoint_path)
        return (
            checkpoint_path,
            config_path,
            {
                "source": "local",
                "local_path": str(checkpoint_path),
                "checkpoint_sha256": _EXPECTED_CHECKPOINT_SHA256[variant],
            },
        )
    raise ValueError(
        f"unknown_variant {variant!r}; expected one of "
        f"{sorted(set(_VARIANT_SUBFOLDERS) | set(_LOCAL_CHECKPOINT_PATHS))}"
    )


def load_model(checkpoint_path: Path):
    """Reconstructs the STARCOP LightningModule saved at `checkpoint_path`,
    returning (model, settings). `settings` is the OmegaConf DictConfig
    ModelModule/ModelModuleRegression.__init__ received via
    save_hyperparameters() at training time.

    weights_only=False is required -- the checkpoint's hyper_parameters is a
    real OmegaConf DictConfig, not just tensors, and weights_only=True's
    restricted unpickler rejects it. Only call this on a path that already
    passed verify_checkpoint_digest (download_checkpoint always does)."""
    import torch

    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    settings = raw["hyper_parameters"]["settings"]
    model_cls = model_class_for_mode(settings.model.model_mode)
    model = model_cls(settings)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    return model, settings


def import_variant(variant: str, stage: str | None) -> None:
    """Downloads `variant`, logs it as an MLflow run (params, tags, raw
    checkpoint artifact, pyfunc-loadable model artifact), and -- unless
    `stage` is None -- registers/promotes it via mlflow_registry's already
    tested register_and_promote."""
    import sys

    _training_dir = str(Path(__file__).resolve().parents[1] / "training")
    if _training_dir not in sys.path:
        sys.path.insert(0, _training_dir)

    import mlflow
    import mlflow.pytorch
    import mlflow_registry
    import mlflow_utils
    from mlflow.tracking import MlflowClient

    log = logging.getLogger(__name__)
    mlflow_utils.require_mlflow_tracking_env()
    mlflow.set_experiment("starcop-baselines")

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path, config_path, provenance_tags = resolve_checkpoint(variant, Path(tmp))
        model, settings = load_model(checkpoint_path)

        model_name = registry_model_name(variant)
        with mlflow.start_run(run_name=f"{model_name}-import") as run:
            mlflow.log_params(mlflow_utils.flatten_hydra_params(settings))
            mlflow.set_tags(
                {
                    **provenance_tags,
                    "variant": variant,
                    "baseline": "true",
                    "sensor": _SENSOR,
                }
            )
            mlflow.log_artifact(str(config_path), artifact_path="config")
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoint")
            # Logged via the pytorch flavor too (MLmodel metadata), same as
            # src/training/train.py, so registered versions are loadable via
            # mlflow.pyfunc.load_model -- the raw checkpoint alone isn't.
            # serialization_format="pickle": mlflow's default ("pt2", torch.export
            # tracing) requires an input_example, which this checkpoint-import path
            # doesn't have.
            mlflow.pytorch.log_model(model, artifact_path="model", serialization_format="pickle")

        log.info("Logged %s as MLflow run %s", variant, run.info.run_id)

        if stage is not None:
            client = MlflowClient()
            version = mlflow_registry.register_and_promote(
                client, run_id=run.info.run_id, model_name=model_name, stage=stage
            )
            log.info("Registered %s version %s at stage %s", model_name, version.version, stage)
