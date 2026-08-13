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
own live path is validated at, see docs/environment_notes.md).
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

_MODEL_MODE_CLASS_NAMES = {
    "segmentation_output": "ModelModule",
    "regression_output": "ModelModuleRegression",
}


def variant_subfolder(variant: str) -> str:
    """Returns the `models/<subfolder>/` path segment `variant` lives under
    in the isp-uv-es/starcop HF repo. Raises ValueError for an unknown
    variant."""
    try:
        return _VARIANT_SUBFOLDERS[variant]
    except KeyError:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {sorted(_VARIANT_SUBFOLDERS)}"
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


def download_checkpoint(variant: str, dest_dir: Path) -> tuple[Path, Path, str]:
    """Downloads config.yaml + checkpoint for `variant` from _HF_REPO into
    dest_dir. Returns (checkpoint_path, config_path, resolved commit sha of
    _HF_REPO's main branch)."""
    from huggingface_hub import HfApi, hf_hub_download

    subfolder = variant_subfolder(variant)
    checkpoint_path = Path(
        hf_hub_download(
            _HF_REPO, f"models/{subfolder}/final_checkpoint_model.ckpt", local_dir=str(dest_dir)
        )
    )
    config_path = Path(
        hf_hub_download(_HF_REPO, f"models/{subfolder}/config.yaml", local_dir=str(dest_dir))
    )
    revision = HfApi().model_info(_HF_REPO).sha
    return checkpoint_path, config_path, revision


def load_model(checkpoint_path: Path):
    """Reconstructs the STARCOP LightningModule saved at `checkpoint_path`,
    returning (model, settings). `settings` is the OmegaConf DictConfig
    ModelModule/ModelModuleRegression.__init__ received via
    save_hyperparameters() at training time."""
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
        checkpoint_path, config_path, revision = download_checkpoint(variant, Path(tmp))
        model, settings = load_model(checkpoint_path)

        model_name = registry_model_name(variant)
        with mlflow.start_run(run_name=f"{model_name}-import") as run:
            mlflow.log_params(mlflow_utils.flatten_hydra_params(settings))
            mlflow.set_tags(
                {
                    "source": "huggingface",
                    "hf_repo": _HF_REPO,
                    "hf_revision": revision,
                    "hf_subfolder": variant_subfolder(variant),
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
            mlflow.pytorch.log_model(model, artifact_path="model")

        log.info("Logged %s as MLflow run %s", variant, run.info.run_id)

        if stage is not None:
            client = MlflowClient()
            version = mlflow_registry.register_and_promote(
                client, run_id=run.info.run_id, model_name=model_name, stage=stage
            )
            log.info("Registered %s version %s at stage %s", model_name, version.version, stage)
