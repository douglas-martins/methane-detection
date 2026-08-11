"""Pure helpers for MLflow run tagging/param-logging.

Kept free of any `mlflow` SDK calls so they're testable without a live
tracking server -- src/training/train.py does the thin SDK glue
(mlflow.set_tags(build_run_tags(...)), mlflow.log_params(flatten_hydra_params(...)))
around these.
"""

from typing import Dict

from omegaconf import DictConfig, OmegaConf


def build_run_tags(
    dataset_version: str, dataset_dirty: bool, machine: str, sensor: str
) -> Dict[str, str]:
    return {
        "dataset_version": dataset_version,
        "dataset_dirty": str(dataset_dirty),
        "machine": machine,
        "sensor": sensor,
    }


def flatten_hydra_params(settings: DictConfig) -> Dict[str, str]:
    """Flattens a (possibly nested) Hydra config into MLflow-safe params.

    MLflow's log_params rejects nested/non-string values, so nested keys
    become dot-separated (dataset.input_products) and non-scalar values
    (e.g. a list like input_products) are stringified.
    """
    container = OmegaConf.to_container(settings, resolve=True)

    flat: Dict[str, str] = {}

    def _walk(prefix: str, obj) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(child_prefix, value)
        else:
            flat[prefix] = str(obj)

    _walk("", container)
    return flat
