"""Builds the Hydra CLI arg list and required credential env vars for a
per-machine training launch script -- see mlops-methane-detection-plan.md
TASK-3.3a. Pure functions only, no subprocess/env access, so
scripts/train_mac.sh (and later train_desktop.sh/train_colab.ipynb) can call
this and stay thin, untested-by-unit-test glue exercised by a real run
instead of duplicating this decision logic in shell.
"""

from typing import Dict, List, Literal, Optional

Machine = Literal["desktop", "macbook", "colab"]

# gpu/1 is STARCOP's own unmodified default (vendor/starcop/scripts/configs/
# config.yaml) -- listed explicitly here (rather than left as an implicit
# fallback) so every launch script is self-documenting about what accelerator
# it actually requested, matching TASK-3.2's own "assert what was requested,
# don't trust what completed" precedent.
_MACHINE_DEFAULTS: Dict[str, Dict[str, str]] = {
    "desktop": {"training.accelerator": "gpu", "training.devices": "1"},
    "macbook": {"training.accelerator": "mps", "training.devices": "1"},
    "colab": {"training.accelerator": "gpu", "training.devices": "1"},
}

# Same three-set split documented in training-runbook.md (MLflow tracking +
# B2 artifact upload); identical across machines today, kept per-machine
# since TASK-3.3c's Colab profile will need additional DVC OAuth vars on top.
_REQUIRED_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


def build_launch_args(
    machine: Machine,
    dataset_name: str,
    overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Returns the Hydra CLI arg list for launching src/training/train.py.

    Raises ValueError for an unrecognized machine. `overrides` replaces any
    of the machine's own accelerator/devices defaults by key instead of
    duplicating the flag -- Hydra errors on a key passed twice.
    """
    if machine not in _MACHINE_DEFAULTS:
        raise ValueError(
            f"machine={machine!r} is not one of {tuple(_MACHINE_DEFAULTS)}."
        )

    merged = {**_MACHINE_DEFAULTS[machine], **(overrides or {})}
    return [f"+machine={machine}", f"+dataset_name={dataset_name}"] + [
        f"{key}={value}" for key, value in merged.items()
    ]


def required_env_vars(machine: Machine) -> List[str]:
    """Returns the credential env var names needed to launch on `machine`,
    for a pre-flight check before the real training subprocess starts.
    """
    return list(_REQUIRED_ENV_VARS)
