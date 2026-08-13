"""Composes configs/training/overlay.yaml (machine, dataset.augmentations.*)
on top of Hydra settings derived from vendor/starcop/scripts/configs/config.yaml
-- see mlops-methane-detection-plan.md TASK-2.2 decision 5. That file is
never edited; new fields are layered on from outside instead.
"""

from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf

VALID_MACHINES = ("desktop", "macbook", "colab")
VALID_DATASET_NAMES = ("starcop_mini", "starcop_raw")


def merge_overlay(hydra_settings: DictConfig, overlay_path: Union[str, Path]) -> DictConfig:
    """Merges overlay_path's defaults under hydra_settings.

    Overlay first, hydra_settings second: Hydra's own CLI-applied overrides
    (e.g. `+machine=desktop`) already live in hydra_settings by the time this
    runs, so they win over the overlay's placeholder/default values, while
    overlay-only keys (e.g. dataset.augmentations.* when not overridden)
    survive the merge.
    """
    overlay = OmegaConf.load(overlay_path)
    return OmegaConf.merge(overlay, hydra_settings)


def validate_machine(settings: DictConfig) -> None:
    """Raises ValueError unless settings.machine is one of VALID_MACHINES."""
    if OmegaConf.is_missing(settings, "machine"):
        raise ValueError(
            "settings.machine is required and was not set -- pass "
            "+machine=desktop|macbook|colab on the CLI."
        )

    machine = settings.machine
    if machine not in VALID_MACHINES:
        raise ValueError(f"settings.machine={machine!r} is not one of {VALID_MACHINES}.")


def validate_dataset_name(settings: DictConfig) -> None:
    """Raises ValueError unless settings.dataset_name is one of VALID_DATASET_NAMES."""
    if OmegaConf.is_missing(settings, "dataset_name"):
        raise ValueError(
            "settings.dataset_name is required and was not set -- pass "
            "+dataset_name=starcop_mini|starcop_raw on the CLI."
        )

    dataset_name = settings.dataset_name
    if dataset_name not in VALID_DATASET_NAMES:
        raise ValueError(
            f"settings.dataset_name={dataset_name!r} is not one of {VALID_DATASET_NAMES}."
        )
