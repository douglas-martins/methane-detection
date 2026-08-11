"""MLflow-tracked STARCOP training entrypoint (TASK-2.2).

Replaces vendor/starcop/scripts/train.py as this project's actual training
command WITHOUT modifying that file or anything else under vendor/starcop/
(see mlops-methane-detection-plan.md TASK-2.2 decision 0). Every STARCOP
building block (Permian2019DataModule, ModelModule via get_model, ImageLogger,
run_validation) is imported unmodified via _vendor_starcop_training.py and
composed with new behavior from outside:
  - Hydra config: vendor/starcop/scripts/configs/config.yaml (unmodified) +
    configs/training/overlay.yaml, merged in settings_overlay.py.
  - Data loading: ProcessedDatasetDataModule (starcop_datamodule.py) subclasses
    Permian2019DataModule, overriding only prepare_data() to source this
    project's own data/processed/<dataset_name>/{patches,splits}/ layout
    instead of STARCOP's Permian2019 file-discovery convention (step 4b).
  - Augmentations: overridden via a plain attribute reassignment on
    data_module.train_dataset after prepare_data() (decision 6).
  - Background F1: model.val_epoch_end rebound via types.MethodType on the
    one model instance (decision 7).
  - Multi-logger image logging: MultiLoggerImageLogger subclass (decision 3).

Run with (Environment A):
    vendor/starcop/.venv/bin/python src/training/train.py \\
        +machine=desktop +dataset_name=starcop_mini

Requires MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
(see docs/environment_notes.md) -- validated up front, before any MLflow call,
so a missing var raises instead of silently training against a local store.
"""

import logging
import os
import sys
import types
from pathlib import Path

import hydra
import matplotlib
import mlflow
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from torch.utils.data import DataLoader

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kornia.augmentation as K  # noqa: E402
import metrics_ext  # noqa: E402
import mlflow_utils  # noqa: E402
import plot_confusion_matrix as pcm  # noqa: E402
import settings_overlay  # noqa: E402
from _vendor_starcop_training import ImageLogger, get_model, run_validation  # noqa: E402
from dvc_dataset_version import get_dataset_version, is_dataset_dirty  # noqa: E402
from mlflow_image_logger import MultiLoggerImageLogger  # noqa: E402
from starcop_datamodule import ProcessedDatasetDataModule  # noqa: E402

_VENDOR_CONFIG_DIR = str(
    Path(__file__).resolve().parents[2] / "vendor" / "starcop" / "scripts" / "configs"
)
_OVERLAY_PATH = Path(__file__).resolve().parents[2] / "configs" / "training" / "overlay.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DVC_LOCK_PATH = _REPO_ROOT / "dvc.lock"
_SENSOR = "AVIRIS-NG"  # both starcop_mini/starcop_raw are 100% AVIRIS-NG, see TASK-1.3


def _patched_val_epoch_end(self, outputs, prefix):
    """Replaces ModelModule.val_epoch_end with one that also logs background
    F1 via metrics_ext.compute_all -- see decision 7. Mirrors the original
    method's structure exactly, just swapping the metric source.

    Also stashes the raw confusion matrix as self._last_confusion_matrix
    before resetting it -- the original method always reset the
    torchmetrics accumulator at the end of every validation epoch (including
    the last one), so computing it again after trainer.fit() returns (for
    the confusion-matrix PNG artifact) would silently read an empty,
    already-reset matrix otherwise.
    """
    cm = self.confusion_matrix.compute()
    self._last_confusion_matrix = cm
    for name, value in metrics_ext.compute_all(cm).items():
        self.log(f"{prefix}_{name}", value)
    self.confusion_matrix.reset()

    if self.settings_model.model_mode == "segmentation_output":
        cm = self.classification_confusion_matrix.compute()
        for name, value in metrics_ext.compute_all(cm).items():
            self.log(f"{prefix}_classification_{name}", value)
        self.classification_confusion_matrix.reset()

    return {}


def _build_augmentations(settings: DictConfig, data_keys) -> K.AugmentationSequential:
    """Rebuilds the augmentation pipeline from settings.dataset.augmentations.*.

    data_keys must come from the augmentation object STARCOP's own
    datamodule.py already built (data_module.train_dataset.spatial_augmentations
    before this replaces it) -- it depends on settings.model.model_mode and
    weight-loss config via branching logic in datamodule.py that this file
    must not duplicate (or silently drift from).
    """
    aug = settings.dataset.augmentations
    return K.AugmentationSequential(
        K.RandomRotation(p=aug.rotation_p, degrees=aug.rotation_degrees),
        K.RandomHorizontalFlip(p=aug.hflip_p),
        K.RandomVerticalFlip(p=aug.vflip_p),
        keepdim=True,
        data_keys=data_keys,
    )


@hydra.main(version_base=None, config_path=_VENDOR_CONFIG_DIR, config_name="config")
def train(hydra_settings: DictConfig) -> None:
    """Hydra entrypoint: runs one MLflow-tracked STARCOP training + validation
    pass -- see module docstring for the composition decisions involved."""
    log = logging.getLogger(__name__)

    settings = settings_overlay.merge_overlay(hydra_settings, _OVERLAY_PATH)
    settings_overlay.validate_machine(settings)
    settings_overlay.validate_dataset_name(settings)

    experiment_path = os.getcwd()
    checkpoint_path = os.path.join(experiment_path, "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)

    OmegaConf.set_struct(settings, False)
    settings["experiment_path"] = experiment_path
    OmegaConf.set_struct(settings, True)

    plt.ioff()
    seed_everything(None if settings.seed == "None" else settings.seed)

    # MLflow run + dual logger setup
    dataset_version = get_dataset_version(settings.dataset_name, _DVC_LOCK_PATH)
    dataset_dirty = is_dataset_dirty(settings.dataset_name, _REPO_ROOT)

    mlflow_utils.require_mlflow_tracking_env()

    # `with` (not manual start_run/end_run) so a crash anywhere below marks
    # the run FAILED on the server instead of leaving it stuck RUNNING forever.
    with mlflow.start_run(run_name=settings.experiment_name) as run:
        mlflow.log_params(mlflow_utils.flatten_hydra_params(settings))
        mlflow.set_tags(
            mlflow_utils.build_run_tags(
                dataset_version=dataset_version,
                dataset_dirty=dataset_dirty,
                machine=settings.machine,
                sensor=_SENSOR,
            )
        )

        wandb_logger = WandbLogger(
            name=settings.experiment_name,
            project=settings.wandb.wandb_project,
            entity=settings.wandb.wandb_entity,
        )
        wandb_logger.experiment.config.update(OmegaConf.to_container(settings, resolve=True))

        mlflow_logger = MLFlowLogger(
            experiment_name=settings.experiment_name,
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            run_id=run.info.run_id,
        )

        # Dataset
        log.info("SETTING UP DATASET")
        data_module = ProcessedDatasetDataModule(
            settings, dataset_name=settings.dataset_name, repo_root=_REPO_ROOT
        )
        data_module.prepare_data()
        existing_data_keys = data_module.train_dataset.spatial_augmentations.data_keys
        data_module.train_dataset.spatial_augmentations = _build_augmentations(
            settings, existing_data_keys
        )

        # Model
        log.info("SETTING UP MODEL")
        settings.model.test = False
        settings.model.train = True
        model = get_model(settings, settings.experiment_name)
        model.val_epoch_end = types.MethodType(_patched_val_epoch_end, model)

        # Checkpointing
        metric_monitor = "val_loss"
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_path,
            save_top_k=True,
            save_last=True,
            verbose=True,
            monitor=metric_monitor,
            mode="min",
        )
        early_stop_callback = EarlyStopping(
            monitor=metric_monitor,
            patience=settings.model.early_stopping_patience,
            strict=False,
            verbose=False,
            mode="min",
        )

        batch_train = next(iter(data_module.train_plot_dataloader(batch_size=settings.plot_samples)))
        batch_test = next(iter(data_module.test_plot_dataloader(batch_size=settings.plot_samples)))
        image_logger = MultiLoggerImageLogger(
            batch_train=batch_train,
            batch_test=batch_test,
            products_plot=settings.products_plot,
            input_products=settings.dataset.input_products,
        )

        log.info("START TRAINING")
        trainer = Trainer(
            logger=[wandb_logger, mlflow_logger],
            callbacks=[checkpoint_callback, early_stop_callback, image_logger],
            default_root_dir=experiment_path,
            accumulate_grad_batches=1,
            gradient_clip_val=0.0,
            benchmark=False,
            accelerator=settings.training.accelerator,
            devices=settings.training.devices,
            max_epochs=settings.training.max_epochs,
            val_check_interval=settings.training.val_check_interval,
            log_every_n_steps=settings.training.train_log_every_n_steps,
        )
        # pytorch_lightning 1.6.4's ckpt_path only accepts a checkpoint *file*
        # (or the literal "best") -- there's no "last" token in this version,
        # unlike newer Lightning releases -- so resuming must point at the
        # last.ckpt file save_last=True (above) writes into checkpoint_path.
        trainer.fit(
            model,
            data_module,
            ckpt_path=(
                os.path.join(checkpoint_path, "last.ckpt")
                if settings.resume_from_checkpoint
                else None
            ),
        )

        final_checkpoint_path = os.path.join(experiment_path, "final_checkpoint_model.ckpt")
        trainer.save_checkpoint(final_checkpoint_path)
        mlflow.log_artifact(final_checkpoint_path, artifact_path="checkpoint")

        mlflow.log_figure(pcm.plot_confusion_matrix(model._last_confusion_matrix), "confusion_matrix.png")

        # run_validation is STARCOP's own unmodified diagnostic/reporting pass
        # (difficulty-stratified metrics + per-sample plots) -- it unconditionally
        # assumes the test set has both "easy" and "hard" no-plume examples,
        # which small/skewed splits like starcop_mini's 9-scene test set may
        # not. That's a real fragility in unmodified STARCOP code we can't
        # patch (decision 0), so a failure here is logged and swallowed
        # rather than failing an otherwise-successful training+tracking run --
        # the MLflow run this task validates has already fully succeeded by
        # this point (metrics, tags, checkpoint, images all logged above).
        log.info("Running validation of val data")
        dataloader_val = data_module.test_plot_dataloader(batch_size=1, num_workers=data_module.num_workers)
        try:
            run_validation(
                model,
                dataloader_val,
                products_plot=settings.products_plot,
                show_plots=False,
                verbose=False,
                path_save_results=os.path.join(experiment_path, "validation"),
            )
        except Exception:
            log.warning("run_validation (val data) failed -- skipping, see TASK-2.2 note", exc_info=True)

        log.info("Running validation of train data")
        dataloader_train = DataLoader(
            data_module.train_dataset_non_tiled,
            batch_size=1,
            num_workers=data_module.num_workers,
            shuffle=False,
        )
        try:
            run_validation(
                model,
                dataloader_train,
                products_plot=settings.products_plot,
                show_plots=False,
                verbose=False,
                path_save_results=os.path.join(experiment_path, "train"),
            )
        except Exception:
            log.warning("run_validation (train data) failed -- skipping, see TASK-2.2 note", exc_info=True)

    log.info(f"Finished: results saved to {experiment_path}")


if __name__ == "__main__":
    train()
