"""MultiLoggerImageLogger -- a subclass of starcop.data.data_logger.ImageLogger
(imported unmodified) that supports PyTorch Lightning's multi-logger mode
(Trainer(logger=[wandb_logger, mlflow_logger])).

The original ImageLogger calls trainer.logger.experiment.log(fig_dict,
commit=False), which is W&B-specific. Verified empirically against the real
pytorch_lightning 1.6.4 API: once a Trainer has multiple loggers,
trainer.logger.experiment is a plain list of the wrapped loggers' .experiment
objects, and calling .log() on it raises AttributeError. This subclass
overrides only the two hook methods that do that dispatch -- on_split_epoch_end
(the figure-building logic) is inherited unchanged from the parent via
super(), so it's never duplicated (see mlops-methane-detection-plan.md
TASK-2.2 decision 0/3).
"""

from _vendor_starcop_training import ImageLogger


class MultiLoggerImageLogger(ImageLogger):
    def on_train_epoch_end(self, trainer, model, unused=None) -> None:
        figures = self.on_split_epoch_end(self.batch_train, model, "train")
        self._log_to_all(trainer, figures)

    def on_validation_epoch_end(self, trainer, model) -> None:
        figures = self.on_split_epoch_end(self.batch_test, model, "val")
        self._log_to_all(trainer, figures)

    def _log_to_all(self, trainer, figures) -> None:
        loggers = getattr(trainer, "loggers", None) or [trainer.logger]
        for logger in loggers:
            if logger is None:
                continue
            if hasattr(logger, "run_id"):
                # MLFlowLogger-style: no batched dict log, one artifact per figure.
                for name, fig in figures.items():
                    logger.experiment.log_figure(
                        run_id=logger.run_id,
                        figure=fig,
                        artifact_file=f"images/{name}_epoch_{trainer.current_epoch}.png",
                    )
            else:
                logger.experiment.log(figures, commit=False)
