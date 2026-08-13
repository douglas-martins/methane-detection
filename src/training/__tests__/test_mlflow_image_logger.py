"""Tests for src/training/mlflow_image_logger.py.

data_logger.ImageLogger (vendor/starcop, unmodified) has zero existing test
coverage. Confirmed against the real pytorch_lightning 1.6.4 API (empirically,
not assumed): Trainer(logger=[a, b]) sets trainer.logger to a LoggerCollection
whose .experiment is a plain list of the wrapped loggers' .experiment objects
-- calling .log(...) on that list raises AttributeError. That's the real
regression a naive WandbLogger -> [wandb_logger, mlflow_logger] swap would hit.
Loggers below are fakes (small recorder stand-ins), not Mock() interaction
checks -- tests assert on recorded state.
"""

import pytest
from _vendor_starcop_training import ImageLogger
from mlflow_image_logger import MultiLoggerImageLogger


class FakeWandbExperiment:
    def __init__(self):
        self.logged = []

    def log(self, data, commit=False):
        self.logged.append((data, commit))


class FakeWandbLogger:
    def __init__(self):
        self.experiment = FakeWandbExperiment()


class FakeMLFlowExperiment:
    def __init__(self):
        self.logged_figures = []

    def log_figure(self, run_id, figure, artifact_file):
        self.logged_figures.append((run_id, figure, artifact_file))


class FakeMLFlowLogger:
    def __init__(self, run_id="run-123"):
        self.run_id = run_id
        self.experiment = FakeMLFlowExperiment()


class FakeLoggerCollection:
    """Stand-in for pytorch_lightning's real LoggerCollection (verified
    empirically: .experiment is a plain list of the wrapped .experiment
    objects, not itself loggable)."""

    def __init__(self, loggers):
        self.experiment = [lg.experiment for lg in loggers]


class FakeTrainer:
    def __init__(self, loggers, current_epoch=0):
        self.loggers = loggers
        self.logger = loggers[0] if len(loggers) == 1 else FakeLoggerCollection(loggers)
        self.current_epoch = current_epoch


def _image_logger_with_canned_figures(cls, figures):
    il = cls(batch_train={}, batch_test={}, input_products=[], products_plot=[])
    il.on_split_epoch_end = lambda batch, model, name: figures
    return il


class TestUnmodifiedImageLoggerBreaksUnderMultiLogger:
    def test_raises_with_unpatched_image_logger_when_trainer_has_multiple_loggers(self):
        il = _image_logger_with_canned_figures(ImageLogger, {"train_batch": "fake-fig"})
        trainer = FakeTrainer(loggers=[FakeWandbLogger(), FakeMLFlowLogger()])

        with pytest.raises(AttributeError):
            il.on_train_epoch_end(trainer, model=None)


class TestMultiLoggerImageLogger:
    def test_still_logs_via_wandb_experiment_when_only_wandb_logger_present(self):
        ml = _image_logger_with_canned_figures(MultiLoggerImageLogger, {"train_batch": "fig"})
        wandb_logger = FakeWandbLogger()
        trainer = FakeTrainer(loggers=[wandb_logger])

        ml.on_train_epoch_end(trainer, model=None)

        assert wandb_logger.experiment.logged == [({"train_batch": "fig"}, False)]

    def test_logs_figure_via_mlflow_log_figure_when_mlflow_logger_present(self):
        ml = _image_logger_with_canned_figures(MultiLoggerImageLogger, {"val_batch": "fig"})
        mlflow_logger = FakeMLFlowLogger()
        trainer = FakeTrainer(loggers=[mlflow_logger], current_epoch=3)

        ml.on_validation_epoch_end(trainer, model=None)

        assert mlflow_logger.experiment.logged_figures == [
            ("run-123", "fig", "images/val_batch_epoch_3.png")
        ]

    def test_logs_to_both_loggers_when_both_present(self):
        ml = _image_logger_with_canned_figures(MultiLoggerImageLogger, {"train_batch": "fig"})
        wandb_logger = FakeWandbLogger()
        mlflow_logger = FakeMLFlowLogger()
        trainer = FakeTrainer(loggers=[wandb_logger, mlflow_logger], current_epoch=1)

        ml.on_train_epoch_end(trainer, model=None)

        assert wandb_logger.experiment.logged == [({"train_batch": "fig"}, False)]
        assert mlflow_logger.experiment.logged_figures == [
            ("run-123", "fig", "images/train_batch_epoch_1.png")
        ]

    def test_logs_each_figure_in_the_dict_separately_to_mlflow(self):
        ml = _image_logger_with_canned_figures(
            MultiLoggerImageLogger, {"train_batch": "fig-a", "extra_batch": "fig-b"}
        )
        mlflow_logger = FakeMLFlowLogger()
        trainer = FakeTrainer(loggers=[mlflow_logger], current_epoch=0)

        ml.on_train_epoch_end(trainer, model=None)

        artifact_files = {f for _, _, f in mlflow_logger.experiment.logged_figures}
        assert artifact_files == {
            "images/train_batch_epoch_0.png",
            "images/extra_batch_epoch_0.png",
        }
