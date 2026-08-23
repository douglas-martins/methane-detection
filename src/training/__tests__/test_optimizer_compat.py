"""Tests for src/training/optimizer_compat.py -- pure hook-rebinding logic
(Test Size: Small, no mocking, a plain fake object standing in for
ModelModule): torch removed the `verbose` kwarg from
torch.optim.lr_scheduler.ReduceLROnPlateau in a later release than the
Environment A pin (torch==1.13.1) targets. STARCOP's own ModelModule
(vendor/starcop/starcop/models/model_module.py, imported unmodified) still
passes `verbose=True` in configure_optimizers() -- see TASK-3.1 in
mlops-methane-detection-plan.md, found running a real training job on
Environment B (torch 2.12.1) on the RTX 5070: `TypeError:
ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`.

Detection is via real introspection of the installed
torch.optim.lr_scheduler.ReduceLROnPlateau signature (not a hardcoded torch
version threshold) -- this repo's established pattern (e.g. TASK-3.2's
mlflow<3.7 pin was verified by wheel inspection, not assumed from a changelog
date) of checking the real installed thing rather than guessing a cutoff.
"""

import optimizer_compat
import pytest
import torch


class _FakeModelModule:
    """Mimics just the surface ModelModule exposes for this fix: the
    optimizer/lr fields configure_optimizers reads, and network.parameters().
    """

    def __init__(self):
        self.settings_model = type("Settings", (), {"optimizer": "adam"})()
        self.network = torch.nn.Linear(2, 2)
        self.lr = 0.01
        self.lr_decay = 0.5
        self.lr_patience = 3

    def configure_optimizers(self):
        raise AssertionError("original configure_optimizers should have been rebound")


class TestBindVerboseFreeConfigureOptimizers:
    def test_rebinds_when_installed_reduce_lr_on_plateau_rejects_verbose(self, monkeypatch):
        class _NoVerboseReduceLROnPlateau:
            def __init__(self, optimizer, mode, factor, patience):
                self.optimizer = optimizer

        monkeypatch.setattr(
            torch.optim.lr_scheduler, "ReduceLROnPlateau", _NoVerboseReduceLROnPlateau
        )
        model = _FakeModelModule()

        optimizer_compat.bind_verbose_free_configure_optimizers(model)
        result = model.configure_optimizers()

        assert result["monitor"] == "val_loss"
        assert isinstance(result["lr_scheduler"], _NoVerboseReduceLROnPlateau)

    def test_is_a_noop_when_installed_reduce_lr_on_plateau_still_accepts_verbose(self, monkeypatch):
        class _VerboseReduceLROnPlateau:
            def __init__(self, optimizer, mode, factor, patience, verbose):
                pass

        monkeypatch.setattr(
            torch.optim.lr_scheduler, "ReduceLROnPlateau", _VerboseReduceLROnPlateau
        )
        model = _FakeModelModule()

        optimizer_compat.bind_verbose_free_configure_optimizers(model)

        with pytest.raises(AssertionError, match="should have been rebound"):
            model.configure_optimizers()

    def test_rebound_optimizer_is_adam_over_the_network_parameters(self, monkeypatch):
        class _NoVerboseReduceLROnPlateau:
            def __init__(self, optimizer, mode, factor, patience):
                pass

        monkeypatch.setattr(
            torch.optim.lr_scheduler, "ReduceLROnPlateau", _NoVerboseReduceLROnPlateau
        )
        model = _FakeModelModule()

        optimizer_compat.bind_verbose_free_configure_optimizers(model)
        result = model.configure_optimizers()

        assert isinstance(result["optimizer"], torch.optim.Adam)

    def test_raises_for_an_unimplemented_optimizer_setting(self, monkeypatch):
        class _NoVerboseReduceLROnPlateau:
            def __init__(self, optimizer, mode, factor, patience):
                pass

        monkeypatch.setattr(
            torch.optim.lr_scheduler, "ReduceLROnPlateau", _NoVerboseReduceLROnPlateau
        )
        model = _FakeModelModule()
        model.settings_model.optimizer = "sgd"

        optimizer_compat.bind_verbose_free_configure_optimizers(model)

        with pytest.raises(Exception, match="No optimizer implemented"):
            model.configure_optimizers()
