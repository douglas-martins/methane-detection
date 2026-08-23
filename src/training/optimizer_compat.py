"""Composition-only fix for torch removing the `verbose` kwarg from
torch.optim.lr_scheduler.ReduceLROnPlateau -- see TASK-3.1 in
mlops-methane-detection-plan.md, found running a real training job on
Environment B (torch 2.12.1) on the RTX 5070: `TypeError:
ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`.
STARCOP's own ModelModule.configure_optimizers (vendor/starcop/starcop/
models/model_module.py, imported unmodified) still passes `verbose=True`.

Detection is via real introspection of the installed
torch.optim.lr_scheduler.ReduceLROnPlateau signature, not a hardcoded torch
version threshold -- a no-op wherever `verbose` is still accepted (Environment
A's pinned torch==1.13.1, where the original method already works).
"""

import inspect
import types

import torch


def _configure_optimizers(self):
    if self.settings_model.optimizer == "adam":
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
    else:
        raise Exception(f"No optimizer implemented for : {self.settings_model.optimizer}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=self.lr_decay, patience=self.lr_patience
    )

    return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


def bind_verbose_free_configure_optimizers(model) -> None:
    """Rebinds ModelModule.configure_optimizers to build ReduceLROnPlateau
    without `verbose` when the installed torch no longer accepts it. A no-op
    when it does -- the original vendor method already works there.
    """
    params = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__).parameters
    if "verbose" in params:
        return

    model.configure_optimizers = types.MethodType(_configure_optimizers, model)
