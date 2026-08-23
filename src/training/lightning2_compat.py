"""Composition-only fix for pytorch-lightning>=2.0's removal of the
validation_epoch_end/test_epoch_end LightningModule hooks -- see TASK-3.1 in
mlops-methane-detection-plan.md, found running a real training job on
Environment B (lightning 2.6.5) on the RTX 5070. STARCOP's own ModelModule
(vendor/starcop/starcop/models/model_module.py, imported unmodified) still
implements those pre-2.0 hook names; Lightning 2.x's own configuration
validator raises NotImplementedError merely because the method is present
(`callable(getattr(model, "validation_epoch_end", None))`), regardless of
whether it is ever called -- so this can't be left alone the way an unused
method normally could be.

No-op under Lightning <2.0 (Environment A's pinned pytorch-lightning==1.6.4,
or the TASK-3.2 1.9.5 override): those versions still expect and call the
old hook names directly. Patching unconditionally would double-invoke
val_epoch_end there -- once via the still-present old hook, once via the
newly-bound new-style one.
"""

import types

import pytorch_lightning


def _on_validation_epoch_end(self) -> None:
    self.val_epoch_end(outputs=None, prefix="val")


def _on_test_epoch_end(self) -> None:
    self.val_epoch_end(outputs=None, prefix="test")


def bind_new_style_epoch_end_hooks(model) -> None:
    """Shadows validation_epoch_end/test_epoch_end on this instance (so
    Lightning 2.x's configuration validator doesn't reject the run) and
    binds on_validation_epoch_end/on_test_epoch_end to call val_epoch_end
    instead -- dynamically via self.val_epoch_end, so this works regardless
    of whether train.py's own val_epoch_end rebind (decision 7, background
    F1) happens before or after this call.
    """
    major_version = int(pytorch_lightning.__version__.split(".")[0])
    if major_version < 2:
        return

    model.validation_epoch_end = None
    model.test_epoch_end = None
    model.on_validation_epoch_end = types.MethodType(_on_validation_epoch_end, model)
    model.on_test_epoch_end = types.MethodType(_on_test_epoch_end, model)
