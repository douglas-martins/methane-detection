"""Tests for src/training/lightning2_compat.py -- pure hook-rebinding logic
(Test Size: Small, no mocking, a plain fake object standing in for
ModelModule): pytorch-lightning>=2.0 removed support for the
validation_epoch_end/test_epoch_end LightningModule hooks -- its own
configuration validator raises NotImplementedError merely because the
method is present (callable(getattr(model, "validation_epoch_end", None))),
regardless of whether it's ever called. STARCOP's own ModelModule (vendor/
starcop/starcop/models/model_module.py) still implements those pre-2.0
names, unmodified (composition-only) -- see TASK-3.1 in
mlops-methane-detection-plan.md, found running a real training job on
Environment B (lightning 2.6.5) on the RTX 5070.
"""

import lightning2_compat
import pytest
import pytorch_lightning


class _FakeModelModule:
    """Mimics just the surface ModelModule exposes for this fix: the old-style
    hooks Lightning 2.x rejects, and the val_epoch_end method they delegate
    to (unused `outputs` param, matching the real class exactly).
    """

    def __init__(self):
        self.calls = []

    def validation_epoch_end(self, outputs) -> None:
        self.val_epoch_end(outputs, prefix="val")

    def test_epoch_end(self, outputs) -> None:
        self.val_epoch_end(outputs, prefix="test")

    def val_epoch_end(self, outputs, prefix):
        self.calls.append(prefix)


class TestBindNewStyleEpochEndHooks:
    def test_shadows_the_old_hook_names_so_lightning_2x_does_not_reject_them(self):
        model = _FakeModelModule()

        lightning2_compat.bind_new_style_epoch_end_hooks(model)

        assert not callable(getattr(model, "validation_epoch_end", None))
        assert not callable(getattr(model, "test_epoch_end", None))

    def test_on_validation_epoch_end_calls_val_epoch_end_with_val_prefix(self):
        model = _FakeModelModule()
        lightning2_compat.bind_new_style_epoch_end_hooks(model)

        model.on_validation_epoch_end()

        assert model.calls == ["val"]

    def test_on_test_epoch_end_calls_val_epoch_end_with_test_prefix(self):
        model = _FakeModelModule()
        lightning2_compat.bind_new_style_epoch_end_hooks(model)

        model.on_test_epoch_end()

        assert model.calls == ["test"]

    def test_new_hooks_pick_up_a_later_rebind_of_val_epoch_end(self):
        """train.py rebinds model.val_epoch_end (background-F1 patch, decision
        7) before calling this -- but order must not matter: the new-style
        hooks call self.val_epoch_end dynamically, not a captured reference,
        so binding this first and rebinding val_epoch_end after still works.
        """
        model = _FakeModelModule()
        lightning2_compat.bind_new_style_epoch_end_hooks(model)

        replaced_calls = []
        model.val_epoch_end = lambda outputs, prefix: replaced_calls.append(prefix)

        model.on_validation_epoch_end()

        assert replaced_calls == ["val"]
        assert model.calls == []

    @pytest.mark.skipif(
        int(pytorch_lightning.__version__.split(".")[0]) < 2,
        reason="only meaningful to assert the no-op path under lightning<2.0's own real version",
    )
    def test_is_not_a_noop_under_the_installed_lightning_2x(self):
        model = _FakeModelModule()

        lightning2_compat.bind_new_style_epoch_end_hooks(model)

        assert getattr(model, "validation_epoch_end", "unset") is None


class TestVersionGuard:
    def test_is_a_noop_under_lightning_below_2_0(self, monkeypatch):
        monkeypatch.setattr(pytorch_lightning, "__version__", "1.6.4")
        model = _FakeModelModule()

        lightning2_compat.bind_new_style_epoch_end_hooks(model)

        assert callable(model.validation_epoch_end)
        assert callable(model.test_epoch_end)
        assert not hasattr(model, "on_validation_epoch_end")
        assert not hasattr(model, "on_test_epoch_end")
