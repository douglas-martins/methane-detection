"""Tests for src/training/mlflow_log_model_compat.py -- pure kwarg-selection
logic (Test Size: Small, no mocking, a plain fake function standing in for
mlflow.pytorch.log_model): mlflow.pytorch.log_model()'s serialization_format
kwarg doesn't exist on every mlflow version this project runs against -- see
TASK-3.3c in mlops-methane-detection-plan.md, found running a real training
job on Colab (Environment A, mlflow<3.7): `TypeError: save() got an
unexpected keyword argument 'serialization_format'`.

serialization_format="pickle" was added for Environment B's newer mlflow,
where the default flips to "pt2" (torch.export tracing, needs an
input_example this project's log_model call doesn't provide). Environment A
pins mlflow<3.7 for the opposite reason (unpinned mlflow's save_model() does
an unconditional torch.export import that torch==1.13.1 can't satisfy), and
that older mlflow line predates serialization_format existing as a
parameter at all -- passing it anyway falls through **kwargs straight into
torch.save(), which rejects it.

Detection is via real introspection of the installed
mlflow.pytorch.log_model signature (not a hardcoded mlflow version
threshold) -- this repo's established pattern (see optimizer_compat.py,
lightning2_compat.py) of checking the real installed thing rather than
guessing a cutoff.
"""

import mlflow_log_model_compat


class TestBuildLogModelKwargs:
    def test_includes_serialization_format_when_log_model_fn_accepts_it(self):
        def _new_style_log_model(pytorch_model, artifact_path, serialization_format="pt2"):
            pass

        kwargs = mlflow_log_model_compat.build_log_model_kwargs("model", _new_style_log_model)

        assert kwargs == {"artifact_path": "model", "serialization_format": "pickle"}

    def test_omits_serialization_format_when_log_model_fn_predates_it(self):
        def _old_style_log_model(pytorch_model, artifact_path, **kwargs):
            pass

        kwargs = mlflow_log_model_compat.build_log_model_kwargs("model", _old_style_log_model)

        assert kwargs == {"artifact_path": "model"}

    def test_returned_kwargs_do_not_break_the_old_style_function_via_kwargs(self):
        """Reproduces the actual bug: calling with a leftover
        serialization_format kwarg falls through **kwargs into whatever the
        old-style function forwards it to. Confirms the built kwargs are
        clean enough that the old-style function's own body (standing in for
        torch.save, which raised the real TypeError) never sees it."""
        received = {}

        def _old_style_log_model(pytorch_model, artifact_path, **kwargs):
            received.update(kwargs)

        kwargs = mlflow_log_model_compat.build_log_model_kwargs("model", _old_style_log_model)
        _old_style_log_model("fake-model", **kwargs)

        assert "serialization_format" not in received
