"""Composition-only fix for mlflow.pytorch.log_model()'s serialization_format
kwarg not existing on every mlflow version this project runs against -- see
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
mlflow.pytorch.log_model signature, not a hardcoded mlflow version
threshold -- this repo's established pattern (see optimizer_compat.py,
lightning2_compat.py) of checking the real installed thing rather than
guessing a cutoff.
"""

import inspect
from typing import Callable


def build_log_model_kwargs(artifact_path: str, log_model_fn: Callable) -> dict:
    """Returns the kwargs to call log_model_fn(pytorch_model, **kwargs) with:
    serialization_format="pickle" only when log_model_fn's own signature
    actually accepts it. A no-op omission on mlflow versions that predate the
    kwarg -- their default behavior already matches plain pickle-based
    torch.save().
    """
    kwargs = {"artifact_path": artifact_path}
    if "serialization_format" in inspect.signature(log_model_fn).parameters:
        kwargs["serialization_format"] = "pickle"
    return kwargs
