"""Pure logic for turning STARCOP's run_validation() metrics dict into
MLflow-loggable scalars (TASK-2.3).

run_validation() (vendor/starcop/starcop/validation.py, unmodified) returns a
metrics dict mixing Python/numpy scalars (aggregate and per-difficulty
metrics) with non-scalar entries -- confusion_matrix / classification_
confusion_matrix tensors, thresholded (a list of per-threshold dicts) --
that mlflow.log_metrics rejects outright. Isolated here as tested pure logic
rather than inlined in train.py's SDK glue.
"""

from numbers import Number
from typing import Dict


def extract_scalar_metrics(metrics: Dict[str, object], prefix: str) -> Dict[str, float]:
    """Returns the scalar (non-tensor, non-container) entries of `metrics`,
    keyed as `f"{prefix}_{name}"` with plain Python float values.
    """
    return {
        f"{prefix}_{name}": float(value)
        for name, value in metrics.items()
        if isinstance(value, Number)
    }
