# Model Registry Promotion Policy

Governs how a training run tracked in MLflow (TASK-2.2) becomes a versioned,
staged model in the MLflow Model Registry — implemented by
`src/registry/promote_model.py` (TASK-2.3).

## Registered model name

All versions of this project's model are registered under the fixed name
`methane-cnn-starcop`. MLflow registered-model names are effectively
permanent identifiers on the server — do not rename once real versions exist.

## Metric name mapping

The promotion criteria (below, and in `mlops-methane-detection-plan.md`
TASK-2.3) are stated in terms of "OA" and "F1 (methane class)". These map to
literal MLflow metric keys as follows — nothing else in the codebase states
this mapping, so it's pinned here:

| Criteria name | Staging metric key | Production metric key | Source |
|---|---|---|---|
| OA (overall accuracy) | `val_accuracy` | `test_accuracy` | `starcop.metrics.accuracy`, logged by `model_module.py`'s `val_epoch_end` (`prefix_accuracy`) / by `run_validation`'s test-set pass (`src/training/validation_metrics.py`, `prefix="test"`) |
| F1 (methane class) | `val_f1score` | `test_f1score` | `starcop.metrics.f1score` (methane is the positive class), same logging paths |

`val_*` metrics are logged once per validation epoch during training (the
final/last-epoch value is what promotion reads). `test_*` metrics come from
`run_validation`'s pass over the held-out test split (`splits/test.csv` —
never used during training), added to `train.py` in TASK-2.3 specifically so
Production has a metric to check against.

## Promotion criteria

(To be updated as more experiments run — the numeric thresholds below are
the plan's initial targets, not yet validated against a real trained model.)

### Staging

- `val_accuracy` ≥ 0.85
- `val_f1score` ≥ 0.70
- No training instability — automated proxy: the population stddev of
  consecutive `val_loss` deltas over the last 5 logged epochs must not
  exceed `0.1` (`src/registry/promotion_criteria.py`,
  `DEFAULT_INSTABILITY_WINDOW` / `DEFAULT_MAX_LOSS_DELTA_STDDEV`), and no
  logged `val_loss` value may be `NaN`/`Inf`. This threshold is a
  conservative starting point, not derived from a real training run yet —
  tune it once real `val_loss` histories exist.

### Production

A run is only eligible for Production if it has already cleared every
Staging criterion above.

- `test_accuracy` ≥ 0.88
- `test_f1score` ≥ 0.75

Both are evaluated against the held-out test split. If `test_accuracy` /
`test_f1score` are absent from the run entirely (e.g. `run_validation`
failed — see `train.py`'s try/except around it, TASK-2.2 note on
STARCOP's `metrics_by_difficulty` `KeyError` on skewed splits), Production
is rejected outright rather than silently skipped.

## Registry API

Uses MLflow's classic stage-based API
(`MlflowClient.transition_model_version_stage(stage="Staging"/"Production")`).
This is soft-deprecated upstream since MLflow 2.9 in favor of aliases
(`set_registered_model_alias`) and MLflow's own docs recommend migrating —
but it's still fully functional in the installed version, and matches the
"Staging"/"Production" terminology already used throughout the project plan.
Noted here as a known future migration, not acted on now.

## Model artifact

`train.py` logs the trained model two ways:

- `mlflow.log_artifact(checkpoint, artifact_path="checkpoint")` — the raw
  PyTorch Lightning `.ckpt` (TASK-2.2), useful for exact resume/inspection.
- `mlflow.pytorch.log_model(model, artifact_path="model")` — added in
  TASK-2.3, carries MLmodel flavor metadata so the registered version is
  loadable via `mlflow.pyfunc.load_model` (needed for Phase 5 serving).

`promote_model.py` registers from the `model` artifact path, not `checkpoint`.

## Running the script

```bash
.venv/bin/python src/registry/promote_model.py --run-id <run_id>
```

`--run-id` may be omitted for manual/ad-hoc use, in which case the script
falls back to the latest run in the default MLflow experiment (id `0` —
`train.py` never calls `mlflow.set_experiment`, so every run lands there
today). The eventual Phase 4 CI call should always pass an explicit
`--run-id` from the training job that produced it, not rely on this fallback.

Exits `0` and prints the promoted stage + registered version on success;
exits `1` and prints the rejection reasons otherwise.
