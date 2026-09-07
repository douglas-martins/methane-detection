# Shared training infrastructure

This directory contains model-agnostic training infrastructure used by the
STARCOP baseline and available to future candidate models:

- `dvc_dataset_version.py`: DVC dataset identity and dirty-state checks.
- `mlflow_utils.py`: MLflow environment validation, config flattening, and tags.
- `mlflow_log_model_compat.py`: version-compatible MLflow logging arguments.

STARCOP-specific orchestration, compatibility hooks, metrics, data-module wiring,
and launch profiles live in [`../baselines/starcop/training/`](../baselines/starcop/training/).
Baseline adapters may import these helpers; shared modules here must not import a
baseline adapter or configure the STARCOP vendor path.

From the repository root, run STARCOP training through
`./scripts/train_mac.sh starcop_mini` (baseline env) or
`./scripts/train_desktop.sh starcop_mini` (research env). Direct commands and
credential setup are documented in
[`internal-docs/runbooks/training.md`](../../internal-docs/runbooks/training.md).
Candidate-specific orchestration belongs with its model under `src/models/`, not
in this directory until a genuinely shared interface has more than one consumer.
