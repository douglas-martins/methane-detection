# Source architecture

This directory separates the pinned STARCOP reference integration from reusable
infrastructure and independently implemented research candidates.

## Ownership map

| Path | Owner and purpose |
| --- | --- |
| `../vendor/starcop/` | Pinned upstream STARCOP submodule. Never edit it. |
| `data/` | Dataset download and preprocessing. Its local vendor seam stays here because data-pipeline ownership is independent of model ownership. |
| `baselines/starcop/` | This project's STARCOP training, checkpoint import, paper evaluation, and BentoML serving adapters. |
| `training/` | Shared MLflow logging, DVC lineage, and MLflow-version compatibility helpers. It is not a second STARCOP trainer. |
| `registry/` | Shared MLflow registry and promotion policy. Baseline checkpoint import is not here. |
| `serving/` | Shared rolling drift mathematics and the `BandStats` value type. It is not a generic inference service. |
| `models/<candidate>/` | Independently implemented thesis candidates. Current scaffold homes are `tiny_ds/`, `tiny_u/`, and `spectral_tiny/`. |
| `comparison/` | Future candidate-versus-STARCOP quality and inference-time comparison. It does not own baseline paper evaluation. |

There are no `__init__.py` files under `src/`; modules use the repository's flat
import convention. Each STARCOP-consuming directory has its own uniquely named
`_vendor_starcop*.py` seam to avoid collisions during combined test collection.

## Environments

- **Baseline env:** `vendor/starcop/.venv` (Python 3.10, torch 1.13.1), for
  reproducing the original stack and supported baseline checks.
- **Research env:** `.venv` (Python 3.12, torch >=2.5), the default for active
  development, registry operations, serving, and candidate work. Blackwell GPU
  training also uses this environment.

STARCOP integration is baseline-specific ownership, not baseline-env-only code:
its tests and selected commands deliberately run in both environments.

## Entrypoints

Run commands from the repository root. Stable wrappers are preferred:

```bash
./scripts/train_mac.sh starcop_mini
./scripts/train_desktop.sh starcop_mini
.venv/bin/python scripts/import_starcop_hf_baseline.py mag1c_rgb --stage Staging
.venv/bin/python scripts/run_starcop_baseline_evaluation.py mag1c_rgb --limit 5
.venv/bin/bentoml serve src.baselines.starcop.serving.service:MethaneDetectionService
```

The Mac wrapper uses the baseline env; the desktop wrapper and BentoML use the
research env. See [`baselines/starcop/README.md`](baselines/starcop/README.md) for
component commands and contracts, and
[`../internal-docs/runbooks/training.md`](../internal-docs/runbooks/training.md)
for credentials and machine-specific setup.

Run `make test-baseline` and `make test-research` for the supported suites. New
candidate code starts with a failing test in its own model directory before it is
added to the shared CI/coverage lists.
