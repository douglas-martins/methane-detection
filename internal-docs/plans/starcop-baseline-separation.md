# Separate STARCOP baseline code from shared infrastructure

> **Status:** BS-01 through BS-09 code work completed on
> `refactor/starcop-baseline-core`; all available local validation passed.
> Final integration acceptance remains blocked on deployed-artifact identity and
> an immutable rollback image digest; no deployment is authorized. See the
> [validation evidence](../starcop-baseline-migration-validation.md).
> **Scope:** Behavior-preserving refactor into `src/baselines/starcop/`, separate
> from the completed [candidate scaffold](../../model-hypotheses-scaffolding-plan.md).

## Goal

A newcomer should immediately distinguish:

- `vendor/starcop/`: untouched, pinned upstream implementation.
- `src/baselines/starcop/`: this project's STARCOP training, evaluation, import,
  and serving integration — usable under the appropriate baseline or research env.
- `src/models/<candidate>/`: independently implemented research candidates.
- Existing top-level infrastructure directories: genuinely reusable utilities.
- `src/comparison/`: future candidate-versus-STARCOP quality and inference-time
  comparison, not another copy of the STARCOP evaluator.

“Baseline-specific code” does **not** mean “baseline-environment-only code.” Much
of this integration already runs under Python 3.12/research env; preserve that.

## Constraints and non-goals

- Never modify anything under `vendor/starcop/`, including its environment.
- Preserve flat imports, no `__init__.py` under `src/`, and distinct vendor seams
  per consuming directory. Do not introduce a new packaging system in this refactor.
- Preserve training behavior, Hydra options, metrics, model names/stages, MLflow
  tags/artifact names, HTTP contracts, monitoring metric names, and data layout.
- No model implementation, new promotion policy, hardware choice, dependency
  upgrade, data migration, or automatic deployment.
- Keep `src/data/` and its STARCOP preprocessing seam where they are: data-pipeline
  ownership is separate from model-baseline ownership.
- Keep `configs/training/overlay.yaml`, datasets, checkpoints, and existing MLflow
  artifacts in place. Update resolution of their paths, not their contents/identity.
- Do not modify unrelated root plans or include them in commits.
- The earlier scaffold's “do not move existing code” constraint applied to that
  scaffold. This separately approved refactor supersedes it only for the files below.
- Minimal path and command corrections in public `docs/` are approved for this
  migration. No public documentation restructuring, methodology changes, or new
  results are in scope. Do not create shims solely to preserve outdated examples.

## Proposed layout and ownership

```text
src/
  README.md                         architecture map and entrypoint guide
  baselines/
    starcop/
      README.md                     baseline contracts, environments, commands
      training/                     STARCOP orchestration and compatibility
        __tests__/
      evaluation/                   STARCOP dataset and paper-metric evaluation
        __tests__/
      registry/                     STARCOP checkpoint import only
        __tests__/
      serving/                      STARCOP BentoML service and inference
        __tests__/
  training/                         shared tracking/data-lineage utilities
    README.md
    __tests__/
  registry/                         shared registry and promotion infrastructure
    __tests__/
  serving/                          shared drift statistics, not a generic service
    __tests__/
  models/{tiny_ds,tiny_u,spectral_tiny}/
  comparison/
```

Do not leave an empty `src/evaluation/` placeholder after its implementation and
all tests move. Retain it only if an approved compatibility entrypoint requires it.
Existing `.gitkeep` and cache files are not implementation: do not copy caches or
create speculative shared packages.

### Initial file disposition

Confirm this manifest against all tracked files and transitive consumers at the
start of implementation. Lack of a direct `starcop` import is not proof of generic
behavior: configuration, checkpoint, metric, and band contracts also count.

| Current directory | Move under `src/baselines/starcop/<directory>/` | Keep shared at current location |
| --- | --- | --- |
| `src/training/` | `train.py`, `_vendor_starcop_training.py`, `starcop_datamodule.py`, `settings_overlay.py`, `launch_profiles.py`, `colab_bootstrap.py`, `accelerator_check.py`, `lightning2_compat.py`, `normalizer_dtype_fix.py`, `optimizer_compat.py`, `metrics_ext.py`, `mlflow_image_logger.py`, `plot_confusion_matrix.py`, `validation_metrics.py` | `mlflow_utils.py`, `dvc_dataset_version.py`, `mlflow_log_model_compat.py` |
| `src/evaluation/` | All seven current implementation modules: `dataset_wiring.py`, `live_verify.py`, `paper_eval_mlflow.py`, `paper_metrics.py`, `run_baseline_eval.py`, `select_docs_examples.py`, `_vendor_starcop_evaluation.py` | None currently identified |
| `src/registry/` | `hf_baseline_import.py`, `_vendor_starcop_baseline.py` | `mlflow_registry.py`, `promotion_criteria.py`, `promote_model.py` |
| `src/serving/` | `service.py`, `inference.py`, `model_loader.py`, `band_baseline.py`, `_vendor_starcop_serving.py` | `drift.py`, after the narrow dependency split below |

Training helpers such as accelerator checks may be reusable in principle, but
currently carry baseline launch/environment policy. Keep them with their current
owner rather than inventing a generic API before another consumer needs it.
`mlflow_log_model_compat.py`, by contrast, introspects a supplied logging function
and does not require STARCOP model internals.

Move each module's tests alongside its owner. In particular,
`test_model_forward.py` belongs with baseline training; shared tracking, registry,
and drift tests remain top-level. Split/update local `conftest.py` path setup rather
than copying one broad fixture file everywhere.

### Narrow shared/baseline split: band statistics

`drift.py` is algorithmically generic but imports `BandStats` from
`band_baseline.py`, which also embeds STARCOP model names and training statistics.
Moving only `band_baseline.py` would make shared infrastructure depend on a baseline.

Through a failing test first, extract only the `BandStats` type into a uniquely
named shared module such as `src/serving/band_statistics.py`. Both drift logic and
the relocated baseline statistics import it. Preserve tuple behavior and existing
numeric results. Do not generalize the hard-coded STARCOP statistics registry now.
Check whether old serialized artifacts reference the original type/module before
removing or replacing that import surface.

## Import and path rules

Dependency direction: baseline adapters may use shared infrastructure; shared
infrastructure must not import baseline modules or configure the STARCOP vendor
path as a side effect. Candidate/comparison code must not depend on baseline
training orchestration merely to access a generic helper.

- Preserve unique `_vendor_starcop_training`, `_vendor_starcop_evaluation`,
  `_vendor_starcop_baseline`, and `_vendor_starcop_serving` seam names.
- Do not name a flat seam simply `starcop.py`, which would shadow upstream.
- Candidate modules/tests retain their candidate prefixes from the scaffold.
- Audit **every** `Path(__file__).parents[...]`, relative sibling-directory
  expression, Hydra config path, `sys.path` modification, and string-based module
  import. Moving files two levels deeper invalidates current root calculations.
- Make repo-root and shared-directory resolution explicit at the entrypoint/seam
  boundaries; avoid a global import-hook workaround or broad recursive sys.path.
- Run imports in clean subprocesses and in the combined suite. A test passing
  only because another suite already inserted its directory is not a passing seam.
- Verify execution from the repository root and from a different working directory
  wherever the existing CLI supports that; preserve existing cwd semantics.

## Compatibility decisions required before moving code

### Commands

Recommended: keep user-facing `scripts/train_mac.sh`, `scripts/train_desktop.sh`,
`scripts/import_starcop_hf_baseline.py`, `scripts/run_starcop_baseline_evaluation.py`,
and `scripts/run_live_verify.py` stable and redirect their internal paths.

**Approved policy:** preserve the script entrypoints above, update repo-owned
direct Python paths and the BentoML service string, and document those changes.
Inventory notebooks, runbooks, external scheduled jobs, and deployed launch commands.
Retain old-path compatibility only for a demonstrated external consumer or a
serialized artifact that needs it, never as a blanket policy.

Every retained shim must have a named owner, documented consumer, regression test,
removal condition, and tracked follow-up task in the migration evidence record.
Command migration alone does not justify removing serialization compatibility.
Do not keep duplicate logic or same-name flat aliases that shadow new modules.

### Existing checkpoints and MLflow models

Relocation can break pickle/cloudpickle references to local classes or rebound
methods even when model weights and vendor imports are unchanged. Before choosing
module renames or deleting old paths:

- Inventory representative existing Lightning checkpoints and MLflow PyTorch
  artifacts, including models logged with local compatibility hooks.
- Load them in fresh processes using the relocated loader, without the old source
  directories on `sys.path`; compare fixed-input predictions with pre-move results.
- Inspect packaged `code_paths`/artifact source and serving dependencies.
- If old local module names are required, preserve a minimal tested import surface
  or approve an explicit artifact migration. Do not silently re-register models,
  relabel versions, retrain, or require deletion of old artifacts.

BS-01 must freeze an artifact matrix covering an existing project-trained Lightning
checkpoint, a project-trained MLflow artifact, an imported pretrained STARCOP model,
and the deployed artifact if different. One artifact may cover multiple categories.
Record identifier/hash, producing environment when known, supported loading
environment, and BS-02 fixed-input reference. Record access gaps explicitly; do not
invent available artifacts. Unavailable artifacts block integration acceptance, not
independent offline code tasks.

### Numerical parity and safe validation

Freeze criteria in BS-02, before relocation: compare within the same environment,
device, seed, and inputs. Require identical config, metric keys, thresholds, and
metadata contracts. Require exact deterministic results; establish justified
numerical tolerances from pre-move behavior for nondeterministic operations, never
from post-move discrepancies. Cross-environment floating-point equality is not
required. Use disposable local tracking stores and no production registry writes.

### Review milestones and deployment authorization

Use two independently validated PRs:

- **Milestone A:** BS-01–BS-06, with a transitional architecture note explaining
  that serving remains at its original location. BS-03 may land here as preparation.
- **Milestone B:** BS-07–BS-09, completing serving and final acceptance.

Both PRs require passing applicable code checks; milestone A is not a claim of
full deployment readiness. Historical-artifact and runtime integration gaps must
be visible to reviewers. Final acceptance and deployment remain blocked by any
required integration gap.

BS-01 must audit actual CD triggers and release/deployment coupling. Before either
PR is merged, verify a manual deployment gate prevents unauthorized automatic
rollout; implement a narrowly scoped gate change if necessary and validate it.
Merging must not implicitly authorize deployment. Record the last known-good image
and launch configuration for rollback. Deployment always requires explicit approval.

## Implementation tasks

Execute one task at a time, validate it, and record evidence before proceeding.
The task order below is the execution sequence; the later migration-phase sections
are cross-cutting checklists, **not** instructions to postpone caller/CI updates
until all folders have moved. Every intermediate state must remain runnable.

All tasks start `Not started`. Track **Code validation** and **Integration
validation** separately, each as `Not started`, `In progress`, `Blocked`, `Passed`,
or `Not applicable` with a reason. Record commands, environments, results, existing
failures, and blocked checks in the evidence record.

Task dependencies permit subsequent code work once prerequisite code validation
passes. Missing historical artifacts or Docker block integration validation, not
otherwise independent code work. A missing supported Python environment still
blocks code validation for that environment. Never mark a task fully validated
until all its applicable checks pass; BS-09 and deployment require all required
integration checks to pass too.

| Task | Deliverable | Depends on | Status |
| --- | --- | --- | --- |
| BS-01 | Approved inventory, compatibility decisions, and starting evidence | — | Code passed; integration blocked on remote artifact identity and immutable rollback digest |
| BS-02 | Characterization fixtures and regression reference | BS-01 | Code passed; integration blocked on remote/deployed artifacts |
| BS-03 | Shared band-statistics type without baseline dependency | BS-02 | Code and task-scope integration passed |
| BS-04 | Baseline training moved, shared training retained | BS-02 | Code and task-scope integration passed |
| BS-05 | Baseline checkpoint import moved, shared registry retained | BS-04 | Code and task-scope integration passed |
| BS-06 | Baseline evaluation moved and all evaluation callers updated | BS-05 | Code and task-scope integration passed |
| BS-07 | Baseline serving moved with complete Bento/CD wiring | BS-03, BS-06 | Code passed; integration blocked on deployed artifact and immutable image digest |
| BS-08 | Final architecture guide and compatibility cleanup | BS-07 | Code and task-scope integration passed |
| BS-09 | Whole-system acceptance and review handoff | BS-08 | Code passed; integration blocked on remote/pretrained/deployed artifacts and immutable rollback digest |

### Rules for every implementation task

- Read the TDD and lint/docstring instructions before Python changes. Add the
  failing regression test before implementing new extraction/path behavior;
  preserve and run existing tests for mechanical moves.
- Move tests with their owner. Update `Makefile`, workflow triggers, Codecov flag
  paths, coverage omissions, and test `conftest.py` **in the same task that changes
  the relevant source path**. Retain coverage of code that has not moved yet.
- Update every affected caller, even if its own folder moves in a later task.
  During the transition, sibling folders can be at different depths; resolve each
  actual location rather than assuming the final layout already exists.
- Run focused tests first, then `make test-baseline` and `make test-research` for
  every relocation task. Verify collection counts/skips against BS-01 so a missing
  suite cannot masquerade as success.
- After Python changes run `make lint` (scope fixes to changed Python files when
  full-repo findings pre-exist) and full-repo `make docstring-coverage`. Use both
  environments for modules currently supported in both; serving remains research.
- Correct active command examples and agent guidance affected by that task now;
  BS-08 consolidates the architecture explanation, not overdue runtime instructions.
- Record `git diff --check`, vendor cleanliness, commands/results, and unresolved
  items. Do not update datasets, dependency locks, or registry state to make tests
  pass. No commit, push, or deployment is implicit in completing a task.

### BS-01 — Inventory and approve compatibility decisions

**Scope:** No source moves or behavior changes.

1. Establish the dedicated refactor branch and explicitly account for the scaffold
   work already in progress; inspect staged, unstaged, and untracked changes.
2. Confirm the file/test disposition above against tracked source and transitive
   imports. List all old-path consumers, including notebooks and deployed commands.
3. Record available environments/tools/artifacts and run starting test collection,
   both full suites, both coverage targets, lint, and docstring coverage. Record
   failures as pre-existing rather than silently fixing unrelated problems.
4. Choose direct-command compatibility, any temporary shim removal milestone, and
   the public-doc permission policy. Inventory historical artifacts without changing
   their registry identity or downloading sensitive data into tracked fixtures.
5. Audit CD/release triggers and establish the manual deployment gate and rollback
   reference required above. Freeze the artifact matrix and assign an owner/removal
   condition/follow-up task for every required shim.
6. Store a concise migration evidence record at
   `internal-docs/starcop-baseline-migration-validation.md` (new internal document).

**Validate:** Every tracked module/test has an owner; all compatibility decisions
are recorded; baseline test/coverage counts and tooling limitations are captured.

**Stop point:** No migration has begun. If compatibility decisions remain open,
resolve them before BS-02 or any source extraction.

### BS-02 — Capture behavior before changing paths

**Scope:** Tests/fixtures and local validation procedures only.

1. Reuse existing tiny fixtures and add missing characterization coverage for
   deterministic forward output, evaluation metrics, config composition, and
   serving response contracts. Do not launch a full training experiment.
2. Capture fixed-input expected logits/masks and metric keys/values, with explicit
   numerical tolerances, seeds, model identity, and environment. Use real small
   fixtures rather than broad interaction mocks.
3. Add fresh-process import/CLI checks for supported entrypoints and cwd behavior.
4. Establish a reproducible local historical-artifact load check and a safe local
   serving smoke procedure. Record artifact locations/hashes, not binary artifacts
   in Git. Distinguish these external checks from the offline unit suite.

**Validate:** Characterization tests pass on the pre-move implementation in their
supported environments; fixture outputs and smoke commands are reproducible.
Run quality gates for any added Python.

**Stop point:** Behavior is documented and testable; no production modules moved.

### BS-03 — Extract the shared band-statistics type

**Scope:** `src/serving/band_statistics.py`, `band_baseline.py`, `drift.py`, and
associated tests. No directory moves yet.

1. Write a failing test that shared drift/statistics imports require neither
   baseline statistics nor the vendor seam, then extract only `BandStats`.
2. Point both consumers at the shared type. Preserve the old type import surface
   where BS-01's artifact/command inventory requires it.
3. Preserve tuple semantics, drift numerical behavior, and STARCOP band mappings.
4. Ensure `bentofile.yaml` packages the new shared file; its current wildcard may
   already do so, but verify the runtime dependency closure.

**Validate:** Focused drift/band-statistics/band-baseline tests, clean-process
imports, research suite, quality gates, and any identified serialized-type check.

**Stop point:** Existing service and old paths still work; shared drift no longer
imports STARCOP-specific statistics. BS-04 can be done independently after BS-02.

### BS-04 — Move baseline training as a complete slice

**Scope:** Training move list and tests; retain the three shared training modules.

1. Add failing new-location/config-root checks, then relocate training modules and
   their tests. Resolve vendor/config roots and access to shared training utilities.
2. Update shell launchers, Bats assertions, notebooks/Colab bootstrap, and any flow
   invocation or documented direct command affected by the training move.
3. Update consumers still at old locations: notably baseline import/evaluation
   modules and their fixtures that reach into `src/training/`.
4. Update both environment test/coverage lists, CI triggers, Codecov paths, and
   omissions without removing shared training coverage. Update AGENTS' training
   seam paths and create/update the shared training README.

**Validate:** Training tests in both envs; both full suites; `make test-scripts`;
CLI/config composition in clean processes; tiny forward/training smoke against
BS-02; both coverage targets with inspected source lists; quality gates.

**Stop point:** All existing launchers work; training lives at the new path;
registry/evaluation/serving may still live at their old paths and must still work.

### BS-05 — Move baseline checkpoint import

**Scope:** `hf_baseline_import.py`, its vendor seam and tests. Generic registry and
promotion modules stay where they are.

1. Add failing new-location/import checks, then move the baseline import modules.
2. Fix repo-root, vendor, relocated training, and shared-registry resolution.
3. Update `scripts/import_starcop_hf_baseline.py`, evaluation consumers such as
   `live_verify.py`, and affected fixtures while evaluation remains at its old path.
4. Update test/coverage wiring and current command references in this slice.
5. Exercise historical model serialization/import compatibility using local
   fixtures or a temporary registry, never by promoting a production model.

**Validate:** Import and generic registry tests in both supported envs; both full
suites; clean-process import CLI smoke; representative artifact loading against
BS-02; both coverage targets and quality gates.

**Stop point:** Checkpoint import works from the new directory; generic registry
imports do not acquire a baseline dependency; old-location evaluation still works.

### BS-06 — Move baseline evaluation

**Scope:** All seven evaluation modules and their tests.

1. Add failing path/seam checks, move evaluation, and fix references to baseline
   training/import and shared registry/tracking modules at their actual locations.
2. Update evaluation/live-verification scripts, evaluation flows and tests, and
   active commands. Keep the serving endpoint contract unchanged.
3. Update test/coverage wiring. Remove the old tracked evaluation directory unless
   an approved compatibility entrypoint still needs it; do not copy caches.
4. Preserve paper-metric definitions, examples selection, MLflow tags/artifacts,
   and evaluation dataset/split semantics.

**Validate:** Focused evaluation and flow tests, both full suites, clean-process
CLI smoke, tiny fixture metrics/logits against BS-02, both coverage targets, and
quality gates. Live verification continues to target a local service only.
Publish the transitional ownership/command note with milestone A; confirm its
applicable checks and deployment gate before requesting merge.

**Stop point:** Training, import, and evaluation work together at new locations;
the not-yet-moved serving service remains usable.

### BS-07 — Move baseline serving and its delivery wiring

**Scope:** Baseline serving modules/tests; shared drift/statistics remain top-level.
Bento configuration, CD change detection, and deployment launch references are
part of this task, not a later deployment task.

1. Add failing loader/path checks, then move baseline serving modules and tests.
2. Resolve shared drift/statistics/registry imports and the vendor seam without
   relying on another test suite to populate `sys.path`.
3. Change the service import string and bundle includes in `bentofile.yaml`.
   Update `.github/workflows/cd.yml`, `deploy/bentoml/`, relevant flow/CLI commands,
   and monitoring references only where paths actually change.
4. Update research test/coverage wiring, current service commands, and any approved
   temporary compatibility entrypoint. Do not expand baseline-env serving support.
5. Load representative old artifacts using the relocated loader and compare outputs.

**Validate:** Research serving/registry/evaluation/flow tests; both full suites;
coverage targets; quality gates; local real `bentoml serve` with predict, health,
and metrics requests; Bento build and container smoke including shared modules.
Unavailable Docker/artifacts block integration validation, but BS-08 code/docs
work may proceed once BS-07 code validation passes.

**Stop point:** A locally validated service bundle uses the new layout. No remote
service has been restarted and no model has been promoted or deployed.

### BS-08 — Finish architecture documentation and audit stale paths

**Scope:** Final ownership documentation and intentional compatibility cleanup.

1. Add/consolidate `src/README.md`, baseline README, and shared training README.
   Show real working commands, both environments, and candidate-versus-baseline
   ownership without implying the shared serving directory is a generic service.
2. Reconcile AGENTS, CONTRIBUTING, active runbooks/notebooks, approved public path
   corrections, hypotheses references, and the completed scaffold's follow-up link.
3. Search tracked files for stale paths/module strings. Classify historical text,
   valid shared paths, and approved shims; fix all unclassified live consumers.
4. Remove a compatibility shim only if its agreed milestone has been met, with a
   failing regression test first for any changed behavior. Otherwise list it as
   intentionally retained, with owner/removal condition, in the evidence record.

**Validate:** Internal links and example paths resolve; Markdown diff check;
`make docs-build` if published documentation changed. Any code cleanup also runs
its focused/full affected tests and Python quality gates. Review the architecture
map against the actual tracked tree, not only the proposed manifest.

**Stop point:** The repository explains its current layout without erasing history.

### BS-09 — Final acceptance and review handoff

**Scope:** Cross-slice validation, evidence consolidation, and rollback readiness.

1. Run the complete validation checklist below on the combined tree: both suites,
   both coverage targets, Bats, quality gates, CLI/config smokes, fixed-input parity,
   historical artifact loading, and local serving/build/container checks.
2. Compare test counts/skips, coverage ownership, outputs and artifact metadata
   against BS-01/BS-02. Explain intended path-only differences; investigate others.
3. Confirm vendor/data/lockfile/registry identities are unchanged and no old source
   directory is accidentally required by fresh-process or packaged execution.
4. Finish the evidence record with per-task results and unresolved blockers.
5. Prepare an auditable PR/commit grouping proposal and record the last known-good
   deployment image/config rollback reference. Commit, push, PR creation, and
   deployment each require the relevant user authorization.

**Validate:** Every task is validated; no required check is blocked; the definition
of done below is satisfied. If environment access prevents this, report completed
slices separately from an explicitly blocked final acceptance.

**Stop point:** Ready for review and an independently authorized deployment, not
already deployed.

## Cross-cutting migration checklists

### 0. Freeze the starting state and capture behavior

1. Work on a dedicated refactor branch via PR, never commit to `main`. Finish or
   explicitly carry the current scaffold changes; do not mix their ownership into
   refactor commits accidentally.
2. Read the repo's TDD and lint/docstring skills/rules before Python edits.
3. Record `git status`, tracked-file inventory, existing failures, test collection
   per environment, and coverage source/flag membership.
4. Capture a deterministic tiny-fixture forward/evaluation result and current
   serving responses. Add characterization tests before changing behavior seams.
5. Search the whole tracked repo, including notebooks, deployment assets, configs,
   scripts, workflows, docs, agent guidance, and tests, for old paths/module strings.
6. Apply the approved compatibility policies and freeze the move manifest.

### 1. Untangle the shared statistics dependency

Write/run the failing test first, extract `BandStats`, and keep existing imports
working as needed. Run drift/band-baseline tests. This is a narrow refactor, not an
excuse to redesign monitoring or relocate all utilities into a new `common/` tree.

### 2. Relocate baseline code and its tests

Move the manifest files with history-preserving Git renames where practical. Fix
root/sibling paths, entrypoint imports, test fixtures, and shared utility access.
Preserve existing function/class names unless a proven collision requires otherwise.

Add migration regression tests before new path-resolution logic: verify the new
seams resolve the real vendor/config locations, baseline and shared tests collect
together, and shared modules import without baseline modules. Preserve Python 3.10
syntax compatibility for code exercised by the baseline environment.

### 3. Update all consumers and existing CI in the same migration

This is **existing baseline integration**, not deferred first-candidate integration.
Update it now; the scaffold's candidate CI deferral does not apply to moved tests.

Explicit checklist:

- `Makefile`: both environments' test paths and coverage paths, preserving the
  prior baseline-vs-research coverage scope without broadening gates accidentally.
- `.github/workflows/tests.yml`: include `src/baselines/starcop/**` path triggers;
  retain shared-path triggers and update comments and any explicit commands.
- `codecov.yml`: map relocated sources into their existing environment flags.
- `pyproject.toml`: review coverage omissions against new paths; do not add broad
  exclusions to hide failures. The existing `*/training/train.py` wildcard may
  still match, but verify deliberately.
- Package-local/root test fixtures and any literal source-path assertions.
- Shell launchers and Bats expectations (`train_mac.bats`, `train_desktop.bats`).
- Python CLI wrappers listed above; `flows/retrain.py`, `flows/eval_baseline.py`,
  their tests, and any Prefect deployment configuration with source-path assumptions.
- `bentofile.yaml`: service string becomes
  `src.baselines.starcop.serving.service:MethaneDetectionService`; include relocated
  serving source, shared `drift.py`/statistics type, `mlflow_registry.py`, and vendor
  source. Verify the complete runtime dependency closure, not only this list.
- `.github/workflows/cd.yml`: source change-detection paths and service references;
  inspect `deploy/bentoml/`, containers, launch commands, and monitoring configuration.
- Notebooks/Colab bootstrap cells, active setup/runbooks, config comments, README,
  CONTRIBUTING, AGENTS, and relevant agent skill examples.

Preserve historical experiment/journal text as historical evidence. Add a migration
note where old commands may otherwise look current; do not rewrite old checkpoint
paths or falsify previous run provenance. Update active architecture references in
`internal-docs/plans/hls4ml-methane-model-hypotheses.md` and link this follow-up from
the completed scaffold instead of rewriting that scaffold as if it moved code.

### 4. Validate before declaring the migration ready

- Both full suites: `make test-baseline` and `make test-research`.
- Both coverage targets: `make coverage` and `make coverage-research`; inspect XML
  paths, omissions, test counts/skips, and Codecov flag membership against phase 0.
- `make lint`, fixing findings on changed Python files; document pre-existing
  full-repo findings. Full-repo `make docstring-coverage` must pass.
- `make test-scripts` for wrapper behavior, with Docker availability reported.
- Clean-process CLI help/config composition smoke tests using the appropriate env;
  a tiny training/evaluation smoke where dependencies and fixtures permit.
- Before/after fixed-input logits, masks, metric keys/values, and artifact metadata.
- Representative historical artifact loading as described above.
- Research-env real `bentoml serve` plus predict/health/metrics smoke; local
  `bentoml build` and container smoke when available. Do not contact production or
  mutate the production registry as a validation shortcut.
- Repo-wide stale-path search: classify remaining matches as compatibility shims,
  historical records, or intentional shared paths; fix unclassified runtime uses.
- Confirm `git diff -- vendor/starcop` is empty and there are no unintended changes
  to data, lockfiles, model artifacts, HTTP contracts, or registry identities.

Missing environment/tool/artifact access must be reported with the exact blocked
check and its code/integration classification. Static path searches are not
substitutes for runtime or deployment tests. Integration gaps do not stop independent
code work, but they block final acceptance and deployment.

### 5. Document, review, and hand off

Add `src/README.md`, baseline README, and shared training README explaining the
ownership map, commands for both envs, and where a new model belongs. Update
`AGENTS.md` to match the new seam paths and architecture.

Prepare coherent Conventional Commit groups per repository rules: narrow dependency
refactor, baseline relocation plus required consumers, CI/build wiring where it can
be separated without a broken intermediate state, and documentation. Group by
actual type/scope; preserve runnable commits rather than forcing an artificial
split. No commit/push/deploy is implied by approval of this draft alone.

Deployment is a separate authorized action. Retain the last known-good Bento/image
and its launch configuration for rollback; revert the migration PR as a unit if
needed. No data/registry rollback should be necessary for a path-only refactor.

## Definition of done

- Baseline-specific model integration has one clear home under
  `src/baselines/starcop/`; generic directories contain no hidden baseline imports.
- Existing baseline and research execution remain supported with explicitly
  documented command compatibility.
- Existing tests remain collected/covered; deployment bundles contain all required
  relocated and shared modules.
- Model predictions, metric semantics, API behavior, artifact compatibility, and
  registry identity are unchanged, with validation evidence or explicit blockers.
- New contributors can locate baseline code, candidate code, shared utilities, and
  comparison responsibilities from `src/README.md` alone.
