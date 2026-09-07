# STARCOP baseline migration — validation evidence

## Starting point

- Milestone A branch: `refactor/starcop-baseline-core`.
- Starting commit: `a7c1d5a` (main after scaffold PR #38).
- Scope: BS-01–BS-06; baseline serving relocation is milestone B.
- No source moves, dependency changes, commits, pushes, or deployments performed
  during this initial audit.
- Starting tracked worktree and index were clean. Four local plans were untracked:
  `docs-reorganization-plan.md`, `mlops-methane-detection-plan.md`,
  `model-hypotheses-scaffolding-plan.md`, and
  `internal-docs/plans/starcop-baseline-separation.md`. Preserve them; the root plans
  are not deliverables for this PR. The migration plan remains local unless its
  inclusion is separately approved.

## Task status

| Task | Code validation | Integration validation | Evidence / next action |
| --- | --- | --- | --- |
| BS-01 | Passed | Blocked | Inventory and local artifact classification complete; workflow is explicit-dispatch-only and actionlint passes; deployed artifact/digest and immutable rollback digest require authorized external read access |
| BS-02 | Passed | Blocked | Five migration-contract tests pass in both envs; local checkpoint parity frozen and repeated exactly; remote artifacts/live service remain deferred integration checks |
| BS-03 | Passed | Passed | Shared type extracted test-first; compatibility and Bento source bundle verified |
| BS-04 | Passed | Passed | Baseline training slice relocated; wrappers, callers, CI/coverage, docs, both environments, fixed-output parity, and script tests verified |
| BS-05 | Passed | Passed | Baseline checkpoint importer relocated; shared registry remains independent; local MLflow, CLI, callers, coverage, and fixed-output parity verified |
| BS-06 | Passed | Passed | Baseline evaluation and tests relocated; wrappers, Prefect flow, CI/coverage, clean-process imports, metrics, and checkpoint parity verified |
| BS-07 | Passed | Blocked | Baseline serving relocated; local source/Bento/container HTTP smokes passed, but deployed artifact parity and immutable running-image identity require authorized read access |
| BS-08 | Passed | Passed | Architecture maps, current commands, stale-path classification, links, and docs build verified; no compatibility shim was removed |
| BS-09 | Passed | Blocked | Combined local acceptance passed; remote MLflow/pretrained/deployed artifact checks and immutable rollback identity require authorized external read access |

## Starting checks

Commands ran with `PYTHONDONTWRITEBYTECODE=1 timeout 180 make <target>` to avoid
writing bytecode under the vendor tree and to bound each invocation. Both existing
interpreters are executable: research Python **3.12.14**, baseline Python **3.10.21**.
No environment installation or synchronization was performed.

| Target | Result before migration |
| --- | --- |
| `make test-baseline` | Passed: 299 collected, 297 passed, 2 skipped, 26 warnings |
| `make test-research` | Passed: 471 collected, 471 passed, 130 warnings |
| `make coverage` | Passed: same baseline test totals; displayed total 87% (359 statements, 45 misses, 94 branches, 4 partial branches) |
| `make coverage-research` | Passed: same research test totals; displayed total 79% (1652 statements, 347 misses, 350 branches, 21 partial branches) |
| `make lint` | Failed before changes: notebook E501 and E402; make stops before its format step |
| `make docstring-coverage` | Passed: 87.9%, 218/248 documented, threshold 80% |
| `.venv/bin/ruff format --check .` | Failed before changes: one file would be reformatted, 156 already formatted |
| `docker info --format '{{.ServerVersion}}'` | Initially blocked by socket permissions; access rechecked successfully after user enabled it: server 29.7.2 |
| `make test-scripts` | All 31 Bats tests reported `ok`; see logging caveat below |
| `git diff -- vendor/starcop` | Empty after checks |

Baseline skips are the two `TestImportVariant` integration tests in
`src/registry/__tests__/test_hf_baseline_import.py`: logging provenance without a
stage, and registering/promoting with a stage. Preserve these as the starting skip
set; do not silently introduce additional skips.

Pre-existing Ruff findings:

- `notebooks/train_colab.ipynb`, cell 3 line 14: E501 (103 characters).
- Same notebook, cell 14 line 23: E402 (`dotenv` import after setup).
- Format check requests changes to embedded examples in
  `.agents/skills/test-driven-development/SKILL.md`; no autofix applied.

Initial raw logs and path-search output were written under
`/tmp/starcop-migration-bs01/`. That temporary directory was no longer present at
the Docker follow-up; these are not durable CI artifacts. The recorded results
above preserve the starting evidence. Coverage commands also produced the standard
ignored XML/JUnit outputs. No badges were regenerated.

### Docker follow-up

After the user enabled Docker access, `docker info` succeeded (server 29.7.2).
The existing `bats/bats:latest` image ID was
`sha256:5322b877351fda0cc435de8c6116de7d0a2ec79d7c680132a0ef329a633bc66f`.
`make test-scripts` executed all 31 script tests in that container; every test
reported `ok`, covering Prefect worker and desktop/mac training wrappers.

Logging caveat: the invocation was piped through `tee` with `pipefail`; `tee`
failed because the temporary log directory no longer existed, so the outer shell
returned 1 despite the complete successful Bats output. No log file was saved and
this is not a test failure. Docker permission is now resolved. This validates the
script suite only, not Bento build, model loading, or containerized serving; those
remain scheduled integration checks. No production service or privileged workaround
was used.

### Coverage ownership frozen for comparison

- Baseline measured sources/Codecov flag: `src/data/download/`,
  `vendor/starcop/scripts/preprocessing/`, `src/training/`.
- Research measured sources/flag: `src/data/preprocessing/`, `src/training/`,
  `src/registry/`, `src/serving/`, `src/evaluation/`, `flows/`.
- Baseline tests also cover registry/evaluation, but their source is **not** part
  of baseline coverage today. Do not accidentally expand it during relocation.
- Coverage omission `*/training/train.py` needs deliberate verification after
  the move; do not broaden it to suppress unrelated modules.

## Ownership and consumer inventory

The initial tracked inventory matches the proposed production-file manifest:

- Baseline training: move the 15 modules listed in the plan; retain
  `mlflow_utils.py`, `dvc_dataset_version.py`, `mlflow_log_model_compat.py` shared.
  Matching tests move except the three shared-utility tests; `test_model_forward.py`
  moves with baseline training. Split its local `conftest.py` appropriately.
- Baseline registry: move `hf_baseline_import.py`, `_vendor_starcop_baseline.py`,
  and `test_hf_baseline_import.py`; retain registry/promotion implementation/tests.
- Baseline evaluation: all seven production modules and all six test modules move;
  its local `conftest.py` must follow actual training/registry locations at each step.
- Baseline serving: all except shared `drift.py` move in milestone B; the four
  current serving test modules split by owner. Extract `BandStats` in BS-03, with
  its tests, without moving serving yet.
- Per-directory fixtures require separate review rather than wholesale copying.
  `.gitkeep` files are not source modules; caches must not be moved.

A tracked-repo search for `src/(training|evaluation|registry|serving)` and dotted
variants was captured before edits. Known consumer categories to resolve in each
slice include:

| Consumer | Required ownership-aware update |
| --- | --- |
| `Makefile`, `codecov.yml`, test workflow, coverage config | Preserve original environment membership while changing paths |
| Training shell launchers, Bats tests, Colab notebook/bootstrap | Keep wrapper names, update direct training/helper paths |
| Import/evaluation/live-verify Python scripts | Keep CLI names, update module search locations |
| Evaluation modules and `hf_baseline_import.py` | Fix cross-directory training/registry references during partial migration |
| Local test `conftest.py` files | Prove imports independently and in combined suites |
| Prefect flows and tests | Audit both explicit paths and constructed subprocess commands |
| Bento config/CD/deployment compose | BS-03 dependency closure now; serving relocation in milestone B |
| AGENTS, CONTRIBUTING, README, active runbooks/public commands | Update live guidance with each moved slice |
| Historical journal entries | Preserve provenance; annotate only where commands appear current |

The transitive audit reviewed 300 tracked path/sys.path references, 123 targeted
baseline imports, and every constructed path in `src/`, `flows/`, `scripts/`, and
tests. The move manifest remains correct, with these concrete transition hazards:

- Training's vendor/config/repo roots currently use `parents[2]`; all become wrong
  after moving two levels deeper. Shared training imports need explicit paths.
- `hf_baseline_import.py` computes its repo root with `parents[2]`, inserts sibling
  training dynamically inside `import_variant`, and is imported directly by both
  evaluation and flows.
- Evaluation inserts current sibling registry/training paths. `live_verify.py` and
  `paper_eval_mlflow.py` span all three milestone-A slices, so each intermediate
  move must update them immediately.
- `flows/eval_baseline.py` inserts both evaluation and registry; retraining keeps a
  valid dependency on shared `promote_model.py` after baseline import moves.
- Shell scripts use cwd-relative `src/training`; Python wrappers use repo-root-based
  paths. Preserve wrapper cwd semantics and validate direct execution separately.
- The Colab notebook imports `colab_bootstrap` through a direct path and executes
  `src/training/train.py`; its traceback text is historical output and must not be
  mistaken for a live command.
- Vendor seams all use depth-sensitive parent traversal. Preserve their unique
  names and test them in fresh processes to avoid module-cache false positives.
- No tracked production module imports a generic helper through a stable package
  namespace; the flat path setup is therefore part of the migration surface.

Tracked consumer categories and concrete files are preserved in the temporary audit
outputs only for convenience; this classification is the durable inventory. Newly
discovered runtime dependencies must be added here if later characterization exposes
them.

## Approved compatibility and parity policies

- Preserve script entrypoints; change repo-owned direct paths and the BentoML
  string when their slice moves. Minimal public-doc path corrections are allowed.
- No blanket old-path shims. Any proven exception requires a consumer, named owner,
  regression test, removal condition, and follow-up task. None approved yet.
- BS-02 must freeze expected outputs before relocation, comparing within the same
  environment/device/seed/input, not across Python/torch versions. Require exact
  deterministic results and pre-established tolerances where necessary.
- Tests use tiny real fixtures and disposable local tracking stores, not production
  model promotion or deployment.

## Artifact matrix — initial discovery

| Category | Current evidence | Remaining validation |
| --- | --- | --- |
| Project-trained Lightning checkpoint | `experiments/starcop_run/2026-08-23_02-05/final_checkpoint_model.ckpt`; SHA-256 `e2619ef61d070d4a097add9e1d25dacc1c2cf31077b0a7f7faa7ee2c8ba45a85` | Confirm producer/config metadata and supported load env; freeze BS-02 predictions before moving code |
| Project-trained MLflow model | Run ID `f16aa1f8330248df855ceea77eb1a281` accompanies the selected 2026-08-23 checkpoint, but no matching complete local MLflow artifact was found. Local `mlruns/` model payloads are only 2517 bytes and appear to be test fixtures. | Use the checkpoint for offline parity; loading the run's remote MLflow artifact is integration-blocked pending authorized read access |
| Imported pretrained STARCOP model | Code pins HF repo `isp-uv-es/starcop` at revision `b5fd9c0d1028321ab2d6791623e16e910fd45289` and checkpoint digests for `mag1c_only`/`mag1c_rgb`; local Varon is DVC-pinned by `models/starcop_baseline.dvc` | Files are unavailable locally: `dvc status` reports `models/starcop_baseline` not in cache. Use pinned identities for tests; actual load parity is integration-blocked until an artifact is obtained through the normal verified path |
| Currently deployed model, if different | Existing internal deployment record identifies `starcop-baseline-mag1c-rgb` version 1 at Staging | Artifact hash and currently running immutable container digest were not queried; obtain through authorized read-only access before final integration acceptance |

Inspected local model `m-4df587306ca74385849850871f45d896` records Python 3.12.14,
torch 2.12.1+cu130, MLflow 3.14.0, `code: null`, and only 2545 model bytes. The
largest discovered local `model.pth` payload is 2517 bytes; these do not establish
that a real STARCOP deployment artifact is available. Artifact access gaps block
integration acceptance, not independent offline work. No checkpoint was
deserialized during BS-01.

## CD audit — merge gate required

Confirmed from checked-in configuration:

1. `.github/workflows/release.yml` runs on main pushes and may publish semantic
   release tags.
2. `.github/workflows/cd.yml` accepts both manual dispatch and `v*.*.*` tag pushes.
3. Its deploy job has no GitHub environment approval gate. Tag handling checks
   serving/deploy changes, then can build, push `:latest`, and call Coolify.
4. BS-03 changes `src/serving/`, which is already a deployment-relevant path.
   Refactor commits alone should not release, but a later release can include
   these changes in its tag diff. Commit type is not an authorization gate.
5. Compose defaults to mutable `:latest` with `pull_policy: always`; this is not
   an immutable last-known-good rollback reference.

Implemented the repository-owned gate in `.github/workflows/cd.yml`: removed the
release-tag trigger, leaving `workflow_dispatch` as the only event; simplified the
obsolete release-diff step to confirm explicit dispatch while retaining its output
for existing guards. The Prefect retraining flow intentionally invokes this explicit
dispatch after its promotion path; ordinary releases and merges cannot invoke CD.
Updated the deployment runbook to describe explicit dispatch and `latest` accurately.

Validation performed without dispatching or deploying:

- Static assertion found no `push`/`tags` trigger in the workflow.
- `rhysd/actionlint:latest` completed with exit 0 against `cd.yml` (image digest
  pulled locally: `sha256:b1934ee5f1c509618f2508e6eb47ee0d3520686341fec936f3b79331f9315667`).
- The full Bats suite already passed after Docker access was restored.

External integration remains blocked on recording the currently running immutable
image digest and deployed model artifact identity through authorized read-only
access. Compose still defaults to mutable `latest`; that is unchanged runtime
behavior, not an adequate rollback identifier. No GitHub workflow was run and no
remote repository, package, Coolify, registry, or service state was changed.

## BS-02 characterization baseline

### Migration-contract tests

Added `src/training/__tests__/test_baseline_migration_contract.py` before moving
production code. It runs each supported entrypoint's `--help` from a temporary
working directory and verifies depth-sensitive repo/config/vendor paths in a fresh
Python process. The stable wrapper/direct entrypoints pinned are:

- `src/training/train.py` (Hydra help);
- `scripts/import_starcop_hf_baseline.py`;
- `scripts/run_starcop_baseline_evaluation.py`;
- `scripts/run_live_verify.py`.

The path contract covers training's repo/overlay/vendor config roots, baseline
import's repo root, and the distinct evaluation/serving vendor seams. The test is
expected to fail after a mechanical move until new roots and callers are correctly
wired. Existing focused tests already pin metric arithmetic, mask digests,
input assembly, inference sigmoid/threshold behavior, response lists, model identity,
and MLflow model-loader output equivalence; BS-02 does not duplicate them.

Validation after adding the five tests:

| Command | Result |
| --- | --- |
| Research focused migration contract | 5 passed |
| Baseline focused migration contract | 5 passed |
| `make test-baseline` | 304 collected; 302 passed, same 2 skips, 26 warnings |
| `make test-research` | 476 collected; 476 passed, 130 warnings |
| Ruff check/format on the new test file | Passed; one file already formatted |
| `make docstring-coverage` | Passed at the unchanged 87.9% |

The five-test increase in each environment is intentional. Research's existing
MLflow import integration tests execute rather than skip; baseline retains exactly
the two BS-01 skips. No test was disabled or newly skipped.

### Frozen configuration identity

- Vendor config SHA-256:
  `1899e005a7dcbc6d07fe5e3b5d78fe2818d8cae32419f652a7901f62a9ddab4b`.
- Overlay SHA-256:
  `3391da073f5e6ed11ca026675ff8123e1b52f8e8394b01f8538be49bd6913818`.
- Selected run's composed Hydra config SHA-256:
  `1e51b152dd5867135c079a43c2e1ac83e52866cb4bb65bfd71d8f24da80b3cee`.
- Selected run ID: `f16aa1f8330248df855ceea77eb1a281`; model is
  `unet_semseg`/MobileNetV2, four ordered Mag1c+RGB channels, segmentation output,
  five epochs, desktop GPU profile, `starcop_mini`.

### Frozen local-checkpoint output

Loaded the selected 80 MB checkpoint read-only through
`hf_baseline_import.load_model` in each environment, put the model in eval mode,
and inferred on CPU with:

```python
torch.linspace(0.0, 80.0, steps=4 * 32 * 32, dtype=torch.float32).reshape(1, 4, 32, 32)
```

The output is contiguous float32 logits with shape `[1, 1, 32, 32]`. Each complete
procedure was repeated and produced the same byte digest within its environment:

| Environment | Torch | Logit SHA-256 | Sum | Min | Max |
| --- | --- | --- | ---: | ---: | ---: |
| Research | 2.12.1+cu130 | `3443c27dc3643ccce944bb87b02e934da0cd7a785138335c84c602aaeae5ab67` | -1492.0 | -5.168102741241455 | 0.6370056867599487 |
| Baseline | 1.13.1+cu117 | `74cd23ce84c2a21f3f8ccfea5f7d713f71ede12b72036708ee84a1aae80c84ae` | -1492.00048828125 | -5.1681060791015625 | 0.6370047926902771 |

Acceptance after relocation is exact digest equality **within the same environment**.
The differing cross-environment digests are expected and are not compared. If a
platform operation later proves nondeterministic, do not weaken this retrospectively;
investigate first and document a separately approved tolerance.

### Safe integration procedures

- Artifact loading uses the local checkpoint and no tracking credentials or writes.
- MLflow integration tests use disposable local stores, as the existing suite does.
- Local serving validation remains the documented `bentoml serve` + predict/health/
  metrics procedure against an explicitly selected artifact; it must not use or
  mutate Production. It is deferred until the serving slice because remote/deployed
  artifacts are not currently available for read-only parity.
- No full training, registry promotion, workflow dispatch, external API request, or
  production service request was made during BS-02.

## BS-03 shared band-statistics extraction

Followed RED → GREEN before moving any serving module:

1. Added `test_band_statistics.py` and changed drift tests to import the intended
   shared module. Focused collection failed with `ModuleNotFoundError: No module
   named 'band_statistics'` (three collection errors), confirming RED.
2. Added `src/serving/band_statistics.py` with the `BandStats` named tuple; changed
   `drift.py` to depend on it and `band_baseline.py` to re-export the shared type.
3. Added a clean-process assertion that importing shared drift does not load
   `band_baseline`, plus a compatibility test that unpickles the original
   `band_baseline.BandStats` global path into the shared type.

Behavior remains unchanged: named-tuple fields/tuple semantics, STARCOP baseline
values, rolling statistics, and KL calculations are covered. Repo search found no
serialized `BandStats` use outside serving, but the old attribute remains available
for defensive pickle/import compatibility.

Validation:

| Command/check | Result |
| --- | --- |
| Focused statistics/drift/baseline tests after GREEN | 21 passed |
| `make test-research` | 480 passed, 130 warnings |
| `make coverage-research` | 480 passed; displayed total remains 79% (1654 statements, 347 misses, 350 branches, 21 partial branches) |
| Ruff check and format on all six changed/new Python files | Passed after fixing one import-order finding |
| `make docstring-coverage` | Passed at 88.4% |
| `.venv/bin/bentoml build` | Built `methane_detection_service:eyhfrtfk6saykimq` locally |
| Bento source inspection | Bundle contains `band_statistics.py`, `drift.py`, and `band_baseline.py` |
| `git diff -- vendor/starcop` | Empty |

The Bento build wrote only to the local Bento store and did not containerize, push,
serve, access MLflow, or deploy. Its dependency resolver reported BentoML 1.4.39 and
a locked image environment; no project dependency/lock file changed. Containerized
serving remains a milestone-B integration check. BS-03 integration passes for its
stated source-bundle boundary; the broader remote artifact/rollback blockers remain
tracked under BS-01/BS-02 and final acceptance.

## BS-04 baseline training relocation

Followed RED → GREEN for the new entrypoint before relocating production code:

1. Changed the migration contract to require
   `src/baselines/starcop/training/train.py`; the focused run failed because the
   file did not exist, while the four unchanged compatibility checks still passed.
2. Moved all 15 baseline-owned training modules and their 13 existing owned test
   modules under `src/baselines/starcop/training/`. Moved the migration contract
   with them and split the test path setup so baseline tests can also import the
   three shared modules retained under `src/training/`.
3. Made the relocated entrypoint and vendor seam resolve the repository, Hydra
   overlay, shared training directory, and unmodified vendor checkout from their
   new depth. Updated evaluation's transitional imports to use baseline
   `validation_metrics.py` and shared `mlflow_utils.py` independently.
4. Preserved the Mac/Desktop wrapper commands while redirecting their internal
   paths. Updated Bats assertions, the active Colab cells, Makefile, test workflow,
   Codecov flags, active runbook/path references, and AGENTS. Added
   `src/training/README.md` to define the remaining shared ownership.

Validation:

| Command/check | Result |
| --- | --- |
| RED migration contract | 1 expected failure: new training entrypoint absent; 4 passed |
| Relocated training + shared training tests, research | 105 passed, 43 warnings |
| Relocated training + shared training tests, baseline | 105 passed, 7 warnings |
| `make test-baseline` | 304 collected; 302 passed, same 2 skips, 26 warnings |
| `make test-research` | 480 passed, 130 warnings |
| `make test-scripts` | All 31 Bats tests passed |
| `make coverage` | 302 passed, same 2 skips; 87%, 359 statements; both shared and baseline training sources present |
| `make coverage-research` | 480 passed; 79%, 1657 statements; both shared and baseline training sources present |
| BS-02 fixed-input checkpoint, research | Exact logit digest `3443c27dc3643ccce944bb87b02e934da0cd7a785138335c84c602aaeae5ab67` |
| BS-02 fixed-input checkpoint, baseline | Exact logit digest `74cd23ce84c2a21f3f8ccfea5f7d713f71ede12b72036708ee84a1aae80c84ae` |
| Ruff check/format on all changed/new Python files | Passed; 49 files already formatted |
| `make docstring-coverage` | Passed at 88.4% |
| Notebook JSON parse, fresh-cwd CLI/config checks | Passed |
| `make docs-build` | Passed with strict mode |
| `rhysd/actionlint` on `tests.yml` | Passed |
| `git diff --check`; `git diff -- vendor/starcop` | Passed; vendor diff empty |

The first parallel coverage invocation allowed the two coverage processes to share
coverage.py's default data file and temporarily contaminated the research terminal
report with baseline-only sources. `make coverage-research` was rerun alone; the
result recorded above and `coverage-research.xml` have the intended source roots.
No test was disabled, and the baseline skip set remains unchanged.

No old-path compatibility shim was added: repo-owned wrappers keep their stable
names and now invoke the new path. The old `src/training/train.py` surface is absent.
The historical traceback retained in the Colab notebook is output provenance, not
an active command. No dataset, lockfile, registry identity, artifact, remote
service, or deployment was changed.

## BS-05 baseline checkpoint importer relocation

Followed RED → GREEN for the new importer location:

1. Changed the clean-process migration contract to require
   `src/baselines/starcop/registry/hf_baseline_import.py`; the focused test failed
   because Python still resolved the old shared-registry location.
2. Moved `hf_baseline_import.py`, `_vendor_starcop_baseline.py`, and their test
   module under `src/baselines/starcop/registry/`. Added a narrow test conftest for
   baseline registry, shared registry, and shared training paths.
3. Updated repository/vendor root resolution and made the importer resolve shared
   MLflow registry and training utilities without an old-path alias.
4. Redirected the stable `scripts/import_starcop_hf_baseline.py` wrapper and updated
   still-unmoved evaluation modules, `flows/eval_baseline.py`, fixtures, Makefile,
   Codecov, AGENTS, and active documentation references.

Validation:

| Command/check | Result |
| --- | --- |
| RED clean-process importer contract | Expected assertion failure: importer still resolved from old location |
| Migration contract after GREEN | 5 passed |
| Shared + baseline registry tests, research | 57 passed, 16 warnings |
| Shared + baseline registry tests, baseline | 55 passed, same 2 skips, 20 warnings |
| `make test-baseline` | 304 collected; 302 passed, same 2 skips, 26 warnings |
| `make test-research` | 480 passed, 130 warnings |
| `make coverage` | 302 passed, same 2 skips; unchanged 87%/359 statements; baseline registry intentionally remains outside this flag |
| `make coverage-research` | 480 passed; 79%/1667 statements; shared and baseline registry sources present |
| Local MLflow import tests | Both real sqlite-backed importer tests passed in research env; no production registry writes |
| BS-02 fixed-input checkpoint, research | Exact logit digest `3443c27dc3643ccce944bb87b02e934da0cd7a785138335c84c602aaeae5ab67` |
| BS-02 fixed-input checkpoint, baseline | Exact logit digest `74cd23ce84c2a21f3f8ccfea5f7d713f71ede12b72036708ee84a1aae80c84ae` |
| Shared-registry clean-process import | Passed without loading importer or vendor seam |
| Ruff check/format on all changed/new Python files | Passed; 54 files already formatted |
| `make docstring-coverage` | Passed at 88.4% |
| `git diff --check`; vendor worktree/diff | Passed; vendor unchanged |

The research coverage flag now owns the relocated importer, while baseline coverage
retains its frozen BS-01 scope: baseline tests execute importer behavior but do not
measure registry sources. The wrapper name and cwd behavior remain stable; the old
`src/registry/hf_baseline_import.py` and vendor-seam surfaces are absent rather than
shimmed. Existing model names, pinned revisions/digests, tags, artifact names,
stages, and registry policy are unchanged.

No remote artifact was downloaded, no production tracking or registry operation
was performed, and no model was promoted. Remote artifact identity and immutable
rollback digest gaps remain final integration blockers rather than BS-05 offline
code blockers.

## BS-06 baseline evaluation relocation

Followed RED → GREEN for the new evaluation seam before moving the slice:

1. Changed the clean-process migration contract to require
   `src/baselines/starcop/evaluation/_vendor_starcop_evaluation.py`; the focused
   test failed with `ModuleNotFoundError` because the new directory did not exist.
2. Moved all seven evaluation modules and six test modules, plus their narrow
   conftest, under `src/baselines/starcop/evaluation/`. Removed the old directory
   and caches; no compatibility shim was approved or added.
3. Corrected the evaluation vendor/repository depth and explicit paths to shared
   training/registry and relocated baseline training/registry modules.
4. Kept the stable evaluation and live-verification wrapper names while redirecting
   their module paths. Updated the Prefect evaluation flow, DVC comment, Makefile,
   Codecov ownership, test workflow, AGENTS seam/ownership guidance, and active
   paper-reference path. Historical benchmark-plan paths remain historical evidence.

Validation:

| Command/check | Result |
| --- | --- |
| RED clean-process evaluation seam contract | Expected failure: new evaluation seam absent |
| Focused evaluation + flow + migration tests, research | 181 passed, 48 warnings |
| Focused evaluation + migration tests, baseline | 130 passed, 11 warnings |
| `make test-baseline` | 304 collected; 302 passed, same 2 skips, 26 warnings |
| `make test-research` | 480 passed, 130 warnings |
| `make coverage` | 302 passed, same 2 skips; unchanged 87%/359 statements; evaluation intentionally remains outside the baseline flag |
| `make coverage-research` | 480 passed; 79%/1667 statements; relocated evaluation sources present |
| `make test-scripts` | All 31 Bats tests passed |
| Evaluation/live wrapper help from repo and external cwd | Passed; stable wrappers expose `--device` and `--base-url` |
| BS-02 fixed-input checkpoint, research | Exact logit digest `3443c27dc3643ccce944bb87b02e934da0cd7a785138335c84c602aaeae5ab67` |
| BS-02 fixed-input checkpoint, baseline | Exact logit digest `74cd23ce84c2a21f3f8ccfea5f7d713f71ede12b72036708ee84a1aae80c84ae` |
| Paper metrics/examples/live comparison fixtures | Passed in both focused suites; definitions and expected values unchanged |
| Ruff check/format on all changed/new Python files | Passed; 64 files already formatted |
| `make docstring-coverage` | Passed at 88.4% |
| Stale live-path search; `git diff --check`; vendor diff | Passed; only classified historical plan references retain `src/evaluation/`; vendor unchanged |

The live-verification implementation still targets only an explicitly supplied
local Bento service; no live or production endpoint was contacted. The CD workflow
remains explicit-dispatch-only, so this migration does not authorize deployment.
No data, lockfile, production artifact, MLflow identity, registry stage, remote
service, or deployment was changed.

## BS-07 baseline serving relocation

Followed RED → GREEN for the serving seam before moving the runtime slice:

1. Changed the clean-process migration contract to require
   `src/baselines/starcop/serving/_vendor_starcop_serving.py`; the focused test
   failed because it still resolved the old seam.
2. Moved `service.py`, `inference.py`, `model_loader.py`, `band_baseline.py`, the
   serving vendor seam, and their three test modules under
   `src/baselines/starcop/serving/`. Kept only shared drift mathematics,
   `BandStats`, and their tests under `src/serving/`; removed the obsolete
   `.gitkeep` and added a narrow baseline-serving conftest.
3. Corrected vendor, shared-serving, and shared-registry path resolution. Preserved
   the bare `band_baseline.BandStats` pickle surface inside the relocated runtime.
4. Updated the Bento service string, source bundle closure, Prefect local-serving
   command, live-verification example, Makefile, Codecov, CD comments, deployment
   and monitoring references, requirements comments, AGENTS, and active status/
   architecture links. CD remains manual `workflow_dispatch` only.

Validation:

| Command/check | Result |
| --- | --- |
| RED clean-process serving seam contract | Expected assertion failure: serving seam resolved from old path |
| Focused shared/baseline serving + flow + migration tests | 109 passed, 53 warnings |
| Clean-process relocated service import | Passed; shared serving path resolved explicitly |
| `make test-baseline` | 304 collected; 302 passed, same 2 skips, 26 warnings |
| `make test-research` | 480 passed, 130 warnings |
| `make coverage` | 302 passed, same 2 skips; unchanged 87%/359 statements |
| `make coverage-research` | 480 passed; 79%/1674 statements; shared and baseline serving sources present |
| `make test-scripts` | All 31 Bats tests passed |
| Local source `bentoml serve` with disposable sqlite MLflow model | `/readyz`, `POST /health`, `POST /predict`, and `GET /metrics` passed; response and custom metric contracts preserved |
| `.venv/bin/bentoml build` | Built `methane_detection_service:y2g4mmfk7wpentkt`; bundle contains baseline serving, shared drift/statistics, shared registry, and vendor source |
| Bento containerization | Built local `linux/amd64` image `methane-detection:bs07-local` |
| Local container smoke with disposable sqlite MLflow model | `/readyz`, health, predict, and metrics passed after mounting the writable local tracking store |
| Real project checkpoint through relocated MLflow loader | Exact research digest `3443c27dc3643ccce944bb87b02e934da0cd7a785138335c84c602aaeae5ab67` after temporary log/register/load round trip |
| Ruff check/format on all changed/new Python files | Passed; 69 files already formatted |
| `make docstring-coverage` | Passed at 88.4% |
| Actionlint + Docker Compose config validation | `tests.yml`, manual-only `cd.yml`, and `deploy/bentoml/docker-compose.yml` passed |

The first container-smoke attempt mounted the disposable sqlite store read-only and
failed because sqlite could not open/write its database; this was a test-fixture
permission issue, not a bundled-source failure. Repeating with a writable mounted
store passed. The local Bento and 11.2 GB Docker image were not pushed or deployed and were
removed after validation.

Remote/deployed artifact compatibility remains integration-blocked: the selected
project checkpoint passed locally, but the deployed MLflow artifact and immutable
running-image digest still require authorized read-only access. This does not block
BS-07's independent local code and packaging acceptance, but still blocks BS-09 and
deployment authorization.

## BS-08 architecture guide and compatibility audit

Completed the final ownership documentation without changing runtime behavior:

1. Added `src/README.md` as the source-tree entrypoint and
   `src/baselines/starcop/README.md` as the baseline contract/command guide.
   Expanded the shared training README so it states dependency direction and both
   supported training environments.
2. Reconciled `README.md`, `CONTRIBUTING.md`, and `AGENTS.md` with the actual tree.
   In particular, `CONTRIBUTING.md` no longer claims application deployment happens
   automatically on merge: CD remains explicit `workflow_dispatch` only.
3. Linked the current ownership map from the hardware-model hypotheses and added
   migration notices to active internal journals whose old paths are retained as
   execution-time provenance rather than current commands.
4. Removed stale pre-BS-07 validation commentary from `bentofile.yaml`; its service
   string and dependency closure remain unchanged from the locally validated BS-07
   configuration.

The tracked/current-worktree stale-path audit found no live code, workflow, shell
script, YAML/TOML configuration, or unclassified command using a removed baseline
path. Remaining old-path strings are intentional:

- `mlops-methane-detection-plan.md`, `internal-docs/plan.md`, and
  `internal-docs/plans/track-a-paper-benchmark-reproduction-plan.md`: historical
  implementation records; the latter two now carry an explicit current-path note.
- this migration plan and evidence record: before/after migration provenance.
- `model-hypotheses-scaffolding-plan.md`: an untracked scaffold-planning record
  explicitly excluded from this migration change set.
- `notebooks/train_colab.ipynb`: captured traceback output only; executable cells
  use the relocated training path.
- `docs-reorganization-plan.md`: an unrelated pre-migration planning draft,
  intentionally left untouched.

No approved compatibility shim exists, and no shim was added or removed. Stable
`scripts/` entrypoints remain the compatibility surface; removed Python source
paths remain absent.

Validation:

| Command/check | Result |
| --- | --- |
| Local Markdown-link/path checker over the architecture and reconciled guides | Passed for 10 files |
| Removed-source existence checks and stale live-consumer search | Passed; old baseline-owned files/directories absent and no unclassified live references |
| `make docs-build` | Passed in strict mode; only the existing MkDocs nav notices were reported |
| `git diff --check` | Passed |
| Vendor worktree/diff | Clean |
| Python tests and quality gates | Not applicable to BS-08: no Python behavior or source was changed |

## BS-09 combined acceptance and review handoff

Ran the complete locally available checklist on the combined BS-01–BS-08 tree.
No environment was changed, no production endpoint or registry was contacted, and
no workflow was dispatched.

### Test and quality comparison

| Check | Final result | Comparison with frozen state |
| --- | --- | --- |
| `make test-baseline` | 304 collected; 302 passed, 2 skipped, 26 warnings | Exactly the original 299 tests plus BS-02's five migration contracts; original two importer skips preserved |
| `make test-research` | 480 passed, 130 warnings | Original 471 plus five migration contracts and four shared-statistics tests; no skips |
| `make coverage` | 302 passed, 2 skipped; 87%, 359 statements | Percentage and statement count unchanged; source roots contain shared and relocated baseline training only |
| `make coverage-research` | 480 passed; 79%, 1674 statements | Matches BS-07; XML contains all shared and relocated baseline source roots |
| `make test-scripts` | 31 passed | Stable Mac/Desktop/Prefect wrappers still target relocated entrypoints |
| Ruff check and format | Passed for all 69 changed/new Python files | Diff-scoped per CI policy |
| `make docstring-coverage` | Passed at 88.4% | Above the 80% gate |
| `make docs-build` | Passed in strict mode | Only existing MkDocs nav notices |

Clean external-working-directory CLI smokes passed for the training entrypoint in
both environments and for the import, evaluation, and live-verification wrappers.
The vendor config retains SHA-256
`1899e005a7dcbc6d07fe5e3b5d78fe2818d8cae32419f652a7901f62a9ddab4b`.
The overlay hash changed because its leading comment names the relocated training
path; `git diff` confirms no configuration value changed.

### Numerical and package acceptance

The selected project checkpoint still has SHA-256
`e2619ef61d070d4a097add9e1d25dacc1c2cf31077b0a7f7faa7ee2c8ba45a85`.
Fixed-input inference reproduced the exact BS-02 byte digests:

- research: `3443c27dc3643ccce944bb87b02e934da0cd7a785138335c84c602aaeae5ab67`;
- baseline: `74cd23ce84c2a21f3f8ccfea5f7d713f71ede12b72036708ee84a1aae80c84ae`.

Built Bento `methane_detection_service:wqxzeuflask3ygii`, inspected its source
closure, and confirmed the relocated baseline service, shared drift/statistics,
shared registry module, and vendor source were included. Containerization produced
a local `linux/amd64` image with digest
`sha256:faf71d3a9c7eb0621bf0024c77fbc61963c69b63d57f8f9050149caa20e2f02f`.
Both source `bentoml serve` and the container passed `/readyz`, `POST /health`,
`POST /predict`, and `GET /metrics` against separate disposable sqlite-backed
MLflow stores and tiny real PyTorch models. The temporary Bento, image, stores, and
containers were removed afterward; the local digest is validation evidence, not a
deployment or rollback reference.

Actionlint passed for `tests.yml` and manual-only `cd.yml`; Docker Compose config
resolution passed for `deploy/bentoml/docker-compose.yml`. The stale-runtime-path
search was empty. `git diff --check` passed, old implementation paths are absent,
and vendor, data, DVC lock, dependency locks, and model identities are unchanged.

### Remaining integration blockers

Final integration acceptance is **blocked**, not failed. Authorized external
read-only access is still required to:

1. identify and load the project-trained remote MLflow artifact associated with run
   `f16aa1f8330248df855ceea77eb1a281`;
2. load the pinned imported pretrained artifacts that are absent from the local DVC
   cache and compare them with BS-02 behavior;
3. identify and test the artifact currently deployed as
   `starcop-baseline-mag1c-rgb` version 1 at Staging, if it differs; and
4. record the currently running immutable image digest as the last-known-good
   rollback target.

The checked-in launch configuration is `deploy/bentoml/docker-compose.yml`, but it
defaults to mutable `latest`; it cannot substitute for item 4. No remote read was
attempted without authorization, and no promotion, dispatch, push, restart, or
deployment occurred. These blockers prevent the plan's full definition of done and
any deployment authorization even though all local code acceptance passed.

### Proposed commit groups (not executed)

1. `fix(cd): require explicit deployment dispatch` — the manual CD gate and its
   deployment guidance, separated because it changes release/deployment safety.
2. `refactor(starcop): separate baseline adapters from shared infrastructure` —
   the statistics extraction, baseline relocations, tests, callers, CI/coverage,
   Bento closure, and command wiring kept atomic so every committed tree remains
   runnable.
3. `docs(architecture): document STARCOP baseline ownership` — source maps,
   contributor guidance, current runbooks, hypotheses references, migration plan,
   and validation evidence.

Use explicit path/patch staging because several shared files contain changes from
multiple slices; never use `git add -A`. The unrelated untracked
`docs-reorganization-plan.md` and `mlops-methane-detection-plan.md` remain excluded.
No commit or push was requested or performed.

## Final state

BS-01 through BS-09 code validation and all available local integration checks have
passed. Final integration acceptance and deployment remain blocked on the four
external read-only checks above. The next action requires explicit authorization
and access to gather those identities; it is not additional local refactor work.
