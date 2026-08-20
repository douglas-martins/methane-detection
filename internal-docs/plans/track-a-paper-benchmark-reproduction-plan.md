# Track A — Reproducing STARCOP's Paper Numbers as a Standing Benchmark

> **Status:** plan drafted, not yet executed
> **Created:** 2026-08-19
> **Target project:** `methane-detection`
> **Scope:** evaluation-only reproduction of the STARCOP paper's Table 1 (MultiSTARCOP) and Table 2 (HyperSTARCOP) numbers, using the paper's own released checkpoints and full held-out test set — wired through MLflow, Prefect, and BentoML, and published as a public, image-led page at `docs/results.md`.
> **Out of scope:** from-scratch retraining reproduction ("Track B"), EMIT zero-shot Table 3 reproduction, wiring MultiSTARCOP into the *production* BentoML deployment.

## Context

The project already ran a smoke-test inference check on a 9-scene demo subset
(`notebooks/starcop_baseline_validation.py` → `docs/baseline_metrics.md`), but that
result isn't comparable to the paper's published Table 1/2 numbers — different metric
aggregation (pooled pixels vs. per-tile stratified) and a tiny, non-representative
scene set.

The full paper-matching test set is already sitting in the repo, unused for this
purpose: `data/starcop_raw/test.csv` (342 rows) — verified to exactly match the
paper's Figure 4 test-set table (166 plume rows split 57 strong/109 weak by the
`qplume` kg/h column, 176 no-plume rows). The paper's own released checkpoints are
also already local, and two of three are already in the MLflow registry.

This plan turns that into a real, defensible benchmark: evaluate the paper's own
checkpoints against the paper's own full test set, using the paper's own evaluation
code (`vendor/starcop/starcop/validation.py::run_validation`) — with one critical fix
(see below) — then wire the result through MLflow, Prefect, and BentoML (the stack
already deployed for this project), and publish it as an accessible, image-led public
page at `docs/results.md`.

**The one bug this plan exists to work around**: `run_validation` computes its own
`difficulty` column from label pixel count (`>1000px = "easy"`), silently shadowing
`test.csv`'s real `difficulty` column, which is actually keyed off `qplume` (kg/h) —
the paper's real strong/weak definition. Confirmed by direct comparison: among
plume rows, `test.csv`'s `difficulty=="easy"` (57 rows) exactly equals `qplume>=1000`
(57), and `"hard"` (109 plume rows) exactly equals `qplume<1000` (109). Any
reproduction that trusts `run_validation`'s internal grouping silently reports the
wrong split. The fix is composition-only (join its per-scene output back to the real
CSV column afterward) — `vendor/starcop/` is never edited, per this project's
established rule ([[feedback-vendor-starcop-composition-only]]).

## Recommended Approach

### Phase 0 — Settle Environment A vs. B (do this first, empirically)

- [ ] Not started

`train.py` already calls this exact same unmodified `run_validation` function today,
in **Environment B** (`.venv`, Python 3.12, MPS-capable), via the
`src/training/_vendor_starcop_training.py` shim — this is proven working code, not
speculative. Environment A (`vendor/starcop/.venv`) is CPU-only (torch 1.13.1 has no
MPS support) and was only used before because the mini-set script never needed
Environment B's newer tooling. For 342 scenes at up to 512×512, Environment B/MPS is
likely meaningfully faster.

Run a `--limit 5` dry pass of checkpoint-loading + one `run_validation` batch under
Environment B first. If clean, do the full run there. If it hits a real snag, fall
back to Environment A (already proven end-to-end by the mini-set script). Either way,
new code should avoid environment-specific imports outside the one vendor shim, so
this stays a runtime choice, not an architecture decision.

**A `--limit`-ed dry pass is diagnostic only.** It must never log to the
`starcop-paper-eval` MLflow experiment (Phase 3's permanent benchmark record) and
must never be combined with `--emit-docs-assets` (Phase 6) — see the CLI validation
in Phase 1. This dry pass exists to de-risk the Environment A/B choice, not to
produce a reportable number.

### Phase 1 — Evaluation core (`src/evaluation/`, new package)

- [ ] Not started

New sibling package to `src/training/`/`src/registry/`/`src/serving/`, same
conventions (flat files, `__tests__/`, one `_vendor_starcop_evaluation.py` shim per
the project's existing vendor-import-seam pattern).

- **`dataset_wiring.py`** — builds a test-only dataloader directly against
  `starcop.data.dataset.STARCOPDataset`, *not* via `Permian2019DataModule.prepare_data()`
  (which unconditionally tiles the full multi-GB training set first — wasteful for an
  eval-only pass). Replicates `Permian2019DataModule.load_dataframe`'s few relevant
  lines (`vendor/starcop/starcop/data/datamodule.py:104`): read `test.csv`, rebuild
  `folder` as `root_folder/id`, rebuild the `window` from the CSV's
  `window_col_off`/`window_row_off`/`window_width`/`window_height` columns (same
  pattern `src/training/starcop_datamodule.py::_load_dataframe` already uses). For
  MultiSTARCOP, calls `feature_extration.extract_features(...)` first (its Varon
  ratio bands are computed features, unlike Hyper's raw `mag1c`+RGB bands).
- **`paper_metrics.py`** (pure, unit-tested) — the actual bug fix. Joins
  `run_validation`'s returned per-scene `out_data` back to `test.csv`'s `qplume`
  column by scene `id`; buckets into `strong` (`qplume>=1000`), `weak` (`has_plume`
  and `qplume<1000`), `no_plume`; sums `TP/FP/TN/FN` per bucket and applies
  `starcop.metrics.{precision,recall,f1score,iou}` the same way `run_validation`
  does internally — just off the corrected bucket. Also derives AUPRC from
  `run_validation`'s `metrics["thresholded"]` list (not directly returned by
  vendor code) — **the AUPRC convention must be pinned before implementation, not
  left to a library default**:
  - `metrics["thresholded"]` is built high-to-low threshold
    (`validation.py:41-42`); that ordering is not guaranteed to be strictly
    ascending in recall once TP/FP/FN come from summed per-bucket confusion
    matrices, so the derived (precision, recall) points must be explicitly
    re-sorted by ascending recall before integration — never just reversed.
  - Pick, explicitly, either (a) non-interpolated average precision (step
    function: sum `(recall_n − recall_{n−1}) × precision_n` over the sorted
    points, no interpolation) or (b) trapezoidal integration
    (`sklearn.metrics.auc`-style linear interpolation) — the two can diverge
    meaningfully on a curve built from only ~13 threshold samples. Check the
    STARCOP paper's own evaluation methodology for which one it used (`vendor/starcop`
    itself never computes AUPRC, so this isn't settled by the vendor code); if the
    paper doesn't say, default to non-interpolated average precision (the more
    common convention), and record that assumption explicitly in
    `paper_comparison.md` (Phase 3) and `docs/results.md` (Phase 6) rather than
    presenting it as an unqualified match to the paper's number.
  - Document how missing recall endpoints (recall=0 / recall=1 — not guaranteed
    by the threshold list, e.g. the highest threshold can yield zero predicted
    positives) are handled by the chosen method, rather than leaving it to
    whatever the integration code does by accident.
  - Add a synthetic `metrics["thresholded"]`-shaped fixture with a deliberately
    non-monotonic precision-recall curve whose expected AUPRC differs between the
    non-interpolated and trapezoidal conventions, so an accidental method swap
    fails a unit test instead of only showing up as an unexplained drift against
    the paper's published number.

  **The join is validated, not assumed 1:1**: asserts `id` is unique
  on both `out_data` and `test.csv` before joining, that the two `id` sets are
  exactly equal (no extra/missing scenes either side), and uses a one-to-one merge
  (e.g. pandas `merge(..., validate="one_to_one")`) so a duplicate or mismatched id
  raises instead of silently duplicating or dropping rows; rejects any null
  `qplume` in the joined result before bucketing, since a silent NaN would
  otherwise fail the `>=1000` strong/weak split without erroring. Unit tests use a
  small hand-built synthetic `out_data`/`test.csv` pair with known expected F1
  values, plus the row-count-integrity assertion (guards against a silent
  id-mismatch dropping rows) and dedicated cases for duplicate ids, mismatched id
  sets, and null `qplume` values, each asserting the join raises.
- **`select_docs_examples.py`** (pure, unit-tested) — deterministic picks for the
  public doc: best strong-plume detection, a weak-plume true positive, a false
  negative if any exist, cleanest no-plume scene.
- **`run_baseline_eval.py`** (thin glue, Large-boundary, real-run validated) —
  loads a checkpoint via `src/registry/hf_baseline_import.py::load_model()` (reused,
  not duplicated), wires the dataloader, calls `run_validation(products_plot=None)`
  for the fast metrics-only pass over all 342 scenes, then a second tiny pass calling
  `starcop.plot.plot_batch(...)` (same call `notebooks/starcop_baseline_validation.py`
  already makes) only for the curated scene ids, to produce sample-mask PNGs — and,
  for those same curated scene ids, persists the offline predicted mask/confidence
  (or a digest) as the artifact Phase 5's `live_verify.py` diffs against.
  Asserts the known counts (342 total, 166/176 has_plume split, 57/109 strong/weak)
  as hard checks, not just eyeballed. Supports `--limit N` (Phase 0) and
  `--emit-docs-assets DIR` (Phase 6) — **but the CLI rejects the two together**:
  `--limit N` is a local smoke-test flag only, and `--emit-docs-assets` requires the
  full, unlimited 342-scene run (the hard-count assertions above gate it). Likewise,
  MLflow logging into `starcop-paper-eval` (Phase 3) is skipped whenever `--limit` is
  set, so a partial run can never land in the permanent benchmark record next to real
  paper-reproduction numbers.
- **CLI**: `scripts/run_starcop_baseline_evaluation.py` — thin argparse glue only,
  mirroring `scripts/import_starcop_hf_baseline.py`'s split.

### Phase 2 — Close the MultiSTARCOP registry gap

- [ ] Not started

`multistarcop_varon` isn't in the MLflow registry yet (`hf_baseline_import.py`'s
variant table only has the two Hyper keys), so Track A currently can't produce a
Table 1 (MultiSTARCOP) number without this. First check whether it's on the same
HuggingFace repo (`isp-uv-es/starcop`) the Hyper variants come from:

- **If yes**: pure extension of the existing pattern — add a `"varon"` entry to the
  variant/digest tables, one new CLI choice, matching unit tests.
- **If no** (it currently lives only at `models/starcop_baseline/multistarcop_varon/`,
  fetched via `gdown`, not HF): add a local-source import path that reuses every
  source-agnostic piece of `hf_baseline_import.py` unchanged (`load_model` doesn't
  even parse `config.yaml` — it reconstructs everything from the checkpoint's own
  embedded `hyper_parameters.settings`, so it's already source-format-agnostic) —
  only the "where does the checkpoint come from" step is new, pinned by a
  once-computed sha256 the same way the HF table already is.

Validated the same way the existing Hyper variants already are: registry lookup
returns a `Staging` version, `mlflow.pytorch.load_model(...)` round-trips cleanly.

### Phase 3 — MLflow: the permanent benchmark record

- [ ] Not started

New experiment `starcop-paper-eval` — deliberately separate from `starcop-baselines`
(which holds *import* runs that register a loadable model artifact) — so this
project's own future candidate models can log evaluation runs into the *same*
experiment for direct comparison against these paper-reproduction runs later.

One run per variant, e.g. `starcop-baseline-mag1c-rgb-paper-eval-<date>`:
- **Tags**: variant, registry model name + version, checkpoint sha256, DVC dataset
  version (reusing `src/training/dvc_dataset_version.py`, same pattern `train.py`
  already uses), `n_test_scenes=342` (sanity-checked, not just recorded),
  `paper_reference=true`, `resolved_device` (which environment/accelerator ran it),
  plus evaluation *software* revision — not just data/checkpoint revision — since
  `run_validation`'s vendor-pinned quirks are exactly what this plan works around:
  this repo's commit and dirty flag (`eval_code_git_sha`/`eval_code_dirty`, same
  pattern as `train.py`'s existing `dataset_dirty` tag — MLflow's automatic
  `mlflow.source.git.commit` tag covers the clean-tree case but not a dirty
  working tree, and doesn't cover the vendored submodule below), and
  `vendor_starcop_sha` (the `vendor/starcop` submodule commit actually checked
  out for this run, since it isn't captured by MLflow's own git auto-tagging and
  a future submodule bump could silently change `run_validation`'s behavior).
- **Metrics**: the corrected `strong_f1score`/`weak_f1score`/`no_plume_FPR`/`auprc`
  (the real Table 1/2 headline numbers — explicitly distinguished by name from
  `run_validation`'s own uncorrected `easy_*`/`hard_*` keys) plus the full aggregate
  metrics via the existing `src/training/validation_metrics.py::extract_scalar_metrics`
  helper.
- **Artifacts**: per-scene results CSV (with the corrected bucket column), the full
  `run_validation` metrics JSON, a generated `paper_comparison.md` (this run's numbers
  next to the paper's published Table 1/2 values — those are entered once, by hand,
  with an explicit table/page citation, never approximated), the curated
  sample-mask PNGs, and a frozen dependency manifest (`pip freeze` or equivalent) from
  whichever environment — A or B, per Phase 0's runtime choice — actually ran the
  evaluation, so a numeric drift between two runs of "the same" evaluation can be
  root-caused against an actual dependency diff instead of guessed at.

### Phase 4 — Prefect: make it a repeatable, auditable run

- [ ] Not started

New `flows/eval_baseline.py`, same `@task`/`@flow` shape and injectable-callable
testing pattern as `flows/retrain.py` (its `pull_dataset`/`notify`/failure-message
helpers are reused directly, not reinvented). Tasks, in order: pull the dataset (dvc
pull), ensure MultiSTARCOP is registered (idempotent check-then-import), run the
evaluation per variant (shelling to Phase 1's CLI, parsing the same
`MLFLOW_RUN_ID=` sentinel convention `retrain.py` already establishes), run the
BentoML live check (Phase 5, still non-fatal — a serving-side outage shouldn't block
regenerating the numbers), **then** emit docs assets (Phase 6), notify. The live
check now runs *before* asset emission, and its pass/fail result is threaded into
`--emit-docs-assets` so the generated page's "How this was verified" section
(Phase 6) states plainly whether the live spot-check passed, failed, or didn't run —
never silently claiming verification that didn't happen.

**The flow never passes `--limit`.** This is the sole path that logs to
`starcop-paper-eval` and emits docs assets, so it always shells out to Phase 1's CLI
for the full, unlimited run; `--limit` stays a manual, interactive-only flag for
Phase 0's dry pass and is never wired into `eval_baseline.py` or `prefect.yaml`.

Deployed to the existing `mac-mps` work pool via a new `prefect.yaml` entry —
**no schedule**: this is an on-demand "regenerate the numbers" operation triggered
manually whenever the vendor pin, dataset snapshot, or eval code changes, not a
recurring job.

### Phase 5 — BentoML: verify the live API agrees

- [ ] Not started

New `src/evaluation/live_verify.py`: for each curated scene, POST its input array to
the live `/predict` endpoint and compare the response against Phase 1's own offline
prediction for that scene, using the same JSON contract `test_inference.py` already
establishes (`predict_response`'s `{"mask": [[int]], "confidence": [[float]]}`) —
**not just a derived plume-pixel count**: two materially different masks can sum to
the same pixel count, so a count-only comparison can pass even when the live API
disagrees pixel-for-pixel with the offline run. Phase 1 persists the offline mask
(or a deterministic digest, e.g. a sha256 of the mask array) and confidence array
for the curated scenes only — not all 342, to avoid bloating the MLflow artifact
store — as an artifact `live_verify.py` diffs against: mask equality (or digest
match) pixel-for-pixel, and confidence values within an explicit numeric tolerance
(e.g. `atol=1e-5`, matching the sigmoid/threshold arithmetic `test_inference.py`
already exercises — not exact float equality, which would spuriously fail across
CPU/MPS numeric differences).

**Pin the served model to the exact checkpoint being verified, not whatever
`Staging` currently resolves to**: `service.py` resolves the `MODEL_NAME`/
`MODEL_STAGE` env vars (default `starcop-baseline-mag1c-rgb`/`Staging`) to whichever
registry version currently holds that stage alias — that can silently drift from
the registry version + checkpoint sha256 Phase 1 evaluated and Phase 3 tagged (a
later promotion could move `Staging` to a different version between the offline run
and the live check). `live_verify.py` must set `MODEL_NAME`/`MODEL_STAGE` for the
`bentoml serve` process it targets to the same registry version/checkpoint sha256
recorded in Phase 3's tags for that run, and assert the served model's identity
matches before comparing any predictions — otherwise a "passing" check could still
be silently comparing against the wrong model.

**Explicit scope limit**: only the two Hyper variants are servable live today (the
service loads exactly one `MODEL_NAME`/`MODEL_STAGE` at a time). Verifying
MultiSTARCOP live means a separate local `bentoml serve` with its own env vars, not
touching the production deployment.

### Phase 6 — Public documentation (`docs/results.md`)

- [ ] Not started

This is `docs/results.md`'s placeholder text come true — "Baseline model comparison
and sample prediction masks" is exactly this deliverable. Per the project's own
in-flight docs-reorg plan (`docs-reorganization-plan.md`), this page must read as
**"here's the model and proof it works,"** not an MLOps-platform showcase —
MLflow/BentoML/Prefect get one short, peripheral paragraph at the end, not top
billing. `docs/baseline_metrics.md`'s existing mini-set content is folded in as a
clearly-labeled historical subsection (not deleted — zero information loss), since
it's already explicit about being a non-representative smoke test.

**Page outline**:
1. Plain-language lead: what the model does, why plume detection matters.
2. **"How well does it work?"** — one comparison table, paper-reported vs.
   this-reproduction, per variant (Strong-plume F1, Weak-plume F1, AUPRC, FPR).
3. **"See it in action"** — the 3–4 curated sample-mask images with plain-language
   captions (a big obvious plume correctly caught; a small weak one; a clean
   no-plume scene correctly cleared), reusing the same TP/FP/FN/TN diff-map visual
   language already established by the existing sample images.
4. **"What do these numbers mean"** — a short non-technical glossary callout for
   precision/recall/F1/false-positive-rate.
5. **"How this was verified"** — brief, links to the MLflow run, states the actual
   outcome of the Phase 5 live spot-check for this run — passed, failed, or didn't
   run — rather than an unconditional "predictions were spot-checked" claim; a
   failed/skipped check is surfaced here as an explicit unverified status, not
   omitted. Deep infra explanation stays in `docs/pipeline/{training,serving}.md`,
   not here.
6. Footer: last-updated date + exact MLflow run ID(s), so staleness is checkable.

**No hand-typed reproduction numbers, no drift**: this applies to this project's own
measured outputs only — the paper-reported side of the comparison table is, by
design, a fixed reference value entered once by hand with an explicit table/page
citation (Phase 3), since it comes from an external publication and cannot be
"generated" by this project's pipeline; that hand-entry is correct, not a drift risk,
and stays that way unless the paper itself changes. `--emit-docs-assets DIR`
(Phase 1) writes the curated PNGs plus a generated Markdown table fragment covering
only this-reproduction's numbers, which `docs/results.md` pulls in via the
`mkdocs-include-markdown-plugin` (`{% include-markdown %}` — already used by
`docs/changelog.md`, so this is a proven mechanism here, not a new one). The
reproduction side of the page is mechanically tied to the last real run's output,
not a manually maintained copy; the paper-reported side is mechanically tied to
`paper_comparison.md`'s one-time hand entry (Phase 3), not re-typed per page edit.
Because `--emit-docs-assets` is rejected in combination with `--limit` (Phase 1) and
is only ever invoked by Phase 4's un-limited flow run, the assets `docs/results.md`
includes can never come from a partial/smoke-test pass.

## Ordering

Phase 0 (env decision) → Phase 2 Step 0 (HF-hosting check) can run in parallel with
Phase 1's pure modules (`paper_metrics.py`, `select_docs_examples.py`, unit tests
only, no dependency on anything else) → Phase 2's registry import → Phase 1's glue
(full run, all three variants) → sanity-check the corrected numbers against the
paper before trusting anything downstream → Phase 3 (already happens as part of the
Phase 1 run) → Phase 4 (flow, one manual trigger) → Phase 5 (live check) → Phase 6
(docs) → `make docs-build --strict` + local preview.

## Critical Files

- `vendor/starcop/starcop/validation.py` — `run_validation`, wrapped not edited; its
  `difficulty` bug (the pixel-count grouping) is what Phase 1's `paper_metrics.py`
  works around.
- `vendor/starcop/starcop/data/datamodule.py` — the `folder = root_folder/id` join
  logic `dataset_wiring.py` must replicate.
- `src/registry/hf_baseline_import.py` — reused for checkpoint loading/digest
  pinning; extended in Phase 2.
- `data/starcop_raw/test.csv` — ground truth for the corrected strong/weak bucketing
  (166/176 has_plume split, 57/109 strong/weak).
- `flows/retrain.py` and `prefect.yaml` — the structural pattern Phase 4 mirrors.
- `docs/results.md`, `docs/baseline_metrics.md`, `mkdocs.yml` — the public doc and
  its existing `include-markdown` mechanism.

## Verification

- **Eval core**: unit tests green (`paper_metrics.py`, `select_docs_examples.py`);
  full run asserts exact known counts (342 scenes, 166/176 split, 57/109 strong/weak).
- **Registry gap**: `MlflowClient().get_latest_versions("starcop-baseline-varon", ...)`
  returns a `Staging` version; `mlflow.pytorch.load_model(...)` round-trips.
- **MLflow**: all three runs visible in `starcop-paper-eval`, every tag/metric/
  artifact from Phase 3 present, all metric values in `[0, 1]`.
- **Prefect**: one manual `prefect deployment run` completes; per-task history
  visible in the Prefect UI; `MLFLOW_RUN_ID=` sentinel parsed for all three variants.
- **BentoML**: live `/predict` plume-pixel counts match the recorded offline counts
  for the curated scenes (Hyper variants only, per the documented scope limit).
- **Docs**: `make docs-build --strict` passes, WIP banner removed, images render via
  `make docs-serve`, table numbers spot-check exactly against the MLflow run.
