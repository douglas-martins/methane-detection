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
  vendor code). Unit tests use a small hand-built synthetic `out_data`/`test.csv`
  pair with known expected F1 values, plus a row-count-integrity assertion (guards
  against a silent id-mismatch dropping rows).
- **`select_docs_examples.py`** (pure, unit-tested) — deterministic picks for the
  public doc: best strong-plume detection, a weak-plume true positive, a false
  negative if any exist, cleanest no-plume scene.
- **`run_baseline_eval.py`** (thin glue, Large-boundary, real-run validated) —
  loads a checkpoint via `src/registry/hf_baseline_import.py::load_model()` (reused,
  not duplicated), wires the dataloader, calls `run_validation(products_plot=None)`
  for the fast metrics-only pass over all 342 scenes, then a second tiny pass calling
  `starcop.plot.plot_batch(...)` (same call `notebooks/starcop_baseline_validation.py`
  already makes) only for the curated scene ids, to produce sample-mask PNGs.
  Asserts the known counts (342 total, 166/176 has_plume split, 57/109 strong/weak)
  as hard checks, not just eyeballed. Supports `--limit N` (Phase 0) and
  `--emit-docs-assets DIR` (Phase 6).
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
  `paper_reference=true`, `resolved_device` (which environment/accelerator ran it).
- **Metrics**: the corrected `strong_f1score`/`weak_f1score`/`no_plume_FPR`/`auprc`
  (the real Table 1/2 headline numbers — explicitly distinguished by name from
  `run_validation`'s own uncorrected `easy_*`/`hard_*` keys) plus the full aggregate
  metrics via the existing `src/training/validation_metrics.py::extract_scalar_metrics`
  helper.
- **Artifacts**: per-scene results CSV (with the corrected bucket column), the full
  `run_validation` metrics JSON, a generated `paper_comparison.md` (this run's numbers
  next to the paper's published Table 1/2 values — those are entered once, by hand,
  with an explicit table/page citation, never approximated), and the curated
  sample-mask PNGs.

### Phase 4 — Prefect: make it a repeatable, auditable run

- [ ] Not started

New `flows/eval_baseline.py`, same `@task`/`@flow` shape and injectable-callable
testing pattern as `flows/retrain.py` (its `pull_dataset`/`notify`/failure-message
helpers are reused directly, not reinvented). Tasks: pull the dataset (dvc pull),
ensure MultiSTARCOP is registered (idempotent check-then-import), run the evaluation
per variant (shelling to Phase 1's CLI, parsing the same `MLFLOW_RUN_ID=` sentinel
convention `retrain.py` already establishes), emit docs assets (Phase 6), run the
BentoML live check (Phase 5, non-fatal if it fails), notify.

Deployed to the existing `mac-mps` work pool via a new `prefect.yaml` entry —
**no schedule**: this is an on-demand "regenerate the numbers" operation triggered
manually whenever the vendor pin, dataset snapshot, or eval code changes, not a
recurring job.

### Phase 5 — BentoML: verify the live API agrees

- [ ] Not started

New `src/evaluation/live_verify.py`: for each curated scene, POST its input array to
the live `/predict` endpoint and compare the returned plume-pixel count against the
value already recorded in Phase 1's `out_data.csv` for that scene — no extra offline
inference needed, it's already in the artifact.

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
5. **"How this was verified"** — brief, links to the MLflow run, notes predictions
   were spot-checked against the live API. Deep infra explanation stays in
   `docs/pipeline/{training,serving}.md`, not here.
6. Footer: last-updated date + exact MLflow run ID(s), so staleness is checkable.

**No hand-typed numbers, no drift**: `--emit-docs-assets DIR` (Phase 1) writes the
curated PNGs plus a generated Markdown table fragment that `docs/results.md` pulls in
via the `mkdocs-include-markdown-plugin` (`{% include-markdown %}` — already used by
`docs/changelog.md`, so this is a proven mechanism here, not a new one). The page is
mechanically tied to the last real run's output, not a manually maintained copy.

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
