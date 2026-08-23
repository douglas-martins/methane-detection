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

**Handling gaps where the paper's methodology is underspecified**: this benchmark's
entire value rests on matching the paper's original settings as closely as possible
— a reproduction that quietly substitutes a different convention wherever the paper
is vague isn't a reproduction. When the paper's Methods section doesn't fully pin
down something needed to compute a reported number, resolve it in this order: (1)
check whether the released `vendor/starcop` code settles it — if the code only ever
does it one way, that's the paper's real methodology regardless of what the text
says; (2) if the code doesn't settle it either, pick whichever convention this
project's own dependency stack already treats as standard (i.e. the choice an
implementation built on the same tooling the authors used would most plausibly have
made), not whichever is easiest to implement; (3) explicitly flag the resulting
number as assumption-dependent, with the reasoning, in both the internal
`paper_reference_metrics.md`/`paper_comparison.md` record (Phase 3) and the public
`docs/results.md` page (Phase 6) — never present it as an unqualified match to the
paper.

Known instance: **AUPRC integration convention** (Phase 1's `paper_metrics.py`). The
paper reports an AUPRC score (page 8, "Metrics") but never states whether it's
non-interpolated or trapezoidal, and `vendor/starcop` never computes AUPRC anywhere
(grepped for `auprc`/`average_precision`/`AveragePrecision` across `starcop/` — the
only hit is `torchmetrics`'s own library file, installed as a dependency but never
imported for this purpose) — so step (1) above doesn't resolve it. Falling through to
step (2): resolved to non-interpolated, matching `torchmetrics.AveragePrecision`
(which the codebase already uses for every other confusion-matrix metric) and
`sklearn.metrics.average_precision_score`. This is the best-supported guess for what
the authors' own tooling likely did, but it remains an assumption, not a verified
match — per the flagging rule above, this must be called out everywhere this
benchmark's AUPRC number is shown, not silently reported as-is.

## Recommended Approach

### Phase 0 — Settle Environment A vs. B (do this first, empirically)

- [x] **Done — decided: Environment A.** Executed both dry passes for real
  (`vendor/starcop/.venv` and `.venv`, `mag1c_rgb`, `--limit 5`). Environment
  B hit a real, hard snag, not a slowdown: `run_validation`
  (`validation.py:45,67,187`) calls `torchmetrics.ConfusionMatrix(num_classes=2)`
  with the pre-0.11 API (no `task=`); Environment B's pinned `torchmetrics`
  (1.9.0) requires `task=` and raises `TypeError` immediately. This is not
  fixable by re-pinning `torchmetrics` in this project's own
  `pyproject.toml`: `vendor/starcop/starcop/models/model_module.py:62-63`
  (also used by every Environment B training run) calls
  `torchmetrics.ConfusionMatrix(num_classes=2, task="binary")` — the
  *modern* API. `model_module.py` and `validation.py` require mutually
  incompatible `torchmetrics` versions, a real inconsistency inside
  `vendor/starcop` itself, so **no single `torchmetrics` version lets
  Environment B run both** — this isn't a Phase-0 timing tradeoff, it's a
  structural blocker. Environment A's pinned `torchmetrics` (0.10.0) matches
  `validation.py`'s expectation and ran cleanly end to end (checkpoint load
  → device placement → dataloader → `run_validation` batch → corrected
  metrics → AUPRC → docs-example picks), confirming the plan's fallback
  path. **Likely secondary finding, not fixed here (out of this plan's
  scope):** `train.py`'s own `run_validation` calls (`train.py:293`, `:317`)
  are wrapped in `try/except Exception: log.warning("run_validation ...
  failed -- skipping")` (`train.py:304-307`, `:326-327`) — the same
  `TypeError` this Phase 0 run hit would be silently swallowed there too, so
  Environment B training runs to date have likely never actually completed
  a `run_validation` diagnostic pass, contrary to this plan's original
  "proven working code" assumption for that call site.
- Two real, in-scope bugs found and fixed as part of getting the dry pass to run
  (both in this project's own code, not `vendor/starcop/`, so composition-only
  wasn't at stake): (1) `src/registry/hf_baseline_import.py::verify_checkpoint_digest`
  used `hashlib.file_digest`, added in Python 3.11 — Environment A is Python
  3.10, so every call there raised `AttributeError`; fixed to a portable
  chunked `hashlib.sha256().update()` loop, with a regression test
  (`test_computes_the_correct_digest_without_hashlib_file_digest`) that
  monkeypatches `hashlib.file_digest` away to prove the fallback path.
  (2) `run_validation`'s `path_save_results=None` default is itself broken
  (`validation.py:72`: unconditional `path_save_results.startswith("gs://")`,
  no None guard) — every existing caller (`train.py:299`, `:323`) always
  passes an explicit path, so nobody had hit this before;
  `run_baseline_eval.py`'s call site now always passes a real path
  (a throwaway tempdir for the metrics-only pass), matching that established
  pattern, without editing `vendor/starcop/`.

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

**Resolved implementation details** (checked against the current repo state before
starting Phase 0):

- **Sequencing.** As written, this dry pass needs `--limit`, checkpoint-loading, and
  a real test-set dataloader — all Phase 1 deliverables, and `src/evaluation/`
  doesn't exist yet. Rather than write a throwaway script that tests something
  different from what actually ships, build the minimal slice of Phase 1 first:
  `dataset_wiring.py` (test-only dataloader) plus a skeletal `run_baseline_eval.py`
  that only loads a checkpoint, wires the dataloader, and calls `run_validation`
  once — no MLflow logging, no `--emit-docs-assets`, no hard-count assertions yet.
  That skeleton is exactly what Phase 1's full `run_baseline_eval.py` grows from, so
  nothing here is thrown away.
- **`root_folder` for the eval dataloader is `data/starcop_raw/STARCOP_test`.**
  Verified empirically: `test.csv`'s own `folder` column is a stale absolute path
  from the original authors' machine (`/AVIRISNG/Permian2019/...`), unusable as-is —
  confirming `folder` must be rebuilt as `root_folder/id` per vendor's
  `load_dataframe` (`datamodule.py:104`), not read from the CSV column. With
  `root_folder = data/starcop_raw/STARCOP_test`, all 342 `test.csv` ids resolve to
  real scene directories on disk.
- **Device placement must be explicit.** `hf_baseline_import.py::load_model()`
  always loads to CPU (`torch.load(..., map_location="cpu")`) and never moves the
  model. `run_validation` derives its device purely from `model.device`
  (`cm_thr.to(model.device)`, `confusion_metric.to(model.device)` in
  `validation.py`). The dry-pass skeleton (and later `run_baseline_eval.py`) must
  call `model.to(device)` right after `load_model()` — otherwise "running under
  Environment B" silently stays on CPU and the whole empirical comparison this phase
  exists to make is invalid.
- **`--limit N` must use a stratified sample, not the first N rows.** Read directly:
  `run_validation`'s difficulty aggregation does unconditional lookups —
  `metrics_by_difficulty.loc[(False,"hard")]`, `.loc[(True,"easy")]`,
  `.loc[(True,"hard")]` — that `KeyError` if the sampled rows don't include at least
  one no-plume scene and at least one plume scene on each side of the (buggy,
  pixel-count) easy/hard split. This is the same fragility `train.py` already
  documents and wraps in `try/except` for small test sets (`train.py:274-280`,
  `304-307`) — and `notebooks/starcop_baseline_validation.py` hit the same problem on
  its own mini subset. Swallowing it here would mean the dry pass "passes" without
  ever exercising the real aggregation path, defeating its purpose as an empirical
  check. Pick the `--limit N` rows deterministically to guarantee all three buckets
  are present (e.g. force-include one no-plume row, one easy-pixel-count plume row,
  one hard-pixel-count plume row, then fill the remainder) rather than taking a
  naive head/tail/random slice.
- **"Real snag" means a crash, an incorrect result, or a device that silently fails
  to engage (see device-placement point above) — not merely "slower."** Environment
  B is already expected to win on speed; a slowdown alone is not a reason to fall
  back to Environment A.

### Phase 1 — Evaluation core (`src/evaluation/`, new package)

- [x] **Done for the two already-registered Hyper variants** (`mag1c_only`,
  `mag1c_rgb`). MultiSTARCOP/`varon` still needs Phase 2's registry gap
  closed before it can run — out of scope for this pass.

  Built `src/evaluation/` exactly as specified below (`dataset_wiring.py`,
  `paper_metrics.py`, `select_docs_examples.py`, `run_baseline_eval.py`,
  `_vendor_starcop_evaluation.py`, `__tests__/`) plus
  `scripts/run_starcop_baseline_evaluation.py`, all TDD'd — 59 unit tests,
  green on both environments (`make test-env-a`: 190 passed;
  `make test-env-b`: 284 passed).

  **Full, unlimited real runs executed against all 342 test scenes** (both
  registered variants, Environment A per Phase 0's decision) — asserted
  counts held (342 total, 166/176 has_plume, 57/109 strong/weak), no
  `KnownDifficultyBucketGapError`. Corrected numbers vs. this repo's own
  reading of the paper's Table 2 (not yet the canonical
  `paper_reference_metrics.md` — that's still Phase 3):

  | Metric | Paper, mag1c-only | This run | Paper, mag1c+rgb | This run |
  |---|---|---|---|---|
  | Strong F1 | 74.15 ± 6.10 | 66.74 | 81.96 ± 3.71 | 83.08 |
  | Weak F1 | 47.57 ± 4.17 | 48.76 | 43.42 ± 5.72 | 42.34 |
  | FPR (tile-level) | 52.11 ± 10.98 | 36.00 | 43.66 ± 7.36 | 40.57 |
  | AUPRC | 49.41 ± 5.49 | 36.19 | 51.99 ± 2.76 | 47.60 |

  mag1c+rgb lands close across all four metrics (within ~1 std). mag1c-only
  is directionally right but noisier — strong F1 (~1.2 std low), FPR
  (~1.5 std low, but the paper's own std here is ±10.98, i.e. high
  run-to-run variance even in the paper), and AUPRC (~2.4 std low) all sit
  further from the paper's mean than mag1c+rgb does. This single
  released checkpoint is not guaranteed to be the modal run of the paper's
  own 5-run average, so some spread is expected — but the AUPRC gap in
  particular is large enough to be worth another look before treating this
  number as settled (candidates: the non-interpolated-AUPRC assumption
  landing differently per variant, or a subtler input/threshold difference
  specific to the mag1c-only input configuration). Recorded here rather
  than smoothed over, per this plan's own fidelity principle (see Context).

  **A second real metric-definition bug was found and fixed by actually
  running this against real data** (not something reading the paper or
  vendor code alone surfaced) — see `paper_metrics.py`'s bullet below:
  `run_validation`'s own `FPR_no_plume` is pixel-summed, not the paper's
  tile-level definition; the first (pre-fix) `mag1c_rgb` run reported
  0.53% against the paper's 43.66% before this was caught and corrected.

  Two real, in-scope bugs also fixed to get the real run working at all
  (both this project's own code, not `vendor/starcop/`) — see Phase 0's
  entry above for the full detail: `hf_baseline_import.py`'s
  Python-3.11-only `hashlib.file_digest` usage, and `run_validation`'s
  broken `path_save_results=None` default (worked around at the call site).

  **Not yet done, deliberately out of this pass's scope:** MLflow
  experiment logging (Phase 3 — headed under a separate heading, not
  "Phase 1's glue" in the literal sense even though the plan's Ordering
  section describes it as a natural byproduct of a full run; skipped here
  since it wasn't asked for and pulls in registry/tagging decisions of its
  own), Prefect/BentoML wiring (Phases 4/5), and `docs/results.md` (Phase
  6) — `--emit-docs-assets` was exercised for real (curated PNGs +
  offline-prediction digests written to a scratch directory) but nothing
  was published to the canonical docs location.

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
  pattern `src/training/starcop_datamodule.py::_load_dataframe` already uses).
  **`root_folder = data/starcop_raw/STARCOP_test`** — verified against the real data:
  `test.csv`'s own `folder` column is a stale absolute path from the original
  authors' machine (`/AVIRISNG/Permian2019/...`) and unusable as-is, but with this
  `root_folder` all 342 `test.csv` ids resolve to real scene directories on disk. For
  MultiSTARCOP, calls `feature_extration.extract_features(...)` first (its Varon
  ratio bands are computed features, unlike Hyper's raw `mag1c`+RGB bands).
- **`paper_metrics.py`** (pure, unit-tested) — the actual bug fix. Joins
  `run_validation`'s returned per-scene `out_data` back to `test.csv`'s `qplume`
  column by scene `id`; buckets into `strong` (`qplume>=1000`), `weak` (`has_plume`
  and `qplume<1000`), `no_plume`; sums `TP/FP/TN/FN` per bucket and applies
  `starcop.metrics.{precision,recall,f1score,iou}` the same way `run_validation`
  does internally — just off the corrected bucket.

  **A second, distinct metric-definition mismatch found by actually running
  this against real data (not caught by reading the paper/code alone):**
  `run_validation`'s own `FPR_no_plume` (and `compute_bucket_metrics`'s
  equivalent, now exposed as `no_plume_fpr_pixel_level` precisely so it's
  never confused with the paper's number) is a pixel-summed ratio
  (`FP/(FP+TN)` over every no-plume pixel). The paper's own reported FPR
  (page 8, "Metrics") is a **tile-level classification rate** instead:
  "Each tile ... is finally marked as containing a plume if the thresholded
  prediction has more than 10 active pixels ... We study the false positive
  rate (FPR) on the subset of the evaluation dataset that does not contain
  any plumes." Confirmed empirically on a real `mag1c_rgb` full run: the
  pixel-summed number came out ~0.5%, against the paper's reported ~44% for
  the same variant — an ~80x gap from comparing two different metrics, not
  noise (strong/weak F1 on that same run landed close to the paper's
  numbers, confirming the pixel-level bucketing itself is correct). Fixed
  with a separate function, `tile_no_plume_fpr`, using
  `out_data`'s own per-scene `pred_pixels_plume` column against the same
  >10-pixel threshold the paper describes; this is the value that actually
  belongs in the `no_plume_FPR` key compared against the paper's Table 1/2
  FPR column.

  Also derives AUPRC from
  `run_validation`'s `metrics["thresholded"]` list (not directly returned by
  vendor code) — **the AUPRC convention must be pinned before implementation, not
  left to a library default**:
  - **Decided: non-interpolated (step-function) average precision.** Checked both
    sources this can come from: the paper's own Methods section (page 8, "Metrics")
    states AUPRC is used but never specifies the integration convention, and the
    released `vendor/starcop` codebase never computes AUPRC anywhere (grepped for
    `auprc`/`average_precision`/`AveragePrecision` across `starcop/` — the only hit
    is `torchmetrics`' own library file, installed as a dependency but never
    imported for this purpose). Since the codebase already standardizes on
    `torchmetrics` for every other confusion-matrix metric (`validation.py`,
    `model_module.py`), and `torchmetrics.AveragePrecision` implements exactly the
    non-interpolated convention (matching `sklearn.metrics.average_precision_score`)
    — this is the best-supported guess for what the paper's own numbers used, and is
    also the standard default in ML tooling generally. Record this reasoning (not
    just the choice) in `paper_comparison.md` (Phase 3) and `docs/results.md`
    (Phase 6) rather than presenting it as an unqualified match to the paper's
    number.
  - `metrics["thresholded"]` is built high-to-low threshold
    (`validation.py:41-42`); that ordering is not guaranteed to be strictly
    ascending in recall once TP/FP/FN come from summed per-bucket confusion
    matrices, so the derived (precision, recall) points must be explicitly
    re-sorted by ascending recall before integration — never just reversed.
  - Non-interpolated average precision: sum `(recall_n − recall_{n−1}) ×
    precision_n` over the sorted points, no interpolation between them.
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
  negative if any exist, cleanest no-plume scene. **Decided selection rule: highest
  IoU per bucket.** Best strong-plume / weak-plume TP = the highest per-scene IoU
  within that bucket; false negative = the highest-`qplume` scene with zero recall
  in its bucket, included only if at least one such scene exists; cleanest no-plume
  = the no-plume scene with the fewest predicted-positive pixels. IoU is the
  standard segmentation-quality metric and per-scene IoU is already produced by
  `paper_metrics.py`'s per-bucket pass, so this needs no extra computation.
- **`run_baseline_eval.py`** (thin glue, Large-boundary, real-run validated) —
  loads a checkpoint via `src/registry/hf_baseline_import.py::load_model()` (reused,
  not duplicated). **`load_model()` always returns a CPU model**
  (`torch.load(..., map_location="cpu")`) and never moves it — `run_validation`
  derives its device purely from `model.device`, so this function must call
  `model.to(device)` immediately after `load_model()`, using whichever device
  Phase 0 settled on. Wires the dataloader, calls `run_validation(products_plot=None)`
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
  paper-reproduction numbers. **`--limit N` must select a stratified sample, not a
  naive head/tail/random slice**: `run_validation`'s difficulty aggregation does
  unconditional lookups (`metrics_by_difficulty.loc[(False,"hard")]`,
  `.loc[(True,"easy")]`, `.loc[(True,"hard")]`) that `KeyError` unless the sample
  includes at least one no-plume scene and at least one plume scene on each side of
  the pixel-count easy/hard split — the same fragility `train.py` already documents
  and wraps in `try/except` for small test sets. Swallowing the error here would let
  `--limit` runs "pass" without ever exercising the real aggregation path, which
  defeats Phase 0's purpose; force-include one row from each required bucket, then
  fill the rest of `N` however's convenient.
- **CLI**: `scripts/run_starcop_baseline_evaluation.py` — thin argparse glue only,
  mirroring `scripts/import_starcop_hf_baseline.py`'s split.

### Phase 2 — Close the MultiSTARCOP registry gap AND make it actually runnable

- [x] **Done — 2026-08-21.** Built TDD (RED → GREEN, no mocking beyond the
  established `download_checkpoint_fn` injection seam): `resolve_checkpoint`
  and `local_checkpoint_dir` in `hf_baseline_import.py` (9 new tests) and
  `derive_features_extract` in `run_baseline_eval.py` (3 new tests) — 199
  passed on `make test-env-a` (was 190), 293 passed on `make test-env-b`
  (was 284).

  **Registered for real**: `scripts/import_starcop_hf_baseline.py varon
  --stage Staging` run live against the MLflow server — `starcop-baseline-varon`
  v1, `Staging`. Round-trip verified: `get_latest_versions(...)` returns the
  version; `mlflow.pytorch.load_model("runs:/<run_id>/model")` loads a real
  `ModelModule` (6.6M params) — via `runs:/`, not `models:/`, per this
  project's own known B2-store quirk ([[project-task-5-1-serving-complete]]);
  confirmed the same quirk affects the two pre-existing Hyper variants too,
  not something this phase introduced. Logged tags show clean local-source
  provenance with no HF fields: `{"source": "local", "local_path":
  ".../multistarcop_varon/final_checkpoint_model.ckpt", "variant": "varon",
  "baseline": "true", "sensor": "AVIRIS-NG"}`.

  **Actually runnable, not just registered**: `vendor/starcop/.venv/bin/python
  scripts/run_starcop_baseline_evaluation.py varon --limit 5` (Environment A,
  per Phase 0) completed end to end — checkpoint resolved from the local
  DVC-tracked path (no network call), `features_extract` auto-derived the
  three Varon ratio-band names from the checkpoint's own `input_products`
  (log confirms: "Feature ratio_wv3_B7_B5_varon21_sum_c_out does not exists.
  It will be generated" ×3), `run_validation` ran real batches, metrics and
  docs-example picks returned. This is a diagnostic-only 5-scene sample
  (noisy, not a reportable number, same as every other `--limit` dry pass in
  this plan) — the point was proving the wiring, not a Table 1 number.

  **Full, unlimited real run also executed against all 342 test scenes**
  (Environment A, ~9 min): asserted counts held (342 total), no
  `KnownDifficultyBucketGapError`. Compared against the paper's own Table 1
  "Our (Varon)" row (the exact variant this checkpoint is — not "Sanchez" or
  "Varon+Sanchez"):

  | Metric | Paper, Varon | This run |
  | --- | --- | --- |
  | Strong F1 | 30.72 ± 2.87 | 28.94 |
  | Weak F1 | 10.35 ± 1.52 | 17.01 |
  | FPR (tile-level) | 87.89 ± 4.67 | 84.57 |
  | AUPRC | 11.92 ± 1.35 | 11.27 |

  Strong F1, FPR, and AUPRC all land within ~1 std of the paper — the same
  reproduction quality already seen for `mag1c_rgb` in Phase 1. Weak F1 is
  the outlier: this run detects weak plumes noticeably better than the
  paper's own 5-run average (17.01 vs. 10.35 ± 1.52, ~4.4 std away),
  directionally the opposite of Phase 1's `mag1c_only` gap (which ran low,
  not high). Recorded here rather than smoothed over, per this plan's own
  fidelity principle (see Context) — worth a closer look before Phase 3
  treats this number as settled, alongside the still-open AUPRC-convention
  question from Phase 1. Not yet logged to `starcop-paper-eval` (Phase 3) or
  `--emit-docs-assets`'d — this was a metrics-only pass to confirm Phase 2's
  wiring produces a real, paper-comparable number end to end.

`multistarcop_varon` isn't in the MLflow registry yet (`hf_baseline_import.py`'s
variant table only has the two Hyper keys), so Track A currently can't produce a
Table 1 (MultiSTARCOP) number without this.

**HF-hosting check — resolved, not left open.** Queried the live repo directly
(`HfApi().list_repo_files("isp-uv-es/starcop")`): only
`models/hyperstarcop_mag1c_only/` and `models/hyperstarcop_mag1c_rgb/` exist.
`multistarcop_varon` is not on HF. It lives only at
`models/starcop_baseline/multistarcop_varon/final_checkpoint_model.ckpt` (+
sibling `config.yaml`) — and, checked against `models/starcop_baseline.dvc` /
`dvc list`, that path **is already DVC-tracked** (contradicts this plan's
earlier "fetched via gdown" framing and a stale note elsewhere in the docs
calling it untracked). So the local-source path needs no runtime `gdown` call
at all — just `dvc pull` (already a Phase 4 prerequisite) plus a direct file
read, same convention `notebooks/starcop_baseline_validation.py` already uses
(`MODELS_ROOT / "multistarcop_varon"`). Pinned digest of the file currently on
disk: sha256 `ccc9a7ec3d0bc8acf7e2f6232e798da0744d0890d02c0e2c54d08b3b1702b37e`.

**Verified empirically that `load_model()` needs zero changes for this
checkpoint**: loaded it directly via the existing function — `hyper_parameters
.settings.model.model_mode == "segmentation_output"`, dispatches to the
already-registered `ModelModule` class, state_dict loads cleanly (6.6M
params). The "already source-format-agnostic" claim holds.

**Scope correction: closing the registry gap alone does NOT make MultiSTARCOP
runnable.** Two call sites outside `hf_baseline_import.py`'s variant/digest
tables are hardcoded to the HF path and will still crash on `varon` after a
naive table-only fix — both are in scope for this phase now, not deferred:

1. **`src/evaluation/run_baseline_eval.py::evaluate_variant()`** calls
   `hf_baseline_import.download_checkpoint(variant, Path(tmp))` directly
   (bypassing the registry entirely) — this is the actual eval entry point,
   and it will raise via `variant_subfolder("varon")` unless fixed.
2. **`hf_baseline_import.py::import_variant()`** itself hardcodes HF-only
   tags (`hf_repo`, `hf_revision`, `hf_subfolder` — the last via an
   unconditional `variant_subfolder(variant)` call), so the registry-import
   path also breaks on a naive table-only extension, not just the eval path.

**Design: one shared source-dispatching resolver, used by both call sites.**
Add to `hf_baseline_import.py`:

- `_LOCAL_CHECKPOINT_PATHS = {"varon": Path("models/starcop_baseline/multistarcop_varon")}`
  (directory, since a local source also has its own `config.yaml` sitting
  next to the checkpoint — no need to fabricate one).
- `_EXPECTED_CHECKPOINT_SHA256["varon"]` added to the existing table, with a
  comment noting the different trust model (see caveat below).
- A new `resolve_checkpoint(variant, dest_dir) -> tuple[Path, Path, dict]`
  that replaces `download_checkpoint` as the single entry point both callers
  use: dispatches on `variant in _VARIANT_SUBFOLDERS` (→ today's HF download,
  returns `{"source": "huggingface", "hf_repo": ..., "hf_revision": ...,
  "hf_subfolder": ...}`) vs. `variant in _LOCAL_CHECKPOINT_PATHS` (→ verify
  digest against the on-disk file, no network, returns `{"source": "local",
  "local_path": str(checkpoint_path)}`); unknown variant raises the same
  `ValueError` shape `variant_subfolder` already does. `download_checkpoint`
  itself can stay as the HF-only implementation `resolve_checkpoint` calls
  into — no need to gut it, just stop calling it directly from either
  `import_variant()` or `evaluate_variant()`.
- `import_variant()` updated to call `resolve_checkpoint` and merge its
  returned provenance dict into the tags it sets, replacing the hardcoded
  `"source": "huggingface", "hf_repo": ...` block.
- `run_baseline_eval.py::evaluate_variant()`'s checkpoint-loading step (today
  `hf_baseline_import.download_checkpoint(variant, Path(tmp))`) updated to
  call `hf_baseline_import.resolve_checkpoint` instead — this is the one-line
  fix that actually closes the "still needs Phase 2 to run" gap Phase 1 flagged.

**`features_extract` wiring — auto-derived, not a new manual table.**
`dataset_wiring.build_test_dataloader` and `evaluate_variant` already accept
`features_extract` (Phase 1 built this generically), but nothing populates it
today — no CLI flag, no per-variant lookup — so a real `varon` run would try
to read `ratio_wv3_B7_B5_varon21_sum_c_out` etc. as raw on-disk bands and
fail. Verified `varon`'s embedded `settings.dataset.input_products` are
exactly the three Varon ratio-band keys, and confirmed those keys exist
verbatim in `vendor/starcop/starcop/data/feature_extration.py`'s
module-level `FEATURES` dict. So derive it automatically inside
`evaluate_variant()` rather than hand-maintaining a per-variant table:
`features_extract = [p for p in input_products if p in
feature_extration.FEATURES]` — empty (falsy) for both Hyper variants since
their raw `mag1c`/RGB band names aren't in `FEATURES`, so this is a strict
generalization with no behavior change for the two already-working variants.

**Local-source digest's weaker trust model — document, don't paper over.**
The HF digest check is anchored against an independent source (HF's own
published blob sha256 via `HfApi().model_info(files_metadata=True)`,
reviewed once against what was downloaded). For `varon` there is no
independent upstream digest to check against — "once-computed" here can only
mean hashing whatever's currently on disk (originally pulled via `gdown` from
Google Drive, now DVC-pinned). That still protects against future silent
corruption/tampering of the tracked file, but it doesn't verify the original
download was authentic the way the HF check does. State this explicitly in
the code comment next to `_EXPECTED_CHECKPOINT_SHA256["varon"]`, per this
plan's own fidelity-flagging principle (see Context).

**CLI**: add `"varon"` to the `choices` list in both
`scripts/import_starcop_hf_baseline.py` and
`scripts/run_starcop_baseline_evaluation.py`.

**Validated — registry AND a real runnable eval, not just a round-trip:**

- Registry: `MlflowClient().get_latest_versions("starcop-baseline-varon", ...)`
  returns a `Staging` version; `mlflow.pytorch.load_model(...)` round-trips
  cleanly (same bar the two Hyper variants were already held to).
- Runnable end to end: `scripts/run_starcop_baseline_evaluation.py varon
  --limit 5` completes without error — checkpoint resolves via the new local
  path, `features_extract` auto-derives and actually produces the ratio
  bands, `run_validation` runs a real batch. This is the concrete proof this
  phase's expanded scope ("fully runnable model after this phase") is met,
  not just that a model artifact is loadable from the registry in isolation.

### Phase 3 — MLflow: the permanent benchmark record

- [x] **Done — 2026-08-21.** Built TDD (RED → GREEN): new `src/evaluation/
  paper_eval_mlflow.py` (9 pure/SDK-glue functions, 20 new tests — pure
  builders unit-tested directly, `check_registry_version_matches` against a
  real sqlite `MlflowClient`, same convention as `test_mlflow_registry.py`),
  `resolve_checkpoint()` extended with a `checkpoint_sha256` tag (existing
  Phase 2 tests updated, RED → GREEN), `evaluate_variant()`'s return dict
  extended with the 4 new keys this phase needs (`joined_scene_results`,
  `run_validation_metrics`, `checkpoint_provenance`, `device`) — strict
  addition, no existing key's shape changed. `internal-docs/plans/
  paper_reference_metrics.md` authored by hand from the paper (page 9 Table
  1, page 10 Table 2), both a human-readable citation table and the fenced
  `yaml` block the loader parses. `make test-env-a`: 219 passed (was 199);
  `make test-env-b`: 313 passed (was 293).

  **A real, previously-undocumented bug was caught by the fail-fast
  registry check working exactly as designed**: the first live run for each
  variant raised `registry drift: checkpoint_sha256=None` — the
  `starcop-baseline-*` versions registered back in Phase 2 (before this
  phase added the `checkpoint_sha256` tag to `resolve_checkpoint`'s
  provenance dict) genuinely had no such tag, so `check_registry_version_
  matches` correctly refused to log against them rather than silently
  tagging a run with an unverifiable registry version. Fixed by re-running
  `scripts/import_starcop_hf_baseline.py <variant> --stage Staging` for all
  three variants (now version 2 each, all carrying the tag), then re-running
  the full evals.

  **All three variants' full, unlimited 342-scene runs executed for real**
  (Environment A, run in parallel — ~12 min wall-clock instead of ~27 min
  sequential, some per-scene slowdown from CPU contention as expected but no
  correctness issue) and logged live to `starcop-paper-eval`:
  `varon` (`8225d7cc737b42d0bf58933b056ed80a`), `mag1c_only`
  (`43d201bfca0a466691a67d54fae420f1`), `mag1c_rgb`
  (`10443a2c81db4cdbb33eb1620e02a599`) — corrected metrics matched Phase
  1/2's earlier documented numbers for all three exactly, confirming the
  pipeline is deterministic across re-runs. Verified against this phase's
  own bar, not just "the run exists": all three runs' tags contain all 10
  expected keys (`variant`, `registry_model_name`, `registry_version`,
  `checkpoint_sha256`, `dvc_dataset_version`, `n_test_scenes`,
  `paper_reference`, `resolved_device`, `eval_code_dirty`,
  `vendor_starcop_sha`) — `dvc_dataset_version`/`vendor_starcop_sha`
  identical across all three (same repo state, as expected),
  `eval_code_dirty=true` correctly reflecting the genuinely-uncommitted
  working tree; 42 metrics each, all within `[0, 1]`; all 4 expected
  artifacts present (`dependency_manifest.txt`, `paper_comparison.md`,
  `per_scene_results.csv`, `run_validation_metrics.json`). Downloaded and
  read `varon`'s `paper_comparison.md` directly — matches this repo's own
  hand-computed comparison from Phase 2 exactly, confirming
  `render_paper_comparison` and `paper_reference_metrics.md` agree end to
  end, not just in isolation.

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
  `run_validation` metrics JSON, the curated sample-mask PNGs (filenames prefixed
  with the variant name, e.g. `mag1c_rgb_sample1.png`, so per-variant runs never
  collide when later merged), and a frozen dependency manifest (`pip freeze` or
  equivalent) from whichever environment — A or B, per Phase 0's runtime choice —
  actually ran the evaluation, so a numeric drift between two runs of "the same"
  evaluation can be root-caused against an actual dependency diff instead of
  guessed at. Also logged: a copy of `paper_comparison.md` (this run's numbers next
  to the paper's published Table 1/2 values for this variant) for per-run
  provenance — but the *canonical* source of the paper-reported values is a single
  repo-committed file, `internal-docs/plans/paper_reference_metrics.md`, entered
  once by hand across all three variants with an explicit table/page citation,
  never approximated and never re-typed per run. Each run's logged
  `paper_comparison.md` copy is generated from that one canonical file, not an
  independent hand entry, so there is exactly one place the paper's numbers can
  drift from the published source.

**Resolved implementation details** (checked against the current repo state before
starting Phase 3 — every item below was answered by inspection, no open questions
needed a call):

- **`paper_reference_metrics.md` doesn't exist yet — it's this phase's own first
  deliverable, not a pre-existing input.** Read directly from the paper (page 9,
  Table 1; page 10, Table 2) rather than approximated:

  | Metric | MultiSTARCOP, Varon (Table 1) | HyperSTARCOP, mag1c-only (Table 2) | HyperSTARCOP, mag1c+rgb (Table 2) |
  | --- | --- | --- | --- |
  | Strong F1 | 30.72 ± 2.87 | 74.15 ± 6.10 | 81.96 ± 3.71 |
  | Weak F1 | 10.35 ± 1.52 | 47.57 ± 4.17 | 43.42 ± 5.72 |
  | FPR (tile-level) | 87.89 ± 4.67 | 52.11 ± 10.98 | 43.66 ± 7.36 |
  | AUPRC | 11.92 ± 1.35 | 49.41 ± 5.49 | 51.99 ± 2.76 |

  (Table 1's "Our (Varon)" row specifically — not "Sanchez" or "Varon+Sanchez",
  since the released `multistarcop_varon` checkpoint is the Varon-ratio variant.)
  Write this table into `internal-docs/plans/paper_reference_metrics.md` verbatim,
  each value tagged with its table/page citation, before any run's
  `paper_comparison.md` is generated from it — the generation step has nothing to
  read from until this file exists.

- **`evaluate_variant()` currently discards exactly what this phase needs to log**
  (checked `src/evaluation/run_baseline_eval.py:181-304` directly): the per-scene
  `joined` dataframe (with the corrected bucket column) never leaves the function;
  `run_validation`'s full metrics dict is written to a `tempfile.TemporaryDirectory()`
  that's deleted before return; `resolve_checkpoint`'s provenance tags are captured
  into `_provenance_tags` and thrown away; nothing tracks which `device` ran it. Its
  return dict needs four new keys — `joined_scene_results` (the per-scene dataframe,
  records-oriented), `run_validation_metrics` (the full in-memory dict, no more
  writing to a throwaway tempdir), `checkpoint_provenance` (the tags dict
  `resolve_checkpoint` already returns), and `device` (`str(device)`) — a strict
  addition, no existing key changes shape, so this doesn't disturb Phase 1/2's
  existing real-run validation of this function.

- **New module, not logic bolted into `evaluate_variant()`**: `src/evaluation/
  paper_eval_mlflow.py`, mirroring `hf_baseline_import.py`'s own split (pure,
  unit-tested tag/manifest builders + one thin SDK-glue function) rather than
  entangling MLflow I/O into `evaluate_variant()`, whose docstring already commits
  it to staying Large-boundary/real-run-validated instead of unit-tested. The CLI
  calls `evaluate_variant()` first, then this module's `log_paper_eval_run(variant,
  result, ...)` afterward — same two-step shape `import_variant()` already uses
  internally (resolve → load → log).

- **`eval_code_git_sha` is redundant, don't add a custom tag for it — verified
  live.** Checked the varon import run's full tag set (including `mlflow.*`
  tags, which this plan's own earlier ad hoc verification calls had been
  filtering out): `mlflow.source.git.commit` auto-populated correctly
  (`d95562002a9f6841e330688ae0347dea20cd4c8e`, matching the real HEAD at
  that time) because the script was invoked as a real file path, not
  `python -c`. Since every Phase 3 invocation goes through
  `scripts/run_starcop_baseline_evaluation.py` (a real file), this auto-tag
  will always fire — only `eval_code_dirty` is genuinely new code; the sha
  itself is already covered.
  `eval_code_dirty` (new, small): `git status --porcelain` against the repo
  root, non-empty output → dirty. Whole-repo scope (matching
  `mlflow.source.git.commit`'s own scope), not limited to `src/evaluation/`.
  Injectable via a `run_git_status_fn` parameter, same DI pattern already used
  for `load_model_fn`/`run_validation_fn`/`download_checkpoint_fn` elsewhere in
  this codebase, so it's unit-testable without shelling out in tests.
  `vendor_starcop_sha` (new, small, confirmed no existing plumbing anywhere in
  the repo): `git -C vendor/starcop rev-parse HEAD` — verified working live,
  returns `c4789268a3fa0395f92357429052f6f5fc748acb` for the currently
  checked-out submodule commit. Same injectable-subprocess pattern.

- **The plan's proposed DVC dataset-version reuse is wrong — verified against
  `dvc.lock` directly, don't build it as originally described.**
  `dvc_dataset_version.get_dataset_version("starcop_raw", ...)` returns the hash
  of `patch_extract@starcop_raw`'s *output* (`data/processed/starcop_raw/patches`,
  9cb8e072...) — this project's own training-tile pipeline artifact, a completely
  different object from `data/starcop_raw/test.csv`/`STARCOP_test` (the paper's
  untouched held-out test set Phase 1/2's eval actually reads). Reusing it would
  tag paper-eval runs with an identifier that doesn't reflect the actual eval
  input, and would spuriously flag "changed" on unrelated training-pipeline edits.
  The correct source is much simpler and already exists: `data/starcop_raw.dvc`
  itself is a plain single-hash DVC pointer for the whole raw directory (checked
  directly — `outs[0].md5 == "e11b16a61ddf235613701fbece9b59d6.dir"`). New tiny
  function, `dvc_tracked_dir_hash(dvc_file_path) -> str`, reads that one field —
  no `dvc.lock`/pipeline-stage coupling at all, and no `dvc status` subprocess
  call needed either (a static YAML read, unlike `is_dataset_dirty`'s live
  `dvc status` shell-out).

- **Dependency manifest: `pip freeze` doesn't work in Environment A — verified
  live.** `vendor/starcop/.venv` is a `uv`-created venv (`uv 0.9.5` per its
  `pyvenv.cfg`) with no `pip` module installed at all (`No module named pip`).
  `uv pip freeze --python <interpreter>` works against **any** venv regardless of
  how it was created or whether a lockfile is present (verified live against
  Environment A with no `uv.lock` in `vendor/starcop/` at all) — use this
  uniformly for whichever environment/interpreter actually ran the eval (matches
  what MLflow's own model-logging step already does for Environment B: "Detected
  uv project... Exported ... dependencies via uv", seen live during Phase 2's
  registry import), rather than a plain `pip freeze` that would silently fail in
  Environment A.

- **Registry-version tag: look it up explicitly, fail fast on mismatch — not
  assumed identical.** `evaluate_variant()` bypasses the registry entirely (calls
  `resolve_checkpoint` directly, per Phase 2's design), so nothing guarantees the
  checkpoint just evaluated is the same content as whatever the registry's
  current `Staging` version happens to be at logging time (e.g. a promotion race).
  `log_paper_eval_run()` calls `mlflow_registry.resolve_stage_version(client,
  registry_model_name(variant), "Staging")` (already exists, reusable
  unmodified), reads that version's own run tags for the checkpoint sha256 it
  was registered with, and raises if it doesn't match the sha256 of the
  checkpoint just evaluated — consistent with this plan's established
  fail-fast style elsewhere (Phase 1's join-integrity checks, Phase 4's
  concurrency-slot failure) rather than silently tagging a version number that
  may not correspond to what was actually run.

- **Logging is automatic on every full run, no new CLI flag** — already implied
  by Phase 1's own text ("MLflow logging into `starcop-paper-eval` is skipped
  whenever `--limit` is set"); the only gate is `--limit`, so `log_paper_eval_run`
  is called unconditionally whenever a full (non-`--limit`) run completes,
  independent of whether `--emit-docs-assets` was also passed.

### Phase 4 — Prefect: make it a repeatable, auditable run

- [x] **Done — 2026-08-22, fully validated end to end.** Built TDD
  (RED → GREEN): new `flows/eval_baseline.py`
  (14 pure/injectable functions + `@task`/`@flow` wrappers, same shape as
  `flows/retrain.py`, reusing its `pull_dataset`/`notify`/`parse_run_id`
  directly via `import retrain`), 42 new tests
  (`flows/__tests__/test_eval_baseline.py`), all fakes/injected — no real
  subprocess, network, or MLflow server touched by the suite. `flows/__tests__`
  was never wired into the Makefile at all (a pre-existing gap, unrelated to
  this phase) — fixed as part of this work so both `retrain.py`'s and
  `eval_baseline.py`'s tests are now part of `make test-env-b`'s gate.
  `make test-env-b`: 417 passed (was 348 before this phase; 69 of those are
  `flows/__tests__`, previously 0).

  **Deployed for real, verified live against the actual production Prefect
  server** (`methane-detection-prefect.ghostface.tech`) — not just written:
  `.venv/bin/prefect deploy --all` registered both `retrain-weekly` and the
  new `eval-baseline` deployment (id `57137f21-130b-4f57-83db-699b3e5f8ead`).
  Inspected the live deployment afterward to confirm the concurrency design
  actually took effect server-side, not just in `prefect.yaml`:
  `global_concurrency_limit` — `limit: 1`, `active: True`,
  `collision_strategy: 'CANCEL_NEW'` — exactly as designed. (Caught one real
  gotcha along the way: `prefect deploy --all` silently falls back to
  spinning up its own ephemeral local server unless `PREFECT_API_URL` is
  exported first — `scripts/prefect_worker_mac.sh` already sets this
  internally when starting the worker, but a bare interactive `prefect
  deploy` from a normal shell needs it exported explicitly too; noted in
  `deploy/prefect/README.md`'s deploy command.)

  **`.env.prefect` extended with the two AWS/B2 credential lines** this
  phase's own resolved-details section called for, copied from
  `.env.mlflow`'s real values (never printed) — verified present
  (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, now 8 lines total).
  `deploy/prefect/README.md` updated to document why (mirrors
  `GITHUB_ACTIONS_PAT`/Pushover's existing pattern, explained inline) and to
  add the `eval-baseline` deploy/run commands alongside `retrain-weekly`'s.

  **First real `prefect deployment run 'eval-baseline/eval-baseline'`
  attempted 2026-08-22 — caught one genuine bug immediately, at the very
  first task.** Every injectable piece of the orchestration was already
  unit-tested against fakes; this was the first time the real subprocess
  chain ran at all, and it found something no fake could have: `pull_dataset`
  (reused as-is from `retrain.py`) failed with `dvc pull failed with exit
  code 1`. Root cause, confirmed by reproducing `dvc pull` directly outside
  the flow: `dvc pull` refuses with `ERROR: failed to pull data from the
  cloud - Can't remove the following unsaved files without confirmation`
  whenever MultiSTARCOP/varon has been evaluated at least once in this
  working tree — `dataset_wiring.py`'s `feature_extration.extract_features`
  call (Phase 1) computes Varon ratio-band features on the fly and writes
  them **directly into `data/starcop_raw/`, a DVC-tracked directory**, as a
  side effect of running the eval, not as tracked content. 999 of the 1017
  files DVC balked at matched `ratio_wv3_*.tif` (this exact pattern);
  the remaining 18 were an unrelated but same-shaped issue,
  `weight_mag1c.tif` byproducts of the earlier STARCOP_mini
  baseline-validation notebook run sitting in `data/processed/starcop_mini/`.
  Neither is unique data — both are fully regenerable computation outputs,
  safe to have DVC simply stop tracking.

  **Fixed at the source, not worked around per-run**: added both patterns
  to `.dvcignore` (previously just the default empty template) rather than
  reaching for `dvc pull --force` in `pull_dataset` itself, which would also
  silently discard any *other* genuine local changes a future run happened
  to have — too blunt for a flow meant to be trustworthy and unattended.
  Verified live: `dvc pull` re-run immediately after the `.dvcignore`
  change exits `0` cleanly, no confirmation needed, same working tree that
  failed moments before.

  **Second real attempt (`devious-ant`) got much further, then caught a
  second genuine bug** — `pull_dataset` and `ensure_multistarcop_registered`
  both completed for real, and the `evaluate[varon]` task ran the actual
  342-scene evaluation to completion (~9 min, matching Phase 1's own timing)
  before crashing at the MLflow-logging step:
  `RuntimeError: Missing required MLflow tracking environment variable(s):
  MLFLOW_TRACKING_URI`. Root cause: `run_evaluation_for_variant` and
  `start_bentoml_serve` both shell out to a subprocess that needs
  `MLFLOW_TRACKING_URI` in its own environment
  (`scripts/run_starcop_baseline_evaluation.py`'s MLflow logging step;
  `service.py`'s `__init__`), but this flow's own process only has that
  value as a Python constant (`eval_baseline.MLFLOW_TRACKING_URI`, matching
  `retrain.py`'s established pattern for its *own* direct `MlflowClient`
  calls) — a Python constant never propagates to a child process on its
  own, unlike `.env.prefect`'s credentials, which the worker process
  inherits automatically. Neither subprocess call site was explicitly
  passing `env=`, so both silently fell back to inheriting `os.environ`
  as-is, which never had this key. **Note this crashed only *after* the
  real 342-scene pass fully completed** — the evaluation logic itself was
  never in question, only the logging step at the very end, so no wasted
  compute, just an unlogged result.

  Fixed via TDD (RED → GREEN): both functions now explicitly build
  `env = {**os.environ, "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI, ...}`
  before calling their subprocess. Caught in passing:
  `start_bentoml_serve` had zero direct test coverage before this (only
  exercised indirectly via `run_live_check_for_variant`'s fakes, which
  never touch its real `env=` construction) — added a dedicated test class
  for it while fixing the same bug there. 5 new tests, all green; `make
  test-env-b`: 422 passed (was 417).

  **Third real attempt (`beneficial-buzzard`) got past the
  `MLFLOW_TRACKING_URI` fix and re-ran `evaluate[varon]`'s full 342-scene
  pass successfully a second time (~9 min again) — then hit a third,
  different real bug at the same logging step**:
  `FileNotFoundError: [Errno 2] No such file or directory: 'uv'`, from
  `paper_eval_mlflow.py`'s `_run_uv_pip_freeze` (the dependency-manifest
  artifact, called right after the `MLFLOW_TRACKING_URI` check that just
  got fixed — one bug hid the next). Root cause, confirmed live: that
  function shelled out to a bare `"uv"`, relying on PATH — but
  `launchctl print gui/$(id -u)/com.methane-detection.prefect-worker`
  shows the worker's actual `default environment` is just
  `PATH => /usr/bin:/bin:/usr/sbin:/sbin`, launchd's own minimal default,
  never the fuller interactive-shell PATH that has `~/.local/bin` (where
  `uv` actually lives on this machine, confirmed via `which uv`). Every
  subprocess this flow spawns inherits that same restricted PATH, so any
  bare-command PATH lookup anywhere in the chain is a latent instance of
  this same bug — `retrain.pull_dataset`/`run_evaluation_for_variant`'s own
  `python`/`dvc` invocations were already safe only because they already
  used absolute paths, not because PATH itself was fine.

  Fixed via TDD (RED → GREEN): new `resolve_uv_binary()` in
  `paper_eval_mlflow.py` — tries `shutil.which("uv")` first (zero behavior
  change for any normal interactive/dev context), falls back to this
  machine's real install location (`~/.local/bin/uv`) if that fails, and
  raises a clear `RuntimeError` naming both places checked if neither
  resolves, rather than the opaque `FileNotFoundError` subprocess raises
  on its own. 3 new tests. Verified live under launchd's *exact* restricted
  PATH (`env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin ...`), not just reasoned
  about: resolved to `/Users/ghostface/.local/bin/uv` correctly. `make
  test-env-a`: 257 passed (was 254); `make test-env-b`: 425 passed (was 422).

  **Still open**: nothing past `evaluate[varon]` has been exercised for
  real yet — `evaluate[mag1c_only]`, `evaluate[mag1c_rgb]`, both
  `live_check`s, `aggregate`, and `publish` all remain unit-tested against
  fakes only. Given three real, distinct bugs surfaced by three consecutive
  attempts (each one only visible once the previous was fixed), treat this
  as the expected pattern for this flow's first full validation, not a
  sign something is unusually broken — every fix so far has been narrow
  and real-run-confirmed, not speculative. Next real run should reach
  `evaluate[mag1c_only]` at minimum.

  **Proactive audit after the `uv` bug, not just a one-off patch**: grepped
  every file reachable from this flow's actual subprocess chain
  (`flows/`, `scripts/`, `src/evaluation/`, `src/registry/`,
  `src/serving/`) for bare-command PATH lookups. Found and fixed two more,
  same class of bug: `paper_eval_mlflow._run_git_status`/
  `_run_git_rev_parse_head` both called a bare `"git"` — safe today only by
  coincidence (`/usr/bin/git` ships on every Mac and happens to already be
  inside launchd's minimal default PATH), not by design. New
  `resolve_git_binary()`, same shape as `resolve_uv_binary`. Wrote this up
  as a standing rule for any future flow-reachable script — see
  `deploy/prefect/README.md`'s new "Rule: never rely on ambient PATH or env
  vars in code a flow can reach" section — rather than leaving it as
  tribal knowledge in this one plan file. Caught and fixed a second,
  unrelated regression along the way: an earlier `ruff --fix` this same
  session had alphabetized `test_paper_eval_mlflow.py`'s imports, silently
  breaking an intentional "import `paper_eval_mlflow` first so its
  sys.path side effect runs before `import mlflow_registry`" ordering —
  invisible in `make test-env-a`/`test-env-b` (a different file's conftest
  happened to prime `sys.path` first when the full suite ran together),
  but broke the file when run standalone. Fixed at the root, not by
  re-pinning import order again: `src/evaluation/__tests__/conftest.py` now
  puts `src/registry`/`src/training` on `sys.path` itself, so no test
  file's import order matters anymore. `make test-env-a`: 259 passed (was
  257); `make test-env-b`: 427 passed (was 425).

  **Fourth real attempt (`mellow-fennec`) got past all three previous
  fixes and re-ran `evaluate[varon]`'s full pass a third time — then hit a
  fourth issue, this one operational, not a code bug**:
  `botocore.exceptions.NoCredentialsError: Unable to locate credentials`
  during the MLflow artifact upload. Root cause, confirmed directly: the
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` lines added to `.env.prefect`
  earlier this session were correct, but the worker process had been
  running continuously since the previous Monday — five days before that
  edit — and env vars are only read at process startup. `ps eww <pid> |
  grep AWS_` on the live process confirmed it: nothing there, the file on
  disk had changed, the running process hadn't. Fixed operationally, not
  by touching code: bounced the worker (`launchctl bootout` +
  `bootstrap`), confirmed the new PID actually has the credentials before
  retrying. Documented as a standing gotcha in
  `deploy/prefect/README.md` (any `.env.prefect` edit needs a worker
  restart to take effect, with the exact verify-don't-assume commands),
  not left as something that'll bite the next person who edits that file.

  **Fifth real attempt (`judicious-guan`) got past the credentials-missing
  error and re-ran `evaluate[varon]` a fourth time — then hit a fifth,
  subtler bug at the exact same logging step**:
  `botocore.exceptions.ClientError: An error occurred (InvalidAccessKeyId)
  when calling the PutObject operation: The AWS Access Key Id you provided
  does not exist in our records.` Confusing on its face — the credentials
  really were present and byte-identical to `.env.mlflow`'s known-good
  values (verified via sha256 hash comparison, not just eyeballing).
  Reproduced standalone to find the real cause: `boto3.client('s3')` with
  no explicit endpoint defaults to real AWS S3
  (`client.meta.endpoint_url == "https://s3.amazonaws.com"`), and a
  Backblaze B2 key presented to *real* AWS fails with exactly this error —
  not a connectivity problem, a wrong-provider problem. `.env.mlflow` has
  always carried `MLFLOW_S3_ENDPOINT_URL` (confirmed: every manual upload
  this session worked because of it); `.env.prefect` never got it, because
  the earlier audit that copied `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`
  across used `grep -o '^[A-Z_]*='` to list `.env.mlflow`'s keys — a
  pattern that silently can't match a var name containing a digit, so
  `MLFLOW_S3_ENDPOINT_URL` (the "3" in "S3") never showed up to be copied.
  A real, if minor, lesson: an audit script's own blind spots don't
  announce themselves — this one looked complete and wasn't.

  Fixed via TDD, following `MLFLOW_TRACKING_URI`'s own precedent rather
  than adding a fourth `.env.prefect` line: `MLFLOW_S3_ENDPOINT_URL` is a
  public S3-compatible endpoint hostname, not a secret, so it's a code
  constant (`eval_baseline.MLFLOW_S3_ENDPOINT_URL`) explicitly injected
  into both subprocess `env` dicts that need it —
  `run_evaluation_for_variant` (artifact upload) and `start_bentoml_serve`
  (the live-check server needs it too, to download the served model's
  weights from B2). 2 new tests. `make test-env-b`: 429 passed (was 427).
  Worker bounced again to pick up the code change, new pid confirmed
  running.

  **Sixth attempt (`gentle-dragonfly`) completed successfully end to end
  — all 8 tasks `Completed`, ~34 minutes total — but both live-checks
  reported `not_run`, not `passed`.** Investigated rather than accepted at
  face value. First cause: `resolve_paper_eval_run`'s original "exactly
  one run per variant, ever" invariant (Phase 5's own design) is
  fundamentally incompatible with Phase 4's "repeatable" goal — every
  successful flow run adds new `starcop-paper-eval` runs without retiring
  old ones, so the very next live-check after the first-ever clean run was
  always going to fail this check. **Decided with the project owner:
  `resolve_paper_eval_run` now picks the most recent run per variant by
  `start_time` instead of demanding exactly one** — old runs are the audit
  trail Phase 3 was built for, not stale duplicates; deleting them after
  every future success was rejected as the alternative since it would
  destroy that trail. Fixed via TDD, `make test-env-b`: 430 passed (was
  429).

  Re-testing surfaced a **second, more serious cause, live-verified end to
  end with a real `bentoml serve` process**: `run_live_check_for_variant`
  calls `live_verify.verify_variant` in-process, not via subprocess, so
  the `MLFLOW_S3_ENDPOINT_URL` fix injected into the two subprocess `env`
  dicts never reached it — its own MLflow artifact download defaulted to
  real AWS S3 and failed, caught silently and reported as `not_run`. Fixed
  via TDD (`os.environ.setdefault(...)` once, right before the in-process
  call — there's no `env=` to inject into for a direct function call).
  `make test-env-b`: 431 passed (was 430).

  With both fixed, a real end-to-end check (fresh `bentoml serve`, no
  fakes) got further but reported `"failed"`, not `"passed"` — 4 of 8
  compared scenes mismatched. **Third bug, and the most consequential**:
  all three variants share one staging directory for the whole duration of
  a single flow run (`run_eval_baseline_cycle`'s one
  `tempfile.TemporaryDirectory()` wraps the entire `evaluate` loop), and
  `persist_offline_predictions` wrote filenames as plain `{scene_id}.json`
  with no variant prefix — flagged as a latent risk in Phase 5's own
  resolved-details section months ago, never triggered until a real flow
  run actually shared one directory across variants for real.
  Confirmed live: `mag1c_only`'s uploaded `offline_predictions/` artifact
  had 8 files (its own 4 plus 4 leftover from `varon`, which ran first in
  the loop); `mag1c_rgb`'s had 12 (its own 4 plus `varon`'s 4 plus
  `mag1c_only`'s 4, compounding since it ran last). `live_verify.py`
  correctly detected the mismatch when comparing `varon`'s stored
  predictions against `mag1c_only`'s live server for those scene ids — a
  real bug surfacing as a real (not flaky) failure. Fixed via TDD in two
  places: `persist_offline_predictions` now prefixes filenames with
  `variant` (matching the sample-mask PNGs' own existing convention), and
  `paper_eval_mlflow.collect_docs_asset_artifacts` now filters
  `offline_predictions/` by that same prefix before upload, not just
  `sample_masks/`. `make test-env-a`: 261 passed (was 260); `make
  test-env-b`: 432 passed (was 431).

  **`gentle-dragonfly`'s own uploaded `mag1c_only`/`mag1c_rgb` runs are
  contaminated by this third bug and should not be trusted for a live
  check** (their real metrics — strong/weak F1, FPR, AUPRC — are unaffected
  and correct, only the `offline_predictions` artifact used for live
  verification is mixed). No manual cleanup needed: `resolve_paper_eval_run`
  now picks the most recent run automatically, so the next successful flow
  run's clean artifacts supersede these without deleting anything.

  **Seventh attempt (`modern-dingo`), with all three fixes in place —
  fully clean, genuinely verified end to end.** All 8 tasks `Completed`,
  no leftover `bentoml serve` processes. Both Hyper variants' live-checks
  report **`passed`**, for real this time — confirmed by inspecting the
  actual MLflow artifacts, not just trusting the status string:
  `mag1c_only` (run `ddf8d8e4...`) and `mag1c_rgb` (run `0116dc44...`) each
  carry exactly 4 `offline_predictions/*.json` files, correctly prefixed
  with their own variant name, zero cross-contamination from `varon` (run
  `b2a1f6cd...`, also clean at 4 files). Published
  `docs/assets/paper_eval/paper_comparison.md` reflects this: `mag1c_only`
  and `mag1c_rgb` both show `Live check: passed`; `varon` correctly shows
  `out of scope` (never claims a check that can't run).

  **Seven real issues found across seven consecutive attempts — six code
  bugs, one operational gotcha, all fixed and real-run-verified.** Phase 4
  is done: the flow runs cleanly, unattended, start to finish, and its own
  live-verification step genuinely passes against the real served models.

New `flows/eval_baseline.py`, same `@task`/`@flow` shape and injectable-callable
testing pattern as `flows/retrain.py` (its `pull_dataset`/`notify`/failure-message
helpers are reused directly, not reinvented). Tasks, in order: pull the dataset (dvc
pull), ensure MultiSTARCOP is registered (idempotent check-then-import), run the
evaluation per variant (shelling to Phase 1's CLI once per variant, each call
already passing that same `--emit-docs-assets STAGING_DIR` per Phase 1, where
`STAGING_DIR` is a fresh, **run-scoped** directory named with this flow run's ID —
never the canonical directory `docs/results.md` reads from — so it writes that
variant's curated PNGs — filenames prefixed with the variant name so the three
invocations sharing one `STAGING_DIR` never collide — and a per-variant metrics
fragment; parsing the same `MLFLOW_RUN_ID=` sentinel convention `retrain.py`
already establishes, the task loops over all three variants and captures and
holds **one run ID per variant**, keyed by variant name, plus the paths to that
variant's per-scene results CSV, masks, and metrics JSON already written as
MLflow artifacts per Phase 3), run the BentoML live check (Phase 5, still
non-fatal — a serving-side outage shouldn't block regenerating the numbers) once
per servable variant, **then aggregate, publish, and notify**. Aggregation is a
new, distinct, lightweight task — not a fourth call into Phase 1's CLI and not
another call into `run_validation`: keyed by variant, it consumes every variant's
already-captured run ID and per-variant docs-asset fragment from `STAGING_DIR`,
plus the canonical `paper_reference_metrics.md` (Phase 3), and merges them into
**one** combined comparison fragment covering all three variants, written into
that same `STAGING_DIR` — never leaving three separate per-variant fragments for
`docs/results.md` to merge by hand. Before publishing, the task validates that
all three variants' run IDs, artifacts, and live-check statuses (pass/fail/not-run
— a live check must have a recorded status, not be silently missing) are present
in `STAGING_DIR`; a flow run that fails this check errors out and never touches
the canonical directory, so a partial run cannot corrupt what's already published.
Only once validation passes does a final **publish** step atomically replace the
canonical directory (e.g. write-then-`rename`/directory-swap, not an in-place
file-by-file overwrite) with `STAGING_DIR`'s contents, so `docs/results.md` (and
any concurrent `make docs-build`) only ever sees either the previous complete
publish or the new complete one, never a directory mid-overwrite. The live check
runs *before* aggregation for exactly this reason, and its per-variant
pass/fail/not-run results are threaded into that same aggregation task so the
generated page's "How this was verified" section (Phase 6) states plainly, per
variant, whether the live spot-check passed, failed, or didn't run — never
silently claiming verification that didn't happen.

**Concurrency protection.** Because this flow is on-demand (no schedule, see
below) it can be triggered more than once before a prior run finishes; the
deployment must set a Prefect concurrency limit of 1 so two overlapping runs
can never both reach the publish step and interleave writes to the canonical
directory or to `STAGING_DIR`'s parent. **Resolved: deployment-scoped, not
tag-scoped** — verified live against the installed `prefect==3.7.7` client
(`DeploymentCreate.concurrency_limit`/`concurrency_options` exist on the
schema; `ConcurrencyLimitStrategy.CANCEL_NEW` is a valid value). A tag-scoped
global concurrency limit would work too, but `mac-mps` is a *shared* work
pool (`retrain-weekly` also runs on it) — a deployment-scoped limit only
throttles `eval-baseline` against itself, never `retrain-weekly`. Set in the
new `prefect.yaml` deployment entry (see below) as
`concurrency_limit: {limit: 1, collision_strategy: CANCEL_NEW}` —
`CANCEL_NEW` is what gives the fail-fast behavior this phase already
requires: a run that can't acquire the slot is cancelled immediately with a
clear "another eval_baseline run is already in progress" status, rather than
queuing silently (`ENQUEUE`, the other available strategy) or racing the
in-progress run.

**The flow never passes `--limit`.** This is the sole path that logs to
`starcop-paper-eval` and emits docs assets, so it always shells out to Phase 1's CLI
for the full, unlimited run; `--limit` stays a manual, interactive-only flag for
Phase 0's dry pass and is never wired into `eval_baseline.py` or `prefect.yaml`.

**This flow itself runs for real on production Prefect** — deployed to the
existing `mac-mps` work pool via a new `prefect.yaml` entry, on the same
single production Prefect server (`methane-detection-prefect.ghostface.tech`)
`retrain-weekly` already deploys to. There is no separate "local Prefect" —
once `eval_baseline.py` exists it gets validated manually
(`prefect deployment run`, same practice `deploy/prefect/README.md` already
establishes for `retrain-weekly`) and then is a real production deployment,
same as every other flow in this project. **No schedule**: this is an
on-demand "regenerate the numbers" operation triggered manually whenever the
vendor pin, dataset snapshot, or eval code changes, not a recurring job. (Do
not confuse this with the separate, narrower "local instance" decision below,
about what one *task inside* this production-deployed flow talks to over
HTTP — that's about a throwaway `bentoml serve` process, not about where the
flow runs.)

**Resolved implementation details** (checked/decided before starting Phase 4,
same spirit as Phase 3's own resolved-details pass — every item below was
either answered by inspection or decided with the project owner, no open
questions left unresolved going into implementation):

- **Sequencing — Phase 5's core built before Phase 4, despite the phase
  numbering.** Phase 4's own task list calls Phase 5's live check as one of
  its steps, but Phase 5 (`live_verify.py`) doesn't exist yet — only its
  MLflow-artifact prerequisite (the curated sample-mask PNGs and offline
  mask-digest/confidence JSON now uploaded by `paper_eval_mlflow.py`, fixed
  and real-run-validated separately). The plan's own `## Ordering` section
  originally listed Phase 4 before Phase 5, which is backwards for this
  dependency — Phase 5 has no dependency on Phase 4 (only on Phase 3's
  artifacts, already fixed), so there's no cost to building it first. Decided
  with the project owner: resequence rather than ship Phase 4 with the
  live-check task stubbed out — see `## Ordering` above. Two live re-runs of
  `mag1c_only`/`mag1c_rgb` with `--emit-docs-assets` already confirm the
  artifact plumbing works end-to-end in MLflow (`sample_masks/`,
  `offline_predictions/` present under both new runs), so Phase 5 has real
  data to build against once it starts.

- **Canonical docs-asset directory: `docs/assets/paper_eval/`.** Neither this
  phase nor Phase 6 ever named the actual path for "the canonical directory
  `docs/results.md` reads from" — found by inspection, not guessed:
  `docs-reorganization-plan.md:188` already establishes the project's own
  convention for this exact page (`docs/results.md`'s images live under
  `docs/assets/<page-slug>/`; it explicitly plans moving the existing
  mini-set images from `docs/baseline_validation/` to
  `docs/assets/baseline_validation/`). Since Phase 6 folds that mini-set in
  as a separate historical subsection rather than merging it with the new
  paper-eval curated masks, the paper-eval assets get their own sibling
  directory under the same convention rather than reusing
  `baseline_validation`'s. The staging→publish swap is therefore:
  `STAGING_DIR` (run-scoped, e.g. under a temp/work directory, never
  committed) → atomically replaces `docs/assets/paper_eval/` (e.g.
  write-to-`docs/assets/paper_eval.new/` then `os.rename` over the old
  directory, so a concurrent `make docs-build` never observes a half-swapped
  directory).

- **Prefect concurrency mechanism: deployment-scoped, not tag-scoped** — see
  the "Concurrency protection" paragraph above for the resolved
  `concurrency_limit`/`CANCEL_NEW` detail, verified live against the
  installed `prefect==3.7.7` client schema.

- **Idempotent MultiSTARCOP-registered check.** No such check exists yet in
  `src/registry/hf_baseline_import.py` (verified — `import_variant` always
  registers a new version unconditionally, there's no "is this already
  registered" guard anywhere in that module). New small function, same
  reuse-not-reinvent spirit as the rest of this plan: resolve the checkpoint
  that *would* be imported (`hf_baseline_import.resolve_checkpoint("varon",
  ...)`, already used elsewhere for its digest), then check whether
  `mlflow_registry.resolve_stage_version(client, "starcop-baseline-varon",
  "Staging")` already points at a version whose own `checkpoint_sha256` tag
  matches that digest (same comparison `paper_eval_mlflow.
  check_registry_version_matches` already makes, reusable as a template, not
  copy-pasted since that function raises on mismatch rather than returning a
  bool) — skip `import_variant` if so, call it if not (mismatch or nothing
  registered yet). Checked live: `starcop-baseline-varon` is currently at
  Staging version 2 with a `checkpoint_sha256` tag present, so this check
  will pass and skip re-importing on the next flow run — but the flow must
  not assume that statically; the check has to run for real every time,
  since a checkpoint or registry change could invalidate it.

- **Marker-file caution, not a redesign.** `paper_eval_mlflow.
  log_paper_eval_run`'s `mlflow_utils.write_run_id_marker` call writes the
  run id to a single shared file at `repo_root/mlflow_run_id.txt` (not
  per-variant) in addition to printing the `MLFLOW_RUN_ID=` stdout sentinel.
  Safe as long as `eval_baseline.py` only ever reads each subprocess's own
  captured stdout for the run id (which is what this phase's design already
  does, mirroring `retrain.py`'s `parse_run_id`) — but if the three variant
  evaluations are ever run concurrently (as Phase 3's manual pass did, to
  cut wall-clock time) and anything reads that shared file instead of
  captured stdout, it will race. Don't read `mlflow_run_id.txt` from
  `eval_baseline.py`; stdout only.

**Second resolved-details pass** (added after Phase 5 was actually built and
real-run validated — these only became visible once there was a real
`live_verify.py` to wire in, not before):

- **B2/AWS credentials: add two lines to `.env.prefect`, don't invent a new
  sourcing mechanism.** Checked live — `.env.prefect` currently has exactly
  six lines (`PREFECT_API_AUTH_STRING`, `GITHUB_ACTIONS_PAT`,
  `PUSHOVER_USER_KEY`, `PUSHOVER_API_TOKEN`, `MLFLOW_TRACKING_USERNAME`,
  `MLFLOW_TRACKING_PASSWORD`), no `AWS_ACCESS_KEY_ID`/
  `AWS_SECRET_ACCESS_KEY`. `retrain.py` never needed them because it shells
  out to `train_mac.sh`, a shell script that self-sources `.env.mlflow`
  (which does have the B2 creds) independently — confirmed by reading both
  files directly. `eval_baseline.py`'s subprocess calls are different: it
  shells out to `scripts/run_starcop_baseline_evaluation.py` (needs B2
  creds to *upload* MLflow artifacts) and `scripts/run_live_verify.py`
  (needs them to *download* the served model + offline_predictions
  artifacts) — both plain Python CLIs, not self-sourcing shell scripts, so
  they only ever see whatever env the Prefect worker process itself already
  has. Grepped the whole codebase for any existing "read `.env.mlflow` from
  Python" helper — none exists; every module just reads `os.environ`
  directly and assumes boto3's own credential chain finds it there. Given
  `.env.prefect` is already the established "everything the worker's
  subprocesses need" file (that's literally how `GITHUB_ACTIONS_PAT`/
  Pushover creds already reach `retrain.py`'s subprocesses — by being
  *in* `.env.prefect`, not read from elsewhere), the consistent fix is
  adding `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` as two more lines
  there, not building a one-off `.env.mlflow`-reading mechanism just for
  this flow. `MLFLOW_TRACKING_URI` itself doesn't need a new line either —
  `eval_baseline.py` should hardcode it as a Python constant, same as
  `retrain.py:35` already does (public, non-secret URL).

- **`eval_baseline.py` must start and stop its own `bentoml serve`
  process(es) — this is real, unwritten glue, not just "call Phase 5."**
  `live_verify.py` assumes a server is already running and reachable;
  it never starts one. Resolved design, mirroring exactly what this
  session's manual Phase 5 validation actually did (and confirmed working
  live): for each servable variant, `subprocess.Popen([venv_python, "-m",
  "bentoml", "serve", "src.serving.service:MethaneDetectionService",
  "--port", PORT], env={**os.environ, "MODEL_NAME":
  hf_baseline_import.registry_model_name(variant), "MODEL_STAGE":
  "Staging"})`, then poll `POST /health` (not `GET` — `service.py`'s own
  convention) until it returns 200 or a timeout elapses, run
  `scripts/run_live_verify.py` against that `base_url`, then terminate the
  process in a `finally` block so a raised exception or a failed check
  still tears the server down. One variant's server at a time (sequential,
  reusing one fixed port), not parallel `bentoml serve` instances — the
  live check is cheap (a handful of curated-scene requests, not a full
  342-scene pass), so there's no real time cost to serializing it, and it
  avoids port-management complexity for no benefit.

- **Startup timeout: 20 minutes, not left unbounded.** This session hit
  the same B2/IPv6 `SYN_SENT` connectivity stall (see Phase 5's own
  write-up) three separate times while validating `live_verify.py` and
  `bentoml serve` locally — twice self-resolving in under 10 minutes, once
  taking over 20 before it cleared. Phase 4 already calls the live check
  "non-fatal," but that only protects the flow if the task actually
  *returns* (pass, fail, or timeout) — an indefinitely-hung `bentoml serve`
  startup would hang the whole flow run, not just fail one task. The
  `/health` poll loop above must have an explicit deadline (20 minutes,
  set from this session's own observed worst case with headroom) that
  raises past which the live-check task is recorded as `"not_run"` (per
  this phase's own existing status vocabulary) and the flow moves on to
  aggregation, exactly like a `bentoml serve` outage would already be
  handled. The IPv4-only `sitecustomize.py` workaround used to unblock
  local validation this session is not something to bake into
  `eval_baseline.py` itself — it's a local workaround for one flaky
  network, not a fix; if this recurs often once the flow runs
  unattended, the real fix belongs at the network/infra level, not in this
  plan's application code.

- **Live-check HTTP target (not to be confused with where the flow itself
  runs, see above): a throwaway `bentoml serve` process for both Hyper
  variants, not the real deployed BentoML service — decided with the
  project owner.** This is entirely about what `live_verify.py`'s `POST
  /predict`/`/health` calls hit, inside one task of the (already
  production-deployed) `eval_baseline` flow — not about Prefect at all.
  The already-running production BentoML deployment
  (`https://api-methane-detection.ghostface.tech`) serves `mag1c_rgb`/
  Staging continuously, and pointing `mag1c_rgb`'s live check at it
  directly would be a stronger verification (proves what's actually
  deployed agrees, not just a freshly-loaded copy of the same checkpoint).
  Rejected in favor of the uniform local-instance approach above: an
  unattended, periodically-triggered background flow should not add load
  to or depend on live production infrastructure, especially given this
  session's own observed network flakiness (a flaky retry loop against
  production is a worse failure mode than one against a throwaway local
  process). `varon` already forces a local instance regardless (per Phase
  5's own scope-limit note), so this also keeps both servable variants on
  one uniform code path rather than special-casing `mag1c_rgb`. Trade-off
  accepted explicitly: this live check proves the registry checkpoint is
  servable and agrees with the offline numbers, not that the currently
  *deployed* production service does — `docs/results.md` (Phase 6) should
  word its "How this was verified" section accordingly, not imply
  production itself was checked.

### Phase 5 — BentoML: verify the live API agrees

- [x] **Done — 2026-08-21.** Built TDD (RED → GREEN): new `src/evaluation/
  live_verify.py` (9 pure/SDK-glue functions, 25 tests — pure
  comparison/parsing logic unit-tested directly, `resolve_paper_eval_run`
  against a real sqlite `MlflowClient`, same convention as
  `test_paper_eval_mlflow.py::TestCheckRegistryVersionMatches`) plus
  `scripts/run_live_verify.py`, thin argparse glue mirroring
  `run_starcop_baseline_evaluation.py`'s split. `make test-env-a`: 254
  passed (was 229 at session start); `make test-env-b`: 348 passed (was
  323).

  **A real bug was caught by the negative-path validation working exactly
  as designed, not by unit tests alone**: the first `assert_model_identity`
  implementation compared only the served model's registry *version
  number* against the expected one, not its *model name*. Since
  `mag1c_only` and `mag1c_rgb` each have their own independent per-name
  version counter, both legitimately sit at version "2" — so verifying
  `mag1c_rgb`'s predictions against a server actually serving `mag1c_only`
  silently passed the identity check and proceeded to compare against the
  wrong model (caught live: the mismatched request then failed downstream
  with a 400 channel-count error instead of the intended fail-fast
  `ModelIdentityMismatch`). This is exactly the "silently comparing against
  the wrong model" failure mode this phase's own text already warned
  about. Fixed properly via TDD (new failing tests first, RED → GREEN):
  `assert_model_identity` now checks both `model_name` and `model_version`
  from `/health`, using `hf_baseline_import.registry_model_name(variant)`
  as the expected name. Re-validated live afterward — see below.

  **Both servable Hyper variants validated live, for real, end to end** —
  not just unit tests: started a real local `bentoml serve` process per
  variant (`MODEL_NAME=starcop-baseline-{mag1c_only,mag1c_rgb}
  MODEL_STAGE=Staging`), confirmed `/health` reported the expected
  registry version, then ran `scripts/run_live_verify.py` against it for
  real. **`mag1c_only`**: all 4 curated scenes passed, mask sha256 digests
  matching Phase 1/3's offline-recorded values bit-for-bit, confidence
  arrays within tolerance, exit code 0. **`mag1c_rgb`**: same result, all 4
  curated scenes passed exactly. Negative path also validated live (not
  just by unit test): pointed `mag1c_rgb` verification at the
  `mag1c_only`-serving process — correctly raised `ModelIdentityMismatch`
  and refused to compare predictions, confirming the bug above is actually
  fixed, not just passing in isolation.

  **One real infrastructure issue hit and worked around, not a code bug**:
  this machine's IPv6 route to the MLflow artifact store
  (`s3.us-east-005.backblazeb2.com`, Backblaze B2) intermittently hangs on
  `SYN_SENT` until OS timeout before falling back to IPv4 — seen across
  several unrelated downloads this session, in one case for over 20
  minutes before self-resolving. Worked around for the `bentoml serve`
  validation runs via a `PYTHONPATH`-injected `sitecustomize.py` that
  forces IPv4-only DNS resolution for that one throwaway process tree — a
  local, non-repo workaround, not a code or infra change. Left unfixed at
  the infra level (outside this plan's scope); worth a network-level fix
  (e.g. disabling IPv6 route advertisement for this host, or pinning
  `AWS_S3_ADDRESSING_STYLE`/endpoint resolution to IPv4 in the actual
  serving/eval environments) if it recurs during Phase 4's automated flow
  runs, where nobody will be watching to intervene.

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

**Resolved implementation details** (checked/decided before starting Phase 5,
same spirit as Phase 3/4's own resolved-details passes):

- **One run per variant, restored.** Backfilling the `sample_masks`/
  `offline_predictions` artifacts (the fix this session made to
  `paper_eval_mlflow.log_paper_eval_run`) was done by re-running
  `mag1c_only`/`mag1c_rgb` rather than editing the original runs in place,
  which left two runs per variant in `starcop-paper-eval` — the original
  Phase 3 run (no docs-asset artifacts) and the new one (has them), both
  tagged `registry_version=2`, genuinely ambiguous for anything doing a
  registry-version-only lookup. **Decided with the project owner: delete the
  two stale runs** (`43d201bf...` for `mag1c_only`, `10443a2c...` for
  `mag1c_rgb`) rather than teach `live_verify.py` permanent
  most-recent-run disambiguation logic — done via `MlflowClient.delete_run`
  (soft delete, recoverable via `restore_run`), verified live:
  `starcop-paper-eval` now holds exactly one run per Hyper variant again
  (`mag1c_only` → `44ab9523...`, `mag1c_rgb` → `6e3c063b...`), matching
  Phase 3's original one-run-per-variant design. `live_verify.py` can
  therefore resolve "the run to verify against" with a plain
  variant+registry-version tag lookup, no most-recent/has-artifacts special
  casing. **Caution for whoever runs this backfill pattern again**: delete
  the prior run first (or extend the CLI to overwrite/replace rather than
  create a new run) instead of accumulating duplicates a second time.

- **Per-scene input-array construction: reuse `dataset_wiring`, verified
  empirically end-to-end, not just by reading the code.** Neither this
  phase's original text nor any other phase ever specified how
  `live_verify.py` builds the array it POSTs to `/predict` for a given
  curated scene id. Resolved recipe: call
  `dataset_wiring.build_test_dataloader(test_csv_path, root_folder,
  input_products=..., output_products=..., weight_loss=...,
  features_extract=..., scene_ids=[scene_id], batch_size=1)` (the same
  function `run_baseline_eval.py`'s curated pass already uses), take the
  first batch's `batch["input"]` tensor — raw, **pre-normalization**
  (`ModelModule.forward` normalizes internally via
  `self.normalizer.normalize_x(x)`, confirmed at
  `vendor/starcop/starcop/models/model_module.py:90-98`, so this is exactly
  the scale `/predict` expects — no separate normalization step needed) —
  squeeze the batch dimension, and POST the resulting `(C, H, W)` array
  as-is. Verified live, not just reasoned about: ran this exact recipe
  against `mag1c_only`'s real checkpoint and curated scene
  `ang20191018t141549_r3900_c244_w151_h151`, fed the resulting array through
  `src/serving/inference.py::predict_response` directly (in-process, same
  function `/predict` calls), and the resulting mask's sha256
  (`979f425a29123d95960da13dae0550d017dfaf4bf8f33e7514ef693d2f15184c`)
  **exactly matched** the digest `run_baseline_eval.py`'s offline pass
  already persisted to MLflow for that same scene — bit-for-bit, no
  tolerance needed on the mask side. This is the strongest evidence
  available short of an actual HTTP round-trip through a running
  `bentoml serve` process (which still needs testing once `live_verify.py`
  exists, since JSON serialization/deserialization and the HTTP layer
  itself remain unexercised by this check) that the array-construction
  assumption this whole phase rests on is correct.

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
citation, since it comes from an external publication and cannot be "generated" by
this project's pipeline; that hand-entry is correct, not a drift risk, and stays
that way unless the paper itself changes. That single hand entry lives in one
repo-committed file, `internal-docs/plans/paper_reference_metrics.md` (Phase 3) —
not an MLflow artifact — precisely so the MkDocs build can read it directly off
disk without needing MLflow connectivity or resolving which run's artifact copy is
current.

**Aggregation, keyed by variant.** Because Phase 4 evaluates all three variants as
separate runs (one `MLFLOW_RUN_ID` each), each variant's own `--emit-docs-assets
STAGING_DIR` call (Phase 1) only ever writes that variant's curated PNGs
(variant-prefixed filenames, so the three invocations sharing one `STAGING_DIR`
never collide) and a per-variant metrics fragment into that run-scoped staging
directory — it is not itself the cross-variant aggregation, and it never touches
the canonical directory `docs/results.md` reads from. Phase 4's final aggregation
task is: given the three captured run IDs, load each variant's per-variant
fragment/metrics from `STAGING_DIR`, join each row against the matching variant
entry in `paper_reference_metrics.md`, and write **one** combined Markdown table
fragment (all three variants, reproduction vs. paper-reported, in a single table)
into that same `STAGING_DIR`, replacing the three per-variant fragments rather
than leaving them for `docs/results.md` to merge. Only after Phase 4's
completeness check passes does the publish step atomically swap this staged
output into the canonical directory; `docs/results.md` pulls the combined
fragment from that canonical directory in via the `mkdocs-include-markdown-plugin`
(`{% include-markdown %}` — already used by `docs/changelog.md`, so this is a
proven mechanism here, not a new one). The reproduction side of the page is
mechanically tied to the last full flow run's aggregated output, not a manually
maintained copy; the paper-reported side is mechanically tied to
`paper_reference_metrics.md`'s one-time hand entry, not re-typed per page edit or
per variant. Because `--emit-docs-assets` is rejected in combination with `--limit`
(Phase 1), each per-variant call only ever runs as part of Phase 4's un-limited
flow, and the aggregation task itself only runs after all three variant
evaluations and their live checks have completed, so the fragment
`docs/results.md` includes can never come from a partial/smoke-test pass or from
only a subset of variants.

## Ordering

Phase 0 (env decision) → Phase 2 Step 0 (HF-hosting check) can run in parallel with
Phase 1's pure modules (`paper_metrics.py`, `select_docs_examples.py`, unit tests
only, no dependency on anything else) → Phase 2's registry import → Phase 1's glue
(full run, all three variants) → sanity-check the corrected numbers against the
paper before trusting anything downstream → Phase 3 (already happens as part of the
Phase 1 run) → **Phase 5's core (`live_verify.py`) built before Phase 4, despite
the phase numbering** (resequenced — see Phase 4's "Resolved implementation
details": Phase 4's flow calls Phase 5's live check as one of its own tasks, so
that entry point has to exist first; Phase 5 has no dependency on Phase 4, only
on Phase 3's artifacts) → Phase 4 (flow, one manual trigger, now wiring in a
real, already-working live check) → Phase 6 (docs) → `make docs-build --strict` + local
preview.

## Critical Files

- `vendor/starcop/starcop/validation.py` — `run_validation`, wrapped not edited; its
  `difficulty` bug (the pixel-count grouping) is what Phase 1's `paper_metrics.py`
  works around.
- `vendor/starcop/starcop/data/datamodule.py` — the `folder = root_folder/id` join
  logic `dataset_wiring.py` must replicate.
- `src/registry/hf_baseline_import.py` — reused for checkpoint loading/digest
  pinning; extended in Phase 2 with a source-dispatching `resolve_checkpoint`
  (HF vs. local) that both the registry import and the eval run call.
- `src/evaluation/run_baseline_eval.py` — `evaluate_variant()`'s checkpoint-loading
  call site also updated in Phase 2 (was hardcoded to the HF-only download path);
  this is what actually makes `varon` runnable, not just registered.
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
