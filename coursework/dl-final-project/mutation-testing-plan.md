# Coursework Mutation Testing — standalone estimate

## Status: all 8 modules complete, 0 unexplained survivors

Every task below is done. Final state, verified with 5+ repeated full runs
after the `track_dependencies` fix (see below) for stability:

| Metric | Count |
|---|---:|
| Mutants generated (in-scope modules) | 2,169 |
| Killed | 1,159 |
| Equivalent/untestable, pragma'd with a reason | ~15 |
| No tests (documented thin glue: `main`/plotting/CLI functions) | 1,000 |
| **Survived, unexplained** | **0** |

`make coursework-test` and `make coursework-lint` both still pass (154
passed, 1 pre-existing skip).

## Why this is a separate document, not an extension of the repo-wide rollout

The repo-wide plan (`/mutation-testing-plan.md`, untracked) and
`.agents/rules/mutation-testing.md` both exclude `coursework/` from mutation
testing — `pyproject.toml`'s `[tool.mutmut] do_not_mutate` lists
`"coursework/*"` explicitly, and the rule says "Never add either
[`vendor/starcop/` or `coursework/`] to `source_paths` or `only_mutate`."
That exclusion stays untouched here; this project is a course deliverable
with its own isolated gates already (`make coursework-test`, `make
coursework-lint` — see the Makefile's "Coursework" block), deliberately
**not** wired into `test`/`lint`/`docstring-coverage`, and the whole folder +
that Makefile block are deleted once the course is graded (Makefile comment,
line ~23).

So this is a **standalone, disposable mutmut setup scoped only to
`coursework/dl-final-project/`**, run manually, never touching the root
`pyproject.toml`, root gate, or CI.

## Mechanics: how the standalone config stays isolated

mutmut reads config from **the current working directory only** — no
`--config` flag. It checks `./pyproject.toml`'s `[tool.mutmut]` first, and
falls back to `./setup.cfg`'s `[mutmut]` only if no `pyproject.toml` exists in
the CWD at all, or it exists without a `[tool.mutmut]` table (confirmed by
reading the installed `mutmut/configuration.py::_config_reader`). Since there
is no `pyproject.toml` in `coursework/dl-final-project/`, `setup.cfg` there is
read cleanly, and running mutmut with `cwd=coursework/dl-final-project` never
touches the root config.

`coursework/dl-final-project/setup.cfg` (tracked in this folder — commit it
if you want the config to persist for whoever else works on this folder):

```ini
[mutmut]
source_paths = .
also_copy =
    __tests__
    r2_manifest_train.csv
    r2_manifest_val.csv
only_mutate =
    early_stopping.py
    preprocessing.py
    sampling.py
    losses.py
    metrics.py
    pr_curve_plots.py
    architectures.py
    dataset.py
    eda.py
    evaluate.py
    train.py
do_not_mutate =
    __tests__/*
    confirm_raw.py
    build_r2_manifest.py
track_dependencies = False
pytest_add_cli_args_test_selection =
    __tests__/test_early_stopping.py
    ... (one entry per module above)
```

`mutants/` is already covered by the repo's global `.gitignore` (`mutants/`,
unanchored), so the generated copy never gets committed regardless of which
directory it's generated under.

Run it from within the folder, using the research env's interpreter (the same
one `coursework-test`/`coursework-train` already use):

```bash
cd coursework/dl-final-project
../../.venv/bin/mutmut run
../../.venv/bin/mutmut results
../../.venv/bin/mutmut show <mutant-id>   # inspect one survivor's diff
```

### Important finding: `track_dependencies` gave false "survived" verdicts

mutmut's default `track_dependencies = True` narrows each mutant's test run
to only the tests its coverage/dependency map thinks touch that line. On this
project that **misattributed real, killable mutants as "survived"** — e.g. a
mutation that dropped `random_state=seed` from a `.sample()` call was
reported as surviving, but manually patching the real source and running the
*same* existing test (`test_deterministic_given_same_seed`) failed
immediately, proving the test does catch it when actually run. Setting
`track_dependencies = False` (small test suite, this cost is negligible —
each full run is still under 30 seconds) made every previously-flaky
survivor resolve consistently across 5+ repeated runs. **If you see a
survivor here that looks like it should obviously be caught, don't trust
`mutmut show` alone** — hand-apply the diff to the real file and run the
specific test directly first, before writing a "kill" test or a pragma.

### Known blocker (fixed): `_REPO_ROOT = Path(__file__).resolve().parents[2]`

Five files (`dataset.py`, `train.py`, `evaluate.py`, `confirm_raw.py`,
`build_r2_manifest.py`) compute the real repo root by climbing a **fixed**
number of parents from `__file__`, to reach `configs/` or `data/processed/`
at the true repo root. mutmut's `mutants/` copy inserts one extra directory
level, so under the copy this fixed climb lands one level short.

**Fix applied** (test-only, no change to any module's runtime behavior): an
autouse fixture in `__tests__/conftest.py` (`_fix_dataset_repo_root_for_copied_trees`)
repatches `dataset._REPO_ROOT` to the real repo root, found by walking up from
conftest's own (also-copied) location until `configs/dataset/` exists,
instead of trusting a fixed-depth climb. `test_dataset.py`'s own hardcoded
`parents[3]` (in `test_channel_order_matches_the_dataset_config`) got the
same fix via the conftest helper. Only `dataset.py`'s `_load_dataset_config`
needed this — `train.py`/`evaluate.py` also use `_REPO_ROOT`, but their tests
never exercise that particular code path directly (mocked or not reached).

## Completed tasks

| # | Task | Result |
|---|---|---|
| 1 | `early_stopping.py` | 27/27 mutants killed. Added boundary/initial-state tests. |
| 2 | `preprocessing.py`, `sampling.py` | 64/64 killed. One equivalent mutant pragma'd (`replace=None` == `replace=False` in numpy). |
| 3 | `losses.py`, `metrics.py` | 286/286 killed. Two equivalent mutants pragma'd (a `zip(strict=True)` that can never see mismatched lengths; a redundant `p=0.5` matching a library default). |
| 4 | `pr_curve_plots.py` | 6/6 killed in the one tested function (`curve_points_from_artifact`); the rest is documented thin glue (matplotlib/mlflow), same pattern as `eda.py`. |
| 5 | `_REPO_ROOT` fix + `architectures.py`, `dataset.py` | 0 survivors. Conftest fixture added; 4 real gaps fixed (window bypass, index-reset, config-path case-sensitivity, rotation-transform coverage); ~6 equivalent kornia/numpy-default mutants either isolated to their own pragma'd line or the redundant argument removed outright. |
| 6 | `eda.py` | 0 survivors. Fixed a real min/max-scaling bug class (min-blind formula), a dtype-cast gap, 2 boundary conditions, and a flaky-vs-real distinction resolved by switching from output-randomness assertions to a `DataFrame.sample` call-spy. |
| 7 | `evaluate.py` | 0 survivors. Fixed 5 real gaps (sweep pooling across batches, `n_batches` accumulator, wall-clock elapsed-vs-summed timestamps, a test double that ignored its input, a threshold boundary). ~14 CUDA-only branches pragma'd as untestable on this CPU-only sandbox (no CUDA device available; this file already has its own `@pytest.mark.skipif(not torch.cuda.is_available())` precedent for the same limitation). |
| 8 | `train.py` | 0 survivors. Only `build_model` had zero coverage; added 2 tests using a builder-dict spy (avoids the real E2/E3 path's ImageNet weight download). |

## Effort actually spent vs. estimated

The original estimate (before doing the work) was 10-15.5 hours based on
extrapolating from the Phase 0 pilot. Actually completing it surfaced two
things the estimate didn't anticipate:

1. The `track_dependencies` false-negative issue (required for correctness,
   not visible until cross-checking a "survived" verdict by hand).
2. `train.py`/`evaluate.py`'s heavy existing test investment (669 + 239
   lines) meant far fewer real survivors than line count alone predicted —
   3 survivors in a 521-line file, all in one untested helper.

`confirm_raw.py` and `build_r2_manifest.py` remain excluded (`do_not_mutate`)
since neither has a test file — both are one-shot scripts run against real
`data/processed/starcop_raw/`, the same "thin glue, exercised via a real
run" shape as this repo's own `*/training/train.py` exclusion.
