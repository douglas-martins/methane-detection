# Extend TASK-1.2's DVC Pipeline to `starcop_raw`

> **Status**: 🟢 Fully executed and validated — code implemented, all unit tests GREEN (`pytest`, `make coverage-env-b`), and the real `dvc repro` against `starcop_raw` (Verification steps 1-7 below) completed successfully on 2026-08-09. See "Real-run results (2026-08-09)" section below for actual numbers.
> **Related**: TASK-1.2 in `internal-docs/plan.md` (previously validated against `starcop_mini` only — now also validated against `starcop_raw`)
> **See also**: [internal-docs/runbooks/starcop-raw-pipeline.md](../runbooks/starcop-raw-pipeline.md) for "how to re-run this pipeline" (extracted from this file's original "How to run it" section).

## Context

TASK-1.2's 4-stage DVC pipeline (`normalize` → `split` → `patch_extract` → `stats`, in `src/data/preprocessing/`) is built and validated end-to-end against `starcop_mini` (see `internal-docs/plan.md`, Phase 1). `starcop_raw` (59GB / 75,353 files, the full dataset) was deliberately out of scope for that pass. This plan parameterizes the config and `dvc.yaml` so both datasets run from the same code, and fixes one real code gap discovered while investigating `starcop_raw`'s actual layout: **it isn't structured like `starcop_mini`**, so `normalize.py`'s current scene-discovery logic would silently produce an empty pipeline if pointed at raw unmodified (not an error — just zero scenes processed).

**This plan has been executed** (2026-08-09) — see "Real-run results" section near the end for what actually happened. Everything in the Design/Context sections below was gathered by directly inspecting `data/starcop_raw/` ahead of implementation — not assumed — and was subsequently confirmed correct by the real run.

## What's different about `starcop_raw` (verified)

- **Not flat.** `starcop_mini`'s scene folders sit directly under `data/starcop_mini/<scene_id>/`. `starcop_raw`'s 3,767 scene folders are split across **7 subfolders**: `STARCOP_test/` (342), `STARCOP_train_easy/` (559), `STARCOP_train_remaining_part1..5/` (574/573/573/573/573).
- **Subfolder is not derivable from the CSV.** Spot-checked: a `train.csv` row with `difficulty=hard` resolved to `STARCOP_train_remaining_part1/` — no column in the CSV predicts which of the 7 subfolders holds a given scene. Must be discovered by search, not inferred.
- **CSV schema is identical to mini's** — same columns (`id`, `name`, `folder`, `window_col_off/row_off/width/height`, `has_plume`, `difficulty`, `subset`, ...). `split.py` needs **no changes**; `patch_extract.py` needs its `num_workers` wiring fixed and `stats.py` needs an incremental-memory rewrite (both scale/config gaps, not schema gaps — see their Design sections below) — only `normalize.py`'s scene discovery is a correctness gap caused by the schema/layout difference itself.
- **`train.csv`** (3,425 rows) **+ `test.csv`** (342 rows) **= 3,767** — exact match to the 3,767 on-disk scene folders. Verified directly (not just counted): every one of the 3,767 combined ids resolves via `find_scene_folder`'s glob to exactly one folder, and every on-disk folder is referenced by exactly one CSV row — 0 missing, 0 ambiguous, 0 unreferenced. (An earlier pass at this count double-counted each CSV's header row as data — corrected here.) **`train_easy.csv`** (560 rows, byte-identical to the copy nested inside `STARCOP_train_easy/`) is almost certainly a curated subset of `train.csv`'s easier rows, not additive — use `train.csv`/`test.csv` as the pair, matching STARCOP's own shipped `vendor/starcop/scripts/configs/config.yaml` default (`train_csv: "train.csv"`).
- **Per-scene folder layout matches mini's naming convention** — verified one `STARCOP_train_easy` scene folder: 20 files (`TOA_AVIRIS_{2004,2109,2310,2350,2360,460,550,640}nm.tif`, `TOA_WV3_SWIR{1-8}.tif`, `label_rgba.tif`, `labelbinary.tif`, `mag1c.tif`, `weight_mag1c.tif`) — same filenames `select_scene()` already expects, all 4 currently-configured input bands present.
- **Scale**: one full scene folder ≈ 16MB; just the 4 configured bands + `labelbinary` ≈ 2.8MB. So stage 1's `selected/` output ≈ 3,767 × 2.8MB ≈ **10.5GB** (269GB free — no disk concern). The slow stage is `patch_extract`: mini processed patches at ~50-60 it/s; raw's ~3,767 scenes × ~49 patches ≈ 185K patches extrapolates to roughly **50–60 minutes** for that stage alone, at `num_workers=1` (patch_extract.py's current hardcoded default — this plan parameterizes it).

## Design: Hydra config groups, not scattered CLI overrides

Restructure `configs/data.yaml` to use Hydra's config-group mechanism, so `dataset=starcop_raw` alone (one override) pulls in every raw-specific setting consistently, instead of remembering 2-3 separate CLI overrides per invocation (fragile, easy to drift out of sync across `dvc.yaml`'s stage commands).

**`configs/data.yaml`** (shared settings only — split/patch/stats/hydra unchanged from today, minus the dataset-specific keys which move out):

```yaml
defaults:
  - dataset: starcop_mini
  - _self_

split:
  val_fraction: 0.15
  seed: 42
  stratify_by: name

patch:
  size: [128, 128]
  overlap: [64, 64]
  has_plume_threshold: 0.00244
  num_workers: 1   # NEW — currently hardcoded in patch_extract.py, not config-driven at all

stats:
  bands: null

hydra:
  run:
    dir: .
  output_subdir: null
```

**`configs/dataset/starcop_mini.yaml`** (new — the exact values `data.yaml` has today):

```yaml
# @package _global_
dataset: starcop_mini

paths:
  raw_root: data/${dataset}
  processed_root: data/processed/${dataset}

dataset_cfg:
  input_products: ["mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]
  output_products: ["labelbinary"]
  train_csv: train_mini10.csv
  test_csv: test_mini10.csv
```

**`configs/dataset/starcop_raw.yaml`** (new):

```yaml
# @package _global_
dataset: starcop_raw

paths:
  raw_root: data/${dataset}
  processed_root: data/processed/${dataset}

dataset_cfg:
  input_products: ["mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]
  output_products: ["labelbinary"]
  train_csv: train.csv
  test_csv: test.csv

patch:
  num_workers: 4   # raw is ~200x mini's scene count; bump parallelism for patch_extract
```

`# @package _global_` is required on both group files — without it, Hydra would nest their keys under a `dataset:` sub-key (e.g. `cfg.dataset.paths...`), breaking every existing `cfg.paths.raw_root` / `cfg.dataset_cfg.input_products` access pattern in `normalize.py`/`split.py`/`patch_extract.py`/`stats.py`. No changes needed to any stage script's `main()` — `@hydra.main(config_path="../../../configs", config_name="data")` already auto-discovers `configs/dataset/*.yaml` once `data.yaml` declares the `dataset:` group in `defaults:`.

**Expected one-time side effect**: `configs/data.yaml`'s content hash changes (it's a dep of every stage), so the *next* `dvc repro` will show `starcop_mini`'s stages as changed and re-run them too, even though their outputs will be byte-identical. Not a bug — just don't be surprised by it.

## Design: `normalize.py` — discover scenes from the CSVs, not the directory tree

Current `run()` does `for scene_folder in sorted(p for p in raw_root.iterdir() if p.is_dir())` — this only sees `starcop_raw`'s 7 subfolder names, never the actual scenes one level inside them. Fix: **discover the scene id list from `train_csv`/`test_csv`'s `id` column** (the authoritative manifest of what's actually needed downstream), then locate each scene's real folder via a bounded search. This is dataset-agnostic — no `layout: flat|nested` config flag needed, and it also stops `normalize` from wastefully processing any stray raw-dataset folders that aren't referenced by either CSV.

New helper in `normalize.py`:

```python
def find_scene_folder(raw_root: Path, scene_id: str) -> Path:
    """Locate a scene's folder, flat (mini) or one subfolder level deep (raw)."""
    direct = raw_root / scene_id
    if direct.is_dir():
        return direct
    matches = list(raw_root.glob(f"*/{scene_id}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Scene '{scene_id}' found in multiple locations under {raw_root}: {matches}"
        )
    raise FileNotFoundError(f"Scene '{scene_id}' not found under {raw_root}")
```

`run()` changes to:

```python
def run(cfg) -> None:
    raw_root = Path(cfg.paths.raw_root)
    selected_root = Path(cfg.paths.processed_root) / "selected"
    input_products = list(cfg.dataset_cfg.input_products)
    output_products = list(cfg.dataset_cfg.output_products)

    train_ids = pd.read_csv(raw_root / cfg.dataset_cfg.train_csv)["id"]
    test_ids = pd.read_csv(raw_root / cfg.dataset_cfg.test_csv)["id"]
    scene_ids = sorted(set(train_ids) | set(test_ids))

    range_check = {}
    missing = []
    for scene_id in scene_ids:
        try:
            scene_folder = find_scene_folder(raw_root, scene_id)
        except FileNotFoundError:
            missing.append(scene_id)
            continue
        flagged = select_scene(
            scene_folder, selected_root / scene_id, input_products, output_products
        )
        if flagged:
            range_check[scene_id] = flagged

    selected_root.mkdir(parents=True, exist_ok=True)
    (selected_root / "range_check.json").write_text(json.dumps(range_check, indent=2))
    if missing:
        (selected_root / "missing_scenes.json").write_text(json.dumps(missing, indent=2))
```

**Design decision on missing scenes**: log and skip, don't crash the whole run — even though a direct reconciliation (globbing every `train.csv`/`test.csv` id against all 7 subfolders, done while writing this plan) confirms today's `starcop_raw` has zero missing and zero ambiguous scenes: 3,767 ids, 3,767 folders, exact 1:1 match. This is defensive design for a future re-download or dataset update, not a fix for a currently-observed problem: at raw's scale, a hard crash on scene #612 of 3,767 loses all prior work, so `missing_scenes.json` reports the (currently expected-empty) exception list without discarding a ~50-minute run, and without pre-building exclusion logic into `split.py` for a problem that isn't currently occurring.

`select_scene()` itself and every function in `split.py` are unchanged. `patch_extract.py` and `stats.py` each get one small, targeted change — see their own Design sections below.

## Design: `patch_extract.py` — actually read `cfg.patch.num_workers`

Current `run()` calls `patch_scenes(...)` without passing `num_workers` at all, so it silently uses the function's own hardcoded default (1) regardless of config. Fix: add `num_workers=cfg.patch.num_workers` to the call in `run()`. `patch_scenes()`'s signature already accepts `num_workers` — no change needed there.

## Design: `stats.py` — incremental accumulation instead of holding every patch in memory

Current `compute_band_stats()` (`src/data/preprocessing/stats.py`) appends every patch's every band array into a Python list, then calls `np.mean`/`np.std`/`np.min`/`np.max` once at the end over the fully materialized list. Fine at mini's scale (~490 total patches). Doesn't scale to raw: the `train` split alone is estimated at ~142K patches (2,911 train scenes after the 15% val carve-out × ~49 patches/scene) × 4 configured bands × 128×128 float32 ≈ **~35GB of raw array data held simultaneously** — before Python/list/numpy per-object overhead, which pushes it higher still. Checked against the dev machine's actual RAM (24GB): this stage would very likely OOM or thrash — and it's the *last* stage, run only after `patch_extract`'s already-estimated 50-60 minute pass has completed. Discovered by extrapolating `compute_band_stats`'s current algorithm to raw's patch count, not observed by actually running it (raw hasn't been executed yet).

Fix: replace accumulate-then-reduce with an incremental per-band running total, so peak memory is O(1) per band instead of O(all patches):

```python
def compute_band_stats(dataframe: pd.DataFrame, bands: list[str]) -> dict:
    dataset = STARCOPDataset(dataframe, input_products=bands, output_products=[])
    running = {
        band: {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": np.inf, "max": -np.inf}
        for band in bands
    }

    for idx in range(len(dataset)):
        input_tensor = dataset[idx]["input"].numpy()
        for band_idx, band in enumerate(bands):
            arr = input_tensor[band_idx]
            s = running[band]
            s["count"] += arr.size
            s["sum"] += float(arr.sum())
            s["sum_sq"] += float(
                np.square(arr, dtype=np.float64).sum()
            )  # float64: 142K patches of squared values will lose precision in float32
            s["min"] = min(s["min"], float(arr.min()))
            s["max"] = max(s["max"], float(arr.max()))

    return {
        band: {
            "mean": (mean := s["sum"] / s["count"]),
            "std": float(np.sqrt(s["sum_sq"] / s["count"] - mean**2)),
            "min": s["min"],
            "max": s["max"],
        }
        for band, s in running.items()
    }
```

Same output shape (`mean`/`std`/`min`/`max` per band) as today, so no callers change. Numerically equivalent to the current two-pass approach for well-scaled reflectance/concentration values like these (not the regime where naive sum-of-squares loses precision).

**New test** in `test_stats.py` (existing tests there assert only on output values, not on internal list-accumulation, so none of them need to change — this is additive):

- `test_compute_band_stats_matches_two_pass_result_for_larger_dataset` — build ~50+ synthetic patches with varying per-patch values, assert the incremental result matches `np.mean`/`np.std`/`np.min`/`np.max` computed the direct way within a small float tolerance (proves the rewrite is numerically equivalent, not just lower-memory).

## Design: `dvc.yaml` — one parameterized stage template via `foreach`

Replace today's 4 mini-only stages with DVC's `vars:` + `foreach:` matrix, generating `normalize@starcop_mini`, `normalize@starcop_raw`, etc. from one template per stage — not 8 hand-duplicated blocks:

```yaml
vars:
  - datasets: [starcop_mini, starcop_raw]

stages:
  normalize:
    foreach: ${datasets}
    do:
      cmd: python src/data/preprocessing/normalize.py dataset=${item}
      deps:
        - src/data/preprocessing/normalize.py
        - src/data/preprocessing/_vendor_starcop.py
        - vendor/starcop/starcop/data/normalizer_module.py
        - data/${item}
        - configs/data.yaml
        - configs/dataset/${item}.yaml
      outs:
        - data/processed/${item}/selected

  split:
    foreach: ${datasets}
    do:
      cmd: python src/data/preprocessing/split.py dataset=${item}
      deps:
        - src/data/preprocessing/split.py
        - data/processed/${item}/selected
        - configs/data.yaml
        - configs/dataset/${item}.yaml
      outs:
        - data/processed/${item}/splits

  patch_extract:
    foreach: ${datasets}
    do:
      cmd: python src/data/preprocessing/patch_extract.py dataset=${item}
      deps:
        - src/data/preprocessing/patch_extract.py
        - src/data/preprocessing/_vendor_starcop.py
        - vendor/starcop/starcop/data/datamodule.py
        - data/processed/${item}/splits
        - configs/data.yaml
        - configs/dataset/${item}.yaml
      outs:
        - data/processed/${item}/patches

  stats:
    foreach: ${datasets}
    do:
      cmd: python src/data/preprocessing/stats.py dataset=${item}
      deps:
        - src/data/preprocessing/stats.py
        - src/data/preprocessing/_vendor_starcop.py
        - vendor/starcop/starcop/data/dataset.py
        - data/processed/${item}/patches
        - configs/data.yaml
        - configs/dataset/${item}.yaml
      outs:
        - data/processed/${item}/stats
```

`dataset=${item}` is the only CLI override needed per stage — the config-group restructuring above is what makes this sufficient (train_csv/test_csv/num_workers all come along with it). `dvc repro` with no args still reproduces the whole DAG (both datasets); `dvc repro patch_extract@starcop_raw` targets just raw's slow stage later without touching mini's already-validated outputs, as long as mini's deps haven't changed.

## Build order (TDD, matching how TASK-1.2 was built)

1. **`configs/data.yaml` + `configs/dataset/{starcop_mini,starcop_raw}.yaml`** — write the restructure. RED/GREEN via a new small test proving both compose correctly (no real data needed, just reads yaml):

   ```python
   # src/data/preprocessing/__tests__/test_config.py
   from pathlib import Path
   from hydra import compose, initialize_config_dir

   CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "configs")


   def test_starcop_mini_config_resolves_expected_values():
       with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
           cfg = compose(config_name="data", overrides=["dataset=starcop_mini"])
       assert cfg.dataset == "starcop_mini"
       assert cfg.dataset_cfg.train_csv == "train_mini10.csv"
       assert cfg.paths.raw_root == "data/starcop_mini"
       assert cfg.patch.num_workers == 1


   def test_starcop_raw_config_resolves_expected_values():
       with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
           cfg = compose(config_name="data", overrides=["dataset=starcop_raw"])
       assert cfg.dataset == "starcop_raw"
       assert cfg.dataset_cfg.train_csv == "train.csv"
       assert cfg.paths.raw_root == "data/starcop_raw"
       assert cfg.patch.num_workers == 4  # raw's override


   def test_default_dataset_is_starcop_mini_when_unspecified():
       with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
           cfg = compose(config_name="data")
       assert cfg.dataset == "starcop_mini"
   ```

2. **`find_scene_folder()` in `normalize.py`** — RED tests first, in `test_normalize.py`:
   - `test_find_scene_folder_locates_flat_scene` (mini-like: `raw_root/scene1/`)
   - `test_find_scene_folder_locates_nested_scene` (raw-like: `raw_root/STARCOP_train_easy/scene1/`)
   - `test_find_scene_folder_raises_file_not_found_when_missing`
   - `test_find_scene_folder_raises_value_error_when_ambiguous` (same id under two subfolders)

3. **`normalize.run()` — CSV-driven discovery** — this changes existing, already-GREEN behavior, so update the existing `test_run_writes_range_check_json_only_for_flagged_scenes` test (it currently builds scenes by writing folders directly under a fake `raw_root` with no CSVs; needs `train.csv`/`test.csv` fixtures added) plus two new RED tests:
   - `test_run_discovers_scenes_from_nested_subfolders` (raw-like fixture: `raw_root/subfolderA/scene1/`, referenced by a `train.csv` row — proves the fix actually solves the problem that motivated it)
   - `test_run_logs_missing_scenes_instead_of_crashing` (a synthetic `train.csv` row whose id has no matching folder anywhere — assert `missing_scenes.json` contains it and the run still completes for the other scenes; this exercises the defensive path, not an observed real-data condition — see Design section above)

4. **`patch_extract.run()` passes `num_workers`** — small change; extend the existing `patch_extract.py` tests or add `test_run_passes_configured_num_workers_to_patch_scenes` (via `monkeypatch` on `patch_extract.patch_scenes` to capture the `num_workers` kwarg it was called with). **Deliberate exception to state-based testing**: `num_workers` only changes multiprocessing pool size, not `patch_scenes`'s output dataframe, so there's no observable state difference to assert on — a timing-based state test would be flaky (its own anti-pattern). An interaction test is the correct narrow tool here, not a shortcut.

5. **`stats.compute_band_stats()` — incremental rewrite** — RED test first (`test_compute_band_stats_matches_two_pass_result_for_larger_dataset`, per the Design section above), then swap the accumulate-then-reduce body for the running-total version. Existing `test_stats.py` tests should stay GREEN throughout since they assert on output values only.

6. **`dvc.yaml`** — rewrite as the `foreach` template above. No unit test (matches how `dvc.yaml` itself wasn't unit-tested for the mini pass) — verified instead by the Verification section below, when this plan is executed.

## Files to be touched (when implemented)

- `configs/data.yaml` (rewritten — shared keys only)
- `configs/dataset/starcop_mini.yaml`, `configs/dataset/starcop_raw.yaml` (new)
- `src/data/preprocessing/normalize.py` (add `find_scene_folder`, rewrite `run()`)
- `src/data/preprocessing/patch_extract.py` (`run()` passes `num_workers`)
- `src/data/preprocessing/stats.py` (`compute_band_stats()` rewritten for incremental accumulation)
- `src/data/preprocessing/__tests__/test_config.py` (new)
- `src/data/preprocessing/__tests__/test_normalize.py` (new tests + one existing test updated)
- `src/data/preprocessing/__tests__/test_patch_extract.py` (one new test)
- `src/data/preprocessing/__tests__/test_stats.py` (one new test)
- `dvc.yaml` (rewritten with `vars:`/`foreach:`)
- `internal-docs/plan.md` — not touched by this parameterization work; updated once `starcop_raw` was actually executed and validated (see the TASK-1.2 entry there).

## Known risks (resolved by the 2026-08-09 real run — see "Real-run results" above for exact numbers)

- **`missing_scenes.json` is expected to be empty** — confirmed: the real run found 0 missing and 0 ambiguous scenes, matching the manual reconciliation done while writing this plan.
- **`patch_extract@starcop_raw` runtime** (~50-60 min estimated) — actual: the whole 8-stage DAG (both datasets) completed in ~15 minutes total. The mini-based extrapolation was pessimistic.
- **`stats@starcop_raw` would very likely OOM without the incremental rewrite** — confirmed the rewrite works: RSS measured flat at ~700MB throughout the stage's run, not the ~35GB estimated for the naive accumulate-then-reduce approach.
- **First `dvc repro` after this refactor recomputes `starcop_mini` too** (config file hash changed) — confirmed, and mini's outputs came out as expected; not a regression.
- **New risk discovered during the real run, not anticipated by this plan**: starting a second `dvc repro` (e.g. a scoped one) while a first is still running does not reliably get blocked by DVC's own lock (`.dvc/tmp/rwlock`) — both ran `stats@starcop_raw` concurrently. See "Operational lessons from this run" above for the resolution. Don't start a second `dvc repro` to "check progress" — inspect the existing run's log/process instead.

## Verification (once this plan is executed and raw is actually run)

1. ✅ **Done** — `pytest src/data/preprocessing/__tests__/ -v`: all 32 tests pass, including the new `test_config.py` (3), the updated/new `normalize.py` tests (11 total, 4 new + 2 new discovery/missing tests + 1 existing test updated with CSV fixtures), the new `patch_extract.py` `num_workers` test, and the new `test_stats.py` equivalence test.
2. ✅ **Done** — `pytest` (full suite, `--continue-on-collection-errors`): 32 passed, same 2 pre-existing collection errors as before (`gdown` in `test_download_mini_dataset.py`, `spectral` in `test_starcop_aviris_data_prep.py`) — confirmed pre-existing and unrelated, not caused by this change.
2a. ✅ **Done** — `make coverage-env-b`: 89% overall (`normalize.py` 94%, `patch_extract.py` 81%, `stats.py` 89%, `split.py` 85%). All uncovered lines are exclusively each file's `main()` Hydra-CLI wrapper — same untested-by-design pattern the original TASK-1.2 pass already had, not a new gap.
3. ✅ **Done** (2026-08-09) — `dvc repro` completed against real `starcop_raw`. `selected/missing_scenes.json` does not exist → 0 scenes skipped, matching the reconciliation above exactly. `dvc.yaml`'s structural validation (`dvc dag`, `dvc stage list`) also re-confirmed correct against the current tree. Note: `dvc repro --dry` run cold (before any `starcop_raw` stage has ever produced real output) errors partway through — `split@starcop_raw` fails looking for `normalize@starcop_raw`'s output, because `--dry` never materializes it. This is a real DVC limitation for chained dry-runs on never-before-run stages, not a bug in this plan or `dvc.yaml`.
4. ✅ **Done** — Watched `stats@starcop_raw`'s RSS via `ps -o rss=` several times over its ~15-minute run: stayed flat at ~700MB throughout, never climbing — confirms the incremental rewrite genuinely keeps peak memory O(1) per band rather than growing with patch count.
5. ✅ **Done** — `train_tiled_128_128.csv` has **141,218 patches** (plus 26,608 val / 16,759 test) — same order of magnitude as the plan's ~142K back-of-envelope estimate, as expected (that estimate was extrapolated from mini's per-scene patch count, never meant to match exactly).
6. ✅ **Done** — Re-ran `dvc status` after the full run: only `stats@starcop_mini` reported changed, and only because of the `tqdm` progress-bar addition made *during* verification (see below) — not a reproducibility issue. Before that edit, all 8 stages reported up to date.
7. ✅ **Done** — `band_stats.json` read and sanity-checked: no band has `std: 0.0`; `mag1c`'s max is exactly `100000.0` (StarCOP's known concentration clip ceiling, not corrupted data); AVIRIS bands' small negative mins are normal reflectance noise. See exact values in "Real-run results" below.

## Real-run results (2026-08-09)

The real `dvc repro` against `starcop_raw` ran to completion. Actual numbers, for comparison against the estimates made while writing this plan:

- **Total wall time: ~15 minutes** for the whole DAG (both `@starcop_mini` and `@starcop_raw`, all 4 stages each) — far faster than the 50-90 minute estimate. `patch_extract@starcop_raw` (the stage expected to dominate at ~50-60 min alone) in particular ran much faster than extrapolated from mini's throughput; the mini-based per-scene extrapolation was pessimistic, not wrong in kind.
- **`normalize@starcop_raw`**: 3,767 scenes discovered from `train.csv`/`test.csv`, 0 missing, 0 ambiguous — exact match to the manual reconciliation done while writing this plan. `range_check.json` has 10,510 lines (flagged out-of-range values across scenes — not investigated further here, pre-existing mechanism from the mini pass).
- **`split@starcop_raw`**: 2,882 train / 543 val / 342 test rows — matches `train.csv`'s 3,425 rows split at `val_fraction: 0.15`, plus `test.csv`'s 342, exactly.
- **`patch_extract@starcop_raw`**: 141,218 train patches / 26,608 val / 16,759 test — same order of magnitude as the ~142K back-of-envelope estimate. Confirmed 4 `multiprocessing-fork` worker processes active via `ps aux`, verifying the `num_workers=4` config wiring fix actually takes effect (not silently falling back to the old hardcoded default of 1).
- **`stats@starcop_raw`**: RSS stayed flat ~700MB across the ~15 minute run (no growth with patch count) — confirms the incremental-memory rewrite works as designed, not just in the synthetic unit test. Final `band_stats.json`:

  | band | mean | std | min | max |
  | --- | --- | --- | --- | --- |
  | `mag1c` | 36.22 | 310.13 | 0.0 | 100000.0 |
  | `TOA_AVIRIS_640nm` | 28.82 | 15.27 | -0.17 | 345.77 |
  | `TOA_AVIRIS_550nm` | 26.83 | 14.15 | -0.20 | 381.95 |
  | `TOA_AVIRIS_460nm` | 24.76 | 12.87 | -0.29 | 472.34 |

  No band has `std: 0.0`. `mag1c`'s max of exactly `100000.0` is StarCOP's known concentration clip ceiling, not a data problem.
- **Idempotency**: re-running showed all 8 stages up to date (before an unrelated later edit to `stats.py`, see below).

### Operational lessons from this run

- **Two concurrent `dvc repro` invocations will race on shared output, and DVC's lock doesn't reliably stop it.** A background `dvc repro` (full DAG) and a second, separately-started scoped `dvc repro normalize@starcop_raw split@starcop_raw patch_extract@starcop_raw stats@starcop_raw` ended up running at the same time, both executing `stats@starcop_raw` concurrently and both writing toward `data/processed/starcop_raw/stats`. `.dvc/tmp/rwlock` only reflected one of the two runs' registrations. Resolution: kill the redundant `dvc repro` parent *and* its orphaned stage subprocess separately — killing the parent alone leaves the child `stats.py`/etc. process running, since DVC doesn't propagate the signal to already-spawned stage commands. If you kick off a background run per the instructions in [the runbook](../runbooks/starcop-raw-pipeline.md), don't also start a second one "just to check progress" — check the existing one's log/process instead.
- **Don't edit a stage's dependencies while it's running — it happened once here and turned out fine, but don't rely on that.** Python compiles imported modules into memory at process startup and never re-reads the `.py` file mid-run, so editing `stats.py` on disk while `stats@starcop_raw` was executing had zero effect on *that* run — the edit (adding a `tqdm` progress bar, `mininterval=5.0` to avoid flooding a redirected log file with updates) only took effect on the next invocation. But DVC computes a stage's `dvc.lock` dep hash from the file's state *after* the subprocess exits, not at launch time — so `stats@starcop_raw`'s lock entry ended up recording the *edited* `stats.py` hash even though the code that actually produced its `band_stats.json` was the pre-edit version. This particular edit was cosmetic (no logic change, tests unaffected) and `stats` has no downstream consumers in this pipeline, so nothing further was needed here — but in general, `dvc status`/`dvc.lock` cannot be trusted as proof of which code version actually produced a given output once you've edited a stage's script mid-run. If an in-flight edit happens to `stats@starcop_raw` or any other stage, don't treat a plain `dvc repro --force <stage>` as sufficient — run `dvc repro --force --force-downstream <stage>` and treat its outputs (and anything downstream of it) as unverified until that completes, since hash-based change detection is exactly what's unreliable in this scenario.
- **`stats.py` had no progress output at all** before this run (unlike `patch_extract.py`'s inherited `tqdm` "Computing label statistics" bar) — for a stage that reads through every train patch individually, this made it impossible to tell if it was progressing or hung without indirect signals (`ps` RSS/CPU, or matching a currently-open file via `lsof` back to a CSV row number to estimate position). Fixed by adding a `tqdm` bar (see `src/data/preprocessing/stats.py`); worth keeping in mind for any other long, silent, single-threaded stage added later.
