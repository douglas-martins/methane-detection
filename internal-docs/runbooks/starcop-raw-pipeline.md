# How to re-run the `starcop_raw` DVC pipeline

Extracted from `internal-docs/plans/starcop-raw-pipeline.md`'s original "How to run it" section — genuine how-to content, kept separate from that file's design/journal narrative since credentials/paths make it an operational runbook, not a public reference. See that file for the full design rationale and 2026-08-09 real-run results this estimate is based on.

Estimate from that run: **~15 minutes** for the full DAG end-to-end (both datasets, all 4 stages) — safe to run in the background, no longer expected to take the original 50-90 minute estimate.

## 1. Kick off the real run, in the background

From the repo root, with Environment B's venv:

```bash
nohup .venv/bin/dvc repro > /tmp/dvc-repro-raw.log 2>&1 & echo PID: $!
```

(single line, no quotes — if your terminal shows a `dquote>`/`quote>` continuation prompt instead of running this, a quote character got mangled in the paste; press `Ctrl+C` to cancel the stuck prompt, then paste this exact line again.)

This reproduces the *whole* DAG (both `@starcop_mini` and `@starcop_raw` for all 4 stages) — expected, because `configs/data.yaml`'s content hash changed, so DVC correctly sees `starcop_mini`'s stages as stale too (mini's own stages are fast, seconds, so this doesn't meaningfully add to the runtime). If you'd rather scope it to just the new raw stages:

```bash
nohup .venv/bin/dvc repro normalize@starcop_raw split@starcop_raw patch_extract@starcop_raw stats@starcop_raw > /tmp/dvc-repro-raw.log 2>&1 & echo PID: $!
```

## 2. Watch progress

```bash
tail -f /tmp/dvc-repro-raw.log
```

You should see stage headers appear in order (`Running stage 'normalize@starcop_raw':`, then `split@starcop_raw`, then `patch_extract@starcop_raw` — this one sits at "0%|..." progress bars for a long stretch, that's expected — then `stats@starcop_raw`). If you started it detached and want to check on it later without `tail -f`:

```bash
ps -p <PID>          # still running?
tail -50 /tmp/dvc-repro-raw.log   # latest output
```

## 3. While `stats@starcop_raw` is running, watch its memory (verifies the OOM fix actually worked)

In a separate terminal, find the PID of the `stats.py` subprocess DVC spawned and sample its resident memory a few times over the run:

```bash
# find it (look for "stats.py" in the process list)
ps aux | grep "[s]tats.py dataset=starcop_raw"

# then sample RSS (in KB) a few times, minutes apart, using that PID
ps -o rss= -p <PID>
```

**What you're checking**: RSS should stay roughly flat (a few hundred MB, not growing) across the run. If it climbs steadily into multi-GB territory as the stage progresses, the incremental rewrite isn't working as intended and needs a second look — file size, band count, and running-total math are the places to check first (see `internal-docs/plans/starcop-raw-pipeline.md`'s `stats.py` Design section).

## 4. Once it finishes: read `missing_scenes.json`

```bash
cat data/processed/starcop_raw/selected/missing_scenes.json 2>/dev/null || echo "file doesn't exist -- 0 missing scenes, as expected"
```

`normalize.run()` only writes this file if `missing` is non-empty, so **no file at all is the expected, good outcome** — it means every `train.csv`/`test.csv` id resolved to a real folder, matching the reconciliation done in the plan doc's "What's different about `starcop_raw`" section. If the file *does* exist, open it — it's a JSON list of scene ids DVC's `normalize` stage couldn't find on disk, worth investigating (partial download? re-extraction changed folder names?) before trusting the rest of the run.

## 5. Sanity-check the pipeline's actual output shape

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('data/processed/starcop_raw/patches/train_tiled_128_128.csv')
print('train patches:', len(df))
print('has_plume True:', df['has_plume'].sum(), f\"({df['has_plume'].mean():.1%})\")
print(df.head(3))
"
```

Compare `len(df)` against the plan's back-of-envelope estimate (~142K train patches) — it won't match exactly (that number was extrapolated from mini's per-scene patch count, not measured), but it should be the same order of magnitude, not off by 10x.

Then load one row through the real dataset class, same shape check as mini's own TASK-1.2 verification:

```bash
.venv/bin/python -c "
import sys
sys.path.insert(0, 'src/data/preprocessing')
sys.path.insert(0, 'vendor/starcop')
from _vendor_starcop import STARCOPDataset
import pandas as pd
df = pd.read_csv('data/processed/starcop_raw/patches/train_tiled_128_128.csv')
ds = STARCOPDataset(df, input_products=['mag1c', 'TOA_AVIRIS_640nm', 'TOA_AVIRIS_550nm', 'TOA_AVIRIS_460nm'], output_products=['labelbinary'])
sample = ds[0]
print('input shape:', sample['input'].shape)   # expect (4, 128, 128)
print('output shape:', sample['output'].shape) # expect (1, 128, 128)
"
```

## 6. Read `band_stats.json` (the actual point of the `stats` stage)

```bash
cat data/processed/starcop_raw/stats/band_stats.json
```

Sanity checks, not exact-value checks (raw's real distribution wasn't computed ahead of time — that's the whole point of this stage): each band's `min`/`max` should roughly bracket its `BAND_NORMALIZATION` clip range in `_vendor_starcop.py` (a lot of raw values falling wildly outside it would suggest something upstream is wrong, not just a few flagged scenes — those are expected and already visible in `selected/range_check.json`), and no band's `std` should be `0.0` (a constant band across 3,767 scenes would be its own red flag).

## 7. Confirm idempotency

```bash
.venv/bin/dvc repro
```

Should print `Data and pipelines are up to date.` for every stage, `@starcop_mini` and `@starcop_raw` alike, with no output re-generated. If it re-runs anything, something's non-deterministic (check first: did `configs/data.yaml` or any dep file change between the two runs?).

## If something fails partway through

`dvc repro`'s per-stage caching means you don't lose completed stages — fix the code, then just re-run the same `dvc repro` command; DVC skips whatever's already up to date and resumes from the failing stage. `patch_extract@starcop_raw` failing partway through (e.g. a corrupt GeoTIFF in one scene) is the most likely failure point given it's the only stage touching all ~3,767 scenes' full label rasters; the traceback will name the offending scene folder under `data/processed/starcop_raw/selected/`.
