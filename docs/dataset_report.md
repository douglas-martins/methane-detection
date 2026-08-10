# STARCOP Dataset Report

> Generated for TASK-1.3 from the DVC pipeline outputs in `data/processed/{starcop_mini,starcop_raw}/` (stages: `normalize` → `split` → `patch_extract` → `stats` → `coordinates`, see `dvc.yaml`). Covers both tracked datasets side by side.
> All pixel-level and per-band statistics below are computed on **`train` split patches only** (`patches/train_tiled_128_128.csv`), matching what the model actually trains on — not the full scene set.
> Generated: 2026-08-09.

## 1. Dataset overview

| | `starcop_mini` | `starcop_raw` |
|---|---|---|
| Scenes discovered (train.csv ∪ test.csv) | 18 | 3,767 |
| Scenes missing on disk | 0 | 0 |
| Train / val / test scene-rows (after split) | 8 / 1 / 9 | 2,882 / 543 / 342 |
| Train / val / test patches (128×128, 64 overlap) | 392 / 49 / 441 | 141,218 / 26,607 / 16,758 |

Scene counts come from `data/processed/<dataset>/selected/` (stage 1 output — every scene referenced by `train.csv`/`test.csv` that was actually found and validated). 0 missing scenes for either dataset — no `missing_scenes.json` was written by `normalize.py` for `starcop_mini`; `starcop_raw` explicitly confirmed 0/3,767 missing.

## 2. Sensor breakdown

| Sensor | `starcop_mini` | `starcop_raw` |
|---|---|---|
| AVIRIS-NG | 18 (100%) | 3,767 (100%) |
| EMIT | 0 (0%) | 0 (0%) |

Every scene ID in both datasets' manifests is `ang<timestamp>...` (AVIRIS-NG). **This is expected, not a data gap**: per `vendor/starcop/README.md`, STARCOP's "full annotated training and evaluation dataset" — what `data/starcop_mini`/`data/starcop_raw` are — is AVIRIS-NG only, sourced from the 2019 Permian Basin survey. EMIT only appears in STARCOP as a small **zero-shot generalization demo** (a separate Colab notebook, `inference_on_raw_EMIT_nc_file.ipynb`, downloading its own small annotated EMIT subset) — it was never part of the bulk training dataset release and was correctly not pulled down here. The plan document's Dataset Registry entry (`mlops-methane-detection-plan.md` §7) listing "AVIRIS-NG, EMIT, WorldView-3" as STARCOP's sensors describes the paper's full scope, not what this project actually downloaded for training; see the correction in that section.

## 3. Geographic coverage

Per-scene coordinates (centroid of each scene's reprojected WGS84 bounds) are in `data/processed/<dataset>/coordinates/scene_coordinates.csv`.

| | `starcop_mini` | `starcop_raw` |
|---|---|---|
| Latitude range | 31.36° – 32.40° N | 31.08° – 34.22° N |
| Longitude range | -104.06° – -101.72° W | -104.76° – -101.23° W |

Both datasets sit entirely within the Permian Basin (West Texas / southeastern New Mexico, USA) — consistent with STARCOP's AVIRIS-NG source survey. `starcop_raw`'s wider spread reflects its ~209x larger scene count covering more of the same survey area, not a different region.

### `starcop_mini` full scene list (18 scenes)

| Scene ID | Lat | Lon |
|---|---|---|
| ang20190923t174142_r5826_c168_w151_h151 | 32.1088 | -103.7698 |
| ang20190924t183641_r14137_c154_w151_h151 | 31.6463 | -101.7645 |
| ang20191005t210402_r9366_c472_w151_h151 | 31.8606 | -101.7923 |
| ang20191005t221554_r5023_c432_w151_h151 | 32.1547 | -101.9295 |
| ang20191008t151045_r5097_c477_w151_h151 | 32.1563 | -101.9323 |
| ang20191010t155034_r16931_c325_w151_h151 | 31.3630 | -101.7751 |
| ang20191011t165345_r4367_c196_w151_h151 | 31.9784 | -104.0630 |
| ang20191011t165345_r5135_c169_w151_h151 | 31.9345 | -104.0303 |
| ang20191011t174241_r4837_c234_w151_h151 | 32.0195 | -103.9451 |
| ang20191018t141549_r3900_c244_w151_h151 | 32.2399 | -101.7846 |
| ang20191018t144405_r2674_c436_w151_h151 | 32.3262 | -101.8128 |
| ang20191018t163108_r14851_c525_w151_h151 | 31.5042 | -101.7231 |
| ang20191018t165503_r2660_c460_w151_h151 | 32.3247 | -101.8128 |
| ang20191018t181457_r4349_c389_w151_h151 | 32.2116 | -101.9352 |
| ang20191018t190719_r1941_c33_w151_h151 | 32.3738 | -101.8449 |
| ang20191018t190719_r2696_c420_w151_h151 | 32.3239 | -101.8117 |
| ang20191021t191828_r4300_c359_w151_h151 | 32.4030 | -103.3848 |
| ang20191025t165545_r6573_c24_w151_h151 | 31.5025 | -101.7237 |

`starcop_raw`'s 3,767-row list is not reproduced here — see `data/processed/starcop_raw/coordinates/scene_coordinates.csv`.

## 4. Class distribution (methane vs. background)

Pixel-level counts over `labelbinary`, computed via `stats.compute_class_distribution` on the train-split patches:

| | `starcop_mini` | `starcop_raw` |
|---|---|---|
| Positive (methane) pixels | 72,764 | 7,333,884 |
| Background pixels | 6,349,764 | 2,306,381,828 |
| Total pixels | 6,422,528 | 2,313,715,712 |
| Positive fraction | 1.13% | 0.32% |
| Imbalance ratio (background : methane) | ~87 : 1 | ~314 : 1 |

`starcop_raw`'s imbalance is substantially more severe than `starcop_mini`'s — the mini demo subset was curated toward plume-visible scenes, while the full dataset includes many more `difficulty=hard`/plume-free patches. **Any loss function or sampling strategy tuned against `starcop_mini` will underestimate the imbalance actually present at `starcop_raw` scale** — worth flagging for Phase 3 (training) as a concrete design input, not just a descriptive stat.

## 5. Per-band statistics

Computed over the same input bands used for training (`mag1c`, `TOA_AVIRIS_640nm`, `TOA_AVIRIS_550nm`, `TOA_AVIRIS_460nm`), train-split patches only.

**`starcop_mini`**

| Band | Mean | Std | Min | Max |
|---|---|---|---|---|
| mag1c | 111.56 | 542.46 | 0.0 | 41,640.05 |
| TOA_AVIRIS_640nm | 30.90 | 22.69 | 0.0 | 215.50 |
| TOA_AVIRIS_550nm | 29.95 | 23.88 | 0.0 | 240.94 |
| TOA_AVIRIS_460nm | 28.70 | 24.41 | 0.0 | 243.70 |

**`starcop_raw`**

| Band | Mean | Std | Min | Max |
|---|---|---|---|---|
| mag1c | 36.22 | 310.13 | 0.0 | 100,000.0 |
| TOA_AVIRIS_640nm | 28.82 | 15.27 | -0.17 | 345.77 |
| TOA_AVIRIS_550nm | 26.83 | 14.15 | -0.20 | 381.95 |
| TOA_AVIRIS_460nm | 24.76 | 12.87 | -0.29 | 472.33 |

Notes:
- `mag1c`'s `starcop_raw` max is exactly `100000.0` — STARCOP's known concentration clip ceiling, not corrupted data (confirmed in `starcop-raw-pipeline-plan.md`).
- The small negative mins on `starcop_raw`'s TOA reflectance bands are normal sensor noise, not a data quality issue.
- `starcop_mini`'s substantially higher `mag1c` mean/std reflects its curation toward plume-heavy scenes (consistent with §4's lower imbalance ratio); `starcop_raw`'s broader, more representative scene mix pulls both down.

## 6. Data quality issues

Sourced from `normalize.py` (stage 1)'s per-scene validation:

| | `starcop_mini` | `starcop_raw` |
|---|---|---|
| Scenes with NaN/Inf in a configured band | 0 (hard failure — `dvc repro` would have aborted) | 0 |
| Scenes missing on disk | 0 | 0 |
| Scenes with ≥1 band flagged outside its normalization clip range | 18 / 18 (100%) | 2,972 / 3,767 (78.9%) |

Full detail: `data/processed/<dataset>/selected/range_check.json`. These are **flags, not rejections** — `mag1c` legitimately exceeds STARCOP's train-time clip range near real methane plumes; STARCOP's own `DataNormalizer` clips these values at train time (see `normalize.py`'s docstring). No scenes are recommended for exclusion based on this signal alone. `starcop_mini` being flagged at 100% vs. `starcop_raw`'s 79% is consistent with `starcop_mini`'s curation toward plume-visible scenes (§4).

## 7. Recommendations for Phase 3 (training)

- Use `starcop_raw`'s class distribution (§4), not `starcop_mini`'s, when choosing a loss function (e.g. weighted BCE/Focal loss pos_weight) or sampling strategy — the two datasets' imbalance ratios differ by ~3.6x.
- No scenes are flagged for exclusion; `range_check.json`'s flags are expected `mag1c` behavior near plumes, already handled by STARCOP's own train-time clipping.
- This report's sensor breakdown (§2) confirms training will be AVIRIS-NG-only unless EMIT data is separately sourced — revisit Open Decision D-05 if hyperspectral cross-sensor generalization is in scope for this project.
