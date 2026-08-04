# STARCOP Baseline Validation

> Confirms the original STARCOP pretrained models produce valid segmentation masks
> under Environment A (`vendor/starcop/.venv`, Python 3.10.19, torch 1.13.1,
> pytorch-lightning 1.6.4) on the MacBook M4 Pro (CPU — torch 1.13.1 has no stable
> MPS support). Ran 2026-07-22.

## Setup

- **Assets**: mini test set + pretrained checkpoints (`STARCOP_mini.zip`,
  `multistarcop_varon.zip`, `hyperstarcop_magic_rgb.zip`) — the same files
  `vendor/starcop/notebooks/model_demos_AVIRIS.ipynb` downloads on Colab, fetched
  locally via `src/data/download_mini_dataset.py`
- **Runner**: `notebooks/starcop_baseline_validation.py` — a local, non-Colab
  equivalent of the notebook's HyperSTARCOP/MultiSTARCOP inference cells (same
  model loading and plotting calls, `device="cpu"` instead of CUDA, no Colab
  magics/pip installs)
- **Test split**: `starcop_mini/test_mini10.csv` — 9 scenes, ~2.36M pixels total
- **Metrics**: computed directly from `pred_binary` vs `output_norm` per pixel
  (accuracy, `sklearn.metrics.balanced_accuracy_score`, `f1_score`,
  `jaccard_score`) rather than via `starcop.validation.run_validation`'s
  built-in aggregation, which assumes the full dataset's easy/hard difficulty
  split — a column the mini subset doesn't carry (confirmed by a `KeyError`
  when called directly against this test set)

## Results

| Model | Overall Accuracy | Balanced Accuracy | F1 (methane) | IoU (methane) |
|---|---|---|---|---|
| **HyperSTARCOP** (AVIRIS hyperspectral, 4ch: mag1c + 3 TOA bands) | 0.9965 | 0.9594 | 0.9065 | 0.8290 |
| **MultiSTARCOP** (WorldView-3 multispectral ratio bands) | 0.9844 | 0.6504 | 0.4197 | 0.2656 |

Full JSON: `docs/baseline_validation/metrics.json`

## Segmentation mask samples

- `docs/baseline_validation/hyperstarcop_sample_mask.png` — HyperSTARCOP on one
  test scene: RGB, mag1c enhancement, ground-truth label, prediction, and a
  correct/false-positive/false-negative difference map. Prediction closely
  tracks the labeled plume shape.
- `docs/baseline_validation/multistarcop_sample_mask.png` — same layout for
  MultiSTARCOP.

## Interpretation

- **No GCS or W&B errors** — validation criterion met without needing the
  `train.py` GCS-skip patch (this run only exercises inference/checkpoint
  loading, not the training script).
- HyperSTARCOP (hyperspectral input) clearly outperforms MultiSTARCOP
  (multispectral ratio-band input) on this baseline, consistent with the
  STARCOP paper's own finding that hyperspectral bands carry much stronger
  methane signal than multispectral proxies — expected given this project's
  focus on hyperspectral imagery.
- Metrics are computed on only 9 mini-test scenes; the notebook's own comment
  on this subset applies here too — absolute numbers are not representative of
  full-dataset performance, only a smoke test that the pretrained baseline
  loads and predicts correctly.

## Patch inventory

- `patches/train-lightning2-and-gcs-skip.patch` — updates `scripts/train.py`
  for Lightning 2.x (`auto_select_gpus`/`auto_lr_find` removed,
  `resume_from_checkpoint` moved to `trainer.fit(ckpt_path=...)`) and gates the
  hardcoded `gs://starcop/` upload behind `STARCOP_NO_GCS_UPLOAD`. Not applied
  automatically — apply with `patch -p1 -d vendor/starcop < patches/train-lightning2-and-gcs-skip.patch`
  before running actual training (Phase 3). Not exercised by this inference-only
  baseline check, since `train.py` isn't invoked here.
