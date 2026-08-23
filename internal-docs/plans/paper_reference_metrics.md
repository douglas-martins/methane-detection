# STARCOP Paper — Canonical Reference Metrics

> **Source:** Semantic segmentation of methane plumes with hyperspectral machine
> learning models (Scientific Reports 13:19999, 2023),
> `Semantic_segmentation_of_methane_plumes_with_hyperspectral_machine_learning_models.pdf`
> (repo root). Entered once by hand, with an explicit table/page citation per
> value — see `track-a-paper-benchmark-reproduction-plan.md` Phase 3. Never
> approximated, never re-typed per run: every `paper_comparison.md` this project
> generates reads from this one file.
>
> All values are the paper's own reported mean ± std across 5 training runs,
> as percentages (0-100), matching the paper's own table formatting.

## Table 1 — MultiSTARCOP (multispectral, WorldView-3-simulated), page 9

Three variants reported; this project's released checkpoint
(`multistarcop_varon`) is the **Varon** ratio variant specifically — not
"Sanchez" or "Varon+Sanchez".

| Variant | Strong F1 | Weak F1 | FPR (tile-level) | AUPRC |
| --- | --- | --- | --- | --- |
| Baseline, ratio + morpho. | 7.44 | 0.5 | 100.0 | N/A |
| Our (Varon) | 30.72 ± 2.87 | 10.35 ± 1.52 | 87.89 ± 4.67 | 11.92 ± 1.35 |
| Our (Sanchez) | 26.59 ± 3.13 | 9.32 ± 1.05 | 94.4 ± 1.30 | 9.96 ± 1.43 |
| Our (Varon+Sanchez) | 31.89 ± 2.44 | 11.04 ± 0.75 | 90.51 ± 4.23 | 13.04 ± 1.96 |

## Table 2 — HyperSTARCOP (hyperspectral, AVIRIS-NG), page 10

| Variant | Strong F1 | Weak F1 | FPR (tile-level) | AUPRC |
| --- | --- | --- | --- | --- |
| Baseline, mag1c + morpho. | 67.45 | 39.95 | 75.43 | N/A |
| HyperSTARCOP, only mag1c | 74.15 ± 6.10 | 47.57 ± 4.17 | 52.11 ± 10.98 | 49.41 ± 5.49 |
| HyperSTARCOP, mag1c + rgb | 81.96 ± 3.71 | 43.42 ± 5.72 | 43.66 ± 7.36 | 51.99 ± 2.76 |

## Machine-readable form

The prose tables above are for human reading and citation; the block below is
what `src/evaluation/paper_eval_mlflow.py::load_paper_reference_metrics` parses
(kept redundant with the tables on purpose — a mismatch between the two is a
signal this file was hand-edited incorrectly, not something to silently trust
one side of). Values are fractions (0-1), matching this project's own metric
scale (`paper_metrics.py` returns fractions, not percentages) — a `/100` of
the percentages above, mean and std both.

```yaml
varon:
  citation: "Table 1, page 9 — row 'Our (Varon)'"
  strong_f1score: {mean: 0.3072, std: 0.0287}
  weak_f1score: {mean: 0.1035, std: 0.0152}
  no_plume_FPR: {mean: 0.8789, std: 0.0467}
  auprc: {mean: 0.1192, std: 0.0135}
mag1c_only:
  citation: "Table 2, page 10 — row 'HyperSTARCOP, only mag1c'"
  strong_f1score: {mean: 0.7415, std: 0.0610}
  weak_f1score: {mean: 0.4757, std: 0.0417}
  no_plume_FPR: {mean: 0.5211, std: 0.1098}
  auprc: {mean: 0.4941, std: 0.0549}
mag1c_rgb:
  citation: "Table 2, page 10 — row 'HyperSTARCOP, mag1c + rgb'"
  strong_f1score: {mean: 0.8196, std: 0.0371}
  weak_f1score: {mean: 0.4342, std: 0.0572}
  no_plume_FPR: {mean: 0.4366, std: 0.0736}
  auprc: {mean: 0.5199, std: 0.0276}
```
