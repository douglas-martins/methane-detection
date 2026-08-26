# Paper comparison using all variants

Each table compares our reproduced numbers against the original paper's reported results for one model variant. **Bold** marks whichever value (paper or reproduced) is actually better for that metric, for F1 and AUPRC, higher wins; for FPR, lower wins.

| Metric | What it measures | Direction |
| --- | --- | --- |
| Strong F1 | Detection accuracy on plumes with a strong, unambiguous methane signal | higher = better |
| Weak F1 | Detection accuracy on plumes with a weak/faint signal (harder cases) | higher = better |
| FPR (tile-level) | Share of non-plume tiles wrongly flagged as containing a plume | lower = better |
| AUPRC | Overall precision/recall trade-off across all confidence thresholds | higher = better |

## MultiSTARCOP: Varon ratio

**Variant:** `varon` · **Live API check:** out of scope (MultiSTARCOP not deployed live)

| Metric | Paper (mean ± std) | Reproduced | How we did |
| --- | ---: | ---: | --- |
| Strong F1 | **30.72 ± 2.87** | 28.94 | 1.8 pts below paper |
| Weak F1 | 10.35 ± 1.52 | **17.01** | 6.7 pts above paper |
| FPR (tile-level) | 87.89 ± 4.67 | **84.57** | 3.3 pts lower (better) |
| AUPRC | **11.92 ± 1.35** | 11.27 | 0.7 pts below paper |

## HyperSTARCOP: mag1c only

**Variant:** `mag1c_only` · **Live API check:** passed

| Metric | Paper (mean ± std) | Reproduced | How we did |
| --- | ---: | ---: | --- |
| Strong F1 | **74.15 ± 6.10** | 66.74 | 7.4 pts below paper |
| Weak F1 | 47.57 ± 4.17 | **48.76** | 1.2 pts above paper |
| FPR (tile-level) | 52.11 ± 10.98 | **36.00** | 16.1 pts lower (better) |
| AUPRC | **49.41 ± 5.49** | 36.19 | 13.2 pts below paper |

## HyperSTARCOP: mag1c + RGB

**Variant:** `mag1c_rgb` · **Live API check:** passed

| Metric | Paper (mean ± std) | Reproduced | How we did |
| --- | ---: | ---: | --- |
| Strong F1 | 81.96 ± 3.71 | **83.08** | 1.1 pts above paper |
| Weak F1 | **43.42 ± 5.72** | 42.34 | 1.1 pts below paper |
| FPR (tile-level) | 43.66 ± 7.36 | **40.57** | 3.1 pts lower (better) |
| AUPRC | **51.99 ± 2.76** | 47.60 | 4.4 pts below paper |
