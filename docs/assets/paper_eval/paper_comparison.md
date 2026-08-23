# Paper comparison — all variants

## MultiSTARCOP — Varon ratio

**Variant:** `varon` · **Live API check:** out of scope (MultiSTARCOP not deployed live)

| Metric | Paper | Reproduced |
| --- | ---: | ---: |
| Strong F1 | 30.72 ± 2.87 | 28.94 |
| Weak F1 | 10.35 ± 1.52 | 17.01 |
| FPR (tile-level) | 87.89 ± 4.67 | 84.57 |
| AUPRC | 11.92 ± 1.35 | 11.27 |

## HyperSTARCOP — mag1c only

**Variant:** `mag1c_only` · **Live API check:** passed

| Metric | Paper | Reproduced |
| --- | ---: | ---: |
| Strong F1 | 74.15 ± 6.10 | 66.74 |
| Weak F1 | 47.57 ± 4.17 | 48.76 |
| FPR (tile-level) | 52.11 ± 10.98 | 36.00 |
| AUPRC | 49.41 ± 5.49 | 36.19 |

## HyperSTARCOP — mag1c + RGB

**Variant:** `mag1c_rgb` · **Live API check:** passed

| Metric | Paper | Reproduced |
| --- | ---: | ---: |
| Strong F1 | 81.96 ± 3.71 | 83.08 |
| Weak F1 | 43.42 ± 5.72 | 42.34 |
| FPR (tile-level) | 43.66 ± 7.36 | 40.57 |
| AUPRC | 51.99 ± 2.76 | 47.60 |
