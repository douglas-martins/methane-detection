# Hardware Benchmark: HyperSTARCOP on the Xilinx ZCU104

On-board/embedded deployment is this project's actual target, not a cloud REST API
(see the research strategy in [Methodology](../methodology.md)). This page summarizes
a real feasibility test of that: the same HyperSTARCOP `mag1c + rgb` model, quantized
to INT8 and run on a Xilinx ZCU104's DPU, compared against the same model in FP32 on
the ZCU104's ARM CPU and on a notebook's x86 CPU.

> [!IMPORTANT]
> This benchmark was carried out in a collaborator's separate repository,
> [`Projeto_VITISAI_hyperstarcop`](https://github.com/Thifjj/Projeto_VITISAI_hyperstarcop),
> as an on-board deployment feasibility study for the same model family used elsewhere
> on this site. The numbers below are curated here for reference, not reproduced by
> this repo's own MLflow/Prefect pipeline. They were measured on the 9-scene
> `STARCOP_mini` test set, not the 342-scene full test set used in the
> [STARCOP reproduction results](starcop-benchmark.md) — see that page's own
> "Historical" section for the same caveat applied to the CPU/FP32 baseline reused
> here.

## Targets evaluated

| Target | Platform | Precision | Runtime |
|---|---|---|---|
| CPU x86 | Notebook, AMD Ryzen 9 5980HX | FP32 | PyTorch |
| CPU ARM | ZCU104, quad-core Cortex-A53 | FP32 | ExecuTorch + XNNPACK |
| DPU | ZCU104, 2× `DPUCZDX8G_ISA1_B4096` @ 300 MHz | INT8 | Xilinx Vitis AI + VART |

The INT8 model was quantization-calibrated separately with 200 images from the full
STARCOP dataset (not the mini set, which has too few images for calibration). All
three targets run on the same logical input (batch = 1, 4 channels, 512×512), laid
out per platform convention: NCHW `[1, 4, 512, 512]` for the CPU targets (PyTorch,
ExecuTorch) and NHWC `[1, 512, 512, 4]` for the DPU (Vitis AI's expected layout).
All three use the same `sigmoid > 0.5` segmentation threshold.

## Does INT8 quantization hurt segmentation quality?

| Target | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| CPU x86 (PyTorch FP32) | 89.2663% | 92.0803% | 90.6515% | 82.9014% |
| CPU ARM (ExecuTorch FP32) | 89.2663% | 92.0803% | 90.6515% | 82.9014% |
| DPU (Vitis AI INT8) | 90.9764% | 90.3945% | 90.6845% | 82.9567% |

ExecuTorch preserved the FP32 mask exactly on these nine scenes. INT8 quantization
traded a little recall for precision but left F1 and IoU essentially unchanged
(+0.033 and +0.055 percentage points) — quantizing for the DPU did not measurably
cost segmentation quality here.

## Inference performance

**Baseline (one image at a time, no concurrency):**

| Target | `model-only` FPS | `model-only` latency | `end-to-end` FPS | `end-to-end` latency |
|---|---:|---:|---:|---:|
| CPU x86 | 2.072 | 482.5 ms | 2.010 | 497.4 ms |
| CPU ARM (4 threads) | 0.497 | 2013.9 ms | 0.464 | 2155.6 ms |
| DPU | **26.472** | **37.8 ms** | **4.821** | **207.4 ms** |

**Best configuration found per platform** (multiple runners/workers, searched
independently per platform):

| Target | `model-only` FPS | `end-to-end` FPS |
|---|---:|---:|
| CPU x86 | 6.076 | 5.305 |
| CPU ARM (4 threads, no runner search) | 0.497 | 0.464 |
| DPU | **50.876** | **21.673** |

Comparing best-found configurations, the DPU reached **≈102× the CPU ARM's
throughput** and **≈8.4× the CPU x86's throughput** in `model-only` mode (≈47× and
≈4.1× respectively in `end-to-end` mode, where TIFF reads, normalization, and
thresholding dominate more on the DPU since inference itself is so fast). A wider
parallelism sweep (37 CPU configurations, 47 DPU configurations) backs these
numbers — see the linked reports below for the full sweep.

## Environment & toolchain

| Item | CPU x86 (notebook) | CPU ARM (ZCU104) | DPU (ZCU104) |
|---|---|---|---|
| OS | Linux, glibc 2.39 | PetaLinux 2022.2 | PetaLinux 2022.2, kernel 5.15.36 |
| Runtime | Python 3.10, PyTorch 2.13 | ExecuTorch 1.3.1 + XNNPACK | Vitis AI 3.5 pipeline |
| Model format | `.pth` checkpoint | `.pte` (XNNPACK-delegated) | `.xmodel` (DPU-compiled) |

> [!NOTE]
> The benchmark code identifies its own flow as Vitis AI 3.5, while `xdputil` run
> directly on the board reports the deployed VART/XIR libraries as version 3.0.0.
> The source report keeps both numbers rather than assuming they match — worth
> preserving here rather than collapsing to one version string.

## Key takeaways

1. INT8 quantization for the DPU preserved segmentation quality: F1 and IoU moved by
   less than 0.06 percentage points from the FP32 reference.
2. The DPU delivered the highest throughput on every scenario tested, from **12.8×**
   the CPU x86's FPS at the shared sequential baseline up to **~102×** the CPU ARM's
   FPS at each platform's best found configuration.
3. The ARM Cortex-A53 CPU inside the ZCU104 ran the identical FP32 model correctly
   via ExecuTorch/XNNPACK, but its own compute budget — not the DPU's INT8 path — is
   the bottleneck for running this model board-side without acceleration.
4. These results support Xilinx Vitis AI and a ZCU104-class DPU as a viable on-board
   deployment path for this model family, ahead of this project committing to its own
   final on-board approach.

## Full reports and raw data

This page deliberately omits the per-image accuracy breakdown, the full 37-run CPU
and 47-run DPU parallelism sweeps, and the FPS/latency derivation walkthrough. For
those, see the source repository:

- [`Projeto_VITISAI_hyperstarcop`](https://github.com/Thifjj/Projeto_VITISAI_hyperstarcop) — full project
- [`comparacao_benchmarks_configuracoes.md`](https://github.com/Thifjj/Projeto_VITISAI_hyperstarcop/blob/main/resultados_zcu104/comparacao_benchmarks_configuracoes.md) — full 3-way comparison (CPU x86, CPU ARM, DPU), all sweep phases
- [`relatorio.md`](https://github.com/Thifjj/Projeto_VITISAI_hyperstarcop/blob/main/resultados_zcu104/relatorio.md) — detailed CPU FP32 × DPU INT8 report, including per-image quality breakdown
