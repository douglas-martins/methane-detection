# Hardware-oriented methane segmentation: repository analysis and model hypotheses

> **Status:** design baseline, not an implementation specification  
> **Analysis date:** 2026-08-19  
> **Target project:** `methane-detection`  
> **Primary deployment tool considered:** [hls4ml](https://fastmachinelearning.org/hls4ml/)  
> **Decision rule:** do not choose an architecture until the target FPGA, system boundary, throughput, power, and full-granule false-alert budget are defined.  
> **Update 2026-08-19:** added RaVAEn on-board flight evidence (Section 6A), a non-FPGA fallback track (Section 8.5), Alternative E (Section 9), and H9 (Section 10), after reading Růžička et al., *Fast Model Inference and Training On-Board of Satellites*, arXiv:2307.08700.

## 1. Executive summary

Three observations should drive the design:

1. **Input representation matters more than making an already large segmentation network larger.** The MARS study reports that changing from Mag1c to WMF improved the single-model tiled F1 from 43.59 to 63.07, while changing among U-Net, U-Net++, DeepLabV3/+, and much larger encoders produced relatively small differences. This favors investment in the spectral/preprocessing contract and hard-negative data before architecture scale.
2. **Full-granule false alerts, not patch accuracy, are the operational bottleneck.** MARS found 3,565 average false alarms for a single RGB+WMF U-Net over its EMIT full-tile set, reduced to 1,501 by a five-model ensemble. A hardware student should therefore be optimized and distilled against object/granule behavior, not only pixel F1.
3. **Neither reference model family is a safe drop-in hls4ml conversion target.** MARS uses a 6.69M-parameter U-Net/MobileNetV3 with unsupported or awkward frontend operations. HyperspectralViTs uses attention, LayerNorm, dynamic interpolation, and custom graph logic; hls4ml lists multi-head attention as unsupported through its PyTorch frontend. The practical path is a small, static, FX-traceable student composed from Conv2D, depthwise Conv2D, BatchNorm, ReLU, fixed nearest-neighbor upsampling, and simple two-input merges.

### Recommended first experiment portfolio

Run these in parallel against identical splits and full-scene evaluation:

| Priority | Candidate                     | Input                      | Main question                                                                                          |
| -------- | ----------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------ |
| P0       | **TinyDS-4**                  | `mag1c/WMF + RGB`          | What is the smallest full-resolution CNN that preserves useful plume morphology?                       |
| P0       | **SpectralTiny-86**           | selected EMIT/AVIRIS bands | Can a 1x1 spectral bottleneck plus a tiny spatial CNN remove matched-filter latency without attention? |
| P1       | **TinyU-4**                   | `mag1c/WMF + RGB`          | Are two spatial scales worth the skip-buffer and conversion complexity?                                |
| P1       | **Distilled TinyDS/TinyU**    | either input regime        | Can one student reproduce the false-positive suppression of the five-model MARS ensemble?              |
| P2       | **On-board retrainable head (H9)** | frozen encoder latents | Can a tiny head be retrained on-board from few-shot labels to adapt without redeployment, as RaVAEn demonstrated for cloud detection? Not comparable to H1–H3 until the [H9.1 protocol](#h91--required-evaluation-protocol-precondition-for-comparison-with-h1h3) is fixed. |
| Control  | **MF threshold + morphology** | `mag1c/WMF`                | What accuracy, latency, and resource floor must learned models beat?                                   |

**Do not begin with a direct U-Net, SegFormer, or EfficientViT conversion.** First prove the conversion, bit-accuracy, synthesis, and end-to-end data path with a deliberately constrained student.

**Also run OpenVino/VPU as a parallel, non-FPGA fallback benchmark (Section 8.5, Alternative E)**, not as the primary path: it is flight-proven (Section 6A) and de-risks the case where hls4ml conversion or synthesis gates block a promising candidate — but the flight evidence used an EOL OpenVino toolchain (2022.3 LTS + MYRIAD/HDDL), so pin that toolchain or re-validate on a current one (Section 8.5) before treating it as an available fallback rather than historical evidence.

---

## 2. Scope and system questions

This document analyzes:

- `methane-detection`, including its current STARCOP-based data, training, registry, and serving assumptions;
- `UNEP-IMEO-MARS/marsml-hyperspectral`;
- `previtus/HyperspectralViTs`;
- current hls4ml documentation and PyTorch converter behavior;
- model, data, evaluation, hardware, software, operational, reproducibility, and licensing risks.

It does **not** assume that “FPGA deployment” means only the neural network. The final system may include:

1. sensor calibration and bad-band handling;
2. orthorectification/reprojection;
3. radiance-to-reflectance conversion;
4. optional Mag1c/MF/WMF computation;
5. normalization and clipping;
6. tiled or streamed neural inference;
7. thresholding, connected components, morphology, ranking, and georeferencing;
8. telemetry/downlink and human review.

A fast hls4ml network does not produce a fast system if WMF, memory traffic, or scene reprojection remains dominant.

---

## 3. Repositories and provenance

The two reference repositories were cloned locally under `vendor/` and excluded in `.git/info/exclude` so they do not alter this project's tracked submodule policy or mix unlicensed source into the MIT repository.

| Repository                         | Local path                    | Branch            | Analyzed commit                            | Repository character                                        |
| ---------------------------------- | ----------------------------- | ----------------- | ------------------------------------------ | ----------------------------------------------------------- |
| MARS hyperspectral                 | `vendor/marsml-hyperspectral` | `main`            | `ebc608bc107abf7aa1510b4556521fbe2d62f598` | Operational research reproduction code                      |
| HyperspectralViTs                  | `vendor/HyperspectralViTs`    | `main`            | `a184a2556430fe7cb3af57558e64817569c8d258` | On-board hyperspectral research prototype                   |
| STARCOP                            | `vendor/starcop`              | project submodule | project-pinned                             | Existing baseline and composition dependency                |
| hls4ml source inspected separately | temporary research clone      | `main`            | `b90fb06736baa0908a8995fc1cf4ac4a7d1c241f` | Converter/backend implementation matching analyzed dev docs |

To reproduce the reference checkout without adding it to Git:

```bash
git clone https://github.com/UNEP-IMEO-MARS/marsml-hyperspectral vendor/marsml-hyperspectral
git clone https://github.com/previtus/HyperspectralViTs vendor/HyperspectralViTs
git -C vendor/marsml-hyperspectral checkout ebc608bc107abf7aa1510b4556521fbe2d62f598
git -C vendor/HyperspectralViTs checkout a184a2556430fe7cb3af57558e64817569c8d258
```

### 3.1 Licensing blocker

Neither cloned code repository contains a `LICENSE`/`COPYING` file, and the GitHub API reports no repository license. The HyperspectralViTs **paper** is CC BY 4.0, but that does not automatically license its source code. Therefore:

- ideas, published results, interfaces, and independently reimplemented concepts may be studied;
- code should **not** be copied into this MIT project until the authors provide an explicit compatible software license;
- if the repositories later become durable dependencies, record exact commits and add them as submodules only after licensing review.

This is a release-blocking concern, not clerical cleanup.

---

## 4. Current project baseline

### 4.1 Data and model contract

The current project trains on AVIRIS-NG STARCOP data with four channels in this exact order:

1. `mag1c`
2. `TOA_AVIRIS_640nm`
3. `TOA_AVIRIS_550nm`
4. `TOA_AVIRIS_460nm`

The existing STARCOP normalization is part of `ModelModule.forward`, not the external data pipeline:

| Input                   | Offset | Factor | Clip after scaling |
| ----------------------- | -----: | -----: | -----------------: |
| `mag1c`                 |      0 |   1750 |             [0, 2] |
| each AVIRIS RGB channel |      0 |     60 |             [0, 2] |

hls4ml performs no automatic input normalization. A hardware implementation must either:

- include these affine/clip operations explicitly in the exported graph;
- implement them in a bit-exact FPGA preprocessing block; or
- require already normalized host input and version that contract with the model.

The third option is easiest for a prototype but weakest for production reproducibility.

### 4.2 Dataset scale and bias

From `docs/dataset_report.md`:

| Property                       |           `starcop_mini` |             `starcop_raw` |
| ------------------------------ | -----------------------: | ------------------------: |
| Train/val/test scenes          |                8 / 1 / 9 |         2,882 / 543 / 342 |
| Train/val/test 128x128 patches |           392 / 49 / 441 | 141,218 / 26,607 / 16,758 |
| Positive pixel fraction        |                    1.13% |                     0.32% |
| Background:methane ratio       |                    ~87:1 |                    ~314:1 |
| Sensor/geography               | AVIRIS-NG, Permian Basin |  AVIRIS-NG, Permian Basin |

Implications:

- `starcop_mini` is suitable only for smoke tests;
- `starcop_raw` must drive class weighting and architecture comparisons;
- the current data cannot validate global, EMIT, PRISMA, or EnMAP generalization;
- current test evidence (HyperSTARCOP F1 0.9065 on nine mini scenes) is not an operational acceptance benchmark;
- sensor, geography, sector, surface material, and plume-strength shift all need separate reporting.

### 4.3 Existing engineering assets worth reusing

The project already provides useful foundations:

- DVC data lineage and deterministic split/patch stages;
- scene-level split isolation;
- MLflow/W&B tracking and model registry;
- full-scene validation hooks from STARCOP;
- input-band serving contracts and drift statistics;
- promotion policy and BentoML serving.

Gaps for hardware work:

- no architecture registry for hardware students;
- no float -> quantized -> hls4ml equivalence pipeline;
- no synthesis/resource/latency artifacts in MLflow;
- no EMIT/MARS full-granule dataset in DVC;
- serving assumes a PyTorch model and CPU tensor semantics;
- no fixed-point preprocessing contract or bit-accurate test vectors;
- no target board/toolchain configuration.

---

## 5. MARS hyperspectral analysis

### 5.1 Purpose and evidence

MARS is designed for an operational analyst-assisted workflow, not on-board FPGA inference. Its paper reports:

- 25,024 full scenes processed in 11 months;
- 2,851 verified methane leaks;
- 834 stakeholder notifications;
- deployment over EMIT, EnMAP, and PRISMA;
- more than 74% false-alert reduction from ensembling compared with prior deep models.

This operational evidence is particularly valuable because it exposes failures hidden by curated patch evaluation.

### 5.2 Data regime

The released work uses global, expert-validated data from three sensors:

- **EMIT:** 60 m, 381–2493 nm, global arid-region coverage;
- **PRISMA:** 30 m, 400–2505 nm;
- **EnMAP:** 30 m, 420–2450 nm.

Training samples are 256x256 source tiles, then 128x128 training patches with 64-pixel overlap. Full granules are retained for realistic evaluation. EMIT splits are temporal; PRISMA/EnMAP splits are spatial.

This is stronger than random patch splitting because it reduces source-scene leakage and better approximates future deployment.

### 5.3 Model and training

The operational model is:

- U-Net from `segmentation_models_pytorch`;
- MobileNetV3 encoder, depth 5;
- decoder channels 256, 128, 64, 32, 16;
- approximately 6.69M parameters;
- RGB reflectance + one matched-filter product, with WMF selected;
- BCE loss, weighted random sampling, per-pixel MF-derived loss weights;
- no-data masking;
- five independently initialized models averaged at deployment.

Optional wind and location bands gave small patch-level gains. Location was not deployed due to concern that it could memorize known source areas and reduce discovery/generalization.

### 5.4 Result most relevant to this project

For EMIT:

| Variant                |     Tiled F1 | Detected |  Missed | False alarms on full-tile set |
| ---------------------- | -----------: | -------: | ------: | ----------------------------: |
| WMF threshold baseline |        23.40 |      225 |      86 |                        79,273 |
| single U-Net RGB+WMF   | 63.07 ± 2.46 | 213 ± 20 | 98 ± 20 |                 3,565 ± 1,830 |
| ensemble, 5x RGB+WMF   |        65.13 |      207 |     104 |                         1,501 |

The ensemble's main value is false-alert suppression, not a dramatic F1 increase. This motivates **ensemble-to-student distillation with full-granule hard negatives**.

Architecture ablation also shows that 4.71M–48.79M models had similar tiled scores. Larger encoders sometimes reduced false alerts but also missed more events. Parameter count alone is therefore a poor optimization target.

### 5.5 System timing insight

On a consumer Mac used in the MARS study:

- download: ~130 s;
- load + orthorectify + WMF: ~20 s;
- one model inference: ~1.6 s;
- five-model inference: approximately 5x the one-model step.

For on-ground processing the network is not the bottleneck. For an FPGA/on-board design, the conclusion depends on whether WMF and geometric/radiometric preprocessing are also accelerated.

### 5.6 Code/repository assessment

Strengths:

- model, data, full-tile evaluation, ensembling, and export code are present;
- full-tile object matching is more operationally meaningful than pixel metrics alone;
- normalization and no-data handling are visible;
- trained models and data are linked.

Risks:

- only three commits in the analyzed history;
- no tests, CI, package metadata, lockfile, or software license;
- several absolute example paths and broad exception handlers;
- fixed thresholds and connected-component rules are embedded in evaluation;
- `EnsembleHandler`'s averaging logic is safe only for the batch-size-one use pattern because it concatenates predictions along the batch dimension before summing;
- some dataset variables assume an MF product is present;
- val/test loaders inherit `shuffle=True` in the custom `DataModule`;
- architecture wrappers are not consistently `nn.Module` subclasses, so conversion should target the underlying network.

Treat this as research reference code, not production dependency code.

---

## 6. HyperspectralViTs analysis

### 6.1 Purpose

HyperspectralViTs studies end-to-end, on-board processing of hyperspectral L1B data. It removes the matched-filter dependency and adapts SegFormer and EfficientViT for high spectral dimensionality.

This directly addresses the largest system-level weakness of RGB+MF models: if the matched filter misses or distorts the plume, the downstream network cannot recover the lost spectral evidence.

### 6.2 Data

The work provides:

- **OxHyperSyntheticCH4:** 796/198/200 train/val/test 512x512 tiles, 86 EMIT bands, 228 GB;
- **OxHyperRealCH4:** 279/91/98 train/val/test tiles, 47 GB;
- **OxHyperMinerals:** 285 bands, 372 GB;
- all-band STARCOP variants using 60 AVIRIS-NG bands.

EMIT training uses 64x64 patches with 32 overlap; STARCOP uses 128x128 with 64 overlap. End-to-end methane models use 86 EMIT bands covering RGB and methane-relevant ranges (1573–1699 nm and 2004–2478 nm). Synthetic pretraining followed by real-data fine-tuning outperformed real-only training.

### 6.3 Architecture ideas worth retaining

The paper identifies an **early information bottleneck**: an ordinary RGB-oriented first layer collapses 86 bands too aggressively. Three modular adaptations are proposed:

1. **1x1 spectral layers** to mix bands without spatial mixing;
2. **learned upscaling blocks** to recover output detail;
3. **reduced early stride** to preserve spatial resolution.

The transferable idea is not necessarily “use a transformer.” It is:

> perform an explicit, learnable spectral projection before spatial compression, and preserve enough spatial detail for narrow plume boundaries.

That idea maps naturally to a much smaller hls4ml-compatible CNN.

### 6.4 Results and timing

On synthetic EMIT methane data:

| Model                          |          F1 |   Tile FPR |
| ------------------------------ | ----------: | ---------: |
| HyperSTARCOP MF+RGB            | 58.08 ± 5.3 | 65.6 ± 6.7 |
| SegFormer base                 | 60.26 ± 1.7 | 27.2 ± 3.1 |
| HyperSegFormer ConvUpStride    | 74.27 ± 2.9 | 18.4 ± 2.4 |
| EfficientViT base              | 68.40 ± 6.9 | 25.0 ± 5.0 |
| HyperEfficientViT ConvUpStride | 72.26 ± 4.9 | 31.8 ± 5.9 |

On STARCOP, HyperSegFormer ConvUpStride reports F1 56.75 versus 50.26 for HyperSTARCOP, but lower strong-plume F1. End-to-end gains are therefore dataset and plume-strength dependent.

For a 1280x1242 EMIT granule split into 100 fixed 128x128 tiles:

| Model                             | Params |   Unibap CPU | Jetson GPU, TensorRT |
| --------------------------------- | -----: | -----------: | -------------------: |
| HyperSTARCOP MF+RGB, including MF | 6.633M |      203.3 s |             370.27 s |
| HyperSegFormer ConvUp             | 4.326M |       30.1 s |               0.64 s |
| HyperEfficientViT ConvUp          | 4.855M | not measured |               0.55 s |

The MF computation dominates the HyperSTARCOP timing. This is the strongest evidence for evaluating an all-band FPGA student rather than accelerating only the post-MF network.

A local forward-pass inspection of the repository's EfficientViT implementation produced these approximate figures at 64x64 input; convolution/linear MAC counts are lower bounds because custom attention tensor operations are not included:

| Variant   | Bands | Parameters | Conv/linear MAC lower bound | Native output               |
| --------- | ----: | ---------: | --------------------------: | --------------------------- |
| B0 ConvUp |     4 |    713,765 |                       10.4M | 16x16 before wrapper resize |
| B0 ConvUp |    86 |    727,131 |                       46.7M | 16x16 before wrapper resize |
| B1 ConvUp |     4 |  4,836,149 |                       56.4M | 16x16 before wrapper resize |
| B1 ConvUp |    86 |  4,855,419 |                       98.7M | 16x16 before wrapper resize |

These numbers illustrate why parameter count alone hides activation and spectral-input cost.

### 6.5 Hardware evidence limitations

The reported “satellite proxy” is a quad-core AMD GX-412HC CPU plus Myriad X VPU and 2 GB RAM at an estimated 18 W. Other tests use Raspberry Pi and a 30 W Jetson AGX Xavier. These are valuable edge benchmarks but are **not FPGA synthesis results**. They do not report LUT, FF, DSP, BRAM/URAM, initiation interval, timing closure, or fixed-point accuracy.

The input is L1B, not raw L0. A real on-board design still needs a feasible calibration/partial preprocessing pipeline or L0-like retraining.

### 6.6 Code/repository assessment

Strengths:

- explicit model implementations and architecture modifications;
- all-band AVIRIS and EMIT loading;
- synthetic-to-real transfer workflow;
- low-compute benchmark methodology;
- fixed-tile deployment pattern compatible with static accelerators.

Risks:

- only four commits in the analyzed history;
- no tests, CI, lockfile, or repository license;
- older tightly pinned ML stack;
- hard-coded paths and broad `except`/`assert False` error handling;
- dynamic control, custom dictionaries, interpolation, and model wrappers complicate FX tracing;
- default evaluation paths do not consistently apply the available no-data mask;
- metric code accumulates full predictions in memory;
- several thresholds and class conventions are architecture-specific;
- the repository setup is a research reproduction environment, not a stable reusable package.

---

## 6A. RaVAEn on D-Orbit ION SCV004: on-board flight evidence

### 6A.1 Why this paper is a different kind of evidence

Unlike MARS (ground reprocessing) and HyperspectralViTs (CPU/VPU/GPU proxy benchmarks, Section 6.5), Růžička et al. (2023, arXiv:2307.08700) report an **actual flight deployment**: RaVAEn, a VAE encoder that maps 32x32x4-band Sentinel-2 tiles to 128-dimensional latents, running on D-Orbit's ION SCV004 CubeSat demonstration mission. The same author group overlaps with this project's STARCOP baseline (Růžička and Mateo-García are co-authors of both). This paper targets exactly the system-boundary question raised in Section 2 and Section 13, but answers it with measured flight timings rather than a ground proxy or synthetic estimate.

### 6A.2 Hardware and evidence

- **Flown hardware**, not a proxy: quad-core x86-64 CPU, Intel Movidius Myriad X VPU, 2GB RAM — the same hardware class HyperspectralViTs used only as a ground-based proxy (Section 6.5).
- Three compute regimes measured on this exact hardware, encoding one file of 225 tiles at batch size 64 (RGB+NIR bands):

  | Regime           | Encoding only | Encoding + IO |
  | ----------------- | -------------: | -------------: |
  | Torch CPU          |         0.327s |         0.825s |
  | OpenVino CPU        |         0.171s |         0.659s |
  | OpenVino Myriad VPU |         0.111s |         0.596s |

- The gap between "encoding only" and "encoding + IO" is roughly constant (~0.49s) across all three devices: load/tiling overhead does not shrink as the encoder gets faster. This is hardware-measured confirmation of the same warning this document already makes from ground-processing timing (Section 5.5) and system-boundary framing (Section 2): **a fast network does not produce a fast system if I/O or preprocessing dominates.**
- The Myriad VPU is not only faster on average but also more *stable*: CPU/PyTorch shows large per-batch timing spikes, while Myriad/OpenVino stays in a tight band (paper's Fig. 3a vs 3b). For a duty-cycled or real-time on-board budget, tail latency may matter as much as mean latency.
- **On-board training, not just inference.** A frozen VAE encoder (pre-trained on the ground) produces 128-dim latents; a tiny one-layer FC classifier (129 trainable parameters) is trained **on-board** from those latents for a cloud-detection few-shot task (1,305 training tiles), reaching AUPRC 0.979, F1 0.956 at threshold 0.5. Average epoch time drops from 0.201s (batch 32) to 0.091s (batch 256) — training is cheap once the encoder is frozen and features are pre-extracted.

### 6A.3 What is transferable to this project

RaVAEn is a scene-level cloud-detection VAE, not a methane segmentation network, and its unit of output is a whole-tile embedding, not a per-pixel mask — it is not a competing architecture for TinyDS/TinyU/SpectralTiny and its accuracy numbers do not transfer. Three findings are transferable as design inputs:

1. **OpenVino + Myriad VPU is a flight-proven, non-FPGA on-board deployment path.** It sidesteps the hls4ml conversion-risk surface documented in Section 8 (unsupported PyTorch ops, LayerNorm/attention limits, FX-tracing fragility) because OpenVino supports a much larger standard-op set out of the box, at the cost of losing HLS's fine-grained fixed-point/resource control. Recorded as Alternative E in Section 9 and Section 8.5.
2. **Frozen encoder + tiny trainable head, retrained on-board, is flight-demonstrated.** This is a different system pattern from H0–H8 in Section 10: instead of shipping a fixed inference-only network, ship a frozen feature extractor and let a small head be (re)trained on-board from few-shot labels collected in orbit. Recorded as H9 in Section 10.
3. **I/O/tiling overhead as a hardware-measured, first-class budget item**, not just a ground-processing anecdote. This strengthens, without changing, the full-pipeline benchmarking requirement already in Section 12.1 and Section 13.

### 6A.4 Limits of this evidence

- RaVAEn's task (scene-level cloud yes/no from a whole-tile VAE latent) is much coarser than per-pixel methane segmentation; none of its accuracy numbers transfer.
- 4-band, 32x32 Sentinel-2 RGB+NIR tiles are smaller and coarser than either this project's 4-channel Mag1c+RGB AVIRIS patches or HyperspectralViTs' 86-band EMIT input; the encoder's compute profile does not generalize to a spectral segmentation network.
- Myriad X / OpenVino is a specific, aging Intel VPU product line. A target-hardware decision should not anchor on this exact chip without checking current availability — Section 13's "FPGA/SoC board and exact part: TBD" row should treat it as one candidate to evaluate or explicitly rule out, not assume.
- No resource utilization, power draw, or radiation/reliability data is reported for the VPU beyond the encode/train timings above — not yet comparable to the hardware-metrics rigor Section 12.1 requires.

---

## 7. Comparison from multiple design angles

| Angle                     | Current STARCOP project  | MARS RGB+WMF                             | HyperspectralViTs all-band              | Hardware implication                                |
| ------------------------- | ------------------------ | ---------------------------------------- | --------------------------------------- | --------------------------------------------------- |
| Input                     | 4ch Mag1c+RGB            | 4ch WMF+RGB                              | 60 AVIRIS or 86 EMIT bands              | 4ch minimizes I/O; all-band removes MF              |
| Sensor scope              | AVIRIS, Permian only     | EMIT/PRISMA/EnMAP                        | EMIT and AVIRIS experiments             | Need sensor-specific validation                     |
| Model                     | U-Net/MobileNetV2        | U-Net/MobileNetV3                        | SegFormer/EfficientViT                  | None is ideal as first hls4ml target                |
| Parameters                | millions                 | 6.69M × 5 deployed                       | ~4.3–4.9M                               | Tiny student should target orders of magnitude less |
| Main operational weakness | limited domain evidence  | full-scene false alerts                  | sensor specificity and complexity       | Optimize full-scene event metrics                   |
| Preprocessing             | Mag1c assumed            | WMF required                             | no MF, but L1B required                 | Define acceleration boundary first                  |
| Generalization            | unproven outside AVIRIS  | strong via common RGB+WMF representation | not zero-shot across spectral samplings | Common feature vs sensor-specific bitstreams        |
| Quantization evidence     | none                     | none                                     | FP16 TensorRT only                      | Need QAT/PTQ and fixed-point tests                  |
| Hardware evidence         | CPU/MPS/GPU pipeline     | ground processing                        | CPU/VPU/GPU proxy                       | Need actual HLS synthesis                           |
| Software maturity         | tested MLOps composition | research code                            | research code                           | Reimplement students locally                        |

RaVAEn (Section 6A) is omitted from this table because it solves a coarser task (scene-level cloud classification from a whole-tile VAE latent, not per-pixel methane segmentation) and so is not comparable row-for-row with the three segmentation systems above. Its value here is orthogonal: it is the only one of the four with **flight-measured**, not proxy or ground, hardware evidence, and it is the source of Alternative E (Section 9) and H9 (Section 10).

---

## 8. hls4ml feasibility

### 8.1 Current capabilities inspected

The analyzed hls4ml documentation reports:

- stable 1.3.0 and development 1.4.0.dev32 at the time of analysis;
- Keras, PyTorch, ONNX, and QONNX frontends;
- Vivado HLS, Vitis HLS, Intel/Quartus, Catapult, and experimental oneAPI backends;
- MLP, 1D/2D CNN, recurrent, Einsum, and experimental attention support;
- Linux support; macOS and Windows are not supported environments;
- fixed-point precision, per-layer precision, reuse factors, resource/latency strategies, streaming I/O, and FIFO-depth optimization.

The PyTorch frontend uses `torch.fx` symbolic tracing and is less mature than Keras. Current direct handlers include Conv1D/2D, depthwise Conv1D/2D, pooling, BatchNorm, limited LayerNorm, common activations, fixed upsampling modules, padding, concatenation, simple merges, and Einsum.

Important constraints found in the converter source:

- PyTorch multi-head attention is marked unsupported in the status table;
- direct activation handlers do not include Hardswish, SiLU, or GELU;
- direct `nn.functional.interpolate` is less safe than a fixed `nn.Upsample` module;
- LayerNorm accepts only a restricted 3D shape;
- PyTorch `io_stream` requires user-provided channels-last input; automatic input transpose is available only for `io_parallel`;
- concatenation supports at most two tensors per node;
- direct Brevitas ingestion is unsupported; Brevitas should export QONNX;
- QONNX conversion works best with constant, scalar, power-of-two quantization scales and zero-point zero.

### 8.2 Reference architecture conversion matrix

| Architecture                      | Likely status                                 | Blocking/awkward operations                                                           | Recommendation                                          |
| --------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| MARS U-Net/MobileNetV3            | Not drop-in                                   | Hardswish, squeeze/excitation/adaptive pooling, large skip buffers, third-party graph | Distill into a small ReLU CNN                           |
| Current STARCOP U-Net/MobileNetV2 | High risk/large                               | third-party encoder, ReLU6, depthwise path, decoder skips, size                       | Use as teacher/baseline only                            |
| HyperSegFormer                    | Not viable through direct PyTorch frontend    | attention unsupported, LayerNorm shape, GELU, dynamic interpolation, custom outputs   | Do not use as first FPGA target                         |
| HyperEfficientViT                 | Not drop-in                                   | custom linear attention, SiLU, dynamic graph/dictionaries, interpolation              | Use as teacher; retain only spectral projection idea    |
| HyperspectralViTs `SimpleCNN`     | Operator-compatible but computationally large | all convolutions remain at full resolution                                            | Useful conversion smoke test, not preferred final model |
| Proposed TinyDS/TinyU             | Designed for compatibility                    | verify depthwise/dilation/resize on selected backend                                  | Preferred hls4ml path                                   |

### 8.3 `io_parallel` versus `io_stream`

- `io_parallel` is useful for the earliest small smoke test but may explode resources for 64x64 or 128x128 images.
- `io_stream` is the realistic starting point for larger CNNs. Each pixel carries all channels in parallel, so 86-band input still creates a wide interface.
- Branches and skip connections require FIFOs and balancing. Run hls4ml's FIFO depth optimization after co-simulation.
- Full-scene streaming minimizes redundant overlap, but hls4ml models have static shapes and the surrounding geospatial pipeline is tile-oriented. A first implementation should use fixed 64x64 or 128x128 tiles, then evaluate whether a custom line-buffered full-scene wrapper is justified.

### 8.4 Quantization path options

1. **PTQ for feasibility:** train float PyTorch, export a simple graph, profile activation ranges, then test 16/12/10/8-bit fixed point.
2. **QAT through Keras/QKeras:** most mature hls4ml quantized route, but requires a maintained Keras mirror.
3. **QAT through Brevitas -> QONNX:** keeps PyTorch training but introduces QONNX cleaning, channels-last conversion, and quantization constraints.
4. **HGQ/HGQ2:** promising model-wise precision inference, primarily Keras-oriented and a larger workflow change.

Start with PTQ to prove architecture and synthesis. Move to QAT only after the float student is competitive and the HLS graph is stable.

### 8.5 Non-FPGA fallback: OpenVino / Myriad VPU

Section 6A reports a flight-proven (not benchmark-proxy) deployment of a PyTorch-trained encoder via OpenVino on an Intel Movidius Myriad X VPU, on the same class of hardware (quad-core x86 CPU + VPU + 2GB RAM) HyperspectralViTs used only as a ground proxy (Section 6.5). This is relevant to Section 8 specifically because it sidesteps most of the conversion blockers in Section 8.1: OpenVino's PyTorch/ONNX import supports attention, LayerNorm, and common activations (Hardswish/SiLU/GELU) that the hls4ml PyTorch frontend does not — subject to the toolchain pin below, since operator coverage is version-specific and should not be assumed from current OpenVino documentation.

**Import success is not device execution.** The claim above is about OpenVino's PyTorch/ONNX *frontend* accepting a layer into an IR graph, not about the **MYRIAD/HDDL plugin** actually executing that layer on the VPU. OpenVino's heterogeneous execution mode can silently fall back unsupported ops to the host CPU while the rest of the graph runs on the VPU; a model that "converts" cleanly may therefore still be partially CPU-bound, which changes both the latency/power numbers Section 8.5's parallel-benchmark use case depends on and whether the VPU-only operator set actually matches what Section 8.1 assumes. Any claim that an operator is "supported" for this fallback path must specify whether that means (i) accepted by the IR frontend, or (ii) confirmed to execute on the MYRIAD/HDDL plugin itself with no CPU fallback in the compiled graph.

**Toolchain pin required.** The RaVAEn flight evidence (Section 6A.2) was produced with OpenVino's legacy **MYRIAD/HDDL plugin** targeting the Intel Movidius Myriad X VPU. Intel deprecated and then removed the MYRIAD/HDDL plugin from mainline OpenVino starting with the 2023.0 release, replacing it with the newer NPU plugin for different silicon; **OpenVino 2022.3 LTS is the last release confirmed to support the MYRIAD/HDDL plugin the flight evidence used.** The flight results do not transfer to a different OpenVino version or a different VPU/NPU chip without re-validation. Before Alternative E is treated as an actionable fallback rather than historical evidence, either:

- (a) pin the toolchain to OpenVino 2022.3 LTS + MYRIAD/HDDL plugin on Myriad X hardware, and accept that this is an unsupported/EOL toolchain not eligible for current Intel support; or
- (b) re-validate the conversion and inference path on a current OpenVino release and a currently available/supported VPU or NPU board.

Section 13's "FPGA/SoC board and exact part: TBD" row must record which of these two applies.

This does not replace the hls4ml/HLS track — it trades away fixed-point/resource-level control (Section 12.1's LUT/FF/DSP/BRAM metrics do not apply to a VPU) in exchange for a much larger supported-operator set and a shorter path to a validated on-board system. Recorded as **Alternative E** in Section 9. Two ways to use it, both conditioned on the toolchain pin above:

1. **As a risk hedge:** if a candidate architecture (e.g. TinyU-4's skip connections, or any dilation/upsampling variant) fails hls4ml conversion or FIFO/BRAM synthesis gates (Section 12.3), the same float or PTQ model can likely still be exported through the pinned OpenVino toolchain, giving a fallback deployment path rather than a dead end.
2. **As a parallel benchmark:** measure OpenVino/VPU latency and power on the pinned toolchain and board alongside HLS synthesis results for the same candidate, to test whether the added engineering cost of HLS is actually justified by this project's specific latency/power targets (Section 13).

**Required smoke tests before relying on this path:**

1. convert a representative H1–H3 float checkpoint through the pinned OpenVino toolchain and confirm it stays within the plugin's supported operator subset for that version (Conv2D, depthwise Conv2D, BatchNorm, ReLU, nearest-neighbor upsample, concatenation — the same restricted op set Section 8.1 already requires for hls4ml, so both paths can share one constrained architecture);
2. inspect the compiled graph (e.g. OpenVino's per-layer device-affinity/execution report) to confirm every operator in that subset is actually assigned to the MYRIAD/HDDL plugin, not silently placed on host-CPU fallback — a CPU-fallback layer invalidates the resource/latency comparison in Section 8.5's parallel-benchmark use case even if the model still runs correctly;
3. run the **float** artifact end-to-end on the pinned VPU/board (or the currently available equivalent chosen under option (b) above), using the same fixed input layout (channels-first vs. channels-last, and tile size) the hls4ml path targets in Section 8.3, and confirm output parity against **the float PyTorch model** (Section 12.2 item 1) on the same golden test set used for the hls4ml float/quantized/C-sim/board comparison — parity must be checked against explicit, pre-registered float-comparison tolerances (e.g. max logit deviation and mask IoU/pixel-agreement threshold), not "matches" left undefined;
4. run a representative **PTQ** artifact (Section 8.4, option 1) end-to-end on the same target, not only the float checkpoint — Section 8.4 and Section 9 both treat PTQ as the intended feasibility path, so a float-only smoke test does not validate the path this section is meant to de-risk. Compare the on-device PTQ output against **the quantized software-emulation reference** (Section 12.2 item 2), not against the float PyTorch model — comparing PTQ-on-device to float would conflate expected quantization error with actual conversion/device fidelity loss, and use its own separate, pre-registered logit/mask tolerances (device-vs-quantized-emulation tolerances are not the same numbers as step 3's device-vs-float tolerances). Report PTQ-vs-float degradation as its own, independent measurement — e.g. against Section 12.3's quantized-degradation gate — rather than folding it into either device-parity check.

Until all four pass, Alternative E's deployment claim should be read as "float/PTQ IR import is plausible," not "float/PTQ inference is validated on-device."

---

## 9. System-boundary alternatives

### A. Feature-based FPGA model: RGB + WMF/Mag1c

**Benefits**

- only four input channels;
- common feature representation supports cross-sensor deployment;
- uses this project's existing data and contracts;
- smallest model and I/O path;
- easiest hls4ml conversion.

**Costs/risks**

- matched-filter computation may dominate latency and energy;
- network cannot recover evidence lost by the MF;
- WMF and Mag1c are not equivalent; the current project trains Mag1c while MARS favors WMF;
- on-board generation of WMF needs separate architecture and verification.

**Use when:** WMF is already produced elsewhere, downlink/ground inference is acceptable, or a complete system benchmark proves MF is not the bottleneck.

### B. End-to-end spectral FPGA model

**Benefits**

- removes matched-filter latency;
- can reject confounders using original spectral evidence;
- supports a genuinely on-board detector.

**Costs/risks**

- 86x input bandwidth and buffering;
- sensor-specific wavelength grids and calibration drift;
- larger external-memory footprint;
- more difficult quantization and data acquisition;
- current project lacks EMIT all-band training/evaluation data.

At 16-bit, one 128x128x86 input is about **2.69 MiB**, versus **128 KiB** for four channels. A 1280x1242x86 granule is roughly **261 MiB** before intermediate buffers. It cannot be treated as an on-chip tensor; streaming/external memory is mandatory.

**Use when:** on-board latency/energy includes MF, sufficient sensor-specific data exists, and the memory interface can sustain the spectral stream.

### C. Hybrid spectral frontend + feature/student network

Possible forms:

- fixed or learned 1x1 projection from 86 bands to 8–16 latent channels;
- a physically initialized spectral filter bank followed by learned spatial cleanup;
- FPGA spectral projection plus CPU/FPGA spatial CNN;
- sensor-specific projection matrices feeding a shared spatial backbone.

This may offer the best balance: retain original spectral information, avoid attention, and isolate sensor-specific wavelengths in a small frontend.

### D. Cascaded detector

- Stage 1: extremely cheap high-recall proposal model;
- Stage 2: more precise student only on proposed crops;
- host/FPGA scheduler skips Stage 2 for empty areas.

A cascade only saves compute if the scheduler can genuinely avoid Stage 2. Putting both stages in a fixed always-running dataflow graph does not provide the same benefit.

### E. Non-FPGA fallback: VPU + OpenVino (flight-proven on an EOL toolchain; historical evidence until re-pinned)

**Toolchain constraint (see Section 8.5 for full detail):** the flight evidence used OpenVino's MYRIAD/HDDL plugin on Myriad X, last supported in **OpenVino 2022.3 LTS**; that plugin is removed from current mainline OpenVino. Treat everything below as **historical evidence of feasibility**, not a currently reproducible fallback, until either the 2022.3 LTS + MYRIAD/HDDL toolchain is explicitly pinned and accepted as EOL, or the path is re-validated on a current OpenVino release and a currently available VPU/NPU board — and until the smoke tests in Section 8.5 pass on whichever toolchain is chosen.

**Benefits**

- flight-proven on real hardware (D-Orbit ION SCV004; Section 6A), not only a benchmark proxy — conditional on reproducing the same toolchain (OpenVino 2022.3 LTS + MYRIAD/HDDL) or re-validating on a current equivalent;
- much larger supported-operator set than hls4ml's PyTorch frontend (Section 8), avoiding the attention/LayerNorm/Hardswish/SiLU/GELU conversion blockers listed in Section 8.1–8.2 — subject to confirming the operator subset on the pinned toolchain version specifically;
- lower engineering risk and shorter iteration loop than HLS synthesis: no C-sim/RTL co-sim/timing-closure cycle;
- supports on-board training of a small head on top of a frozen encoder (Section 6A.2), which hls4ml does not target at all.

**Costs/risks**

- loses HLS's fine-grained fixed-point precision and per-layer resource control, so it is not a like-for-like substitute if the FPGA/power/resource budget is the hard constraint;
- Myriad X is a specific, aging Intel VPU product line, and its OpenVino plugin (MYRIAD/HDDL) is deprecated and removed from mainline OpenVino releases after 2022.3 LTS — there is no guarantee that a *current* board/VPU/NPU offers equivalent OpenVino support or availability (Section 13);
- no LUT/FF/DSP/BRAM data exists for this path because it is not an FPGA target; the resource/power comparisons in Section 12.1 don't directly apply;
- still subject to the same I/O/tiling overhead documented in Section 6A.2 and Section 5.5.

**Use when:** the FPGA resource/power budget is not yet the binding constraint, a faster path to a validated on-board system is more valuable than HLS-level control, or as a parallel benchmark to sanity-check whether hls4ml's added engineering cost is justified for this project's actual latency/power targets — **and** the toolchain pin and smoke tests in Section 8.5 have actually passed. Scheduling those steps is not itself sufficient. Until the toolchain is pinned and the smoke tests pass, keep this alternative in the experiment portfolio as motivating historical evidence only — not as a deployment option, a benchmark comparison, or a fallback.

---

## 10. Falsifiable model hypotheses

### H0 — Threshold/morphology is the hardware floor

**Hypothesis:** a fixed WMF threshold plus 3x3 opening can meet a useful recall floor at negligible neural cost, but cannot meet the operational false-alert budget.

**Test:** implement bit-exact threshold, erosion, dilation, connected components, and minimum-area filtering. Compare per-granule alerts and event recall against learned models.

**Value:** establishes the minimum latency/resource baseline and validates the system data path.

### H1 — TinyDS-4 can replace a large post-MF U-Net

**Architecture sketch, 128x128:**

```text
4ch input
 -> Conv3x3 4->8 + BN + ReLU
 -> [DepthwiseConv3x3 + PointwiseConv1x1 + BN + ReLU] x4
 -> Conv1x1 8->1 logits
```

Use constant width or a small 8->12->16->16->8 progression. Optional dilation `[1, 2, 4, 8]` enlarges context without down/up sampling, but must pass backend conversion and line-buffer synthesis tests.

A representative 8/12/16 version is about **1.4k convolution weights** and **23.5M MACs per 128x128 tile** before BN folding. Exact counts must be generated from the implemented graph.

**Hypothesis:** explicit MF input already contains the target signal, so a shallow spatial confounder filter can retain most of the useful accuracy while reducing model storage by orders of magnitude.

**Failure mode:** insufficient receptive field causes fragmented or missed plume tails.

### H2 — TinyU-4 improves morphology enough to justify skip buffers

**Architecture sketch:**

```text
4x128x128
 -> 8x128x128
 -> stride-2 DSConv -> 12x64x64
 -> stride-2 DSConv -> 16x32x32
 -> nearest upsample + concat(12x64x64) -> DSConv
 -> nearest upsample + concat(8x128x128) -> DSConv
 -> 1x1 logits
```

Use only two scales, nearest-neighbor `nn.Upsample`, and pairwise concatenation. Avoid transposed convolution and attention.

**Hypothesis:** one or two low-resolution context stages reduce false positives from roads/buildings and better connect plume shapes than H1.

**Failure mode:** skip FIFOs dominate BRAM, and tiled edges create artifacts.

### H3 — SpectralTiny-86 removes MF without transformer complexity

**Architecture sketch:**

```text
86x64x64 or 86x128x128
 -> Conv1x1 86->16 + BN + ReLU       # spectral projection
 -> four depthwise-separable spatial blocks
 -> Conv1x1 16->1 logits
```

A 16-channel full-resolution version is approximately **3.0k convolution weights** and **49M MACs per 128x128 tile**. The 1x1 spectral layer contributes about 22.5M MACs at that size.

**Hypothesis:** the key HyperspectralViTs gain comes from explicit spectral mixing and preserved resolution, not attention. A tiny CNN can learn sufficient methane-sensitive projections when pretrained on synthetic data and fine-tuned on real data.

**Failure modes:**

- sensor-specific overfitting;
- 86-channel stream width and memory dominate energy;
- a 16-channel bottleneck still discards weak spectral signatures;
- quantization damages low-amplitude absorption evidence.

### H4 — Fixed physical projection improves SpectralTiny generalization

Initialize some 1x1 filters from methane transmittance/matched-filter templates and train them with either constraints or a mixture of fixed and learned filters.

**Hypothesis:** physically informed initialization reduces data requirements and stabilizes low-bit training while learned channels model confounders/background.

**Counter-test:** compare fully learned, fully fixed, and hybrid projection banks under identical parameter and bit budgets.

### H5 — Ensemble distillation captures operational false-positive suppression

Teacher: average logits/probabilities from the five MARS U-Nets or a high-performing HyperSegFormer/EfficientViT ensemble. Student: H1, H2, or H3.

Suggested objective:

```text
L = BCE_or_Focal(student, label)
  + λ1 * soft_logit_distillation(student, teacher)
  + λ2 * boundary_or_Dice_loss(student, label)
  + λ3 * false-positive hard-negative loss
```

**Hypothesis:** most ensemble benefit is smooth uncertainty/regularization that a compact student can learn, yielding a larger false-alert improvement than ordinary label-only training.

**Critical data requirement:** distill on broad full-granule background/OOD crops, not only plume-centered tiles.

### H6 — Hard-negative curriculum matters more than another model block

Mine false positives from cities, roads, rivers, mountains, solar panels, roofs, dunes, and minerals. Include sensor, geography, sector, and season labels where possible.

**Hypothesis:** adding diverse full-granule negatives produces a greater reduction in alerts/granule than doubling model channels.

This follows MARS's conclusion that OOD background is underrepresented in plume-centered training data.

### H7 — Lower-resolution proposals plus host postprocessing are sufficient

Predict at 1/2 or 1/4 resolution, use fixed nearest upsampling, and apply connected-component ranking externally.

**Hypothesis:** for alert generation, exact boundaries are less important than event localization, so reduced-resolution output can greatly reduce compute without hurting event recall.

**Failure mode:** small/weak plume events and narrow tails disappear. Evaluate by plume strength and size, not aggregate F1 only.

### H8 — A two-head confidence design improves ranking

Output:

- segmentation logits;
- a coarse tile/event confidence computed from a fixed pooling path.

**Hypothesis:** jointly learned confidence ranks connected components better than using only maximum pixel probability.

Keep the pooling shape static. If multi-output conversion proves fragile, compute confidence in postprocessing instead.

### H9 — Frozen encoder + on-board retrainable head enables in-orbit adaptation

Motivated directly by the RaVAEn flight result (Section 6A.2): a frozen spectral/spatial encoder produces a compact per-tile or per-pixel latent representation; a small trainable head (linear or shallow MLP, analogous to RaVAEn's 129-parameter classifier) is (re)trained **on-board** from a small number of newly labeled or newly confirmed tiles, without a ground uplink of new weights.

**Architecture sketch:**

```text
frozen spatial/spectral encoder (e.g. TinyDS/SpectralTiny backbone, weights fixed after ground training)
 -> per-tile or per-pixel latent vector
 -> small trainable head (linear / 1-2 layer MLP), retrained on-board from few-shot labels
```

**Hypothesis:** most of the domain/sensor drift and hard-negative false-alert problems identified in H5/H6 do not require a full model redeployment to correct; a small head retrained on-board from a handful of confirmed detections or confirmed false alarms can recover much of the accuracy loss, at training cost consistent with RaVAEn's measured 0.09–0.2s/epoch for a comparably sized head.

**Relationship to existing hypotheses:** this is complementary to, not a replacement for, H1–H3 (which decide the frozen encoder's architecture) and H8 (two-head confidence design — the on-board-retrainable head could be the confidence head specifically, which is lower-risk than retraining the segmentation head itself).

**Failure modes:**

- a segmentation encoder's latent space may not be as linearly separable as RaVAEn's cloud-detection VAE latents, so a linear/shallow head may underfit;
- on-board retraining needs a labeling mechanism (confirmed detection, operator uplink, or heuristic pseudo-label) that this project does not yet have;
- retraining introduces a state-management problem (versioning, rollback, drift monitoring) absent from a fixed-weight deployment, and is out of scope for hls4ml/HLS synthesis — this hypothesis is realistic only on the VPU/OpenVino or CPU fallback path (Alternative E in Section 9), not on a static HLS bitstream.

**Priority:** exploratory / P2. Do not block the P0/P1 portfolio in Section 1 on this hypothesis; treat it as a system-level extension to evaluate once a frozen encoder from H1–H3 is validated.

### H9.1 — Required evaluation protocol (precondition for comparison with H1–H3)

H9 answers a different question from H1–H3 (on-board adaptability, not frozen-weight segmentation accuracy) and Section 12's metric hierarchy and go/no-go gates do not cover it. Before H9 is compared against H1–H3 in the Section 1 portfolio table or promoted past exploratory status, define:

1. **Frozen-encoder checkpoint:** the exact H1–H3 architecture, weights, training data, and checkpoint version that is frozen and reused as the encoder. H9 has no independent encoder; without pinning this, "H9's accuracy" is undefined.
2. **Tile-level versus pixel-level task:** whether the on-board head predicts a whole-tile label (RaVAEn's task) or a per-pixel mask (H1–H3's task). These are not interchangeable, and Section 12.1's primary metrics assume per-pixel/per-event output.
3. **Adaptation-label budget:** the number and source of on-board few-shot labels available for retraining the head (e.g., N confirmed detections + N confirmed false alarms), analogous to RaVAEn's 1,305-tile budget, fixed in advance rather than tuned post hoc.
4. **Held-out scenes:** scenes/sensors excluded from encoder pretraining, head adaptation, *and* the rollback guard set (item 6), used exclusively to report final post-adaptation gain or regression. Held-out scenes must never be consulted by any automatic on-board decision — only by final reporting — or the reported numbers are no longer an unbiased estimate.
5. **Adaptation time:** the wall-clock/epoch budget for on-board retraining under the target on-board compute envelope (VPU/CPU per Section 8.5/Alternative E, not FPGA), reported against RaVAEn's measured 0.09–0.2 s/epoch as a reference point, not an assumption that this project's head/hardware will match it.
6. **Rollback guard set and rollback rule:** rollback is an automatic on-board decision, so it cannot use the held-out scenes from item 4 without contaminating their status as an unbiased final-report set. Pin a separate, labeled **rollback guard set** — distinct from both the adaptation-label budget (item 3) and the held-out scenes (item 4), carried on-board for this purpose only and excluded from all final accuracy/regression reporting — and define an explicit, automatic condition for reverting the head to its last known-good state if on-board retraining degrades performance on that guard set (e.g., revert if guard-set event recall at the fixed false-alert budget drops by more than X points). If no separate guard set is feasible under the on-board storage/label budget, rollback must instead be a post-run, ground-side decision made after downlinking adaptation telemetry, with held-out scenes still reserved for evaluation-only reporting.
7. **Acceptance metric:** the specific metric and threshold for accepting an on-board-adapted head as an improvement over the frozen baseline, expressed in Section 12.1's primary operational terms (event recall at a fixed false-alert budget) and measured on the held-out scenes only, not pixel F1/AUPRC alone and not the rollback guard set — since pixel-level metrics are what H1–H3 optimize for and may not be reported for a tile-level H9 head, and guard-set performance is a decision signal, not a reportable accuracy result.

Until these seven items are fixed, H9's row in the Section 1 portfolio table records a research direction, not a result comparable to H1–H3's "main question" framing; treat any accuracy or timing claim attributed to H9 as provisional.

### H9.2 — Acceptance gates

H9 stays **exploratory/P2** regardless of gate outcome; passing these gates makes H9 a reportable result, not a promotion into the P0/P1 portfolio (Section 1) or the general go/no-go gates (Section 12.3), which do not cover on-board retraining. Report all five together from the same adaptation run — a single passing metric without the rest is not sufficient. They are not all held-out-set measurements: gates 1 and 5 are run metadata (the label budget actually used and the measured training cost), while gates 2–4 are the held-out-scene evaluation outputs (adapted and reverted-baseline performance) that budget and cost apply to:

1. **Few-shot label budget:** adaptation uses no more than the budget pinned in H9.1 item 3; report the result at exactly that budget, not a best-of-several-budgets figure chosen after the fact.
2. **Held-out-scene evaluation:** post-adaptation gains are measured only on the scenes/sensors held out per H9.1 item 4 — never seen during encoder pretraining, head adaptation, or the rollback guard set/decision (H9.1 item 6).
3. **Segmentation/per-pixel accuracy:** if the H9 head is per-pixel rather than tile-level (H9.1 item 2), it must be scored with the same per-pixel/per-event metrics H1–H3 use (Section 12.1) on the held-out set. **RaVAEn's binary per-tile AUPRC 0.979 / F1 0.956 (Section 6A.2) is evidence for tile-level cloud classification only; it is not evidence for methane segmentation accuracy or for a per-pixel H9 head, and must not be cited as such.**
4. **Rollback to frozen baseline:** the adapted head is reverted automatically per the H9.1 item 6 rollback rule, using the separate rollback guard set (or a post-run ground-side decision) — never the held-out scenes — if performance regresses versus the frozen baseline; report both the adapted and the reverted-baseline numbers on the held-out set, not the adapted number alone.
5. **Measured on-board training cost:** report actual wall-clock/epoch training time on the pinned on-board target (Alternative E's pinned toolchain, Section 8.5). RaVAEn's 0.09–0.2 s/epoch (Section 6A.2) is a reference point only, not a substitute for measuring this project's own head size and hardware.

---

## 11. Training and data strategy

### 11.1 Required datasets

Use a staged program:

1. **STARCOP mini:** conversion and training smoke tests only.
2. **STARCOP raw:** AVIRIS model comparison and quantization development.
3. **OxHyperSyntheticCH4:** spectral-student pretraining.
4. **OxHyperRealCH4 or MARS EMIT:** real EMIT fine-tuning.
5. **MARS full-granule sets:** false-alert and object-level acceptance.
6. **PRISMA/EnMAP:** cross-sensor tests for feature-based models; separate projection/fine-tuning tests for all-band models.

Track every dataset/split with DVC. Prevent source-granule, site, and temporal leakage.

### 11.2 Imbalance and sampling

Do not rely on one method alone:

- scene/tile sampler balancing;
- pixel loss weighting or focal/Tversky loss;
- explicit no-plume and confounder tiles;
- hard-negative mining from model errors;
- report natural-prevalence validation separately from balanced training batches.

MF-weighted loss helps focus on difficult high-enhancement pixels but can also bias a supposedly end-to-end model toward MF behavior. For H3/H4, compare training with and without MF-derived weights.

### 11.3 Augmentation

Preserve physically valid transformations:

- rotations/flips are generally acceptable for plume morphology;
- spatial jitter is useful;
- spectral jitter should model calibrated sensor uncertainty, not arbitrary color augmentation;
- band dropout can test robustness but may not represent real correlated sensor failure;
- simulate quantization noise during QAT;
- include no-data boundaries and tile seams.

If wind is used, rotate/flip wind vectors consistently. Wind gave only a small MARS gain and costs two channels; treat it as an ablation, not a default.

### 11.4 Synthetic-to-real transfer

Follow the evidence from HyperspectralViTs:

1. pretrain H3/H4 on synthetic methane inserted into clean EMIT L1B;
2. fine-tune on real expert-validated plumes;
3. calibrate thresholds on a held-out temporal validation split;
4. test on untouched future/full-granule data.

Synthetic-only success is insufficient because zero-shot synthetic-to-real performance was weak in the reference study.

---

## 12. Evaluation specification

### 12.1 Metric hierarchy

**Primary operational metrics**

- event recall at a fixed false-alert budget;
- false alerts per granule and per megapixel/km²;
- detected/missed events by plume size, flux/strength, sensor, sector, geography, and surface type;
- precision-recall curve for ranked connected components;
- workload: candidates an analyst must inspect to find N true plumes.

**Secondary segmentation metrics**

- AUPRC;
- methane-class F1 and IoU;
- precision and recall;
- boundary F1 or shape overlap;
- calibration error/Brier score.

**Hardware metrics**

- LUT, FF, DSP, BRAM, URAM;
- achieved clock and timing slack;
- latency/tile, initiation interval, tiles/s, effective megapixels/s;
- input/output bandwidth;
- power/energy per tile and per granule where measurable;
- C-sim and RTL co-sim agreement.

### 12.2 Required comparisons

For each candidate compare:

1. float PyTorch;
2. quantized software emulation;
3. hls4ml C simulation;
4. RTL co-simulation;
5. board output;
6. teacher and classical baseline.

Use the same serialized test vectors and compare intermediate layers where possible. Validate logits before thresholded masks because a single threshold can hide numerical drift.

### 12.3 Suggested go/no-go gates

Finalize values only after the target platform is selected. Initial gates:

- no unsupported/fallback operators in the exported graph;
- float student retains at least 95% of the selected teacher's primary validation score **and** stays within the full-granule false-alert budget;
- quantized degradation no more than 1 absolute F1/AUPRC point and no more than 5% relative event-recall loss at fixed false alerts;
- C-sim and Python produce identical masks on the golden set after the agreed threshold;
- synthesis uses no more than 70% of each critical FPGA resource, leaving integration margin;
- timing closes with at least 10% clock margin;
- end-to-end throughput, including preprocessing and I/O, exceeds the sensor production rate by at least 2x.

These are starting hypotheses, not universal guarantees.

---

## 13. Hardware sizing questions that must be answered

Create a target-platform record before model selection:

| Requirement                                                 | Value needed |
| ----------------------------------------------------------- | ------------ |
| FPGA/SoC board and exact part                               | TBD          |
| HLS vendor/version                                          | TBD          |
| Clock target                                                | TBD          |
| Allowed LUT/FF/DSP/BRAM/URAM                                | TBD          |
| External memory and sustainable bandwidth                   | TBD          |
| Power/energy budget and duty cycle                          | TBD          |
| Input processing level: L0/L0.5/L1B/reflectance/MF          | TBD          |
| Sensor and number/order of bands                            | TBD          |
| Tile size/overlap or full-scene stream                      | TBD          |
| Required granules/day and maximum latency/granule           | TBD          |
| Output: mask, boxes/components, score, compressed telemetry | TBD          |
| Radiation/reliability constraints                           | TBD          |
| Reconfiguration allowed?                                    | TBD          |

Without these values, “small enough for FPGA” is not a testable statement.

### 13.1 Memory realities

- Four-channel 128x128 input at 16-bit: 128 KiB.
- 86-channel 128x128 input at 16-bit: ~2.69 MiB.
- 6.69M weights: ~26.8 MB FP32, ~13.4 MB int16, ~6.69 MB int8.
- Five 6.69M models: ~134 MB FP32 or ~33.5 MB int8, excluding activations.
- A 3k-weight student: only a few KiB at 8–16 bit; activation and line buffers become dominant.

Thus, a tiny student changes the bottleneck from weight memory to pixels, FIFOs, line buffers, and external I/O—the right regime for streaming HLS.

---

## 14. Implementation plan in this project

### Phase 0 — Decisions and baselines

1. Select target board/toolchain and fill Section 13.
2. Decide whether WMF/Mag1c is inside or outside the accelerated boundary.
3. Add MARS/OxHyper datasets as DVC datasets only after checking their data licenses.
4. Reproduce float baselines on natural-prevalence and full-granule splits.
5. Freeze metric and connected-component definitions.

### Phase 1 — Conversion spike

Implement a new local model rather than modifying vendor code:

```text
src/models/hls/
  tiny_ds.py
  tiny_u.py
  spectral_tiny.py
  export.py
  equivalence.py
configs/model/
  tiny_ds_4ch.yaml
  tiny_u_4ch.yaml
  spectral_tiny_86ch.yaml
configs/hardware/
  <board>.yaml
```

Requirements:

- plain `nn.Module`;
- static input shape;
- module-based activations and upsampling;
- no data-dependent Python control flow;
- no Hardswish/SiLU/GELU initially;
- explicit pairwise merges;
- unit tests for shape, FX trace, export, and numerical equivalence.

Convert a one-block model first, then add blocks one at a time. This isolates frontend failures.

### Phase 2 — Float architecture search

Use a constrained search space:

- spectral width: 8/12/16/24;
- spatial width: 8/12/16/24;
- blocks: 2–6;
- scales: 1–3;
- tile: 64 or 128;
- standard versus depthwise convolution;
- optional dilation;
- H1/H2/H3 input regimes.

Optimize a multi-objective score, not only F1:

```text
maximize event recall and AUPRC
minimize false alerts, MACs, peak activation, and estimated FPGA resources
subject to conversion compatibility
```

### Phase 3 — Distillation and hard negatives

- generate teacher probability maps on full granules;
- cache maps with teacher/model/data hashes;
- mine high-confidence false positives;
- train students with label + teacher + boundary/hard-negative terms;
- recalibrate connected-component thresholds after distillation.

### Phase 4 — Quantization

- profile post-BN-folding weights/activations;
- test fixed 16, 12, 10, and 8-bit baselines;
- assign wider accumulators than weights/activations;
- inspect overflow and saturation by layer;
- introduce QAT if PTQ misses gates;
- keep input normalization constants and thresholds in the bit-exact contract.

### Phase 5 — HLS and board validation

For each promoted experiment, log to MLflow:

- Git/data/model/config/toolchain hashes;
- frontend/backend and FPGA part;
- precision/reuse/strategy per layer;
- generated HLS project artifact;
- C-sim/RTL co-sim reports;
- synthesis utilization/timing;
- board latency/power;
- float, quantized, C-sim, and board metrics on the same golden set.

Run FIFO-depth optimization for `io_stream` designs after functional co-simulation.

### Phase 6 — System validation

Benchmark complete granules including:

- storage read;
- calibration/reprojection;
- optional MF;
- tiling and overlap;
- accelerator transfer;
- model;
- stitching, connected components, ranking, and geospatial output.

A network-only speedup is not a deployment result.

---

## 15. Risks and mitigations

| Risk                                   | Consequence                             | Mitigation                                             |
| -------------------------------------- | --------------------------------------- | ------------------------------------------------------ |
| No software license in reference repos | Cannot safely copy/integrate code       | Independent implementation; request license            |
| Target hardware unspecified            | Architecture decisions are arbitrary    | Freeze board/clock/power/I/O first                     |
| MF excluded from benchmark             | Misleading system speedup               | Report network-only and end-to-end timing              |
| Patch-centric training                 | Excess full-scene false alerts          | Full-granule negatives, hard mining, object metrics    |
| Current AVIRIS-only data               | EMIT/global claims unsupported          | Add temporal/global multi-sensor datasets              |
| All-band bandwidth                     | Model compute ceases to be bottleneck   | Spectral projection, streaming, bandwidth model        |
| Quantization of weak signals           | Recall loss for faint plumes            | QAT, per-layer precision, strength-stratified metrics  |
| Tile boundaries                        | Broken plume masks and duplicate alerts | overlap/stitch tests or line-streamed inference        |
| hls4ml PyTorch frontend maturity       | Conversion failures or wrong layouts    | restrictive graph, Keras/QONNX fallback, golden layers |
| Skip/branch FIFOs                      | BRAM explosion/deadlock                 | shallow branches, FIFO optimization, synthesis gates   |
| Cross-sensor spectral mismatch         | all-band model fails zero-shot          | sensor-specific projection/frontends and fine-tuning   |
| Ensemble benefit lost in student       | false alerts return                     | distill on full-granule OOD data and validate workload |
| Threshold overfitting                  | optimistic reported F1                  | validation-only calibration; locked test threshold     |
| L1B assumption on-board                | unavailable input product               | model partial preprocessing or simulate L0-like data   |
| Radiation/toolchain/platform limits    | board demo not flight ready             | separate ML feasibility from flight qualification      |
| hls4ml/HLS the only deployment path evaluated | a convertible-but-blocked candidate is abandoned instead of shipped | benchmark OpenVino/VPU (Alternative E, Section 8.5) in parallel as a fallback, but first pin the toolchain (OpenVino 2022.3 LTS + MYRIAD/HDDL is EOL; a current release needs re-validation) and pass the Section 8.5 conversion/inference smoke tests — otherwise treat Alternative E as historical evidence, not an available fallback |
| On-board retraining (H9) has no labeling/versioning mechanism | drift correction stays theoretical | scope H9 as exploratory/P2; do not depend on it for the P0/P1 portfolio |

---

## 16. Decisions recommended now

1. **Adopt TinyDS-4 and SpectralTiny-86 as the two lead hypotheses.** They test the central system trade-off directly.
2. **Use MARS U-Net ensemble and HyperSegFormer/EfficientViT only as teachers/reference points.** Do not make direct conversion the critical path.
3. **Make full-granule false alerts and event recall primary promotion metrics.** Pixel F1 remains necessary but not sufficient.
4. **Include preprocessing in every end-to-end benchmark.** Report WMF cost explicitly.
5. **Keep the first graph deliberately boring.** Conv2D/depthwise Conv2D/BN/ReLU/fixed resize/simple merge gives the best chance of bit-accurate hls4ml success.
6. **Treat hard-negative data and distillation as first-class model components.** The reference results imply they may matter more than model width.
7. **Resolve software and dataset licenses before integrating any reference assets.**
8. **Create an isolated Linux hardware environment.** The current macOS development machine is not a supported hls4ml/HLS synthesis platform.

---

## 17. Source references

### Repositories

- [UNEP-IMEO-MARS/marsml-hyperspectral](https://github.com/UNEP-IMEO-MARS/marsml-hyperspectral), analyzed at `ebc608b`.
- [previtus/HyperspectralViTs](https://github.com/previtus/HyperspectralViTs), analyzed at `a184a25`.
- [spaceml-org/STARCOP](https://github.com/spaceml-org/STARCOP), project-pinned submodule.
- [fastmachinelearning/hls4ml](https://github.com/fastmachinelearning/hls4ml), converter source inspected at `b90fb06`.

### Papers

- Růžička et al., [Operational machine learning for remote spectroscopic detection of CH4 point sources](https://arxiv.org/abs/2511.07719), v2.
- Růžička and Markham, [HyperspectralViTs: General Hyperspectral Models for On-Board Remote Sensing](https://arxiv.org/abs/2410.17248), IEEE JSTARS 2025.
- Růžička et al., [Semantic segmentation of methane plumes with hyperspectral machine learning models](https://www.nature.com/articles/s41598-023-44918-6), Scientific Reports 2023.
- Růžička et al., [Fast Model Inference and Training On-Board of Satellites](https://arxiv.org/abs/2307.08700), arXiv:2307.08700, 2023 — analyzed in Section 6A; motivates H9 and Alternative E.

### hls4ml documentation

- [Status and Features](https://fastmachinelearning.org/hls4ml/intro/status.html)
- [Concepts: I/O, reuse, and strategy](https://fastmachinelearning.org/hls4ml/api/concepts.html)
- [Configuration](https://fastmachinelearning.org/hls4ml/api/configuration.html)
- [PyTorch frontend](https://fastmachinelearning.org/hls4ml/frontend/pytorch.html)
- [ONNX/QONNX frontend](https://fastmachinelearning.org/hls4ml/frontend/qonnx.html)
- [Profiling](https://fastmachinelearning.org/hls4ml/advanced/profiling.html)
- [Automatic precision inference](https://fastmachinelearning.org/hls4ml/advanced/auto.html)
- [FIFO depth optimization](https://fastmachinelearning.org/hls4ml/advanced/fifo_depth.html)
- [Hardware-aware optimization](https://fastmachinelearning.org/hls4ml/advanced/model_optimization.html)

### Internal project evidence

- [`dataset_report.md`](dataset_report.md)
- [`baseline_metrics.md`](baseline_metrics.md)
- [`../configs/dataset/starcop_raw.yaml`](../configs/dataset/starcop_raw.yaml)
- [`../src/training/train.py`](../src/training/train.py)
- [`../src/serving/inference.py`](../src/serving/inference.py)
