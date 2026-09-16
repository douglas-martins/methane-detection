# First run: Architecture & Results 

## 0. Glossary

| Term | Meaning |
| --- | --- |
| **E1** | `TinyUNet`: a small U-Net built from scratch for this course project, no pretraining. |
| **E2** | U-Net decoder + **MobileNetV2** encoder, ImageNet-pretrained the existing STARCOP baseline architecture. |
| **E3** | LinkNet decoder + **MobileNetV3-small-minimal** encoder, ImageNet-pretrained a much smaller candidate (reproduced from Herec et al.'s on-board methane detection work). |
| **`starcop_mini`** (**mini** tier) | Small, curated subset of STARCOP (392 train / 49 val / 441 test patches) fast iteration loop, source of the graded headline comparison. |
| **`starcop_raw`** | The full STARCOP dataset (141,218 train / 26,607 val / 16,758 test patches) used to confirm mini-scale conclusions hold at real scale. |
| **R1** | *Contract confirmation*: does the code run correctly on real `starcop_raw` data (shapes, dtypes, no OOM)? Minutes, no training. |
| **R2** | *Subsample confirmation*: E1/E2/E3 trained on a seeded ~6,076-patch subsample of `starcop_raw`'s train split. |
| **R3** | *Full-scale confirmation*: E2/E3 trained on `starcop_raw`'s entire 141,218-patch train split (E1 skipped, least informative, by design). |
| **PR-AUC** | Area under the Precision-Recall curve, computed over a 101-point decision-threshold sweep, threshold-independent, unlike a single Precision/Recall/F1 number at a fixed cutoff. |
| **`pos_weight`** | Background:positive pixel ratio fed into the loss (`BCEWithLogitsLoss`) to counter class imbalance, differs per tier since imbalance itself differs (mini ≈87, R2 ≈270, R3 ≈314). |
| **"Detected at all"** | Per-patch metric: among patches with ≥1 true methane pixel, how many got ≥1 predicted positive pixel, a coarser, "did it notice anything" companion to the pixel-level metrics. |
| **TOA** | Top-Of-Atmosphere reflectance the satellite/airborne sensor's raw calibrated measurement, before atmospheric correction. |
| **`mag1c`** | A physically-derived methane concentration-enhancement product one of this project's 4 input channels, not a raw reflectance band. |

## 1. Domain context: STARCOP and `mag1c`

**STARCOP** (Růžička et al., 2023, *Scientific Reports*) is a benchmark for
automated detection of methane point-source plumes from airborne
imaging-spectrometer data (AVIRIS-NG overflights). The task this project
tackles is **per-pixel binary segmentation**: for each pixel of a
128×128 patch, does it show part of a methane plume?

Each scene provides several candidate input products; this project uses
**4 as the model's input channels**:

- **`mag1c`**: a matched-filter-family algorithm (alongside classical
  alternatives like MF/CEM/ACE) that estimates a per-pixel methane
  concentration-enhancement signal directly from the spectrometer's
  spectral bands, *before* any learning happens. It's a
  physically-motivated feature, not a raw reflectance value, and by far
  the single strongest predictor of plume location, but it can be noisy
  or miss real plumes on its own.
- **`TOA_AVIRIS_640/550/460nm`**: three visible-wavelength reflectance
  bands, standing in for a simple RGB-like composite. These give the
  model context to reject false positives that the `mag1c` signal alone
  can't distinguish (e.g. certain bright/dark surfaces that confuse the
  matched filter).

All three architectures compared here (E1/E2/E3) take the **same
4-channel input contract** and produce a **single-channel raw-logit
output** (sigmoid + a 0.5 decision threshold applied downstream, for the
headline numbers), the input/output shape is held fixed; what differs is
everything in between.

## 2. Project scope, in one picture

> [!NOTE]
> `starcop_mini` develops and produces the graded numbers; `starcop_raw` is
> used throughout as a confirmation dataset, at three increasing levels of
> rigor:

```mermaid
flowchart LR
    Mini["starcop_mini<br/>392 train / 49 val / 441 test<br/>fast dev loop, graded headline numbers"]
    R1["R1: contract check<br/>few hundred real starcop_raw patches<br/>minutes, no training"]
    R2Node["R2: subsample confirmation<br/>~6,076 train patches (seeded)<br/>~1h per configuration"]
    R3Node["R3: full-scale confirmation<br/>141,218 train patches<br/>E2/E3 only, overnight"]
    Raw["starcop_raw test split<br/>16,758 patches<br/>real-scale ground truth for every comparison"]

    Mini --> R1
    R1 --> R2Node
    R2Node --> R3Node
    R2Node --> Raw
    R3Node --> Raw
    Mini -.cross-tier eval.-> Raw
```

## 3. Architectures: macro comparison

```mermaid
flowchart TB
    subgraph E1G["E1: TinyUNet (from scratch) 487,361 params, no pretraining"]
        direction LR
        E1in["Input<br/>4×128×128"] --> E1enc["Encoder<br/>4 downsampling stages"]
        E1enc --> E1bn["Bottleneck<br/>128 channels"]
        E1bn --> E1dec["Decoder<br/>4 upsampling stages<br/>concat skip connections"]
        E1dec --> E1out["Output<br/>1×128×128 raw logits"]
    end
    subgraph E2G["E2: U-Net + MobileNetV2 (ImageNet-pretrained) 6,629,233 params"]
        direction LR
        E2in["Input<br/>4×128×128"] --> E2enc["MobileNetV2 encoder<br/>5 stages"]
        E2enc --> E2bn["Bottleneck<br/>1280 channels"]
        E2bn --> E2dec["U-Net decoder<br/>5 stages<br/>concat skip connections"]
        E2dec --> E2out["Output<br/>1×128×128 raw logits"]
    end
    subgraph E3G["E3: LinkNet + MobileNetV3-small-minimal (ImageNet-pretrained) 856,635 params"]
        direction LR
        E3in["Input<br/>4×128×128"] --> E3enc["MobileNetV3-small encoder<br/>5 stages"]
        E3enc --> E3bn["Bottleneck<br/>576 channels"]
        E3bn --> E3dec["LinkNet decoder<br/>5 stages<br/>ADDITIVE skip connections"]
        E3dec --> E3out["Output<br/>1×128×128 raw logits"]
    end
    E1G ~~~ E2G ~~~ E3G
```

> [!IMPORTANT]
> The one structural difference worth remembering: E1/E2 merge encoder
> skip connections by **concatenation** (U-Net style); E3 merges them by
> **addition** (LinkNet style) cheaper, and part of why E3's decoder is
> so much smaller relative to its encoder.

## 4. Architectures: micro view (real, code-verified structure)

### 4.1 E1: TinyUNet

Every block is a `_ConvBlock` = two (Conv3×3 → BatchNorm → ReLU) layers
(`architectures.py`). Built entirely from scratch, no pretrained weights
anywhere in this graph.

```mermaid
flowchart TB
    In["Input: 4×128×128<br/>(mag1c, TOA 640/550/460nm)"] --> Enc1["enc1: ConvBlock<br/>4→8 ch, 128×128"]
    Enc1 --> Pool1["MaxPool /2"] --> Enc2["enc2: ConvBlock<br/>8→16 ch, 64×64"]
    Enc2 --> Pool2["MaxPool /2"] --> Enc3["enc3: ConvBlock<br/>16→32 ch, 32×32"]
    Enc3 --> Pool3["MaxPool /2"] --> Enc4["enc4: ConvBlock<br/>32→64 ch, 16×16"]
    Enc4 --> Pool4["MaxPool /2"] --> BN["bottleneck: ConvBlock<br/>64→128 ch, 8×8"]
    BN --> Up4["ConvTranspose /2<br/>128→64 ch"]
    Up4 & Enc4 --> Cat4["concat: 128 ch, 16×16"] --> Dec4["dec4: ConvBlock<br/>128→64 ch"]
    Dec4 --> Up3["ConvTranspose /2<br/>64→32 ch"]
    Up3 & Enc3 --> Cat3["concat: 64 ch, 32×32"] --> Dec3["dec3: ConvBlock<br/>64→32 ch"]
    Dec3 --> Up2["ConvTranspose /2<br/>32→16 ch"]
    Up2 & Enc2 --> Cat2["concat: 32 ch, 64×64"] --> Dec2["dec2: ConvBlock<br/>32→16 ch"]
    Dec2 --> Up1["ConvTranspose /2<br/>16→8 ch"]
    Up1 & Enc1 --> Cat1["concat: 16 ch, 128×128"] --> Dec1["dec1: ConvBlock<br/>16→8 ch"]
    Dec1 --> Head["head: 1×1 Conv<br/>8→1 ch"] --> Out["Output: 1×128×128<br/>raw logits"]
```

### 4.2 E2: U-Net + MobileNetV2

Encoder/decoder channel counts read directly from the instantiated model
(`build_e2().encoder.out_channels` / `.decoder.blocks`), not estimated.

```mermaid
flowchart TB
    In["Input: 4×128×128"] --> S0["Stage 0 (input passthrough)<br/>4 ch, 128×128"]
    S0 --> S1["Stage 1 (MobileNetV2)<br/>16 ch, 64×64"]
    S1 --> S2["Stage 2<br/>24 ch, 32×32"]
    S2 --> S3["Stage 3<br/>32 ch, 16×16"]
    S3 --> S4["Stage 4<br/>96 ch, 8×8"]
    S4 --> S5["Stage 5: bottleneck<br/>1280 ch, 4×4"]
    S5 --> D0["Decoder block 0<br/>1280+96 → 256 ch, 8×8"]
    S4 -.skip.-> D0
    D0 --> D1["Decoder block 1<br/>256+32 → 128 ch, 16×16"]
    S3 -.skip.-> D1
    D1 --> D2["Decoder block 2<br/>128+24 → 64 ch, 32×32"]
    S2 -.skip.-> D2
    D2 --> D3["Decoder block 3<br/>64+16 → 32 ch, 64×64"]
    S1 -.skip.-> D3
    D3 --> D4["Decoder block 4<br/>32 → 16 ch, 128×128<br/>(no further skip)"]
    D4 --> Head["Segmentation head: 1×1 Conv<br/>16→1 ch"] --> Out["Output: 1×128×128"]
```

### 4.3 E3: LinkNet + MobileNetV3-small-minimal

Same verification approach. Note the **additive** merge (`+`) instead of
`concat`, LinkNet's decoder blocks are 1×1-reduce → transpose-conv
upsample → 1×1-expand, then add the same-resolution encoder skip.

```mermaid
flowchart TB
    In["Input: 4×128×128"] --> S0["Stage 0 (input passthrough)<br/>4 ch, 128×128"]
    S0 --> S1["Stage 1 (MobileNetV3-small)<br/>16 ch, 64×64"]
    S1 --> S2["Stage 2<br/>16 ch, 32×32"]
    S2 --> S3["Stage 3<br/>24 ch, 16×16"]
    S3 --> S4["Stage 4<br/>48 ch, 8×8"]
    S4 --> S5["Stage 5: bottleneck<br/>576 ch, 4×4"]
    S5 --> D0["Decoder 0: reduce 576→144<br/>upsample, expand→48"]
    D0 --> Add0(("+"))
    S4 -.skip 48ch.-> Add0
    Add0 --> D1["Decoder 1: reduce 48→12<br/>upsample, expand→24"]
    D1 --> Add1(("+"))
    S3 -.skip 24ch.-> Add1
    Add1 --> D2["Decoder 2: reduce 24→6<br/>upsample, expand→16"]
    D2 --> Add2(("+"))
    S2 -.skip 16ch.-> Add2
    Add2 --> D3["Decoder 3: reduce 16→4<br/>upsample, expand→16"]
    D3 --> Add3(("+"))
    S1 -.skip 16ch.-> Add3
    Add3 --> D4["Decoder 4: reduce 16→4<br/>upsample, expand→32<br/>(no further skip)"]
    D4 --> Head["Segmentation head: 1×1 Conv<br/>32→1 ch"] --> Out["Output: 1×128×128"]
```

**Why E3's total param count (856,635) sits between E1 and E2, despite the
smallest encoder**: its decoder+head (427,603 params) is almost exactly
balanced with its encoder (429,032) LinkNet's additive, narrow decoder
is cheap. E2's decoder+head alone (4,405,073 params) is **~2× its own
encoder** (2,224,160) and **~5× E3's entire decoder**, U-Net's
concatenative decoder is the expensive half of that architecture, not the
pretrained encoder.

## 5. Code structure

```mermaid
classDiagram
    class architectures_py {
        <<module>>
        +build_e1() TinyUNet
        +build_e2(pretrained) smp.Unet
        +build_e3(pretrained) smp.Linknet
    }
    class TinyUNet {
        +enc1..enc4, bottleneck, dec1..dec4, head
        +forward(x) Tensor
    }
    class ConvBlock {
        +net: Conv3x3-BN-ReLU x2
        +forward(x) Tensor
    }
    TinyUNet *-- ConvBlock
    architectures_py ..> TinyUNet : builds E1

    class dataset_py {
        <<module>>
    }
    class PatchDataset {
        +patches_df
        +input_products, output_products
        +augmenter
        +__getitem__(index) dict
    }
    dataset_py *-- PatchDataset

    class losses_py {
        <<module>>
        +compute_pos_weight(df) float
        +build_loss(pos_weight) BCEWithLogitsLoss
        +is_degenerate(probs) bool
    }

    class early_stopping_py {
        <<module>>
    }
    class EarlyStopper {
        +patience, mode, best, counter, is_best
        +step(value) bool
    }
    early_stopping_py *-- EarlyStopper

    class metrics_py {
        <<module>>
        +confusion_matrix_counts(pred, target) dict
        +precision_from_counts(counts) float
        +recall_from_counts(counts) float
        +f1_from_counts(counts) float
        +sweep_confusion_counts(probs, target, thresholds) Tensor
        +average_precision_from_sweep(sweep) float
        +patch_detection_counts(pred, target) dict
        +detection_rate_from_counts(counts) float
    }

    class train_py {
        <<module>>
        +build_model(architecture) Module
        +set_seed(seed)
        +fit(model, train_df, val_df, ...) dict
        +evaluate(model, loader, loss_fn, device) dict
    }
    train_py ..> architectures_py : build_model()
    train_py ..> dataset_py : PatchDataset
    train_py ..> losses_py : build_loss(), compute_pos_weight()
    train_py ..> early_stopping_py : EarlyStopper

    class evaluate_py {
        <<module>>
        +load_checkpoint(architecture, path, device) Module
        +evaluate_full_metrics(model, loader, device, ...) dict
        +main()
    }
    evaluate_py ..> train_py : build_model()
    evaluate_py ..> dataset_py : PatchDataset
    evaluate_py ..> metrics_py : confusion-matrix / PR-AUC / patch functions
```

## 6. Data flow

### 6.1 Training (`train.py::fit`)

```mermaid
sequenceDiagram
    participant Script as train.py::fit()
    participant Loader as DataLoader
    participant DS as PatchDataset
    participant Model as E1 / E2 / E3
    participant Loss as BCEWithLogitsLoss
    participant Optimizer as Adam optimizer
    participant MLflow

    loop each epoch
        loop each training batch
            Script->>Loader: request next batch
            Loader->>DS: __getitem__(index) × batch_size
            DS-->>Loader: normalized input (4×128×128) + label (1×128×128)
            Loader-->>Script: batched input/output
            Script->>Model: forward(input)
            Model-->>Script: logits (1×128×128)
            Script->>Loss: loss(logits, label, pos_weight)
            Loss-->>Script: scalar loss
            Script->>Script: loss.backward()
            Script->>Optimizer: optimizer.step()
        end
        Script->>Model: forward(val batch), no_grad
        Model-->>Script: val logits
        Script->>Script: val_loss, val_f1 = evaluate()
        Script->>MLflow: log_metrics(val_loss, val_f1, seconds_per_epoch)
        alt val_loss improved
            Script->>Script: save checkpoint, reset patience
        else no improvement
            Script->>Script: increment patience counter
        end
    end
```

### 6.2 Inference / evaluation (`evaluate.py::main`)

```mermaid
sequenceDiagram
    participant Script as evaluate.py::main()
    participant Loader as DataLoader
    participant DS as PatchDataset
    participant Model as E1 / E2 / E3 (eval mode)
    participant Metrics as metrics.py
    participant MLflow

    Script->>Model: load_checkpoint(architecture, path)
    loop each batch (val or test split)
        Script->>Loader: request next batch
        Loader->>DS: __getitem__(index) × batch_size
        DS-->>Loader: normalized input + label
        Loader-->>Script: batched input/output
        Script->>Model: forward(input), no_grad
        Model-->>Script: logits
        Script->>Script: sigmoid + threshold (0.5) -> binary prediction
        Script->>Metrics: confusion_matrix_counts, sweep_confusion_counts,<br/>patch_detection_counts
        Metrics-->>Script: incremental TP/FP/FN/TN, PR-sweep counts, patch counts
        Note over Script,Metrics: accumulated across batches, never materializing the<br/>full split in memory, avoids the OOM bug that hit<br/>starcop_raw's 26,607-patch val split during Section 6/7
    end
    Script->>Script: precision, recall, F1, PR-AUC, confusion matrix,<br/>per-patch detection rate from the accumulated counts
    Script->>Script: assert patches_processed == len(split)<br/>(catches silent truncation)
    Script->>MLflow: log_metrics(...), log_dict(PR curve)
```

> [!IMPORTANT]
> No backward pass, no gradients here the key structural difference.
> This is also where the GPU-vs-CPU throughput numbers in
> [Section 9.5](#95-gpu-vs-cpu-inference-throughput) come from: timing
> wraps this whole loop (data loading + forward + metric accumulation),
> matching how `wall_clock_seconds` is already defined for training in
> [Section 6.1](#61-training-trainpyfit).

## 8. Training loop as a state machine (`EarlyStopper`)

```mermaid
stateDiagram-v2
    [*] --> Training
    Training --> Validating: epoch's training batches done
    Validating --> CheckBest: val_loss computed
    CheckBest --> NewBest: val_loss < best seen
    CheckBest --> NoImprovement: val_loss >= best seen
    NewBest --> SaveCheckpoint
    SaveCheckpoint --> ResetPatienceCounter
    ResetPatienceCounter --> CheckEpochLimit
    NoImprovement --> IncrementPatienceCounter
    IncrementPatienceCounter --> CheckPatience
    CheckPatience --> Stopped: counter >= patience
    CheckPatience --> CheckEpochLimit: counter < patience
    CheckEpochLimit --> Training: epochs remain
    CheckEpochLimit --> Stopped: max_epochs reached
    Stopped --> [*]: return best_epoch's metrics<br/>(not the last epoch's)
```

> [!NOTE]
> `patience=10` for every configuration/tier, the one hyperparameter held
> fixed everywhere, so no configuration gets an unfair extra-patience
> advantage. Only the epoch cap and learning rate are allowed to differ
> (E1: 1e-3 from scratch; E2/E3: 1e-4 fine-tuning a pretrained encoder)
> both differences are principled (fine-tuning warrants a smaller LR), not
> ad hoc tuning per configuration.

## 9. Results

### 9.1 Architecture summary

| Config | Full name | Params | Encoder | Params vs. E1 |
| --- | --- | ---: | --- | ---: |
| **E1** | TinyUNet | 487,361 | none (from scratch) | 1.0× (baseline) |
| **E2** | U-Net + MobileNetV2 | 6,629,233 | ImageNet-pretrained | 13.6× |
| **E3** | LinkNet + MobileNetV3-small-minimal | 856,635 | ImageNet-pretrained | 1.76× |

### 9.2 `mini` tier: val and test (graded headline comparison)

| Config | Split | Precision | Recall | F1 | PR-AUC | Detected patches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1 | val (49) | **0.4569** | **1.0000** | **0.6272** | **0.8599** | 9/9 |
| E1 | test (441) | **0.9328** | 0.6960 | **0.7972** | **0.8314** | 113/125 |
| E2 | val (49) | 0.1878 | 0.9954 | 0.3160 | 0.6170 | 9/9 |
| E2 | test (441) | 0.6811 | 0.8165 | 0.7427 | 0.8031 | 115/125 |
| E3 | val (49) | 0.0098 | 0.9969 | 0.0194 | 0.7196 | 9/9 |
| E3 | test (441) | 0.1925 | **0.9296** | 0.3190 | 0.6766 | **125/125** |

*Bold = best in its column, compared within the same split (val vs. test
compared separately different scale, not a fair cross-comparison). Val's
"Detected patches" is a 3-way tie (9/9 for all three) not bolded, since
nothing wins there. E1 sweeps every val column; on test, E1 still leads
Precision/F1/PR-AUC but E3 takes Recall and detects every positive patch.*

### 9.3 Cross-tier: `mini`-trained checkpoints scored on `starcop_raw`'s real test split (16,758 patches)

| Config | Precision | Recall | F1 | PR-AUC | Detected patches |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | **0.7247** | 0.3816 | **0.5000** | **0.4835** | 966/1,706 |
| E2 | 0.5071 | 0.4664 | 0.4859 | 0.3997 | 936/1,706 |
| E3 | 0.0369 | **0.8253** | 0.0706 | 0.2562 | **1,706/1,706** |

*Bold = best per column across all three rows (same split for all three,
so directly comparable).*

E1 > E2 > E3 by F1: **same ordering as `mini`'s own test split**
([Section 9.2](#92-mini-tier-val-and-test-graded-headline-comparison)),
despite training on a completely different, much smaller dataset.

### 9.4 `raw` tier: R2/R3, on `starcop_raw`'s real test split (16,758 patches)

| Config | Tier | Precision | Recall | F1 | PR-AUC | Detected patches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1 | R2 (6,076 train patches) | 0.0823 | 0.9770 | 0.1518 | 0.2610 | 1,668/1,706 |
| E2 | R2 | 0.0614 | **0.9847** | 0.1156 | 0.1783 | 1,636/1,706 |
| E3 | R2 | 0.0895 | 0.9594 | 0.1637 | 0.2968 | **1,686/1,706** |
| E2 | R3 (141,218 train patches) | **0.1524** | 0.9739 | **0.2636** | **0.4445** | 1,671/1,706 |
| E3 | R3 | 0.1302 | 0.9826 | 0.2300 | 0.4389 | 1,684/1,706 |

*Bold = best per column across all five rows (same split throughout, R2
and R3 directly comparable).* E2-R3 takes Precision/F1/PR-AUC, E2-R2 edges
out Recall, and E3-R2 edges out detection rate, individual columns are
close, but:

> [!IMPORTANT]
> **R3 clearly beats R2 as a group** on the identical test split (E2's F1
> more than doubles: 0.1156 → 0.2636) the cleanest "more real training
> data helps" evidence produced this session.

> [!WARNING]
> All five rows above are precision ≤0.15 with recall ≥0.96, read at
> face value, that looks like every `raw`-trained model badly
> over-predicts. It's largely a **threshold-calibration artifact, not a
> quality gap**: `pos_weight` (the loss's imbalance correction) scales
> hard with tier, mini≈87, R2≈270, R3≈314 ([Section 0](#0-glossary)), so
> the fixed 0.5 decision threshold stops being well-calibrated for models
> trained under a much larger `pos_weight`. PR-AUC (threshold-independent)
> tells a fairer story here than this table's fixed-threshold columns
> alone, see the PR curve discussion in
> [Section 9.5](#95-gpu-vs-cpu-inference-throughput) for the same effect,
> visualized.

### 9.5 GPU vs. CPU inference throughput

Same checkpoints, same `num_workers=4` data-loading config, only
`device` varied. Classification metrics matched to 3-4 decimals between
GPU/CPU runs; only speed differs.

| Config | Scale | GPU (patches/s) | CPU (patches/s) | Speedup |
| --- | --- | ---: | ---: | ---: |
| E1 | mini test (441) | 212.4 | 76.5 | 2.78× |
| E2 | mini test (441) | **217.3** | 47.4 | **4.59×** |
| E3 | mini test (441) | 215.9 | **78.1** | 2.76× |
| E1 | raw test (16,758) | **224.9** | 78.6 | 2.86× |
| E2 | raw test (16,758) | 223.8 | 48.5 | **4.62×** |
| E3 | raw test (16,758) | 224.1 | **83.9** | 2.67× |

*Bold = best per column, compared within the same scale (mini vs. raw
rows compared separately). "Best speedup" here just marks the largest
GPU/CPU ratio E2, the biggest model, benefits most because it's the
least I/O-bound of the three.*

**GPU throughput is nearly identical across all three architectures**
(~212-225 patches/s) despite a 13.6× spread in parameter count, the GPU
runs are **I/O-bound** (limited by data loading), not compute-bound. CPU
throughput, by contrast, clearly tracks model size: E2 is ~1.6-1.7×
slower than E1/E3 on CPU.

> [!CAUTION]
> Don't read the "Speedup" column above as a FLOPs/compute-efficiency
> comparison, it mostly measures "how much the GPU hides compute behind
> I/O" for each architecture, not raw arithmetic throughput. Because GPU
> time here is dominated by `num_workers=4` data loading rather than the
> model itself, E2's higher speedup (~4.6×) reflects it having the *most*
> compute to hide, not the GPU handling it more efficiently than E1/E3's
> compute. A `num_workers=0` "pure compute" rerun would isolate the real
> per-architecture GPU/CPU gap, uncontaminated by I/O.

### 9.6 Everything on one line per configuration

| Config | Params | Best test F1 (mini) | Best test PR-AUC (mini) | GPU throughput (raw) | CPU throughput (raw) |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1: TinyUNet | 487,361 | 0.7972 | 0.8314 | 224.9 p/s | 78.6 p/s |
| E2: U-Net + MobileNetV2 | 6,629,233 | 0.7427 | 0.8031 | 223.8 p/s | 48.5 p/s |
| E3: LinkNet + MobileNetV3-small | 856,635 | 0.3190 | 0.6766 | 224.1 p/s | 83.9 p/s |

## 10. Key findings, condensed

- **PR-AUC vs. fixed-threshold F1 tell different stories.** At threshold
  0.5, `raw`-trained models (R2/R3) show high recall / low precision
  (`pos_weight` scales hard with tier: mini≈87, R2≈270, R3≈314) PR-AUC
  (threshold-independent) is the fairer comparison across tiers
  ([Section 9.4](#94-raw-tier-r2r3-on-starcop_raws-real-test-split-16758-patches)).
- **R3 ≫ R2 on the same real test split**: more real training data
  measurably helps, and R2's own small/curated val split had understated
  how poorly it generalizes
  ([Section 9.4](#94-raw-tier-r2r3-on-starcop_raws-real-test-split-16758-patches)).
- **The mini-tier F1 ordering (E1 > E2 > E3) survives cross-tier
  evaluation** on a completely different, much larger dataset
  ([Section 9.3](#93-cross-tier-mini-trained-checkpoints-scored-on-starcop_raws-real-test-split-16758-patches)),
  real evidence the small-scale comparison isn't purely a data-size
  artifact.
- **E3 "detects" every positive patch at raw scale (1,706/1,706) only
  because it over-triggers almost everywhere** (FP=15.1M vs. TP=579K,
  cross-tier), "detects at all" and "detects precisely" are different
  claims
  ([Section 9.3](#93-cross-tier-mini-trained-checkpoints-scored-on-starcop_raws-real-test-split-16758-patches)
  / [9.4](#94-raw-tier-r2r3-on-starcop_raws-real-test-split-16758-patches)).
- **GPU inference is I/O-bound at this patch/model scale**: near-equal
  throughput across a 13.6× parameter-count spread on GPU, but clearly
  compute-bound on CPU
  ([Section 9.5](#95-gpu-vs-cpu-inference-throughput)). A pure-compute
  (`num_workers=0`) variant would isolate the real per-architecture GPU speedup further, if useful
  later.
