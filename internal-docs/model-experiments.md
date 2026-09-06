# Model Experiments & On-Board Conversion — Living Tracker

> New file, per `docs-reorganization-plan.md` Section 11.2 — a living research journal
> for model-iteration and on-board-conversion work, explicitly **internal** (not
> MkDocs): a running record of what's been tried and ruled out, so a future session
> (human or AI agent) doesn't re-try a dead end. Related: [decisions.md](decisions.md)
> D-07 (Linux-only on-board tooling), public
> [Hardware Benchmark](../docs/results/hardware-benchmark.md) page (a collaborator's
> completed Vitis AI + ZCU104 feasibility test on the same model family).

## Toolchain investigations

### hls4ml ONNX/PyTorch conversion — first probe, 2026-09-06

**Why**: `docs-reorganization-plan.md` Section 11.4 named hls4ml + Vitis AI as the
working on-board direction, but flagged as unresolved that "STARCOP's U-Net-style
encoder-decoder with skip connections needs checking against hls4ml's actual
supported layer set — not assumed to work out of the box." This project's completed
hardware deployment work (the collaborator's ZCU104 benchmark, see the public
Hardware Benchmark page) only exercised Vitis AI — hls4ml itself had never actually
been tried against this architecture. This is that first try.

**Model under test**: this repo's own most recent local checkpoint,
`experiments/starcop_run/2026-08-23_02-05/final_checkpoint_model.ckpt` —
`unet_semseg`, MobileNetV2 encoder, 4-channel input (`mag1c` + 3 TOA bands), same
architecture family as HyperSTARCOP. `hls4ml==1.3.0`, `torch==2.12.1+cu130`.

**Packages installed for this probe** (`uv pip install`, research env
`.venv` — transient, **not** added to `pyproject.toml`/lock, so they won't survive
a `uv sync`): `onnx`, `onnxscript`, `onnxruntime`, `hls4ml` (`==1.3.0`, per above),
`qonnx`. Exact resolved versions for the other four weren't captured at probe
time (no `pip freeze`/lockfile from that install survives) — `uv pip install
onnx onnxscript onnxruntime qonnx` pulls whatever's current, which may not
match this run; if a retry behaves differently, that drift is the first thing
to suspect.

**Attempt 1 — legacy `torch.onnx.export` (TorchScript-based, `dynamo=False`) → hls4ml ONNX import**:
Export itself succeeded (15 op types, `onnx.checker` passed) — the architecture
survives ONNX export cleanly, consistent with the collaborator's ExecuTorch/XNNPACK
result on the same architecture family. hls4ml's ONNX importer then failed on both
`config_from_onnx_model` and `convert_from_onnx_model`:
```text
RuntimeError: Could not find the shape for input onnx::Sub_916
```
Reproduced with the normalizer stripped out too (network-only export), same failure
class against a different constant tensor (`onnx::Conv_715`) — not specific to the
normalizer. `onnx.shape_inference.infer_shapes()` as a pre-pass did not fix it.
Looks like a shape-metadata gap in how the legacy exporter represents certain
constants/initializers, that hls4ml 1.3.0's ONNX frontend doesn't tolerate.

**Attempt 2 — newer `torch.export`-based ONNX exporter (`dynamo=True`) → hls4ml ONNX import**:
Produces a much cleaner graph (6 op types: `Add`, `Clip`, `Concat`, `Conv`, `Relu`,
`Resize` — no `Shape`/`Slice`/`Sub`/`Cast` scaffolding). hls4ml's importer got
further, with a specific, expected complaint instead of an opaque one:
```text
RuntimeError: Please convert the model to channels-last format with qonnx-to-channels-last
```
(hls4ml wants NHWC internally — same convention the collaborator's DPU flow already
uses.) Ran `qonnx-cleanup` then `qonnx-to-channels-last --make-input-channels-last`;
that hit a second, separate bug in `qonnx`'s own channels-last transform:
```text
Exception: Required attribute kernel_shape unspecified in a Conv node
```
The dynamo exporter omits `kernel_shape` as a `Conv` node attribute (valid ONNX —
it's inferable from the weight tensor's shape), but `qonnx`'s shape-inference for
its channels-last `Conv` custom-op assumes the attribute is always explicitly
present. Not pursued further (would mean patching `qonnx` or pre-processing the
ONNX graph to inject the attribute).

**Attempt 3 — hls4ml's native PyTorch converter (`convert_from_pytorch_model`, skips ONNX entirely)**:
Failed immediately and specifically, at `config_from_pytorch_model`:
```text
Exception: Unsupported layer ReLU6
```
This is the real finding, not a tooling/format quirk. The encoder is MobileNetV2,
and `ReLU6` (ReLU clipped at 6) is MobileNetV2's signature activation, used
throughout its inverted-residual blocks — hls4ml 1.3.0's PyTorch frontend has no
layer handler for it. The ONNX paths (attempts 1–2) never hit this because ONNX
export already lowers `ReLU6` to `Clip`, which hls4ml's ONNX frontend does accept
as an op type — so the ONNX route may still be the more viable path once its
tooling issues (attempt 2) are worked through, since it sidesteps the PyTorch
frontend's `ReLU6` gap entirely.

**Conclusion so far**: hls4ml conversion of this architecture is **not proven
infeasible**, but is genuinely blocked today, for three distinct reasons depending
on the path taken (a legacy-exporter shape bug, a dynamo-exporter/qonnx attribute
bug, and a real missing-layer gap in hls4ml's native PyTorch frontend). Vitis AI
remains the only toolchain proven end-to-end on this architecture (the
collaborator's ZCU104 work).

**Not yet tried / options for a follow-up session**:
- Patch/pre-process the dynamo-exported ONNX graph to inject explicit `kernel_shape`
  attributes on `Conv` nodes, then retry `qonnx-to-channels-last` (attempt 2's
  blocker) — most promising, narrowest fix, doesn't touch the model itself.
- Replace `ReLU6` with plain `ReLU` in the encoder for the on-board candidate
  specifically, then retry hls4ml's native PyTorch frontend (attempt 3) — changes
  model behavior, would need re-validation/fine-tuning, not a free substitution.
- Check hls4ml's GitHub issues/newer releases for existing `ReLU6` support or
  fixes to either bug above, before spending more time working around them locally.
- Revisit whether hls4ml is worth pursuing at all for *this* architecture, given
  Vitis AI already has a fully proven path — i.e. treat "hls4ml + Vitis AI" (the
  plan's original framing) as "Vitis AI, with hls4ml evaluated and currently
  blocked" rather than two co-equal toolchains, until one of the above unblocks it.
