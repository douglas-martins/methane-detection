# Results

This section answers two separate questions about the model:

1. **Does it reproduce the STARCOP paper's own reported accuracy?** Evaluated against
   the paper's own released checkpoints and full held-out test set — a reproduction
   check, not a new model this project trained.
2. **Is it actually feasible to run on real on-board/embedded hardware?** This
   project's target isn't a cloud GPU — see a real feasibility test on a Xilinx
   ZCU104.

- **[STARCOP Reproduction Results](starcop-benchmark.md)** — paper-accuracy
  reproduction on the full 342-scene held-out test set, plus a live-API check
  confirming the deployed model matches the offline benchmark.
- **[Hardware Benchmark: ZCU104](hardware-benchmark.md)** — INT8 quantization
  quality check and inference throughput/latency across a notebook CPU, the
  ZCU104's ARM CPU, and the ZCU104's DPU.
