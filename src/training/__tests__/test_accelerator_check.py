"""Tests for src/training/accelerator_check.py -- pure guard logic (Test
Size: Small, no mocking): pytorch-lightning versions before 1.7.0 (Environment
A originally pinned 1.6.4) silently resolve an unrecognized accelerator
string like "mps" to CPUAccelerator instead of raising -- a run would
otherwise complete FINISHED with real metrics while never touching the GPU
(see TASK-3.2 in mlops-methane-detection-plan.md). This module fails loudly
instead of trusting that what was requested is what Lightning resolved.

The "gpu" case (TASK-3.1) is the same defensive shape extended to CUDA:
Lightning's GPUAccelerator resolves to a device with `.type == "cuda"`, not
"gpu" -- see pytorch_lightning/accelerators/gpu.py's own
`root_device.type != "cuda"` check.
"""

import accelerator_check
import pytest


class TestAssertResolvedAccelerator:
    def test_passes_when_mps_requested_and_resolved_to_mps(self):
        accelerator_check.assert_resolved_accelerator("mps", "mps")  # should not raise

    def test_raises_when_mps_requested_but_resolved_to_cpu(self):
        with pytest.raises(RuntimeError, match="mps"):
            accelerator_check.assert_resolved_accelerator("mps", "cpu")

    def test_passes_when_gpu_requested_and_resolved_to_cuda(self):
        accelerator_check.assert_resolved_accelerator("gpu", "cuda")  # should not raise

    def test_raises_when_gpu_requested_but_resolved_to_cpu(self):
        with pytest.raises(RuntimeError, match="gpu"):
            accelerator_check.assert_resolved_accelerator("gpu", "cpu")

    def test_is_a_noop_when_cpu_accelerator_requested_regardless_of_resolved_device(self):
        accelerator_check.assert_resolved_accelerator("cpu", "cpu")  # should not raise
