"""Tests for src/training/accelerator_check.py -- pure guard logic (Test
Size: Small, no mocking): pytorch-lightning versions before 1.7.0 (Environment
A originally pinned 1.6.4) silently resolve an unrecognized accelerator
string like "mps" to CPUAccelerator instead of raising -- a run would
otherwise complete FINISHED with real metrics while never touching the GPU
(see TASK-3.2 in mlops-methane-detection-plan.md). This module fails loudly
instead of trusting that what was requested is what Lightning resolved.
"""

import accelerator_check
import pytest


class TestAssertResolvedAccelerator:
    def test_passes_when_mps_requested_and_resolved_to_mps(self):
        accelerator_check.assert_resolved_accelerator("mps", "mps")  # should not raise

    def test_raises_when_mps_requested_but_resolved_to_cpu(self):
        with pytest.raises(RuntimeError, match="mps"):
            accelerator_check.assert_resolved_accelerator("mps", "cpu")

    @pytest.mark.parametrize("accelerator", ("cpu", "gpu"))
    def test_is_a_noop_when_non_mps_accelerator_requested_regardless_of_resolved_device(
        self, accelerator
    ):
        accelerator_check.assert_resolved_accelerator(accelerator, "cpu")  # should not raise
