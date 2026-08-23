"""Guards against pytorch-lightning silently resolving an unrecognized
accelerator string to CPU instead of raising -- see TASK-3.2 in
mlops-methane-detection-plan.md. Versions before 1.7.0 (Environment A
originally pinned pytorch-lightning==1.6.4) have no MPSAccelerator at all,
so `Trainer(accelerator="mps")` resolves to CPUAccelerator without error.

The "gpu" case (TASK-3.1) is the same defensive check extended to CUDA: a
resolved device type of "cuda" is what Lightning's own GPUAccelerator
requires (see pytorch_lightning/accelerators/gpu.py's `root_device.type !=
"cuda"` check), so "gpu" maps to "cuda" rather than to itself.
"""

_EXPECTED_RESOLVED_DEVICE_TYPE = {
    "mps": "mps",
    "gpu": "cuda",
}


def assert_resolved_accelerator(requested_accelerator: str, resolved_device_type: str) -> None:
    """Raises RuntimeError if "mps" or "gpu" was requested but Lightning
    resolved to a different device type. A no-op for any other requested
    accelerator.
    """
    expected = _EXPECTED_RESOLVED_DEVICE_TYPE.get(requested_accelerator)
    if expected is not None and resolved_device_type != expected:
        raise RuntimeError(
            f"training.accelerator={requested_accelerator} was requested but "
            f"resolved to device type {resolved_device_type!r} (expected "
            f"{expected!r}) -- check that pytorch-lightning>=1.7.0 is "
            "installed in vendor/starcop/.venv (1.6.4 has no MPSAccelerator "
            "and silently falls back to CPU instead of erroring) and, for "
            "gpu, that CUDA is actually available/functional on this machine."
        )
