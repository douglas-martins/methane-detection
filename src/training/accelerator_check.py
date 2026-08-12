"""Guards against pytorch-lightning silently resolving an unrecognized
accelerator string to CPU instead of raising -- see TASK-3.2 in
mlops-methane-detection-plan.md. Versions before 1.7.0 (Environment A
originally pinned pytorch-lightning==1.6.4) have no MPSAccelerator at all,
so `Trainer(accelerator="mps")` resolves to CPUAccelerator without error.
"""


def assert_resolved_accelerator(requested_accelerator: str, resolved_device_type: str) -> None:
    """Raises RuntimeError if "mps" was requested but Lightning resolved to
    a different device type. A no-op for any other requested accelerator.
    """
    if requested_accelerator == "mps" and resolved_device_type != "mps":
        raise RuntimeError(
            f"training.accelerator=mps was requested but resolved to device "
            f"type {resolved_device_type!r} -- check that pytorch-lightning"
            ">=1.7.0 is installed in vendor/starcop/.venv (1.6.4 has no "
            "MPSAccelerator and silently falls back to CPU instead of erroring)."
        )
