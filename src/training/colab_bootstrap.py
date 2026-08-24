"""Pure helpers for notebooks/train_colab.ipynb -- see mlops-methane-
detection-plan.md TASK-3.3c. Importable without any Colab-specific package
(no `google.colab`) so it's testable outside Colab; the notebook itself
supplies the actual secret values and subprocess execution.

`dvc_service_account_setup_commands` returns argv lists rather than
hand-built `.dvc/config.local` INI text deliberately: TASK-3.3c's step 1b
spike proved `dvc remote modify --local ...` works end-to-end on a real
Colab session (no OAuth prompt, real pull succeeded), so this mirrors those
exact proven commands instead of a parallel, untested file-writing path
that would only need to agree with DVC's own config parser by coincidence.
"""

from typing import Callable, List, Mapping, Optional

import launch_profiles

# The DVC_GDRIVE_SERVICE_ACCOUNT_JSON secret holds the same service-account
# key already created for the mac-mps Prefect worker (D-01) -- reused here,
# not a second key. See internal-docs/setup/environment-notes.md's "DVC pull
# via the D-01 service account" section for the confirmed-working recipe
# these commands come from.
_DVC_GDRIVE_SERVICE_ACCOUNT_SECRET = "DVC_GDRIVE_SERVICE_ACCOUNT_JSON"


def dvc_service_account_setup_commands(service_account_json_path: str) -> List[List[str]]:
    """Returns the two `dvc remote modify --local` argv lists that point the
    gdrive remote at a service-account key instead of interactive OAuth.
    Raises ValueError if the path is empty rather than building a broken
    command silently.
    """
    if not service_account_json_path:
        raise ValueError("service_account_json_path must not be empty.")
    return [
        ["dvc", "remote", "modify", "--local", "gdrive", "gdrive_use_service_account", "true"],
        [
            "dvc",
            "remote",
            "modify",
            "--local",
            "gdrive",
            "gdrive_service_account_json_file_path",
            service_account_json_path,
        ],
    ]


def required_colab_secrets() -> List[str]:
    """Every Colab Secret name the notebook needs, composed on top of
    launch_profiles.required_env_vars("colab") rather than re-declaring the
    same MLflow/AWS names a second time.
    """
    return launch_profiles.required_env_vars("colab") + [_DVC_GDRIVE_SERVICE_ACCOUNT_SECRET]


def read_secret(
    name: str,
    userdata_get: Optional[Callable[[str], Optional[str]]],
    environ: Mapping[str, str],
) -> Optional[str]:
    """Reads a credential by name, preferring Colab Secrets when reachable
    and falling back to a plain environment variable otherwise -- lets the
    same notebook run against the Colab web UI (google.colab.userdata is
    reachable) or a VS Code-attached Colab kernel (no browser frontend, so
    userdata.get() raises instead of returning a value) without two
    separate code paths per cell.

    `userdata_get` should be `google.colab.userdata.get` when importable, or
    None if the import itself failed. Any exception from calling it (not
    just it being unavailable at import time) falls back to `environ` too --
    a call-time failure is exactly the VS Code-without-a-frontend case this
    exists for, not just a missing import.
    """
    if userdata_get is not None:
        try:
            value = userdata_get(name)
        except Exception:
            value = None
        if value:
            return value
    return environ.get(name)


def resolve_wandb_mode(
    wandb_api_key: Optional[str], requested_mode: Optional[str]
) -> Optional[str]:
    """D-09's WANDB_MODE default, as a pure function -- see scripts/train_mac.sh
    and scripts/train_desktop.sh for the equivalent bash logic. An explicit
    `requested_mode` always wins; otherwise defaults to "disabled" when no
    API key is present (Colab has no cached wandb login, same as the other
    two machines' first unattended run), and leaves WANDB_MODE untouched
    when a real key is present.
    """
    if requested_mode:
        return requested_mode
    if not wandb_api_key:
        return "disabled"
    return None
