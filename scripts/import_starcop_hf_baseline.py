"""CLI for src/registry/hf_baseline_import.py -- imports a pretrained STARCOP
checkpoint from HuggingFace (isp-uv-es/starcop) into MLflow as a comparison
baseline. All logic lives in hf_baseline_import.py (unit tested); this file
is argparse glue only, same split as src/training/train.py vs mlflow_utils.py.

Run with (same credentials as scripts/train_mac.sh):
    set -a; source .env.mlflow; set +a
    .venv/bin/python scripts/import_starcop_hf_baseline.py mag1c_only
    .venv/bin/python scripts/import_starcop_hf_baseline.py mag1c_rgb
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "registry"))

import hf_baseline_import  # noqa: E402


def main() -> None:
    """Parses CLI args and imports the requested checkpoint variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant", choices=["mag1c_only", "mag1c_rgb"], help="STARCOP checkpoint variant to import"
    )
    parser.add_argument(
        "--stage",
        default="Staging",
        choices=["Staging", "Production", "None"],
        help="Registry stage to promote to after logging the run (default: Staging). "
        "Pass 'None' to log the run without creating a registered model version.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    hf_baseline_import.import_variant(
        args.variant, stage=None if args.stage == "None" else args.stage
    )


if __name__ == "__main__":
    main()
