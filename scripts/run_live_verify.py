"""CLI for src/evaluation/live_verify.py -- verifies a live BentoML
`/predict` endpoint agrees with Phase 1/3's offline predictions for a
variant's curated scenes. All logic lives in live_verify.py (unit tested);
this file is argparse glue only, same split as
scripts/run_starcop_baseline_evaluation.py.

Only the two Hyper variants are servable live today (see Phase 5's
"Explicit scope limit" note) -- point --base-url at a `bentoml serve`
process whose MODEL_NAME/MODEL_STAGE env vars match the variant being
verified, e.g.:

    MODEL_NAME=starcop-baseline-mag1c-only MODEL_STAGE=Staging \\
        .venv/bin/bentoml serve src.serving.service:MethaneDetectionService
    .venv/bin/python scripts/run_live_verify.py mag1c_only

Exits 0 if every curated scene passed, 1 otherwise (or on any error) --
the exit code is what Phase 4's flow reads to record the live-check status.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))

import live_verify  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TEST_CSV = _REPO_ROOT / "data" / "starcop_raw" / "test.csv"
_DEFAULT_ROOT_FOLDER = _REPO_ROOT / "data" / "starcop_raw" / "STARCOP_test"
_DEFAULT_BASE_URL = "http://localhost:3000"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "variant",
        choices=list(live_verify.SERVABLE_VARIANTS),
        help="STARCOP checkpoint variant to verify (Hyper variants only -- see module docstring)",
    )
    parser.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help="Base URL of the running bentoml serve process",
    )
    parser.add_argument(
        "--tracking-uri", default=None, help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI)"
    )
    parser.add_argument("--test-csv", default=str(_DEFAULT_TEST_CSV))
    parser.add_argument("--root-folder", default=str(_DEFAULT_ROOT_FOLDER))
    parser.add_argument(
        "--atol",
        type=float,
        default=live_verify.DEFAULT_ATOL,
        help="Confidence comparison tolerance",
    )
    args = parser.parse_args()

    tracking_uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        parser.error("--tracking-uri not given and MLFLOW_TRACKING_URI is not set")

    result = live_verify.verify_variant(
        args.variant,
        base_url=args.base_url,
        tracking_uri=tracking_uri,
        test_csv_path=args.test_csv,
        root_folder=args.root_folder,
        atol=args.atol,
    )

    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
