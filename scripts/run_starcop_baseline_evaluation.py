"""CLI for src/evaluation/run_baseline_eval.py -- evaluates a paper-baseline
checkpoint against the full STARCOP paper test set. All logic lives in
run_baseline_eval.py (unit tested); this file is argparse glue only, same
split as scripts/import_starcop_hf_baseline.py.

Run with:
    .venv/bin/python scripts/run_starcop_baseline_evaluation.py mag1c_rgb --limit 5
    .venv/bin/python scripts/run_starcop_baseline_evaluation.py mag1c_rgb
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))

import paper_eval_mlflow  # noqa: E402
import run_baseline_eval  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TEST_CSV = _REPO_ROOT / "data" / "starcop_raw" / "test.csv"
_DEFAULT_ROOT_FOLDER = _REPO_ROOT / "data" / "starcop_raw" / "STARCOP_test"
_REFERENCE_METRICS_PATH = _REPO_ROOT / "internal-docs" / "plans" / "paper_reference_metrics.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant",
        choices=["mag1c_only", "mag1c_rgb", "varon"],
        help="STARCOP checkpoint variant to evaluate",
    )
    parser.add_argument("--device", default="cpu", help="torch device to run on (e.g. cpu, mps)")
    parser.add_argument("--test-csv", default=str(_DEFAULT_TEST_CSV))
    parser.add_argument("--root-folder", default=str(_DEFAULT_ROOT_FOLDER))
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--limit", type=int, default=None, help="Diagnostic-only smoke test over N scenes (Phase 0)"
    )
    limit_group.add_argument(
        "--emit-docs-assets",
        default=None,
        metavar="DIR",
        help="Write curated sample PNGs + offline predictions to DIR "
        "(requires the full, unlimited run)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = run_baseline_eval.evaluate_variant(
        args.variant,
        device=torch.device(args.device),
        test_csv_path=args.test_csv,
        root_folder=args.root_folder,
        limit=args.limit,
        emit_docs_assets_dir=Path(args.emit_docs_assets) if args.emit_docs_assets else None,
    )

    # Bulky, not meant for a terminal -- kept in `result` for log_paper_eval_run,
    # stripped from the printed summary.
    _bulky_keys = ("joined_scene_results", "run_validation_metrics")
    summary = {k: v for k, v in result.items() if k not in _bulky_keys}
    print(json.dumps(summary, indent=2, default=str))

    # Automatic on every full run, per Phase 1's own established gating --
    # --limit is the only thing that skips it, so a partial run can never
    # land in the permanent starcop-paper-eval record.
    if args.limit is None and result.get("status") == "ok":
        run_id = paper_eval_mlflow.log_paper_eval_run(
            args.variant,
            result,
            repo_root=_REPO_ROOT,
            reference_metrics_path=_REFERENCE_METRICS_PATH,
            emit_docs_assets_dir=Path(args.emit_docs_assets) if args.emit_docs_assets else None,
        )
        print(f"Logged to starcop-paper-eval as run {run_id}")


if __name__ == "__main__":
    main()
