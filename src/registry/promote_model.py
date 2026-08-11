"""CLI entrypoint for the MLflow model registry promotion workflow (TASK-2.3).

Composes promotion_criteria.py (pure decision logic) and mlflow_registry.py
(MLflow SDK glue) into the experiment -> staging -> production ladder: a run
is only considered for Production once it has cleared Staging.

Run with (Environment B):
    .venv/bin/python src/registry/promote_model.py --run-id <run_id>

Falls back to the latest run in the default MLflow experiment if --run-id is
omitted -- convenient for manual use, but the Phase 4 CI call should always
pass an explicit --run-id from the training job that produced it (see
mlops-methane-detection-plan.md TASK-2.3 decision on "latest run" scoping).
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow_registry  # noqa: E402
import promotion_criteria  # noqa: E402

DEFAULT_MODEL_NAME = "methane-cnn-starcop"
DEFAULT_EXPERIMENT_ID = "0"


@dataclass
class PromotionOutcome:
    """Result of decide_and_promote: which stage (if any) a run was promoted to."""

    run_id: str
    stage: Optional[str]
    model_version: Optional[ModelVersion]
    decision: promotion_criteria.PromotionDecision


def decide_and_promote(
    client: MlflowClient,
    run_id: Optional[str],
    model_name: str = DEFAULT_MODEL_NAME,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
) -> PromotionOutcome:
    """Resolves `run_id` (or the latest run in `experiment_id`), evaluates it
    against Staging criteria and then Production criteria, and registers +
    promotes the model to the highest stage it qualifies for.
    """
    resolved_run_id = mlflow_registry.resolve_run_id(client, run_id, experiment_id=experiment_id)
    metrics = mlflow_registry.fetch_run_metrics(client, resolved_run_id)
    val_loss_history = mlflow_registry.fetch_metric_history(client, resolved_run_id, "val_loss")

    staging_decision = promotion_criteria.evaluate_staging(metrics, val_loss_history)
    if not staging_decision.promote:
        return PromotionOutcome(
            run_id=resolved_run_id, stage=None, model_version=None, decision=staging_decision
        )

    production_decision = promotion_criteria.evaluate_production(metrics)
    if production_decision.promote:
        model_version = mlflow_registry.register_and_promote(
            client, resolved_run_id, model_name, stage="Production"
        )
        return PromotionOutcome(
            run_id=resolved_run_id,
            stage="Production",
            model_version=model_version,
            decision=production_decision,
        )

    model_version = mlflow_registry.register_and_promote(
        client, resolved_run_id, model_name, stage="Staging"
    )
    return PromotionOutcome(
        run_id=resolved_run_id,
        stage="Staging",
        model_version=model_version,
        decision=staging_decision,
    )


def main() -> None:
    """CLI entrypoint: parses args, runs decide_and_promote, and reports the outcome."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=None,
        help="MLflow run to evaluate. Defaults to the latest run in --experiment-id.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="Defaults to the MLFLOW_TRACKING_URI environment variable.",
    )
    args = parser.parse_args()

    if not args.tracking_uri:
        parser.error("--tracking-uri or MLFLOW_TRACKING_URI must be set")

    client = MlflowClient(tracking_uri=args.tracking_uri)
    outcome = decide_and_promote(
        client, run_id=args.run_id, model_name=args.model_name, experiment_id=args.experiment_id
    )

    if outcome.stage is None:
        log.info("Run %s NOT promoted. Reasons:", outcome.run_id)
        for reason in outcome.decision.reasons:
            log.info("  - %s", reason)
        raise SystemExit(1)

    log.info(
        "Run %s promoted to %s as %s v%s",
        outcome.run_id,
        outcome.stage,
        outcome.model_version.name,
        outcome.model_version.version,
    )


if __name__ == "__main__":
    main()
