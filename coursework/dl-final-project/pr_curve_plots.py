"""PR-curve comparison plots for the DL course final project (plan Section 8's final
Activities checklist item).

Reads each run's `test_precision_recall_curve.json` artifact -- logged by
`evaluate.py`'s `main()` for every evaluation this section already ran (mini,
cross-tier, R2, R3) -- and renders two comparison figures, matching `eda.py`'s
own established split: `curve_points_from_artifact`/`sort_points_by_recall`
(reused from `metrics.py`) are the tested logic; `plot_pr_curve_comparison`
itself (matplotlib rendering) is thin glue, exercised by running `main()` for
real and inspecting the saved figures, the same way `eda.py`'s own
`plot_patch_level_balance`/`plot_example_patches` are (neither has a unit
test either -- see `__tests__/test_eda.py`).

Both required plots are scored on the *same* evaluation split so the
comparison is apples-to-apples, not an artifact of different test sets:

- **Within a tier** (E1 vs. E2 vs. E3): all three trained *and* evaluated on
  `mini`'s own test split (441 patches) -- the headline, graded comparison.
- **Across tiers** (one configuration, E2): `mini`-trained (scored via the
  cross-tier eval), `r2`-trained, and `raw-full`-trained, all three scored on
  `starcop_raw`'s own 16,758-patch test split -- the only split all three
  configurations share, so the curves differ because of training data, not
  evaluation data.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script writes figures to disk, never shows a window
import matplotlib.pyplot as plt
import mlflow
from metrics import sort_points_by_recall

_COURSEWORK_ROOT = Path(__file__).resolve().parent
_MLFLOW_TRACKING_URI = f"sqlite:///{_COURSEWORK_ROOT / 'mlflow.db'}"
_MLFLOW_EXPERIMENT = "dl-final-project"


def curve_points_from_artifact(artifact: dict) -> list[tuple[float, float]]:
    """Extract and sort `(recall, precision)` points from a loaded PR-curve JSON artifact.

    `evaluate.py` logs `{"precision_recall_curve": [[recall, precision], ...]}`
    in threshold order -- JSON round-trips each pair as a list, and plotting
    wants tuples sorted ascending by recall (matching `average_precision_from_sweep`'s
    own convention), so both conversions happen here in one place.
    """
    points = [
        (float(recall), float(precision))
        for recall, precision in artifact["precision_recall_curve"]
    ]
    return sort_points_by_recall(points)


def load_precision_recall_curve(run_name: str, split: str = "test") -> list[tuple[float, float]]:
    """Find the `mlflow.db` run named `run_name` and return its `split`'s PR curve.

    Raises `ValueError` if no run with that exact name exists -- silently
    plotting an empty curve would be a worse failure than crashing here.
    """
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(_MLFLOW_EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])
    matches = [run for run in runs if run.data.tags.get("mlflow.runName") == run_name]
    if not matches:
        raise ValueError(f"no run named {run_name!r} found in experiment {_MLFLOW_EXPERIMENT!r}")

    artifact = mlflow.artifacts.load_dict(
        f"runs:/{matches[0].info.run_id}/{split}_precision_recall_curve.json"
    )
    return curve_points_from_artifact(artifact)


def plot_pr_curve_comparison(
    curves: dict[str, list[tuple[float, float]]], title: str, out_path: Path
) -> None:
    """Save a precision-recall line plot with one line per `curves` entry (label -> points)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, points in curves.items():
        recalls, precisions = zip(*points, strict=True)
        ax.plot(recalls, precisions, marker=".", markersize=3, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate this checklist item's two required figures from already-recorded eval runs."""
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    figures_dir = _COURSEWORK_ROOT / "figures"

    within_tier_curves = {
        architecture: load_precision_recall_curve(f"{architecture}-mini-eval")
        for architecture in ["E1", "E2", "E3"]
    }
    plot_pr_curve_comparison(
        within_tier_curves,
        "E1 vs. E2 vs. E3 -- mini tier, test split",
        figures_dir / "pr_curve_within_tier_mini.png",
    )

    across_tier_curves = {
        "mini (cross-tier)": load_precision_recall_curve("E2-mini-on-raw-full-eval"),
        "r2": load_precision_recall_curve("E2-r2-eval"),
        "raw-full": load_precision_recall_curve("E2-raw-full-eval"),
    }
    plot_pr_curve_comparison(
        across_tier_curves,
        "E2 across tiers -- all scored on starcop_raw's test split",
        figures_dir / "pr_curve_across_tiers_e2.png",
    )

    print(f"wrote {figures_dir / 'pr_curve_within_tier_mini.png'}")
    print(f"wrote {figures_dir / 'pr_curve_across_tiers_e2.png'}")


if __name__ == "__main__":
    main()
