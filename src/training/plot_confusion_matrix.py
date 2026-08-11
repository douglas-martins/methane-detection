"""Renders a binary confusion matrix tensor (STARCOP's [[TN,FP],[FN,TP]]
layout) as a matplotlib figure, for logging as an MLflow artifact.

Pure torch + matplotlib -- no `starcop` import needed, so this could run
under either environment; kept in src/training/ since it's consumed by the
Environment-A training entrypoint (see TASK-2.2 step 8).
"""

import matplotlib.pyplot as plt

CLASS_NAMES = ("background", "methane")


def plot_confusion_matrix(cm, class_names=CLASS_NAMES) -> plt.Figure:
    """Renders a 2x2 confusion matrix tensor/array as an annotated heatmap figure."""
    cm_np = cm.detach().cpu().numpy() if hasattr(cm, "detach") else cm

    fig, ax = plt.subplots()
    ax.imshow(cm_np)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            ax.text(j, i, str(int(cm_np[i, j])), ha="center", va="center")

    fig.tight_layout()
    return fig
