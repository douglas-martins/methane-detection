"""Tests for src/training/plot_confusion_matrix.py (Test Size: Small, pure --
asserts on actual rendered content, not just "doesn't crash")."""

import matplotlib
import torch

matplotlib.use("agg")

import plot_confusion_matrix as pcm


class TestPlotConfusionMatrix:
    def test_returns_figure_with_expected_cell_values(self):
        cm = torch.tensor([[90, 10], [5, 95]])

        fig = pcm.plot_confusion_matrix(cm)

        ax = fig.axes[0]
        rendered_values = sorted(int(t.get_text()) for t in ax.texts)
        assert rendered_values == [5, 10, 90, 95]

    def test_labels_axes_as_background_and_methane(self):
        cm = torch.tensor([[90, 10], [5, 95]])

        fig = pcm.plot_confusion_matrix(cm)

        ax = fig.axes[0]
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert x_labels == ["background", "methane"]
        assert y_labels == ["background", "methane"]
