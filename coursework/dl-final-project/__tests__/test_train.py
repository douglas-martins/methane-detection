import numpy as np
import pandas as pd
import pytest
import torch
from losses import build_loss
from train import augment_for, evaluate, fit


class TestAugmentFor:
    def test_e1_gets_no_augmentation(self):
        # Section 6: E1 is the deliberately plain baseline -- no
        # pretraining, no augmentation, minimal regularization. Direct
        # unit coverage after this was silently wrong once already (main()
        # never passed `augment=` to fit(), so E1 trained with kornia
        # augmentation on by default until this was caught during review).
        assert augment_for("E1") is False

    def test_e2_and_e3_get_augmentation(self):
        assert augment_for("E2") is True
        assert augment_for("E3") is True


class _FixedLogitModel(torch.nn.Module):
    """Returns pre-baked logits regardless of input -- lets tests control predictions exactly."""

    def __init__(self, logits_by_call: list[torch.Tensor]):
        super().__init__()
        self._logits_by_call = iter(logits_by_call)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return next(self._logits_by_call)


class TestEvaluate:
    def test_accumulates_across_batches_without_holding_the_full_set_in_memory(self):
        # Batch 1: 1 true positive correctly predicted, 1 false positive.
        # Batch 2: 1 true positive missed (false negative).
        # Correct pooled F1 requires summing tp/fp/fn across *both* batches,
        # not computing per-batch and averaging (which this asserts against).
        batches = [
            {
                "input": torch.zeros(2, 4, 2, 2),
                "output": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            },
            {
                "input": torch.zeros(2, 4, 2, 2),
                "output": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            },
        ]
        # logits: batch1 predicts positive at [0,0,0,0] (matches) and [0,0,0,1] (false positive)
        logits_batch1 = torch.tensor([[[[5.0, 5.0], [-5.0, -5.0]]], [[[-5.0, -5.0], [-5.0, -5.0]]]])
        # batch2 predicts nothing positive -- misses the true positive at [0,0,0,0]
        logits_batch2 = torch.full((2, 1, 2, 2), -5.0)
        model = _FixedLogitModel([logits_batch1, logits_batch2])
        loss_fn = build_loss(pos_weight=1.0)

        result = evaluate(model, batches, loss_fn, device="cpu")

        # pooled: tp=1, fp=1, fn=1 -> f1=0.5 (see test_metrics.py's hand-computed case)
        assert result["f1"] == pytest.approx(0.5)
        assert result["degenerate"] is False

    def test_flags_all_zero_predictions_as_degenerate_across_batches(self):
        batches = [
            {"input": torch.zeros(2, 4, 2, 2), "output": torch.zeros(2, 1, 2, 2)},
            {"input": torch.zeros(2, 4, 2, 2), "output": torch.zeros(2, 1, 2, 2)},
        ]
        model = _FixedLogitModel([torch.full((2, 1, 2, 2), -20.0), torch.full((2, 1, 2, 2), -20.0)])
        loss_fn = build_loss(pos_weight=1.0)

        result = evaluate(model, batches, loss_fn, device="cpu")

        assert result["degenerate"] is True


def _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=0):
    bands = {
        "mag1c": np.random.default_rng(0).uniform(0, 500, (size, size)).astype("float32"),
        "TOA_AVIRIS_640nm": np.random.default_rng(1).uniform(0, 40, (size, size)).astype("float32"),
        "TOA_AVIRIS_550nm": np.random.default_rng(2).uniform(0, 40, (size, size)).astype("float32"),
        "TOA_AVIRIS_460nm": np.random.default_rng(3).uniform(0, 40, (size, size)).astype("float32"),
        "labelbinary": np.zeros((size, size), dtype="float32"),
    }
    bands["labelbinary"].flat[:positive_pixels] = 1.0
    for band_name, array in bands.items():
        tiny_geotiff_factory(folder / f"{band_name}.tif", array)


def _patches_df(folder, n_rows, size=16):
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "id": f"patch_{i}",
                "name": "scene0",
                "folder": str(folder),
                "window_col_off": 0,
                "window_row_off": 0,
                "window_width": size,
                "window_height": size,
                "has_plume": i % 3 == 0,
                "frac_positives": 0.05 if i % 3 == 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


class TestFit:
    def test_runs_for_the_requested_epochs_and_saves_a_checkpoint(
        self, tmp_path, tiny_geotiff_factory
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=6, size=16)
        val_df = _patches_df(folder, n_rows=4, size=16)
        checkpoint_path = tmp_path / "checkpoints" / "e1.pt"

        model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, 1),
        )
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=2,
            patience=10,
            checkpoint_path=checkpoint_path,
            log_to_mlflow=False,
        )

        assert result["epochs_run"] == 2
        assert checkpoint_path.exists()
        assert "loss" in result
        assert "f1" in result
        assert "degenerate" in result

    def test_returns_the_best_epochs_metrics_not_the_last_epochs(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        # A model can wander after its best epoch before early stopping's
        # patience is exhausted (real behavior seen on a real R2 run: best
        # at epoch 8, stopped at epoch 18 with a worse loss). The returned
        # metrics -- and the checkpoint -- must reflect the best epoch,
        # not whatever the last one happened to be, since reporting the
        # last epoch's worse numbers would misrepresent the saved model.
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        canned_losses = [0.5, 0.1, 0.4, 0.6]  # best is epoch 2 (index 1)
        calls = iter(canned_losses)

        def _fake_evaluate(model, loader, loss_fn, device):
            return {
                "loss": next(calls),
                "f1": 0.0,
                "degenerate": False,
                "positive_fraction": 0.0,
            }

        monkeypatch.setattr("train.evaluate", _fake_evaluate)

        model = torch.nn.Conv2d(4, 1, 1)
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=4,
            patience=100,
            log_to_mlflow=False,
        )

        assert result["loss"] == 0.1
        assert result["best_epoch"] == 2

    def test_max_steps_stops_training_early_regardless_of_max_epochs(
        self, tmp_path, tiny_geotiff_factory
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=6, size=16)
        val_df = _patches_df(folder, n_rows=4, size=16)

        model = torch.nn.Sequential(torch.nn.Conv2d(4, 1, 1))
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=50,
            patience=1000,
            max_steps=3,
            log_to_mlflow=False,
        )

        assert result["steps_run"] == 3
        assert result["epochs_run"] < 50
