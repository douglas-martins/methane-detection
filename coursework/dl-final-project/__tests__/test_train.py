import numpy as np
import pandas as pd
import pytest
import segmentation_models_pytorch as smp
import torch
from losses import build_loss
from train import augment_for, count_parameters, evaluate, fit, library_versions, set_seed


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

        def _fake_evaluate(model, loader, loss_fn, device, max_batches=None):
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


class TestSetSeed:
    def test_sets_torch_and_numpy_rng_deterministically(self):
        # Plan Section 7.1 Phase A1: `set_seed` is the single call site that
        # has to cover every RNG source `fit()` touches (torch CPU, numpy --
        # kornia augmentation and DataLoader shuffling draw from these), so
        # that calling it twice with the same value reproduces the same
        # draws from both.
        set_seed(123)
        first_torch = torch.rand(3)
        first_numpy = np.random.rand(3)

        set_seed(123)
        second_torch = torch.rand(3)
        second_numpy = np.random.rand(3)

        assert torch.equal(first_torch, second_torch)
        assert np.array_equal(first_numpy, second_numpy)


class TestFitSeed:
    def test_same_seed_produces_identical_metrics_across_runs(self, tmp_path, tiny_geotiff_factory):
        # This is the literal Phase A1 gate (plan Section 7.0): "same cell
        # run twice -> identical val_loss to 4 decimals." Augmentation is on
        # (E2/E3's real setting) so the test also covers kornia's randomness,
        # not just DataLoader shuffle order.
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=6, size=16)
        val_df = _patches_df(folder, n_rows=4, size=16)

        def _run():
            torch.manual_seed(0)  # identical starting weights each call
            model = torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 1, 1),
            )
            return fit(
                model,
                train_df,
                val_df,
                dataset="starcop_mini",
                lr=1e-3,
                batch_size=2,
                max_epochs=2,
                patience=10,
                seed=42,
                augment=True,
                log_to_mlflow=False,
            )

        first = _run()
        second = _run()

        assert first["loss"] == second["loss"]
        assert first["f1"] == second["f1"]

    def test_seed_is_wired_into_the_train_loaders_generator(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        # `_patches_df` points every row at the same window, so patch
        # *content* can't distinguish shuffle orders -- assert on the wiring
        # instead: `fit(seed=N)` must construct its (shuffled) train loader
        # with a generator seeded from that same N, per Phase A1's "add the
        # DataLoader's generator=" requirement.
        import train as train_module

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        real_data_loader = train_module.DataLoader
        captured_generators = []

        def _recording_data_loader(*args, **kwargs):
            captured_generators.append(kwargs.get("generator"))
            return real_data_loader(*args, **kwargs)

        monkeypatch.setattr(train_module, "DataLoader", _recording_data_loader)

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            max_steps=1,
            seed=7,
            log_to_mlflow=False,
        )

        train_generator = captured_generators[0]
        assert train_generator is not None
        assert train_generator.initial_seed() == 7


class TestCountParameters:
    def test_counts_every_parameter_tensor_element(self):
        # Linear(4, 2): weight (4*2=8) + bias (2) = 10 -- a hand-computable
        # case rather than a real architecture, so the test doesn't depend
        # on `architectures.py`'s exact parameter counts staying fixed.
        model = torch.nn.Linear(4, 2)
        assert count_parameters(model) == 10

    def test_matches_the_sum_of_numel_across_parameters(self):
        # Cross-checks against the direct computation for a multi-layer
        # model, so the helper isn't just right for the single-layer case.
        model = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 3), torch.nn.Conv2d(8, 1, 1))
        expected = sum(p.numel() for p in model.parameters())
        assert count_parameters(model) == expected


class TestLibraryVersions:
    def test_reports_torch_smp_and_cuda_versions_for_mlflow_params(self):
        # Plan Section 7.1 Phase A2: these three go into every run's MLflow
        # params so Section 9's numbers don't depend on retyping versions
        # from memory later. Checked against the actual imported modules,
        # not hardcoded strings, so this fails the day a version drifts.
        versions = library_versions()

        assert versions["torch_version"] == torch.__version__
        assert versions["smp_version"] == smp.__version__
        assert versions["cuda_version"] == (torch.version.cuda or "cpu")


class TestFitTiming:
    def test_returns_nonzero_wall_clock_and_seconds_per_epoch(self, tmp_path, tiny_geotiff_factory):
        # Plan Section 7.1 Phase A3: R3's epoch-cap sizing has to come from
        # a *measured* per-epoch time, not the `stats.py` I/O estimate
        # carried forward in Section 7's own prose -- so these have to be
        # real, non-zero numbers coming out of an actual `fit()` call.
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        model = torch.nn.Conv2d(4, 1, 1)
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=2,
            patience=10,
            log_to_mlflow=False,
        )

        assert result["wall_clock_seconds"] > 0
        assert result["seconds_per_epoch"] > 0

    def test_seconds_per_epoch_is_wall_clock_divided_by_epochs_run(
        self, tmp_path, tiny_geotiff_factory
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        model = torch.nn.Conv2d(4, 1, 1)
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=3,
            patience=10,
            log_to_mlflow=False,
        )

        assert result["seconds_per_epoch"] == pytest.approx(
            result["wall_clock_seconds"] / result["epochs_run"]
        )


class TestEvaluateMaxBatches:
    def test_stops_after_max_batches_and_ignores_the_rest(self):
        # Plan Section 7.1 Phase A4: without this, a "2-step smoke test" on
        # `raw-full` still runs a full 26,607-patch `evaluate()` pass every
        # epoch. Three batches with distinguishable logits -- capping at 2
        # must change the averaged loss, proving the third was never read.
        batches = [
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.zeros(1, 1, 2, 2)},
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.zeros(1, 1, 2, 2)},
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.zeros(1, 1, 2, 2)},
        ]
        logits = [
            torch.full((1, 1, 2, 2), -5.0),
            torch.full((1, 1, 2, 2), 0.0),
            torch.full((1, 1, 2, 2), 5.0),
        ]
        model = _FixedLogitModel(logits)
        loss_fn = build_loss(pos_weight=1.0)

        result = evaluate(model, batches, loss_fn, device="cpu", max_batches=2)

        target = torch.zeros(1, 1, 2, 2)
        expected_loss = (loss_fn(logits[0], target).item() + loss_fn(logits[1], target).item()) / 2
        assert result["loss"] == pytest.approx(expected_loss)

    def test_none_processes_every_batch_unchanged(self):
        # Default behavior (no cap) must match the pre-A4 code exactly --
        # regression guard against `max_batches` accidentally becoming a
        # non-optional truncation.
        batches = [
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.zeros(1, 1, 2, 2)},
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.zeros(1, 1, 2, 2)},
        ]
        model = _FixedLogitModel([torch.full((1, 1, 2, 2), -5.0), torch.full((1, 1, 2, 2), 5.0)])
        loss_fn = build_loss(pos_weight=1.0)

        result = evaluate(model, batches, loss_fn, device="cpu", max_batches=None)

        target = torch.zeros(1, 1, 2, 2)
        expected_loss = (
            loss_fn(torch.full((1, 1, 2, 2), -5.0), target).item()
            + loss_fn(torch.full((1, 1, 2, 2), 5.0), target).item()
        ) / 2
        assert result["loss"] == pytest.approx(expected_loss)


class TestFitMaxValBatches:
    def test_max_val_batches_is_threaded_through_to_evaluate(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        captured_max_batches = []

        def _recording_evaluate(model, loader, loss_fn, device, max_batches=None):
            captured_max_batches.append(max_batches)
            return {"loss": 0.0, "f1": 0.0, "degenerate": False, "positive_fraction": 0.0}

        monkeypatch.setattr("train.evaluate", _recording_evaluate)

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            max_val_batches=3,
            log_to_mlflow=False,
        )

        assert captured_max_batches == [3]

    def test_defaults_to_no_cap(self, tmp_path, tiny_geotiff_factory, monkeypatch):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        captured_max_batches = []

        def _recording_evaluate(model, loader, loss_fn, device, max_batches=None):
            captured_max_batches.append(max_batches)
            return {"loss": 0.0, "f1": 0.0, "degenerate": False, "positive_fraction": 0.0}

        monkeypatch.setattr("train.evaluate", _recording_evaluate)

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            log_to_mlflow=False,
        )

        assert captured_max_batches == [None]
