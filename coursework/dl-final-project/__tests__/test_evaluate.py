import pytest
import torch
from architectures import build_e1
from evaluate import _build_run_name, evaluate_full_metrics, load_checkpoint


class _FixedLogitModel(torch.nn.Module):
    """Returns pre-baked logits regardless of input -- lets tests control predictions exactly.

    Duplicated from test_train.py's own helper of the same name rather than
    imported -- each test module owns its fixtures here, matching this
    project's established per-file convention (e.g. `_make_scene`/
    `_patches_df` are similarly duplicated, not shared, across test files).
    """

    def __init__(self, logits_by_call: list[torch.Tensor]):
        super().__init__()
        self._logits_by_call = iter(logits_by_call)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return next(self._logits_by_call)


class TestEvaluateFullMetrics:
    def test_pools_counts_across_batches_instead_of_averaging_per_batch_scores(self):
        # Batch 1: tp=1 (correct), fp=1 (wrong). Batch 2: fn=1 (missed tp).
        # Pooled: tp=1, fp=1, fn=1 -> precision=recall=f1=0.5 (test_metrics.py's
        # own hand-computed case) -- averaging per-batch scores instead would
        # not equal this, since batch 2 alone has an undefined precision.
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
        logits_batch1 = torch.tensor([[[[5.0, 5.0], [-5.0, -5.0]]], [[[-5.0, -5.0], [-5.0, -5.0]]]])
        logits_batch2 = torch.full((2, 1, 2, 2), -5.0)
        model = _FixedLogitModel([logits_batch1, logits_batch2])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5
        assert result["confusion_matrix"] == [[13.0, 1.0], [1.0, 1.0]]

    def test_perfect_prediction_scores_one_on_both_precision_and_recall(self):
        batches = [
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])}
        ]
        logits = torch.tensor([[[[5.0, -5.0], [-5.0, -5.0]]]])
        model = _FixedLogitModel([logits])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_respects_a_non_default_threshold(self):
        # A single 0.6-probability pixel: counted positive under threshold=0.5
        # but negative under threshold=0.7 -- proves `threshold` is actually
        # applied, not hardcoded to 0.5 inside the accumulation loop.
        logit_for_p06 = torch.logit(torch.tensor(0.6))
        batches = [{"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])}]
        model_default = _FixedLogitModel([torch.full((1, 1, 1, 1), logit_for_p06.item())])
        model_strict = _FixedLogitModel([torch.full((1, 1, 1, 1), logit_for_p06.item())])

        default_result = evaluate_full_metrics(model_default, batches, device="cpu")
        strict_result = evaluate_full_metrics(model_strict, batches, device="cpu", threshold=0.7)

        assert default_result["recall"] == 1.0
        assert strict_result["recall"] == 0.0

    def test_counts_total_patches_processed_across_batches(self):
        # Plan Section 8's own R1 validation requirement: a full-split run
        # must be verified by the processed-patch count matching the
        # manifest, not by the run merely exiting 0 -- so this count has to
        # be a real, checkable number the caller can compare, not inferred
        # from batch count (the last batch of a real split is often smaller
        # than `batch_size`).
        batches = [
            {"input": torch.zeros(2, 4, 1, 1), "output": torch.zeros(2, 1, 1, 1)},
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.zeros(1, 1, 1, 1)},
        ]
        model = _FixedLogitModel([torch.zeros(2, 1, 1, 1), torch.zeros(1, 1, 1, 1)])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert result["patches_processed"] == 3

    def test_stops_after_max_batches(self):
        # Same reasoning as train.py's own evaluate(max_batches=...): a full
        # raw-tier split (up to 16,758 test patches) must never be forced to
        # run in full just to smoke-test this function.
        batches = [
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])},
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])},
        ]
        # Second batch's logit would flip the outcome (a huge negative logit
        # misses the true positive) -- only detectable if it were read.
        model = _FixedLogitModel([torch.full((1, 1, 1, 1), 5.0), torch.full((1, 1, 1, 1), -5.0)])

        result = evaluate_full_metrics(model, batches, device="cpu", max_batches=1)

        assert result["recall"] == 1.0

    def test_stops_after_exactly_max_batches_out_of_more_than_two(self):
        # 3 batches with max_batches=2 -- distinguishes a broken n_batches
        # accumulator (e.g. always resetting to 1, or incrementing by 2 each
        # time) from a real running count, which test_stops_after_max_batches
        # above (only 2 batches) can't: after exactly one increment, a
        # hardcoded "= 1" and a real "+= 1" both happen to read 1.
        batches = [
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])}
            for _ in range(3)
        ]
        model = _FixedLogitModel([torch.zeros(1, 1, 1, 1)] * 3)

        result = evaluate_full_metrics(model, batches, device="cpu", max_batches=2)

        assert result["patches_processed"] == 2

    def test_model_receives_the_batchs_actual_input_tensor(self):
        # A model that ignores its argument (like _FixedLogitModel) can't
        # distinguish a correctly-passed input from a dropped one -- record
        # what's actually received instead.
        received = []

        class _RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                received.append(x)
                return torch.zeros(1, 1, 1, 1)

        batch_input = torch.full((1, 4, 1, 1), 3.0)
        batches = [{"input": batch_input, "output": torch.zeros(1, 1, 1, 1)}]

        evaluate_full_metrics(_RecordingModel(), batches, device="cpu")

        assert received[0] is not None
        torch.testing.assert_close(received[0], batch_input)

    def test_probability_exactly_at_threshold_is_not_predicted_positive(self):
        # Strict `>` -- sigmoid(0) == 0.5 exactly, so this must miss the one
        # true positive (fn=1), not count it (tp=1).
        batches = [{"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])}]
        model = _FixedLogitModel([torch.zeros(1, 1, 1, 1)])

        result = evaluate_full_metrics(model, batches, device="cpu", threshold=0.5)

        assert result["confusion_matrix"] == [[0.0, 0.0], [1.0, 0.0]]

    def test_wall_clock_is_the_elapsed_duration_not_a_sum_of_timestamps(self, monkeypatch):
        # perf_counter() is an arbitrary large reference point, not zero-based
        # -- summing two calls instead of subtracting would give a huge wrong
        # duration that a mere "> 0" check can't distinguish from a real one.
        timestamps = iter([1000.0, 1000.25])
        monkeypatch.setattr("evaluate.time.perf_counter", lambda: next(timestamps))
        batches = [{"input": torch.zeros(1, 4, 1, 1), "output": torch.zeros(1, 1, 1, 1)}]
        model = _FixedLogitModel([torch.zeros(1, 1, 1, 1)])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert result["wall_clock_seconds"] == pytest.approx(0.25)


class TestEvaluateFullMetricsPRAUC:
    def test_computes_pr_auc_over_the_given_threshold_sweep(self):
        # Same probs/targets/thresholds as test_metrics.py's
        # TestAveragePrecisionFromSweep hand-computed case (AP=0.5), driven
        # through a real model + evaluate_full_metrics this time.
        batches = [
            {
                "input": torch.zeros(4, 4, 1, 1),
                "output": torch.tensor([[[[1.0]]], [[[1.0]]], [[[0.0]]], [[[0.0]]]]),
            }
        ]
        probs = torch.tensor([0.9, 0.4, 0.1, 0.6])
        logits = torch.logit(probs).reshape(4, 1, 1, 1)
        model = _FixedLogitModel([logits])

        result = evaluate_full_metrics(
            model, batches, device="cpu", pr_thresholds=torch.tensor([0.0, 0.5, 1.0])
        )

        assert result["pr_auc"] == pytest.approx(0.5, abs=1e-4)
        assert result["precision_recall_curve"] == pytest.approx(
            [(1.0, 0.5), (0.5, 0.5), (0.0, 0.0)], abs=1e-4
        )

    def test_sweep_pools_across_batches_not_just_the_last_one(self):
        # Batch 1: tp=1 at threshold 0.5 (correct positive). Batch 2: fn=1
        # (missed positive). Pooled: recall=0.5, precision=1.0 -> AP=0.5.
        # If only the last batch's sweep counted (a broken `=` instead of
        # `+=` accumulator), it would see only batch 2's tp=0, fn=1 -> AP=0.0.
        batches = [
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])},
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])},
        ]
        model = _FixedLogitModel([torch.full((1, 1, 1, 1), 5.0), torch.full((1, 1, 1, 1), -5.0)])

        result = evaluate_full_metrics(
            model, batches, device="cpu", pr_thresholds=torch.tensor([0.5])
        )

        assert result["pr_auc"] == pytest.approx(0.5)

    def test_defaults_to_a_101_point_threshold_sweep(self):
        batches = [{"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])}]
        model = _FixedLogitModel([torch.full((1, 1, 1, 1), 5.0)])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert len(result["precision_recall_curve"]) == 101

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_works_when_model_and_batches_are_on_cuda(self):
        # Real bug found while running evaluate.py for real on this machine's
        # GPU: `_DEFAULT_PR_THRESHOLDS` is a plain CPU tensor, so the sweep
        # crashed with a device-mismatch error the CPU-only test suite could
        # never catch -- `pr_thresholds`/`sweep_totals` must follow `device`.
        batches = [
            {
                "input": torch.zeros(1, 4, 1, 1, device="cuda"),
                "output": torch.tensor([[[[1.0]]]], device="cuda"),
            }
        ]
        model = _FixedLogitModel([torch.full((1, 1, 1, 1), 5.0, device="cuda")])

        result = evaluate_full_metrics(model, batches, device="cuda")

        assert result["recall"] == 1.0
        assert result["pr_auc"] > 0.0


class TestEvaluateFullMetricsPerPatchDetection:
    def test_computes_per_patch_detection_counts_and_rate(self):
        batches = [
            {
                "input": torch.zeros(2, 4, 2, 2),
                "output": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[1.0, 0.0], [0.0, 0.0]]]]),
            }
        ]
        # sample0 detected (tp at 0,0); sample1 missed entirely.
        logits = torch.tensor([[[[5.0, -5.0], [-5.0, -5.0]]], [[[-5.0, -5.0], [-5.0, -5.0]]]])
        model = _FixedLogitModel([logits])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert result["per_patch_positive_patches"] == 2
        assert result["per_patch_detected_patches"] == 1
        assert result["per_patch_detection_rate"] == 0.5


class TestEvaluateFullMetricsTiming:
    def test_returns_positive_wall_clock_and_consistent_throughput(self):
        # GPU-vs-CPU inference comparison needs a real, checkable throughput
        # number -- not just a duration -- so `patches_per_second` must be
        # derivable and consistent with `patches_processed`, the same
        # consistency contract `train.py::fit`'s own `seconds_per_epoch`
        # already has against `wall_clock_seconds`/`epochs_run`.
        batches = [
            {"input": torch.zeros(2, 4, 1, 1), "output": torch.zeros(2, 1, 1, 1)},
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.zeros(1, 1, 1, 1)},
        ]
        model = _FixedLogitModel([torch.zeros(2, 1, 1, 1), torch.zeros(1, 1, 1, 1)])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert result["wall_clock_seconds"] > 0
        assert result["patches_per_second"] == pytest.approx(
            result["patches_processed"] / result["wall_clock_seconds"]
        )


class TestBuildRunName:
    def test_same_tier_default_device_has_no_suffix(self):
        # Default (auto-detected) device must not change any existing run
        # name already referenced in report.md's "Métricas de avaliação".
        assert _build_run_name("E1", "mini", "mini", None, "cuda") == "E1-mini-eval"

    def test_cross_tier_default_device_has_no_suffix(self):
        assert _build_run_name("E2", "raw-full", "mini", None, "cuda") == "E2-mini-on-raw-full-eval"

    def test_explicit_device_appends_a_suffix(self):
        # An explicit `device=` override means a GPU-vs-CPU comparison run --
        # it must not collide with (or overwrite the identity of) the
        # canonical same-tier run already logged under the un-suffixed name.
        assert _build_run_name("E1", "mini", "mini", "cpu", "cpu") == "E1-mini-eval-cpu"

    def test_cross_tier_with_explicit_device_appends_a_suffix(self):
        assert (
            _build_run_name("E2", "raw-full", "mini", "cpu", "cpu")
            == "E2-mini-on-raw-full-eval-cpu"
        )


class TestLoadCheckpoint:
    def test_loads_saved_weights_and_reproduces_the_same_forward_pass(self, tmp_path):
        original = build_e1()
        checkpoint_path = tmp_path / "e1.pt"
        torch.save(original.state_dict(), checkpoint_path)

        loaded = load_checkpoint("E1", checkpoint_path, device="cpu")

        sample_input = torch.randn(1, 4, 16, 16)
        original.eval()
        with torch.no_grad():
            assert torch.equal(loaded(sample_input), original(sample_input))
