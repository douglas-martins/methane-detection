import pytest
import torch
from architectures import build_e1
from evaluate import evaluate_full_metrics, load_checkpoint


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
