"""Tests for src/registry/promotion_criteria.py -- pure functions, no MLflow
SDK calls, no live server needed (Test Size: Small)."""

import math

import promotion_criteria


class TestCheckThresholds:
    def test_returns_no_reasons_when_all_metrics_meet_threshold(self):
        reasons = promotion_criteria.check_thresholds(
            metrics={"val_accuracy": 0.9, "val_f1score": 0.8},
            thresholds={"val_accuracy": 0.85, "val_f1score": 0.70},
        )

        assert reasons == []

    def test_returns_no_reasons_when_metric_exactly_equals_threshold(self):
        reasons = promotion_criteria.check_thresholds(
            metrics={"val_accuracy": 0.85},
            thresholds={"val_accuracy": 0.85},
        )

        assert reasons == []

    def test_returns_a_reason_when_metric_is_below_threshold(self):
        reasons = promotion_criteria.check_thresholds(
            metrics={"val_accuracy": 0.80, "val_f1score": 0.8},
            thresholds={"val_accuracy": 0.85, "val_f1score": 0.70},
        )

        assert len(reasons) == 1
        assert "val_accuracy" in reasons[0]
        assert "0.8" in reasons[0]
        assert "0.85" in reasons[0]

    def test_returns_a_reason_when_metric_is_missing(self):
        reasons = promotion_criteria.check_thresholds(
            metrics={"val_f1score": 0.8},
            thresholds={"val_accuracy": 0.85, "val_f1score": 0.70},
        )

        assert len(reasons) == 1
        assert "val_accuracy" in reasons[0]
        assert "missing" in reasons[0].lower()


class TestIsLossHistoryStable:
    def test_true_for_a_smoothly_decreasing_history(self):
        history = [0.90, 0.72, 0.61, 0.55, 0.51, 0.48, 0.46, 0.45]

        assert promotion_criteria.is_loss_history_stable(history) is True

    def test_false_when_history_contains_nan(self):
        history = [0.5, 0.4, math.nan, 0.3, 0.28]

        assert promotion_criteria.is_loss_history_stable(history) is False

    def test_false_when_history_contains_inf(self):
        history = [0.5, 0.4, math.inf, 0.3, 0.28]

        assert promotion_criteria.is_loss_history_stable(history) is False

    def test_false_for_a_wildly_spiking_history(self):
        history = [0.5, 0.9, 0.05, 0.85, 0.02]

        assert promotion_criteria.is_loss_history_stable(history) is False

    def test_false_for_empty_history(self):
        assert promotion_criteria.is_loss_history_stable([]) is False

    def test_true_for_a_history_shorter_than_the_window(self):
        assert promotion_criteria.is_loss_history_stable([0.5, 0.48], window=5) is True


class TestEvaluateStaging:
    def test_promotes_when_metrics_pass_and_loss_is_stable(self):
        decision = promotion_criteria.evaluate_staging(
            metrics={"val_accuracy": 0.9, "val_f1score": 0.8},
            val_loss_history=[0.9, 0.7, 0.6, 0.55, 0.5, 0.48],
        )

        assert decision.promote is True
        assert decision.reasons == []

    def test_rejects_with_reason_when_val_accuracy_below_threshold(self):
        decision = promotion_criteria.evaluate_staging(
            metrics={"val_accuracy": 0.5, "val_f1score": 0.8},
            val_loss_history=[0.9, 0.7, 0.6, 0.55, 0.5, 0.48],
        )

        assert decision.promote is False
        assert any("val_accuracy" in reason for reason in decision.reasons)

    def test_rejects_with_reason_when_loss_history_is_unstable(self):
        decision = promotion_criteria.evaluate_staging(
            metrics={"val_accuracy": 0.9, "val_f1score": 0.8},
            val_loss_history=[0.5, 0.9, 0.05, 0.85, 0.02],
        )

        assert decision.promote is False
        assert any("loss" in reason.lower() for reason in decision.reasons)


class TestEvaluateProduction:
    def test_promotes_when_test_metrics_pass(self):
        decision = promotion_criteria.evaluate_production(
            metrics={"test_accuracy": 0.9, "test_f1score": 0.8}
        )

        assert decision.promote is True
        assert decision.reasons == []

    def test_rejects_with_reason_when_test_metrics_are_entirely_missing(self):
        decision = promotion_criteria.evaluate_production(metrics={"val_accuracy": 0.95})

        assert decision.promote is False
        assert any("test_accuracy" in reason for reason in decision.reasons)
        assert any("test_f1score" in reason for reason in decision.reasons)

    def test_rejects_with_reason_when_test_metric_below_threshold(self):
        decision = promotion_criteria.evaluate_production(
            metrics={"test_accuracy": 0.80, "test_f1score": 0.8}
        )

        assert decision.promote is False
        assert any("test_accuracy" in reason for reason in decision.reasons)
