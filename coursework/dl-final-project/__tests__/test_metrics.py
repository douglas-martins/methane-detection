import torch
from metrics import pixel_f1


class TestPixelF1:
    def test_perfect_prediction_scores_one(self):
        predictions = torch.tensor([1.0, 0.0, 1.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert pixel_f1(predictions, targets) == 1.0

    def test_all_wrong_scores_zero(self):
        predictions = torch.tensor([1.0, 1.0])
        targets = torch.tensor([0.0, 0.0])
        assert pixel_f1(predictions, targets) == 0.0

    def test_no_predicted_or_true_positives_scores_zero_not_nan(self):
        # The degenerate all-negative case this section explicitly worries
        # about must not silently produce a NaN F1.
        predictions = torch.zeros(4)
        targets = torch.zeros(4)
        result = pixel_f1(predictions, targets)
        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_matches_hand_computed_precision_recall(self):
        # tp=1, fp=1, fn=1 -> precision=0.5, recall=0.5, f1=0.5
        predictions = torch.tensor([1.0, 1.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0])
        assert pixel_f1(predictions, targets) == 0.5
