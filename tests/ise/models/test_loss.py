import pytest
import torch

from ise.models.loss import (
    WeightedMSELoss,
    WeightedMSELossWithSignPenalty,
    WeightedMSEPCALoss,
    MSEDeviationLoss,
    WeightedPCALoss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t(*values):
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# WeightedMSELoss
# ---------------------------------------------------------------------------

class TestWeightedMSELoss:
    @pytest.fixture
    def criterion(self):
        return WeightedMSELoss(data_mean=0.0, data_std=1.0, weight_factor=2.0)

    def test_zero_loss_on_perfect_predictions(self, criterion):
        x = _t(1.0, 2.0, 3.0)
        loss = criterion(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_positive_on_error(self, criterion):
        pred = _t(0.0, 0.0, 0.0)
        target = _t(1.0, 2.0, 3.0)
        loss = criterion(pred, target)
        assert loss.item() > 0.0

    def test_extreme_values_weighted_more_than_std_mse(self):
        # Extreme target → weighted loss > plain MSE
        plain_mse = torch.nn.MSELoss()
        criterion = WeightedMSELoss(data_mean=0.0, data_std=1.0, weight_factor=5.0)

        pred = _t(0.0)
        target = _t(10.0)   # 10 std devs from mean → high weight
        weighted_loss = criterion(pred, target).item()
        standard_loss = plain_mse(pred, target).item()
        assert weighted_loss > standard_loss

    def test_near_mean_target_weight_close_to_one(self):
        criterion = WeightedMSELoss(data_mean=0.0, data_std=1.0, weight_factor=1.0)
        plain_mse = torch.nn.MSELoss()
        pred = _t(0.0)
        target = _t(0.0)   # exactly at mean → weight = 1
        assert criterion(pred, target).item() == pytest.approx(plain_mse(pred, target).item(), abs=1e-6)


# ---------------------------------------------------------------------------
# WeightedMSELossWithSignPenalty
# ---------------------------------------------------------------------------

class TestWeightedMSELossWithSignPenalty:
    @pytest.fixture
    def criterion(self):
        return WeightedMSELossWithSignPenalty(
            data_mean=0.0, data_std=1.0, weight_factor=1.0, sign_penalty_factor=2.0
        )

    def test_wrong_sign_prediction_penalised_more(self, criterion):
        target = _t(1.0)
        correct_sign_pred = _t(0.5)    # same sign
        wrong_sign_pred   = _t(-0.5)   # opposite sign, same magnitude error
        loss_correct = criterion(correct_sign_pred, target)
        loss_wrong   = criterion(wrong_sign_pred,   target)
        assert loss_wrong.item() > loss_correct.item()

    def test_zero_loss_on_perfect_predictions(self, criterion):
        x = _t(1.0, -1.0, 0.5)
        assert criterion(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_correct_sign_gives_nonnegative_loss(self, criterion):
        pred   = _t(0.3, -0.4)
        target = _t(1.0, -1.0)
        assert criterion(pred, target).item() >= 0.0


# ---------------------------------------------------------------------------
# MSEDeviationLoss
# ---------------------------------------------------------------------------

class TestMSEDeviationLoss:
    @pytest.fixture
    def criterion(self):
        return MSEDeviationLoss(threshold=1.0, penalty_multiplier=3.0)

    def test_small_errors_approx_mse(self, criterion):
        pred   = _t(0.0, 0.0)
        target = _t(0.5, 0.5)   # error = 0.5 < threshold=1.0
        plain_mse = torch.nn.MSELoss()(pred, target).item()
        loss = criterion(pred, target).item()
        assert loss == pytest.approx(plain_mse, rel=1e-5)

    def test_large_errors_exceed_plain_mse(self, criterion):
        pred   = _t(0.0)
        target = _t(5.0)   # error = 5.0 > threshold=1.0
        plain_mse = torch.nn.MSELoss()(pred, target).item()
        loss = criterion(pred, target).item()
        assert loss > plain_mse

    def test_zero_loss_on_perfect_predictions(self, criterion):
        x = _t(1.0, 2.0)
        assert criterion(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_penalty_multiplier_scales_extra_loss(self):
        low_penalty  = MSEDeviationLoss(threshold=0.1, penalty_multiplier=1.0)
        high_penalty = MSEDeviationLoss(threshold=0.1, penalty_multiplier=10.0)
        pred   = _t(0.0)
        target = _t(5.0)
        assert high_penalty(pred, target).item() > low_penalty(pred, target).item()


# ---------------------------------------------------------------------------
# WeightedPCALoss
# ---------------------------------------------------------------------------

class TestWeightedPCALoss:
    def test_higher_component_weight_gives_higher_loss(self):
        low_weight  = WeightedPCALoss(component_weights=[1.0, 1.0])
        high_weight = WeightedPCALoss(component_weights=[10.0, 1.0])

        pred   = torch.zeros(4, 2)
        target = torch.ones(4, 2)

        assert high_weight(pred, target).item() > low_weight(pred, target).item()

    def test_zero_loss_on_perfect_predictions(self):
        criterion = WeightedPCALoss(component_weights=[1.0, 2.0])
        x = torch.rand(4, 2)
        assert criterion(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_mismatched_shape_raises(self):
        criterion = WeightedPCALoss(component_weights=[1.0, 2.0])
        with pytest.raises(ValueError, match="shape"):
            criterion(torch.zeros(4, 2), torch.zeros(4, 3))


# ---------------------------------------------------------------------------
# WeightedMSEPCALoss
# ---------------------------------------------------------------------------

class TestWeightedMSEPCALoss:
    def test_zero_loss_on_perfect_predictions(self):
        criterion = WeightedMSEPCALoss(data_mean=0.0, data_std=1.0)
        x = _t(1.0, 2.0)
        assert criterion(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_mismatched_shape_raises(self):
        criterion = WeightedMSEPCALoss(data_mean=0.0, data_std=1.0)
        with pytest.raises(ValueError, match="shape"):
            criterion(torch.zeros(3), torch.zeros(4))
