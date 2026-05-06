import numpy as np
import pytest
import torch

from ise.models.deep_ensemble import DeepEnsemble
from ise.models.iseflow import ISEFlow, smooth_projections
from ise.models.lstm import LSTM
from ise.models.normalizing_flow import NormalizingFlow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nf(input_size=8):
    nf = NormalizingFlow(
        input_size=input_size, output_size=1, num_flow_transforms=2, flow_hidden_features=8
    )
    nf.model_dir = None
    return nf


def _make_trained_nf(input_size=8):
    nf = _make_nf(input_size)
    nf.trained = True
    nf.best_loss = 0.5
    nf.epochs_trained = 5
    return nf


def _make_lstm(input_size=9):
    m = LSTM(lstm_num_layers=1, lstm_hidden_size=16, input_size=input_size, output_size=1)
    m.trained = True
    m.sequence_length = 5
    m.best_loss = 0.1
    m.epochs_trained = 1
    return m


def _make_de(input_size=8, latent_dim=1):
    members = [_make_lstm(input_size + latent_dim), _make_lstm(input_size + latent_dim)]
    de = DeepEnsemble(ensemble_members=members)
    return de


@pytest.fixture
def trained_iseflow():
    nf = _make_trained_nf(input_size=8)
    de = _make_de(input_size=8, latent_dim=1)
    model = ISEFlow(de, nf)
    model.trained = True
    return model


@pytest.fixture
def small_features():
    return torch.rand(86, 8)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestISEFlowConstructor:
    def test_accepts_correct_types(self):
        nf = _make_trained_nf(8)
        de = _make_de(8, 1)
        model = ISEFlow(de, nf)
        assert isinstance(model, ISEFlow)

    def test_raises_on_wrong_de_type(self):
        nf = _make_trained_nf(8)
        with pytest.raises(ValueError, match="DeepEnsemble"):
            ISEFlow("not_a_de", nf)

    def test_raises_on_wrong_nf_type(self):
        de = _make_de(8, 1)
        with pytest.raises(ValueError, match="NormalizingFlow"):
            ISEFlow(de, "not_a_nf")

    def test_trained_reflects_submodule_state(self):
        nf = _make_trained_nf(8)
        de = _make_de(8, 1)
        model = ISEFlow(de, nf)
        assert model.trained == (de.trained and nf.trained)

    def test_save_raises_if_not_trained(self, tmp_path):
        nf = _make_nf(8)
        de = _make_de(8, 1)
        model = ISEFlow(de, nf)
        model.trained = False
        with pytest.raises(ValueError, match="trained"):
            model.save(str(tmp_path / "iseflow_dir"))

    def test_save_raises_if_path_is_pth_file(self, tmp_path):
        nf = _make_trained_nf(8)
        de = _make_de(8, 1)
        model = ISEFlow(de, nf)
        model.trained = True
        with pytest.raises(ValueError, match="directory"):
            model.save(str(tmp_path / "model.pth"))


# ---------------------------------------------------------------------------
# forward — uncertainty dict structure
# ---------------------------------------------------------------------------


class TestISEFlowForward:
    def test_returns_tuple(self, trained_iseflow, small_features):
        result = trained_iseflow.forward(small_features)
        assert isinstance(result, tuple) and len(result) == 2

    def test_uncertainty_keys(self, trained_iseflow, small_features):
        _, uncertainties = trained_iseflow.forward(small_features)
        assert set(uncertainties.keys()) == {"total", "epistemic", "aleatoric"}

    def test_predictions_shape(self, trained_iseflow, small_features):
        predictions, _ = trained_iseflow.forward(small_features)
        assert predictions.shape[0] == small_features.shape[0]

    def test_total_equals_epistemic_plus_aleatoric(self, trained_iseflow, small_features):
        _, uncertainties = trained_iseflow.forward(small_features)
        np.testing.assert_allclose(
            uncertainties["total"],
            uncertainties["epistemic"] + uncertainties["aleatoric"],
            rtol=1e-5,
        )

    def test_aleatoric_non_negative(self, trained_iseflow, small_features):
        _, uncertainties = trained_iseflow.forward(small_features)
        assert (uncertainties["aleatoric"] >= 0).all()


# ---------------------------------------------------------------------------
# smooth_projections
# ---------------------------------------------------------------------------


class TestSmoothProjections:
    def test_no_op_when_window_zero(self):
        data = np.random.rand(86)
        result = smooth_projections(data, window_size=0)
        np.testing.assert_array_equal(result, data)

    def test_no_op_when_window_one(self):
        data = np.random.rand(86)
        result = smooth_projections(data, window_size=1)
        np.testing.assert_array_equal(result, data)

    def test_1d_output_shape_preserved(self):
        data = np.random.rand(86)
        result = smooth_projections(data, window_size=5)
        assert result.shape == data.shape

    def test_2d_output_shape_preserved(self):
        data = np.random.rand(86, 1)
        result = smooth_projections(data, window_size=5)
        assert result.shape == data.shape

    def test_multi_projection_shape_preserved(self):
        data = np.random.rand(86 * 3)
        result = smooth_projections(data, window_size=5, projection_length=86)
        assert result.shape == data.shape

    def test_no_bleed_across_projection_boundary(self):
        # Two projections with very different values — if bleed happened the
        # values near the boundary would shift toward the other projection's level.
        proj1 = np.zeros(86)
        proj2 = np.ones(86) * 100.0
        data = np.concatenate([proj1, proj2])

        result = smooth_projections(data, window_size=11, projection_length=86)

        # Last value of proj1 segment must still be 0 (no bleed from proj2)
        assert result[85] == pytest.approx(0.0, abs=1e-6)
        # First value of proj2 segment must still be 100 (no bleed from proj1)
        assert result[86] == pytest.approx(100.0, abs=1e-6)

    def test_even_window_auto_adjusted(self):
        # Even window sizes are internally incremented to odd — should not crash
        data = np.random.rand(86)
        result = smooth_projections(data, window_size=4)
        assert result.shape == data.shape

    def test_output_differs_from_input_when_window_gt_1(self):
        rng = np.random.default_rng(0)
        data = rng.random(86)
        result = smooth_projections(data, window_size=7)
        assert not np.allclose(result, data)
