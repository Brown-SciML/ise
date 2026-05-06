import pytest
import torch

from ise.models.deep_ensemble import DeepEnsemble
from ise.models.lstm import LSTM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lstm(input_size=8):
    return LSTM(lstm_num_layers=1, lstm_hidden_size=16, input_size=input_size, output_size=1)


def _make_trained_lstm(input_size=8):
    m = _make_lstm(input_size)
    m.trained = True
    m.sequence_length = 5
    m.best_loss = 0.1
    m.epochs_trained = 1
    return m


@pytest.fixture
def small_input():
    """(N=86, features=9) feature matrix — one full projection (8 base + 1 latent)."""
    return torch.rand(86, 9)


# ---------------------------------------------------------------------------
# Constructor — auto-generated members
# ---------------------------------------------------------------------------


class TestDeepEnsembleAutoConstructor:
    def test_creates_correct_number_of_members(self):
        de = DeepEnsemble(input_size=8, num_ensemble_members=3, latent_dim=0)
        assert len(de.ensemble_members) == 3

    def test_members_are_lstm_instances(self):
        de = DeepEnsemble(input_size=8, num_ensemble_members=3, latent_dim=0)
        assert all(isinstance(m, LSTM) for m in de.ensemble_members)

    def test_input_size_includes_latent(self):
        de = DeepEnsemble(input_size=8, num_ensemble_members=2, latent_dim=1)
        assert de.input_size == 9

    def test_trained_false_when_no_members_trained(self):
        de = DeepEnsemble(input_size=8, num_ensemble_members=2, latent_dim=0)
        assert de.trained is False


# ---------------------------------------------------------------------------
# Constructor — explicit members
# ---------------------------------------------------------------------------


class TestDeepEnsembleExplicitConstructor:
    def test_accepts_list_of_lstms(self):
        members = [_make_lstm(9), _make_lstm(9)]
        de = DeepEnsemble(ensemble_members=members)
        assert len(de.ensemble_members) == 2

    def test_invalid_member_type_raises(self):
        with pytest.raises(ValueError, match="LSTM"):
            DeepEnsemble(ensemble_members=["not_an_lstm"])

    def test_trained_true_when_all_members_trained(self):
        members = [_make_trained_lstm(9), _make_trained_lstm(9)]
        de = DeepEnsemble(ensemble_members=members)
        assert de.trained is True

    def test_trained_false_when_some_untrained(self):
        members = [_make_trained_lstm(9), _make_lstm(9)]
        de = DeepEnsemble(ensemble_members=members)
        assert de.trained is False


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestDeepEnsembleForward:
    @pytest.fixture
    def trained_de(self):
        members = [_make_trained_lstm(9), _make_trained_lstm(9)]
        de = DeepEnsemble(ensemble_members=members)
        return de

    def test_returns_tuple_of_two(self, trained_de, small_input):
        result = trained_de.forward(small_input)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mean_shape(self, trained_de, small_input):
        mean, _ = trained_de.forward(small_input)
        assert mean.shape[0] == small_input.shape[0]

    def test_epistemic_shape(self, trained_de, small_input):
        _, epistemic = trained_de.forward(small_input)
        assert epistemic.shape[0] == small_input.shape[0]

    def test_untrained_emits_warning(self):
        # LSTMs need sequence_length set to run predict(); untrained flag triggers warning
        m1 = _make_lstm(9)
        m1.sequence_length = 5
        m2 = _make_lstm(9)
        m2.sequence_length = 5
        de = DeepEnsemble(ensemble_members=[m1, m2])
        x = torch.rand(86, 9)
        with pytest.warns(UserWarning):
            de.forward(x)

    def test_epistemic_non_negative(self, trained_de, small_input):
        # std across members is always >= 0
        _, epistemic = trained_de.forward(small_input)
        assert (epistemic >= 0).all()


# ---------------------------------------------------------------------------
# predict is an alias for forward in eval mode
# ---------------------------------------------------------------------------


class TestDeepEnsemblePredict:
    def test_predict_returns_same_shapes_as_forward(self):
        members = [_make_trained_lstm(9), _make_trained_lstm(9)]
        de = DeepEnsemble(ensemble_members=members)
        x = torch.rand(86, 9)
        mean_f, ep_f = de.forward(x)
        mean_p, ep_p = de.predict(x)
        assert mean_f.shape == mean_p.shape
        assert ep_f.shape == ep_p.shape
