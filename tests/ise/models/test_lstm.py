import json
import os

import pytest
import torch
import numpy as np

from ise.models.lstm import LSTM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_lstm():
    """Tiny LSTM (1 layer, 16 hidden) for fast CPU tests."""
    return LSTM(lstm_num_layers=1, lstm_hidden_size=16, input_size=8, output_size=1)


@pytest.fixture
def random_batch():
    """(batch=4, seq=5, features=8) input tensor."""
    return torch.rand(4, 5, 8)


@pytest.fixture
def random_train_data():
    """Small (N=86, features=8) train set — one full projection."""
    X = torch.rand(86, 8)
    y = torch.rand(86, 1)
    return X, y


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestLSTMConstructor:
    def test_num_layers_stored(self, small_lstm):
        assert small_lstm.lstm_num_layers == 1

    def test_hidden_size_stored(self, small_lstm):
        assert small_lstm.lstm_num_hidden == 16

    def test_input_size_stored(self, small_lstm):
        assert small_lstm.input_size == 8

    def test_output_size_stored(self, small_lstm):
        assert small_lstm.output_size == 1

    def test_trained_false_on_init(self, small_lstm):
        assert small_lstm.trained is False

    def test_sequence_length_none_on_init(self, small_lstm):
        assert small_lstm.sequence_length is None

    def test_dropout_none_when_zero(self, small_lstm):
        assert small_lstm.dropout is None

    def test_dropout_module_when_nonzero(self):
        m = LSTM(lstm_num_layers=2, lstm_hidden_size=16, input_size=8, dropout=0.3)
        assert isinstance(m.dropout, torch.nn.Dropout)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestLSTMForward:
    def test_output_shape(self, small_lstm, random_batch):
        out = small_lstm(random_batch)
        assert out.shape == (4, 1)

    def test_output_is_float32(self, small_lstm, random_batch):
        out = small_lstm(random_batch)
        assert out.dtype == torch.float32

    def test_different_batch_sizes(self, small_lstm):
        for bs in [1, 8, 32]:
            x = torch.rand(bs, 5, 8)
            out = small_lstm(x)
            assert out.shape == (bs, 1)


# ---------------------------------------------------------------------------
# Predict (no training required — checks shape only)
# ---------------------------------------------------------------------------

class TestLSTMPredict:
    def test_predict_output_shape(self, small_lstm, random_train_data):
        X, _ = random_train_data
        preds = small_lstm.predict(X, sequence_length=5)
        assert preds.shape[0] == X.shape[0]
        assert preds.shape[1] == 1

    def test_predict_accepts_numpy(self, small_lstm, random_train_data):
        X, _ = random_train_data
        preds = small_lstm.predict(X.numpy(), sequence_length=5)
        assert preds.shape[0] == X.shape[0]

    def test_predict_returns_tensor(self, small_lstm, random_train_data):
        X, _ = random_train_data
        preds = small_lstm.predict(X, sequence_length=5)
        assert isinstance(preds, torch.Tensor)


# ---------------------------------------------------------------------------
# Save / Load round-trip (manually set trained=True to avoid running fit)
# ---------------------------------------------------------------------------

class TestLSTMSaveLoad:
    def test_save_raises_if_not_trained(self, small_lstm, tmp_path):
        with pytest.raises(ValueError, match="Train"):
            small_lstm.save(str(tmp_path / "model.pth"))

    def test_save_creates_files(self, small_lstm, tmp_path):
        small_lstm.trained = True
        small_lstm.sequence_length = 5
        small_lstm.best_loss = 0.1
        small_lstm.epochs_trained = 1
        path = str(tmp_path / "model.pth")
        small_lstm.save(path)
        assert os.path.isfile(path)
        meta = path.replace(".pth", "_metadata.json")
        assert os.path.isfile(meta)

    def test_load_restores_architecture(self, small_lstm, tmp_path):
        small_lstm.trained = True
        small_lstm.sequence_length = 5
        small_lstm.best_loss = 0.1
        small_lstm.epochs_trained = 1
        path = str(tmp_path / "model.pth")
        small_lstm.save(path)

        loaded = LSTM.load(path)
        assert loaded.lstm_num_layers == small_lstm.lstm_num_layers
        assert loaded.lstm_num_hidden == small_lstm.lstm_num_hidden
        assert loaded.input_size == small_lstm.input_size

    def test_load_predictions_match(self, small_lstm, tmp_path, random_train_data):
        X, _ = random_train_data
        small_lstm.trained = True
        small_lstm.sequence_length = 5
        small_lstm.best_loss = 0.1
        small_lstm.epochs_trained = 1
        path = str(tmp_path / "model.pth")
        small_lstm.save(path)

        loaded = LSTM.load(path)
        with torch.no_grad():
            orig = small_lstm.predict(X, sequence_length=5)
            reloaded = loaded.predict(X, sequence_length=5)
        assert torch.allclose(orig, reloaded, atol=1e-5)

    def test_load_missing_metadata_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LSTM.load(str(tmp_path / "nonexistent.pth"))

    def test_metadata_json_content(self, small_lstm, tmp_path):
        small_lstm.trained = True
        small_lstm.sequence_length = 5
        small_lstm.best_loss = 0.1
        small_lstm.epochs_trained = 1
        path = str(tmp_path / "model.pth")
        small_lstm.save(path)
        meta_path = path.replace(".pth", "_metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["model_type"] == "LSTM"
        assert meta["architecture"]["lstm_num_layers"] == 1
        assert meta["architecture"]["input_size"] == 8
