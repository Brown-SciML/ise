import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from ise.data.dataclasses import EmulatorDataset, PyTorchDataset, TSDataset, ScenarioDataset

### ---------------------- EmulatorDataset Tests ---------------------- ###
@pytest.fixture
def sample_emulator_data():
    """Fixture to create sample EmulatorDataset data"""
    X = np.random.rand(100, 10)  # 100 time steps, 10 features
    y = np.random.rand(100, 1)   # 100 target values
    return X, y

def test_emulator_tensor_conversion(sample_emulator_data):
    """Check if EmulatorDataset correctly converts data to tensors"""
    X, y = sample_emulator_data
    dataset = EmulatorDataset(X, y)

    assert isinstance(dataset.X, torch.Tensor)
    assert isinstance(dataset.y, torch.Tensor)
    assert dataset.X.dtype == torch.float32
    assert dataset.y.dtype == torch.float32

def test_emulator_dataset_length(sample_emulator_data):
    """Check dataset length calculation"""
    X, y = sample_emulator_data
    dataset = EmulatorDataset(X, y)
    
    assert len(dataset) == X.shape[0]  # Should match the number of rows

def test_emulator_getitem(sample_emulator_data):
    """Check sequence retrieval and target values"""
    X, y = sample_emulator_data
    dataset = EmulatorDataset(X, y, sequence_length=5)

    x_seq, y_target = dataset[10]  # Get the 10th item
    
    assert x_seq.shape == (5, X.shape[1])  # Sequence length x Features
    assert y_target.shape == (1, )  # Target should be a single value

def test_emulator_y_none(sample_emulator_data):
    """Ensure EmulatorDataset works when y is None"""
    X, _ = sample_emulator_data  # Only provide X
    dataset = EmulatorDataset(X, y=None)

    x_seq = dataset[10]  # Should return only X sequence
    assert isinstance(x_seq, torch.Tensor)
    assert x_seq.shape == (5, X.shape[1])

### ---------------------- PyTorchDataset Tests ---------------------- ###
@pytest.fixture
def sample_pytorch_data():
    X = torch.rand(50, 10)  # 50 samples, 10 features
    y = torch.rand(50, 1)
    return X, y

def test_pytorch_dataset_length(sample_pytorch_data):
    X, y = sample_pytorch_data
    dataset = PyTorchDataset(X, y)
    assert len(dataset) == 50  # Should match number of samples

def test_pytorch_getitem(sample_pytorch_data):
    X, y = sample_pytorch_data
    dataset = PyTorchDataset(X, y)

    x_sample, y_sample = dataset[5]  # Get 5th item
    assert x_sample.shape == (10,)
    assert y_sample.shape == (1,)

def test_pytorch_y_none(sample_pytorch_data):
    X, _ = sample_pytorch_data
    dataset = PyTorchDataset(X, y=None)

    x_sample = dataset[5]  # Should return only X
    assert x_sample.shape == (10,)

### ---------------------- TSDataset Tests ---------------------- ###
@pytest.fixture
def sample_ts_data():
    X = torch.rand(30, 5)  # 30 timesteps, 5 features
    y = torch.rand(30, 1)
    return X, y

def test_ts_dataset_length(sample_ts_data):
    X, y = sample_ts_data
    dataset = TSDataset(X, y, sequence_length=5)
    assert len(dataset) == 30  # Should match number of samples

def test_ts_getitem_padding(sample_ts_data):
    X, y = sample_ts_data
    dataset = TSDataset(X, y, sequence_length=5)

    x_seq, y_target = dataset[2]  # Early sequence (requires padding)
    assert x_seq.shape == (5, X.shape[1])  # Sequence length x Features

    # Ensure padding (first rows should match first row in dataset)
    assert torch.equal(x_seq[0], X[0])

def test_ts_getitem_normal(sample_ts_data):
    X, y = sample_ts_data
    dataset = TSDataset(X, y, sequence_length=5)

    x_seq, y_target = dataset[10]  # Middle sequence
    assert x_seq.shape == (5, X.shape[1])  # Sequence length x Features

def test_ts_y_none(sample_ts_data):
    X, _ = sample_ts_data
    dataset = TSDataset(X, y=None, sequence_length=5)

    x_seq = dataset[5]  # Should return only X sequence
    assert isinstance(x_seq, torch.Tensor)

### ---------------------- ScenarioDataset Tests ---------------------- ###
@pytest.fixture
def sample_scenario_data():
    features = torch.rand(40, 6)  # 40 samples, 6 features
    labels = torch.randint(0, 2, (40, 1))  # Binary labels
    return features, labels

def test_scenario_dataset_length(sample_scenario_data):
    features, labels = sample_scenario_data
    dataset = ScenarioDataset(features, labels)
    assert len(dataset) == 40

def test_scenario_getitem(sample_scenario_data):
    features, labels = sample_scenario_data
    dataset = ScenarioDataset(features, labels)

    x_sample, y_sample = dataset[7]
    assert x_sample.shape == (6,)
    assert y_sample.shape == (1,)
