import pytest
import torch
import numpy as np
import os
from ise.utils.functions import to_tensor
from ise.data.scaler import StandardScaler, RobustScaler, LogScaler

### ---------------------- Fixtures for Sample Data ---------------------- ###
@pytest.fixture
def sample_tensor():
    """Generates a sample tensor for testing"""
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

@pytest.fixture
def standard_scaler():
    """Creates an instance of StandardScaler"""
    return StandardScaler()

@pytest.fixture
def robust_scaler():
    """Creates an instance of RobustScaler"""
    return RobustScaler()

@pytest.fixture
def log_scaler():
    """Creates an instance of LogScaler"""
    return LogScaler()

### ---------------------- StandardScaler Tests ---------------------- ###
def test_standard_scaler_fit_transform(sample_tensor, standard_scaler):
    """Test StandardScaler fitting and transformation"""
    scaler = standard_scaler
    scaler.fit(sample_tensor)

    assert scaler.mean_ is not None
    assert scaler.scale_ is not None

    transformed = scaler.transform(sample_tensor)
    assert transformed.shape == sample_tensor.shape

    # Ensure mean is approximately 0 after transformation
    assert torch.allclose(torch.mean(transformed, dim=0), torch.zeros_like(scaler.mean_), atol=1e-5)

def test_standard_scaler_inverse_transform(sample_tensor, standard_scaler):
    """Ensure inverse transformation recovers the original data"""
    scaler = standard_scaler
    scaler.fit(sample_tensor)
    transformed = scaler.transform(sample_tensor)
    recovered = scaler.inverse_transform(transformed)

    assert torch.allclose(recovered, sample_tensor, atol=1e-5)

def test_standard_scaler_transform_before_fit(standard_scaler):
    """Ensure transform raises an error if fit() is not called first"""
    scaler = standard_scaler
    with pytest.raises(RuntimeError):
        scaler.transform(torch.tensor([[1.0, 2.0, 3.0]]))

def test_standard_scaler_save_load(tmp_path, standard_scaler, sample_tensor):
    """Ensure StandardScaler saves and loads correctly"""
    scaler = standard_scaler
    scaler.fit(sample_tensor)

    save_path = tmp_path / "standard_scaler.pth"
    scaler.save(save_path)

    loaded_scaler = StandardScaler.load(save_path)

    assert torch.allclose(scaler.mean_, loaded_scaler.mean_)
    assert torch.allclose(scaler.scale_, loaded_scaler.scale_)

### ---------------------- RobustScaler Tests ---------------------- ###
def test_robust_scaler_fit_transform(sample_tensor, robust_scaler):
    """Test RobustScaler fitting and transformation"""
    scaler = robust_scaler
    scaler.fit(sample_tensor)

    assert scaler.median_ is not None
    assert scaler.iqr_ is not None

    transformed = scaler.transform(sample_tensor)
    assert transformed.shape == sample_tensor.shape

    # Median should be approximately 0 after transformation
    assert torch.allclose(torch.median(transformed, dim=0).values, torch.zeros_like(scaler.median_), atol=1e-5)

def test_robust_scaler_inverse_transform(sample_tensor, robust_scaler):
    """Ensure inverse transformation recovers the original data"""
    scaler = robust_scaler
    scaler.fit(sample_tensor)
    transformed = scaler.transform(sample_tensor)
    recovered = scaler.inverse_transform(transformed)

    assert torch.allclose(recovered, sample_tensor, atol=1e-5)

def test_robust_scaler_transform_before_fit(robust_scaler):
    """Ensure transform raises an error if fit() is not called first"""
    scaler = robust_scaler
    with pytest.raises(RuntimeError):
        scaler.transform(torch.tensor([[1.0, 2.0, 3.0]]))

def test_robust_scaler_save_load(tmp_path, robust_scaler, sample_tensor):
    """Ensure RobustScaler saves and loads correctly"""
    scaler = robust_scaler
    scaler.fit(sample_tensor)

    save_path = tmp_path / "robust_scaler.pth"
    scaler.save(save_path)

    loaded_scaler = RobustScaler.load(save_path)

    assert torch.allclose(scaler.median_, loaded_scaler.median_)
    assert torch.allclose(scaler.iqr_, loaded_scaler.iqr_)

### ---------------------- LogScaler Tests ---------------------- ###
def test_log_scaler_fit_transform(sample_tensor, log_scaler):
    """Test LogScaler fitting and transformation"""
    scaler = log_scaler
    scaler.fit(sample_tensor)

    assert scaler.min_value is not None

    transformed = scaler.transform(sample_tensor)
    assert transformed.shape == sample_tensor.shape

    # Log values should be increasing with input values
    assert torch.all(transformed[1, :] > transformed[0, :])

def test_log_scaler_inverse_transform(sample_tensor, log_scaler):
    """Ensure inverse transformation recovers the original data"""
    scaler = log_scaler
    scaler.fit(sample_tensor)
    transformed = scaler.transform(sample_tensor)
    recovered = scaler.inverse_transform(transformed)

    assert torch.allclose(recovered, sample_tensor, atol=1e-5)

def test_log_scaler_transform_with_zero(log_scaler):
    """Ensure LogScaler handles zero values correctly"""
    zero_tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    scaler = log_scaler
    scaler.fit(zero_tensor)
    
    transformed = scaler.transform(zero_tensor)
    assert not torch.isnan(transformed).any()

def test_log_scaler_save_load(tmp_path, log_scaler, sample_tensor):
    """Ensure LogScaler saves and loads correctly"""
    scaler = log_scaler
    scaler.fit(sample_tensor)

    save_path = tmp_path / "log_scaler.pth"
    scaler.save(save_path)

    loaded_scaler = LogScaler.load(save_path)

    assert scaler.epsilon == loaded_scaler.epsilon
    assert scaler.min_value == loaded_scaler.min_value
