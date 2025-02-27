import pytest
import numpy as np
import torch
import xarray as xr
from ise.evaluation.metrics import (
    sum_by_sector, r2_score, mean_squared_error_sector, kl_divergence, js_divergence,
    crps, mape, relative_squared_error, kolmogorov_smirnov, t_test, calculate_ece,
    mean_squared_error, mean_absolute_error
)
from scipy.spatial.distance import jensenshannon
from scipy.stats import kstest, ttest_ind

### ---------------------- Fixtures for Sample Data ---------------------- ###
@pytest.fixture
def sample_grid():
    """Creates a mock grid file as an xarray dataset with known sector values."""
    grid_data = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4]
    ])
    return xr.Dataset({"sectors": (["x", "y"], grid_data)})

@pytest.fixture
def sample_array():
    """Creates a small 2D array for sum_by_sector testing."""
    return np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

### ---------------------- sum_by_sector Tests ---------------------- ###
# def test_sum_by_sector(sample_array, sample_grid):
#     """Test sum_by_sector function with known input and expected output."""
#     expected_result = np.array([[14, 22, 46, 54]])  # Precomputed sum per sector
#     result = sum_by_sector(sample_array, sample_grid)
#     assert np.array_equal(result, expected_result)

### ---------------------- Error Metric Tests ---------------------- ###
def test_r2_score():
    """Test R² score with fixed input values."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])  # Perfect predictions
    assert r2_score(y_true, y_pred) == 1.0  # R² should be exactly 1.0

def test_mean_squared_error():
    """Test MSE with a fixed calculation."""
    y_true = np.array([2, 4, 6, 8])
    y_pred = np.array([1, 3, 5, 7])  # Each value is off by 1
    expected_mse = np.mean((y_true - y_pred) ** 2)  # (1² + 1² + 1² + 1²) / 4 = 1.0
    assert mean_squared_error(y_true, y_pred) == expected_mse

def test_mean_absolute_error():
    """Test MAE with fixed values."""
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 25])
    expected_mae = np.mean(np.abs(y_true - y_pred))  # (|10-12| + |20-18| + |30-25|) / 3 = 3.00
    print(mean_absolute_error(y_true, y_pred))
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(3.00, rel=1e-2)

def test_mape():
    """Test MAPE with a simple case."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([90, 210, 310])
    expected_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # 5%
    assert mape(y_true, y_pred) == expected_mape

### ---------------------- Divergence & Statistical Tests ---------------------- ###
def test_kl_divergence():
    """Test KL divergence with known probability distributions."""
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.1, 0.4, 0.5])
    expected_kl = np.sum(p * np.log(p / q))  # Manually computed
    assert kl_divergence(p, q) == pytest.approx(expected_kl, rel=1e-5)

def test_js_divergence():
    """Test JS divergence with a fixed probability distribution."""
    p = np.array([0.5, 0.5])
    q = np.array([0.9, 0.1])
    expected_js = jensenshannon(p, q) ** 2  # Jensen-Shannon returns sqrt(divergence), so square it
    assert js_divergence(p, q) == pytest.approx(expected_js, rel=1e-5)

### ---------------------- Distribution Tests ---------------------- ###
def test_kolmogorov_smirnov():
    """Test Kolmogorov-Smirnov statistic with known data."""
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = np.array([1, 2, 3, 4, 5])  # Identical distributions
    stat, p_value = kolmogorov_smirnov(x1, x2)
    assert stat == 0.0  # No difference in distributions
    assert p_value == 1.0  # p-value should be 1

def test_t_test():
    """Test t-test statistic with simple means."""
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = np.array([2, 3, 4, 5, 6])
    expected_t, expected_p = ttest_ind(x1, x2)
    stat, p_value = t_test(x1, x2)
    assert stat == pytest.approx(expected_t, rel=1e-5)
    assert p_value == pytest.approx(expected_p, rel=1e-5)

### ---------------------- Expected Calibration Error Tests ---------------------- ###
def test_calculate_ece():
    """Test Expected Calibration Error (ECE) with a known distribution."""
    predictions = np.array([1.0, 2.0, 3.0, 4.0])
    uncertainties = np.array([0.1, 0.2, 0.3, 0.4])
    true_values = np.array([1.1, 1.9, 3.2, 4.1])
    
    expected_ece = 0.0345  # Precomputed based on sample values
    assert calculate_ece(predictions, uncertainties, true_values, bins=5) == pytest.approx(expected_ece, rel=1e-2)

### ---------------------- Edge Case Tests ---------------------- ###
def test_mape_with_zeros():
    """Ensure MAPE correctly handles zero values."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    assert mape(y_true, y_pred) == np.inf  # Should return infinity

# def test_kl_divergence_invalid_input():
#     """Ensure KL divergence raises an error for invalid probability distributions."""
#     with pytest.raises(ValueError):
#         kl_divergence(np.array(["0", 0.5]), np.array([0.0, 0.0]))  # q must sum to 1
