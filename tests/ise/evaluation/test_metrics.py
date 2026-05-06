import numpy as np
import pytest
import xarray as xr
from scipy.spatial.distance import jensenshannon
from scipy.stats import ttest_ind

from ise.evaluation.metrics import (
    calculate_ece,
    crps,
    js_divergence,
    kl_divergence,
    kolmogorov_smirnov,
    mape,
    mean_absolute_error,
    mean_prediction_interval_width,
    mean_squared_error,
    r2_score,
    relative_squared_error,
    t_test,
    winkler_score,
)


### ---------------------- Fixtures for Sample Data ---------------------- ###
@pytest.fixture
def sample_grid():
    """Creates a mock grid file as an xarray dataset with known sector values."""
    grid_data = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    return xr.Dataset({"sectors": (["x", "y"], grid_data)})


@pytest.fixture
def sample_array():
    """Creates a small 2D array for sum_by_sector testing."""
    return np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])


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
    assert calculate_ece(predictions, uncertainties, true_values, bins=5) == pytest.approx(
        expected_ece, rel=1e-2
    )


### ---------------------- Edge Case Tests ---------------------- ###
def test_mape_with_zeros():
    """Ensure MAPE correctly handles zero values."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    assert mape(y_true, y_pred) == np.inf  # Should return infinity


### ---------------------- CRPS Tests ---------------------- ###
def test_crps_perfect_low():
    """Perfect mean with tiny std should yield near-zero CRPS."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    y_std = np.full(3, 1e-6)
    scores = crps(y_true, y_pred, y_std)
    assert np.mean(scores) < 0.01


def test_crps_returns_array():
    """crps should return one score per sample."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    y_std = np.array([0.5, 0.5, 0.5])
    scores = crps(y_true, y_pred, y_std)
    assert scores.shape == y_true.shape


def test_crps_larger_error_gives_higher_score():
    """A larger prediction error with the same std should increase CRPS."""
    y_true = np.array([0.0])
    y_std = np.array([1.0])
    score_close = np.mean(crps(y_true, np.array([0.1]), y_std))
    score_far = np.mean(crps(y_true, np.array([5.0]), y_std))
    assert score_far > score_close


### ---------------------- relative_squared_error Tests ---------------------- ###
def test_rse_perfect_predictions():
    """Perfect predictions give RSE = 0."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert relative_squared_error(y, y) == pytest.approx(0.0, abs=1e-9)


def test_rse_mean_baseline():
    """Predicting the mean for every sample gives RSE = 1."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    baseline = np.full_like(y, np.mean(y))
    assert relative_squared_error(y, baseline) == pytest.approx(1.0, rel=1e-9)


def test_rse_positive_for_imperfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 3.5])
    assert relative_squared_error(y_true, y_pred) > 0.0


### ---------------------- mean_prediction_interval_width Tests ---------------------- ###
def test_mpiw_known_value():
    """MPIW = mean(upper - lower)."""
    upper = np.array([3.0, 4.0, 5.0])
    lower = np.array([1.0, 2.0, 3.0])
    assert mean_prediction_interval_width(upper, lower) == pytest.approx(2.0, rel=1e-9)


def test_mpiw_zero_width_intervals():
    x = np.array([1.0, 2.0, 3.0])
    assert mean_prediction_interval_width(x, x) == pytest.approx(0.0, abs=1e-9)


### ---------------------- winkler_score Tests ---------------------- ###
def test_winkler_within_interval_equals_width():
    """True values inside interval: score = width (no violation penalty)."""
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])
    lower = np.array([0.0, 1.0])
    upper = np.array([2.0, 3.0])
    width = np.mean(upper - lower)  # 2.0
    score = winkler_score(y_true, y_pred, lower, upper, alpha=0.05)
    assert score == pytest.approx(width, rel=1e-9)


def test_winkler_violation_increases_score():
    """A true value outside the interval raises the score above the interval width."""
    y_true_in = np.array([1.0])
    y_true_out = np.array([10.0])
    y_pred = np.array([1.0])
    lower = np.array([0.0])
    upper = np.array([2.0])
    score_in = winkler_score(y_true_in, y_pred, lower, upper)
    score_out = winkler_score(y_true_out, y_pred, lower, upper)
    assert score_out > score_in
