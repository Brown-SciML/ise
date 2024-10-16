import numpy as np
import torch
import xarray as xr
from scipy.spatial.distance import jensenshannon
from scipy.stats import kstest, ttest_ind
from sklearn.metrics import r2_score as r2
import properscoring as ps


def sum_by_sector(array, grid_file):
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach().numpy()
    if isinstance(grid_file, str):
        grids = xr.open_dataset(grid_file)
        sector_name = "sectors" if "ais" in grid_file.lower() else "ID"
    elif isinstance(grid_file, xr.Dataset):
        sector_name = "ID" if "Rignot" in grids.Description else "sectors"
    else:
        raise ValueError("grid_file must be a string or an xarray Dataset.")

    if len(array.shape) == 3:
        num_timesteps = array.shape[0]
    elif len(array.shape) == 2:
        num_timesteps = 1
        array = array.reshape((1, array.shape[0], array.shape[1]))

    # if len(array.shape) == 3:
    #     grids = grids.expand_dims(dim={'time': num_timesteps})
    sectors = grids[sector_name].values

    ice_sheet = "AIS" if 761 in array.shape else "GIS"
    num_sectors = 18 if ice_sheet == "AIS" else 6

    sums_by_sector = np.zeros((num_timesteps, num_sectors))
    for i in range(array.shape[0]):
        for sector in range(1, num_sectors + 1):
            sums_by_sector[i, sector - 1] = np.sum(array[i, :, :][sectors == sector])
    return sums_by_sector

def r2_score(y_true, y_pred):
    return r2(y_true, y_pred)
    

def mean_squared_error_sector(sum_sectors_true, sum_sectors_pred):
    return np.mean((sum_sectors_true - sum_sectors_pred) ** 2)


def kl_divergence(p: np.ndarray, q: np.ndarray):
    """Calculates the Kullback-Leibler Divergence between two distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize p and q to ensure they are probability distributions
    p /= np.sum(p)
    q /= np.sum(q)
    
    # Clip values to avoid numerical instability
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    
    # Compute KL divergence
    return np.sum(p * np.log(p / q))



def js_divergence(p: np.ndarray, q: np.ndarray):
    """Calculates the Jensen-Shannon Divergence between two distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize p and q to ensure they are probability distributions
    p /= np.sum(p)
    q /= np.sum(q)
    
    # Clip values to avoid numerical instability
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    
    # Calculate the Jensen-Shannon Divergence
    jsd = jensenshannon(p, q) ** 2  # The function returns the square root, so square it for the divergence
    
    return jsd

def crps(y_true, y_pred, y_std):
    return ps.crps_gaussian(y_true, mu=y_pred, sig=y_std)

def mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
    - y_true: numpy array or a list of actual numbers
    - y_pred: numpy array or a list of predicted numbers

    Returns:
    - mape: Mean Absolute Percentage Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return np.inf
    mape = (
        np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))
        * 100
    )
    return mape


def relative_squared_error(y_true, y_pred):
    """
    Calculate Relative Squared Error (RSE).

    Args:
    - y_true: numpy array or a list of actual numbers
    - y_pred: numpy array or a list of predicted numbers

    Returns:
    - rse: Relative Squared Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    rse = ss_res / ss_tot
    return rse


def kolmogorov_smirnov(x1, x2):
    res = kstest(x1, x2)
    return res.statistic, res.pvalue


def t_test(x1, x2):
    res = ttest_ind(x1, x2)
    return res.statistic, res.pvalue


def calculate_ece(predictions, uncertainties, true_values, bins=10):
    """
    Calculate the Expected Calibration Error (ECE) for regression model predictions.
    
    Args:
    predictions (numpy.ndarray): Array of predicted means by the model.
    uncertainties (numpy.ndarray): Array of predicted standard deviations (uncertainty estimates).
    true_values (numpy.ndarray): Array of actual values.
    bins (int): Number of bins to use for grouping predictions by their uncertainty.

    Returns:
    float: The Expected Calibration Error.
    """
    bin_limits = np.linspace(np.min(uncertainties), np.max(uncertainties), bins+1)
    ece = 0.0
    total_count = len(predictions)

    for i in range(bins):
        bin_mask = (uncertainties >= bin_limits[i]) & (uncertainties < bin_limits[i+1])
        if np.sum(bin_mask) == 0:
            continue
        bin_predictions = predictions[bin_mask]
        bin_uncertainties = uncertainties[bin_mask]
        bin_true_values = true_values[bin_mask]

        # Assume Gaussian distribution: about 95.4% of data should fall within ±2 standard deviations
        lower_bounds = bin_predictions - 2 * bin_uncertainties
        upper_bounds = bin_predictions + 2 * bin_uncertainties
        in_range = (bin_true_values >= lower_bounds) & (bin_true_values <= upper_bounds)
        observed_probability = np.mean(in_range)
        expected_probability = 0.954  # For ±2 standard deviations in Gaussian distribution

        # Calculate the absolute difference weighted by the number of elements in the bin
        ece += np.abs(observed_probability - expected_probability) * np.sum(bin_mask) / total_count

    return ece

def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE).

    Args:
    - y_true: numpy array or a list of actual numbers
    - y_pred: numpy array or a list of predicted numbers

    Returns:
    - mse: Mean Squared Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).

    Args:
    - y_true: numpy array or a list of actual numbers
    - y_pred: numpy array or a list of predicted numbers

    Returns:
    - mae: Mean Absolute Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae