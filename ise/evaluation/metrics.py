import numpy as np
import torch
import xarray as xr
from scipy.spatial.distance import jensenshannon
from scipy.stats import kstest, ttest_ind
from sklearn.metrics import r2_score as r2
import properscoring as ps


def sum_by_sector(array, grid_file):
    """
    Computes the sum of values in a given array by predefined sectors using a grid file.

    Args:
        array (numpy.ndarray or torch.Tensor): A 2D or 3D array containing values to be summed by sector.
        grid_file (str or xarray.Dataset): Path to the grid file defining sector boundaries or an xarray dataset.

    Returns:
        numpy.ndarray: A 2D array where each row represents a timestep and each column represents a sector.

    Raises:
        ValueError: If grid_file is not a valid string or xarray dataset.
    """

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
    """
    Computes the coefficient of determination (R² score).

    Args:
        y_true (numpy.ndarray or list): The true values.
        y_pred (numpy.ndarray or list): The predicted values.

    Returns:
        float: The R² score, where 1 indicates perfect predictions.
    """

    return r2(y_true, y_pred)
    

def mean_squared_error_sector(sum_sectors_true, sum_sectors_pred):
    """
    Computes the mean squared error (MSE) between true and predicted sector-wise sums.

    Args:
        sum_sectors_true (numpy.ndarray): The true summed sector values.
        sum_sectors_pred (numpy.ndarray): The predicted summed sector values.

    Returns:
        float: The mean squared error (MSE).
    """

    return np.mean((sum_sectors_true - sum_sectors_pred) ** 2)


def kl_divergence(p: np.ndarray, q: np.ndarray):
    """
    Computes the Kullback-Leibler (KL) Divergence between two probability distributions.

    Args:
        p (numpy.ndarray): The first probability distribution.
        q (numpy.ndarray): The second probability distribution.

    Returns:
        float: The KL divergence value.

    Notes:
        - The distributions p and q must be normalized (i.e., sum to 1).
        - Small epsilon values are used to avoid numerical instability.
    """

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
    """
    Computes the Jensen-Shannon Divergence (JSD) between two probability distributions.

    Args:
        p (numpy.ndarray): The first probability distribution.
        q (numpy.ndarray): The second probability distribution.

    Returns:
        float: The Jensen-Shannon divergence value.

    Notes:
        - JSD is a smoothed and symmetric version of KL divergence.
        - The function normalizes the distributions before computation.
    """

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
    """
    Computes the Continuous Ranked Probability Score (CRPS) for a Gaussian distribution.

    Args:
        y_true (numpy.ndarray): The true values.
        y_pred (numpy.ndarray): The predicted mean values.
        y_std (numpy.ndarray): The predicted standard deviations.

    Returns:
        numpy.ndarray: The computed CRPS values for each prediction.
    """

    return ps.crps_gaussian(y_true, mu=y_pred, sig=y_std)

def mape(y_true, y_pred):
    """
    Computes the Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (numpy.ndarray or list): The true values.
        y_pred (numpy.ndarray or list): The predicted values.

    Returns:
        float: The MAPE value, expressed as a percentage.

    Notes:
        - MAPE ignores zero values in y_true to prevent division by zero.
        - If all true values are zero, returns infinity.
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
    Computes the Relative Squared Error (RSE), measuring the error relative to the variance in y_true.

    Args:
        y_true (numpy.ndarray or list): The true values.
        y_pred (numpy.ndarray or list): The predicted values.

    Returns:
        float: The computed RSE value.

    Notes:
        - A lower RSE indicates better performance, with RSE=0 indicating perfect predictions.
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    rse = ss_res / ss_tot
    return rse


def kolmogorov_smirnov(x1, x2):
    """
    Computes the Kolmogorov-Smirnov (KS) statistic to compare two distributions.

    Args:
        x1 (numpy.ndarray or list): The first dataset.
        x2 (numpy.ndarray or list): The second dataset.

    Returns:
        tuple: (KS statistic, p-value).
    """

    res = kstest(x1, x2)
    return res.statistic, res.pvalue


def t_test(x1, x2):
    """
    Performs an independent two-sample t-test to compare the means of two distributions.

    Args:
        x1 (numpy.ndarray or list): The first dataset.
        x2 (numpy.ndarray or list): The second dataset.

    Returns:
        tuple: (t-statistic, p-value).
    """

    res = ttest_ind(x1, x2)
    return res.statistic, res.pvalue


def calculate_ece(predictions, uncertainties, true_values, bins=10):
    """
    Computes the Expected Calibration Error (ECE) for a regression model.

    Args:
        predictions (numpy.ndarray): The predicted mean values.
        uncertainties (numpy.ndarray): The predicted standard deviations.
        true_values (numpy.ndarray): The true values.
        bins (int, optional): The number of bins for uncertainty grouping. Defaults to 10.

    Returns:
        float: The Expected Calibration Error (ECE).

    Notes:
        - ECE measures how well predicted uncertainties align with actual errors.
        - A lower ECE indicates better-calibrated uncertainty estimates.
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
    Computes the Mean Squared Error (MSE).

    Args:
        y_true (numpy.ndarray or list): The true values.
        y_pred (numpy.ndarray or list): The predicted values.

    Returns:
        float: The Mean Squared Error (MSE).
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def mean_absolute_error(y_true, y_pred):
    """
    Computes the Mean Absolute Error (MAE).

    Args:
        y_true (numpy.ndarray or list): The true values.
        y_pred (numpy.ndarray or list): The predicted values.

    Returns:
        float: The Mean Absolute Error (MAE).
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae