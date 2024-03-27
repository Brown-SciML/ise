import numpy as np
import torch
import xarray as xr
from scipy.spatial.distance import jensenshannon
from scipy.stats import kstest, ttest_ind


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


def mean_squared_error_sector(sum_sectors_true, sum_sectors_pred):
    return np.mean((sum_sectors_true - sum_sectors_pred) ** 2)


def kl_divergence(p: np.ndarray, q: np.ndarray):
    """Calculates the Kullback-Leibler Divergence between two distributions. Q is typically a
    'known' distirubtion and should be the true values, whereas P is typcically the test distribution,
    or the predicted distribution. Note the the KL divergence is assymetric, and near-zero values for
    p with a non-near zero values for q cause the KL divergence to inflate [citation].

    Args:
        p (np.ndarray): Test distribution
        q (np.ndarray): Known distribution

    Returns:
        float: KL Divergence
    """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def js_divergence(p: np.ndarray, q: np.ndarray):
    """Calculates the Jensen-Shannon Divergence between two distributions. Q is typically a
    'known' distirubtion and should be the true values, whereas P is typcically the test distribution,
    or the predicted distribution. Note the the JS divergence, unlike the KL divergence, is symetric.

    Args:
        p (np.ndarray): Test distribution
        q (np.ndarray): Known distribution

    Returns:
        float: JS Divergence
    """
    return jensenshannon(p, q)


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
