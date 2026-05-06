"""Utilities for I/O, data loading, and package paths.

This package provides functions for loading/saving data,
path helpers (e.g. ismip6_model_configs_path), and tensor/data transformations.
"""

import os

from .functions import get_data, get_X_y, get_device, unscale_output

__all__ = [
    "get_data",
    "get_X_y",
    "get_device",
    "unscale_output",
    "ismip6_model_configs_path",
]

ismip6_model_configs_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "data_files", "ismip6_model_configs.json"
)
