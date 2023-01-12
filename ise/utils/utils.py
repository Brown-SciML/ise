"""Utility functions for handling various parts of the package, including argument checking and
formatting and file traversal."""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

np.random.seed(10)


file_dir = os.path.dirname(os.path.realpath(__file__))


def check_input(input: str, options: List[str], argname: str = None):
    """Checks validity of input argument. Not used frequently due to error raising being better practice.

    Args:
        input (str): Input value.
        options (List[str]): Valid options for the input value.
        argname (str, optional): Name of the argument being tested. Defaults to None.
    """
    # simple assert that input is in the designated options (readability purposes only)
    if isinstance(input, str):
        input = input.lower()
    if input not in options:
        if argname is not None:
            raise ValueError(f"{argname} must be in {options}, received {input}")
        raise ValueError(f"input must be in {options}, received {input}")


def get_all_filepaths(path: str, filetype: str = None, contains: str = None):
    """Retrieves all filepaths for files within a directory. Supports subsetting based on filetype
    and substring search.

    Args:
        path (str): Path to directory to be searched.
        filetype (str, optional): File type to be returned (e.g. csv, nc). Defaults to None.
        contains (str, optional): Substring that files found must contain. Defaults to None.

    Returns:
        List[str]: list of files within the directory matching the input criteria.
    """
    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        all_files += [os.path.join(dirpath, file) for file in filenames]

    if filetype:
        if filetype.lower() != "all":
            all_files = [file for file in all_files if file.endswith(filetype)]

    if contains:
        all_files = [file for file in all_files if contains in file]

    return all_files


def _structure_emulatordata_args(input_args: dict, time_series: bool):
    """Formats kwargs for EmulatorData processing. Includes establishing defaults if values are not
    supplied.

    Args:
        input_args (dict): Dictionary containin kwargs for EmulatorData.process()
        time_series (bool): Flag denoting whether the processing is time-series.

    Returns:
        dict: EmulatorData.process() kwargs formatted with defaults.
    """
    emulator_data_defaults = dict(
        target_column="sle",
        drop_missing=True,
        drop_columns=["groupname", "experiment"],
        boolean_indices=True,
        scale=True,
        split_type="batch",
        drop_outliers="explicit",
        drop_expression=[("ivaf", "<", -1e13)],
        time_series=time_series,
        lag=None,
    )

    if time_series:
        emulator_data_defaults["lag"] = 5

    # If no other args are supplied, use defaults
    if input_args is None:
        return emulator_data_defaults
    # else, replace provided key value pairs in the default dict and reassign
    else:
        for key in input_args.keys():
            emulator_data_defaults[key] = input_args[key]
        output_args = emulator_data_defaults

    return output_args


def _structure_architecture_args(architecture, time_series):
    """Formats the arguments for model architectures.

    Args:
        architecture (dict): User input for desired architecture.
        time_series (bool): Flag denoting whether to use time series model arguments or traditional.

    Returns:
        architecture (dict): Formatted architecture argument.
    """

    # Check to make sure inappropriate args are not used
    if not time_series and (
        "num_rnn_layers" in architecture.keys()
        or "num_rnn_hidden" in architecture.keys()
    ):
        raise AttributeError(
            f"Time series architecture args must be in [num_linear_layers, nodes], received {architecture}"
        )
    if time_series and (
        "nodes" in architecture.keys() or "num_linear_layers" in architecture.keys()
    ):
        raise AttributeError(
            f"Time series architecture args must be in [num_rnn_layers, num_rnn_hidden], received {architecture}"
        )

    if architecture is None:
        if time_series:
            architecture = {
                "num_rnn_layers": 3,
                "num_rnn_hidden": 128,
            }
        else:
            architecture = {
                "num_linear_layers": 4,
                "nodes": [128, 64, 32, 1],
            }
    else:
        return architecture
    return architecture
