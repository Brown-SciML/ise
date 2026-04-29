"""Time conversion and subsetting utilities for xarray datasets.

This module provides convert_and_subset_times for standardizing time coordinates
to a 2015–2100 range and handling various time formats (cftime, numeric, datetime64).
"""

import warnings
from datetime import datetime

import cftime
import numpy as np
import xarray as xr


def convert_and_subset_times(dataset):
    """
    Converts time variables in an xarray dataset to a uniform format and subsets time to the range 2015-2100.

    Args:
        dataset (xarray.Dataset): The dataset with time values to be converted and subset.

    Returns:
        xarray.Dataset: The dataset with standardized time format and subset to the correct time range.

    Raises:
        ValueError: If time values are not in a recognizable format.
    """

    # This part of your code remains the same
    if "time" not in dataset and "year" in dataset:
        dataset = dataset.rename({"year": "time"})
        dataset["time"] = np.array(
            [np.datetime64(f"{int(x)}-01-01") for x in dataset.time.values], dtype="datetime64[ns]"
        )

        # Check if the length is less than 86 and needs padding
        if len(dataset.time) < 86:
            warnings.warn(
                f"Dataset has {len(dataset.time)} time points. Padding to 86 by repeating the last value."
            )

            # 1. Determine the first year to create a full 86-year index
            start_year = dataset.time.dt.year.min().item()

            # 2. Create the target 86-year time index
            target_years = np.arange(start_year, start_year + 86)
            target_time_index = [np.datetime64(f"{year}-01-01", "ns") for year in target_years]

            # 3. Reindex the dataset to the new time coordinate, filling missing values
            #    by carrying forward the last available data point ('ffill').
            dataset = dataset.reindex(time=target_time_index, method="ffill")

        # Optional: You can also warn if the dataset is unexpectedly long
        elif len(dataset.time) > 86:
            warnings.warn(
                f"Dataset has {len(dataset.time)} time points, which is more than the expected 86."
            )

    if isinstance(dataset.time.values[0], cftime._cftime.DatetimeNoLeap) or isinstance(
        dataset.time.values[0], cftime._cftime.Datetime360Day
    ):
        datetimeindex = dataset.indexes["time"].to_datetimeindex()
        dataset["time"] = datetimeindex

    elif (
        isinstance(dataset.time.values[0], np.float32)
        or isinstance(dataset.time.values[0], np.float64)
        or isinstance(dataset.time.values[0], np.int32)
        or isinstance(dataset.time.values[0], np.int64)
    ):
        try:
            units = dataset.time.attrs["units"]
        except KeyError:
            units = dataset.time.attrs["unit"]
        units = units.replace("days since ", "").split(" ")[0]

        if units == "2000-1-0":  # VUB AISMPALEO
            units = "2000-1-1"
        elif units == "day":  # NCAR CISM exp7 - "day as %Y%m%d.%f"?
            units = "2014-1-1"

        if units == "seconds":  # VUW PISM -- seconds since 1-1-1 00:00:00
            start_date = np.datetime64(
                datetime.strptime("0001-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
            )
            dataset["time"] = np.array(
                [start_date + np.timedelta64(int(x), "s") for x in dataset.time.values]
            )
        elif units == "2008-1-1" and dataset.time[-1] == 157785.0:  # UAF?
            # every 5 years but still len(time) == 86.. assume we keep them all for 2015-2100
            dataset["time"] = np.array(
                [
                    np.datetime64(datetime.strptime(f"{x}-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"))
                    for x in range(2015, 2101)
                ]
            )
        else:
            try:
                start_date = np.datetime64(
                    datetime.strptime(units.replace("days since ", ""), "%Y-%m-%d")
                )
            except ValueError:
                start_date = np.datetime64(
                    datetime.strptime(units.replace("days since ", ""), "%d-%m-%Y")
                )

            time_values = np.array(
                [start_date + np.timedelta64(int(x), "D") for x in dataset.time.values],
                dtype="datetime64[ns]",
            )
            dataset["time"] = time_values
    elif isinstance(dataset.time.values[0], np.datetime64):
        pass
    else:
        raise ValueError(f"Time values are not recognized: {type(dataset.time.values[0])}")

    if len(dataset.time) > 86:
        # make sure the max date is 2100
        # dataset = dataset.sel(time=slice(np.datetime64('2014-01-01'), np.datetime64('2101-01-01')))
        dataset = dataset.sortby("time")
        dataset = dataset.sel(time=slice("2012-01-01", "2101-01-01"))

        # if you still have more than 86, take the previous 86 values from 2100
        if len(dataset.time) > 86:
            # LSCE GRISLI has two 2015 measurements

            # dataset = dataset.sel(time=slice(dataset.time.values[len(dataset.time) - 86], dataset.time.values[-1]))
            start_idx = len(dataset.time) - 86
            dataset = dataset.isel(time=slice(start_idx, len(dataset.time)))
        elif len(dataset.time) < 86:
            warnings.warn(
                f"Dataset has {len(dataset.time)} time points after subsetting. Padding to 86 by repeating the last value."
            )

            _, unique_indices = np.unique(dataset["time"], return_index=True)
            dataset = dataset.isel(time=unique_indices)
            # 1. Determine the first year to create a full 86-year index
            start_year = dataset.time.dt.year.min().item()

            # 2. Create the target 86-year time index
            target_years = np.arange(start_year, start_year + 86)
            target_time_index = [np.datetime64(f"{year}-01-01", "ns") for year in target_years]

            # 3. Reindex the dataset to the new time coordinate, filling missing values
            #    by carrying forward the last available data point ('ffill').
            dataset = dataset.reindex(time=target_time_index, method="ffill")

    if len(dataset.time) != 86:
        warnings.warn(
            "After subsetting there are still not 86 time points. Go back and check logs."
        )
        print(f"dataset_length={len(dataset.time)} -- {dataset.attrs}")

    return dataset
