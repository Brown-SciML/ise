import os
import time
import warnings
from datetime import datetime

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from ise.data.scaler import LogScaler, RobustScaler, StandardScaler
from ise.models.grid import PCA
from ise.utils.functions import get_all_filepaths


class GridProcessor:
    pass


class SectorProcessor:
    pass


class ProjectionProcessor:
    """
    A class for processing ice sheet data.

    Attributes:
    - ice_sheet (str): Ice sheet to be processed. Must be 'AIS' or 'GIS'.
    - forcings_directory (str): The path to the directory containing the forcings data.
    - projections_directory (str): The path to the directory containing the projections data.
    - scalefac_path (str): The path to the netCDF file containing scaling factors for each grid cell.
    - densities_path (str): The path to the CSV file containing ice and ocean density (rhow/rhoi) data for each experiment.

    Methods:
    - __init__(self, ice_sheet, forcings_directory, projections_directory, scalefac_path=None, densities_path=None): Initializes the Processor object.
    - process_forcings(self): Processes the forcings data.
    - process_projections(self, output_directory): Processes the projections data.
    - _calculate_ivaf_minus_control(self, data_directory, densities_fp, scalefac_path): Calculates the ice volume above flotation (IVAF) for each file in the given data directory, subtracting out the control projection IVAF if applicable.
    - _calculate_ivaf_single_file(self, directory, densities, scalefac_model, ctrl_proj=False): Calculates the ice volume above flotation (IVAF) for a single file.
    """

    def __init__(
        self,
        ice_sheet,
        forcings_directory,
        projections_directory,
        scalefac_path=None,
        densities_path=None,
    ):
        self.forcings_directory = forcings_directory
        self.projections_directory = projections_directory
        self.densities_path = densities_path
        self.scalefac_path = scalefac_path
        self.ice_sheet = ice_sheet.upper()
        if self.ice_sheet.lower() in ("gris", "gis"):
            self.ice_sheet = "GIS"
        self.resolution = 5 if self.ice_sheet == "GIS" else 8

    def process(
        self,
    ):
        """
        Process the ISMIP6 projections by calculating IVAF for both control
        and experiments, subtracting out the control IVAF from experiments,
        and exporting ivaf files.

        Args:
            output_directory (str): The directory to save the processed projections.

        Raises:
            ValueError: If projections_directory or output_directory is not specified.

        Returns:
            int: 1 indicating successful processing.
        """
        if self.projections_directory is None:
            raise ValueError("Projections path must be specified")

        # if the last ivaf file is missing, assume none of them are and calculate and export all ivaf files
        if (
            self.ice_sheet == "AIS"
        ):  # and not os.path.exists(f"{self.projections_directory}/VUW/PISM/exp08/ivaf_GIS_VUW_PISM_exp08.nc"):
            self._calculate_ivaf_minus_control(
                self.projections_directory, self.densities_path, self.scalefac_path
            )
        elif (
            self.ice_sheet == "GIS"
        ):  # and not os.path.exists(f"{self.projections_directory}/VUW/PISM/exp04/ivaf_AIS_VUW_PISM_exp04.nc"):
            self._calculate_ivaf_minus_control(
                self.projections_directory, self.densities_path, self.scalefac_path
            )

        return 1

    def _calculate_ivaf_minus_control(
        self, data_directory: str, densities_fp: str, scalefac_path: str
    ):
        """
        Calculates the ice volume above flotation (IVAF) for each file in the given data directory,
        subtracting out the control projection IVAF if applicable.

        Args:
        - data_directory (str): path to directory containing the data files to process
        - densities_fp (str or pd.DataFrame): filepath to CSV file containing density data, or a pandas DataFrame
        - scalefac_path (str): path to netCDF file containing scaling factors for each grid cell

        Returns:
        - int: 1 indicating successful calculation.

        Raises:
        - ValueError: if densities_fp is None or not a string or pandas DataFrame

        """

        # error handling for densities argument (must be str filepath or dataframe)
        if densities_fp is None:
            raise ValueError(
                "densities_fp must be specified. Run get_model_densities() to get density data."
            )
        if isinstance(densities_fp, str):
            densities = pd.read_csv(densities_fp)
        elif isinstance(densities_fp, pd.DataFrame):
            pass
        else:
            raise ValueError("densities argument must be a string or a pandas DataFrame.")

        # open scaling model
        scalefac_model = xr.open_dataset(scalefac_path)
        scalefac_model = np.transpose(scalefac_model.af2.values, (1, 0))

        # adjust scaling model based on desired resolution
        if self.ice_sheet == "AIS":
            scalefac_model = scalefac_model[:: self.resolution, :: self.resolution]
        elif self.ice_sheet == "GIS" and scalefac_model.shape != (337, 577):
            if scalefac_model.shape[0] == 6081:
                raise ValueError(
                    f"Scalefac model must be 337x577 for GIS, received {scalefac_model.shape}. Make sure you are using the GIS scaling model and not the AIS."
                )
            raise ValueError(
                f"Scalefac model must be 337x577 for GIS, received {scalefac_model.shape}."
            )

        # get all files in directory with "ctrl_proj" and "exp" in them and store separately
        ctrl_proj_dirs = []
        exp_dirs = []
        for root, dirs, _ in os.walk(data_directory):
            for directory in dirs:
                if "ctrl_proj" in directory:
                    ctrl_proj_dirs.append(os.path.join(root, directory))
                elif "exp" in directory:
                    exp_dirs.append(os.path.join(root, directory))
                else:
                    pass

        # first calculate ivaf for control projections
        for directory in ctrl_proj_dirs:
            self._calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=True)

        # then, for each experiment, calculate ivaf and subtract out control
        # exp_dirs = exp_dirs[65:]
        for directory in exp_dirs:
            self._calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=False)

        return 1

    def _calculate_ivaf_single_file(self, directory, densities, scalefac_model, ctrl_proj=False):
        """
        Calculate the Ice Volume Above Floatation (IVAF) for a single file.

        Args:
            directory (str): The directory path of the file.
            densities (pandas.DataFrame): A DataFrame containing density values for different groups and models.
            scalefac_model (float): The scale factor for the model.
            ctrl_proj (bool, optional): Flag indicating whether the projection is a control projection. Defaults to False.

        Returns:
            int: 1 if the processing is successful, -1 otherwise.


        """

        # directory = r"/gpfs/data/kbergen/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Projection-GrIS/AWI/ISSM1/exp09"
        # get metadata from path

        path = directory.split("/")
        exp = path[-1]
        model = path[-2]
        group = path[-3]

        # Determine which control to use based on experiment (only applies to AIS) per Nowicki, 2020
        if not ctrl_proj:
            if self.ice_sheet == "AIS":
                if exp in (
                    "exp01",
                    "exp02",
                    "exp03",
                    "exp04",
                    "exp11",
                    "expA1",
                    "expA2",
                    "expA3",
                    "expA4",
                    "expB1",
                    "expB2",
                    "expB3",
                    "expB4",
                    "expB5",
                    "expC2",
                    "expC5",
                    "expC8",
                    "expC11",
                    "expE1",
                    "expE2",
                    "expE3",
                    "expE4",
                    "expE5",
                    "expE11",
                    "expE12",
                    "expE13",
                    "expE14",
                ):
                    ctrl_path = os.path.join(
                        "/".join(path[:-1]),
                        f"ctrl_proj_open/ivaf_{self.ice_sheet}_{group}_{model}_ctrl_proj_open.nc",
                    )
                elif (
                    exp
                    in (
                        "exp05",
                        "exp06",
                        "exp07",
                        "exp08",
                        "exp09",
                        "exp10",
                        "exp12",
                        "exp13",
                        "expA5",
                        "expA6",
                        "expA7",
                        "expA8",
                        "expB6",
                        "expB7",
                        "expB8",
                        "expB9",
                        "expB10",
                        "expC3",
                        "expC6",
                        "expC9",
                        "expC12",
                        "expE6",
                        "expE7",
                        "expE8",
                        "expE9",
                        "expE10",
                        "expE15",
                        "expE16",
                        "expE17",
                        "expE18",
                    )
                    or "expD" in exp
                ):
                    ctrl_path = os.path.join(
                        "/".join(path[:-1]),
                        f"ctrl_proj_std/ivaf_{self.ice_sheet}_{group}_{model}_ctrl_proj_std.nc",
                    )
                elif exp in (
                    "expC1",
                    "expC4",
                    "expC7",
                    "expC10",
                ):  # N/A value for ocean_forcing in Nowicki, 2020 table A2
                    return -1
                else:
                    print(f"Experiment {exp} not recognized. Skipped.")
                    return -1

            else:
                # GrIS doesn't have ctrl_proj_open vs ctrl_proj_std
                ctrl_path = os.path.join(
                    "/".join(path[:-1]),
                    f"ctrl_proj/ivaf_{self.ice_sheet}_{group}_{model}_ctrl_proj.nc",
                )

            # for some reason there is no ctrl_proj_open for AWI and JPL1, skip
            if group == "AWI" and "ctrl_proj_open" in ctrl_path:
                return -1
            if group == "JPL1" and "ctrl_proj_open" in ctrl_path:
                return -1

        # MUN_GISM1 is corrupted, skip
        if group == "MUN" and model == "GSM1":
            return -1
        # folder is empty, skip
        elif group == "IMAU" and exp == "exp11":
            return -1
        # bed file in NCAR_CISM/expD10 is empty, skip
        elif group == "NCAR" and exp in ("expD10", "expD11"):
            return -1

        # lookup densities from csv
        subset_densities = densities[(densities.group == group) & (densities.model == model)]
        rhoi = subset_densities.rhoi.values[0]
        rhow = subset_densities.rhow.values[0]

        # load data
        if self.ice_sheet == "AIS" and group == "ULB":
            # ULB uses fETISh for AIS naming, not actual model name (fETISh_16km or fETISh_32km)
            naming_convention = f"{self.ice_sheet}_{group}_fETISh_{exp}.nc"

        else:
            naming_convention = f"{self.ice_sheet}_{group}_{model}_{exp}.nc"

        # load data
        bed = get_xarray_data(
            os.path.join(directory, f"topg_{naming_convention}"), ice_sheet=self.ice_sheet
        )
        thickness = get_xarray_data(
            os.path.join(directory, f"lithk_{naming_convention}"), ice_sheet=self.ice_sheet
        )
        mask = get_xarray_data(
            os.path.join(directory, f"sftgif_{naming_convention}"), ice_sheet=self.ice_sheet
        )
        ground_mask = get_xarray_data(
            os.path.join(directory, f"sftgrf_{naming_convention}"), ice_sheet=self.ice_sheet
        )

        # bed = xr.open_dataset(os.path.join(directory, f'topg_{naming_convention}'), decode_times=False)
        # thickness = xr.open_dataset(os.path.join(directory, f'lithk_{naming_convention}'), decode_times=False)
        # mask = xr.open_dataset(os.path.join(directory, f'sftgif_{naming_convention}'), decode_times=False)
        # ground_mask = xr.open_dataset(os.path.join(directory, f'sftgrf_{naming_convention}'), decode_times=False)
        length_time = len(thickness.time)
        # note on decode_times=False -- by doing so, it stays in "days from" rather than trying to infer a type. Makes handling much more predictable.

        try:
            bed = bed.transpose("x", "y", "time", ...)
            thickness = thickness.transpose("x", "y", "time", ...)
            mask = mask.transpose("x", "y", "time", ...)
            ground_mask = ground_mask.transpose("x", "y", "time", ...)
        except ValueError:
            bed = bed.transpose("x", "y", ...)
            thickness = thickness.transpose("x", "y", ...)
            mask = mask.transpose("x", "y", ...)
            ground_mask = ground_mask.transpose("x", "y", ...)

        # if time is not a dimension, add copies for each time step
        if "time" not in bed.dims or bed.dims["time"] == 1:
            try:
                bed = bed.drop_vars(
                    [
                        "time",
                    ]
                )
            except ValueError:
                pass
            bed = bed.expand_dims(dim={"time": length_time})

            if length_time == 86:
                bed["time"] = thickness[
                    "time"
                ]  # most times just the bed file is missing the time index
            elif length_time > 86:
                if len(thickness.time.values) != len(set(thickness.time.values)):  # has duplicates
                    keep_indices = np.unique(thickness["time"], return_index=True)[
                        1
                    ]  # find non-duplicates
                    bed = bed.isel(time=keep_indices)  # only select non-duplicates
                    thickness = thickness.isel(time=keep_indices)
                    mask = mask.isel(time=keep_indices)
                    ground_mask = ground_mask.isel(time=keep_indices)
                else:
                    warnings.warn(
                        f"At least one file in {exp} does not have a time index formatted correctly. Attempting to fix."
                    )
                    start_idx = len(bed.time) - 86
                    bed = bed.sel(time=slice(bed.time.values[start_idx], len(bed.time)))
                    thickness = thickness.sel(
                        time=slice(thickness.time[start_idx], thickness.time[-1])
                    )
                    mask = mask.sel(time=slice(mask.time[start_idx], mask.time[-1]))
                    ground_mask = ground_mask.sel(
                        time=slice(ground_mask.time[start_idx], ground_mask.time[-1])
                    )

                try:
                    bed["time"] = thickness["time"].copy()
                except ValueError:
                    print(
                        f"Cannot fix time index for {exp} due to duplicate index values. Skipped."
                    )
                    return -1

            else:
                print(f"Only {len(bed.time)} time points for {exp}. Skipped.")
                return -1

        # if -9999 instead of np.nan, replace (come back and optimize? couldn't figure out with xarray)
        if bed.topg[0, 0, 0] <= -9999.0 or bed.topg[0, 0, 0] >= 9999:
            topg = bed.topg.values
            topg[(np.where((topg <= -9999.0) | (topg >= 9999)))] = np.nan
            bed["topg"].values = topg
            del topg

            lithk = thickness.lithk.values
            lithk[(np.where((lithk <= -9999.0) | (lithk >= 9999)))] = np.nan
            thickness["lithk"].values = lithk
            del lithk

            sftgif = mask.sftgif.values
            sftgif[(np.where((sftgif <= -9999.0) | (sftgif >= 9999)))] = np.nan
            mask["sftgif"].values = sftgif
            del sftgif

            sftgrf = ground_mask.sftgrf.values
            sftgrf[(np.where((sftgrf <= -9999.0) | (sftgrf >= 9999)))] = np.nan
            ground_mask["sftgrf"].values = sftgrf
            del sftgrf

        # converts time (in "days from X" to numpy.datetime64) and subsets time from 2015 to 2100

        # a few datasets do not have the time index formatted correctly
        if len(bed.time.attrs) == 0:

            if len(bed.time) == 86:
                bed["time"] = thickness[
                    "time"
                ]  # most times just the bed file is missing the time index
            elif len(bed.time) > 86:
                # bed['time'] = thickness['time'].copy()
                warnings.warn(
                    f"At least one file in {exp} does not have a time index formatted correctly. Attempting to fix."
                )
                start_idx = len(bed.time) - 86
                bed = bed.sel(time=slice(bed.time.values[start_idx], len(bed.time)))
                thickness = thickness.sel(time=slice(thickness.time[start_idx], thickness.time[-1]))
                mask = mask.sel(time=slice(mask.time[start_idx], mask.time[-1]))
                ground_mask = ground_mask.sel(
                    time=slice(ground_mask.time[start_idx], ground_mask.time[-1])
                )

                try:
                    bed["time"] = thickness["time"]
                except ValueError:
                    print(
                        f"Cannot fix time index for {exp} due to duplicate index values. Skipped."
                    )
                    return -1

            else:
                print(f"Only {len(bed.time)} time points for {exp}. Skipped.")
                return -1

        bed = convert_and_subset_times(bed)
        thickness = convert_and_subset_times(thickness)
        mask = convert_and_subset_times(mask)
        ground_mask = convert_and_subset_times(ground_mask)
        length_time = len(thickness.time)

        # Interpolate values for x & y, for formatting purposes only, does not get used
        if len(set(thickness.y.values)) != len(scalefac_model):
            bed["x"], bed["y"] = interpolate_values(bed)
            thickness["x"], thickness["y"] = interpolate_values(thickness)
            mask["x"], mask["y"] = interpolate_values(mask)
            ground_mask["x"], ground_mask["y"] = interpolate_values(ground_mask)

        # clip masks if they are below 0 or above 1
        if np.min(mask.sftgif.values) < 0 or np.max(mask.sftgif.values) > 1:
            mask["sftgif"] = np.clip(mask.sftgif, 0.0, 1.0)
        if np.min(ground_mask.sftgrf.values) < 0 or np.max(ground_mask.sftgrf.values) > 1:
            ground_mask["sftgrf"] = np.clip(ground_mask.sftgrf, 0.0, 1.0)

        # if time is not a dimension, add copies for each time step
        # if 'time' not in bed.dims or bed.dims['time'] == 1:
        #     try:
        #         bed = bed.drop_vars(['time',])
        #     except ValueError:
        #         pass
        #     bed = bed.expand_dims(dim={'time': length_time})

        # flip around axes so the order is (x, y, time)
        bed = bed.transpose("x", "y", "time", ...)
        bed_data = bed.topg.values

        thickness = thickness.transpose("x", "y", "time", ...)
        thickness_data = thickness.lithk.values

        mask = mask.transpose("x", "y", "time", ...)
        mask_data = mask.sftgif.values

        ground_mask = ground_mask.transpose("x", "y", "time", ...)
        ground_mask_data = ground_mask.sftgrf.values

        # for each time step, calculate ivaf
        ivaf = np.zeros(bed_data.shape)
        for i in range(length_time):

            # get data slices for current time
            thickness_i = thickness_data[:, :, i].copy()
            bed_i = bed_data[:, :, i].copy()
            mask_i = mask_data[:, :, i].copy()
            ground_mask_i = ground_mask_data[:, :, i].copy()

            # set data slices to zero where mask = 0 or any value is NaN
            thickness_i[
                (mask_i == 0)
                | (np.isnan(mask_i))
                | (np.isnan(thickness_i))
                | (np.isnan(ground_mask_i))
                | (np.isnan(bed_i))
            ] = 0
            bed_i[
                (mask_i == 0)
                | (np.isnan(mask_i))
                | (np.isnan(thickness_i))
                | (np.isnan(ground_mask_i))
                | (np.isnan(bed_i))
            ] = 0
            ground_mask_i[
                (mask_i == 0)
                | np.isnan(mask_i)
                | np.isnan(thickness_i)
                | np.isnan(ground_mask_i)
                | np.isnan(bed_i)
            ] = 0
            mask_i[
                (mask_i == 0)
                | (np.isnan(mask_i))
                | (np.isnan(thickness_i))
                | (np.isnan(ground_mask_i))
                | (np.isnan(bed_i))
            ] = 0

            # take min(bed_i, 0)
            bed_i[bed_i > 0] = 0

            # calculate IVAF (based on MATLAB processing scripts from Seroussi, 2021)
            hf_i = thickness_i + ((rhow / rhoi) * bed_i)
            masked_output = hf_i * ground_mask_data[:, :, i] * mask_data[:, :, i]
            ivaf[:, :, i] = masked_output * scalefac_model * (self.resolution * 1000) ** 2

        # subtract out control if for an experment
        ivaf_nc = bed.copy()  # copy file structure and metadata for ivaf file
        if not ctrl_proj:
            # open control dataset
            ivaf_ctrl = xr.open_dataset(
                ctrl_path,
            ).transpose("x", "y", "time", ...)

            # subtract out control
            ivaf = ivaf_ctrl.ivaf.values - ivaf

        # save ivaf file (copied format from bed_data, change accordingly.)
        ivaf_nc["ivaf"] = (("x", "y", "time"), ivaf)
        ivaf_nc = ivaf_nc.drop_vars(
            [
                "topg",
            ]
        )
        ivaf_nc["sle"] = ivaf_nc.ivaf / 1e9 / 362.5
        ivaf_nc.to_netcdf(
            os.path.join(directory, f"ivaf_{self.ice_sheet}_{group}_{model}_{exp}.nc")
        )

        print(f"{group}_{model}_{exp}: Processing successful.")

        return 1


def convert_and_subset_times(
    dataset,
):
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

            dataset["time"] = np.array(
                [start_date + np.timedelta64(int(x), "D") for x in dataset.time.values]
            )
    else:
        raise ValueError(f"Time values are not recognized: {type(dataset.time.values[0])}")

    if len(dataset.time) > 86:
        # make sure the max date is 2100
        # dataset = dataset.sel(time=slice(np.datetime64('2014-01-01'), np.datetime64('2101-01-01')))
        dataset = dataset.sel(time=slice("2012-01-01", "2101-01-01"))

        # if you still have more than 86, take the previous 86 values from 2100
        if len(dataset.time) > 86:
            # LSCE GRISLI has two 2015 measurements

            # dataset = dataset.sel(time=slice(dataset.time.values[len(dataset.time) - 86], dataset.time.values[-1]))
            start_idx = len(dataset.time) - 86
            dataset = dataset.isel(time=slice(start_idx, len(dataset.time)))

    if len(dataset.time) != 86:
        warnings.warn(
            "After subsetting there are still not 86 time points. Go back and check logs."
        )
        print(f"dataset_length={len(dataset.time)} -- {dataset.attrs}")

    return dataset


def get_model_densities(zenodo_directory: str, output_path: str = None):
    """
    Extracts values for rhoi and rhow from NetCDF files in the specified directory and returns a pandas DataFrame
    containing the group, model, rhoi, and rhow values for each file.

    Args:
        zenodo_directory (str): The path to the directory containing the NetCDF files.
        output_path (str, optional): The path to save the resulting DataFrame as a CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the group, model, rhoi, and rhow values for each file.
    """
    results = []
    for root, dirs, files in os.walk(zenodo_directory):
        for file in files:
            if file.endswith(".nc"):  # Check if the file is a NetCDF file
                file_path = os.path.join(root, file)
                try:
                    # Open the NetCDF file using xarray
                    dataset = xr.open_dataset(file_path, decode_times=False).transpose(
                        "x", "y", "time", ...
                    )

                    # Extract values for rhoi and rhow
                    if "rhoi" in dataset and "rhow" in dataset:
                        rhoi_values = dataset["rhoi"].values
                        rhow_values = dataset["rhow"].values

                        # Append the filename and values to the results list
                        results.append({"filename": file, "rhoi": rhoi_values, "rhow": rhow_values})

                    # Close the dataset
                    dataset.close()
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    densities = []
    for file in results:
        if "ctrl_proj" in file["filename"] or "hist" in file["filename"]:
            continue

        elif "ILTS" in file["filename"]:
            fp = file["filename"].split("_")
            group = "ILTS_PIK"
            model = fp[-2]

        elif "ULB_fETISh" in file["filename"]:
            fp = file["filename"].split("_")
            group = "ULB"
            model = "fETISh_32km" if "32km" in file["filename"] else "fETISh_16km"

        else:
            fp = file["filename"].split("_")
            group = fp[-3]
            model = fp[-2]
        densities.append([group, model, file["rhoi"], file["rhow"]])

    df = pd.DataFrame(densities, columns=["group", "model", "rhoi", "rhow"])
    df["rhoi"], df["rhow"] = df.rhoi.astype("float"), df.rhow.astype("float")
    df = df.drop_duplicates()

    ice_sheet = "AIS" if "AIS" in file["filename"] else "GIS"

    if output_path is not None:
        if output_path.endswith("/"):
            df.to_csv(f"{output_path}/{ice_sheet}_densities.csv", index=False)
        else:
            df.to_csv(output_path, index=False)

    return df


def interpolate_values(data):
    """
    Interpolates missing values in the x and y dimensions of the input NetCDF data using linear interpolation.

    Args:
        data: A NetCDF file containing x and y dimensions with missing values.

    Returns:
        A tuple containing the interpolated x and y arrays.
    """
    y = pd.Series(data.y.values)
    y = y.replace(0, np.NaN)
    y = np.array(y.interpolate())

    # first and last are NaNs, replace with correct values
    y[0] = y[1] - (y[2] - y[1])
    y[-1] = y[-2] + (y[-2] - y[-3])

    x = pd.Series(data.x.values)
    x = x.replace(0, np.NaN)
    x = np.array(x.interpolate())

    # first and last are NaNs, replace with correct values
    x[0] = x[1] - (x[2] - x[1])
    x[-1] = x[-2] + (x[-2] - x[-3])

    return x, y


class DimensionalityReducer:
    def __init__(self, forcing_dir, projection_dir, output_dir, ice_sheet=None, scaling_method=None):
        super().__init__()
        if forcing_dir is None:
            raise ValueError("Forcing directory must be specified.")
        if output_dir is None:
            raise ValueError("Output directory must be specified.")
        self.forcing_dir = forcing_dir
        self.projection_dir = projection_dir
        self.output_dir = output_dir
        self.forcing_paths = {"all": None, "atmosphere": None, "ocean": None}

        # check inputs
        if os.path.exists(f"{self.output_dir}/pca_models/"):
            self.pca_model_directory = f"{self.output_dir}/pca_models/"
        else:
            self.pca_model_directory = None

        if os.path.exists(f"{self.output_dir}/scalers/"):
            self.scaler_directory = f"{self.output_dir}/scalers/"
        else:
            self.scaler_directory = None
        self.scaling_method = scaling_method

        if ice_sheet not in ("AIS", "GrIS"):
            raise ValueError("Ice sheet must be specified and must be 'AIS' or 'GrIS'.")
        else:
            self.ice_sheet = ice_sheet

        if self.ice_sheet.lower() == "gris":
            atmospheric_files = get_all_filepaths(
                path=self.forcing_dir,
                filetype="nc",
                contains="Atmosphere_Forcing/aSMB_observed/v1",
            )
            atmospheric_files = [x for x in atmospheric_files if "combined" in x]

            # files in atmopheric directory are separated by year, needs to be combined
            if not atmospheric_files:
                combine_gris_forcings(self.forcing_dir)

            oceanic_files = get_all_filepaths(
                path=self.forcing_dir,
                filetype="nc",
                contains="Ocean_Forcing/Melt_Implementation/v4",
            )
            self.forcing_paths["all"] = atmospheric_files + oceanic_files
            self.forcing_paths["atmosphere"] = atmospheric_files
            self.forcing_paths["ocean"] = oceanic_files
        else:
            all_forcing_fps = get_all_filepaths(
                path=self.forcing_dir,
                filetype="nc",
                contains="1995-2100",
                not_contains="Ice_Shelf_Fracture",
            )
            self.forcing_paths["all"] = [x for x in all_forcing_fps if "8km" in x and "v1" not in x]
            self.forcing_paths["atmosphere"] = [
                x for x in self.forcing_paths["all"] if "Atmosphere_Forcing" in x
            ]
            self.forcing_paths["ocean"] = [
                x for x in self.forcing_paths["all"] if "Ocean_Forcing" in x
            ]

        all_projection_fps = get_all_filepaths(
            path=self.projection_dir, filetype="nc", contains="ivaf", not_contains="ctrl_proj"
        )
        self.projection_paths = all_projection_fps

    # def reduce_dimensionlity(self, forcing_dir: str=None, output_dir: str=None):
    # generate pca models
    # convert each forcing file to pca space

    def generate_pca_models(self, num_forcing_pcs, num_projection_pcs, scaling_method="standard"):
        """
        Generate principal component analysis (PCA) models for atmosphere and ocean variables.

        Parameters:
        - atmosphere_fps (list): List of file paths for atmosphere data.
        - ocean_fps (list): List of file paths for ocean data.
        - save_dir (str): Directory to save the generated PCA models and results.

        Returns:
            int: 0 if successful.
        """

        # check inputs and make directories for outputted models
        if not os.path.exists(f"{self.output_dir}/pca_models/"):
            os.mkdir(f"{self.output_dir}/pca_models/")
        self.pca_model_directory = f"{self.output_dir}/pca_models/"
        if not os.path.exists(f"{self.output_dir}/scalers/"):
            os.mkdir(f"{self.output_dir}/scalers/")
        self.scaler_directory = f"{self.output_dir}/scalers/"
        self.scaling_method = scaling_method

        # Train PCA models for each atmospheric and oceanic forcing variable and save
        if self.ice_sheet == "AIS":
            self._generate_ais_atmosphere_pcas(
                self.forcing_paths["atmosphere"],
                self.pca_model_directory,
                num_pcs=num_forcing_pcs,
                scaler_dir=self.scaler_directory,
                scaling_method=scaling_method,
            )
            self._generate_ais_ocean_pcas(
                self.forcing_paths["ocean"],
                self.pca_model_directory,
                num_pcs=num_forcing_pcs,
                scaler_dir=self.scaler_directory,
                scaling_method=scaling_method,
            )
        else:
            self._generate_gris_atmosphere_pcas(
                self.forcing_paths["atmosphere"],
                self.pca_model_directory,
                num_pcs=num_forcing_pcs,
                scaler_dir=self.scaler_directory,
                scaling_method=scaling_method,
            )
            self._generate_gris_ocean_pcas(
                self.forcing_paths["ocean"],
                self.pca_model_directory,
                num_pcs=num_forcing_pcs,
                scaler_dir=self.scaler_directory,
                scaling_method=scaling_method,
            )

        # Train PCA model for SLE and save
        sle_paths = get_all_filepaths(
            path=self.projection_dir, filetype="nc", contains="ivaf", not_contains="ctrl"
        )
        self._generate_sle_pca(
            sle_paths,
            save_dir=self.pca_model_directory,
            num_pcs=num_projection_pcs,
            scaler_dir=self.scaler_directory,
            scaling_method=scaling_method,
        )

        return 0

    def convert_forcings(
        self, forcing_files: list = None, pca_model_directory: str = None, output_dir: str = None, scaling_method=None,
    ):
        """
        Converts atmospheric and oceanic forcing files to PCA space using pretrained PCA models.

        Args:
            forcing_files (list, optional): List of specific forcing files to convert. If not provided, all files in the directory will be used. Default is None.
            pca_model_directory (str, optional): Directory containing the pretrained PCA models. If not provided, the directory specified during object initialization will be used. Default is None.
            output_dir (str, optional): Directory to save the converted files. If not provided, the directory specified during object initialization will be used. Default is None.

        Returns:
            int: 0 indicating successful conversion.
        """

        # check inputs for validity
        output_dir = self.output_dir if output_dir is None else output_dir
        if self.pca_model_directory is None and pca_model_directory is None:
            raise ValueError(
                "PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first."
            )
        if self.scaling_method is None and scaling_method is None:
            raise ValueError(
                "Scalers must be generated first, or scaling_method must be identified if they already exist. Run DimensionalityReducer.generate_pca_models first."
            )
            
            
        if scaling_method is not None:
            self.scaling_method = scaling_method
            
        if pca_model_directory is not None:
            self.pca_model_directory = pca_model_directory

        # if user supplies specific forcing files (rather than entire directory), use that instead
        # TODO: test this..
        if forcing_files is not None:
            warnings.warn(
                "By using specific forcing files, forcing_paths attribute will be overwritten."
            )
            self.forcing_paths["all"] = forcing_files
            self.forcing_paths["atmosphere"] = [
                x for x in self.forcing_paths["all"] if "Atmosphere_Forcing" in x
            ]
            self.forcing_paths["ocean"] = [
                x for x in self.forcing_paths["all"] if "Ocean_Forcing" in x
            ]

        # ATMOSPHERIC FORCINGS

        if not os.path.exists(f"{output_dir}/forcings/"):
            os.mkdir(f"{output_dir}/forcings/")

        # for each atmospheric forcing file, convert each variable to PCA space with pretrained PCA model
        for i, path in tqdm(
            enumerate(self.forcing_paths["atmosphere"]),
            total=len(self.forcing_paths["atmosphere"]),
            desc="Converting atmospheric forcing files to PCA space",
        ):
            # dataset = xr.open_dataset(path, decode_times=False, engine='netcdf4', ).transpose('time', 'y', 'x', ...)  # open the dataset
            # if len(dataset.dims) > 3:
            #     drop_dims = [x for x in list(dataset.dims) if x not in ('time', 'x', 'y')]
            #     dataset = dataset.drop_dims(drop_dims)
            dataset = get_xarray_data(path, ice_sheet=self.ice_sheet, convert_and_subset=True)
            forcing_name = path.replace(".nc", "").split("/")[-1]  # get metadata (model, ssp, etc.)

            # transform each variable in the dataset with their respective trained PCA model
            transformed_data = {}
            if self.ice_sheet == "AIS":

                for var in [
                    "evspsbl_anomaly",
                    "mrro_anomaly",
                    "pr_anomaly",
                    "smb_anomaly",
                    "ts_anomaly",
                ]:
                    try:
                        transformed = self.transform(
                            dataset[var].values,
                            var_name=var,
                            pca_model_directory=self.pca_model_directory,
                            scaler_directory=self.scaler_directory,
                            scaling_method=self.scaling_method,
                        )
                    except KeyError:  # if a variable is missing (usually mrro_anomaly), skip it
                        warnings.warn(f"Variable {var} not found in {forcing_name}. Skipped.")
                        continue
                    transformed_data[
                        var
                    ] = transformed  # store in dict with structure {'var_name': transformed_var}
            else:
                var = path.split("_")[-2]
                try:
                    transformed = self.transform(
                        dataset[var].values,
                        var_name=var,
                        pca_model_directory=self.pca_model_directory,
                        scaler_directory=self.scaler_directory,
                        scaling_method=self.scaling_method,
                    )
                except KeyError:
                    warnings.warn(f"Variable {var} not found in {forcing_name}. Skipped.")
                transformed_data[
                    var
                ] = transformed  # store in dict with structure {'var_name': transformed_var}

                if transformed.isnan().any() or transformed.isinf().any():
                    warnings.warn(f"NaN or inf values found in converted {forcing_name}.")

            # create a dataframe with rows corresponding to time (106 total) and columns corresponding to each variables principal components
            compiled_transformed_forcings = pd.DataFrame()
            for var in transformed_data.keys():
                var_df = pd.DataFrame(
                    transformed_data[var],
                    columns=[f"{var}_pc{i+1}" for i in range(transformed_data[var].shape[1])],
                )
                compiled_transformed_forcings = pd.DataFrame(
                    pd.concat([compiled_transformed_forcings, var_df], axis=1)
                )

            pd.DataFrame(compiled_transformed_forcings).to_csv(
                f"{output_dir}/forcings/PCA_{forcing_name}.csv", index=False
            )

        print(
            f"{len(self.forcing_paths['atmosphere'])}/{len(self.forcing_paths['atmosphere'])} atmospheric forcing files converted to PCA space."
        )
        print(
            f"Finished converting atmospheric forcings to PCA space, files outputted to {output_dir}."
        )

        # OCEANIC FORCINGS

        # for each ocean forcing file, convert each variable to PCA space with pretrained PCA model
        for i, path in tqdm(
            enumerate(self.forcing_paths["ocean"]),
            total=len(self.forcing_paths["ocean"]),
            desc="Converting oceanic forcing files",
        ):

            # open the dataset
            forcing_name = path.replace(".nc", "").split("/")[-1]  # get metadata (model, ssp, etc.)

            # get variable name by splitting the filepath name
            if self.ice_sheet == "AIS":
                var = self.forcing_paths["ocean"][i].split("/")[-1].split("_")[-4]
            else:
                metadata = self.forcing_paths["ocean"][i].split("/")[-1].split("_")
                if "basinRunoff" in metadata:
                    var = "basin_runoff"
                elif "oceanThermalForcing" in metadata:
                    var = "thermal_forcing"
                else:
                    var = self.forcing_paths["ocean"][i].split("/")[-1].split("_")[-2]
            if var == "forcing" or var == "thermal":
                var = "thermal_forcing"

            # get forcing array (requires mean value over z dimensions, see get_xarray_data())
            forcing_array = get_xarray_data(
                path, var_name=var, ice_sheet=self.ice_sheet, convert_and_subset=True
            )

            # transform each variable in the dataset with their respective trained PCA model
            transformed_data = {}
            transformed = self.transform(
                forcing_array,
                var_name=var,
                pca_model_directory=self.pca_model_directory,
                scaler_directory=self.scaler_directory,
                scaling_method=self.scaling_method,
            )
            transformed_data[
                var
            ] = transformed  # store in dict with structure {'var_name': transformed_var}

            # create a dataframe with rows corresponding to time (86 total) and columns corresponding to each variables principal components
            variable_df = pd.DataFrame(
                transformed_data[var],
                columns=[f"{var}_pc{i+1}" for i in range(transformed_data[var].shape[1])],
            )
            variable_df.to_csv(f"{output_dir}/forcings/PCA_{forcing_name}.csv", index=False)

        print(
            f"{len(self.forcing_paths['ocean'])}/{len(self.forcing_paths['ocean'])} oceanic forcing files converted to PCA space."
        )
        print(
            f"Finished converting oceanic forcings to PCA space, files outputted to {output_dir}."
        )

        return 0

    def convert_projections(
        self, projection_files: list = None, pca_model_directory: str = None, output_dir: str = None, scaling_method=None,
    ):

        # check inputs for validity
        output_dir = self.output_dir if output_dir is None else output_dir
        if self.pca_model_directory is None and pca_model_directory is None:
            raise ValueError(
                "PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first."
            )
        
        if self.scaling_method is None and scaling_method is None:
            raise ValueError(
                "Scalers must be generated first, or scaling_method must be identified if they already exist. Run DimensionalityReducer.generate_pca_models first."
            )
            
            
        if scaling_method is not None:
            self.scaling_method = scaling_method
        
        if pca_model_directory is not None:
            self.pca_model_directory = pca_model_directory

        # if user supplies specific projection files (rather than entire directory), use that instead
        if projection_files is not None:
            warnings.warn(
                "By using specific projection files, projection_paths attribute will be overwritten."
            )
            self.projection_paths = projection_files

        # make a folder in output directory for converted projections
        if not os.path.exists(f"{output_dir}/projections/"):
            os.mkdir(f"{output_dir}/projections/")

        # for each projection file, convert ivaf to PCA space with pretrained PCA model
        for i, path in tqdm(
            enumerate(self.projection_paths),
            total=len(self.projection_paths),
            desc="Converting projection files to PCA space",
        ):
            # get forcing array (requires mean value over z dimensions, see get_xarray_data())
            try:
                projection_array = get_xarray_data(path, var_name="sle", ice_sheet=self.ice_sheet)
            except:
                projection_array = get_xarray_data(path, var_name="ivaf", ice_sheet=self.ice_sheet)
                projection_array = projection_array / 1e9 / 362.5

            # nan_indices = np.argwhere(np.isnan(projection_array))
            # print(len(nan_indices))
            # continue

            # projection_array = np.nan_to_num(projection_array)  # deal with np.nans
            var = "sle"
            # projection_array = np.nan_to_num(projection_array)  # there shouldn't be nans...
            projection_name = path.replace(".nc", "").split("/")[
                -1
            ]  # get metadata (model, ssp, etc.)

            # transform each variable in the dataset with their respective trained PCA model
            transformed_data = {}
            transformed = self.transform(
                projection_array,
                var_name=var,
                pca_model_directory=self.pca_model_directory,
                scaler_directory=self.scaler_directory,
                scaling_method=self.scaling_method,
            )
            transformed_data[
                var
            ] = transformed  # store in dict with structure {'var_name': transformed_var}

            # create a dataframe with rows corresponding to time (86 total) and columns corresponding to each variables principal components
            variable_df = pd.DataFrame(
                transformed_data[var],
                columns=[f"{var}_pc{i+1}" for i in range(transformed_data[var].shape[1])],
            )
            variable_df.to_csv(f"{output_dir}/projections/PCA_{projection_name}.csv", index=False)

        print(
            f"{len(self.projection_paths)}/{len(self.projection_paths)} projection files converted to PCA space."
        )
        print(f"Finished converting projections to PCA space, files outputted to {output_dir}.")

    def _generate_ais_atmosphere_pcas(
        self,
        atmosphere_fps: list,
        save_dir: str,
        num_pcs="95%",
        scaler_dir: str = None,
        scaling_method="standard",
    ):
        """
        Generate principal component analysis (PCA) for atmospheric variables.

        Args:
            atmosphere_fps (list): List of file paths to atmospheric CMIP files.
            save_dir (str): Directory to save the PCA results.

        Returns:
            int: 0 if successful.
        """

        # if no separate directory for saving scalers is specified, use the pca save_dir
        if scaler_dir is None:
            scaler_dir = save_dir

        # for each variable

        var_names = ["pr_anomaly", "evspsbl_anomaly", "mrro_anomaly", "smb_anomaly", "ts_anomaly"]
        for i, var in tqdm(
            enumerate(var_names), total=len(var_names), desc="Processing atmospheric PCA"
        ):
            variable_array = np.zeros([len(atmosphere_fps), 86, 761 * 761])

            # loop through each atmospheric CMIP file and combine them into one big array
            for i, fp in enumerate(atmosphere_fps):

                # get the variable you need (rather than the entire dataset)
                dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet, convert_and_subset=True)
                # data_array = convert_and_subset_times(dataset)
                try:
                    data_flattened = dataset[var].values.reshape(86, 761 * 761)
                except KeyError:
                    data_flattened = np.nan
                # store it in the total array
                variable_array[i, :, :] = data_flattened

            # deal with np.nans -- since it's an anomaly, replace with 0
            variable_array = np.nan_to_num(variable_array)

            # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
            variable_array = variable_array.reshape(len(atmosphere_fps) * 86, 761 * 761)

            # scale data
            if scaling_method.lower() == "standard":
                variable_scaler = StandardScaler()
            elif scaling_method.lower() == "robust":
                variable_scaler = RobustScaler()
            elif scaling_method.lower() == "log":
                variable_scaler = LogScaler()
            variable_scaler.fit(variable_array)
            variable_array = variable_scaler.transform(variable_array)

            # run PCA
            pca, _ = self._run_PCA(variable_array, num_pcs=num_pcs)

            # output pca object
            save_path = f"{save_dir}/AIS_pca_{var}.pth"
            pca.save(save_path)
            # and scaler
            save_path = f"{scaler_dir}/AIS_{var}_scaler.pth"
            variable_scaler.save(save_path)

        return 0

    def _generate_ais_ocean_pcas(
        self,
        ocean_fps: list,
        save_dir: str,
        num_pcs="95%",
        scaler_dir: str = None,
        scaling_method="standard",
    ):
        """
        Generate principal component analysis (PCA) for ocean variables.

        Args:
            ocean_fps (list): List of file paths for ocean variables.
            save_dir (str): Directory to save the PCA results.

        Returns:
            int: 0 if PCA generation is successful, -1 otherwise.
        """

        if scaler_dir is None:
            scaler_dir = save_dir

        thermal_forcing_fps = [x for x in ocean_fps if "thermal_forcing" in x]
        salinity_fps = [x for x in ocean_fps if "salinity" in x]
        temperature_fps = [x for x in ocean_fps if "temperature" in x]

        thermal_forcing_array = np.zeros([len(thermal_forcing_fps), 86, 761 * 761])
        salinity_array = np.zeros([len(salinity_fps), 86, 761 * 761])
        temperature_array = np.zeros([len(temperature_fps), 86, 761 * 761])

        # get the variables you need (rather than the entire dataset)
        print("Processing thermal_forcing PCA model.")
        for i, fp in enumerate(thermal_forcing_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet, convert_and_subset=True)
            # data_array = convert_and_subset_times(dataset)
            thermal_forcing_array[i, :, :] = dataset["thermal_forcing"].values.reshape(
                86, 761 * 761
            )  # store
        print("Processing salinity PCA model.")
        for i, fp in enumerate(salinity_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet, convert_and_subset=True)
            # data_array = convert_and_subset_times(dataset)
            salinity_array[i, :, :] = dataset["salinity"].values.reshape(86, 761 * 761)  # store
        print("Processing temperature PCA model.")
        for i, fp in enumerate(temperature_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet, convert_and_subset=True)
            # data_array = convert_and_subset_times(dataset)
            temperature_array[i, :, :] = dataset["temperature"].values.reshape(86, 761 * 761)

        # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
        thermal_forcing_array = thermal_forcing_array.reshape(
            len(thermal_forcing_fps) * 86, 761 * 761
        )
        salinity_array = salinity_array.reshape(len(salinity_fps) * 86, 761 * 761)
        temperature_array = temperature_array.reshape(len(temperature_fps) * 86, 761 * 761)

        # remove nans
        thermal_forcing_array = np.nan_to_num(thermal_forcing_array)
        salinity_array = np.nan_to_num(salinity_array)
        temperature_array = np.nan_to_num(temperature_array)

        # scale data
        if scaling_method.lower() == "standard":
            therm_scaler = StandardScaler()
        elif scaling_method.lower() == "robust":
            therm_scaler = RobustScaler()
        elif scaling_method.lower() == "log":
            therm_scaler = LogScaler()
        else:
            raise ValueError(f"Scaler method {scaling_method} not recognized.")
        therm_scaler.fit(thermal_forcing_array)
        thermal_forcing_array = therm_scaler.transform(thermal_forcing_array)

        if scaling_method.lower() == "standard":
            salinity_scaler = StandardScaler()
        elif scaling_method.lower() == "robust":
            salinity_scaler = RobustScaler()
        elif scaling_method.lower() == "log":
            salinity_scaler = LogScaler()
        else:
            raise ValueError(f"Scaler method {scaling_method} not recognized.")
        salinity_scaler.fit(salinity_array)
        salinity_array = salinity_scaler.transform(salinity_array)

        if scaling_method.lower() == "standard":
            temp_scaler = StandardScaler()
        elif scaling_method.lower() == "robust":
            temp_scaler = RobustScaler()
        elif scaling_method.lower() == "log":
            temp_scaler = LogScaler()
        else:
            raise ValueError(f"Scaler method {scaling_method} not recognized.")
        temp_scaler.fit(temperature_array)
        temperature_array = temp_scaler.transform(temperature_array)

        # run PCA
        pca_tf, _ = self._run_PCA(thermal_forcing_array, num_pcs=num_pcs)
        pca_sal, _ = self._run_PCA(salinity_array, num_pcs=num_pcs)
        pca_temp, _ = self._run_PCA(temperature_array, num_pcs=num_pcs)

        # get percent explained
        save_path = f"{save_dir}/AIS_pca_thermal_forcing.pth"
        pca_tf.save(save_path)

        save_path = f"{save_dir}/AIS_pca_salinity.pth"
        pca_sal.save(save_path)

        save_path = f"{save_dir}/AIS_pca_temperature.pth"
        pca_temp.save(save_path)

        # save scalers
        save_path = f"{scaler_dir}/AIS_scaler_thermal_forcing.pth"
        therm_scaler.save(save_path)

        save_path = f"{scaler_dir}/AIS_scaler_temperature.pth"
        temp_scaler.save(save_path)

        save_path = f"{scaler_dir}/AIS_scaler_salinity.pth"
        salinity_scaler.save(save_path)

        return 0

    def _generate_gris_atmosphere_pcas(
        self,
        atmosphere_fps: list,
        save_dir: str,
        num_pcs="95%",
        scaler_dir: str = None,
        scaling_method="standard",
    ):

        # if no separate directory for saving scalers is specified, use the pca save_dir
        if scaler_dir is None:
            scaler_dir = save_dir

        # get SMB and ST paths
        test_num = 5
        aSMB_fps = [x for x in atmosphere_fps if "aSMB_combined" in x][0:test_num]
        aST_fps = [x for x in atmosphere_fps if "aST_combined" in x][0:test_num]

        # allocate memory
        flattened_xy_dim = 337 * 577

        smb_forcing_array = np.zeros([len(aSMB_fps), 86, flattened_xy_dim])
        st_forcing_array = np.zeros([len(aST_fps), 86, flattened_xy_dim])
        # get xarray dataset, format it, and put it in preallocated array
        print("Processing aSMB PCA model.")
        for i, fp in enumerate(aSMB_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet, convert_and_subset=True)
            smb_forcing_array[i, :, :] = dataset["aSMB"].values.reshape(
                86, flattened_xy_dim
            )  # store
        print("Processing aST PCA model.")
        for i, fp in enumerate(aST_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet, convert_and_subset=True)
            st_forcing_array[i, :, :] = dataset["aST"].values.reshape(86, flattened_xy_dim)  # store

        # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
        smb_forcing_array = smb_forcing_array.reshape(
            len(aSMB_fps) * len(dataset.time), flattened_xy_dim
        )
        st_forcing_array = st_forcing_array.reshape(
            len(aST_fps) * len(dataset.time), flattened_xy_dim
        )

        # remove nans
        smb_forcing_array = np.nan_to_num(smb_forcing_array)
        st_forcing_array = np.nan_to_num(st_forcing_array)

        # scale data
        if scaling_method.lower() == "standard":
            smb_scaler = StandardScaler()
        elif scaling_method.lower() == "robust":
            smb_scaler = RobustScaler()
        elif scaling_method.lower() == "log":
            smb_scaler = LogScaler()
        else:
            raise ValueError(f"Scaler method {scaling_method} not recognized.")
        smb_scaler.fit(smb_forcing_array)
        smb_forcing_array = smb_scaler.transform(smb_forcing_array)

        if scaling_method.lower() == "standard":
            st_scaler = StandardScaler()
        elif scaling_method.lower() == "robust":
            st_scaler = RobustScaler()
        elif scaling_method.lower() == "log":
            st_scaler = LogScaler()
        else:
            raise ValueError(f"Scaler method {scaling_method} not recognized.")
        st_scaler.fit(st_forcing_array)
        st_forcing_array = st_scaler.transform(st_forcing_array)

        # run PCA
        pca_smb, _ = self._run_PCA(smb_forcing_array, num_pcs=num_pcs)
        pca_st, _ = self._run_PCA(st_forcing_array, num_pcs=num_pcs)

        # save pca models
        save_path = f"{save_dir}/GrIS_pca_aSMB.pth"
        pca_smb.save(save_path)

        save_path = f"{save_dir}/GrIS_pca_aST.pth"
        pca_st.save(save_path)

        # save scalers
        save_path = f"{scaler_dir}/GrIS_aSMB_scaler.pth"
        smb_scaler.save(save_path)

        save_path = f"{scaler_dir}/GrIS_aST_scaler.pth"
        st_scaler.save(save_path)

        return 0

    def _generate_gris_ocean_pcas(
        self,
        ocean_fps: list,
        save_dir: str,
        num_pcs="95%",
        scaler_dir: str = None,
        scaling_method="standard",
    ):

        # if no separate directory for saving scalers is specified, use the pca save_dir
        if scaler_dir is None:
            scaler_dir = save_dir

        basin_runoff_fps = [x for x in ocean_fps if "basinRunoff" in x]
        thermal_forcing_fps = [x for x in ocean_fps if "oceanThermalForcing" in x]

        # allocate memory
        flattened_xy_dim = 337 * 577
        basin_runoff_array = np.zeros([len(basin_runoff_fps), 86, flattened_xy_dim])
        thermal_forcing_array = np.zeros([len(thermal_forcing_fps), 86, flattened_xy_dim])

        # get xarray dataset, format it, and put it in preallocated array
        print("Processing basin_runoff PCA model.")
        for i, fp in enumerate(basin_runoff_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet, convert_and_subset=True)
            basin_runoff_array[i, :, :] = dataset["basin_runoff"].values.reshape(
                86, flattened_xy_dim
            )
        print("Processing thermal_forcing PCA model.")
        for i, fp in enumerate(thermal_forcing_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet, convert_and_subset=True)
            thermal_forcing_array[i, :, :] = dataset["thermal_forcing"].values.reshape(
                86, flattened_xy_dim
            )

        # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
        basin_runoff_array = basin_runoff_array.reshape(
            len(basin_runoff_fps) * len(dataset.time), flattened_xy_dim
        )
        thermal_forcing_array = thermal_forcing_array.reshape(
            len(thermal_forcing_fps) * len(dataset.time), flattened_xy_dim
        )

        # remove nans
        basin_runoff_array = np.nan_to_num(basin_runoff_array)
        thermal_forcing_array = np.nan_to_num(thermal_forcing_array)

        # scale data
        if scaling_method.lower() == "standard":
            basin_runoff_scaler = StandardScaler()
        elif scaling_method.lower() == "robust":
            basin_runoff_scaler = RobustScaler()
        elif scaling_method.lower() == "log":
            basin_runoff_scaler = LogScaler()
        else:
            raise ValueError(f"Scaler method {scaling_method} not recognized.")
        basin_runoff_scaler.fit(basin_runoff_array)
        basin_runoff_array = basin_runoff_scaler.transform(basin_runoff_array)

        if scaling_method.lower() == "standard":
            thermal_forcing_scaler = StandardScaler()
        elif scaling_method.lower() == "robust":
            thermal_forcing_scaler = RobustScaler()
        elif scaling_method.lower() == "log":
            thermal_forcing_scaler = LogScaler()
        else:
            raise ValueError(f"Scaler method {scaling_method} not recognized.")
        thermal_forcing_scaler.fit(thermal_forcing_array)
        thermal_forcing_array = thermal_forcing_scaler.transform(thermal_forcing_array)

        # run PCA
        pca_br, _ = self._run_PCA(basin_runoff_array, num_pcs=num_pcs)
        pca_tf, _ = self._run_PCA(thermal_forcing_array, num_pcs=num_pcs)

        # save PCA
        save_path = f"{save_dir}/GrIS_pca_basin_runoff.pth"
        pca_br.save(save_path)

        save_path = f"{save_dir}/GrIS_pca_thermal_forcing.pth"
        pca_tf.save(save_path)

        # save scalers
        save_path = f"{scaler_dir}/GrIS_basin_runoff_scaler.pth"
        basin_runoff_scaler.save(save_path)

        save_path = f"{scaler_dir}/GrIS_thermal_forcing_scaler.pth"
        thermal_forcing_scaler.save(save_path)

        return 0

    def _generate_sle_pca(
        self, sle_fps: list, save_dir: str, num_pcs="99%", scaler_dir=None, scaling_method="standard"
    ):
        """
        Generate principal component analysis (PCA) for sea level equivalent (SLE) variables.

        Args:
            sle_fps (list): List of file paths for SLE variables.
            save_dir (str): Directory to save the PCA results.

        Returns:
            int: 0 if PCA generation is successful, -1 otherwise.
        """

        if scaler_dir is None:
            scaler_dir = save_dir

        # get the flattened xy dimension
        if self.ice_sheet == "AIS":
            flattened_xy_dim = 761 * 761
        else:
            flattened_xy_dim = 337 * 577

        # allocate memory
        sle_array = np.zeros([len(sle_fps), 86, flattened_xy_dim])

        # loop through each SLE (IVAF) projection file
        for i, fp in tqdm(enumerate(sle_fps), total=len(sle_fps), desc="Aggregating SLE files"):
            # get the variable
            try:
                data_flattened = get_xarray_data(fp, var_name="sle", ice_sheet=self.ice_sheet)
            except:
                data_flattened = get_xarray_data(fp, var_name="ivaf", ice_sheet=self.ice_sheet)
                data_flattened = data_flattened / 1e9 / 362.5

            # store it in the total array
            sle_array[i, :, :] = data_flattened

        # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
        sle_array = sle_array.reshape(len(sle_fps) * 86, flattened_xy_dim)

        # since the array is so large (350*85, 761*761) = (29750, 579121), randomly sample N rows and run PCA
        sle_array = sle_array[np.random.choice(sle_array.shape[0], 1590, replace=False), :]

        # deal with np.nans
        sle_array = np.nan_to_num(sle_array)

        # scale sle
        if scaling_method.lower() == "standard":
            scaler = StandardScaler()
        elif scaling_method.lower() == "robust":
            scaler = RobustScaler()
        elif scaling_method.lower() == "log":
            scaler = LogScaler()
        else:
            raise ValueError(f"Scaler method {scaling_method} not recognized.")
        scaler.fit(sle_array)
        sle_array = scaler.transform(sle_array)

        # run pca
        pca, _ = self._run_PCA(
            sle_array,
            num_pcs=num_pcs,
        )

        # output pca object
        save_path = f"{save_dir}/{self.ice_sheet}_pca_sle.pth"
        pca.save(save_path)

        # and scaler
        save_path = f"{scaler_dir}/{self.ice_sheet}_scaler_sle.pth"
        scaler.save(save_path)

        return 0

    def _run_PCA(
        self,
        variable_array,
        num_pcs,
    ):
        """
        Runs Principal Component Analysis (PCA) on the given variable array.

        Args:
            variable_array (array-like): The input array containing the variables.
            num_pcs (int, optional): The number of principal components to keep.
                If not specified, all components will be kept.

        Returns:
            tuple: A tuple containing the fitted PCA model and the transformed array.

        """
        if isinstance(num_pcs, str) and not num_pcs.endswith("%"):
            raise ValueError("num_pcs must be an integer, float, or string ending with '%'")

        # if num_pcs is a string, convert it to a float
        if isinstance(num_pcs, str) and num_pcs.endswith("%"):
            num_pcs = float(num_pcs.replace("%", ""))
            if num_pcs > 1:
                num_pcs /= 100

        # run PCA
        pca = PCA(
            n_components=num_pcs,
        )

        # fit and transform the variable array
        pca = pca.fit(variable_array)
        pca_array = pca.transform(variable_array)
        return pca, pca_array

    def _load_pca_models(self, pca_model_directory, var_name="all"):
        if self.pca_model_directory is None and pca_model_directory is None:
            raise ValueError(
                "PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first."
            )
        if pca_model_directory is not None:
            self.pca_model_directory = pca_model_directory

        # get all pca model paths
        pca_models_paths = os.listdir(self.pca_model_directory)
        pca_models_paths = [x for x in pca_models_paths if "pca" in x and self.ice_sheet in x]

        # load pca models
        if self.ice_sheet == "AIS":

            if var_name not in [
                "all",
                "evspsbl_anomaly",
                "mrro_anomaly",
                "pr_anomaly",
                "smb_anomaly",
                "ts_anomaly",
                "thermal_forcing",
                "salinity",
                "temperature",
                "sle",
                None,
            ]:
                raise ValueError(f"Variable name {var_name} not recognized.")

            if var_name == "all" or var_name is None:
                evspsbl_model = [x for x in pca_models_paths if "evspsbl" in x][0]
                mrro_model = [x for x in pca_models_paths if "mrro" in x][0]
                pr_model = [x for x in pca_models_paths if "pr" in x][0]
                smb_model = [x for x in pca_models_paths if "smb" in x][0]
                ts_model = [x for x in pca_models_paths if "ts" in x][0]
                thermal_forcing_model = [x for x in pca_models_paths if "thermal_forcing" in x][0]
                salinity_model = [x for x in pca_models_paths if "salinity" in x][0]
                temperature_model = [x for x in pca_models_paths if "temperature" in x][0]
                sle_model = [x for x in pca_models_paths if "sle" in x][0]

                pca_models = dict(
                    evspsbl_anomaly=PCA.load(f"{self.pca_model_directory}/{evspsbl_model}"),
                    mrro_anomaly=PCA.load(f"{self.pca_model_directory}/{mrro_model}"),
                    pr_anomaly=PCA.load(f"{self.pca_model_directory}/{pr_model}"),
                    smb_anomaly=PCA.load(f"{self.pca_model_directory}/{smb_model}"),
                    ts_anomaly=PCA.load(f"{self.pca_model_directory}/{ts_model}"),
                    thermal_forcing=PCA.load(f"{self.pca_model_directory}/{thermal_forcing_model}"),
                    salinity=PCA.load(f"{self.pca_model_directory}/{salinity_model}"),
                    temperature=PCA.load(f"{self.pca_model_directory}/{temperature_model}"),
                    sle=PCA.load(f"{self.pca_model_directory}/{sle_model}"),
                )
            else:
                pca_models = {}
                model_path = [x for x in pca_models_paths if var_name in x][0]
                pca_models[var_name] = PCA.load(
                    f"{self.pca_model_directory}/{model_path}",
                )
        else:
            if var_name not in [
                "all",
                "aST",
                "aSMB",
                "basin_runoff",
                "thermal_forcing",
                "sle",
                None,
            ]:
                raise ValueError(f"Variable name {var_name} not recognized.")

            if var_name == "all" or var_name is None:
                aSMB_model = [x for x in pca_models_paths if "aSMB" in x][0]
                aST_model = [x for x in pca_models_paths if "aST" in x][0]
                basin_runoff_model = [x for x in pca_models_paths if "basin_runoff" in x][0]
                thermal_forcing_model = [x for x in pca_models_paths if "thermal_forcing" in x][0]

                pca_models = dict(
                    aSMB=PCA.load(f"{self.pca_model_directory}/{aSMB_model}"),
                    aST=PCA.load(f"{self.pca_model_directory}/{aST_model}"),
                    basin_runoff=PCA.load(f"{self.pca_model_directory}/{basin_runoff_model}"),
                    thermal_forcing=PCA.load(f"{self.pca_model_directory}/{thermal_forcing_model}"),
                    sle=PCA.load(f"{self.pca_model_directory}/{sle_model}"),
                )
            else:
                pca_models = {}
                model_path = [x for x in pca_models_paths if var_name in x][0]
                pca_models[var_name] = PCA.load(f"{self.pca_model_directory}/{model_path}")

        return pca_models

    def _load_scalers(self, scaler_directory, var_name="all", scaling_method="standard"):
        
        if scaling_method.lower() == "standard":
            scaler_class = StandardScaler
        elif scaling_method.lower() == "robust":
            scaler_class = RobustScaler
        elif scaling_method.lower() == "log":
            scaler_class = LogScaler
        else:
            raise ValueError(f"Scaler method {scaling_method} not recognized.")
        if self.scaler_directory is None and scaler_directory is None:
            warnings.warn(
                "self.scaler_directory is None, resorting to using self.pca_model_directory"
            )
            if self.pca_model_directory is None:
                raise ValueError(
                    "Scaler directory must be specified, or DimensionalityReducer.generate_pca_models must be run first."
                )
            self.scaler_directory = self.pca_model_directory
        if scaler_directory is not None:
            self.scaler_directory = scaler_directory

        # get all scaler model paths
        scaler_paths = os.listdir(self.scaler_directory)
        scaler_paths = [x for x in scaler_paths if "scaler" in x and self.ice_sheet in x]

        # load scaler models
        if self.ice_sheet == "AIS":

            if var_name not in [
                "all",
                "evspsbl_anomaly",
                "mrro_anomaly",
                "pr_anomaly",
                "smb_anomaly",
                "ts_anomaly",
                "thermal_forcing",
                "salinity",
                "temperature",
                "sle",
                None,
            ]:
                raise ValueError(f"Variable name {var_name} not recognized.")

            if var_name == "all" or var_name is None:
                evspsbl_model = [x for x in scaler_paths if "evspsbl" in x][0]
                mrro_model = [x for x in scaler_paths if "mrro" in x][0]
                pr_model = [x for x in scaler_paths if "pr" in x][0]
                smb_model = [x for x in scaler_paths if "smb" in x][0]
                ts_model = [x for x in scaler_paths if "ts" in x][0]
                thermal_forcing_model = [x for x in scaler_paths if "thermal_forcing" in x][0]
                salinity_model = [x for x in scaler_paths if "salinity" in x][0]
                temperature_model = [x for x in scaler_paths if "temperature" in x][0]
                sle_model = [x for x in scaler_paths if "sle" in x][0]

                scalers = dict(
                    evspsbl_anomaly=scaler_class.load(f"{self.scaler_directory}/{evspsbl_model}"),
                    mrro_anomaly=scaler_class.load(f"{self.scaler_directory}/{mrro_model}"),
                    pr_anomaly=scaler_class.load(f"{self.scaler_directory}/{pr_model}"),
                    smb_anomaly=scaler_class.load(f"{self.scaler_directory}/{smb_model}"),
                    ts_anomaly=scaler_class.load(f"{self.scaler_directory}/{ts_model}"),
                    thermal_forcing=scaler_class.load(
                        f"{self.scaler_directory}/{thermal_forcing_model}"
                    ),
                    salinity=scaler_class.load(f"{self.scaler_directory}/{salinity_model}"),
                    temperature=scaler_class.load(f"{self.scaler_directory}/{temperature_model}"),
                    sle=scaler_class.load(f"{self.scaler_directory}/{sle_model}"),
                )
            else:
                scalers = {}
                scaler_path = [x for x in scaler_paths if var_name in x][0]
                scalers[var_name] = scaler_class.load(f"{self.scaler_directory}/{scaler_path}")

        else:  # GrIS
            if var_name not in [
                "all",
                "aST",
                "aSMB",
                "basin_runoff",
                "thermal_forcing",
                "sle",
                None,
            ]:
                raise ValueError(f"Variable name {var_name} not recognized.")

            if var_name == "all" or var_name is None:
                aSMB_model = [x for x in scaler_paths if "aSMB" in x][0]
                aST_model = [x for x in scaler_paths if "aST" in x][0]
                basin_runoff_model = [x for x in scaler_paths if "basin_runoff" in x][0]
                thermal_forcing_model = [x for x in scaler_paths if "thermal_forcing" in x][0]
                sle_model = [x for x in scaler_paths if "sle" in x][0]

                scalers = dict(
                    aSMB=scaler_class.load(
                        f"{self.scaler_directory}/{aSMB_model}",
                    ),
                    aST=scaler_class.load(
                        f"{self.scaler_directory}/{aST_model}",
                    ),
                    basin_runoff=scaler_class.load(
                        f"{self.scaler_directory}/{basin_runoff_model}",
                    ),
                    thermal_forcing=scaler_class.load(
                        f"{self.scaler_directory}/{thermal_forcing_model}",
                    ),
                    sle=scaler_class.load(
                        f"{self.scaler_directory}/{sle_model}",
                    ),
                )
            else:
                scalers = {}
                scaler_path = [x for x in scaler_paths if var_name in x][0]
                scalers[var_name] = scaler_class.load(f"{self.scaler_directory}/{scaler_path}")

        return scalers

    def transform(self, x, var_name, num_pcs=None, pca_model_directory=None, scaler_directory=None, scaling_method="standard"):
        """
        Transform the given variable into PCA space.

        Args:
            x (array-like): The input array containing the variables.
            variable (str): The name of the variable to transform.
            pca_models_paths (dict): A dictionary containing the filepaths for the PCA models.

        Returns:
            array-like: The transformed array.
        """
        #
        if pca_model_directory is None and self.pca_model_directory is None:
            raise ValueError(
                "PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first."
            )

        if pca_model_directory is not None:
            self.pca_model_directory = pca_model_directory

        if scaler_directory is None and self.scaler_directory is None:
            raise ValueError(
                "PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first."
            )

        if scaler_directory is not None:
            self.scaler_directory = scaler_directory

        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)

        # load pca and scaler models
        pca_models = self._load_pca_models(self.pca_model_directory, var_name=var_name)
        scalers = self._load_scalers(self.scaler_directory, var_name=var_name, scaling_method=scaling_method)
        pca = pca_models[var_name]
        scaler = scalers[var_name]
        x = np.nan_to_num(x)

        # scale and transform
        scaled = scaler.transform(x)
        transformed = pca.transform(scaled)

        # if num_pcs is a string, convert it to a float
        if num_pcs is not None and num_pcs.endswith("%"):
            exp_var_pca = pca.explained_variance_ratio_
            cum_sum_eigenvalues = np.cumsum(exp_var_pca)
            num_pcs_cutoff = cum_sum_eigenvalues > float(num_pcs.replace("%", "")) / 100
            if ~num_pcs_cutoff.any():
                warnings.warn(
                    f"Explained variance cutoff ({num_pcs}) not reached, using all PCs available ({len(cum_sum_eigenvalues)})."
                )
                num_pcs = len(cum_sum_eigenvalues)
            else:
                num_pcs = np.argmax(num_pcs_cutoff) + 1

        return transformed[:, :num_pcs]

    def invert(self, pca_x, var_name, pca_model_directory=None, scaler_directory=None):
        """
        Invert the given variable from PCA space.

        Args:
            pca_x (array-like): The input array containing the variables in PCA space.
            variable (str): The name of the variable to transform.
            pca_models_paths (dict): A dictionary containing the filepaths for the PCA models.

        Returns:
            array-like: The inverted array.
        """
        if pca_model_directory is None and self.pca_model_directory is None:
            raise ValueError(
                "PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first."
            )

        if pca_model_directory is not None:
            self.pca_model_directory = pca_model_directory

        # load pca and calculate inverse
        pca_models = self._load_pca_models(pca_model_directory, var_name=var_name)
        pca = pca_models[var_name]
        inverted = pca.inverse_transform(pca_x)

        # unscale pca inverse
        scalers = self._load_scalers(scaler_directory, var_name=var_name)
        scaler = scalers[var_name]
        unscaled = scaler.inverse_transform(inverted)
        return unscaled


def get_xarray_data(dataset_fp, var_name=None, ice_sheet="AIS", convert_and_subset=False):
    """
    Retrieves data from an xarray dataset.

    Args:
        dataset_fp (str): The file path to the xarray dataset.
        var_name (str, optional): The name of the variable to retrieve from the dataset. Defaults to None.
        ice_sheet (str, optional): The ice sheet type. Defaults to 'AIS'.
        convert_and_subset (bool, optional): Flag indicating whether to convert and subset the dataset. Defaults to False.

    Returns:
        np.ndarray or xr.Dataset: The retrieved data from the dataset.
    """

    dataset = xr.open_dataset(
        dataset_fp,
        decode_times=False,
        engine="netcdf4",
    )
    try:
        dataset = dataset.transpose("time", "x", "y", ...)
    except:
        pass

    if "ivaf" in dataset.variables:
        pass

    else:

        # handle extra dimensions and variables
        try:
            dataset = dataset.drop_dims("nv4")
        except ValueError:
            pass

        for var in [
            "z_bnds",
            "lat",
            "lon",
            "mapping",
            "time_bounds",
            "lat2d",
            "lon2d",
            "polar_stereographic",
        ]:
            try:
                dataset = dataset.drop(labels=[var])
            except ValueError:
                pass
        if "z" in dataset.dims:
            dataset = dataset.mean(dim="z", skipna=True)

    # subset the dataset for 5km resolution (GrIS)
    if dataset.dims["x"] == 1681 and dataset.dims["y"] == 2881:
        dataset = dataset.sel(x=dataset.x.values[::5], y=dataset.y.values[::5])

    if convert_and_subset:
        dataset = convert_and_subset_times(dataset)

    if var_name is not None:
        try:
            data = dataset[var_name].values
        except KeyError:
            return np.nan, np.nan

        x_dim = 761 if ice_sheet.lower() == "ais" else 337
        y_dim = 761 if ice_sheet.lower() == "ais" else 577
        if (
            "time" not in dataset.dims
            or dataset.dims["time"] == 1
            or (data.shape[1] == y_dim and data.shape[2] == x_dim)
        ):
            pass
        else:
            # TODO: fix this. this is just a weird way of tranposing, not sure if it even happens.
            grid_indices = np.array([0, 1, 2])[
                (np.array(data.shape) == x_dim) | (np.array(data.shape) == y_dim)
            ]
            data = np.moveaxis(data, list(grid_indices), [1, 2])

        if "time" not in dataset.dims:
            data_flattened = data.reshape(
                -1,
            )
        else:
            data_flattened = data.reshape(len(dataset.time), -1)
        return data_flattened

    return dataset


class DatasetMerger:
    """
    A class for merging datasets from forcing and projection files.
    """

    def __init__(self, ice_sheet, forcings, projections, experiment_file, output_dir):
        """
        Initializes a DatasetMerger object.

        Args:
            ice_sheet (str): The ice sheet name.
            forcing_dir (str): The directory path for forcing files.
            projection_dir (str): The directory path for projection files.
            experiment_file (str): The path to the experiment file (CSV or JSON).
            output_dir (str): The directory path to save the merged dataset.
        """
        self.ice_sheet = ice_sheet
        self.forcings = forcings
        self.projections = projections
        self.experiment_file = experiment_file
        self.output_dir = output_dir

        if self.experiment_file.endswith(".csv"):
            self.experiments = pd.read_csv(experiment_file)
            self.experiments.ice_sheet = self.experiments.ice_sheet.apply(lambda x: x.lower())
        elif self.experiment_file.endswith(".json"):
            self.experiments = pd.read_json(experiment_file).T
        else:
            raise ValueError("Experiment file must be a CSV or JSON file.")

        self.forcing_paths = get_all_filepaths(
            path=self.forcings,
            filetype="csv",
        )
        self.projection_paths = get_all_filepaths(
            path=self.projections,
            filetype="csv",
        )
        self.forcing_metadata = self._get_forcing_metadata()

    def merge_dataset(self):
        """
        Merges the forcing and projection files and creates a dataset.

        Returns:
            int: Returns 0 after successfully merging and saving the dataset.
        """
        full_dataset = pd.DataFrame()
        self.experiments["exp"] = self.experiments["exp"].apply(lambda x: x.lower())

        for i, projection in enumerate(
            tqdm(
                self.projection_paths,
                total=len(self.projection_paths),
                desc="Merging forcing & projection files",
            )
        ):
            # get experiment from projection filepath
            exp = projection.replace(".csv", "").split("/")[-1].split("_")[-1]

            # make sure cases match when doing table lookup

            # get AOGCM value from table lookup
            try:
                aogcm = self.experiments.loc[
                    (self.experiments.exp == exp.lower())
                    & (self.experiments.ice_sheet == self.ice_sheet.lower())
                ]["AOGCM"].values[0]
            except IndexError:
                aogcm = self.experiments.loc[self.experiments.exp == exp.lower()]["AOGCM"].values[0]
            proj_cmip_model = aogcm.split("_")[0]
            proj_pathway = aogcm.split("_")[-1]

            # names of CMIP models are slightly different, adjust based on AIS/GrIS directories
            if self.ice_sheet == "AIS":
                if proj_cmip_model == "csiro-mk3.6":
                    proj_cmip_model = "csiro-mk3-6-0"
                elif proj_cmip_model == "ipsl-cm5-mr":
                    proj_cmip_model = "ipsl-cm5a-mr"
                elif proj_cmip_model == "cnrm-esm2" or proj_cmip_model == "cnrm-cm6":
                    proj_cmip_model = f"{proj_cmip_model}-1"
            elif self.ice_sheet == "GrIS":
                if proj_cmip_model.lower() == "noresm1-m":
                    proj_cmip_model = "noresm1"
                elif proj_cmip_model.lower() == "ipsl-cm5-mr":
                    proj_cmip_model = "ipsl-cm5"
                elif proj_cmip_model.lower() == "access1-3":
                    proj_cmip_model = "access1.3"
                elif proj_cmip_model.lower() == "ukesm1-0-ll":
                    proj_cmip_model = "ukesm1-cm6"

            # get forcing file from table lookup that matches projection
            forcing_files = self.forcing_metadata.file.loc[
                (self.forcing_metadata.cmip_model == proj_cmip_model)
                & (self.forcing_metadata.pathway == proj_pathway)
            ]

            if forcing_files.empty:
                raise IndexError(
                    f"Could not find forcing file for {aogcm}. Check formatting of experiment file."
                )

            if len(forcing_files) > 1:
                forcings = pd.DataFrame()
                for file in forcing_files.values:
                    forcings = pd.concat(
                        [forcings, pd.read_csv(f"{self.forcings}/{file}.csv")], axis=1
                    )
            else:
                forcing_file = forcing_files.values[0]
                forcings = pd.read_csv(f"{self.forcings}/{forcing_file}.csv")

            # load forcing and projection datasets
            projections = pd.read_csv(projection)
            # if forcings are longer than projections, cut off the beginning of the forcings
            if len(forcings) > len(projections):
                forcings = forcings.iloc[-len(projections) :].reset_index(drop=True)

            # add forcings and projections together and add some metadata
            merged_dataset = pd.concat([forcings, projections], axis=1)
            merged_dataset["time"] = np.arange(1, len(merged_dataset) + 1)
            merged_dataset["cmip_model"] = proj_cmip_model
            merged_dataset["pathway"] = proj_pathway
            merged_dataset["exp"] = exp
            merged_dataset["id"] = i

            # now add to dataset with all forcing/projection pairs
            full_dataset = pd.concat([full_dataset, merged_dataset])

        # save the full dataset
        full_dataset.to_csv(f"{self.output_dir}/dataset.csv", index=False)

        return 0

    def merge_sectors(self, forcings_file=None, projections_file=None, save_dir=None):

        pass

    # def merge(self, inputs='pca', outputs='sectors', save_dir=None):
    #     if save_dir is None:
    #         save_dir = self.output_dir

    #     full_dataset = pd.DataFrame()
    #     self.experiments['exp'] = self.experiments['exp'].apply(lambda x: x.lower())

    #     if outputs.lower() == 'average' or outputs.lower() == 'sectors':
    #         paths = get_all_filepaths(path=self.projection_dir, filetype='nc', contains='rm', not_contains='historical')
    #         paths = [x for x in paths if 'ctrl' not in x]

    #     for i, projection in enumerate(tqdm(paths, total=len(paths), desc="Merging forcing & projection files")):
    #         # get experiment from projection filepath

    #         exp = projection.replace('.nc', '').replace('.csv', '').split('/')[-1].split('_')[-1]

    #         # make sure cases match when doing table lookup

    #         # get AOGCM value from table lookup
    #         try:
    #             aogcm = self.experiments.loc[(self.experiments.exp == exp.lower()) & (self.experiments.ice_sheet ==self.ice_sheet.lower())]['AOGCM'].values[0]
    #         except IndexError:
    #             aogcm = self.experiments.loc[self.experiments.exp == exp.lower()]['AOGCM'].values[0]
    #         proj_cmip_model = aogcm.split('_')[0]
    #         proj_pathway = aogcm.split('_')[-1]

    #         # names of CMIP models are slightly different, adjust based on AIS/GrIS directories
    #         if self.ice_sheet == 'AIS':
    #             if proj_cmip_model == 'csiro-mk3.6':
    #                 proj_cmip_model = 'csiro-mk3-6-0'
    #             elif proj_cmip_model == 'ipsl-cm5-mr':
    #                 proj_cmip_model = 'ipsl-cm5a-mr'
    #             elif proj_cmip_model == 'cnrm-esm2' or proj_cmip_model == 'cnrm-cm6':
    #                 proj_cmip_model = f'{proj_cmip_model}-1'
    #         elif self.ice_sheet == 'GrIS':
    #             if proj_cmip_model.lower() == 'noresm1-m':
    #                 proj_cmip_model = 'noresm1'
    #             elif proj_cmip_model.lower() == 'ipsl-cm5-mr':
    #                 proj_cmip_model = 'ipsl-cm5'
    #             elif proj_cmip_model.lower() == 'access1-3':
    #                 proj_cmip_model = 'access1'

    #                     # get forcing file from table lookup that matches projection
    #         forcing_files = self.forcing_metadata.file.loc[(self.forcing_metadata.cmip_model == proj_cmip_model) & (self.forcing_metadata.pathway == proj_pathway)]

    #         if forcing_files.empty:
    #             raise IndexError(f"Could not find forcing file for {aogcm}. Check formatting of experiment file.")

    #         if len(forcing_files) > 1:
    #             forcings = pd.DataFrame()
    #             for file in forcing_files.values:
    #                 forcings = pd.concat([forcings, pd.read_csv(f"{self.forcing_dir}/{file}.csv")], axis=1)
    #         else:
    #             forcing_file = forcing_files.values[0]
    #             forcings = pd.read_csv(f"{self.forcing_dir}/{forcing_file}.csv")

    #         # load forcing and projection datasets
    #         if 'nc' in projection:
    #             projections = xr.open_dataset(projection)
    #             projections = projections.to_dataframe()
    #             projections = projections[[x for x in projections.columns if 'ivaf' in x]]
    #             projections = projections / 1e9 / 362.5
    #         else:
    #             projections = pd.read_csv(projection)

    #         # if forcings are longer than projections, cut off the beginning of the forcings
    #         if len(forcings) > len(projections):
    #             forcings = forcings.iloc[-len(projections):].reset_index(drop=True)

    #          # add forcings and projections together and add some metadata
    #         merged_dataset = pd.concat([forcings, projections], axis=1)
    #         merged_dataset['cmip_model'] = proj_cmip_model
    #         merged_dataset['pathway'] = proj_pathway
    #         merged_dataset['exp'] = exp
    #         merged_dataset['id'] = i

    #         # now add to dataset with all forcing/projection pairs
    #         full_dataset = pd.concat([full_dataset, merged_dataset])

    #     # save the full dataset
    #     full_dataset.to_csv(f"{self.output_dir}/dataset.csv", index=False)

    def _get_forcing_metadata(self):
        """
        Retrieves the metadata for the forcing files.

        Returns:
            df (pandas.DataFrame): DataFrame containing the metadata for the forcing files.
                The DataFrame has three columns: 'file', 'cmip_model', and 'pathway'.
        """
        pairs = {}
        # loop through forcings, looking for cmip model and pathway
        for forcing in self.forcing_paths:
            if (
                forcing
                == r"/oscar/home/pvankatw/scratch/pca/AIS/forcings/PCA_IPSL-CM5A-MR_RCP26_salinity_8km_x_60m.csv"
            ):
                stop = "stop"
            forcing = forcing.replace(".csv", "").split("/")[-1]
            cmip_model = forcing.split("_")[1]

            # GrIS has MAR3.9 in name, ignore
            if cmip_model == "MAR3.9":
                cmip_model = forcing.split("_")[2]
            elif cmip_model.lower() == "gris":
                cmip_model = forcing.split("_")[2]

            if "rcp" in forcing.lower() or "ssp" in forcing.lower():
                for substring in forcing.lower().split("_"):
                    if "rcp" in substring or "ssp" in substring:
                        pathway = substring.lower()
                        if len(pathway.split("-")) > 1 and (
                            "rcp" in pathway.split("-")[-1] or "ssp" in pathway.split("-")[-1]
                        ):
                            if len(pathway.split("-")) > 2:
                                cmip_model = "-".join(pathway.split("-")[0:2])
                                pathway = pathway.split("-")[-1]
                            else:
                                cmip_model = pathway.split("-")[0]
                                pathway = pathway.split("-")[-1]
                        break
            else:
                pathway = "rcp85"
            if self.ice_sheet == "GrIS":
                if cmip_model.lower() == "noresm1-m":
                    cmip_model = "noresm1"
                elif cmip_model.lower() == "ipsl-cm5-mr":
                    cmip_model = "ipsl-cm5"
                elif cmip_model.lower() == "access1-3":
                    cmip_model = "access1.3"
                elif cmip_model.lower() == "ukesm1-0-ll":
                    cmip_model = "ukesm1-cm6"

            pairs[forcing] = [cmip_model.lower(), pathway.lower()]
        df = pd.DataFrame(pairs).T
        df = pd.DataFrame(pairs).T.reset_index()
        df.columns = ["file", "cmip_model", "pathway"]

        return df


def combine_gris_forcings(forcing_dir):
    """
    Combine GrIS forcings from multiple CMIP directories into a single NetCDF file.

    Parameters:
    - forcing_dir (str): The directory containing the GrIS forcings.

    Returns:
    - int: 0 indicating successful completion of the function.
    """

    atmosphere_dir = f"{forcing_dir}/GrIS/Atmosphere_Forcing/aSMB_observed/v1/"
    cmip_directories = next(os.walk(atmosphere_dir))[1]
    for cmip_dir in tqdm(
        cmip_directories, total=len(cmip_directories), desc="Processing CMIP directories"
    ):
        for var in [f"aSMB", f"aST"]:
            files = os.listdir(f"{atmosphere_dir}/{cmip_dir}/{var}")
            files = np.array([x for x in files if x.endswith(".nc")])
            years = np.array([int(x.replace(".nc", "").split("-")[-1]) for x in files])
            year_files = files[(years >= 2015) & (years <= 2100)]

            for i, file in enumerate(year_files):
                # first iteration, open dataset and store
                if i == 0:
                    dataset = xr.open_dataset(f"{atmosphere_dir}/{cmip_dir}/{var}/{file}")
                    for dim in ["nv", "nv4", "mapping"]:
                        try:
                            dataset = dataset.drop_dims(dim)
                        except:
                            pass
                    dataset = dataset.drop("mapping")
                    dataset = dataset.sel(x=dataset.x.values[::5], y=dataset.y.values[::5])
                    continue

                # following iterations, open dataset and concatenate
                data = xr.open_dataset(f"{atmosphere_dir}/{cmip_dir}/{var}/{file}")
                for dim in ["nv", "nv4"]:
                    try:
                        data = data.drop_dims(dim)
                    except:
                        pass
                data = data.drop("mapping")
                data = data.sel(x=data.x.values[::5], y=data.y.values[::5])
                # data['time'] = pd.to_datetime(year, format='%Y')
                dataset = xr.concat([dataset, data], dim="time")

            # Now you have the dataset with the files loaded and time dimension set
            dataset.to_netcdf(
                os.path.join(atmosphere_dir, cmip_dir, f"GrIS_{cmip_dir}_{var}_combined.nc")
            )

    return 0


def process_GrIS_atmospheric_sectors(forcing_directory, grid_file):

    start_time = time.time()
    path_to_forcings = f"Atmosphere_Forcing/aSMB_observed/v1/"
    af_directory = (
        f"{forcing_directory}/{path_to_forcings}"
        if not forcing_directory.endswith(path_to_forcings)
        else forcing_directory
    )

    # check to see if GrIS forcings have been combined
    filepaths = get_all_filepaths(path=af_directory, contains="combined", filetype="nc")
    if not filepaths:
        combine_gris_forcings(af_directory)
        filepaths = get_all_filepaths(path=af_directory, contains="combined", filetype="nc")
        if not filepaths:
            raise ValueError("No combined files found. Check combine_gris_forcings function.")

    aogcm_directories = os.listdir(af_directory)
    aogcm_directories = [x for x in aogcm_directories if "DS_Store" not in x and "README" not in x]

    sectors = _format_grid_file(grid_file)
    unique_sectors = np.unique(sectors)
    all_data = []
    for i, fp in enumerate(aogcm_directories):
        print("")
        print(f"Directory {i+1} / {len(aogcm_directories)}")
        print(f'Directory: {fp.split("/")[-1]}')
        print(f"Time since start: {(time.time()-start_time) // 60} minutes")

        files = get_all_filepaths(path=f"{af_directory}/{fp}", contains="combined", filetype="nc")
        if len(files) != 2:
            raise ValueError(f"There should only be 2 combined files in each firectory, see {fp}.")

        st_and_smb = []
        for file in files:
            dataset = xr.open_dataset(file, decode_times=False)
            dataset = convert_and_subset_times(dataset)

            # handle extra dimensions and variables
            try:
                dataset = dataset.drop_dims("nv4")
            except ValueError:
                pass

            for var in [
                "z_bnds",
                "lat",
                "lon",
                "mapping",
                "time_bounds",
                "lat2d",
                "lon2d",
                "polar_stereographic",
            ]:
                try:
                    dataset = dataset.drop(labels=[var])
                except ValueError:
                    pass
            if "z" in dataset.dims:
                dataset = dataset.mean(dim="z", skipna=True)

            dataset["sector"] = sectors

            formatted_aogcm = fp.rsplit("-", 1)
            formatted_aogcm = "_".join(formatted_aogcm).lower()

            aogcm_data = []
            for sector in unique_sectors:
                mask = dataset.sector == sector
                sector_averages = dataset.where(mask, drop=True).mean(dim=["x", "y"])
                sector_averages = sector_averages.to_dataframe()
                sector_averages["aogcm"] = formatted_aogcm
                sector_averages["year"] = np.arange(1, 87)
                sector_averages = sector_averages.reset_index(drop=True)
                aogcm_data.append(sector_averages)
            st_and_smb.append(pd.concat(aogcm_data))

        all_data.append(pd.concat(st_and_smb, axis=1))

    return pd.concat(all_data)


def process_AIS_atmospheric_sectors(forcing_directory, grid_file):

    ice_sheet = "AIS"

    start_time = time.time()
    path_to_forcings = "AIS/Atmosphere_Forcing/"
    af_directory = (
        f"{forcing_directory}/{path_to_forcings}"
        if not forcing_directory.endswith(path_to_forcings)
        else forcing_directory
    )

    filepaths = get_all_filepaths(path=af_directory, filetype="nc")
    filepaths = [f for f in filepaths if "1995-2100" in f]
    filepaths = [f for f in filepaths if "8km" in f]

    sectors = _format_grid_file(grid_file)
    unique_sectors = np.unique(sectors)
    all_data = []
    for i, fp in enumerate(filepaths):
        fp = r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing//AIS/Atmosphere_Forcing/miroc-esm-chem_rcp8.5/Regridded_8km/MIROC-ESM-CHEM_8km_anomaly_1995-2100.nc"
        print("")
        print(f"File {i+1} / {len(filepaths)}")
        print(f'File: {fp.split("/")[-1]}')
        print(f"Time since start: {(time.time()-start_time) // 60} minutes")

        dataset = xr.open_dataset(fp, decode_times=False)
        dataset = convert_and_subset_times(dataset)

        # handle extra dimensions and variables
        try:
            dataset = dataset.drop_dims("nv4")
        except ValueError:
            pass

        for var in ["z_bnds", "lat", "lon", "mapping", "time_bounds", "lat2d", "lon2d"]:
            try:
                dataset = dataset.drop(labels=[var])
            except ValueError:
                pass
        if "z" in dataset.dims:
            dataset = dataset.mean(dim="z", skipna=True)

        # dataset = dataset.transpose("time", "x", "y", ...)
        dataset["sector"] = sectors

        aogcm_data = []
        for sector in unique_sectors:
            mask = dataset.sector == sector
            sector_averages = dataset.where(
                mask,
            ).mean(dim=["x", "y"], skipna=True)
            sector_averages = sector_averages.to_dataframe()
            sector_averages["aogcm"] = fp.split("/")[-3].lower()
            sector_averages["year"] = np.arange(1, 87)
            sector_averages = sector_averages.reset_index(drop=True)
            aogcm_data.append(sector_averages)

        all_data.append(pd.concat(aogcm_data))
    atmospheric_df = pd.concat(all_data)
    atmospheric_df = atmospheric_df.loc[:, ~atmospheric_df.columns.duplicated()]
    return atmospheric_df


def process_AIS_oceanic_sectors(forcing_directory, grid_file):

    start_time = time.time()
    directory = (
        f"{forcing_directory}/Ocean_Forcing/"
        if not forcing_directory.endswith("Ocean_Forcing/")
        else forcing_directory
    )
    # Get all NC files that contain data from 1995-2100
    filepaths = get_all_filepaths(path=directory, filetype="nc")
    filepaths = [f for f in filepaths if "1995-2100" in f]
    filepaths = [f for f in filepaths if "8km" in f]

    # In the case of ocean forcings, use the filepaths of the files to determine
    # which directories need to be used for OceanForcing processing. Change to
    # those directories rather than individual files.
    aogcms = list(set([f.split("/")[-3] for f in filepaths]))
    filepaths = [f"{directory}/{aogcm}/" for aogcm in aogcms]

    # Useful progress prints
    print("Files to be processed...")
    print([f.split("/")[-2] for f in filepaths])

    sectors = _format_grid_file(grid_file)
    unique_sectors = np.unique(sectors)
    all_data = []
    for i, directory in enumerate(filepaths):
        print("")
        print(f"File {i+1} / {len(filepaths)}")
        print(f'File: {directory.split("/")[-1]}')
        print(f"Time since start: {(time.time()-start_time) // 60} minutes")

        files = os.listdir(f"{directory}/1995-2100/")
        if len(files) != 3:
            warnings.warn(f"Directory {directory} does not contain 3 files.")

        thermal_forcing_file = [f for f in files if "thermal_forcing" in f][0]
        salinity_file = [f for f in files if "salinity" in f][0]
        temperature_file = [f for f in files if "temperature" in f][0]

        thermal_forcing = xr.open_dataset(
            f"{directory}/1995-2100/{thermal_forcing_file}", decode_times=False
        )
        salinity = xr.open_dataset(f"{directory}/1995-2100/{salinity_file}", decode_times=False)
        temperature = xr.open_dataset(
            f"{directory}/1995-2100/{temperature_file}", decode_times=False
        )

        thermal_forcing = convert_and_subset_times(thermal_forcing)
        salinity = convert_and_subset_times(salinity)
        temperature = convert_and_subset_times(temperature)

        data = {
            "thermal_forcing": thermal_forcing,
            "salinity": salinity,
            "temperature": temperature,
        }
        aogcm_data = {"thermal_forcing": [], "salinity": [], "temperature": []}
        for name, dataset in data.items():
            # handle extra dimensions and variables
            try:
                dataset = dataset.drop_dims("nv4")
            except ValueError:
                pass

            for var in [
                "z_bnds",
                "lat",
                "lon",
                "mapping",
                "time_bounds",
                "lat2d",
                "lon2d",
                "polar_stereographic",
            ]:
                try:
                    dataset = dataset.drop(labels=[var])
                except ValueError:
                    pass
            if "z" in dataset.dims:
                dataset = dataset.mean(dim="z", skipna=True)

            try:
                dataset["sector"] = sectors
            except ValueError:
                dataset["time"] = np.arange(1, 87)
                dataset["sector"] = sectors

            for sector in unique_sectors:
                mask = dataset.sector == sector
                sector_averages = dataset.where(mask, drop=True).mean(dim=["x", "y"])
                sector_averages = sector_averages.to_dataframe()
                sector_averages["aogcm"] = _format_AIS_ocean_aogcm_name(
                    directory.split("/")[-2].lower()
                )
                sector_averages["year"] = np.arange(1, 87)
                sector_averages = sector_averages.reset_index(drop=True)
                aogcm_data[name].append(sector_averages)
        df = pd.concat(
            [
                pd.concat(aogcm_data["thermal_forcing"]),
                pd.concat(aogcm_data["salinity"]),
                pd.concat(aogcm_data["temperature"]),
            ],
            axis=1,
        )
        df = df.loc[:, ~df.columns.duplicated()]
        all_data.append(df)
    return pd.concat(all_data)


def process_GrIS_oceanic_sectors(forcing_directory, grid_file):

    start_time = time.time()
    path_to_forcing = "Ocean_Forcing/Melt_Implementation/v4/"
    forcing_directory = (
        f"{forcing_directory}/{path_to_forcing}"
        if not forcing_directory.endswith(path_to_forcing)
        else forcing_directory
    )

    aogcm_directories = os.listdir(forcing_directory)
    aogcm_directories = [x for x in aogcm_directories if "DS_Store" not in x and "README" not in x]

    sectors = _format_grid_file(grid_file)
    unique_sectors = np.unique(sectors)
    all_data = []
    for i, directory in enumerate(aogcm_directories):
        print("")
        print(f"Directory {i+1} / {len(aogcm_directories)}")
        print(f'Directory: {directory.split("/")[-1]}')
        print(f"Time since start: {(time.time()-start_time) // 60} minutes")

        files = os.listdir(f"{forcing_directory}/{directory}")
        if len(files) != 2:
            warnings.warn(f"Directory {directory} does not contain 2 files.")

        thermal_forcing_file = [f for f in files if "thermalforcing" in f.lower()][0]
        basin_runoff_file = [f for f in files if "basinrunoff" in f.lower()][0]

        thermal_forcing = xr.open_dataset(
            f"{forcing_directory}/{directory}/{thermal_forcing_file}", decode_times=False
        )
        basin_runoff = xr.open_dataset(
            f"{forcing_directory}/{directory}/{basin_runoff_file}", decode_times=False
        )

        # subset the dataset for 5km resolution (GrIS)
        if thermal_forcing.dims["x"] == 1681 and thermal_forcing.dims["y"] == 2881:
            thermal_forcing = thermal_forcing.sel(
                x=thermal_forcing.x.values[::5], y=thermal_forcing.y.values[::5]
            )
            basin_runoff = basin_runoff.sel(
                x=basin_runoff.x.values[::5], y=basin_runoff.y.values[::5]
            )

        thermal_forcing = convert_and_subset_times(thermal_forcing)
        basin_runoff = convert_and_subset_times(basin_runoff)

        data = {
            "thermal_forcing": thermal_forcing,
            "basin_runoff": basin_runoff,
        }
        aogcm_data = {
            "thermal_forcing": [],
            "basin_runoff": [],
        }
        for name, dataset in data.items():
            # handle extra dimensions and variables
            try:
                dataset = dataset.drop_dims("nv4")
            except ValueError:
                pass

            for var in [
                "z_bnds",
                "lat",
                "lon",
                "mapping",
                "time_bounds",
                "lat2d",
                "lon2d",
                "polar_stereographic",
            ]:
                try:
                    dataset = dataset.drop(labels=[var])
                except ValueError:
                    pass
            if "z" in dataset.dims:
                dataset = dataset.mean(dim="z", skipna=True)

            try:
                dataset["sector"] = sectors
            except ValueError:
                dataset["time"] = np.arange(1, 87)
                dataset["sector"] = sectors

            for sector in unique_sectors:
                mask = dataset.sector == sector
                sector_averages = dataset.where(mask, drop=True).mean(dim=["x", "y"])
                sector_averages = sector_averages.to_dataframe()
                sector_averages["aogcm"] = _format_GrIS_ocean_aogcm_name(directory)
                sector_averages["year"] = np.arange(1, 87)
                sector_averages = sector_averages.reset_index(drop=True)
                aogcm_data[name].append(sector_averages)
        df = pd.concat(
            [
                pd.concat(aogcm_data["thermal_forcing"]),
                pd.concat(aogcm_data["basin_runoff"]),
            ],
            axis=1,
        )
        df = df.loc[:, ~df.columns.duplicated()]
        all_data.append(df)
    return pd.concat(all_data)


def _format_grid_file(grid_file):
    if isinstance(grid_file, str):
        grids = xr.open_dataset(grid_file)  # .transpose('x', 'y',)
        sector_name = "sectors" if "8km" in grid_file.lower() else "ID"
    elif isinstance(grid_file, xr.Dataset):
        sector_name = "ID" if "Rignot" in grids.Description else "sectors"
    else:
        raise ValueError("grid_file must be a string or an xarray Dataset.")

    grids = grids.expand_dims(dim={"time": 86})
    sectors = grids[sector_name]
    grids = grids.transpose("time", "x", "y", ...)

    return sectors


def process_AIS_outputs(
    zenodo_directory,
):

    directory = (
        f"{zenodo_directory}/ComputedScalarsPaper/"
        if not zenodo_directory.endswith("ComputedScalarsPaper")
        else zenodo_directory
    )
    files = get_all_filepaths(directory, contains="ivaf_minus_ctrl_proj", filetype="nc")
    count = 0

    all_files_data = []
    for i, f in enumerate(files):
        exp = f.replace(".nc", "").split("/")[-1].split("_")[-1]
        model = f"{f.replace('.nc', '').split('/')[-1].split('_')[-3]}_{f.replace('.nc', '').split('/')[-1].split('_')[-2]}"

        dataset = xr.open_dataset(f, decode_times=False)

        if len(dataset.time) == 85:
            count += 1
            warnings.warn(
                f"{f.split('/')[-1]} does not contain 86 years. Inserting a copy into the first year."
            )

            # Copy the first entry
            first_entry = dataset.isel({"time": 0})
            # Assuming numeric coordinates, create a new coordinate value
            new_coord_value = (
                first_entry["time"].values - 1
            )  # Adjust this calculation based on your coordinate system
            # Set the new coordinate value for the copied entry
            first_entry["time"] = new_coord_value
            # Concatenate the new entry with the original dataset
            dataset = xr.concat([first_entry, dataset], dim="time")

        # dataset = convert_and_subset_times(dataset)
        all_sectors = []
        for sector in range(1, 19):
            sector_x_data = dataset[f"ivaf_sector_{sector}"].to_dataframe().reset_index(drop=True)
            sector_x_data.rename(columns={f"ivaf_sector_{sector}": "ivaf"}, inplace=True)
            sector_x_data["sector"] = sector
            sector_x_data["year"] = np.arange(1, 87)
            sector_x_data["id"] = f"{model}_sector{sector}"

            all_sectors.append(sector_x_data)
        full_dataset = pd.concat(all_sectors, axis=0)
        full_dataset["exp"] = exp
        full_dataset["model"] = model
        all_files_data.append(full_dataset)
    outputs = pd.concat(all_files_data)
    outputs["sle"] = outputs["ivaf"] / 362.5 / 1e9

    return outputs


def merge_datasets(forcings, projections, experiments_file, ice_sheet="AIS", export_directory=None):

    if isinstance(experiments_file, str):
        experiments = pd.read_csv(experiments_file)
    elif isinstance(experiments_file, pd.DataFrame):
        experiments = experiments_file
    else:
        raise ValueError("experiments_file must be a string or a pandas DataFrame.")

    experiments = experiments[experiments.ice_sheet == ice_sheet]
    projections = pd.merge(projections, experiments, on="exp", how="inner")
    formatting_function = (
        _format_AIS_forcings_aogcm_name if ice_sheet == "AIS" else _format_GrIS_forcings_aogcm_name
    )
    forcings["aogcm"] = forcings["aogcm"].apply(formatting_function)
    projections.rename(columns={"AOGCM": "aogcm"}, inplace=True)
    dataset = pd.merge(forcings, projections, on=["aogcm", "year", "sector"], how="inner")

    return dataset


def process_GrIS_outputs(
    zenodo_directory,
):

    directory = (
        f"{zenodo_directory}/v7_CMIP5_pub/"
        if not zenodo_directory.endswith("v7_CMIP5_pub")
        else zenodo_directory
    )
    files = get_all_filepaths(directory, contains="rm", not_contains="ctrl_proj", filetype="nc")
    files = [f for f in files if "historical" not in f]
    count = 0

    all_files_data = []
    for f in files:
        exp = f.replace(".nc", "").split("/")[-1].split("_")[-1]
        exp = exp.replace("_05", "")
        model = f"{f.replace('.nc', '').split('/')[-1].split('_')[-3]}_{f.replace('.nc', '').split('/')[-1].split('_')[-2]}"
        dataset = xr.open_dataset(f, decode_times=False)

        if len(dataset.time) == 85:
            count += 1
            warnings.warn(
                f"{f.split('/')[-1]} does not contain 86 years. Inserting a copy into the first year."
            )

            # Copy the first entry
            first_entry = dataset.isel({"time": 0})
            # Assuming numeric coordinates, create a new coordinate value
            new_coord_value = (
                first_entry["time"].values - 1
            )  # Adjust this calculation based on your coordinate system
            # Set the new coordinate value for the copied entry
            first_entry["time"] = new_coord_value
            # Concatenate the new entry with the original dataset
            dataset = xr.concat([first_entry, dataset], dim="time")

        sector_mapping = {"1": "no", "2": "ne", "3": "se", "4": "sw", "5": "cw", "6": "nw"}
        # dataset = convert_and_subset_times(dataset)
        all_sectors = []
        for sector in range(1, 7):
            var_name = f"ivaf_{sector_mapping[str(sector)]}"
            sector_x_data = dataset[var_name].to_dataframe().reset_index(drop=True)
            sector_x_data.rename(columns={var_name: "ivaf"}, inplace=True)
            sector_x_data["sector"] = sector
            sector_x_data["year"] = np.arange(1, 87)
            sector_x_data["id"] = f"{model}_{exp}_sector{sector}"

            all_sectors.append(sector_x_data)
        full_dataset = pd.concat(all_sectors, axis=0)
        full_dataset["exp"] = exp
        full_dataset["model"] = model
        all_files_data.append(full_dataset)
    outputs = pd.concat(all_files_data)
    outputs["sle"] = outputs["ivaf"] / 362.5 / 1e9

    return outputs


def process_sectors(
    ice_sheet,
    forcing_directory,
    grid_file,
    zenodo_directory,
    experiments_file,
    export_directory=None,
    overwrite=False,
):

    forcing_exists = os.path.exists(f"{export_directory}/forcings.csv")
    if not forcing_exists or (forcing_exists and overwrite):
        atmospheric_df = (
            process_AIS_atmospheric_sectors(forcing_directory, grid_file)
            if ice_sheet == "AIS"
            else process_GrIS_atmospheric_sectors(forcing_directory, grid_file)
        )
        atmospheric_df.to_csv(f"{export_directory}/{ice_sheet}_atmospheric.csv", index=False)
        # oceanic_df = (
        #     process_AIS_oceanic_sectors(forcing_directory, grid_file)
        #     if ice_sheet == "AIS"
        #     else process_GrIS_oceanic_sectors(forcing_directory, grid_file)
        # )
        # oceanic_df.to_csv(f"{export_directory}/{ice_sheet}_oceanic.csv", index=False)

        # atmospheric_df = pd.read_csv(f"{export_directory}/{ice_sheet}_atmospheric.csv")
        oceanic_df = pd.read_csv(f"{export_directory}/{ice_sheet}_oceanic.csv")
        # atmospheric_df = atmospheric_df[[x for x in atmospheric_df.columns if '.1' not in x]]
        # oceanic_df = oceanic_df[[x for x in oceanic_df.columns if '.1' not in x]]

        atmospheric_df = atmospheric_df.loc[:, ~atmospheric_df.columns.duplicated()]
        oceanic_df = oceanic_df.loc[:, ~oceanic_df.columns.duplicated()]
        forcings = pd.merge(
            atmospheric_df,
            oceanic_df,
            on=[
                "aogcm",
                "year",
                "sector",
            ],
            how="inner",
        )
        forcings.to_csv(f"{export_directory}/forcings.csv", index=False)
    else:
        forcings = pd.read_csv(f"{export_directory}/forcings.csv")

    projections_exists = os.path.exists(f"{export_directory}/projections.csv")
    if not projections_exists or (projections_exists and overwrite):
        projections = (
            process_AIS_outputs(
                zenodo_directory,
            )
            if ice_sheet == "AIS"
            else process_GrIS_outputs(
                zenodo_directory,
            )
        )
        projections.to_csv(f"{export_directory}/projections.csv", index=False)
    else:
        projections = pd.read_csv(f"{export_directory}/projections.csv")

    projections = projections.loc[:, ~projections.columns.duplicated()]
    dataset = merge_datasets(
        forcings,
        projections,
        experiments_file,
        ice_sheet,
    )
    dataset = dataset[[x for x in dataset.columns if ".1" not in x]]

    if export_directory is not None:
        dataset.to_csv(f"{export_directory}/dataset.csv", index=False)

    return dataset


def _format_AIS_ocean_aogcm_name(aogcm):
    aogcm = aogcm.lower()
    if (
        aogcm == "ipsl-cm5a-mr_rcp2.6"
        or aogcm == "ipsl-cm5a-mr_rcp8.5"
        or aogcm == "hadgem2-es_rcp8.5"
        or aogcm == "csiro-mk3-6-0_rcp8.5"
    ):
        aogcm = aogcm.replace(".", "")
    elif (
        aogcm == "cnrm-cm6-1_ssp585"
        or aogcm == "cnrm-esm2-1_ssp585"
        or aogcm == "cnrm-cm6-1_ssp126"
    ):
        aogcm = aogcm.replace("-1", "")
        aogcm = aogcm.replace("-", "_")
    elif aogcm == "ukesm1-0-ll_ssp585":
        aogcm = "ukesm1-0-ll"
    else:
        pass
    return aogcm


def _format_AIS_forcings_aogcm_name(aogcm):
    aogcm = aogcm.lower()
    if (
        aogcm == "noresm1-m_rcp2.6"
        or aogcm == "noresm1-m_rcp8.5"
        or aogcm == "miroc-esm-chem_rcp8.5"
        or aogcm == "ccsm4_rcp8.5"
    ):
        aogcm = aogcm.replace(".", "")
    elif aogcm == "csiro-mk3-6-0_rcp85":
        aogcm = "csiro-mk3.6_rcp85"
    elif aogcm == "ipsl-cm5a-mr_rcp26" or aogcm == "ipsl-cm5a-mr_rcp85":
        aogcm = aogcm.replace("a", "")
    else:
        pass
    return aogcm


def _format_GrIS_forcings_aogcm_name(aogcm):
    aogcm = aogcm.lower()
    if aogcm == "noresm1_rcp85":
        aogcm = "noresm1-m_rcp85"
    elif aogcm == "ukesm1-cm6_ssp585":
        aogcm = "ukesm1-0-ll_ssp585"
    else:
        pass
    return aogcm


def format_GrIS_atmospheric_aogcm_name(aogcm):
    modified_string = aogcm.rsplit("-", 1)
    return "_".join(modified_string).lower()


def _format_GrIS_ocean_aogcm_name(aogcm):
    aogcm = aogcm.lower()
    if aogcm == "access1-3_rcp8.5":
        aogcm = "access1.3_rcp85"
    elif aogcm == "csiro-mk3.6_rcp8.5":
        aogcm = aogcm.replace("8.5", "85")
    elif aogcm in (
        "hadgem2-es_rcp8.5",
        "ipsl-cm5-mr_rcp8.5",
        "miroc5_rcp2.6",
        "miroc5_rcp8.5",
        "miroc5_rcp8.5",
    ):
        aogcm = aogcm.replace(".", "")
    elif aogcm == "noresm1-m_rcp8.5":
        aogcm = "noresm1_rcp85"
    elif aogcm == "ukesm1-0-ll_ssp585":
        aogcm = "ukesm1-cm6_ssp585"
    else:
        pass
    return aogcm
