"""Processing functions for ISMIP6 atmospheric, oceanic, and ice-collapse forcings found in 
the [Globus ISMIP6 Archive](https://app.globus.org/file-manager?origin_id=ad1a6ed8-4de0-4490-93a9-8258931766c7&origin_path=%2F)
"""
import time
import numpy as np
import pandas as pd
import xarray as xr
from ise.utils.utils import get_all_filepaths, check_input

np.random.seed(10)


def process_forcings(
    forcing_directory: str,
    grids_directory: str,
    export_directory: str,
    to_process: str = "all",
    verbose: bool = False,
) -> None:
    """Perform preprocessing of atmospheric, oceanic, and ice-collapse forcing from [Globus ISMIP6
    Directory](https://app.globus.org/file-manager?origin_id=ad1a6ed8-4de0-4490-93a9-8258931766c7
    &origin_path=%2F).

    Args:
        forcing_directory (str): Directory containing grid data files
        export_directory (str): Directory to export processed files.
        to_process (str, optional): Forcings to process, options=[all,
            atmosphere, ocean, ice_collapse],
        verbose (bool, optional): Flag denoting whether to output logs
            in terminal, defaults to False
    defaults to 'all'
    """
    # check inputs
    to_process_options = ["all", "atmosphere", "ocean", "ice_collapse"]
    if isinstance(to_process, str):
        if to_process.lower() not in to_process_options:
            raise ValueError(
                f"to_process arg must be in [{to_process_options}], \
                received {to_process}"
            )
    elif isinstance(to_process, list):
        to_process_valid = all(s in to_process_options for s in to_process)
        if not to_process_valid:
            raise ValueError(
                f"to_process arg must be in [{to_process_options}], \
                received {to_process}"
            )

    if to_process.lower() == "all":
        to_process = ["atmosphere", "ocean", "ice_collapse"]

    if verbose:
        print("Processing...")

    # Process each using respective functions
    curr_time = time.time()
    if "atmosphere" in to_process:
        af_directory = f"{forcing_directory}/Atmosphere_Forcing/"
        aggregate_atmosphere(
            af_directory,
            grids_directory,
            export=export_directory,
        )
        if verbose:
            prev_time, curr_time = curr_time, time.time()
            curr_time = time.time()
            print(
                f"Finished processing atmosphere, Total Running Time: \
                {(curr_time - prev_time) // 60} minutes"
            )

    if "ocean" in to_process:
        of_directory = f"{forcing_directory}/Ocean_Forcing/"
        aggregate_ocean(
            of_directory,
            grids_directory,
            export=export_directory,
        )
        if verbose:
            prev_time, curr_time = curr_time, time.time()
            curr_time = time.time()
            print(
                f"Finished processing ocean, Total Running Time: \
                {(curr_time - prev_time) // 60} minutes"
            )

    if "ice_collapse" in to_process:
        ice_directory = f"{forcing_directory}/Ice_Shelf_Fracture"
        aggregate_icecollapse(
            ice_directory,
            grids_directory,
            export=export_directory,
        )
        if verbose:
            prev_time, curr_time = curr_time, time.time()
            curr_time = time.time()
            print(
                f"Finished processing ice_collapse, Total Running Time: \
                {(curr_time - prev_time) // 60} minutes"
            )
    if verbose:
        print(f"Finished. Data exported to {export_directory}")


class GridSectors:
    """Class for grid sector data and attributes."""

    def __init__(
        self,
        grids_dir: str,
        grid_size: int = 8,
        filetype: str = "nc",
        format_index: bool = True,
    ):
        """Initializes class and opens/stores data.

        Args:
            grids_dir (str): Directory containing grid data.
            grid_size (int, optional): KM grid size to be used, must be
                [4, 8, 16, 32] defaults to 8
            filetype (str, optional): Filetype of data, must be in [nc,
                csv], defaults to 'nc'
            format_index (bool, optional): Flag denoting whether to fix
                index so that join works appropriately, defaults to True
        """
        check_input(grid_size, [4, 8, 16, 32])
        check_input(filetype.lower(), ["nc", "csv"])
        self.grids_dir = grids_dir

        if filetype.lower() == "nc":
            self.path = self.grids_dir + f"sectors_{grid_size}km.nc"
            self.data = xr.open_dataset(self.path, decode_times=False)
            self._to_dataframe()
            if format_index:
                self._format_index()
        elif filetype.lower() == "csv":
            self.path = self.grids_dir + f"sector_{grid_size}.csv"
            self.data = pd.read_csv(self.path)
        else:
            raise NotImplementedError('Only "NetCDF" and "CSV" are currently supported')

    def _to_dataframe(self):
        """Converts self.data to dataframe.

        Returns:
            self: GridSectors: GridSectors object with data as
            dataframe.
        """
        if not isinstance(self, pd.DataFrame):
            self.data = self.data.to_dataframe()
        return self

    def _format_index(self):
        """Formats indices from 0 to 761 so merge with forcing data is possible.

        Returns:
            self: GridSectors: GridSectors object with indices
            formatted.
        """
        index_array = list(np.arange(0, 761))
        self.data.index = pd.MultiIndex.from_product(
            [index_array, index_array], names=["x", "y"]
        )
        return self


class AtmosphereForcing:
    """Class for atmospheric forcing data and attributes."""

    def __init__(self, path: str):
        """Initializes class and opens/stores data.

        Args:
            path (str): Filepath to atmospheric forcing file.
        """
        self.forcing_type = "atmosphere"
        self.path = path
        self.aogcm = path.split("/")[-3]  # 3rd to last folder in directory structure

        if path[-2:] == "nc":
            self.data = xr.open_dataset(self.path, decode_times=False)
            self.datatype = "NetCDF"

        elif path[-3:] == "csv":
            self.data = pd.read_csv(
                self.path,
            )
            self.datatype = "CSV"

    def aggregate_dims(
        self,
    ):
        """Aggregates over excess dimesions, particularly over time or grid cells.

        Returns:
            self: AtmosphereForcing: AtmosphereForcing object with
            dimensions reduced.
        """
        dims = self.data.dims
        if "time" in dims:
            self.data = self.data.mean(dim="time")
        if "nv4" in dims:
            self.data = self.data.mean(dim="nv4")
        return self

    def add_sectors(self, grids: GridSectors):
        """Adds information on which sector each grid cell belongs to. This is done through a merge
        of grid cell data with a sectors NC file.

        Args:
            grids (GridSectors): GridSectors class containing grid cell
                information and attributes

        Returns:
            self: AtmosphereForcing: AtmosphereForcing class with
            sectors added.
        """
        for col in ["lon_bnds", "lat_bnds", "lat2d", "lon2d"]:
            try:  
                self.data = self.data.drop(labels=[col])
            except ValueError:
                pass
        self.data = self.data.to_dataframe().reset_index(level="time", drop=True)
        # merge forcing data with grid data
        self.data = pd.merge(
            self.data, grids.data, left_index=True, right_index=True, how="outer"
        )
        return self


class OceanForcing:
    """Class for oceanic forcing data and attributes."""

    def __init__(self, aogcm_dir: str):
        """Initializes class and opens/stores data.

        Args:
            aogcm_dir (str): Directory path to oceanic forcings.
        """
        self.forcing_type = "ocean"
        self.path = f"{aogcm_dir}/1995-2100/"
        self.aogcm = aogcm_dir.split("/")[
            -2
        ]  # 3rd to last folder in directory structure

        # Load all data: thermal forcing, salinity, and temperature
        files = get_all_filepaths(path=self.path, filetype="nc")
        for file in files:
            if "salinity" in file:
                self.salinity_data = xr.open_dataset(file)
            elif "thermal_forcing" in file:
                self.thermal_forcing_data = xr.open_dataset(file)
            elif "temperature" in file:
                self.temperature_data = xr.open_dataset(file)
            else:
                pass

    def aggregate_dims(
        self,
    ):
        """Aggregates over excess dimesions, particularly over time or grid cells.

        Returns:
            self: AtmosphereForcing: AtmosphereForcing object with
            dimensions reduced.
        """
        dims = self.data.dims
        if "z" in dims:
            self.data = self.data.mean(dim="time")
        if "nbounds" in dims:
            self.data = self.data.mean(dim="nv4")
        return self

    def add_sectors(self, grids: GridSectors):
        """Adds information on which sector each grid cell belongs to. This is done through a merge
        of grid cell data with a sectors NC file.

        Args:
            grids (GridSectors): GridSectors class containing grid cell
                information and attributes

        Returns:
            self: OceanForcing: OceanForcing class with sectors added.
        """
        self.salinity_data = self.salinity_data.drop(labels=["z_bnds", "lat", "lon"])
        # Take mean over all z values (only found in oceanic forcings)
        self.salinity_data = self.salinity_data.mean(
            dim="z", skipna=True
        ).to_dataframe()
        self.salinity_data = self.salinity_data.reset_index(
            level="time",
        )
        # merge with grid data
        self.salinity_data = pd.merge(
            self.salinity_data,
            grids.data,
            left_index=True,
            right_index=True,
            how="outer",
        )
        self.salinity_data["year"] = self.salinity_data["time"].apply(lambda x: x.year)
        self.salinity_data = self.salinity_data.drop(columns=["time", "mapping"])

        self.thermal_forcing_data = self.thermal_forcing_data.drop(labels=["z_bnds"])
        self.thermal_forcing_data = (
            self.thermal_forcing_data.mean(dim="z", skipna=True)
            .to_dataframe()
            .reset_index(
                level="time",
            )
        )
        self.thermal_forcing_data = pd.merge(
            self.thermal_forcing_data,
            grids.data,
            left_index=True,
            right_index=True,
            how="outer",
        )
        self.thermal_forcing_data["year"] = self.thermal_forcing_data["time"].apply(
            lambda x: x.year
        )
        self.thermal_forcing_data = self.thermal_forcing_data.drop(
            columns=["time", "mapping"]
        )

        self.temperature_data = self.temperature_data.drop(labels=["z_bnds"])
        self.temperature_data = (
            self.temperature_data.mean(dim="z", skipna=True)
            .to_dataframe()
            .reset_index(
                level="time",
            )
        )
        self.temperature_data = pd.merge(
            self.temperature_data,
            grids.data,
            left_index=True,
            right_index=True,
            how="outer",
        )
        self.temperature_data["year"] = self.temperature_data["time"].apply(
            lambda x: x.year
        )
        self.temperature_data = self.temperature_data.drop(columns=["time", "mapping"])

        return self


class IceCollapse:
    """Class for ice collapse forcing data and attributes."""

    def __init__(self, aogcm_dir: str):
        """Initializes class and opens/stores data.

        Args:
            aogcm_dir (str): Directory path to ice collapse forcings
                forcings.
        """
        self.forcing_type = "ice_collapse"
        self.path = f"{aogcm_dir}"
        self.aogcm = aogcm_dir.split("/")[-2]  # last folder in directory structure

        # Load all data: thermal forcing, salinity, and temperature
        files = get_all_filepaths(path=self.path, filetype="nc")
        files = [f for f in files if "8km" in f]
        if len(files) > 1:  # if there is a "v2" file in the directory, use that one
            for file in files:
                if "v2" in file:
                    self.data = xr.open_dataset(file)
                else:
                    pass
        else:
            self.data = xr.open_dataset(files[0])

    def add_sectors(self, grids: GridSectors):
        """Adds information on which sector each grid cell belongs to. This is done through a merge
        of grid cell data with a sectors NC file.

        Args:
            grids (GridSectors): GridSectors class containing grid cell
                information and attributes

        Returns:
            self: IceCollapse: IceCollapse class with sectors added.
        """
        for col in ["lon_bnds", "lat_bnds", "lat2d", "lon2d"]:
            try:  
                self.data = self.data.drop(labels=[col])
            except ValueError:
                pass
        self.data = self.data.to_dataframe().reset_index(level="time", drop=False)
        self.data = pd.merge(
            self.data, grids.data, left_index=True, right_index=True, how="outer"
        )
        self.data["year"] = self.data["time"].apply(lambda x: x.year)
        self.data = self.data.drop(
            columns=[
                "time",
                "mapping",
                "lat_x",
                "lat_y",
                "lon_x",
                "lon_y",
            ]
        )
        return self


def aggregate_by_sector(path: str, grids_dir: str):
    """Takes a atmospheric forcing dataset, adds sector numbers to it,
    and gets aggregate data based on sector and year. Returns atmospheric
    forcing data object.

    Args:
        path (str): Filepath to atmospheric forcing nc file.
        grids_dir (str): Directory containing grid data.

    Returns:
        forcing: AtmosphereForcing: AtmosphereForcing instance with aggregated data
    """
    # Load grid data with 8km grid size
    print("")

    # Load in Atmospheric forcing data and add the sector numbers to it
    if "Atmosphere" in path:
        grids = GridSectors(
            grids_dir,
            grid_size=8,
        )
        forcing = AtmosphereForcing(path=path)

    elif "Ocean" in path:
        grids = GridSectors(grids_dir, grid_size=8, format_index=False)
        forcing = OceanForcing(aogcm_dir=path)

    elif "Ice" in path:
        grids = GridSectors(
            grids_dir,
            grid_size=8,
        )
        forcing = IceCollapse(path)

    forcing = forcing.add_sectors(grids)

    # Group the dataset and assign aogcm column to the aogcm simulation
    if forcing.forcing_type in ("atmosphere", "ice_collapse"):
        forcing.data = forcing.data.groupby(["sectors", "year"]).mean()
        forcing.data["aogcm"] = forcing.aogcm.lower()
    elif forcing.forcing_type == "ocean":
        forcing.salinity_data = forcing.salinity_data.groupby(
            ["sectors", "year"]
        ).mean()
        forcing.salinity_data["aogcm"] = forcing.aogcm.lower()
        forcing.temperature_data = forcing.temperature_data.groupby(
            ["sectors", "year"]
        ).mean()
        forcing.temperature_data["aogcm"] = forcing.aogcm.lower()
        forcing.thermal_forcing_data = forcing.thermal_forcing_data.groupby(
            ["sectors", "year"]
        ).mean()
        forcing.thermal_forcing_data["aogcm"] = forcing.aogcm.lower()

    return forcing


def aggregate_atmosphere(
    directory: str,
    grids_directory: str,
    export: str,
):
    """Loops through every NC file in the provided forcing directory
    from 1995-2100 and applies the aggregate_by_sector function. It then outputs
    the concatenation of all processed data to all_data.csv

    Args:
        directory (str): Directory containing forcing files
        grids_directory (str): Directory containing grid data.
        export (str): Directory to export output files.
    """

    start_time = time.time()

    # Get all NC files that contain data from 1995-2100
    filepaths = get_all_filepaths(path=directory, filetype="nc")
    filepaths = [f for f in filepaths if "1995-2100" in f]
    filepaths = [f for f in filepaths if "8km" in f]

    # Useful progress prints
    print("Files to be processed...")
    print([f.split("/")[-1] for f in filepaths])

    # Loop over each file specified above
    all_data = pd.DataFrame()
    for i, fp in enumerate(filepaths):
        print("")
        print(f"File {i+1} / {len(filepaths)}")
        print(f'File: {fp.split("/")[-1]}')
        print(f"Time since start: {(time.time()-start_time) // 60} minutes")

        # attach the sector to the data and groupby sectors & year
        forcing = aggregate_by_sector(fp, grids_dir=grids_directory)

        # Handle files that don't have mrro_anomaly input (ISPL RCP 85?)
        try:
            forcing.data["mrro_anomaly"]
        except KeyError:
            forcing.data["mrro_anomaly"] = np.nan

        # Keep selected columns and output each file individually
        forcing.data = forcing.data[
            [
                "pr_anomaly",
                "evspsbl_anomaly",
                "mrro_anomaly",
                "smb_anomaly",
                "ts_anomaly",
                "regions",
                "aogcm",
            ]
        ]

        # meanwhile, create a concatenated dataset
        all_data = pd.concat([all_data, forcing.data])

        print(" -- ")

    if export:
        all_data.to_csv(f"{export}/atmospheric_forcing.csv")


def aggregate_ocean(
    directory,
    grids_directory,
    export,
):
    """Loops through every NC file in the provided forcing directory
    from 1995-2100 and applies the aggregate_by_sector function. It then outputs
    the concatenation of all processed data to all_data.csv.

    Args:
        directory (str): Directory containing forcing files
        grids_directory (str): Directory containing grid data.
        export (str): Directory to export output files.
    """
    start_time = time.time()

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

    # Loop over each directory specified above
    salinity_data = pd.DataFrame()
    temperature_data = pd.DataFrame()
    thermal_forcing_data = pd.DataFrame()
    for i, fp in enumerate(filepaths):
        print("")
        print(f"Directory {i+1} / {len(filepaths)}")
        print(f'Directory: {fp.split("/")[-2]}')
        print(f"Time since start: {(time.time()-start_time) // 60} minutes")

        # attach the sector to the data and groupby sectors & year
        forcing = aggregate_by_sector(fp, grids_dir=grids_directory)

        forcing.salinity_data = forcing.salinity_data[["salinity", "regions", "aogcm"]]
        forcing.temperature_data = forcing.temperature_data[
            ["temperature", "regions", "aogcm"]
        ]
        forcing.thermal_forcing_data = forcing.thermal_forcing_data[
            ["thermal_forcing", "regions", "aogcm"]
        ]

        # meanwhile, create a concatenated dataset
        salinity_data = pd.concat([salinity_data, forcing.salinity_data])
        temperature_data = pd.concat([temperature_data, forcing.temperature_data])
        thermal_forcing_data = pd.concat(
            [thermal_forcing_data, forcing.thermal_forcing_data]
        )

    print(" -- ")

    if export:
        salinity_data.to_csv(export + "/salinity.csv")
        temperature_data.to_csv(export + "/temperature.csv")
        thermal_forcing_data.to_csv(export + "/thermal_forcing.csv")


def aggregate_icecollapse(
    directory,
    grids_directory,
    export,
):
    """Loops through every NC file in the provided forcing directory
    from 1995-2100 and applies the aggregate_by_sector function. It then outputs
    the concatenation of all processed data to all_data.csv.

    Args:
        directory (str): Directory containing forcing files
        grids_directory (str): Directory containing grid data.
        export (str): Directory to export output files.
    """
    start_time = time.time()

    # Get all NC files that contain data from 1995-2100
    filepaths = get_all_filepaths(path=directory, filetype="nc")

    # In the case of ocean forcings, use the filepaths of the files to determine
    # which directories need to be used for processing. Change to
    # those directories rather than individual files.
    aogcms = list(set([f.split("/")[-2] for f in filepaths]))
    filepaths = [f"{directory}/{aogcm}/" for aogcm in aogcms]
    # filepaths = [f for f in filepaths if "8km" in f]

    # Useful progress prints
    print("Files to be processed...")
    print([f.split("/")[-2] for f in filepaths])

    # Loop over each directory specified above
    ice_collapse = pd.DataFrame()
    for i, fp in enumerate(filepaths):
        print("")
        print(f"Directory {i+1} / {len(filepaths)}")
        print(f'Directory: {fp.split("/")[-2]}')
        print(f"Time since start: {(time.time()-start_time) // 60} minutes")

        # attach the sector to the data and groupby sectors & year
        forcing = aggregate_by_sector(fp, grids_dir=grids_directory)

        forcing.data = forcing.data[["mask", "regions", "aogcm"]]

        # meanwhile, create a concatenated dataset
        ice_collapse = pd.concat([ice_collapse, forcing.data])

    print(" -- ")

    if export:
        ice_collapse.to_csv(export + "/ice_collapse.csv")
