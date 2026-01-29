"""CMIP6 atmospheric and oceanic forcing processing.

This module provides process_atmos_forcings and process_ocean_forcings for
converting NetCDF CMIP6 forcings to sector-averaged DataFrames for use in
ice sheet emulation.
"""
from ise.data.forcings import ForcingFile
from ise.data.grids import GridFile
from ise.utils import functions as f
import numpy as np
import pandas as pd
import os
import warnings


def process_atmos_forcings(forcing_filepath: str, grid_filepath: str, aogcm_name: str) -> pd.DataFrame:
    """
    Process atmospheric CMIP6 forcing NetCDF into a sector-averaged DataFrame.

    Args:
        forcing_filepath (str): Path to the atmospheric forcing NetCDF file.
        grid_filepath (str): Path to the AIS sector grid NetCDF file.
        aogcm_name (str): Name of the atmosphere-ocean GCM (for labeling).

    Returns:
        pandas.DataFrame: Sector-averaged atmospheric forcings with columns
            including time, sector, aogcm, and year (years since 2014).
    """
    datafile = ForcingFile(
        ice_sheet="AIS",
        realm="atmos",
        filepath=forcing_filepath,
    )
    
    # load data
    datafile.load(decode_times = False)
    datafile.format_timestamps()
    gridfile = GridFile(ice_sheet="AIS", filepath=grid_filepath)
    gridfile.load()
    gridfile.format_grids()

    # trim dataset
    datafile.drop_vars(["nv4", "z_bnds", "lat", "lon", "mapping", "time_bounds", "lat2d", "lon2d", "bnds", "areacella"])
    datafile.data = datafile.data.rename(name_dict={"sectors": "sector"},)
    
    assert datafile._check_averaged_sectors(), "The data is not pre-averaged as expected."
    
    # Average over all sectors
    sector_averages = datafile.average_over_sector()
    
    # Format dataframe
    sector_averages = sector_averages.to_dataframe()
    sector_averages["aogcm"] = aogcm_name
    sector_averages = sector_averages.reset_index()
    sector_averages["year"] = sector_averages.time.apply(lambda x: x.year) - 2014
    
    return sector_averages

def process_ocean_forcings(tf_filepath: str, sal_filepath: str, ts_filepath: str, grid_filepath: str, aogcm_name: str):
    """
    Process oceanic CMIP6 forcing NetCDFs (thermal forcing, salinity, temperature) into a combined DataFrame.

    Args:
        tf_filepath (str): Path to thermal forcing NetCDF.
        sal_filepath (str): Path to salinity NetCDF.
        ts_filepath (str): Path to temperature NetCDF.
        grid_filepath (str): Path to the AIS sector grid NetCDF file.
        aogcm_name (str): Name of the atmosphere-ocean GCM (for labeling).

    Returns:
        pandas.DataFrame: Combined oceanic forcings (depth-averaged, sector-averaged)
            with thermal_forcing, salinity, temperature, sector, aogcm, and year.
    """
    gridfile = GridFile(ice_sheet="AIS", filepath=grid_filepath)
    gridfile.load()
    gridfile.format_grids()
    
    tffile = ForcingFile(ice_sheet="AIS", realm="ocean", filepath=tf_filepath, varname="thermal_forcing")
    salfile = ForcingFile(ice_sheet="AIS", realm="ocean", filepath=sal_filepath, varname="salinity")
    tempfile = ForcingFile(ice_sheet="AIS", realm="ocean", filepath=ts_filepath, varname="temperature")
    aogcm_data = {"thermal_forcing": [], "salinity": [], "temperature": []}
    
    for datafile in [tffile, salfile, tempfile, ]:
        datafile.load(decode_times = False)
        datafile.format_timestamps()
        datafile.drop_vars(["nv4", "z_bnds", "lat", "lon", "mapping", "time_bounds", "lat2d", "lon2d", "n_bounds", "bnds", "areacella", "nbounds"])
        datafile.aggregate_depth(method="mean")
        datafile.assign_sectors(gridfile)
        sectors = gridfile.get_sectors()
        unique_sectors = np.unique(sectors)
        
        for sector in unique_sectors:
            sector_averages = datafile.average_over_sector(sector_number=int(sector))
            sector_averages = sector_averages.to_dataframe()
            sector_averages["sector"] = sector
            sector_averages["aogcm"] = aogcm_name
            sector_averages = sector_averages.reset_index()
            sector_averages["year"] = np.arange(1, 87)
            # sector_averages["year"] = sector_averages.time.apply(lambda x: x.year) - 2014
            aogcm_data[datafile.varname].append(sector_averages)
    
    df = pd.concat(
        [
            pd.concat(aogcm_data["thermal_forcing"]),
            pd.concat(aogcm_data["salinity"]),
            pd.concat(aogcm_data["temperature"]),
        ],
        axis=1,
    )
    df = df.loc[:, ~df.columns.duplicated()]
    return df


if __name__ == "__main__":
    dataset_path = r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/CMIP/dataset"
    grid_file = r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Grid_Files/AIS_sectors_8km.nc"
    overwrite = False
    
    aogcms = os.listdir(dataset_path)
    
    
    for aogcm in aogcms:
        
        aogcm_path = os.path.join(dataset_path, aogcm)
        if not os.path.isdir(aogcm_path):
            continue
        print('\n\n')
        print("-" * 40)
        print(f"Processing AOGCM: {aogcm}")
        
        ssps = [x for x in os.listdir(aogcm_path) if x.startswith("ssp")]
        
        for ssp in ssps:
            ssp_path = os.path.join(aogcm_path, ssp)
            if not os.path.isdir(ssp_path):
                continue
            print('\n')
            print(f"  Scenario: {ssp}")
            
            # for realm in ["ocean"]:
            for realm in ["atmosphere", "ocean"]:
                realm_dir = os.path.join(ssp_path, realm)
                if not os.path.isdir(realm_dir):
                    continue
                print(f"    Realm: {realm}")

                if realm == "atmosphere":
                    if not overwrite and os.path.exists(os.path.join(realm_dir, ".complete")):
                        print(f"    Atmospheric forcings already processed for {aogcm} {ssp}. Skipping.")
                        continue
                    atmos_forcing_file = os.path.join(realm_dir, f"{aogcm}_{ssp}_anomaly.nc")
                    if os.path.exists(atmos_forcing_file):
                        try:
                            atmos_df = process_atmos_forcings(atmos_forcing_file, grid_file, aogcm)
                            atmos_df["ssp"] = ssp
                            atmos_df.to_csv(f"{realm_dir}/{aogcm}_{ssp}_atmospheric.csv", index=False)
                            with open(f"{realm_dir}/.complete", "w") as c:
                                pass
                        except Exception as e:
                            warnings.warn(f"Failed to process atmospheric forcings for {aogcm} {ssp}: {e}")
                            continue    
                    else:
                        warnings.warn(f"Atmospheric forcing file not found: {atmos_forcing_file}")
                        continue
                    
                elif realm == "ocean":
                    if not overwrite and os.path.exists(os.path.join(realm_dir, ".complete")):
                        print(f"    Oceanic forcings already processed for {aogcm} {ssp}. Skipping.")
                        continue
                    
                    ocean_dir = os.path.join(realm_dir, '1995-2100')
                    if not os.path.isdir(ocean_dir):
                        warnings.warn(f"Ocean data directory not found: {ocean_dir}. Skipping ocean processing.")
                        continue
                    files = os.listdir(ocean_dir)
                    try:
                        tf_file = os.path.join(ocean_dir, [f for f in files if "thermal_forcing" in f][0])
                        sal_file = os.path.join(ocean_dir, [f for f in files if "salinity" in f][0])
                        ts_file = os.path.join(ocean_dir, [f for f in files if "temperature" in f][0])
                    except IndexError:
                        warnings.warn(f"One or more ocean forcing files missing in {ocean_dir}. Skipping ocean processing.")
                        continue
                    
                    if all(os.path.exists(f) for f in [tf_file, sal_file, ts_file]):
                        try:
                            ocean_df = process_ocean_forcings(tf_file, sal_file, ts_file, grid_file, aogcm)
                            ocean_df["ssp"] = ssp
                            ocean_df.to_csv(f"{ocean_dir}/{aogcm}_{ssp}_oceanic.csv", index=False)
                            with open(f"{realm_dir}/.complete", "w") as c:
                                pass
                        except Exception as e:
                            warnings.warn(f"Failed to process ocean forcings for {aogcm} {ssp}: {e}")    
                            continue
                    else:
                        warnings.warn(f"One or more ocean forcing files not found for {aogcm} {ssp}. Skipping ocean processing.")
                        continue
                    

    # create combined CSVs for Atmosphere and Ocean
    aogcms = os.listdir(dataset_path)
    for aogcm in aogcms:
        aogcm_path = os.path.join(dataset_path, aogcm)
        if not os.path.isdir(aogcm_path):
            continue
        
        ssps = [x for x in os.listdir(aogcm_path) if x.startswith("ssp")]
        for ssp in ssps:
            
            data = []
            for realm in ["atmosphere", "ocean"]:
                realm_dir = os.path.join(aogcm_path, ssp, realm)
                if not os.path.isdir(realm_dir):
                    continue
                
                csv_file = os.path.join(realm_dir, f"{aogcm}_{ssp}_atmospheric.csv") if realm == "atmosphere" else os.path.join(realm_dir, "1995-2100", f"{aogcm}_{ssp}_oceanic.csv")
                if os.path.exists(csv_file):
                    data.append(csv_file)
                    
            if len(data) == 2:
                ssp_data = pd.concat([pd.read_csv(x) for x in data], axis=1)
                ssp_data = ssp_data.loc[:, ~ssp_data.columns.duplicated()]
                ssp_data.to_csv(os.path.join(aogcm_path, f"{aogcm}_{ssp}_combined.csv"), index=False)


    # Combine all Atmos/Ocean CSVs into one CSV
    data_paths = f.get_all_filepaths(
        path=dataset_path,
        filetype=".csv",
        contains="combined"
    )
    data = []
    for path in data_paths:
        df = pd.read_csv(path)
        print(f"Loaded {path} with shape {df.shape}")
        data.append(df)

    if data:
        combined = pd.concat(data, ignore_index=True)
        print(f"Combined DataFrame shape: {combined.shape}")
    else:
        combined = pd.DataFrame()
        print("No data files loaded; combined DataFrame is empty.")
    
    combined.to_csv(os.path.join(dataset_path, "CMIP6_AIS_combined.csv"), index=False)