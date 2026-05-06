import pandas as pd

from ise.data.feature_engineer import FeatureEngineer
from ise.data.process import process_sectors

ICE_SHEET = "AIS"


if ICE_SHEET == "GrIS":
    ISMIP6_FORCINGS = (
        r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/GrIS"
    )
    ISMIP6_GRIDS = r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Grid_Files/GrIS_Basins_Rignot_sectors_5km.nc"
    ISMIP6_OUTPUTS = (
        r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Zenodo_Outputs/v7_CMIP5_pub"
    )
else:
    ISMIP6_FORCINGS = (
        r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS"
    )
    ISMIP6_GRIDS = (
        r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Grid_Files/AIS_sectors_8km.nc"
    )
    ISMIP6_OUTPUTS = (
        r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Zenodo_Outputs/ComputedScalarsPaper"
    )


EXPORT_DIR = f"/oscar/home/pvankatw/research/ise/supplemental/dataset/tests/{ICE_SHEET}"

dataset = process_sectors(
    ice_sheet=ICE_SHEET,
    forcing_directory=ISMIP6_FORCINGS,
    grid_file=ISMIP6_GRIDS,
    zenodo_directory=ISMIP6_OUTPUTS,
    export_directory=EXPORT_DIR,
    overwrite=False,
    with_ctrl=False,
)


fe = FeatureEngineer(
    ice_sheet=ICE_SHEET,
    data=pd.read_csv(f"{EXPORT_DIR}/dataset.csv"),
    split_dataset=False,
    output_directory=None,
)

fe.data = fe.data.drop(columns="mrro_anomaly") if ICE_SHEET == "AIS" else fe.data
fe.add_model_characteristics()
fe.scale_data(save_dir=f"{EXPORT_DIR}/scalers/")
fe.add_lag_variables(lag=5)
fe.drop_outliers("quantile", "sle", quantiles=[0.005, 1 - 0.005])
fe.split_data(
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    output_directory=EXPORT_DIR,
)
