from ise.data.process import process_sectors
from ise.data.feature_engineer import FeatureEngineer
import pandas as pd

dataset = process_sectors(
    ice_sheet=r'GrIS',
    forcing_directory=r'/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/GrIS', 
    grid_file=r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Grid_Files/GrIS_Basins_Rignot_sectors_5km.nc", 
    zenodo_directory=r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Zenodo_Outputs/v7_CMIP5_pub", 
    experiments_file=r"/users/pvankatw/research/ise/ise/utils/data_files/ismip6_experiments_updated.csv",
    export_directory=r"/users/pvankatw/research/ise/dataset/GrIS_slc",
    overwrite=False, 
    with_ctrl=False, 
)


fe = FeatureEngineer(
        ice_sheet='GrIS',
        data=pd.read_csv(r"/users/pvankatw/research/ise/dataset/GrIS_slc/dataset.csv"),
        split_dataset=False,
        output_directory=None,
)

# fe.data = fe.data.drop(columns='mrro_anomaly')
fe.add_model_characteristics()
fe.scale_data(save_dir=r"/users/pvankatw/research/ise/dataset/GrIS_slc/scalers/")
fe.add_lag_variables(lag=5)
fe.drop_outliers('quantile', 'sle', quantiles=[0.005, 1-0.005])
fe.split_data(
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    output_directory=r"/users/pvankatw/research/ise/dataset/GrIS_slc/"
)