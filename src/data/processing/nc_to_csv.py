directory = "/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/"
files = get_all_filepaths(path=directory, filetype='netcdf')

for i, nc_filepath in enumerate(nc_files):
    print(f"Iteration: {i}, File: {nc_filepath}")
    forcing = AtmosphereForcing(path=nc_filepath)
    forcing = forcing.aggregate_dims()
    forcing = forcing.save_as_df()

