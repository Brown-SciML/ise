from ise.pipelines import processing, feature_engineering, training

forcing_directory = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/"
ismip6_output_directory = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Zenodo_Outputs/"
processed_forcing_outputs = r"/users/pvankatw/emulator/untracked_folder/processed_forcing_outputs/"
ml_data_directory = r"/users/pvankatw/emulator/untracked_folder/ml_data_directory/"
saved_model_path = r"/users/pvankatw/emulator/untracked_folder/saved_models/"

master, inputs, outputs = processing.process_data(
    forcing_directory=forcing_directory, 
    ismip6_output_directory=ismip6_output_directory,
    export_directory=processed_forcing_outputs,
)

feature_engineering.feature_engineer(
    data_directory=processed_forcing_outputs, 
    time_series=True, 
    export_directory=ml_data_directory,
)

model, metrics, test_preds = training.train_timeseries_network(
    data_directory=ml_data_directory, 
    save_model=saved_model_path,
    verbose=True, 
    epochs=10, 
)

