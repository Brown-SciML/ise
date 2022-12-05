from ise.pipelines import training

data_directory = r'/users/pvankatw/emulator/old'
model, metrics, test_preds = training.train_timeseries_network(data_directory=data_directory, verbose=True, epochs=1)

# TODO: Make testing package, module, and pipeline
# Make visualization package, module, and pipeline