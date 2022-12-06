from ise.models.training.iterative import rnn_architecture_test, lag_sequence_test
from ise.models.testing.pretrained import test_pretrained_model
from ise.pipelines import training
from ise.pipelines import feature_engineering
from ise.models import timeseries

processed_output_files = r"/users/pvankatw/emulator/ise/data/datasets/processed_output_files/"
processed_data = r"/users/pvankatw/emulator/old/"

if __name__ == '__main__':
    # lag_sequence_test(
    #     lag_array=[1, 3, 5, 10],
    #     sequence_array=[3, 5, 10],
    #     iterations=5
    # )
    
    # run_network()

    # rnn_architecture_test(
    #     rnn_layers_array=[12], 
    #     hidden_nodes_array=[128, 256, 512], 
    #     iterations=5,
    #     )
    
    # feature_engineering.feature_engineer(data_dir=processed_output_files, time_series=True, export_directory=processed_data,)
    
    model_name = "04-12-2022 12.41.39.pt"
    metrics, preds = test_pretrained_model(
        model_path=f"/users/pvankatw/emulator/ise/models/pretrained/{model_name}", 
        model_class=timeseries.TimeSeriesEmulator,
        architecture={'num_rnn_layers': 12,'num_rnn_hidden': 256,},
        data_directory=processed_data,
        time_series=True,
    )
    # import pandas as pd
    # pd.DataFrame(preds).to_csv(r'preds.csv')

stop = ''


# TODO: Make visualization package, module, and pipeline


