"""Pipeline functions for analyzing a trained network, including model testing, automatic generation
of descriptive plots, and analyzing the accuracy of uncertainty bounds."""
import os
from ise.models.timeseries import TimeSeriesEmulator
from ise.utils.data import combine_testing_results, load_ml_data, calculate_distribution_metrics
from ise.models.testing import test_pretrained_model, binned_sle_table
from ise.utils.models import load_model
from ise.visualization import Plotter
import pandas as pd

def analyze_model(data_directory: str, model_path: str, architecture: dict, model_class, 
                  time_series: bool=True, mc_dropout: bool=True, dropout_prob: float=0.1, 
                  mc_iterations: int=100, verbose: bool=True, save_directory: str=None):
    """Analyzes the performance of a pretrained model. Includes running model evaluation with test
    metrics on testing data, creating a results dataframe for easy analysis, and automatic generation
    of plots for both test cases and error analysis.

    Args:
        data_directory (str): Directory containing training and testing data.
        model_path (str): Path to the pretrained model. Must be a '.pt' model. Can also be a loaded model if the model was trained in the same script.
        architecture (dict): Architecture arguments used to train the model. 
        model_class (_type_): Model class used to train the model.
        time_series (bool, optional): Flag denoting wether model was trained with time-series data. Defaults to True.
        mc_dropout (bool, optional): Flag denoting whether the model was trained with MC dropout protocol. Defaults to True.
        dropout_prob (float, optional): Dropout probability in MC dropout protocol. Unused if mc_dropout=False. Defaults to 0.1.
        mc_iterations (int, optional): MC iterations to be used in testing. Unused if mc_dropout=False. Defaults to 100.
        verbose (bool, optional): Flag denoting whether to output logs to terminal. Defaults to True.
        save_directory (str, optional): Directory to save outputs. Defaults to None.
    """    

    if verbose:
        print('1/4: Calculating test metrics')
    metrics, preds, bounds = test_pretrained_model(
        model_path=model_path,
        model_class=model_class,
        architecture=architecture,
        data_directory=data_directory,
        time_series=time_series,
        mc_dropout=mc_dropout,
        dropout_prob=dropout_prob,
        mc_iterations=mc_iterations,
        verbose=False
    )
    
    
    if verbose:
        print('2/4: Creating results dataframe')
    dataset = combine_testing_results(
        data_directory=data_directory,
        preds=preds,
        bounds=bounds,
        save_directory=save_directory
    )
    _, _, test_features, _, _ = load_ml_data(data_directory)
    architecture['input_layer_size'] = test_features.shape[1]
    
    if isinstance(model_path, str):
        model = load_model(
            model_path=model_path,
            model_class=TimeSeriesEmulator,
            architecture=architecture,
            mc_dropout=mc_dropout,
            dropout_prob=dropout_prob,
        )
    else:
        model = model_path
    distribution_metrics = calculate_distribution_metrics(dataset)
    print(f"""True vs Predicted Distribution Closeness:
KL Divergence: {distribution_metrics['kl']}
JS Divergence: {distribution_metrics['js']}
""")
    
    binned = binned_sle_table(dataset, bins=5)
    
    print(f"\nBinned SLE Metrics: \n{binned}")

    if verbose:
        print('3/4: Generating plots')

    plotter = Plotter.Plotter(results_dataset=dataset, save_directory=save_directory)
    plotter.plot_ensemble(save=f'{save_directory}/ensemble_plot.png')
    plotter.plot_ensemble_mean(save=f'{save_directory}/ensemble_means.png')
    plotter.plot_distributions(year=2100, save=f'{save_directory}/distributions.png')
    plotter.plot_histograms(year=2100, save=f'{save_directory}/histograms.png')
    plotter.plot_callibration(alpha=0.5, save=f'{save_directory}/callibration.png')

    if verbose:
        print('4/4: Generating example test cases')

    test_case_dir = f"{save_directory}/test cases/"
    if not os.path.exists(path=test_case_dir):
        os.mkdir(test_case_dir)

    plotter.plot_test_series(
        model=model, data_directory=data_directory, save_directory=f'{save_directory}/test cases/'
    )

    return metrics, preds, plotter
