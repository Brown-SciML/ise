import os
import pandas as pd
from ise.models.timeseries import TimeSeriesEmulator
from ise.utils.data import combine_testing_results, load_ml_data
from ise.models.testing import test_pretrained_model
from ise.utils.models import load_model
from ise.visualization import Plotter
from ise.visualization.testing import plot_test_series

def analyze_model(data_directory, model_path, architecture, 
                       model_class, time_series=True, mc_dropout=True, 
                       dropout_prob=0.1, mc_iterations=100, verbose=True, save_directory=None):
    
    if verbose:
        print('1/4: Calculating test metrics')
    metrics, preds = test_pretrained_model(
        model_path=model_path,
        model_class=model_class,
        architecture=architecture,
        data_directory=data_directory,
        time_series=time_series,
        mc_dropout=mc_dropout,
        dropout_prob=dropout_prob, 
        mc_iterations=mc_iterations, 
        verbose=False)
    
    print(f"""Test Metrics
MSE: {metrics['MSE']:0.6f}
MAE: {metrics['MAE']:0.6f}
RMSE: {metrics['RMSE']:0.6f}
R2: {metrics['R2']:0.6f}""")
    
    if verbose:
        print('2/4: Creating results dataframe')
    combine_testing_results(
        data_directory=data_directory, 
        preds=preds, 
        save_directory=save_directory
    )
    
    _, _, test_features, _, _ = load_ml_data(data_directory)
     
    architecture['input_layer_size'] = test_features.shape[1]
    model = load_model(
        model_path=model_path, 
        model_class=TimeSeriesEmulator, 
        architecture=architecture,
        mc_dropout=mc_dropout,
        dropout_prob=dropout_prob,
    )

    if verbose:
        print('3/4: Generating plots')
        
    dataset = pd.read_csv(f"{save_directory}/results.csv")
    p = Plotter.Plotter(results_dataset=dataset, save_directory=save_directory)
    p.plot_ensemble(save=f'{save_directory}/ensemble_plot.png')
    p.plot_ensemble_mean(save=f'{save_directory}/ensemble_means.png')
    p.plot_distributions(year=2100, save=f'{save_directory}/distributions.png')
    p.plot_histograms(year=2100, save=f'{save_directory}/histograms.png')
    p.plot_test_series(model=model, data_directory=data_directory, save_directory=save_directory)
    p.plot_callibration(alpha=0.5, save=f'{save_directory}/ensemble_means.png')

    # TODO: Add calculation for KL and JS Divergence here.. need access to create_distribution outputs in Plotter, make them an attribute    
    # print(f"KL Divergence: {kl_divergence()}")
    
    if verbose:
        print('4/4: Generating example test cases')
        
    test_case_dir = f"{save_directory}/test cases/"
    if not os.path.exists(path=test_case_dir):
        os.mkdir(test_case_dir)
    
    plot_test_series(
        model=model, 
        data_directory=data_directory, 
        time_series=time_series, 
        approx_dist=mc_dropout, 
        mc_iterations=mc_iterations,
        confidence='95', 
        draws='random', 
        k=20, 
        save_directory=save_directory
    )