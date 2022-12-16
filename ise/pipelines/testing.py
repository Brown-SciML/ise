import os
from ise.models.timeseries import TimeSeriesEmulator
from ise.utils.data import combine_testing_results, load_ml_data, calculate_distribution_metrics
from ise.models.testing import test_pretrained_model
from ise.utils.models import load_model
from ise.visualization import Plotter

def analyze_model(data_directory, model_path, architecture, model_class, time_series=True, mc_dropout=True,
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
        verbose=False
    )

    if verbose:
        print('2/4: Creating results dataframe')
    dataset = combine_testing_results(
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
    distribution_metrics = calculate_distribution_metrics(dataset)
    print(f"""True vs Predicted Distribution Closeness:
KL Divergence: {distribution_metrics['kl']}
JS Divergence: {distribution_metrics['js']}
""")

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
