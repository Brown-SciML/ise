"""Pipeline functions for analyzing a trained network, including model testing, automatic generation
of descriptive plots, and analyzing the accuracy of uncertainty bounds."""
import json
import os

import pandas as pd

from ise.evaluation._tests import binned_sle_table, test_pretrained_model
from ise.evaluation.plots import SectorPlotter
from ise.models.sector import TimeSeriesEmulator
from ise.utils.functions import (
    calculate_distribution_metrics,
    combine_testing_results,
    load_ml_data,
    load_model,
)


def analyze_model(
    data_directory: str,
    model_path: str,
    architecture: dict,
    model_class,
    time_series: bool = True,
    mc_dropout: bool = True,
    dropout_prob: float = 0.1,
    mc_iterations: int = 100,
    verbose: bool = True,
    save_directory: str = None,
    plot: bool = True,
):
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
        plot (bool, optional): Flag denoting whether to output plots. Defaults to True.
    """

    # TODO: write functionality to save a file with the model metadata used (path, architecture, etc.)
    # Test the pretrained model to generate metrics and predictions
    if verbose:
        print("1/4: Calculating test metrics")
    metrics, preds, sd = test_pretrained_model(
        model_path=model_path,
        model_class=model_class,
        architecture=architecture,
        data_directory=data_directory,
        time_series=time_series,
        mc_dropout=mc_dropout,
        dropout_prob=dropout_prob,
        mc_iterations=mc_iterations,
        verbose=False,
    )

    if save_directory:
        # save metrics, preds, and bounds
        with open(f"{save_directory}/metrics.txt", "w") as metrics_file:
            metrics_file.write(json.dumps(metrics))

        pd.Series(preds, name="preds").to_csv(f"{save_directory}/NN_predictions.csv")
        # with open(f"{save_directory}/bounds.txt", "w") as bounds_file:
        #     bounds_file.write(json.dumps(bounds))

    # Create the results dataframe, which is the undummified version
    if verbose:
        print("2/4: Creating results dataframe")
    dataset = combine_testing_results(
        data_directory=data_directory,
        preds=preds,
        sd=sd,
        save_directory=save_directory,
    )
    _, _, test_features, _, _ = load_ml_data(data_directory)
    architecture["input_layer_size"] = test_features.shape[1]

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

    # Calculate the distribution metrics, tables, etc.
    distribution_metrics = calculate_distribution_metrics(dataset)
    print(
        f"""True vs Predicted Distribution Closeness:
KL Divergence: {distribution_metrics['kl']}
JS Divergence: {distribution_metrics['js']}
"""
    )

    binned = binned_sle_table(dataset, bins=5)
    if save_directory:
        binned.to_csv(f"{save_directory}/binned_sle_table.csv")

    print(f"\nBinned SLE Metrics: \n{binned}")

    if verbose:
        print("3/4: Generating plots")

    if plot:
        plotter = SectorPlotter(results_dataset=dataset, save_directory=save_directory)
        plotter.plot_ensemble(save=f"{save_directory}/ensemble_plot.png")
        plotter.plot_ensemble_mean(save=f"{save_directory}/ensemble_means.png")
        plotter.plot_distributions(year=2100, save=f"{save_directory}/distributions.png")
        plotter.plot_histograms(year=2100, save=f"{save_directory}/histograms.png")
        plotter.plot_callibration(alpha=0.5, save=f"{save_directory}/callibration.png")

    if verbose:
        print("4/4: Generating example test cases")

    if plot:
        test_case_dir = f"{save_directory}/test cases/"
        if not os.path.exists(path=test_case_dir):
            os.mkdir(test_case_dir)
        plotter.plot_test_series(
            model=model,
            data_directory=data_directory,
            save_directory=f"{save_directory}/test cases/",
        )

    return metrics, preds, plotter
