"""Plotting functions for analyzing and comparing the ensembles from simulated data and emulated data. Generally compares distributions at given years or plots all paths over the entire series."""


import numpy as np

np.random.seed(10)
import matplotlib.pyplot as plt
import pandas as pd
from ise.utils.data import (
    group_by_run,
    get_uncertainty_bands,
    create_distribution,
    kl_divergence,
)
import seaborn as sns


class UncertaintyBounds:
    def __init__(self, data, confidence="95", quantiles=[0.05, 0.95]):
        self.data = data
        (
            self.mean,
            self.sd,
            self.upper_ci,
            self.lower_ci,
            self.upper_q,
            self.lower_q,
        ) = get_uncertainty_bands(data, confidence=confidence, quantiles=quantiles)


def plot_ensemble(
    dataset: pd.DataFrame,
    uncertainty: str = "quantiles",
    column: str = None,
    condition: str = None,
    save: str = None,
    cache: dict = None,
):
    """Generates a plot of the comparison of ensemble results from the true simulations and the predicted emulation.
    Adds uncertainty bounds and plots them side-by-side.

    Args:
        dataset (pd.DataFrame): testing results dataframe, result from [ise.utils.data.combine_testing_results](https://brown-sciml.github.io/ise/ise/utils/data.html#combine_testing_results).
        uncertainty (str, optional): Type of uncertainty for creating bounds, must be in [quantiles, confidence]. Defaults to 'quantiles'.
        column (str, optional): Column to subset on, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/utils/data.html#group_by_run). Defaults to None.
        condition (str, optional): Condition to subset with, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/utils/data.html#group_by_run). Can be int, str, float, etc. Defaults to None.
        save (str, optional): Path to save plot. Defaults to None.
        cache (dict, optional): Cached results from previous calculation, used internally in [ise.visualization.Plotter](https://brown-sciml.github.io/ise/ise/visualization/Plotter.html#Plotter). Defaults to None.
    """

    if cache is None:
        all_trues, all_preds, scenarios = group_by_run(
            dataset, column=column, condition=condition
        )
        (
            mean_true,
            true_sd,
            true_upper_ci,
            true_lower_ci,
            true_upper_q,
            true_lower_q,
        ) = get_uncertainty_bands(
            all_trues,
        )
        (
            mean_pred,
            pred_sd,
            pred_upper_ci,
            pred_lower_ci,
            pred_upper_q,
            pred_lower_q,
        ) = get_uncertainty_bands(
            all_preds,
        )
    else:
        all_trues = cache["true_sle_runs"]
        all_preds = cache["pred_sle_runs"]
        t = cache["true_bounds"]
        p = cache["pred_bounds"]
        mean_true, true_upper_ci, true_lower_ci, true_upper_q, true_lower_q = (
            t.mean,
            t.upper_ci,
            t.lower_ci,
            t.upper_q,
            t.lower_q,
        )
        mean_pred, pred_upper_ci, pred_lower_ci, pred_upper_q, pred_lower_q = (
            p.mean,
            p.upper_ci,
            p.lower_ci,
            p.upper_q,
            p.lower_q,
        )

    true_df = pd.DataFrame(all_trues).transpose()
    pred_df = pd.DataFrame(all_preds).transpose()

    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True, sharex=True)
    axs[0].plot(true_df)
    axs[0].plot(mean_true, "r-", linewidth=4, label="Mean")
    axs[1].plot(pred_df)
    axs[1].plot(mean_pred, "r-", linewidth=4, label="Mean")
    if uncertainty and uncertainty.lower() == "confidence":
        axs[0].plot(true_upper_ci, "b--", linewidth=3, label="5/95% Confidence (True)")
        axs[0].plot(true_lower_ci, "b--", linewidth=3)
        axs[1].plot(
            pred_upper_ci, "b--", linewidth=3, label="5/95% Confidence (Predicted)"
        )
        axs[1].plot(pred_lower_ci, "b--", linewidth=3)

    elif uncertainty and uncertainty.lower() == "quantiles":
        axs[0].plot(
            pred_upper_q, "b--", linewidth=3, label="5/95% Percentile (Predicted)"
        )
        axs[0].plot(pred_lower_q, "b--", linewidth=3)
        axs[1].plot(true_upper_q, "b--", linewidth=3, label="5/95% Percentile (True)")
        axs[1].plot(true_lower_q, "b--", linewidth=3)

    elif uncertainty and uncertainty.lower() == "both":
        axs[0].plot(true_upper_ci, "r--", linewidth=2, label="5/95% Confidence (True)")
        axs[0].plot(true_lower_ci, "r--", linewidth=2)
        axs[1].plot(
            pred_upper_ci, "b--", linewidth=2, label="5/95% Confidence (Predicted)"
        )
        axs[1].plot(pred_lower_ci, "b--", linewidth=2)
        axs[1].plot(
            pred_upper_q, "o--", linewidth=2, label="5/95% Percentile (Predicted)"
        )
        axs[1].plot(pred_lower_q, "o--", linewidth=2)
        axs[0].plot(true_upper_q, "k--", linewidth=2, label="5/95% Percentile (True)")
        axs[0].plot(true_lower_q, "k--", linewidth=2)

    elif uncertainty and uncertainty.lower() not in ["confidence", "quantiles"]:
        raise AttributeError(
            f"uncertainty argument must be in ['confidence', 'quantiles'], received {uncertainty}"
        )

    axs[0].title.set_text("True")
    axs[0].set_ylabel("True SLE (mm)")
    axs[1].title.set_text("Predicted")
    plt.xlabel("Years since 2015")
    if column is not None and condition is not None:
        plt.suptitle(f"Time Series of ISM Ensemble - where {column} == {condition}")
    else:
        plt.suptitle(f"Time Series of ISM Ensemble")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.legend()


    # TODO: FileNotFoundError: [Errno 2] No such file or directory: 'None/ensemble_plot.png'
    if save:
        plt.savefig(save)


def plot_ensemble_mean(
    dataset: pd.DataFrame,
    uncertainty: str = False,
    column=None,
    condition=None,
    save=None,
    cache=None,
):
    """Generates a plot of the mean sea level contribution from the true simulations and the predicted emulation.

    Args:
        dataset (pd.DataFrame): testing results dataframe, result from [ise.utils.data.combine_testing_results](https://brown-sciml.github.io/ise/ise/utils/data.html#combine_testing_results).
        uncertainty (str, optional): Type of uncertainty for creating bounds. If not None/False, must be in [quantiles, confidence]. Defaults to 'quantiles'.
        column (str, optional): Column to subset on, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/utils/data.html#group_by_run). Defaults to None.
        condition (str, optional): Condition to subset with, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/utils/data.html#group_by_run). Can be int, str, float, etc. Defaults to None.
        save (str, optional): Path to save plot. Defaults to None.
        cache (dict, optional): Cached results from previous calculation, used internally in [ise.visualization.Plotter](https://brown-sciml.github.io/ise/ise/visualization/Plotter.html#Plotter). Defaults to None.
    """

    if cache is None:
        all_trues, all_preds, scenarios = group_by_run(
            dataset, column=column, condition=condition
        )
        (
            mean_true,
            true_sd,
            true_upper_ci,
            true_lower_ci,
            true_upper_q,
            true_lower_q,
        ) = get_uncertainty_bands(
            all_trues,
        )
        (
            mean_pred,
            pred_sd,
            pred_upper_ci,
            pred_lower_ci,
            pred_upper_q,
            pred_lower_q,
        ) = get_uncertainty_bands(
            all_preds,
        )
    else:
        all_trues = cache["true_sle_runs"]
        all_preds = cache["pred_sle_runs"]
        t = cache["true_bounds"]
        p = cache["pred_bounds"]
        mean_true, true_upper_ci, true_lower_ci, true_upper_q, true_lower_q = (
            t.mean,
            t.upper_ci,
            t.lower_ci,
            t.upper_q,
            t.lower_q,
        )
        mean_pred, pred_upper_ci, pred_lower_ci, pred_upper_q, pred_lower_q = (
            p.mean,
            p.upper_ci,
            p.lower_ci,
            p.upper_q,
            p.lower_q,
        )

    plt.figure(figsize=(15, 6))
    plt.plot(mean_true, label="True Mean SLE")
    plt.plot(mean_pred, label="Predicted Mean SLE")

    if uncertainty and uncertainty.lower() == "confidence":
        plt.plot(true_upper_ci, "r--", linewidth=2, label="5/95% Percentile (True)")
        plt.plot(true_lower_ci, "r--", linewidth=2)
        plt.plot(
            pred_upper_ci, "b--", linewidth=2, label="5/95% Percentile (Predicted)"
        )
        plt.plot(pred_lower_ci, "b--", linewidth=2)

    elif uncertainty and uncertainty.lower() == "quantiles":
        plt.plot(pred_upper_q, "r--", linewidth=2, label="5/95% Confidence (Predicted)")
        plt.plot(pred_lower_q, "r--", linewidth=2)
        plt.plot(true_upper_q, "b--", linewidth=2, label="5/95% Confidence (True)")
        plt.plot(true_lower_q, "b--", linewidth=2)

    elif uncertainty and uncertainty.lower() == "both":
        plt.plot(true_upper_ci, "r--", linewidth=2, label="5/95% Percentile (True)")
        plt.plot(true_lower_ci, "r--", linewidth=2)
        plt.plot(
            pred_upper_ci, "b--", linewidth=2, label="5/95% Percentile (Predicted)"
        )
        plt.plot(pred_lower_ci, "b--", linewidth=2)
        plt.plot(pred_upper_q, "o--", linewidth=2, label="5/95% Confidence (Predicted)")
        plt.plot(pred_lower_q, "o--", linewidth=2)
        plt.plot(true_upper_q, "k--", linewidth=2, label="5/95% Confidence (True)")
        plt.plot(true_lower_q, "k--", linewidth=2)

    elif uncertainty and uncertainty.lower() not in ["confidence", "quantiles"]:
        raise AttributeError(
            f"uncertainty argument must be in ['confidence', 'quantiles'], received {uncertainty}"
        )

    else:
        pass

    if column is not None and condition is not None:
        plt.suptitle(f"ISM Ensemble Mean SLE over Time - where {column} == {condition}")
    else:
        plt.suptitle(f"ISM Ensemble Mean over Time")
    plt.xlabel("Years since 2015")
    plt.ylabel("Mean SLE (mm)")
    plt.legend()

    if save:
        plt.savefig(save)


def plot_distributions(
    dataset: pd.DataFrame,
    year: int = 2100,
    column: str = None,
    condition: str = None,
    save: str = None,
    cache: dict = None,
):
    """Generates a plot of comparison of distributions at a given time slice (year) from the true simulations and the predicted emulation.

    Args:
        dataset (pd.DataFrame): testing results dataframe, result from [ise.utils.data.combine_testing_results](https://brown-sciml.github.io/ise/ise/utils/data.html#combine_testing_results).
        year (int, optional): Distribution year (time slice). Defaults to 2100.
        column (str, optional): Column to subset on, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/utils/data.html#group_by_run). Defaults to None.
        condition (str, optional): Condition to subset with, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/utils/data.html#group_by_run). Can be int, str, float, etc. Defaults to None.
        save (str, optional): Path to save plot. Defaults to None.
        cache (dict, optional): Cached results from previous calculation, used internally in [ise.visualization.Plotter](https://brown-sciml.github.io/ise/ise/visualization/Plotter.html#Plotter). Defaults to None.
    """

    if cache is None:
        all_trues, all_preds, scenarios = group_by_run(
            dataset, column=column, condition=condition
        )
    else:
        all_trues = cache["true_sle_runs"]
        all_preds = cache["pred_sle_runs"]

    true_dist, true_support = create_distribution(year=year, dataset=all_trues)
    pred_dist, pred_support = create_distribution(year=year, dataset=all_preds)
    plt.figure(figsize=(15, 8))
    plt.plot(true_support, true_dist, label="True")
    plt.plot(pred_support, pred_dist, label="Predicted")
    plt.title(
        f"Distribution Comparison at year {year}, KL Divergence: {kl_divergence(pred_dist, true_dist):0.3f}"
    )
    plt.xlabel("SLE (mm)")
    plt.ylabel("Probability")
    plt.legend()
    if save:
        plt.savefig(save)


def plot_histograms(
    dataset: pd.DataFrame,
    year: int = 2100,
    column: str = None,
    condition: str = None,
    save: str = None,
    cache: dict = None,
):
    """Generates a plot of comparison of histograms at a given time slice (year) from the true simulations and the predicted emulation.

    Args:
        dataset (pd.DataFrame): testing results dataframe, result from [ise.utils.data.combine_testing_results](https://brown-sciml.github.io/ise/ise/utils/data.html#combine_testing_results).
        year (int, optional): Histogram year (time slice). Defaults to 2100.
        column (str, optional): Column to subset on, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/utils/data.html#group_by_run). Defaults to None.
        condition (str, optional): Condition to subset with, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/utils/data.html#group_by_run). Can be int, str, float, etc. Defaults to None.
        save (str, optional): Path to save plot. Defaults to None.
        cache (dict, optional): Cached results from previous calculation, used internally in [ise.visualization.Plotter](https://brown-sciml.github.io/ise/ise/visualization/Plotter.html#Plotter). Defaults to None.
    """
    if cache is None:
        all_trues, all_preds, scenarios = group_by_run(
            dataset, column=column, condition=condition
        )

    else:
        all_trues = cache["true_sle_runs"]
        all_preds = cache["pred_sle_runs"]

    fig = plt.figure(figsize=(15, 8))
    ax1 = plt.subplot(
        1,
        2,
        1,
    )
    sns.histplot(
        all_preds[:, year - 2101],
        label="Predicted Distribution",
        color="blue",
        alpha=0.3,
    )
    plt.legend()
    plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    sns.histplot(
        all_trues[:, year - 2101], label="True Distribution", color="red", alpha=0.3
    )
    plt.suptitle(f"Histograms of Predicted vs True SLE at year {year}")
    plt.ylabel("")
    plt.legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig(save)
