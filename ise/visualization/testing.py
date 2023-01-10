from ise.utils.data import load_ml_data
import pandas as pd
import seaborn as sns
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)


def plot_test_series(
    model,
    data_directory,
    time_series,
    approx_dist=True,
    mc_iterations=100,
    confidence="95",
    draws="random",
    k=10,
    save_directory=None,
):
    _, _, test_features, test_labels, test_scenarios = load_ml_data(
        data_directory, time_series=time_series
    )

    sectors = list(set(test_features.sectors))
    sectors.sort()

    if draws == "random":
        data = random.sample(test_scenarios, k=k)
    elif draws == "first":
        data = test_scenarios[:k]
    else:
        raise ValueError(f"draws must be in [random, first], received {draws}")

    for scen in data:
        single_scenario = scen
        test_model = single_scenario[0]
        test_exp = single_scenario[2]
        test_sector = single_scenario[1]
        single_test_features = torch.tensor(
            np.array(
                test_features[
                    (test_features[test_model] == 1)
                    & (test_features[test_exp] == 1)
                    & (test_features.sectors == test_sector)
                ],
                dtype=np.float64,
            ),
            dtype=torch.float,
        )
        single_test_labels = np.array(
            test_labels[
                (test_features[test_model] == 1)
                & (test_features[test_exp] == 1)
                & (test_features.sectors == test_sector)
            ],
            dtype=np.float64,
        )
        preds, means, upper_ci, lower_ci, quantiles = model.predict(
            single_test_features,
            approx_dist=approx_dist,
            mc_iterations=mc_iterations,
            confidence=confidence,
        )  # TODO: this doesn't work with traditional

        if not approx_dist:
            plt.figure(figsize=(15, 8))
            plt.plot(single_test_labels, "r-", label="True")
            plt.plot(preds, "b-", label="Predicted")
            plt.xlabel("Time (years since 2015)")
            plt.ylabel("SLE (mm)")
            plt.title(
                f"Model={test_model}, Exp={test_exp}, sector={sectors.index(test_sector)+1}"
            )
            plt.legend()
            if save_directory:
                plt.savefig(f"{save_directory}/{test_model}_{test_exp}_test_sector.png")
        else:
            preds = pd.DataFrame(preds).transpose()
            plt.figure(figsize=(15, 8))
            plt.plot(
                preds,
                alpha=0.2,
            )
            plt.plot(means, "b-", label="Predicted")
            plt.plot(upper_ci, "k-", label=f"{confidence}% CI")
            plt.plot(
                lower_ci,
                "k-",
            )
            plt.plot(quantiles[0, :], "k--", label=f"Quantiles")
            plt.plot(quantiles[1, :], "k--")
            plt.plot(
                lower_ci,
                "k-",
            )
            plt.plot(single_test_labels, "r-", label="True")

            plt.xlabel("Time (years since 2015)")
            plt.ylabel("SLE (mm)")
            plt.title(
                f"Model={test_model}, Exp={test_exp}, sector={sectors.index(test_sector)+1}"
            )
            plt.legend()
            if save_directory:
                plt.savefig(
                    f'{save_directory}/{test_model.replace("-", "_")}_{test_exp}_test_sector.png'
                )


def plot_callibration(
    dataset, column=None, condition=None, color_by=None, alpha=0.2, save=None
):

    # TODO: Add ability to subset multiple columns and conditions. Not needed now so saving for later...
    if column is None and condition is None:
        subset = dataset
    elif column is not None and condition is not None:
        subset = dataset[(dataset[column] == condition)]
    else:
        raise ValueError(
            "Column and condition type must be the same (None & None, not None & not None)."
        )

    plt.figure(figsize=(15, 8))
    sns.scatterplot(data=subset, x="true", y="pred", hue=color_by, alpha=alpha)
    plt.plot(
        [min(subset.true), max(subset.true)],
        [min(subset.true), max(subset.true)],
        "r-",
    )

    # TODO: Add density plots (below)
    # sns.kdeplot(data=subset, x='true', y='pred', hue=color_by, fill=True)
    # plt.plot([min(subset.true),max(subset.true)], [min(subset.true),max(subset.true)], 'r-',)

    # TODO: add plotly export
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("Callibration Plot")

    if color_by is not None:
        plt.legend()

    if save:
        plt.savefig(save)
