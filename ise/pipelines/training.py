""""Pipeline functions for training various kinds of emulators, including traditional and time-based
neural networks as well as a gaussian process-based emulator."""
from ise.models.training.Trainer import Trainer
from ise.models.traditional import ExploratoryModel
from ise.models.timeseries import TimeSeriesEmulator
from ise.models.traditional.ExploratoryModel import ExploratoryModel
from ise.models.gp.GaussianProcess import GP
from torch import nn
import pandas as pd
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

np.random.seed(10)
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA


def train_timeseries_network(
    data_directory: str,
    architecture: dict = None,
    epochs: int = 20,
    batch_size: int = 100,
    model_class=TimeSeriesEmulator,
    loss=nn.MSELoss(),
    mc_dropout: bool = True,
    dropout_prob: float = 0.1,
    tensorboard: bool = False,
    save_model: str = False,
    performance_optimized: bool = False,
    verbose: bool = False,
    tensorboard_comment: str = None,
):
    """Pipeline function for training a time-series neural network emulator. Loads processed data, trains
    network and saves it to the desired location. Outputs test metrics and predictions.

    Args:
        data_directory (str): Directory containing training and testing data.
        architecture (dict, optional): Architecture arguments used to train the model. Defaults to None.
        epochs (int, optional): Number of epochs to train the network. Defaults to 20.
        batch_size (int, optional): Batch size of training. Defaults to 100.
        model_class (ModelClass, optional): Model class used to train the model. Defaults to TimeSeriesEmulator.
        loss (nn.Loss, optional): PyTorch loss to be used in training. Defaults to nn.MSELoss().
        mc_dropout (bool, optional): Flag denoting whether the model was trained with MC dropout protocol. Defaults to True.
        dropout_prob (float, optional): Dropout probability in MC dropout protocol. Unused if mc_dropout=False. Defaults to 0.1.
        tensorboard (bool, optional): Flag denoting whether to output logs to Tensorboard. Defaults to False.
        save_model (str, optional): Directory to save model. Can be False if model is not to be saved. Defaults to False.
        performance_optimized (bool, optional): Flag denoting whether to optimize the training for faster performace. Leaves out in-loop validation testing, extra logging, etc. Defaults to False.
        verbose (bool, optional): Flag denoting whether to output logs to terminal. Defaults to False.
        tensorboard_comment (str, optional): Comment to be included in the tensorboard logs. Defaults to None.

    Returns:
        tuple: Tuple containing [model, metrics, test_preds], or the model, test metrics, and test predictions.
    """

    if verbose:
        print("1/3: Loading processed data...")
    try:
        test_features = pd.read_csv(f"{data_directory}/ts_test_features.csv")
        train_features = pd.read_csv(f"{data_directory}/ts_train_features.csv")
        test_labels = pd.read_csv(f"{data_directory}/ts_test_labels.csv")
        train_labels = pd.read_csv(f"{data_directory}/ts_train_labels.csv")
        scenarios = pd.read_csv(
            f"{data_directory}/ts_test_scenarios.csv"
        ).values.tolist()
    except FileNotFoundError:
        raise FileNotFoundError(
            'Files not found. Format must be in format "ts_train_features.csv"'
        )

    data_dict = {
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels,
    }

    trainer = Trainer()
    if verbose:
        print("2/3: Training Model...")

    if architecture is None:
        architecture = {
            "num_rnn_layers": 4,
            "num_rnn_hidden": 128,
        }

    print("Architecture: ")
    print(architecture)

    trainer.train(
        model_class=model_class,
        architecture=architecture,
        data_dict=data_dict,
        criterion=loss,
        epochs=epochs,
        batch_size=batch_size,
        mc_dropout=mc_dropout,
        dropout_prob=dropout_prob,
        tensorboard=tensorboard,
        save_model=save_model,
        performance_optimized=performance_optimized,
        sequence_length=5,
        verbose=verbose,
        tensorboard_comment=tensorboard_comment,
    )

    if verbose:
        print("3/3: Evaluating Model")
    model = trainer.model
    metrics, test_preds = trainer.evaluate(verbose=verbose)
    return model, metrics, test_preds


def train_traditional_network(
    data_directory: str,
    architecture: dict = None,
    epochs: int = 20,
    batch_size: int = 100,
    model_class=ExploratoryModel,
    loss=nn.MSELoss(),
    tensorboard: bool = False,
    save_model: str = False,
    performance_optimized: bool = False,
    verbose: bool = False,
    tensorboard_comment: str = None,
):
    """Pipeline function for training a traditional neural network emulator. Loads processed data, trains
    network and saves it to the desired location. Outputs test metrics and predictions.

    Args:
        data_directory (str): Directory containing training and testing data.
        architecture (dict, optional): Architecture arguments used to train the model. Defaults to None.
        epochs (int, optional): Number of epochs to train the network. Defaults to 20.
        batch_size (int, optional): Batch size of training. Defaults to 100.
        model_class (ModelClass, optional): Model class used to train the model. Defaults to ExploratoryModel.
        loss (nn.Loss, optional): PyTorch loss to be used in training. Defaults to nn.MSELoss().
        tensorboard (bool, optional): Flag denoting whether to output logs to Tensorboard. Defaults to False.
        save_model (str, optional): Directory to save model. Can be False if model is not to be saved. Defaults to False.
        performance_optimized (bool, optional): Flag denoting whether to optimize the training for faster performace. Leaves out in-loop validation testing, extra logging, etc. Defaults to False.
        verbose (bool, optional): Flag denoting whether to output logs to terminal. Defaults to False.
        tensorboard_comment (str, optional): Comment to be included in the tensorboard logs. Defaults to None.

    Returns:
        tuple: Tuple containing [model, metrics, test_preds], or the model, test metrics, and test predictions.
    """

    if verbose:
        print("1/3: Loading processed data")

    try:
        test_features = pd.read_csv(f"{data_directory}/traditional_test_features.csv")
        train_features = pd.read_csv(f"{data_directory}/traditional_train_features.csv")
        test_labels = pd.read_csv(f"{data_directory}/traditional_test_labels.csv")
        train_labels = pd.read_csv(f"{data_directory}/traditional_train_labels.csv")
        scenarios = pd.read_csv(
            f"{data_directory}/traditional_test_scenarios.csv"
        ).values.tolist()
    except FileNotFoundError:
        raise FileNotFoundError(
            'Files not found. Format must be in format "traditional_train_features.csv"'
        )

    if "lag" in train_features.columns:
        raise AttributeError(
            "Data must be processed using timeseries=True in feataure_engineering. Rerun feature engineering to train traditional network."
        )

    data_dict = {
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels,
    }

    trainer = Trainer()
    if verbose:
        print("2/3: Training Model")

    if architecture is None:
        architecture = {
            "num_linear_layers": 4,
            "nodes": [128, 64, 32, 1],
        }

    trainer.train(
        model_class=model_class,
        architecture=architecture,
        data_dict=data_dict,
        criterion=loss,
        epochs=epochs,
        batch_size=batch_size,
        tensorboard=tensorboard,
        save_model=save_model,
        performance_optimized=performance_optimized,
        sequence_length=5,
        verbose=verbose,
        tensorboard_comment=tensorboard_comment,
    )

    if verbose:
        print("3/3: Evaluating Model")
    model = trainer.model
    metrics, test_preds = trainer.evaluate(verbose=verbose)
    return model, metrics, test_preds


def train_gaussian_process(
    data_directory: str,
    n: int,
    features: list[str] = ["temperature"],
    sampling_method: str = "first_n",
    kernel=None,
    verbose: bool = False,
    save_directory: str = None,
):
    """Pipeline function for training a gaussian process emulator. Loads processed data and trains
    gaussian process. Outputs test metrics and predictions.

    Args:
        data_directory (str): Directory containing training and testing data.
        n (int): Number of observations to use for training.
        features (list[str], optional): List of columns to use for training. May also contain list of [pc1, pc2, pc3, ...] to use N principal components. Defaults to ['temperature'].
        sampling_method (str, optional): Method of sampling n rows, must be in [first_n, random]. First_n takes the first n rows in the dataset, whereas random uses random sampling. First_n is recommended. Defaults to 'first_n'.
        kernel (sklearn.kernels, optional): Sklean kernel to be used for GP training. Defaults to None.
        verbose (bool, optional): Flag denoting whether to output logs to terminal. Defaults to False.
        save_directory (str, optional): Directory to save outputs. Defaults to None.

    Returns:
        tuple: Tuple containing [preds, std_prediction, metrics], or the test predictions, uncertainty, and test metrics.
    """

    if verbose:
        print("1/3: Loading processed data...")

    try:
        test_features = pd.read_csv(f"{data_directory}/traditional_test_features.csv")
        train_features = pd.read_csv(f"{data_directory}/traditional_train_features.csv")
        test_labels = pd.read_csv(f"{data_directory}/traditional_test_labels.csv")
        train_labels = pd.read_csv(f"{data_directory}/traditional_train_labels.csv")
        scenarios = pd.read_csv(
            f"{data_directory}/traditional_test_scenarios.csv"
        ).values.tolist()
    except FileNotFoundError:
        test_features = pd.read_csv(f"{data_directory}/ts_test_features.csv")
        train_features = pd.read_csv(f"{data_directory}/ts_train_features.csv")
        test_labels = pd.read_csv(f"{data_directory}/ts_test_labels.csv")
        train_labels = pd.read_csv(f"{data_directory}/ts_train_labels.csv")
        scenarios = pd.read_csv(
            f"{data_directory}/ts_test_scenarios.csv"
        ).values.tolist()

    if not isinstance(features, list):
        raise ValueError(f"features argument must be a list, received {type(features)}")

    # type check features
    if not isinstance(features, list):
        raise AttributeError(
            f"features argument must be of type list, received {type(features)}"
        )

    # See if features argument contain columns or principal components
    features_are_pcs = all([f.lower().startswith("pc") for f in features])
    features_are_columns = all([f in train_features.columns for f in features])

    if features_are_columns:
        # subset dataframe based on columns
        gp_train_features = train_features[features]
        gp_test_features = test_features[features]

    elif features_are_pcs:
        # run PCA with the len(features) be the number of PCs
        train_features["set"] = "train"
        test_features["set"] = "test"
        pca_features = pd.concat([train_features, test_features])
        pca = PCA(n_components=len(features))
        principalComponents = pca.fit_transform(pca_features.drop(columns=["set"]))
        gp_train_features = pd.DataFrame(
            principalComponents[pca_features["set"] == "train"].squeeze()
        )
        gp_test_features = pd.DataFrame(
            principalComponents[pca_features["set"] == "test"].squeeze()
        )

    else:
        raise ValueError(
            "Features must all be in train_features.columns or must all be PCs, e.g. [pc1, pc2, pc3]"
        )

    # Handle subsetting data -- randomly drawn or first N
    if sampling_method.lower() == "random":
        gp_train_features = gp_train_features.sample(n)
    elif sampling_method.lower() == "first_n":
        gp_train_features = gp_train_features[:n]
    else:
        raise ValueError(
            f"sampling method must be in [random, first_n], received {sampling_method}"
        )

    # Format datasets
    gp_train_labels = np.array(train_labels.loc[gp_train_features.index])
    if isinstance(gp_train_features, pd.Series) or gp_train_features.shape[1] == 1:
        gp_train_features = np.array(gp_train_features).reshape(-1, 1)
    if isinstance(gp_test_features, pd.Series) or gp_test_features.ndim == 1:
        gp_test_features = np.array(gp_test_features).reshape(-1, 1)

    # Initiate model classes
    if kernel is None:
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e2))
    gaussian_process = GP(kernel=kernel)

    if verbose:
        print("2/3: Training Model...")

    # fit data (same as gaussian_process.fit())
    gaussian_process.train(
        gp_train_features,
        gp_train_labels,
    )

    if verbose:
        print("3/3: Evaluating Model")

    # evaluate on test
    preds, std_prediction, metrics = gaussian_process.test(
        gp_test_features, test_labels
    )

    if save_directory:
        if isinstance(save_directory, str):
            preds_path = f"{save_directory}/preds.csv"
            uq_path = f"{save_directory}/std.csv"

        elif isinstance(save_directory, bool):
            preds_path = f"preds.csv"
            uq_path = f"std.csv"

        pd.Series(preds, name="preds").to_csv(preds_path, index=False)
        pd.Series(std_prediction, name="std_prediction").to_csv(uq_path, index=False)

    return preds, std_prediction, metrics
