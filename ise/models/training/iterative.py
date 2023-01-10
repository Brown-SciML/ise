from ise.data.EmulatorData import EmulatorData
from ise.models.training.Trainer import Trainer
from ise.models.timeseries import TimeSeriesEmulator
from ise.models.traditional.ExploratoryModel import ExploratoryModel
from ise.utils.utils import _structure_emulatordata_args, _structure_architecture_args
from datetime import datetime
from torch import nn
from ise.utils.data import load_ml_data


def lag_sequence_test(
    data_directory,
    lag_array,
    sequence_array,
    iterations,
    model_class=TimeSeriesEmulator,
    emulator_data_args=None,
    architecture=None,
    verbose=True,
    epochs=100,
    batch_size=100,
    loss=nn.MSELoss(),
):

    if verbose:
        print("1/3: Loading processed data...")

    emulator_data_args = _structure_emulatordata_args(
        emulator_data_args, time_series=True
    )
    architecture = _structure_architecture_args(architecture, time_series=True)

    count = 0
    for iteration in range(1, iterations + 1):
        for lag in lag_array:
            for sequence_length in sequence_array:

                print(
                    f"Training... Lag: {lag}, Sequence Length: {sequence_length}, Iteration: {iteration}, Trained {count} models"
                )

                emulator_data = EmulatorData(directory=data_directory)
                emulator_data_args["lag"] = lag
                (
                    emulator_data,
                    train_features,
                    test_features,
                    train_labels,
                    test_labels,
                ) = emulator_data.process(**emulator_data_args)

                data_dict = {
                    "train_features": train_features,
                    "train_labels": train_labels,
                    "test_features": test_features,
                    "test_labels": test_labels,
                }

                trainer = Trainer()

                current_time = datetime.now().strftime(r"%d-%m-%Y %H.%M.%S")
                trainer.train(
                    model_class=model_class,
                    architecture=architecture,
                    data_dict=data_dict,
                    criterion=loss,
                    epochs=epochs,
                    batch_size=batch_size,
                    tensorboard=True,
                    save_model=False,
                    performance_optimized=False,
                    verbose=verbose,
                    sequence_length=sequence_length,
                    tensorboard_comment=f" -- {current_time}, lag={lag}, sequence_length={sequence_length}",
                )

                # not verbose because if verbose==True, this is already calculated in training loop
                if not verbose:
                    metrics, preds = trainer.evaluate()

                count += 1

    print(f"Finished trainin {count} models.")


def rnn_architecture_test(
    data_directory: str,
    rnn_layers_array: list[int],
    hidden_nodes_array: list[int],
    iterations: int,
    model_class=TimeSeriesEmulator,
    verbose: bool = True,
    epochs: int = 100,
    batch_size: int = 100,
    loss=nn.MSELoss(),
    mc_dropout: bool = False,
    dropout_prob: float = None,
    performance_optimized: bool = False,
    save_model: str = True,
    tensorboard: bool = True,
):
    """Tests various configurations of the RNN architecture specified. It will take the (Cartesian)
    product of the rnn_layers_array and hidden_nodes_array and train N networks (iterations) and logs
    the performance.

    Args:
        data_directory (str): Directory containing training and testing data.
        rnn_layers_array (list[int]): List of possible RNN layer numbers to be tested.
        hidden_nodes_array (list[int]): List of possible hidden layer nodes to be tested.
        iterations (int): Number of times each combination of rnn_layers_array and hidden_nodes_array will be tested.
        model_class (ModelClass, optional): Model class to be trained. Defaults to TimeSeriesEmulator.
        verbose (bool, optional): Flag denoting whether to output logs to terminal. Defaults to True.
        epochs (int, optional): Number of epochs to train the network. Defaults to 100.
        batch_size (int, optional): Batch size of training. Defaults to 100.
        loss (_type_, optional): PyTorch loss to be used in training. Defaults to nn.MSELoss().
        mc_dropout (bool, optional): Flag denoting whether the model was trained with MC dropout protocol. Defaults to False.
        dropout_prob (float, optional): Dropout probability in MC dropout protocol. Unused if mc_dropout=False. Defaults to None.
        performance_optimized (bool, optional): Flag denoting whether to optimize the training for faster performace. Leaves out in-loop validation testing, extra logging, etc. Defaults to False.
        save_model (str, optional): Directory to save model. Can be False if model is not to be saved. Defaults to True.
        tensorboard (bool, optional): Flag denoting whether to output logs to Tensorboard. Defaults to True.
    """

    train_features, train_labels, test_features, test_labels, _ = load_ml_data(
        data_directory=data_directory, time_series=True
    )

    data_dict = {
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels,
    }

    count = 0
    for iteration in range(1, iterations + 1):
        for num_rnn_layers in rnn_layers_array:
            for num_rnn_hidden in hidden_nodes_array:
                print(
                    f"Training... RNN Layers: {num_rnn_layers}, Hidden: {num_rnn_hidden}, Iteration: {iteration}, Trained {count} models"
                )

                trainer = Trainer()
                architecture = {
                    "num_rnn_layers": num_rnn_layers,
                    "num_rnn_hidden": num_rnn_hidden,
                }
                current_time = datetime.now().strftime(r"%d-%m-%Y %H.%M.%S")
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
                    verbose=verbose,
                    sequence_length=5,
                    tensorboard_comment=f" -- {current_time}, num_rnn={num_rnn_layers}, num_hidden={num_rnn_hidden}",
                )

                if not verbose:
                    metrics, preds = trainer.evaluate()

                count += 1

    print("")
    print(f"Finished training {count} models.")


def traditional_architecture_test(
    data_directory,
    architectures: list[dict],
    iterations,
    model_class=ExploratoryModel,
    verbose=True,
    epochs=100,
    batch_size=100,
    loss=nn.MSELoss(),
):

    train_features, train_labels, test_features, test_labels = load_ml_data(
        data_directory=data_directory, time_series=False
    )

    data_dict = {
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels,
    }

    count = 0
    for iteration in range(1, iterations + 1):
        for architecture in architectures:
            num_linear_layers = architecture["num_linear_layers"]
            nodes = architecture["nodes"]
            print(
                f"Training... Linear Layers: {num_linear_layers}, Nodes: {nodes}, Iteration: {iteration}, Trained {count} models"
            )

            trainer = Trainer()
            current_time = datetime.now().strftime(r"%d-%m-%Y %H.%M.%S")
            trainer.train(
                model_class=model_class,
                architecture=architecture,
                data_dict=data_dict,
                criterion=loss,
                epochs=epochs,
                batch_size=batch_size,
                tensorboard=True,
                save_model=True,
                performance_optimized=False,
                verbose=verbose,
                sequence_length=5,
                tensorboard_comment=f" -- {current_time}, num_linear={num_linear_layers}, nodes={nodes}",
            )

            if not verbose:
                metrics, preds = trainer.evaluate()

            count += 1

    print(f"Finished trainin {count} models.")


# TODO: Write tests for the above
# TODO: make dense_architecture_test() that tests what dense layers should be at the end of the RNN
# TODO: Write function for dataset_tests and other tests I've done before (reproducibility!!!), use dict{'dataset1':['columns']} to loop
