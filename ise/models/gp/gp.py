
import numpy as np
from joblib import dump, load
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, _check_length_scale
from sklearn.metrics import r2_score


class PowerExponentialKernel(RBF):
    """
    A modified Radial Basis Function (RBF) kernel with a power exponential component.

    This kernel generalizes the standard RBF kernel by allowing a custom 
    exponential factor in the distance computation.

    Attributes:
        exponential (float): The power to which distances are raised in the kernel function.
        length_scale (float): Characteristic length scale for the kernel.
        length_scale_bounds (tuple): Lower and upper bounds for the length scale.
    """

    def __init__(
        self,
        exponential=2.0,
        length_scale=1.0,
        length_scale_bounds=(1e-5, 1e5),
    ):
        super().__init__(
            length_scale_bounds=length_scale_bounds,
            length_scale=length_scale,
        )
        self.exponential = exponential

    # OVERWRITE CALL METHOD FROM SKLEARN.GAUSSIAN_PROCESS.KERNELS.RBF
    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Computes the kernel matrix between inputs X and Y.

        Args:
            X (np.ndarray): Input data of shape (n_samples_X, n_features).
            Y (np.ndarray, optional): Input data of shape (n_samples_Y, n_features). If None, self-similarity is computed. Defaults to None.
            eval_gradient (bool, optional): If True, computes the gradient of the kernel with respect to the log of the length scale. Defaults to False.

        Returns:
            np.ndarray: Computed kernel matrix of shape (n_samples_X, n_samples_Y).
            np.ndarray (optional): Gradient of the kernel function, returned only if `eval_gradient=True`.

        Raises:
            ValueError: If `eval_gradient` is True but `Y` is not None.
        """

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** self.exponential / (
                    length_scale**2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K


class NuggetKernel(WhiteKernel):
    """
    A noise kernel (White Kernel) that models independent noise in Gaussian Processes.

    This kernel adds noise to the diagonal of the covariance matrix to account for 
    observation noise.

    Attributes:
        noise_level (float): The variance of the noise.
        noise_level_bounds (tuple): The bounds for the noise variance.
    """

    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        super().__init__(noise_level=noise_level, noise_level_bounds=noise_level_bounds)


class GP(GaussianProcessRegressor):
    """
    Gaussian Process (GP) regression model with a specified kernel.

    This class extends `GaussianProcessRegressor` from `sklearn` and allows 
    training and testing with additional functionality.

    Attributes:
        kernel (sklearn.gaussian_process.kernels.Kernel): The kernel function used for regression.
        verbose (bool): If True, prints progress and evaluation metrics.
    """

    def __init__(self, kernel, verbose=True):
        super().__init__(
            n_restarts_optimizer=9,
        )
        self.kernel = kernel
        self.verbose = verbose

    def train(self, train_features, train_labels):
        """
        Trains the Gaussian Process regression model.

        Args:
            train_features (np.ndarray): Feature matrix of shape (n_samples, n_features).
            train_labels (np.ndarray): Target values of shape (n_samples,).

        Returns:
            GP: The trained Gaussian Process model.
        """

        self.train_features, self.train_labels = train_features, train_labels
        self.fit(
            train_features,
            train_labels,
        )
        return self
    def test(self, test_features, test_labels):
        """
        Evaluates the Gaussian Process regression model on test data.

        Args:
            test_features (np.ndarray): Feature matrix of shape (n_samples, n_features).
            test_labels (np.ndarray): Ground truth target values of shape (n_samples,).

        Returns:
            tuple:
                - np.ndarray: Predicted values.
                - np.ndarray: Standard deviation of predictions.
                - dict: Evaluation metrics including MSE, MAE, RMSE, and R².

        Raises:
            ValueError: If `test_features` or `test_labels` have incorrect dimensions.
        """

        self.test_features, self.test_labels = test_features, test_labels
        preds, std_prediction = self.predict(test_features, return_std=True)
        test_labels = np.array(test_labels.squeeze())
        mse = sum((preds - test_labels) ** 2) / len(preds)
        mae = sum(abs((preds - test_labels))) / len(preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_labels, preds)

        metrics = {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}

        if self.verbose:
            print(
                f"""Test Metrics
MSE: {mse:0.6f}
MAE: {mae:0.6f}
RMSE: {rmse:0.6f}
R2: {r2:0.6f}"""
            )
        return preds, std_prediction, metrics

    def save(self, path):
        """
        Saves the trained Gaussian Process model to a file.

        Args:
            path (str): Path where the model should be saved. Must end with `.joblib`.

        Raises:
            ValueError: If the file path does not end with `.joblib`.
        """

        if not path.endswith(".joblib"):
            raise ValueError("Path must end with .joblib")
        dump(self, path)

    def load(self, path):
        """
        Loads a Gaussian Process model from a saved file.

        Args:
            path (str): Path to the saved model file. Must end with `.joblib`.

        Returns:
            GP: The loaded Gaussian Process model.

        Raises:
            ValueError: If the file path does not end with `.joblib`.
        """

        if not path.endswith(".joblib"):
            raise ValueError("Path must end with .joblib")
        self = load(path)
        return self


# TODO write Emulandice based on train_gp.py