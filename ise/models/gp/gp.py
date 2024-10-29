
import numpy as np
from joblib import dump, load
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, _check_length_scale
from sklearn.metrics import r2_score


class PowerExponentialKernel(RBF):
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
    def __call__(
        self,
        X,
        Y=None,
        eval_gradient=False,
    ):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
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
    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        super().__init__(noise_level=noise_level, noise_level_bounds=noise_level_bounds)


class GP(GaussianProcessRegressor):
    def __init__(self, kernel, verbose=True):
        super().__init__(
            n_restarts_optimizer=9,
        )
        self.kernel = kernel
        self.verbose = verbose

    def train(
        self,
        train_features,
        train_labels,
    ):
        self.train_features, self.train_labels = train_features, train_labels
        self.fit(
            train_features,
            train_labels,
        )
        return self

    def test(self, test_features, test_labels):
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
        """Save model to path."""
        if not path.endswith(".joblib"):
            raise ValueError("Path must end with .joblib")
        dump(self, path)

    def load(self, path):
        """Load model from path."""
        if not path.endswith(".joblib"):
            raise ValueError("Path must end with .joblib")
        self = load(path)
        return self


# TODO write Emulandice based on train_gp.py