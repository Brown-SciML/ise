"""Custom Kernels required for gaussian process regression."""

from sklearn.gaussian_process.kernels import RBF, _check_length_scale, WhiteKernel
from scipy.spatial.distance import pdist, cdist, squareform
import numpy as np


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
                K_gradient = (
                    X[:, np.newaxis, :] - X[np.newaxis, :, :]
                ) ** self.exponential / (length_scale**2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K


class NuggetKernel(WhiteKernel):
    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        super().__init__(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
