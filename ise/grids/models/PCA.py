import torch

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.singular_values = None

    def fit(self, X):
        # Center the data
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean

        # Compute low-rank PCA
        U, S, V = torch.pca_lowrank(X_centered, q=self.n_components)

        self.components = V
        self.singular_values = S

    def transform(self, X):
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet.")
        X_centered = X - self.mean
        return torch.mm(X_centered, self.components)

    def inverse_transform(self, X):
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet.")
        return torch.mm(X, self.components.t()) + self.mean

    def save(self, path):
        torch.save({
            'n_components': self.n_components,
            'mean': self.mean,
            'components': self.components,
            'singular_values': self.singular_values
        }, path)

    @staticmethod
    def load(path):
        checkpoint = torch.load(path)
        model = PCA_LowRank(checkpoint['n_components'])
        model.mean = checkpoint['mean']
        model.components = checkpoint['components']
        model.singular_values = checkpoint['singular_values']
        return model
