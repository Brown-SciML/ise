import gpytorch
import numpy as np
import pandas as pd
import torch

DATA_DIRECTORY = r"/users/pvankatw/emulator/untracked_folder/ml_data"

train_features = pd.read_csv(f"{DATA_DIRECTORY}/ts_train_features.csv")
train_labels = pd.read_csv(f"{DATA_DIRECTORY}/ts_train_labels.csv")
test_features = pd.read_csv(f"{DATA_DIRECTORY}/ts_test_features.csv")
test_labels = pd.read_csv(f"{DATA_DIRECTORY}/ts_test_labels.csv")


train_features = torch.from_numpy(np.array(train_features))
train_labels = torch.from_numpy(np.array(train_labels))
test_features = torch.from_numpy(np.array(test_features))
test_labels = torch.from_numpy(np.array(test_labels))


class GPyTorchModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPyTorchModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPyTorchModel(train_features, train_labels, likelihood)

train_features = train_features.cuda()
train_labels = train_labels.cuda()
model = model.cuda()
likelihood = likelihood.cuda()

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_features)
    # Calc loss and backprop gradients
    loss = -mll(output, train_labels)
    loss.backward()
    print(
        "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
        % (
            i + 1,
            training_iter,
            loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item(),
        )
    )
    optimizer.step()


print(model)
