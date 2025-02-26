import torch


class WeightedGridLoss(torch.nn.Module):
    """
    Custom loss function that penalizes errors based on the total variation of a grid.

    This loss function consists of two components:
    1. **Pixel-wise Weighted Mean Squared Error (MSE):** Higher weight is assigned to extreme values.
    2. **Total Variation Regularization (TVR):** Enforces spatial smoothness by penalizing large differences 
       between adjacent grid values.

    Attributes:
        device (str): The device on which the model runs ('cuda' or 'cpu').

    Methods:
        - total_variation_regularization: Computes TVR penalty for smoothness.
        - weighted_pixelwise_mse: Computes MSE loss with per-pixel weighting.
        - forward: Computes the final weighted loss.

    """


    def __init__(self):
        super(WeightedGridLoss, self).__init__()
        self.to(self.device)

    def total_variation_regularization(self, grid):
        """
        Computes the total variation regularization (TVR) loss for spatial smoothness.

        Args:
            grid (Tensor): A 2D tensor representing spatial data.

        Returns:
            Tensor: The total variation loss.
        """

        # Calculate the sum of horizontal and vertical differences
        horizontal_diff = torch.abs(torch.diff(grid, axis=2))
        vertical_diff = torch.abs(torch.diff(grid, axis=1))
        total_variation = torch.sum(horizontal_diff, axis=(1, 2)) + torch.sum(
            vertical_diff, axis=(1, 2)
        )
        return torch.mean(total_variation)

    def weighted_pixelwise_mse(self, true, predicted, weights):
        """
        Computes the pixel-wise mean squared error (MSE) with custom weights.

        Args:
            true (Tensor): Ground truth values.
            predicted (Tensor): Model predictions.
            weights (Tensor): Weighting factor for each pixel.

        Returns:
            Tensor: Weighted mean squared error.
        """

        # Compute the squared error
        squared_error = (true - predicted) ** 2
        # Apply weights
        weighted_error = weights * squared_error
        # Return the mean of the weighted error
        return torch.mean(weighted_error)
    
    
    def forward(self, true, predicted, smoothness_weight=0.001, extreme_value_threshold=1e-6):
        """
        Computes the final weighted loss combining pixel-wise MSE and TVR.

        Args:
            true (Tensor): Ground truth values.
            predicted (Tensor): Model predictions.
            smoothness_weight (float, optional): Weighting factor for the TVR loss. Defaults to 0.001.
            extreme_value_threshold (float, optional): Threshold to define extreme values. Defaults to 1e-6.

        Returns:
            Tensor: The total computed loss.
        """

        true = torch.tensor(true, dtype=torch.float32, device=self.device)
        predicted = torch.tensor(predicted, dtype=torch.float32, device=self.device)

        # Determine weights based on extreme values
        if extreme_value_threshold is not None:
            # Identify extreme values in the true data
            extreme_mask = torch.abs(true) > extreme_value_threshold
            # Assign higher weight to extreme values, 1 to others
            weights = torch.where(extreme_mask, 10.0 * torch.ones_like(true), torch.ones_like(true))
        else:
            # If no threshold is provided, use uniform weights
            weights = torch.ones_like(true)

        pixelwise_mse = self.weighted_pixelwise_mse(true, predicted, weights)
        tvr = self.total_variation_regularization(predicted)
        return pixelwise_mse + smoothness_weight * tvr


class WeightedMSELoss(torch.nn.Module):
    """
    Custom loss function that applies a weighted penalty to extreme values.

    This function increases the weight of extreme values based on their deviation from the 
    dataset mean, normalizing by the standard deviation.

    Attributes:
        data_mean (Tensor): Mean value of the dataset.
        data_std (Tensor): Standard deviation of the dataset.
        weight_factor (Tensor): Factor controlling how much extreme values are penalized.

    Methods:
        - forward: Computes the weighted mean squared error loss.
    """

    def __init__(self, data_mean, data_std, weight_factor=1.0):
        super(WeightedMSELoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_mean = torch.tensor(data_mean, dtype=torch.float32, device=self.device)
        self.data_std = torch.tensor(data_std, dtype=torch.float32, device=self.device)
        self.weight_factor = torch.tensor(weight_factor, dtype=torch.float32, device=self.device)
        self.to(self.device)
        
    def forward(self, input, target):
        """
        Computes the Weighted Mean Squared Error (MSE) Loss.

        Args:
            input (Tensor): Predicted values.
            target (Tensor): Ground truth values.

        Returns:
            Tensor: Computed loss.
        """

        # Ensure data_mean, data_std, and weight_factor are on the same device as input
        input = input.to(self.device)
        target = target.to(self.device)

        # Calculate the deviation of each target value from the mean
        deviation = torch.abs(target - self.data_mean)

        # Scale deviations by the standard deviation to normalize them
        # normalized_deviation = torch.tensor(deviation / self.data_std, dtype=torch.float32, device=self.device)
        normalized_deviation = deviation / self.data_std

        # Compute weights: increase penalty for extreme values
        weights = 1 + (normalized_deviation * self.weight_factor)

        # Compute the squared error
        squared_error = torch.nn.functional.mse_loss(input, target, reduction="none")

        # Apply the weights and take the mean to get the final loss
        weighted_squared_error = weights * squared_error
        loss = torch.mean(weighted_squared_error)

        return loss


class WeightedMSEPCALoss(torch.nn.Module):
    """
    Extension of WeightedMSELoss that allows for custom per-batch weighting.

    This loss function enables additional user-defined weights to further adjust penalties for different predictions.

    Attributes:
        data_mean (Tensor): Mean of the dataset.
        data_std (Tensor): Standard deviation of the dataset.
        weight_factor (Tensor): Controls the penalty for extreme values.
        custom_weights (Tensor, optional): User-defined weight tensor.

    Methods:
        - forward: Computes the batch-weighted MSE loss.
    """
    def __init__(self, data_mean, data_std, weight_factor=1.0, custom_weights=None):

        super(WeightedMSEPCALoss, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        self.data_mean = torch.tensor(data_mean, dtype=torch.float32, device=self.device)
        self.data_std = torch.tensor(data_std, dtype=torch.float32, device=self.device)
        self.weight_factor = torch.tensor(weight_factor, dtype=torch.float32, device=self.device)
        self.custom_weights = (
            torch.tensor(custom_weights, dtype=torch.float32, device=self.device)
            if custom_weights is not None
            else None
        )

    def forward(self, input, target):
        """
        Computes the batch-weighted mean squared error loss.

        Args:
            input (Tensor): Predicted values.
            target (Tensor): Ground truth values.

        Returns:
            Tensor: Computed loss.
        """


        input = input.to(self.device)
        target = target.to(self.device)

        # Ensure input and target are of the same shape
        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape.")

        # Calculate the deviation of each target value from the mean
        deviation = torch.abs(target - self.data_mean)

        # Scale deviations by the standard deviation to normalize them
        normalized_deviation = deviation / self.data_std

        # Compute weights: increase penalty for extreme values
        weights = 1 + (normalized_deviation * self.weight_factor)

        # If custom weights are provided, multiply them by the calculated weights
        if self.custom_weights is not None:
            # Expand custom weights to match batch size if necessary
            if self.custom_weights.dim() == 1:
                self.custom_weights = self.custom_weights.unsqueeze(0)  # Make it a 2D tensor
            if self.custom_weights.shape != weights.shape:
                raise ValueError("Custom weights shape must match input/target shape.")
            weights *= self.custom_weights

        # Compute the squared error for each element in the batch without reducing
        squared_error = (input - target) ** 2

        # Apply the weights to the squared error
        weighted_squared_error = weights * squared_error

        # Take the mean across all dimensions to get the final loss
        loss = torch.mean(weighted_squared_error)

        return loss


class WeightedMSELossWithSignPenalty(torch.nn.Module):
    """
    Custom loss function that penalizes errors on extreme values and opposite sign predictions.

    This function extends WeightedMSELoss by adding a penalty when the sign of the prediction differs from the target.

    Attributes:
        data_mean (Tensor): Mean of the dataset.
        data_std (Tensor): Standard deviation of the dataset.
        weight_factor (Tensor): Factor controlling extreme value weighting.
        sign_penalty_factor (Tensor): Factor controlling penalty for opposite sign predictions.

    Methods:
        - forward: Computes the weighted loss with sign penalties.
    """

    def __init__(self, data_mean, data_std, weight_factor=1.0, sign_penalty_factor=1.0):

        super(WeightedMSELossWithSignPenalty, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_mean = torch.tensor(data_mean, dtype=torch.float32, device=self.device)
        self.data_std = torch.tensor(data_std, dtype=torch.float32, device=self.device)
        self.weight_factor = torch.tensor(weight_factor, dtype=torch.float32, device=self.device)
        self.sign_penalty_factor = torch.tensor(
            sign_penalty_factor, dtype=torch.float32, device=self.device
        )
        self.to(self.device)

    def forward(self, input, target):
        """
        Computes the Weighted MSE Loss with an additional sign penalty.

        Args:
            input (Tensor): Predicted values.
            target (Tensor): Ground truth values.

        Returns:
            Tensor: Computed loss.
        """

        # Calculate the deviation of each target value from the mean
        deviation = torch.abs(target - self.data_mean)

        # Scale deviations by the standard deviation to normalize them
        normalized_deviation = deviation / self.data_std

        # Compute weights: increase penalty for extreme values
        weights = 1 + (normalized_deviation * self.weight_factor)

        # Compute the squared error
        squared_error = torch.nn.functional.mse_loss(input, target, reduction="none")

        # Calculate sign penalty
        sign_penalty = torch.where(
            torch.sign(input) != torch.sign(target),
            torch.abs(input - target) * self.sign_penalty_factor,
            torch.zeros_like(input),
        )

        # Apply the weights and sign penalty, then take the mean to get the final loss
        weighted_squared_error = weights * (squared_error + sign_penalty)
        loss = torch.mean(weighted_squared_error)

        return loss


class GridCriterion(torch.nn.Module):
    """
    Custom loss function enforcing spatial smoothness using total variation regularization.

    This function encourages smoothness in spatial predictions by penalizing large variations.

    Methods:
        - total_variation_regularization: Computes the smoothness loss.
        - forward: Computes the final loss.
    """

    def __init__(
        self,
    ):
        super(GridCriterion, self).__init__()

    def total_variation_regularization(self, grid):
        """
        Computes total variation regularization (TVR) loss.

        Args:
            grid (Tensor): A 2D tensor representing spatial data.

        Returns:
            Tensor: TVR loss.
        """

        # Calculate the sum of horizontal and vertical differences
        horizontal_diff = torch.abs(torch.diff(grid, axis=2))
        vertical_diff = torch.abs(torch.diff(grid, axis=1))
        total_variation = torch.sum(horizontal_diff, axis=(1, 2)) + torch.sum(
            vertical_diff, axis=(1, 2)
        )
        return torch.mean(total_variation)

    # def spatial_loss(self, true, predicted, smoothness_weight=0.001):
    def forward(self, true, predicted, smoothness_weight=0.001):
        """
        Computes the final loss by combining pixel-wise MSE and TVR.

        Args:
            true (Tensor): Ground truth values.
            predicted (Tensor): Model predictions.
            smoothness_weight (float, optional): Weight for TVR. Defaults to 0.001.

        Returns:
            Tensor: Computed loss.
        """

        pixelwise_mse = torch.mean(
            torch.abs(true - predicted) ** 2,
        )  # loss for each image in the batch (batch_size,)
        tvr = self.total_variation_regularization(
            predicted,
        )
        return pixelwise_mse + smoothness_weight * tvr

    # def forward(self, true, predicted, x, y, flow, predictor_weight=0.5, nf_weight=0.5,):
    #     if predictor_weight + nf_weight != 1:
    #         raise ValueError("The sum of predictor_weight and nf_weight must be 1")
    #     predictor_loss = self.spatial_loss(true, predicted, smoothness_weight=0.2)
    #     nf_loss = -flow.log_prob(inputs=y, context=x)
    #     return predictor_weight*predictor_loss + nf_weight*nf_loss


class WeightedPCALoss(torch.nn.Module):
    """
    Custom loss function applying different weights to errors in principal component analysis.

    This function allows assigning higher penalties to the first components.

    Attributes:
        component_weights (Tensor): Weighting factors for each principal component.
        reduction (str): Specifies reduction mode ('mean', 'sum', or 'none').

    Methods:
        - forward: Computes the weighted PCA loss.
    """

    def __init__(self, component_weights, reduction="mean"):
        super(WeightedPCALoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.component_weights = torch.tensor(
            component_weights, dtype=torch.float32, device=self.device
        )
        if len(self.component_weights.size()) == 1:
            self.component_weights = self.component_weights.unsqueeze(0)  # Make it a row vector
        self.reduction = reduction
        self.to(self.device)

    def forward(self, input, target):
        """
        Computes the weighted PCA loss.

        Args:
            input (Tensor): Predicted principal components.
            target (Tensor): Actual principal components.

        Returns:
            Tensor: Computed loss.
        """


        input = input.to(self.device)
        target = target.to(self.device)

        # Ensure input and target are of the same shape
        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape")

        # Calculate the squared error
        squared_error = (input - target) ** 2

        # Apply weights to the squared error
        weighted_error = squared_error * self.component_weights.to(input.device)

        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(weighted_error)
        elif self.reduction == "sum":
            return torch.sum(weighted_error)
        else:
            return weighted_error


class MSEDeviationLoss(torch.nn.Module):
    """
    Custom MSE Loss with an additional penalty for large deviations.

    This function penalizes predictions that deviate significantly from the target.

    Attributes:
        threshold (float): Deviation threshold for applying penalties.
        penalty_multiplier (float): Multiplier controlling penalty severity.

    Methods:
        - forward: Computes the loss with deviation penalties.
    """

    def __init__(self, threshold=1.0, penalty_multiplier=2.0):
        super(MSEDeviationLoss, self).__init__()
        self.threshold = threshold
        self.penalty_multiplier = penalty_multiplier

    def forward(self, predictions, targets):
        """
        Computes the MSE loss with an additional deviation penalty.

        Args:
            predictions (Tensor): Predicted values.
            targets (Tensor): Ground truth values.

        Returns:
            Tensor: Computed loss.
        """

        mse_loss = torch.mean((predictions - targets) ** 2)
        large_deviation_penalty = torch.mean(
            torch.where(
                torch.abs(predictions - targets) > self.threshold,
                self.penalty_multiplier * (predictions - targets) ** 2,
                torch.tensor(0.0, device=predictions.device),
            )
        )
        return mse_loss + large_deviation_penalty
