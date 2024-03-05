import torch

class WeightedGridLoss(torch.nn.Module):
    def __init__(self):
        super(WeightedGridLoss, self).__init__()
        
    def total_variation_regularization(self, grid):
        # Calculate the sum of horizontal and vertical differences
        horizontal_diff = torch.abs(torch.diff(grid, axis=2))
        vertical_diff = torch.abs(torch.diff(grid, axis=1))
        total_variation = torch.sum(horizontal_diff, axis=(1,2)) + torch.sum(vertical_diff, axis=(1,2))
        return torch.mean(total_variation)

    def weighted_pixelwise_mse(self, true, predicted, weights):
        # Compute the squared error
        squared_error = (true - predicted) ** 2
        # Apply weights
        weighted_error = weights * squared_error
        # Return the mean of the weighted error
        return torch.mean(weighted_error)
    
    def forward(self, true, predicted, smoothness_weight=0.001, extreme_value_threshold=1e-6):
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
    def __init__(self, data_mean, data_std, weight_factor=1.0):
        """
        Custom loss function that penalizes errors on extreme values more.
        
        Args:
            data_mean (float): Mean of the target variable in the training set.
            data_std (float): Standard deviation of the target variable in the training set.
            weight_factor (float): Factor to adjust the weighting. Higher values will increase
                                   the penalty on extremes. Default is 1.0.
        """
        super(WeightedMSELoss, self).__init__()
        self.data_mean = data_mean
        self.data_std = data_std
        self.weight_factor = weight_factor
    
    def forward(self, input, target):
        """
        Calculate the Weighted MSE Loss.
        
        Args:
            input (tensor): Predicted values.
            target (tensor): Actual values.
        
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
        squared_error = torch.nn.functional.mse_loss(input, target, reduction='none')
        
        # Apply the weights and take the mean to get the final loss
        weighted_squared_error = weights * squared_error
        loss = torch.mean(weighted_squared_error)
        
        return loss
    
class WeightedMSEPCALoss(torch.nn.Module):
    def __init__(self, data_mean, data_std, weight_factor=1.0, custom_weights=None):
        """
        Custom loss function that penalizes errors on extreme values more and allows for custom weighting of each prediction
        in a batched manner.
        
        Args:
            data_mean (float): Mean of the target variable in the training set.
            data_std (float): Standard deviation of the target variable in the training set.
            weight_factor (float): Factor to adjust the weighting. Higher values will increase the penalty on extremes. Default is 1.0.
            custom_weights (torch.Tensor, optional): A tensor of weights corresponding to each y-value in the batch. Default is None.
        """
        super(WeightedMSEPCALoss, self).__init__()
        self.data_mean = data_mean
        self.data_std = data_std
        self.weight_factor = weight_factor
        self.custom_weights = custom_weights
    
    def forward(self, input, target):
        """
        Calculate the Weighted MSE Loss for batched inputs and outputs.
        
        Args:
            input (tensor): Predicted values with shape (batch_size, num_targets).
            target (tensor): Actual values with shape (batch_size, num_targets).
        
        Returns:
            Tensor: Computed loss.
        """
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
    def __init__(self, data_mean, data_std, weight_factor=1.0, sign_penalty_factor=1.0):
        """
        Custom loss function that penalizes errors on extreme values more and adds a penalty for opposite sign predictions.
        
        Args:
            data_mean (float): Mean of the target variable in the training set.
            data_std (float): Standard deviation of the target variable in the training set.
            weight_factor (float): Factor to adjust the weighting for extremes. Higher values will increase
                                   the penalty on extremes. Default is 1.0.
            sign_penalty_factor (float): Factor to adjust the penalty for opposite sign predictions.
                                         Higher values increase the penalty. Default is 1.0.
        """
        super(WeightedMSELossWithSignPenalty, self).__init__()
        self.data_mean = data_mean
        self.data_std = data_std
        self.weight_factor = weight_factor
        self.sign_penalty_factor = sign_penalty_factor
    
    def forward(self, input, target):
        """
        Calculate the Weighted MSE Loss with an additional penalty for opposite sign predictions.
        
        Args:
            input (tensor): Predicted values.
            target (tensor): Actual values.
        
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
        squared_error = torch.nn.functional.mse_loss(input, target, reduction='none')
        
        # Calculate sign penalty
        sign_penalty = torch.where(torch.sign(input) != torch.sign(target),
                                   torch.abs(input - target) * self.sign_penalty_factor,
                                   torch.zeros_like(input))
        
        # Apply the weights and sign penalty, then take the mean to get the final loss
        weighted_squared_error = weights * (squared_error + sign_penalty)
        loss = torch.mean(weighted_squared_error)
        
        return loss

class GridCriterion(torch.nn.Module):
    def __init__(self,):
        super(GridCriterion, self).__init__()
        
    def total_variation_regularization(self, grid, ):
        # Calculate the sum of horizontal and vertical differences
        horizontal_diff = torch.abs(torch.diff(grid, axis=2))
        vertical_diff = torch.abs(torch.diff(grid, axis=1))
        total_variation = torch.sum(horizontal_diff, axis=(1,2)) + torch.sum(vertical_diff, axis=(1,2))
        return torch.mean(total_variation)

    # def spatial_loss(self, true, predicted, smoothness_weight=0.001):
    def forward(self, true, predicted, smoothness_weight=0.001):
        pixelwise_mse = torch.mean(torch.abs(true - predicted)**2,) # loss for each image in the batch (batch_size,)
        tvr = self.total_variation_regularization(predicted,)
        return pixelwise_mse + smoothness_weight * tvr


    # def forward(self, true, predicted, x, y, flow, predictor_weight=0.5, nf_weight=0.5,):
    #     if predictor_weight + nf_weight != 1:
    #         raise ValueError("The sum of predictor_weight and nf_weight must be 1")
    #     predictor_loss = self.spatial_loss(true, predicted, smoothness_weight=0.2)
    #     nf_loss = -flow.log_prob(inputs=y, context=x)
    #     return predictor_weight*predictor_loss + nf_weight*nf_loss
    
    
class WeightedPCALoss(torch.nn.Module):
    def __init__(self, component_weights, reduction='mean'):
        """
        Custom loss function that applies different weights to the error of each principal component prediction.
        
        Args:
            component_weights (list or torch.Tensor): Weights for each principal component's error, 
                                                      where the first component has the highest weight.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(WeightedPCALoss, self).__init__()
        self.component_weights = torch.tensor(component_weights)
        if len(self.component_weights.size()) == 1:
            self.component_weights = self.component_weights.unsqueeze(0)  # Make it a row vector
        self.reduction = reduction
    
    def forward(self, input, target):
        """
        Calculate the weighted loss for principal component predictions.
        
        Args:
            input (tensor): Predicted principal components.
            target (tensor): Actual principal components.
            
        Returns:
            Tensor: Computed Weighted PCA Loss.
        """
        # Ensure input and target are of the same shape
        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape")

        # Calculate the squared error
        squared_error = (input - target) ** 2
        
        # Apply weights to the squared error
        weighted_error = squared_error * self.component_weights.to(input.device)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(weighted_error)
        elif self.reduction == 'sum':
            return torch.sum(weighted_error)
        else:
            return weighted_error