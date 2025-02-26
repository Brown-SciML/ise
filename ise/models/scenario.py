import torch
import torch.nn as nn
import torch.optim as optim
from ise.utils import functions as f


class ScenarioPredictor(nn.Module):

    def __init__(self, input_size, hidden_layers=[128, 64], output_size=1, dropout_rate=0.1):
        """
        Initializes the ScenarioPredictor model.

        Args:
            input_size (int): Number of input features.
            hidden_layers (list of int, optional): List specifying the number of neurons in each hidden layer. Defaults to [128, 64].
            output_size (int, optional): Number of output neurons. Defaults to 1.
            dropout_rate (float, optional): Dropout rate applied after each hidden layer. Defaults to 0.1.

        Attributes:
            device (str): Device to run the model on ('cuda' if available, otherwise 'cpu').
            input_layer (torch.nn.Linear): First linear layer of the network.
            hidden_layers (torch.nn.ModuleList): List of hidden layers.
            output_layer (torch.nn.Linear): Output layer of the network.
            activation (torch.nn.ReLU): ReLU activation function.
            dropout (torch.nn.Dropout): Dropout layer.
            sigmoid (torch.nn.Sigmoid): Sigmoid activation function for output.
            criterion (torch.nn.BCELoss): Binary Cross-Entropy loss function.
            optimizer (torch.optim.Adam): Adam optimizer for training.
        """

        super(ScenarioPredictor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize network layers
        self.input_layer = nn.Linear(input_size, hidden_layers[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        
        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Sigmoid activation for the output
        self.sigmoid = nn.Sigmoid()

        # Loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters())

        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor with probabilities in the range [0,1].
        """


        x = self.activation(self.input_layer(x))
        x = self.dropout(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            # x = self.dropout(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
      
    def fit(self, train_loader, val_loader=None, epochs=10, lr=1e-3, print_every=1, save_checkpoint=True):
        """
        Trains the model on the given dataset.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset. Defaults to None.
            epochs (int, optional): Number of epochs for training. Defaults to 10.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            print_every (int, optional): Interval for printing training progress. Defaults to 1.
            save_checkpoint (bool, optional): Whether to save model checkpoints based on validation loss. Defaults to True.

        Returns:
            None
        """

        # Set the model to training mode
        self.train()

        # Update the optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.best_val_loss = 10000
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1).float())  # Ensure targets are correctly shaped and typed
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % print_every == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')

                if val_loader is not None:
                    # Evaluate the model on the validation set
                    self.eval()  # Set the model to evaluation mode
                    val_loss, val_accuracy = self.evaluate(val_loader)
                    if save_checkpoint:
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            torch.save(self.state_dict(), 'checkpoint.pth')
                    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
                    self.train()  # Set the model back to training mode

    def evaluate(self, data_loader):
        """
        Evaluates the model on a dataset.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.

        Returns:
            tuple: A tuple containing:
                - avg_loss (float): Average loss over the dataset.
                - accuracy (float): Accuracy of predictions (0 to 1).
        """

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1).float())
                total_loss += loss.item()

                predicted = outputs.round()  # Round probabilities to obtain binary predictions
                correct_predictions += (predicted == targets.unsqueeze(1)).sum().item()
                total_predictions += targets.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions
        

        return avg_loss, accuracy
    
    def predict(self, x):
        """
        Predicts the output for a given input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted output tensor with probabilities in the range [0,1].
        """

        self.eval()
        x = f.to_tensor(x)
        with torch.no_grad():
            x = x.to(self.device)
            output = self.forward(x)
        return output

    def load(self, path):
        """
        Loads the model state from a file.

        Args:
            path (str): Path to the file containing the model state.

        Returns:
            None
        """

        self.load_state_dict(torch.load(path, map_location=self.device))
