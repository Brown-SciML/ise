import torch
import torch.nn as nn
import torch.optim as optim
from ise.utils import functions as f


class ScenarioPredictor(nn.Module):

    
    def __init__(self, input_size, hidden_layers=[128, 64], output_size=1, dropout_rate=0.1):
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

        x = self.activation(self.input_layer(x))
        x = self.dropout(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            # x = self.dropout(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
      
    def fit(self, train_loader, val_loader=None, epochs=10, lr=1e-3, print_every=1):

        # Set the model to training mode
        self.train()

        # Update the optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

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
                    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
                    self.train()  # Set the model back to training mode

    def evaluate(self, data_loader):
        """
        Evaluates the model on a dataset.

        Args:
            data_loader (DataLoader): DataLoader for the dataset to evaluate.

        Returns:
            tuple: A tuple containing the average loss and accuracy on the dataset.
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
            torch.Tensor: Predicted output tensor.
        """
        self.eval()
        x = f.to_tensor(x)
        with torch.no_grad():
            x = x.to(self.device)
            output = self.forward(x)
        return output

