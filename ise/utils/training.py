import torch
import warnings

class CheckpointSaver:
    """
    A class to handle saving and loading of model checkpoints during training.

    This class monitors the model's loss and saves the model's state when an improvement is detected.
    It can also be configured to save the model at every epoch.

    Attributes:
        checkpoint_path (str): Path where the checkpoint will be saved.
        model (torch.nn.Module): The PyTorch model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        best_loss (float): The best recorded loss value. Initially set to infinity.
        verbose (bool): If True, logs messages when a checkpoint is saved.
        log (str or None): Stores log messages for saving actions.

    Methods:
        __call__(loss, epoch, save_best_only=True):
            Checks whether to save the checkpoint based on loss improvement.

        _determine_if_better(loss):
            Determines if the new loss is an improvement over the best recorded loss.

        _update_best_loss(loss):
            Updates the best recorded loss.

        save_checkpoint(epoch, loss, path=None):
            Saves the model's state, optimizer state, and epoch information.

        load_checkpoint(path=None):
            Loads a saved checkpoint and restores model and optimizer states.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str, verbose: bool = False):
        """
        Initializes the CheckpointSaver instance.

        Args:
            model (torch.nn.Module): The PyTorch model to be saved and restored.
            optimizer (torch.optim.Optimizer): The optimizer associated with the model.
            checkpoint_path (str): Path to save the checkpoint file.
            verbose (bool, optional): Whether to print logs when saving checkpoints. Defaults to False.
        """

        self.checkpoint_path = checkpoint_path
        self.model = model
        self.optimizer = optimizer
        self.best_loss = float('inf')
        self.verbose = verbose
        self.log = None
        
    def __call__(self, loss, epoch, save_best_only=True):
        """
        Determines whether to save the checkpoint based on the loss.

        Args:
            loss (float): The current loss value.
            epoch (int): The current training epoch.
            save_best_only (bool, optional): If True, saves the checkpoint only when the loss improves.
                                            If False, saves the checkpoint at every call. Defaults to True.

        Returns:
            bool: True if a checkpoint was saved, False otherwise.
        """

        is_better = self._determine_if_better(loss) if save_best_only else True

        if is_better or not save_best_only:  # Save if loss improves or save_best_only is False
            self.save_checkpoint(epoch, loss, self.checkpoint_path)
            if self.verbose:
                self.log = f"Loss decreased ({self.best_loss:.6f} --> {loss:.6f}). Saving checkpoint to {self.checkpoint_path}."
            self._update_best_loss(loss)
            return True
        else:
            self.log = ""
        return False
        
    def _determine_if_better(self, loss: float):
        """
        Checks if the new loss value is lower than the best recorded loss.

        Args:
            loss (float): The current loss value.

        Returns:
            bool: True if the loss has improved, False otherwise.
        """

        # Determine if current loss is better than best_loss
        return loss < self.best_loss
        
    def _update_best_loss(self, loss):
        """
        Updates the best recorded loss with the new value.

        Args:
            loss (float): The new best loss value.
        """

        self.best_loss = loss
    
    def save_checkpoint(self, epoch, loss, path: str = None):
        """
        Saves the model checkpoint, including model state, optimizer state, and epoch.

        Args:
            epoch (int): The current epoch number.
            loss (float): The loss value associated with this checkpoint.
            path (str, optional): The file path to save the checkpoint. If None, the default path is used.
        """

        checkpoint_path = path or self.checkpoint_path
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        # if self.verbose:
        #     print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, path: str = None):
        """
        Loads a checkpoint and restores the model and optimizer states.

        Args:
            path (str, optional): The file path to load the checkpoint from. If None, the default path is used.

        Returns:
            int: The epoch number from which training should resume.
        """

        checkpoint_path = path or self.checkpoint_path
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1
        if self.verbose:
            print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
        return start_epoch

class EarlyStoppingCheckpointer(CheckpointSaver):
    """
    A class that extends CheckpointSaver to implement early stopping.

    This class tracks model performance and stops training when the validation loss does not improve 
    for a specified number of epochs (patience).

    Attributes:
        patience (int): The number of epochs with no improvement before stopping.
        counter (int): Tracks the number of epochs since the last improvement.
        early_stop (bool): Flag indicating whether early stopping should occur.

    Methods:
        __call__(loss, epoch, save_best_only=True):
            Saves the checkpoint and updates early stopping conditions.
    """

    def __init__(self, model, optimizer, checkpoint_path='checkpoint.pt', patience=10, verbose=False):
        """
        Initializes the EarlyStoppingCheckpointer.

        Args:
            model (torch.nn.Module): The PyTorch model to be saved and monitored for early stopping.
            optimizer (torch.optim.Optimizer): The optimizer used during training.
            checkpoint_path (str, optional): Path to save the checkpoint file. Defaults to 'checkpoint.pt'.
            patience (int, optional): Number of epochs to wait before stopping if no improvement is detected. Defaults to 10.
            verbose (bool, optional): Whether to print logs when early stopping is triggered. Defaults to False.
        """

        super().__init__(model, optimizer, checkpoint_path, verbose)
        self.patience = patience
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss, epoch, save_best_only=True):
        """
        Saves the checkpoint and updates the early stopping counter.

        Args:
            loss (float): The current loss value.
            epoch (int): The current training epoch.
            save_best_only (bool, optional): If True, saves the checkpoint only when loss improves.
                                            If False, saves the checkpoint at every call. Defaults to True.

        Side Effects:
            - Resets the early stopping counter if the checkpoint is saved.
            - Increments the counter if no improvement is observed.
            - Sets the `early_stop` flag to True if the counter reaches the patience threshold.
        """

        saved = super().__call__(loss, epoch, save_best_only,)
        if saved:
            self.counter = 0  # Reset counter if the model improved
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
