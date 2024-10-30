import torch
import warnings

class CheckpointSaver:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str, verbose: bool = False):
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.optimizer = optimizer
        self.best_loss = float('inf')
        self.verbose = verbose
        self.log = None
        
    def __call__(self, loss, epoch, save_best_only=True,):
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
        # Determine if current loss is better than best_loss
        return loss < self.best_loss
        
    def _update_best_loss(self, loss):
        self.best_loss = loss
    
    def save_checkpoint(self, epoch, loss, path: str = None):
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
    def __init__(self, model, optimizer, checkpoint_path='checkpoint.pt', patience=10, verbose=False):
        super().__init__(model, optimizer, checkpoint_path, verbose)
        self.patience = patience
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss, epoch, save_best_only=True,):
        saved = super().__call__(loss, epoch, save_best_only,)
        if saved:
            self.counter = 0  # Reset counter if the model improved
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
