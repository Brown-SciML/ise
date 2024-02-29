import numpy as np
import matplotlib.pyplot as plt

class EvaluationPlotter:
    def __init__(self,  save_dir='.'):
    
        self.save_dir = save_dir
    
    def spatial_side_by_side(self, y_true, y_pred, timestep=None, save_path=None, cmap=plt.cm.RdBu):
        # Create a custom colormap for masked values (white)
        
        if len(y_true.shape) == 3 and len(y_pred.shape) == 3 and timestep is None:
            raise ValueError("timestep must be specified for 3D arrays")
        elif len(y_true.shape) == 3 and len(y_pred.shape) == 3 and timestep is not None:
            self.y_true = y_true[timestep-1, :, :]
            self.y_pred = y_pred[timestep-1, :, :]
        else:
            self.y_true = y_true
            self.y_pred = y_pred
        difference = np.abs(self.y_pred - self.y_true)
        masked_y_true = np.ma.masked_equal(self.y_true, 0)
        masked_y_pred = np.ma.masked_equal(self.y_pred, 0)
        masked_difference = np.ma.masked_equal(difference, 0)
        global_min = min(masked_y_true.min(), masked_y_pred.min())
        global_max = max(masked_y_true.max(), masked_y_pred.max())
        
        global_extreme = max(abs(global_min), abs(global_max))
        cmap.set_bad(color='white')

        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot y_true with mask, align color scale
        cax1 = axs[0].imshow(masked_y_true, cmap=cmap, vmin=global_extreme*-1, vmax=global_extreme)
        fig.colorbar(cax1, ax=axs[0], orientation='vertical')
        axs[0].set_title('True Y')

        # Plot y_pred with mask, align color scale
        cax2 = axs[1].imshow(masked_y_pred, cmap=cmap, vmin=global_extreme*-1, vmax=global_extreme)
        fig.colorbar(cax2, ax=axs[1], orientation='vertical')
        axs[1].set_title('Predicted Y')

        # Plot absolute difference with mask, using 'Reds' colormap
        cax3 = axs[2].imshow(masked_difference, cmap='Reds')
        fig.colorbar(cax3, ax=axs[2], orientation='vertical')
        axs[2].set_title('Absolute Difference |Y_pred - Y_true|')

        # Show plot
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(f"{self.save_dir}/{save_path}", dpi=600)
        plt.show()