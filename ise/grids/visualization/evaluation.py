import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import warnings
from tqdm import tqdm
import xarray as xr
from ise.grids.evaluation.metrics import mean_squared_error_sector

class EvaluationPlotter:
    def __init__(self,  save_dir='.'):
    
        self.save_dir = save_dir
        self.video = False
    
    def spatial_side_by_side(self, y_true, y_pred, timestep=None, save_path=None, cmap=plt.cm.RdBu, video=False):
        
        if video and timestep:
            warnings.warn("Video will be generated, ignoring timestep argument.")
        # Create a custom colormap for masked values (white)
        
        if video:
            self.video = True
            self._generate_side_by_side_video(y_true, y_pred, fps=3)
            return self
        
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
            if self.video:
                plt.savefig(f"{self.save_dir}/{save_path}", )
            else:
                plt.savefig(f"{self.save_dir}/{save_path}", dpi=600)
            
        plt.close('all')
        
    def _generate_side_by_side_video(self, y_true, y_pred, fps=3):
        if not (len(y_true.shape) == 3 and len(y_pred.shape) == 3):
            raise ValueError("y_true and y_pred must be 3D arrays with shape (timesteps, height, width)")

        timesteps = y_true.shape[0]

        for timestep in tqdm(range(timesteps), total=timesteps, desc="Generating video"):
            save_path = f"timestep_{timestep}.png"  # Save each frame with timestep
            self.spatial_side_by_side(y_true, y_pred, timestep, save_path, cmap=plt.cm.viridis, video=False)

        images = []
        # Improved sorting function that handles unexpected filenames more gracefully
        try:
            files = sorted(os.listdir(self.save_dir), key=lambda x: int(x.replace("timestep_", "").split(".")[0]))
        except ValueError:
            raise ValueError("Unexpected filenames found in save directory. Expected format: 'timestep_#.png'")
        for filename in files:
            if filename.endswith(".png"):
                image_path = os.path.join(self.save_dir, filename)
                images.append(imageio.imread(image_path))

        # Create a video from the images
        video_path = f'{self.save_dir}/plot_video.mp4'
        imageio.mimwrite(video_path, images, fps=fps, codec='libx264')  # fps is frames per second

    
    def sector_side_by_side(self, y_true, y_pred, grid_file, outline_array_true=None, outline_array_pred=None, timestep=None, save_path=None, cmap=plt.cm.RdBu,):
    
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        if y_pred.shape[1] != 18 and y_pred.shape[1] != 6:
            raise ValueError("y_pred must have 18 sectors.")

        if len(y_true.shape) == 2 and len(y_pred.shape) == 2 and timestep is None:
            raise ValueError("timestep must be specified for 2D arrays")
        elif len(y_true.shape) == 2 and len(y_pred.shape) == 2 and timestep is not None:
            self.y_true = y_true[timestep-1, :]
            self.y_pred = y_pred[timestep-1, :]
            outline_array_pred = outline_array_pred[timestep-1, :]
            outline_array_true = outline_array_true[timestep-1, :]
        else:
            self.y_true = y_true
            self.y_pred = y_pred
            
        if isinstance(grid_file, str):
            grids = xr.open_dataset(grid_file).transpose('x', 'y', ...)
            sector_name = 'sectors' if 'ais' in grid_file.lower() else 'ID'
        elif isinstance(grid_file, xr.Dataset):
            sector_name = 'ID' if 'Rignot' in grids.Description else 'sectors'
        else:
            raise ValueError("grid_file must be a string or an xarray Dataset.")
        
        sectors = grids[sector_name].values
        true_plot_data = np.zeros_like(sectors)
        pred_plot_data = np.zeros_like(sectors)
        
        num_sectors = 18 if sector_name == 'sectors' else 6

        for sector in range(1, num_sectors+1):
            true_plot_data[sectors == sector] = self.y_true[sector-1]
            pred_plot_data[sectors == sector] = self.y_pred[sector-1]
            
        # Convert outline arrays to binary masks
        outline_mask_true = np.where(outline_array_true != 0, 1, 0)
        outline_mask_pred = np.where(outline_array_pred != 0, 1, 0)

        # Define the color scale based on the combined range of true and predicted matrices
        vmin = min(true_plot_data.min(), pred_plot_data.min())
        vmax = max(true_plot_data.max(), pred_plot_data.max())

        # Create a figure and a set of subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'wspace': 0.5})

        # Plot the modified outline array for the true matrix (black for non-zero values, white elsewhere)
        axs[0].imshow(np.flipud(outline_mask_true.T), cmap='Greys', interpolation='nearest')
        # Plot the true matrix with slight transparency
        cax1 = axs[0].imshow(np.flipud(true_plot_data.T), cmap='Reds', interpolation='nearest', vmin=vmin, vmax=vmax, alpha=0.90)
        fig.colorbar(cax1, ax=axs[0], fraction=0.046, pad=0.04)
        axs[0].set_title('True')

        # Plot the modified outline array for the predicted matrix (black for non-zero values, white elsewhere)
        axs[1].imshow(np.flipud(outline_mask_pred.T), cmap='Greys', interpolation='nearest')
        # Plot the predicted matrix with slight transparency
        cax2 = axs[1].imshow(np.flipud(pred_plot_data.T), cmap='Reds', interpolation='nearest', vmin=vmin, vmax=vmax, alpha=0.90)
        fig.colorbar(cax2, ax=axs[1], fraction=0.046, pad=0.04)
        axs[1].set_title('Predicted')
        
        mse = mean_squared_error_sector(sum_sectors_true, sum_sectors_pred)
        plt.suptitle(f"Mean Squared Error: {mse:0.2f}")
        # plt.tight_layout()
        
        
        if save_path is not None:
            plt.savefig(f"{self.save_dir}/{save_path}", dpi=600)
            
            
        plt.close('all')
        
        
        stop = ''
        
        
        
