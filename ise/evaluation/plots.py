import numpy as np
import os
import imageio
import warnings
from tqdm import tqdm
import xarray as xr
from ise.evaluation.metrics import sum_by_sector, mean_squared_error_sector
from ise.evaluation.plots import UncertaintyBounds, plot_ensemble, plot_ensemble_mean, plot_distributions, plot_histograms, plot_test_series, plot_callibration
from ise.utils.functions import group_by_run, get_uncertainty_bands, create_distribution, kl_divergence, js_divergence, load_ml_data
import ise
import pandas as pd
import seaborn as sns
import random
import torch
import matplotlib.pyplot as plt
        


class SectorPlotter:
    def __init__(
        self, results_dataset, column=None, condition=None, save_directory=None
    ):
        super().__init__()
        self.dataset = results_dataset
        self.save_directory = save_directory
        self.trues, self.preds, self.scenarios = group_by_run(
            self.dataset, column=column, condition=condition
        )
        self.true_bounds = UncertaintyBounds(self.trues)
        self.pred_bounds = UncertaintyBounds(self.preds)
        self.cache = {
            "true_sle_runs": self.trues,
            "pred_sle_runs": self.preds,
            "true_bounds": self.true_bounds,
            "pred_bounds": self.pred_bounds,
        }
        self.true_distribution, self.support = create_distribution(
            year=2100, dataset=self.trues
        )
        self.pred_distribution, _ = create_distribution(year=2100, dataset=self.preds)
        self.distribution_metrics = {
            "kl": kl_divergence(self.pred_distribution, self.true_distribution),
            "js": js_divergence(self.pred_distribution, self.true_distribution),
        }
        self.model = None
        self.ml_directory = None

    def plot_ensemble(
        self,
        uncertainty="quantiles",
        column=None,
        condition=None,
        save=None,
    ):
        return plot_ensemble(
            dataset=self.dataset,
            uncertainty=uncertainty,
            column=column,
            condition=condition,
            save=save,
            cache=self.cache,
        )

    def plot_ensemble_mean(
        self,
        uncertainty=False,
        column=None,
        condition=None,
        save=None,
    ):
        return plot_ensemble_mean(
            dataset=self.dataset,
            uncertainty=uncertainty,
            column=column,
            condition=condition,
            save=save,
            cache=self.cache,
        )

    def plot_distributions(
        self,
        year,
        column=None,
        condition=None,
        save=None,
    ):
        return plot_distributions(
            dataset=self.dataset,
            year=year,
            column=column,
            condition=condition,
            save=save,
            cache=self.cache,
        )

    def plot_histograms(
        self,
        year,
        column=None,
        condition=None,
        save=None,
    ):
        return plot_histograms(
            dataset=self.dataset,
            year=year,
            column=column,
            condition=condition,
            save=save,
            cache=self.cache,
        )

    def plot_test_series(
        self,
        model,
        data_directory,
        time_series=True,
        approx_dist=True,
        mc_iterations=100,
        confidence="95",
        draws="random",
        k=10,
        save_directory=None,
    ):
        if not isinstance(model, ise.models.timeseries.TimeSeriesEmulator):
            raise NotImplementedError(
                "currently the only model compatible with this function is TimeSeriesEmulator."
            )
        self.model = model
        self.ml_directory = data_directory
        return plot_test_series(
            model=model,
            data_directory=data_directory,
            time_series=time_series,
            approx_dist=approx_dist,
            mc_iterations=mc_iterations,
            confidence=confidence,
            draws=draws,
            k=k,
            save_directory=save_directory,
        )

    def plot_callibration(
        self, color_by=None, alpha=0.2, column=None, condition=None, save=None
    ):
        return plot_callibration(
            dataset=self.dataset,
            column=column,
            condition=condition,
            color_by=color_by,
            alpha=alpha,
            save=save,
        )


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
        
        sum_by_sector_true = sum_by_sector(self.y_true, grid_file)
        sum_by_sector_pred = sum_by_sector(self.y_pred, grid_file)
        
        mse = mean_squared_error_sector(sum_by_sector_true, sum_by_sector_pred)
        plt.suptitle(f"Mean Squared Error: {mse:0.2f}")
        # plt.tight_layout()
        
        
        if save_path is not None:
            plt.savefig(f"{self.save_dir}/{save_path}", dpi=600)
            
            
        plt.close('all')
        
        
        stop = ''
        



class UncertaintyBounds:
    def __init__(self, data, confidence="95", quantiles=None):
        if quantiles is None:
            quantiles = [0.05, 0.95]
        self.data = data
        (
            self.mean,
            self.sd,
            self.upper_ci,
            self.lower_ci,
            self.upper_q,
            self.lower_q,
        ) = get_uncertainty_bands(data, confidence=confidence, quantiles=quantiles)


def plot_ensemble(
    dataset: pd.DataFrame,
    uncertainty: str = "quantiles",
    column: str = None,
    condition: str = None,
    save: str = None,
    cache: dict = None,
):
    """Generates a plot of the comparison of ensemble results from the true simulations and the predicted emulation.
    Adds uncertainty bounds and plots them side-by-side.

    Args:
        dataset (pd.DataFrame): testing results dataframe, result from [ise.utils.data.combine_testing_results](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#combine_testing_results).
        uncertainty (str, optional): Type of uncertainty for creating bounds, must be in [quantiles, confidence]. Defaults to 'quantiles'.
        column (str, optional): Column to subset on, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#group_by_run). Defaults to None.
        condition (str, optional): Condition to subset with, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#group_by_run). Can be int, str, float, etc. Defaults to None.
        save (str, optional): Path to save plot. Defaults to None.
        cache (dict, optional): Cached results from previous calculation, used internally in [ise.visualization.Plotter](https://brown-sciml.github.io/ise/ise/sectors/visualization/Plotter.html#Plotter). Defaults to None.
    """

    if cache is None:
        all_trues, all_preds, _ = group_by_run(
            dataset, column=column, condition=condition
        )
        (
            mean_true,
            _,
            true_upper_ci,
            true_lower_ci,
            true_upper_q,
            true_lower_q,
        ) = get_uncertainty_bands(
            all_trues,
        )
        (
            mean_pred,
            _,
            pred_upper_ci,
            pred_lower_ci,
            pred_upper_q,
            pred_lower_q,
        ) = get_uncertainty_bands(
            all_preds,
        )
    else:
        all_trues = cache["true_sle_runs"]
        all_preds = cache["pred_sle_runs"]
        t = cache["true_bounds"]
        p = cache["pred_bounds"]
        mean_true, true_upper_ci, true_lower_ci, true_upper_q, true_lower_q = (
            t.mean,
            t.upper_ci,
            t.lower_ci,
            t.upper_q,
            t.lower_q,
        )
        mean_pred, pred_upper_ci, pred_lower_ci, pred_upper_q, pred_lower_q = (
            p.mean,
            p.upper_ci,
            p.lower_ci,
            p.upper_q,
            p.lower_q,
        )

    true_df = pd.DataFrame(all_trues).transpose()
    pred_df = pd.DataFrame(all_preds).transpose()

    _, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True, sharex=True)
    axs[0].plot(true_df)
    axs[0].plot(mean_true, "r-", linewidth=4, label="Mean")
    axs[1].plot(pred_df)
    axs[1].plot(mean_pred, "r-", linewidth=4, label="Mean")
    if uncertainty and uncertainty.lower() == "confidence":
        axs[0].plot(true_upper_ci, "b--", linewidth=3, label="5/95% Confidence (True)")
        axs[0].plot(true_lower_ci, "b--", linewidth=3)
        axs[1].plot(
            pred_upper_ci, "b--", linewidth=3, label="5/95% Confidence (Predicted)"
        )
        axs[1].plot(pred_lower_ci, "b--", linewidth=3)

    elif uncertainty and uncertainty.lower() == "quantiles":
        axs[0].plot(
            pred_upper_q, "b--", linewidth=3, label="5/95% Percentile (Predicted)"
        )
        axs[0].plot(pred_lower_q, "b--", linewidth=3)
        axs[1].plot(true_upper_q, "b--", linewidth=3, label="5/95% Percentile (True)")
        axs[1].plot(true_lower_q, "b--", linewidth=3)

    elif uncertainty and uncertainty.lower() == "both":
        axs[0].plot(true_upper_ci, "r--", linewidth=2, label="5/95% Confidence (True)")
        axs[0].plot(true_lower_ci, "r--", linewidth=2)
        axs[1].plot(
            pred_upper_ci, "b--", linewidth=2, label="5/95% Confidence (Predicted)"
        )
        axs[1].plot(pred_lower_ci, "b--", linewidth=2)
        axs[1].plot(
            pred_upper_q, "o--", linewidth=2, label="5/95% Percentile (Predicted)"
        )
        axs[1].plot(pred_lower_q, "o--", linewidth=2)
        axs[0].plot(true_upper_q, "k--", linewidth=2, label="5/95% Percentile (True)")
        axs[0].plot(true_lower_q, "k--", linewidth=2)

    elif uncertainty and uncertainty.lower() not in ["confidence", "quantiles"]:
        raise AttributeError(
            f"uncertainty argument must be in ['confidence', 'quantiles'], received {uncertainty}"
        )

    axs[0].title.set_text("True")
    axs[0].set_ylabel("True SLE (mm)")
    axs[1].title.set_text("Predicted")
    plt.xlabel("Years since 2015")
    if column is not None and condition is not None:
        plt.suptitle(f"Time Series of ISM Ensemble - where {column} == {condition}")
    else:
        plt.suptitle("Time Series of ISM Ensemble")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.legend()

    # TODO: FileNotFoundError: [Errno 2] No such file or directory: 'None/ensemble_plot.png'
    if save:
        plt.savefig(save)


def plot_ensemble_mean(
    dataset: pd.DataFrame,
    uncertainty: str = False,
    column=None,
    condition=None,
    save=None,
    cache=None,
):
    """Generates a plot of the mean sea level contribution from the true simulations and the predicted emulation.

    Args:
        dataset (pd.DataFrame): testing results dataframe, result from [ise.utils.data.combine_testing_results](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#combine_testing_results).
        uncertainty (str, optional): Type of uncertainty for creating bounds. If not None/False, must be in [quantiles, confidence]. Defaults to 'quantiles'.
        column (str, optional): Column to subset on, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#group_by_run). Defaults to None.
        condition (str, optional): Condition to subset with, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#group_by_run). Can be int, str, float, etc. Defaults to None.
        save (str, optional): Path to save plot. Defaults to None.
        cache (dict, optional): Cached results from previous calculation, used internally in [ise.visualization.Plotter](https://brown-sciml.github.io/ise/ise/sectors/visualization/Plotter.html#Plotter). Defaults to None.
    """

    if cache is None:
        all_trues, all_preds, _ = group_by_run(
            dataset, column=column, condition=condition
        )
        (
            mean_true,
            _,
            true_upper_ci,
            true_lower_ci,
            true_upper_q,
            true_lower_q,
        ) = get_uncertainty_bands(
            all_trues,
        )
        (
            mean_pred,
            _,
            pred_upper_ci,
            pred_lower_ci,
            pred_upper_q,
            pred_lower_q,
        ) = get_uncertainty_bands(
            all_preds,
        )
    else:
        all_trues = cache["true_sle_runs"]
        all_preds = cache["pred_sle_runs"]
        t = cache["true_bounds"]
        p = cache["pred_bounds"]
        mean_true, true_upper_ci, true_lower_ci, true_upper_q, true_lower_q = (
            t.mean,
            t.upper_ci,
            t.lower_ci,
            t.upper_q,
            t.lower_q,
        )
        mean_pred, pred_upper_ci, pred_lower_ci, pred_upper_q, pred_lower_q = (
            p.mean,
            p.upper_ci,
            p.lower_ci,
            p.upper_q,
            p.lower_q,
        )

    plt.figure(figsize=(15, 6))
    plt.plot(mean_true, label="True Mean SLE")
    plt.plot(mean_pred, label="Predicted Mean SLE")

    if uncertainty and uncertainty.lower() == "confidence":
        plt.plot(true_upper_ci, "r--", linewidth=2, label="5/95% Percentile (True)")
        plt.plot(true_lower_ci, "r--", linewidth=2)
        plt.plot(
            pred_upper_ci, "b--", linewidth=2, label="5/95% Percentile (Predicted)"
        )
        plt.plot(pred_lower_ci, "b--", linewidth=2)

    elif uncertainty and uncertainty.lower() == "quantiles":
        plt.plot(pred_upper_q, "r--", linewidth=2, label="5/95% Confidence (Predicted)")
        plt.plot(pred_lower_q, "r--", linewidth=2)
        plt.plot(true_upper_q, "b--", linewidth=2, label="5/95% Confidence (True)")
        plt.plot(true_lower_q, "b--", linewidth=2)

    elif uncertainty and uncertainty.lower() == "both":
        plt.plot(true_upper_ci, "r--", linewidth=2, label="5/95% Percentile (True)")
        plt.plot(true_lower_ci, "r--", linewidth=2)
        plt.plot(
            pred_upper_ci, "b--", linewidth=2, label="5/95% Percentile (Predicted)"
        )
        plt.plot(pred_lower_ci, "b--", linewidth=2)
        plt.plot(pred_upper_q, "o--", linewidth=2, label="5/95% Confidence (Predicted)")
        plt.plot(pred_lower_q, "o--", linewidth=2)
        plt.plot(true_upper_q, "k--", linewidth=2, label="5/95% Confidence (True)")
        plt.plot(true_lower_q, "k--", linewidth=2)

    elif uncertainty and uncertainty.lower() not in ["confidence", "quantiles"]:
        raise AttributeError(
            f"uncertainty argument must be in ['confidence', 'quantiles'], received {uncertainty}"
        )

    else:
        pass

    if column is not None and condition is not None:
        plt.suptitle(f"ISM Ensemble Mean SLE over Time - where {column} == {condition}")
    else:
        plt.suptitle("ISM Ensemble Mean over Time")
    plt.xlabel("Years since 2015")
    plt.ylabel("Mean SLE (mm)")
    plt.legend()

    if save:
        plt.savefig(save)


def plot_distributions(
    dataset: pd.DataFrame,
    year: int = 2100,
    column: str = None,
    condition: str = None,
    save: str = None,
    cache: dict = None,
):
    """Generates a plot of comparison of distributions at a given time slice (year) from the true simulations and the predicted emulation.

    Args:
        dataset (pd.DataFrame): testing results dataframe, result from [ise.utils.data.combine_testing_results](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#combine_testing_results).
        year (int, optional): Distribution year (time slice). Defaults to 2100.
        column (str, optional): Column to subset on, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#group_by_run). Defaults to None.
        condition (str, optional): Condition to subset with, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#group_by_run). Can be int, str, float, etc. Defaults to None.
        save (str, optional): Path to save plot. Defaults to None.
        cache (dict, optional): Cached results from previous calculation, used internally in [ise.visualization.Plotter](https://brown-sciml.github.io/ise/ise/sectors/visualization/Plotter.html#Plotter). Defaults to None.
    """

    if cache is None:
        all_trues, all_preds, _ = group_by_run(
            dataset, column=column, condition=condition
        )
    else:
        all_trues = cache["true_sle_runs"]
        all_preds = cache["pred_sle_runs"]

    true_dist, true_support = create_distribution(year=year, dataset=all_trues)
    pred_dist, pred_support = create_distribution(year=year, dataset=all_preds)
    plt.figure(figsize=(15, 8))
    plt.plot(true_support, true_dist, label="True")
    plt.plot(pred_support, pred_dist, label="Predicted")
    plt.title(
        f"Distribution Comparison at year {year}, KL Divergence: {kl_divergence(pred_dist, true_dist):0.3f}"
    )
    plt.xlabel("SLE (mm)")
    plt.ylabel("Probability")
    plt.legend()
    if save:
        plt.savefig(save)


def plot_histograms(
    dataset: pd.DataFrame,
    year: int = 2100,
    column: str = None,
    condition: str = None,
    save: str = None,
    cache: dict = None,
):
    """Generates a plot of comparison of histograms at a given time slice (year) from the true simulations and the predicted emulation.

    Args:
        dataset (pd.DataFrame): testing results dataframe, result from [ise.utils.data.combine_testing_results](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#combine_testing_results).
        year (int, optional): Histogram year (time slice). Defaults to 2100.
        column (str, optional): Column to subset on, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#group_by_run). Defaults to None.
        condition (str, optional): Condition to subset with, used in [ise.utils.data.group_by_run](https://brown-sciml.github.io/ise/ise/sectors/utils/data.html#group_by_run). Can be int, str, float, etc. Defaults to None.
        save (str, optional): Path to save plot. Defaults to None.
        cache (dict, optional): Cached results from previous calculation, used internally in [ise.visualization.Plotter](https://brown-sciml.github.io/ise/ise/sectors/visualization/Plotter.html#Plotter). Defaults to None.
    """
    if cache is None:
        all_trues, all_preds, _ = group_by_run(
            dataset, column=column, condition=condition
        )

    else:
        all_trues = cache["true_sle_runs"]
        all_preds = cache["pred_sle_runs"]

    fig = plt.figure(figsize=(15, 8))
    ax1 = plt.subplot(
        1,
        2,
        1,
    )
    sns.histplot(
        all_preds[:, year - 2101],
        label="Predicted Distribution",
        color="blue",
        alpha=0.3,
    )
    plt.legend()
    plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    sns.histplot(
        all_trues[:, year - 2101], label="True Distribution", color="red", alpha=0.3
    )
    plt.suptitle(f"Histograms of Predicted vs True SLE at year {year}")
    plt.ylabel("")
    plt.legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig(save)
        


def plot_test_series(
    model,
    data_directory,
    time_series,
    approx_dist=True,
    mc_iterations=100,
    confidence="95",
    draws="random",
    k=10,
    save_directory=None,
):
    _, _, test_features, test_labels, test_scenarios = load_ml_data(
        data_directory, time_series=time_series
    )

    sectors = list(set(test_features.sectors))
    sectors.sort()

    if draws == "random":
        data = random.sample(test_scenarios, k=k)
    elif draws == "first":
        data = test_scenarios[:k]
    else:
        raise ValueError(f"draws must be in [random, first], received {draws}")

    for scen in data:
        single_scenario = scen
        test_model = single_scenario[0]
        test_exp = single_scenario[2]
        test_sector = single_scenario[1]
        single_test_features = torch.tensor(
            np.array(
                test_features[
                    (test_features[test_model] == 1)
                    & (test_features[test_exp] == 1)
                    & (test_features.sectors == test_sector)
                ],
                dtype=np.float64,
            ),
            dtype=torch.float,
        )
        single_test_labels = np.array(
            test_labels[
                (test_features[test_model] == 1)
                & (test_features[test_exp] == 1)
                & (test_features.sectors == test_sector)
            ],
            dtype=np.float64,
        )
        preds, means, sd = model.predict(
            single_test_features,
            approx_dist=approx_dist,
            mc_iterations=mc_iterations,
            confidence=confidence,
        )  # TODO: this doesn't work with traditional
        
        quantiles = np.quantile(preds, [0.05, 0.95], axis=0)
        lower_ci = means - 1.96*sd
        upper_ci = means + 1.96*sd
        upper_q = quantiles[1, :]
        lower_q = quantiles[0, :]

        if not approx_dist:
            plt.figure(figsize=(15, 8))
            plt.plot(single_test_labels, "r-", label="True")
            plt.plot(preds, "b-", label="Predicted")
            plt.xlabel("Time (years since 2015)")
            plt.ylabel("SLE (mm)")
            plt.title(
                f"Model={test_model}, Exp={test_exp}, sector={sectors.index(test_sector)+1}"
            )
            plt.legend()
            if save_directory:
                plt.savefig(f"{save_directory}/{test_model}_{test_exp}_test_sector.png")
        else:
            preds = pd.DataFrame(preds).transpose()
            plt.figure(figsize=(15, 8))
            plt.plot(
                preds,
                alpha=0.2,
            )
            plt.plot(means, "b-", label="Predicted")
            plt.plot(upper_ci, "k-", label=f"{confidence}% CI")
            plt.plot(
                lower_ci,
                "k-",
            )
            plt.plot(quantiles[0, :], "k--", label=f"Quantiles")
            plt.plot(quantiles[1, :], "k--")
            plt.plot(
                lower_ci,
                "k-",
            )
            plt.plot(single_test_labels, "r-", label="True")

            plt.xlabel("Time (years since 2015)")
            plt.ylabel("SLE (mm)")
            plt.title(
                f"Model={test_model}, Exp={test_exp}, sector={sectors.index(test_sector)+1}"
            )
            plt.legend()
            if save_directory:
                plt.savefig(
                    f'{save_directory}/{test_model.replace("-", "_")}_{test_exp}_test_sector.png'
                )


def plot_callibration(
    dataset, column=None, condition=None, color_by=None, alpha=0.2, save=None
):

    # TODO: Add ability to subset multiple columns and conditions. Not needed now so saving for later...
    if column is None and condition is None:
        subset = dataset
    elif column is not None and condition is not None:
        subset = dataset[(dataset[column] == condition)]
    else:
        raise ValueError(
            "Column and condition type must be the same (None & None, not None & not None)."
        )

    plt.figure(figsize=(15, 8))
    sns.scatterplot(data=subset, x="true", y="pred", hue=color_by, alpha=alpha)
    plt.plot(
        [min(subset.true), max(subset.true)],
        [min(subset.true), max(subset.true)],
        "r-",
    )

    # TODO: Add density plots (below)
    # sns.kdeplot(data=subset, x='true', y='pred', hue=color_by, fill=True)
    # plt.plot([min(subset.true),max(subset.true)], [min(subset.true),max(subset.true)], 'r-',)

    # TODO: add plotly export
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("Callibration Plot")

    if color_by is not None:
        plt.legend()

    if save:
        plt.savefig(save)


