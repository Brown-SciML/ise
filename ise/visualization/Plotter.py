from ise.utils.data import group_by_run
from ise.visualization import ensemble, testing
from ise.utils.data import create_distribution, kl_divergence, js_divergence
import ise



class Plotter:
    def __init__(self, results_dataset, column=None, condition=None, save_directory=None):
        super().__init__()
        self.dataset = results_dataset
        self.save_directory = save_directory
        self.trues, self.preds, self.scenarios = group_by_run(self.dataset, column=column,
                                                              condition=condition)
        self.true_bounds = ensemble.UncertaintyBounds(self.trues)
        self.pred_bounds = ensemble.UncertaintyBounds(self.preds)
        self.cache = {
            'true_sle_runs': self.trues,
            'pred_sle_runs': self.preds,
            'true_bounds': self.true_bounds,
            'pred_bounds': self.pred_bounds
        }
        self.true_distribution, self.support = create_distribution(year=2100, dataset=self.trues)
        self.pred_distribution, _ = create_distribution(year=2100, dataset=self.preds)
        self.distribution_metrics = {
            'kl': kl_divergence(self.pred_distribution, self.true_distribution),
            'js': js_divergence(self.pred_distribution, self.true_distribution)
        }
        self.model = None
        self.ml_directory = None

    def plot_ensemble(self, uncertainty='quantiles', column=None, condition=None, save=None,):
        return ensemble.plot_ensemble(
            dataset=self.dataset, uncertainty=uncertainty, column=column, 
            condition=condition, save=save, cache=self.cache,
        )

    def plot_ensemble_mean(self, uncertainty=False, column=None, condition=None, save=None, ):
        return ensemble.plot_ensemble_mean(
            dataset=self.dataset, uncertainty=uncertainty, column=column, 
            condition=condition, save=save, cache=self.cache,
        )

    def plot_distributions(self, year, column=None, condition=None, save=None,):
        return ensemble.plot_distributions(
            dataset=self.dataset, year=year, column=column, 
            condition=condition, save=save, cache=self.cache,
        )

    def plot_histograms(self, year, column=None, condition=None, save=None,):
        return ensemble.plot_histograms(
            dataset=self.dataset, year=year, column=column,
            condition=condition, save=save, cache=self.cache,
        )

    def plot_test_series(self, model, data_directory, time_series=True, 
                         approx_dist=True, mc_iterations=100, confidence='95', 
                         draws='random', k=10, save_directory=None):
        if not isinstance(model, ise.models.timeseries.TimeSeriesEmulator):
            raise NotImplementedError(
                'currently the only model compatible with this function is TimeSeriesEmulator.'
            )
        self.model = model
        self.ml_directory = data_directory
        return testing.plot_test_series(
            model=model, data_directory=data_directory, time_series=time_series,
            approx_dist=approx_dist, mc_iterations=mc_iterations,
            confidence=confidence, draws=draws, k=k, save_directory=save_directory
        )

    def plot_callibration(self, color_by=None, alpha=0.2, column=None, condition=None, save=None):
        return testing.plot_callibration(
            dataset=self.dataset, column=column, condition=condition,
            color_by=color_by, alpha=alpha, save=save
        )
