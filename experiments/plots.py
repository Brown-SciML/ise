from experiments.simulation import SimulationEnsemble
import matplotlib.pyplot as plt
import pandas as pd
import os

class EnsemblePlotter:
    
    def __init__(self, ensemble: SimulationEnsemble, ensemble_df: pd.DataFrame|str=None):
        self.ensemble = ensemble
        
        if not self.ensemble.simulations and ensemble_df is None:
            self.ensemble.load_all_simulations()
        
        if isinstance(ensemble_df, str):
            if os.path.exists(ensemble_df):
                self.ensemble_df = pd.read_csv(ensemble_df)
            else:
                raise FileNotFoundError(f"File {ensemble_df} does not exist.")
        elif isinstance(ensemble_df, pd.DataFrame):
            self.ensemble_df = ensemble_df
        elif ensemble_df is None:
            self.ensemble_df = self.ensemble.to_df()
        else:
            raise ValueError("ensemble_df must be a pandas DataFrame, a valid file path, or None.")
        
        
        self.colors = {
            'ssp126': 'purple',
            'ssp245': 'blue',
            'ssp370': 'orange',
            'ssp585': 'red'
        }
        
    def plot_ensemble(self, sector: int|str='total', export_path: str=None):
        plt.figure(figsize=(10, 6))
        
        data = self.ensemble_df
        sim_ids = data['sim_id'].unique()
        
        for sim in sim_ids:
            
            sim_data = data[data['sim_id'] == sim]

            if isinstance(sector, int):
                plot_data = sim_data[sim_data['sector'] == sector]
                if plot_data.empty:
                    continue
            elif isinstance(sector, str):
                if sector == 'total':
                    plot_data = sim_data[sim_data['sector'] == 0.0]
                else:
                    raise ValueError(f"Invalid sector string: {sector}. Use 'total' or an integer sector ID.")
            else:
                raise ValueError("Sector must be an integer or 'total'.")
            
            year = plot_data['year']
            pred = plot_data['prediction_mm_sle'].squeeze()
            plt.plot(year, pred, alpha=0.1, color=self.colors[plot_data.scenario.values[0]])
        
        plt.xlabel('Year')
        plt.ylabel('Projected Sea Level Equivalent (mm SLE)')
        plt.title('ISEFlow-AIS Sea Level Projection Ensemble')
        plt.grid(True)
        plt.savefig(export_path if export_path else 'ISEFlow_simulation_ensemble.png')
        plt.show()
        plt.close('all')
        