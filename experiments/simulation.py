

from dataclasses import dataclass
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyarrow import parquet as pq

@dataclass
class SimulationIndex:
    simulation_dir: str=None
    
    def __post_init__(self):
        self.load_index()

    def load_index(self,):
        metadata_path = os.path.join(self.simulation_dir, "simulation_index.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file {metadata_path} not found.")
        
        self.metadata = pd.read_csv(metadata_path)
        return self.metadata

    def get_simulation_by_id(self, sim_id: str):
        if not hasattr(self, 'metadata'):
            self.load_index()
        
        idx = self.metadata[self.metadata['sim_id'] == sim_id]
        if idx.empty:
            raise ValueError(f"Simulation ID {sim_id} not found in metadata.")
        return idx.iloc[0]
    
    def get_simulation_by_filepath(self, filepath: str):
        if not hasattr(self, 'metadata'):
            self.load_index()
        
        idx = self.metadata[self.metadata['filepath'] == filepath]
        if idx.empty:
            raise ValueError(f"Filepath {filepath} not found in metadata.")
        return idx.iloc[0]
    
    def get_simulation_by_params(self, aogcm: str, scenario: str, ism: str, ocean_forcing_type: str, sensitivity: float, ice_shelf_fracture: bool):
        if not hasattr(self, 'metadata'):
            self.load_index()
        
        idx = self.metadata.loc[
            (self.metadata['aogcm'] == aogcm)  &
            (self.metadata['scenario'] == scenario) &
            (self.metadata['ism'] == ism) &
            (self.metadata['ocean_forcing_type'] == ocean_forcing_type) &
            (self.metadata['sensitivity'] == sensitivity) &
            (self.metadata['ice_shelf_fracture'] == ice_shelf_fracture)
        ]
        if idx.empty:
            raise ValueError("No matching simulation found for the provided parameters.")
        return idx.iloc[0]
    
    def get_completed_simulations(self,):
        if not hasattr(self, 'metadata'):
            self.load_index()
        
        completed = self.metadata[self.metadata['status'] == 'completed']
        return completed

@dataclass
class Simulation:
    sim_id: str=None
    filepath: str=None
    
    aogcm: str=None,
    scenario: str=None,
    ism: str=None,
    ocean_forcing_type: str=None,
    sensitivity: float=None,
    ice_shelf_fracture: bool=None,
    
    simulation_dir: str=None,

    def __post_init__(self):
        self._check_inputs()
        self.index = SimulationIndex(simulation_dir=self.simulation_dir)

    def _check_inputs(self):
        if self.sim_id is None and self.filepath is None and not all([self.aogcm, self.scenario, self.ism, self.ocean_forcing_type, self.sensitivity, self.ice_shelf_fracture]):
            raise ValueError("Must provide either sim_id, filepath, or all simulation parameters.")

    
    def _get_simulation_metadata(self):
        if self.sim_id is not None:
            return self.index.get_simulation_by_id(self.sim_id)
            
        elif self.sim_id is None and self.filepath is not None:
            return self.index.get_simulation_by_filepath(self.filepath)
            
        elif self.sim_id is None and self.filepath is None:
            return self.index.get_simulation_by_params(
                aogcm=self.aogcm,
                scenario=self.scenario,
                ism=self.ism,
                ocean_forcing_type=self.ocean_forcing_type,
                sensitivity=self.sensitivity,
                ice_shelf_fracture=self.ice_shelf_fracture
            )
        else:
            raise ValueError("Invalid state in _get_simulation_metadata.")
    
    def load_simulation(self, metadata: pd.DataFrame=None):
        if metadata is None:
            self.sim_metadata = self._get_simulation_metadata()
        else:
            self.sim_metadata = metadata

        self.sim_id = self.sim_metadata['sim_id']
        self.filepath = self.sim_metadata['filepath']

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Simulation file {self.filepath} not found.")
        
        self.data = pq.read_table(self.filepath)
        self.data = self.data.to_pandas()
        
        return self.data
    
    def plot(self, sector: int|str='total', export_path: str=None):
        if self.data is None:
            self.load_simulation()

        if isinstance(sector, int):
            plot_data = self.data[self.data['sector'] == sector]
            if plot_data.empty:
                raise ValueError(f"No data found for sector {sector}.")
        elif isinstance(sector, str):
            if sector == 'total':
                plot_data = self.data[self.data['sector'] == 0.0]
            else:
                raise ValueError(f"Invalid sector string: {sector}. Use 'total' or an integer sector ID.")
        else:
            raise ValueError("Sector must be an integer or 'total'.")
        
        year = plot_data['year']
        pred = plot_data['prediction_mm_sle']
        uq = {
            'aleatoric': plot_data['aleatoric_uncertainty'],
            'epistemic': plot_data['epistemic_uncertainty'],
            'total': plot_data['total_uncertainty']
        }
        
        plt.figure(figsize=(10, 6))
        pred = pred.squeeze()
        aleatoric = uq['aleatoric'].squeeze()
        epistemic = uq['epistemic'].squeeze()
        total = aleatoric + epistemic.squeeze()
        plt.fill_between(year, pred - epistemic, pred + epistemic, color='blue', alpha=0.5, label=r'Emulator Uncertainty (2$\sigma$)')
        plt.fill_between(year, pred - total, pred + total, color='green', alpha=0.5, label='Data Coverage Uncertainty')
        plt.plot(year, pred, color='red', label='Prediction')

        plt.xlabel('Year')
        plt.ylabel('Projected Sea Level Equivalent (mm SLE)')
        plt.title('ISEFlow-AIS Sea Level Projection')
        plt.legend()
        plt.grid(True)
        plt.savefig(export_path if export_path else f'ISEFlow_simulation_{self.sim_id}.png')
        plt.show()
        plt.close('all')



@dataclass
class SimulationEnsemble:
    simulations: list[Simulation]=None
    simulation_dir: str=None

    def __post_init__(self):
        self._check_inputs()
        self.simulations = [] if self.simulations is None else self.simulations
        self.index = None
        self.df = None
    
    def _check_inputs(self):
        if self.simulations is None and self.simulation_dir is None:
            raise ValueError("Must provide either a list of simulations or a simulation directory to load simulations from.")
    
    def load_all_simulations(self,):
        if self.simulation_dir is None:
            raise ValueError("simulation_dir must be set to load simulations.")
        self.index = SimulationIndex(simulation_dir=self.simulation_dir)
        completed = self.index.get_completed_simulations()
        
        print(f"[INFO] Loading {len(completed)} completed simulations from {self.simulation_dir}...")
        print(f'[INFO] Total ensemble size: {completed.file_size_mb.sum():.2f} MB')

        for _, row in tqdm(completed.iterrows(), total=len(completed), desc="Loading simulations", unit="sim", dynamic_ncols=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}]" ):

            try:
                sim = Simulation(
                    sim_id=row.sim_id, 
                    filepath=row.filepath,
                    simulation_dir=self.simulation_dir
                )
                sim.load_simulation(metadata=row)
                self.add_simulation(sim)
            except pd.errors.EmptyDataError:
                print(f"[WARNING] Empty dataframe at sim_id={sim.sim_id}. Skipping...")
            except KeyError as e:
                print(f"[WARNING] {e} not found in dataframe at sim_id={sim.sim_id}. Skipping...")
        
        self.to_df()

    def add_simulation(self, sim: Simulation):
        if self.simulations is None:
            self.simulations = []
        self.simulations.append(sim)
    
    def clean_simulations(self, ):
        if self.simulation_dir is None:
            raise ValueError("simulation_dir must be set to load simulations.")
        
        if self.index is None:
            self.index = SimulationIndex(simulation_dir=self.simulation_dir)
        completed = self.index.get_completed_simulations()
        filepaths = completed.filepath.unique().tolist()

        total_deleted = 0
        for fp in tqdm(filepaths, desc="Cleaning simulations", unit="sim", dynamic_ncols=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}]" ):
            d = pd.read_parquet(fp)
            if d.empty:
                os.remove(fp)
                print(f'[INFO] Empty dataframe at {fp}, deleted. ')
                total_deleted += 1
            elif 'sim_id' in d.columns:
                os.remove(fp)
                print(f"[INFO] 'sim_id' not found at {fp}, deleted.")
                total_deleted += 1
            else:
                continue
        print(f"[INFO] Total deleted files: {total_deleted}")
        
    def to_df(self, export_path: str=None):
        if not self.simulations:
            self.load_all_simulations()
            
        if self.df is not None:
            if export_path:
                self.df.to_csv(export_path, index=False)
                print(f"[INFO] Ensemble dataframe exported to {export_path}")
            return self.df
        
        records = []
        for sim in self.simulations:
            if sim.data is None:
                sim.load_simulation()
            df = sim.data.copy()
            df['sim_id'] = sim.sim_id
            df['aogcm'] = sim.sim_metadata['aogcm']
            df['scenario'] = sim.sim_metadata['scenario']
            df['ism'] = sim.sim_metadata['ism']
            df['ocean_forcing_type'] = sim.sim_metadata['ocean_forcing_type']
            df['sensitivity'] = sim.sim_metadata['sensitivity']
            df['ice_shelf_fracture'] = sim.sim_metadata['ice_shelf_fracture']
            records.append(df)
        
        ensemble_df = pd.concat(records, ignore_index=True)
        self.df = ensemble_df
        
        if export_path:
            self.df.to_csv(export_path, index=False)
            print(f"[INFO] Ensemble dataframe exported to {export_path}")
        
        return self.df