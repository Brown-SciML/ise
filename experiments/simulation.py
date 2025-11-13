

from dataclasses import dataclass
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyarrow import parquet as pq
import json
import hashlib
import numpy as np
from datetime import datetime

from ise.models.ISEFlow import ISEFlow_AIS
from ise.data.inputs import ISEFlowAISInputs
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
    
    def query_simulations(self, **filters):
        if not hasattr(self, 'metadata'):
            self.load_index()
        
        df = self.metadata.copy()
        for key, value in filters.items():
            if key not in df.columns:
                raise KeyError(f"Column '{key}' not found in simulation metadata.")
            df = df[df[key] == value]
        
        if df.empty:
            print(f"[INFO] No simulations found for filters: {filters}")
        
        return df

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
        self.data = None
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
            df['filepath'] = sim.filepath
            records.append(df)
        
        ensemble_df = pd.concat(records, ignore_index=True)
        self.df = ensemble_df
        
        if export_path:
            self.df.to_csv(export_path, index=False)
            print(f"[INFO] Ensemble dataframe exported to {export_path}")
        
        return self.df
    
    
class SimulationManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir        
        self.metadata_file = os.path.join(output_dir, "simulation_index.csv")
        self.log_file = None
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(self.metadata_file):
            pd.DataFrame().to_csv(self.metadata_file, index=False)
    
    def _generate_simulation_id(self, params):
        param_string = f"{params['aogcm']}_{params['scenario']}_{params['ism']}_" \
                      f"{params['sensitivity']}_{params['ice_shelf_fracture']}_{params['ocean_forcing_type']}"
        param_string += json.dumps(params.get('override_params', {}), sort_keys=True)
        return hashlib.md5(param_string.encode()).hexdigest()[:8]
    
    def _generate_filename(self, params, sim_id):
        fracture_str = "frac-true" if params['ice_shelf_fracture'] else "frac-false"
        filename = (f"sim_{params['aogcm']}_{params['scenario']}_{params['ism']}_"
                   f"sens-{params['sensitivity']}_{fracture_str}_"
                   f"ocean-{params['ocean_forcing_type']}_{sim_id}.parquet")
        if 'override_params' in params and params['override_params']:
            filename = filename.replace(".parquet", "")
            for key, value in params['override_params'].items():
                filename += f"_{key}-{value}"
            filename += ".parquet"
        return filename
    
    def _load_metadata(self):
        try:
            return pd.read_csv(self.metadata_file)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()
        
    def _save_metadata(self, df):
        df.to_csv(self.metadata_file, index=False)
        
    def _append_metadata(self, row_dict):
        df_row = pd.DataFrame([row_dict])
        metadata = self._load_metadata()
        metadata = pd.concat([metadata, df_row], ignore_index=True)
        self._save_metadata(metadata)

    def _log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    def simulation_exists(self, params):
        sim_id = self._generate_simulation_id(params)
        filename = self._generate_filename(params, sim_id)
        filepath = os.path.join(self.output_dir, filename)
        return os.path.exists(filepath), sim_id, filepath
    
    def simulation_logged(self, params):
        sim_id = self._generate_simulation_id(params)
        metadata = self._load_metadata()
        if metadata.empty:
            return False, None
        
        exists_in_metadata = sim_id in metadata['sim_id'].values
        status = metadata[metadata['sim_id'] == sim_id]['status'].values[-1] if exists_in_metadata else None

        return exists_in_metadata, status

    def _previously_missing_data(self, params):
        logged, status = self.simulation_logged(params)
        return logged and ".csv not found" in status
    
    def run_simulation(self, params, force_rerun=False, override_params=None):
        
        if override_params:
            params['override_params'] = override_params
                    
        exists, sim_id, filepath = self.simulation_exists(params)
        
        # set up log dir
        logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.log_file = os.path.join(logs_dir, f"log_{sim_id}.txt")

        # if these params have run before and the data was missing, skip it
        if self._previously_missing_data(params) and not force_rerun:
            self._log_message(f"Previous simulation {sim_id} failed due to missing data, skipping...")
            return sim_id, None

        if exists and not force_rerun:
            self._log_message(f"Simulation {sim_id} already exists, skipping...")
            return sim_id, filepath

        self._log_message(f"Starting simulation {sim_id} with parameters: {params}")
        start_time = datetime.now()
    
        try:
            data = self.get_aogcm_data(params['aogcm'], params['scenario'], params['data_dir'])
            
            iseflowais = ISEFlow_AIS(version="v1.1.0")
            
            all_results = {}
            sectors_processed = 0
            
            for sector in data.sector.unique():
                sector_data = data[data.sector == sector]
                
                if len(sector_data) != 86:
                    self._log_message(f"Skipping sector {sector} due to incomplete data ({len(sector_data)} years)")
                    continue
            
                inputs = ISEFlowAISInputs(
                    year=sector_data.year.values,
                    sector=sector_data.sector.values,
                    pr_anomaly=sector_data.pr_anomaly.values,
                    evspsbl_anomaly=sector_data.evspsbl_anomaly.values,
                    smb_anomaly=sector_data.smb_anomaly.values,
                    ts_anomaly=sector_data.ts_anomaly.values,
                    ocean_thermal_forcing=sector_data.thermal_forcing.values,
                    ocean_salinity=sector_data.salinity.values,
                    ocean_temperature=sector_data.temperature.values,
                    ocean_forcing_type=params['ocean_forcing_type'],
                    ocean_sensitivity=params['sensitivity'],
                    ice_shelf_fracture=params['ice_shelf_fracture'],
                    model_configs=params['ism'],
                    override_params=override_params,
                )
                
                pred, uq = iseflowais.predict(inputs, smoothing_window=0)
                
                all_results[sector] = {
                    'year': sector_data.year.values,
                    'prediction': pred.squeeze(),
                    'aleatoric': uq['aleatoric'].tolist(),
                    'epistemic': uq['epistemic'].tolist(),
                    'sector_data': sector_data.to_dict('records')
                }
                sectors_processed += 1
                
            if all_results:
                predictions = [all_results[sector]['prediction'] for sector in all_results.keys()]
                total_prediction = np.sum(predictions, axis=0)
                
                aleatoric_uncertainties = [np.square(all_results[sector]['aleatoric']) for sector in all_results.keys()]
                epistemic_uncertainties = [np.square(all_results[sector]['epistemic']) for sector in all_results.keys()]
                
                total_aleatoric = np.sqrt(np.sum(aleatoric_uncertainties, axis=0))
                total_epistemic = np.sqrt(np.sum(epistemic_uncertainties, axis=0))                                 
                
                all_results['total'] = {
                    'year': list(all_results.values())[0]['year'],
                    'prediction': total_prediction,
                    'aleatoric': total_aleatoric,
                    'epistemic': total_epistemic
                }
            
            results_df = self._format_results_for_parquet(all_results, params)
            results_df.to_parquet(filepath, index=False)

            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds()
            
            self._append_metadata({
                'sim_id': sim_id,
                'filename': filepath.split('/')[-1],
                'filepath': filepath,
                'aogcm': params['aogcm'],
                'scenario': params['scenario'],
                'ism': params['ism'],
                'sensitivity': params['sensitivity'],
                'ice_shelf_fracture': params['ice_shelf_fracture'],
                'ocean_forcing_type': params['ocean_forcing_type'],
                'start_time': start_time,
                'end_time': end_time,
                'runtime_seconds': runtime,
                'sectors_processed': sectors_processed,
                'total_sectors': len(data.sector.unique()),
                'file_size_mb': os.path.getsize(filepath) / (1024 * 1024),
                'status': 'completed'
            })

            self._log_message(f"Simulation {sim_id} completed in {runtime:.1f}s, processed {sectors_processed} sectors")
            return sim_id, filepath
        
        except Exception as e:
            error_time = datetime.now()
            # Update metadata with error info
            
            self._append_metadata({
                'sim_id': sim_id,
                'filename': filepath.split('/')[-1],
                'filepath': filepath,
                'aogcm': params['aogcm'],
                'scenario': params['scenario'],
                'ism': params['ism'],
                'sensitivity': params['sensitivity'],
                'ice_shelf_fracture': params['ice_shelf_fracture'],
                'ocean_forcing_type': params['ocean_forcing_type'],
                'start_time': start_time,
                'end_time': error_time,
                'runtime_seconds': (error_time - start_time).total_seconds(),
                'sectors_processed': 0,
                'total_sectors': len(data.sector.unique()) if 'data' in locals() else 0,
                'file_size_mb': 0,
                'status': f'error: {str(e)}'
            })
            self._log_message(f"Simulation {sim_id} failed: {str(e)}")
            raise
            

    def _format_results_for_parquet(self, all_results, params):
        """Convert nested results to flat DataFrame for parquet storage"""
        rows = []
        
        for sector, sector_results in all_results.items():
            if sector == 'total':
                continue  # Handle total separately
                
            years = sector_results['year']
            predictions = sector_results['prediction'].squeeze()
            
            # Extract uncertainty components if available
            aleatoric = np.array(sector_results.get('aleatoric', np.zeros_like(predictions))).squeeze()
            epistemic = np.array(sector_results.get('epistemic', np.zeros_like(predictions))).squeeze()
            
            # Create rows for each year
            for i, year in enumerate(years):
                row = {
                    # Simulation parameters
                    'aogcm': params['aogcm'],
                    'scenario': params['scenario'], 
                    'ism': params['ism'],
                    'sensitivity': params['sensitivity'],
                    'ice_shelf_fracture': params['ice_shelf_fracture'],
                    'ocean_forcing_type': params['ocean_forcing_type'],
                    
                    # Time and location
                    'year': year,
                    'sector': sector,
                    
                    # Results
                    'prediction_mm_sle': predictions[i] if predictions.ndim > 0 else predictions,
                    'aleatoric_uncertainty': aleatoric[i] if aleatoric.ndim > 0 else aleatoric,
                    'epistemic_uncertainty': epistemic[i] if epistemic.ndim > 0 else epistemic,
                    'total_uncertainty': (aleatoric[i] if aleatoric.ndim > 0 else aleatoric) + 
                                       (epistemic[i] if epistemic.ndim > 0 else epistemic)
                }
                rows.append(row)
        
        # Add total prediction if available
        if 'total' in all_results:
            total_results = all_results['total']
            years = total_results['year']
            predictions = total_results['prediction'].squeeze()
            aleatoric = total_results.get('aleatoric', np.full_like(predictions, np.nan)).squeeze()
            epistemic = total_results.get('epistemic', np.full_like(predictions, np.nan)).squeeze()

            
            for i, year in enumerate(years):
                row = {
                    'aogcm': params['aogcm'],
                    'scenario': params['scenario'],
                    'ism': params['ism'], 
                    'sensitivity': params['sensitivity'],
                    'ice_shelf_fracture': params['ice_shelf_fracture'],
                    'ocean_forcing_type': params['ocean_forcing_type'],
                    'year': year,
                    'sector': 0,
                    'prediction_mm_sle': predictions[i] if predictions.ndim > 0 else predictions,
                    'aleatoric_uncertainty': aleatoric[i],
                    'epistemic_uncertainty': epistemic[i],
                    'total_uncertainty': aleatoric[i] + epistemic[i]
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
            

    def get_aogcm_data(self, aogcm, ssp, data_dir):
        aogcm_dir = os.path.join(data_dir, aogcm)
        filepath = os.path.join(aogcm_dir, f"{aogcm}_{ssp}_combined.csv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        df = pd.read_csv(filepath)
        return df
    
    def load_simulation_results(self, sim_id):
        metadata = self._load_metadata()
        if sim_id not in metadata['sim_id'].values:
            raise ValueError(f"Simulation {sim_id} not found")

        filepath = metadata.loc[metadata['sim_id'] == sim_id, 'filepath'].values[0]
        return pd.read_parquet(filepath)
    