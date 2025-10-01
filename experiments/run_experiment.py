import json
from ise.models.ISEFlow import ISEFlow_AIS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ise.utils.functions as f
from ise.data.inputs import ISEFlowAISInputs
import os
import hashlib
from datetime import datetime

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
        return hashlib.md5(param_string.encode()).hexdigest()[:8]
    
    def _generate_filename(self, params, sim_id):
        fracture_str = "frac-true" if params['ice_shelf_fracture'] else "frac-false"
        filename = (f"sim_{params['aogcm']}_{params['scenario']}_{params['ism']}_"
                   f"sens-{params['sensitivity']}_{fracture_str}_"
                   f"ocean-{params['ocean_forcing_type']}_{sim_id}.parquet")
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
        
        exists_in_metadata = sim_id in metadata['sim_id'].values
        status = metadata[metadata['sim_id'] == sim_id]['status'].values[-1] if exists_in_metadata else None

        return exists_in_metadata, status

    def _previously_missing_data(self, params):
        logged, status = self.simulation_logged(params)
        return logged and ".csv not found" in status
    
    def run_simulation(self, params, force_rerun=False, skip_missing_data=False):
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
    

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run ISEFlow-AIS Experiment")
    parser.add_argument('--output_dir', type=str, 
                        default='/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/CMIP/projections',
                       help='Output directory for simulations')
    parser.add_argument('--data_dir', type=str, 
                       default="/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/CMIP/dataset",
                       help='Directory containing AOGCM data')
    
    # Single simulation parameters
    parser.add_argument('--aogcm', type=str, help='AOGCM model name')
    parser.add_argument('--scenario', type=str, choices=['ssp126', 'ssp245', 'ssp370', 'ssp585'])
    parser.add_argument('--ism', type=str, choices=['AWI_PISM1', 'DOE_MALI', 'ILTS_PIK_SICOPOLIS', 'IMAU_IMAUICE1', 'IMAU_IMAUICE2', 'JPL1_ISSM', 'LSCE_GRISLI',  'NCAR_CISM', 'PIK_PISM1', 'PIK_PISM2', 'UCIJPL_ISSM', 'ULB_FETISH_16km', 'ULB_FETISH_32km', 'UTAS_ElmerIce', 'VUB_AISMPALEO', 'VUW_PISM'])
    parser.add_argument('--sensitivity', type=str, default='medium', 
                       choices=['low', 'medium', 'high'])
    parser.add_argument('--ice_shelf_fracture', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ocean_forcing_type', type=str, default='standard', 
                       choices=['standard', 'open'])
    parser.add_argument('--force_rerun', action='store_true', 
                       help='Force rerun even if simulation exists')
    return parser.parse_args()
        

def main():
    args = get_args()
    params = {
        'aogcm': args.aogcm,
        'scenario': args.scenario,
        'ism': args.ism,
        'sensitivity': args.sensitivity,
        'ice_shelf_fracture': args.ice_shelf_fracture,
        'ocean_forcing_type': args.ocean_forcing_type,
        'data_dir': args.data_dir
    }
    sim_manager = SimulationManager(output_dir=args.output_dir)
    sim_id, filepath = sim_manager.run_simulation(params, force_rerun=args.force_rerun, skip_missing_data=True)
    print(f"Simulation completed: {sim_id}")
    print(f"Results saved to: {filepath}")

if __name__ == "__main__":
    main()