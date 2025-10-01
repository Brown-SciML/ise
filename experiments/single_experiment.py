from ise.models.ISEFlow import ISEFlow_AIS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ise.utils.functions as f
from ise.data.inputs import ISEFlowAISInputs
import os

def get_aogcm_data(aogcm, ssp, data_dir):
    aogcm_dir = os.path.join(data_dir, aogcm)
    filepath = os.path.join(aogcm_dir, f"{aogcm}_{ssp}_combined.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
    df = pd.read_csv(filepath)
    return df



def main():
    args = get_args()
    data = get_aogcm_data(args.aogcm, args.scenario, data_dir=args.data_dir)
    iseflowais = ISEFlow_AIS(version="v1.1.0", )

    all_data = {}
    for sector in data.sector.unique():
        sector_data = data[data.sector == sector]
        
        if len(sector_data) != 86:
            print(f"Skipping sector {sector} due to incomplete data ({len(sector_data)} years)")
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
            ocean_forcing_type=args.ocean_forcing_type,
            ocean_sensitivity=args.sensitivity,
            ice_shelf_fracture=args.ice_shelf_fracture,
            model_configs=args.ism,
        )
        
        
        pred, uq = iseflowais.predict(inputs, smoothing_window=0)
        
        if args.plot:
            year = sector_data.year.values
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
            plt.title(f'ISEFlow-AIS Sea Level Projection: {args.aogcm.upper()} {args.scenario.upper()} - Sector {sector}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'ISEFlow_{args.ism}_{args.aogcm}_{args.scenario}_sector_{sector}.png')
            plt.show()
            plt.close('all')
            
            all_data[sector] = pred
        
    
    if args.plot:
        # total for all sectors
        total_pred = np.sum(list(all_data.values()), axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(year, total_pred, color='red', label='Total Prediction')
        plt.xlabel('Year')
        plt.ylabel('Projected Sea Level Equivalent (mm SLE)')
        plt.title(f'ISEFlow-AIS Total Sea Level Projection: {args.aogcm.upper()} {args.scenario.upper()}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'ISEFlow_{args.ism}_{args.aogcm}_{args.scenario}_total.png')
        plt.show()
        plt.close('all')
    

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run ISEFlow-AIS Experiment")
    parser.add_argument('--aogcm', type=str, required=True, help='AOGCM model name')
    parser.add_argument('--scenario', type=str, required=True, help='SSP scenario', choices=['ssp126', 'ssp245', 'ssp370', 'ssp585'])
    parser.add_argument('--ism', type=str, required=True, help='Ice Sheet Model', choices=['AWI_PISM1', 'DOE_MALI', 'ILTS_PIK_SICOPOLIS', 'IMAU_IMAUICE1', 'IMAU_IMAUICE2', 'JPL1_ISSM', 'LSCE_GRISLI',  'NCAR_CISM', 'PIK_PISM1', 'PIK_PISM2', 'UCIJPL_ISSM', 'ULB_FETISH_16km', 'ULB_FETISH_32km', 'UTAS_ElmerIce', 'VUB_AISMPALEO', 'VUW_PISM'])
    parser.add_argument('--sensitivity', type=str, default="medium", choices=['low', 'medium', 'high'], help='Ocean sensitivity setting')
    parser.add_argument('--ice_shelf_fracture', type=bool, default=False, help='Enable ice shelf fracture')
    parser.add_argument('--ocean_forcing_type', type=str, default='standard', choices=['standard', 'open'], help='Type of ocean forcing')
    parser.add_argument('--data_dir', type=str, default="/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/CMIP/dataset", help='Directory containing AOGCM data')
    parser.add_argument('--plot', action='store_true', help='Whether to generate plots')
    return parser.parse_args()
        


if __name__ == "__main__":
    main()