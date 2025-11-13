import json
from experiments.simulation import SimulationManager



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
    parser.add_argument('--override_params', type=json.loads, default=None,
                    help='JSON string of model parameters to override defaults')

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
    sim_id, filepath = sim_manager.run_simulation(params, force_rerun=args.force_rerun, override_params=args.override_params)
    print(f"Simulation completed: {sim_id}")
    print(f"Results saved to: {filepath}")

if __name__ == "__main__":
    main()